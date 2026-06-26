"""
The configuration is the SINGLE SOURCE OF TRUTH for a simulation run. It is a
plain nested dict (loaded from JSON); this module owns loading it, validating it
against itself, and computing the derived quantities the rest of the package
needs (voxel_shape shape as (Nz,Ny,Nx), sinogram shape, etc.).

Schema (four sections):

  voxel_space : the emitting/attenuating voxel space grid
      axis_order     : "xyz" (documents the order of the triples below)
      grid_size_mm   : [dx, dy, dz]   physical size of one voxel
      num_voxels     : [Nx, Ny, Nz]   number of voxels per axis

  scanner : detector hardware
      symmetry_axis          : "z"
      axial_fov_mm        : detector length along the symmetry axis (FOVz)
      transaxial_fov_mm      : detector length along the transverse axis (FOVxy)
      radius_mm              : detector ring radius
      num_rings              : number of detector rings
      num_detectors_per_ring : crystals per ring

  sinogram : how lines of response are binned (the michelogram)
      num_angular_bins, num_radial_bins, num_radial_trim,
      max_ring_difference, span, num_axial_planes

  mcgpu : physics + acquisition knobs (passed through to the .in file)

Two families of numbers that must never be confused (the source of a past bug):
  VOXEL SPACE side   -> voxel_space.num_voxels / grid_size_mm
  DETECTOR side -> scanner.* and sinogram.*
They live in separate sections precisely so a detector number can't be typed
into an voxel space slot.

Axis-order discipline: the triples in voxel_space are unpacked ONCE here (into
named nx,ny,nz / dx,dy,dz) so no other module indexes a raw [0]/[1]/[2] and can
guess wrong.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path


# ----------------------------------------------------------------------------
# Loading / merging
# ----------------------------------------------------------------------------

def load_config(*paths):
    """Load and section-wise merge JSON configs. Later files win per field.

    Pattern: load_config("template.json", "run_42/delta.json")
    Sections merge (not replace): within a section, later keys override earlier
    ones; keys present only in an earlier file survive.
    """
    merged = {}
    for p in paths:
        cfg = json.loads(Path(p).read_text())
        for section, fields in cfg.items():
            if isinstance(fields, dict):
                merged.setdefault(section, {}).update(fields)
            else:
                merged[section] = fields
    return merged

def default_config():
    template_path = Path(__file__).parent / "templates" / "template.json"
    return load_config(template_path)

# ----------------------------------------------------------------------------
# Axis-order unpacking -- the ONLY place that turns triples into named axes
# ----------------------------------------------------------------------------

def _check_axis_order(vs):
    order = vs.get("axis_order", "xyz")
    if order != "xyz":
        raise ValueError(
            f"voxel_space.axis_order={order!r}; only 'xyz' is supported. The "
            f"triples grid_size_mm and num_voxels are interpreted as [x, y, z]."
        )


def num_voxels(config):
    """Return (Nx, Ny, Nz) from voxel_space.num_voxels, honoring axis_order."""
    vs = config["voxel_space"]
    _check_axis_order(vs)
    nx, ny, nz = vs["num_voxels"]
    return (int(nx), int(ny), int(nz))


def grid_size_mm(config):
    """Return (dx, dy, dz) from voxel_space.grid_size_mm, honoring axis_order."""
    vs = config["voxel_space"]
    _check_axis_order(vs)
    dx, dy, dz = vs["grid_size_mm"]
    return (float(dx), float(dy), float(dz))


def voxel_space_shape_zyx(config):
    """Voxel space shape (Nz, Ny, Nx) -- the convention used by VoxelGrid and the
    .vox writer (axial slice first, x fastest on ravel)."""
    nx, ny, nz = num_voxels(config)
    return (nz, ny, nx)


def voxel_space_shape_xyz(config):
    """Voxel space shape as (Nx, Ny, Nz) -- the order VoxelSpaceBuilder takes."""
    return num_voxels(config)


def voxel_space_extent_mm(config):
    """Physical size of the voxel space bounding box (x, y, z) in mm."""
    nx, ny, nz = num_voxels(config)
    dx, dy, dz = grid_size_mm(config)
    return (nx * dx, ny * dy, nz * dz)


# ----------------------------------------------------------------------------
# Sinogram shape -- ported verbatim from MCGPU-PET (read_input / kernel)
# ----------------------------------------------------------------------------

@dataclass
class SegmentInfo:
    """One michelogram segment's layout in the stored sinogram.

    Storage order is iseg = 0, 1, 2, 3, ... which corresponds to signed
    segment numbers 0, -1, +1, -2, +2, ... (negative slope gets the lower,
    odd iseg). All fields are derived from the kernel's own arithmetic.

    iseg            : storage index (0,1,2,...); the segment's block is the
                      izm range [start_plane, start_plane + n_planes).
    segment_number  : signed michelogram segment (0, -1, +1, -2, ...).
    ring_diff_min/max : inclusive band of ring differences |iz2-iz1| this
                      segment covers (span-wide; segment 0 covers 0..(span-1)/2).
    start_plane     : first plane index (izm) of this segment in the NSINOS axis.
    n_planes        : number of planes in this segment.
    axial_sum_min/max : inclusive range of (iz1+iz2) covered. The mean ring
                      position of a plane is (iz1+iz2)/2; SSRB maps a plane to
                      the direct slice at that midpoint.
    """
    iseg: int
    segment_number: int
    ring_diff_min: int
    ring_diff_max: int
    start_plane: int
    n_planes: int
    axial_sum_min: int
    axial_sum_max: int


def segment_table(config):
    """Return the michelogram segment layout as a list of SegmentInfo, in
    storage order (iseg = 0, 1, 2, ...).

    This is the authoritative per-segment breakdown of the stored sinogram's
    axial (NSINOS) axis. It is computed by replaying the kernel's exact binning
    arithmetic (MCGPU-PET_kernel.cu): for every ring pair (iz1, iz2) with
    |iz2-iz1| <= MRD, the kernel computes a segment index and a plane index izm;
    we aggregate those into contiguous per-segment blocks.

    Rebinning (SSRB/FORE) needs exactly this: which planes belong to which
    segment, each segment's ring-difference band (-> polar angle), and the
    axial-sum range (-> which direct slice a plane maps to).
    """
    s = config["sinogram"]
    sc = config["scanner"]
    nzs = s["num_axial_planes"]
    mrd = s["max_ring_difference"]
    span = s["span"]
    nrows = sc["num_rings"]

    # Number of segments MCGPU-PET allocates (read_input's NSEG). The kernel's
    # per-event `incl` formula can round a ring difference near MRD up to a
    # segment index BEYOND this count; those events are out of bounds and the
    # binary drops them. We must apply the same cutoff, or we would invent
    # spurious single-plane segments at the end (a 2-plane overcount surfaces at
    # small span, e.g. span=3, though not at span=11 where it happens to align).
    nseg_alloc = 2 * (mrd // span) + 1

    # Aggregate, per iseg, the set of izm and the (iz1+iz2), |iz2-iz1| seen.
    izm_by_seg: dict[int, set] = {}
    sum_by_seg: dict[int, set] = {}
    delta_by_seg: dict[int, set] = {}

    half = (span + 1) // 2
    for iz1 in range(nrows):
        for iz2 in range(nrows):
            delta = abs(iz2 - iz1)
            if delta > mrd:
                continue
            incl = (delta - 1 + half) // span
            iseg = 2 * incl
            if iz2 < iz1 and iseg > 0:
                iseg -= 1
            if iseg >= nseg_alloc:
                # Beyond the allocated segments: MCGPU-PET discards these events.
                continue
            # cumulative offset of this segment's block (ofseg in the kernel)
            ofseg = iseg * nzs
            for kka in range(0, iseg + 1):
                if kka > 1:
                    ofseg -= (span + 1)
                if kka > 3:
                    ofseg -= ((kka - 2) // 2) * 2 * span
            ofz = (half + (incl - 1) * span) if incl > 0 else 0
            izm = ofseg + (iz1 + iz2) - ofz
            izm_by_seg.setdefault(iseg, set()).add(izm)
            sum_by_seg.setdefault(iseg, set()).add(iz1 + iz2)
            delta_by_seg.setdefault(iseg, set()).add(delta)

    table = []
    for iseg in sorted(izm_by_seg):
        izm = sorted(izm_by_seg[iseg])
        deltas = sorted(delta_by_seg[iseg])
        sums = sorted(sum_by_seg[iseg])
        table.append(SegmentInfo(
            iseg=iseg,
            segment_number=_signed_segment(iseg),
            ring_diff_min=deltas[0],
            ring_diff_max=deltas[-1],
            start_plane=izm[0],
            n_planes=len(izm),
            axial_sum_min=sums[0],
            axial_sum_max=sums[-1],
        ))
    return table


def _signed_segment(iseg):
    """iseg storage index -> signed michelogram segment number.
    0->0, 1->-1, 2->+1, 3->-2, 4->+2, ..."""
    if iseg == 0:
        return 0
    n = (iseg + 1) // 2
    return -n if iseg % 2 == 1 else n


def sinogram_shape(config):
    """Compute the 3D sinogram layout the binary will produce, from config alone.

    NSINOS (the number of stored planes) is the SUM of plane counts over all
    michelogram segments (see segment_table). It is NOT a compression of NZS =
    num_axial_planes: NZS is the per-segment plane cap (segment 0, the direct
    segment, reaches it, with NZS = 2*num_rings - 1), while NSINOS is the grand
    total across segments. Both are axial quantities at different stages.

    The kernel writes a flat int32 buffer of length NBINS; the per-event index is
        ibin = izm*(NANGLES*NRAD) + ith*NRAD + ir
    so the C-order reshape is (NSINOS, NANGLES, NRAD).
    """
    s = config["sinogram"]
    nrad = s["num_radial_bins"]
    nang = s["num_angular_bins"]
    nzs = s["num_axial_planes"]

    segs = segment_table(config)
    nsinos = sum(seg.n_planes for seg in segs)
    nbins = nrad * nang * nsinos
    return {
        "NSEG": len(segs),
        "NSINOS": nsinos,
        "NBINS": nbins,
        "shape_csr": (nsinos, nang, nrad),  # (planes, angular, radial), C-order
        "NRAD": nrad,
        "NANGLES": nang,
        "NZS": nzs,
    }


def radial_fov_mm(config):
    """The transverse FOV diameter actually covered by the radial bins.

    MCGPU-PET's radial coordinate is combinatorial (from the crystal-index
    difference), and maps to the LOR's perpendicular distance from the axis by
    the chord relation  s = R * cos(pi * d / NCRYSTALS),  where d is the angular
    crystal separation. This mapping is NONUNIFORM across the FOV (bins are
    coarsest at the center, finest near the edge), which is why downstream
    reconstruction needs arc correction.

    Returns the covered diameter 2*s_max for the retained NRAD bins.
    """
    s = config["sinogram"]; sc = config["scanner"]
    ncryst = sc["num_detectors_per_ring"]
    nang = s["num_angular_bins"]
    nrad = s["num_radial_bins"]
    R = sc["radius_mm"]
    r_max = nrad // 2
    d = nang - r_max
    s_max = R * abs(math.cos(math.pi * d / ncryst))
    return 2.0 * s_max


# ----------------------------------------------------------------------------
# Validation: config against itself
# ----------------------------------------------------------------------------

def validate_config(config):
    """Validate the config against itself. Raises ValueError on hard
    contradictions (would crash or silently corrupt a run); warns on soft
    mismatches (legal but probably not intended)."""
    for sec in ("voxel_space", "scanner", "sinogram", "mcgpu"):
        if sec not in config:
            raise ValueError(f"config missing required section: {sec!r}")
    _validate_voxel_space(config)
    _validate_scanner(config)
    _validate_sinogram(config)
    _validate_voxel_space_vs_scanner(config)
    _validate_mcgpu(config["mcgpu"])


def _validate_voxel_space(config):
    vs = config["voxel_space"]
    _check_axis_order(vs)
    dx, dy, dz = grid_size_mm(config)
    if any(d <= 0 for d in (dx, dy, dz)):
        raise ValueError(f"grid_size_mm must be all positive: {vs['grid_size_mm']}")
    nx, ny, nz = num_voxels(config)
    if any(n <= 0 for n in (nx, ny, nz)):
        raise ValueError(f"num_voxels must be all positive: {vs['num_voxels']}")


def _validate_scanner(config):
    sc = config["scanner"]
    if sc.get("symmetry_axis", "z") != "z":
        raise ValueError(
            f"scanner.symmetry_axis={sc.get('symmetry_axis')!r}; MCGPU-PET assumes "
            f"the axial direction is z."
        )
    if sc["radius_mm"] <= 0:
        raise ValueError(f"scanner.radius_mm must be > 0, got {sc['radius_mm']}.")
    if sc["axial_fov_mm"] <= 0:
        raise ValueError(
            f"scanner.axial_fov_mm must be > 0, got {sc['axial_fov_mm']}."
        )
    if sc["transaxial_fov_mm"] <= 0:
        raise ValueError(
            f"scanner.transaxial_fov_mm must be > 0, got {sc['transaxial_fov_mm']}."
        )
    if sc["num_rings"] <= 0 or sc["num_detectors_per_ring"] <= 0:
        raise ValueError("scanner.num_rings and num_detectors_per_ring must be > 0.")


def _validate_sinogram(config):
    s = config["sinogram"]
    sc = config["scanner"]
    ndet = sc["num_detectors_per_ring"]

    # Radial bins <-> detector count + radial trim (hard identity).
    expected_rad = ndet + 1 - 2 * s["num_radial_trim"]
    if s["num_radial_bins"] != expected_rad:
        raise ValueError(
            f"num_radial_bins={s['num_radial_bins']} but expected {expected_rad} "
            f"= num_detectors_per_ring + 1 - 2*num_radial_trim "
            f"({ndet} + 1 - 2*{s['num_radial_trim']})."
        )
    
    # Coverage: do the retained radial bins span the transaxial FOV?
    covered = radial_fov_mm(config)
    fov_xy = sc["transaxial_fov_mm"]
    if covered < fov_xy - 1e-6:
        warnings.warn(
            f"radial bins cover only {covered:.1f} mm diameter but transaxial_fov_mm "
            f"is {fov_xy:.1f} mm; LORs from the outer FOV are discarded. To cover the "
            f"full FOV, reduce num_radial_trim (increase num_radial_bins)."
        )

    # Over-coverage: retained bins reach beyond the declared FOV. Not a
    # correctness problem (no LORs lost) -- just larger sinograms, and the
    # outermost bins are the most arc-distorted.
    if covered > fov_xy * 1.02:
        warnings.warn(
            f"radial bins cover {covered:.1f} mm diameter, beyond "
            f"transaxial_fov_mm={fov_xy:.1f} mm; the extra outer bins are the most "
            f"arc-distorted and enlarge the sinogram. Consider increasing num_radial_trim to "
            f"tighten coverage to the FOV."
        )

    # Angular bins <-> detectors/2 (convention; breakable, e.g. angular mashing).
    half = ndet // 2
    if s["num_angular_bins"] != half:
        warnings.warn(
            f"num_angular_bins={s['num_angular_bins']} != num_detectors_per_ring/2 "
            f"={half}. Fine if intentional (angular mashing)."
        )

    # MRD must fit within the ring count (hard).
    if s["max_ring_difference"] >= sc["num_rings"]:
        raise ValueError(
            f"max_ring_difference={s['max_ring_difference']} must be "
            f"< num_rings={sc['num_rings']}."
        )

    # span: conventionally a positive odd integer.
    if s["span"] < 1 or s["span"] % 2 == 0:
        warnings.warn(
            f"span={s['span']} is conventionally a positive odd integer; unusual "
            f"values may not lay out the michelogram as expected."
        )

    # num_axial_planes is the per-segment cap; for a full 3D acquisition it
    # equals 2*num_rings - 1 (segment 0's extent). Warn if not.
    expected_nzs = 2 * sc["num_rings"] - 1
    if s["num_axial_planes"] != expected_nzs:
        warnings.warn(
            f"num_axial_planes={s['num_axial_planes']} but 2*num_rings-1="
            f"{expected_nzs}. num_axial_planes is the maximum planes in one segment "
            f"(the direct segment reaches it); for a full 3D scan it equals "
            f"2*num_rings-1."
        )


def _validate_voxel_space_vs_scanner(config):
    """Soft checks relating the voxel_space grid to the scanner geometry. These are
    warnings, not errors: an voxel_space that doesn't fill the FOV still runs."""
    ex, ey, ez = voxel_space_extent_mm(config)
    sc = config["scanner"]
    dx, dy, dz = grid_size_mm(config)

    # Axial: voxel_space should fill (not exceed) the axial FOV.
    fov_z = sc["axial_fov_mm"]
    if ez > fov_z + 0.5 * dz:
        warnings.warn(
            f"voxel_space axial extent {ez:.1f} mm exceeds scanner axial fov "
            f"{fov_z:.1f} mm; the protruding ends won't be detected."
        )
    elif ez < fov_z - dz:
        warnings.warn(
            f"voxel_space axial extent {ez:.1f} mm is shorter than scanner axial fov "
            f"{fov_z:.1f} mm; the end rings will see little/no activity."
        )

    # Transverse: voxel_space must fit inside the bore (hard) and ideally inside the
    # ring radius (soft, since corners with only air are harmless).
    # Ideally, voxel space should not exceed the transaxial FOV.
    bore_d = 2.0 * sc["radius_mm"]
    if ex > bore_d or ey > bore_d:
        raise ValueError(
            f"voxel_space transverse extent ({ex:.1f} x {ey:.1f} mm) exceeds detector "
            f"bore diameter {bore_d:.1f} mm."
        )
    corner_r = 0.5 * math.sqrt(ex ** 2 + ey ** 2)
    if corner_r > sc["radius_mm"] + max(dx, dy):
        warnings.warn(
            f"voxel_space bbox corners reach radius {corner_r:.1f} mm > scanner radius "
            f"{sc['radius_mm']:.1f} mm; corner regions sit outside the detector "
            f"(harmless if they contain only air)."
        )
    fov_xy = sc["transaxial_fov_mm"]
    if ex > fov_xy + 0.5 * dx or ey > fov_xy + 0.5 * dy:
        warnings.warn(
            f"voxel_space transverse extent ({ex:.1f} x {ey:.1f} mm) exceeds scanner transaxial length "
            f"{fov_xy:.1f} mm; the protruding ends won't be detected correctly."
        )


def _validate_mcgpu(m):
    if m["energy_window_low_eV"] >= m["energy_window_high_eV"]:
        raise ValueError(
            f"energy_window_low_eV ({m['energy_window_low_eV']}) must be < "
            f"energy_window_high_eV ({m['energy_window_high_eV']})."
        )
    if m["acquisition_time_s"] <= 0:
        raise ValueError(f"acquisition_time_s must be > 0, got {m['acquisition_time_s']}.")
    if m.get("isotope_mean_life_s", 1) <= 0:
        raise ValueError(
            f"isotope_mean_life_s must be > 0, got {m.get('isotope_mean_life_s')}."
        )
    mats = m.get("materials", [])
    if not 1 <= len(mats) <= 15:
        raise ValueError(
            f"materials has {len(mats)} entries; MCGPU-PET supports 1..15 "
            f"(#define MAX_MATERIALS 15). Material ID k in the .vox refers to the "
            f"k-th file in this list."
        )
    for flag in ("tally_material_dose", "tally_voxel_dose"):
        v = m.get(flag, "NO")
        if v not in ("YES", "NO"):
            raise ValueError(f"{flag} must be 'YES' or 'NO', got {v!r}.")
