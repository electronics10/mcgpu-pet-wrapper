"""
rebinning.py  (auxiliary; not part of the core wrapper)

Rebinning collapses a 3D PET sinogram (direct + oblique michelogram segments)
into a stack of 2D direct sinograms -- one per axial slice -- that a 2D
reconstruction method can handle. Two standard methods:

  SSRB (Single-Slice Rebinning): assign each oblique LOR to the direct slice at
        the AXIAL MIDPOINT of its two rings. Pure bookkeeping; exact only for
        activity near the axis, but cheap and robust.

  FORE (Fourier Rebinning): transform each oblique sinogram over (radial s,
        angular phi), then use the frequency-distance relation to reassign each
        2D Fourier component to a direct slice. Much more accurate for larger
        objects / steeper obliqueness.

Both consume the per-segment arrays + geometry from the mcgpu_backend package
(config.segment_table, data_reader.read_sinogram_segments). This module is kept
OUT of the core wrapper -- it is reconstruction-adjacent processing, provided as
a convenience; a serious recon pipeline may supersede it.

Arc correction: MCGPU-PET exports the sinogram in raw arc coordinates (the
radial axis is nonuniformly sampled in physical distance s; see arc_correct).
FORE and analytic FBP assume uniform s. Both ssrb() and fore() therefore
arc-correct their input internally by default (arc_correction=True), so the
usual pipeline is simply:

    read -> ssrb/fore -> FBP

Notes:
  - arc_correct consumes the per-segment list (the input form of ssrb/fore), so
    it is always applied BEFORE rebinning, never after (ssrb returns a stacked
    array, not a segment list).
  - SSRB is radial-agnostic, so arc-correcting before it does not change the
    axial bookkeeping; it only puts the radial axis on a uniform grid for the
    subsequent FBP. FORE genuinely REQUIRES uniform s for its Fourier step.
  - For a PET-specific iterative reconstructor (STIR/parallelproj) that models
    the arc geometry itself, pass arc_correction=False and let the reconstructor
    handle the geometry; or call arc_correct manually for full control.

Conventions:
  - A "direct sinogram stack" has shape (NSLICES, NANGLES, NRAD), where
    NSLICES = 2*num_rings - 1 axial positions indexed by (iz1+iz2), i.e. twice
    the mean-ring coordinate. Slice index = iz1+iz2 ranges 0 .. 2*(num_rings-1).
  - Segment polar angle theta: tan(theta) = (ring_diff * ring_spacing) /
    (2 * ring_radius), with ring_spacing = axial_length / num_rings. (Small-angle
    PET geometry; ring_diff is the segment's mean ring difference.)
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _ring_spacing_mm(config):
    sc = config["scanner"]
    return sc["axial_fov_mm"] / sc["num_rings"]


def _num_direct_slices(config):
    # axial positions indexed by (iz1+iz2): 0 .. 2*(num_rings-1)
    return 2 * config["scanner"]["num_rings"] - 1


def _segment_mean_ring_diff(seg):
    return 0.5 * (seg["ring_diff_min"] + seg["ring_diff_max"])


# ----------------------------------------------------------------------------
# Arc correction
# ----------------------------------------------------------------------------
#
# MCGPU-PET stores the sinogram in RAW ARC COORDINATES: the radial bin index
# `ir` comes from the crystal-index difference, and the physical perpendicular
# distance of the LOR from the axis is the NONUNIFORM chord mapping
#
#       s(m) = R * cos(pi * m / NCRYSTALS),
#
# where m is the crystal-index separation and R = scanner radius. Equal steps in
# `ir` are NOT equal steps in s (bins are widest at the center, bunch toward the
# edge). Algorithms that work in the Fourier domain of s -- FORE, and analytic
# FBP -- assume UNIFORM s sampling, so the sinogram must be resampled onto a
# uniform s grid first.
#
# Pipeline ordering: arc_correct consumes the per-segment list, so it always
# runs BEFORE rebinning (ssrb/fore call it internally by default). FORE needs
# uniform s for its Fourier step; SSRB is radial-agnostic but its output still
# feeds analytic FBP, which also needs uniform s -- so both correct first.
# For an iterative reconstructor that models arc geometry, skip it
# (arc_correction=False).
#
# CAVEAT: the exact ir<->s correspondence (the half-bin offset from the +NRAD/2
# centering, and the sign convention) is derived analytically from the kernel's
# crystal indexing and is NOT yet verified against a point-source simulation.
# Treat quantitative results as provisional until that calibration is done.


def radial_bin_positions_mm(config):
    """Physical perpendicular distance s (mm) of each radial bin from the axis,
    under the analytic arc mapping s = R*cos(pi*m/NCRYSTALS).

    Returns an array of length NRAD giving the (signed) s of each stored radial
    bin ir = 0 .. NRAD-1. The center bin (ir = NRAD//2) maps to s = 0; bins on
    either side map to increasing |s| out toward the edge of the FOV. The
    spacing is nonuniform (densest near the edge).
    """
    s = config["sinogram"]
    sc = config["scanner"]
    nrad = s["num_radial_bins"]
    ncryst = sc["num_detectors_per_ring"]
    nang = s["num_angular_bins"]
    R = sc["radius_mm"]

    ir = np.arange(nrad)
    # ir is centered by +NRAD/2 in the kernel; undo to get signed radial index r.
    r = ir - nrad // 2
    # r is the departure (in crystal-index units) from the diametric pair, which
    # has angular separation m = NANGLES - r. s = R*cos(pi*m/NCRYSTALS), carrying
    # the sign of r so the two halves of the sinogram are distinguished.
    m = nang - np.abs(r)
    s_abs = R * np.abs(np.cos(np.pi * m / ncryst))
    return np.sign(r) * s_abs


def _arc_resample_map(config):
    """Precompute the linear-interpolation map from the nonuniform arc grid onto
    a uniform s grid. Returns (lo, hi, w, s_uniform):

        out[..., j] = w[j]*src[..., lo[j]] + (1-w[j])*src[..., hi[j]]

    computed ONCE (independent of plane/angle/segment), so the actual resampling
    is pure vectorized indexing over the radial axis.
    """
    s_src = radial_bin_positions_mm(config)          # (NRAD,) nonuniform, signed
    nrad = s_src.size

    # Source must be monotonically increasing for np.interp-style bracketing.
    order = np.argsort(s_src)
    s_sorted = s_src[order]

    # Uniform target grid spanning the same radial extent, same bin count.
    s_uniform = np.linspace(s_sorted[0], s_sorted[-1], nrad)

    # For each target position, find the bracketing pair in the sorted source.
    hi_in_sorted = np.searchsorted(s_sorted, s_uniform, side="left")
    hi_in_sorted = np.clip(hi_in_sorted, 1, nrad - 1)
    lo_in_sorted = hi_in_sorted - 1

    s_lo = s_sorted[lo_in_sorted]
    s_hi = s_sorted[hi_in_sorted]
    denom = (s_hi - s_lo)
    denom[denom == 0] = 1.0
    # weight on the LOW sample
    w = (s_hi - s_uniform) / denom

    # Map sorted indices back to original (unsorted) radial bin indices.
    lo = order[lo_in_sorted]
    hi = order[hi_in_sorted]
    return lo, hi, w, s_uniform


def arc_correct(segments, config, in_place=False):
    """Resample a per-segment sinogram list from raw arc coordinates onto a
    UNIFORM radial (s) grid.

    Parameters
    ----------
    segments : list of per-segment dicts from
               mcgpu_backend.data_reader.read_sinogram_segments. Each dict's
               "data" is (n_planes, NANGLES, NRAD) in arc coordinates.
    config   : run config (geometry).
    in_place : if False (default) return new dicts with resampled "data" and the
               other keys copied; if True, overwrite each dict's "data".

    Returns
    -------
    list of per-segment dicts, same structure, with "data" resampled onto the
    uniform s grid. A "s_uniform_mm" key (length NRAD) is added giving the
    physical radial position of each output bin.

    Speed
    -----
    The interpolation map (lo, hi, w) is computed ONCE from the geometry and
    applied to every segment's whole (n_planes, NANGLES, NRAD) block with a
    single vectorized fancy-index along the radial axis -- no per-row Python
    loop. This is O(total voxels) with NumPy-level constants.

    Must run BEFORE fore() (FORE assumes uniform s) and before analytic FBP.
    Do not use ahead of a PET-specific iterative reconstructor that models the
    arc geometry itself.
    """
    lo, hi, w, s_uniform = _arc_resample_map(config)

    out = []
    for seg in segments:
        data = seg["data"]                       # (n_planes, NANGLES, NRAD)
        # Vectorized linear interpolation along the last (radial) axis:
        resampled = w * data[..., lo] + (1.0 - w) * data[..., hi]
        if in_place:
            seg["data"] = resampled
            seg["s_uniform_mm"] = s_uniform
            out.append(seg)
        else:
            new = dict(seg)
            new["data"] = resampled
            new["s_uniform_mm"] = s_uniform
            out.append(new)
    return out


# ----------------------------------------------------------------------------
# SSRB
# ----------------------------------------------------------------------------

def ssrb(segments, config, arc_correction=True) -> np.ndarray:
    """Single-Slice Rebinning.

    Parameters
    ----------
    segments : list of per-segment dicts from
               mcgpu_backend.data_reader.read_sinogram_segments.
    config   : the run config (for the number of direct slices).
    arc_correction : if True (default), resample the segments onto a uniform
               radial grid (arc_correct) before rebinning, so the output is
               ready for analytic FBP. SSRB itself is radial-agnostic, so this
               does not change the axial bookkeeping; it only fixes the radial
               axis. Set False if the segments are already arc-corrected (to
               avoid resampling twice) or if you will feed an iterative
               reconstructor that models the arc geometry itself.

    Returns
    -------
    ndarray (NSLICES, NANGLES, NRAD): a stack of 2D direct sinograms, where
    NSLICES = 2*num_rings - 1 indexed by (iz1+iz2). Each oblique plane is added
    into the slice at its axial midpoint (which, in (iz1+iz2) coordinates, is
    simply (iz1+iz2) -- the same index the plane already carries).

    Because the planes within a segment are stored in increasing (iz1+iz2)
    order from axial_sum_min, plane p of a segment maps to direct-slice
    (axial_sum_min + p).
    """
    if arc_correction:
        segments = arc_correct(segments, config)
    nslices = _num_direct_slices(config)
    # infer (NANGLES, NRAD) from the first segment
    _, nang, nrad = segments[0]["data"].shape
    out = np.zeros((nslices, nang, nrad), dtype=np.float64)
    counts = np.zeros(nslices, dtype=np.int64)  # how many planes hit each slice

    for seg in segments:
        data = seg["data"]                       # (n_planes, NANGLES, NRAD)
        base = seg["axial_sum_min"]
        for p in range(data.shape[0]):
            slice_idx = base + p                 # = (iz1+iz2) for this plane
            if 0 <= slice_idx < nslices:
                out[slice_idx] += data[p]
                counts[slice_idx] += 1

    return out


# ----------------------------------------------------------------------------
# FORE
# ----------------------------------------------------------------------------

def fore(segments, config, omega_min=1e-6, klim=None, arc_correction=True) -> np.ndarray:
    """Fourier Rebinning (vectorized).

    Implements the frequency-distance relation: a 2D Fourier component of an
    oblique sinogram at angular harmonic k and radial frequency omega belongs,
    to first order, to a direct sinogram shifted axially by

        dz = - (k / omega) * tan(theta) / ring_spacing      [in slice units]

    where theta is the segment's polar angle. Components with omega == 0 or
    |omega| < omega_min (and optionally |k| > klim) fall back to SSRB-like
    assignment, since the relation is undefined / unreliable at very low radial
    frequency.

    This version is vectorized: per segment it builds the dz(k, omega) target-
    slice map ONCE for the whole Fourier plane, then for each plane scatters the
    entire 2D FFT into the accumulator with np.add.at -- no Python loop over
    individual Fourier bins. It also tracks, per Fourier bin, how many oblique
    contributions landed in each direct slice, and normalizes by that count
    (the averaging that standard FORE requires where segments overlap).

    Parameters
    ----------
    segments  : per-segment dicts from read_sinogram_segments.
    config    : run config (geometry).
    omega_min : radial-frequency magnitude below which FORE falls back to SSRB
                (default 1e-6). The DC component (omega == 0) always falls back.
    klim      : optional cap on |k| (angular harmonic) handled by FORE.
    arc_correction : if True (default), resample the segments onto a uniform
                radial grid (arc_correct) BEFORE the Fourier step. FORE works in
                the Fourier domain of the radial coordinate s and therefore
                REQUIRES uniform s sampling; on raw arc-coordinate data the
                frequency-distance relation uses a warped frequency axis and the
                rebinning is biased. Set False only if the segments are already
                arc-corrected (to avoid resampling twice).

    Returns
    -------
    ndarray (NSLICES, NANGLES, NRAD): direct sinogram stack, NSLICES =
    2*num_rings - 1, axial index = (iz1+iz2).

    Notes
    -----
    Convenience implementation. It uses nearest-slice assignment (round), not
    linear interpolation, and independent (omega_min, klim) cutoffs rather than
    the full Defrise consistency wedge. For production scatter work a tested
    implementation (e.g. STIR's FORE) is recommended.
    """
    if arc_correction:
        segments = arc_correct(segments, config)
    nslices = _num_direct_slices(config)
    _, nang, nrad = segments[0]["data"].shape
    ring_spacing = _ring_spacing_mm(config)
    ring_radius = config["scanner"]["radius_mm"]
    if klim is None:
        klim = nang  # effectively no cap

    # Fourier-space accumulator and per-bin contribution counter.
    acc = np.zeros((nslices, nang, nrad), dtype=np.complex128)
    cnt = np.zeros((nslices, nang, nrad), dtype=np.float64)

    # Frequency axes, broadcast to the full (NANGLES, NRAD) Fourier grid.
    omega = np.fft.fftfreq(nrad)              # (NRAD,)  cycles per radial bin
    kvec = np.fft.fftfreq(nang) * nang        # (NANGLES,) angular harmonics
    K = kvec[:, None]                         # (NANGLES, 1)
    W = omega[None, :]                        # (1, NRAD)
    # Fixed angular/radial bin indices over the whole grid (for scatter).
    ik_grid, iw_grid = np.meshgrid(np.arange(nang), np.arange(nrad),
                                   indexing="ij")  # both (NANGLES, NRAD)

    # Mask of Fourier bins that use the FORE shift vs the SSRB fallback. This
    # depends only on (k, omega), so it is the same for every segment/plane.
    fallback = (W == 0.0) | (np.abs(W) < omega_min) | (np.abs(K) > klim)

    for seg in segments:
        data = seg["data"].astype(np.float64)        # (n_planes, NANGLES, NRAD)
        base = seg["axial_sum_min"]
        rd = _segment_mean_ring_diff(seg)
        tan_theta = (rd * ring_spacing) / (2.0 * ring_radius)

        # dz(k, omega) for this segment, computed for the whole grid at once.
        with np.errstate(divide="ignore", invalid="ignore"):
            dz = -(K / W) * tan_theta / ring_spacing     # (NANGLES, NRAD)
        dz = np.where(fallback, 0.0, dz)                 # fallback -> no shift

        for p in range(data.shape[0]):
            slice_idx0 = base + p                        # SSRB target (iz1+iz2)
            tgt = np.round(slice_idx0 + dz).astype(np.int64)  # (NANGLES, NRAD)
            valid = (tgt >= 0) & (tgt < nslices)
            if not valid.any():
                continue
            F = np.fft.fft2(data[p])                     # (NANGLES, NRAD)
            # Scatter the whole Fourier plane into the accumulator in one shot.
            tv = tgt[valid]; iv = ik_grid[valid]; wv = iw_grid[valid]
            np.add.at(acc, (tv, iv, wv), F[valid])
            np.add.at(cnt, (tv, iv, wv), 1.0)

    # Normalize each Fourier bin by its contribution count (averaging overlaps),
    # then inverse-transform every direct plane back to a real sinogram.
    nz = cnt > 0
    acc[nz] /= cnt[nz]
    out = np.zeros((nslices, nang, nrad), dtype=np.float64)
    for z in range(nslices):
        if np.any(acc[z]):
            out[z] = np.real(np.fft.ifft2(acc[z]))
    return out