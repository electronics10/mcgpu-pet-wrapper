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
    return sc["axial_length_mm"] / sc["num_rings"]


def _num_direct_slices(config):
    # axial positions indexed by (iz1+iz2): 0 .. 2*(num_rings-1)
    return 2 * config["scanner"]["num_rings"] - 1


def _segment_mean_ring_diff(seg):
    return 0.5 * (seg["ring_diff_min"] + seg["ring_diff_max"])


# ----------------------------------------------------------------------------
# SSRB
# ----------------------------------------------------------------------------

def ssrb(segments, config) -> np.ndarray:
    """Single-Slice Rebinning.

    Parameters
    ----------
    segments : list of per-segment dicts from
               mcgpu_backend.data_reader.read_sinogram_segments.
    config   : the run config (for the number of direct slices).

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

def fore(segments, config, omega_min=1e-6, klim=None) -> np.ndarray:
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