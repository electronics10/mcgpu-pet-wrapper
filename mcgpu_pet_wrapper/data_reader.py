"""
Read MCGPU-PET's raw OUTPUT files into correctly-shaped numpy arrays.

(For reading/writing the INPUT voxel geometry, see vox_io.py -- that owns the
VoxelGrid <-> .vox round-trip. This module only parses simulator outputs.)

Outputs:
  sinogram_Trues.raw.gz / sinogram_Scatter.raw.gz
      flat int32 of length NBINS. The kernel's per-event flat index is
          ibin = izm*(NANGLES*NRAD) + ith*NRAD + ir
      so the C-order reshape is (NSINOS, NANGLES, NRAD): plane slowest, angular
      middle, radial fastest. NSINOS/NBINS come from config.sinogram_shape,
      which replays the binary's own arithmetic.

  image_Trues.raw.gz / image_Scatter.raw.gz
      flat int32 of length Nx*Ny*Nz, reshaping to (Nz, Ny, Nx) C-order -- the
      same shape as the voxel grid (these are per-emitting-voxel coincidence
      counts, NOT reconstructions).

The stored sinogram concatenates all michelogram segments along the NSINOS axis.
read_sinogram returns that full concatenated block; read_sinogram_segments
splits it into a list of per-segment 3D arrays (using config.segment_table), so
each segment's ring-difference and axial mapping are known -- the form rebinning
(SSRB/FORE) actually consumes.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .config import sinogram_shape, segment_table, num_voxels


def read_sinogram(path, config: dict) -> np.ndarray:
    """Read a sinogram .raw.gz into shape (NSINOS, NANGLES, NRAD) int32."""
    info = sinogram_shape(config)
    raw = _read_int32(path)
    if raw.size != info["NBINS"]:
        raise ValueError(
            f"{path}: got {raw.size} int32 values but config implies NBINS="
            f"{info['NBINS']} (shape {info['shape_csr']}). Geometry/config mismatch."
        )
    return raw.reshape(info["shape_csr"], order="C")


def read_sinogram_segments(path, config: dict) -> list[dict]:
    """Read a sinogram and split it into per-segment 3D arrays.

    Returns a list (in storage order) of dicts:
        {
          "segment_number": signed michelogram segment (0, -1, +1, ...),
          "ring_diff_min", "ring_diff_max": the segment's ring-difference band,
          "axial_sum_min", "axial_sum_max": (iz1+iz2) range; midpoint = sum/2,
          "data": ndarray (n_planes, NANGLES, NRAD),
        }

    This is usually more useful than the flat concatenated block: each segment
    has a known polar angle (from ring difference) and axial mapping, which is
    exactly what SSRB/FORE need.
    """
    full = read_sinogram(path, config)         # (NSINOS, NANGLES, NRAD)
    segs = segment_table(config)
    out = []
    for seg in segs:
        block = full[seg.start_plane: seg.start_plane + seg.n_planes]
        out.append({
            "segment_number": seg.segment_number,
            "ring_diff_min": seg.ring_diff_min,
            "ring_diff_max": seg.ring_diff_max,
            "axial_sum_min": seg.axial_sum_min,
            "axial_sum_max": seg.axial_sum_max,
            "data": block,
        })
    return out


def read_emission_image(path, config: dict) -> np.ndarray:
    """Read an emission image .raw.gz into shape (Nz, Ny, Nx) int32 -- the same
    shape as the voxel grid. Uses the axis-order-aware num_voxels accessor."""
    nx, ny, nz = num_voxels(config)
    raw = _read_int32(path)
    expected = nx * ny * nz
    if raw.size != expected:
        raise ValueError(
            f"{path}: got {raw.size} int32 values but num_voxels implies "
            f"{expected} ((Nz,Ny,Nx)=({nz},{ny},{nx}))."
        )
    return raw.reshape((nz, ny, nx), order="C")


def summarize_sinogram(path, config: dict) -> dict:
    """Quick stats for a sinogram file (dataset sanity checks)."""
    s = read_sinogram(path, config)
    return {
        "shape": s.shape,
        "total_counts": int(s.sum()),
        "max_bin": int(s.max()),
        "nonzero_bins": int((s > 0).sum()),
        "fraction_nonzero": float((s > 0).mean()),
    }


def _read_int32(path) -> np.ndarray:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.int32)