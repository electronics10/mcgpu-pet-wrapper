"""
Serialize the voxel space to MCGPU-PET's .vox format, and read one back.

Format: penEasy 2008 voxel geometry + a third column for activity (Bq/voxel).
Header declares voxel counts and sizes; body is one line per voxel in x-fastest
order (C-order ravel of the (Nz,Ny,Nx) arrays). Blank inter-cycle lines are
optional and we omit them (BLANK LINES flag = 0).

MCGPU reads this through zlib, so .vox and .vox.gz are both accepted; the
filename in the .in must match the file on disk exactly.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .voxel_grid import VoxelGrid


class VoxFileGenerator:
    """Write the voxel space to .vox (optionally gzipped)."""

    def __init__(self, voxel_space: VoxelGrid):
        voxel_space.validate()
        self.voxel_space = voxel_space

    def _header(self) -> str:
        nx, ny, nz = self.voxel_space.shape_xyz
        dx_cm, dy_cm, dz_cm = (d / 10.0 for d in self.voxel_space.grid_size_mm)
        return "\n".join([
            "[SECTION VOXELS HEADER v.2008-04-13]",
            f"{nx} {ny} {nz}       No. OF VOXELS IN X,Y,Z",
            f"{dx_cm:.6g} {dy_cm:.6g} {dz_cm:.6g}       VOXEL SIZE (cm) ALONG X,Y,Z",
            "1                    COLUMN NUMBER WHERE MATERIAL ID IS LOCATED",
            "2                    COLUMN NUMBER WHERE THE MASS DENSITY [g/cm3] IS LOCATED",
            "0                    BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)",
            "[END OF VXH SECTION]",
            "",
        ])

    def _body(self) -> str:
        mat = self.voxel_space.material_id.ravel(order="C")
        rho = self.voxel_space.density.ravel(order="C")
        act = self.voxel_space.activity.ravel(order="C")
        parts = [
            f"{m} {d:.6g} {a:.6g}"
            for m, d, a in zip(mat.tolist(), rho.tolist(), act.tolist())
        ]
        return "\n".join(parts) + "\n"

    def write(self, run_dir, config) -> Path:
        """Write the grid to run_dir/filename. Compression is decided by the
        filename: a name ending in '.gz' is written gzipped, otherwise plain.
        This makes the filename the single source of truth, so the .in reference
        and the file on disk can never disagree about compression."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        filename = config["mcgpu"]["voxel_space_file"]
        out = run_dir / filename
        payload = self._header() + self._body()
        if filename.endswith(".gz"):
            with gzip.open(out, "wt") as f:
                f.write(payload)
        else:
            out.write_text(payload)
        return out


def read_vox(run_dir, config=None) -> VoxelGrid:
    """Parse a .vox (or .vox.gz) back into a voxel space. Used for round-trip tests
    and re-loading recorded runs."""
    run_dir = Path(run_dir)
    path = run_dir / config["mcgpu"]["voxel_space_file"]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        text = f.read()

    lines = text.split("\n")
    h0 = next(i for i, l in enumerate(lines) if l.startswith("[SECTION VOXELS HEADER"))
    nx, ny, nz = (int(x) for x in lines[h0 + 1].split()[:3])
    dx, dy, dz = (float(x) * 10.0 for x in lines[h0 + 2].split()[:3])  # cm -> mm
    end = next(i for i, l in enumerate(lines) if l.startswith("[END OF VXH SECTION]"))

    body = [l for l in lines[end + 1:] if l.strip()]
    n = nx * ny * nz
    if len(body) != n:
        raise ValueError(f"expected {n} voxel lines, found {len(body)}")

    mat = np.empty(n, dtype=np.uint8)
    rho = np.empty(n, dtype=np.float32)
    act = np.empty(n, dtype=np.float32)
    for idx, l in enumerate(body):
        p = l.split()
        mat[idx] = int(p[0]); rho[idx] = float(p[1]); act[idx] = float(p[2])

    mat_names = []
    if config is not None:
        mats = config["mcgpu"]["materials"]      # 1-based: id k -> mats[k-1]
        max_id = int(mat.max())
        if max_id > len(mats):
            raise ValueError(
                f"{path} references material id {max_id} but config lists only "
                f"{len(mats)} materials.")
        mat_names = list(mats)

    shape = (nz, ny, nx)
    return VoxelGrid(
        material_id=mat.reshape(shape, order="C"),
        density=rho.reshape(shape, order="C"),
        activity=act.reshape(shape, order="C"),
        grid_size_mm=(dx, dy, dz),
        material_names=mat_names
    )
