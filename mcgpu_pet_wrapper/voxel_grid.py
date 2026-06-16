"""
The voxel object: per-voxel material ID, density, and activity. This is the
single in-memory source of truth for the simulation source/object. It is a pure
value object -- it knows nothing about the scanner, serialization, or config.

  VoxelGrid          : the data container + derived quantities + self-validation.
  VoxelSpaceBuilder  : paint primitives (background, cylinder, sphere) into a
                     Phantom. Geometry comes from the caller (typically the
                     factory, which reads it from config).

Coordinate convention (matches MCGPU-PET / penEasy 2008):
  - Bounding box in the first octant; voxel (1,1,1) cornered at origin.
  - Array shape (Nz, Ny, Nx): arr[z] is an axial slice, arr[z, y, x] a voxel.
  - Voxel [k, j, i] center at ((i+0.5)dx, (j+0.5)dy, (k+0.5)dz) mm.
  - There is NO placement freedom: the object is always origin-cornered. To
    center an object in the FOV, paint it at the bbox center (builder helper).

Units: grid_size_mm in mm; density g/cm^3; activity stored as Bq/voxel.
The builder accepts activity as Bq/mL (physicist-native) and converts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ----------------------------------------------------------------------------
# Phantom: the data object
# ----------------------------------------------------------------------------

@dataclass
class VoxelGrid:
    """Voxelized object. All three arrays have shape (Nz, Ny, Nx)."""
    material_id: np.ndarray   # (Nz, Ny, Nx) uint8, values >= 1
    density:     np.ndarray   # (Nz, Ny, Nx) float32, g/cm^3
    activity:    np.ndarray   # (Nz, Ny, Nx) float32, Bq per voxel
    grid_size_mm: tuple[float, float, float]
    material_names: list[str] = field(default_factory=list)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(self.material_id.shape)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        nz, ny, nx = self.material_id.shape
        return (nx, ny, nz)

    @property
    def bbox_size_mm(self) -> tuple[float, float, float]:
        nx, ny, nz = self.shape_xyz
        dx, dy, dz = self.grid_size_mm
        return (nx * dx, ny * dy, nz * dz)

    @property
    def voxel_volume_mL(self) -> float:
        dx, dy, dz = self.grid_size_mm
        return (dx * dy * dz) / 1000.0  # mm^3 -> cm^3(=mL)

    @property
    def total_activity_Bq(self) -> float:
        return float(self.activity.sum())

    @property
    def total_volume_mL(self) -> float:
        nx, ny, nz = self.shape_xyz
        return (nx * ny * nz) * self.voxel_volume_mL

    @property
    def total_mass_g(self) -> float:
        # mass = sum_voxel(density[g/cm^3] * voxel_volume[cm^3]); units cancel to g.
        # Direct sum, NOT mean-density*volume (which hides the equal-volume assumption).
        return float(self.density.sum()) * self.voxel_volume_mL

    def validate(self) -> None:
        """Raise ValueError on any MCGPU-PET invariant violation."""
        s = self.material_id.shape
        if self.density.shape != s or self.activity.shape != s:
            raise ValueError(
                f"shape mismatch: material_id {s}, density {self.density.shape}, "
                f"activity {self.activity.shape}"
            )
        if self.material_id.min() < 1:
            n_bad = int((self.material_id < 1).sum())
            raise ValueError(
                f"{n_bad} voxels have material_id < 1 (vacuum not allowed in MCGPU)."
            )
        if (self.density < 0).any():
            raise ValueError("Negative densities not allowed.")
        if (self.activity < 0).any():
            raise ValueError("Negative activities not allowed.")
        if any(d <= 0 for d in self.grid_size_mm):
            raise ValueError(f"grid_size_mm must be positive: {self.grid_size_mm}")


# ----------------------------------------------------------------------------
# PhantomBuilder: paint primitives into a Phantom
# ----------------------------------------------------------------------------

class VoxelSpaceBuilder:
    """Paint primitives into an initially-empty volume.

    Coordinates in mm from the bbox origin (first octant). Use bbox_center_mm
    for the FOV center. Activity in Bq/mL, converted to Bq/voxel on paint.

    Paint semantics: later calls overwrite earlier at overlapping voxels. Order
    matters: background first, then large compartments, then small features.
    """

    def __init__(
        self,
        shape_xyz: tuple[int, int, int],
        grid_size_mm: tuple[float, float, float],
        material_names: list[str] | None = None,
    ):
        nx, ny, nz = shape_xyz
        if any(n <= 0 for n in (nx, ny, nz)):
            raise ValueError(f"shape must be positive: {shape_xyz}")
        if any(d <= 0 for d in grid_size_mm):
            raise ValueError(f"grid_size_mm must be positive: {grid_size_mm}")

        self.shape_xyz = (nx, ny, nz)
        self.grid_size_mm = tuple(grid_size_mm)
        self.material_names = list(material_names) if material_names else []

        self.material_id = np.zeros((nz, ny, nx), dtype=np.uint8)
        self.density     = np.zeros((nz, ny, nx), dtype=np.float32)
        self.activity    = np.zeros((nz, ny, nx), dtype=np.float32)

        dx, dy, dz = self.grid_size_mm
        self._x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
        self._y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
        self._z = (np.arange(nz, dtype=np.float64) + 0.5) * dz

    @property
    def bbox_size_mm(self) -> tuple[float, float, float]:
        nx, ny, nz = self.shape_xyz
        dx, dy, dz = self.grid_size_mm
        return (nx * dx, ny * dy, nz * dz)

    @property
    def bbox_center_mm(self) -> tuple[float, float, float]:
        bx, by, bz = self.bbox_size_mm
        return (bx / 2.0, by / 2.0, bz / 2.0)

    def _voxel_volume_mL(self) -> float:
        dx, dy, dz = self.grid_size_mm
        return (dx * dy * dz) / 1000.0

    def _paint(self, mask: np.ndarray, material_id: int, density: float,
               activity_Bq_per_mL: float) -> int:
        if not (1 <= material_id <= 255):
            raise ValueError(f"material_id must be in 1..255, got {material_id}")
        if density < 0:
            raise ValueError(f"density must be >= 0, got {density}")
        if activity_Bq_per_mL < 0:
            raise ValueError(f"activity must be >= 0, got {activity_Bq_per_mL}")
        self.material_id[mask] = material_id
        self.density[mask] = density
        self.activity[mask] = activity_Bq_per_mL * self._voxel_volume_mL()
        return int(mask.sum())

    def _bbox_slices(self, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
        nx, ny, nz = self.shape_xyz
        dx, dy, dz = self.grid_size_mm
        i0, i1 = max(0, int(np.floor(x_lo / dx))), min(nx, int(np.ceil(x_hi / dx)) + 1)
        j0, j1 = max(0, int(np.floor(y_lo / dy))), min(ny, int(np.ceil(y_hi / dy)) + 1)
        k0, k1 = max(0, int(np.floor(z_lo / dz))), min(nz, int(np.ceil(z_hi / dz)) + 1)
        return slice(k0, k1), slice(j0, j1), slice(i0, i1)

    def fill_background(self, material_id: int, density: float,
                        activity_Bq_per_mL: float = 0.0) -> int:
        """Fill the whole volume. Usually called first (e.g. air)."""
        return self._paint(np.ones_like(self.material_id, dtype=bool),
                            material_id, density, activity_Bq_per_mL)

    def add_cylinder(self, center_mm, radius_mm, height_mm,
                     axis: Literal["x", "y", "z"], material_id, density,
                     activity_Bq_per_mL: float = 0.0) -> int:
        """Axis-aligned cylinder. center_mm=(x,y,z), height along `axis`."""
        if radius_mm <= 0 or height_mm <= 0:
            raise ValueError("radius_mm and height_mm must be > 0")
        cx, cy, cz = center_mm
        half = height_mm / 2.0
        if axis == "z":
            bb = (cx - radius_mm, cx + radius_mm, cy - radius_mm, cy + radius_mm,
                  cz - half, cz + half)
        elif axis == "x":
            bb = (cx - half, cx + half, cy - radius_mm, cy + radius_mm,
                  cz - radius_mm, cz + radius_mm)
        elif axis == "y":
            bb = (cx - radius_mm, cx + radius_mm, cy - half, cy + half,
                  cz - radius_mm, cz + radius_mm)
        else:
            raise ValueError(f"axis must be 'x','y','z'; got {axis!r}")

        ks, js, is_ = self._bbox_slices(*bb)
        X = self._x[is_][None, None, :]
        Y = self._y[js][None, :, None]
        Z = self._z[ks][:, None, None]
        if axis == "z":
            cross = (X - cx) ** 2 + (Y - cy) ** 2 <= radius_mm ** 2
            within = np.abs(Z - cz) <= half
        elif axis == "x":
            cross = (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2
            within = np.abs(X - cx) <= half
        else:  # y
            cross = (X - cx) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2
            within = np.abs(Y - cy) <= half
        local = cross & within
        if not local.any():
            return 0
        full = np.zeros_like(self.material_id, dtype=bool)
        full[ks, js, is_] = local
        return self._paint(full, material_id, density, activity_Bq_per_mL)

    def add_sphere(self, center_mm, radius_mm, material_id, density,
                   activity_Bq_per_mL: float = 0.0) -> int:
        if radius_mm <= 0:
            raise ValueError("radius_mm must be > 0")
        cx, cy, cz = center_mm
        ks, js, is_ = self._bbox_slices(cx - radius_mm, cx + radius_mm,
                                        cy - radius_mm, cy + radius_mm,
                                        cz - radius_mm, cz + radius_mm)
        X = self._x[is_][None, None, :]
        Y = self._y[js][None, :, None]
        Z = self._z[ks][:, None, None]
        local = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2
        if not local.any():
            return 0
        full = np.zeros_like(self.material_id, dtype=bool)
        full[ks, js, is_] = local
        return self._paint(full, material_id, density, activity_Bq_per_mL)

    def build(self) -> VoxelGrid:
        vg = VoxelGrid(
            material_id=self.material_id.copy(),
            density=self.density.copy(),
            activity=self.activity.copy(),
            grid_size_mm=self.grid_size_mm,
            material_names=list(self.material_names),
        )
        vg.validate()
        return vg