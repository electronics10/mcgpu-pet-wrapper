"""
Factory functions: config -> VoxelGrid. A "phantom" is a standard test object
with known geometry; these factories build common ones. Geometry (voxel count
and size) is read FROM the config, so the grid cannot disagree with the
simulation geometry by construction.

These factories return a VoxelGrid (the general data structure); they live here,
separate from voxel_grid.py, because they encode *specific* standardized objects
rather than the general grid machinery.

Materials are passed explicitly as role -> (material_id, density_g_per_cm3).
There is no hidden assumption about which cross-section files exist. PMMA, if you
lack a real cross-section file, is faked as the water material ID at PMMA density
(1.19); at 511 keV Compton scattering dominates and the electron-density gap is
about 1%.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .config import voxel_space_shape_xyz, grid_size_mm
from .voxel_grid import VoxelGrid, VoxelSpaceBuilder

# NEMA NU 4-2008 Image Quality phantom dimensions (mm).
_IQ_BODY_DIAMETER = 30.0
_IQ_BODY_LENGTH = 50.0
_IQ_UNIFORM_LENGTH = 30.0
_IQ_ROD_REGION_LENGTH = 20.0
_IQ_COLD_INSERT_DIAMETER = 8.0
_IQ_COLD_INSERT_LENGTH = 15.0
_IQ_COLD_INSERT_SEPARATION = 15.0
_IQ_ROD_DIAMETERS = (1.0, 2.0, 3.0, 4.0, 5.0)
_IQ_ROD_PITCH = 7.0
_IQ_WALL_THICKNESS = 3.0

# Default material roles. air=1, water=2; pmma reuses the water cross-sections
# at PMMA density (see module docstring).
_DEFAULT_MATERIALS = {
    "air":   (1, 0.00120479),
    "water": (2, 1.0),
    "pmma":  (2, 1.19),
}


def _require_roles(materials, roles):
    if materials is None:
        materials = dict(_DEFAULT_MATERIALS)
    for role in roles:
        if role not in materials:
            raise ValueError(f"materials missing required role: {role!r}")
    return materials


def nema_iq_preclinical(
    config: dict,
    hot_activity_Bq_per_mL: float = 3700.0,
    materials: Dict[str, Tuple[int, float]] | None = None,
) -> VoxelGrid:
    """NEMA NU 4-2008 Image Quality phantom, sized from the config.

    The voxel count and size come from config['voxel_space'] (via the config
    accessors), so the grid always matches the simulation geometry. The phantom
    is centered transversely and axially in the voxel space bounding box, so the
    50 mm body sits in the middle of the axial extent.

    Geometry (standard NEMA NU 4-2008): a 30 mm-diameter, 50 mm-long PMMA
    cylinder; a rod region (five hot rods of diameter 1-5 mm on a 7 mm pitch
    circle) at one axial end; a uniform hot region in the middle; two cold
    inserts (one water, one air) at the other end.

    materials roles required: 'air', 'water', 'pmma'.
    """
    materials = _require_roles(materials, ("air", "water", "pmma"))
    air_id, air_rho = materials["air"]
    water_id, water_rho = materials["water"]
    pmma_id, pmma_rho = materials["pmma"]

    shape_xyz = voxel_space_shape_xyz(config)            # (Nx, Ny, Nz)
    voxel_size = grid_size_mm(config)               # (dx, dy, dz)
    nx, ny, nz = shape_xyz
    dx, dy, dz = voxel_size

    # Fit checks (the wrapper doesn't police placement, but a phantom that can't
    # physically fit the grid is a construction error worth catching here).
    transverse_needed = _IQ_BODY_DIAMETER + 2 * _IQ_WALL_THICKNESS
    if nx * dx < transverse_needed or ny * dy < transverse_needed:
        raise ValueError(
            f"voxel space transverse {nx*dx:.1f}x{ny*dy:.1f} mm too small for the "
            f"NEMA IQ phantom (needs >= {transverse_needed:.1f} mm each side)."
        )
    if nz * dz < _IQ_BODY_LENGTH:
        raise ValueError(
            f"voxel space axial {nz*dz:.1f} mm too small for the {_IQ_BODY_LENGTH} "
            f"mm NEMA IQ phantom."
        )

    b = VoxelSpaceBuilder(shape_xyz, voxel_size,
                          material_names=["air", "water/pmma"])
    cx, cy, cz = b.bbox_center_mm

    # Axial layout: center the 50 mm body in the voxel space axial extent.
    z0 = cz - _IQ_BODY_LENGTH / 2.0          # bottom face of the body
    z_body_top = z0 + _IQ_BODY_LENGTH
    body_cz = z0 + _IQ_BODY_LENGTH / 2.0
    rod_center_z = z0 + _IQ_ROD_REGION_LENGTH / 2.0

    inner_r = _IQ_BODY_DIAMETER / 2.0
    outer_r = inner_r + _IQ_WALL_THICKNESS

    # 1. air background
    b.fill_background(air_id, air_rho, 0.0)
    # 2. PMMA shell (full body footprint)
    b.add_cylinder((cx, cy, body_cz), outer_r, _IQ_BODY_LENGTH, "z",
                   pmma_id, pmma_rho, 0.0)
    # 3. hot fill through the whole inner cavity
    b.add_cylinder((cx, cy, body_cz), inner_r, _IQ_BODY_LENGTH, "z",
                   water_id, water_rho, hot_activity_Bq_per_mL)
    # 4. rod region: re-fill with PMMA, then paint the 5 hot rods back in
    b.add_cylinder((cx, cy, rod_center_z), inner_r, _IQ_ROD_REGION_LENGTH, "z",
                   pmma_id, pmma_rho, 0.0)
    for k, diam in enumerate(_IQ_ROD_DIAMETERS):
        ang = 2 * math.pi * k / len(_IQ_ROD_DIAMETERS)
        rx = cx + _IQ_ROD_PITCH * math.cos(ang)
        ry = cy + _IQ_ROD_PITCH * math.sin(ang)
        b.add_cylinder((rx, ry, rod_center_z), diam / 2.0, _IQ_ROD_REGION_LENGTH,
                       "z", water_id, water_rho, hot_activity_Bq_per_mL)
    # 5. cold inserts (water + air) flush with the top face of the body
    insert_cz = z_body_top - _IQ_COLD_INSERT_LENGTH / 2.0
    half_sep = _IQ_COLD_INSERT_SEPARATION / 2.0
    b.add_cylinder((cx + half_sep, cy, insert_cz), _IQ_COLD_INSERT_DIAMETER / 2.0,
                   _IQ_COLD_INSERT_LENGTH, "z", water_id, water_rho, 0.0)
    b.add_cylinder((cx - half_sep, cy, insert_cz), _IQ_COLD_INSERT_DIAMETER / 2.0,
                   _IQ_COLD_INSERT_LENGTH, "z", air_id, air_rho, 0.0)

    return b.build()


def uniform_cylinder(
    config: dict,
    diameter_mm: float,
    length_mm: float,
    activity_Bq_per_mL: float,
    materials: Dict[str, Tuple[int, float]] | None = None,
) -> VoxelGrid:
    """A uniform fillable cylinder centered in the voxel space (axis = z).

    Useful as a NEMA scatter/sensitivity phantom or a normalization source.
    materials roles required: 'air', 'water'.
    """
    materials = _require_roles(materials, ("air", "water"))
    air_id, air_rho = materials["air"]
    water_id, water_rho = materials["water"]

    shape_xyz = voxel_space_shape_xyz(config)
    voxel_size = grid_size_mm(config)
    nx, ny, nz = shape_xyz
    dx, dy, dz = voxel_size

    if nx * dx < diameter_mm or ny * dy < diameter_mm:
        raise ValueError(
            f"voxel space transverse {nx*dx:.1f}x{ny*dy:.1f} mm too small for a "
            f"{diameter_mm:.1f} mm cylinder."
        )
    if nz * dz < length_mm:
        raise ValueError(
            f"voxel space axial {nz*dz:.1f} mm too small for a {length_mm:.1f} mm "
            f"cylinder."
        )

    b = VoxelSpaceBuilder(shape_xyz, voxel_size, material_names=["air", "water"])
    cx, cy, cz = b.bbox_center_mm
    b.fill_background(air_id, air_rho, 0.0)
    b.add_cylinder((cx, cy, cz), diameter_mm / 2.0, length_mm, "z",
                   water_id, water_rho, activity_Bq_per_mL)
    return b.build()


def point_source(
    config: dict,
    activity_Bq_per_mL: float = 1.0e6,
    radius_mm: float = 1.0,
    offset_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    materials: Dict[str, Tuple[int, float]] | None = None,
) -> VoxelGrid:
    """A small hot sphere in air, centered in the voxel space (plus optional
    offset). Useful for point-spread-function / resolution checks.

    materials roles required: 'air', 'water' (the sphere is water-filled).
    """
    materials = _require_roles(materials, ("air", "water"))
    air_id, air_rho = materials["air"]
    water_id, water_rho = materials["water"]

    shape_xyz = voxel_space_shape_xyz(config)
    voxel_size = grid_size_mm(config)

    b = VoxelSpaceBuilder(shape_xyz, voxel_size, material_names=["air", "water"])
    cx, cy, cz = b.bbox_center_mm
    ox, oy, oz = offset_mm
    b.fill_background(air_id, air_rho, 0.0)
    b.add_sphere((cx + ox, cy + oy, cz + oz), radius_mm, water_id, water_rho,
                 activity_Bq_per_mL)
    return b.build()