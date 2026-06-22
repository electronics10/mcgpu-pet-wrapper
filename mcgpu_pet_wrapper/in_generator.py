"""
Generate an MCGPU-PET.in file from a validated config. The .in is a thin
projection of the config: section headers and comments are preserved verbatim
(the parser locates sections by exact header strings), only value lines change.

The activity-per-material table in SOURCE PET SCAN is intentionally left as the
sentinel "1 0.0 / 0 0.0": the template itself documents that this input is NOT
used -- activity is read from the .vox third column. So the voxel grid owns the
spatial activity distribution; the .in owns physics + geometry; no overlap.

Field paths follow the four-section config schema (voxel_space, scanner,
sinogram, mcgpu). The axis-order-aware accessors from config.py (num_voxels) are
used so this module never indexes a raw [0]/[1]/[2] triple.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_config, validate_config, num_voxels


# Schema: readable name -> (section header, value-line index within section).
# Value-line index counts non-empty, non-comment lines, resetting per section.
# These addresses are into template.in; the values are in MCGPU native units
# (cm, eV), with unit conversion done in _translate.
SCHEMA: dict[str, tuple[str, int]] = {
    "random_seed":           ("SIMULATION CONFIG", 0),
    "gpu_number":            ("SIMULATION CONFIG", 1),
    "gpu_threads_per_block": ("SIMULATION CONFIG", 2),
    "density_scale_factor":  ("SIMULATION CONFIG", 3),

    "acquisition_time":      ("SOURCE PET SCAN", 0),
    "isotope_mean_life":     ("SOURCE PET SCAN", 1),
    # value-lines 2,3 are the material-activity sentinel; left untouched.

    "psf_filename":          ("PHASE SPACE FILE", 0),
    "detector_geometry":     ("PHASE SPACE FILE", 1),  # "X Y Z H RADIUS" cm
    "psf_max_elements":      ("PHASE SPACE FILE", 2),
    "report_trues_scatter":  ("PHASE SPACE FILE", 3),
    "report_psf_sinogram":   ("PHASE SPACE FILE", 4),

    "tally_material_dose":   ("DOSE DEPOSITION", 0),
    "tally_voxel_dose":      ("DOSE DEPOSITION", 1),
    "dose_filename":         ("DOSE DEPOSITION", 2),
    "dose_roi_x":            ("DOSE DEPOSITION", 3),
    "dose_roi_y":            ("DOSE DEPOSITION", 4),
    "dose_roi_z":            ("DOSE DEPOSITION", 5),

    "energy_resolution":     ("ENERGY PARAMETERS", 0),
    "energy_window_low":     ("ENERGY PARAMETERS", 1),
    "energy_window_high":    ("ENERGY PARAMETERS", 2),

    "axial_fov_cm":          ("SINOGRAM PARAMETERS", 0),
    "num_rings":             ("SINOGRAM PARAMETERS", 1),
    "total_crystals":        ("SINOGRAM PARAMETERS", 2),
    "num_angular_bins":      ("SINOGRAM PARAMETERS", 3),
    "num_radial_bins":       ("SINOGRAM PARAMETERS", 4),
    "num_z_slices":          ("SINOGRAM PARAMETERS", 5),
    "image_resolution":      ("SINOGRAM PARAMETERS", 6),
    "num_energy_bins":       ("SINOGRAM PARAMETERS", 7),
    "max_ring_difference":   ("SINOGRAM PARAMETERS", 8),
    "span":                  ("SINOGRAM PARAMETERS", 9),

    "voxel_space_file":      ("VOXELIZED GEOMETRY FILE", 0),

    "material1_file":        ("MATERIAL FILE LIST", 0),
    "material2_file":        ("MATERIAL FILE LIST", 1),
}

_SECTION_RE = re.compile(r'^\s*#\[SECTION (.+?)\]')
_VALUE_COLUMN = 31  # cosmetic comment alignment


class InFileGenerator:
    def __init__(self, template_path: str | Path = None):
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "template.in"
        self.lines = Path(template_path).read_text().splitlines()
        self._index = self._build_index()

    def _build_index(self) -> dict[tuple[str, int], int]:
        index: dict[tuple[str, int], int] = {}
        section: str | None = None
        n = 0
        for i, line in enumerate(self.lines):
            m = _SECTION_RE.match(line)
            if m:
                section = m.group(1).split(' v.')[0].strip()
                n = 0
                continue
            stripped = line.strip()
            if section and stripped and not stripped.startswith('#'):
                index[(section, n)] = i
                n += 1
        return index

    def _set_addr(self, section: str, line_no: int, value: Any) -> None:
        try:
            idx = self._index[(section, line_no)]
        except KeyError:
            raise KeyError(
                f"({section!r}, {line_no}) not in template. Check header text "
                f"and value-line count."
            )
        old = self.lines[idx]
        cpos = old.find('#')
        comment = old[cpos:] if cpos != -1 else ''
        leading = old[: len(old) - len(old.lstrip())]
        val_str = f"{leading}{value}"
        pad = max(_VALUE_COLUMN - len(val_str), 1)
        self.lines[idx] = f"{val_str}{' ' * pad}{comment}".rstrip()

    def apply(self, flat: dict[str, Any]) -> None:
        for key, value in flat.items():
            if key not in SCHEMA:
                raise KeyError(f"Unknown schema field {key!r}.")
            section, line_no = SCHEMA[key]
            self._set_addr(section, line_no, value)

    def from_config(self, config: dict) -> None:
        validate_config(config)
        self.apply(self._translate(config))

    @staticmethod
    def _translate(config: dict) -> dict[str, Any]:
        """Config (mm, s, eV) -> flat schema dict in MCGPU units (cm, s, eV)."""
        sc = config["scanner"]
        sg = config["sinogram"]
        m = config["mcgpu"]

        radius_cm = sc["radius_mm"] / 10.0
        axial_fov_cm = sc["axial_fov_mm"] / 10.0
        # Detector: X Y Z H RADIUS in cm; negative RADIUS => center on the voxel
        # grid bounding box (MCGPU convention).
        detector_geom = f"0.0 0.0 0.0 {axial_fov_cm} -{radius_cm}"

        # Dose ROI: only meaningful if voxel-dose tally is on. We set it to the
        # full voxel-grid extent so it can never reference voxels that don't
        # exist. num_voxels honors voxel_space.axis_order.
        nx, ny, nz = num_voxels(config)

        # density_scale_factor: dropped from the schema; the .in line still needs
        # a value. Default 1.0 (no scaling).
        density_scale = m.get("density_scale_factor", 1.0)

        # image_resolution (RES): dead code in current MCGPU-PET (see notes), but
        # the .in parser still reads the line. Default to the transverse voxel
        # count so the line is self-documenting; the binary ignores it.
        image_resolution = m.get("image_resolution", nx)

        return {
            "random_seed":           m["random_seed"],
            "gpu_number":            m["gpu_number"],
            "gpu_threads_per_block": m["gpu_threads_per_block"],
            "density_scale_factor":  density_scale,

            "acquisition_time":      m["acquisition_time_s"],
            "isotope_mean_life":     m["isotope_mean_life_s"],

            "psf_filename":          m["psf_filename"],
            "detector_geometry":     detector_geom,
            "psf_max_elements":      m["psf_max_elements"],
            "report_trues_scatter":  m["report_trues_scatter"],
            "report_psf_sinogram":   m["report_psf_sinogram"],

            "tally_material_dose":   m["tally_material_dose"],
            "tally_voxel_dose":      m["tally_voxel_dose"],
            "dose_filename":         m["dose_filename"],
            "dose_roi_x":            f"1 {nx}",
            "dose_roi_y":            f"1 {ny}",
            "dose_roi_z":            f"1 {nz}",

            "energy_resolution":     m["energy_resolution"],
            "energy_window_low":     m["energy_window_low_eV"],
            "energy_window_high":    m["energy_window_high_eV"],

            "axial_fov_cm":          axial_fov_cm,
            "num_rings":             sc["num_rings"],
            "total_crystals":        sc["num_detectors_per_ring"],
            "num_angular_bins":      sg["num_angular_bins"],
            "num_radial_bins":       sg["num_radial_bins"],
            "num_z_slices":          sg["num_axial_planes"],
            "image_resolution":      image_resolution,
            "num_energy_bins":       m["num_energy_bins"],
            "max_ring_difference":   sg["max_ring_difference"],
            "span":                  sg["span"],

            "voxel_space_file":      m["voxel_space_file"],

            "material1_file":        m["materials"][0],
            "material2_file":        m["materials"][1],
        }

    def write(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "MCGPU-PET.in"
        out.write_text('\n'.join(self.lines) + '\n')
        return out

    # convenience re-export so callers can do InFileGenerator.load_config(...)
    load_config = staticmethod(load_config)