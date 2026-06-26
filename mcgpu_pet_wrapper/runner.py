"""
Orchestrate a single MCGPU-PET run.

A run directory is self-contained and reproducible: it holds config.json,
MCGPU-PET.in, the voxel-space .vox, a symlink to materials/ and the binary, and
(after running) the outputs. You can re-run or inspect a directory later.

Pipeline (build_run does the staging; Runner.__call__ does the launch):
  1. record config.json into run_dir (the source of truth)
  2. InFileGenerator(config).write(run_dir)        -> MCGPU-PET.in
  3. VoxFileGenerator(voxel_space).write(run_dir)         -> <voxel_space_file>
  4. Runner()(run_dir)                             -> launch + collect outputs

The wrapper does not second-guess experiment design (whether the voxel_space fits the
scanner sensibly, etc.); that is the user's concern when building the VoxelGrid.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from .config import load_config, validate_config
from .in_generator import InFileGenerator
from .voxel_grid import VoxelGrid
from .vox_io import VoxFileGenerator

PKG_DIR = Path(__file__).parent


@dataclass
class RunResult:
    run_dir: Path
    sinogram_trues: Path
    sinogram_scatter: Path
    image_trues: Path
    image_scatter: Path
    energy_spectrum: Path
    log: Path
    wall_time_s: float
    returncode: int


def build_run(
    run_dir,
    config: dict,
    voxel_space: VoxelGrid,
) -> Path:
    """Stage a run directory from a config + voxel_space. Returns run_dir.

    Records config.json, writes MCGPU-PET.in and the .vox. Does NOT launch the
    simulation -- call Runner()(run_dir) afterward.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    validate_config(config)

    # Patch a copy so the recorded config.json matches what ran.
    cfg = json.loads(json.dumps(config))  # deep copy

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    gen = InFileGenerator()
    gen.from_config(cfg)
    gen.write(run_dir)

    VoxFileGenerator(voxel_space).write(run_dir, cfg)
    return run_dir


class Runner:
    """Launch MCGPU-PET.x in a staged run directory and collect outputs."""

    def __init__(self, binary=PKG_DIR / "MCGPU-PET.x",
                 materials=PKG_DIR / "materials"):
        self.binary = Path(binary)
        self.materials = Path(materials)

    def __call__(
        self,
        run_dir,
        on_existing: Literal["error", "overwrite", "skip"] = "error",
        verbose: bool = True,
        timeout_s: Optional[float] = 3600,
        line_callback: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

        existing = self._handle_existing(run_dir, on_existing)
        if existing is not None:
            return existing

        self._stage(run_dir)
        self._preflight(run_dir)
        rc, dt = self._execute(run_dir, verbose, timeout_s, line_callback)
        return self._collect(run_dir, rc, dt)

    def _stage(self, d: Path) -> None:
        for src, name in [(self.binary, "MCGPU-PET.x"),
                          (self.materials, "materials")]:
            link = d / name
            if not link.exists():
                link.symlink_to(src.resolve())

    def _preflight(self, d: Path) -> None:
        # The .vox filename lives in the recorded config.json; read it back so
        # we check for the exact file the .in references.
        cfg = load_config(d / "config.json")
        vox = cfg["mcgpu"]["voxel_space_file"]
        required = ["MCGPU-PET.x", "MCGPU-PET.in", "materials", vox]
        missing = [r for r in required if not (d / r).exists()]
        if missing:
            raise FileNotFoundError(f"Missing in {d}: {missing}")

    def _execute(self, d, verbose, timeout_s, line_callback):
        log = d / "MCGPU-PET.out"
        t0 = time.perf_counter()
        with open(log, "w") as f:
            p = subprocess.Popen(
                ["./MCGPU-PET.x", "MCGPU-PET.in"],
                cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            timer = threading.Timer(timeout_s, p.kill) if timeout_s else None
            if timer:
                timer.start()
            try:
                for line in p.stdout:
                    if verbose:
                        print(line, end="")
                    if line_callback:
                        line_callback(line)
                    f.write(line)
                p.wait()
            finally:
                if timer:
                    timer.cancel()
        dt = time.perf_counter() - t0
        if p.returncode < 0:
            raise RuntimeError(
                f"Run exceeded {timeout_s}s, killed (signal {-p.returncode})"
            )
        return p.returncode, dt

    def _collect(self, d: Path, rc: int, dt: float) -> RunResult:
        expected = [
            "sinogram_Trues.raw.gz", "sinogram_Scatter.raw.gz",
            "image_Trues.raw.gz", "image_Scatter.raw.gz",
            "Energy_Sinogram_Spectrum.dat",
        ]
        missing = [f for f in expected
                   if not (d / f).exists() or (d / f).stat().st_size == 0]
        if rc != 0 or missing:
            raise RuntimeError(f"Run failed (rc={rc}); missing/empty: {missing}")
        return RunResult(
            run_dir=d,
            sinogram_trues=d / "sinogram_Trues.raw.gz",
            sinogram_scatter=d / "sinogram_Scatter.raw.gz",
            image_trues=d / "image_Trues.raw.gz",
            image_scatter=d / "image_Scatter.raw.gz",
            energy_spectrum=d / "Energy_Sinogram_Spectrum.dat",
            log=d / "MCGPU-PET.out",
            wall_time_s=dt, returncode=rc,
        )

    def _handle_existing(self, d: Path, mode: str):
        outputs = list(d.glob("*.raw.gz"))
        if not outputs:
            return None
        if mode == "error":
            raise FileExistsError(f"Outputs exist in {d}")
        if mode == "skip":
            return self._collect(d, rc=0, dt=0.0)
        if mode == "overwrite":
            for f in outputs + list(d.glob("*.dat")) + [d / "MCGPU-PET.out"]:
                if f.exists():
                    f.unlink()
            return None
        raise ValueError(f"unknown on_existing mode: {mode!r}")