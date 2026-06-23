## Introduction
This is a Python wrapper which wraps MCGPU-PET (Badal et al., Comput. Phys. Commun. 295 (2024) 109008, doi:10.1016/j.cpc.2023.109008), developed at the US FDA and in the public domain (17 U.S.C. §105). The bundled MCGPU-PET.x binary and the simulation engine are their work; this repository adds a Python configuration, phantom-building, and I/O layer. If you use this in published work, please cite the original MCGPU-PET paper.

## Installation

This package wraps [MCGPU-PET](https://github.com/DIDSR/MCGPU-PET) and is managed with [pixi](https://pixi.sh). If you don't have pixi, install it first by following the instructions at https://pixi.sh.

> **Note on the binary and GPU.** A compiled `MCGPU-PET.x` is included, but GPU binaries are architecture-specific. If you clone this on a machine with a different GPU and the simulator fails to run, rebuild it from the [MCGPU-PET source](https://github.com/DIDSR/MCGPU-PET) and replace `mcgpu_pet_wrapper/MCGPU-PET.x`. The Python API (building voxel spaces, reading outputs) works without a GPU; only running an actual simulation (`Runner`) needs an NVIDIA GPU with CUDA.

### Step 1 — Clone this repository

Pick a location to keep the wrapper. A simple convention is to keep the wrapper and the projects that use it **side by side** in one folder:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/electronics10/mcgpu-pet-wrapper.git
```

You now have `~/projects/mcgpu-pet-wrapper`.

### Step 2 — Create your own project next to it

```bash
cd ~/projects
mkdir my-pet-project
cd my-pet-project
pixi init
```

`pixi init` creates a `pixi.toml` file. Your layout now looks like this:

```
~/projects/
├── mcgpu-pet-wrapper/      <- the wrapper you cloned
└── my-pet-project/         <- your project, where you write code
    └── pixi.toml
```

### Step 3 — Add the wrapper as a dependency

Open `my-pet-project/pixi.toml` in a text editor and add these lines at the end:

```toml
[pypi-dependencies]
mcgpu-pet-wrapper = { path = "../mcgpu-pet-wrapper", editable = true }
```

The path `../mcgpu-pet-wrapper` means "go up one folder, then into mcgpu-pet-wrapper" -- it is relative to _your project_. This works because of the side-by-side layout above. (If you cloned the wrapper somewhere else, use the full absolute path instead, e.g. `path = "/home/you/tools/mcgpu-pet-wrapper"`.)

`editable = true` means that if you ever edit the wrapper's code, the changes take effect immediately, with no reinstall.

`[pypi-dependencies]`  is used (not `[tool.pixi.pypi-dependencies]`) because in a standalone `pixi.toml` the section is top-level; the `tool.pixi.` prefix is only for when the config lives inside a `pyproject.toml`. Since `pixi init` creates a `pixi.toml`, the shorter form is correct. Worth knowing if you have a `pyproject.toml`-based project instead — then you'd need the prefixed form.

Then install:

```bash
pixi add python
pixi install
```

Supplement: 
1. You may add a `pyrightconfig.json` in the local root, a version-controllable way to configure Pyright-based tools (e.g., Pylance), to resolve static analysis of Pylance for better experience.
2. Ask AI for help if you don't know how to install the module.


### Step 4 — Write and run a script

Create a file `my-pet-project/main.py`:

```python
import mcgpu_pet_wrapper as mpw

# Load the built-in default configuration (no file path needed).
cfg = mpw.default_config()
mpw.validate_config(cfg) # validate

# Build a simple test object: a point source.
voxel_space = mpw.point_source(cfg)

# Stage a run directory (writes config.json, MCGPU-PET.in, voxel_space.vox).
mpw.build_run("data/run_0", cfg, voxel_space)

print("Setup OK. Run directory created at data/run_0.")
```

Run it **through pixi** so it uses the project's environment:

```bash
pixi run python main.py
```

> Always use `pixi run python ...`, not plain `python ...`. The wrapper lives inside the project's pixi environment, so plain `python` won't find it (you'd get `ModuleNotFoundError: No module named 'mcgpu_pet_wrapper'`).

If you see `Setup OK.`, everything is installed correctly.

### Step 5 — Run an actual simulation (needs a GPU)

The step above builds the input files but does not run the simulator. To run it (on a machine with an NVIDIA GPU and the compiled binary):

```python
result = mpw.Runner()("data/run_0", on_existing="overwrite")

trues   = mpw.read_sinogram(result.sinogram_trues,   cfg)   # (NSINOS, NANGLES, NRAD)
scatter = mpw.read_sinogram(result.sinogram_scatter, cfg)
print("trues:", trues.shape, "total counts:", trues.sum())
```

See the sections below for the full walkthrough of every module.

## 1 Prerequisite for MCGPU-PET

(You may skip this part and will only need to read this if you encounter GPU compatibility problems.)

[MCGPU-PET](https://github.com/DIDSR/MCGPU-PET.git) is a research code; the authors likely developed it by extending NVIDIA CUDA sample programs. Therefore, CUDA samples dependencies such as `helper_functions.h` are essential but not shipped with the distribution. Moreover, since GPU binaries are architecture-specific (unlike x86-64 CPU binaries), distributing a prebuilt `.x` binary is impractical. (Although this is what we ship in the repo, for simplicity, which may cause problems. Continue reading to if you encountered any compatibility problems.)

### 1.1 Installation

Clone the repository:
```bash
git clone https://github.com/DIDSR/MCGPU-PET.git
cd MCGPU-PET
```

The expected environment is Linux. To get the executive file `MCGPU-PET.x`, run
```bash
make
```
However, running `make` directly will likely fail due to the two issues above. If so, fix them as shown in the details.

<details>

1. **CUDA samples dependency**

Generally, find or install `helper_functions.h`:
```bash
# Option A: install via apt (replace version as needed)
sudo apt install cuda-samples-12-0

# Option B: clone NVIDIA's samples repo
git clone https://github.com/NVIDIA/cuda-samples.git
```
Then set `CUDA_SDK_PATH` in the `Makefile` to the `Common/inc` directory of whichever option you used.

Personally, I did
```bash
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0
```
Since the release is 12.0, I clone the CUDA samples by
```bash
# Clone the matching samples version to ~/cuda-samples
git clone --branch v12.0 https://github.com/NVIDIA/cuda-samples.git ~/cuda-samples
```

Then in `Makefile`, I changed the line:
```makefile
CUDA_SDK_PATH = $(HOME)/cuda-samples/Common/
```

2. **GPU compute capability**

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```
Update `GPU_COMPUTE_CAPABILITY` in the `Makefile` accordingly, e.g. for capability 7.5:
```makefile
GPU_COMPUTE_CAPABILITY = -gencode=arch=compute_75,code=sm_75
```

(Personally, mine is the same as the authors so I don't need to change anything.)

Then compile:
```bash
make
```

(Compilation may succeeds with warnings, which are safe to ignore.)
</details>

### 1.2 Basic IO

After you get the `MCGPU-PET.x` binary file, it becomes an independent executive binary file (which serves as a MC PET simulator). That is to say, you can deploy the file to any places/projects without any worries, it would work fine on your computer. Personally, I used it as a backend for my Python project. According to the authors, the simulator takes in a `.in` file (`MCGPU-PET.in` in sample example) and is typically used by prompting 
```bash
time ./MCGPU-PET.x MCGPU-PET.in | tee MCGPU-PET.out
```

Note that the `.in` file requires a `.vox` file (simulation object such as a phantom) and some `.gz` files (materials). 


## 2 A Closer look in MCGPU-PET

### 2.1 The First Run

A single run with the default template produces:

| File | Format | Content |
|---|---|---|
| `image_Trues.raw.gz`   | gzipped `int32`, shape $(N_z, N_y, N_x)$ | per-voxel count of *true* coincidences emitted from that voxel |
| `image_Scatter.raw.gz` | same | same, for scattered coincidences |
| `sinogram_Trues.raw.gz`   | gzipped `int32`, flat buffer | trues binned into a 3D PET sinogram (michelogram) |
| `sinogram_Scatter.raw.gz` | same | same, for scatter |
| `Energy_Sinogram_Spectrum.dat` | text | energy spectrum of detected events |
| `MCGPU-PET.out` | text | our tee'd log |

**Emission images** (`image_Trues.raw.gz`, `image_Scatter.raw.gz`) are *not* reconstructions. They are forward counts — for each voxel, the count of coincidences (Monte Carlo) within the energy window that *originated* there. A real scanner cannot observe this; only the simulator can. They're useful as a sanity check (Trues + Scatter should approximate the input activity map, Poisson-noisy and sensitivity-weighted) and as a ground-truth reference. They are sized by the voxel object (`.vox` input).

**Energy spectrum** is a text file we don't use downstream; mentioned for completeness.

**Sinograms** are what reconstruction acts on. `sinogram_Trues.raw.gz` and `sinogram_Scatter.raw.gz` are written as flat int32 buffers of length NBINS. The kernel's per-event flat index is `ibin = izm*(NANGLES*NRAD) + ith*NRAD + ir` so the C-order reshape is (NSINOS, NANGLES, NRAD): axial-plane slowest, angular middle, radial fastest. They 3D layout is dictated by the `SINOGRAM PARAMETERS` block in the `.in` file — specifically `num_rings`, `MRD`, and `span`. The data format is something worth inspection and deal with in the later sections.

### 2.2 phantom_9x9x9cm.vox

```vox
[SECTION VOXELS HEADER v.2008-04-13]
9 9 9   No. OF VOXELS IN X,Y,Z
1.0 1.0 1.0   VOXEL SIZE (cm) ALONG X,Y,Z
 1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED
 2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED
 1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)
[END OF VXH SECTION]  # MCGPU-PET voxel format: Material  Density  Activity
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0
1 0.0012 0.0

1 0.0012 0.0
1 0.0012 0.0
```

According to the [paper]([https://doi.org/10.1016/j.cpc.2023.109008](https://doi.org/10.1016/j.cpc.2023.109008)), which described loosely, the file is in penEasy 2008 format (penEasy 2008 voxel geometry file), plus an activity column (per-voxel activity in Bq).

Note that density is not determined by the material. This is because, to my understanding, by material, MCGPU-PET actually mean something like $\mu/\rho$ ( for which $\mu$ is the linear attenuation coefficient and $\rho$ is the density). To be more precise, $\mu/\rho = \frac{Z}{A}N_A\sigma$, which is exactly specified by the material in microscopic scale (and energy).

Key conventions, all verified against `MCGPU-PET.cu`:

- Index order: x fastest, y slower, z slowest. For example:
```vox
voxel (x=0, y=0, z=0)      ← line 1
voxel (x=1, y=0, z=0)      ← line 2   x increments first
voxel (x=2, y=0, z=0)      ← line 3
...
voxel (x=Nx-1, y=0, z=0)
voxel (x=0,    y=1, z=0)   ← x resets, y increments
voxel (x=1,    y=1, z=0)
...
voxel (x=0, y=0, z=1)      ← only after a full x,y plane does z increment
```
- The bounding box sits in the first octant with voxel `{1,1,1}`  (index start from 1) having a corner at the origin. So if you want a sphere centered in the FOV, you have to compute the center yourself. (At first glance, it seems uneccessary to talk about the origin. However, it is easier for us to to reference what we are actually indicating. We'll see the importance of it when dealing with the scanner placement in the following section.)
- Vacuum (material 0) is illegal. Air must be material ≥ 1 (index start from 1).
- Blank lines between x- and y-cycles are optional (the header flag controls it). Skip them — fewer bytes, faster to write, and the parser handles either.
- Gzip is fine; the binary opens `.vox.gz` directly via zlib.

### 2.3 MCGPU-PET.in

```in
#[SECTION PHASE SPACE FILE v.2016-07-05]
 MCGPU_PET.psf                  # OUTPUT PHASE SPACE FILE FILE NAME
 0.0  0.0  0.0  12.656  -9.05   # CYLINDRIC DETECTOR CENTER, HEIGHT, AND RADIUS: X, Y, Z, H, RADIUS [cm] (IF RADIUS<0: DETECTOR CENTERED AT THE CENTER OF THE VOXELIZED GEOMETRY)
```

- phase space file, to my understanding, is basically something like a listmode file (haven't really inspect it yet).
- In the previous subsection, where we discuss the `.vox` file, we've mentioned that the bounding box sits in the first octant, of a coordinate system. Now, we can define the placement of the scanner (cylindric detector center). Typically, we adopt the negative radius convention accroding to MCGPU-PET, where the scanner is expended from the center of the voxelized geometry (phantom/voxel space), automatically calculated, with half the specified height along both the positive and negative z direction (MCGPU-PET hardcoded the axial direction as the z direction) and rings of the specified radius transversely.

```
#[SECTION DOSE DEPOSITION v.2012-12-12]
YES                             # TALLY MATERIAL DOSE? [YES/NO] (electrons not transported, x-ray energy locally deposited at interaction)
NO                              # TALLY 3D VOXEL DOSE? [YES/NO] (dose measured separately for each voxel)
mc-gpu_dose.dat                 # OUTPUT VOXEL DOSE FILE NAME
  1  128                        # VOXEL DOSE ROI: X-index min max (first voxel has index 1)
  1  128                        # VOXEL DOSE ROI: Y-index min max
  1  159                        # VOXEL DOSE ROI: Z-index min max
```

- We basically don't need this part. Set both of the tally to no and ignore the dose ROI.
- Dose is a radiation-physics quantity: the amount of ionizing energy absorbed per unit mass of material, measured in grays (Gy = joules per kilogram). When a 511 keV photon Compton-scatters or photoelectrically absorbs in tissue, it dumps some energy there. Dose tallying adds up that deposited energy. It answers "how much radiation did the object absorb?" — relevant for patient/animal safety studies, not for image formation. So in a PET _imaging_ simulation, dose is usually a byproduct you ignore. The simulator can report it because it's tracking the photons anyway, but it's orthogonal to the sinograms.
- One caveat the comment flags: `electrons not transported, x-ray energy locally deposited at interaction`. This means the dose estimate is approximate — when a photon interacts, the energy that _would_ go to a recoil electron is just deposited on the spot rather than following the electron's path. For PET energies this is a reasonable approximation but not a rigorous dosimetry calculation. Another reason it's a rough side-output, not the simulator's main job.
- Tally material dose aggregates absorbed energy by material type. The output is one number per material: total energy absorbed in air, total in water, etc. It does _not_ tell you _where_ in space the energy landed, only _what kind of stuff_ absorbed it. Cheap (a handful of accumulators, one per material) and low-memory. It is reported in the text log.
- Tally 3D voxel dose aggregates absorbed energy per voxel, producing a full 3D map the same shape as the object (`mc-gpu_dose.dat`). This tells you _spatially_ where energy was deposited — a dose image you could overlay on the anatomy. Much more expensive: it needs an accumulator for every voxel (millions of them), hence the ROI lines to limit which voxels are tracked and save memory. That's what the ROI is for: when voxel dose is `ON`, you often only care about dose in a sub-region (say, a tumor), so you restrict the per-voxel tally to an index box — `X: 1..128, Y: 1..128, Z: 1..159` — instead of the whole grid. Since yours is `OFF`, those ROI lines are inert; they're read by the parser but nothing uses them.

```
#[SECTION SINOGRAM PARAMETERS v.2019-04-25]
12.656 # AXIAL FIELD OF VIEW (FOVz) in cm //22.0 16.4
80     # NUMBER OF ROWS
336    # TOTAL NUMBER OF CRYSTALS
168    # NUMBER OF ANGULAR BINS (NCRYSTALS/2)
147    # NUMBER OF RADIAL BINS // 391
159    # NUMBER OF Z SLICES 
128    # IMAGE RESOLUTION (NUMBER OF BINS IN THE IMAGE) // legacy, not used
700    # NUMBER OF ENERGY BINS (NE)
79     # MAXIMUM RING DIFFERENCE (MRD)
11     # SPAN
```

- MCGPU-PET assumes that the axial direction is the z direction. With the above specification, a scanner of axial FOV 12.656 cm is generated, which forms a cylinder consists of 80 equally spaced rings, with 336 detectors (crystals) attached to each ring. 
- The image resolution field is, in the current MCGPU-PET, effectively nothing. It's read into `RES`, used only to compute `NVOXS`, and `NVOXS` is referenced only in dead/commented code. It's a vestige of an earlier design (probably when the emission image was a fixed square `RES×RES×NZS` grid independent of the voxel object). The authors flagged it as possibly unnecessary. (`RES` feeds only `NVOXS = RES*RES*NZS`, and `NVOXS` is passed into the kernel but used only in a commented-out line (`//if (ivox>=0 && ivox<*NVOXS)`), and annotated by the authors at line 183: `//FEB2022 !!DeBuG!! Is this input necessary? And should it be NVOX_SIM?`)
- One thing worth mention is the emission image (`image_True.raw.gz`, `image_Scatter.raw.gz`). The emission image is exactly the voxel-object size (determined by the `.vox` input). The source code confirms it: it increments `Imagen_T_dev[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y]`, and the grid is one block per voxel. The emission image indexes by emitting voxel, so it must be voxel-object-shaped and has nothing to do with the `.in` file specifications.
- As for the rest fields, they are mainly parameters for sinograms and specifications of how the LOR are binned in 3D (Michelogram). One may need to know the basics of sinogram and Michellogram in advance.  
	- The number of Z (axial) slices `NZS` is normally two times the number of rings (row) minus one. There are $N_{rings}$ direct planes (a ring paired with itself) and $N_{rings} − 1$ planes that sit _between_ adjacent rings (ring i with ring i+1), giving $2·N_{rings} − 1$ total. `NZS = 159` is the maximum number of planes a single segment can hold (the direct segment, segment 0, achieves it; `159 = 2·N_rings − 1`). It is used as the per-segment offset stride. The total stored plane count `NSINOS = 1293` is the _sum_ of the actual plane counts across all 15 segments (159+2·(147+125+103+81+59+37+15)), each segment holding fewer planes than segment 0 because span compression and the MRD cutoff progressively limit the reachable axial positions. The kernel lays out the sinogram by giving every segment the same stride `NZS` in a provisional layout (`ofseg = iseg*NZS`), then subtracts correction terms (`-(SPAN+1)`, `-floor(ofk)*2*SPAN`) to pack the shorter segments tightly. `NZS` is the stride constant the packing arithmetic is built around. The code needs it as an input because it's the scaffold the offset computation hangs on — it can't be silently derived inside the kernel without rederiving the whole michelogram layout. (Whether they _should_ have made you specify it versus computing it from `N_rings` is a design choice; functionally `159 = 2·80−1` so it's determined by the ring count, but they chose to read it explicitly.)

	- After the simulation, we'll recieve the `sinogram.raw.gz` files. The flat buffer reshapes to `(NSINOS, N_ang, N_rad) = (1293, 168, 147)`, planes slowest, radial fastest (`ibin = izm·N_ang·N_rad + ith·N_rad + ir`). The plane index `izm` encodes the michelogram segment. A _segment_ groups a span-wide band of ring differences: with `delta = |r2 − r1|`, the segment magnitude is `incl = floor((delta − 1 + (span+1)/2)/span)`, so segment 0 is the _direct_ band (`delta` near 0), segment ±1 the next band, etc. The code's segment counter is `iseg = 2·incl`, decremented by 1 for negative slope (`r2 < r1`), giving storage order `seg_0, seg_−1, seg_+1, seg_−2, seg_+2, …`. The `NSINOS = 1293` planes are these segments concatenated, each segment contributing fewer planes than the last as the span compression and MRD cutoff (`delta > MRD` is discarded) remove oblique combinations. 

  - The number of angular bins is normally set us half the number of the detectors (crystals) $N_d/2$ since it only ranges from $0$ to $\pi$. 

	- The number of the raidal bins, inside the transverse field of view, is normally $N_{d}+1$. The "+1" arises because radial positions are sampled symmetrically about the central LOR (the line through the scanner axis): with $N_d$ even, the symmetric arrangement yields an odd count $N_d + 1$.

  - The radial axis is in arc coordinates, not uniform distance — and MCGPU-PET does not arc-correct. This matters for reconstruction. The radial bin index `ir` is computed purely from the *crystal-index difference* of the two detectors that fired (`ir = |ix2 − ix1 − NANGLES|`, then centered by `+NRAD/2`); it is never converted to a physical distance. The physical perpendicular distance `s` of a line of response from the scanner axis relates to the crystal-index separation `m` by the chord geometry of a ring of radius `R`: $$s = R\cos\frac{\pi m}{N_{\text{crystals}}}$$ (Radial bins are widest near the center and bunch together toward the edge of the field of view. This nonuniform sampling is the natural "arc" geometry of a ring scanner.) This was verified against the source at both ends: the kernel stores `ir` directly with no remapping, and the export path copies the sinogram buffer from the GPU and gzwrites it to disk untouched (only a sum-for-reporting happens in between). So the file received is in raw, arc-uncorrected coordinates.

  - Consequence for reconstruction. The sinogram has the structure of a Radon transform (axes = view angle $\phi$, signed radial offset $s$). But standard FBP assumes uniformly sampled $s$. Applied to the raw arc-coordinate sinogram, it produces a geometrically distorted image — roughly right near the center, increasingly wrong toward the edge. Two correct paths: (a) arc-correct first — resample each sinogram row from the arc spacing onto a uniform s grid using the cosine map above — then FBP; or (b) use an iterative reconstructor (MLEM/OSEM, as the real scanner uses) that encodes the true arc geometry in its system matrix and consumes the non-arc-corrected data directly, which is preferable because it avoids the noise correlation that resampling introduces. (Caveat on exactness. The cosine mapping is derived from (and consistent with) the crystal-indexing code, but the precise correspondence between a given `ir` and a physical $s$ — including the half-bin offset from the `+NRAD/2` centering and the sign convention — should be confirmed empirically with a point-source simulation (place a source at a known radius, see which radial bin fills) before trusting it for quantitative reconstruction.)

```
		seg 0:           159 planes
		seg ±1 (1,2):    147 each
		seg ±2 (3,4):    125 each
		seg ±3 (5,6):    103 each
		seg ±4 (7,8):     81 each
		seg ±5 (9,10):    59 each
		seg ±6 (11,12):   37 each
		seg ±7 (13,14):   15 each
		                 ─────
		sum =            1293
```


```
#[SECTION MATERIAL FILE LIST v.2009-11-30]  
materials/air_5-515keV.mcgpu.gz             # 1
materials/water_5-515keV.mcgpu.gz           # 2
```

- There are only two materials, namely air and water, shipped in the github repository of MCGPU-PET.
- This is mostly fine — but the reason it's fine is specific, and there's one caveat. MCGPU-PET explicitly checks that each material's cross-section table extends up to the annihilation energy (511 keV) and aborts with an error otherwise. At 511 keV in light tissue, Compton (incoherent) scattering dominates every other photon process — photoelectric and Rayleigh are smaller by roughly two orders of magnitude. The Compton interaction rate per unit volume is governed by electron density: $$n_e = \rho \cdot \frac{Z}{A} \cdot N_A.$$For soft tissues $Z/A$ is nearly constant (water 0.555, muscle ~0.550, fat ~0.557 — all within ~1%). So once you scale water by the right $\rho$, you reproduce the correct electron density, and therefore the correct scatter behavior, to ~1% for any soft tissue. The clinical world relies on exactly this: PET attenuation correction maps CT to 511 keV with a bilinear water/bone rule, because soft tissue tracks water and only bone needs a separate anchor. The one caveat: bone. Cortical bone has $Z/A \approx 0.53$ (lower than water) and higher effective $Z$. Two consequences: First, using water composition at bone density slightly _overestimates_ electron density (by the $Z/A$ ratio, ~4–5%), hence overestimates Compton/scatter in those voxels. Secondly, the small residual photoelectric/coherent contribution that bone's higher-$Z$ elements (Ca, P) add is missed. For preclinical mouse imaging this is a minor effect: bone is thin, low-volume, and the scatter sinogram is a smooth quantity that integrates over the whole body, so a few-percent error confined to bone voxels largely averages out. MCGPU-PET's own validation is only quoted as ~10% agreement against GATE/PeneloPET, so a bone-composition error well under that is in the noise.

- Baically, water + per-voxel density is enough. Treat density as a heterogeneity knob. (If you later want bone fidelity or generalization to bony/contrast-bearing regions: generate a proper bone material valid to 511 keV with PENELOPE's material program, and assign bone voxels that material rather than up-scaling water density.)

## 3 Building a Python Wrapper for MCGPU-PET

This part of the document walks through the `mcgpu_pet_wrapper` Python wrapper.

The static file system of MCGPU-PET make it difficult to untangle and manipulate the parameters, and the dependencies are dangerous since MCGPU-PET will often still run even if your input is unreasonable. That silent kind of failure is the worst kind.

A singel MCGPU-PET run needs several files that all have to agree with each other:
- an input file `MCGPU-PET.in`, describing GPU, tracer, scanner, materials, image, sinogram, etc.
- a voxel object `phantom_9x9x9cm.vox`, describing what the input voxelized space (phantom + background) with its material, density (not determined by the material, elaborated in the following sections), and activity distribution
- material files `xxx.gz` (used in the `.in` file)

To remain integraity, we keep only one file that we edit, and everything else is generated from it. That one file is `config.json`.

### 3.1 Configuration

```json
{
  "voxel_space":{
	"axis_order": "xyz",
    "grid_size_mm": [1.0, 1.0, 1.0],
    "num_voxels": [80, 80, 150]
  },
  "scanner":{
	"symmetry_axis": "z",
    "axial_fov_mm": 150.0,
    "transaxial_fov_mm": 80.0,
    "radius_mm": 52.5,
    "num_rings": 80,
    "num_detectors_per_ring": 336
  },
  "sinogram":{
    "num_angular_bins": 168,
    "num_radial_bins": 257,
    "num_radial_trim": 40,
    "max_ring_difference": 79,
    "span": 11,
    "num_axial_planes": 159
  },
  "mcgpu": {
    "random_seed": 0,
    "gpu_number": 0,
    "gpu_threads_per_block": 32,

    "acquisition_time_s": 600.0,
    "isotope_mean_life_s": 9502.0,

    "psf_filename": "MCGPU_PET.psf",
    "psf_max_elements": 10000000,
    "report_trues_scatter": 0,
    "report_psf_sinogram": 2,

    "tally_material_dose": "NO",
    "tally_voxel_dose": "NO",
    "dose_filename": "mc-gpu_dose.dat",

    "energy_resolution": 0.15,
    "energy_window_low_eV": 358000.0,
    "energy_window_high_eV": 664000.0,
    "num_energy_bins": 700,

    "voxel_space_file": "voxel_space.vox",
    "materials": [
      "materials/air_5-515keV.mcgpu.gz",
      "materials/water_5-515keV.mcgpu.gz"
    ]
  }
}
```

 We rename the input `.vox` file from `phantom_9x9x9cm.vox` to `voxel_space.vox` to avoid ambiguity (phantom + background; or just some voxelized space in general). The "axis_order" and "symmetry_axis" declare the interpretation of the coordinate triples; currently only "xyz" is supported and the loader enforces it. We adopt the convention where the scanner expand it self from the center of the voxel space (the negative-radius convention in the `.in`). Activity outside the detector cylinder is emitted but never detected. The validator warns on axial mismatch and on the bounding box exceeding the bore/FOV, but it cannot see where the activity is (that needs the built VoxelGrid), so keeping active regions inside the transaxial FOV is the user's responsibility.

Several fields are coupled — changing one without its partner breaks consistency (the validator will catch most, but understand why):

1. `num_radial_bins = num_detectors_per_ring + 1 − 2·num_radial_trim`. Here 336 + 1 − 80 = 257. Edit trim and bins together.
2. `num_angular_bins = num_detectors_per_ring / 2 = 168` (convention; only change for angular mashing).
3. `num_axial_planes = 2·num_rings − 1 = 159`. This is the per-segment plane cap, not the total stored planes (which is NSINOS, derived). 

On the other hand, `num_radial_trim` controls how much of the transverse FOV the sinogram covers, via the nonuniform arc mapping `s = R·cos(π·m/N_crystals)` (not a uniform mm-per-bin)where `R = radius_mm` and `m` is the index separation. With trim 75 → 187 bins, the sinogram covers ~80 mm diameter, matching `transaxial_fov_mm`. Larger trim = smaller FOV coverage but smaller/faster sinograms. The validator warns if coverage falls short of `transaxial_fov_mm`.


We write a module `config.py` for loading the json file, checking it for internal contradictions, and computing some derived numbers that the rest of the package needs. If the config is wrong, we want to fail here, loudly, with some helpful messages.

For example:
```python
import mcgpu_pet_wrapper as mpw


cfg = mpw.load_config("mcgpu_pet_wrapper/templates/template.json")
mpw.validate_config(cfg)

print(mpw.sinogram_shape(cfg))
print(mpw.segment_table(cfg)[0])
```

<details>

The template is dedicated to a specific configuration as stated in the following:

1. The scanner block is a discrete proxy for a continuous-crystal scanner. The real Bruker 7T PET has 3 physical rings of 8 monolithic LYSO blocks with continuous light-distribution decoding and 10-layer DOI — a detector model MCGPU-PET cannot represent. MCGPU-PET requires discrete crystals in discrete rings. So `num_rings`, `num_detectors_per_ring`, `span`, and `max_ring_difference` are not the real hardware; they are a self-consistent discrete stand-in chosen so the sampling is fine enough not to be the bottleneck. What is physically real and must match the scanner: `radius_mm` (52.5), `axial_fov_mm` (150), `transaxial_fov_mm` (80), and the energy fields. Validation is therefore image-domain and scatter-fraction, never sinogram-structure.

2. `energy_window_low_eV` (358000) and `energy_window_high_eV` (664000) are the scanner's real "30%" window, and `energy_resolution` (0.15 = ~15% @ 511 keV) is its real blurring. These three dominate the scatter fraction — the quantity you validate against (6.9% mouse, 14.2% rat). Change them to match a different real acquisition, not to tune results.

3. `isotope_mean_life_s: 9508`: simulating F-18.

</details>

### 3.2 Voxel Space
#### 3.2.1 VoxelGrid Class and VoxelSpaceBuilder
We define a VoxelGrid class which serves as a data container of per-voxel material ID, density, and activity. 

We also define a VoxelSpaceBuilder class which has some predefined building function of different geometries. One may check the source code for more details. 

```python
voxel_space = mpw.VoxelSpaceBuilder(mpw.voxel_space_shape_xyz(cfg), mpw.grid_size_mm(cfg))
# 1. fill everything with air (material 1, low density, no activity)
voxel_space.fill_background(material_id=1, density=0.0012, activity_Bq_per_mL=0.0)

# 2. a water cylinder down the middle, with no activity
cx, cy, cz = voxel_space.bbox_center_mm        # the center of the object, in mm
voxel_space.add_cylinder(center_mm=(cx, cy, cz), radius_mm=45.0, height_mm=94.0,
               axis="z", material_id=2, density=1.0, activity_Bq_per_mL=0.0)

# 3. a small hot (high-activity) sphere inside it
voxel_space.add_sphere(center_mm=(cx+2, cy+13, cz+20), radius_mm=6.0,
             material_id=2, density=1.0, activity_Bq_per_mL=37000.0)

voxel_space = voxel_space.build() # validates and returns the voxel_space


print(voxel_space)
```

**Order matters**: later paints cover earlier ones where they overlap. So the recipe is always: background first, big things next, small details last.

#### 3.2.2 Vox input output

MCGPU-PET does not read Python objects; it reads a text file in a specific format (penEasy 2008, with one extra column for activity). This module turns a voxel space into that file, and can read one back.

`VoxFileGenerator(voxel_space).write(run_dir, filename)` writes the file.
The format is a short header (voxel counts, voxel sizes in cm, a couple of
bookkeeping lines) followed by one line per voxel: `material density activity`.

```python
mpw.VoxFileGenerator(voxel_space).write("data/run_0", cfg["mcgpu"]["voxel_space_file"])
```

The one subtle thing the writer gets right is voxel order. The file lists
voxels with the x-index changing fastest, then y, then z. Our arrays are stored as
`(Nz, Ny, Nx)`, and flattening them in standard ("C") order gives exactly
x-fastest. If this were wrong, the phantom would come out transposed and the
simulation would be quietly nonsense - so the unit tests check it explicitly.

`read_vox(path)` parses a `.vox` (or `.vox.gz`) back into a `VoxelGrid`. We use
it mainly to confirm a round-trip: write a phantom, read it back, check it is
identical. It is also handy for re-loading a recorded run.

```python
vox = mpw.read_vox("data/run_0/" + cfg["mcgpu"]["voxel_space_file"])

print(vox.shape_zyx)
```

### 3.3 Input file generator
The `.in` file is the actual input to MCGPU-PET. It is a fussy text file: the simulator finds each setting by looking for exact section headers and counting lines, so we cannot just write it freely. Instead we keep a `template.in` with all the headers and comments intact, and this module overwrites only the *value* lines, using numbers from the config.

There is a table (`SCHEMA`) mapping each readable setting name to its location in the template
(which section, which value line). The generator reads the config, converts units
where MCGPU expects different ones (millimetres to centimetres, etc.), and writes
each value into its slot, leaving headers and comments untouched.

```python
gen = mpw.InFileGenerator()        # loads templates/template.in
gen.from_config(cfg)              # validates the config, then fills in values
gen.write("data/run_0")        # writes data/run_0/MCGPU-PET.in
```

- The "activity per material" lines are left alone. The template documents
  that MCGPU ignores them ("INPUT NOT USED") - activity comes from the `.vox`
  file's third column instead. So the phantom owns *where* the activity is; the
  `.in` owns the physics and geometry. No overlap, no contradiction.
- The detector is set to auto-center on the object. The detector line uses a
  negative radius, which tells MCGPU "put the detector at the center of the voxel
  object." This is why the placement check (next module) can assume the detector
  sits at the object's center.

### 3.4 Runner

Now, we have basically everything we need for a simulation. Let's write a final runner wrapper that handle everything in a shot. Running MCGPU-PET means: put the right files in a directory,
launch the binary there, wait, and collect the outputs. This module does all of that.

There are two pieces.

`build_run(run_dir, config, phantom)` prepares a run directory. It:

1. validates the config,
2. writes a copy of `config.json` into the directory (so the run is
   self-contained and you can re-read exactly what produced it),
3. generates `MCGPU-PET.in`,
4. writes `voxel_space.vox.gz`.

It also keeps the filename in sync between the `.vox` file and the `.in`
file.

```python
mpw.build_run("data/run_1", cfg, voxel_space)
```

`Runner()(run_dir)` does the actual run. It symlinks the binary and the
materials folder into the directory, checks all required files are present, runs
`MCGPU-PET.x`, streams its log to both your screen and `MCGPU-PET.out`, and then
collects the output files into a tidy `RunResult` object:

```python
result = mpw.Runner()("data/run_1", on_existing="error")

print(result.sinogram_trues)    # path to sinogram_Trues.raw.gz
print(result.wall_time_s)       # how long it took
```

The `on_existing` argument decides what to do if outputs already exist:
`"error"` (refuse), `"overwrite"` (delete and rerun), or `"skip"` (reuse the
existing outputs). `"skip"` is what makes a big batch *resumable* - if it dies
halfway, rerun and it picks up where it left off.

Why a separate `build_run` and `Runner`? So you can stage a directory on one
machine (no GPU needed) and run it later on the GPU box. The run directory is the
unit of work.

Till now, we've pretty much done everything. In the next subsection, we provide some useful  functions.

### 3.5 Utilities

#### 3.5.1 phantoms.py

Painting shapes by hand is fine for experiments, but for real work you may want standard, repeatable phantoms. This module provides factory functions: you hand them a config, they hand you back a finished `Phantom`. The crucial design choice is that they read the object's size and voxel size from the config, so the phantom automatically matches the simulation geometry - they cannot disagree.

 `nema_iq_preclinical(config, ...)` builds the NEMA NU 4-2008 Image Quality phantom - a standard small-animal PET test object. It is a good first phantom because the MCGPU-PET paper validates against this exact phantom, so your results have something to be compared to.

Its geometry (all standard, baked into the function): a 30 mm-wide, 50 mm-long plastic (PMMA) cylinder, with three regions stacked along its length - five hot rods of increasing diameter (1 to 5 mm) at one end, a uniform hot region in the middle, and two "cold" inserts (one water, one air, no activity) at the other end.

```python
cfg = mpw.load_config("mcgpu_pet_wrapper/templates/template.json")

voxel_space = mpw.nema_iq_preclinical(cfg, hot_activity_Bq_per_mL=37000.0)

```

About the `materials` argument: the function needs three material roles - `air`,
`water`, and `pmma`. By default it uses your existing air and water cross-section
files and *fakes* PMMA as "water cross-sections at PMMA's density (1.19)." This is
accurate to about 1% at PET energies because the relevant physics (Compton
scattering) depends mostly on density, and it lets the phantom run with the two
material files you already have. If you later obtain a real PMMA cross-section file
(say, as material 3), you pass `materials={"air": (1, 0.0012), "water": (2, 1.0), "pmma": (3, 1.19)}` and nothing else changes.

`uniform_cylinder(config, ...)` is a simpler factory: one uniform fillable cylinder centered in the field of view. Useful for the NEMA scatter-fraction test or for computing normalization factors.

```python
voxel_space = mpw.uniform_cylinder(cfg, diameter_mm=50.0, length_mm=120.0,
                         activity_Bq_per_mL=1000.0)
```

`point_source` is a small hot sphere in air, optionally offset from center. Handy for resolution/PSF checks and as the simplest possible sanity-test object (if a point source doesn't reconstruct to a point, something's wrong).

```python
voxel_space = mpw.point_source(cfg)
```

#### 3.5.2 data_reader.py

MCGPU-PET writes its sinograms as a flat stream of numbers in a `.gz` file. To use them we must reshape that stream into a 3D array - and to reshape correctly we need the exact dimensions, which come from the config.

`read_sinogram(path, config)` does this. It reads the file, checks the number
of values matches what the config predicts (raising a clear error if not - that is
your geometry-mismatch alarm), and reshapes to `(NSINOS, NANGLES, NRAD)`:

```python
sino = mpw.read_sinogram("data/run_0/sinogram_Trues.raw.gz", cfg)

print(sino.shape)
```

The shape order - planes, then angular, then radial - comes straight from how the
simulator's GPU code computes each event's position in the flat array. We read
that code to get it right rather than guessing.

`read_sinogram_segments` splits the flat sinogram into per-segment 3D arrays with their ring-difference and axial labelling. You get a list of clean, labelled per-segment sinograms instead of one opaque stack.

```python
sino = mpw.read_sinogram_segments("data/run_0/sinogram_Trues.raw.gz", cfg)

print(sino[5]["data"].shape)
```

`summarize_sinogram` for sanity checks: 

```python
print(mpw.summarize_sinogram("data/run_0/sinogram_Trues.raw.gz", cfg))
```

`read_emission_image(path, config)` reads the "emission images" (per-voxel
counts of detected coincidences) into the object's shape `(Nz, Ny, Nx)`. These are
a ground-truth reference, not something a real scanner could measure.

```python
img = mpw.read_emission_image("data/run_0/image_Trues.raw.gz", cfg)

print(img.shape)
```

### 3.6 Auxiliary: Rebinning
The rebinning is not part of the core wrapper. Not going to the technical details, we provide SSRB and FORE rebinning with the aid of `read_sinogram_segments`.

```python
import mcgpu_pet_wrapper as mpw
from mcgpu_pet_wrapper import rebinning as rb

cfg_path = "mcgpu_pet_wrapper/templates/template.json"
run_dir = "data/run_0"
sino_path = run_dir + "/sinogram_Trues.raw.gz"

# Load configuration
cfg = mpw.load_config(cfg_path)

# Load sinogram segments
sino = mpw.read_sinogram_segments(sino_path, cfg)

print(rb.ssrb(sino, cfg).shape)
print(rb.fore(sino, cfg).shape)
```

Note that arc correction is done by default so one doesn't have to worry about it downstream. (You may call `sino = rb.arc_correct(sino, cfg)` to see it explicitly.) If you plan on doing iterative reconstruction later, you may want to turn it off.

## 4 A Full Run

```python
import mcgpu_pet_wrapper as mpw


cfg_path = "mcgpu_pet_wrapper/templates/template.json"
run_dir = "data/run_0"

# Load configuration
cfg = mpw.load_config(cfg_path)

# Define voxel space object
voxel_space = mpw.point_source(cfg)

# Build simulation directory and files
mpw.build_run(run_dir, cfg, voxel_space)

# Run simulation
simulation = mpw.Runner()(run_dir, "overwrite")
```

To read the data, try:

```python
import mcgpu_pet_wrapper as mpw


cfg_path = "mcgpu_pet_wrapper/templates/template.json"
run_dir = "data/run_0"
img_path = run_dir + "/image_Trues.raw.gz"
sino_path = run_dir + "/sinogram_Trues.raw.gz"

# Load configuration
cfg = mpw.load_config(cfg_path)

# Load emission image
image = mpw.read_emission_image(img_path, cfg)
print(image.shape)

# Load sinogram segments
sino = mpw.read_sinogram_segments(sino_path, cfg)
print(sino[5]["data"].shape)
```
