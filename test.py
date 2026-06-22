import mcgpu_pet_wrapper as mpw
from mcgpu_pet_wrapper import rebinning as rb

# Load the built-in default configuration (no file path needed).
cfg = mpw.default_config()

# Build a simple test object: a point source.
voxel_space = mpw.point_source(cfg)

# Stage a run directory (writes config.json, MCGPU-PET.in, voxel_space.vox).
mpw.build_run("data/run_0", cfg, voxel_space)

print("Setup OK. Run directory created at data/run_0.")

result = mpw.Runner()("data/run_0", on_existing="overwrite")

trues   = mpw.read_sinogram(result.sinogram_trues,   cfg)   # (NSINOS, NANGLES, NRAD)
scatter = mpw.read_sinogram(result.sinogram_scatter, cfg)
print("trues:", trues.shape, "total counts:", trues.sum())

sino = mpw.read_sinogram_segments("data/run_0/sinogram_Trues.raw.gz", cfg)
# sino = rb.arc_correct(sino, cfg)
print(rb.ssrb(sino, cfg).shape)
print(rb.fore(sino, cfg).shape)