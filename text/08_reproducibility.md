% 8 - Reproducibility

# Running the portable experiment

The package includes harmonized borehole tables and a cached MiniLM embedding matrix, so the default workflow does not download an embedding model.

1. Install `uv`.
1. Run `training/start.bat`.
1. Open `notebooks/00_full_reproduction.ipynb`.
1. Execute the smoke test before a complete run.

The locked environment uses CPU PyTorch by default to avoid CUDA and driver-specific DLL failures. CUDA can be enabled with `training/setup_cuda.bat` after verifying driver compatibility.

Full LOBO training contains 95 folds and can be slow on CPU. The smoke test checks installation and data flow but is not a scientific result. Random seed, feature configuration, split protocol, epoch count, and uncertainty settings are stored with each completed run and should accompany reported metrics.
