% 10 - Reproducibility

# Results-only and full-reproduction routes

The Results Explorer contains the compact summary tables and held-out predictions needed by the `Training_*` notebooks. It does not import PyTorch, download an embedding model, or train a network.

1. Install `uv`.
1. Run `training/start.bat`.
1. Open the notebooks in numerical order.

The Full Reproduction package additionally contains the 95 harmonized borehole tables, cached MiniLM embeddings, and training code. Its locked environment uses CPU PyTorch by default to avoid CUDA and driver-specific DLL failures. CUDA can be enabled only after verifying compatibility.

A complete LOBO run contains 95 folds and can be slow on CPU. The smoke test validates installation and data flow but does not constitute a scientific result. Reported experiments should retain the feature configuration, split protocol, epoch count, random seed, and uncertainty settings stored with each run.
