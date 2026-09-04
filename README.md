# BOVINO Full Reproduction

This package reruns the principal MiniLM PCA16 + XYZ/depth Transformer experiment with MC Dropout. It includes the 95 harmonized borehole tables and a precomputed MiniLM embedding cache, so the default run does not need to download the embedding model.

## Recommended installation order

1. Install a fresh Python 3.11 virtual environment with `setup_cpu.bat` or `setup_cuda.bat`.
2. Run `verify_torch.py` before starting training.
3. Run `run_lobo.bat` for leave-one-borehole-out validation.
4. Run `run_loco.bat` for leave-one-campaign-out validation.

The CPU setup deliberately installs PyTorch separately from the remaining dependencies. This avoids silently receiving an incompatible CUDA build on a machine without a working CUDA runtime.

## Runtime expectations

The full 95-fold LOBO experiment is computationally expensive on CPU. For an installation smoke test, run `run_smoke_test.bat`, which evaluates only one fold and 2 epochs. Output is written under `results/`.

## PyTorch troubleshooting

If `verify_torch.py` fails, do not start training. Delete `.venv` and rerun the appropriate setup script. The Results Explorer package remains usable because it does not depend on PyTorch.
