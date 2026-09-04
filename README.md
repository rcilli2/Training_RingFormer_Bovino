# BOVINO Full Reproduction

Portable JupyterLab reproduction of the MiniLM PCA16 + XYZ/depth MiniTransformer experiment with MC Dropout uncertainty estimation. The package includes 95 harmonized borehole tables and a precomputed MiniLM embedding cache, so the default workflow does not download an embedding model.

## Quick start on Windows

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run [`training/start.bat`](training/start.bat).
3. Open `notebooks/00_full_reproduction.ipynb` in JupyterLab.

The project uses Python 3.12 and a reproducible `uv.lock`. PyTorch is explicitly installed as the CPU build to avoid CUDA/driver DLL issues on workshop machines.

## Running the experiments

```powershell
uv run python verify_torch.py --expect cpu
run_smoke_test.bat
run_lobo.bat
run_loco.bat
```

The smoke test runs one fold for two epochs. The complete 95-fold LOBO run is computationally expensive on CPU. Generated LOBO, LOCO, and smoke-test outputs are ignored by Git; the supplied reference smoke-test outputs remain in `results/smoke_test_direct/`.

## Project layout

- `src/`: training code and MiniTransformer implementation.
- `data/`: borehole tables and cached MiniLM embeddings.
- `notebooks/`: interactive workflow.
- `results/`: supplied reference output and locally generated runs.
- `training/`: JupyterLab launcher compatible with the RING training template.
