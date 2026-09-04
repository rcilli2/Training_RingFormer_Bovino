# BOVINO Full Reproduction

Portable JupyterLab reproduction of the MiniLM PCA16 + XYZ/depth MiniTransformer experiment with MC Dropout uncertainty estimation. The package includes 95 harmonized borehole tables and a precomputed MiniLM embedding cache, so the default workflow does not download an embedding model.

## Quick start on Windows

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run [`training/start.bat`](training/start.bat).
3. Run the `Training_*` notebooks in numerical order.
4. Use `90_Full_Reproduction.ipynb` only when you want to launch new training runs.

The Full package is a superset of the Results Explorer. It includes the same scientific narrative and compact benchmark tables, followed by the raw borehole data, embedding cache, and training code needed for reproduction.

The six `Training_*` notebooks use only saved CSV tables and do not import PyTorch. Run `uv run python verify_results.py` to check this lightweight path. PyTorch is needed only for `90_Full_Reproduction.ipynb` and the training commands below.

The project uses Python 3.12 and a reproducible `uv.lock`. PyTorch is explicitly installed as the CPU build to avoid CUDA/driver DLL issues on workshop machines.

## Optional NVIDIA CUDA support

The default environment is CPU-only. On a machine with a compatible NVIDIA driver, run:

```powershell
training\setup_cuda.bat
```

The default CUDA wheel is `cu126`. Pass another PyTorch wheel tag when required by the driver, for example `training\setup_cuda.bat cu128`. The launcher replaces only the local PyTorch installation, then runs `verify_torch.py --expect cuda`. It does not modify `uv.lock`.

To return to the portable CPU environment:

```powershell
uv sync --reinstall
```

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
