# BOVINO Full Reproduction

Portable JupyterLab reproduction of the MiniLM PCA16 + XYZ/depth MiniTransformer experiment with MC Dropout uncertainty estimation. The package includes 95 harmonized borehole tables and a precomputed MiniLM embedding cache, so the default workflow does not download an embedding model.

## Quick start on Windows

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run [`training/start.bat`](training/start.bat).
3. Open the notebooks in numerical order in JupyterLab.

The training now provides two complementary routes:

- `00_full_reproduction.ipynb` validates PyTorch and launches new smoke, LOBO, or LOCO runs.
- `01_results_explorer.ipynb` reproduces the principal protocol comparison from packaged predictions without training.
- `02_uq_method_comparison.ipynb` compares calibration and confidence-based error detection across five UQ methods.
- `03_distribution_shift.ipynb` examines spatial exclusion buffers and synthetic feature perturbations.

The conceptual chapters in `text/` explain the dataset, architecture, validation protocols, uncertainty methods, metrics, and limitations.

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
- `notebooks/`: full reproduction and saved-results analysis workflows.
- `results/`: supplied reference output and locally generated runs.
- `text/`: self-contained background chapters for the RING training interface.
- `training/`: JupyterLab launcher compatible with the RING training template.
