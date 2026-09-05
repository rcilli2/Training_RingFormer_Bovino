% 14 - Reproducing the figures

# Explore saved experiments or run a new model

## The saved-results route

The Results Explorer contains compact benchmark tables and held-out predictions. The Training notebooks use these files to construct figures and comparisons. They do not require PyTorch.

1. Install uv.
2. Run training/start.bat to open JupyterLab.
3. Open the Training notebooks in numerical order.

Each notebook corresponds to a question:

| Notebook | Question | Packaged evidence |
|---|---|---|
| Training_00 | What is the dataset and target? | Held-out prediction table and study-area assets |
| Training_01 | What do features and representations contribute? | representation_lobo_metrics.csv and representation_loco_metrics.csv |
| Training_02 | How well do UQ scores rank errors? | Representation benchmark metrics |
| Training_03 | What happens as spatial support decreases? | buffer_uq_method_metrics.csv |
| Training_04 | What happens within a borehole? | Saved interval predictions |
| Training_05 | Where do performance and uncertainty vary? | Saved predictions and borehole coordinates |

A plotted curve can therefore be a recombination of previously evaluated results. It is not evidence that the notebook has just trained a model.

## The full-reproduction route

The Full package additionally contains 95 harmonized borehole tables, cached MiniLM embeddings and the training code. Start with 90_Full_Reproduction.ipynb.

The default locked environment uses CPU PyTorch. The optional CUDA setup is available through training/setup_cuda.bat.

Run the smoke test first to check data flow. It trains a small test configuration, not the complete scientific experiment. A complete LOBO evaluation trains across 95 held-out folds and takes substantially longer.

The supplied launchers reproduce the MiniLM PCA16 MC-dropout configuration. The cached five-method benchmark represents a broader set of experiments than this training entry point alone.

## Reading and extending an experiment

For a comparison, keep the feature set, split and evaluation aggregation explicit. Changing PCA dimension, input variables or the held-out group changes the question being evaluated.

For new runs, retain the configuration and held-out predictions alongside the metrics. This lets the figures be traced back to the model and observations that generated them.

Return to the [chapter index](00_index.md).
