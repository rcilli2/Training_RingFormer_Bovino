@echo off
setlocal
cd /d "%~dp0"
set "DEVICE=%~1"
if "%DEVICE%"=="" set "DEVICE=cpu"
uv run --no-sync python -u src\scripts\train_sequence_labelling_uq.py ^
  --data-glob "data\boreholes\*.csv" ^
  --output-dir "results\smoke_test" ^
  --embedding-cache "data\embedding_cache_minilm.npz" ^
  --embedding-model all-MiniLM-L6-v2 ^
  --embedding-pca-dim 16 ^
  --epochs 2 --batch-size 95 --learning-rate 3e-4 --weight-decay 0.01 ^
  --optimizer adam --grad-clip 1.0 --min-epochs 2 --no-early-stopping ^
  --d-model 128 --num-heads 16 --num-layers 1 --dropout 0.1 ^
  --pooling-power 5 --valid-pooling-threshold 0.5 ^
  --uq-method mc_dropout --mc-samples 3 ^
  --model-selection-fraction 0 --temperature-fraction 0 --conformal-fraction 0 ^
  --device %DEVICE% --num-workers 0 --seed 13 --split-mode lobo --fold-index 0 ^
  --train-on-valid-only --eval-on-valid-only --no-temperature-scaling ^
  --ignore-valid-as-feature --use-x-as-feature --use-y-as-feature --use-z-as-feature ^
  --use-depth-as-feature --use-embeddings-as-feature
endlocal
