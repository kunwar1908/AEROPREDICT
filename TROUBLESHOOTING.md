# Troubleshooting Guide

## Common Issues

### Dataset files contain HTML instead of numeric rows
Cause: an old download URL saved webpage content instead of the real NASA files.

Fix:
```bash
run_pipeline.cmd
```

### Missing dataset file
Fix:
```bash
run_pipeline.cmd
```

### Torch import or dependency errors
Fix:
```bash
python -m pip install --upgrade -r requirements.txt
```

### Training is slow
Tips:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
python src/train.py --dataset FD001 --epochs 10 --batch-size 64
```

### Out-of-memory during training
Try:
```bash
python src/train.py --dataset FD001 --batch-size 32 --seq-length 30
```

## Diagnostic Commands

### Full environment check
```bash
test_environment.cmd
```

### Data loading check
```bash
python -c "from src.data_loader import load_data; train_df, test_df, rul_df = load_data('FD001'); print(train_df.shape, test_df.shape, rul_df.shape)"
```

### Evaluation check
```bash
python src/evaluate.py --dataset FD001 --mc-samples 10
```

## Reset Steps
```bash
cd /d "%~dp0"
if exist .venv\Scripts\python.exe .venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
python src/download_data.py
test_environment.cmd
```
