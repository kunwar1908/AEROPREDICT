# Setup Summary & Quick Start

## Current State
- Project uses the real NASA C-MAPSS `FD001` files
- Training/evaluation pipeline is built around a PyTorch LSTM baseline
- Validation split, early stopping, checkpointing, and uncertainty estimation are enabled
- Metrics are now intended to come from the real benchmark pipeline, not synthetic data

## Run Order

### Refresh official dataset files
```bash
run_pipeline.cmd
```

### Train the baseline
```bash
run_pipeline.cmd
```

### Evaluate the saved model
```bash
test_environment.cmd
```

### Full pipeline
```bash
run_pipeline.cmd
```

## Diagnostics
```bash
test_environment.cmd
```

## Main Behavior
- Reads `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`
- Computes capped RUL targets with default cap `125`
- Keeps only informative channels from the FD001 training set
- Uses train-only scaling
- Saves the best validation checkpoint to `models/lstm_rul.pth`
- Saves the scaler to `models/scaler.pkl`
- Saves epoch metrics to `models/training_history.json`

## Key Defaults
- Sequence length: `50`
- Hidden size: `64`
- Number of LSTM layers: `2`
- Dropout: `0.2`
- Batch size: `128`
- Epochs: `30`
- Early stopping patience: `7`

## Notes
- This is a regression setup, so read RMSE/MAE instead of percentage accuracy
- `train_advanced.py` now routes to the main training entrypoint
- The older synthetic-data summaries are obsolete after this upgrade
