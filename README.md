# Predictive Maintenance of Aircraft Engines - NASA C-MAPSS RUL

Deep learning baseline for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan benchmark. The project now trains on the real `FD001` subset instead of synthetic demo data.

## What The Pipeline Does
- Loads `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`
- Parses engine-wise time-series records from NASA C-MAPSS
- Computes capped training RUL targets
- Builds fixed-length sequences per engine
- Drops constant or near-constant channels on FD001
- Normalizes features using train-only statistics
- Trains a 2-layer PyTorch LSTM with validation and early stopping
- Evaluates RMSE, MAE, NASA score, and Monte Carlo dropout uncertainty

## Quick Start

### 1. Refresh the official NASA data
```bash
run_pipeline.cmd
```

### 2. Train on FD001
```bash
run_pipeline.cmd
```

### 3. Evaluate the saved checkpoint
```bash
test_environment.cmd
```

### 4. Run the complete pipeline
```bash
run_pipeline.cmd
```

## Model
- Architecture: 2-layer LSTM
- Hidden size: 64
- Dropout: 0.2
- Default sequence length: 50 cycles
- Default capped RUL: 125
- Optimizer: Adam
- Loss: MSE

## Data Notes
- C-MAPSS rows contain `unit id`, `cycle`, `3 operational settings`, and `21 sensor measurements`
- FD001 contains one operating condition and one fault mode
- Feature count after filtering may be lower than 24 because constant channels are removed

## Saved Artifacts
- `models/lstm_rul.pth`: best validation checkpoint
- `models/scaler.pkl`: train-fit feature scaler
- `models/training_history.json`: epoch-by-epoch metrics

## Diagnostics
```bash
test_environment.cmd
```

## Notes
- This is a regression problem, so the primary metrics are RMSE and MAE rather than classification accuracy
- Monte Carlo dropout is used for uncertainty estimation after deterministic prediction
- `train_advanced.py` is kept as a thin alias to the main training entrypoint

## References
- NASA C-MAPSS: [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
