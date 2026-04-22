- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements - Skipped, requirements already specified in the issue.

- [x] Scaffold the Project - Complete with Python source structure, data directory, and models directory.

- [x] Customize the Project - LSTM RUL prediction model with uncertainty estimation implemented.

- [x] Install Required Extensions - No VS Code extensions needed for this Python project.

- [x] Compile the Project - All dependencies installed (PyTorch 2.8, Pandas, NumPy, Scikit-learn).

- [x] Create and Run Task - Training scripts verified and working (train.py, train_advanced.py).

- [x] Launch the Project - Model trained and evaluated with RMSE 26.90, MAE 24.35.

- [x] Ensure Documentation is Complete - README with setup guide, TROUBLESHOOTING guide, and test script.

## Quick Commands

```bash
# Full diagnostic test
test_environment.cmd

# Run complete pipeline
run_pipeline.cmd

# Advanced training with plots
python src/train_advanced.py

# Quick evaluation
python src/evaluate.py
```

## Project Summary

**Deep Learning for Aircraft Engine RUL Prediction**
- Model: 2-layer LSTM with 64 hidden units
- Input: 50-cycle sequences from 24 sensors
- Output: Remaining Useful Life (RUL) prediction
- Uncertainty: MC Dropout (100 samples)
- Performance: RMSE ~26.9 cycles, MAE ~24.4 cycles

**Key Files:**
- `src/model.py` - LSTM architecture
- `src/train_advanced.py` - Training with visualization
- `src/evaluate.py` - Model evaluation
- `README.md` - Comprehensive guide
- `TROUBLESHOOTING.md` - Error solutions

**Generated Artifacts:**
- `/models/lstm_rul.pth` - Trained model weights
- `/models/scaler.pkl` - Data normalizer
- `/models/training_loss.png` - Loss history
- `/models/predictions_analysis.png` - Prediction uncertainty plots

**Python Environment:** `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Unix-like shells