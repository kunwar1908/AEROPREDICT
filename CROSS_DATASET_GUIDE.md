# C-MAPSS Cross-Dataset Generalization Guide

## 1. Pipeline Code Changes

The core of the refactoring was in `src/train.py`, which is now entirely rewritten to handle rigorous cross-dataset domain generalization:

*   **Multi-Source Training:** You can now provide multiple source domains using `--train-datasets FD001 FD002`. Features are pooled but `unit_id`s are sequentially offset to ensure the train/validation split isolates entire engine lifecycles without overlaps and prevents data leakage.
*   **Leakage Prevention:** `StandardScaler` is strictly `fit()` on the generated training features *only*. The validation and multiple test sets (e.g. `--test-datasets FD001 FD003`) are transformed against the source features distribution.
*   **Handling Variable Features:** If you train on FD001 (which models out constant features) and test on FD003 (which has varying operating conditions), the pipeline explicitly maps available features and zeros out unknown dimensions to guarantee matrix shape compatibility. 
*   **Evaluation Artifacts:** Generates cross-domain analytical plots for *every* test dataset requested:
    *   `actual_vs_predicted_FD00X.png` 
    *   `error_histogram_FD00X.png`
    *   A combined console summary table of RMSE, MAE, and NASA score across datasets.

`api_server.py` and `Main_Dashboard.html` were updated to read and display explicit metadata flags under the Model Summary board.
*   **Trained On**, **Validated On**, **Tested On** (Space-separated identifiers like FD001 FD004)
*   **Experiment Mode:** Auto-identified as `IN-DISTRIBUTION`, `CROSS-DATASET`, or `MULTI-SOURCE`.

## 2. Baseline Experiment Config Structure

You can run experiments using the refactored CLI:

### Baseline 1: In-Distribution (Control)
```bash
python src/train.py --train-datasets FD001 --test-datasets FD001
```

### Baseline 2: Single-Source Transfer (Cross-Dataset)
Train on one condition, evaluate on another.
```bash
python src/train.py --train-datasets FD001 --test-datasets FD004
python src/train.py --train-datasets FD002 --test-datasets FD004
python src/train.py --train-datasets FD003 --test-datasets FD004
```

### Baseline 3: Multi-Source Domain Generalization
Train on multiple diverse domains to build a generalized boundary for unseen domains constraint logic.
```bash
python src/train.py --train-datasets FD001 FD002 FD003 --test-datasets FD004
```

## 3. Why Cross-Dataset Performance May Be Low

Transferring from small domains (FD001: sea-level, 1 operating condition) to complex ones (FD004: 6 operating conditions) triggers extreme **Covariate Shift**:
1. **Unseen Feature Spaces:** In FD001, Altitude or Mach numbers are largely constant and pruned. On target domains, these sensors fluctuate, leaving the LSTM without calibrated weights yielding zero-padded defaults.
2. **Distinct Wear Mechanisms:** Alternate operating envelopes create distinct wear profiles (e.g., thermal vs. mechanical distress). A trajectory modeling 50 cycles of degradation in FD001 may actually equal 100 cycles in FD004 based on the physical engine layout.
3. **Out-of-Distribution Normalization:** Because we rigorously fit the scaler *only* on the source data (FD001 explicitly), applying FD004 sets the distributions drastically outside $N(0,1)$, breaking gradient activation.

## 4. Recommendations for Improving Transfer Performance

If basic model transferring yields high RMSE, implement these advanced strategies:

*   **Operating Condition Normalization:** Group observations locally by clustering the 3 operational condition fields. Standardize the 21 sensor strings relative to their specific cluster constraints instead of across the global timeline.
*   **Domain Adaptation (CORAL):** Formulate a composite PyTorch loss function to penalize distance distributions (Correlation Alignment) capturing hidden shifts in source and target domains.
*   **Temporal Convolutional Networks (TCN) or Transformers:** Basic LSTMs are sensitive to length variation shifts. Shift to robust Attention-based models with sliding temporal convolution embeddings.
*   **Fine-Tuning Hooks:** Enable `--fine-tune` on `train.py` freezing early RNN gates and consuming 5-10% of target dataset trajectories into a recalibrated Dense projection head.

## 5. Assumptions

*   Script execution now uses repo-relative logic and the local `.venv` if present.
*   Standard piece-wise linear function variables like `max-rul` remaining at ceiling $125$ mapping accurately to target transfer mappings.
*   Changes applied to `aerospace-dashboard` retain backward compatibility avoiding regressions for live data visual polling via explicit DOM parsing modifications.
