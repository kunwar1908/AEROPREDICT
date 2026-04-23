#!/bin/bash

echo "================================"
echo "System Diagnostics & Testing"
echo "================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

echo ""
echo "1. Checking Python environment..."
$PYTHON_BIN --version
$PYTHON_BIN -c "import sys; print('   Python executable:', sys.executable)"

echo ""
echo "2. Checking critical dependencies..."
$PYTHON_BIN -c "import torch; print('   ✓ PyTorch version:', torch.__version__)"
$PYTHON_BIN -c "import pandas; print('   ✓ Pandas version:', pandas.__version__)"
$PYTHON_BIN -c "import numpy; print('   ✓ NumPy version:', numpy.__version__)"
$PYTHON_BIN -c "import sklearn; print('   ✓ Scikit-learn version:', sklearn.__version__)"

echo ""
echo "3. Validating NASA C-MAPSS data files..."
$PYTHON_BIN -c "
from src.data_loader import load_data, prepare_train_data
train_df, test_df, rul_df = load_data('FD001')
prepared_train_df, features, _ = prepare_train_data(train_df)
print(f'   ✓ Training rows: {train_df.shape}')
print(f'   ✓ Test rows: {test_df.shape}')
print(f'   ✓ RUL rows: {rul_df.shape}')
print(f'   ✓ Informative features kept: {len(features)}')
print(f'   ✓ Train engines: {prepared_train_df.unit_id.nunique()}')
"

echo ""
echo "4. Testing sequence generation..."
$PYTHON_BIN -c "
from src.data_loader import load_data, prepare_train_data, create_sequences_per_engine
train_df, _, _ = load_data('FD001')
prepared_train_df, features, target = prepare_train_data(train_df)
X_train, y_train = create_sequences_per_engine(prepared_train_df, features, target, seq_length=50)
print(f'   ✓ Sequence tensor shape: {X_train.shape}')
print(f'   ✓ Label shape: {y_train.shape}')
"

echo ""
echo "5. Testing model architecture..."
$PYTHON_BIN -c "
import torch
from src.data_loader import load_data, prepare_train_data
from src.model import LSTMRULPredictor
train_df, _, _ = load_data('FD001')
_, features, _ = prepare_train_data(train_df)
model = LSTMRULPredictor(input_size=len(features), hidden_size=64, num_layers=2)
x = torch.randn(8, 50, len(features))
output = model(x)
print(f'   ✓ Model output shape: {output.shape}')
print(f'   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}')
"

echo ""
echo "6. Testing saved checkpoint files..."
if [ -f "$SCRIPT_DIR/models/lstm_rul.pth" ]; then
    echo "   ✓ Trained model found: lstm_rul.pth"
else
    echo "   ✗ Trained model NOT found"
fi

if [ -f "$SCRIPT_DIR/models/scaler.pkl" ]; then
    echo "   ✓ Scaler found: scaler.pkl"
else
    echo "   ✗ Scaler NOT found"
fi

echo ""
echo "7. Quick end-to-end evaluation..."
$PYTHON_BIN src/evaluate.py --dataset FD001 --mc-samples 10

echo ""
echo "================================"
echo "Diagnostics Complete!"
echo "================================"
