from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{idx}" for idx in range(1, 4)]
    + [f"sensor_{idx}" for idx in range(1, 22)]
)
DEFAULT_DATASET = "FD001"
DEFAULT_MAX_RUL = 125


def get_dataset_paths(dataset: str = DEFAULT_DATASET) -> dict[str, Path]:
    dataset = dataset.upper()
    return {
        "train": DATA_DIR / f"train_{dataset}.txt",
        "test": DATA_DIR / f"test_{dataset}.txt",
        "rul": DATA_DIR / f"RUL_{dataset}.txt",
    }


def _ensure_numeric_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()

    if not first_line:
        raise ValueError(f"Dataset file is empty: {path}")

    if first_line.lower().startswith("<!doctype html"):
        raise ValueError(
            f"Dataset file contains HTML instead of C-MAPSS data: {path}. "
            "Run src/download_data.py to refresh the official NASA files."
        )


def _read_cmapss_table(path: Path, column_names: list[str]) -> pd.DataFrame:
    _ensure_numeric_dataset(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")

    if frame.shape[1] != len(column_names):
        raise ValueError(
            f"Unexpected column count in {path}: expected {len(column_names)}, got {frame.shape[1]}"
        )

    frame.columns = column_names
    return frame


def load_data(dataset: str = DEFAULT_DATASET) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if dataset.upper() == "MULTI":
        return load_combined_data(["FD001", "FD002", "FD003", "FD004"])
    paths = get_dataset_paths(dataset)
    train_df = _read_cmapss_table(paths["train"], list(CMAPSS_COLUMNS))
    test_df = _read_cmapss_table(paths["test"], list(CMAPSS_COLUMNS))
    rul_df = _read_cmapss_table(paths["rul"], ["RUL"])
    return train_df, test_df, rul_df


def load_combined_data(datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_train = []
    combined_test = []
    combined_rul = []
    max_train_unit_id = 0
    max_test_unit_id = 0

    for ds in datasets:
        train_df, test_df, rul_df = load_data(ds)
        
        train_df = train_df.copy()
        test_df = test_df.copy()
        rul_df = rul_df.copy()

        train_df["unit_id"] += max_train_unit_id
        max_train_unit_id = train_df["unit_id"].max()
        combined_train.append(train_df)

        test_df["unit_id"] += max_test_unit_id
        max_test_unit_id = test_df["unit_id"].max()
        combined_test.append(test_df)

        combined_rul.append(rul_df)

    return (
        pd.concat(combined_train, ignore_index=True),
        pd.concat(combined_test, ignore_index=True),
        pd.concat(combined_rul, ignore_index=True),
    )


def _select_feature_columns(
    train_df: pd.DataFrame,
    near_constant_threshold: float = 1e-8,
) -> list[str]:
    # Return all 24 sensory and setting columns equally to ensure cross-dataset dimension matching
    return [name for name in train_df.columns if name not in {"unit_id", "cycle", "RUL"}]


def prepare_train_data(
    train_df: pd.DataFrame,
    max_rul: int | None = DEFAULT_MAX_RUL,
    near_constant_threshold: float = 1e-8,
) -> tuple[pd.DataFrame, list[str], str]:
    prepared = train_df.copy()
    prepared["RUL"] = prepared.groupby("unit_id")["cycle"].transform("max") - prepared["cycle"]
    if max_rul is not None:
        prepared["RUL"] = prepared["RUL"].clip(upper=max_rul)
    feature_columns = _select_feature_columns(prepared, near_constant_threshold=near_constant_threshold)
    return prepared, feature_columns, "RUL"


def create_sequences_per_engine(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    seq_length: int = 50,
    unit_ids: list[int] | np.ndarray | None = None,
    pad_short: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[float] = []

    engine_ids = unit_ids if unit_ids is not None else df["unit_id"].unique()
    for unit_id in engine_ids:
        engine_data = df[df["unit_id"] == unit_id].sort_values("cycle")
        if len(engine_data) < seq_length and not pad_short:
            continue
        values = engine_data[features].to_numpy(dtype=np.float32)
        targets = engine_data[target].to_numpy(dtype=np.float32)
        if len(engine_data) < seq_length and pad_short:
            pad_rows = np.repeat(values[:1], seq_length - len(engine_data), axis=0)
            sequences.append(np.vstack([pad_rows, values]))
            labels.append(targets[-1])
            continue
        for start_idx in range(len(engine_data) - seq_length + 1):
            end_idx = start_idx + seq_length
            sequences.append(values[start_idx:end_idx])
            labels.append(targets[end_idx - 1])

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def prepare_test_data(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    features: list[str],
    seq_length: int = 50,
    max_rul: int | None = DEFAULT_MAX_RUL,
) -> tuple[np.ndarray, np.ndarray]:
    sequences, labels, _ = prepare_test_samples(test_df, rul_df, features, seq_length=seq_length, max_rul=max_rul)
    return sequences, labels


def prepare_test_samples(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    features: list[str],
    seq_length: int = 50,
    max_rul: int | None = DEFAULT_MAX_RUL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_cycle = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    prepared = test_df.merge(last_cycle, on="unit_id")
    prepared["RUL"] = prepared["max_cycle"] - prepared["cycle"]
    prepared = prepared.merge(
        rul_df.reset_index(names="unit_id").assign(unit_id=lambda df: df["unit_id"] + 1),
        on="unit_id",
        suffixes=("", "_final"),
    )
    prepared["RUL"] = prepared["RUL"] + prepared["RUL_final"]
    if max_rul is not None:
        prepared["RUL"] = prepared["RUL"].clip(upper=max_rul)

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    unit_ids: list[int] = []
    for unit_id in prepared["unit_id"].unique():
        engine_data = prepared[prepared["unit_id"] == unit_id].sort_values("cycle")
        values = engine_data[features].to_numpy(dtype=np.float32)
        if len(engine_data) < seq_length:
            pad_rows = np.repeat(values[:1], seq_length - len(engine_data), axis=0)
            values = np.vstack([pad_rows, values])
        else:
            values = values[-seq_length:]
        sequences.append(values)
        labels.append(float(engine_data.iloc[-1]["RUL"]))
        unit_ids.append(int(unit_id))

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int32),
    )


def train_validation_split(
    train_df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    unit_ids = np.sort(train_df["unit_id"].unique())
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unit_ids)
    validation_size = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_ids = np.sort(shuffled[:validation_size]).tolist()
    training_ids = np.sort(shuffled[validation_size:]).tolist()
    if not training_ids:
        raise ValueError("Validation split consumed all engines; lower validation_fraction.")
    return training_ids, validation_ids


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_pred) - np.asarray(y_true)
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


if __name__ == "__main__":
    train_df, test_df, rul_df = load_data()
    prepared_train_df, features, target = prepare_train_data(train_df)
    X_train, y_train = create_sequences_per_engine(prepared_train_df, features, target)
    X_test, y_test = prepare_test_data(test_df, rul_df, features)
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Feature count after filtering: {len(features)}")
    print(f"Train sequences: {X_train.shape}, labels: {y_train.shape}")
    print(f"Test sequences: {X_test.shape}, labels: {y_test.shape}")
