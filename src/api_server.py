from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from data_loader import (
    DEFAULT_DATASET,
    load_data,
    prepare_test_samples,
    prepare_train_data,
)
from model import LSTMRULPredictor, predict_with_uncertainty
from captum.attr import IntegratedGradients

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DASHBOARD_DIR = BASE_DIR / "aerospace-dashboard"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {
        "state_dict": checkpoint,
        "config": {
            "dataset": DEFAULT_DATASET,
            "seq_length": 50,
            "max_rul": 125,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "feature_columns": None,
        },
        "metrics": {},
        "history": [],
    }


class ModelApiService:
    def __init__(self) -> None:
        self.checkpoint_path = MODELS_DIR / "lstm_rul.pth"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.history_path = MODELS_DIR / "training_history.json"
        self.device = get_device()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {self.scaler_path}")

        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.config = self.checkpoint["config"]
        self.metrics = self.checkpoint.get("metrics", {})
        train_datasets = self.config.get("train_datasets", [DEFAULT_DATASET])
        self.train_datasets = [d.upper() for d in train_datasets]
        if len(self.train_datasets) == 1:
            self.dataset = self.train_datasets[0]
        else:
            self.dataset = "MULTI"
        self.test_datasets = [d.upper() for d in self.config.get("test_datasets", self.train_datasets)]
        self.mode = self.config.get("mode", "in-distribution")
        self.seq_length = int(self.config.get("seq_length", 50))
        self.max_rul = int(self.config.get("max_rul", 125))

        self.history = self._load_history()
        self.model = self._load_model()
        with self.scaler_path.open("rb") as handle:
            self.scaler = pickle.load(handle)

        self._summary_cache: dict[str, Any] | None = None
        self._test_cache = self._build_test_cache()
        self._per_dataset_cache: dict[str, dict] = {}
        self._ensure_artifacts()

    def _load_history(self) -> list[dict[str, Any]]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        return list(self.checkpoint.get("history", []))

    def _load_model(self) -> LSTMRULPredictor:
        feature_columns = self.config.get("feature_columns") or []
        model = LSTMRULPredictor(
            input_size=int(self.config.get("input_size", len(feature_columns))),
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.2)),
        ).to(self.device)
        model.load_state_dict(self.checkpoint["state_dict"])
        model.eval()
        return model

    def _build_test_cache(self) -> dict[str, Any]:
        train_df, test_df, rul_df = load_data(self.dataset)
        prepared_train_df, feature_columns, _ = prepare_train_data(train_df, max_rul=self.max_rul)
        configured_features = self.config.get("feature_columns")
        if configured_features:
            feature_columns = list(configured_features)
        X_test, y_test, unit_ids = prepare_test_samples(
            test_df, rul_df, feature_columns, seq_length=self.seq_length, max_rul=self.max_rul
        )
        X_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        rul_indexed = rul_df.reset_index(names="unit_id").assign(unit_id=lambda df: df["unit_id"] + 1)
        rul_indexed = rul_indexed.rename(columns={"RUL": "RUL_final"})
        prepared_test_df = test_df.copy()
        last_cycle = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
        prepared_test_df = prepared_test_df.merge(last_cycle.reset_index(), on="unit_id")
        prepared_test_df["RUL"] = prepared_test_df["max_cycle"] - prepared_test_df["cycle"]
        prepared_test_df = prepared_test_df.merge(rul_indexed, on="unit_id")
        prepared_test_df["RUL"] = (prepared_test_df["max_cycle"] - prepared_test_df["cycle"]) + prepared_test_df["RUL_final"]
        if self.max_rul:
            prepared_test_df["RUL"] = prepared_test_df["RUL"].clip(upper=self.max_rul)

        return {
            "train_df": train_df,
            "test_df": test_df,
            "rul_df": rul_df,
            "prepared_train_df": prepared_train_df,
            "prepared_test_df": prepared_test_df,
            "feature_columns": feature_columns,
            "X_test": X_test,
            "X_test_scaled": X_scaled,
            "X_test_tensor": X_tensor,
            "y_test": y_test,
            "unit_ids": unit_ids,
            "predictions": predictions,
            "scaler": self.scaler,
        }

    def _ensure_artifacts(self) -> None:
        self._plot_training_history()
        self._plot_predictions_analysis()

    def _plot_training_history(self) -> None:
        if not self.history:
            return
        output_path = MODELS_DIR / "training_loss.png"
        epochs = [item["epoch"] for item in self.history]
        losses = [item["loss"] for item in self.history]
        val_rmse = [item["val_rmse"] for item in self.history]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, color="#58A6FF", linewidth=2, label="Training Loss (MSE)")
        plt.plot(epochs, val_rmse, color="#39D0D8", linewidth=2, label="Validation RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training Progress on NASA C-MAPSS FD001")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _plot_predictions_analysis(self) -> None:
        output_path = MODELS_DIR / "predictions_analysis.png"
        y_test = self._test_cache["y_test"]
        predictions = self._test_cache["predictions"]
        _, std_pred = predict_with_uncertainty(self.model, self._test_cache["X_test_tensor"], n_samples=20)
        std_pred = std_pred.flatten()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_test, predictions, alpha=0.7, s=40, color="#58A6FF")
        line_start = min(y_test.min(), predictions.min())
        line_end = max(y_test.max(), predictions.max())
        axes[0].plot([line_start, line_end], [line_start, line_end], "r--", linewidth=2)
        axes[0].set_xlabel("Actual RUL")
        axes[0].set_ylabel("Predicted RUL")
        axes[0].set_title("Predicted vs Actual RUL")
        axes[0].grid(True, alpha=0.3)

        sample_count = min(25, len(y_test))
        indices = np.arange(sample_count)
        axes[1].errorbar(
            indices,
            predictions[:sample_count],
            yerr=std_pred[:sample_count],
            fmt="o",
            capsize=4,
            color="#39D0D8",
            label="Prediction ± uncertainty",
        )
        axes[1].plot(indices, y_test[:sample_count], "r*", markersize=10, label="Actual RUL")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("RUL")
        axes[1].set_title("Sample Predictions With Uncertainty")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close(fig)

    def get_feature_importance(self) -> dict[str, float]:
        X_test_tensor = self._test_cache["X_test_tensor"]
        samples = X_test_tensor[:50].requires_grad_()
        ig = IntegratedGradients(self.model)
        baseline = torch.zeros_like(samples)
        attributions, _ = ig.attribute(samples, baseline, return_convergence_delta=True)
        attr_mean = attributions.mean(dim=0).mean(dim=0).detach().cpu().numpy()
        
        feature_columns = self.config.get("feature_columns") or []
        if len(feature_columns) != len(attr_mean):
            return {"error": "Feature mismatch"}
            
        importance = {feat: float(abs(val)) for feat, val in zip(feature_columns, attr_mean)}
        return importance

    def _compute_uncertainty_summary(self) -> dict[str, float]:
        mean_pred, std_pred = predict_with_uncertainty(self.model, self._test_cache["X_test_tensor"], n_samples=20)
        del mean_pred
        std_pred = std_pred.flatten()
        return {
            "averageStd": float(std_pred.mean()),
            "minStd": float(std_pred.min()),
            "maxStd": float(std_pred.max()),
        }

    def get_summary(self) -> dict[str, Any]:
        if self._summary_cache is not None:
            return self._summary_cache

        train_df = self._test_cache["train_df"]
        test_df = self._test_cache["test_df"]
        feature_columns = self.config.get("feature_columns") or []
        uncertainty = self._compute_uncertainty_summary()

        test_metrics = self.metrics.get("test_metrics", {})
        if test_metrics:
            first_ds = list(test_metrics.keys())[0]
            primary_rmse = float(test_metrics[first_ds]["rmse"])
            primary_mae = float(test_metrics[first_ds]["mae"])
            primary_nasa = float(test_metrics[first_ds]["nasa_score"])
        else:
            primary_rmse = float(self.metrics.get("test_rmse", 0.0))
            primary_mae = float(self.metrics.get("test_mae", 0.0))
            primary_nasa = float(self.metrics.get("test_nasa_score", 0.0))

        summary = {
            "dataset": self.dataset,
            "device": str(self.device),
            "config": {
                "seqLength": self.seq_length,
                "maxRul": self.max_rul,
                "hiddenSize": int(self.config.get("hidden_size", 64)),
                "numLayers": int(self.config.get("num_layers", 2)),
                "dropout": float(self.config.get("dropout", 0.2)),
                "featureCount": len(feature_columns),
                "featureColumns": feature_columns,
            },
            "metrics": {
                "bestEpoch": int(self.metrics.get("best_epoch", 0)),
                "bestValRmse": float(self.metrics.get("best_val_rmse", 0.0)),
                "testRmse": primary_rmse,
                "testMae": primary_mae,
                "nasaScore": primary_nasa,
                "averageUncertainty": uncertainty["averageStd"],
                "uncertaintyMin": uncertainty["minStd"],
                "uncertaintyMax": uncertainty["maxStd"],
                "perDatasetMetrics": {k: {"rmse": float(v["rmse"]), "mae": float(v["mae"])} for k, v in test_metrics.items()},
            },
            "metadata": {
                "trainedOn": self.train_datasets,
                "validatedOn": self.train_datasets,
                "testedOn": self.test_datasets,
                "experimentMode": self.mode,
            },
            "data": {
                "trainRows": int(len(train_df)),
                "testRows": int(len(test_df)),
                "trainEngines": int(train_df["unit_id"].nunique()),
                "testEngines": int(test_df["unit_id"].nunique()),
                "operatingSettings": 3,
                "sensorCount": 21,
                "retainedFeatureCount": len(feature_columns),
            },
            "artifacts": {
                "trainingLossUrl": "/artifacts/training_loss.png",
                "predictionsAnalysisUrl": "/artifacts/predictions_analysis.png",
            },
            "sampleEngineIds": [int(x) for x in self._test_cache["unit_ids"][:10]],
        }
        self._summary_cache = summary
        return summary

    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    def predict_sequence(self, sequence: np.ndarray, mc_samples: int = 20) -> dict[str, Any]:
        feature_count = len(self._test_cache["feature_columns"])
        if sequence.shape != (self.seq_length, feature_count):
            raise ValueError(
                f"Expected sequence shape {(self.seq_length, feature_count)}, got {tuple(sequence.shape)}"
            )

        scaled = self.scaler.transform(sequence).reshape(1, self.seq_length, feature_count)
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = float(self.model(tensor).cpu().numpy().flatten()[0])
        mean_pred, std_pred = predict_with_uncertainty(self.model, tensor, n_samples=mc_samples)
        return {
            "predictedRul": prediction,
            "meanPrediction": float(mean_pred.flatten()[0]),
            "uncertaintyStd": float(std_pred.flatten()[0]),
            "mcSamples": mc_samples,
        }

    def get_sample_prediction(self, engine_id: int | None = None) -> dict[str, Any]:
        unit_ids = self._test_cache["unit_ids"]
        index = 0
        if engine_id is not None:
            matches = np.where(unit_ids == engine_id)[0]
            if len(matches) == 0:
                raise KeyError(f"Engine {engine_id} is not available in the test set.")
            index = int(matches[0])

        sequence = self._test_cache["X_test"][index]
        actual_rul = float(self._test_cache["y_test"][index])
        prediction = self.predict_sequence(sequence, mc_samples=20)
        prediction.update(
            {
                "engineId": int(unit_ids[index]),
                "actualRul": actual_rul,
                "absoluteError": abs(prediction["predictedRul"] - actual_rul),
            }
        )
        return prediction


app = Flask(__name__)
CORS(app)
service = ModelApiService()

# ── fresh-dataset cache (keyed by dataset name) ──────────────────────────────
_fresh_cache: dict[str, dict] = {}


def _get_dataset_cache(dataset: str) -> dict:
    """Return (and memoize) predictions for any of the 4 C-MAPSS datasets.

    For datasets the model was trained on (in service.train_datasets), uses
    service.scaler (fitted on all training data combined). For unseen datasets,
    fits a fresh scaler on that dataset's training data.
    """
    dataset = dataset.upper()

    if dataset in _fresh_cache:
        return _fresh_cache[dataset]

    from sklearn.preprocessing import StandardScaler

    train_df, test_df, rul_df = load_data(dataset)
    _, feature_columns, _ = prepare_train_data(train_df, max_rul=service.max_rul)

    trained_features = service.config.get("feature_columns", [])
    available = [f for f in trained_features if f in feature_columns]
    missing   = [f for f in trained_features if f not in feature_columns]

    X_test, y_test, unit_ids = prepare_test_samples(
        test_df, rul_df, available, seq_length=service.seq_length, max_rul=service.max_rul
    )

    if missing:
        zeros = np.zeros((X_test.shape[0], X_test.shape[1], len(missing)), dtype=np.float32)
        X_test = np.concatenate([X_test, zeros], axis=-1)

    use_original_scaler = dataset in service.train_datasets
    if use_original_scaler:
        scaler = service.scaler
    else:
        from data_loader import create_sequences_per_engine
        prepared, _, target = prepare_train_data(train_df, max_rul=service.max_rul)
        X_train_ds, _ = create_sequences_per_engine(prepared, available, target, seq_length=service.seq_length)
        if missing:
            zeros_tr = np.zeros((X_train_ds.shape[0], X_train_ds.shape[1], len(missing)), dtype=np.float32)
            X_train_ds = np.concatenate([X_train_ds, zeros_tr], axis=-1)
        scaler = StandardScaler()
        scaler.fit(X_train_ds.reshape(-1, X_train_ds.shape[-1]))

    X_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=service.device)
    with torch.no_grad():
        predictions = service.model(X_tensor).cpu().numpy().flatten()

    rul_indexed = rul_df.reset_index(names="unit_id").assign(unit_id=lambda df: df["unit_id"] + 1)
    rul_indexed = rul_indexed.rename(columns={"RUL": "RUL_final"})
    prepared_test_df = test_df.copy()
    last_cycle = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    prepared_test_df = prepared_test_df.merge(last_cycle.reset_index(), on="unit_id")
    prepared_test_df["RUL"] = prepared_test_df["max_cycle"] - prepared_test_df["cycle"]
    prepared_test_df = prepared_test_df.merge(rul_indexed, on="unit_id")
    prepared_test_df["RUL"] = (prepared_test_df["max_cycle"] - prepared_test_df["cycle"]) + prepared_test_df["RUL_final"]
    if service.max_rul:
        prepared_test_df["RUL"] = prepared_test_df["RUL"].clip(upper=service.max_rul)

    cache = {
        "dataset": dataset,
        "train_df": train_df,
        "test_df": test_df,
        "rul_df": rul_df,
        "X_test": X_test,
        "X_test_scaled": X_scaled,
        "X_test_tensor": X_tensor,
        "y_test": y_test,
        "unit_ids": unit_ids,
        "predictions": predictions,
        "feature_columns": available,
        "missing_features": missing,
        "prepared_test_df": prepared_test_df,
        "scaler": scaler,
        "use_original_scaler": use_original_scaler,
    }

    _fresh_cache[dataset] = cache
    return cache



@app.get("/api/summary")
def summary() -> Any:
    return jsonify(service.get_summary())


@app.get("/api/history")
def history() -> Any:
    dataset = request.args.get("dataset", service.dataset).upper()
    history = service.get_history()
    return jsonify(history)


@app.get("/api/sample-prediction")
def sample_prediction() -> Any:
    dataset  = request.args.get("dataset", service.dataset).upper()
    engine_id = request.args.get("engineId", type=int)
    use_random = request.args.get("random", "false").lower() == "true"
    try:
        cache = _get_dataset_cache(dataset)
        unit_ids = cache["unit_ids"]
        if use_random:
            index = int(np.random.randint(0, len(unit_ids)))
        elif engine_id is not None:
            matches = np.where(unit_ids == engine_id)[0]
            if len(matches) == 0:
                raise KeyError(f"Engine {engine_id} not found in {dataset}.")
            index = int(matches[0])
        else:
            index = 0

        sequence   = cache["X_test"][index]
        actual_rul = float(cache["y_test"][index])
        result     = service.predict_sequence(sequence, mc_samples=20)
        result.update({
            "engineId":      int(unit_ids[index]),
            "actualRul":     actual_rul,
            "absoluteError": abs(result["predictedRul"] - actual_rul),
            "dataset":       dataset,
            "isFreshData":   dataset != service.dataset,
        })
        return jsonify(result)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/all-predictions")
def all_predictions() -> Any:
    """Returns actual vs predicted RUL for scatter chart. Supports ?dataset=FD00X."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        data = [
            {
                "engineId":    int(uid),
                "actualRul":   float(y),
                "predictedRul": float(p),
                "error":       float(abs(p - y)),
            }
            for uid, y, p in zip(cache["unit_ids"], cache["y_test"], cache["predictions"])
        ]
        rmse = float(np.sqrt(np.mean((cache["predictions"] - cache["y_test"]) ** 2)))
        mae  = float(np.mean(np.abs(cache["predictions"] - cache["y_test"])))
        return jsonify({"dataset": dataset, "rmse": round(rmse,4), "mae": round(mae,4), "engines": data})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/engine-ids")
def engine_ids() -> Any:
    """Returns sorted engine IDs for a given dataset. Supports ?dataset=FD00X."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        uid_list = sorted(int(x) for x in set(cache["unit_ids"]))
        return jsonify(uid_list)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/explorer")
def explorer() -> Any:
    dataset = request.args.get("dataset", service.dataset).upper()
    limit = request.args.get("limit", default=25, type=int) or 25

    try:
        cache = _get_dataset_cache(dataset)
        train_df, test_df, _ = load_data(dataset)
        latest_rows = test_df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)

        prediction_by_engine = {
            int(uid): {"actualRul": float(actual), "predictedRul": float(predicted)}
            for uid, actual, predicted in zip(cache["unit_ids"], cache["y_test"], cache["predictions"])
        }

        explorer_rows: list[dict[str, Any]] = []
        for _, row in latest_rows.head(limit).iterrows():
            engine_id = int(row["unit_id"])
            prediction = prediction_by_engine.get(engine_id, {"actualRul": 0.0, "predictedRul": 0.0})
            predicted_rul = float(prediction["predictedRul"])
            actual_rul = float(prediction["actualRul"])
            health = max(0.0, min(100.0, (predicted_rul / service.max_rul) * 100.0))
            if predicted_rul < 20:
                status = "CRITICAL"
            elif predicted_rul < 60:
                status = "WARNING"
            else:
                status = "NOMINAL"

            engine_history = test_df[test_df["unit_id"] == engine_id].sort_values("cycle")
            trend_source = engine_history["sensor_2"].tail(8).to_numpy(dtype=np.float32)
            if len(trend_source) == 0:
                trend_source = np.zeros(8, dtype=np.float32)
            trend_min = float(trend_source.min())
            trend_max = float(trend_source.max())
            if trend_max - trend_min < 1e-8:
                trend = [50.0 for _ in trend_source]
            else:
                trend = [float(((value - trend_min) / (trend_max - trend_min)) * 100.0) for value in trend_source]

            explorer_rows.append(
                {
                    "engineId": engine_id,
                    "cycle": int(row["cycle"]),
                    "status": status,
                    "t2": float(row["sensor_2"]),
                    "p30": float(row["sensor_3"]),
                    "n1": float(row["sensor_4"]),
                    "vibe": float(row["sensor_11"]),
                    "health": round(health, 1),
                    "actualRul": round(actual_rul, 1),
                    "predictedRul": round(predicted_rul, 1),
                    "error": round(abs(predicted_rul - actual_rul), 1),
                    "trend": trend,
                }
            )

        summary = {
            "dataset": dataset,
            "engineCount": len(explorer_rows),
            "criticalCount": sum(1 for row in explorer_rows if row["status"] == "CRITICAL"),
            "warningCount": sum(1 for row in explorer_rows if row["status"] == "WARNING"),
            "averageHealth": round(float(np.mean([row["health"] for row in explorer_rows])) if explorer_rows else 0.0, 1),
        }

        return jsonify({"summary": summary, "rows": explorer_rows})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/notifications")
def notifications() -> Any:
    """Returns lightweight dashboard alerts derived from current model outputs."""
    dataset = request.args.get("dataset", service.dataset).upper()
    limit = request.args.get("limit", default=5, type=int) or 5

    try:
        cache = _get_dataset_cache(dataset)
        predictions = cache["predictions"]
        unit_ids = cache["unit_ids"]

        alerts: list[dict[str, Any]] = []
        for uid, pred in zip(unit_ids, predictions):
            predicted_rul = float(pred)
            if predicted_rul < 20:
                level = "critical"
                title = f"Engine {int(uid)} critical"
                message = f"Predicted RUL is {predicted_rul:.1f} cycles. Immediate maintenance recommended."
            elif predicted_rul < 60:
                level = "warning"
                title = f"Engine {int(uid)} warning"
                message = f"Predicted RUL is {predicted_rul:.1f} cycles. Schedule maintenance soon."
            else:
                continue

            alerts.append(
                {
                    "id": f"{dataset}-{int(uid)}-{level}",
                    "level": level,
                    "title": title,
                    "message": message,
                    "dataset": dataset,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )

        alerts.sort(key=lambda item: 0 if item["level"] == "critical" else 1)
        alerts = alerts[: max(1, min(limit, 20))]

        if not alerts:
            return jsonify(
                {
                    "dataset": dataset,
                    "unreadCount": 0,
                    "notifications": [],
                    "checkedAt": datetime.now(timezone.utc).isoformat(),
                }
            )

        return jsonify(
            {
                "dataset": dataset,
                "unreadCount": len(alerts),
                "notifications": alerts,
                "checkedAt": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/evaluate-fresh")
def evaluate_fresh() -> Any:
    """
    Runs the trained model against any C-MAPSS dataset.
    Now uses per-dataset normalization for proper cross-dataset evaluation.
    """
    dataset = request.args.get("dataset", "FD002").upper().strip()
    if dataset not in ("FD001", "FD002", "FD003", "FD004"):
        return jsonify({"error": "dataset must be one of FD001, FD002, FD003, FD004"}), 400

    try:
        cache = _get_dataset_cache(dataset)
        y_test = cache["y_test"]
        predictions = cache["predictions"]
        unit_ids = cache["unit_ids"]

        errors = np.abs(predictions - y_test)
        rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
        mae = float(np.mean(errors))
        diffs = predictions - y_test
        nasa = float(np.sum(np.where(diffs < 0, np.exp(-diffs / 13) - 1, np.exp(diffs / 10) - 1)))

        per_engine = [
            {
                "engineId": int(uid),
                "actualRul": float(y),
                "predictedRul": float(p),
                "error": float(abs(p - y)),
            }
            for uid, y, p in zip(unit_ids, y_test, predictions)
        ]

        return jsonify({
            "dataset": dataset,
            "trainedOn": service.dataset,
            "engineCount": len(per_engine),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "nasaScore": round(nasa, 2),
            "missingFeatures": cache.get("missing_features", []),
            "usedFeatures": cache.get("feature_columns", []),
            "predictions": per_engine,
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    sequence = payload.get("sequence")
    mc_samples = int(payload.get("mcSamples", 20))
    if sequence is None:
        return jsonify({"error": "Missing sequence payload."}), 400

    try:
        sequence_array = np.asarray(sequence, dtype=np.float32)
        result = service.predict_sequence(sequence_array, mc_samples=mc_samples)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/engine-history")
def engine_history() -> Any:
    """Returns the full historical telemetry and RUL predictions for a specific engine."""
    engine_id = request.args.get("engineId", type=int)
    dataset = request.args.get("dataset", service.dataset).upper()

    if engine_id is None:
        return jsonify({"error": "Missing engineId parameter"}), 400

    try:
        cache = _get_dataset_cache(dataset)
        prepared_test_df = cache.get("prepared_test_df")

        if prepared_test_df is None:
            return jsonify({"error": f"Prepared test data not available for {dataset}"}), 500

        engine_data = prepared_test_df[prepared_test_df["unit_id"] == engine_id].sort_values("cycle")
        if engine_data.empty:
            return jsonify({"error": f"Engine {engine_id} not found in {dataset}"}), 404

        cycles = engine_data["cycle"].tolist()
        actual_ruls = engine_data["RUL"].tolist()

        feature_cols = cache.get("feature_columns", [])
        raw_test_df = cache.get("test_df")
        raw_engine_data = None
        if raw_test_df is not None and engine_id in raw_test_df["unit_id"].values:
            raw_engine_data = raw_test_df[raw_test_df["unit_id"] == engine_id].sort_values("cycle")

        sensor_data = {}
        for col in feature_cols:
            if raw_engine_data is not None and col in raw_engine_data.columns:
                sensor_data[col] = raw_engine_data[col].tolist()
            elif col in engine_data.columns:
                sensor_data[col] = engine_data[col].tolist()
            else:
                sensor_data[col] = [0.0] * len(cycles)

        seq_length = service.seq_length
        n_cycles = len(cycles)
        step = 1 if n_cycles < 150 else 5
        step_indices = [i for i in range(n_cycles) if i % step == 0 or i == n_cycles - 1]
        n_pred = len(step_indices)

        raw_features = engine_data[feature_cols].to_numpy(dtype=np.float32)
        n_features = len(feature_cols)

        scaler = cache.get("scaler") or service.scaler

        if scaler is not None:
            raw_flat = raw_features.reshape(-1, n_features)
            scaled_flat = scaler.transform(raw_flat).reshape(n_cycles, n_features)
        else:
            scaled_flat = raw_features

        scaled_sequences = []
        for idx in step_indices:
            i = step_indices.index(idx)
            if idx + 1 < seq_length:
                sub_seq = scaled_flat[:idx+1]
                pad_rows = np.repeat(sub_seq[:1], seq_length - len(sub_seq), axis=0)
                sub_seq = np.vstack([pad_rows, sub_seq])
            else:
                sub_seq = scaled_flat[idx-seq_length+1:idx+1]
            scaled_sequences.append((i, idx, sub_seq))

        batch_tensor = torch.tensor(
            np.stack([s[2] for s in scaled_sequences]),
            dtype=torch.float32,
            device=service.device
        )
        with torch.no_grad():
            batch_preds = service.model(batch_tensor).cpu().numpy().flatten()

        predicted_ruls = [None] * n_cycles
        for s, pred in zip(scaled_sequences, batch_preds):
            predicted_ruls[s[1]] = float(pred)

        last_idx = -1
        for i in range(n_cycles):
            if predicted_ruls[i] is not None:
                if last_idx != -1 and i - last_idx > 1:
                    start_val = predicted_ruls[last_idx]
                    end_val = predicted_ruls[i]
                    for j in range(last_idx + 1, i):
                        predicted_ruls[j] = start_val + (end_val - start_val) * (j - last_idx) / (i - last_idx)
                last_idx = i

        return jsonify({
            "engineId": engine_id,
            "dataset": dataset,
            "cycles": cycles,
            "actualRul": actual_ruls,
            "predictedRul": predicted_ruls,
            "sensors": sensor_data
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/artifacts/<path:filename>")
def artifacts(filename: str) -> Any:
    return send_from_directory(MODELS_DIR, filename)


@app.get("/")
def index() -> Any:
    return send_from_directory(DASHBOARD_DIR, "Main_Dashboard.html")


@app.get("/<path:filename>")
def static_pages(filename: str) -> Any:
    return send_from_directory(DASHBOARD_DIR, filename)


@app.get("/api/explain")
def api_explain() -> Any:
    # Use first 50 samples for speed
    X_test_tensor = service._test_cache["X_test_tensor"]
    samples = X_test_tensor[:50].requires_grad_()
    
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(service.model)
    baseline = torch.zeros_like(samples)
    attributions, _ = ig.attribute(samples, baseline, return_convergence_delta=True)
    # Average over batch and sequence dimension
    attr_mean = attributions.mean(dim=0).mean(dim=0).detach().cpu().numpy()
    
    feature_columns = service.config.get("feature_columns") or []
    if len(feature_columns) != len(attr_mean):
        return jsonify({"error": "Feature mismatch"}), 500
        
    importance = {feat: float(abs(val)) for feat, val in zip(feature_columns, attr_mean)}
    
    # Sort by importance and format properly
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return jsonify({
        "features": [i[0] for i in sorted_importance],
        "values": [i[1] for i in sorted_importance]
    })


@app.get("/api/latent-state")
def api_latent_state() -> Any:
    """Returns the hidden state activations of the LSTM for a sample sequence."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        # Use first sample from test set
        X_tensor = cache["X_test_tensor"][:1]
        
        # We need the LSTM output before the FC layer
        # Since we can't easily modify the forward without changing model.py,
        # we'll use a hook or just access the internal layers if possible.
        # Alternatively, just rerun the lstm part here.
        model = service.model
        with torch.no_grad():
            # Initial states
            h0 = torch.zeros(model.num_layers, X_tensor.size(0), model.hidden_size).to(X_tensor.device)
            c0 = torch.zeros(model.num_layers, X_tensor.size(0), model.hidden_size).to(X_tensor.device)
            
            # Forward pass through LSTM layer only
            out, _ = model.lstm(X_tensor, (h0, c0))
            # out shape: [batch, seq_len, hidden_size] -> [1, 50, 64]
            activations = out[0].cpu().numpy().tolist()
            
        return jsonify({
            "activations": activations, # 50 x 64
            "seq_length": service.seq_length,
            "hidden_size": model.hidden_size
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/accuracy-stats")
def api_accuracy_stats() -> Any:
    """Returns RMSE/MAE grouped by RUL buckets to show how accuracy changes near failure."""
    dataset = request.args.get("dataset", service.dataset).upper()
    try:
        cache = _get_dataset_cache(dataset)
        y_true = cache["y_test"]
        y_pred = cache["predictions"]
        
        buckets = [
            (0, 25, "0-25 (Critical)"),
            (25, 50, "25-50 (Late)"),
            (50, 80, "50-80 (Mid)"),
            (80, 125, "80+ (Early)")
        ]
        
        stats = []
        for low, high, label in buckets:
            mask = (y_true >= low) & (y_true < high)
            if np.any(mask):
                true_b = y_true[mask]
                pred_b = y_pred[mask]
                rmse = float(np.sqrt(np.mean((true_b - pred_b)**2)))
                mae = float(np.mean(np.abs(true_b - pred_b)))
                count = int(np.sum(mask))
            else:
                rmse, mae, count = 0.0, 0.0, 0
            stats.append({"label": label, "rmse": rmse, "mae": mae, "count": count})
            
        return jsonify(stats)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True,
        # Keep reloader off in this script entrypoint to avoid SystemExit
        # under debugpy/VS Code and single-process launcher confusion.
        use_reloader=False,
    )
