from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .database import Database, utc_now
from .dataset import build_yolo_training_dataset


@dataclass
class TrainingPolicy:
    minimum_verified: int = 20
    minimum_classes: int = 2
    epochs: int = 10
    image_size: int = 640
    patience: int = 4
    minimum_map_improvement: float = 0.0
    base_model: str = "yolo11n.pt"


class TrainingCoordinator:
    """Frequently fine-tunes a detector using complete, diverse keyframes."""

    SETTINGS_KEY = "training_policy"

    def __init__(self, db: Database, data_dir: str | Path):
        self.db = db
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        saved = db.get_setting(self.SETTINGS_KEY, {})
        defaults = asdict(TrainingPolicy())
        saved_policy = (
            {key: value for key, value in saved.items() if key in defaults}
            if isinstance(saved, dict)
            else {}
        )
        self.policy = TrainingPolicy(**{**defaults, **saved_policy})
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._message = "Idle"
        self._progress = 0

    def update_policy(self, **changes: Any) -> None:
        values = asdict(self.policy)
        values.update({key: value for key, value in changes.items() if key in values})
        self.policy = TrainingPolicy(**values)
        self.db.set_setting(self.SETTINGS_KEY, asdict(self.policy))

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = bool(self._thread and self._thread.is_alive())
            return {"running": running, "message": self._message, "progress": self._progress, "policy": asdict(self.policy)}

    def _set_status(self, message: str, progress: int) -> None:
        with self._lock:
            self._message = message
            self._progress = max(0, min(100, progress))

    def readiness(self) -> dict[str, Any]:
        stats = self.db.training_stats()
        last_revision = int(self.db.get_setting("last_trained_dataset_revision", 0) or 0)
        new_changes = max(0, stats["revision"] - last_revision)
        can_train = (
            stats["examples"] >= self.policy.minimum_verified
            and stats["classes"] >= self.policy.minimum_classes
        )
        return {**stats, "new_changes": new_changes, "can_train": can_train}

    def request_training(self, *, reason: str = "manual request") -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
        stats = self.db.training_stats()
        if stats["examples"] < self.policy.minimum_verified or stats["classes"] < self.policy.minimum_classes:
            self._set_status(
                f"Need at least {self.policy.minimum_verified} selected training boxes across {self.policy.minimum_classes} species",
                0,
            )
            return False
        active = self.db.active_model()
        base_model = active["weights_path"] if active and Path(active["weights_path"]).is_file() else self.policy.base_model
        run = self.db.create_training_run(reason, stats["examples"], base_model)
        thread = threading.Thread(target=self._train, args=(run["id"], base_model, stats), daemon=True, name="finframe-training")
        with self._lock:
            self._thread = thread
        thread.start()
        return True

    @staticmethod
    def _metric_from_results(results: Any) -> float | None:
        candidates = []
        if hasattr(results, "results_dict"):
            candidates.append(results.results_dict)
        if isinstance(results, dict):
            candidates.append(results)
        for metrics in candidates:
            for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95"):
                if key in metrics:
                    try:
                        return float(metrics[key])
                    except (TypeError, ValueError):
                        pass
        return None

    def _train(self, run_id: str, base_model: str, stats: dict[str, int]) -> None:
        run_dir = self.data_dir / "training" / run_id
        dataset_dir = run_dir / "dataset"
        output_dir = run_dir / "runs"
        self.db.update_training_run(run_id, status="running", started_at=utc_now(), dataset_path=str(dataset_dir))
        self._set_status("Building dataset from complete, selected keyframes", 10)
        try:
            dataset = build_yolo_training_dataset(self.db, dataset_dir)
            self._set_status(f"Fine-tuning on {dataset['examples']} reviewed boxes", 25)
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install the 'training' dependencies to enable detector retraining") from exc
            model = YOLO(base_model)
            results = model.train(
                data=str(dataset["yaml"]),
                epochs=self.policy.epochs,
                imgsz=self.policy.image_size,
                patience=self.policy.patience,
                project=str(output_dir),
                name="finframe",
                exist_ok=True,
                verbose=False,
            )
            self._set_status("Evaluating candidate model", 90)
            metric = self._metric_from_results(results)
            trainer_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", output_dir / "finframe"))
            best_weights = trainer_dir / "weights" / "best.pt"
            if not best_weights.is_file():
                raise RuntimeError("Training completed without producing best.pt")
            model_dir = self.data_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            version = self.db.next_model_version()
            destination = model_dir / f"finframe_v{version:04d}.pt"
            shutil.copy2(best_weights, destination)
            candidate = self.db.register_model(str(destination), metric, stats["examples"], run_id)
            active = self.db.active_model()
            active_metric = active.get("map50_95") if active else None
            should_activate = active is None or (
                metric is not None
                and active_metric is not None
                and metric >= float(active_metric) + self.policy.minimum_map_improvement
            )
            if active is not None and active_metric is None and metric is not None:
                should_activate = True
            if should_activate:
                self.db.activate_model(candidate["id"])
                outcome = "activated"
            else:
                self.db.reject_model(candidate["id"], "Candidate did not improve held-out mAP50-95")
                outcome = "retained previous model"
            metrics = {"map50_95": metric, "outcome": outcome, "dataset": {key: value for key, value in dataset.items() if key not in {"path", "yaml", "species"}}}
            self.db.update_training_run(run_id, status="completed", metrics_json=json.dumps(metrics), completed_at=utc_now())
            self.db.set_setting("last_trained_verified_count", stats["examples"])
            self.db.set_setting("last_trained_dataset_revision", stats["revision"])
            self._set_status(f"Training complete — {outcome}", 100)
        except Exception as exc:
            self.db.update_training_run(run_id, status="failed", error=str(exc), completed_at=utc_now())
            self._set_status(f"Training failed: {exc}", 0)
