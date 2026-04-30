"""Minimal YOLO train script: edit TRAIN_HYP for optimizer, schedule, aug, and loss gains."""

import os
from typing import Any

from ultralytics import YOLO, settings

# Hyperparameters passed to model.train() (see ultralytics/cfg/default.yaml "Hyperparameters" section).
TRAIN_HYP: dict[str, Any] = {
    # Optimizer & LR schedule
    "optimizer": "AdamW",  # SGD, AdamW, Adam, NAdam, …
    "lr0": 0.001,  # Adam/AdamW typical start; SGD often 0.01
    "lrf": 0.01,  # final LR = lr0 * lrf (OneCycle tail)
    "momentum": 0.937,  # SGD momentum / Adam beta1
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
    "patience": 50,  # early stopping on val plateau
    # Batch / loader (batch=-1 = AutoBatch)
    "batch": 16,
    "workers": 8,
    # Augmentation (raise for harder regularization on large data; lower on tiny sets like coco8)
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10,
    # Loss gains (detect)
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}

# Optional: full default hyp YAML from Ultralytics, then TRAIN_HYP still overrides overlapping keys.
# Example: HYP_YAML = "path/to/custom_hyp.yaml"
HYP_YAML: str | None = None


def configure_tracking() -> None:
    """Enable Ultralytics -> MLflow integration."""
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",  # mlflow server --backend-store-uri runs/mlflow
    )
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "ultralytics-yolo-hpo")
    os.environ.setdefault("MLFLOW_RUN", "yolo26n-hpo")

    # Keep this enabled so Ultralytics logs params/metrics/artifacts to MLflow.
    settings.update({"mlflow": True})


def main() -> None:
    configure_tracking()

    # Load a pretrained YOLO model (recommended for training).
    model = YOLO("yolo26n.pt")

    # Train: run name/project make tracking outputs easy to distinguish.
    train_kw: dict[str, Any] = {
        "data": "coco8.yaml",  # bundled cfg/datasets/coco8.yaml
        "epochs": 3,
        "imgsz": 640,
        "project": "runs/train",
        "name": "yolo26n-hpo",
        **TRAIN_HYP,
    }
    if HYP_YAML:
        train_kw["hyp"] = HYP_YAML
    model.train(**train_kw)

    # Validate and predict; these stages are also tracked when mlflow is enabled.
    model.val()

    # Perform inference on 'bus.jpg'
    results = model("bus.jpg")  
    if results and hasattr(results[0], "save"):
        save_dir = (
            model.trainer.save_dir
            if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir")
            else "runs/train"
        )
        out_path = os.path.join(save_dir, f"predict_{os.path.basename(results[0].path)}")
        results[0].save(filename=out_path)
 

    # # Export model artifact.
    # model.export(format="onnx")


if __name__ == "__main__":
    main()