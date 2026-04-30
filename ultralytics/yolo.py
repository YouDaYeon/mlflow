import os

from ultralytics import YOLO, settings


def configure_tracking() -> None:
    """Enable Ultralytics -> MLflow integration."""
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",  # mlflow server --backend-store-uri runs/mlflow
    )
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "ultralytics-yolo")
    os.environ.setdefault("MLFLOW_RUN", "yolo26n-exp1")

    # Keep this enabled so Ultralytics logs params/metrics/artifacts to MLflow.
    settings.update({"mlflow": True})


def main() -> None:
    configure_tracking()

    # Load a pretrained YOLO model (recommended for training).
    model = YOLO("yolo26n.pt")

    # Train: run name/project make tracking outputs easy to distinguish.
    model.train(
        data="coco8.yaml",   # cfg/datasets/coco8.yaml
        epochs=3,
        imgsz=640,
        project="runs/train",
        name="yolo26n-exp1",
    )

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