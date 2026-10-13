
from pathlib import Path
from multiprocessing import freeze_support

import torch
from ultralytics import YOLO


def main():
    ROOT = Path(__file__).resolve().parents[2]

    DATASET = ROOT / "ml" / "dataset_preparado" / "data.yaml"
    MODELO = ROOT / "ml" / "models" / "yolo11n.pt"
    RESULTADOS = ROOT / "ml" / "runs" / "train"

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATASET}")

    if not MODELO.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODELO}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    print("=" * 50)
    print("POCAR - ENTRENAMIENTO YOLO11n")
    print("=" * 50)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATASET)
    print("Modelo:", MODELO)

    model = YOLO(str(MODELO))

    model.train(
        data=str(DATASET),
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        seed=42,
        deterministic=True,
        workers=0,
        amp=True,
        plots=True,
        project=str(RESULTADOS),
        name="yolo11n_base",
        exist_ok=False,
        cache=False,
        mosaic=0.0,
    )

    print("\nENTRENAMIENTO FINALIZADO")


if __name__ == "__main__":
    freeze_support()
    main()
