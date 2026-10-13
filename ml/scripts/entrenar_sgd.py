
from pathlib import Path
from multiprocessing import freeze_support

import torch
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[2]

    dataset = root / "ml/dataset_preparado/data.yaml"
    modelo = root / "ml/models/yolo11n.pt"
    resultados = root / "ml/runs/train"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    print("=" * 50)
    print("POCAR - EXPERIMENTO 2: SGD")
    print("=" * 50)

    model = YOLO(str(modelo))

    model.train(
        data=str(dataset),
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        patience=10,
        seed=42,
        deterministic=True,
        workers=0,
        cache=False,
        mosaic=0.0,
        amp=True,
        plots=True,
        project=str(resultados),
        name="yolo11n_sgd",
        exist_ok=False,
    )

    print("\nEXPERIMENTO SGD FINALIZADO")


if __name__ == "__main__":
    freeze_support()
    main()
