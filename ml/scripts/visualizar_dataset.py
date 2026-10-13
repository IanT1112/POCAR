
from pathlib import Path
import random
import cv2
import numpy as np

DATASET = Path("ml/dataset")
SALIDA = Path("ml/images/muestra_dataset.jpg")

random.seed(42)

imagenes = []

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    archivos = [
        p for p in carpeta.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    seleccion = random.sample(archivos, min(4, len(archivos)))

    for ruta in seleccion:
        imagen = cv2.imread(str(ruta))

        if imagen is None:
            print(f"No se pudo abrir: {ruta}")
            continue

        alto, ancho = imagen.shape[:2]

        etiqueta = (
            DATASET / split / "labels" / f"{ruta.stem}.txt"
        )

        cantidad = 0

        for linea in etiqueta.read_text(
            encoding="utf-8"
        ).splitlines():
            cls, x, y, w, h = map(float, linea.split())

            x1 = int((x - w / 2) * ancho)
            y1 = int((y - h / 2) * alto)
            x2 = int((x + w / 2) * ancho)
            y2 = int((y + h / 2) * alto)

            cv2.rectangle(
                imagen, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cantidad += 1

        imagen = cv2.resize(imagen, (320, 240))

        cv2.putText(
            imagen,
            f"{split.upper()} | Baches: {cantidad}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        imagenes.append(imagen)

if len(imagenes) != 12:
    raise RuntimeError(
        f"Se esperaban 12 imagenes y se obtuvieron {len(imagenes)}"
    )

filas = []

for i in range(0, 12, 3):
    fila = np.hstack(imagenes[i:i + 3])
    filas.append(fila)

mosaico = np.vstack(filas)

SALIDA.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(SALIDA), mosaico)

print("Visualizacion generada correctamente")
print(f"Archivo: {SALIDA.resolve()}")
