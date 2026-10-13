
from pathlib import Path
import math

DATASET = Path("ml/dataset")
EXTENSIONES = {".jpg", ".jpeg", ".png"}

total_imagenes = 0
total_cajas = 0
total_vacias = 0
errores = []

for split in ["train", "valid", "test"]:
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    imagenes = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in EXTENSIONES
    ]

    etiquetas = list(labels_dir.glob("*.txt"))
    nombres_imagenes = {p.stem for p in imagenes}

    cajas = 0
    vacias = 0

    for imagen in imagenes:
        etiqueta = labels_dir / f"{imagen.stem}.txt"

        if not etiqueta.exists():
            errores.append(f"{split}: falta {etiqueta.name}")
            continue

        lineas = etiqueta.read_text(
            encoding="utf-8"
        ).splitlines()

        if not lineas:
            vacias += 1

        for numero, linea in enumerate(lineas, 1):
            try:
                valores = list(map(float, linea.split()))

                if len(valores) != 5:
                    raise ValueError("Formato incorrecto")

                cls, x, y, w, h = valores

                if not all(math.isfinite(v) for v in valores):
                    raise ValueError("Valor no finito")

                if cls != 0:
                    raise ValueError("Clase incorrecta")

                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("Centro fuera de rango")

                if not (0 < w <= 1 and 0 < h <= 1):
                    raise ValueError("Dimensiones invalidas")

                tol = 1e-8

                if (
                    x - w / 2 < -tol
                    or y - h / 2 < -tol
                    or x + w / 2 > 1 + tol
                    or y + h / 2 > 1 + tol
                ):
                    raise ValueError("Caja fuera de imagen")

                cajas += 1

            except ValueError as e:
                errores.append(
                    f"{split}/{etiqueta.name}:"
                    f"{numero} - {e}"
                )

    for etiqueta in etiquetas:
        if etiqueta.stem not in nombres_imagenes:
            errores.append(
                f"{split}: etiqueta sin imagen: {etiqueta.name}"
            )

    total_imagenes += len(imagenes)
    total_cajas += cajas
    total_vacias += vacias

    print(f"\n{split.upper()}")
    print(f"Imagenes: {len(imagenes)}")
    print(f"Etiquetas: {len(etiquetas)}")
    print(f"Cajas validas: {cajas}")
    print(f"Etiquetas vacias: {vacias}")

print("\nRESUMEN FINAL")
print("-" * 40)
print(f"Total imagenes: {total_imagenes}")
print(f"Total cajas validas: {total_cajas}")
print(f"Etiquetas vacias: {total_vacias}")
print(f"Errores restantes: {len(errores)}")

for error in errores[:30]:
    print(error)
