
from pathlib import Path
from itertools import combinations
from PIL import Image
import imagehash

DATASET = Path("ml/dataset")
EXTENSIONES = {".jpg", ".jpeg", ".png"}

imagenes = []

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    for ruta in carpeta.iterdir():
        if ruta.suffix.lower() not in EXTENSIONES:
            continue

        try:
            with Image.open(ruta) as img:
                phash = imagehash.phash(img)

            imagenes.append((split, ruta.name, phash))

        except Exception as e:
            print(f"Error leyendo {ruta.name}: {e}")

print(f"Imagenes analizadas: {len(imagenes)}")

similares = []

for (split1, nombre1, hash1), (
    split2, nombre2, hash2
) in combinations(imagenes, 2):

    if split1 == split2:
        continue

    distancia = hash1 - hash2

    if distancia <= 5:
        similares.append(
            (split1, nombre1, split2, nombre2, distancia)
        )

similares.sort(key=lambda x: x[4])

print("\nANALISIS DE SIMILITUD VISUAL")
print("-" * 45)
print(f"Pares sospechosos: {len(similares)}")

for s1, n1, s2, n2, distancia in similares[:20]:
    print(
        f"{s1} / {n1}  <->  "
        f"{s2} / {n2} | Distancia: {distancia}"
    )
