
from pathlib import Path
from collections import Counter, defaultdict
import re

DATASET = Path("ml/dataset")
EXTENSIONES = {".jpg", ".jpeg", ".png"}

grupos = defaultdict(list)

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    for imagen in carpeta.iterdir():
        if imagen.suffix.lower() not in EXTENSIONES:
            continue

        nombre = imagen.name.split(".rf.")[0]

        if nombre.startswith("frame_"):
            grupo = "frame"
        elif re.match(r"^G\d+", nombre):
            grupo = "G"
        elif nombre.startswith("cracks-and-potholes"):
            grupo = "cracks-and-potholes"
        else:
            grupo = nombre.split("_")[0]

        grupos[grupo].append((split, imagen.name))

print("\nANALISIS DE SECUENCIAS")
print("-" * 50)

for grupo, archivos in sorted(
    grupos.items(),
    key=lambda item: len(item[1]),
    reverse=True
):
    distribucion = Counter(
        split for split, _ in archivos
    )

    print(
        f"{grupo}: {len(archivos)} imagenes | "
        f"Train={distribucion['train']} | "
        f"Valid={distribucion['valid']} | "
        f"Test={distribucion['test']}"
    )

print("\nTOTAL DE GRUPOS:", len(grupos))
