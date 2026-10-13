
from pathlib import Path
from collections import defaultdict
import hashlib

DATASET = Path("ml/dataset")
EXTENSIONES = {".jpg", ".jpeg", ".png"}

hashes = defaultdict(list)
total = 0

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    for imagen in carpeta.iterdir():
        if imagen.suffix.lower() not in EXTENSIONES:
            continue

        sha = hashlib.sha256()

        with open(imagen, "rb") as archivo:
            for bloque in iter(
                lambda: archivo.read(1024 * 1024), b""
            ):
                sha.update(bloque)

        hashes[sha.hexdigest()].append(
            (split, imagen.name)
        )
        total += 1

duplicados = {
    h: archivos
    for h, archivos in hashes.items()
    if len(archivos) > 1
}

duplicados_cruzados = {
    h: archivos
    for h, archivos in duplicados.items()
    if len({split for split, _ in archivos}) > 1
}

print("\nANALISIS DE DUPLICADOS")
print("-" * 45)
print(f"Imagenes analizadas: {total}")
print(f"Hashes unicos: {len(hashes)}")
print(f"Grupos duplicados exactos: {len(duplicados)}")
print(
    "Grupos duplicados entre particiones:",
    len(duplicados_cruzados)
)

if duplicados_cruzados:
    print("\nDUPLICADOS ENTRE PARTICIONES")

    for archivos in list(
        duplicados_cruzados.values()
    )[:20]:
        for split, nombre in archivos:
            print(f"{split}: {nombre}")
        print("-" * 30)
else:
    print("\nNo se encontraron duplicados exactos entre particiones.")
