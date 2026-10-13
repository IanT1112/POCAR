
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from PIL import Image
import imagehash
import csv

DATASET = Path("ml/dataset")
SALIDA = Path("ml/runs/auditoria_similitud.csv")
EXTENSIONES = {".jpg", ".jpeg", ".png"}
UMBRAL = 5

registros = []

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue

        with Image.open(ruta) as imagen:
            huella = imagehash.phash(imagen.convert("RGB"))

        registros.append({
            "split": split,
            "ruta": str(ruta),
            "nombre": ruta.name,
            "hash": huella
        })

padres = list(range(len(registros)))

def encontrar(i):
    while padres[i] != i:
        padres[i] = padres[padres[i]]
        i = padres[i]
    return i

def unir(a, b):
    raiz_a = encontrar(a)
    raiz_b = encontrar(b)

    if raiz_a != raiz_b:
        padres[raiz_b] = raiz_a

# Comparamos todos los pares, incluidos los del mismo split.
for i, j in combinations(range(len(registros)), 2):
    distancia = registros[i]["hash"] - registros[j]["hash"]

    if distancia <= UMBRAL:
        unir(i, j)

grupos = defaultdict(list)

for i, registro in enumerate(registros):
    grupos[encontrar(i)].append(registro)

grupos_mixtos = [
    grupo for grupo in grupos.values()
    if len({r["split"] for r in grupo}) > 1
]

SALIDA.parent.mkdir(parents=True, exist_ok=True)

with SALIDA.open("w", newline="", encoding="utf-8") as archivo:
    writer = csv.writer(archivo)
    writer.writerow(["grupo", "split", "imagen"])

    for numero, grupo in enumerate(grupos_mixtos, 1):
        for registro in grupo:
            writer.writerow([
                numero,
                registro["split"],
                registro["nombre"]
            ])

print("\nAUDITORIA DE SIMILITUD")
print("-" * 45)
print(f"Imagenes analizadas: {len(registros)}")
print(f"Grupos de similitud: {len(grupos)}")
print(f"Grupos que cruzan particiones: {len(grupos_mixtos)}")
print(
    "Imagenes en grupos mixtos:",
    sum(len(g) for g in grupos_mixtos)
)

print("\n10 GRUPOS MIXTOS MAS GRANDES")

for numero, grupo in enumerate(
    sorted(grupos_mixtos, key=len, reverse=True)[:10], 1
):
    distribucion = Counter(r["split"] for r in grupo)
    print(
        f"Grupo {numero}: {len(grupo)} imagenes | "
        f"Train={distribucion['train']} | "
        f"Valid={distribucion['valid']} | "
        f"Test={distribucion['test']}"
    )

print(f"\nReporte guardado: {SALIDA.resolve()}")
