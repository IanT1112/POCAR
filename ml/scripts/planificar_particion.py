
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from PIL import Image
import imagehash
import random

DATASET = Path("ml/dataset")
UMBRAL = 5
SEMILLA = 42

registros = []

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        with Image.open(ruta) as img:
            huella = imagehash.phash(img.convert("RGB"))

        registros.append((split, ruta, huella))

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

for i, j in combinations(range(len(registros)), 2):
    if registros[i][2] - registros[j][2] <= UMBRAL:
        unir(i, j)

grupos = defaultdict(list)

for i in range(len(registros)):
    grupos[encontrar(i)].append(i)

lista_grupos = list(grupos.values())

random.seed(SEMILLA)
random.shuffle(lista_grupos)
lista_grupos.sort(key=len, reverse=True)

total = len(registros)

objetivos = {
    "train": round(total * 0.70),
    "valid": round(total * 0.20),
}
objetivos["test"] = total - sum(objetivos.values())

asignacion = defaultdict(list)

for grupo in lista_grupos:
    destino = max(
        objetivos,
        key=lambda split: objetivos[split] - len(asignacion[split])
    )

    asignacion[destino].extend(grupo)

print("\nSIMULACION DE PARTICION")
print("-" * 45)
print(f"Total de imagenes: {total}")
print(f"Grupos de similitud: {len(lista_grupos)}")

for split in ["train", "valid", "test"]:
    cantidad = len(asignacion[split])
    porcentaje = cantidad / total * 100

    print(
        f"{split.upper()}: {cantidad} imagenes "
        f"({porcentaje:.2f}%) | "
        f"Objetivo: {objetivos[split]}"
    )

print("\nNo se han movido archivos.")
