
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from PIL import Image
import imagehash
import random
import shutil
import csv

ORIGEN = Path("ml/dataset")
DESTINO = Path("ml/dataset_preparado")
UMBRAL = 5
SEMILLA = 42

registros = []

for split in ["train", "valid", "test"]:
    carpeta = ORIGEN / split / "images"

    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        with Image.open(ruta) as img:
            huella = imagehash.phash(img.convert("RGB"))

        etiqueta = ORIGEN / split / "labels" / (
            ruta.stem + ".txt"
        )

        if not etiqueta.exists():
            raise FileNotFoundError(etiqueta)

        registros.append({
            "split_original": split,
            "imagen": ruta,
            "etiqueta": etiqueta,
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

# Construir grupos de similitud.
for i, j in combinations(range(len(registros)), 2):
    if registros[i]["hash"] - registros[j]["hash"] <= UMBRAL:
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
    "valid": round(total * 0.20)
}
objetivos["test"] = total - sum(objetivos.values())

asignacion = defaultdict(list)

for grupo in lista_grupos:
    split = max(
        objetivos,
        key=lambda s: objetivos[s] - len(asignacion[s])
    )
    asignacion[split].extend(grupo)

# Evitar mezclar archivos de una ejecucion anterior.
if DESTINO.exists():
    if any(DESTINO.iterdir()):
        raise RuntimeError(
            f"{DESTINO} ya contiene archivos. "
            "Revisa su contenido antes de continuar."
        )

for split in ["train", "valid", "test"]:
    (DESTINO / split / "images").mkdir(
        parents=True, exist_ok=True
    )
    (DESTINO / split / "labels").mkdir(
        parents=True, exist_ok=True
    )

manifiesto = []

for split, indices in asignacion.items():
    for i in indices:
        registro = registros[i]
        imagen = registro["imagen"]
        etiqueta = registro["etiqueta"]

        nueva_imagen = DESTINO / split / "images" / imagen.name
        nueva_etiqueta = DESTINO / split / "labels" / etiqueta.name

        if nueva_imagen.exists() or nueva_etiqueta.exists():
            raise RuntimeError(
                f"Nombre duplicado detectado: {imagen.name}"
            )

        shutil.copy2(imagen, nueva_imagen)
        shutil.copy2(etiqueta, nueva_etiqueta)

        manifiesto.append({
            "imagen": imagen.name,
            "origen": registro["split_original"],
            "destino": split,
            "grupo": encontrar(i)
        })

# Crear archivo de configuracion YOLO.
yaml = f"""path: {DESTINO.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: pothole
"""

(DESTINO / "data.yaml").write_text(
    yaml, encoding="utf-8"
)

# Guardar trazabilidad de la nueva particion.
with (DESTINO / "particion.csv").open(
    "w", newline="", encoding="utf-8"
) as archivo:
    writer = csv.DictWriter(
        archivo,
        fieldnames=["imagen", "origen", "destino", "grupo"]
    )
    writer.writeheader()
    writer.writerows(manifiesto)

print("\nDATASET PREPARADO")
print("-" * 45)

for split in ["train", "valid", "test"]:
    cantidad = len(asignacion[split])
    print(f"{split.upper()}: {cantidad} imagenes")

print(f"\nTotal: {total}")
print(f"Grupos de similitud: {len(lista_grupos)}")
print(f"Ubicacion: {DESTINO.resolve()}")
print("Dataset original conservado.")
