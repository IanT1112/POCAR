
from pathlib import Path
from itertools import combinations
from PIL import Image, ImageDraw, ImageOps
import imagehash

DATASET = Path("ml/dataset")
SALIDA = Path("ml/images/comparacion_similares.jpg")

registros = []

for split in ["train", "valid", "test"]:
    carpeta = DATASET / split / "images"

    for archivo in carpeta.iterdir():
        if archivo.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        with Image.open(archivo) as img:
            huella = imagehash.phash(img)

        registros.append((split, archivo, huella))

pares = []

for a, b in combinations(registros, 2):
    if a[0] == b[0]:
        continue

    distancia = a[2] - b[2]

    if distancia <= 5:
        pares.append((distancia, a, b))

pares.sort(key=lambda x: x[0])

seleccionados = pares[:12]

ANCHO = 800
ALTO_FILA = 260
ANCHO_IMAGEN = 390
ALTO_IMAGEN = 220

mosaico = Image.new(
    "RGB",
    (ANCHO, ALTO_FILA * len(seleccionados)),
    "white"
)

dibujo = ImageDraw.Draw(mosaico)

for indice, (distancia, a, b) in enumerate(seleccionados):
    y = indice * ALTO_FILA

    for columna, registro in enumerate([a, b]):
        split, ruta, _ = registro

        with Image.open(ruta) as imagen:
            imagen = ImageOps.fit(
                imagen.convert("RGB"),
                (ANCHO_IMAGEN, ALTO_IMAGEN)
            )

        x = columna * 400 + 5
        mosaico.paste(imagen, (x, y + 25))

        dibujo.text(
            (x, y + 5),
            f"{split.upper()} | {ruta.name[:28]}",
            fill="black"
        )

    dibujo.text(
        (10, y + 247),
        f"Par {indice + 1} | Distancia pHash: {distancia}",
        fill="black"
    )

SALIDA.parent.mkdir(parents=True, exist_ok=True)
mosaico.save(SALIDA, quality=90)

print("\nCOMPARACION GENERADA")
print(f"Pares encontrados: {len(pares)}")
print(f"Pares visualizados: {len(seleccionados)}")
print(f"Archivo: {SALIDA.resolve()}")
