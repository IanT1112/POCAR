
from pathlib import Path
import shutil

DATASET = Path("ml/dataset")
BACKUP = DATASET / "labels_original"

corregidas = 0
archivos_modificados = 0

for split in ["train", "valid", "test"]:
    labels_dir = DATASET / split / "labels"
    backup_dir = BACKUP / split

    backup_dir.mkdir(parents=True, exist_ok=True)

    for label in labels_dir.glob("*.txt"):
        nuevas_lineas = []
        cambios_archivo = 0

        for line in label.read_text(
            encoding="utf-8"
        ).splitlines():

            valores = line.split()

            if len(valores) != 5:
                nuevas_lineas.append(line)
                continue

            cls, x, y, w, h = map(float, valores)

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            exceso = max(
                0, -x1, -y1, x2 - 1, y2 - 1
            )

            # Solo corregir desviaciones pequeñas.
            if 0 < exceso <= 0.0001:
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))

                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                nuevas_lineas.append(
                    f"{int(cls)} {x:.10f} {y:.10f} "
                    f"{w:.10f} {h:.10f}"
                )
                cambios_archivo += 1
            else:
                nuevas_lineas.append(line)

        if cambios_archivo:
            destino_backup = backup_dir / label.name

            if not destino_backup.exists():
                shutil.copy2(label, destino_backup)

            label.write_text(
                "\n".join(nuevas_lineas) + "\n",
                encoding="utf-8"
            )

            corregidas += cambios_archivo
            archivos_modificados += 1

print("\nRESULTADOS DE LIMPIEZA")
print("-" * 40)
print(f"Anotaciones corregidas: {corregidas}")
print(f"Archivos modificados: {archivos_modificados}")
print(f"Respaldo: {BACKUP}")
