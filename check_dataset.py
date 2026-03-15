import sys
from pathlib import Path
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fail(msg: str) -> None:
    print(f"[ERREUR] {msg}")
    sys.exit(1)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/PlantVillage")

    if not data_dir.exists() or not data_dir.is_dir():
        fail(f"DATA_DIR n'existe pas ou n'est pas un dossier : {data_dir}")

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(class_dirs) != 38:
        fail(
            f"DATA_DIR doit contenir exactement 38 sous-dossiers (classes), trouvé {len(class_dirs)}."
        )

    class_names = sorted(d.name for d in class_dirs)
    print(f"Nombre de classes : {len(class_names)}")
    print("Quelques classes (5) :", ", ".join(class_names[:5]))

    counts = {}
    total_images = 0

    for d in class_dirs:
        imgs = [
            p
            for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not imgs:
            fail(
                f"La classe '{d.name}' ne contient aucune image valide (extensions acceptées : {IMAGE_EXTS})."
            )
        counts[d.name] = len(imgs)
        total_images += len(imgs)

    print(f"Nombre total d'images : {total_images}")

    top10 = Counter(counts).most_common(10)
    print("Top 10 des classes par nombre d'images :")
    for cls, n in top10:
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()

