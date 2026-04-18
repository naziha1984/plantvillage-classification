import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fail(msg: str) -> None:
    print(f"[ERREUR] {msg}")
    sys.exit(1)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/PlantVillage")

    if not data_dir.exists() or not data_dir.is_dir():
        fail(f"DATA_DIR n'existe pas ou n'est pas un dossier : {data_dir}")

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        fail(
            f"Aucune classe détectée dans {data_dir}. "
            "Ajoutez au moins un sous-dossier de classe contenant des images."
        )

    counts = {}
    total_images = 0

    for d in sorted(class_dirs, key=lambda x: x.name.lower()):
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

    min_class, min_count = min(counts.items(), key=lambda item: item[1])
    max_class, max_count = max(counts.items(), key=lambda item: item[1])
    is_imbalanced = max_count > 2 * min_count

    sep = "=" * 64
    print(sep)
    print("RAPPORT DE VÉRIFICATION DU DATASET")
    print(sep)
    print(f"Dossier analysé      : {data_dir}")
    print(f"Nombre de classes    : {len(class_dirs)}")
    print(f"Nombre total d'images: {total_images}")
    print()
    print("-" * 64)
    print("IMAGES PAR CLASSE")
    print("-" * 64)
    for cls in sorted(counts):
        print(f"{cls:<40} {counts[cls]:>6}")
    print()
    print("-" * 64)
    print("STATISTIQUES CLÉS")
    print("-" * 64)
    print(f"Classe la plus petite : {min_class} ({min_count} images)")
    print(f"Classe la plus grande : {max_class} ({max_count} images)")
    print(
        "Déséquilibre détecté  : "
        + ("Oui (max > 2 x min)" if is_imbalanced else "Non")
    )
    print(sep)


if __name__ == "__main__":
    main()

