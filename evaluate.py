"""
Évaluation du modèle PlantVillage — affichage clair pour rendu TP.
"""
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="white")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _short_name(name: str, max_len: int = 22) -> str:
    """Réduit le nom de classe pour l'affichage (matrice de confusion)."""
    s = name.replace("___", "_").replace("__", "_")
    return s[:max_len] + "…" if len(s) > max_len else s


def main() -> None:
    set_seed(42)

    data_dir = Path("data/PlantVillage")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    img_size = 224
    test_split = 0.1

    model_path = output_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Lance d'abord train.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    num_classes = len(dataset.classes)
    classes = dataset.classes

    indices = list(range(len(dataset)))
    _, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=42,
        shuffle=True,
        stratify=dataset.targets,
    )
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    idx_to_class = torch.load(output_dir / "idx_to_class.pt")
    target_names = [idx_to_class[i] for i in range(num_classes)]

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=target_names, output_dict=True
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    n_test = len(all_labels)

    # ---------- Affichage clair (console) ----------
    sep = "=" * 60
    print()
    print(sep)
    print("  RÉSULTATS D'ÉVALUATION — Classification PlantVillage (TP)")
    print(sep)
    print()
    print("  Dataset      : PlantVillage (ImageFolder)")
    print("  Modèle       : ResNet18 pré-entraîné ImageNet, head fine-tunée")
    print("  Appareil     :", device)
    print("  Classes      :", num_classes)
    print("  Échantillons test :", n_test)
    print()
    print("-" * 60)
    print("  MÉTRIQUES GLOBALES")
    print("-" * 60)
    print("  Accuracy (test)  : {:.4f}  ({:.2f} %)".format(acc, acc * 100))
    print("  Macro F1         : {:.4f}".format(macro_f1))
    print("  Weighted F1      : {:.4f}".format(weighted_f1))
    print()
    print("-" * 60)
    print("  CLASSIFICATION REPORT (precision / recall / F1 par classe)")
    print("-" * 60)
    print(report)
    print(sep)
    print()

    # ---------- Fichiers de sortie ----------
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Classification PlantVillage — Rapport détaillé\n")
        f.write("=" * 50 + "\n\n")
        f.write("Accuracy (test): {:.4f}\n\n".format(acc))
        f.write(report)

    # Résumé court pour rendu TP (à imprimer ou joindre au rapport)
    resume_path = output_dir / "resume_evaluation_tp.txt"
    with open(resume_path, "w", encoding="utf-8") as f:
        f.write("RÉSUMÉ ÉVALUATION — TP Classification maladies des plantes\n")
        f.write("Projet : PlantVillage — ResNet18\n")
        f.write("Date : {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
        f.write("Résultats sur l'ensemble de test :\n")
        f.write("  - Accuracy  : {:.4f} ({:.2f} %)\n".format(acc, acc * 100))
        f.write("  - Macro F1  : {:.4f}\n".format(macro_f1))
        f.write("  - Weighted F1 : {:.4f}\n\n".format(weighted_f1))
        f.write("Fichiers générés :\n")
        f.write("  - confusion_matrix.png\n")
        f.write("  - classification_report.txt\n")
    print("  Résumé TP enregistré :", resume_path)

    # ---------- Export JSON pour l'interface graphique ----------
    classes_data = []
    for name in target_names:
        if name in report_dict and isinstance(report_dict[name], dict):
            r = report_dict[name]
            classes_data.append({
                "name": name,
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1-score": round(r["f1-score"], 4),
                "support": int(r["support"]),
            })
    results = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_test": n_test,
        "num_classes": num_classes,
        "report_text": report,
        "classes": classes_data,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("  Données pour l'interface : outputs/results.json")

    # ---------- Matrice de confusion (figure lisible) ----------
    cm = confusion_matrix(all_labels, all_preds)
    short_names = [_short_name(c) for c in target_names]

    fig, ax = plt.subplots(figsize=(14, 12))
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=short_names,
            yticklabels=short_names,
            ax=ax,
            cbar_kws={"label": "Nombre d\'échantillons"},
            linewidths=0.5,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, label="Nombre d'échantillons")
        tick_marks = np.arange(len(short_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(short_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(short_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title("Matrice de confusion — PlantVillage (ensemble de test)")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("  Matrice de confusion :", cm_path)
    print()


if __name__ == "__main__":
    main()
