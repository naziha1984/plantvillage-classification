import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """Active/désactive le calcul du gradient pour un module."""
    for p in model.parameters():
        p.requires_grad = requires_grad


def visualize_data_augmentation(
    data_dir: Path, train_transform: transforms.Compose, output_dir: Path
) -> None:
    dataset_no_transform = datasets.ImageFolder(str(data_dir))
    if len(dataset_no_transform.samples) == 0:
        fail_msg = (
            "Aucune image trouvée pour visualiser l'augmentation. "
            f"Vérifie le dossier dataset: {data_dir}"
        )
        raise RuntimeError(fail_msg)

    image_path, _ = dataset_no_transform.samples[0]
    original_pil = datasets.folder.default_loader(image_path)

    original_for_plot = transforms.Resize((224, 224))(original_pil)
    augmented_1 = _denormalize_image(train_transform(original_pil))
    augmented_2 = _denormalize_image(train_transform(original_pil))
    augmented_3 = _denormalize_image(train_transform(original_pil))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_for_plot)
    axes[0].set_title("Original")
    axes[1].imshow(augmented_1)
    axes[1].set_title("Augmented 1")
    axes[2].imshow(augmented_2)
    axes[2].set_title("Augmented 2")
    axes[3].imshow(augmented_3)
    axes[3].set_title("Augmented 3")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "augmentation_preview.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Aperçu des augmentations sauvegardé : {output_path}")


def calibrate_temperature(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device
) -> float:
    """Ajuste une température simple sur les logits du set de validation."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs)
            labels_list.append(labels.to(device))

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = torch.ones(1, device=device, requires_grad=True)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=1e-3)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-3).item())


def main() -> None:
    set_seed(42)

    data_dir = Path("data/PlantVillage")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    phase1_epochs = 3
    lr_phase1 = 1e-3
    # Total epochs (phase1 + phase2) = 15
    phase2_epochs = 12
    lr_phase2 = 1e-4
    val_split = 0.1
    img_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(img_size, img_size), scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    print("Advanced data augmentation enabled")
    print("Astuce rapport : lance `python train.py --preview-aug` pour capturer l'aperçu.")

    if "--preview-aug" in sys.argv:
        visualize_data_augmentation(data_dir, train_transform, output_dir)
        return

    dataset = datasets.ImageFolder(str(data_dir))
    num_classes = len(dataset.classes)
    print(f"Nombre de classes : {num_classes}")

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        shuffle=True,
        stratify=dataset.targets,
    )

    train_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(data_dir), transform=val_transform)

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(val_dataset, val_idx)

    train_targets = [dataset.targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = len(train_targets) / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Using class weights to handle imbalance")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    torch.save(idx_to_class, output_dir / "idx_to_class.pt")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Dropout est utilisé pour réduire l'overfitting : il désactive aléatoirement
    # une partie des activations pendant l'entraînement, ce qui améliore la
    # généralisation.
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model.to(device)

    # Label smoothing: réduit la surconfiance en "adouçissant" les labels,
    # ce qui améliore souvent la généralisation (et donc la confiance globale).
    print("Using label smoothing to improve generalization")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.1
    )

    def train_phase(
        phase_label: str,
        num_epochs_phase: int,
        optimizer_phase: optim.Optimizer,
        best_val_acc: float,
        best_val_loss: float,
        epochs_no_improve: int,
        early_stop_patience: int,
    ) -> tuple[float, float, int, bool]:
        for epoch in range(1, num_epochs_phase + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(
                train_loader,
                desc=f"{phase_label} — Epoch {epoch}/{num_epochs_phase} - train",
            ):
                images, labels = images.to(device), labels.to(device)

                optimizer_phase.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_phase.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            model.eval()
            val_correct = 0
            val_total = 0
            running_val_loss = 0.0
            with torch.no_grad():
                for images, labels in tqdm(
                    val_loader,
                    desc=f"{phase_label} — Epoch {epoch}/{num_epochs_phase} - val",
                ):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss_val = criterion(outputs, labels)
                    running_val_loss += loss_val.item() * labels.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            val_loss = running_val_loss / val_total
            print(
                f"{phase_label} | Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Meilleur modèle sauvegardé dans {best_model_path} "
                    f"avec val_acc={best_val_acc:.4f}"
                )

            # Early stopping logic based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered")
                return best_val_acc, best_val_loss, epochs_no_improve, True

        return best_val_acc, best_val_loss, epochs_no_improve, False

    best_model_path = output_dir / "best_model.pth"
    best_val_acc = 0.0

    # Early stopping (monitor validation loss)
    best_val_loss = float("inf")
    early_stop_patience = 3
    epochs_no_improve = 0

    print("Phase 1: feature extraction")
    set_requires_grad(model, False)
    set_requires_grad(model.fc, True)
    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=lr_phase1)
    best_val_acc, best_val_loss, epochs_no_improve, early_stopped = train_phase(
        "Phase 1: feature extraction",
        phase1_epochs,
        optimizer_phase1,
        best_val_acc,
        best_val_loss,
        epochs_no_improve,
        early_stop_patience,
    )

    if early_stopped:
        print("Entraînement terminé (early stopping).")
        return

    print("Phase 2: fine-tuning")
    set_requires_grad(model.layer4, True)
    set_requires_grad(model.fc, True)
    optimizer_phase2 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr_phase2
    )
    best_val_acc, best_val_loss, epochs_no_improve, early_stopped = train_phase(
        "Phase 2: fine-tuning",
        phase2_epochs,
        optimizer_phase2,
        best_val_acc,
        best_val_loss,
        epochs_no_improve,
        early_stop_patience,
    )

    if early_stopped:
        print("Entraînement terminé (early stopping).")
    else:
        print("Entraînement terminé.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    temperature = calibrate_temperature(model, val_loader, device)
    torch.save({"temperature": temperature}, output_dir / "temperature.pt")
    print("Model calibrated with temperature scaling")


if __name__ == "__main__":
    main()

