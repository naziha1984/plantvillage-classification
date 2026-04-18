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


def main() -> None:
    set_seed(42)

    data_dir = Path("data/PlantVillage")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    val_split = 0.1
    img_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(img_size, img_size), scale=(0.9, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    print("Data augmentation enabled")
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
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs} - train"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch}/{num_epochs} - val"
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Meilleur modèle sauvegardé dans {best_model_path} "
                f"avec val_acc={best_val_acc:.4f}"
            )

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()

