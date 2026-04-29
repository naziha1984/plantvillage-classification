import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def load_model(output_dir: Path, device: torch.device):
    model_path = output_dir / "best_model.pth"
    idx_to_class_path = output_dir / "idx_to_class.pt"
    temperature_path = output_dir / "temperature.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Lance d'abord train.py."
        )
    if not idx_to_class_path.exists():
        raise FileNotFoundError(
            f"Mapping introuvable : {idx_to_class_path}. Lance d'abord train.py."
        )

    idx_to_class = torch.load(idx_to_class_path, map_location="cpu")
    num_classes = len(idx_to_class)

    state_dict = torch.load(model_path, map_location=device)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Compatibilité : ancien modèle (fc linéaire) ou nouveau (Dropout + Linear)
    has_sequential_head = any(k.startswith("fc.1.") for k in state_dict.keys())
    if has_sequential_head:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    temperature = 1.0
    if temperature_path.exists():
        temperature_data = torch.load(temperature_path, map_location="cpu")
        temperature = float(temperature_data.get("temperature", 1.0))

    return model, idx_to_class, temperature


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_image(image_path: Path) -> None:
    output_dir = Path("outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    model, idx_to_class, temperature = load_model(output_dir, device)
    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        logits = logits / max(temperature, 1e-3)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    pred_idx = int(pred_idx.item())
    confidence = float(confidence.item())

    try:
        predicted_label = idx_to_class[pred_idx]
    except KeyError:
        predicted_label = idx_to_class.get(str(pred_idx), str(pred_idx))

    print(f"Predicted: {predicted_label}")
    print(f"Confidence: {confidence * 100:.1f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        f"Predicted: {predicted_label}\nConfidence: {confidence * 100:.1f}%",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    predict_image(image_path)


if __name__ == "__main__":
    main()
