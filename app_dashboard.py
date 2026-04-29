"""
Interface graphique des résultats d'évaluation PlantVillage.
Lance un serveur local et ouvre le tableau de bord dans le navigateur.

Usage : python app_dashboard.py
        (exécuter evaluate.py une fois avant pour générer outputs/results.json et confusion_matrix.png)
"""
import json
import uuid
import webbrowser
from pathlib import Path
from threading import Timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from torchvision import models, transforms

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

_PREDICTOR = None


def _load_predictor():
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    best_model_path = OUTPUT_DIR / "best_model.pth"
    idx_to_class_path = OUTPUT_DIR / "idx_to_class.pt"
    temperature_path = OUTPUT_DIR / "temperature.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {best_model_path}. Lance d'abord `python train.py`."
        )
    if not idx_to_class_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {idx_to_class_path}. Lance d'abord `python train.py`."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_to_class = torch.load(idx_to_class_path, map_location="cpu")
    num_classes = len(idx_to_class)
    if num_classes <= 0:
        raise RuntimeError("Mapping de classes vide (idx_to_class.pt).")

    state_dict = torch.load(best_model_path, map_location=device)
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

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    temperature = 1.0
    if temperature_path.exists():
        temperature_data = torch.load(temperature_path, map_location="cpu")
        temperature = float(temperature_data.get("temperature", 1.0))

    _PREDICTOR = {
        "device": device,
        "model": model,
        "idx_to_class": idx_to_class,
        "preprocess": preprocess,
        "temperature": temperature,
    }
    return _PREDICTOR


@app.route("/")
def index():
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Erreur</title></head><body style='font-family:sans-serif;padding:2rem;'>"
            "<h1>Données manquantes</h1><p>Exécutez d'abord : <code>python evaluate.py</code></p>"
            "<p>Puis rechargez cette page.</p></body></html>",
            200,
        )
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return render_template("dashboard.html", **data)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if "image" not in request.files:
        return render_template("predict.html", error="Aucune image envoyée.")

    file = request.files["image"]
    if not file or file.filename == "":
        return render_template("predict.html", error="Veuillez sélectionner une image.")

    # Charger modèle + mapping
    try:
        predictor = _load_predictor()
    except Exception as e:
        return render_template("predict.html", error=str(e))

    device = predictor["device"]
    model = predictor["model"]
    idx_to_class = predictor["idx_to_class"]
    preprocess = predictor["preprocess"]
    temperature = predictor["temperature"]

    # Sauvegarder l'image pour l'afficher sur la page
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"  # fallback simple

    filename = f"upload_{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / filename
    file.save(saved_path)

    # Prétraitement + prédiction
    try:
        img = Image.open(saved_path).convert("RGB")
    except Exception:
        return render_template("predict.html", error="Impossible de lire le fichier image.")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        logits = logits / max(temperature, 1e-3)
        probs = F.softmax(logits, dim=1)
        conf_tensor, pred_tensor = probs.max(dim=1)

    pred_idx = int(pred_tensor.item())
    confidence = float(conf_tensor.item())

    # Compatibilité légère si les clés sont des int ou des str
    try:
        predicted_class = idx_to_class[pred_idx]
    except KeyError:
        predicted_class = idx_to_class.get(str(pred_idx), str(pred_idx))

    return render_template(
        "predict.html",
        predicted_class=predicted_class,
        confidence=confidence,
        uploaded_image_url=f"/outputs/uploads/{filename}",
    )


def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    print("Tableau de bord : http://127.0.0.1:5000/")
    print("Fermez le serveur avec Ctrl+C.")
    Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
