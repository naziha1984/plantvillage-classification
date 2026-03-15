"""
Interface graphique des résultats d'évaluation PlantVillage.
Lance un serveur local et ouvre le tableau de bord dans le navigateur.

Usage : python app_dashboard.py
        (exécuter evaluate.py une fois avant pour générer outputs/results.json et confusion_matrix.png)
"""
import json
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, render_template, send_from_directory

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))


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


def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    print("Tableau de bord : http://127.0.0.1:5000/")
    print("Fermez le serveur avec Ctrl+C.")
    Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
