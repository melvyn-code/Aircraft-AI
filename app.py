
import os
import json
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)
from PIL import Image, ImageDraw
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION DU PROJET
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent
DOSSIER_MODELES = DOSSIER_PROJET / "models"

DOSSIER_UPLOADS = DOSSIER_PROJET / "uploads"
DOSSIER_RESULTS = DOSSIER_PROJET / "results"

DOSSIER_UPLOADS.mkdir(exist_ok=True)
DOSSIER_RESULTS.mkdir(exist_ok=True)

MODELE_DETECTION = DOSSIER_MODELES / "modele_detection_avions.pt"
MODELE_CLASSIFICATION = DOSSIER_MODELES / "modele_classification_avions.pth"

EXTENSIONS_AUTORISEES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 Mo maximum


# ============================================================
# 2. CONFIGURATION DU CALCUL
# ============================================================

APPAREIL = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {APPAREIL}")


# ============================================================
# 3. VERIFICATION DES MODELES
# ============================================================

if not MODELE_DETECTION.is_file():
    raise FileNotFoundError(
        f"Modèle de détection introuvable : {MODELE_DETECTION}"
    )

if not MODELE_CLASSIFICATION.is_file():
    raise FileNotFoundError(
        f"Modèle de classification introuvable : {MODELE_CLASSIFICATION}"
    )


# ============================================================
# 4. CHARGEMENT DU MODELE YOLO
# ============================================================

print("Chargement du modèle YOLO...")

detecteur = YOLO(str(MODELE_DETECTION))

print("Modèle YOLO chargé avec succès.")


# ============================================================
# 5. CHARGEMENT DU MODELE RESNET50
# ============================================================

print("Chargement du modèle ResNet50...")

try:
    checkpoint = torch.load(
        MODELE_CLASSIFICATION,
        map_location=APPAREIL,
        weights_only=True
    )
except TypeError:
    # Compatibilité avec certaines anciennes versions de PyTorch
    checkpoint = torch.load(
        MODELE_CLASSIFICATION,
        map_location=APPAREIL
    )

if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
    raise ValueError(
        "Le fichier de classification ne contient pas la clé 'model_state'."
    )

categories = checkpoint["categories"]
TAILLE_IMAGE = int(checkpoint.get("taille_image", 224))

nombre_classes = len(categories)

classificateur = models.resnet50(weights=None)
classificateur.fc = nn.Linear(
    classificateur.fc.in_features,
    nombre_classes
)

classificateur.load_state_dict(checkpoint["model_state"])
classificateur.to(APPAREIL)
classificateur.eval()

print("Modèle ResNet50 chargé avec succès.")
print("Catégories :", categories)


# ============================================================
# 6. PRETRAITEMENT DES IMAGES
# ============================================================

transformation = transforms.Compose([
    transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. FONCTION DE CLASSIFICATION
# ============================================================

def classifier_avion(image_avion):
    """
    Classe une image contenant un avion avec ResNet50.
    Retourne la catégorie prédite et sa confiance.
    """

    image_avion = image_avion.convert("RGB")

    image_tensor = transformation(image_avion)
    image_tensor = image_tensor.unsqueeze(0).to(APPAREIL)

    with torch.no_grad():
        predictions = classificateur(image_tensor)
        probabilites = torch.softmax(predictions, dim=1)

        confiance, indice = torch.max(probabilites, dim=1)

    indice = indice.item()
    confiance = confiance.item()

    categorie = categories[indice]

    return categorie, confiance


# ============================================================
# 8. ANALYSE D'UNE IMAGE
# ============================================================

def analyser_image(chemin_image):
    """
    Détecte les avions avec YOLO puis classe chaque avion
    avec ResNet50.
    """

    image = Image.open(chemin_image).convert("RGB")

    largeur, hauteur = image.size

    image_annotee = image.copy()
    dessin = ImageDraw.Draw(image_annotee)

    resultats = detecteur.predict(
        source=str(chemin_image),
        conf=0.25,
        verbose=False
    )

    avions_detectes = []

    for resultat in resultats:

        if resultat.boxes is None:
            continue

        for boite in resultat.boxes:

            coordonnees = boite.xyxy[0].cpu().tolist()

            xmin, ymin, xmax, ymax = [
                int(round(valeur)) for valeur in coordonnees
            ]

            # Sécurisation des coordonnées
            xmin = max(0, min(xmin, largeur - 1))
            ymin = max(0, min(ymin, hauteur - 1))
            xmax = max(0, min(xmax, largeur))
            ymax = max(0, min(ymax, hauteur))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Extraction de l'avion
            crop = image.crop((xmin, ymin, xmax, ymax))

            # Classification de l'avion
            categorie, confiance = classifier_avion(crop)

            avions_detectes.append({
                "class": str(categorie),
                "confidence": round(confiance, 4),
                "coordinates": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            })

            # Dessin du rectangle
            dessin.rectangle(
                (xmin, ymin, xmax, ymax),
                outline="red",
                width=3
            )

            # Affichage de la catégorie
            texte = f"{categorie} ({confiance * 100:.1f}%)"

            dessin.text(
                (xmin, max(0, ymin - 20)),
                texte,
                fill="yellow"
            )

    return image_annotee, avions_detectes


# ============================================================
# 9. PAGE D'ACCUEIL
# ============================================================

@app.route("/")
def accueil():
    return render_template("index.html")


# ============================================================
# 10. ANALYSE DE L'IMAGE ENVOYEE
# ============================================================

@app.route("/analyser", methods=["POST"])
def analyser():

    if "image" not in request.files:
        return jsonify({
            "erreur": "Aucune image n'a été envoyée."
        }), 400

    fichier = request.files["image"]

    if fichier.filename == "":
        return jsonify({
            "erreur": "Veuillez sélectionner une image."
        }), 400

    extension = Path(fichier.filename).suffix.lower()

    if extension not in EXTENSIONS_AUTORISEES:
        return jsonify({
            "erreur": "Format non autorisé. Utilisez JPG, PNG ou TIFF."
        }), 400

    identifiant = uuid.uuid4().hex

    nom_image = f"{identifiant}{extension}"
    nom_resultat = f"{identifiant}_annotee.jpg"
    nom_json = f"{identifiant}.json"

    chemin_image = DOSSIER_UPLOADS / nom_image
    chemin_resultat = DOSSIER_RESULTS / nom_resultat
    chemin_json = DOSSIER_RESULTS / nom_json

    try:
        fichier.save(str(chemin_image))

        # Vérification que le fichier est bien une image
        with Image.open(chemin_image) as image_test:
            image_test.verify()

        image_annotee, avions = analyser_image(chemin_image)

        image_annotee.save(chemin_resultat, format="JPEG", quality=95)

        donnees_json = {
            nom_image: avions
        }

        with open(chemin_json, "w", encoding="utf-8") as fichier_json:
            json.dump(
                donnees_json,
                fichier_json,
                indent=4,
                ensure_ascii=False
            )

        return jsonify({
            "success": True,
            "nombre_avions": len(avions),
            "avions": avions,
            "image_annotee": f"/images/{nom_resultat}",
            "telechargement_image": f"/telecharger/{nom_resultat}",
            "telechargement_json": f"/telecharger/{nom_json}"
        })

    except Exception as erreur:
        print("Erreur pendant l'analyse :", erreur)

        return jsonify({
            "erreur": f"Une erreur est survenue pendant l'analyse : {str(erreur)}"
        }), 500

    finally:
        # Suppression de l'image envoyée après son analyse
        if chemin_image.exists():
            chemin_image.unlink()


# ============================================================
# 11. AFFICHAGE DE L'IMAGE ANNOTEE
# ============================================================

@app.route("/images/<filename>")
def afficher_image(filename):

    return send_from_directory(
        DOSSIER_RESULTS,
        filename,
        as_attachment=False
    )


# ============================================================
# 12. TELECHARGEMENT DES RESULTATS
# ============================================================

@app.route("/telecharger/<filename>")
def telecharger(filename):

    return send_from_directory(
        DOSSIER_RESULTS,
        filename,
        as_attachment=True
    )


# ============================================================
# 13. LANCEMENT DU SERVEUR
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("       AIRCRAFT AI")
    print("===================================")
    print("Serveur disponible sur : http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
