
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Aircraft AI",
    page_icon="✈️",
    layout="wide"
)

DOSSIER_PROJET = Path(__file__).resolve().parent
DOSSIER_MODELES = DOSSIER_PROJET / "models"

MODELE_DETECTION = DOSSIER_MODELES / "modele_detection_avions.pt"
MODELE_CLASSIFICATION = DOSSIER_MODELES / "modele_classification_avions.pth"

APPAREIL = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. CHARGEMENT DES MODELES
# ==========================================

@st.cache_resource
def charger_modeles():

    if not MODELE_DETECTION.is_file():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {MODELE_DETECTION}"
        )

    if not MODELE_CLASSIFICATION.is_file():
        raise FileNotFoundError(
            f"Modèle ResNet50 introuvable : {MODELE_CLASSIFICATION}"
        )

    detecteur = YOLO(str(MODELE_DETECTION))

    try:
        checkpoint = torch.load(
            MODELE_CLASSIFICATION,
            map_location=APPAREIL,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODELE_CLASSIFICATION,
            map_location=APPAREIL
        )

    categories = checkpoint["categories"]
    taille_image = int(checkpoint.get("taille_image", 224))

    classificateur = models.resnet50(weights=None)

    classificateur.fc = nn.Linear(
        classificateur.fc.in_features,
        len(categories)
    )

    classificateur.load_state_dict(checkpoint["model_state"])
    classificateur.to(APPAREIL)
    classificateur.eval()

    transformation = transforms.Compose([
        transforms.Resize((taille_image, taille_image)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return detecteur, classificateur, categories, transformation


# ==========================================
# 3. CLASSIFICATION D'UN AVION
# ==========================================

def classifier_avion(image, classificateur, categories, transformation):

    image = image.convert("RGB")

    tensor = transformation(image).unsqueeze(0).to(APPAREIL)

    with torch.no_grad():
        predictions = classificateur(tensor)
        probabilites = torch.softmax(predictions, dim=1)
        confiance, indice = torch.max(probabilites, dim=1)

    return categories[indice.item()], confiance.item()


# ==========================================
# 4. ANALYSE DE L'IMAGE
# ==========================================

def analyser_image(image, detecteur, classificateur, categories, transformation):

    image = image.convert("RGB")
    largeur, hauteur = image.size

    image_annotee = image.copy()
    dessin = ImageDraw.Draw(image_annotee)

    resultats = detecteur.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    avions = []

    for resultat in resultats:

        if resultat.boxes is None:
            continue

        for boite in resultat.boxes:

            coords = boite.xyxy[0].cpu().tolist()

            xmin, ymin, xmax, ymax = [
                int(round(v)) for v in coords
            ]

            xmin = max(0, min(xmin, largeur - 1))
            ymin = max(0, min(ymin, hauteur - 1))
            xmax = max(0, min(xmax, largeur))
            ymax = max(0, min(ymax, hauteur))

            if xmax <= xmin or ymax <= ymin:
                continue

            crop = image.crop((xmin, ymin, xmax, ymax))

            categorie, confiance = classifier_avion(
                crop,
                classificateur,
                categories,
                transformation
            )

            avions.append({
                "class": str(categorie),
                "confidence": round(confiance, 4),
                "coordinates": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            })

            dessin.rectangle(
                (xmin, ymin, xmax, ymax),
                outline="red",
                width=3
            )

            dessin.text(
                (xmin, max(0, ymin - 20)),
                f"{categorie} ({confiance * 100:.1f}%)",
                fill="yellow",
                stroke_width=1,
                stroke_fill="black"
            )

    return image_annotee, avions


# ==========================================
# 5. INTERFACE AIRCRAFT AI
# ==========================================

st.title("✈️ Aircraft AI")

st.subheader("Détection et classification intelligente d'avions")

st.write(
    "Importez une image pour détecter les avions "
    "et identifier leur catégorie automatiquement."
)

st.divider()

# Chargement des modèles
try:
    with st.spinner("Chargement des modèles IA..."):
        detecteur, classificateur, categories, transformation = charger_modeles()

except Exception as erreur:
    st.error(f"Impossible de charger les modèles : {erreur}")
    st.stop()

st.success("Les modèles IA sont prêts.")

# Importation de l'image
fichier = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if fichier is not None:

    image = Image.open(fichier).convert("RGB")

    st.subheader("Image sélectionnée")
    st.image(image, use_container_width=True)

    if st.button("🔍 Analyser l'image", type="primary"):

        try:
            with st.spinner("Détection et classification en cours..."):

                image_annotee, avions = analyser_image(
                    image,
                    detecteur,
                    classificateur,
                    categories,
                    transformation
                )

            st.divider()
            st.header("Résultats de l'analyse")

            st.metric(
                "Nombre d'avions détectés",
                len(avions)
            )

            st.subheader("Image annotée")
            st.image(image_annotee, use_container_width=True)

            donnees_json = {
                fichier.name: avions
            }

            st.download_button(
                "Télécharger l'image annotée",
                data=_image_en_bytes(image_annotee),
                file_name="image_annotee.jpg",
                mime="image/jpeg"
            )

            st.download_button(
                "Télécharger les résultats JSON",
                data=json.dumps(
                    donnees_json,
                    indent=4,
                    ensure_ascii=False
                ),
                file_name="resultats.json",
                mime="application/json"
            )

            st.subheader("Détail des avions détectés")

            if not avions:
                st.info("Aucun avion n'a été détecté.")

            else:
                for i, avion in enumerate(avions, start=1):

                    with st.expander(f"Avion {i} — {avion['class']}"):

                        st.write(
                            f"**Classe :** {avion['class']}"
                        )

                        st.write(
                            f"**Confiance :** {avion['confidence'] * 100:.1f}%"
                        )

                        st.json(avion["coordinates"])

        except Exception as erreur:
            st.error(f"Erreur pendant l'analyse : {erreur}")


def _image_en_bytes(image):
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    return buffer.getvalue()
