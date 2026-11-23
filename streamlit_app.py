
import streamlit as st
import torch
import torch.nn as nn
import json
import io

from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aircraft AI",
    page_icon="✈️",
    layout="wide"
)

DOSSIER = Path(__file__).resolve().parent

MODELE_DETECTION = (
    DOSSIER / "models" / "modele_detection_avions.pt"
)

MODELE_CLASSIFICATION = (
    DOSSIER / "models" / "modele_classification_avions.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONVERSION IMAGE EN BYTES
# ============================================================

def _image_en_bytes(image):

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return buffer.getvalue()


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

@st.cache_resource
def charger_modeles():

    if not MODELE_DETECTION.exists():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {MODELE_DETECTION}"
        )

    if not MODELE_CLASSIFICATION.exists():
        raise FileNotFoundError(
            f"Modèle ResNet50 introuvable : {MODELE_CLASSIFICATION}"
        )

    # Chargement YOLO
    modele_yolo = YOLO(str(MODELE_DETECTION))

    # Chargement ResNet50
    checkpoint = torch.load(
        MODELE_CLASSIFICATION,
        map_location=DEVICE,
        weights_only=False
    )

    categories = checkpoint["categories"]

    taille_image = checkpoint.get("taille_image", 224)

    if isinstance(categories, dict):
        categories = [
            categories[i]
            for i in sorted(categories.keys())
        ]

    # Création du réseau ResNet50
    modele_resnet = models.resnet50(weights=None)

    nombre_classes = len(categories)

    modele_resnet.fc = nn.Linear(
        modele_resnet.fc.in_features,
        nombre_classes
    )

    modele_resnet.load_state_dict(
        checkpoint["model_state"]
    )

    modele_resnet = modele_resnet.to(DEVICE)

    modele_resnet.eval()

    # Prétraitement des images
    transformation = transforms.Compose([

        transforms.Resize(
            (taille_image, taille_image)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return (
        modele_yolo,
        modele_resnet,
        categories,
        transformation
    )


# ============================================================
# CLASSIFICATION D'UN AVION
# ============================================================

def classifier_avion(
    image,
    modele_resnet,
    categories,
    transformation
):

    image = image.convert("RGB")

    image_tensor = transformation(image).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        predictions = modele_resnet(image_tensor)

        probabilites = torch.softmax(
            predictions,
            dim=1
        )

        confiance, indice = torch.max(
            probabilites,
            dim=1
        )

    classe = categories[indice.item()]

    confiance = confiance.item()

    return classe, confiance


# ============================================================
# DETECTION ET CLASSIFICATION
# ============================================================

def analyser_image(
    image,
    modele_yolo,
    modele_resnet,
    categories,
    transformation,
    seuil_confiance=0.25
):

    image = image.convert("RGB")

    # Détection avec YOLO
    resultats = modele_yolo.predict(
        source=image,
        conf=seuil_confiance,
        verbose=False
    )

    resultat = resultats[0]

    avions = []

    # Copie de l'image originale
    image_annotee = image.copy()

    dessin = ImageDraw.Draw(image_annotee)

    if resultat.boxes is not None:

        for boite in resultat.boxes:

            # Coordonnées de la détection
            coordonnees = boite.xyxy[0].cpu().numpy()

            xmin, ymin, xmax, ymax = [
                int(valeur)
                for valeur in coordonnees
            ]

            # Limites de l'image
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            # Découpage de l'avion
            crop = image.crop(
                (xmin, ymin, xmax, ymax)
            )

            # Classification du même avion
            classe, confiance = classifier_avion(
                crop,
                modele_resnet,
                categories,
                transformation
            )

            # Numéro unique
            numero = len(avions) + 1

            # Enregistrement des résultats
            avions.append({

                "numero": numero,

                "class": str(classe),

                "confidence": round(
                    confiance,
                    4
                ),

                "coordinates": {

                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax

                }

            })

            # Dessin du rectangle
            couleur = "lime"

            dessin.rectangle(
                (xmin, ymin, xmax, ymax),
                outline=couleur,
                width=4
            )

            # Texte affiché au-dessus du rectangle
            etiquette = (
                f"Avion {numero} : {classe} "
                f"({confiance * 100:.1f}%)"
            )

            boite_texte = dessin.textbbox(
                (0, 0),
                etiquette
            )

            largeur = (
                boite_texte[2] - boite_texte[0]
            )

            hauteur = (
                boite_texte[3] - boite_texte[1]
            )

            texte_x = xmin

            texte_y = max(
                0,
                ymin - hauteur - 10
            )

            # Fond de l'étiquette
            dessin.rectangle(
                (
                    texte_x,
                    texte_y,
                    texte_x + largeur + 10,
                    texte_y + hauteur + 8
                ),
                fill=couleur
            )

            # Texte de classification
            dessin.text(
                (
                    texte_x + 5,
                    texte_y + 4
                ),
                etiquette,
                fill="black"
            )

    return image_annotee, avions


# ============================================================
# INTERFACE AIRCRAFT AI
# ============================================================

st.title("✈️ Aircraft AI")

st.subheader(
    "Détection et classification automatique d'avions"
)

st.write(
    "Importe une image pour détecter les avions présents "
    "et identifier leur catégorie grâce à l'intelligence artificielle."
)

st.divider()


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

try:

    with st.spinner(
        "Chargement des modèles d'intelligence artificielle..."
    ):

        (
            modele_yolo,
            modele_resnet,
            categories,
            transformation
        ) = charger_modeles()

    st.success(
        "Les modèles sont chargés et prêts à fonctionner."
    )

except Exception as erreur:

    st.error(
        f"Impossible de charger les modèles : {erreur}"
    )

    st.stop()


# ============================================================
# IMPORTATION DE L'IMAGE
# ============================================================

fichier_image = st.file_uploader(
    "Choisis une image contenant un ou plusieurs avions",
    type=["jpg", "jpeg", "png"]
)


if fichier_image is not None:

    image = Image.open(
        fichier_image
    ).convert("RGB")

    st.subheader("Image importée")

    st.image(
        image,
        caption="Image originale",
        width=500
    )

    st.divider()

    # Seuil de confiance
    seuil = st.slider(
        "Seuil de confiance de détection",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    # ========================================================
    # ANALYSE
    # ========================================================

    if st.button(
        "🔍 Analyser l'image",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyse en cours..."
            ):

                image_annotee, avions = analyser_image(
                    image,
                    modele_yolo,
                    modele_resnet,
                    categories,
                    transformation,
                    seuil
                )

            st.success("Analyse terminée !")

            # =================================================
            # NOMBRE D'AVIONS
            # =================================================

            nombre_avions = len(avions)

            st.subheader("Résultats de l'analyse")

            st.metric(
                label="Nombre d'avions détectés",
                value=nombre_avions
            )

            # =================================================
            # IMAGE ANNOTÉE
            # =================================================

            st.subheader(
                "Image avec les avions détectés et classifiés"
            )

            st.image(
                image_annotee,
                caption="Résultat Aircraft AI",
                width=500
            )

            # =================================================
            # CLASSIFICATION
            # =================================================

            st.subheader(
                "Classification des avions"
            )

            if nombre_avions == 0:

                st.warning(
                    "Aucun avion n'a été détecté dans cette image."
                )

            else:

                for avion in avions:

                    with st.expander(
                        f"Avion {avion['numero']} — "
                        f"Classe : {avion['class']}"
                    ):

                        st.write(
                            f"**Numéro :** {avion['numero']}"
                        )

                        st.write(
                            f"**Classe détectée :** "
                            f"{avion['class']}"
                        )

                        st.write(
                            f"**Confiance :** "
                            f"{avion['confidence'] * 100:.2f} %"
                        )

                        st.write(
                            "**Coordonnées :**"
                        )

                        st.json(
                            avion["coordinates"]
                        )

            # =================================================
            # JSON
            # =================================================

            donnees_json = {
                fichier_image.name: avions
            }

            json_bytes = json.dumps(
                donnees_json,
                indent=4,
                ensure_ascii=False
            ).encode("utf-8")

            # =================================================
            # TELECHARGEMENTS
            # =================================================

            st.divider()

            st.subheader(
                "Télécharger les résultats"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    label="📥 Télécharger l'image annotée",

                    data=_image_en_bytes(
                        image_annotee
                    ),

                    file_name="aircraft_ai_resultat.png",

                    mime="image/png",

                    use_container_width=True
                )

            with col2:

                st.download_button(
                    label="📄 Télécharger les résultats JSON",

                    data=json_bytes,

                    file_name="resultats_aircraft_ai.json",

                    mime="application/json",

                    use_container_width=True
                )

        except Exception as erreur:

            st.error(
                f"Erreur pendant l'analyse : {erreur}"
            )

else:

    st.info(
        "Importe une image pour commencer l'analyse."
    )


# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Aircraft AI — Détection et classification d'avions "
    "par intelligence artificielle."
)

st.caption(
    "Aircraft AI — Détection et classification d'avions par intelligence artificielle."
)
