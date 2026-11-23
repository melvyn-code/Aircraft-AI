
import io
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO


# ==========================================================
# CONFIGURATION
# ==========================================================

DOSSIER = Path(__file__).resolve().parent

MODELE_DETECTION = DOSSIER / "models" / "modele_detection_avions.pt"
MODELE_CLASSIFICATION = DOSSIER / "models" / "modele_classification_avions.pth"
CSS_PATH = DOSSIER / "static" / "style.css"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Aircraft AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# CHARGEMENT DU DESIGN
# ==========================================================

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"<style>\n{css}\n</style>",
        unsafe_allow_html=True
    )


# ==========================================================
# FONCTIONS UTILITAIRES
# ==========================================================

def image_en_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def charger_police(taille=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", taille)
    except OSError:
        return ImageFont.load_default()


# ==========================================================
# CHARGEMENT DES MODELES IA
# ==========================================================

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

    # Modèle YOLO
    modele_yolo = YOLO(str(MODELE_DETECTION))

    # Chargement du modèle ResNet50
    checkpoint = torch.load(
        MODELE_CLASSIFICATION,
        map_location=DEVICE,
        weights_only=False
    )

    categories = checkpoint["categories"]

    modele_resnet = models.resnet50(weights=None)

    modele_resnet.fc = nn.Linear(
        modele_resnet.fc.in_features,
        len(categories)
    )

    modele_resnet.load_state_dict(
        checkpoint["model_state"]
    )

    modele_resnet.to(DEVICE)
    modele_resnet.eval()

    taille_image = checkpoint.get("taille_image", 224)

    transformation = transforms.Compose([
        transforms.Resize((taille_image, taille_image)),
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


# ==========================================================
# CLASSIFICATION D'UN AVION
# ==========================================================

def classifier_avion(
    crop,
    modele_resnet,
    categories,
    transformation
):

    image_tensor = transformation(crop)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sorties = modele_resnet(image_tensor)
        probabilites = torch.softmax(sorties, dim=1)

        confiance, indice = torch.max(
            probabilites,
            dim=1
        )

    classe = categories[indice.item()]
    score = confiance.item()

    return classe, score


# ==========================================================
# DETECTION ET CLASSIFICATION
# ==========================================================

def analyser_image(
    image,
    modele_yolo,
    modele_resnet,
    categories,
    transformation,
    seuil
):

    image_originale = image.convert("RGB")
    image_annotee = image_originale.copy()

    dessin = ImageDraw.Draw(image_annotee)
    police = charger_police(18)

    resultats = modele_yolo.predict(
        source=image_originale,
        conf=seuil,
        verbose=False
    )

    avions = []
    numero = 0

    for resultat in resultats:

        if resultat.boxes is None:
            continue

        for boite in resultat.boxes:

            x1, y1, x2, y2 = map(
                int,
                boite.xyxy[0].tolist()
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_originale.width, x2)
            y2 = min(image_originale.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Découpage de l'avion
            crop = image_originale.crop(
                (x1, y1, x2, y2)
            )

            # Classification ResNet50
            classe, confiance = classifier_avion(
                crop,
                modele_resnet,
                categories,
                transformation
            )

            numero += 1

            libelle = (
                f"Avion {numero} : {classe} "
                f"({confiance * 100:.1f}%)"
            )

            # Rectangle autour de l'avion
            dessin.rectangle(
                (x1, y1, x2, y2),
                outline="#38BDF8",
                width=4
            )

            # Position du texte
            position_texte = (
                x1,
                max(0, y1 - 32)
            )

            boite_texte = dessin.textbbox(
                position_texte,
                libelle,
                font=police
            )

            # Fond du texte
            dessin.rounded_rectangle(
                (
                    boite_texte[0] - 6,
                    boite_texte[1] - 5,
                    boite_texte[2] + 6,
                    boite_texte[3] + 5
                ),
                radius=6,
                fill="#071426"
            )

            dessin.text(
                position_texte,
                libelle,
                fill="white",
                font=police
            )

            # Enregistrement des informations
            avions.append({
                "numero": numero,
                "class": str(classe),
                "confidence": round(float(confiance), 4),
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

    return image_annotee, avions


# ==========================================================
# EN-TETE AIRCRAFT AI
# ==========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-top">
            <div class="hero-icon">✈</div>

            <div class="hero-status">
                <span class="status-dot"></span>
                SYSTÈME IA ACTIF
            </div>
        </div>

        <h1>Aircraft <span>AI</span></h1>

        <p class="hero-subtitle">
            Détection et classification intelligente d'aéronefs
        </p>

        <div class="hero-line"></div>

        <p class="hero-description">
            Analysez vos images grâce à l'intelligence artificielle.
            Détectez les avions et identifiez automatiquement leurs catégories.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CHARGEMENT DES MODELES
# ==========================================================

try:

    with st.spinner("Chargement des modèles IA..."):

        (
            modele_yolo,
            modele_resnet,
            categories,
            transformation
        ) = charger_modeles()

    st.markdown(
        """
        <div class="success-card">
            <div class="success-icon">✓</div>

            <div>
                <strong>Modèles chargés avec succès</strong>
                <p>YOLO et ResNet50 sont prêts pour l'analyse.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as erreur:

    st.error(
        f"Impossible de charger les modèles : {erreur}"
    )

    st.stop()


# ==========================================================
# PARAMETRES
# ==========================================================

st.markdown(
    """
    <div class="section-heading">
        <div class="section-number">01</div>

        <div>
            <h2>Paramètres d'analyse</h2>
            <p>Configurez la détection de vos aéronefs.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    seuil = st.slider(
        "Seuil de confiance YOLO",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

with col2:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-label">
                CLASSES DISPONIBLES
            </div>

            <div class="info-card-value">
                {len(categories)}
            </div>

            <div class="info-card-description">
                Catégories reconnues par ResNet50
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# IMPORTATION DE L'IMAGE
# ==========================================================

st.markdown(
    """
    <div class="section-heading">
        <div class="section-number">02</div>

        <div>
            <h2>Importer une image</h2>
            <p>Choisissez une photographie contenant un ou plusieurs avions.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

fichier_image = st.file_uploader(
    "Sélectionner une image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==========================================================
# ANALYSE DE L'IMAGE
# ==========================================================

if fichier_image is not None:

    try:

        image = Image.open(fichier_image).convert("RGB")

        st.markdown(
            """
            <div class="section-heading">
                <div class="section-number">03</div>

                <div>
                    <h2>Image importée</h2>
                    <p>Vérifiez votre image avant de lancer l'analyse.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            caption="Image originale",
            width=500
        )

        lancer = st.button(
            "✈ LANCER L'ANALYSE",
            type="primary",
            use_container_width=True
        )

        if lancer:

            with st.spinner("Analyse des aéronefs en cours..."):

                image_annotee, avions = analyser_image(
                    image,
                    modele_yolo,
                    modele_resnet,
                    categories,
                    transformation,
                    seuil
                )

            # Résultats
            st.markdown(
                """
                <div class="section-heading">
                    <div class="section-number">04</div>

                    <div>
                        <h2>Résultats de l'analyse</h2>
                        <p>Voici les aéronefs détectés et classifiés.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            nombre_avions = len(avions)

            st.markdown(
                f"""
                <div class="result-banner">

                    <div>
                        <div class="result-label">
                            AÉRONEFS DÉTECTÉS
                        </div>

                        <div class="result-number">
                            {nombre_avions}
                        </div>
                    </div>

                    <div class="result-symbol">✈</div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # Image annotée
            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown("### Image avec détections")

            st.image(
                image_annotee,
                caption="Détections et classifications",
                width=500
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # Liste des avions
            st.markdown("### Détails des aéronefs")

            if avions:

                for avion in avions:

                    st.markdown(
                        f"""
                        <div class="aircraft-card">

                            <div class="aircraft-card-top">
                                <span class="aircraft-number">
                                    AVION {avion["numero"]:02d}
                                </span>

                                <span class="aircraft-confidence">
                                    {avion["confidence"] * 100:.1f} %
                                </span>
                            </div>

                            <h3>{avion["class"]}</h3>

                            <p>
                                Coordonnées :
                                X1 {avion["coordinates"]["x1"]} ·
                                Y1 {avion["coordinates"]["y1"]} ·
                                X2 {avion["coordinates"]["x2"]} ·
                                Y2 {avion["coordinates"]["y2"]}
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.warning(
                    "Aucun avion détecté. "
                    "Essaie de diminuer le seuil de confiance."
                )

            # Téléchargements
            st.markdown(
                """
                <div class="section-heading">
                    <div class="section-number">05</div>

                    <div>
                        <h2>Exporter les résultats</h2>
                        <p>Téléchargez les images et les données.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            donnees_json = {
                fichier_image.name: avions
            }

            json_bytes = json.dumps(
                donnees_json,
                indent=4,
                ensure_ascii=False
            ).encode("utf-8")

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "Télécharger l'image annotée",
                    data=image_en_bytes(image_annotee),
                    file_name="aircraft_ai_resultat.png",
                    mime="image/png",
                    use_container_width=True
                )

            with col2:

                st.download_button(
                    "Télécharger les résultats JSON",
                    data=json_bytes,
                    file_name="aircraft_ai_resultats.json",
                    mime="application/json",
                    use_container_width=True
                )

    except Exception as erreur:

        st.error(
            f"Une erreur est survenue pendant l'analyse : {erreur}"
        )


# ==========================================================
# PIED DE PAGE
# ==========================================================

st.markdown(
    """
    <div class="footer">

        <div class="footer-logo">
            ✈ AIRCRAFT AI
        </div>

        <p>
            Détection et classification d'aéronefs par intelligence artificielle
        </p>

        <span>
            YOLO · RESNET50 · STREAMLIT
        </span>

    </div>
    """,
    unsafe_allow_html=True
)
