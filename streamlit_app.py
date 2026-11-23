
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
    layout="wide",
    initial_sidebar_state="collapsed"
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
# CHARGEMENT DU DESIGN CSS
# ============================================================

CSS_PATH = DOSSIER / "static" / "style.css"

if CSS_PATH.exists():

    with open(CSS_PATH, encoding="utf-8") as fichier:

        st.markdown(
            f"<style>{fichier.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# FONCTION IMAGE EN BYTES
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

    modele_yolo = YOLO(str(MODELE_DETECTION))

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

    modele_resnet = models.resnet50(weights=None)

    modele_resnet.fc = nn.Linear(
        modele_resnet.fc.in_features,
        len(categories)
    )

    modele_resnet.load_state_dict(
        checkpoint["model_state"]
    )

    modele_resnet = modele_resnet.to(DEVICE)

    modele_resnet.eval()

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

    return classe, confiance.item()


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

    resultats = modele_yolo.predict(
        source=image,
        conf=seuil_confiance,
        verbose=False
    )

    resultat = resultats[0]

    avions = []

    image_annotee = image.copy()

    dessin = ImageDraw.Draw(image_annotee)

    if resultat.boxes is not None:

        for boite in resultat.boxes:

            coordonnees = boite.xyxy[0].cpu().numpy()

            xmin, ymin, xmax, ymax = [
                int(valeur) for valeur in coordonnees
            ]

            xmin = max(0, xmin)
            ymin = max(0, ymin)

            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            crop = image.crop(
                (xmin, ymin, xmax, ymax)
            )

            classe, confiance = classifier_avion(
                crop,
                modele_resnet,
                categories,
                transformation
            )

            numero = len(avions) + 1

            avions.append({

                "numero": numero,

                "class": str(classe),

                "confidence": round(confiance, 4),

                "coordinates": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }

            })

            # Rectangle autour de l'avion
            dessin.rectangle(
                (xmin, ymin, xmax, ymax),
                outline="#38f8a5",
                width=4
            )

            # Etiquette
            etiquette = (
                f"Avion {numero} : {classe} "
                f"({confiance * 100:.1f}%)"
            )

            boite_texte = dessin.textbbox(
                (0, 0),
                etiquette
            )

            largeur = boite_texte[2] - boite_texte[0]

            hauteur = boite_texte[3] - boite_texte[1]

            texte_x = xmin

            texte_y = max(
                0,
                ymin - hauteur - 10
            )

            dessin.rectangle(
                (
                    texte_x,
                    texte_y,
                    texte_x + largeur + 10,
                    texte_y + hauteur + 8
                ),
                fill="#38f8a5"
            )

            dessin.text(
                (
                    texte_x + 5,
                    texte_y + 4
                ),
                etiquette,
                fill="#071426"
            )

    return image_annotee, avions


# ============================================================
# EN-TETE
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">✈</div>
        <div class="hero-label">ARTIFICIAL INTELLIGENCE</div>
        <h1>AIRCRAFT AI</h1>
        <p>
            Advanced aircraft detection and classification system
        </p>
        <div class="hero-status">
            <span class="status-dot"></span>
            SYSTEM READY
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-heading">
        <span class="section-number">01</span>
        <span>IMAGE ANALYSIS</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

try:

    with st.spinner("Initialisation des modèles IA..."):

        (
            modele_yolo,
            modele_resnet,
            categories,
            transformation
        ) = charger_modeles()

except Exception as erreur:

    st.error(
        f"Impossible de charger les modèles : {erreur}"
    )

    st.stop()


# ============================================================
# IMPORTATION DE L'IMAGE
# ============================================================

fichier_image = st.file_uploader(
    "Importer une image aéronautique",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG et PNG"
)


if fichier_image is not None:

    image = Image.open(
        fichier_image
    ).convert("RGB")

    col_image, col_info = st.columns(
        [1.5, 1]
    )

    with col_image:

        st.markdown(
            '<div class="card-title">IMAGE SOURCE</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            width=500
        )

    with col_info:

        st.markdown(
            '<div class="card-title">FILE INFORMATION</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">FILENAME</div>
                <div class="info-value">{fichier_image.name}</div>
                <div class="info-divider"></div>
                <div class="info-label">IMAGE DIMENSIONS</div>
                <div class="info-value">
                    {image.width} × {image.height} px
                </div>
                <div class="info-divider"></div>
                <div class="info-label">IMAGE FORMAT</div>
                <div class="info-value">RGB</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-heading"><span class="section-number">02</span><span>DETECTION SETTINGS</span></div>',
        unsafe_allow_html=True
    )

    seuil = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    st.markdown(
        f'<div class="threshold-value">CONFIDENCE THRESHOLD : {seuil * 100:.0f}%</div>',
        unsafe_allow_html=True
    )

    lancer_analyse = st.button(
        "✦  LAUNCH AIRCRAFT ANALYSIS",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # ANALYSE
    # ========================================================

    if lancer_analyse:

        try:

            with st.spinner(
                "Detection and classification in progress..."
            ):

                image_annotee, avions = analyser_image(
                    image,
                    modele_yolo,
                    modele_resnet,
                    categories,
                    transformation,
                    seuil
                )

            nombre_avions = len(avions)

            st.markdown(
                """
                <div class="section-heading">
                    <span class="section-number">03</span>
                    <span>ANALYSIS RESULTS</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Cartes statistiques
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:

                st.metric(
                    "AIRCRAFT DETECTED",
                    nombre_avions
                )

            with col_stat2:

                st.metric(
                    "CLASSES IDENTIFIED",
                    len(set(a["class"] for a in avions))
                )

            with col_stat3:

                confiance_moyenne = (
                    sum(a["confidence"] for a in avions)
                    / nombre_avions * 100
                    if nombre_avions > 0
                    else 0
                )

                st.metric(
                    "AVERAGE CONFIDENCE",
                    f"{confiance_moyenne:.1f}%"
                )

            # Image annotée
            st.markdown(
                '<div class="card-title">DETECTION VISUALIZATION</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_annotee,
                width=500
            )

            # Liste des avions
            st.markdown(
                """
                <div class="section-heading">
                    <span class="section-number">04</span>
                    <span>CLASSIFICATION DETAILS</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            if nombre_avions == 0:

                st.warning(
                    "Aucun avion n'a été détecté dans cette image."
                )

            else:

                for avion in avions:

                    with st.expander(
                        f"✈  AIRCRAFT {avion['numero']}  |  "
                        f"{avion['class']}"
                    ):

                        c1, c2 = st.columns(2)

                        with c1:

                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="info-label">
                                        IDENTIFIED CLASS
                                    </div>
                                    <div class="result-class">
                                        {avion['class']}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with c2:

                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="info-label">
                                        CLASSIFICATION CONFIDENCE
                                    </div>
                                    <div class="result-class">
                                        {avion['confidence'] * 100:.2f}%
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.write("Detection coordinates")

                        st.json(
                            avion["coordinates"]
                        )

            # =================================================
            # TELECHARGEMENT
            # =================================================

            donnees_json = {
                fichier_image.name: avions
            }

            json_bytes = json.dumps(
                donnees_json,
                indent=4,
                ensure_ascii=False
            ).encode("utf-8")

            st.markdown(
                """
                <div class="section-heading">
                    <span class="section-number">05</span>
                    <span>EXPORT RESULTS</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:

                st.download_button(
                    "↓  DOWNLOAD ANNOTATED IMAGE",
                    data=_image_en_bytes(image_annotee),
                    file_name="aircraft_ai_resultat.png",
                    mime="image/png",
                    use_container_width=True
                )

            with col_dl2:

                st.download_button(
                    "↓  DOWNLOAD JSON DATA",
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

    st.markdown(
        """
        <div class="upload-placeholder">
            <div class="upload-icon">✈</div>
            <h3>READY FOR ANALYSIS</h3>
            <p>
                Import an aircraft image to begin detection
                and classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div class="footer-line"></div>
        <p>AIRCRAFT AI · INTELLIGENT AVIATION ANALYSIS</p>
        <span>POWERED BY YOLO & RESNET50</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Aircraft AI — Détection et classification d'avions par intelligence artificielle."
)
