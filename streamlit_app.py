
from pathlib import Path
import io
import json

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CSS_PATH = BASE_DIR / "static" / "style.css"

YOLO_PATH = MODELS_DIR / "modele_detection_avions.pt"
CLASSIFIER_PATH = MODELS_DIR / "modele_classification_avions.pth"

st.set_page_config(
    page_title="Aircraft AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CSS
# --------------------------------------------------

def charger_css():
    if CSS_PATH.exists():
        css = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


charger_css()


# --------------------------------------------------
# HTML HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">ARTIFICIAL INTELLIGENCE · AVIATION</div>
    <h1>Aircraft <span>AI</span></h1>
    <p class="hero-subtitle">
        Une intelligence artificielle dédiée à la détection
        et à la classification des aéronefs.
    </p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def charger_modeles():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from ultralytics import YOLO

    if not YOLO_PATH.exists():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {YOLO_PATH}"
        )

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Modèle de classification introuvable : {CLASSIFIER_PATH}"
        )

    # Chargement du modèle de détection
    modele_detection = YOLO(str(YOLO_PATH))

    # Chargement du checkpoint de classification
    try:
        checkpoint = torch.load(
            str(CLASSIFIER_PATH),
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            str(CLASSIFIER_PATH),
            map_location="cpu"
        )

    if not isinstance(checkpoint, dict):
        raise ValueError("Le fichier de classification est invalide.")

    categories = checkpoint.get("categories")

    if not categories:
        raise ValueError(
            "La liste des catégories est absente du checkpoint."
        )

    if isinstance(categories, dict):
        categories = [
            categories[key]
            for key in sorted(categories.keys())
        ]

    categories = list(categories)

    taille_image = int(checkpoint.get("taille_image", 224))

    # Architecture ResNet50
    modele_classification = models.resnet50(weights=None)
    modele_classification.fc = nn.Linear(
        modele_classification.fc.in_features,
        len(categories)
    )

    poids = checkpoint.get(
        "model_state",
        checkpoint.get("state_dict", checkpoint)
    )

    # Retire un éventuel préfixe "module."
    poids_corriges = {}
    for key, value in poids.items():
        nouveau_key = key.replace("module.", "", 1)
        poids_corriges[nouveau_key] = value

    modele_classification.load_state_dict(
        poids_corriges,
        strict=True
    )

    modele_classification.eval()

    # Prétraitement identique à celui d'un ResNet50 ImageNet
    transformation = transforms.Compose([
        transforms.Resize((taille_image, taille_image)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return (
        modele_detection,
        modele_classification,
        categories,
        transformation,
        torch
    )


# --------------------------------------------------
# AIRCRAFT CLASSIFICATION
# --------------------------------------------------

def classifier_avion(
    image_avion,
    modele_classification,
    categories,
    transformation,
    torch
):
    image_avion = image_avion.convert("RGB")
    tensor_image = transformation(image_avion).unsqueeze(0)

    with torch.no_grad():
        predictions = modele_classification(tensor_image)
        probabilites = torch.softmax(predictions, dim=1)
        confiance, indice = torch.max(probabilites, dim=1)

    classe = categories[indice.item()]
    confiance = float(confiance.item())

    return classe, confiance


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

def analyser_image(image):
    (
        modele_detection,
        modele_classification,
        categories,
        transformation,
        torch
    ) = charger_modeles()

    image_originale = image.convert("RGB")
    image_annotee = image_originale.copy()

    dessin = ImageDraw.Draw(image_annotee)
    largeur, hauteur = image_annotee.size

    try:
        police = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        police = ImageFont.load_default()

    resultats = modele_detection.predict(
        source=image_originale,
        conf=0.25,
        verbose=False
    )

    avions_detectes = []
    numero = 1

    for resultat in resultats:
        if resultat.boxes is None:
            continue

        for boite in resultat.boxes:
            coordonnees = boite.xyxy[0].cpu().tolist()
            confiance_detection = float(boite.conf[0].cpu())

            x1, y1, x2, y2 = [
                int(max(0, valeur))
                for valeur in coordonnees
            ]

            x2 = min(x2, largeur)
            y2 = min(y2, hauteur)

            if x2 <= x1 or y2 <= y1:
                continue

            # Découpage de l'aéronef détecté
            crop = image_originale.crop((x1, y1, x2, y2))

            # Classification de l'aéronef
            classe, confiance_classification = classifier_avion(
                crop,
                modele_classification,
                categories,
                transformation,
                torch
            )

            # Dessin du rectangle
            dessin.rectangle(
                (x1, y1, x2, y2),
                outline="#38bdf8",
                width=4
            )

            texte = f"{classe} ({confiance_classification:.1%})"

            boite_texte = dessin.textbbox(
                (x1, max(0, y1 - 28)),
                texte,
                font=police
            )

            dessin.rectangle(
                boite_texte,
                fill="#071426"
            )

            dessin.text(
                (x1, max(0, y1 - 28)),
                texte,
                fill="#38bdf8",
                font=police
            )

            avions_detectes.append({
                "numero": numero,
                "class": str(classe),
                "confidence": round(confiance_classification, 4),
                "detection_confidence": round(
                    confiance_detection, 4
                ),
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

            numero += 1

    return image_annotee, avions_detectes


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ✈️ Aircraft AI")
    st.caption("Plateforme d'analyse aéronautique")

    st.markdown("---")
    st.markdown("### Fonctionnalités")

    st.markdown("""
    - Détection des aéronefs
    - Classification automatique
    - Estimation de confiance
    - Export des résultats JSON
    """)

    st.markdown("---")
    st.caption("Développé avec Python, YOLO et ResNet50")


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

st.markdown("""
<div class="section-heading">
    <span class="section-number">01</span>
    <div>
        <h2>Analyse d'une image</h2>
        <p>Importe une image pour détecter et identifier les aéronefs.</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_upload, col_info = st.columns([1.5, 1])

with col_upload:
    fichier = st.file_uploader(
        "Importer une image",
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés : JPG, JPEG et PNG."
    )

with col_info:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">◎</div>
        <h3>Comment ça fonctionne ?</h3>
        <p>
            YOLO repère les aéronefs dans l'image.
            ResNet50 analyse ensuite chaque zone détectée
            pour identifier sa catégorie.
        </p>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# DISPLAY AND ANALYSIS
# --------------------------------------------------

if fichier is not None:
    image = Image.open(fichier).convert("RGB")

    st.markdown("### Image importée")
    st.image(
        image,
        caption="Image originale",
        use_container_width=True
    )

    lancer_analyse = st.button(
        "Lancer l'analyse",
        type="primary",
        use_container_width=True
    )

    if lancer_analyse:
        try:
            with st.spinner(
                "Analyse en cours : détection et classification..."
            ):
                image_resultat, avions = analyser_image(image)

            st.success("Analyse terminée.")

            st.markdown("## Résultat de l'analyse")

            col_image, col_stats = st.columns([1.6, 1])

            with col_image:
                st.image(
                    image_resultat,
                    caption="Aéronefs détectés et classifiés",
                    use_container_width=True
                )

            with col_stats:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="result-label">AÉRONEFS DÉTECTÉS</p>
                        <div class="result-number">{len(avions)}</div>
                        <p class="result-description">
                            Nombre total d'aéronefs identifiés
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if avions:
                st.markdown("### Détails des aéronefs")

                for avion in avions:
                    st.markdown(
                        f"""
                        <div class="aircraft-card">
                            <div>
                                <span class="aircraft-index">
                                    AÉRONEF {avion["numero"]:02d}
                                </span>
                                <h3>{avion["class"]}</h3>
                            </div>
                            <div class="confidence">
                                {avion["confidence"]:.1%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.caption(
                        "Confiance de détection : "
                        f"{avion['detection_confidence']:.1%}"
                    )
            else:
                st.warning(
                    "Aucun aéronef n'a été détecté dans cette image."
                )

            # Export JSON
            donnees_json = {
                fichier.name: avions
            }

            json_bytes = json.dumps(
                donnees_json,
                indent=4,
                ensure_ascii=False
            ).encode("utf-8")

            st.download_button(
                label="Télécharger les résultats en JSON",
                data=json_bytes,
                file_name="resultats_aircraft_ai.json",
                mime="application/json",
                use_container_width=True
            )

        except Exception as erreur:
            st.error("Une erreur est survenue pendant l'analyse.")
            st.exception(erreur)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    <div class="footer-line"></div>
    <p>Aircraft <span>AI</span> · Détection intelligente d'aéronefs</p>
    <small>Powered by Python · YOLO · ResNet50 · Streamlit</small>
</div>
""", unsafe_allow_html=True)
