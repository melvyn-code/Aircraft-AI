
import streamlit as st
import torch
import torch.nn as nn
import json
import io

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_PATH = BASE_DIR / "models" / "modele_detection_avions.pt"
CLASSIFIER_PATH = BASE_DIR / "models" / "modele_classification_avions.pth"
CSS_PATH = BASE_DIR / "static" / "style.css"

st.set_page_config(
    page_title="Aircraft AI",
    page_icon="✈️",
    layout="wide"
)


# ==================================================
# CHARGEMENT DU CSS
# ==================================================

def charger_css():

    if CSS_PATH.exists():

        css = CSS_PATH.read_text(encoding="utf-8")

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


charger_css()


# ==================================================
# HEADER
# ==================================================

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


# ==================================================
# CHARGEMENT DES MODELES
# ==================================================

@st.cache_resource
def charger_modeles():

    if not YOLO_PATH.exists():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {YOLO_PATH}"
        )

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Modèle ResNet50 introuvable : {CLASSIFIER_PATH}"
        )

    # Chargement du modèle YOLO
    modele_detection = YOLO(str(YOLO_PATH))

    # Chargement du checkpoint ResNet50
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
        raise ValueError("Checkpoint de classification invalide.")

    categories = checkpoint.get("categories")

    if not categories:
        raise ValueError("Les catégories sont absentes du modèle.")

    if isinstance(categories, dict):

        categories = [
            categories[key]
            for key in sorted(categories.keys())
        ]

    categories = list(categories)

    taille_image = int(
        checkpoint.get("taille_image", 224)
    )

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

    # Correction des éventuels préfixes
    poids_corriges = {}

    for key, value in poids.items():

        nouveau_key = key.replace("module.", "", 1)

        poids_corriges[nouveau_key] = value

    modele_classification.load_state_dict(
        poids_corriges,
        strict=True
    )

    modele_classification.eval()

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
        modele_detection,
        modele_classification,
        categories,
        transformation
    )


# ==================================================
# CLASSIFICATION D'UN AVION
# ==================================================

def classifier_avion(
    image_avion,
    modele_classification,
    categories,
    transformation
):

    image_avion = image_avion.convert("RGB")

    tensor_image = transformation(
        image_avion
    ).unsqueeze(0)

    with torch.no_grad():

        predictions = modele_classification(
            tensor_image
        )

        probabilites = torch.softmax(
            predictions,
            dim=1
        )

        confiance, indice = torch.max(
            probabilites,
            dim=1
        )

    classe = categories[indice.item()]

    confiance = float(confiance.item())

    return classe, confiance


# ==================================================
# ANALYSE DE L'IMAGE
# ==================================================

def analyser_image(image):

    (
        modele_detection,
        modele_classification,
        categories,
        transformation
    ) = charger_modeles()

    image_originale = image.convert("RGB")

    image_annotee = image_originale.copy()

    dessin = ImageDraw.Draw(image_annotee)

    largeur, hauteur = image_annotee.size

    try:

        police = ImageFont.truetype(
            "arial.ttf",
            18
        )

    except OSError:

        police = ImageFont.load_default()

    # Détection YOLO
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

            x1, y1, x2, y2 = [
                int(round(valeur))
                for valeur in coordonnees
            ]

            # Limitation aux dimensions de l'image
            x1 = max(0, min(x1, largeur))
            y1 = max(0, min(y1, hauteur))
            x2 = max(0, min(x2, largeur))
            y2 = max(0, min(y2, hauteur))

            if x2 <= x1 or y2 <= y1:
                continue

            # Découpage de l'avion
            crop = image_originale.crop(
                (x1, y1, x2, y2)
            )

            # Classification ResNet50
            classe, confiance = classifier_avion(
                crop,
                modele_classification,
                categories,
                transformation
            )

            # Dessin du rectangle
            dessin.rectangle(
                (x1, y1, x2, y2),
                outline="#38bdf8",
                width=4
            )

            texte = f"{classe} ({confiance:.1%})"

            position_texte = (
                x1,
                max(0, y1 - 28)
            )

            boite_texte = dessin.textbbox(
                position_texte,
                texte,
                font=police
            )

            dessin.rectangle(
                boite_texte,
                fill="#071426"
            )

            dessin.text(
                position_texte,
                texte,
                fill="#38bdf8",
                font=police
            )

            # Enregistrement au format JSON demandé
            avions_detectes.append({

                "class": str(classe),

                "coordinates": {

                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2

                }

            })

            numero += 1

    return image_annotee, avions_detectes


# ==================================================
# INITIALISATION DES RESULTATS
# ==================================================

if "resultats_json" not in st.session_state:

    st.session_state.resultats_json = {}


if "images_analysees" not in st.session_state:

    st.session_state.images_analysees = {}


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## ✈️ Aircraft AI")

    st.caption("Plateforme d'analyse aéronautique")

    st.markdown("---")

    st.markdown("### Fonctionnalités")

    st.markdown("""
    - Détection des aéronefs
    - Classification automatique
    - Identification des coordonnées
    - Exportation JSON
    - Téléchargement des images
    """)

    st.markdown("---")

    st.caption("Python · YOLO · ResNet50 · Streamlit")


# ==================================================
# INTERFACE PRINCIPALE
# ==================================================

st.markdown("""
<div class="section-heading">
    <span class="section-number">01</span>
    <div>
        <h2>Analyse d'une image</h2>
        <p>
            Importe une image pour détecter et identifier les aéronefs.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


col_upload, col_info = st.columns([1.5, 1])

with col_upload:

    fichier = st.file_uploader(
        "Importer une image",
        type=["jpg", "jpeg", "png"],
        key="upload_image"
    )


with col_info:

    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">◎</div>
        <h3>Comment ça fonctionne ?</h3>
        <p>
            YOLO détecte les aéronefs dans l'image.
            ResNet50 analyse ensuite chaque zone
            pour identifier sa catégorie.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# ANALYSE
# ==================================================

if fichier is not None:

    image = Image.open(fichier).convert("RGB")

    st.markdown("### Image importée")

    st.image(
        image,
        caption=fichier.name,
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
                "Détection et classification en cours..."
            ):

                image_resultat, avions = analyser_image(image)

            # Sauvegarde des résultats de cette image
            st.session_state.resultats_json[fichier.name] = avions

            # Sauvegarde de l'image annotée en mémoire
            buffer_image = io.BytesIO()

            image_resultat.save(
                buffer_image,
                format="PNG"
            )

            st.session_state.images_analysees[fichier.name] = (
                buffer_image.getvalue()
            )

            st.success("Analyse terminée avec succès !")

        except Exception as erreur:

            st.error("Une erreur est survenue pendant l'analyse.")

            st.exception(erreur)


# ==================================================
# AFFICHAGE DU DERNIER RESULTAT
# ==================================================

if fichier is not None and fichier.name in st.session_state.resultats_json:

    avions = st.session_state.resultats_json[fichier.name]

    image_bytes = st.session_state.images_analysees[fichier.name]

    image_resultat = Image.open(
        io.BytesIO(image_bytes)
    )

    st.markdown("## Résultats de l'analyse")

    col_image, col_stats = st.columns([1.6, 1])

    with col_image:

        st.image(
            image_resultat,
            caption="Image analysée",
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

        for index, avion in enumerate(avions, start=1):

            st.markdown(
                f"""
                <div class="aircraft-card">
                    <div>
                        <span class="aircraft-index">
                            AÉRONEF {index:02d}
                        </span>
                        <h3>{avion["class"]}</h3>
                    </div>
                    <div class="confidence">
                        {avion["class"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "Aucun aéronef n'a été détecté dans cette image."
        )


# ==================================================
# TELECHARGEMENT DE L'IMAGE
# ==================================================

    st.markdown("### Télécharger l'image analysée")

    nom_image = Path(fichier.name).stem

    st.download_button(
        label="📷 Télécharger l'image analysée",
        data=image_bytes,
        file_name=f"{nom_image}_analyse.png",
        mime="image/png",
        use_container_width=True
    )


# ==================================================
# TELECHARGEMENT DU JSON GLOBAL
# ==================================================

if st.session_state.resultats_json:

    st.markdown("---")

    st.markdown("## Exportation des résultats")

    donnees_json = st.session_state.resultats_json

    json_bytes = json.dumps(
        donnees_json,
        indent=2,
        ensure_ascii=False
    ).encode("utf-8")

    st.download_button(
        label="📄 Télécharger tous les résultats JSON",
        data=json_bytes,
        file_name="resultats_aircraft_ai.json",
        mime="application/json",
        use_container_width=True
    )

    st.caption(
        f"{len(donnees_json)} image(s) enregistrée(s) dans le fichier JSON."
    )

    if st.button("Effacer tous les résultats"):

        st.session_state.resultats_json = {}

        st.session_state.images_analysees = {}

        st.rerun()



# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    <div class="footer-line"></div>
    <p>
        Aircraft <span>AI</span> · Détection intelligente d'aéronefs
    </p>
    <small>
        Powered by Python · YOLO · ResNet50 · Streamlit
    </small>
</div>
""", unsafe_allow_html=True)
