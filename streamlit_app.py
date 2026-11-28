import base64
import io
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
YOLO_PATH = MODELS_DIR / "modele_detection_avions.pt"
CLASSIFIER_PATH = MODELS_DIR / "modele_classification_avions.pth"
CSS_PATH = ROOT / "static" / "style.css"

IMAGE_SIZE_DEFAULT = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# STYLE
# ============================================================

def charger_css():
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )


st.set_page_config(
    page_title="Aircraft AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)
charger_css()


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

@st.cache_resource
def charger_modeles():
    if not YOLO_PATH.exists():
        raise FileNotFoundError(f"Modèle YOLO introuvable : {YOLO_PATH}")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Modèle de classification introuvable : {CLASSIFIER_PATH}"
        )

    modele_detection = YOLO(str(YOLO_PATH))

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    categories = checkpoint.get("categories")
    if not categories:
        raise ValueError(
            "Le checkpoint de classification ne contient pas la clé 'categories'."
        )

    taille_image = int(checkpoint.get("taille_image", IMAGE_SIZE_DEFAULT))

    # Les catégories peuvent être une liste ou un dictionnaire index -> classe.
    if isinstance(categories, dict):
        try:
            categories = [
                categories[k] for k in sorted(categories, key=lambda x: int(x))
            ]
        except (ValueError, TypeError):
            categories = list(categories.values())
    else:
        categories = list(categories)

    modele_classification = models.resnet50(weights=None)
    modele_classification.fc = torch.nn.Linear(
        modele_classification.fc.in_features,
        len(categories)
    )

    state_dict = checkpoint.get("model_state", checkpoint)
    modele_classification.load_state_dict(state_dict)
    modele_classification.to(DEVICE)
    modele_classification.eval()

    preprocessing = transforms.Compose([
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
        preprocessing
    )


# ============================================================
# ANALYSE D'UNE IMAGE
# ============================================================

def obtenir_police(taille=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", taille)
    except OSError:
        return ImageFont.load_default()


def analyser_image(
    image_pil,
    modele_detection,
    modele_classification,
    categories,
    preprocessing,
    seuil_detection
):
    image_originale = image_pil.convert("RGB")
    image_annotee = image_originale.copy()
    dessin = ImageDraw.Draw(image_annotee)
    police = obtenir_police(18)

    largeur, hauteur = image_originale.size
    detections_json = []

    resultats = modele_detection.predict(
        source=image_originale,
        conf=seuil_detection,
        verbose=False
    )

    for resultat in resultats:
        if resultat.boxes is None:
            continue

        for boite in resultat.boxes:
            coords = boite.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in coords]

            # Garantir que les coordonnées restent dans l'image.
            x1 = max(0, min(x1, largeur - 1))
            y1 = max(0, min(y1, hauteur - 1))
            x2 = max(x1 + 1, min(x2, largeur))
            y2 = max(y1 + 1, min(y2, hauteur))

            crop = image_originale.crop((x1, y1, x2, y2))

            if crop.width == 0 or crop.height == 0:
                continue

            entree = preprocessing(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = modele_classification(entree)
                indice = int(torch.argmax(logits, dim=1).item())

            classe = str(categories[indice])

            # Format demandé : uniquement class et coordinates.
            detections_json.append({
                "class": classe,
                "coordinates": {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2
                }
            })

            couleur = (56, 248, 165)
            dessin.rectangle((x1, y1, x2, y2), outline=couleur, width=3)

            texte = classe
            bbox_texte = dessin.textbbox((x1, y1), texte, font=police)
            largeur_texte = bbox_texte[2] - bbox_texte[0]
            hauteur_texte = bbox_texte[3] - bbox_texte[1]
            haut_texte = max(0, y1 - hauteur_texte - 8)

            dessin.rectangle(
                (x1, haut_texte, x1 + largeur_texte + 10, y1),
                fill=couleur
            )
            dessin.text(
                (x1 + 5, haut_texte + 2),
                texte,
                fill=(7, 20, 38),
                font=police
            )

    return image_annotee, detections_json


# ============================================================
# SESSION
# ============================================================

if "resultats_json" not in st.session_state:
    st.session_state.resultats_json = {}

if "images_analysees" not in st.session_state:
    st.session_state.images_analysees = {}

if "images_originales" not in st.session_state:
    st.session_state.images_originales = {}


# ============================================================
# ENTETE
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">INTELLIGENCE ARTIFICIELLE · AÉRONAUTIQUE</div>
    <h1>Aircraft <span>AI</span></h1>
    <p>Détection et classification intelligente d'aéronefs</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    Importez vos images, détectez les avions et exportez les résultats
    avec l'image annotée directement intégrée au fichier JSON.
</div>
""", unsafe_allow_html=True)


# ============================================================
# PARAMETRES ET MODELES
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    seuil = st.slider(
        "Seuil de détection YOLO",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )
    st.caption("Un seuil plus élevé réduit les détections moins sûres.")

    st.markdown("---")
    st.markdown("### 📦 Modèles")
    st.caption("YOLO : détection des avions")
    st.caption("ResNet50 : classification des aéronefs")

try:
    with st.spinner("Chargement des modèles IA..."):
        (
            modele_detection,
            modele_classification,
            categories,
            preprocessing
        ) = charger_modeles()
    modeles_ok = True
except Exception as erreur:
    modeles_ok = False
    st.error("Impossible de charger les modèles IA.")
    st.code(str(erreur))
    st.info(
        "Vérifiez que les deux fichiers de poids sont présents dans le dossier "
        "'models' et que le fichier packages.txt est à la racine du dépôt."
    )


# ============================================================
# IMPORT DES IMAGES
# ============================================================

st.markdown('<div class="section-heading">01 · Importer des images</div>',
            unsafe_allow_html=True)

fichiers_images = st.file_uploader(
    "Sélectionnez une ou plusieurs images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Formats acceptés : JPG, JPEG, PNG et WEBP."
)

if fichiers_images:
    st.success(f"{len(fichiers_images)} image(s) sélectionnée(s).")

    if modeles_ok:
        for fichier in fichiers_images:
            try:
                image_originale = Image.open(fichier).convert("RGB")
            except Exception:
                st.warning(f"Le fichier {fichier.name} n'est pas une image valide.")
                continue

            st.markdown(f"### 🖼️ {fichier.name}")
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image_originale,
                    caption="Image originale",
                    use_container_width=True
                )

            with col2:
                if fichier.name in st.session_state.images_analysees:
                    image_bytes = st.session_state.images_analysees[fichier.name]
                    st.image(
                        Image.open(io.BytesIO(image_bytes)),
                        caption="Dernière image analysée",
                        use_container_width=True
                    )
                else:
                    st.info("Cette image n'a pas encore été analysée.")

            if st.button(
                f"🔍 Analyser {fichier.name}",
                key=f"analyse_{fichier.name}",
                use_container_width=True
            ):
                with st.spinner(f"Analyse de {fichier.name}..."):
                    image_annotee, detections = analyser_image(
                        image_originale,
                        modele_detection,
                        modele_classification,
                        categories,
                        preprocessing,
                        seuil
                    )

                    # Encoder l'image annotée en JPEG pour l'intégrer au JSON.
                    buffer_jpg = io.BytesIO()
                    image_annotee.save(
                        buffer_jpg,
                        format="JPEG",
                        quality=95
                    )
                    image_base64 = base64.b64encode(
                        buffer_jpg.getvalue()
                    ).decode("utf-8")

                    # Format JSON demandé : nom d'image -> liste.
                    st.session_state.resultats_json[fichier.name] = [
                        {"image": image_base64},
                        *detections
                    ]

                    # Garder l'image annotée dans la session pour les boutons.
                    buffer_png = io.BytesIO()
                    image_annotee.save(buffer_png, format="PNG")
                    st.session_state.images_analysees[fichier.name] = (
                        buffer_png.getvalue()
                    )

                    buffer_original = io.BytesIO()
                    image_originale.save(buffer_original, format="PNG")
                    st.session_state.images_originales[fichier.name] = (
                        buffer_original.getvalue()
                    )

                st.success(
                    f"Analyse terminée : {len(detections)} avion(s) détecté(s)."
                )
                st.rerun()


# ============================================================
# RESULTATS ET TELECHARGEMENTS PAR IMAGE
# ============================================================

if st.session_state.resultats_json:
    st.markdown(
        '<div class="section-heading">02 · Résultats analysés</div>',
        unsafe_allow_html=True
    )

    for nom_fichier, elements in st.session_state.resultats_json.items():
        detections = [
            element for element in elements
            if "class" in element and "coordinates" in element
        ]

        image_bytes = st.session_state.images_analysees.get(nom_fichier)

        with st.expander(
            f"✈️ {nom_fichier} — {len(detections)} avion(s)",
            expanded=True
        ):
            if image_bytes:
                image_annotee = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.image(
                    image_annotee,
                    caption="Image analysée avec annotations",
                    use_container_width=True
                )

                nom_sans_extension = Path(nom_fichier).stem

                col_png, col_jpg = st.columns(2)

                with col_png:
                    st.download_button(
                        "📷 Télécharger en PNG",
                        data=image_bytes,
                        file_name=f"{nom_sans_extension}_analyse.png",
                        mime="image/png",
                        key=f"png_{nom_fichier}",
                        use_container_width=True
                    )

                with col_jpg:
                    buffer_jpg = io.BytesIO()
                    image_annotee.save(
                        buffer_jpg,
                        format="JPEG",
                        quality=95
                    )
                    st.download_button(
                        "📷 Télécharger en JPG",
                        data=buffer_jpg.getvalue(),
                        file_name=f"{nom_sans_extension}_analyse.jpg",
                        mime="image/jpeg",
                        key=f"jpg_{nom_fichier}",
                        use_container_width=True
                    )

            st.json(detections)


# ============================================================
# TELECHARGEMENT JSON GLOBAL
# ============================================================

if st.session_state.resultats_json:
    st.markdown(
        '<div class="section-heading">03 · Exporter les données</div>',
        unsafe_allow_html=True
    )

    json_bytes = json.dumps(
        st.session_state.resultats_json,
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
        "Le JSON contient l'image annotée encodée en Base64, suivie des "
        "détections avec leurs classes et coordonnées."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <div class="footer-line"></div>
    <p>Aircraft <span>AI</span> · Détection intelligente d'aéronefs</p>
    <small>Powered by Python · YOLO · ResNet50 · Streamlit</small>
</div>
""", unsafe_allow_html=True)

    </small>
</div>
""", unsafe_allow_html=True)
