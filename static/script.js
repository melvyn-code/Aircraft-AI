
// ======================================================
// AIRCRAFT AI - SCRIPT PRINCIPAL
// ======================================================


// Récupération des éléments HTML
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyseButton = document.getElementById("analyseButton");
const status = document.getElementById("status");

const results = document.getElementById("results");
const aircraftCount = document.getElementById("aircraftCount");
const resultImage = document.getElementById("resultImage");

const downloadImage = document.getElementById("downloadImage");
const downloadJson = document.getElementById("downloadJson");

const aircraftList = document.getElementById("aircraftList");


// ======================================================
// 1. SELECTION D'UNE IMAGE
// ======================================================

imageInput.addEventListener("change", function () {

    const fichier = imageInput.files[0];

    if (!fichier) {
        return;
    }

    // Vérification du format
    const formatsAutorises = [
        "image/jpeg",
        "image/png",
        "image/tiff"
    ];

    const extension = fichier.name.split(".").pop().toLowerCase();
    const extensionsAutorisees = ["jpg", "jpeg", "png", "tif", "tiff"];

    if (
        !formatsAutorises.includes(fichier.type) &&
        !extensionsAutorisees.includes(extension)
    ) {
        afficherStatut(
            "Format non autorisé. Choisissez une image JPG, PNG ou TIFF.",
            "error"
        );

        imageInput.value = "";
        preview.removeAttribute("src");
        return;
    }

    // Vérification de la taille (20 Mo maximum)
    if (fichier.size > 20 * 1024 * 1024) {

        afficherStatut(
            "L'image est trop volumineuse. Maximum : 20 Mo.",
            "error"
        );

        imageInput.value = "";
        preview.removeAttribute("src");
        return;
    }

    // Affichage de l'aperçu
    const imageURL = URL.createObjectURL(fichier);

    preview.src = imageURL;
    preview.style.display = "block";

    // Réinitialisation des anciens résultats
    results.style.display = "none";
    aircraftList.replaceChildren();

    afficherStatut(
        "Image sélectionnée. Vous pouvez lancer l'analyse.",
        "success"
    );

});


// ======================================================
// 2. LANCEMENT DE L'ANALYSE
// ======================================================

analyseButton.addEventListener("click", async function () {

    const fichier = imageInput.files[0];

    if (!fichier) {

        afficherStatut(
            "Veuillez d'abord sélectionner une image.",
            "error"
        );

        return;
    }

    // Désactivation du bouton pendant l'analyse
    analyseButton.disabled = true;
    analyseButton.textContent = "Analyse en cours...";

    afficherStatut(
        "Détection et classification des avions en cours...",
        "loading"
    );

    results.style.display = "none";

    // Préparation de l'image pour Flask
    const formulaire = new FormData();

    formulaire.append("image", fichier);

    try {

        // Envoi de l'image au serveur Python
        const reponse = await fetch("/analyser", {
            method: "POST",
            body: formulaire
        });

        const donnees = await reponse.json();

        // Vérification des erreurs du serveur
        if (!reponse.ok || !donnees.success) {

            throw new Error(
                donnees.erreur || "Une erreur est survenue pendant l'analyse."
            );

        }

        // Affichage des résultats
        afficherResultats(donnees);

        afficherStatut(
            "Analyse terminée avec succès !",
            "success"
        );

    } catch (erreur) {

        console.error("Erreur Aircraft AI :", erreur);

        afficherStatut(
            erreur.message || "Impossible de contacter le serveur.",
            "error"
        );

    } finally {

        // Réactivation du bouton
        analyseButton.disabled = false;
        analyseButton.textContent = "Analyser l'image";

    }

});


// ======================================================
// 3. AFFICHAGE DES RESULTATS
// ======================================================

function afficherResultats(donnees) {

    // Nombre d'avions détectés
    aircraftCount.textContent = donnees.nombre_avions;

    // Image annotée
    resultImage.src = donnees.image_annotee + "?t=" + Date.now();
    resultImage.style.display = "block";

    // Liens de téléchargement
    downloadImage.href = donnees.telechargement_image;
    downloadImage.download = "image_annotee.jpg";

    downloadJson.href = donnees.telechargement_json;
    downloadJson.download = "resultats.json";

    // Suppression des anciens avions affichés
    aircraftList.replaceChildren();

    // Affichage de chaque avion
    if (donnees.avions.length === 0) {

        const message = document.createElement("p");

        message.textContent = "Aucun avion n'a été détecté dans cette image.";

        aircraftList.appendChild(message);

    } else {

        donnees.avions.forEach(function (avion, index) {

            const carte = document.createElement("div");
            carte.className = "aircraft-item";

            const titre = document.createElement("h3");
            titre.textContent = "Avion " + (index + 1);

            const classe = document.createElement("p");
            classe.textContent = "Classe : " + avion.class;

            const confiance = document.createElement("p");

            confiance.textContent =
                "Confiance : " +
                (avion.confidence * 100).toFixed(1) +
                "%";

            const coordonnees = document.createElement("p");

            coordonnees.textContent =
                "Coordonnées : " +
                "xmin = " + avion.coordinates.xmin +
                ", ymin = " + avion.coordinates.ymin +
                ", xmax = " + avion.coordinates.xmax +
                ", ymax = " + avion.coordinates.ymax;

            carte.appendChild(titre);
            carte.appendChild(classe);
            carte.appendChild(confiance);
            carte.appendChild(coordonnees);

            aircraftList.appendChild(carte);

        });

    }

    // Affichage de la section résultats
    results.style.display = "block";

}


// ======================================================
// 4. AFFICHAGE DES MESSAGES
// ======================================================

function afficherStatut(message, type) {

    status.textContent = message;

    status.className = "";

    status.classList.add("status-" + type);

}


// ======================================================
// 5. VERIFICATION DU CHARGEMENT DE LA PAGE
// ======================================================

document.addEventListener("DOMContentLoaded", function () {

    console.log("Aircraft AI est prêt.");

});
