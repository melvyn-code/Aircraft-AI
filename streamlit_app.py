
            # --------------------------------------------------
            # TELECHARGEMENT DE L'IMAGE ANNOTÉE
            # --------------------------------------------------

            buffer_image = io.BytesIO()
            image_resultat.save(buffer_image, format="PNG")
            buffer_image.seek(0)

            nom_image = Path(fichier.name).stem

            st.download_button(
                label="Télécharger l'image analysée",
                data=buffer_image.getvalue(),
                file_name=f"{nom_image}_analyse.png",
                mime="image/png",
                use_container_width=True
            )

            # --------------------------------------------------
            # TELECHARGEMENT DU JSON
            # --------------------------------------------------

            donnees_json = {
                fichier.name: avions
            }

            json_bytes = json.dumps(
                donnees_json,
                indent=4,
                ensure_ascii=False
            ).encode("utf-8")

            st.download_button(
                label="Télécharger les résultats JSON",
                data=json_bytes,
                file_name=f"{nom_image}.json",
                mime="application/json",
                use_container_width=True
            )
