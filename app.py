
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analyse du choléra — PNECHOL-MD",
    page_icon="🧪",
    layout="wide"
)

DOSSIER = Path(__file__).parent
FICHIER = DOSSIER / "resultats_comparaison_modeles_cholera.xlsx"

st.title("🧪 Analyse et comparaison des modèles de prédiction du choléra")
st.caption(
    "Mémoire de Master 2 — Intelligence artificielle et science des données"
)

if not FICHIER.exists():
    st.error("Le fichier de résultats est introuvable.")
    st.stop()

df = pd.read_excel(FICHIER, sheet_name="Comparaison")

pages = [
    "Accueil",
    "Résultats expérimentaux",
    "Comparaison des modèles",
    "Visualisations",
    "Méthodologie et limites"
]

page = st.sidebar.radio("Navigation", pages)

if page == "Accueil":
    st.header("Présentation du projet")

    st.write("""
    Cette application accompagne un mémoire de Master 2 consacré à la
    comparaison de techniques d'apprentissage automatique appliquées aux
    données du PNECHOL-MD.

    Quatre modèles sont étudiés :
    - Random Forest ;
    - XGBoost ;
    - TabNet ;
    - FT-Transformer (prototype).
    """)

    c1, c2 = st.columns(2)
    c1.metric("Modèles comparés", len(df))
    c2.metric("Métriques évaluées", 6)

    st.info(
        "Cette application est un outil de recherche. Elle ne constitue "
        "pas un dispositif de diagnostic médical."
    )

elif page == "Résultats expérimentaux":
    st.header("Résultats enregistrés")

    st.dataframe(
        df.style.format({
            c: "{:.4f}" for c in df.columns if c != "Modèle"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Télécharger les résultats Excel",
        data=FICHIER.read_bytes(),
        file_name=FICHIER.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif page == "Comparaison des modèles":
    st.header("Comparaison des performances")

    metrique = st.selectbox(
        "Choisir une métrique",
        [
            "Accuracy",
            "Précision décès",
            "Rappel décès",
            "F1 décès",
            "AUC-ROC",
            "AUC-PR"
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    barres = ax.bar(df["Modèle"], df[metrique])

    ax.set_title(f"Comparaison des modèles — {metrique}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)

    for barre, valeur in zip(barres, df[metrique]):
        ax.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height(),
            f"{valeur:.3f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Visualisations":
    st.header("Visualisation des métriques")

    choix = st.multiselect(
        "Sélectionner les métriques",
        [
            "Précision décès",
            "Rappel décès",
            "F1 décès",
            "AUC-ROC",
            "AUC-PR"
        ],
        default=["Précision décès", "Rappel décès", "F1 décès"]
    )

    if choix:
        fig, ax = plt.subplots(figsize=(11, 5))

        df.plot(
            x="Modèle",
            y=choix,
            kind="bar",
            ax=ax
        )

        ax.set_title("Comparaison des performances")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif page == "Méthodologie et limites":
    st.header("Méthodologie et limites")

    st.write("""
    Les résultats affichés proviennent du fichier Excel généré lors
    des expérimentations du mémoire.

    La classe DECEDE étant minoritaire, l'accuracy ne suffit pas pour
    évaluer les modèles. La précision, le rappel, le F1-score et
    l'AUC-PR doivent également être examinés.
    """)

    st.warning("""
    Cette version présente les résultats expérimentaux. Elle ne réalise
    pas de prédiction individuelle. Le FT-Transformer est un prototype
    simplifié et doit être présenté comme tel dans le mémoire.
    """)

st.divider()
st.caption("Projet de recherche — PNECHOL-MD")
