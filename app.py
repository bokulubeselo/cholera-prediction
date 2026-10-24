
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(
    page_title="CholeraPredict - PNECHOL-MD",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 CholeraPredict")
st.subheader("Analyse et prédiction des issues cliniques du choléra")

# --------------------------------------------------
# BARRE LATÉRALE DE NAVIGATION
# --------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Données & Exploration",
        "Analyse descriptive",
        "Comparaison des modèles",
        "Prédiction d'issue"
    ]
)

# Chemins vers les fichiers utiles
EXCEL_PATH = "resultats_comparaison_modeles_cholera.xlsx"
MODELS_DIR = "models"

if menu == "Accueil":
    st.header("Présentation du projet de mémoire")
    st.markdown("""
    Cette application est le fruit d'un travail de mémoire de Master 2 en Intelligence Artificielle et Data Science.
    
    **Objectif principal :**
    Comparer différentes techniques d'apprentissage automatique appliquées aux données épidémiologiques et cliniques du **PNECHOL-MD** afin de classifier l'issue des patients atteints de choléra (**GUÉRI** ou **DÉCÉDÉ**).

    **Modèles étudiés :**
    - **Random Forest** (Forêt d'arbres décisionnels)
    - **XGBoost** (Extreme Gradient Boosting)
    - **TabNet** (Architecture Deep Learning pour données tabulaires)
    - **FT-Transformer** (Feature Tokenizer Transformer léger)
    """)
    st.info("⚠️ Note importante : Les prédictions générées par cet outil ont un but académique et expérimental et ne sauraient remplacer un avis médical professionnel.")

elif menu == "Données & Exploration":
    st.header("Données & Exploration")
    st.markdown("Importez un extrait de données pour l'examiner de manière interactive.")
    
    fichier = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])
    if fichier is not None:
        try:
            if fichier.name.lower().endswith(".csv"):
                donnees = pd.read_csv(fichier)
            else:
                donnees = pd.read_excel(fichier)
            st.success("Fichier chargé avec succès !")
            st.write(f"**Dimensions :** {donnees.shape[0]} lignes, {donnees.shape[1]} colonnes.")
            st.dataframe(donnees.head(15))
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

elif menu == "Analyse descriptive":
    st.header("Analyse descriptive")
    st.markdown("Obtenez un aperçu rapide des statistiques globales d'un jeu de données.")
    
    fichier = st.file_uploader("Importer vos données pour analyse", type=["csv", "xlsx"], key="analyse")
    if fichier is not None:
        if fichier.name.lower().endswith(".csv"):
            donnees = pd.read_csv(fichier)
        else:
            donnees = pd.read_excel(fichier)
        
        st.subheader("Résumé statistique")
        st.dataframe(donnees.describe(include="all").T)
        
        st.subheader("Analyse des données manquantes")
        manquantes = donnees.isna().sum()
        manquantes = manquantes[manquantes > 0]
        if not manquantes.empty:
            st.bar_chart(manquantes)
        else:
            st.success("Aucune valeur manquante détectée !")

elif menu == "Comparaison des modèles":
    st.header("Comparaison des performances des modèles")
    
    if os.path.exists(EXCEL_PATH):
        df_comp = pd.read_excel(EXCEL_PATH, sheet_name="Comparaison")
        st.subheader("Tableau des performances obtenues")
        st.dataframe(df_comp.style.format({
            'Accuracy': '{:.3f}', 'Précision décès': '{:.3f}', 
            'Rappel décès': '{:.3f}', 'F1 décès': '{:.3f}', 
            'AUC-ROC': '{:.3f}', 'AUC-PR': '{:.3f}'
        }))
        
        # Graphique des métriques
        st.subheader("Visualisation comparative (Classe DÉCÉDÉ)")
        fig, ax = plt.subplots(figsize=(10, 4))
        df_comp.set_index('Modèle')[['Précision décès', 'Rappel décès', 'F1 décès']].plot(kind='bar', ax=ax)
        plt.title("Comparaison des scores d'évaluation")
        plt.ylabel("Valeur du score")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15)
        st.pyplot(fig)
    else:
        st.warning("Le fichier des résultats de comparaison est introuvable. Veuillez l'ajouter au projet.")

elif menu == "Prédiction d'issue":
    st.header("🔮 Prédiction d'issue clinique en temps réel")
    
    # Vérifier la présence des fichiers de modèles
    required_files = ["imp_num.joblib", "scaler.joblib", "imp_cat.joblib", "enc.joblib", "model_xgb.joblib", "num_cols.joblib", "cat_cols.joblib"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(MODELS_DIR, f))]
    
    if missing:
        st.error(f"Certains fichiers de modèles ou de prétraitement sont manquants dans '{MODELS_DIR}' : {missing}")
    else:
        # Charger les artefacts
        imp_num = joblib.load(f"{MODELS_DIR}/imp_num.joblib")
        scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
        imp_cat = joblib.load(f"{MODELS_DIR}/imp_cat.joblib")
        enc = joblib.load(f"{MODELS_DIR}/enc.joblib")
        xgb_model = joblib.load(f"{MODELS_DIR}/model_xgb.joblib")
        num_cols = joblib.load(f"{MODELS_DIR}/num_cols.joblib")
        cat_cols = joblib.load(f"{MODELS_DIR}/cat_cols.joblib")
        
        st.markdown("Saisissez les caractéristiques du patient pour évaluer le risque d'issue fatale (DÉCÉDÉ).")
        
        # Formulaire utilisateur
        col1, col2 = st.columns(2)
        
        with col1:
            sexe = st.selectbox("Sexe du patient", ["M", "F"])
            age = st.number_input("Âge (en années/mois)", min_value=0, max_value=120, value=25)
            statut_vaccin = st.selectbox("Statut Vaccinal", ["NON", "OUI", "INCONNU"])
            deg_deshydrat = st.selectbox("Degré de déshydratation", ["Sévère", "Modéré", "Léger/Aucun"])
            
        with col2:
            province = st.text_input("Province de provenance", value="SUD KIVU")
            zone_sante = st.text_input("Zone de santé", value="Bukavu")
            sem_epi = st.slider("Semaine Épidémiologique", 1, 53, 24)
            source_eau = st.text_input("Source d'approvisionnement en eau", value="Régie des eaux")

        # Bouton d'évaluation
        if st.button("Lancer la prédiction"):
            # Créer un dictionnaire de saisie vide avec toutes les colonnes requises
            all_features = num_cols + cat_cols
            user_data = {col: np.nan for col in all_features}
            
            # Remplir avec les données saisies par l'utilisateur
            user_data['Sexe'] = sexe
            user_data['Ageannee'] = str(age)
            user_data['StatutVaccinal'] = statut_vaccin
            user_data['Degrededeshydratation'] = deg_deshydrat
            user_data['ProvinceProVnance'] = province
            user_data['Zonedesanteprovenance'] = zone_sante
            user_data['SemEpi'] = sem_epi
            user_data['SourceApprovisionnement Eau'] = source_eau
            
            # Convertir en DataFrame pandas
            input_df = pd.DataFrame([user_data])
            
            # 1. Prétraitement Numérique
            input_num = imp_num.transform(input_df[num_cols])
            input_num_scaled = scaler.transform(input_num)
            
            # 2. Prétraitement Catégoriel
            input_cat_df = input_df[cat_cols].astype(object).where(input_df[cat_cols].notna(), np.nan)
            input_cat_imp = imp_cat.transform(input_cat_df)
            input_cat_enc = enc.transform(input_cat_imp)
            
            # Fusionner les matrices
            final_features = np.column_stack([input_num_scaled, input_cat_enc]).astype(np.float32)
            
            # Prédiction
            proba_deces = xgb_model.predict_proba(final_features)[0, 1]
            classe_pred = (proba_deces >= 0.5)
            
            # Affichage des résultats
            st.subheader("Résultat de la prédiction :")
            if classe_pred:
                st.error(f"🔴 Risque d'issue fatale ÉLEVÉ (Probabilité estimée de décès : {proba_deces*100:.2f}%)")
            else:
                st.success(f"🟢 Pronostic de guérison favorable (Probabilité de survie : {(1-proba_deces)*100:.2f}%)")

st.divider()
st.caption("CholeraPredict — Développé pour la valorisation scientifique des données de santé du PNECHOL-MD.")
