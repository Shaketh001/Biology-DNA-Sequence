# BioSeqML — DNA Sequence Classification Dashboard

An interactive Streamlit web application (plus zero-dependency standalone web build) for Computational Biology DNA sequence classification using extracted nucleotide features and classical machine learning (`DecisionTreeClassifier` & `RandomForestClassifier`).

## Features

1. **Dataset Management & 6-Feature Extraction**
   - Default dataset (`S1`: `ATGCGATC`, `S2`: `ATGCGTAC` -> `Group A`; `S3`: `GCGCGTAT`, `S4`: `GCGCGCGC` -> `Group B`).
   - Interactive dataset editor (`st.data_editor`), sidebar quick-add form, group filter, sortable Feature Table, CSV export, and real-time validation.
   - Extracts all 6 mandatory features: `Count of 'A'`, `Count of 'T'`, `Count of 'G'`, `Count of 'C'`, `Sequence Length`, and `GC Content (%) = ((G + C) / Length) * 100`.
   - Interactive Plotly GC Content (%) bar chart and group-level feature centroid summary.

2. **Model Training & Configuration (`scikit-learn`)**
   - Choice of **Random Forest Classifier** or **Decision Tree Classifier** (Gini/Entropy, `max_depth`, `n_estimators`, `max_features`, `min_samples_split`, `random_state`).
   - Validation strategies: **Full-Fit**, **Leave-One-Out CV (LOOCV)**, and **Stratified Train/Test Split**.
   - Displays Accuracy, Macro-F1, interactive Confusion Matrix heatmap, per-class Precision/Recall/F1, normalized Feature Importance bar chart, and learned split rules (`export_text`).

3. **Inference / Prediction Engine**
   - Pre-filled with target sequence `ATGCGCGATC` (`Test`).
   - Strict `A/T/G/C` nucleotide validator, visual nucleotide strip, 6-feature badge cards, prominent predicted group & probability breakdown, decision-path/tree-vote explanation, and OOD experiment presets.

4. **Academic Deliverable Mode (Analysis & Discussion)**
   - Live data-backed discussion covering feature utility, sanity checks, out-of-distribution behavior, real-world genomics applications, and limitations of AI/ML in genomics.
   - One-click **Export Report (PDF)** and **Download Markdown (.md)**.

---

## Deploy to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` (plus optional `.streamlit/config.toml`) to the root of your GitHub repository — `app.py` is self-contained with no local package subfolder dependencies.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud will automatically install dependencies from `requirements.txt` and apply `.streamlit/config.toml`.

## Run Locally with Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Standalone Fallback (Zero Dependencies)

If running in an offline environment without `streamlit` or `scikit-learn`, open `index.html` directly or run:

```bash
python server.py
```
