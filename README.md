# BioSeqML — DNA Sequence Classification Dashboard

An interactive, Streamlit-inspired scientific web dashboard for Computational Biology DNA sequence classification using extracted nucleotide features and classical machine learning (Decision Tree & Random Forest).

## Features

1. **Dataset Management & 6-Feature Extraction**
   - Default dataset (`S1`: `ATGCGATC`, `S2`: `ATGCGTAC` -> `Group A`; `S3`: `GCGCGTAT`, `S4`: `GCGCGCGC` -> `Group B`).
   - Inline table editor (Sequence ID, DNA Sequence, Group), add/delete rows, group filtering, column sorting, CSV export, and real-time validation.
   - Extracts all 6 mandatory features: `Count of 'A'`, `Count of 'T'`, `Count of 'G'`, `Count of 'C'`, `Sequence Length`, and `GC Content (%) = ((G + C) / Length) * 100`.
   - Interactive GC Content (%) bar chart and group-level feature centroid summary.

2. **Model Training & Configuration**
   - Choice of **Random Forest Classifier** or **Decision Tree Classifier** (Gini/Entropy, `max_depth`, `n_estimators`, `max_features`, `min_samples_split`, `random_state`).
   - Validation strategies: **Full-Fit**, **Leave-One-Out CV (LOOCV)**, and **Stratified Train/Test Split**.
   - Displays Accuracy, Macro-F1, Confusion Matrix, per-class Precision/Recall/F1, normalized Feature Importance bar chart, and learned split rules.

3. **Inference / Prediction Engine**
   - Pre-filled with target sequence `ATGCGCGATC` (`Test`).
   - Strict `A/T/G/C` nucleotide validator, visual nucleotide strip, 6-feature badge cards, prominent predicted group & probability breakdown, tree-vote/decision-path explanation, and OOD experiment presets.

4. **Academic Deliverable Mode (Analysis & Discussion)**
   - Live data-backed discussion covering feature utility, sanity checks, out-of-distribution behavior, real-world genomics applications, and limitations of AI/ML in genomics.
   - One-click **Export Report / Print PDF**, **Download Markdown (.md)**, and **Copy Markdown**.

## How to Run

### Option A: Open Directly
Double-click `index.html` in any modern browser (works offline with zero dependencies).

### Option B: Serve Locally via Python (Port 8501)
```bash
python server.py
```
Then open `http://127.0.0.1:8501/index.html` in your browser.
