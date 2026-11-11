"""Self-contained Streamlit entrypoint for the DNA Sequence Classification Dashboard."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import streamlit as st

st.set_page_config(
    page_title="DNA Sequence Classification Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Sample Datasets
# ---------------------------------------------------------------------------

VALID_BASES = {"A", "T", "G", "C"}

FEATURE_COLUMNS = [
    "Count of 'A'",
    "Count of 'T'",
    "Count of 'G'",
    "Count of 'C'",
    "Sequence Length",
    "GC Content (%)",
]

DEFAULT_DATASET = [
    {"Sequence ID": "S1", "DNA Sequence": "ATGCGATC", "Group": "Group A"},
    {"Sequence ID": "S2", "DNA Sequence": "ATGCGTAC", "Group": "Group A"},
    {"Sequence ID": "S3", "DNA Sequence": "GCGCGTAT", "Group": "Group B"},
    {"Sequence ID": "S4", "DNA Sequence": "GCGCGCGC", "Group": "Group B"},
]

DEFAULT_UNKNOWN = {
    "id": "Test",
    "sequence": "ATGCGCGATC",
}

OOD_PRESETS = [
    {
        "label": "Default Test",
        "id": "Test",
        "sequence": "ATGCGCGATC",
        "note": "10 bp, 60.0% GC - assignment target sequence",
    },
    {
        "label": "AT-Rich Promoter",
        "id": "OOD-AT",
        "sequence": "ATATATATAT",
        "note": "10 bp, 0.0% GC - extreme AT-rich sequence",
    },
    {
        "label": "CpG Island Repeat",
        "id": "OOD-CpG",
        "sequence": "GCGCGCGCGCGC",
        "note": "12 bp, 100.0% GC - pure GC dinucleotide repeat",
    },
    {
        "label": "Long 50% GC",
        "id": "OOD-Long50",
        "sequence": "ATGCATGCATGCATGCATGC",
        "note": "20 bp, 50.0% GC - tests raw count vs GC% conflict",
    },
    {
        "label": "Poly-A Homopolymer",
        "id": "OOD-PolyA",
        "sequence": "AAAAAAAAAAAAAAAAAAAA",
        "note": "20 bp, 0.0% GC - out-of-distribution homopolymer",
    },
]

GROUP_COLORS = ["#0d9488", "#2563eb", "#d97706", "#9333ea", "#e11d48"]


# ---------------------------------------------------------------------------
# Feature Extraction & Validation
# ---------------------------------------------------------------------------


def _is_missing(val: Any) -> bool:
    return val is None or (isinstance(val, float) and val != val)


def _clean_text(val: Any, fallback: str) -> str:
    if _is_missing(val):
        return fallback
    cleaned = str(val).strip()
    return cleaned if cleaned else fallback


def normalize_sequence(raw: Any) -> str:
    """Strip surrounding whitespace and uppercase a DNA sequence string."""
    if _is_missing(raw):
        return ""
    return str(raw).strip().upper()


def validate_sequence(raw: Any) -> dict[str, Any]:
    """Validate that a DNA sequence is non-empty and contains only A, T, G, C."""
    cleaned = normalize_sequence(raw)
    if not cleaned:
        return {
            "valid": False,
            "cleaned": "",
            "invalid_chars": [],
            "message": "Sequence cannot be empty. Enter bases using A, T, G, C.",
        }

    invalid: list[tuple[str, int]] = []
    for idx, ch in enumerate(cleaned, start=1):
        if ch not in VALID_BASES:
            invalid.append(("SPACE" if ch == " " else ch, idx))

    if invalid:
        unique_bad = list(dict.fromkeys(ch for ch, _ in invalid))[:5]
        preview = ", ".join(f"'{ch}' @ pos {pos}" for ch, pos in invalid[:4])
        return {
            "valid": False,
            "cleaned": cleaned,
            "invalid_chars": invalid,
            "message": (
                f"Invalid base(s) detected ({', '.join(unique_bad)}): {preview}. "
                "Only A, T, G, C are permitted."
            ),
        }

    return {
        "valid": True,
        "cleaned": cleaned,
        "invalid_chars": [],
        "message": f"Valid DNA sequence ({len(cleaned)} bp, bases A/T/G/C only).",
    }


def extract_features(raw_sequence: Any) -> dict[str, Any]:
    """Compute the 6 mandatory numerical features for a DNA sequence."""
    seq = normalize_sequence(raw_sequence)
    count_a = seq.count("A")
    count_t = seq.count("T")
    count_g = seq.count("G")
    count_c = seq.count("C")
    length = len(seq)
    gc_content = ((count_g + count_c) / length) * 100.0 if length > 0 else 0.0

    return {
        "sequence": seq,
        "count_a": count_a,
        "count_t": count_t,
        "count_g": count_g,
        "count_c": count_c,
        "length": length,
        "gc_content": gc_content,
        "vector": [count_a, count_t, count_g, count_c, length, gc_content],
    }


def extract_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a raw dataset DataFrame with validation flags and extracted 6 features."""
    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict(orient="records"), start=1):
        seq_id = _clean_text(row.get("Sequence ID"), f"S{idx}")
        group = _clean_text(row.get("Group"), "Unassigned")
        val = validate_sequence(row.get("DNA Sequence", ""))
        feats = extract_features(val["cleaned"])
        enriched.append(
            {
                "Sequence ID": seq_id,
                "Group": group,
                "DNA Sequence": val["cleaned"],
                "Count of 'A'": feats["count_a"],
                "Count of 'T'": feats["count_t"],
                "Count of 'G'": feats["count_g"],
                "Count of 'C'": feats["count_c"],
                "Sequence Length": feats["length"],
                "GC Content (%)": round(feats["gc_content"], 2),
                "Valid": val["valid"],
                "Validation Note": val["message"],
            }
        )
    return pd.DataFrame(enriched)


def compute_group_summaries(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean feature values per biological group for valid sequences."""
    valid_df = enriched_df[enriched_df["Valid"]].copy()
    if valid_df.empty:
        return valid_df

    return (
        valid_df.groupby("Group", as_index=False)
        .agg(
            N=("Sequence ID", "count"),
            **{col: (col, "mean") for col in FEATURE_COLUMNS},
        )
        .sort_values("Group")
    )


# ---------------------------------------------------------------------------
# Machine Learning Pipeline (scikit-learn)
# ---------------------------------------------------------------------------


def build_classifier(config: dict[str, Any]):
    """Instantiate a scikit-learn DecisionTreeClassifier or RandomForestClassifier."""
    criterion = "entropy" if config.get("criterion") == "entropy" else "gini"
    max_depth = None if config.get("unlimited_depth") else int(config.get("max_depth") or 5)
    min_samples_split = max(2, int(config.get("min_samples_split") or 2))
    random_state = int(config.get("random_state") if config.get("random_state") is not None else 42)

    if config.get("classifier") == "decision_tree":
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )

    max_features = config.get("max_features")
    if max_features == "all":
        max_features = None
    elif max_features not in ("sqrt", "log2"):
        max_features = "sqrt"

    return RandomForestClassifier(
        n_estimators=max(1, min(200, int(config.get("n_estimators") or 50))),
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=random_state,
    )


def describe_learned_rules(model, feature_names: list[str]) -> str:
    """Format Decision Tree rules or top Random Forest estimators as readable text."""
    if isinstance(model, DecisionTreeClassifier):
        if model.tree_.node_count == 1:
            return "Single root leaf (no discriminatory splits needed)."
        return export_text(model, feature_names=feature_names, decimals=2)

    if isinstance(model, RandomForestClassifier):
        blocks: list[str] = []
        total = len(model.estimators_)
        for idx, est in enumerate(model.estimators_[:3], start=1):
            if est.tree_.node_count == 1:
                rules = "  (Single leaf node from bootstrap sample)"
            else:
                rules = export_text(est, feature_names=feature_names, decimals=2)
            blocks.append(f"[Tree #{idx} of {total}]\n{rules}")
        return "\n".join(blocks)

    return "Rule inspection unavailable for this estimator."


def _compute_eval_bundle(
    classes: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_ids: list[str],
) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    correct = int(np.sum(y_true == y_pred))
    total = int(len(y_true))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class_df = pd.DataFrame(
        {
            "Class": classes,
            "Precision": [round(float(v) * 100, 1) for v in prec],
            "Recall": [round(float(v) * 100, 1) for v in rec],
            "F1 (%)": [round(float(v) * 100, 1) for v in f1],
            "Support": [int(v) for v in sup],
        }
    )

    eval_predictions_df = pd.DataFrame(
        {
            "Sequence ID": eval_ids,
            "Actual": list(y_true),
            "Predicted": list(y_pred),
            "Correct": [bool(a == p) for a, p in zip(y_true, y_pred)],
        }
    )

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_df": per_class_df,
        "eval_predictions_df": eval_predictions_df,
    }


def train_and_evaluate(enriched_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Train classifier and compute evaluation metrics under the chosen strategy."""
    valid_df = enriched_df[enriched_df["Valid"]].reset_index(drop=True)
    if len(valid_df) < 2:
        return {"error": "At least 2 valid sequences are required to train a classifier."}

    X = valid_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = valid_df["Group"].astype(str).to_numpy()
    ids = valid_df["Sequence ID"].astype(str).tolist()
    classes = sorted(set(y.tolist()))

    base_model = build_classifier(config)

    if len(classes) < 2:
        deployed_model = clone(base_model).fit(X, y)
        bundle = _compute_eval_bundle(classes, y, y, ids)
        imp_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": deployed_model.feature_importances_,
                "Importance (%)": deployed_model.feature_importances_ * 100.0,
            }
        ).sort_values("Importance", ascending=False)
        return {
            "error": None,
            "warning": "Only 1 group is present in valid sequences. Add samples from at least 2 groups for meaningful classification.",
            "classes": classes,
            "eval_mode": config.get("eval_mode", "full"),
            "eval_description": "Single-class dataset (trivial fit)",
            "deployed_model": deployed_model,
            "feature_importances_df": imp_df,
            "rules_text": describe_learned_rules(deployed_model, FEATURE_COLUMNS),
            "sample_count": len(valid_df),
            **bundle,
        }

    eval_mode = config.get("eval_mode", "full")
    seed = int(config.get("random_state") if config.get("random_state") is not None else 42)

    if eval_mode == "loocv":
        loo = LeaveOneOut()
        y_true_list: list[str] = []
        y_pred_list: list[str] = []
        eval_ids: list[str] = []
        for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
            fold_model = clone(base_model)
            if hasattr(fold_model, "random_state"):
                fold_model.set_params(random_state=seed + fold_idx)
            fold_model.fit(X[train_idx], y[train_idx])
            y_true_list.append(str(y[test_idx[0]]))
            y_pred_list.append(str(fold_model.predict(X[test_idx])[0]))
            eval_ids.append(ids[test_idx[0]])

        deployed_model = clone(base_model).fit(X, y)
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        eval_desc = (
            f"Leave-One-Out Cross-Validation ({len(valid_df)} folds; "
            f"deployed model fitted on all {len(valid_df)} samples)"
        )
    elif eval_mode == "split":
        class_counts = Counter(y.tolist())
        if min(class_counts.values()) < 2:
            return {
                "error": (
                    "Stratified Train/Test Split requires at least 2 sequences per group. "
                    "Add more sequences or use Full-Fit / Leave-One-Out CV."
                )
            }
        test_ratio = float(config.get("test_ratio") or 0.5)
        indices = np.arange(len(valid_df))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed, stratify=y
        )
        deployed_model = clone(base_model).fit(X[train_idx], y[train_idx])
        y_true_arr = y[test_idx]
        y_pred_arr = deployed_model.predict(X[test_idx])
        eval_ids = [ids[i] for i in test_idx]
        eval_desc = (
            f"Stratified Train/Test Split ({len(train_idx)} train / {len(test_idx)} test samples)"
        )
    else:
        deployed_model = clone(base_model).fit(X, y)
        y_true_arr = y
        y_pred_arr = deployed_model.predict(X)
        eval_ids = ids
        eval_desc = f"Full-Fit Resubstitution (trained & evaluated on all {len(valid_df)} samples)"

    bundle = _compute_eval_bundle(classes, y_true_arr, y_pred_arr, eval_ids)
    importances = deployed_model.feature_importances_
    imp_df = (
        pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": importances,
                "Importance (%)": importances * 100.0,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "error": None,
        "warning": None,
        "classes": classes,
        "eval_mode": eval_mode,
        "eval_description": eval_desc,
        "deployed_model": deployed_model,
        "feature_importances_df": imp_df,
        "rules_text": describe_learned_rules(deployed_model, FEATURE_COLUMNS),
        "sample_count": len(valid_df),
        **bundle,
    }


def explain_unknown_prediction(model, x_vec: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Build human-readable decision path or Random Forest tree vote breakdown."""
    if isinstance(model, DecisionTreeClassifier):
        tree = model.tree_
        node_indicator = model.decision_path(x_vec)
        node_indices = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]
        steps: list[str] = []
        for node_id in node_indices:
            if tree.children_left[node_id] == tree.children_right[node_id]:
                continue
            feat_idx = int(tree.feature[node_id])
            thr = float(tree.threshold[node_id])
            val = float(x_vec[0, feat_idx])
            direction = "<=" if val <= thr else ">"
            steps.append(f"{FEATURE_COLUMNS[feat_idx]} ({val:.2f}) {direction} {thr:.2f}")
        return {"mode": "decision_tree", "steps": steps}

    if isinstance(model, RandomForestClassifier):
        votes: Counter[str] = Counter({cls: 0 for cls in classes})
        root_splits: Counter[str] = Counter()
        for est in model.estimators_:
            pred_idx = int(est.predict(x_vec)[0])
            votes[str(model.classes_[pred_idx])] += 1
            if est.tree_.node_count > 1:
                root_feat_idx = int(est.tree_.feature[0])
                if 0 <= root_feat_idx < len(FEATURE_COLUMNS):
                    root_splits[FEATURE_COLUMNS[root_feat_idx]] += 1
        return {
            "mode": "random_forest",
            "votes": dict(votes),
            "total_trees": len(model.estimators_),
            "root_splits": dict(root_splits),
        }

    return {"mode": "generic"}


def predict_unknown_sequence(model, raw_seq: str, seq_id: str) -> dict[str, Any]:
    """Validate an unknown sequence, extract features, and run classifier inference."""
    val = validate_sequence(raw_seq)
    feats = extract_features(val["cleaned"])
    clean_id = str(seq_id or "").strip() or "Unknown"

    if not val["valid"] or model is None:
        return {
            "valid": val["valid"],
            "id": clean_id,
            "validation": val,
            "features": feats,
            "error": True,
        }

    x_vec = np.array([feats["vector"]], dtype=float)
    classes = [str(c) for c in model.classes_]
    probs_arr = model.predict_proba(x_vec)[0]
    probabilities = {cls: float(p) for cls, p in zip(classes, probs_arr)}
    predicted_class = str(model.predict(x_vec)[0])
    confidence = probabilities.get(predicted_class, float(np.max(probs_arr)))
    explanation = explain_unknown_prediction(model, x_vec, classes)

    return {
        "valid": True,
        "id": clean_id,
        "validation": val,
        "features": feats,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "explanation": explanation,
        "error": False,
    }


# ---------------------------------------------------------------------------
# Academic Discussion & Report Exports
# ---------------------------------------------------------------------------


def _find_feature_collisions(valid_df: pd.DataFrame) -> list[list[str]]:
    groups: dict[tuple, list[tuple[str, str]]] = {}
    for _, row in valid_df.iterrows():
        key = tuple(round(float(row[c]), 4) for c in FEATURE_COLUMNS)
        groups.setdefault(key, []).append((str(row["Sequence ID"]), str(row["DNA Sequence"])))

    collisions: list[list[str]] = []
    for items in groups.values():
        distinct_seqs = {seq for _, seq in items}
        if len(items) > 1 and len(distinct_seqs) > 1:
            collisions.append([f"{sid} ({seq})" for sid, seq in items])
    return collisions


def build_discussion_sections(
    enriched_df: pd.DataFrame,
    train_result: dict[str, Any] | None,
    prediction_result: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Synthesize the 5 academic discussion answers backed by live pipeline metrics."""
    valid_df = enriched_df[enriched_df["Valid"]]
    n_samples = len(valid_df)
    lengths = valid_df["Sequence Length"].tolist() if n_samples else [0]
    min_len = min(lengths)
    max_len = max(lengths)

    if train_result and not train_result.get("error"):
        imp_df = train_result["feature_importances_df"]
        active_feats = imp_df[imp_df["Importance"] > 0.001]
        zero_feats = imp_df[imp_df["Importance"] <= 0.001]["Feature"].tolist()
        if not active_feats.empty:
            top_str = ", ".join(
                f"{row['Feature']} ({row['Importance (%)']:.1f}%)"
                for _, row in active_feats.head(3).iterrows()
            )
        else:
            top_str = "No active discriminatory splits (all features tied at 0%)"
        acc_pct = f"{train_result['accuracy'] * 100:.1f}%"
        f1_pct = f"{train_result['macro_f1'] * 100:.1f}%"
        eval_desc = train_result["eval_description"]
    else:
        top_str = "N/A (model not trained)"
        zero_feats = []
        acc_pct = "N/A"
        f1_pct = "N/A"
        eval_desc = "N/A"

    depth_lbl = "None" if config.get("unlimited_depth") else str(config.get("max_depth", 5))
    if config.get("classifier") == "decision_tree":
        model_label = f"Decision Tree Classifier (criterion={config.get('criterion')}, max_depth={depth_lbl})"
    else:
        model_label = (
            f"Random Forest Classifier (n_estimators={config.get('n_estimators')}, "
            f"criterion={config.get('criterion')}, max_depth={depth_lbl})"
        )

    zero_note = (
        f"Features contributing minimal or zero split gain include {', '.join(zero_feats)}"
        + (
            f" (notably, Sequence Length is constant at {min_len} bp across all {n_samples} training sequences, yielding zero variance)."
            if min_len == max_len and "Sequence Length" in zero_feats
            else "."
        )
        if zero_feats
        else "All six extracted features contributed non-zero impurity reduction across tree splits."
    )

    sec1 = [
        f"In the active {model_label}, the top discriminatory features ranked by normalized impurity decrease are: {top_str}.",
        zero_note,
        (
            "Biologically and mathematically, GC Content (%) and Count of 'G' / 'C' (or complementary 'A' depletion) "
            "capture base-composition shifts between balanced sequences (e.g., Group A at 50.0% GC) and GC-enriched "
            "sequences (e.g., Group B at 62.5%-100.0% GC). Unlike raw base counts, GC Content (%) normalizes by sequence "
            "length and remains scale-invariant across sequences of differing lengths."
        ),
    ]

    collisions = _find_feature_collisions(valid_df)
    collision_note = (
        "Feature-space inspection reveals an exact collision between "
        + "; ".join(" and ".join(grp) for grp in collisions)
        + ": despite differing nucleotide orders, they map to identical 6-feature vectors."
        if collisions
        else "All valid training sequences currently map to distinct 6-feature vectors."
    )

    if prediction_result and not prediction_result.get("error"):
        f = prediction_result["features"]
        probs_str = ", ".join(
            f"{cls}: {p * 100:.1f}%"
            for cls, p in sorted(prediction_result["probabilities"].items(), key=lambda kv: -kv[1])
        )
        len_mismatch = f["length"] < min_len or f["length"] > max_len
        pred_sanity = (
            f"For target sequence \"{prediction_result['id']}\" ({f['sequence']}, {f['length']} bp, "
            f"GC={f['gc_content']:.1f}%), the model predicts {prediction_result['predicted_class']} ({probs_str}). "
            + (
                f"Because this sequence length ({f['length']} bp) differs from the training length range ({min_len}-{max_len} bp), "
                f"raw counts (A={f['count_a']}, G={f['count_g']}) can diverge from length-normalized GC Content ({f['gc_content']:.1f}%): "
                f"A={f['count_a']} equals an 8 bp Group A raw count while representing {f['count_a']/f['length']*100:.1f}% A-content, "
                f"whereas GC={f['gc_content']:.1f}% and G={f['count_g']} align with GC-rich Group B."
                if len_mismatch
                else f"Its length ({f['length']} bp) sits within the training length range ({min_len}-{max_len} bp), keeping raw base counts and GC% consistent."
            )
        )
    else:
        pred_sanity = "Execute a valid prediction in the Predict Unknown tab to audit inference behavior against training bounds."

    sec2 = [
        f"Under {eval_desc}, the classifier achieved {acc_pct} accuracy (Macro-F1: {f1_pct}) across {n_samples} valid sequences.",
        collision_note,
        (
            f"Sanity check on small-sample splits: With N={n_samples} training observations, multiple features "
            "(such as Count of 'A', Count of 'G', and GC Content %) can tie for identical impurity gain at the root split. "
            "A single Decision Tree breaks ties using seeded feature permutation, whereas Random Forest aggregates bootstrap trees "
            "over feature subsets—yielding calibrated ensemble probabilities that are less sensitive to a single tie choice."
        ),
        pred_sanity,
    ]

    sec3 = [
        (
            f"Tree-based classifiers partition the 6-feature space using axis-aligned thresholds learned strictly from training "
            f"sequences (lengths {min_len}-{max_len} bp). When evaluated on out-of-distribution (OOD) sequences, tree models cannot "
            "flag inputs as 'unknown' or 'out-of-domain'—every input is routed into a leaf belonging to an existing training class."
        ),
        (
            "Two OOD failure modes arise directly from unnormalized counts: (1) Length-scaling drift: a 20 bp balanced sequence "
            "(e.g., ATGCATGCATGCATGCATGC, 50% GC) has Count of 'G'=5 and Count of 'C'=5, exceeding Group B thresholds on raw G/C counts "
            "even though its 50% GC matches Group A. Conversely, a 2 bp fragment like 'GC' (100% GC) has G=1, C=1 and falls below Group B count thresholds."
        ),
        (
            "(2) Unseen compositional extremes: homopolymers such as Poly-A (20 bp, 0% GC) or Poly-T (8 bp, 0% GC) lie far outside training "
            "distributions yet still receive confident predictions because leaf probabilities reflect training node purity rather than distance from the training manifold."
        ),
    ]

    sec4 = [
        (
            "1. CpG Island Detection: Mammalian gene promoters frequently contain unmethylated CpG islands—stretches of 200-2000 bp "
            "with GC content >50-55% and elevated CpG dinucleotide density. Composition classifiers separate GC-rich regulatory islands "
            "(analogous to Group B) from AT-rich bulk genomic background."
        ),
        (
            "2. Promoter & Regulatory Element Identification: Core promoters (e.g., AT-rich TATA-box regions) versus housekeeping gene "
            "promoters (GC-rich Sp1 binding sites) exhibit measurable compositional signatures useful in first-pass genome annotation."
        ),
        (
            "3. Viral Strain Typing & Metagenomic Binning: Bacterial taxa, viral clades, and horizontally transferred pathogenicity islands "
            "carry characteristic genomic GC signatures. Rapid k-mer and GC-profile Random Forests help bin metagenomic contigs and flag anomalous inserts."
        ),
    ]

    sec5 = [
        (
            "1. Positional & Motif Context Loss (Bag-of-Bases Bottleneck): Counting A, T, G, C discards nucleotide order completely. "
            "Biological function depends on exact motifs (e.g., TATAAT boxes, EcoRI site GAATTC, GT/AG splice junctions) and dinucleotide order. "
            "For example, GCGCGCGC (CpG island repeat) and GGGGCCCC (homopolymer blocks) yield identical 6-feature vectors "
            "[A=0, T=0, G=4, C=4, Length=8, GC=100%], just as S1 (ATGCGATC) and S2 (ATGCGTAC) collide. Modern genomics addresses this with "
            "k-mer spectra, Position Weight Matrices (PWMs), 1D-CNNs, and genomic foundation models (e.g., DNABERT)."
        ),
        (
            "2. Unnormalized Feature Confounding: Mixing raw base counts with Sequence Length lets classifiers overfit dataset-specific "
            "read or amplicon lengths rather than underlying biological composition."
        ),
        (
            "3. Small-Sample Overfitting & Dataset Bias: Limited labeled datasets (like N=4), PCR GC-amplification bias, class imbalance, "
            "and evolutionary homology across samples can inflate naive evaluation accuracy unless validated with strict cross-validation or homology-aware splits."
        ),
    ]

    return [
        {"num": "01", "title": "Most Useful Features (Feature Importance Analysis)", "paragraphs": sec1},
        {"num": "02", "title": "Model Sanity Check & Correctness Analysis", "paragraphs": sec2},
        {"num": "03", "title": "Out-of-Distribution (OOD) Sequence Behavior", "paragraphs": sec3},
        {"num": "04", "title": "Real-World Genomics Applications", "paragraphs": sec4},
        {"num": "05", "title": "Fundamental Limitations of AI/ML in Genomics", "paragraphs": sec5},
    ]


def build_markdown_report(
    enriched_df: pd.DataFrame,
    train_result: dict[str, Any] | None,
    prediction_result: dict[str, Any] | None,
    config: dict[str, Any],
) -> str:
    """Generate a structured Markdown deliverable with tables, metrics, and discussion."""
    valid_df = enriched_df[enriched_df["Valid"]]
    discussion = build_discussion_sections(enriched_df, train_result, prediction_result, config)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Computational Biology Assignment Report: DNA Sequence Classification",
        "",
        f"**Generated:** {now_str}",
        f"**Classifier:** {'Decision Tree Classifier' if config.get('classifier') == 'decision_tree' else 'Random Forest Classifier'}",
        (
            f"**Hyperparameters:** `criterion={config.get('criterion')}`, "
            f"`max_depth={'None' if config.get('unlimited_depth') else config.get('max_depth')}`, "
            f"`min_samples_split={config.get('min_samples_split')}`"
            + (f", `n_estimators={config.get('n_estimators')}`" if config.get("classifier") != "decision_tree" else "")
            + f", `random_state={config.get('random_state')}`"
        ),
        f"**Evaluation Mode:** {train_result['eval_description'] if train_result and not train_result.get('error') else 'N/A'}",
        "",
        "## 1. Dataset & Extracted 6-Feature Table",
        "",
        "| Seq ID | Group | DNA Sequence | A | T | G | C | Length | GC Content (%) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in valid_df.iterrows():
        ca = int(r["Count of 'A'"])
        ct = int(r["Count of 'T'"])
        cg = int(r["Count of 'G'"])
        cc = int(r["Count of 'C'"])
        seq_len = int(r["Sequence Length"])
        gc_pct = float(r["GC Content (%)"])
        lines.append(
            f"| {r['Sequence ID']} | {r['Group']} | `{r['DNA Sequence']}` | "
            f"{ca} | {ct} | {cg} | {cc} | {seq_len} | {gc_pct:.2f}% |"
        )
    lines.append("")

    if train_result and not train_result.get("error"):
        classes = train_result["classes"]
        cm = train_result["confusion_matrix"]
        lines.extend(
            [
                "## 2. Model Evaluation Metrics",
                "",
                f"- **Accuracy:** {train_result['accuracy'] * 100:.2f}% ({train_result['correct']}/{train_result['total']})",
                f"- **Macro-F1 Score:** {train_result['macro_f1'] * 100:.2f}%",
                "",
                "### Confusion Matrix (Rows = Actual, Columns = Predicted)",
                "",
                "| Actual \\ Predicted | " + " | ".join(classes) + " |",
                "| :--- | " + " | ".join("---:" for _ in classes) + " |",
            ]
        )
        for r_idx, cls in enumerate(classes):
            counts = " | ".join(str(int(v)) for v in cm[r_idx])
            lines.append(f"| **{cls}** | {counts} |")
        lines.extend(
            [
                "",
                "### Feature Importances",
                "",
                "| Feature | Normalized Importance |",
                "| :--- | ---: |",
            ]
        )
        for _, row in train_result["feature_importances_df"].iterrows():
            lines.append(f"| {row['Feature']} | {row['Importance (%)']:.2f}% |")
        lines.append("")

    if prediction_result and not prediction_result.get("error"):
        f = prediction_result["features"]
        probs = " | ".join(
            f"{cls}: {p * 100:.1f}%"
            for cls, p in sorted(prediction_result["probabilities"].items(), key=lambda kv: -kv[1])
        )
        lines.extend(
            [
                "## 3. Unknown Sequence Prediction",
                "",
                f"- **Sequence ID:** {prediction_result['id']}",
                f"- **Input Sequence:** `{f['sequence']}`",
                (
                    f"- **Extracted Features:** A={f['count_a']}, T={f['count_t']}, G={f['count_g']}, "
                    f"C={f['count_c']}, Length={f['length']} bp, GC Content={f['gc_content']:.2f}%"
                ),
                f"- **Predicted Group:** **{prediction_result['predicted_class']}**",
                f"- **Class Probabilities:** {probs}",
                "",
            ]
        )

    lines.extend(["## 4. Academic Analysis & Critical Discussion", ""])
    for sec in discussion:
        lines.append(f"### {sec['num']}. {sec['title']}")
        lines.append("")
        for p in sec["paragraphs"]:
            lines.append(p)
            lines.append("")

    return "\n".join(lines)


def _ascii_safe(text: str) -> str:
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2192": "->",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00d7": "x",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_to_bytes(pdf) -> bytes:
    import fpdf

    major_ver = int(str(getattr(fpdf, "__version__", "2")).split(".")[0])
    out = pdf.output(dest="S") if major_ver < 2 else pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode("latin-1", errors="replace")


def build_pdf_report(
    enriched_df: pd.DataFrame,
    train_result: dict[str, Any] | None,
    prediction_result: dict[str, Any] | None,
    config: dict[str, Any],
) -> bytes:
    """Render a structured academic PDF summary using fpdf2 / fpdf."""
    from fpdf import FPDF

    valid_df = enriched_df[enriched_df["Valid"]]
    discussion = build_discussion_sections(enriched_df, train_result, prediction_result, config)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _ascii_safe("DNA Sequence Classification - Academic Report"), ln=1)
    pdf.set_font("Helvetica", "", 9.5)
    model_name = "Decision Tree Classifier" if config.get("classifier") == "decision_tree" else "Random Forest Classifier"
    pdf.cell(
        0,
        6,
        _ascii_safe(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Model: {model_name}"),
        ln=1,
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "1. Training Dataset & Extracted Features", ln=1)
    pdf.set_font("Helvetica", "B", 8.5)
    col_widths = [18, 22, 42, 12, 12, 12, 12, 16, 22]
    headers = ["ID", "Group", "Sequence", "A", "T", "G", "C", "Len", "GC %"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8.5)
    for _, r in valid_df.iterrows():
        vals = [
            str(r["Sequence ID"])[:10],
            str(r["Group"])[:12],
            str(r["DNA Sequence"])[:22],
            str(int(r["Count of 'A'"])),
            str(int(r["Count of 'T'"])),
            str(int(r["Count of 'G'"])),
            str(int(r["Count of 'C'"])),
            str(int(r["Sequence Length"])),
            f"{float(r['GC Content (%)']):.1f}%",
        ]
        for w, v in zip(col_widths, vals):
            pdf.cell(w, 6, _ascii_safe(v), border=1)
        pdf.ln()
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "2. Model Evaluation & Feature Importance", ln=1)
    pdf.set_font("Helvetica", "", 9.5)
    if train_result and not train_result.get("error"):
        pdf.multi_cell(
            0,
            5.5,
            _ascii_safe(
                f"Evaluation: {train_result['eval_description']}\n"
                f"Accuracy: {train_result['accuracy']*100:.1f}% ({train_result['correct']}/{train_result['total']}) | "
                f"Macro-F1: {train_result['macro_f1']*100:.1f}%"
            ),
        )
        pdf.ln(1)
        for _, row in train_result["feature_importances_df"].iterrows():
            pdf.cell(0, 5, _ascii_safe(f"  - {row['Feature']}: {row['Importance (%)']:.1f}%"), ln=1)
    else:
        pdf.cell(0, 6, "Model training unavailable.", ln=1)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. Unknown Sequence Prediction", ln=1)
    pdf.set_font("Helvetica", "", 9.5)
    if prediction_result and not prediction_result.get("error"):
        f = prediction_result["features"]
        probs = ", ".join(
            f"{cls}: {p*100:.1f}%"
            for cls, p in sorted(prediction_result["probabilities"].items(), key=lambda kv: -kv[1])
        )
        pdf.multi_cell(
            0,
            5.5,
            _ascii_safe(
                f"Target: {prediction_result['id']} ({f['sequence']})\n"
                f"Features: A={f['count_a']}, T={f['count_t']}, G={f['count_g']}, C={f['count_c']}, "
                f"Length={f['length']} bp, GC={f['gc_content']:.1f}%\n"
                f"Predicted Group: {prediction_result['predicted_class']} ({probs})"
            ),
        )
    else:
        pdf.cell(0, 6, "No valid inference result available.", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "4. Critical Discussion & Analysis", ln=1)
    for sec in discussion:
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 5.5, _ascii_safe(f"{sec['num']}. {sec['title']}"))
        pdf.set_font("Helvetica", "", 9.2)
        for p in sec["paragraphs"]:
            pdf.multi_cell(0, 5, _ascii_safe(p))
            pdf.ln(1)
        pdf.ln(1.5)

    return _pdf_to_bytes(pdf)


# ---------------------------------------------------------------------------
# Streamlit UI Rendering
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        .bio-callout {
            border: 1px solid #0d9488;
            background: linear-gradient(135deg, #ffffff 0%, rgba(13, 148, 136, 0.10) 100%);
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .bio-callout-eyebrow {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #0d9488;
        }
        .bio-callout-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.2rem 0;
        }
        .bio-callout-sub {
            font-size: 0.9rem;
            color: #475569;
        }
        .dna-ribbon {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 0.5rem 0 0.8rem;
        }
        .base-tile {
            width: 30px;
            padding: 4px 0;
            border-radius: 6px;
            text-align: center;
            font-family: monospace;
            font-weight: 700;
            font-size: 0.85rem;
            line-height: 1.15;
            display: inline-flex;
            flex-direction: column;
            align-items: center;
        }
        .base-tile small {
            font-size: 0.6rem;
            font-weight: 500;
            opacity: 0.78;
        }
        .base-A { background: #dcfce7; color: #15803d; }
        .base-T { background: #fef3c7; color: #b45309; }
        .base-G { background: #dbeafe; color: #1d4ed8; }
        .base-C { background: #f3e8ff; color: #7e22ce; }
        .base-X { background: #fee2e2; color: #dc2626; }
        .discussion-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.9rem;
        }
        .discussion-badge {
            display: inline-block;
            background: rgba(13, 148, 136, 0.14);
            color: #0f766e;
            font-family: monospace;
            font-weight: 800;
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 6px;
            margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "raw_dataset" not in st.session_state:
        st.session_state.raw_dataset = pd.DataFrame(DEFAULT_DATASET)
    if "editor_version" not in st.session_state:
        st.session_state.editor_version = 0
    if "unknown_id_input" not in st.session_state:
        st.session_state.unknown_id_input = DEFAULT_UNKNOWN["id"]
    if "unknown_seq_input" not in st.session_state:
        st.session_state.unknown_seq_input = DEFAULT_UNKNOWN["sequence"]


def _apply_preset_callback(seq_id: str, sequence: str) -> None:
    st.session_state.unknown_id_input = seq_id
    st.session_state.unknown_seq_input = sequence


def _reset_dataset_callback() -> None:
    st.session_state.raw_dataset = pd.DataFrame(DEFAULT_DATASET)
    st.session_state.editor_version += 1


def render_dna_ribbon(sequence: str) -> str:
    chars = list(sequence[:80])
    if not chars:
        return "<p style='color:#64748b;font-size:0.85rem;'>Enter a DNA sequence to inspect nucleotide positions.</p>"
    tiles = []
    for idx, ch in enumerate(chars, start=1):
        cls = f"base-{ch}" if ch in ("A", "T", "G", "C") else "base-X"
        tiles.append(f"<span class='base-tile {cls}'>{html.escape(ch)}<small>{idx}</small></span>")
    extra = (
        f"<span style='align-self:center;color:#64748b;font-size:0.8rem;margin-left:6px;'>+{len(sequence)-80} more bp</span>"
        if len(sequence) > 80
        else ""
    )
    return f"<div class='dna-ribbon'>{''.join(tiles)}{extra}</div>"


def render_sidebar(enriched_df: pd.DataFrame) -> dict[str, Any]:
    with st.sidebar:
        st.title("BioSeqML")
        st.caption("DNA Sequence Classification Studio")

        valid_df = enriched_df[enriched_df["Valid"]]
        n_groups = int(valid_df["Group"].nunique()) if not valid_df.empty else 0
        st.info(f"**{len(valid_df)}** valid sequence(s) · **{n_groups}** group(s)")

        st.subheader("1. Training Dataset")
        with st.form("quick_add_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            seq_id = c1.text_input("Seq ID", placeholder="S5", max_chars=20)
            group = c2.text_input("Group", placeholder="Group A", max_chars=24)
            dna_seq = st.text_input("DNA Sequence (A/T/G/C)", placeholder="ATGCGTAC")
            submitted = st.form_submit_button("+ Add to Dataset", use_container_width=True)
            if submitted:
                val = validate_sequence(dna_seq)
                if not val["valid"]:
                    st.error(val["message"])
                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "Sequence ID": (seq_id or f"S{len(st.session_state.raw_dataset) + 1}").strip(),
                                "DNA Sequence": val["cleaned"],
                                "Group": (group or "Group A").strip(),
                            }
                        ]
                    )
                    st.session_state.raw_dataset = pd.concat(
                        [st.session_state.raw_dataset, new_row], ignore_index=True
                    )
                    st.session_state.editor_version += 1
                    st.rerun()

        st.button("Reset Default Dataset", on_click=_reset_dataset_callback, use_container_width=True)

        st.divider()
        st.subheader("2. Model & Hyperparameters")
        algo_label = st.selectbox(
            "Classifier Algorithm",
            ["Random Forest Classifier", "Decision Tree Classifier"],
            index=0,
        )
        classifier = "random_forest" if algo_label.startswith("Random Forest") else "decision_tree"

        crit_label = st.selectbox(
            "Split Criterion",
            ["Gini Impurity", "Entropy (Info Gain)"],
            index=0,
        )
        criterion = "gini" if crit_label.startswith("Gini") else "entropy"

        unlimited_depth = st.checkbox("Unlimited max_depth (None)", value=False)
        max_depth = st.slider("max_depth", min_value=1, max_value=10, value=5, disabled=unlimited_depth)

        n_estimators = 50
        max_features = "sqrt"
        if classifier == "random_forest":
            n_estimators = st.slider("n_estimators (Trees)", min_value=5, max_value=150, value=50, step=5)
            mf_label = st.selectbox("max_features (Per Split)", ["sqrt (2 of 6 features)", "all (6 features)"], index=0)
            max_features = "sqrt" if mf_label.startswith("sqrt") else "all"

        c_ms, c_rs = st.columns(2)
        min_samples_split = c_ms.number_input("min_samples_split", min_value=2, max_value=10, value=2, step=1)
        random_state = c_rs.number_input("random_state", min_value=0, max_value=99999, value=42, step=1)

        st.divider()
        st.subheader("3. Training & Validation")
        eval_label = st.selectbox(
            "Validation Strategy",
            [
                "Full-Fit (All Samples)",
                "Leave-One-Out CV (LOOCV)",
                "Stratified Train / Test Split",
            ],
            index=0,
        )
        if eval_label.startswith("Leave-One-Out"):
            eval_mode = "loocv"
        elif eval_label.startswith("Stratified"):
            eval_mode = "split"
        else:
            eval_mode = "full"

        test_ratio = 0.5
        if eval_mode == "split":
            test_pct = st.slider("Test Split Size (%)", min_value=25, max_value=50, value=50, step=25)
            test_ratio = test_pct / 100.0

    return {
        "classifier": classifier,
        "criterion": criterion,
        "max_depth": max_depth,
        "unlimited_depth": unlimited_depth,
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_split": int(min_samples_split),
        "random_state": int(random_state),
        "eval_mode": eval_mode,
        "test_ratio": test_ratio,
    }


def render_kpi_row(enriched_df: pd.DataFrame, config: dict, train_result: dict, pred_result: dict) -> None:
    valid_df = enriched_df[enriched_df["Valid"]]
    n_seq = len(valid_df)
    n_groups = int(valid_df["Group"].nunique()) if n_seq else 0
    avg_len = float(valid_df["Sequence Length"].mean()) if n_seq else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dataset Sequences", f"{n_seq}", f"{n_groups} groups · {avg_len:.1f} bp avg")

    if config["classifier"] == "decision_tree":
        depth_str = "∞" if config["unlimited_depth"] else str(config["max_depth"])
        k2.metric("Active Classifier", "Decision Tree", f"{config['criterion']} · depth {depth_str}")
    else:
        k2.metric("Active Classifier", "Random Forest", f"{config['n_estimators']} trees · {config['criterion']}")

    if not train_result.get("error"):
        mode_tag = {"full": "Full-Fit", "loocv": "LOOCV", "split": "Test Split"}.get(config["eval_mode"], "Eval")
        k3.metric(
            "Model Accuracy",
            f"{train_result['accuracy'] * 100:.1f}%",
            f"{mode_tag} ({train_result['correct']}/{train_result['total']})",
        )
    else:
        k3.metric("Model Accuracy", "—", "Insufficient data")

    if pred_result and not pred_result.get("error"):
        k4.metric(
            "Unknown Prediction",
            pred_result["predicted_class"],
            f"{pred_result['id']} · {pred_result['confidence'] * 100:.1f}% confidence",
        )
    else:
        k4.metric("Unknown Prediction", "—", "Awaiting valid sequence")


def render_tab_features(enriched_df: pd.DataFrame) -> None:
    st.subheader("Training Dataset Editor & 6-Feature Extraction")
    st.caption("Edit Sequence ID, DNA Sequence (A/T/G/C), or Group inline, or add/delete rows below.")

    edited = st.data_editor(
        st.session_state.raw_dataset,
        num_rows="dynamic",
        use_container_width=True,
        key=f"dataset_editor_{st.session_state.editor_version}",
        column_config={
            "Sequence ID": st.column_config.TextColumn("Sequence ID", required=True),
            "DNA Sequence": st.column_config.TextColumn("DNA Sequence (A/T/G/C)", required=True),
            "Group": st.column_config.TextColumn("Group", required=True),
        },
    )

    if not edited.fillna("").astype(str).equals(st.session_state.raw_dataset.fillna("").astype(str)):
        st.session_state.raw_dataset = edited
        st.rerun()

    invalid_rows = enriched_df[~enriched_df["Valid"]]
    if not invalid_rows.empty:
        msgs = [f"**{r['Sequence ID']}**: {r['Validation Note']}" for _, r in invalid_rows.iterrows()]
        st.error("Validation alert in training dataset — " + " | ".join(msgs))

    st.markdown("#### Extracted Feature Table (Sortable)")
    valid_df = enriched_df[enriched_df["Valid"]].copy()
    groups = sorted(valid_df["Group"].unique().tolist()) if not valid_df.empty else []

    col_filter, col_dl = st.columns([3, 1])
    selected_group = col_filter.selectbox("Filter by Group", ["All Groups"] + groups, index=0)
    display_df = valid_df if selected_group == "All Groups" else valid_df[valid_df["Group"] == selected_group]

    table_cols = ["Sequence ID", "Group", "DNA Sequence", *FEATURE_COLUMNS]
    st.dataframe(
        display_df[table_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "GC Content (%)": st.column_config.ProgressColumn(
                "GC Content (%)", min_value=0.0, max_value=100.0, format="%.1f%%"
            )
        },
    )

    csv_bytes = display_df[table_cols].to_csv(index=False).encode("utf-8")
    col_dl.download_button(
        "Export Features CSV",
        data=csv_bytes,
        file_name="dna_features_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

    c_chart, c_cent = st.columns(2)
    with c_chart:
        st.markdown("#### GC Content (%) Across Sequences")
        if valid_df.empty:
            st.info("Add valid sequences to render the GC Content chart.")
        else:
            color_map = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(groups)}
            fig_gc = px.bar(
                valid_df,
                x="Sequence ID",
                y="GC Content (%)",
                color="Group",
                color_discrete_map=color_map,
                text="GC Content (%)",
                hover_data=["DNA Sequence", "Sequence Length", "Count of 'A'", "Count of 'T'", "Count of 'G'", "Count of 'C'"],
            )
            fig_gc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_gc.add_hline(y=50.0, line_dash="dash", line_color="#94a3b8", annotation_text="50% GC Ref")
            fig_gc.update_layout(
                yaxis_range=[0, 114],
                margin=dict(t=24, b=24, l=24, r=16),
                height=340,
                legend_title_text="Group",
            )
            st.plotly_chart(fig_gc, use_container_width=True)

    with c_cent:
        st.markdown("#### Group-Level Feature Centroids")
        summary_df = compute_group_summaries(enriched_df)
        if summary_df.empty:
            st.info("No valid sequences to summarize.")
        else:
            formatted = summary_df.copy()
            for col in FEATURE_COLUMNS:
                formatted[col] = formatted[col].round(2)
            st.dataframe(formatted, use_container_width=True, hide_index=True)


def render_tab_training(train_result: dict, config: dict) -> None:
    if train_result.get("error"):
        st.error(train_result["error"])
        return

    model_title = (
        f"Decision Tree ({config['criterion']}, max_depth={'None' if config['unlimited_depth'] else config['max_depth']})"
        if config["classifier"] == "decision_tree"
        else f"Random Forest ({config['n_estimators']} trees, {config['criterion']}, max_depth={'None' if config['unlimited_depth'] else config['max_depth']})"
    )
    st.info(f"**Active Pipeline:** {model_title} | **Validation:** {train_result['eval_description']}")

    if train_result.get("warning"):
        st.warning(train_result["warning"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy Score", f"{train_result['accuracy'] * 100:.1f}%", f"{train_result['correct']} / {train_result['total']} matched")
    m2.metric("Macro F1-Score", f"{train_result['macro_f1'] * 100:.1f}%", "Balanced class average")

    imp_df = train_result["feature_importances_df"]
    if not imp_df.empty and imp_df.iloc[0]["Importance"] > 0:
        m3.metric(
            "Top Discriminating Feature",
            str(imp_df.iloc[0]["Feature"]),
            f"{imp_df.iloc[0]['Importance (%)']:.1f}% impurity gain",
        )
    else:
        m3.metric("Top Discriminating Feature", "None", "No splits required")

    c_cm, c_imp = st.columns(2)
    with c_cm:
        st.markdown("#### Confusion Matrix")
        classes = train_result["classes"]
        cm = train_result["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Teal",
            labels={"x": "Predicted Group", "y": "Actual Group", "color": "Count"},
        )
        fig_cm.update_layout(margin=dict(t=24, b=24, l=24, r=16), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("**Per-Class Metrics**")
        st.dataframe(train_result["per_class_df"], use_container_width=True, hide_index=True)

        with st.expander("Sample-Level Evaluation Predictions"):
            st.dataframe(train_result["eval_predictions_df"], use_container_width=True, hide_index=True)

    with c_imp:
        st.markdown("#### Feature Importance (Impurity Decrease)")
        plot_df = imp_df.sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            plot_df,
            x="Importance (%)",
            y="Feature",
            orientation="h",
            text="Importance (%)",
            color="Importance (%)",
            color_continuous_scale=["#94a3b8", "#2563eb", "#0d9488"],
        )
        fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_imp.update_layout(
            xaxis_range=[0, max(15.0, float(plot_df["Importance (%)"].max()) * 1.22)],
            margin=dict(t=24, b=24, l=16, r=24),
            height=340,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("Learned Decision Rules Inspector", expanded=True):
        st.code(train_result["rules_text"], language="text")


def render_tab_predict(enriched_df: pd.DataFrame, train_result: dict, config: dict) -> dict:
    st.subheader("Unknown DNA Sequence Inference")

    st.markdown("**Experiment & Out-of-Distribution Presets**")
    preset_cols = st.columns(len(OOD_PRESETS))
    for col, preset in zip(preset_cols, OOD_PRESETS):
        col.button(
            preset["label"],
            help=preset["note"],
            on_click=_apply_preset_callback,
            args=(preset["id"], preset["sequence"]),
            use_container_width=True,
            key=f"preset_{preset['id']}",
        )

    c_in, c_out = st.columns([1.25, 1.0])

    with c_in:
        seq_id = st.text_input("Sequence ID", key="unknown_id_input", max_chars=24)
        raw_seq = st.text_input("DNA Sequence (A, T, G, C)", key="unknown_seq_input")

        val = validate_sequence(raw_seq)
        if val["valid"]:
            st.success(val["message"])
        else:
            st.error(val["message"])

        can_predict = val["valid"] and not train_result.get("error")
        st.button("Predict Group", type="primary", disabled=not can_predict, use_container_width=True)
        st.button(
            "Reset to Test (ATGCGCGATC)",
            on_click=_apply_preset_callback,
            args=(DEFAULT_UNKNOWN["id"], DEFAULT_UNKNOWN["sequence"]),
            use_container_width=True,
        )

        st.markdown("**Nucleotide Composition Strip**")
        st.markdown(render_dna_ribbon(val["cleaned"]), unsafe_allow_html=True)

    model = train_result.get("deployed_model") if not train_result.get("error") else None
    pred_result = predict_unknown_sequence(model, raw_seq, seq_id)

    with c_out:
        if pred_result.get("error"):
            st.warning("Enter a valid A/T/G/C sequence and ensure the classifier is trained to view predictions.")
        else:
            feats = pred_result["features"]
            model_tag = "Decision Tree" if config["classifier"] == "decision_tree" else f"Random Forest ({config['n_estimators']} trees)"
            st.markdown(
                f"""
                <div class="bio-callout">
                    <div class="bio-callout-eyebrow">Inference Output · {html.escape(model_tag)}</div>
                    <div class="bio-callout-title">Predicted: {html.escape(pred_result['predicted_class'])}</div>
                    <div class="bio-callout-sub">
                        Confidence: <strong>{pred_result['confidence'] * 100:.1f}%</strong> ·
                        Sequence "{html.escape(pred_result['id'])}" ({feats['length']} bp, GC {feats['gc_content']:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Class Probabilities**")
            for cls, prob in sorted(pred_result["probabilities"].items(), key=lambda kv: -kv[1]):
                st.progress(min(1.0, max(0.0, float(prob))), text=f"{cls}: {prob * 100:.1f}%")

            exp = pred_result["explanation"]
            if exp.get("mode") == "decision_tree":
                path_str = " -> ".join(exp["steps"]) if exp.get("steps") else "Single root leaf"
                st.info(f"**Decision Path:** `{path_str}` -> **{pred_result['predicted_class']}**")
            elif exp.get("mode") == "random_forest":
                votes_str = " · ".join(
                    f"{cls}: {cnt}/{exp['total_trees']} trees"
                    for cls, cnt in sorted(exp["votes"].items(), key=lambda kv: -kv[1])
                )
                top_roots = ", ".join(
                    f"{feat} ({cnt})"
                    for feat, cnt in sorted(exp["root_splits"].items(), key=lambda kv: -kv[1])[:3]
                )
                st.info(f"**Ensemble Votes:** {votes_str}  \n**Top Root Splits:** {top_roots or 'Constant leaves'}")

    st.markdown("#### Extracted Features for Input Sequence")
    feats = extract_features(val["cleaned"])
    length = feats["length"]
    f_cols = st.columns(6)
    f_cards = [
        ("Count of 'A'", feats["count_a"], f"{(feats['count_a']/length*100):.1f}%" if length else "0.0%"),
        ("Count of 'T'", feats["count_t"], f"{(feats['count_t']/length*100):.1f}%" if length else "0.0%"),
        ("Count of 'G'", feats["count_g"], f"{(feats['count_g']/length*100):.1f}%" if length else "0.0%"),
        ("Count of 'C'", feats["count_c"], f"{(feats['count_c']/length*100):.1f}%" if length else "0.0%"),
        ("Sequence Length", f"{length} bp", "Total bases"),
        ("GC Content (%)", f"{feats['gc_content']:.1f}%", f"G+C = {feats['count_g'] + feats['count_c']} bp"),
    ]
    for col, (lbl, val_str, delta_str) in zip(f_cols, f_cards):
        col.metric(lbl, val_str, delta_str)

    summary_df = compute_group_summaries(enriched_df)
    if val["valid"] and not summary_df.empty and length > 0:
        st.markdown("#### Length-Normalized Composition Comparison (%)")
        st.caption("Compares relative nucleotide percentages against group centroids to reveal raw-count vs length scaling differences.")
        comp_metrics = ["A %", "T %", "G %", "C %", "GC %"]
        fig_comp = go.Figure()
        fig_comp.add_bar(
            name=f"Unknown ({pred_result['id']})",
            x=comp_metrics,
            y=[
                feats["count_a"] / length * 100,
                feats["count_t"] / length * 100,
                feats["count_g"] / length * 100,
                feats["count_c"] / length * 100,
                feats["gc_content"],
            ],
            marker_color="#0f172a",
        )
        for i, (_, row) in enumerate(summary_df.iterrows()):
            m_len = row["Sequence Length"] or 1.0
            fig_comp.add_bar(
                name=f"{row['Group']} (Mean)",
                x=comp_metrics,
                y=[
                    row["Count of 'A'"] / m_len * 100,
                    row["Count of 'T'"] / m_len * 100,
                    row["Count of 'G'"] / m_len * 100,
                    row["Count of 'C'"] / m_len * 100,
                    row["GC Content (%)"],
                ],
                marker_color=GROUP_COLORS[i % len(GROUP_COLORS)],
            )
        fig_comp.update_layout(barmode="group", yaxis_range=[0, 108], height=320, margin=dict(t=24, b=24, l=24, r=16))
        st.plotly_chart(fig_comp, use_container_width=True)

    return pred_result


def render_tab_discussion(enriched_df: pd.DataFrame, train_result: dict, pred_result: dict, config: dict) -> None:
    st.subheader("Academic Deliverable Mode — Report & Critical Discussion")
    st.caption("Live synthesis dynamically backed by your dataset, model importances, confusion matrix, and inference diagnostics.")

    md_text = build_markdown_report(enriched_df, train_result, pred_result, config)
    pdf_bytes = build_pdf_report(enriched_df, train_result, pred_result, config)

    dl1, dl2, _ = st.columns([1.1, 1.1, 2])
    dl1.download_button(
        "Export Report (PDF)",
        data=pdf_bytes,
        file_name="dna_classification_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    dl2.download_button(
        "Download Markdown (.md)",
        data=md_text.encode("utf-8"),
        file_name="dna_classification_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    sections = build_discussion_sections(enriched_df, train_result, pred_result, config)
    for sec in sections:
        paragraphs_html = "".join(
            f"<p style='color:#334155;line-height:1.6;'>{html.escape(p)}</p>" for p in sec["paragraphs"]
        )
        st.markdown(
            f"""
            <div class="discussion-card">
                <h4 style="margin-top:0;margin-bottom:0.65rem;">
                    <span class="discussion-badge">{html.escape(sec['num'])}</span>
                    {html.escape(sec['title'])}
                </h4>
                {paragraphs_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Preview Generated Markdown Deliverable"):
        st.markdown(md_text)


def main() -> None:
    inject_custom_css()
    init_state()

    enriched_df = extract_dataframe_features(st.session_state.raw_dataset)
    config = render_sidebar(enriched_df)
    train_result = train_and_evaluate(enriched_df, config)

    model = train_result.get("deployed_model") if not train_result.get("error") else None
    current_pred = predict_unknown_sequence(
        model, st.session_state.unknown_seq_input, st.session_state.unknown_id_input
    )

    st.title("DNA Sequence Classification Dashboard")
    st.caption("Computational Biology Assignment · 6-Feature Nucleotide Composition & Tree-Based Machine Learning")
    render_kpi_row(enriched_df, config, train_result, current_pred)

    tab_feat, tab_train, tab_pred, tab_disc = st.tabs(
        ["Feature Extraction", "Model Training", "Predict Unknown", "Analysis & Discussion"]
    )

    with tab_feat:
        render_tab_features(enriched_df)
    with tab_train:
        render_tab_training(train_result, config)
    with tab_pred:
        pred_result = render_tab_predict(enriched_df, train_result, config)
    with tab_disc:
        render_tab_discussion(enriched_df, train_result, pred_result, config)


if __name__ == "__main__":
    main()
