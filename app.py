"""Streamlit entrypoint for the DNA Sequence Classification Dashboard."""
from __future__ import annotations

import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bioseq.features import (
    DEFAULT_DATASET,
    DEFAULT_UNKNOWN,
    FEATURE_COLUMNS,
    OOD_PRESETS,
    compute_group_summaries,
    extract_dataframe_features,
    extract_features,
    validate_sequence,
)
from bioseq.modeling import predict_unknown_sequence, train_and_evaluate
from bioseq.report import (
    build_discussion_sections,
    build_markdown_report,
    build_pdf_report,
)

st.set_page_config(
    page_title="DNA Sequence Classification Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROUP_COLORS = ["#0d9488", "#2563eb", "#d97706", "#9333ea", "#e11d48"]


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


def render_sidebar(enriched_df: pd.DataFrame) -> dict:
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

    # 6 Extracted Feature Badge Cards
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

    # Length-Normalized Composition Comparison
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
        paragraphs_html = "".join(f"<p style='color:#334155;line-height:1.6;'>{html.escape(p)}</p>" for p in sec["paragraphs"])
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
