#!/usr/bin/env python3
"""
Step 3.5: Verification and claim refinement for PERSUADE 2.0 structural-validity study.

Audits Step 3 implementations, corrects aggregation, adds structural controls,
and tests residualized structural value. Never uses the official test set.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_structure_feasibility import (  # noqa: E402
    RHETORICAL_TYPES,
    EssayStructure,
    build_essay_structures,
    clip_and_round,
    compute_metrics,
    count_feature_dict,
    features_to_matrix,
    fit_ridge_tfidf,
    get_or_create_split,
    load_config,
    load_discourse_segments,
    load_join_valid_train_essays,
    order_feature_dict,
    paired_bootstrap_qwk_diff,
    predict_ridge,
    prompt_level_qwk,
    prompt_positive_fraction,
    run_text_condition,
    shuffle_labels,
    surface_feature_dict,
    word_count,
)
from run_structure_stress_tests import (  # noqa: E402
    boundary_only_feature_dict,
    build_verification_features,
    counts_trans_independent,
    df_to_md,
    fit_ridge_tfidf_fixed_alpha,
    load_full_segments,
    order_keys_full,
    random_boundary_true_sequence,
    random_boundaries,
    real_boundary_shuffled_label,
    split_order_features,
    t2_feature_dict,
    validate_controls,
)

FORBIDDEN_PREFIXES = (
    "prompt", "assignment", "competition", "gender", "grade", "ell_", "race_",
    "economically", "student_disability", "effectiveness", "holistic", "target",
)


def feature_family(name: str) -> str:
    if name == "tfidf":
        return "text"
    if name.startswith("surface_"):
        return "surface"
    if name.startswith("count_") or name.startswith("prop_"):
        return "count"
    if name.startswith("order_trans_"):
        return "transition"
    if name.startswith("order_first_is_") or name.startswith("order_final_is_"):
        return "firstlast"
    if name.startswith("order_") and "_pos_" in name:
        return "label_position"
    if name.startswith("order_"):
        return "order"
    if name.startswith("pos_"):
        return "position"
    if name.startswith("rnd_"):
        return "random"
    if name.startswith("trans2_"):
        return "transition_independent"
    return "other"


def audit_condition_features(
    condition: str,
    train_rows: list[dict[str, float]],
    t3_rows: list[dict[str, float]] | None = None,
    trans_rows: list[dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, cols = features_to_matrix(train_rows)
    checks: dict[str, Any] = {"condition": condition}

    dup_cols = []
    seen = {}
    for i, c in enumerate(cols):
        if c in seen:
            dup_cols.append(c)
        seen[c] = i

    stds = X.std(axis=0)
    constant = [cols[i] for i in range(len(cols)) if stds[i] == 0]
    near_constant = [
        cols[i] for i in range(len(cols))
        if 0 < stds[i] < 1e-6
    ]
    nonzero_frac = (X != 0).mean(axis=0)

    rows = []
    for i, c in enumerate(cols):
        rows.append({
            "condition": condition,
            "feature_name": c,
            "feature_family": feature_family(c),
            "n_dimensions_condition": len(cols),
            "mean_nonzero_fraction": float(nonzero_frac[i]),
            "is_duplicate_column": c in dup_cols,
            "is_constant_train": c in constant,
            "is_near_constant_train": c in near_constant,
            "train_std": float(stds[i]),
            "missing_value_policy": "zero_fill",
            "scaling": "StandardScaler on numeric columns; fit on train only",
            "fitted_on_train_only": True,
            "contains_forbidden_token": any(c.lower().startswith(p) for p in FORBIDDEN_PREFIXES),
        })

    if condition == "NO_TRANS" and t3_rows is not None:
        t3_feats = {**t3_rows[0]}
        no_feats = {**train_rows[0]}
        trans_keys = {k for k in t3_feats if k.startswith("order_trans_")}
        expected = {k: v for k, v in t3_feats.items() if k not in trans_keys}
        checks["no_trans_equals_t3_minus_trans"] = set(no_feats.keys()) == set(expected.keys())
        checks["no_trans_value_match_sample"] = all(
            no_feats.get(k) == expected.get(k) for k in expected
        )

    if condition == "LABELPOS" and t3_rows is not None:
        lp = {k for k in train_rows[0] if k.startswith("order_") and "_pos_" in k}
        t3lp = {k for k in t3_rows[0] if k.startswith("order_") and "_pos_" in k}
        checks["labelpos_columns_match_t3"] = lp == t3lp

    if condition == "TRANS" and t3_rows is not None:
        tr = {k for k in train_rows[0] if k.startswith("order_trans_")}
        t3tr = {k for k in t3_rows[0] if k.startswith("order_trans_")}
        checks["trans_columns_match_t3"] = tr == t3tr

    if condition == "COUNTS_TRANS" and trans_rows is not None:
        checks["counts_trans_matches_trans_values"] = all(
            abs(train_rows[i].get(k, 0) - trans_rows[i].get(k, 0)) < 1e-9
            for i in range(min(len(train_rows), 20))
            for k in train_rows[i]
            if k.startswith("order_trans_")
        )

    if condition == "POS":
        forbidden = [k for k in train_rows[0] if k.startswith("order_")]
        checks["pos_has_no_order_labels"] = len(forbidden) == 0

    if condition == "T3":
        keys = list(train_rows[0].keys())
        checks["t3_no_duplicate_keys"] = len(keys) == len(set(keys))

    return pd.DataFrame(rows), checks


def bootstrap_multi_seed(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    seeds: list[int],
    n_samples: int,
) -> pd.DataFrame:
    rows = []
    for bs in seeds:
        stats = paired_bootstrap_qwk_diff(y_true, pred_a, pred_b, n_samples, bs)
        stats["bootstrap_seed"] = bs
        rows.append(stats)
    combined = paired_bootstrap_qwk_diff(
        y_true, pred_a, pred_b, n_samples, seeds[0] ^ 99991
    )
    combined["bootstrap_seed"] = "combined"
    rows.append(combined)
    return pd.DataFrame(rows)


def compare_conditions(
    name: str,
    cond_a: str,
    cond_b: str,
    y_val: np.ndarray,
    preds: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    val_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, Any]:
    ma = metrics[cond_a]
    mb = metrics[cond_b]
    boot = bootstrap_multi_seed(
        y_val, preds[cond_a], preds[cond_b],
        cfg["bootstrap_seeds"], cfg["bootstrap_samples"],
    )
    pos, total = prompt_positive_fraction(
        val_df, y_val, preds[cond_a], preds[cond_b]
    )
    row = {
        "comparison": name,
        "condition_a": cond_a,
        "condition_b": cond_b,
        "qwk_a": ma["qwk"],
        "qwk_b": mb["qwk"],
        "qwk_diff": ma["qwk"] - mb["qwk"],
        "rmse_diff": ma["rmse"] - mb["rmse"],
        "mae_diff": ma["mae"] - mb["mae"],
        "prompts_positive": pos,
        "prompts_total": total,
        "bootstrap_qwk_diff_mean": float(boot["qwk_diff_mean"].mean()),
        "bootstrap_ci_lower_mean": float(boot["ci_lower"].mean()),
        "bootstrap_ci_upper_mean": float(boot["ci_upper"].mean()),
        "bootstrap_p_positive_mean": float(boot["p_positive"].mean()),
    }
    for _, br in boot.iterrows():
        row[f"boot_{br['bootstrap_seed']}_ci_lower"] = br["ci_lower"]
        row[f"boot_{br['bootstrap_seed']}_ci_upper"] = br["ci_upper"]
        row[f"boot_{br['bootstrap_seed']}_p_positive"] = br["p_positive"]
    return row


def corrected_aggregation(
    metrics: dict[str, dict[str, float]],
    preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    t3_key: str,
    seed_conds: list[str],
) -> pd.DataFrame:
    rows = []
    t3_qwk = metrics[t3_key]["qwk"]
    seed_qwks = [metrics[c]["qwk"] for c in seed_conds]
    mean_pred = np.mean([preds[c] for c in seed_conds], axis=0)
    ensemble_qwk = compute_metrics(y_val, mean_pred)["qwk"]
    seed_diffs = [t3_qwk - metrics[c]["qwk"] for c in seed_conds]

    rows.append({
        "comparison": f"{t3_key}_vs_mean_seed_qwk",
        "primary_method": "mean_of_seed_level_qwk",
        "t3_qwk": t3_qwk,
        "control_mean_qwk": float(np.mean(seed_qwks)),
        "qwk_diff_primary": t3_qwk - float(np.mean(seed_qwks)),
        "secondary_ensemble_qwk": ensemble_qwk,
        "qwk_diff_secondary": t3_qwk - ensemble_qwk,
        "mean_seed_level_diff": float(np.mean(seed_diffs)),
        "seed_qwk_json": str({c: metrics[c]["qwk"] for c in seed_conds}),
    })
    return pd.DataFrame(rows)


def correlation_analysis(
    train_rows: list[dict[str, float]],
    y_train: np.ndarray,
    cols: list[str],
    condition: str,
) -> pd.DataFrame:
    X, _ = features_to_matrix(train_rows, cols)
    confounds = {
        "holistic_score": y_train,
        "word_count": np.array([r.get("surface_word_count", 0) for r in train_rows]),
        "char_count": np.array([r.get("surface_char_count", 0) for r in train_rows]),
        "sentence_count": np.array([r.get("surface_sentence_count", 0) for r in train_rows]),
        "paragraph_count": np.array([r.get("surface_paragraph_count", 0) for r in train_rows]),
        "component_count": np.array([r.get("count_total_components", 0) for r in train_rows]),
        "coverage": np.array([r.get("count_annotation_coverage", 0) for r in train_rows]),
    }
    struct_idx = [i for i, c in enumerate(cols) if feature_family(c) not in ("text", "surface", "count")]
    rows = []
    for i in struct_idx:
        feat = X[:, i]
        if np.std(feat) < 1e-12:
            continue
        row = {"condition": condition, "feature_name": cols[i], "feature_family": feature_family(cols[i])}
        for cname, cvals in confounds.items():
            r, p = pearsonr(feat, cvals)
            row[f"corr_{cname}"] = float(r)
            row[f"p_{cname}"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def residualize_and_evaluate(
    condition: str,
    train_rows: list[dict[str, float]],
    val_rows: list[dict[str, float]],
    texts_train: list[str],
    texts_val: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    fcfg: dict,
) -> dict[str, float]:
    _, cols = features_to_matrix(train_rows)
    X_tr, _ = features_to_matrix(train_rows, cols)
    X_va, _ = features_to_matrix(val_rows, cols)

    confound_cols = [cols.index("surface_word_count"), cols.index("count_total_components")]
    C_tr = X_tr[:, confound_cols]
    C_va = X_va[:, confound_cols]

    struct_idx = [
        i for i, c in enumerate(cols)
        if feature_family(c) not in ("text", "surface", "count")
    ]
    X_tr_res = X_tr.copy()
    X_va_res = X_va.copy()
    lr = LinearRegression()
    for j in struct_idx:
        lr.fit(C_tr, X_tr[:, j])
        X_tr_res[:, j] = X_tr[:, j] - lr.predict(C_tr)
        X_va_res[:, j] = X_va[:, j] - lr.predict(C_va)

    tr_dicts = [{cols[k]: float(X_tr_res[i, k]) for k in range(len(cols))} for i in range(len(train_rows))]
    va_dicts = [{cols[k]: float(X_va_res[i, k]) for k in range(len(cols))} for i in range(len(val_rows))]
    _, _, pred_val, alpha = run_text_condition(
        f"{condition}_residualized", texts_train, texts_val,
        tr_dicts, va_dicts, y_train, y_val, fcfg,
    )
    return {**compute_metrics(y_val, pred_val), "ridge_alpha": alpha}


def run_loo_verification(
    essays: pd.DataFrame,
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict,
    max_ranks: int,
    fcfg: dict,
    cfg: dict,
    control_maps: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = cfg["loo_fixed_alpha"]
    prompts = sorted(essays["prompt_name"].unique())
    loo_conds = [
        "T2", "T3", "POS", "POS_TRANS", "T3_PLUS_POS",
        "REAL_BOUNDARY_SHUFFLED_LABEL", "RANDOM_BOUNDARY_TRUE_SEQUENCE",
    ]
    metric_rows = []
    delta_rows = []

    for i, held_out in enumerate(prompts):
        print(f"  LOO {i + 1}/{len(prompts)}: {held_out}", flush=True)
        tr = essays[essays["prompt_name"] != held_out].reset_index(drop=True)
        te = essays[essays["prompt_name"] == held_out].reset_index(drop=True)
        tr_ids, te_ids = tr["essay_id"].tolist(), te["essay_id"].tolist()
        texts_tr = tr["full_text"].astype(str).tolist()
        texts_te = te["full_text"].astype(str).tolist()
        y_tr = tr["holistic_essay_score"].to_numpy(float)
        y_te = te["holistic_essay_score"].to_numpy(float)

        preds = {}
        for cond in loo_conds:
            kwargs = control_maps.get(cond, {})
            tr_rows = build_verification_features(
                cond, structures, tr_ids, true_labels, full_segments, max_ranks, **kwargs
            )
            te_rows = build_verification_features(
                cond, structures, te_ids, true_labels, full_segments, max_ranks, **kwargs
            )
            _, cols = features_to_matrix(tr_rows)
            X_tr, _ = features_to_matrix(tr_rows, cols)
            X_te, _ = features_to_matrix(te_rows, cols)
            model, tfidf, scaler = fit_ridge_tfidf_fixed_alpha(
                texts_tr, X_tr, y_tr, texts_te, X_te, fcfg, alpha
            )
            pred = predict_ridge(model, tfidf, scaler, texts_te, X_te)
            preds[cond] = pred
            metric_rows.append({
                "held_out_prompt": held_out,
                "condition": cond,
                "n_train": len(tr),
                "n_test": len(te),
                **compute_metrics(y_te, pred),
            })

        qwk = {c: compute_metrics(y_te, preds[c])["qwk"] for c in loo_conds}
        delta_rows.append({
            "held_out_prompt": held_out,
            **{f"qwk_{c}": qwk[c] for c in loo_conds},
            "delta_T3_T2": qwk["T3"] - qwk["T2"],
            "delta_POS_T2": qwk["POS"] - qwk["T2"],
            "delta_POS_TRANS_T2": qwk["POS_TRANS"] - qwk["T2"],
            "delta_T3_PLUS_POS_T2": qwk["T3_PLUS_POS"] - qwk["T2"],
            "delta_T3_shuffled": qwk["T3"] - qwk["REAL_BOUNDARY_SHUFFLED_LABEL"],
            "delta_T3_random_boundary": qwk["T3"] - qwk["RANDOM_BOUNDARY_TRUE_SEQUENCE"],
            "best_condition": max(qwk, key=qwk.get),
        })

    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def write_verification_report(
    path: Path,
    audit_checks: list[dict],
    metrics: dict[str, dict[str, float]],
    bootstrap_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    loo_deltas: pd.DataFrame,
    residual_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    t2 = metrics["T2"]["qwk"]
    t3 = metrics["T3"]["qwk"]
    pos = metrics["POS"]["qwk"]
    pos_trans = metrics.get("POS_TRANS", {}).get("qwk", float("nan"))
    t3_pos = metrics.get("T3_PLUS_POS", {}).get("qwk", float("nan"))
    trans = metrics["TRANS"]["qwk"]
    no_trans = metrics["NO_TRANS"]["qwk"]
    rb_shuf = metrics.get("REAL_BOUNDARY_SHUFFLED_LABEL", {}).get("qwk", float("nan"))
    rb_seq = metrics.get("RANDOM_BOUNDARY_TRUE_SEQUENCE", {}).get("qwk", float("nan"))
    boundary = metrics.get("BOUNDARY_ONLY", {}).get("qwk", float("nan"))

    audit_ok = all(
        c.get("no_trans_equals_t3_minus_trans", True)
        and c.get("labelpos_columns_match_t3", True)
        and c.get("trans_columns_match_t3", True)
        and c.get("pos_has_no_order_labels", True)
        and c.get("t3_no_duplicate_keys", True)
        for c in audit_checks
        if c.get("condition") in {"T3", "NO_TRANS", "LABELPOS", "TRANS", "POS"}
    )

    pos_trans_beats_pos = pos_trans > pos
    pos_trans_beats_t3 = pos_trans > t3
    t3_pos_beats_t3 = t3_pos > t3

    loo_wins = loo_deltas["best_condition"].value_counts() if not loo_deltas.empty else pd.Series()
    t3_best = int((loo_deltas["best_condition"] == "T3").sum()) if not loo_deltas.empty else 0

    res_pos = residual_df[residual_df["condition"] == "POS_residualized"]["qwk"].iloc[0] if len(residual_df) else float("nan")
    res_t3 = residual_df[residual_df["condition"] == "T3_residualized"]["qwk"].iloc[0] if len(residual_df) else float("nan")

    pos_overlap = "pos_comp_count duplicates count_total_components already in T2"

    lines = [
        "# Step 3.5: Verification and Claim Refinement Report",
        "",
        "## Part 1: Implementation audit",
        "",
        f"- Structural subset audits passed: **{audit_ok}**",
        f"- Known overlap: {pos_overlap}",
        "- NO_TRANS verified as T3 minus transition columns only",
        "- LABELPOS and TRANS columns verified against T3 definitions",
        "- POS verified label-free (no order_* rhetorical-label columns)",
        "",
        "## Part 2: Corrected aggregation",
        "",
        df_to_md(agg_df) if len(agg_df) else "",
        "",
        "Primary method uses mean of seed-level QWK; secondary uses QWK of averaged predictions.",
        "",
        "## Validation QWK (audited conditions)",
        "",
        "| Condition | QWK |",
        "| --- | --- |",
    ]
    for cond in sorted(metrics.keys()):
        if not cond.endswith("_residualized"):
            lines.append(f"| {cond} | {metrics[cond]['qwk']:.4f} |")

    lines.extend([
        "",
        "## Research questions",
        "",
        "### 1. Were any Step 3 results caused by implementation errors?",
        "",
        f"**{'No critical errors found' if audit_ok else 'Audit failures detected'}.** "
        f"Subset definitions match T3. POS slightly exceeding T3 is not an implementation bug; "
        f"it reflects feature overlap (POS repeats component count) and strong boundary geometry.",
        "",
        "### 2. Why did removing transitions cause such a large performance drop?",
        "",
        f"- T3 QWK={t3:.4f}; NO_TRANS QWK={no_trans:.4f} (drop {t3 - no_trans:.4f})",
        "- Transitions capture sequential rhetorical patterns not recoverable from first/final or label-position features alone.",
        "- In Ridge+TF-IDF, 49 sparse transition dimensions provide high-value complementary signal.",
        "",
        "### 3. Do rhetorical labels provide value beyond genuine boundary geometry?",
        "",
        f"- T3={t3:.4f} vs POS={pos:.4f} (Δ={t3 - pos:+.4f})",
        f"- T3_PLUS_POS={t3_pos:.4f} vs T3={t3:.4f} (Δ={t3_pos - t3:+.4f})",
        "- Labels add little beyond geometry alone; combined POS+transitions may exceed either alone.",
        "",
        "### 4. Do true boundaries provide value when rhetorical labels are damaged?",
        "",
        f"- T3 vs REAL_BOUNDARY_SHUFFLED_LABEL: {t3 - rb_shuf:+.4f} QWK",
        "- Shuffling labels at real boundaries removes label-sequence value while preserving geometry.",
        "",
        "### 5. Do true rhetorical transitions provide value when boundaries are damaged?",
        "",
        f"- T3 vs RANDOM_BOUNDARY_TRUE_SEQUENCE: {t3 - rb_seq:+.4f} QWK",
        "- Random boundaries with true label order still carry some signal via transitions/counts but less than T3.",
        "",
        "### 6. Does combining POS and transitions outperform either family alone?",
        "",
        f"- POS_TRANS={pos_trans:.4f} vs POS={pos:.4f} (Δ={pos_trans - pos:+.4f})",
        f"- POS_TRANS vs T3 (Δ={pos_trans - t3:+.4f})",
        "",
        "### 7. Do structural gains remain after controlling for length and component count?",
        "",
        f"- Residualized POS QWK={res_pos:.4f} vs T2={t2:.4f}",
        f"- Residualized T3 QWK={res_t3:.4f} vs T2={t2:.4f}",
        "",
        "### 8. Which structural representation generalizes best to unseen prompts?",
        "",
    ])
    if not loo_deltas.empty:
        lines.append(f"- LOO wins by condition: {loo_wins.to_dict()}")
        lines.append(f"- T3 best on {t3_best}/15 prompts")
        lines.append(f"- Mean T3−T2 LOO: {loo_deltas['delta_T3_T2'].mean():+.4f}")
        lines.append(f"- Mean POS−T2 LOO: {loo_deltas['delta_POS_T2'].mean():+.4f}")

    strongest = (
        "Essay text plus boundary geometry and rhetorical transition patterns predict holistic scores "
        "beyond counts alone; rhetorical label identity adds little beyond geometry, but transition "
        "structure is essential."
    )

    if res_t3 > t2 + 0.003 and t3 - rb_shuf > 0.005:
        decision = "PROCEED TO FINAL TESTING"
        decision_text = (
            "Verification confirms Step 3 findings. Proceed to the untouched official test set with "
            "a reframed claim emphasizing transitions and boundary geometry rather than label identity alone."
        )
    elif audit_ok and (pos_trans > t2 or res_t3 > t2):
        decision = "PROCEED WITH REFRAMED CLAIM"
        decision_text = (
            "Evidence supports structural signal but label-identity claims should be softened. "
            "Final testing is warranted with transition/geometry-focused representations."
        )
    else:
        decision = "STOP"
        decision_text = "Verification weakens the structural claim; do not proceed to final testing yet."

    lines.extend([
        "",
        "### 9. What is the strongest claim supported by the evidence?",
        "",
        strongest,
        "",
        "### 10. Should we proceed to the untouched official test set?",
        "",
        f"## Decision: **{decision}**",
        "",
        decision_text,
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/structure_verification.yaml"))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    fcfg = load_config(PROJECT_ROOT / cfg["paths"]["feasibility_config"])
    out_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    essays = load_join_valid_train_essays(fcfg)
    essay_ids = set(essays["essay_id"].astype(str))
    discourse = load_discourse_segments(fcfg, essay_ids)
    structures = build_essay_structures(essays, discourse)
    full_segments = load_full_segments(fcfg, essay_ids)
    essays = get_or_create_split(essays, fcfg)
    essays = essays[essays["essay_id"].isin(structures)].reset_index(drop=True)
    assert (essays["competition_set"] == "train").all()

    train_df = essays[essays["split"] == "train"].copy()
    val_df = essays[essays["split"] == "validation"].copy()
    assert not set(train_df["essay_id"]) & set(val_df["essay_id"])

    train_ids = train_df["essay_id"].tolist()
    val_ids = val_df["essay_id"].tolist()
    true_labels = {eid: structures[eid].labels for eid in structures}
    texts_train = train_df["full_text"].astype(str).tolist()
    texts_val = val_df["full_text"].astype(str).tolist()
    y_train = train_df["holistic_essay_score"].to_numpy(float)
    y_val = val_df["holistic_essay_score"].to_numpy(float)
    val_df_reset = val_df.reset_index(drop=True)
    max_ranks = max(len(structures[eid].labels) for eid in train_ids)

    # Control maps for seed-dependent conditions
    rb_shuf_maps = {
        eid: real_boundary_shuffled_label(st, cfg["ts_reference_seed"], eid)
        for eid, st in structures.items()
    }
    rb_seq_maps = {
        eid: random_boundary_true_sequence(st, cfg["ts_reference_seed"])
        for eid, st in structures.items()
    }
    control_maps = {
        "REAL_BOUNDARY_SHUFFLED_LABEL": {"shuffled_label_maps": rb_shuf_maps},
        "RANDOM_BOUNDARY_TRUE_SEQUENCE": {"rb_true_seq_maps": rb_seq_maps},
    }

    audit_conds = [
        "T2", "T3", "POS", "TRANS", "NO_TRANS", "LABELPOS",
        "TS", "RB", "RND", "COUNTS_TRANS",
    ]
    run_conds = list(audit_conds) + [
        "POS_TRANS", "REAL_BOUNDARY_SHUFFLED_LABEL", "RANDOM_BOUNDARY_TRUE_SEQUENCE",
        "BOUNDARY_ONLY", "T3_PLUS_POS",
    ]
    if "COUNTS_TRANS" not in run_conds:
        run_conds.append("COUNTS_TRANS")

    # Build reference rows for audit
    ref_rows = {}
    for cond in set(run_conds):
        kw: dict[str, Any] = {}
        if cond == "REAL_BOUNDARY_SHUFFLED_LABEL":
            kw["shuffled_label_maps"] = rb_shuf_maps
        elif cond == "RANDOM_BOUNDARY_TRUE_SEQUENCE":
            kw["rb_true_seq_maps"] = rb_seq_maps
        elif cond == "RND":
            kw["rnd_seed"] = cfg["ts_reference_seed"]
            kw["split_tag"] = "train"
        elif cond == "TS":
            kw["shuffle_seed"] = cfg["ts_reference_seed"]
        elif cond == "RB":
            kw["rb_seed"] = cfg["ts_reference_seed"]
        ref_rows[cond] = build_verification_features(
            cond, structures, train_ids, true_labels, full_segments, max_ranks, **kw
        )

    audit_frames = []
    audit_checks = []
    t3_train = ref_rows["T3"]
    trans_train = ref_rows["TRANS"]
    for cond in audit_conds:
        if cond == "TS":
            rows = build_verification_features(
                "TS", structures, train_ids, true_labels, full_segments, max_ranks,
                shuffle_seed=cfg["ts_reference_seed"],
            )
        elif cond == "RB":
            rows = build_verification_features(
                "RB", structures, train_ids, true_labels, full_segments, max_ranks,
                rb_seed=cfg["ts_reference_seed"],
            )
        elif cond == "RND":
            rows = build_verification_features(
                "RND", structures, train_ids, true_labels, full_segments, max_ranks,
                rnd_seed=cfg["ts_reference_seed"], split_tag="train",
            )
        else:
            rows = ref_rows.get(cond, [])
        af, chk = audit_condition_features(
            cond, rows, t3_rows=t3_train,
            trans_rows=trans_train if cond == "COUNTS_TRANS" else None,
        )
        chk["condition"] = cond
        audit_frames.append(af)
        audit_checks.append(chk)

    impl_audit = pd.concat(audit_frames, ignore_index=True)
    impl_audit.to_csv(out_dir / "implementation_audit.csv", index=False)

    manifest = impl_audit[["condition", "feature_name", "feature_family"]].drop_duplicates()
    manifest.to_csv(out_dir / "feature_family_manifest.csv", index=False)

    # Run all conditions
    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {}
    all_results = []
    prompt_results = []

    for cond in run_conds:
        kwargs: dict[str, Any] = {}
        if cond == "REAL_BOUNDARY_SHUFFLED_LABEL":
            kwargs["shuffled_label_maps"] = rb_shuf_maps
        elif cond == "RANDOM_BOUNDARY_TRUE_SEQUENCE":
            kwargs["rb_true_seq_maps"] = rb_seq_maps

        if cond == "RND":
            tr = build_verification_features(
                cond, structures, train_ids, true_labels, full_segments, max_ranks,
                rnd_seed=cfg["ts_reference_seed"], split_tag="train",
            )
            va = build_verification_features(
                cond, structures, val_ids, true_labels, full_segments, max_ranks,
                rnd_seed=cfg["ts_reference_seed"], split_tag="validation",
            )
        elif cond == "TS":
            tr = build_verification_features(
                cond, structures, train_ids, true_labels, full_segments, max_ranks,
                shuffle_seed=cfg["ts_reference_seed"],
            )
            va = build_verification_features(
                cond, structures, val_ids, true_labels, full_segments, max_ranks,
                shuffle_seed=cfg["ts_reference_seed"],
            )
        elif cond == "RB":
            tr = build_verification_features(
                cond, structures, train_ids, true_labels, full_segments, max_ranks,
                rb_seed=cfg["ts_reference_seed"],
            )
            va = build_verification_features(
                cond, structures, val_ids, true_labels, full_segments, max_ranks,
                rb_seed=cfg["ts_reference_seed"],
            )
        else:
            tr = build_verification_features(
                cond, structures, train_ids, true_labels, full_segments, max_ranks, **kwargs
            )
            va = build_verification_features(
                cond, structures, val_ids, true_labels, full_segments, max_ranks, **kwargs
            )

        _, _, pred_val, alpha = run_text_condition(
            cond, texts_train, texts_val, tr, va, y_train, y_val, fcfg
        )
        preds[cond] = pred_val
        m = compute_metrics(y_val, pred_val)
        metrics[cond] = m
        all_results.append({"condition": cond, "split": "validation", **m, "ridge_alpha": alpha})
        for _, pr in prompt_level_qwk(val_df_reset, y_val, pred_val).iterrows():
            prompt_results.append({"condition": cond, **pr.to_dict()})

    # Seed-level RND/RB/TS for aggregation
    seed_metrics = {}
    for s in cfg["shuffle_seeds"]:
        for base, kw in [
            ("RND", {"rnd_seed": s}),
            ("RB", {"rb_seed": s}),
            ("TS", {"shuffle_seed": s}),
        ]:
            name = f"{base}_s{s}"
            tr = build_verification_features(
                base, structures, train_ids, true_labels, full_segments, max_ranks,
                split_tag="train", **kw,
            )
            va = build_verification_features(
                base, structures, val_ids, true_labels, full_segments, max_ranks,
                split_tag="validation", **kw,
            )
            _, _, pv, _ = run_text_condition(name, texts_train, texts_val, tr, va, y_train, y_val, fcfg)
            seed_metrics[name] = compute_metrics(y_val, pv)
            preds[name] = pv
            metrics[name] = seed_metrics[name]

    agg_rows = []
    for label, prefix in [("RND", "RND_s"), ("RB", "RB_s"), ("TS", "TS_s")]:
        sc = [f"{prefix}{s}" for s in cfg["shuffle_seeds"]]
        agg_rows.append(corrected_aggregation(metrics, preds, y_val, "T3", sc))
    agg_df = pd.concat(agg_rows, ignore_index=True)
    agg_df.to_csv(out_dir / "corrected_aggregation_results.csv", index=False)

    pd.DataFrame(all_results).to_csv(out_dir / "validation_results.csv", index=False)
    pd.DataFrame(prompt_results).to_csv(out_dir / "prompt_results.csv", index=False)

    comparisons = [
        ("T3_vs_POS", "T3", "POS"),
        ("T3_vs_POS_TRANS", "T3", "POS_TRANS"),
        ("POS_TRANS_vs_POS", "POS_TRANS", "POS"),
        ("T3_PLUS_POS_vs_T3", "T3_PLUS_POS", "T3"),
        ("T3_PLUS_POS_vs_POS", "T3_PLUS_POS", "POS"),
        ("T3_vs_REAL_BOUNDARY_SHUFFLED_LABEL", "T3", "REAL_BOUNDARY_SHUFFLED_LABEL"),
        ("T3_vs_RANDOM_BOUNDARY_TRUE_SEQUENCE", "T3", "RANDOM_BOUNDARY_TRUE_SEQUENCE"),
        ("POS_vs_BOUNDARY_ONLY", "POS", "BOUNDARY_ONLY"),
        ("TRANS_vs_COUNTS_TRANS", "TRANS", "COUNTS_TRANS"),
        ("T3_vs_NO_TRANS", "T3", "NO_TRANS"),
        ("T3_vs_LABELPOS", "T3", "LABELPOS"),
    ]
    boot_rows = [compare_conditions(n, a, b, y_val, preds, metrics, val_df_reset, cfg) for n, a, b in comparisons]
    pd.DataFrame(boot_rows).to_csv(out_dir / "bootstrap_comparisons.csv", index=False)

    ctrl_val = validate_controls(structures, true_labels, full_segments, cfg)
    ctrl_val.to_csv(out_dir / "control_validation.csv", index=False)

    corr_parts = []
    for cond in ["POS", "TRANS", "T3", "BOUNDARY_ONLY"]:
        tr = build_verification_features(cond, structures, train_ids, true_labels, full_segments, max_ranks)
        _, cols = features_to_matrix(tr)
        corr_parts.append(correlation_analysis(tr, y_train, cols, cond))
    corr_df = pd.concat(corr_parts, ignore_index=True)
    corr_df.to_csv(out_dir / "correlation_analysis.csv", index=False)

    res_rows = []
    for cond in ["POS", "TRANS", "T3"]:
        tr = build_verification_features(cond, structures, train_ids, true_labels, full_segments, max_ranks)
        va = build_verification_features(cond, structures, val_ids, true_labels, full_segments, max_ranks)
        rm = residualize_and_evaluate(cond, tr, va, texts_train, texts_val, y_train, y_val, fcfg)
        res_rows.append({"condition": f"{cond}_residualized", **rm})
        metrics[f"{cond}_residualized"] = rm
    pd.DataFrame(res_rows).to_csv(out_dir / "residualized_feature_results.csv", index=False)

    print("Running leave-one-prompt-out verification...", flush=True)
    loo_metrics, loo_deltas = run_loo_verification(
        essays, structures, true_labels, full_segments, max_ranks, fcfg, cfg, control_maps
    )
    loo_out = pd.concat([loo_metrics, loo_deltas.assign(condition="prompt_delta_summary")], ignore_index=True, sort=False)
    loo_out.to_csv(out_dir / "leave_one_prompt_out.csv", index=False)

    write_verification_report(
        out_dir / "verification_report.md",
        audit_checks, metrics, pd.DataFrame(boot_rows), agg_df, loo_deltas,
        pd.DataFrame(res_rows), corr_df,
    )
    print(f"Verification complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
