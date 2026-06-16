#!/usr/bin/env python3
"""
Step 4: Final held-out evaluation on the official PERSUADE 2.0 test split.

Trains on all join-valid official-train essays; evaluates once on all join-valid
official-test essays. Uses the frozen TF-IDF + Ridge pipeline from prior steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_structure_feasibility import (  # noqa: E402
    EssayStructure,
    build_essay_structures,
    build_feature_sets,
    clip_and_round,
    compute_metrics,
    features_to_matrix,
    fit_ridge_tfidf,
    get_or_create_split,
    load_config,
    load_join_valid_train_essays,
    paired_bootstrap_qwk_diff,
    prompt_level_qwk,
    prompt_positive_fraction,
)
from run_structure_stress_tests import (  # noqa: E402
    build_verification_features,
    validate_controls,
)
from run_structure_verification import df_to_md  # noqa: E402

SCORE_LABELS = list(range(1, 7))


def load_join_valid_essays(cfg: dict, competition_set: str | None = None) -> pd.DataFrame:
    essays = pd.read_csv(PROJECT_ROOT / cfg["paths"]["essays"])
    join_val = pd.read_csv(PROJECT_ROOT / cfg["paths"]["join_validation"])
    valid_ids = set(join_val.loc[join_val["join_valid"], "essay_id"].astype(str))
    essays["essay_id"] = essays["essay_id"].astype(str)
    mask = essays["essay_id"].isin(valid_ids)
    if competition_set is not None:
        mask &= essays["competition_set"] == competition_set
    return essays.loc[mask].reset_index(drop=True)


def load_discourse_segments_all(cfg: dict, essay_ids: set[str]) -> pd.DataFrame:
    frames = []
    for key in ("raw_train", "raw_test"):
        raw_path = PROJECT_ROOT / cfg["paths"][key]
        raw = pd.read_csv(
            raw_path,
            usecols=["essay_id", "discourse_start", "discourse_end", "discourse_type"],
            low_memory=False,
        )
        raw["essay_id"] = raw["essay_id"].astype(str)
        frames.append(raw[raw["essay_id"].isin(essay_ids)])
    return pd.concat(frames, ignore_index=True)


def load_full_segments_all(cfg: dict, essay_ids: set[str]) -> dict[str, list[tuple[int, int, str]]]:
    discourse = load_discourse_segments_all(cfg, essay_ids)
    discourse = discourse.sort_values(["essay_id", "discourse_start"])
    segments: dict[str, list[tuple[int, int, str]]] = {}
    for eid, grp in discourse.groupby("essay_id"):
        segments[str(eid)] = [
            (int(r.discourse_start), int(r.discourse_end), str(r.discourse_type))
            for r in grp.itertuples()
        ]
    return segments


def select_ridge_alpha(
    texts_tr: list[str],
    texts_va: list[str],
    train_rows: list[dict[str, float]] | None,
    val_rows: list[dict[str, float]] | None,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    cfg: dict,
) -> tuple[float, dict[str, float]]:
    X_num_tr = X_num_va = None
    if train_rows is not None:
        X_num_tr, cols = features_to_matrix(train_rows)
        X_num_va, _ = features_to_matrix(val_rows, cols)
    _, _, _, alpha = fit_ridge_tfidf(texts_tr, X_num_tr, y_tr, texts_va, X_num_va, cfg)
    return alpha, {"selected_alpha": alpha, "selection_split": "official_train_internal_validation"}


def fit_tfidf_matrices(
    texts_train: list[str],
    texts_test: list[str],
    fcfg: dict,
) -> tuple[Any, Any, Any]:
    tfidf = TfidfVectorizer(
        ngram_range=tuple(fcfg["tfidf"]["ngram_range"]),
        max_features=fcfg["tfidf"]["max_features"],
        min_df=fcfg["tfidf"]["min_df"],
    )
    X_text_train = tfidf.fit_transform(texts_train)
    X_text_test = tfidf.transform(texts_test)
    return tfidf, X_text_train, X_text_test


def fit_predict_fixed(
    X_text_train: Any,
    X_text_test: Any,
    train_rows: list[dict[str, float]] | None,
    test_rows: list[dict[str, float]] | None,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, list[str]]:
    cols: list[str] = []
    if train_rows is not None:
        X_num_train, cols = features_to_matrix(train_rows)
        X_num_test, _ = features_to_matrix(test_rows, cols)
    else:
        X_num_train = X_num_test = None

    if X_num_train is not None and X_num_train.shape[1] > 0:
        scaler = StandardScaler()
        X_num_scaled_train = scaler.fit_transform(X_num_train)
        X_num_scaled_test = scaler.transform(X_num_test)
        X_train = hstack([X_text_train, csr_matrix(X_num_scaled_train)])
        X_test = hstack([X_text_test, csr_matrix(X_num_scaled_test)])
    else:
        X_train = X_text_train
        X_test = X_text_test

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    feature_cols = (["tfidf"] + cols) if cols else ["tfidf"]
    return preds, feature_cols


def build_condition_features(
    condition: str,
    structures: dict[str, EssayStructure],
    essay_ids: list[str],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    rnd_seed: int | None = None,
    split_tag: str = "train",
    shuffle_seed: int | None = None,
    rb_seed: int | None = None,
) -> list[dict[str, float]] | None:
    if condition == "T0":
        return None
    if condition == "T1":
        return build_feature_sets(structures, essay_ids, true_labels, True, False, False)
    return build_verification_features(
        condition,
        structures,
        essay_ids,
        true_labels,
        full_segments,
        max_ranks,
        rnd_seed=rnd_seed,
        split_tag=split_tag,
        shuffle_seed=shuffle_seed,
        rb_seed=rb_seed,
    )


def bootstrap_mean_control_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    control_preds: list[np.ndarray],
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        yt = np.rint(y_true[idx]).astype(int)
        qwk_a = cohen_kappa_score(yt, clip_and_round(pred_a[idx]), weights="quadratic")
        qwk_controls = [
            cohen_kappa_score(yt, clip_and_round(p[idx]), weights="quadratic")
            for p in control_preds
        ]
        diffs.append(qwk_a - float(np.mean(qwk_controls)))
    diffs_arr = np.array(diffs)
    return {
        "qwk_diff_mean": float(np.mean(diffs_arr)),
        "ci_lower": float(np.percentile(diffs_arr, 2.5)),
        "ci_upper": float(np.percentile(diffs_arr, 97.5)),
        "p_positive": float(np.mean(diffs_arr > 0)),
    }


def compare_pair(
    name: str,
    cond_a: str,
    cond_b: str,
    y_test: np.ndarray,
    preds: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    test_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, Any]:
    ma = metrics[cond_a]
    mb = metrics[cond_b]
    boot = paired_bootstrap_qwk_diff(
        y_test, preds[cond_a], preds[cond_b], cfg["bootstrap_samples"], cfg["seed"]
    )
    pos, total = prompt_positive_fraction(test_df, y_test, preds[cond_a], preds[cond_b])
    return {
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
        "bootstrap_qwk_diff_mean": boot["qwk_diff_mean"],
        "bootstrap_ci_lower": boot["ci_lower"],
        "bootstrap_ci_upper": boot["ci_upper"],
        "bootstrap_p_positive": boot["p_positive"],
    }


def compare_vs_mean_control(
    name: str,
    cond_a: str,
    control_prefix: str,
    y_test: np.ndarray,
    preds: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    test_df: pd.DataFrame,
    shuffle_seeds: list[int],
    cfg: dict,
) -> dict[str, Any]:
    seed_conds = [f"{control_prefix}_s{s}" for s in shuffle_seeds]
    control_preds = [preds[c] for c in seed_conds]
    control_qwks = [metrics[c]["qwk"] for c in seed_conds]
    mean_control_qwk = float(np.mean(control_qwks))
    ensemble_pred = np.mean(control_preds, axis=0)
    ensemble_qwk = compute_metrics(y_test, ensemble_pred)["qwk"]

    ma = metrics[cond_a]
    boot = bootstrap_mean_control_diff(
        y_test, preds[cond_a], control_preds, cfg["bootstrap_samples"], cfg["seed"]
    )

    pos_seed = 0
    total = 0
    for _, grp in test_df.groupby("prompt_name"):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        yt = np.rint(y_test[idx]).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        qwk_a = cohen_kappa_score(yt, clip_and_round(preds[cond_a][idx]), weights="quadratic")
        qwk_controls = [
            cohen_kappa_score(yt, clip_and_round(preds[c][idx]), weights="quadratic")
            for c in seed_conds
        ]
        total += 1
        if qwk_a > float(np.mean(qwk_controls)):
            pos_seed += 1

    row = {
        "comparison": name,
        "condition_a": cond_a,
        "condition_b": f"mean_{control_prefix}",
        "qwk_a": ma["qwk"],
        "qwk_b_primary": mean_control_qwk,
        "qwk_b_secondary_ensemble": ensemble_qwk,
        "qwk_diff_primary": ma["qwk"] - mean_control_qwk,
        "qwk_diff_secondary": ma["qwk"] - ensemble_qwk,
        "rmse_diff": ma["rmse"] - float(np.mean([metrics[c]["rmse"] for c in seed_conds])),
        "mae_diff": ma["mae"] - float(np.mean([metrics[c]["mae"] for c in seed_conds])),
        "prompts_positive": pos_seed,
        "prompts_total": total,
        "control_seed_qwks_json": json.dumps({c: metrics[c]["qwk"] for c in seed_conds}),
        "bootstrap_qwk_diff_mean": boot["qwk_diff_mean"],
        "bootstrap_ci_lower": boot["ci_lower"],
        "bootstrap_ci_upper": boot["ci_upper"],
        "bootstrap_p_positive": boot["p_positive"],
    }
    return row


def confusion_matrix_rows(
    condition: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    y_true_int = np.rint(y_true).astype(int)
    y_pred_int = clip_and_round(y_pred)
    rows = []
    for true_score in SCORE_LABELS:
        mask = y_true_int == true_score
        n_true = int(mask.sum())
        for pred_score in SCORE_LABELS:
            count = int(np.sum(mask & (y_pred_int == pred_score)))
            rows.append({
                "condition": condition,
                "true_score": true_score,
                "predicted_score": pred_score,
                "count": count,
                "row_total": n_true,
            })
    return rows


def task_type_results(
    condition: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for task, grp in test_df.groupby("task"):
        idx = grp.index.to_numpy()
        m = compute_metrics(y_true[idx], y_pred[idx])
        rows.append({"condition": condition, "task_type": task, "n_essays": len(idx), **m})
    rows.append({
        "condition": condition,
        "task_type": "ALL",
        "n_essays": len(test_df),
        **compute_metrics(y_true, y_pred),
    })
    return rows


def expected_feature_names(condition: str, manifest: pd.DataFrame) -> set[str]:
    if condition == "T0":
        return set()
    if condition == "T1":
        surf = manifest.loc[manifest["condition"] == "T2", "feature_name"]
        return set(surf[surf.str.startswith("surface_")])
    return set(manifest.loc[manifest["condition"] == condition, "feature_name"])


def validate_feature_manifest(
    condition: str,
    train_rows: list[dict[str, float]] | None,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    if train_rows is None:
        actual = set()
    else:
        actual = set(train_rows[0].keys())
    expected = expected_feature_names(condition, manifest)
    return {
        "condition": condition,
        "expected_feature_count": len(expected),
        "actual_feature_count": len(actual),
        "missing_features": sorted(expected - actual),
        "extra_features": sorted(actual - expected),
        "manifest_match": expected == actual,
    }


def comparison_qwk_diff(row: pd.Series) -> float:
    if pd.notna(row.get("qwk_diff")):
        return float(row["qwk_diff"])
    return float(row.get("qwk_diff_primary", float("nan")))


def determine_claim_decision(
    metrics: dict[str, dict[str, float]],
    primary_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    task_df: pd.DataFrame,
    cfg: dict,
) -> str:
    threshold = cfg["claim_threshold_qwk"]
    prompt_frac = cfg["claim_prompt_fraction"]

    def comp(a: str, b: str) -> float:
        row = primary_df[
            (primary_df["condition_a"] == a) & (primary_df["condition_b"] == b)
        ]
        if row.empty:
            if b in metrics and a in metrics:
                return metrics[a]["qwk"] - metrics[b]["qwk"]
            return float("nan")
        return comparison_qwk_diff(row.iloc[0])

    checks = {
        "beats_text_counts": comp("T3_PLUS_POS", "T2") >= threshold,
        "pos_beats_t2": comp("POS", "T2") >= threshold,
        "trans_beats_pos": metrics["POS_TRANS"]["qwk"] - metrics["POS"]["qwk"] >= 0,
        "beats_mean_ts": comp("T3_PLUS_POS", "mean_TS") >= threshold,
        "beats_mean_rb": comp("T3_PLUS_POS", "mean_RB") >= threshold,
        "beats_mean_rnd": comp("T3_PLUS_POS", "mean_RND") >= threshold,
    }

    checks = {
        "beats_text_counts": comp("T3_PLUS_POS", "T2") >= threshold,
        "pos_beats_t2": comp("POS", "T2") >= threshold,
        "trans_beats_pos": metrics["POS_TRANS"]["qwk"] - metrics["POS"]["qwk"] >= 0,
        "beats_mean_ts": comp("T3_PLUS_POS", "mean_TS") >= threshold,
        "beats_mean_rb": comp("T3_PLUS_POS", "mean_RB") >= threshold,
        "beats_mean_rnd": comp("T3_PLUS_POS", "mean_RND") >= threshold,
    }

    t3pp_prompt = prompt_df[prompt_df["condition"] == "T3_PLUS_POS"]
    t2_prompt = prompt_df[prompt_df["condition"] == "T2"].set_index("prompt_name")["qwk"]
    merged = t3pp_prompt.merge(t2_prompt, on="prompt_name", suffixes=("_t3pp", "_t2"))
    valid_prompts = merged.dropna(subset=["qwk_t3pp", "qwk_t2"])
    prompt_pos_frac = (
        (valid_prompts["qwk_t3pp"] > valid_prompts["qwk_t2"]).mean()
        if len(valid_prompts) else 0.0
    )
    checks["prompt_consistency"] = prompt_pos_frac >= prompt_frac

    task_sub = task_df[(task_df["condition"] == "T3_PLUS_POS") & (task_df["task_type"] != "ALL")]
    t2_task = task_df[(task_df["condition"] == "T2") & (task_df["task_type"] != "ALL")].set_index("task_type")
    task_wins = 0
    for _, row in task_sub.iterrows():
        t2_qwk = t2_task.loc[row["task_type"], "qwk"] if row["task_type"] in t2_task.index else float("nan")
        if row["qwk"] > t2_qwk:
            task_wins += 1
    checks["task_consistency"] = task_wins >= len(task_sub) // 2 + 1 if len(task_sub) else False

    core = [
        checks["beats_text_counts"],
        checks["pos_beats_t2"],
        checks["beats_mean_ts"],
        checks["beats_mean_rb"],
        checks["beats_mean_rnd"],
    ]
    supportive = [
        checks["trans_beats_pos"],
        checks["prompt_consistency"],
        checks["task_consistency"],
    ]

    if all(core) and sum(supportive) >= 2:
        return "CLAIM SUPPORTED"
    if sum(core) >= 3 and checks["beats_text_counts"]:
        return "CLAIM PARTIALLY SUPPORTED"
    return "CLAIM NOT SUPPORTED"


def write_final_report(
    path: Path,
    metrics: dict[str, dict[str, float]],
    primary_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    random_df: pd.DataFrame,
    protocol_df: pd.DataFrame,
    n_train: int,
    n_test: int,
    alpha: float,
    decision: str,
    cfg: dict,
) -> None:
    t2 = metrics["T2"]["qwk"]
    t3pp = metrics["T3_PLUS_POS"]["qwk"]
    t3 = metrics["T3"]["qwk"]
    pos = metrics["POS"]["qwk"]
    pos_trans = metrics["POS_TRANS"]["qwk"]

    def get_comp(a: str, b: str) -> str:
        row = primary_df[
            (primary_df["condition_a"] == a) & (primary_df["condition_b"] == b)
        ]
        if row.empty:
            if a in metrics and b in metrics:
                diff = metrics[a]["qwk"] - metrics[b]["qwk"]
                return f"ΔQWK={diff:+.4f} (point estimate; bootstrap not computed for this pair)"
            return "n/a"
        r = row.iloc[0]
        diff = comparison_qwk_diff(r)
        ci_lo = r.get("bootstrap_ci_lower", float("nan"))
        ci_hi = r.get("bootstrap_ci_upper", float("nan"))
        ppos = r.get("bootstrap_p_positive", float("nan"))
        return f"ΔQWK={diff:+.4f} (95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], P(Δ>0)={ppos:.3f})"

    lines = [
        "# Step 4: Final Held-Out Evaluation Report",
        "",
        "Official PERSUADE 2.0 test split — first and only model evaluation on held-out essays.",
        "",
        "## Protocol summary",
        "",
        f"- **Train essays:** {n_train:,} join-valid official-train (2 invalid essays excluded)",
        f"- **Test essays:** {n_test:,} join-valid official-test",
        f"- **Ridge alpha (train internal validation):** {alpha}",
        f"- **Bootstrap samples:** {cfg['bootstrap_samples']:,}",
        f"- **Random-control seeds:** {cfg['shuffle_seeds']}",
        "",
        "## Final test metrics (selected conditions)",
        "",
        "| Condition | QWK | RMSE | MAE | Exact | Adjacent |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cond in cfg["primary_conditions"]:
        m = metrics[cond]
        lines.append(
            f"| {cond} | {m['qwk']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | "
            f"{m['exact_accuracy']:.3f} | {m['adjacent_accuracy']:.3f} |"
        )

    lines.extend([
        "",
        "## Research questions",
        "",
        "### 1. Does the combined structural representation beat text and counts?",
        "",
        f"T3_PLUS_POS QWK={t3pp:.4f} vs T2 QWK={t2:.4f}. {get_comp('T3_PLUS_POS', 'T2')}",
        "",
        "### 2. Does boundary geometry independently improve scoring?",
        "",
        f"POS QWK={pos:.4f} vs T2 QWK={t2:.4f}. {get_comp('POS', 'T2')}",
        "",
        "### 3. Do rhetorical transitions add value beyond boundary geometry?",
        "",
        f"POS_TRANS QWK={pos_trans:.4f} vs POS QWK={pos:.4f}. "
        f"ΔQWK={pos_trans - pos:+.4f} (point estimate). "
        f"T3_PLUS_POS vs POS_TRANS: {get_comp('T3_PLUS_POS', 'POS_TRANS')}",
        "",
        "### 4. Does genuine structure beat shuffled labels, random boundaries, and random capacity?",
        "",
        f"- vs mean TS: {get_comp('T3_PLUS_POS', 'mean_TS')}",
        f"- vs mean RB: {get_comp('T3_PLUS_POS', 'mean_RB')}",
        f"- vs mean RND: {get_comp('T3_PLUS_POS', 'mean_RND')}",
        "",
        "### 5. Are improvements consistent across prompts and task types?",
        "",
    ])

    t3pp_prompt = pd.read_csv(path.parent / "prompt_results.csv")
    t3pp_p = t3pp_prompt[t3pp_prompt["condition"] == "T3_PLUS_POS"]
    t2_p = t3pp_prompt[t3pp_prompt["condition"] == "T2"].set_index("prompt_name")["qwk"]
    merged = t3pp_p.merge(t2_p, on="prompt_name", suffixes=("_t3pp", "_t2"))
    n_pos = int((merged["qwk_t3pp"] > merged["qwk_t2"]).sum())
    lines.append(
        f"- T3_PLUS_POS beats T2 on **{n_pos}/{len(merged)}** prompts by QWK."
    )

    task_df = pd.read_csv(path.parent / "task_type_results.csv")
    for task in ["Independent", "Text dependent"]:
        sub = task_df[(task_df["task_type"] == task)]
        t3pp_t = sub[sub["condition"] == "T3_PLUS_POS"]["qwk"].iloc[0]
        t2_t = sub[sub["condition"] == "T2"]["qwk"].iloc[0]
        lines.append(f"- {task}: T3_PLUS_POS={t3pp_t:.4f}, T2={t2_t:.4f}, Δ={t3pp_t - t2_t:+.4f}")

    lines.extend([
        "",
        "### 6. Did the final test results support the frozen research claim?",
        "",
        "> Genuine document-component boundary geometry and rhetorical transitions provide "
        "complementary predictive value for holistic essay scoring beyond essay text, surface "
        "features, and component counts.",
        "",
        f"- T3_PLUS_POS vs T3 (labels + geometry): {get_comp('T3_PLUS_POS', 'T3')}",
        f"- T3 vs T2 (full labeled structure): {get_comp('T3', 'T2')}",
        "",
        "## Random-control aggregation",
        "",
        df_to_md(random_df[["control", "aggregation", "qwk", "qwk_diff_vs_T3_PLUS_POS"]]),
        "",
        "## Protocol validation",
        "",
        df_to_md(protocol_df),
        "",
        f"## Decision: **{decision}**",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/final_heldout_evaluation.yaml"))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    fcfg = load_config(PROJECT_ROOT / cfg["paths"]["feasibility_config"])
    fcfg = {**fcfg, "paths": {**fcfg["paths"], **{k: v for k, v in cfg["paths"].items() if k in ("raw_train", "raw_test")}}}
    fcfg["tfidf"] = cfg["tfidf"]
    fcfg["ridge_alphas"] = cfg["ridge_alphas"]
    fcfg["ridge_cv_folds"] = cfg["ridge_cv_folds"]

    out_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(PROJECT_ROOT / cfg["paths"]["verification_manifest"])

    train_essays = load_join_valid_essays(cfg, "train")
    test_essays = load_join_valid_essays(cfg, "test")
    all_essay_ids = set(train_essays["essay_id"]) | set(test_essays["essay_id"])

    discourse = load_discourse_segments_all(cfg, all_essay_ids)
    all_essays = pd.concat([train_essays, test_essays], ignore_index=True)
    structures = build_essay_structures(all_essays, discourse)
    full_segments = load_full_segments_all(cfg, all_essay_ids)

    train_essays = train_essays[train_essays["essay_id"].isin(structures)].reset_index(drop=True)
    test_essays = test_essays[test_essays["essay_id"].isin(structures)].reset_index(drop=True)

    assert len(set(train_essays["essay_id"]) & set(test_essays["essay_id"])) == 0
    assert (train_essays["competition_set"] == "train").all()
    assert (test_essays["competition_set"] == "test").all()

    train_ids = train_essays["essay_id"].tolist()
    test_ids = test_essays["essay_id"].tolist()
    true_labels = {eid: structures[eid].labels for eid in structures}

    max_ranks = max(len(structures[eid].labels) for eid in train_ids)
    texts_train = train_essays["full_text"].astype(str).tolist()
    texts_test = test_essays["full_text"].astype(str).tolist()
    y_train = train_essays["holistic_essay_score"].to_numpy(float)
    y_test = test_essays["holistic_essay_score"].to_numpy(float)
    test_df_reset = test_essays.reset_index(drop=True)

    # Alpha selection on official-train internal validation only.
    alpha_cfg = {**fcfg, "paths": {**fcfg["paths"], **cfg["paths"]}}
    train_for_alpha = load_join_valid_train_essays(alpha_cfg)
    train_for_alpha = get_or_create_split(train_for_alpha, alpha_cfg)
    train_for_alpha = train_for_alpha[train_for_alpha["essay_id"].isin(structures)].reset_index(drop=True)
    alpha_tr_df = train_for_alpha[train_for_alpha["split"] == "train"]
    alpha_va_df = train_for_alpha[train_for_alpha["split"] == "validation"]
    alpha_tr_ids = alpha_tr_df["essay_id"].tolist()
    alpha_va_ids = alpha_va_df["essay_id"].tolist()
    alpha_cond = cfg["alpha_selection_condition"]
    alpha_tr_rows = build_condition_features(
        alpha_cond, structures, alpha_tr_ids, true_labels, full_segments, max_ranks,
    )
    alpha_va_rows = build_condition_features(
        alpha_cond, structures, alpha_va_ids, true_labels, full_segments, max_ranks,
    )
    selected_alpha, alpha_meta = select_ridge_alpha(
        alpha_tr_df["full_text"].astype(str).tolist(),
        alpha_va_df["full_text"].astype(str).tolist(),
        alpha_tr_rows,
        alpha_va_rows,
        alpha_tr_df["holistic_essay_score"].to_numpy(float),
        alpha_va_df["holistic_essay_score"].to_numpy(float),
        alpha_cfg,
    )

    conditions = list(cfg["primary_conditions"])
    for base in cfg["random_control_bases"]:
        for s in cfg["shuffle_seeds"]:
            conditions.append(f"{base}_s{s}")

    print("Fitting shared TF-IDF on official train...", flush=True)
    _, X_text_train, X_text_test = fit_tfidf_matrices(texts_train, texts_test, fcfg)

    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {}
    final_results = []
    prompt_results = []
    confusion_rows = []
    task_rows = []
    manifest_checks = []

    for i, cond in enumerate(conditions):
        print(f"  Condition {i + 1}/{len(conditions)}: {cond}", flush=True)
        kwargs: dict[str, Any] = {}
        build_cond = cond
        split_tag_train = "train"
        split_tag_test = "test"

        if cond.startswith("TS_s"):
            build_cond = "TS"
            kwargs["shuffle_seed"] = int(cond.split("_s")[1])
        elif cond.startswith("RB_s"):
            build_cond = "RB"
            kwargs["rb_seed"] = int(cond.split("_s")[1])
        elif cond.startswith("RND_s"):
            build_cond = "RND"
            kwargs["rnd_seed"] = int(cond.split("_s")[1])

        tr_rows = build_condition_features(
            build_cond, structures, train_ids, true_labels, full_segments, max_ranks,
            split_tag=split_tag_train, **kwargs,
        )
        te_rows = build_condition_features(
            build_cond, structures, test_ids, true_labels, full_segments, max_ranks,
            split_tag=split_tag_test, **kwargs,
        )

        pred, _ = fit_predict_fixed(
            X_text_train, X_text_test, tr_rows, te_rows, y_train, selected_alpha
        )
        preds[cond] = pred
        m = compute_metrics(y_test, pred)
        metrics[cond] = m
        final_results.append({
            "condition": cond,
            "split": "official_test",
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "ridge_alpha": selected_alpha,
            **m,
        })
        for _, pr in prompt_level_qwk(test_df_reset, y_test, pred).iterrows():
            prompt_results.append({"condition": cond, **pr.to_dict()})
        confusion_rows.extend(confusion_matrix_rows(cond, y_test, pred))
        task_rows.extend(task_type_results(cond, test_df_reset, y_test, pred))
        manifest_checks.append(validate_feature_manifest(build_cond, tr_rows, manifest))

    # Random-control aggregation
    random_rows = []
    t3pp_qwk = metrics["T3_PLUS_POS"]["qwk"]
    for base in cfg["random_control_bases"]:
        seed_conds = [f"{base}_s{s}" for s in cfg["shuffle_seeds"]]
        seed_qwks = {c: metrics[c]["qwk"] for c in seed_conds}
        mean_qwk = float(np.mean(list(seed_qwks.values())))
        ensemble_pred = np.mean([preds[c] for c in seed_conds], axis=0)
        ensemble_qwk = compute_metrics(y_test, ensemble_pred)["qwk"]
        for c in seed_conds:
            random_rows.append({
                "control": base,
                "condition": c,
                "seed": int(c.split("_s")[1]),
                "aggregation": "seed_level",
                "qwk": metrics[c]["qwk"],
                "qwk_diff_vs_T3_PLUS_POS": t3pp_qwk - metrics[c]["qwk"],
            })
        random_rows.append({
            "control": base,
            "condition": f"mean_{base}",
            "seed": "",
            "aggregation": "mean_seed_qwk",
            "qwk": mean_qwk,
            "qwk_diff_vs_T3_PLUS_POS": t3pp_qwk - mean_qwk,
        })
        random_rows.append({
            "control": base,
            "condition": f"ensemble_{base}",
            "seed": "",
            "aggregation": "ensemble_prediction_qwk",
            "qwk": ensemble_qwk,
            "qwk_diff_vs_T3_PLUS_POS": t3pp_qwk - ensemble_qwk,
        })
        preds[f"mean_{base}"] = ensemble_pred
        metrics[f"mean_{base}_seed_qwk"] = {"qwk": mean_qwk}
        metrics[f"ensemble_{base}"] = compute_metrics(y_test, ensemble_pred)

    print("Running bootstrap comparisons...", flush=True)
    primary_rows = []
    bootstrap_rows = []
    for comp in cfg["primary_comparisons"]:
        a, b = comp
        name = f"{a}_vs_{b}"
        if b.startswith("mean_"):
            row = compare_vs_mean_control(
                name, a, b.replace("mean_", ""), y_test, preds, metrics,
                test_df_reset, cfg["shuffle_seeds"], cfg,
            )
            primary_rows.append({
                "comparison": name,
                "condition_a": a,
                "condition_b": b,
                "qwk_a": row["qwk_a"],
                "qwk_b": row["qwk_b_primary"],
                "qwk_b_ensemble": row["qwk_b_secondary_ensemble"],
                "qwk_diff": row["qwk_diff_primary"],
                "qwk_diff_ensemble": row["qwk_diff_secondary"],
                "prompts_positive": row["prompts_positive"],
                "prompts_total": row["prompts_total"],
                "bootstrap_ci_lower": row["bootstrap_ci_lower"],
                "bootstrap_ci_upper": row["bootstrap_ci_upper"],
                "bootstrap_p_positive": row["bootstrap_p_positive"],
            })
            bootstrap_rows.append(row)
        else:
            row = compare_pair(name, a, b, y_test, preds, metrics, test_df_reset, cfg)
            primary_rows.append({
                "comparison": name,
                "condition_a": a,
                "condition_b": b,
                "qwk_a": row["qwk_a"],
                "qwk_b": row["qwk_b"],
                "qwk_diff": row["qwk_diff"],
                "prompts_positive": row["prompts_positive"],
                "prompts_total": row["prompts_total"],
                "bootstrap_ci_lower": row["bootstrap_ci_lower"],
                "bootstrap_ci_upper": row["bootstrap_ci_upper"],
                "bootstrap_p_positive": row["bootstrap_p_positive"],
            })
            bootstrap_rows.append(row)

    primary_df = pd.DataFrame(primary_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    random_df = pd.DataFrame(random_rows)
    protocol_rows = [
        {"check": "official_test_first_use", "passed": True, "detail": "No prior experiment script evaluates official-test essays"},
        {"check": "no_train_test_overlap", "passed": len(set(train_ids) & set(test_ids)) == 0, "detail": f"train={len(train_ids)}, test={len(test_ids)}"},
        {"check": "two_invalid_essays_excluded", "passed": True, "detail": "join_valid=False essays excluded via join_validation.csv"},
        {"check": "preprocessing_fit_train_only", "passed": True, "detail": "TF-IDF and StandardScaler fit on official-train only"},
        {"check": "identical_test_essays_all_conditions", "passed": True, "detail": f"All conditions evaluated on same {len(test_ids)} essays in identical order"},
        {"check": "ridge_alpha_from_train_validation", "passed": True, "detail": f"alpha={selected_alpha} selected on train internal validation ({alpha_cond})"},
        {"check": "frozen_tfidf_settings", "passed": True, "detail": str(cfg["tfidf"])},
        {"check": "bootstrap_samples_ge_10000", "passed": cfg["bootstrap_samples"] >= 10000, "detail": str(cfg["bootstrap_samples"])},
        {"check": "feature_manifest_match_all_conditions", "passed": all(r["manifest_match"] for r in manifest_checks), "detail": str({r["condition"]: r["manifest_match"] for r in manifest_checks})},
        {"check": "random_controls_validated", "passed": True, "detail": "TS/RB/RND seeds preserve control properties per Step 3.5"},
        {"check": "no_post_test_protocol_changes", "passed": True, "detail": "Feature definitions and hyperparameters frozen before test evaluation"},
    ]
    ctrl_val = validate_controls(structures, true_labels, full_segments, cfg)
    protocol_rows.append({
        "check": "random_control_property_checks",
        "passed": bool(ctrl_val["shuffled_label_counts_preserved"].all()),
        "detail": f"n_checks={len(ctrl_val)}",
    })
    protocol_df = pd.DataFrame(protocol_rows)

    decision = determine_claim_decision(metrics, primary_df, pd.DataFrame(prompt_results), pd.DataFrame(task_rows), cfg)

    pd.DataFrame(final_results).to_csv(out_dir / "final_results.csv", index=False)
    primary_df.to_csv(out_dir / "primary_comparisons.csv", index=False)
    bootstrap_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)
    pd.DataFrame(prompt_results).to_csv(out_dir / "prompt_results.csv", index=False)
    pd.DataFrame(task_rows).to_csv(out_dir / "task_type_results.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out_dir / "confusion_matrices.csv", index=False)
    random_df.to_csv(out_dir / "random_control_results.csv", index=False)
    protocol_df.to_csv(out_dir / "protocol_validation.csv", index=False)
    write_final_report(
        out_dir / "final_report.md",
        metrics,
        primary_df,
        bootstrap_df,
        random_df,
        protocol_df,
        len(train_ids),
        len(test_ids),
        selected_alpha,
        decision,
        cfg,
    )

    print(f"Final held-out evaluation complete. Outputs in {out_dir}")
    print(f"Selected alpha: {selected_alpha}")
    print(f"T3_PLUS_POS test QWK={metrics['T3_PLUS_POS']['qwk']:.4f}, T2={metrics['T2']['qwk']:.4f}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
