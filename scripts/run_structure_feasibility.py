#!/usr/bin/env python3
"""
Fast feasibility screening: does rhetorical order predict holistic scores
beyond text, surface features, and discourse counts?

Uses join-valid official-train essays only. Never touches the official test set.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RHETORICAL_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]


@dataclass
class EssayStructure:
    essay_id: str
    full_text: str
    labels: list[str]
    starts: list[int]
    ends: list[int]
    unannotated_count: int
    total_segments: int
    char_count: int
    coverage_chars: int


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    return max(1, sum(1 for p in parts if p.strip()))


def paragraph_count(text: str) -> int:
    parts = re.split(r"\n\s*\n", text.strip())
    return max(1, sum(1 for p in parts if p.strip()))


def mean_word_length(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_join_valid_train_essays(cfg: dict) -> pd.DataFrame:
    essays = pd.read_csv(PROJECT_ROOT / cfg["paths"]["essays"])
    join_val = pd.read_csv(PROJECT_ROOT / cfg["paths"]["join_validation"])
    valid_ids = set(join_val.loc[join_val["join_valid"], "essay_id"].astype(str))
    essays["essay_id"] = essays["essay_id"].astype(str)
    mask = (
        essays["essay_id"].isin(valid_ids)
        & (essays["competition_set"] == "train")
    )
    return essays.loc[mask].reset_index(drop=True)


def load_discourse_segments(cfg: dict, essay_ids: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["raw_train"],
        usecols=["essay_id", "discourse_start", "discourse_end", "discourse_type"],
        low_memory=False,
    )
    raw["essay_id"] = raw["essay_id"].astype(str)
    return raw[raw["essay_id"].isin(essay_ids)].copy()


def build_essay_structures(essays: pd.DataFrame, discourse: pd.DataFrame) -> dict[str, EssayStructure]:
    text_map = essays.set_index("essay_id")["full_text"].to_dict()
    structures: dict[str, EssayStructure] = {}

    for eid, grp in discourse.groupby("essay_id"):
        eid = str(eid)
        full_text = str(text_map.get(eid, ""))
        grp = grp.sort_values("discourse_start")
        labels = grp["discourse_type"].astype(str).tolist()
        starts = [int(x) for x in grp["discourse_start"]]
        ends = [int(x) for x in grp["discourse_end"]]
        coverage = sum(e - s + 1 for s, e in zip(starts, ends))
        unannotated = sum(1 for lb in labels if lb == "Unannotated")
        rhetorical = [lb for lb in labels if lb != "Unannotated"]
        structures[eid] = EssayStructure(
            essay_id=eid,
            full_text=full_text,
            labels=rhetorical,
            starts=[s for s, lb in zip(starts, labels) if lb != "Unannotated"],
            ends=[e for e, lb in zip(ends, labels) if lb != "Unannotated"],
            unannotated_count=unannotated,
            total_segments=len(labels),
            char_count=len(full_text),
            coverage_chars=coverage,
        )
    return structures


def get_or_create_split(essays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    split_path = PROJECT_ROOT / cfg["paths"]["split_output"]
    essays = essays.copy()
    essays["essay_id"] = essays["essay_id"].astype(str)

    if split_path.exists():
        split_df = pd.read_csv(split_path)
        split_df["essay_id"] = split_df["essay_id"].astype(str)
        merged = essays.merge(split_df[["essay_id", "split"]], on="essay_id", how="left")
        if merged["split"].notna().all():
            return merged

    essays["stratum"] = (
        essays["prompt_name"].astype(str) + "__" + essays["holistic_essay_score"].astype(int).astype(str)
    )
    min_stratum = essays["stratum"].value_counts().min()
    stratify = essays["stratum"] if min_stratum >= 2 else None

    train_ids, val_ids = train_test_split(
        essays["essay_id"],
        test_size=1 - cfg["train_fraction"],
        random_state=cfg["seed"],
        stratify=stratify,
    )
    split_df = pd.DataFrame({"essay_id": essays["essay_id"]})
    split_df["split"] = np.where(split_df["essay_id"].isin(train_ids), "train", "validation")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_path, index=False)
    return essays.merge(split_df, on="essay_id")


def shuffle_labels(structure: EssayStructure, seed: int, essay_id: str) -> list[str]:
    rng = np.random.default_rng(seed ^ (hash(essay_id) & 0xFFFFFFFF))
    labels = list(structure.labels)
    if len(labels) <= 1:
        return labels
    perm = rng.permutation(len(labels))
    return [labels[i] for i in perm]


def surface_feature_dict(text: str) -> dict[str, float]:
    wc = word_count(text)
    cc = len(text)
    return {
        "surface_word_count": wc,
        "surface_char_count": cc,
        "surface_sentence_count": sentence_count(text),
        "surface_paragraph_count": paragraph_count(text),
        "surface_mean_word_length": mean_word_length(text),
    }


def count_feature_dict(structure: EssayStructure) -> dict[str, float]:
    total = max(structure.total_segments, 1)
    type_counts = {t: 0 for t in RHETORICAL_TYPES}
    for lb in structure.labels:
        if lb in type_counts:
            type_counts[lb] += 1

    feats: dict[str, float] = {
        "count_total_components": len(structure.labels),
        "count_total_segments": structure.total_segments,
        "count_annotation_coverage": structure.coverage_chars / max(structure.char_count, 1),
        "count_unannotated_gaps": structure.unannotated_count,
        "count_unannotated_proportion": structure.unannotated_count / total,
    }
    for t in RHETORICAL_TYPES:
        feats[f"count_{t.replace(' ', '_')}"] = type_counts[t]
        feats[f"prop_{t.replace(' ', '_')}"] = type_counts[t] / max(len(structure.labels), 1)
    return feats


def order_feature_dict(structure: EssayStructure, labels: list[str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    text_len = max(structure.char_count, 1)
    seq = labels

    for t in RHETORICAL_TYPES:
        key = t.replace(" ", "_")
        feats[f"order_first_is_{key}"] = float(seq[0] == t) if seq else 0.0
        feats[f"order_final_is_{key}"] = float(seq[-1] == t) if seq else 0.0

    trans = {f"order_trans_{a.replace(' ', '_')}__{b.replace(' ', '_')}": 0.0
             for a, b in product(RHETORICAL_TYPES, RHETORICAL_TYPES)}
    if len(seq) >= 2:
        for a, b in zip(seq[:-1], seq[1:]):
            if a in RHETORICAL_TYPES and b in RHETORICAL_TYPES:
                k = f"order_trans_{a.replace(' ', '_')}__{b.replace(' ', '_')}"
                trans[k] += 1.0
        denom = len(seq) - 1
        for k in trans:
            trans[k] /= denom
    feats.update(trans)

    positions_by_type: dict[str, list[float]] = {t: [] for t in RHETORICAL_TYPES}
    for lb, s, e in zip(labels, structure.starts, structure.ends):
        if lb in RHETORICAL_TYPES:
            positions_by_type[lb].append(((s + e) / 2.0) / text_len)

    for t in RHETORICAL_TYPES:
        key = t.replace(" ", "_")
        pos = positions_by_type[t]
        if pos:
            feats[f"order_mean_pos_{key}"] = float(np.mean(pos))
            feats[f"order_first_pos_{key}"] = float(pos[0])
            feats[f"order_final_pos_{key}"] = float(pos[-1])
            feats[f"order_std_pos_{key}"] = float(np.std(pos)) if len(pos) > 1 else 0.0
        else:
            feats[f"order_mean_pos_{key}"] = 0.0
            feats[f"order_first_pos_{key}"] = 0.0
            feats[f"order_final_pos_{key}"] = 0.0
            feats[f"order_std_pos_{key}"] = 0.0
    return feats


def build_feature_row(
    structure: EssayStructure,
    labels: list[str],
    include_surface: bool,
    include_counts: bool,
    include_order: bool,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    if include_surface:
        feats.update(surface_feature_dict(structure.full_text))
    if include_counts:
        feats.update(count_feature_dict(structure))
    if include_order:
        feats.update(order_feature_dict(structure, labels))
    return feats


def validate_shuffle(
    structure: EssayStructure,
    true_labels: list[str],
    shuffled_labels: list[str],
) -> dict[str, Any]:
    true_counts = count_feature_dict(structure)
    shuf_structure = EssayStructure(
        essay_id=structure.essay_id,
        full_text=structure.full_text,
        labels=shuffled_labels,
        starts=structure.starts,
        ends=structure.ends,
        unannotated_count=structure.unannotated_count,
        total_segments=structure.total_segments,
        char_count=structure.char_count,
        coverage_chars=structure.coverage_chars,
    )
    shuf_counts = count_feature_dict(shuf_structure)

    preserved_keys = [
        "count_total_components",
        "count_total_segments",
        "count_annotation_coverage",
        "count_unannotated_gaps",
        "count_unannotated_proportion",
    ] + [f"count_{t.replace(' ', '_')}" for t in RHETORICAL_TYPES] + [
        f"prop_{t.replace(' ', '_')}" for t in RHETORICAL_TYPES
    ]
    preserved_ok = all(true_counts[k] == shuf_counts[k] for k in preserved_keys)

    true_order = order_feature_dict(structure, true_labels)
    shuf_order = order_feature_dict(structure, shuffled_labels)
    order_changed = true_order != shuf_order

    return {
        "essay_id": structure.essay_id,
        "n_components": len(true_labels),
        "preserved_count_features_ok": preserved_ok,
        "order_features_changed": order_changed,
        "labels_changed": true_labels != shuffled_labels,
        "text_unchanged": True,
        "boundaries_unchanged": True,
    }


def features_to_matrix(rows: list[dict[str, float]], columns: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if not rows:
        return np.empty((0, 0)), []
    if columns is None:
        columns = sorted({k for row in rows for k in row})
    mat = np.array([[row.get(c, 0.0) for c in columns] for row in rows], dtype=float)
    return mat, columns


def clip_and_round(y_pred: np.ndarray, lo: int = 1, hi: int = 6) -> np.ndarray:
    clipped = np.clip(y_pred, lo, hi)
    return np.rint(clipped).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_int = np.rint(y_true).astype(int)
    y_pred_clip = np.clip(y_pred, 1, 6)
    y_pred_round = clip_and_round(y_pred)

    return {
        "qwk": float(cohen_kappa_score(y_true_int, y_pred_round, weights="quadratic")),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clip))),
        "mae": float(mean_absolute_error(y_true, y_pred_clip)),
        "exact_accuracy": float(np.mean(y_true_int == y_pred_round)),
        "adjacent_accuracy": float(np.mean(np.abs(y_true_int - y_pred_round) <= 1)),
    }


def prompt_level_qwk(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    rounded = clip_and_round(y_pred)
    y_true_int = np.rint(y_true).astype(int)
    for prompt, grp in df.groupby("prompt_name"):
        idx = grp.index.to_numpy()
        if len(idx) < 2 or len(np.unique(y_true_int[idx])) < 2:
            qwk = float("nan")
        else:
            qwk = float(cohen_kappa_score(y_true_int[idx], rounded[idx], weights="quadratic"))
        rows.append({"prompt_name": prompt, "n_essays": len(idx), "qwk": qwk})
    return pd.DataFrame(rows)


def fit_hgb(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> HistGradientBoostingRegressor:
    params = cfg["hist_gradient_boosting"]
    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def fit_ridge_tfidf(
    texts: list[str],
    X_num_train: np.ndarray | None,
    y_train: np.ndarray,
    texts_val: list[str],
    X_num_val: np.ndarray | None,
    cfg: dict,
) -> tuple[Any, Any, StandardScaler | None, float]:
    tfidf = TfidfVectorizer(
        ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        max_features=cfg["tfidf"]["max_features"],
        min_df=cfg["tfidf"]["min_df"],
    )
    X_text = tfidf.fit_transform(texts)

    scaler = None
    if X_num_train is not None and X_num_train.shape[1] > 0:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num_train)
        X_train = hstack([X_text, csr_matrix(X_num_scaled)])
    else:
        X_train = X_text

    alphas = cfg["ridge_alphas"]
    grid = GridSearchCV(
        Ridge(),
        param_grid={"alpha": alphas},
        cv=cfg["ridge_cv_folds"],
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    best_alpha = float(grid.best_params_["alpha"])
    model = grid.best_estimator_

    X_text_val = tfidf.transform(texts_val)
    if scaler is not None and X_num_val is not None:
        X_num_val_scaled = scaler.transform(X_num_val)
        X_val = hstack([X_text_val, csr_matrix(X_num_val_scaled)])
    else:
        X_val = X_text_val

    return model, tfidf, scaler, best_alpha


def predict_ridge(model, tfidf, scaler, texts: list[str], X_num: np.ndarray | None) -> np.ndarray:
    X_text = tfidf.transform(texts)
    if scaler is not None and X_num is not None and X_num.shape[1] > 0:
        X = hstack([X_text, csr_matrix(scaler.transform(X_num))])
    else:
        X = X_text
    return model.predict(X)


def paired_bootstrap_qwk_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        qwk_a = cohen_kappa_score(
            np.rint(y_true[idx]).astype(int),
            clip_and_round(pred_a[idx]),
            weights="quadratic",
        )
        qwk_b = cohen_kappa_score(
            np.rint(y_true[idx]).astype(int),
            clip_and_round(pred_b[idx]),
            weights="quadratic",
        )
        diffs.append(qwk_a - qwk_b)
    diffs = np.array(diffs)
    return {
        "qwk_diff_mean": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_positive": float(np.mean(diffs > 0)),
    }


def prompt_positive_fraction(
    df_val: pd.DataFrame,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> tuple[int, int]:
    positive = 0
    total = 0
    for _, grp in df_val.groupby("prompt_name"):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        yt = np.rint(y_true[idx]).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        qwk_a = cohen_kappa_score(yt, clip_and_round(pred_a[idx]), weights="quadratic")
        qwk_b = cohen_kappa_score(yt, clip_and_round(pred_b[idx]), weights="quadratic")
        total += 1
        if qwk_a > qwk_b:
            positive += 1
    return positive, total


@dataclass
class RunResult:
    condition: str
    seed: int | None
    split: str
    metrics: dict[str, float]
    predictions: np.ndarray
    feature_columns: list[str] = field(default_factory=list)
    best_alpha: float | None = None


def run_numeric_condition(
    name: str,
    train_rows: list[dict[str, float]],
    val_rows: list[dict[str, float]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int | None = None,
) -> tuple[RunResult, np.ndarray, np.ndarray]:
    X_train, cols = features_to_matrix(train_rows)
    X_val, _ = features_to_matrix(val_rows, cols)
    model = fit_hgb(X_train, y_train, cfg)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    result = RunResult(
        condition=name,
        seed=seed,
        split="",
        metrics={},
        predictions=pred_val,
        feature_columns=cols,
    )
    return result, pred_train, pred_val


def run_text_condition(
    name: str,
    texts_train: list[str],
    texts_val: list[str],
    num_train: list[dict[str, float]] | None,
    num_val: list[dict[str, float]] | None,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int | None = None,
) -> tuple[RunResult, np.ndarray, np.ndarray, float]:
    cols: list[str] = []
    X_num_train = X_num_val = None
    if num_train is not None:
        X_num_train, cols = features_to_matrix(num_train)
        X_num_val, _ = features_to_matrix(num_val, cols)

    model, tfidf, scaler, alpha = fit_ridge_tfidf(
        texts_train, X_num_train, y_train, texts_val, X_num_val, cfg
    )
    pred_train = predict_ridge(model, tfidf, scaler, texts_train, X_num_train)
    pred_val = predict_ridge(model, tfidf, scaler, texts_val, X_num_val)
    result = RunResult(
        condition=name,
        seed=seed,
        split="",
        metrics={},
        predictions=pred_val,
        feature_columns=(["tfidf"] + cols),
        best_alpha=alpha,
    )
    return result, pred_train, pred_val, alpha


def build_feature_sets(
    structures: dict[str, EssayStructure],
    essay_ids: list[str],
    label_map: dict[str, list[str]],
    include_surface: bool,
    include_counts: bool,
    include_order: bool,
) -> list[dict[str, float]]:
    rows = []
    for eid in essay_ids:
        st = structures[eid]
        rows.append(
            build_feature_row(
                st,
                label_map[eid],
                include_surface=include_surface,
                include_counts=include_counts,
                include_order=include_order,
            )
        )
    return rows


def df_to_md(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.values)
    return "\n".join([headers, sep, body])


def write_feasibility_report(
    path: Path,
    results_val: dict[str, dict[str, float]],
    bootstrap_df: pd.DataFrame,
    cfg: dict,
    n_train: int,
    n_val: int,
) -> None:
    t3_qwk = results_val.get("T3", {}).get("qwk", float("nan"))
    t2_qwk = results_val.get("T2", {}).get("qwk", float("nan"))
    n2_qwk = results_val.get("N2", {}).get("qwk", float("nan"))
    n1_qwk = results_val.get("N1", {}).get("qwk", float("nan"))

    ts_qwks = [results_val.get(f"TS_s{s}", {}).get("qwk", float("nan")) for s in cfg["shuffle_seeds"]]
    ns_qwks = [results_val.get(f"NS_s{s}", {}).get("qwk", float("nan")) for s in cfg["shuffle_seeds"]]
    mean_ts = float(np.nanmean(ts_qwks))
    mean_ns = float(np.nanmean(ns_qwks))

    t3_t2 = bootstrap_df[bootstrap_df["comparison"] == "T3_vs_T2"]
    t3_ts_rows = bootstrap_df[bootstrap_df["comparison"].str.startswith("T3_vs_TS")]
    n2_n1 = bootstrap_df[bootstrap_df["comparison"] == "N2_vs_N1"]
    n2_ns_rows = bootstrap_df[bootstrap_df["comparison"].str.startswith("N2_vs_NS")]

    t3_beats_t2 = t3_qwk - t2_qwk >= cfg["decision"]["green_qwk_threshold"]
    t3_beats_ts = t3_qwk - mean_ts >= cfg["decision"]["green_qwk_threshold"]
    n2_beats_n1 = n2_qwk - n1_qwk >= cfg["decision"]["green_qwk_threshold"]
    n2_beats_ns = n2_qwk - mean_ns >= cfg["decision"]["green_qwk_threshold"]

    t3_ci_pos = not t3_t2.empty and t3_t2.iloc[0]["ci_lower"] > 0
    prompt_frac = (
        t3_t2.iloc[0]["prompts_positive"] / t3_t2.iloc[0]["prompts_total"]
        if not t3_t2.empty and t3_t2.iloc[0]["prompts_total"] > 0
        else 0.0
    )

    green = (
        t3_beats_t2
        and t3_beats_ts
        and t3_ci_pos
        and prompt_frac >= cfg["decision"]["green_prompt_fraction"]
    )
    red = (
        (n2_qwk <= mean_ns or n2_qwk <= n1_qwk)
        and (t3_qwk <= mean_ts or t3_qwk <= t2_qwk)
    )
    if green:
        decision = "GREEN"
        decision_text = (
            "Proceed with the full structural-validity study. True rhetorical order "
            "shows a reliable advantage over counts and shuffled controls when text is available."
        )
    elif red:
        decision = "RED"
        decision_text = (
            "Do not invest in the full order-based pipeline yet. True order is not "
            "meaningfully better than shuffled order under this screening setup."
        )
    else:
        decision = "YELLOW"
        decision_text = (
            "The signal is weak or inconsistent. A full study may still be justified "
            "with richer structural representations, but this screening is inconclusive."
        )

    lines = [
        "# Structural Feasibility Report (PERSUADE 2.0)",
        "",
        "## Setup",
        "",
        f"- Join-valid official-train essays only: **{n_train + n_val:,}**",
        f"- Train / validation: **{n_train:,}** / **{n_val:,}** (80/20, stratified by prompt × score)",
        f"- Official test set: **not used**",
        f"- Bootstrap samples: **{cfg['bootstrap_samples']:,}**",
        "",
        "## Validation QWK by condition",
        "",
        "| Condition | QWK | RMSE | MAE | Exact | Adjacent |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cond in [
        "N0", "N1", "N2",
        *[f"NS_s{s}" for s in cfg["shuffle_seeds"]],
        "T0", "T1", "T2", "T3",
        *[f"TS_s{s}" for s in cfg["shuffle_seeds"]],
    ]:
        if cond in results_val:
            m = results_val[cond]
            lines.append(
                f"| {cond} | {m['qwk']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | "
                f"{m['exact_accuracy']:.4f} | {m['adjacent_accuracy']:.4f} |"
            )

    boot_md = df_to_md(bootstrap_df)
    lines.extend(
        [
            "",
            "## Primary comparisons (validation QWK difference, A − B)",
            "",
            boot_md,
            "",
            "## Research questions",
            "",
            "### 1. Does true rhetorical order outperform counts alone?",
            "",
            f"- **Numeric:** N2 QWK={n2_qwk:.4f} vs N1 QWK={n1_qwk:.4f} (Δ={n2_qwk - n1_qwk:+.4f})",
            f"- **Text:** T3 QWK={t3_qwk:.4f} vs T2 QWK={t2_qwk:.4f} (Δ={t3_qwk - t2_qwk:+.4f})",
            "",
            "### 2. Does true rhetorical order outperform shuffled order?",
            "",
            f"- **Numeric:** N2 vs mean NS QWK={mean_ns:.4f} (Δ={n2_qwk - mean_ns:+.4f})",
            f"- **Text:** T3 vs mean TS QWK={mean_ts:.4f} (Δ={t3_qwk - mean_ts:+.4f})",
            "",
            "### 3. Does order still help when essay text is already available?",
            "",
            f"- T3−T2 QWK difference: **{t3_qwk - t2_qwk:+.4f}**",
            f"- Bootstrap 95% CI lower bound: **{t3_t2.iloc[0]['ci_lower']:.4f}**" if not t3_t2.empty else "",
            "",
            "### 4. Are improvements consistent across prompts?",
            "",
        ]
    )
    if not t3_t2.empty:
        lines.append(
            f"- T3 vs T2: **{int(t3_t2.iloc[0]['prompts_positive'])}/{int(t3_t2.iloc[0]['prompts_total'])}** "
            f"prompts show positive QWK difference ({prompt_frac:.1%})"
        )
    if not n2_n1.empty:
        lines.append(
            f"- N2 vs N1: **{int(n2_n1.iloc[0]['prompts_positive'])}/{int(n2_n1.iloc[0]['prompts_total'])}** "
            f"prompts show positive QWK difference"
        )

    lines.extend(
        [
            "",
            "### 5. Should we proceed with the full structural-validity study?",
            "",
            f"## Decision: **{decision}**",
            "",
            decision_text,
            "",
            "### Decision criteria reference",
            "",
            f"- GREEN threshold: ≥{cfg['decision']['green_qwk_threshold']} QWK over T2 and mean TS, "
            f"positive bootstrap CI, ≥{cfg['decision']['green_prompt_fraction']:.0%} prompts positive",
            "- YELLOW: small or inconsistent advantage",
            "- RED: true order ≈ shuffled order in numeric and text experiments",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/structure_feasibility.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    out_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    essays = load_join_valid_train_essays(cfg)
    essay_ids = set(essays["essay_id"].astype(str))
    discourse = load_discourse_segments(cfg, essay_ids)
    structures = build_essay_structures(essays, discourse)

    essays = get_or_create_split(essays, cfg)
    essays = essays[essays["essay_id"].isin(structures)].reset_index(drop=True)

    train_df = essays[essays["split"] == "train"].copy()
    val_df = essays[essays["split"] == "validation"].copy()
    train_ids = train_df["essay_id"].tolist()
    val_ids = val_df["essay_id"].tolist()

    true_labels = {eid: structures[eid].labels for eid in structures}

    shuffle_validations = []
    shuffled_label_maps: dict[int, dict[str, list[str]]] = {}
    for s in cfg["shuffle_seeds"]:
        shuffled_label_maps[s] = {}
        for eid, st in structures.items():
            shuffled = shuffle_labels(st, seed=s, essay_id=eid)
            shuffled_label_maps[s][eid] = shuffled
            shuffle_validations.append(validate_shuffle(st, true_labels[eid], shuffled))
    pd.DataFrame(shuffle_validations).to_csv(out_dir / "shuffle_validation.csv", index=False)

    y_train = train_df["holistic_essay_score"].to_numpy(dtype=float)
    y_val = val_df["holistic_essay_score"].to_numpy(dtype=float)
    texts_train = train_df["full_text"].astype(str).tolist()
    texts_val = val_df["full_text"].astype(str).tolist()

    # Feature manifests and condition definitions
    condition_specs = {
        "N0": dict(include_surface=True, include_counts=False, include_order=False, shuffled_seed=None, model="numeric"),
        "N1": dict(include_surface=True, include_counts=True, include_order=False, shuffled_seed=None, model="numeric"),
        "N2": dict(include_surface=True, include_counts=True, include_order=True, shuffled_seed=None, model="numeric"),
        "T0": dict(include_surface=False, include_counts=False, include_order=False, shuffled_seed=None, model="text"),
        "T1": dict(include_surface=True, include_counts=False, include_order=False, shuffled_seed=None, model="text"),
        "T2": dict(include_surface=True, include_counts=True, include_order=False, shuffled_seed=None, model="text"),
        "T3": dict(include_surface=True, include_counts=True, include_order=True, shuffled_seed=None, model="text"),
    }
    for s in cfg["shuffle_seeds"]:
        condition_specs[f"NS_s{s}"] = dict(
            include_surface=True, include_counts=True, include_order=True,
            shuffled_seed=s, model="numeric",
        )
        condition_specs[f"TS_s{s}"] = dict(
            include_surface=True, include_counts=True, include_order=True,
            shuffled_seed=s, model="text",
        )

    manifest_rows = []
    all_results: list[dict] = []
    prompt_results: list[dict] = []
    val_predictions: dict[str, np.ndarray] = {}
    results_val_metrics: dict[str, dict[str, float]] = {}

    for cond, spec in condition_specs.items():
        label_map = true_labels
        if spec["shuffled_seed"] is not None:
            label_map = shuffled_label_maps[spec["shuffled_seed"]]

        train_feats = build_feature_sets(
            structures, train_ids, label_map,
            spec["include_surface"], spec["include_counts"], spec["include_order"],
        )
        val_feats = build_feature_sets(
            structures, val_ids, label_map,
            spec["include_surface"], spec["include_counts"], spec["include_order"],
        )

        if spec["model"] == "numeric":
            _, cols = features_to_matrix(train_feats)
            for c in cols:
                manifest_rows.append({"condition": cond, "feature": c, "feature_group": _feature_group(c)})
            result, pred_train, pred_val = run_numeric_condition(
                cond, train_feats, val_feats, y_train, y_val, cfg, spec["shuffled_seed"]
            )
            best_alpha = None
        else:
            if cond == "T0":
                num_train = num_val = None
                manifest_rows.append({"condition": cond, "feature": "tfidf", "feature_group": "text"})
            else:
                num_train = build_feature_sets(
                    structures, train_ids, label_map,
                    spec["include_surface"], spec["include_counts"], spec["include_order"],
                )
                num_val = build_feature_sets(
                    structures, val_ids, label_map,
                    spec["include_surface"], spec["include_counts"], spec["include_order"],
                )
                _, cols = features_to_matrix(num_train)
                for c in cols:
                    manifest_rows.append({"condition": cond, "feature": c, "feature_group": _feature_group(c)})
                manifest_rows.append({"condition": cond, "feature": "tfidf", "feature_group": "text"})

            result, pred_train, pred_val, best_alpha = run_text_condition(
                cond, texts_train, texts_val, num_train, num_val, y_train, y_val, cfg, spec["shuffled_seed"]
            )

        val_predictions[cond] = pred_val
        for split_name, y_true, y_pred, df_split in [
            ("train", y_train, pred_train, train_df),
            ("validation", y_val, pred_val, val_df),
        ]:
            metrics = compute_metrics(y_true, y_pred)
            row = {"condition": cond, "split": split_name, "seed": spec["shuffled_seed"], **metrics}
            if best_alpha is not None:
                row["ridge_alpha"] = best_alpha
            all_results.append(row)
            if split_name == "validation":
                results_val_metrics[cond] = metrics
                for _, pr in prompt_level_qwk(df_split.reset_index(drop=True), y_true, y_pred).iterrows():
                    prompt_results.append({"condition": cond, "seed": spec["shuffled_seed"], **pr.to_dict()})

    pd.DataFrame(all_results).to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(prompt_results).to_csv(out_dir / "prompt_results.csv", index=False)
    pd.DataFrame(manifest_rows).drop_duplicates().to_csv(out_dir / "feature_manifest.csv", index=False)

    # Bootstrap comparisons
    bootstrap_rows = []
    comparisons = [("N2_vs_N1", "N2", "N1"), ("T3_vs_T2", "T3", "T2")]
    for s in cfg["shuffle_seeds"]:
        comparisons.append((f"N2_vs_NS_s{s}", "N2", f"NS_s{s}"))
        comparisons.append((f"T3_vs_TS_s{s}", "T3", f"TS_s{s}"))

    val_df_reset = val_df.reset_index(drop=True)
    for comp_name, cond_a, cond_b in comparisons:
        stats = paired_bootstrap_qwk_diff(
            y_val,
            val_predictions[cond_a],
            val_predictions[cond_b],
            cfg["bootstrap_samples"],
            cfg["seed"],
        )
        pos, total = prompt_positive_fraction(
            val_df_reset, y_val, val_predictions[cond_a], val_predictions[cond_b]
        )
        bootstrap_rows.append(
            {
                "comparison": comp_name,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "qwk_a": results_val_metrics[cond_a]["qwk"],
                "qwk_b": results_val_metrics[cond_b]["qwk"],
                "qwk_diff": results_val_metrics[cond_a]["qwk"] - results_val_metrics[cond_b]["qwk"],
                "prompts_positive": pos,
                "prompts_total": total,
                **stats,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)

    write_feasibility_report(
        out_dir / "feasibility_report.md",
        results_val_metrics,
        bootstrap_df,
        cfg,
        len(train_df),
        len(val_df),
    )
    print(f"Feasibility experiment complete. Outputs in {out_dir}")


def _feature_group(name: str) -> str:
    if name == "tfidf":
        return "text"
    if name.startswith("surface_"):
        return "surface"
    if name.startswith("count_") or name.startswith("prop_"):
        return "count"
    if name.startswith("order_"):
        return "order"
    return "other"


if __name__ == "__main__":
    main()
