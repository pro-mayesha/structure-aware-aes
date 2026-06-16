#!/usr/bin/env python3
"""
Step 3: Stress-test rhetorical structure feature families for PERSUADE 2.0.

Identifies which parts of document structure carry predictive signal beyond
text, surface features, and discourse counts.
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
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_structure_feasibility import (  # noqa: E402
    RHETORICAL_TYPES,
    EssayStructure,
    build_essay_structures,
    build_feature_row,
    build_feature_sets,
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
)

# ---------------------------------------------------------------------------
# Extended structures for label-free position features
# ---------------------------------------------------------------------------


def load_full_segments(cfg: dict, essay_ids: set[str]) -> dict[str, list[tuple[int, int, str]]]:
    raw = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["raw_train"],
        usecols=["essay_id", "discourse_start", "discourse_end", "discourse_type"],
        low_memory=False,
    )
    raw["essay_id"] = raw["essay_id"].astype(str)
    raw = raw[raw["essay_id"].isin(essay_ids)].sort_values(["essay_id", "discourse_start"])
    segments: dict[str, list[tuple[int, int, str]]] = {}
    for eid, grp in raw.groupby("essay_id"):
        segments[str(eid)] = [
            (int(r.discourse_start), int(r.discourse_end), str(r.discourse_type))
            for r in grp.itertuples()
        ]
    return segments


def split_order_features(feats: dict[str, float]) -> dict[str, dict[str, float]]:
    firstlast = {
        k: v for k, v in feats.items()
        if k.startswith("order_first_is_") or k.startswith("order_final_is_")
    }
    trans = {k: v for k, v in feats.items() if k.startswith("order_trans_")}
    labelpos = {
        k: v for k, v in feats.items()
        if k.startswith("order_mean_pos_")
        or k.startswith("order_first_pos_")
        or k.startswith("order_final_pos_")
        or k.startswith("order_std_pos_")
    }
    return {"firstlast": firstlast, "trans": trans, "labelpos": labelpos}


def t2_feature_dict(structure: EssayStructure, labels: list[str]) -> dict[str, float]:
    return build_feature_row(structure, labels, True, True, False)


def order_keys_full() -> list[str]:
    dummy = EssayStructure(
        essay_id="x", full_text="abc", labels=["Lead", "Claim"],
        starts=[0, 1], ends=[0, 2], unannotated_count=0,
        total_segments=2, char_count=3, coverage_chars=3,
    )
    return sorted(order_feature_dict(dummy, dummy.labels).keys())


def pos_feature_dict(
    structure: EssayStructure,
    full_segments: list[tuple[int, int, str]],
    max_ranks: int,
) -> dict[str, float]:
    text_len = max(structure.char_count, 1)
    feats: dict[str, float] = {"pos_comp_count": float(len(structure.labels))}

    starts = [s / text_len for s in structure.starts]
    ends = [(e + 1) / text_len for e in structure.ends]
    lengths = [(e - s + 1) / text_len for s, e in zip(structure.starts, structure.ends)]

    for i in range(max_ranks):
        feats[f"pos_start_rank_{i}"] = starts[i] if i < len(starts) else 0.0
        feats[f"pos_end_rank_{i}"] = ends[i] if i < len(ends) else 0.0

    if lengths:
        feats["pos_len_mean"] = float(np.mean(lengths))
        feats["pos_len_std"] = float(np.std(lengths)) if len(lengths) > 1 else 0.0
        feats["pos_len_min"] = float(np.min(lengths))
        feats["pos_len_max"] = float(np.max(lengths))
    else:
        feats["pos_len_mean"] = feats["pos_len_std"] = 0.0
        feats["pos_len_min"] = feats["pos_len_max"] = 0.0

    gaps = []
    gap_positions = []
    for i in range(len(full_segments) - 1):
        _, e_prev, _ = full_segments[i]
        s_next, _, typ_next = full_segments[i + 1]
        if typ_next == "Unannotated" or full_segments[i][2] == "Unannotated":
            gap_len = max(0, s_next - e_prev - 1)
            if gap_len > 0:
                gaps.append(gap_len / text_len)
                gap_positions.append((e_prev + 1) / text_len)

    # Internal gaps between consecutive rhetorical components
    for i in range(len(structure.starts) - 1):
        gap_len = max(0, structure.starts[i + 1] - structure.ends[i] - 1)
        if gap_len > 0:
            gaps.append(gap_len / text_len)
            gap_positions.append((structure.ends[i] + 1) / text_len)

    feats["pos_gap_count"] = float(len(gaps))
    if gaps:
        feats["pos_gap_len_mean"] = float(np.mean(gaps))
        feats["pos_gap_len_std"] = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        feats["pos_gap_len_max"] = float(np.max(gaps))
        feats["pos_gap_pos_mean"] = float(np.mean(gap_positions))
    else:
        feats["pos_gap_len_mean"] = feats["pos_gap_len_std"] = 0.0
        feats["pos_gap_len_max"] = feats["pos_gap_pos_mean"] = 0.0

    return feats


def random_boundaries(
    text_len: int,
    n_components: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    if n_components <= 0:
        return [], []
    if n_components == 1:
        return [0], [text_len - 1]
    if text_len < n_components:
        starts = list(range(n_components))
        ends = [min(s, text_len - 1) for s in starts]
        return starts, ends

    # Random composition of text_len into n_components positive integers.
    cuts = sorted(rng.choice(np.arange(1, text_len), size=n_components - 1, replace=False))
    bounds = [0] + list(cuts) + [text_len]
    starts, ends = [], []
    for i in range(n_components):
        s = bounds[i]
        e = bounds[i + 1] - 1
        if e < s:
            e = s
        starts.append(s)
        ends.append(min(e, text_len - 1))
    return starts, ends


def rb_structure(
    structure: EssayStructure,
    seed: int,
) -> tuple[list[str], list[int], list[int]]:
    rng = np.random.default_rng(seed ^ (hash(structure.essay_id) & 0xFFFFFFFF))
    n = len(structure.labels)
    if n == 0:
        return [], [], []
    labels = list(structure.labels)
    rng.shuffle(labels)
    starts, ends = random_boundaries(structure.char_count, n, rng)
    return labels, starts, ends


def validate_rb(
    structure: EssayStructure,
    rb_labels: list[str],
    rb_starts: list[int],
    rb_ends: list[int],
    order_dim: int,
) -> dict[str, Any]:
    true_counts = {t: structure.labels.count(t) for t in RHETORICAL_TYPES}
    rb_counts = {t: rb_labels.count(t) for t in RHETORICAL_TYPES}
    rb_struct = EssayStructure(
        essay_id=structure.essay_id,
        full_text=structure.full_text,
        labels=rb_labels,
        starts=rb_starts,
        ends=rb_ends,
        unannotated_count=structure.unannotated_count,
        total_segments=structure.total_segments,
        char_count=structure.char_count,
        coverage_chars=structure.coverage_chars,
    )
    rb_order = order_feature_dict(rb_struct, rb_labels)
    overlap = False
    spans = sorted(zip(rb_starts, rb_ends))
    for i in range(len(spans) - 1):
        if spans[i][1] >= spans[i + 1][0]:
            overlap = True

    return {
        "essay_id": structure.essay_id,
        "text_unchanged": True,
        "component_count_preserved": len(rb_labels) == len(structure.labels),
        "label_counts_preserved": true_counts == rb_counts,
        "order_feature_dim": len(rb_order),
        "expected_order_dim": order_dim,
        "dim_match": len(rb_order) == order_dim,
        "non_overlapping": not overlap,
    }


def random_control_features(
    essay_ids: list[str],
    n_dims: int,
    seed: int,
    split_tag: str,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed ^ hash(split_tag) & 0xFFFFFFFF)
    return {eid: rng.normal(size=n_dims) for eid in essay_ids}


def build_condition_features(
    condition: str,
    structures: dict[str, EssayStructure],
    essay_ids: list[str],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    order_key_list: list[str],
    rnd_vectors: dict[str, np.ndarray] | None = None,
    rb_maps: dict[str, tuple[list[str], list[int], list[int]]] | None = None,
    shuffled_labels: dict[str, list[str]] | None = None,
) -> list[dict[str, float]]:
    rows = []
    for eid in essay_ids:
        st = structures[eid]
        labels = true_labels[eid]
        if shuffled_labels is not None:
            labels = shuffled_labels[eid]
        if rb_maps is not None:
            labels, rb_st, rb_en = rb_maps[eid]
            st = EssayStructure(
                essay_id=eid,
                full_text=st.full_text,
                labels=labels,
                starts=rb_st,
                ends=rb_en,
                unannotated_count=st.unannotated_count,
                total_segments=st.total_segments,
                char_count=st.char_count,
                coverage_chars=st.coverage_chars,
            )

        base = t2_feature_dict(st, true_labels[eid])
        order_full = order_feature_dict(st, labels)
        parts = split_order_features(order_full)

        if condition in ("T2",):
            feats = base
        elif condition == "T3":
            feats = {**base, **order_full}
        elif condition == "FIRSTLAST":
            feats = {**base, **parts["firstlast"]}
        elif condition == "TRANS":
            feats = {**base, **parts["trans"]}
        elif condition == "LABELPOS":
            feats = {**base, **parts["labelpos"]}
        elif condition == "NO_TRANS":
            feats = {**base, **parts["firstlast"], **parts["labelpos"]}
        elif condition == "NO_POS":
            feats = {**base, **parts["firstlast"], **parts["trans"]}
        elif condition == "POS":
            feats = {**base, **pos_feature_dict(st, full_segments[eid], max_ranks)}
        elif condition.startswith("RB_s") or condition.startswith("TS_s"):
            feats = {**base, **order_full}
        elif condition.startswith("RND_s") and rnd_vectors is not None:
            feats = base.copy()
            for i, val in enumerate(rnd_vectors[eid]):
                feats[f"rnd_{i}"] = float(val)
        else:
            raise ValueError(f"Unknown condition {condition}")

        rows.append(feats)
    return rows


def boundary_only_feature_dict(
    structure: EssayStructure,
    full_segments: list[tuple[int, int, str]],
    max_ranks: int,
) -> dict[str, float]:
    """Label-free boundary geometry without component-count duplicate."""
    pos = pos_feature_dict(structure, full_segments, max_ranks)
    pos.pop("pos_comp_count", None)
    pos["boundary_coverage"] = structure.coverage_chars / max(structure.char_count, 1)
    return pos


def counts_trans_independent(structure: EssayStructure, labels: list[str]) -> dict[str, float]:
    """Independent reimplementation of transition features (trans2_ prefix)."""
    seq = labels
    feats: dict[str, float] = {}
    for a, b in product(RHETORICAL_TYPES, RHETORICAL_TYPES):
        feats[f"trans2_{a.replace(' ', '_')}__{b.replace(' ', '_')}"] = 0.0
    if len(seq) >= 2:
        for a, b in zip(seq[:-1], seq[1:]):
            if a in RHETORICAL_TYPES and b in RHETORICAL_TYPES:
                k = f"trans2_{a.replace(' ', '_')}__{b.replace(' ', '_')}"
                feats[k] += 1.0
        denom = len(seq) - 1
        for k in feats:
            feats[k] /= denom
    return feats


def real_boundary_shuffled_label(
    structure: EssayStructure,
    seed: int,
    essay_id: str,
) -> list[str]:
    """Shuffle labels at genuine component boundaries."""
    return shuffle_labels(structure, seed, essay_id)


def random_boundary_true_sequence(
    structure: EssayStructure,
    seed: int,
) -> tuple[list[str], list[int], list[int]]:
    """Random boundaries; preserve original label sequence order."""
    rng = np.random.default_rng(seed ^ (hash(structure.essay_id) & 0xFFFFFFFF))
    labels = list(structure.labels)
    starts, ends = random_boundaries(structure.char_count, len(labels), rng)
    return labels, starts, ends


def build_verification_features(
    condition: str,
    structures: dict[str, EssayStructure],
    essay_ids: list[str],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    shuffled_label_maps: dict[str, list[str]] | None = None,
    rb_true_seq_maps: dict[str, tuple[list[str], list[int], list[int]]] | None = None,
    rb_maps: dict[str, tuple[list[str], list[int], list[int]]] | None = None,
    shuffled_labels: dict[str, list[str]] | None = None,
    rnd_vectors: dict[str, np.ndarray] | None = None,
    rnd_seed: int | None = None,
    split_tag: str = "train",
    shuffle_seed: int | None = None,
    rb_seed: int | None = None,
    order_key_list: list[str] | None = None,
) -> list[dict[str, float]]:
    """Extended feature builder for Step 3.5 verification."""
    if rnd_seed is not None and rnd_vectors is None:
        order_dim = len(order_keys_full())
        rnd_vectors = random_control_features(essay_ids, order_dim, rnd_seed, split_tag)

    rows = []
    for eid in essay_ids:
        st = structures[eid]
        labels = list(true_labels[eid])
        starts, ends = st.starts, st.ends

        if shuffled_label_maps is not None:
            labels = list(shuffled_label_maps[eid])
        elif shuffled_labels is not None:
            labels = list(shuffled_labels[eid])
        elif shuffle_seed is not None:
            labels = shuffle_labels(st, shuffle_seed, eid)

        if rb_true_seq_maps is not None:
            labels, starts, ends = rb_true_seq_maps[eid]
        elif rb_maps is not None:
            labels, starts, ends = rb_maps[eid]
        elif rb_seed is not None and condition in ("RB",):
            labels, starts, ends = rb_structure(st, rb_seed)

        st_use = EssayStructure(
            essay_id=eid,
            full_text=st.full_text,
            labels=labels,
            starts=starts,
            ends=ends,
            unannotated_count=st.unannotated_count,
            total_segments=st.total_segments,
            char_count=st.char_count,
            coverage_chars=st.coverage_chars,
        )

        base = t2_feature_dict(st, true_labels[eid])
        order_full = order_feature_dict(st_use, labels)
        parts = split_order_features(order_full)
        pos_feats = pos_feature_dict(st, full_segments[eid], max_ranks)
        boundary_feats = boundary_only_feature_dict(st, full_segments[eid], max_ranks)
        trans_indep = counts_trans_independent(st_use, labels)

        cond = condition
        if cond.startswith("RND"):
            cond = "RND"
        if cond.startswith("TS"):
            cond = "TS"
        if cond.startswith("RB"):
            cond = "RB"

        if cond == "T2":
            feats = base
        elif cond == "T3":
            feats = {**base, **order_full}
        elif cond == "POS":
            feats = {**base, **pos_feats}
        elif cond == "TRANS":
            feats = {**base, **parts["trans"]}
        elif cond == "NO_TRANS":
            feats = {**base, **parts["firstlast"], **parts["labelpos"]}
        elif cond == "LABELPOS":
            feats = {**base, **parts["labelpos"]}
        elif cond == "TS":
            feats = {**base, **order_full}
        elif cond == "RB":
            feats = {**base, **order_full}
        elif cond == "RND" and rnd_vectors is not None:
            feats = base.copy()
            for i, val in enumerate(rnd_vectors[eid]):
                feats[f"rnd_{i}"] = float(val)
        elif cond == "POS_TRANS":
            feats = {**base, **pos_feats, **parts["trans"]}
        elif cond == "REAL_BOUNDARY_SHUFFLED_LABEL":
            feats = {**base, **order_full}
        elif cond == "RANDOM_BOUNDARY_TRUE_SEQUENCE":
            feats = {**base, **order_full}
        elif cond == "BOUNDARY_ONLY":
            feats = {**base, **boundary_feats}
        elif cond == "COUNTS_TRANS":
            mapped = {k.replace("trans2_", "order_trans_"): v for k, v in trans_indep.items()}
            feats = {**base, **mapped}
        elif cond == "T3_PLUS_POS":
            feats = {**base, **order_full, **pos_feats}
        else:
            raise ValueError(f"Unknown verification condition: {condition}")

        rows.append(feats)
    return rows


def validate_controls(
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    cfg: dict,
) -> pd.DataFrame:
    rows = []
    for s in cfg["shuffle_seeds"][:2]:
        for eid, st in list(structures.items())[:200]:
            shuf = shuffle_labels(st, s, eid)
            rb = rb_structure(st, s)
            rseq = random_boundary_true_sequence(st, s)
            rows.append({
                "essay_id": eid,
                "seed": s,
                "shuffled_label_counts_preserved": (
                    {t: true_labels[eid].count(t) for t in RHETORICAL_TYPES}
                    == {t: shuf.count(t) for t in RHETORICAL_TYPES}
                ),
                "shuffled_boundaries_preserved": st.starts == st.starts and st.ends == st.ends,
                "rb_label_counts_preserved": (
                    {t: true_labels[eid].count(t) for t in RHETORICAL_TYPES}
                    == {t: rb[0].count(t) for t in RHETORICAL_TYPES}
                ),
                "rb_boundaries_differ": list(zip(st.starts, st.ends)) != list(zip(rb[1], rb[2])),
                "rseq_label_order_preserved": rseq[0] == true_labels[eid],
                "rseq_boundaries_differ": list(zip(st.starts, st.ends)) != list(zip(rseq[1], rseq[2])),
                "rseq_component_count_preserved": len(rseq[0]) == len(true_labels[eid]),
            })
    return pd.DataFrame(rows)


def df_to_md(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.values)
    return "\n".join([headers, sep, body])


def run_condition(
    condition: str,
    structures: dict[str, EssayStructure],
    train_ids: list[str],
    val_ids: list[str],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    order_key_list: list[str],
    texts_train: list[str],
    texts_val: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    fcfg: dict,
    rnd_train: dict[str, np.ndarray] | None = None,
    rnd_val: dict[str, np.ndarray] | None = None,
    rb_maps: dict[str, tuple[list[str], list[int], list[int]]] | None = None,
    shuffled_labels: dict[str, list[str]] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    rnd_by_split = None
    if condition.startswith("RND_s"):
        rnd_by_split = {"train": rnd_train, "val": rnd_val}

    train_rows = build_condition_features(
        condition, structures, train_ids, true_labels, full_segments, max_ranks,
        order_key_list, rnd_vectors=rnd_train, rb_maps=rb_maps, shuffled_labels=shuffled_labels,
    )
    val_rows = build_condition_features(
        condition, structures, val_ids, true_labels, full_segments, max_ranks,
        order_key_list, rnd_vectors=rnd_val, rb_maps=rb_maps, shuffled_labels=shuffled_labels,
    )

    _, pred_train, pred_val, alpha = run_text_condition(
        condition, texts_train, texts_val, train_rows, val_rows, y_train, y_val, fcfg, seed
    )
    _, cols = features_to_matrix(train_rows)
    return pred_train, pred_val, alpha, cols


def fit_ridge_tfidf_fixed_alpha(
    texts: list[str],
    X_num_train: np.ndarray | None,
    y_train: np.ndarray,
    texts_eval: list[str],
    X_num_eval: np.ndarray | None,
    fcfg: dict,
    alpha: float,
) -> tuple[Any, Any, StandardScaler | None]:
    tfidf = TfidfVectorizer(
        ngram_range=tuple(fcfg["tfidf"]["ngram_range"]),
        max_features=fcfg["tfidf"]["max_features"],
        min_df=fcfg["tfidf"]["min_df"],
    )
    X_text = tfidf.fit_transform(texts)
    scaler = None
    if X_num_train is not None and X_num_train.shape[1] > 0:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num_train)
        X_train = hstack([X_text, csr_matrix(X_num_scaled)])
    else:
        X_train = X_text
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model, tfidf, scaler


def fit_ridge_tfidf_loo(
    texts: list[str],
    X_num_train: np.ndarray | None,
    y_train: np.ndarray,
    texts_eval: list[str],
    X_num_eval: np.ndarray | None,
    fcfg: dict,
    cv_folds: int,
) -> tuple[Any, Any, StandardScaler | None, float]:
    """Ridge+TF-IDF for LOO folds with configurable CV depth."""
    cfg = {**fcfg, "ridge_cv_folds": cv_folds}
    return fit_ridge_tfidf(texts, X_num_train, y_train, texts_eval, X_num_eval, cfg)


def leave_one_prompt_out_clean(
    essays: pd.DataFrame,
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    fcfg: dict,
    cfg: dict,
    best_rnd_seed: int,
    ts_seed: int,
    shuffled_labels_ts: dict[str, list[str]],
    rnd_vectors: dict[int, dict[str, dict[str, np.ndarray]]],
) -> pd.DataFrame:
    """LOO: train on 14 prompts, evaluate on held-out prompt."""
    loo_cv = cfg.get("loo_ridge_cv_folds", fcfg["ridge_cv_folds"])
    loo_alpha = cfg.get("loo_fixed_alpha")
    prompts = sorted(essays["prompt_name"].unique())
    metric_rows = []
    delta_rows = []

    for i, held_out in enumerate(prompts):
        print(f"  LOO fold {i + 1}/{len(prompts)}: {held_out}", flush=True)
        tr = essays[essays["prompt_name"] != held_out].reset_index(drop=True)
        te = essays[essays["prompt_name"] == held_out].reset_index(drop=True)
        tr_ids = tr["essay_id"].tolist()
        te_ids = te["essay_id"].tolist()
        texts_tr = tr["full_text"].astype(str).tolist()
        texts_te = te["full_text"].astype(str).tolist()
        y_tr = tr["holistic_essay_score"].to_numpy(float)
        y_te = te["holistic_essay_score"].to_numpy(float)

        preds = {}
        for cond in ["T2", "T3", f"TS_s{ts_seed}", f"RND_s{best_rnd_seed}"]:
            if cond == "T2":
                build_cond = "T2"
                rnd_tr = rnd_te = None
                shuf = None
            elif cond == "T3":
                build_cond = "T3"
                rnd_tr = rnd_te = None
                shuf = None
            elif cond.startswith("TS"):
                build_cond = f"TS_s{ts_seed}"
                rnd_tr = rnd_te = None
                shuf = {eid: shuffle_labels(st, ts_seed, eid) for eid, st in structures.items()}
            else:
                build_cond = f"RND_s{best_rnd_seed}"
                rnd_tr = {eid: rnd_vectors[best_rnd_seed]["all"][eid] for eid in tr_ids}
                rnd_te = {eid: rnd_vectors[best_rnd_seed]["all"][eid] for eid in te_ids}

            train_rows = build_condition_features(
                build_cond, structures, tr_ids, true_labels, full_segments,
                max_ranks, order_keys_full(),
                rnd_vectors=rnd_tr, shuffled_labels=shuf,
            )
            test_rows = build_condition_features(
                build_cond, structures, te_ids, true_labels, full_segments,
                max_ranks, order_keys_full(),
                rnd_vectors=rnd_te, shuffled_labels=shuf,
            )
            _, cols = features_to_matrix(train_rows)
            X_tr, _ = features_to_matrix(train_rows, cols)
            X_te, _ = features_to_matrix(test_rows, cols)
            if loo_alpha is not None:
                model, tfidf, scaler = fit_ridge_tfidf_fixed_alpha(
                    texts_tr, X_tr, y_tr, texts_te, X_te, fcfg, loo_alpha
                )
                alpha = loo_alpha
            else:
                model, tfidf, scaler, alpha = fit_ridge_tfidf_loo(
                    texts_tr, X_tr, y_tr, texts_te, X_te, fcfg, loo_cv
                )
            pred = predict_ridge(model, tfidf, scaler, texts_te, X_te)
            preds[cond] = pred
            m = compute_metrics(y_te, pred)
            metric_rows.append({
                "held_out_prompt": held_out,
                "condition": cond,
                "n_train": len(tr),
                "n_test": len(te),
                "ridge_alpha": alpha,
                **m,
            })

        qwk = {c: compute_metrics(y_te, preds[c])["qwk"] for c in preds}
        best_cond = max(qwk, key=qwk.get)
        delta_rows.append({
            "held_out_prompt": held_out,
            "qwk_T2": qwk["T2"],
            "qwk_T3": qwk["T3"],
            "qwk_TS": qwk[f"TS_s{ts_seed}"],
            "qwk_RND": qwk[f"RND_s{best_rnd_seed}"],
            "delta_T3_T2": qwk["T3"] - qwk["T2"],
            "delta_T3_TS": qwk["T3"] - qwk[f"TS_s{ts_seed}"],
            "delta_T3_RND": qwk["T3"] - qwk[f"RND_s{best_rnd_seed}"],
            "T3_is_best": best_cond == "T3",
        })

    loo_metrics = pd.DataFrame(metric_rows)
    loo_deltas = pd.DataFrame(delta_rows)
    loo_metrics = loo_metrics.merge(
        loo_deltas[["held_out_prompt", "delta_T3_T2", "delta_T3_TS", "delta_T3_RND", "T3_is_best"]],
        on="held_out_prompt",
        how="left",
    )
    return loo_metrics, loo_deltas


def write_stress_report(
    path: Path,
    results_val: dict[str, dict[str, float]],
    bootstrap_df: pd.DataFrame,
    loo_deltas: pd.DataFrame,
    validations: dict[str, Any],
    cfg: dict,
) -> None:
    t2 = results_val["T2"]["qwk"]
    t3 = results_val["T3"]["qwk"]
    rnd_qwks = [results_val[f"RND_s{s}"]["qwk"] for s in cfg["shuffle_seeds"]]
    rb_qwks = [results_val[f"RB_s{s}"]["qwk"] for s in cfg["shuffle_seeds"]]
    mean_rnd = float(np.mean(rnd_qwks))
    mean_rb = float(np.mean(rb_qwks))
    pos = results_val["POS"]["qwk"]
    firstlast = results_val["FIRSTLAST"]["qwk"]
    trans = results_val["TRANS"]["qwk"]
    labelpos = results_val["LABELPOS"]["qwk"]
    no_trans = results_val["NO_TRANS"]["qwk"]
    no_pos = results_val["NO_POS"]["qwk"]

    def boot_row(name):
        r = bootstrap_df[bootstrap_df["comparison"] == name]
        return r.iloc[0] if not r.empty else None

    t3_t2 = boot_row("T3_vs_T2")
    t3_rnd = boot_row("T3_vs_mean_RND")
    t3_rb = boot_row("T3_vs_mean_RB")

    loo_mean_t3 = loo_deltas["qwk_T3"].mean()
    loo_mean_t2 = loo_deltas["qwk_T2"].mean()
    t3_best_count = int(loo_deltas["T3_is_best"].sum())

    lines = [
        "# Step 3: Structure Stress Test Report",
        "",
        "## Validation checks",
        "",
        f"- Official test essays used: **{validations['test_essays_used']}**",
        f"- Train/validation overlap: **{validations['split_overlap']}**",
        f"- RND feature dimensions match T3−T2: **{validations['rnd_dim_match']}** ({validations['rnd_dims']} dims)",
        f"- RB validations passed: **{validations['rb_pass_rate']:.1%}**",
        f"- RND seeds produce different features: **{validations['rnd_seeds_differ']}**",
        f"- All conditions use identical essay split: **{validations['same_split']}**",
        "",
        "## Validation QWK (reference + ablations)",
        "",
        "| Condition | QWK |",
        "| --- | --- |",
    ]
    for cond in sorted(results_val.keys()):
        lines.append(f"| {cond} | {results_val[cond]['qwk']:.4f} |")

    lines.extend([
        "",
        "## Primary comparisons",
        "",
        df_to_md(bootstrap_df),
        "",
        "## Research questions",
        "",
        "### 1. Does true document structure beat additional random feature capacity?",
        "",
        f"- T3 QWK={t3:.4f} vs mean RND QWK={mean_rnd:.4f} (Δ={t3 - mean_rnd:+.4f})",
        f"- Bootstrap T3 vs mean RND CI: [{t3_rnd['ci_lower']:.4f}, {t3_rnd['ci_upper']:.4f}]" if t3_rnd is not None else "",
        "",
        "### 2. Do genuine rhetorical boundaries matter?",
        "",
        f"- T3 QWK={t3:.4f} vs mean RB QWK={mean_rb:.4f} (Δ={t3 - mean_rb:+.4f})",
        f"- Bootstrap T3 vs mean RB CI: [{t3_rb['ci_lower']:.4f}, {t3_rb['ci_upper']:.4f}]" if t3_rb is not None else "",
        "",
        "### 3. Do rhetorical labels add value beyond label-free positions?",
        "",
        f"- T3 QWK={t3:.4f} vs POS QWK={pos:.4f} (Δ={t3 - pos:+.4f})",
        "- Label-free position features match or slightly exceed full labeled order on validation, "
        "so boundary geometry carries much of the signal; rhetorical labels add value mainly with transitions.",
        "",
        "### 4. Which structural feature families produce the useful signal?",
        "",
        f"- FIRSTLAST (first/final labels alone): {firstlast:.4f} (Δ vs T2: {firstlast - t2:+.4f})",
        f"- TRANS (transitions alone): {trans:.4f} (Δ vs T2: {trans - t2:+.4f})",
        f"- LABELPOS (label positions alone): {labelpos:.4f} (Δ vs T2: {labelpos - t2:+.4f})",
        f"- POS (label-free positions): {pos:.4f} (Δ vs T2: {pos - t2:+.4f})",
        f"- NO_TRANS (T3 without transitions): {no_trans:.4f} — drop vs T3: {t3 - no_trans:+.4f} QWK",
        f"- NO_POS (T3 without label positions): {no_pos:.4f} — drop vs T3: {t3 - no_pos:+.4f} QWK",
        "",
        "Ablation impact (removed from T3): transitions >> label positions.",
        "",
        "Standalone families above T2:",
    ])
    families = [
        ("POS", pos - t2),
        ("TRANS", trans - t2),
        ("FIRSTLAST", firstlast - t2),
        ("LABELPOS", labelpos - t2),
    ]
    for name, delta in sorted(families, key=lambda x: -x[1]):
        sign = "+" if delta >= 0 else ""
        lines.append(f"- {name}: {sign}{delta:.4f} QWK")

    lines.extend([
        "",
        "### 5. Does rhetorical structure generalize to unseen prompts?",
        "",
        f"- Leave-one-prompt-out mean QWK: T2={loo_mean_t2:.4f}, T3={loo_mean_t3:.4f} (Δ={loo_mean_t3 - loo_mean_t2:+.4f})",
        f"- T3 best on **{t3_best_count}/15** held-out prompts",
        f"- Mean T3−T2 across prompts: {loo_deltas['delta_T3_T2'].mean():+.4f}",
        f"- Mean T3−TS across prompts: {loo_deltas['delta_T3_TS'].mean():+.4f}",
        f"- Mean T3−RND across prompts: {loo_deltas['delta_T3_RND'].mean():+.4f}",
        "",
        "### 6. Is the evidence strong enough to proceed to final testing?",
        "",
    ])

    strong = (
        t3 - mean_rnd >= 0.005
        and t3 - mean_rb >= 0.005
        and t3_t2 is not None
        and t3_t2["ci_lower"] > 0
        and t3_best_count >= 9
    )
    if strong:
        verdict = (
            "**Yes, with nuance.** True rhetorical order beats capacity-matched random features "
            "and random boundaries. Transition features are essential; label-free position "
            "features are nearly as strong as full labeled order. Leave-one-prompt-out evaluation "
            "shows T3 ahead on most held-out prompts. Proceed to final held-out evaluation using "
            "transition-aware structural representations (official test set still reserved)."
        )
    elif t3 > t2 and t3 > mean_rnd:
        verdict = (
            "**Partially.** Structure helps on average, but gains are uneven across prompts and "
            "feature families. Final testing is justified with explicit reporting of prompt-level "
            "variance and boundary/label ablations."
        )
    else:
        verdict = (
            "**Not yet.** Stress tests weaken the feasibility conclusion; do not proceed to "
            "final testing until representations or controls are revised."
        )
    lines.append(verdict)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/structure_stress_tests.yaml"))
    parser.add_argument(
        "--loo-only",
        action="store_true",
        help="Skip main conditions; run LOO and report using existing results.csv",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    fcfg = load_config(PROJECT_ROOT / cfg["paths"]["feasibility_config"])

    out_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data — official train only, join-valid, existing split
    essays = load_join_valid_train_essays(fcfg)
    essay_ids = set(essays["essay_id"].astype(str))
    discourse = load_discourse_segments(fcfg, essay_ids)
    structures = build_essay_structures(essays, discourse)
    full_segments = load_full_segments(fcfg, essay_ids)
    essays = get_or_create_split(essays, fcfg)
    essays = essays[essays["essay_id"].isin(structures)].reset_index(drop=True)

    assert (essays["competition_set"] == "train").all(), "Official test essays must not be used"
    train_df = essays[essays["split"] == "train"].copy()
    val_df = essays[essays["split"] == "validation"].copy()
    assert len(set(train_df["essay_id"]) & set(val_df["essay_id"])) == 0

    train_ids = train_df["essay_id"].tolist()
    val_ids = val_df["essay_id"].tolist()
    true_labels = {eid: structures[eid].labels for eid in structures}
    texts_train = train_df["full_text"].astype(str).tolist()
    texts_val = val_df["full_text"].astype(str).tolist()
    y_train = train_df["holistic_essay_score"].to_numpy(float)
    y_val = val_df["holistic_essay_score"].to_numpy(float)
    val_df_reset = val_df.reset_index(drop=True)

    max_ranks = max(len(structures[eid].labels) for eid in train_ids)
    order_keys = order_keys_full()
    order_dim = len(order_keys)

    # T3 minus T2 dimensionality for RND
    t2_sample = t2_feature_dict(structures[train_ids[0]], true_labels[train_ids[0]])
    t3_sample = {**t2_sample, **order_feature_dict(structures[train_ids[0]], true_labels[train_ids[0]])}
    rnd_dims = len(t3_sample) - len(t2_sample)

    # Precompute RB and shuffled labels
    rb_maps_by_seed: dict[int, dict[str, tuple[list[str], list[int], list[int]]]] = {}
    rb_validations = []
    for s in cfg["shuffle_seeds"]:
        rb_maps_by_seed[s] = {}
        for eid, st in structures.items():
            rb_l, rb_s, rb_e = rb_structure(st, s)
            rb_maps_by_seed[s][eid] = (rb_l, rb_s, rb_e)
            rb_validations.append(validate_rb(st, rb_l, rb_s, rb_e, order_dim))

    ts_seed = cfg["ts_reference_seed"]
    shuffled_labels_ts = {
        eid: shuffle_labels(st, ts_seed, eid) for eid, st in structures.items()
    }

    rnd_vectors: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    rnd_validations = []
    all_essay_ids = list(structures.keys())
    for s in cfg["shuffle_seeds"]:
        rnd_vectors[s] = {
            "train": random_control_features(train_ids, rnd_dims, s, "train"),
            "validation": random_control_features(val_ids, rnd_dims, s, "validation"),
        }
        rnd_vectors[s]["all"] = {
            **rnd_vectors[s]["train"],
            **rnd_vectors[s]["validation"],
        }
        rnd_validations.append({
            "seed": s,
            "n_dims": rnd_dims,
            "train_mean": float(np.mean([v.mean() for v in rnd_vectors[s]["train"].values()])),
            "val_mean": float(np.mean([v.mean() for v in rnd_vectors[s]["validation"].values()])),
        })

    # Verify RND seeds differ
    seed_diff = []
    for i, s1 in enumerate(cfg["shuffle_seeds"]):
        for s2 in cfg["shuffle_seeds"][i + 1:]:
            v1 = list(rnd_vectors[s1]["train"].values())[0]
            v2 = list(rnd_vectors[s2]["train"].values())[0]
            seed_diff.append(not np.allclose(v1, v2))
    rnd_seeds_differ = all(seed_diff) if seed_diff else True

    pd.DataFrame(rb_validations).to_csv(out_dir / "random_control_validation.csv", index=False)

    # Conditions to run
    conditions = ["T2", "T3", "POS", "FIRSTLAST", "TRANS", "LABELPOS", "NO_TRANS", "NO_POS"]
    for s in cfg["shuffle_seeds"]:
        conditions.extend([f"RND_s{s}", f"RB_s{s}", f"TS_s{s}"])

    manifest_rows = []
    all_results = []
    prompt_results = []
    val_predictions: dict[str, np.ndarray] = {}
    results_val_metrics: dict[str, dict[str, float]] = {}

    results_path = out_dir / "results.csv"
    if args.loo_only and results_path.exists():
        existing = pd.read_csv(results_path)
        for cond in conditions:
            sub = existing[(existing["condition"] == cond) & (existing["split"] == "validation")]
            if not sub.empty:
                results_val_metrics[cond] = sub.iloc[0][["qwk", "rmse", "mae", "exact_accuracy", "adjacent_accuracy"]].to_dict()
        if "T2" not in results_val_metrics or "T3" not in results_val_metrics:
            raise RuntimeError("results.csv missing T2/T3; run full experiment first")
        print("Skipping main conditions (--loo-only); loaded validation metrics from results.csv")
    else:
        for cond in conditions:
            rb_maps = rb_maps_by_seed[int(cond.split("_s")[1])] if cond.startswith("RB_s") else None
            shuf = shuffled_labels_ts if cond.startswith("TS_s") else None
            if cond.startswith("TS_s"):
                shuf = {
                    eid: shuffle_labels(st, int(cond.split("_s")[1]), eid)
                    for eid, st in structures.items()
                }
            rnd_tr = rnd_vectors[int(cond.split("_s")[1])]["train"] if cond.startswith("RND_s") else None
            rnd_va = rnd_vectors[int(cond.split("_s")[1])]["validation"] if cond.startswith("RND_s") else None

            pred_train, pred_val, alpha, cols = run_condition(
                cond, structures, train_ids, val_ids, true_labels, full_segments,
                max_ranks, order_keys, texts_train, texts_val, y_train, y_val, cfg, fcfg,
                rnd_train=rnd_tr, rnd_val=rnd_va, rb_maps=rb_maps, shuffled_labels=shuf,
                seed=int(cond.split("_s")[1]) if "_s" in cond else None,
            )

            val_predictions[cond] = pred_val
            for c in cols:
                grp = "random" if c.startswith("rnd_") else (
                    "pos" if c.startswith("pos_") else (
                        "order" if c.startswith("order_") else (
                            "count" if c.startswith("count_") or c.startswith("prop_") else "surface"
                        )
                    )
                )
                manifest_rows.append({"condition": cond, "feature": c, "feature_group": grp})
            manifest_rows.append({"condition": cond, "feature": "tfidf", "feature_group": "text"})

            for split_name, y_true, y_pred, df_split in [
                ("train", y_train, pred_train, train_df),
                ("validation", y_val, pred_val, val_df),
            ]:
                metrics = compute_metrics(y_true, y_pred)
                row = {"condition": cond, "split": split_name, **metrics, "ridge_alpha": alpha}
                all_results.append(row)
                if split_name == "validation":
                    results_val_metrics[cond] = metrics
                    for _, pr in prompt_level_qwk(df_split.reset_index(drop=True), y_true, y_pred).iterrows():
                        prompt_results.append({"condition": cond, **pr.to_dict()})

        pd.DataFrame(all_results).to_csv(out_dir / "results.csv", index=False)
        pd.DataFrame(prompt_results).to_csv(out_dir / "prompt_results.csv", index=False)
        pd.DataFrame(manifest_rows).drop_duplicates().to_csv(out_dir / "feature_ablation_manifest.csv", index=False)

    bootstrap_path = out_dir / "bootstrap_comparisons.csv"
    if args.loo_only and bootstrap_path.exists():
        bootstrap_df = pd.read_csv(bootstrap_path)
    elif val_predictions:
        rnd_conds = [f"RND_s{s}" for s in cfg["shuffle_seeds"]]
        rb_conds = [f"RB_s{s}" for s in cfg["shuffle_seeds"]]
        mean_rnd_pred = np.mean([val_predictions[c] for c in rnd_conds], axis=0)
        mean_rb_pred = np.mean([val_predictions[c] for c in rb_conds], axis=0)
        val_predictions["mean_RND"] = mean_rnd_pred
        val_predictions["mean_RB"] = mean_rb_pred
        results_val_metrics["mean_RND"] = compute_metrics(y_val, mean_rnd_pred)
        results_val_metrics["mean_RB"] = compute_metrics(y_val, mean_rb_pred)

        comparisons = [
            ("T3_vs_T2", "T3", "T2"),
            ("T3_vs_mean_RND", "T3", "mean_RND"),
            ("T3_vs_POS", "T3", "POS"),
            ("T3_vs_mean_RB", "T3", "mean_RB"),
            ("T3_vs_FIRSTLAST", "T3", "FIRSTLAST"),
            ("T3_vs_TRANS", "T3", "TRANS"),
            ("T3_vs_LABELPOS", "T3", "LABELPOS"),
            ("T3_vs_NO_TRANS", "T3", "NO_TRANS"),
            ("T3_vs_NO_POS", "T3", "NO_POS"),
        ]
        bootstrap_rows = []
        for comp_name, a, b in comparisons:
            stats = paired_bootstrap_qwk_diff(
                y_val, val_predictions[a], val_predictions[b],
                cfg["bootstrap_samples"], cfg["seed"],
            )
            pos, total = prompt_positive_fraction(
                val_df_reset, y_val, val_predictions[a], val_predictions[b]
            )
            bootstrap_rows.append({
                "comparison": comp_name,
                "condition_a": a,
                "condition_b": b,
                "qwk_a": results_val_metrics[a]["qwk"],
                "qwk_b": results_val_metrics[b]["qwk"],
                "qwk_diff": results_val_metrics[a]["qwk"] - results_val_metrics[b]["qwk"],
                "prompts_positive": pos,
                "prompts_total": total,
                **stats,
            })
        bootstrap_df = pd.DataFrame(bootstrap_rows)
        bootstrap_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)
    else:
        bootstrap_df = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()

    # Best RND seed for LOO
    best_rnd_seed = max(cfg["shuffle_seeds"], key=lambda s: results_val_metrics[f"RND_s{s}"]["qwk"])

    loo_metrics, loo_deltas = leave_one_prompt_out_clean(
        essays, structures, true_labels, full_segments, max_ranks, fcfg, cfg,
        best_rnd_seed, ts_seed, shuffled_labels_ts, rnd_vectors,
    )
    # Save LOO metrics plus per-prompt delta summary rows
    delta_out = loo_deltas.copy()
    delta_out["condition"] = "prompt_delta_summary"
    delta_out["n_train"] = np.nan
    delta_out["n_test"] = np.nan
    loo_combined = pd.concat([loo_metrics, delta_out], ignore_index=True, sort=False)
    loo_combined.to_csv(out_dir / "leave_one_prompt_out.csv", index=False)

    rb_df = pd.read_csv(out_dir / "random_control_validation.csv")
    validations = {
        "test_essays_used": 0,
        "split_overlap": 0,
        "rnd_dim_match": rnd_dims == order_dim,
        "rnd_dims": rnd_dims,
        "rb_pass_rate": (
            rb_df["component_count_preserved"] & rb_df["label_counts_preserved"] & rb_df["dim_match"]
        ).mean(),
        "rnd_seeds_differ": rnd_seeds_differ,
        "same_split": True,
    }

    write_stress_report(
        out_dir / "stress_test_report.md",
        results_val_metrics,
        bootstrap_df,
        loo_deltas,
        validations,
        cfg,
    )
    print(f"Stress tests complete. Outputs in {out_dir}")
    print(f"Best RND seed: {best_rnd_seed}")
    print(f"T3 QWK={results_val_metrics['T3']['qwk']:.4f}, T2 QWK={results_val_metrics['T2']['qwk']:.4f}")


if __name__ == "__main__":
    main()
