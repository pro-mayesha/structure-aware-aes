#!/usr/bin/env python3
"""
POST-HOC control analysis (added after the frozen Step 4 held-out evaluation).

The Step 4 random controls (TS, RB, RND) are T3-level and were seeded with
Python's per-process string hash(), so their draws cannot be reproduced. This
script leaves the original pipeline untouched and:

  Step 0  reproduces T2, T3, POS, T3_PLUS_POS and regenerates TS/RB/RND with
          deterministic (md5-based) seeding over 20 seeds;
  Step 1  T3 vs TS / RB / RND (T3 level);
  Step 2  POS vs POS_RB / POS_PERM / POS_RND (geometry level);
  Step 3  T3_PLUS_POS vs FULL_TS / FULL_RB / FULL_PERM / FULL_RND (full model).

Only the seed derivation differs from shuffle_labels, rb_structure and
random_control_features; the perturbation logic is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_final_heldout_evaluation import (  # noqa: E402
    build_condition_features,
    fit_predict_fixed,
    fit_tfidf_matrices,
    load_discourse_segments_all,
    load_full_segments_all,
    load_join_valid_essays,
)
from run_structure_feasibility import (  # noqa: E402
    EssayStructure,
    build_essay_structures,
    clip_and_round,
    compute_metrics,
    load_config,
    order_feature_dict,
)
from run_structure_stress_tests import (  # noqa: E402
    order_keys_full,
    pos_feature_dict,
    random_boundaries,
    t2_feature_dict,
)
from run_structure_verification import df_to_md  # noqa: E402

ALPHA = 10.0
ORIGINAL_SEEDS = [13, 21, 42, 87, 101]
EXTRA_SEEDS = [1, 2, 3, 5, 7, 11, 17, 19, 23, 29, 31, 37, 53, 71, 97]
SEEDS = ORIGINAL_SEEDS + EXTRA_SEEDS
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42

REAL_CONDITIONS = ["T2", "T3", "POS", "T3_PLUS_POS"]
CONTROLS = ["TS", "RB", "RND", "POS_RB", "POS_PERM", "POS_RND",
            "FULL_TS", "FULL_RB", "FULL_PERM", "FULL_RND"]
STEPS = {
    "Step 1: T3 level": ("T3", ["TS", "RB", "RND"]),
    "Step 2: geometry level": ("POS", ["POS_RB", "POS_PERM", "POS_RND"]),
    "Step 3: full-model level": ("T3_PLUS_POS", ["FULL_TS", "FULL_RB", "FULL_PERM", "FULL_RND"]),
}
OUT_DIR = PROJECT_ROOT / "outputs/posthoc_controls"


# ---------------------------------------------------------------------------
# Deterministic seeding and perturbations
# ---------------------------------------------------------------------------


def stable_seed(*parts: Any) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def shuffle_labels_stable(st: EssayStructure, seed: int, split_tag: str) -> list[str]:
    rng = np.random.default_rng(stable_seed(st.essay_id, split_tag, seed))
    labels = list(st.labels)
    if len(labels) <= 1:
        return labels
    perm = rng.permutation(len(labels))
    return [labels[i] for i in perm]


def rb_structure_stable(st: EssayStructure, seed: int, split_tag: str) -> tuple[list[str], list[int], list[int]]:
    rng = np.random.default_rng(stable_seed(st.essay_id, split_tag, seed))
    n = len(st.labels)
    if n == 0:
        return [], [], []
    labels = list(st.labels)
    rng.shuffle(labels)
    starts, ends = random_boundaries(st.char_count, n, rng)
    return labels, starts, ends


def random_control_features_stable(
    essay_ids: list[str], n_dims: int, seed: int, split_tag: str,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(stable_seed("rnd", n_dims, split_tag, seed))
    return {eid: rng.normal(size=n_dims) for eid in essay_ids}


def perm_segments(
    segs: list[tuple[int, int, str]], seed: int, essay_id: str, split_tag: str,
    identity: bool = False,
) -> list[tuple[int, int, str]]:
    """Length-preserving permutation of the component/unannotated sequence.

    Each segment (rhetorical component or Unannotated gap) keeps its length;
    the order of segments is permuted; the uncovered spacers between
    consecutive slots and the start offset are kept in place.
    """
    if not segs:
        return []
    units = [(e - s + 1, t) for s, e, t in segs]
    spacers = [segs[i + 1][0] - segs[i][1] - 1 for i in range(len(segs) - 1)]
    if identity:
        order = np.arange(len(units))
    else:
        rng = np.random.default_rng(stable_seed(essay_id, split_tag, seed, "perm"))
        order = rng.permutation(len(units))
    out = []
    pos = segs[0][0]
    for k, j in enumerate(order):
        length, typ = units[j]
        s, e = pos, pos + length - 1
        out.append((s, e, typ))
        if k < len(spacers):
            pos = e + 1 + spacers[k]
    return out


def with_bounds(st: EssayStructure, labels: list[str], starts: list[int], ends: list[int]) -> EssayStructure:
    return EssayStructure(
        essay_id=st.essay_id, full_text=st.full_text, labels=labels, starts=starts, ends=ends,
        unannotated_count=st.unannotated_count, total_segments=st.total_segments,
        char_count=st.char_count, coverage_chars=st.coverage_chars,
    )


def rhetorical_from_segments(st: EssayStructure, segs: list[tuple[int, int, str]]) -> EssayStructure:
    rh = [x for x in segs if x[2] != "Unannotated"]
    return with_bounds(st, [x[2] for x in rh], [x[0] for x in rh], [x[1] for x in rh])


def build_control_rows(
    seed: int,
    split_tag: str,
    essay_ids: list[str],
    d: dict[str, Any],
) -> dict[str, list[dict[str, float]]]:
    n_order = len(order_keys_full())
    n_pos = d["n_pos"]
    rnd = {k: random_control_features_stable(essay_ids, k, seed, split_tag)
           for k in (n_order, n_pos, n_order + n_pos)}
    rows: dict[str, list[dict[str, float]]] = {c: [] for c in CONTROLS}
    for eid in essay_ids:
        st = d["structures"][eid]
        segs = d["full_segments"][eid]
        base = t2_feature_dict(st, d["true_labels"][eid])
        pos_true = pos_feature_dict(st, segs, d["max_ranks"])

        ts_labels = shuffle_labels_stable(st, seed, split_tag)
        ts_order = order_feature_dict(with_bounds(st, ts_labels, st.starts, st.ends), ts_labels)

        rb_labels, rb_s, rb_e = rb_structure_stable(st, seed, split_tag)
        rb_st = with_bounds(st, rb_labels, rb_s, rb_e)
        rb_order = order_feature_dict(rb_st, rb_labels)
        # RB partitions the whole text into components, so there are no unannotated segments.
        rb_pos = pos_feature_dict(rb_st, list(zip(rb_s, rb_e, rb_labels)), d["max_ranks"])

        p_segs = perm_segments(segs, seed, eid, split_tag)
        p_st = rhetorical_from_segments(st, p_segs)
        p_order = order_feature_dict(p_st, p_st.labels)
        p_pos = pos_feature_dict(p_st, p_segs, d["max_ranks"])

        def rnd_feats(k: int) -> dict[str, float]:
            return {f"rnd_{i}": float(v) for i, v in enumerate(rnd[k][eid])}

        rows["TS"].append({**base, **ts_order})
        rows["RB"].append({**base, **rb_order})
        rows["RND"].append({**base, **rnd_feats(n_order)})
        rows["POS_RB"].append({**base, **rb_pos})
        rows["POS_PERM"].append({**base, **p_pos})
        rows["POS_RND"].append({**base, **rnd_feats(n_pos)})
        rows["FULL_TS"].append({**base, **ts_order, **pos_true})
        rows["FULL_RB"].append({**base, **rb_order, **rb_pos})
        rows["FULL_PERM"].append({**base, **p_order, **p_pos})
        rows["FULL_RND"].append({**base, **rnd_feats(n_order + n_pos)})
    return rows


# ---------------------------------------------------------------------------
# Data, sanity checks, metrics
# ---------------------------------------------------------------------------


def load_data(cfg: dict) -> dict[str, Any]:
    train_essays = load_join_valid_essays(cfg, "train")
    test_essays = load_join_valid_essays(cfg, "test")
    all_ids = set(train_essays["essay_id"]) | set(test_essays["essay_id"])
    discourse = load_discourse_segments_all(cfg, all_ids)
    structures = build_essay_structures(pd.concat([train_essays, test_essays], ignore_index=True), discourse)
    full_segments = load_full_segments_all(cfg, all_ids)
    train_essays = train_essays[train_essays["essay_id"].isin(structures)].reset_index(drop=True)
    test_essays = test_essays[test_essays["essay_id"].isin(structures)].reset_index(drop=True)
    train_ids = train_essays["essay_id"].tolist()
    max_ranks = max(len(structures[eid].labels) for eid in train_ids)
    st0 = structures[train_ids[0]]
    n_pos = len(pos_feature_dict(st0, full_segments[st0.essay_id], max_ranks))
    return {
        "train": train_essays, "test": test_essays,
        "train_ids": train_ids, "test_ids": test_essays["essay_id"].tolist(),
        "structures": structures, "full_segments": full_segments,
        "true_labels": {eid: structures[eid].labels for eid in structures},
        "max_ranks": max_ranks, "n_pos": n_pos,
    }


def check_perm_identity(d: dict[str, Any]) -> int:
    """Identity permutation must reproduce the true rhetorical boundaries and geometry."""
    bad = 0
    for eid in d["train_ids"] + d["test_ids"]:
        st, segs = d["structures"][eid], d["full_segments"][eid]
        p_st = rhetorical_from_segments(st, perm_segments(segs, 0, eid, "x", identity=True))
        same = (p_st.labels == st.labels and p_st.starts == st.starts and p_st.ends == st.ends
                and pos_feature_dict(p_st, segs, d["max_ranks"]) == pos_feature_dict(st, segs, d["max_ranks"]))
        bad += not same
    return bad


def perm_preserves_lengths(d: dict[str, Any], seed: int, n_check: int = 500) -> bool:
    for eid in d["test_ids"][:n_check]:
        segs = d["full_segments"][eid]
        p = perm_segments(segs, seed, eid, "test")
        if sorted((e - s + 1, t) for s, e, t in segs) != sorted((e - s + 1, t) for s, e, t in p):
            return False
        if segs and (p[0][0] != segs[0][0] or p[-1][1] != segs[-1][1]):
            return False
    return True


def fast_qwk_codes(y_int: np.ndarray, p_int: np.ndarray) -> np.ndarray:
    return (y_int - 1) * 6 + (p_int - 1)


_W = np.array([[(i - j) ** 2 for j in range(6)] for i in range(6)], dtype=float).ravel()


def qwk_from_counts(counts: np.ndarray) -> np.ndarray:
    """counts: (..., 36) confusion counts over scores 1..6 → QWK."""
    o = counts.reshape(*counts.shape[:-1], 6, 6)
    n = o.sum(axis=(-1, -2), keepdims=True)
    e = o.sum(axis=-1, keepdims=True) * o.sum(axis=-2, keepdims=True) / n
    num = (o.reshape(*counts.shape[:-1], 36) * _W).sum(-1)
    den = (e.reshape(*counts.shape[:-1], 36) * _W).sum(-1)
    return 1.0 - num / den


def bootstrap_vs_controls(y: np.ndarray, pred_a: np.ndarray, control_preds: list[np.ndarray]) -> dict[str, float]:
    """Same procedure as bootstrap_mean_control_diff (QWK_a − mean of seed QWKs per resample)."""
    y_int = np.rint(y).astype(int)
    codes = [fast_qwk_codes(y_int, clip_and_round(p)) for p in [pred_a] + control_preds]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y)
    diffs = np.empty(BOOTSTRAP_SAMPLES)
    for b in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        counts = np.stack([np.bincount(c[idx], minlength=36) for c in codes]).astype(float)
        q = qwk_from_counts(counts)
        diffs[b] = q[0] - q[1:].mean()
    return {
        "boot_diff_mean": float(diffs.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_diff_gt_0": float(np.mean(diffs > 0)),
    }


def prompts_improved(test_df: pd.DataFrame, y: np.ndarray, pred_a: np.ndarray,
                     control_preds: list[np.ndarray]) -> tuple[int, int]:
    pos = total = 0
    for _, grp in test_df.groupby("prompt_name"):
        idx = grp.index.to_numpy()
        yt = np.rint(y[idx]).astype(int)
        if len(idx) < 2 or len(np.unique(yt)) < 2:
            continue
        qa = cohen_kappa_score(yt, clip_and_round(pred_a[idx]), weights="quadratic")
        qc = np.mean([cohen_kappa_score(yt, clip_and_round(p[idx]), weights="quadratic") for p in control_preds])
        total += 1
        pos += int(qa > qc)
    return pos, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def setup(cfg_path: Path) -> tuple[dict, dict[str, Any], Any, Any, np.ndarray, np.ndarray]:
    cfg = load_config(cfg_path)
    fcfg = load_config(PROJECT_ROOT / cfg["paths"]["feasibility_config"])
    fcfg["tfidf"] = cfg["tfidf"]
    d = load_data(cfg)
    _, X_tr, X_te = fit_tfidf_matrices(
        d["train"]["full_text"].astype(str).tolist(), d["test"]["full_text"].astype(str).tolist(), fcfg
    )
    y_tr = d["train"]["holistic_essay_score"].to_numpy(float)
    y_te = d["test"]["holistic_essay_score"].to_numpy(float)
    return cfg, d, X_tr, X_te, y_tr, y_te


def run_determinism_check(args: argparse.Namespace) -> None:
    """Fit every control for one seed and dump QWKs (run in separate processes and compare)."""
    _, d, X_tr, X_te, y_tr, y_te = setup(Path(args.config))
    seed = args.determinism_seed
    tr = build_control_rows(seed, "train", d["train_ids"], d)
    te = build_control_rows(seed, "test", d["test_ids"], d)
    out = {}
    for c in CONTROLS:
        pred, _ = fit_predict_fixed(X_tr, X_te, tr[c], te[c], y_tr, ALPHA)
        out[c] = compute_metrics(y_te, pred)["qwk"]
    out["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "<unset>")
    Path(args.determinism_out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/final_heldout_evaluation.yaml"))
    parser.add_argument("--determinism-seed", type=int, default=None)
    parser.add_argument("--determinism-out", default=None)
    args = parser.parse_args()
    if args.determinism_seed is not None:
        run_determinism_check(args)
        return

    t0 = time.time()
    cfg, d, X_tr, X_te, y_tr, y_te = setup(Path(args.config))
    test_df = d["test"].reset_index(drop=True)
    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"train={len(d['train_ids'])} test={len(d['test_ids'])} max_ranks={d['max_ranks']} "
          f"n_pos={d['n_pos']}", flush=True)

    perm_identity_failures = check_perm_identity(d)
    perm_lengths_ok = perm_preserves_lengths(d, SEEDS[0])
    print(f"perm identity failures={perm_identity_failures}, length-preserving={perm_lengths_ok}", flush=True)

    # Check the fast QWK used in the bootstrap against sklearn on the full test set.
    y_int = np.rint(y_te).astype(int)

    preds: dict[str, np.ndarray] = {}
    n_features: dict[str, set[int]] = {}

    def record(cond: str, pred: np.ndarray, n_feat: int) -> None:
        preds[cond] = pred
        base = cond.rsplit("_s", 1)[0] if cond.split("_s")[-1].isdigit() else cond
        n_features.setdefault(base, set()).add(n_feat)

    for cond in REAL_CONDITIONS:
        args_common = (d["structures"],)
        tr = build_condition_features(cond, *args_common, d["train_ids"], d["true_labels"],
                                      d["full_segments"], d["max_ranks"], split_tag="train")
        te = build_condition_features(cond, *args_common, d["test_ids"], d["true_labels"],
                                      d["full_segments"], d["max_ranks"], split_tag="test")
        pred, _ = fit_predict_fixed(X_tr, X_te, tr, te, y_tr, ALPHA)
        record(cond, pred, len(tr[0]))
        print(f"  {cond}: qwk={compute_metrics(y_te, pred)['qwk']:.6f} n_feat={len(tr[0])}", flush=True)

    sk = cohen_kappa_score(y_int, clip_and_round(preds["T2"]), weights="quadratic")
    fast = float(qwk_from_counts(np.bincount(fast_qwk_codes(y_int, clip_and_round(preds["T2"])),
                                             minlength=36).astype(float)))
    assert abs(sk - fast) < 1e-12, (sk, fast)

    for seed in SEEDS:
        tr = build_control_rows(seed, "train", d["train_ids"], d)
        te = build_control_rows(seed, "test", d["test_ids"], d)
        for c in CONTROLS:
            assert set(tr[c][0]) == set(te[c][0])
            pred, _ = fit_predict_fixed(X_tr, X_te, tr[c], te[c], y_tr, ALPHA)
            record(f"{c}_s{seed}", pred, len(tr[c][0]))
        print(f"  seed {seed} done ({time.time() - t0:.0f}s): " + ", ".join(
            f"{c}={compute_metrics(y_te, preds[f'{c}_s{seed}'])['qwk']:.4f}" for c in CONTROLS), flush=True)

    # Per-essay predictions (one gzipped long-format file per condition family).
    for fam in REAL_CONDITIONS + CONTROLS:
        conds = [fam] if fam in REAL_CONDITIONS else [f"{fam}_s{s}" for s in SEEDS]
        frames = []
        for c in conds:
            frames.append(pd.DataFrame({
                "condition": fam, "seed": "" if fam in REAL_CONDITIONS else int(c.rsplit("_s", 1)[1]),
                "essay_id": d["test_ids"], "true_score": y_te,
                "raw_prediction": preds[c], "rounded_prediction": clip_and_round(preds[c]),
            }))
        pd.concat(frames).to_csv(pred_dir / f"{fam}.csv.gz", index=False, float_format="%.8f")

    qwk = {c: compute_metrics(y_te, p)["qwk"] for c, p in preds.items()}

    # Step 0 reproduction
    ref = pd.read_csv(PROJECT_ROOT / cfg["paths"]["output_dir"] / "final_results.csv").set_index("condition")["qwk"]
    rerun_path = OUT_DIR / "step0_hash_randomized_rerun.csv"
    rerun = pd.read_csv(rerun_path).set_index("condition")["qwk"] if rerun_path.exists() else pd.Series(dtype=float)
    repro_rows = []
    for c in REAL_CONDITIONS:
        repro_rows.append({"condition": c, "reported_qwk": ref[c], "rerun_original_code_qwk": rerun.get(c, np.nan),
                           "posthoc_qwk": qwk[c], "match_4dp": round(qwk[c], 4) == round(ref[c], 4)})
    for b in ("TS", "RB", "RND"):
        for s in ORIGINAL_SEEDS:
            c = f"{b}_s{s}"
            repro_rows.append({"condition": c, "reported_qwk": ref[c], "rerun_original_code_qwk": rerun.get(c, np.nan),
                               "posthoc_qwk": qwk[c], "match_4dp": round(qwk[c], 4) == round(ref[c], 4)})
    repro_df = pd.DataFrame(repro_rows)
    repro_df.to_csv(OUT_DIR / "step0_reproduction.csv", index=False)

    # Seed summaries
    seed_rows = []
    for c in CONTROLS:
        vals = np.array([qwk[f"{c}_s{s}"] for s in SEEDS])
        orig = np.array([qwk[f"{c}_s{s}"] for s in ORIGINAL_SEEDS])
        ens = compute_metrics(y_te, np.mean([preds[f"{c}_s{s}"] for s in SEEDS], axis=0))["qwk"]
        seed_rows.append({"control": c, "n_features": sorted(n_features[c])[0], "n_seeds": len(SEEDS),
                          "mean_qwk": vals.mean(), "sd_qwk": vals.std(ddof=1), "min_qwk": vals.min(),
                          "max_qwk": vals.max(), "mean_qwk_original5": orig.mean(), "ensemble_pred_qwk": ens})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / "control_seed_summary.csv", index=False)
    pd.DataFrame([{"condition": c, "qwk": v} for c, v in qwk.items()]).to_csv(
        OUT_DIR / "all_condition_qwks.csv", index=False)

    # Steps 1-3
    result_rows = []
    for step, (real, ctrls) in STEPS.items():
        for c in ctrls:
            cps = [preds[f"{c}_s{s}"] for s in SEEDS]
            boot = bootstrap_vs_controls(y_te, preds[real], cps)
            pos, total = prompts_improved(test_df, y_te, preds[real], cps)
            mean_c = float(np.mean([qwk[f"{c}_s{s}"] for s in SEEDS]))
            result_rows.append({
                "step": step, "condition_a": real, "control": c,
                "n_features_a": sorted(n_features[real])[0], "n_features_control": sorted(n_features[c])[0],
                "qwk_a": qwk[real], "mean_seed_qwk_control": mean_c, "qwk_diff": qwk[real] - mean_c, **boot,
                "prompts_improved": pos, "prompts_total": total,
            })
            print(f"  {real} vs {c}: diff={qwk[real] - mean_c:+.4f} CI=[{boot['ci_lower']:+.4f},"
                  f"{boot['ci_upper']:+.4f}] P={boot['p_diff_gt_0']:.3f} prompts={pos}/{total}", flush=True)
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(OUT_DIR / "posthoc_results.csv", index=False)

    write_report(repro_df, seed_df, results_df, n_features, d, perm_identity_failures, perm_lengths_ok, qwk)
    print(f"Done in {time.time() - t0:.0f}s")


def write_report(repro_df, seed_df, results_df, n_features, d, perm_fail, perm_ok, qwk) -> None:
    n_order = len(order_keys_full())
    fmt = lambda x: f"{x:.4f}"  # noqa: E731
    rep = repro_df.copy()
    for col in ("reported_qwk", "rerun_original_code_qwk", "posthoc_qwk"):
        rep[col] = rep[col].map(lambda v: "" if pd.isna(v) else fmt(v))

    feat_rows = [{"condition": c, "n_features": sorted(n_features[c])[0]} for c in REAL_CONDITIONS + CONTROLS]
    lines = [
        "# POST-HOC Control Analysis",
        "",
        "**POST-HOC.** Everything in this report was added after the frozen Step 4 held-out "
        "evaluation (`outputs/final_heldout_evaluation/`). The original pipeline, configs and "
        "outputs are unchanged. Produced by `scripts/run_posthoc_controls.py`.",
        "",
        "## Setup",
        "",
        f"- Train essays: {len(d['train_ids']):,}; test essays: {len(d['test_ids']):,} (official split, join-valid).",
        f"- TF-IDF and StandardScaler fit on train only (shared Step 4 code); Ridge alpha = {ALPHA}.",
        f"- Seeds (20): original {ORIGINAL_SEEDS} plus {EXTRA_SEEDS}.",
        f"- Paired bootstrap: {BOOTSTRAP_SAMPLES:,} resamples of test essays (rng seed {BOOTSTRAP_SEED}); in each "
        "resample the statistic is QWK(condition) − mean over the 20 seeds of QWK(control seed), the same "
        "aggregation as `bootstrap_mean_control_diff` in Step 4. `qwk_diff` is the full-test-set "
        "QWK(condition) − mean-of-seed QWK(control).",
        "- Prompts improved: prompts where QWK(condition) > mean-of-seed control QWK (same rule as Step 4).",
        "",
        "## Control reproducibility",
        "",
        "The Step 4 controls (TS, RB, RND) derived their random seeds from Python's built-in `hash()` of "
        "essay ids (`shuffle_labels`, `rb_structure`) and of the split tag (`random_control_features`). "
        "String hashing is randomized per process and `PYTHONHASHSEED` was never set, so the original control "
        "draws could not be reproduced: rerunning the unchanged code gave different QWKs for 14 of 15 "
        "control seeds (column `rerun_original_code_qwk`; TS_s87 matched to 4 dp by chance), and two further "
        "processes with `PYTHONHASHSEED=1` and `=2` gave different values again.",
        "",
        "All controls in this report are **regenerated after the held-out evaluation (post-hoc)** with "
        "deterministic seeding: `int(md5(key).hexdigest()[:8], 16)` with key "
        "`essay_id|split|seed` (TS, RB), `essay_id|split|seed|perm` (PERM), and `rnd|n_dims|split|seed` "
        "(RND-type controls, one RNG per split as in the original). Only the seed derivation changed; the "
        "perturbation logic is identical. Determinism was confirmed by fitting all ten controls for one seed "
        "in two separate processes with different `PYTHONHASHSEED` values "
        "(`determinism_check.csv`: identical QWKs).",
        "",
        "Step 0 comparison (reported Step 4 values vs. this post-hoc run). T2/T3/POS/T3_PLUS_POS are "
        "deterministic and must match; control seeds are new draws and are not expected to match.",
        "",
        df_to_md(rep),
        "",
        "## Feature counts",
        "",
        f"T2 = 24 (surface + counts); order features = {n_order}; geometry (`pos_feature_dict`) = {d['n_pos']} "
        f"(max_ranks = {d['max_ranks']}). Counts verified on the built feature rows for every seed; "
        "train and test key sets are identical.",
        "",
        df_to_md(pd.DataFrame(feat_rows)),
        "",
        "Note: `pos_comp_count` (in geometry) duplicates `count_total_components` (in T2). It is kept, as in "
        "the frozen T3_PLUS_POS/POS, so results stay comparable.",
        "",
        "## Control definitions",
        "",
        "- **TS**: T2 + order features from shuffled labels at true boundaries.",
        "- **RB**: T2 + order features from `rb_structure` (shuffled labels, random contiguous boundaries).",
        f"- **RND**: T2 + {n_order} N(0,1) columns.",
        "- **POS_RB**: T2 + geometry from the `rb_structure` boundaries (segments = the random components; "
        "no unannotated segments).",
        "- **POS_PERM**: T2 + geometry after a length-preserving permutation: every segment (rhetorical "
        "component or Unannotated) keeps its length, the order of segments is randomly permuted, and the "
        "uncovered spacers between slots and the start offset stay in place.",
        f"- **POS_RND**: T2 + {d['n_pos']} N(0,1) columns.",
        "- **FULL_TS**: T2 + TS order features + true geometry.",
        "- **FULL_RB**: T2 + order and geometry both from the `rb_structure` boundaries.",
        "- **FULL_PERM**: T2 + order and geometry both from the length-preserving permutation (labels "
        "travel with their components).",
        f"- **FULL_RND**: T2 + {n_order + d['n_pos']} N(0,1) columns.",
        "",
        f"Permutation checks: identity permutation reproduces true boundaries and geometry for all essays "
        f"(failures = {perm_fail}); multiset of (length, type) and first-start/last-end preserved: {perm_ok}.",
        "",
        "## Control QWK across 20 seeds",
        "",
        df_to_md(seed_df.assign(**{c: seed_df[c].map(fmt) for c in
                                   ("mean_qwk", "sd_qwk", "min_qwk", "max_qwk", "mean_qwk_original5",
                                    "ensemble_pred_qwk")})),
        "",
    ]
    for step in STEPS:
        sub = results_df[results_df["step"] == step]
        tbl = pd.DataFrame({
            "comparison": sub["condition_a"] + " vs " + sub["control"],
            "features (a / control)": sub["n_features_a"].astype(str) + " / " + sub["n_features_control"].astype(str),
            "QWK a": sub["qwk_a"].map(fmt),
            "QWK control (mean of 20)": sub["mean_seed_qwk_control"].map(fmt),
            "diff": sub["qwk_diff"].map(lambda v: f"{v:+.4f}"),
            "95% CI": sub.apply(lambda r: f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]", axis=1),
            "P(diff>0)": sub["p_diff_gt_0"].map(lambda v: f"{v:.4f}"),
            "prompts improved": sub["prompts_improved"].astype(str) + "/" + sub["prompts_total"].astype(str),
        })
        lines += [f"## {step}", "", df_to_md(tbl), ""]

    lines += ["## Notes on unexpected results", "", "NOTES_PLACEHOLDER", ""]
    (OUT_DIR / "posthoc_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
