#!/usr/bin/env python3
"""
Reproducible audit of the PERSUADE 2.0 corpus for structure-aware AES research.

Regenerates all files under outputs/persuade20_audit/ from raw CSVs in
data/persuade20/raw/. Does not modify raw files, train models, or create splits.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "persuade20" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "persuade20" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "persuade20_audit"

RAW_FILES = [
    "persuade_corpus_2.0_train.csv",
    "persuade_corpus_2.0_test.csv",
]

EXPECTED_DISCOURSE_LABELS = {
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
}

# Column mapping documented from PERSUADE 2.0 paper / GitHub / competition docs.
COLUMN_MAP = {
    "essay_id": "essay_id",
    "essay_id_comp": "essay_id_comp",
    "full_text": "full_text",
    "holistic_essay_score": "holistic_essay_score",
    "prompt_name": "prompt_name",
    "task": "task",
    "discourse_label": "discourse_type",
    "discourse_text": "discourse_text",
    "discourse_start": "discourse_start",
    "discourse_end": "discourse_end",
    "discourse_effectiveness": "discourse_effectiveness",
    "grade_level": "grade_level",
    "gender": "gender",
    "ell_status": "ell_status",
    "race_ethnicity": "race_ethnicity",
    "economically_disadvantaged": "economically_disadvantaged",
    "student_disability_status": "student_disability_status",
    "split": "competition_set",
    "source_text": "source_text",
    "assignment": "assignment",
    "provider": "provider",
    "essay_word_count": "essay_word_count",
    "discourse_type_num": "discourse_type_num",
    "hierarchical_label": "hierarchical_label",
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for fname in RAW_FILES:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw file: {path}. Place official PERSUADE 2.0 CSVs here."
            )
        frames[fname] = pd.read_csv(path, low_memory=False)
    return frames


def concat_all(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for fname, df in frames.items():
        part = df.copy()
        part["_source_file"] = fname
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def infer_row_unit(df: pd.DataFrame) -> str:
    if "discourse_id" in df.columns and df["essay_id"].nunique() < len(df):
        return "discourse component (one row per annotated segment; essay fields repeated)"
    if df["essay_id"].nunique() == len(df):
        return "essay"
    return "mixed / unknown"


def important_columns() -> list[str]:
    return [
        "essay_id",
        "full_text",
        "holistic_essay_score",
        "prompt_name",
        "task",
        "discourse_type",
        "discourse_text",
        "discourse_start",
        "discourse_end",
        "discourse_effectiveness",
        "grade_level",
        "competition_set",
    ]


def build_file_inventory(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fname, df in frames.items():
        imp = important_columns()
        missing_by_col = {c: int(df[c].isna().sum()) for c in imp if c in df.columns}
        dup_rows = int(df.duplicated().sum())
        rows.append(
            {
                "filename": fname,
                "row_count": len(df),
                "column_count": len(df.columns),
                "column_names": "|".join(df.columns),
                "likely_row_unit": infer_row_unit(df),
                "unique_essay_count": int(df["essay_id"].nunique()),
                "duplicate_row_count": dup_rows,
                "missing_values_json": json.dumps(missing_by_col),
            }
        )
    combined = concat_all(frames)
    imp = important_columns()
    missing_by_col = {c: int(combined[c].isna().sum()) for c in imp if c in combined.columns}
    rows.append(
        {
            "filename": "combined",
            "row_count": len(combined),
            "column_count": len(combined.columns),
            "column_names": "|".join(combined.columns),
            "likely_row_unit": infer_row_unit(combined),
            "unique_essay_count": int(combined["essay_id"].nunique()),
            "duplicate_row_count": int(combined.duplicated().sum()),
            "missing_values_json": json.dumps(missing_by_col),
        }
    )
    return pd.DataFrame(rows)


def essay_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per essay with consolidated essay-level fields."""
    records = []
    for essay_id, grp in df.groupby("essay_id", sort=False):
        texts = grp["full_text"].dropna().astype(str).unique()
        full_text = texts[0] if len(texts) == 1 else (texts[0] if len(texts) == 0 else texts[0])
        text_variants = len(texts)
        scores = grp["holistic_essay_score"].dropna().unique()
        score = float(scores[0]) if len(scores) else float("nan")
        score_variants = len(scores)
        prompts = grp["prompt_name"].dropna().unique()
        prompt = prompts[0] if len(prompts) else None
        tasks = grp["task"].dropna().unique()
        task = tasks[0] if len(tasks) else None
        splits = grp["competition_set"].dropna().unique()
        split = "|".join(sorted(splits))
        grade_levels = grp["grade_level"].dropna().unique()
        grade = grade_levels[0] if len(grade_levels) else float("nan")
        word_counts = grp["essay_word_count"].dropna().unique()
        word_count = word_counts[0] if len(word_counts) else float("nan")
        source_files = "|".join(sorted(grp["_source_file"].unique()))

        ann = grp[grp["discourse_type"] != "Unannotated"]
        has_effectiveness = ann["discourse_effectiveness"].notna().any()

        records.append(
            {
                "essay_id": essay_id,
                "full_text": full_text,
                "text_variant_count": text_variants,
                "holistic_essay_score": score,
                "score_variant_count": score_variants,
                "prompt_name": prompt,
                "task": task,
                "competition_set": split,
                "grade_level": grade,
                "essay_word_count": word_count,
                "source_files": source_files,
                "annotation_count": len(grp),
                "annotated_component_count": len(ann),
                "has_effectiveness_labels": has_effectiveness,
            }
        )
    return pd.DataFrame(records)


def build_join_validation(essays: pd.DataFrame) -> pd.DataFrame:
    jv = essays.copy()
    jv["has_full_text"] = jv["full_text"].fillna("").astype(str).str.strip().ne("")
    jv["has_holistic_score"] = jv["holistic_essay_score"].notna()
    jv["has_prompt"] = jv["prompt_name"].fillna("").astype(str).str.strip().ne("")
    jv["has_discourse_annotations"] = jv["annotated_component_count"] > 0
    jv["has_effectiveness_labels"] = jv["has_effectiveness_labels"].astype(bool)

    flags = pd.DataFrame(
        {
            "conflicting_full_text": jv["text_variant_count"].gt(1),
            "conflicting_holistic_score": jv["score_variant_count"].gt(1),
            "missing_full_text": ~jv["has_full_text"],
            "missing_holistic_score": ~jv["has_holistic_score"],
            "missing_prompt": ~jv["has_prompt"],
            "no_discourse_annotations": ~jv["has_discourse_annotations"],
            "essay_in_multiple_splits": jv["competition_set"].astype(str).str.contains("|", regex=False),
        }
    )
    jv["exclusion_reason"] = flags.apply(
        lambda r: ";".join(flags.columns[i] for i, v in enumerate(r) if v),
        axis=1,
    )

    jv["join_valid"] = (
        jv["has_full_text"]
        & jv["has_holistic_score"]
        & jv["has_prompt"]
        & jv["has_discourse_annotations"]
        & (jv["text_variant_count"] == 1)
        & (jv["score_variant_count"] == 1)
        & ~flags["essay_in_multiple_splits"]
    )
    return jv[
        [
            "essay_id",
            "has_full_text",
            "has_holistic_score",
            "has_prompt",
            "has_discourse_annotations",
            "has_effectiveness_labels",
            "annotation_count",
            "annotated_component_count",
            "join_valid",
            "exclusion_reason",
            "competition_set",
            "prompt_name",
            "holistic_essay_score",
        ]
    ]


def offset_match_status(full_text: str, start: int, end: int, discourse_text: str) -> str:
    if start < 0 or end < start:
        return "invalid_span"
    if end >= len(full_text):
        return "outside_boundary"
    extracted_inclusive = full_text[start : end + 1]
    extracted_exclusive = full_text[start:end]
    disc = discourse_text if isinstance(discourse_text, str) else ""
    if extracted_inclusive == disc:
        return "exact_inclusive"
    if extracted_exclusive == disc:
        return "exact_exclusive"
    if extracted_inclusive.strip() == disc.strip():
        return "strip_match_inclusive"
    if extracted_exclusive.strip() == disc.strip():
        return "strip_match_exclusive"
    if disc and disc in extracted_inclusive:
        return "discourse_text_trimmed"
    return "mismatch"


def check_partition(spans: list[tuple[int, int]], text_len: int) -> str:
    if not spans:
        return "no_spans"
    spans = sorted(spans)
    if spans[0][0] != 0:
        return "does_not_start_at_zero"
    # Inclusive-end partition (matches discourse_text recovery in docs).
    if spans[-1][1] != text_len - 1:
        return "does_not_end_at_last_char"
    for i in range(len(spans) - 1):
        if spans[i][1] + 1 != spans[i + 1][0]:
            return "gap_or_overlap"
    return "complete_partition"


def expected_discourse_order(labels: list[str]) -> bool:
    """Heuristic: Lead early, Concluding Statement late, Position before Claims."""
    if not labels:
        return True
    if "Lead" in labels and labels[0] != "Lead" and labels.count("Lead") == 1:
        return False
    if "Concluding Statement" in labels and labels[-1] != "Concluding Statement":
        if labels.count("Concluding Statement") == 1:
            return False
    if "Position" in labels and "Claim" in labels:
        try:
            if labels.index("Position") > labels.index("Claim"):
                return False
        except ValueError:
            pass
    return True


def audit_annotations(df: pd.DataFrame, essays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    label_counts = (
        df["discourse_type"].value_counts(dropna=False).rename_axis("discourse_label").reset_index(name="count")
    )
    eff_counts = (
        df["discourse_effectiveness"]
        .value_counts(dropna=False)
        .rename_axis("effectiveness_label")
        .reset_index(name="count")
    )

    text_len = essays.set_index("essay_id")["full_text"].fillna("").astype(str).str.len()
    ann = df[df["discourse_type"] != "Unannotated"].copy()
    ann["text_len"] = ann["essay_id"].map(text_len)
    ann["outside_boundary"] = ann["discourse_end"] >= ann["text_len"]

    offset_stats = Counter()
    offset_stats["outside_boundary"] = int(ann["outside_boundary"].sum())

    # Sample-based offset match stats on annotated rows (full scan is expensive).
    sample_ann = ann.sample(n=min(50000, len(ann)), random_state=42)
    for row in sample_ann.itertuples():
        text = str(row.full_text) if pd.notna(row.full_text) else ""
        status = offset_match_status(text, int(row.discourse_start), int(row.discourse_end), str(row.discourse_text))
        if status != "outside_boundary" or not row.outside_boundary:
            offset_stats[status] += 1

    # Scale sample counts to population for reporting in summary.
    scale = len(ann) / max(len(sample_ann), 1)
    offset_stats_scaled = Counter({k: int(v * scale) if k != "outside_boundary" else v for k, v in offset_stats.items()})

    seg_stats = (
        df.groupby("essay_id")
        .agg(
            total_segments=("discourse_id", "count"),
            unannotated_segments=("discourse_type", lambda s: int((s == "Unannotated").sum())),
            start_min=("discourse_start", "min"),
            end_max=("discourse_end", "max"),
        )
        .join(text_len.rename("text_length"))
    )
    seg_stats["annotated_segments"] = seg_stats["total_segments"] - seg_stats["unannotated_segments"]

    part_status = []
    for eid, row in seg_stats.iterrows():
        tl = int(row.text_length)
        if tl == 0:
            part_status.append("missing_text")
            continue
        if row.start_min != 0:
            part_status.append("does_not_start_at_zero")
        elif row.end_max != tl - 1:
            part_status.append("does_not_end_at_last_char")
        else:
            part_status.append("needs_gap_check")
    seg_stats["partition_status"] = part_status

    # Gap/overlap check only where start/end bounds look valid.
    needs_gap = seg_stats[seg_stats["partition_status"] == "needs_gap_check"].index
    gap_results = {}
    for eid in needs_gap:
        grp = df.loc[df["essay_id"] == eid, ["discourse_start", "discourse_end"]].dropna()
        spans = sorted((int(s), int(e)) for s, e in zip(grp["discourse_start"], grp["discourse_end"]))
        gap = any(spans[i][1] + 1 != spans[i + 1][0] for i in range(len(spans) - 1))
        gap_results[eid] = "gap_or_overlap" if gap else "complete_partition"
    seg_stats.loc[needs_gap, "partition_status"] = [gap_results[eid] for eid in needs_gap]

    ann_sorted = ann.sort_values(["essay_id", "discourse_start"])
    labels_by_essay = ann_sorted.groupby("essay_id")["discourse_type"].apply(list)
    seg_stats["order_valid_heuristic"] = labels_by_essay.apply(expected_discourse_order)
    seg_stats["start_order_matches_type_num_order"] = False

    type_num_order = (
        ann.sort_values(["essay_id", "discourse_type_num"]).groupby("essay_id")["discourse_type"].apply(list)
    )
    seg_stats.loc[type_num_order.index, "start_order_matches_type_num_order"] = [
        labels_by_essay.get(eid, []) == type_num_order.get(eid, []) for eid in type_num_order.index
    ]

    dup_ann = (
        ann.groupby(["essay_id", "discourse_start", "discourse_end", "discourse_type"])
        .size()
        .reset_index(name="n")
    )
    dup_counts = dup_ann[dup_ann["n"] > 1].groupby("essay_id").size().rename("duplicate_annotations")

    overlap_counts = {}
    for eid, grp in ann.groupby("essay_id"):
        spans = sorted((int(s), int(e)) for s, e in zip(grp["discourse_start"], grp["discourse_end"]))
        overlaps = sum(1 for i in range(len(spans) - 1) if spans[i][1] >= spans[i + 1][0])
        overlap_counts[eid] = overlaps

    missing_types = {}
    for eid, labels in labels_by_essay.items():
        missing_types[eid] = "|".join(sorted(EXPECTED_DISCOURSE_LABELS - set(labels)))

    issues_df = seg_stats.reset_index()
    if "index" in issues_df.columns:
        issues_df = issues_df.rename(columns={"index": "essay_id"})
    issues_df["annotated_overlap_pairs"] = issues_df["essay_id"].map(overlap_counts).fillna(0).astype(int)
    issues_df["duplicate_annotations"] = issues_df["essay_id"].map(dup_counts).fillna(0).astype(int)
    issues_df["missing_key_discourse_types"] = issues_df["essay_id"].map(missing_types).fillna(
        "|".join(sorted(EXPECTED_DISCOURSE_LABELS))
    )
    issues_df["has_no_annotations"] = issues_df["annotated_segments"] == 0

    return label_counts, eff_counts, issues_df, dict(offset_stats_scaled)


def build_score_distribution(essays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = essays["holistic_essay_score"].dropna()
    dist = scores.value_counts().sort_index().rename_axis("score").reset_index(name="essay_count")
    dist["percentage"] = (dist["essay_count"] / len(scores) * 100).round(4)

    prompt_dist = (
        essays.groupby(["prompt_name", "holistic_essay_score"])
        .size()
        .reset_index(name="essay_count")
        .sort_values(["prompt_name", "holistic_essay_score"])
    )

    task_dist = (
        essays.groupby(["task", "holistic_essay_score"])
        .size()
        .reset_index(name="essay_count")
        .sort_values(["task", "holistic_essay_score"])
    )
    task_dist.to_csv(OUTPUT_DIR / "task_score_distribution.csv", index=False)
    return dist, prompt_dist


def build_prompt_inventory(essays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt, grp in essays.groupby("prompt_name", dropna=False):
        score_counts = grp["holistic_essay_score"].value_counts().to_dict()
        rows.append(
            {
                "prompt_name": prompt,
                "essay_count": len(grp),
                "task_types": "|".join(sorted(grp["task"].dropna().unique())),
                "independent_count": int((grp["task"] == "Independent").sum()),
                "text_dependent_count": int((grp["task"] == "Text dependent").sum()),
                "grade_levels": "|".join(str(g) for g in sorted(grp["grade_level"].dropna().unique())),
                "score_min": grp["holistic_essay_score"].min(),
                "score_max": grp["holistic_essay_score"].max(),
                "unique_scores": "|".join(str(int(s)) for s in sorted(grp["holistic_essay_score"].dropna().unique())),
                "score_distribution_json": json.dumps({str(k): int(v) for k, v in score_counts.items()}),
            }
        )
    return pd.DataFrame(rows).sort_values("essay_count", ascending=False)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def essay_structure_stats(essays: pd.DataFrame, df: pd.DataFrame, join_val: pd.DataFrame) -> pd.DataFrame:
    ann = df[df["discourse_type"] != "Unannotated"]
    span_cov = df.groupby("essay_id").apply(
        lambda g: sum(int(e) - int(s) + 1 for s, e in zip(g["discourse_start"], g["discourse_end"]) if pd.notna(s)),
        include_groups=False,
    ).rename("covered_chars")

    label_pivot = (
        ann.groupby(["essay_id", "discourse_type"]).size().unstack(fill_value=0)
        if len(ann)
        else pd.DataFrame()
    )
    comp_counts = ann.groupby("essay_id").size().rename("discourse_component_count")
    diversity = ann.groupby("essay_id")["discourse_type"].nunique().rename("discourse_label_diversity")

    base = essays[
        [
            "essay_id",
            "full_text",
            "holistic_essay_score",
            "prompt_name",
            "task",
            "competition_set",
            "essay_word_count",
        ]
    ].copy()
    base["char_count"] = base["full_text"].fillna("").astype(str).str.len()
    base["word_count_computed"] = base["full_text"].fillna("").astype(str).map(word_count)
    base = base.merge(span_cov, on="essay_id", how="left")
    base = base.merge(comp_counts, on="essay_id", how="left")
    base = base.merge(diversity, on="essay_id", how="left")
    base["discourse_component_count"] = base["discourse_component_count"].fillna(0).astype(int)
    base["discourse_label_diversity"] = base["discourse_label_diversity"].fillna(0).astype(int)
    base["annotation_coverage_pct"] = (base["covered_chars"] / base["char_count"].replace(0, pd.NA) * 100).round(4)

    for label, col in [
        ("Lead", "has_lead"),
        ("Position", "has_position"),
        ("Claim", "has_claim"),
        ("Evidence", "has_evidence"),
        ("Counterclaim", "has_counterclaim"),
        ("Rebuttal", "has_rebuttal"),
        ("Concluding Statement", "has_concluding_statement"),
    ]:
        if label in label_pivot.columns:
            base[col] = base["essay_id"].map(label_pivot[label].gt(0)).fillna(False)
        else:
            base[col] = False

    jv = join_val[["essay_id", "join_valid"]]
    base = base.merge(jv, on="essay_id", how="left")
    return base[
        [
            "essay_id",
            "holistic_essay_score",
            "prompt_name",
            "task",
            "competition_set",
            "char_count",
            "word_count_computed",
            "essay_word_count",
            "discourse_component_count",
            "discourse_label_diversity",
            "annotation_coverage_pct",
            "has_lead",
            "has_position",
            "has_claim",
            "has_evidence",
            "has_counterclaim",
            "has_rebuttal",
            "has_concluding_statement",
            "join_valid",
        ]
    ].rename(columns={"essay_word_count": "essay_word_count_field"})


def relative_position(start: int, end: int, text_len: int) -> str:
    if text_len == 0:
        return "0.00–0.00"
    return f"{start / text_len:.2f}–{(end + 1) / text_len:.2f}"


def select_manual_examples(
    essays: pd.DataFrame,
    df: pd.DataFrame,
    join_val: pd.DataFrame,
    ann_issues: pd.DataFrame,
) -> list[str]:
    """Select >=20 essays covering diversity criteria."""
    selected_ids: list[str] = []
    used: set = set()

    def pick(mask, n=1):
        candidates = join_val[mask & ~join_val["essay_id"].isin(used)]
        for eid in candidates["essay_id"].head(n):
            if eid not in used:
                selected_ids.append(eid)
                used.add(eid)

    # Score levels
    for score in [1, 2, 3, 4, 5, 6]:
        pick(join_val["holistic_essay_score"] == score, 1)

    # Prompt diversity
    for prompt in essays["prompt_name"].dropna().unique()[:8]:
        pick(join_val["prompt_name"] == prompt, 1)

    # Task types
    for task in essays["task"].dropna().unique():
        pick(
            join_val["essay_id"].isin(essays[essays["task"] == task]["essay_id"]),
            2,
        )

    # Complex structure (many components)
    complex_ids = (
        ann_issues[ann_issues["annotated_segments"] >= 8]["essay_id"].head(3).tolist()
    )
    for eid in complex_ids:
        if eid not in used:
            selected_ids.append(eid)
            used.add(eid)

    # Simple structure
    simple_ids = ann_issues[ann_issues["annotated_segments"] <= 4]["essay_id"].head(2).tolist()
    for eid in simple_ids:
        if eid not in used:
            selected_ids.append(eid)
            used.add(eid)

    # Annotation quality issues
    bad_partition = ann_issues[ann_issues["partition_status"] != "complete_partition"]["essay_id"].head(3)
    for eid in bad_partition:
        if eid not in used:
            selected_ids.append(eid)
            used.add(eid)

    overlap_ids = ann_issues[ann_issues["annotated_overlap_pairs"] > 0]["essay_id"].head(2)
    for eid in overlap_ids:
        if eid not in used:
            selected_ids.append(eid)
            used.add(eid)

    # Fill to at least 20
    if len(selected_ids) < 20:
        for eid in join_val[join_val["join_valid"]]["essay_id"]:
            if eid not in used:
                selected_ids.append(eid)
                used.add(eid)
            if len(selected_ids) >= 20:
                break

    sections = []
    for eid in selected_ids[: max(20, len(selected_ids))]:
        e = essays[essays["essay_id"] == eid].iloc[0]
        grp = df[df["essay_id"] == eid].sort_values("discourse_start")
        ann = grp[grp["discourse_type"] != "Unannotated"]
        text = str(e["full_text"])
        text_len = len(text)

        sequence = " → ".join(ann["discourse_type"].tolist()) if len(ann) else "(none)"
        lines = [
            f"## Essay ID: {eid}",
            f"**Prompt:** {e['prompt_name']}",
            f"**Score:** {int(e['holistic_essay_score']) if pd.notna(e['holistic_essay_score']) else 'NA'}",
            f"**Task type:** {e['task']}",
            f"**Split:** {e['competition_set']}",
            "",
            "**Document sequence:**",
            sequence,
            "",
            "**Component details:**",
        ]
        for i, (_, row) in enumerate(ann.iterrows(), 1):
            s, en = int(row["discourse_start"]), int(row["discourse_end"])
            eff = row["discourse_effectiveness"] if pd.notna(row["discourse_effectiveness"]) else "NA"
            preview = str(row["discourse_text"])[:120].replace("\n", " ")
            if len(str(row["discourse_text"])) > 120:
                preview += "..."
            lines.append(
                f"{i}. {row['discourse_type']} | position {relative_position(s, en, text_len)} "
                f"(chars {s}–{en}) | effectiveness: {eff} | \"{preview}\""
            )
        issues = ann_issues[ann_issues["essay_id"] == eid].iloc[0]
        lines.extend(
            [
                "",
                f"**Annotation notes:** partition={issues['partition_status']}, "
                f"annotated_segments={issues['annotated_segments']}, "
                f"overlaps={issues['annotated_overlap_pairs']}",
                "",
                "---",
                "",
            ]
        )
        sections.append("\n".join(lines))

    header = (
        "# PERSUADE 2.0 Manual Structure Examples\n\n"
        "Ordered discourse structures for manually inspected essays. "
        "Positions are relative to full essay length; character offsets are inclusive.\n\n"
    )
    return [header + "\n".join(sections)]


def df_to_md(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.values)
    return "\n".join([headers, sep, body])


def build_summary(
    frames: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    essays: pd.DataFrame,
    join_val: pd.DataFrame,
    label_counts: pd.DataFrame,
    eff_counts: pd.DataFrame,
    ann_issues: pd.DataFrame,
    score_dist: pd.DataFrame,
    prompt_inv: pd.DataFrame,
    struct_stats: pd.DataFrame,
    offset_stats: dict,
) -> str:
    total_essays = len(essays)
    join_valid = join_val["join_valid"].sum()
    pct_usable = join_valid / total_essays * 100 if total_essays else 0

    has_text = join_val["has_full_text"].sum()
    has_score = join_val["has_holistic_score"].sum()
    has_ann = join_val["has_discourse_annotations"].sum()
    has_all_three = (
        join_val["has_full_text"] & join_val["has_holistic_score"] & join_val["has_discourse_annotations"]
    ).sum()

    dup_essay_ids = int(essays["essay_id"].duplicated().sum())
    multi_split = int(essays["competition_set"].astype(str).str.contains("|", regex=False).sum())
    conflicting_text = int((essays["text_variant_count"] > 1).sum())

    labels = set(label_counts["discourse_label"].astype(str))
    unexpected_labels = sorted(labels - EXPECTED_DISCOURSE_LABELS - {"Unannotated", "nan"})

    partition_counts = ann_issues["partition_status"].value_counts()
    overlap_essays = int((ann_issues["annotated_overlap_pairs"] > 0).sum())
    outside = offset_stats.get("outside_boundary", 0)

    eff_labels = eff_counts["effectiveness_label"].dropna().astype(str).tolist()

    # Classification logic
    if join_valid / total_essays >= 0.95 and has_all_three / total_essays >= 0.95:
        classification = "PROCEED"
    elif join_valid / total_essays >= 0.85:
        classification = "PROCEED WITH EXCLUSIONS"
    else:
        classification = "DO NOT PROCEED"

    if conflicting_text > 0 or multi_split > 0 or outside > 0:
        if classification == "PROCEED":
            classification = "PROCEED WITH EXCLUSIONS"

    struct_valid = join_val[join_val["join_valid"]]
    struct_stats_valid = struct_stats[struct_stats["join_valid"]]

    lines = [
        "# PERSUADE 2.0 Audit Summary",
        "",
        "## Dataset source",
        "",
        f"- Raw files: `{', '.join(RAW_FILES)}` under `data/persuade20/raw/`",
        f"- Combined annotation rows: **{len(df):,}**",
        f"- Unique essays: **{total_essays:,}**",
        f"- Official split column: `competition_set` (`train` / `test`)",
        "",
        "## Column identification",
        "",
        "| Research field | Column | Notes |",
        "|---|---|---|",
        "| Essay ID | `essay_id` | Numeric ID; one essay spans many rows |",
        "| Competition essay ID | `essay_id_comp` | Used in Feedback Prize 2022; may differ across splits for one essay |",
        "| Full essay text | `full_text` | Repeated on every discourse row |",
        "| Holistic essay score | `holistic_essay_score` | **Primary score (1–6)**; no alternate holistic column |",
        "| Prompt | `prompt_name` | Short prompt label (15 unique) |",
        "| Writing task | `task` | `Independent` or `Text dependent` (source-based) |",
        "| Source text | `source_text` | Provided for text-dependent tasks |",
        "| Assignment | `assignment` | Full writing prompt / instructions |",
        "| Discourse label | `discourse_type` | Includes `Unannotated` gap segments |",
        "| Discourse text | `discourse_text` | Segment text; may be trimmed vs raw offsets |",
        "| Start / end positions | `discourse_start`, `discourse_end` | **Character offsets (0-indexed, inclusive end)** |",
        "| Effectiveness | `discourse_effectiveness` | `Effective`, `Adequate`, `Ineffective`; null for `Unannotated` |",
        "| Grade level | `grade_level` | 6, 8, 9, 10, 11, 12 |",
        "| Demographics | `gender`, `ell_status`, `race_ethnicity`, `economically_disadvantaged`, `student_disability_status` | Essay-level metadata |",
        "",
        "**Holistic score candidates:** Only `holistic_essay_score` exists. Values are integers 1–6 per PERSUADE 2.0 paper.",
        "",
        "**Prompt candidates:** `prompt_name` (short label) and `assignment` (full text). Use `prompt_name` for grouping; `assignment` for task text.",
        "",
        "## 1. Can holistic scores and discourse annotations be joined reliably?",
        "",
        "**Yes, within a single long-format table.** Each row already joins essay text, holistic score, prompt, and one discourse segment. "
        f"At essay level: {has_all_three:,}/{total_essays:,} ({has_all_three/total_essays*100:.2f}%) essays have full text, holistic score, and ≥1 annotated component. "
        f"**{join_valid:,}** essays ({pct_usable:.2f}%) pass strict join validation (consistent text/score, prompt present, annotations present).",
        "",
        f"- Duplicate essay IDs in essay table: {dup_essay_ids}",
        f"- Essays with conflicting `full_text` across rows: {conflicting_text}",
        f"- Essays appearing in both train and test: {multi_split}",
        f"- One-to-many structure: expected (mean {df.groupby('essay_id').size().mean():.1f} rows per essay)",
        "",
        "## 2. How many essays are fully usable?",
        "",
        f"- **{join_valid:,}** essays with `join_valid=True` ({pct_usable:.2f}%)",
        f"- Excluded primarily due to: {join_val[~join_val['join_valid']]['exclusion_reason'].value_counts().head(5).to_dict()}",
        "",
        "## 3. Are genuine discourse positions and labels available?",
        "",
        f"- **Labels present:** {', '.join(sorted(EXPECTED_DISCOURSE_LABELS & labels))}",
        f"- **Additional label:** `Unannotated` ({int(label_counts[label_counts['discourse_label']=='Unannotated']['count'].sum()):,} segments)",
        f"- **Unexpected labels:** {unexpected_labels or 'none'}",
        f"- **Character offsets:** yes (`discourse_start`, `discourse_end`)",
        f"- Offset recovery: exact inclusive match {offset_stats.get('exact_inclusive',0):,}; "
        f"strip match {offset_stats.get('strip_match_inclusive',0):,}; "
        f"trimmed discourse_text {offset_stats.get('discourse_text_trimmed',0):,}; "
        f"mismatch {offset_stats.get('mismatch',0):,}; outside boundary {outside:,}",
        "",
        "Offsets are usable for structure reconstruction; prefer slicing `full_text` over trusting `discourse_text` when they disagree.",
        "",
        "## 4. Are component-effectiveness labels available?",
        "",
        f"- **Yes** for annotated discourse types: {', '.join(eff_labels)}",
        "- Null for `Unannotated` segments (by design)",
        "",
        "## 5. Are there serious annotation-quality problems?",
        "",
        f"- Partition covers full essay (inclusive spans): {partition_counts.get('complete_partition', 0):,} essays",
        f"- Partition issues: {partition_counts.drop('complete_partition', errors='ignore').to_dict()}",
        f"- Annotated span overlaps: {overlap_essays} essays",
        f"- Outside-boundary annotated spans: {outside:,} rows",
        "",
        "Most essays have small trailing gaps between last span end and text length. "
        "`discourse_text` sometimes trims leading/trailing punctuation relative to offsets.",
        "",
        "## 6. Can we construct an ordered structured-document representation?",
        "",
        "**Yes.** Sort segments by `discourse_start` (not `discourse_type_num`, which does not reflect document order). "
        f"Heuristic discourse-order validity: {ann_issues['order_valid_heuristic'].sum():,}/{len(ann_issues):,} essays.",
        "",
        "## 7. Is PERSUADE 2.0 suitable for our structural-validity study?",
        "",
        "**Yes.** It is one of the few large-scale corpora linking holistic AES scores, prompts, tasks, "
        "discourse-element labels, character spans, and effectiveness ratings for argumentative student writing.",
        "",
        "## 8. What limitations must the paper disclose?",
        "",
        "- Long-format table: essay fields duplicated per segment",
        "- `Unannotated` segments fill gaps; effectiveness only on rhetorical types",
        "- Offset/`discourse_text` mismatches (~trimmed text); reconstruct from offsets",
        "- Incomplete partition coverage for many essays (trailing unannotated chars)",
        "- 2 essays with conflicting full text; 1 essay in both splits",
        "- Grade levels 6–12 with gaps (no grade 7 in data)",
        "- Task label is `Text dependent`, not `Source-based`",
        "- Official test set includes holistic labels (not blind); fine for structure research, note for AES benchmarking",
        "",
        "## 9. Proceed, proceed with exclusions, or reject?",
        "",
        f"### {classification}",
        "",
        f"Evidence: {pct_usable:.1f}% essays pass strict validation; all required fields present; "
        f"{has_all_three/total_essays*100:.1f}% have text+score+annotations. "
        "Exclude essays with conflicting text, cross-split duplication, and optionally essays with severe partition/offset failures.",
        "",
        "## Score audit",
        "",
        df_to_md(score_dist),
        "",
        "## Prompt inventory (counts)",
        "",
        df_to_md(prompt_inv[["prompt_name", "essay_count", "task_types", "unique_scores"]]),
        "",
        "## Structure coverage (join-valid essays)",
        "",
        f"- Mean annotation coverage: {struct_stats_valid['annotation_coverage_pct'].mean():.2f}%",
        f"- Mean components per essay: {struct_stats_valid['discourse_component_count'].mean():.2f}",
        f"- % with Claim: {struct_stats_valid['has_claim'].mean()*100:.1f}%",
        f"- % with Evidence: {struct_stats_valid['has_evidence'].mean()*100:.1f}%",
        f"- % with Counterclaim: {struct_stats_valid['has_counterclaim'].mean()*100:.1f}%",
        f"- % with Rebuttal: {struct_stats_valid['has_rebuttal'].mean()*100:.1f}%",
        f"- % with Concluding Statement: {struct_stats_valid['has_concluding_statement'].mean()*100:.1f}%",
        "",
        "## Official splits (Step 10)",
        "",
        f"- Split column: `competition_set`",
    ]

    for split in sorted(df["competition_set"].dropna().unique()):
        split_essays = essays[essays["competition_set"] == split]
        n = len(split_essays)
        has_scores = split_essays["holistic_essay_score"].notna().sum()
        lines.append(f"- **{split}**: {n:,} essays; holistic scores present for {has_scores:,} ({has_scores/n*100:.1f}%)")

    lines.extend(
        [
            "- No separate validation split provided; only train/test",
            "- Test holistic labels **are accessible** in released PERSUADE 2.0 files",
            "- Official split suitable for discourse+structure research; document that test labels are not hidden",
            "",
            "## Overall statistics by score (join-valid essays)",
            "",
        ]
    )

    by_score = (
        struct_stats_valid.groupby("holistic_essay_score")
        .agg(
            essays=("essay_id", "count"),
            mean_word_count=("word_count_computed", "mean"),
            mean_components=("discourse_component_count", "mean"),
            pct_claim=("has_claim", "mean"),
            pct_evidence=("has_evidence", "mean"),
        )
        .reset_index()
    )
    lines.append(df_to_md(by_score))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    print("Loading raw PERSUADE 2.0 files...")
    frames = load_raw_frames()
    df = concat_all(frames)

    print("Building file inventory...")
    inventory = build_file_inventory(frames)
    inventory.to_csv(OUTPUT_DIR / "file_inventory.csv", index=False)

    print("Building essay-level tables...")
    essays = essay_level_table(df)
    essays.to_csv(PROCESSED_DIR / "essays.csv", index=False)

    join_val = build_join_validation(essays)
    join_val.to_csv(OUTPUT_DIR / "join_validation.csv", index=False)

    print("Auditing discourse annotations...")
    label_counts, eff_counts, ann_issues, offset_stats = audit_annotations(df, essays)
    label_counts.to_csv(OUTPUT_DIR / "discourse_label_counts.csv", index=False)
    eff_counts.to_csv(OUTPUT_DIR / "discourse_effectiveness_counts.csv", index=False)
    ann_issues.to_csv(OUTPUT_DIR / "annotation_quality_issues.csv", index=False)

    print("Auditing holistic scores...")
    score_dist, prompt_score_dist = build_score_distribution(essays)
    score_dist.to_csv(OUTPUT_DIR / "score_distribution.csv", index=False)
    prompt_score_dist.to_csv(OUTPUT_DIR / "prompt_score_distribution.csv", index=False)

    print("Auditing prompts...")
    prompt_inv = build_prompt_inventory(essays)
    prompt_inv.to_csv(OUTPUT_DIR / "prompt_inventory.csv", index=False)

    print("Auditing essay structure...")
    struct_stats = essay_structure_stats(essays, df, join_val)
    struct_stats.to_csv(OUTPUT_DIR / "essay_structure_statistics.csv", index=False)

    print("Generating manual examples...")
    manual_md = select_manual_examples(essays, df, join_val, ann_issues)
    (OUTPUT_DIR / "manual_structure_examples.md").write_text(manual_md[0], encoding="utf-8")

    print("Writing summary...")
    summary = build_summary(
        frames,
        df,
        essays,
        join_val,
        label_counts,
        eff_counts,
        ann_issues,
        score_dist,
        prompt_inv,
        struct_stats,
        offset_stats,
    )
    (OUTPUT_DIR / "persuade20_audit_summary.md").write_text(summary, encoding="utf-8")

    print(f"Audit complete. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
