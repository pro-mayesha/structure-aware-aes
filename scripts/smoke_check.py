#!/usr/bin/env python3
"""
Lightweight repository smoke check for CI and reviewers.

Does not require PERSUADE 2.0 data. Verifies directory layout, key outputs,
configs, and core Python imports.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "configs",
    "scripts",
    "outputs/final_heldout_evaluation",
    "outputs/persuade20_audit",
    "data/persuade20/raw",
]

REQUIRED_SCRIPTS = [
    "scripts/audit_persuade20.py",
    "scripts/run_structure_feasibility.py",
    "scripts/run_structure_stress_tests.py",
    "scripts/run_structure_verification.py",
    "scripts/run_final_heldout_evaluation.py",
    "scripts/run_transformer_robustness.py",
]

REQUIRED_CONFIGS = [
    "configs/structure_feasibility.yaml",
    "configs/structure_stress_tests.yaml",
    "configs/structure_verification.yaml",
    "configs/final_heldout_evaluation.yaml",
    "configs/transformer_robustness.yaml",
]

REQUIRED_OUTPUTS = [
    "outputs/final_heldout_evaluation/final_results.csv",
    "outputs/final_heldout_evaluation/primary_comparisons.csv",
    "outputs/final_heldout_evaluation/final_report.md",
    "outputs/persuade20_audit/join_validation.csv",
]

CORE_IMPORTS = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "yaml",
]

FORBIDDEN_GIT_PATTERNS = [
    "data/persuade_corpus_2.0_",
    "data/persuade20/raw/persuade_corpus_2.0_",
]


def check_paths() -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_DIRS:
        if not (PROJECT_ROOT / rel).is_dir():
            errors.append(f"Missing directory: {rel}")
    for rel in REQUIRED_SCRIPTS + REQUIRED_CONFIGS + REQUIRED_OUTPUTS:
        if not (PROJECT_ROOT / rel).is_file():
            errors.append(f"Missing file: {rel}")
    return errors


def check_no_tracked_raw_data() -> list[str]:
    """Fail only if raw PERSUADE files are in the git index (not merely on disk)."""
    errors: list[str] = []
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            text=True,
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return errors

    for path in tracked:
        if any(path.startswith(prefix) for prefix in FORBIDDEN_GIT_PATTERNS):
            errors.append(f"Raw dataset file tracked in git (must be removed): {path}")
    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    for mod in CORE_IMPORTS:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            errors.append(f"Import failed: {mod} ({exc})")
    return errors


def check_headline_numbers() -> list[str]:
    """Spot-check shipped main results without modifying them."""
    errors: list[str] = []
    import pandas as pd

    results = PROJECT_ROOT / "outputs/final_heldout_evaluation/final_results.csv"
    primary = PROJECT_ROOT / "outputs/final_heldout_evaluation/primary_comparisons.csv"
    if not results.exists() or not primary.exists():
        return errors

    df = pd.read_csv(results)
    t3pp = df.loc[df["condition"] == "T3_PLUS_POS", "qwk"]
    t2 = df.loc[df["condition"] == "T2", "qwk"]
    if t3pp.empty or t2.empty:
        errors.append("final_results.csv missing T3_PLUS_POS or T2 row")
        return errors

    qwk_t3pp = float(t3pp.iloc[0])
    qwk_t2 = float(t2.iloc[0])
    if round(qwk_t3pp, 4) != 0.8488:
        errors.append(f"Unexpected T3_PLUS_POS QWK: {qwk_t3pp:.4f} (expected 0.8488)")
    if round(qwk_t2, 4) != 0.8361:
        errors.append(f"Unexpected T2 QWK: {qwk_t2:.4f} (expected 0.8361)")

    pc = pd.read_csv(primary)
    row = pc.loc[pc["comparison"] == "T3_PLUS_POS_vs_T2"]
    if row.empty:
        errors.append("primary_comparisons.csv missing T3_PLUS_POS_vs_T2")
    else:
        if int(row.iloc[0]["prompts_positive"]) != 15 or int(row.iloc[0]["prompts_total"]) != 15:
            errors.append("Prompt consistency not 15/15 in primary_comparisons.csv")

    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_paths())
    errors.extend(check_no_tracked_raw_data())
    errors.extend(check_imports())
    errors.extend(check_headline_numbers())

    if errors:
        print("SMOKE CHECK FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("SMOKE CHECK PASSED")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Scripts: {len(REQUIRED_SCRIPTS)}")
    print(f"  Core imports: {', '.join(CORE_IMPORTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
