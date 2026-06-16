# Reviewer Checklist

Use this list when assessing reproducibility and transparency for the DocEng / NLP-for-education submission.

## Repository artifacts

- [ ] **Scripts** — All experiment stages have entry points under `scripts/` (audit, feasibility, stress, verification, final held-out, optional transformer).
- [ ] **Configs** — Frozen YAML configs under `configs/` match the paper methods section.
- [ ] **Main outputs** — Precomputed Step 4 results ship in `outputs/final_heldout_evaluation/`.
- [ ] **Paper mapping** — `PAPER_OUTPUT_MAP.md` links each claim/table to a specific output file.
- [ ] **Reproducibility guide** — `REPRODUCIBILITY.md` documents setup, data placement, and run order.

## Data policy

- [ ] **Raw PERSUADE 2.0 not in git** — `.gitignore` blocks `data/persuade20/raw/*.csv` and top-level corpus files.
- [ ] **Data README** — `data/persuade20/raw/README.md` explains required filenames and symlinks.
- [ ] **Reviewer can reproduce after obtaining data** — Steps 1→4 pipeline is documented; `make reproduce-all` runs the main chain.

## Scientific transparency

- [ ] **Oracle structure stated** — README and reproducibility docs state that human discourse annotations are used (upper-bound analysis).
- [ ] **Not SOTA neural AES** — README states TF-IDF + Ridge is intentional for controlled feature comparison.
- [ ] **Single official-test use** — Step 4 is the first evaluation on the official test split; protocol validation CSV documents checks.
- [ ] **Falsification controls** — Shuffled labels, random boundaries, and random features mapped in `PAPER_OUTPUT_MAP.md` to `random_control_results.csv`.
- [ ] **Transformer supplementary** — Step 5 marked optional; frozen-encoder caveat in `transformer_robustness_report.md`.

## Quick verification (no dataset)

```bash
pip install -r requirements.txt
python scripts/smoke_check.py
```

Expected: all structure checks pass; core imports succeed.

## Quick verification (with dataset)

```bash
make setup
make audit
make reproduce-all    # Steps 1–4; long-running
```

Compare regenerated `outputs/final_heldout_evaluation/final_results.csv` to the shipped copy (metrics should match under the frozen protocol).

## Key numbers to spot-check

| Claim | File | Field |
|-------|------|-------|
| T3_PLUS_POS QWK = 0.8488 | `outputs/final_heldout_evaluation/final_results.csv` | `qwk`, condition `T3_PLUS_POS` |
| T2 QWK = 0.8361 | same | condition `T2` |
| ΔQWK +0.0128 | `outputs/final_heldout_evaluation/primary_comparisons.csv` | `T3_PLUS_POS_vs_T2` → `qwk_diff` |
| 15/15 prompts | same row | `prompts_positive` / `prompts_total` |
| n_train = 15591, n_test = 10401 | `final_results.csv` | `n_train`, `n_test` |
| 2 excluded essays | `outputs/persuade20_audit/join_validation.csv` | `join_valid=False` count |

## License & citation

- [ ] **LICENSE** — MIT License present.
- [ ] **CITATION.cff** — Author metadata and repository URL present.

## Open items (authors)

- [ ] Add camera-ready PDF / DOI when published.
- [ ] Update `CITATION.cff` conference field when acceptance is confirmed.
- [ ] Optional: full CUDA DeBERTa fine-tune for stronger Step 5 evidence.
