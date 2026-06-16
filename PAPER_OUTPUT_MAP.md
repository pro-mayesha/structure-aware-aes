# Paper Output Map

Maps paper claims and tables to **exact repository files**. All paths are relative to the repository root. Precomputed outputs are included in git; re-running scripts regenerates the corresponding `outputs/` subdirectory.

## Main result table (official test, TF-IDF + Ridge)

**Conditions:** T0, T1, T2, T3, POS, POS_TRANS, T3_PLUS_POS

| Paper element | Repository file | Notes |
|---------------|-----------------|-------|
| Per-condition QWK, RMSE, MAE, accuracies | `outputs/final_heldout_evaluation/final_results.csv` | Rows `condition`, `split=official_test` |
| Narrative + decision | `outputs/final_heldout_evaluation/final_report.md` | Decision: **CLAIM SUPPORTED** |
| Task-type breakdown | `outputs/final_heldout_evaluation/task_type_results.csv` | Independent vs Text dependent |
| Score confusion matrices | `outputs/final_heldout_evaluation/confusion_matrices.csv` | Per condition |

**Headline numbers (from shipped outputs, do not edit):**

| Metric | Value | Source row |
|--------|-------|------------|
| T3_PLUS_POS QWK | 0.8488 | `final_results.csv`, condition `T3_PLUS_POS` |
| T2 QWK | 0.8361 | `final_results.csv`, condition `T2` |
| ΔQWK (T3_PLUS_POS − T2) | +0.0128 | `primary_comparisons.csv`, comparison `T3_PLUS_POS_vs_T2` |
| Bootstrap 95% CI | [+0.0092, +0.0162] | same row, `bootstrap_ci_lower` / `bootstrap_ci_upper` |
| P(ΔQWK > 0) | 1.000 | same row, `bootstrap_p_positive` |

## Primary comparisons (bootstrap on official test)

| Paper element | Repository file |
|---------------|-----------------|
| T3_PLUS_POS vs T2, T3, POS, POS_TRANS | `outputs/final_heldout_evaluation/primary_comparisons.csv` |
| Full bootstrap detail | `outputs/final_heldout_evaluation/bootstrap_comparisons.csv` |

## Falsification controls (official test)

| Control | Seeds | Repository file |
|---------|-------|-----------------|
| Shuffled rhetorical labels (TS) | 13, 21, 42, 87, 101 | `outputs/final_heldout_evaluation/random_control_results.csv` (`control=TS`) |
| Random boundaries (RB) | same | same (`control=RB`) |
| Capacity-matched random features (RND) | same | same (`control=RND`) |
| Mean-seed vs ensemble aggregation | — | same file, `aggregation` column |

Primary contrasts vs controls: `primary_comparisons.csv` rows `T3_PLUS_POS_vs_mean_TS`, `_RB`, `_RND`.

Development-phase control validation: `outputs/structure_verification/control_validation.csv`, `outputs/structure_stress_tests/random_control_validation.csv`.

## Prompt-level result

| Claim | Repository file | Verification |
|-------|-----------------|--------------|
| T3_PLUS_POS beats T2 on **15/15** prompts | `outputs/final_heldout_evaluation/primary_comparisons.csv` | Row `T3_PLUS_POS_vs_T2`: `prompts_positive=15`, `prompts_total=15` |
| Per-prompt QWK by condition | `outputs/final_heldout_evaluation/prompt_results.csv` | Compare `condition` T3_PLUS_POS vs T2 per `prompt_name` |

## Dataset validation & exclusions

| Claim | Repository file | Verification |
|-------|-----------------|--------------|
| **15,591** official-train essays (join-valid) | `outputs/final_heldout_evaluation/final_results.csv` | `n_train` column |
| **10,401** official-test essays (join-valid) | same | `n_test` column |
| **2** excluded join-invalid essays | `outputs/persuade20_audit/join_validation.csv` | `join_valid=False` (2 rows) |
| **99.99%** join-valid rate | `outputs/persuade20_audit/persuade20_audit_summary.md` | Section 2 |
| Exclusion reasons | `outputs/persuade20_audit/join_validation.csv` | `exclusion_reason` column |

## Protocol & feature audit

| Paper element | Repository file |
|---------------|-----------------|
| Held-out protocol checks | `outputs/final_heldout_evaluation/protocol_validation.csv` |
| Frozen feature families | `outputs/structure_verification/feature_family_manifest.csv` |
| Implementation audit | `outputs/structure_verification/implementation_audit.csv` |

## Development pipeline (train-only; supports paper methods section)

| Step | Report | Results |
|------|--------|---------|
| Feasibility | `outputs/structure_feasibility/feasibility_report.md` | `outputs/structure_feasibility/results.csv` |
| Stress tests | `outputs/structure_stress_tests/stress_test_report.md` | `outputs/structure_stress_tests/results.csv` |
| Verification | `outputs/structure_verification/verification_report.md` | `outputs/structure_verification/validation_results.csv` |
| LOO prompt analysis | — | `outputs/structure_verification/leave_one_prompt_out.csv` |

## Transformer robustness (optional / supplementary only)

**Do not use as primary evidence.** Frozen-encoder DeBERTa on non-CUDA hardware; see limitations in `outputs/transformer_robustness/compute_report.md`.

| Paper element | Repository file |
|---------------|-----------------|
| Summary & decision | `outputs/transformer_robustness/transformer_robustness_report.md` | **TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED** |
| D0–D_RANDOM test metrics | `outputs/transformer_robustness/test_results.csv` |
| Primary comparisons | `outputs/transformer_robustness/primary_comparisons.csv` |
| Training logs | `outputs/transformer_robustness/logs/*.json` |

## Configs (frozen hyperparameters)

| Experiment | Config |
|------------|--------|
| Feasibility | `configs/structure_feasibility.yaml` |
| Stress tests | `configs/structure_stress_tests.yaml` |
| Verification | `configs/structure_verification.yaml` |
| **Final held-out** | `configs/final_heldout_evaluation.yaml` |
| Transformer (optional) | `configs/transformer_robustness.yaml` |

## Missing / not in repository

| Item | Status |
|------|--------|
| Raw PERSUADE 2.0 CSVs | **Not included** — obtain separately |
| `data/persuade20/processed/essays.csv` | **Not included** — generated by Step 1 audit |
| DeBERTa embedding cache | **Not included** — regenerated by Step 5 |
| Published PDF / camera-ready | **TODO** — add URL when available |
