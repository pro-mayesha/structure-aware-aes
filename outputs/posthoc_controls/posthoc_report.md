# POST-HOC Control Analysis

**POST-HOC.** Everything in this report was added after the frozen Step 4 held-out evaluation (`outputs/final_heldout_evaluation/`). The original pipeline, configs and outputs are unchanged. Produced by `scripts/run_posthoc_controls.py`.

## Setup

- Train essays: 15,591; test essays: 10,401 (official split, join-valid).
- TF-IDF and StandardScaler fit on train only (shared Step 4 code); Ridge alpha = 10.0.
- Seeds (20): original [13, 21, 42, 87, 101] plus [1, 2, 3, 5, 7, 11, 17, 19, 23, 29, 31, 37, 53, 71, 97].
- Paired bootstrap: 10,000 resamples of test essays (rng seed 42); in each resample the statistic is QWK(condition) − mean over the 20 seeds of QWK(control seed), the same aggregation as `bootstrap_mean_control_diff` in Step 4. `qwk_diff` is the full-test-set QWK(condition) − mean-of-seed QWK(control).
- Prompts improved: prompts where QWK(condition) > mean-of-seed control QWK (same rule as Step 4).

## Control reproducibility

The Step 4 controls (TS, RB, RND) derived their random seeds from Python's built-in `hash()` of essay ids (`shuffle_labels`, `rb_structure`) and of the split tag (`random_control_features`). String hashing is randomized per process and `PYTHONHASHSEED` was never set, so the original control draws could not be reproduced: rerunning the unchanged code gave different QWKs for 14 of 15 control seeds (column `rerun_original_code_qwk`; TS_s87 matched to 4 dp by chance), and two further processes with `PYTHONHASHSEED=1` and `=2` gave different values again.

All controls in this report are **regenerated after the held-out evaluation (post-hoc)** with deterministic seeding: `int(md5(key).hexdigest()[:8], 16)` with key `essay_id|split|seed` (TS, RB), `essay_id|split|seed|perm` (PERM), and `rnd|n_dims|split|seed` (RND-type controls, one RNG per split as in the original). Only the seed derivation changed; the perturbation logic is identical. Determinism was confirmed by fitting all ten controls for one seed in two separate processes with different `PYTHONHASHSEED` values (`determinism_check.csv`: identical QWKs).

Step 0 comparison (reported Step 4 values vs. this post-hoc run). T2/T3/POS/T3_PLUS_POS are deterministic and must match; control seeds are new draws and are not expected to match.

| condition | reported_qwk | rerun_original_code_qwk | posthoc_qwk | match_4dp |
| --- | --- | --- | --- | --- |
| T2 | 0.8361 | 0.8361 | 0.8361 | True |
| T3 | 0.8439 | 0.8439 | 0.8439 | True |
| POS | 0.8437 | 0.8437 | 0.8437 | True |
| T3_PLUS_POS | 0.8488 | 0.8488 | 0.8488 | True |
| TS_s13 | 0.8391 | 0.8416 | 0.8396 | False |
| TS_s21 | 0.8396 | 0.8385 | 0.8402 | False |
| TS_s42 | 0.8381 | 0.8413 | 0.8384 | False |
| TS_s87 | 0.8409 | 0.8409 | 0.8397 | False |
| TS_s101 | 0.8388 | 0.8393 | 0.8405 | False |
| RB_s13 | 0.8377 | 0.8393 | 0.8386 | False |
| RB_s21 | 0.8385 | 0.8379 | 0.8382 | False |
| RB_s42 | 0.8379 | 0.8394 | 0.8385 | False |
| RB_s87 | 0.8396 | 0.8385 | 0.8381 | False |
| RB_s101 | 0.8385 | 0.8384 | 0.8394 | False |
| RND_s13 | 0.8361 | 0.8340 | 0.8361 | True |
| RND_s21 | 0.8362 | 0.8357 | 0.8342 | False |
| RND_s42 | 0.8368 | 0.8350 | 0.8355 | False |
| RND_s87 | 0.8356 | 0.8360 | 0.8364 | False |
| RND_s101 | 0.8362 | 0.8364 | 0.8367 | False |

## Feature counts

T2 = 24 (surface + counts); order features = 91; geometry (`pos_feature_dict`) = 62 (max_ranks = 26). Counts verified on the built feature rows for every seed; train and test key sets are identical.

| condition | n_features |
| --- | --- |
| T2 | 24 |
| T3 | 115 |
| POS | 86 |
| T3_PLUS_POS | 177 |
| TS | 115 |
| RB | 115 |
| RND | 115 |
| POS_RB | 86 |
| POS_PERM | 86 |
| POS_RND | 86 |
| FULL_TS | 177 |
| FULL_RB | 177 |
| FULL_PERM | 177 |
| FULL_RND | 177 |

Note: `pos_comp_count` (in geometry) duplicates `count_total_components` (in T2). It is kept, as in the frozen T3_PLUS_POS/POS, so results stay comparable.

## Control definitions

- **TS**: T2 + order features from shuffled labels at true boundaries.
- **RB**: T2 + order features from `rb_structure` (shuffled labels, random contiguous boundaries).
- **RND**: T2 + 91 N(0,1) columns.
- **POS_RB**: T2 + geometry from the `rb_structure` boundaries (segments = the random components; no unannotated segments).
- **POS_PERM**: T2 + geometry after a length-preserving permutation: every segment (rhetorical component or Unannotated) keeps its length, the order of segments is randomly permuted, and the uncovered spacers between slots and the start offset stay in place.
- **POS_RND**: T2 + 62 N(0,1) columns.
- **FULL_TS**: T2 + TS order features + true geometry.
- **FULL_RB**: T2 + order and geometry both from the `rb_structure` boundaries.
- **FULL_PERM**: T2 + order and geometry both from the length-preserving permutation (labels travel with their components).
- **FULL_RND**: T2 + 153 N(0,1) columns.

Permutation checks: identity permutation reproduces true boundaries and geometry for all essays (failures = 0); multiset of (length, type) and first-start/last-end preserved: True.

## Control QWK across 20 seeds

| control | n_features | n_seeds | mean_qwk | sd_qwk | min_qwk | max_qwk | mean_qwk_original5 | ensemble_pred_qwk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TS | 115 | 20 | 0.8398 | 0.0008 | 0.8382 | 0.8416 | 0.8397 | 0.8411 |
| RB | 115 | 20 | 0.8387 | 0.0009 | 0.8370 | 0.8411 | 0.8386 | 0.8399 |
| RND | 115 | 20 | 0.8363 | 0.0009 | 0.8342 | 0.8374 | 0.8358 | 0.8363 |
| POS_RB | 86 | 20 | 0.8402 | 0.0004 | 0.8391 | 0.8409 | 0.8399 | 0.8403 |
| POS_PERM | 86 | 20 | 0.8402 | 0.0005 | 0.8395 | 0.8414 | 0.8401 | 0.8407 |
| POS_RND | 86 | 20 | 0.8364 | 0.0007 | 0.8352 | 0.8381 | 0.8361 | 0.8364 |
| FULL_TS | 177 | 20 | 0.8457 | 0.0009 | 0.8437 | 0.8476 | 0.8461 | 0.8477 |
| FULL_RB | 177 | 20 | 0.8417 | 0.0008 | 0.8404 | 0.8435 | 0.8423 | 0.8432 |
| FULL_PERM | 177 | 20 | 0.8423 | 0.0009 | 0.8407 | 0.8440 | 0.8422 | 0.8437 |
| FULL_RND | 177 | 20 | 0.8359 | 0.0008 | 0.8347 | 0.8374 | 0.8360 | 0.8359 |

## Step 1: T3 level

| comparison | features (a / control) | QWK a | QWK control (mean of 20) | diff | 95% CI | P(diff>0) | prompts improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T3 vs TS | 115 / 115 | 0.8439 | 0.8398 | +0.0040 | [+0.0016, +0.0065] | 0.9993 | 10/15 |
| T3 vs RB | 115 / 115 | 0.8439 | 0.8387 | +0.0052 | [+0.0025, +0.0079] | 1.0000 | 12/15 |
| T3 vs RND | 115 / 115 | 0.8439 | 0.8363 | +0.0076 | [+0.0046, +0.0105] | 1.0000 | 14/15 |

## Step 2: geometry level

| comparison | features (a / control) | QWK a | QWK control (mean of 20) | diff | 95% CI | P(diff>0) | prompts improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| POS vs POS_RB | 86 / 86 | 0.8437 | 0.8402 | +0.0035 | [+0.0010, +0.0061] | 0.9968 | 12/15 |
| POS vs POS_PERM | 86 / 86 | 0.8437 | 0.8402 | +0.0035 | [+0.0012, +0.0058] | 0.9985 | 14/15 |
| POS vs POS_RND | 86 / 86 | 0.8437 | 0.8364 | +0.0074 | [+0.0046, +0.0102] | 1.0000 | 15/15 |

## Step 3: full-model level

| comparison | features (a / control) | QWK a | QWK control (mean of 20) | diff | 95% CI | P(diff>0) | prompts improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T3_PLUS_POS vs FULL_TS | 177 / 177 | 0.8488 | 0.8457 | +0.0031 | [+0.0010, +0.0053] | 0.9975 | 12/15 |
| T3_PLUS_POS vs FULL_RB | 177 / 177 | 0.8488 | 0.8417 | +0.0071 | [+0.0042, +0.0099] | 1.0000 | 14/15 |
| T3_PLUS_POS vs FULL_PERM | 177 / 177 | 0.8488 | 0.8423 | +0.0065 | [+0.0039, +0.0091] | 1.0000 | 14/15 |
| T3_PLUS_POS vs FULL_RND | 177 / 177 | 0.8488 | 0.8359 | +0.0129 | [+0.0097, +0.0161] | 1.0000 | 15/15 |

## Notes on unexpected results

- Step 0: T2, T3, POS and T3_PLUS_POS reproduce the reported Step 4 QWKs exactly. Of the 15 original control seeds, RND_s13 happens to match the reported value to 4 dp in the regenerated run (it is a new draw); the others differ, as expected.
- FULL_TS (shuffled-label order features + true geometry) has mean QWK 0.8457, which is the closest of all controls to T3_PLUS_POS (0.8488). The T3_PLUS_POS margin over FULL_TS is +0.0031. In Step 4, the reported margin over the T3-level mean TS was +0.0095.
- POS_RB and POS_PERM (geometry from random or permuted boundaries) both have mean QWK 0.8402. That is 0.0041 above T2 (0.8361) and above the T3-level TS and RB means (0.8398, 0.8387).
- By construction, POS_RB has no gaps: RB boundaries tile the full text, so every gap feature is 0 and geometry varies only through component starts, ends and lengths.
- FULL_RND (153 random columns) has mean QWK 0.8359, slightly below T2 (0.8361). POS_RND (62 columns) is 0.8364 and RND (91 columns) is 0.8363.
- Seed SD is ≤ 0.0009 for every control. The mean over the original 5 seeds is within 0.0006 of the 20-seed mean for every control.
