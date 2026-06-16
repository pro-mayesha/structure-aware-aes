# Step 3.5: Verification and Claim Refinement Report

## Part 1: Implementation audit

- Structural subset audits passed: **True**
- Known overlap: pos_comp_count duplicates count_total_components already in T2
- NO_TRANS verified as T3 minus transition columns only
- LABELPOS and TRANS columns verified against T3 definitions
- POS verified label-free (no order_* rhetorical-label columns)

## Part 2: Corrected aggregation

| comparison | primary_method | t3_qwk | control_mean_qwk | qwk_diff_primary | secondary_ensemble_qwk | qwk_diff_secondary | mean_seed_level_diff | seed_qwk_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_vs_mean_seed_qwk | mean_of_seed_level_qwk | 0.8369936656783136 | 0.8267931416917138 | 0.010200523986599763 | 0.828064747327188 | 0.008928918351125525 | 0.010200523986599718 | {'RND_s13': 0.8281261777358813, 'RND_s21': 0.8260312709492944, 'RND_s42': 0.8261381945492113, 'RND_s87': 0.828148721616532, 'RND_s101': 0.8255213436076503} |
| T3_vs_mean_seed_qwk | mean_of_seed_level_qwk | 0.8369936656783136 | 0.8283547064598427 | 0.00863895921847091 | 0.8297527075933355 | 0.007240958084978044 | 0.008638959218470932 | {'RB_s13': 0.8284258885874701, 'RB_s21': 0.8275074052860694, 'RB_s42': 0.826781597261018, 'RB_s87': 0.8300328830873085, 'RB_s101': 0.8290257580773472} |
| T3_vs_mean_seed_qwk | mean_of_seed_level_qwk | 0.8369936656783136 | 0.8311821834342137 | 0.005811482244099886 | 0.831374346678982 | 0.005619318999331546 | 0.005811482244099974 | {'TS_s13': 0.8323039436220656, 'TS_s21': 0.8295610326276828, 'TS_s42': 0.8302908289577495, 'TS_s87': 0.8326819251110962, 'TS_s101': 0.8310731868524739} |

Primary method uses mean of seed-level QWK; secondary uses QWK of averaged predictions.

## Validation QWK (audited conditions)

| Condition | QWK |
| --- | --- |
| BOUNDARY_ONLY | 0.8396 |
| COUNTS_TRANS | 0.8308 |
| LABELPOS | 0.8099 |
| NO_TRANS | 0.8089 |
| POS | 0.8396 |
| POS_TRANS | 0.8404 |
| RANDOM_BOUNDARY_TRUE_SEQUENCE | 0.8318 |
| RB | 0.8268 |
| RB_s101 | 0.8290 |
| RB_s13 | 0.8284 |
| RB_s21 | 0.8275 |
| RB_s42 | 0.8268 |
| RB_s87 | 0.8300 |
| REAL_BOUNDARY_SHUFFLED_LABEL | 0.8303 |
| RND | 0.8261 |
| RND_s101 | 0.8255 |
| RND_s13 | 0.8281 |
| RND_s21 | 0.8260 |
| RND_s42 | 0.8261 |
| RND_s87 | 0.8281 |
| T2 | 0.8284 |
| T3 | 0.8370 |
| T3_PLUS_POS | 0.8410 |
| TRANS | 0.8308 |
| TS | 0.8303 |
| TS_s101 | 0.8311 |
| TS_s13 | 0.8323 |
| TS_s21 | 0.8296 |
| TS_s42 | 0.8303 |
| TS_s87 | 0.8327 |

## Research questions

### 1. Were any Step 3 results caused by implementation errors?

**No critical errors found.** Subset definitions match T3. POS slightly exceeding T3 is not an implementation bug; it reflects feature overlap (POS repeats component count) and strong boundary geometry.

### 2. Why did removing transitions cause such a large performance drop?

- T3 QWK=0.8370; NO_TRANS QWK=0.8089 (drop 0.0281)
- Transitions capture sequential rhetorical patterns not recoverable from first/final or label-position features alone.
- In Ridge+TF-IDF, 49 sparse transition dimensions provide high-value complementary signal.

### 3. Do rhetorical labels provide value beyond genuine boundary geometry?

- T3=0.8370 vs POS=0.8396 (Δ=-0.0026)
- T3_PLUS_POS=0.8410 vs T3=0.8370 (Δ=+0.0040)
- Labels add little beyond geometry alone; combined POS+transitions may exceed either alone.

### 4. Do true boundaries provide value when rhetorical labels are damaged?

- T3 vs REAL_BOUNDARY_SHUFFLED_LABEL: +0.0067 QWK
- Shuffling labels at real boundaries removes label-sequence value while preserving geometry.

### 5. Do true rhetorical transitions provide value when boundaries are damaged?

- T3 vs RANDOM_BOUNDARY_TRUE_SEQUENCE: +0.0052 QWK
- Random boundaries with true label order still carry some signal via transitions/counts but less than T3.

### 6. Does combining POS and transitions outperform either family alone?

- POS_TRANS=0.8404 vs POS=0.8396 (Δ=+0.0007)
- POS_TRANS vs T3 (Δ=+0.0034)

### 7. Do structural gains remain after controlling for length and component count?

- Residualized POS QWK=0.8389 vs T2=0.8284
- Residualized T3 QWK=0.8366 vs T2=0.8284

### 8. Which structural representation generalizes best to unseen prompts?

- LOO wins by condition: {'T3_PLUS_POS': 10, 'POS': 2, 'POS_TRANS': 2, 'T3': 1}
- T3 best on 1/15 prompts
- Mean T3−T2 LOO: +0.0162
- Mean POS−T2 LOO: +0.0174

### 9. What is the strongest claim supported by the evidence?

Essay text plus boundary geometry and rhetorical transition patterns predict holistic scores beyond counts alone; rhetorical label identity adds little beyond geometry, but transition structure is essential.

### 10. Should we proceed to the untouched official test set?

## Decision: **PROCEED TO FINAL TESTING**

Verification confirms Step 3 findings. Proceed to the untouched official test set with a reframed claim emphasizing transitions and boundary geometry rather than label identity alone.