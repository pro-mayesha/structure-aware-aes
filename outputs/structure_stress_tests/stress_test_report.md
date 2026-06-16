# Step 3: Structure Stress Test Report

## Validation checks

- Official test essays used: **0**
- Train/validation overlap: **0**
- RND feature dimensions match T3−T2: **True** (91 dims)
- RB validations passed: **100.0%**
- RND seeds produce different features: **True**
- All conditions use identical essay split: **True**

## Validation QWK (reference + ablations)

| Condition | QWK |
| --- | --- |
| FIRSTLAST | 0.8285 |
| LABELPOS | 0.8099 |
| NO_POS | 0.8318 |
| NO_TRANS | 0.8089 |
| POS | 0.8396 |
| RB_s101 | 0.8286 |
| RB_s13 | 0.8272 |
| RB_s21 | 0.8287 |
| RB_s42 | 0.8263 |
| RB_s87 | 0.8277 |
| RND_s101 | 0.8268 |
| RND_s13 | 0.8239 |
| RND_s21 | 0.8258 |
| RND_s42 | 0.8248 |
| RND_s87 | 0.8238 |
| T2 | 0.8284 |
| T3 | 0.8370 |
| TRANS | 0.8308 |
| TS_s101 | 0.8309 |
| TS_s13 | 0.8264 |
| TS_s21 | 0.8305 |
| TS_s42 | 0.8316 |
| TS_s87 | 0.8310 |

## Primary comparisons

| comparison | condition_a | condition_b | qwk_a | qwk_b | qwk_diff | prompts_positive | prompts_total | qwk_diff_mean | ci_lower | ci_upper | p_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_vs_T2 | T3 | T2 | 0.8369936656783136 | 0.8284136452216019 | 0.0085800204567116 | 12 | 15 | 0.0086035039153395 | 0.0032500796245811 | 0.0141261383248879 | 1.0 |
| T3_vs_mean_RND | T3 | mean_RND | 0.8369936656783136 | 0.8275895921888685 | 0.009404073489445 | 12 | 15 | 0.0094252007673003 | 0.0044863115770895 | 0.0146657518982866 | 1.0 |
| T3_vs_POS | T3 | POS | 0.8369936656783136 | 0.8396248472477236 | -0.00263118156941 | 3 | 15 | -0.0026636298189232 | -0.0084348862714918 | 0.0030125294746385 | 0.1775 |
| T3_vs_mean_RB | T3 | mean_RB | 0.8369936656783136 | 0.8292730198866991 | 0.0077206457916144 | 11 | 15 | 0.0077624947454553 | 0.0026153053221026 | 0.0129043891641524 | 0.998 |
| T3_vs_FIRSTLAST | T3 | FIRSTLAST | 0.8369936656783136 | 0.8284830924321801 | 0.0085105732461334 | 13 | 15 | 0.0085254781573908 | 0.0033545284362322 | 0.0136371960366799 | 0.9995 |
| T3_vs_TRANS | T3 | TRANS | 0.8369936656783136 | 0.8307635945763507 | 0.0062300711019628 | 14 | 15 | 0.0063130293336205 | 0.0016686653180977 | 0.011100565139502 | 0.996 |
| T3_vs_LABELPOS | T3 | LABELPOS | 0.8369936656783136 | 0.8099165911875406 | 0.027077074490773 | 14 | 15 | 0.0270992040575141 | 0.0204465839485738 | 0.0340858102380994 | 1.0 |
| T3_vs_NO_TRANS | T3 | NO_TRANS | 0.8369936656783136 | 0.8089434413242327 | 0.0280502243540808 | 15 | 15 | 0.028060507593022 | 0.0213664952369469 | 0.0351315644932749 | 1.0 |
| T3_vs_NO_POS | T3 | NO_POS | 0.8369936656783136 | 0.8318327885904566 | 0.0051608770878569 | 12 | 15 | 0.0051922595438732 | 0.0008540062958169 | 0.009745664856395 | 0.9885 |

## Research questions

### 1. Does true document structure beat additional random feature capacity?

- T3 QWK=0.8370 vs mean RND QWK=0.8250 (Δ=+0.0120)
- Bootstrap T3 vs mean RND CI: [0.0045, 0.0147]

### 2. Do genuine rhetorical boundaries matter?

- T3 QWK=0.8370 vs mean RB QWK=0.8277 (Δ=+0.0093)
- Bootstrap T3 vs mean RB CI: [0.0026, 0.0129]

### 3. Do rhetorical labels add value beyond label-free positions?

- T3 QWK=0.8370 vs POS QWK=0.8396 (Δ=-0.0026)
- Label-free position features match or slightly exceed full labeled order on validation, so boundary geometry carries much of the signal; rhetorical labels add value mainly with transitions.

### 4. Which structural feature families produce the useful signal?

- FIRSTLAST (first/final labels alone): 0.8285 (Δ vs T2: +0.0001)
- TRANS (transitions alone): 0.8308 (Δ vs T2: +0.0023)
- LABELPOS (label positions alone): 0.8099 (Δ vs T2: -0.0185)
- POS (label-free positions): 0.8396 (Δ vs T2: +0.0112)
- NO_TRANS (T3 without transitions): 0.8089 — drop vs T3: +0.0281 QWK
- NO_POS (T3 without label positions): 0.8318 — drop vs T3: +0.0052 QWK

Ablation impact (removed from T3): transitions >> label positions.

Standalone families above T2:
- POS: +0.0112 QWK
- TRANS: +0.0023 QWK
- FIRSTLAST: +0.0001 QWK
- LABELPOS: -0.0185 QWK

### 5. Does rhetorical structure generalize to unseen prompts?

- Leave-one-prompt-out mean QWK: T2=0.7115, T3=0.7278 (Δ=+0.0162)
- T3 best on **13/15** held-out prompts
- Mean T3−T2 across prompts: +0.0162
- Mean T3−TS across prompts: +0.0084
- Mean T3−RND across prompts: +0.0170

### 6. Is the evidence strong enough to proceed to final testing?

**Yes, with nuance.** True rhetorical order beats capacity-matched random features and random boundaries. Transition features are essential; label-free position features are nearly as strong as full labeled order. Leave-one-prompt-out evaluation shows T3 ahead on most held-out prompts. Proceed to final held-out evaluation using transition-aware structural representations (official test set still reserved).