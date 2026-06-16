# Step 4: Final Held-Out Evaluation Report

Official PERSUADE 2.0 test split — first and only model evaluation on held-out essays.

## Protocol summary

- **Train essays:** 15,591 join-valid official-train (2 invalid essays excluded)
- **Test essays:** 10,401 join-valid official-test
- **Ridge alpha (train internal validation):** 10.0
- **Bootstrap samples:** 10,000
- **Random-control seeds:** [13, 21, 42, 87, 101]

## Final test metrics (selected conditions)

| Condition | QWK | RMSE | MAE | Exact | Adjacent |
| --- | --- | --- | --- | --- | --- |
| T0 | 0.7364 | 0.6815 | 0.5410 | 0.542 | 0.972 |
| T1 | 0.8039 | 0.6015 | 0.4677 | 0.613 | 0.983 |
| T2 | 0.8361 | 0.5542 | 0.4317 | 0.646 | 0.990 |
| T3 | 0.8439 | 0.5440 | 0.4232 | 0.660 | 0.990 |
| POS | 0.8437 | 0.5432 | 0.4226 | 0.656 | 0.990 |
| POS_TRANS | 0.8448 | 0.5416 | 0.4207 | 0.660 | 0.990 |
| T3_PLUS_POS | 0.8488 | 0.5366 | 0.4167 | 0.668 | 0.990 |

## Research questions

### 1. Does the combined structural representation beat text and counts?

T3_PLUS_POS QWK=0.8488 vs T2 QWK=0.8361. ΔQWK=+0.0128 (95% CI [+0.0092, +0.0162], P(Δ>0)=1.000)

### 2. Does boundary geometry independently improve scoring?

POS QWK=0.8437 vs T2 QWK=0.8361. ΔQWK=+0.0077 (95% CI [+0.0046, +0.0107], P(Δ>0)=1.000)

### 3. Do rhetorical transitions add value beyond boundary geometry?

POS_TRANS QWK=0.8448 vs POS QWK=0.8437. ΔQWK=+0.0011 (point estimate). T3_PLUS_POS vs POS_TRANS: ΔQWK=+0.0040 (95% CI [+0.0015, +0.0065], P(Δ>0)=0.999)

### 4. Does genuine structure beat shuffled labels, random boundaries, and random capacity?

- vs mean TS: ΔQWK=+0.0095 (95% CI [+0.0065, +0.0125], P(Δ>0)=1.000)
- vs mean RB: ΔQWK=+0.0104 (95% CI [+0.0072, +0.0135], P(Δ>0)=1.000)
- vs mean RND: ΔQWK=+0.0126 (95% CI [+0.0093, +0.0160], P(Δ>0)=1.000)

### 5. Are improvements consistent across prompts and task types?

- T3_PLUS_POS beats T2 on **15/15** prompts by QWK.
- Independent: T3_PLUS_POS=0.8502, T2=0.8346, Δ=+0.0156
- Text dependent: T3_PLUS_POS=0.8103, T2=0.7969, Δ=+0.0133

### 6. Did the final test results support the frozen research claim?

> Genuine document-component boundary geometry and rhetorical transitions provide complementary predictive value for holistic essay scoring beyond essay text, surface features, and component counts.

- T3_PLUS_POS vs T3 (labels + geometry): ΔQWK=+0.0050 (95% CI [+0.0023, +0.0077], P(Δ>0)=1.000)
- T3 vs T2 (full labeled structure): ΔQWK=+0.0078 (95% CI [+0.0046, +0.0110], P(Δ>0)=1.000)

## Random-control aggregation

| control | aggregation | qwk | qwk_diff_vs_T3_PLUS_POS |
| --- | --- | --- | --- |
| TS | seed_level | 0.8390995921234399 | 0.00972857545768 |
| TS | seed_level | 0.8396240736216305 | 0.0092040939594894 |
| TS | seed_level | 0.838148271468703 | 0.0106798961124169 |
| TS | seed_level | 0.8408607786863355 | 0.0079673888947844 |
| TS | seed_level | 0.8387857250553141 | 0.0100424425258057 |
| TS | mean_seed_qwk | 0.8393036881910845 | 0.0095244793900354 |
| TS | ensemble_prediction_qwk | 0.8397158012908841 | 0.0091123662902358 |
| RB | seed_level | 0.8376927937530328 | 0.011135373828087 |
| RB | seed_level | 0.8384635571083788 | 0.010364610472741 |
| RB | seed_level | 0.8379497552131059 | 0.010878412368014 |
| RB | seed_level | 0.8396372765760292 | 0.0091908910050907 |
| RB | seed_level | 0.8385352525881318 | 0.0102929149929881 |
| RB | mean_seed_qwk | 0.8384557270477357 | 0.0103724405333841 |
| RB | ensemble_prediction_qwk | 0.8390175141496405 | 0.0098106534314793 |
| RND | seed_level | 0.8360760162805733 | 0.0127521513005466 |
| RND | seed_level | 0.8361959085807888 | 0.012632259000331 |
| RND | seed_level | 0.8367980049730768 | 0.0120301626080431 |
| RND | seed_level | 0.8356185410359079 | 0.0132096265452119 |
| RND | seed_level | 0.8362072250625013 | 0.0126209425186185 |
| RND | mean_seed_qwk | 0.8361791391865696 | 0.0126490283945502 |
| RND | ensemble_prediction_qwk | 0.8362320411815264 | 0.0125961263995935 |

## Protocol validation

| check | passed | detail |
| --- | --- | --- |
| official_test_first_use | True | No prior experiment script evaluates official-test essays |
| no_train_test_overlap | True | train=15591, test=10401 |
| two_invalid_essays_excluded | True | join_valid=False essays excluded via join_validation.csv |
| preprocessing_fit_train_only | True | TF-IDF and StandardScaler fit on official-train only |
| identical_test_essays_all_conditions | True | All conditions evaluated on same 10401 essays in identical order |
| ridge_alpha_from_train_validation | True | alpha=10.0 selected on train internal validation (T2) |
| frozen_tfidf_settings | True | {'ngram_range': [1, 2], 'max_features': 30000, 'min_df': 3} |
| bootstrap_samples_ge_10000 | True | 10000 |
| feature_manifest_match_all_conditions | True | {'T0': True, 'T1': True, 'T2': True, 'T3': True, 'POS': True, 'POS_TRANS': True, 'T3_PLUS_POS': True, 'TS': True, 'RB': True, 'RND': True} |
| random_controls_validated | True | TS/RB/RND seeds preserve control properties per Step 3.5 |
| no_post_test_protocol_changes | True | Feature definitions and hyperparameters frozen before test evaluation |
| random_control_property_checks | True | n_checks=400 |

## Decision: **CLAIM SUPPORTED**
