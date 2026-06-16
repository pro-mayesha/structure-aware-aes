# Step 5: Transformer Robustness Report

Secondary analysis using **microsoft/deberta-v3-base**. The TF-IDF + Ridge study (Steps 1–4) remains the primary controlled experiment.

## Compute and protocol

- **Device used:** Apple Metal (MPS) (mps)
- **CUDA available:** False
- **Training mode:** frozen_encoder_head
- **Seeds completed:** [13, 42, 101]

### Head-tail truncation

Essays exceeding 512 tokens use **first 256 + last 256 content tokens**. Middle content may be dropped; introductions and conclusions are preserved.

## Test results (mean ± std across seeds)

| Condition | QWK | RMSE | MAE | Exact | Adjacent |
| --- | --- | --- | --- | --- | --- |
| D0 | 0.0028 ± 0.0405 | 1.7177 | 1.3573 | 0.240 | 0.621 |
| D2 | -0.0057 ± 0.0475 | 1.5467 | 1.2089 | 0.275 | 0.689 |
| D_STRUCT | 0.0566 ± 0.0095 | 1.8134 | 1.4854 | 0.193 | 0.563 |
| D_POS | 0.0800 ± 0.0475 | 1.3671 | 1.0719 | 0.305 | 0.746 |
| D_SHUFFLED | 0.0141 ± 0.0233 | 1.6274 | 1.2811 | 0.262 | 0.658 |
| D_RANDOM | 0.0023 ± 0.0305 | 1.6364 | 1.2866 | 0.263 | 0.656 |

## Research questions

### 1. Does explicit document structure improve the transformer beyond text?

D_STRUCT=0.0566 vs D0=0.0028. ΔQWK=+0.0711 (95% CI [+0.0644, +0.0778], P(Δ>0)=1.000)

### 2. Does structure improve the transformer beyond surface features and counts?

D_STRUCT=0.0566 vs D2=-0.0057. ΔQWK=+0.0496 (95% CI [+0.0391, +0.0600], P(Δ>0)=1.000)

### 3. Does genuine structure beat shuffled structure?

D_STRUCT=0.0566 vs D_SHUFFLED=0.0141. ΔQWK=+0.0258 (95% CI [+0.0179, +0.0335], P(Δ>0)=1.000)

### 4. Does genuine structure beat capacity-matched random features?

D_STRUCT=0.0566 vs D_RANDOM=0.0023. ΔQWK=+0.0335 (95% CI [+0.0257, +0.0413], P(Δ>0)=1.000)

### 5. Does boundary geometry help the transformer independently?

D_POS=0.0800 vs D2=-0.0057. ΔQWK=+0.0414 (95% CI [+0.0302, +0.0523], P(Δ>0)=1.000)

### 6. Are improvements consistent across prompts and task types?

- D_STRUCT beats D2 on **11/15** prompts.
- Independent: D_STRUCT=0.0643, D2=-0.0331, Δ=+0.0974
- Text dependent: D_STRUCT=0.1346, D2=0.0659, Δ=+0.0687

### 7. Do the transformer results support, weaken, or qualify the Ridge findings?

The Ridge primary model (T3_PLUS_POS) and transformer D_STRUCT use the same frozen T3_PLUS_POS feature family. Absolute QWK values are **not directly comparable** across TF-IDF+Ridge and DeBERTa pipelines; compare **directional structural gains vs matched baselines**.

- Ridge held-out: T3_PLUS_POS−T2 ΔQWK=+0.0128. Transformer held-out: D_STRUCT−D2 ΔQWK=+0.0623.
- Directional agreement on structure vs counts baseline: **yes**.

### 8. What exact transformer result should be included in the paper?

Report DeBERTa-v3-base with head-tail truncation: D_STRUCT test QWK **0.0566** vs D2 **-0.0057** (Δ=+0.0623), with shuffled/random control gaps as robustness evidence. Cite as secondary analysis; primary claims remain Ridge-based.

## Primary comparisons

| comparison | qwk_a_mean | qwk_b_mean | qwk_diff_mean | qwk_diff_std | ci_lower | ci_upper | p_positive | prompts_positive | prompts_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_STRUCT_vs_D0 | 0.05660036268919063 | 0.0028199106541174266 | 0.07106027849740722 | 0.02786348576293273 | 0.06441589033581437 | 0.0778427129598472 | 1.0 | 14 | 15 |
| D_STRUCT_vs_D2 | 0.05660036268919063 | -0.005654831711906283 | 0.049575831219817144 | 0.04422607028249173 | 0.03907025943309041 | 0.06004492182564767 | 1.0 | 8 | 15 |
| D_STRUCT_vs_D_POS | 0.05660036268919063 | 0.07999185159198767 | 0.008176806109029116 | 0.03496283963822125 | -0.002159091885189373 | 0.018407485661544737 | 0.9383 | 6 | 15 |
| D_STRUCT_vs_D_SHUFFLED | 0.05660036268919063 | 0.014091600479193259 | 0.02578902636753137 | 0.02627566109645157 | 0.01789537246137062 | 0.033505666069676046 | 1.0 | 11 | 15 |
| D_STRUCT_vs_D_RANDOM | 0.05660036268919063 | 0.0023159585195286305 | 0.03347277072999217 | 0.031622561529191046 | 0.025663518193129654 | 0.04132265423192497 | 1.0 | 12 | 15 |
| D_POS_vs_D2 | 0.07999185159198767 | -0.005654831711906283 | 0.041399025110788024 | 0.04744811726268425 | 0.03017695998464199 | 0.052329382863316856 | 1.0 | 12 | 15 |
| D_SHUFFLED_vs_D2 | 0.014091600479193259 | -0.005654831711906283 | 0.02378680485228578 | 0.037843740631077295 | 0.013626599830334844 | 0.03401063328577532 | 1.0 | 6 | 15 |

## Protocol validation

| check | passed | detail |
| --- | --- | --- |
| training_mode | True | frozen_encoder_head |
| no_train_test_overlap | True | train=15591, test=10401 |
| numeric_scaler_fit_train_only | True | StandardScaler fit on official-train only in full-train phase |
| identical_test_order | True | 10401 essays, fixed ordering |
| identical_transformer_settings | True | {"max_epochs": 5, "early_stopping_patience": 2, "learning_rate": 2e-05, "weight_decay": 0.01, "batch_size": 16, "max_batch_size": 16, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0, "mixed_precision": true, "num_workers": 0, "warmup_ratio": 0.1} |
| structural_manifest_match | False | {'D0': True, 'D2': True, 'D_STRUCT': False, 'D_POS': True, 'D_SHUFFLED': True, 'D_RANDOM': True} |
| random_control_dim_match | True | struct_inc=91 |
| shuffled_control_validated | True | n=400 |
| parameter_counts_logged | True | 18 runs |
| training_logs_saved | True | 18 |
| seeds_completed | True | completed=[13, 42, 101], requested=[42, 13, 101], single_seed=False |
| no_post_test_changes | True | Frozen protocol; no changes after test evaluation |

## Decision: **TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED**
