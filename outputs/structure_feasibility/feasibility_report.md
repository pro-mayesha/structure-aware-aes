# Structural Feasibility Report (PERSUADE 2.0)

## Setup

- Join-valid official-train essays only: **15,591**
- Train / validation: **12,472** / **3,119** (80/20, stratified by prompt × score)
- Official test set: **not used**
- Bootstrap samples: **2,000**

## Validation QWK by condition

| Condition | QWK | RMSE | MAE | Exact | Adjacent |
| --- | --- | --- | --- | --- | --- |
| N0 | 0.7721 | 0.6654 | 0.4983 | 0.5909 | 0.9689 |
| N1 | 0.8267 | 0.5830 | 0.4429 | 0.6412 | 0.9843 |
| N2 | 0.8324 | 0.5717 | 0.4344 | 0.6486 | 0.9849 |
| NS_s13 | 0.8252 | 0.5839 | 0.4449 | 0.6371 | 0.9849 |
| NS_s21 | 0.8217 | 0.5882 | 0.4469 | 0.6348 | 0.9827 |
| NS_s42 | 0.8244 | 0.5864 | 0.4468 | 0.6380 | 0.9833 |
| NS_s87 | 0.8205 | 0.5868 | 0.4461 | 0.6319 | 0.9836 |
| NS_s101 | 0.8222 | 0.5867 | 0.4464 | 0.6374 | 0.9830 |
| T0 | 0.7193 | 0.6874 | 0.5476 | 0.5348 | 0.9699 |
| T1 | 0.7996 | 0.6116 | 0.4689 | 0.6220 | 0.9792 |
| T2 | 0.8284 | 0.5723 | 0.4438 | 0.6448 | 0.9862 |
| T3 | 0.8370 | 0.5559 | 0.4308 | 0.6550 | 0.9888 |
| TS_s13 | 0.8308 | 0.5652 | 0.4379 | 0.6460 | 0.9865 |
| TS_s21 | 0.8300 | 0.5643 | 0.4364 | 0.6467 | 0.9865 |
| TS_s42 | 0.8279 | 0.5683 | 0.4409 | 0.6403 | 0.9872 |
| TS_s87 | 0.8285 | 0.5673 | 0.4405 | 0.6435 | 0.9862 |
| TS_s101 | 0.8290 | 0.5646 | 0.4374 | 0.6441 | 0.9865 |

## Primary comparisons (validation QWK difference, A − B)

| comparison | condition_a | condition_b | qwk_a | qwk_b | qwk_diff | prompts_positive | prompts_total | qwk_diff_mean | ci_lower | ci_upper | p_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2_vs_N1 | N2 | N1 | 0.8324401054974595 | 0.8266951727081251 | 0.0057449327893344115 | 10 | 15 | 0.005758167172443551 | -0.00027536683882620195 | 0.011537063421004745 | 0.97 |
| T3_vs_T2 | T3 | T2 | 0.8369936656783136 | 0.8284136452216019 | 0.00858002045671169 | 12 | 15 | 0.008603503915339568 | 0.003250079624581173 | 0.014126138324887905 | 1.0 |
| N2_vs_NS_s13 | N2 | NS_s13 | 0.8324401054974595 | 0.8252116541164132 | 0.00722845138104633 | 10 | 15 | 0.007247042866897998 | 0.0014612961644219935 | 0.013033423668788307 | 0.997 |
| T3_vs_TS_s13 | T3 | TS_s13 | 0.8369936656783136 | 0.8307901720431462 | 0.0062034936351673675 | 12 | 15 | 0.006182322177371084 | 0.0007658764023343561 | 0.011710450477408966 | 0.988 |
| N2_vs_NS_s21 | N2 | NS_s21 | 0.8324401054974595 | 0.821732036268785 | 0.01070806922867451 | 14 | 15 | 0.010695478199361407 | 0.005307563215942756 | 0.016071439043562275 | 1.0 |
| T3_vs_TS_s21 | T3 | TS_s21 | 0.8369936656783136 | 0.8299796984309425 | 0.007013967247371022 | 11 | 15 | 0.007039968011036795 | 0.001945576930665985 | 0.012334183665366663 | 0.9965 |
| N2_vs_NS_s42 | N2 | NS_s42 | 0.8324401054974595 | 0.8244380710929422 | 0.0080020344045173 | 12 | 15 | 0.007945132796250417 | 0.0026678625258652667 | 0.013557949545078675 | 0.999 |
| T3_vs_TS_s42 | T3 | TS_s42 | 0.8369936656783136 | 0.8278525022462566 | 0.009141163432056953 | 10 | 15 | 0.009188821937098435 | 0.0040131253485254596 | 0.014242048323533136 | 1.0 |
| N2_vs_NS_s87 | N2 | NS_s87 | 0.8324401054974595 | 0.8204830917707275 | 0.011957013726731969 | 14 | 15 | 0.011929691664312745 | 0.00596781476729176 | 0.018024861301454052 | 1.0 |
| T3_vs_TS_s87 | T3 | TS_s87 | 0.8369936656783136 | 0.8284644371874127 | 0.008529228490900875 | 12 | 15 | 0.008632839927277758 | 0.003343731704689451 | 0.01384040041256789 | 1.0 |
| N2_vs_NS_s101 | N2 | NS_s101 | 0.8324401054974595 | 0.8221513625351349 | 0.010288742962324626 | 12 | 15 | 0.010296814301583095 | 0.004296017490136977 | 0.015956039266939528 | 0.9995 |
| T3_vs_TS_s101 | T3 | TS_s101 | 0.8369936656783136 | 0.8290366355418478 | 0.00795703013646576 | 11 | 15 | 0.00800040238413907 | 0.003190520167231678 | 0.012941967535901884 | 0.9995 |

## Research questions

### 1. Does true rhetorical order outperform counts alone?

- **Numeric:** N2 QWK=0.8324 vs N1 QWK=0.8267 (Δ=+0.0057)
- **Text:** T3 QWK=0.8370 vs T2 QWK=0.8284 (Δ=+0.0086)

### 2. Does true rhetorical order outperform shuffled order?

- **Numeric:** N2 vs mean NS QWK=0.8228 (Δ=+0.0096)
- **Text:** T3 vs mean TS QWK=0.8292 (Δ=+0.0078)

### 3. Does order still help when essay text is already available?

- T3−T2 QWK difference: **+0.0086**
- Bootstrap 95% CI lower bound: **0.0033**

### 4. Are improvements consistent across prompts?

- T3 vs T2: **12/15** prompts show positive QWK difference (80.0%)
- N2 vs N1: **10/15** prompts show positive QWK difference

### 5. Should we proceed with the full structural-validity study?

## Decision: **GREEN**

Proceed with the full structural-validity study. True rhetorical order shows a reliable advantage over counts and shuffled controls when text is available.

### Decision criteria reference

- GREEN threshold: ≥0.005 QWK over T2 and mean TS, positive bootstrap CI, ≥60% prompts positive
- YELLOW: small or inconsistent advantage
- RED: true order ≈ shuffled order in numeric and text experiments
