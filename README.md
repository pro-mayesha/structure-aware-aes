# Structure-Aware AES (PERSUADE 2.0)

Document-structure validity study for holistic essay scoring on PERSUADE 2.0. Essays are represented as structured documents with discourse components (Lead, Position, Claim, Evidence, etc.).

**Primary experiment:** TF-IDF + Ridge (Steps 1–4).  
**Secondary robustness:** DeBERTa-v3-base (Step 5).

## Data setup

Download [PERSUADE 2.0](https://github.com/scenario-nlp/REALISE) train and test CSVs and place them in `data/`:

- `data/persuade_corpus_2.0_train.csv`
- `data/persuade_corpus_2.0_test.csv`

Symlinks are created automatically under `data/persuade20/raw/` when you run the audit script.

## Pipeline

```bash
# Step 1: Dataset audit
python scripts/audit_persuade20.py

# Step 2: Feasibility screening (train only)
python scripts/run_structure_feasibility.py

# Step 3: Stress tests
python scripts/run_structure_stress_tests.py

# Step 3.5: Verification
python scripts/run_structure_verification.py

# Step 4: Final held-out evaluation (first official-test use)
python scripts/run_final_heldout_evaluation.py

# Step 5: Transformer robustness (requires PyTorch + transformers)
python scripts/run_transformer_robustness.py
```

## Key results

- **Step 4 decision:** CLAIM SUPPORTED (`outputs/final_heldout_evaluation/final_report.md`)
- **Step 5 decision:** TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED (`outputs/transformer_robustness/transformer_robustness_report.md`)

## Requirements

- Python 3.11+
- `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`
- Step 5: `torch`, `transformers` (CUDA recommended for full DeBERTa fine-tune; MPS uses frozen-encoder fallback)
