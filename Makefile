.PHONY: setup setup-transformer smoke audit reproduce-feasibility reproduce-stress reproduce-verification reproduce-main reproduce-controls reproduce-all reproduce-transformer compile

PYTHON ?= python3
PIP ?= pip

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup-transformer: setup
	$(PIP) install -r requirements-transformer.txt

smoke:
	$(PYTHON) scripts/smoke_check.py

compile:
	$(PYTHON) -m compileall scripts

# Step 1: dataset audit (requires raw PERSUADE CSVs in data/persuade20/raw/)
audit:
	$(PYTHON) scripts/audit_persuade20.py

# Step 2: feasibility screening — official train only, never touches official test
reproduce-feasibility:
	$(PYTHON) scripts/run_structure_feasibility.py --config configs/structure_feasibility.yaml

# Step 3: stress tests
reproduce-stress:
	$(PYTHON) scripts/run_structure_stress_tests.py --config configs/structure_stress_tests.yaml

# Step 3.5: verification & claim refinement
reproduce-verification:
	$(PYTHON) scripts/run_structure_verification.py --config configs/structure_verification.yaml

# Step 4: final held-out evaluation (paper main result)
reproduce-main:
	$(PYTHON) scripts/run_final_heldout_evaluation.py --config configs/final_heldout_evaluation.yaml

# Steps 2–3.5: train-only development & falsification controls
reproduce-controls: reproduce-feasibility reproduce-stress reproduce-verification

# Steps 1–4: full primary pipeline
reproduce-all: audit reproduce-controls reproduce-main

# Step 5: optional transformer robustness (install requirements-transformer.txt first)
reproduce-transformer:
	$(PYTHON) scripts/run_transformer_robustness.py --config configs/transformer_robustness.yaml

# Compute check only (no training)
transformer-compute-check:
	$(PYTHON) scripts/run_transformer_robustness.py --compute-check-only
