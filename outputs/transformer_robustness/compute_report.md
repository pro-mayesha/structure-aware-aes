# Step 5: Compute Report

## Hardware detection

- **CUDA available:** False
- **MPS available:** True
- **PyTorch:** 2.10.0
- **Selected device:** mps
- **Device name:** Apple Metal (MPS)
- **Note:** CUDA is unavailable. Full end-to-end fine-tune would require ~15–20 hours on MPS. This run freezes pretrained DeBERTa-v3-base weights, precomputes pooled embeddings once, and fine-tunes only the numeric projection + regression head using the same MSE objective, early stopping, and validation protocol. Results qualify the Ridge findings but do not replace a CUDA full-finetune replication.

## Resource estimates

- **Training mode:** frozen_encoder_head
- **Full fine-tune available:** False
- **Memory requirements:** DeBERTa-v3-base (~140M params) with batch size 16, seq length 512, fp16/mixed precision: approximately 6-10 GB accelerator memory.
- **Estimated minutes per condition run:** 8
- **Estimated total hours (all seeds):** 2.4

## Head-tail truncation limitation

Essays longer than ~512 tokens retain the **first 256** and **last 256** content tokens (plus special tokens). Middle paragraphs may be omitted. This favors introductions and conclusions but can remove centrally placed evidence or counterclaims.

## Run status

**Status: CLEARED TO RUN** on detected accelerator.

Recommended command:
```bash
python scripts/run_transformer_robustness.py --config configs/transformer_robustness.yaml
```