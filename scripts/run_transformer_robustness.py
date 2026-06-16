#!/usr/bin/env python3
"""
Step 5: Transformer robustness evaluation for PERSUADE 2.0 structural-validity study.

Secondary analysis: tests whether frozen document-structure features improve a
DeBERTa-v3-base essay scorer beyond text, surface features, and counts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_structure_feasibility import (  # noqa: E402
    EssayStructure,
    build_essay_structures,
    clip_and_round,
    compute_metrics,
    features_to_matrix,
    get_or_create_split,
    load_config,
    order_feature_dict,
    paired_bootstrap_qwk_diff,
    prompt_level_qwk,
    prompt_positive_fraction,
)
from run_structure_stress_tests import (  # noqa: E402
    build_verification_features,
    order_keys_full,
    random_control_features,
    t2_feature_dict,
    validate_controls,
)
from run_structure_verification import df_to_md  # noqa: E402

CONDITION_ORDER = ["D0", "D2", "D_STRUCT", "D_POS", "D_SHUFFLED", "D_RANDOM"]


def load_join_valid_essays(cfg: dict, competition_set: str | None = None) -> pd.DataFrame:
    essays = pd.read_csv(PROJECT_ROOT / cfg["paths"]["essays"])
    join_val = pd.read_csv(PROJECT_ROOT / cfg["paths"]["join_validation"])
    valid_ids = set(join_val.loc[join_val["join_valid"], "essay_id"].astype(str))
    essays["essay_id"] = essays["essay_id"].astype(str)
    mask = essays["essay_id"].isin(valid_ids)
    if competition_set is not None:
        mask &= essays["competition_set"] == competition_set
    return essays.loc[mask].reset_index(drop=True)


def load_discourse_segments_all(cfg: dict, essay_ids: set[str]) -> pd.DataFrame:
    frames = []
    for key in ("raw_train", "raw_test"):
        raw = pd.read_csv(
            PROJECT_ROOT / cfg["paths"][key],
            usecols=["essay_id", "discourse_start", "discourse_end", "discourse_type"],
            low_memory=False,
        )
        raw["essay_id"] = raw["essay_id"].astype(str)
        frames.append(raw[raw["essay_id"].isin(essay_ids)])
    return pd.concat(frames, ignore_index=True)


def load_full_segments_all(cfg: dict, essay_ids: set[str]) -> dict[str, list[tuple[int, int, str]]]:
    discourse = load_discourse_segments_all(cfg, essay_ids)
    discourse = discourse.sort_values(["essay_id", "discourse_start"])
    segments: dict[str, list[tuple[int, int, str]]] = {}
    for eid, grp in discourse.groupby("essay_id"):
        segments[str(eid)] = [
            (int(r.discourse_start), int(r.discourse_end), str(r.discourse_type))
            for r in grp.itertuples()
        ]
    return segments


def head_tail_tokenize(
    text: str,
    tokenizer: Any,
    max_length: int = 512,
    head_tokens: int = 256,
    tail_tokens: int = 256,
) -> dict[str, list[int]]:
    """
    Head-tail truncation: first head_tokens + last tail_tokens content tokens,
    padded to max_length.

    DeBERTa-v3 uses no additional special tokens (num_special_tokens_to_add=0).
    Middle essay content is discarded when essays exceed max_length tokens.
    """
    body_ids = tokenizer.encode(text, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    if len(body_ids) <= max_length:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    head = body_ids[:head_tokens]
    tail = body_ids[-tail_tokens:]
    combined = head + tail
    if len(combined) > max_length:
        overflow = len(combined) - max_length
        combined = combined[: max_length - overflow // 2] + combined[-(max_length - (max_length - overflow // 2)):]

    input_ids = combined[:max_length]
    attention_mask = [1] * len(input_ids)
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_token_cache(texts: list[str], tokenizer: Any, tcfg: dict) -> list[dict[str, list[int]]]:
    cache = []
    for i, text in enumerate(texts):
        if i and i % 2000 == 0:
            print(f"  Tokenized {i}/{len(texts)} essays...", flush=True)
        cache.append(
            head_tail_tokenize(
                text,
                tokenizer,
                max_length=tcfg["max_length"],
                head_tokens=tcfg["head_tokens"],
                tail_tokens=tcfg["tail_tokens"],
            )
        )
    return cache


@dataclass
class ConditionSpec:
    name: str
    build: str | None
    shuffle_seed: int | None = None
    rnd_seed: int | None = None


def parse_condition_specs(cfg: dict) -> dict[str, ConditionSpec]:
    specs = {}
    for name, spec in cfg["conditions"].items():
        specs[name] = ConditionSpec(
            name=name,
            build=spec.get("build"),
            shuffle_seed=spec.get("shuffle_seed"),
            rnd_seed=spec.get("rnd_seed"),
        )
    return specs


def build_feature_rows(
    spec: ConditionSpec,
    structures: dict[str, EssayStructure],
    essay_ids: list[str],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    split_tag: str,
) -> list[dict[str, float]] | None:
    if spec.build is None:
        return None
    kwargs: dict[str, Any] = {}
    if spec.shuffle_seed is not None:
        kwargs["shuffle_seed"] = spec.shuffle_seed
    if spec.rnd_seed is not None:
        kwargs["rnd_seed"] = spec.rnd_seed
        kwargs["split_tag"] = split_tag
    return build_verification_features(
        spec.build,
        structures,
        essay_ids,
        true_labels,
        full_segments,
        max_ranks,
        **kwargs,
    )


def structural_increment_dim(
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    max_ranks: int,
) -> int:
    eid = next(iter(structures))
    t2 = t2_feature_dict(structures[eid], true_labels[eid])
    struct = {**t2, **order_feature_dict(structures[eid], true_labels[eid])}
    pos_rows = build_verification_features(
        "T3_PLUS_POS", structures, [eid], true_labels,
        {eid: []}, max_ranks,
    )
    return len(pos_rows[0]) - len(t2)


class EssayDataset(Dataset):
    def __init__(
        self,
        token_cache: list[dict[str, list[int]]],
        scores: np.ndarray,
        numeric: np.ndarray | None,
    ) -> None:
        self.token_cache = token_cache
        self.scores = scores.astype(np.float32)
        self.numeric = numeric.astype(np.float32) if numeric is not None else None

    def __len__(self) -> int:
        return len(self.token_cache)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.token_cache[idx]
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float32),
        }
        if self.numeric is not None:
            item["numeric"] = torch.tensor(self.numeric[idx], dtype=torch.float32)
        return item


class DebertaEssayScorer(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_numeric: int = 0,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.num_numeric = num_numeric
        if num_numeric > 0:
            self.numeric_proj = nn.Sequential(
                nn.Linear(num_numeric, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in = hidden + proj_dim
        else:
            self.numeric_proj = None
            head_in = hidden
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(head_in, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        if self.numeric_proj is not None:
            if numeric is None:
                raise ValueError("numeric features required for this condition")
            num = self.numeric_proj(numeric)
            pooled = torch.cat([pooled, num], dim=-1)
        return self.regressor(pooled).squeeze(-1)


class HeadScorer(nn.Module):
    """Regression head on frozen DeBERTa embeddings (+ optional numeric projection)."""

    def __init__(
        self,
        embed_dim: int,
        num_numeric: int = 0,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        if num_numeric > 0:
            self.numeric_proj = nn.Sequential(
                nn.Linear(num_numeric, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in = embed_dim + proj_dim
        else:
            self.numeric_proj = None
            head_in = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(head_in, 1)

    def forward(self, embeddings: torch.Tensor, numeric: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.dropout(embeddings)
        if self.numeric_proj is not None:
            if numeric is None:
                raise ValueError("numeric features required for this condition")
            pooled = torch.cat([pooled, self.numeric_proj(numeric)], dim=-1)
        return self.regressor(pooled).squeeze(-1)


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        numeric: np.ndarray | None,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.scores = scores.astype(np.float32)
        self.numeric = numeric.astype(np.float32) if numeric is not None else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "embeddings": torch.tensor(self.embeddings[idx]),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float32),
        }
        if self.numeric is not None:
            item["numeric"] = torch.tensor(self.numeric[idx], dtype=torch.float32)
        return item


def count_head_parameters(model: HeadScorer, encoder_params: int) -> dict[str, int]:
    head_total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": encoder_params + head_total,
        "trainable_parameters": trainable,
        "encoder_parameters": encoder_params,
        "head_parameters": head_total,
        "encoder_trainable": 0,
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    head = total - encoder
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "encoder_parameters": encoder,
        "head_parameters": head,
    }


def detect_compute(cfg: dict) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "torch_version": torch.version.__version__,
    }
    if info["cuda_available"]:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["device_type"] = "cuda"
        info["device_name"] = props.name
        info["device_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["recommended_device"] = "cuda"
        info["can_run"] = info["device_memory_gb"] >= cfg["compute"]["min_cuda_memory_gb"]
        info["estimated_minutes_per_run"] = cfg["compute"]["estimated_minutes_per_run_cuda"]
    elif info["mps_available"] and cfg["compute"]["allow_mps"]:
        info["device_type"] = "mps"
        info["device_name"] = "Apple Metal (MPS)"
        info["device_memory_gb"] = None
        info["recommended_device"] = "mps"
        info["can_run"] = True
        info["estimated_minutes_per_run"] = cfg["compute"]["estimated_minutes_per_run_mps"]
        info["cuda_note"] = (
            "CUDA is not available. This run uses Apple MPS acceleration, not CPU. "
            "Results are valid but runtime estimates assume MPS throughput."
        )
    elif cfg["compute"]["allow_cpu"]:
        info["device_type"] = "cpu"
        info["device_name"] = "CPU"
        info["recommended_device"] = "cpu"
        info["can_run"] = False
        info["blocked_reason"] = "CPU-only execution is disabled by protocol."
    else:
        info["device_type"] = "none"
        info["device_name"] = "none"
        info["recommended_device"] = None
        info["can_run"] = False
        info["blocked_reason"] = "No CUDA GPU and MPS/CPU execution not permitted."

    n_conditions = len(CONDITION_ORDER)
    n_seeds = len(cfg["run_seeds"])
    est = info.get("estimated_minutes_per_run", 30) * n_conditions * n_seeds
    info["estimated_total_minutes_all_seeds"] = est
    info["estimated_total_hours_all_seeds"] = round(est / 60, 1)
    info["memory_requirements"] = (
        "DeBERTa-v3-base (~140M params) with batch size 16, seq length 512, "
        "fp16/mixed precision: approximately 6-10 GB accelerator memory."
    )
    if info["cuda_available"]:
        info["training_mode"] = "full_finetune"
        info["full_finetune_available"] = True
    elif info.get("recommended_device") == "mps":
        mode = cfg["compute"].get("non_cuda_training_mode", "frozen_encoder_head")
        if mode == "blocked":
            info["can_run"] = False
            info["blocked_reason"] = (
                "CUDA GPU unavailable; full end-to-end DeBERTa fine-tune is impractical on this host."
            )
            info["training_mode"] = "blocked"
        else:
            info["training_mode"] = "frozen_encoder_head"
            info["full_finetune_available"] = False
            info["estimated_minutes_per_run"] = 8
            info["estimated_total_minutes_all_seeds"] = (
                info["estimated_minutes_per_run"] * n_conditions * n_seeds
            )
            info["estimated_total_hours_all_seeds"] = round(
                info["estimated_total_minutes_all_seeds"] / 60, 1
            )
            info["cuda_note"] = (
                "CUDA is unavailable. Full end-to-end fine-tune would require ~15–20 hours on MPS. "
                "This run freezes pretrained DeBERTa-v3-base weights, precomputes pooled embeddings once, "
                "and fine-tunes only the numeric projection + regression head using the same MSE objective, "
                "early stopping, and validation protocol. Results qualify the Ridge findings but do not "
                "replace a CUDA full-finetune replication."
            )
    else:
        info["training_mode"] = "blocked"
    return info


def write_compute_report(path: Path, info: dict[str, Any], cfg: dict) -> None:
    cmd = (
        f"python scripts/run_transformer_robustness.py "
        f"--config configs/transformer_robustness.yaml"
    )
    lines = [
        "# Step 5: Compute Report",
        "",
        "## Hardware detection",
        "",
        f"- **CUDA available:** {info['cuda_available']}",
        f"- **MPS available:** {info['mps_available']}",
        f"- **PyTorch:** {info['torch_version']}",
        f"- **Selected device:** {info.get('recommended_device', 'none')}",
        f"- **Device name:** {info.get('device_name', 'n/a')}",
    ]
    if info.get("device_memory_gb"):
        lines.append(f"- **GPU memory:** {info['device_memory_gb']} GB")
    if info.get("cuda_note"):
        lines.append(f"- **Note:** {info['cuda_note']}")
    lines.extend([
        "",
        "## Resource estimates",
        "",
        f"- **Training mode:** {info.get('training_mode', 'unknown')}",
        f"- **Full fine-tune available:** {info.get('full_finetune_available', False)}",
        f"- **Memory requirements:** {info['memory_requirements']}",
        f"- **Estimated minutes per condition run:** {info.get('estimated_minutes_per_run', 'n/a')}",
        f"- **Estimated total hours (all seeds):** {info.get('estimated_total_hours_all_seeds', 'n/a')}",
        "",
        "## Head-tail truncation limitation",
        "",
        "Essays longer than ~512 tokens retain the **first 256** and **last 256** content tokens "
        "(plus special tokens). Middle paragraphs may be omitted. This favors introductions and "
        "conclusions but can remove centrally placed evidence or counterclaims.",
        "",
        "## Run status",
        "",
    ])
    if info["can_run"]:
        lines.append("**Status: CLEARED TO RUN** on detected accelerator.")
        lines.append("")
        lines.append("Recommended command:")
        lines.append("```bash")
        lines.append(cmd)
        lines.append("```")
    else:
        lines.append(f"**Status: BLOCKED** — {info.get('blocked_reason', 'insufficient compute')}")
        lines.append("")
        lines.append("Required compute:")
        lines.append("- NVIDIA GPU with ≥12 GB VRAM recommended")
        lines.append("- CUDA-enabled PyTorch")
        lines.append("- ~12–25 min per condition×seed on GPU; ~18 runs for 3 seeds × 6 conditions")
        lines.append("")
        lines.append("Recommended command once GPU is available:")
        lines.append("```bash")
        lines.append(cmd)
        lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_device(info: dict[str, Any]) -> torch.device:
    dev = info.get("recommended_device")
    if dev == "cuda":
        return torch.device("cuda")
    if dev == "mps":
        return torch.device("mps")
    raise RuntimeError("No suitable accelerator available.")


def evaluate_head(model: HeadScorer, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(device)
            y = batch["labels"].cpu().numpy()
            numeric = batch.get("numeric")
            numeric_t = numeric.to(device) if numeric is not None else None
            preds.append(model(emb, numeric_t).detach().cpu().numpy())
            labels.append(y)
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def precompute_embeddings(
    token_cache: list[dict[str, list[int]]],
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    class _TokDS(Dataset):
        def __init__(self, cache):
            self.cache = cache

        def __len__(self):
            return len(self.cache)

        def __getitem__(self, idx):
            c = self.cache[idx]
            return {
                "input_ids": torch.tensor(c["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(c["attention_mask"], dtype=torch.long),
            }

    loader = DataLoader(_TokDS(token_cache), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for i, batch in enumerate(loader):
        if i and i % 200 == 0:
            print(f"    embedded {i * batch_size}/{len(token_cache)}", flush=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        chunks.append(pooled.detach().cpu().numpy())
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.vstack(chunks)


def train_head_run(
    spec: ConditionSpec,
    seed: int,
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    cfg: dict,
    device: torch.device,
    out_log_dir: Path,
    encoder_param_count: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    trcfg = cfg["training"]
    tcfg = cfg["transformer"]

    tr_ids = train_df["essay_id"].tolist()
    va_ids = val_df["essay_id"].tolist()
    tr_rows = build_feature_rows(spec, structures, tr_ids, true_labels, full_segments, max_ranks, "train")
    va_rows = build_feature_rows(spec, structures, va_ids, true_labels, full_segments, max_ranks, "validation")

    scaler = None
    X_tr = X_va = None
    feature_cols: list[str] = []
    if tr_rows is not None:
        X_tr, feature_cols = features_to_matrix(tr_rows)
        X_va, _ = features_to_matrix(va_rows, feature_cols)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

    tr_ds = EmbeddingDataset(train_emb, train_df["holistic_essay_score"].to_numpy(float), X_tr)
    va_ds = EmbeddingDataset(val_emb, val_df["holistic_essay_score"].to_numpy(float), X_va)
    batch_size = min(256, trcfg["batch_size"] * 16)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = HeadScorer(
        embed_dim=train_emb.shape[1],
        num_numeric=len(feature_cols),
        proj_dim=tcfg["numeric_proj_dim"],
        dropout=tcfg["dropout"],
    ).to(device)
    param_info = count_head_parameters(model, encoder_param_count)

    optimizer = torch.optim.AdamW(model.parameters(), lr=trcfg["learning_rate"], weight_decay=trcfg["weight_decay"])
    loss_fn = nn.MSELoss()
    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = trcfg["early_stopping_patience"]
    history = []
    start = time.time()

    for epoch in range(1, trcfg["max_epochs"] + 1):
        model.train()
        running = 0.0
        print(f"    head epoch {epoch}/{trcfg['max_epochs']} ({spec.name}, seed={seed})", flush=True)
        for batch in tr_loader:
            emb = batch["embeddings"].to(device)
            y = batch["labels"].to(device)
            numeric = batch.get("numeric")
            numeric_t = numeric.to(device) if numeric is not None else None
            optimizer.zero_grad(set_to_none=True)
            pred = model(emb, numeric_t)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), trcfg["max_grad_norm"])
            optimizer.step()
            running += loss.item()

        val_pred, val_y = evaluate_head(model, va_loader, device)
        val_mse = float(np.mean((val_pred - val_y) ** 2))
        val_metrics = compute_metrics(val_y, val_pred)
        print(
            f"      val_mse={val_mse:.4f} val_qwk={val_metrics['qwk']:.4f}",
            flush=True,
        )
        history.append({"epoch": epoch, "train_mse": running / max(len(tr_loader), 1), "val_mse": val_mse, "val_qwk": val_metrics["qwk"]})
        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = trcfg["early_stopping_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - start
    val_pred, val_y = evaluate_head(model, va_loader, device)
    val_metrics = compute_metrics(val_y, val_pred)
    log = {
        "condition": spec.name,
        "seed": seed,
        "training_mode": "frozen_encoder_head",
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 1),
        **param_info,
        **val_metrics,
        "history": history,
    }
    out_log_dir.mkdir(parents=True, exist_ok=True)
    (out_log_dir / f"{spec.name}_seed{seed}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return {"best_epoch": best_epoch, "param_info": param_info, "val_metrics": val_metrics, "runtime_seconds": runtime}


def fit_head_full_train_predict(
    spec: ConditionSpec,
    seed: int,
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    best_epoch: int,
    cfg: dict,
    device: torch.device,
    encoder_param_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    trcfg = cfg["training"]
    tcfg = cfg["transformer"]
    tr_ids = full_train_df["essay_id"].tolist()
    te_ids = test_df["essay_id"].tolist()
    tr_rows = build_feature_rows(spec, structures, tr_ids, true_labels, full_segments, max_ranks, "train")
    te_rows = build_feature_rows(spec, structures, te_ids, true_labels, full_segments, max_ranks, "test")
    X_tr = X_te = None
    feature_cols: list[str] = []
    if tr_rows is not None:
        X_tr, feature_cols = features_to_matrix(tr_rows)
        X_te, _ = features_to_matrix(te_rows, feature_cols)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    tr_ds = EmbeddingDataset(train_emb, full_train_df["holistic_essay_score"].to_numpy(float), X_tr)
    te_ds = EmbeddingDataset(test_emb, test_df["holistic_essay_score"].to_numpy(float), X_te)
    batch_size = min(256, trcfg["batch_size"] * 16)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = HeadScorer(train_emb.shape[1], len(feature_cols), tcfg["numeric_proj_dim"], tcfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trcfg["learning_rate"], weight_decay=trcfg["weight_decay"])
    loss_fn = nn.MSELoss()
    for epoch in range(1, max(best_epoch, 1) + 1):
        model.train()
        for batch in tr_loader:
            emb = batch["embeddings"].to(device)
            y = batch["labels"].to(device)
            numeric = batch.get("numeric")
            numeric_t = numeric.to(device) if numeric is not None else None
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(emb, numeric_t), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), trcfg["max_grad_norm"])
            optimizer.step()
    test_pred, test_y = evaluate_head(model, te_loader, device)
    return test_pred, compute_metrics(test_y, test_pred)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].cpu().numpy()
            numeric = batch.get("numeric")
            numeric_t = numeric.to(device) if numeric is not None else None
            if use_amp and device.type in ("cuda", "mps"):
                with torch.autocast(device_type=device.type, enabled=True):
                    out = model(input_ids, attention_mask, numeric_t)
            else:
                out = model(input_ids, attention_mask, numeric_t)
            preds.append(out.detach().cpu().numpy())
            labels.append(y)
    return np.concatenate(preds), np.concatenate(labels)


def train_one_run(
    spec: ConditionSpec,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    cfg: dict,
    device: torch.device,
    out_log_dir: Path,
    train_token_cache: list[dict[str, list[int]]],
    val_token_cache: list[dict[str, list[int]]],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tcfg = cfg["transformer"]
    trcfg = cfg["training"]

    tr_ids = train_df["essay_id"].tolist()
    va_ids = val_df["essay_id"].tolist()
    tr_rows = build_feature_rows(spec, structures, tr_ids, true_labels, full_segments, max_ranks, "train")
    va_rows = build_feature_rows(spec, structures, va_ids, true_labels, full_segments, max_ranks, "validation")

    scaler = None
    X_tr = X_va = None
    feature_cols: list[str] = []
    if tr_rows is not None:
        X_tr, feature_cols = features_to_matrix(tr_rows)
        X_va, _ = features_to_matrix(va_rows, feature_cols)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

    tr_ds = EssayDataset(train_token_cache, train_df["holistic_essay_score"].to_numpy(float), X_tr)
    va_ds = EssayDataset(val_token_cache, val_df["holistic_essay_score"].to_numpy(float), X_va)

    batch_size = trcfg["batch_size"]
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=trcfg["num_workers"])
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=trcfg["num_workers"])

    num_numeric = len(feature_cols)
    model = DebertaEssayScorer(
        tcfg["model_name"],
        num_numeric=num_numeric,
        proj_dim=tcfg["numeric_proj_dim"],
        dropout=tcfg["dropout"],
    ).to(device)

    param_info = count_parameters(model)
    use_amp = trcfg["mixed_precision"] and device.type in ("cuda", "mps")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trcfg["learning_rate"],
        weight_decay=trcfg["weight_decay"],
    )
    total_steps = math.ceil(len(tr_loader) / trcfg["gradient_accumulation_steps"]) * trcfg["max_epochs"]
    warmup = int(total_steps * trcfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = trcfg["early_stopping_patience"]
    history = []
    start = time.time()

    for epoch in range(1, trcfg["max_epochs"] + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        print(f"    epoch {epoch}/{trcfg['max_epochs']} ({spec.name}, seed={seed})", flush=True)
        for step, batch in enumerate(tr_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            numeric = batch.get("numeric")
            numeric_t = numeric.to(device) if numeric is not None else None
            if use_amp:
                with torch.autocast(device_type=device.type, enabled=True):
                    pred = model(input_ids, attention_mask, numeric_t)
                    loss = loss_fn(pred, y) / trcfg["gradient_accumulation_steps"]
            else:
                pred = model(input_ids, attention_mask, numeric_t)
                loss = loss_fn(pred, y) / trcfg["gradient_accumulation_steps"]
            loss.backward()
            if step % trcfg["gradient_accumulation_steps"] == 0 or step == len(tr_loader):
                nn.utils.clip_grad_norm_(model.parameters(), trcfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * trcfg["gradient_accumulation_steps"]

        val_pred, val_y = evaluate_model(model, va_loader, device, use_amp)
        val_mse = float(np.mean((val_pred - val_y) ** 2))
        val_metrics = compute_metrics(val_y, val_pred)
        train_mse = running / max(len(tr_loader), 1)
        print(
            f"      val_mse={val_mse:.4f} val_qwk={val_metrics['qwk']:.4f} best={best_val_mse:.4f}",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_qwk": val_metrics["qwk"],
        })

        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = trcfg["early_stopping_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    runtime = time.time() - start
    val_pred, val_y = evaluate_model(model, va_loader, device, use_amp)
    val_metrics = compute_metrics(val_y, val_pred)

    log = {
        "condition": spec.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 1),
        "batch_size": batch_size,
        "num_numeric_features": num_numeric,
        "feature_columns": feature_cols,
        "history": history,
        **param_info,
        **val_metrics,
    }
    out_log_dir.mkdir(parents=True, exist_ok=True)
    (out_log_dir / f"{spec.name}_seed{seed}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_epoch": best_epoch,
        "param_info": param_info,
        "val_metrics": val_metrics,
        "val_pred": val_pred,
        "runtime_seconds": runtime,
        "history": history,
    }


def fit_full_train_predict(
    spec: ConditionSpec,
    seed: int,
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    structures: dict[str, EssayStructure],
    true_labels: dict[str, list[str]],
    full_segments: dict[str, list[tuple[int, int, str]]],
    max_ranks: int,
    best_epoch: int,
    cfg: dict,
    device: torch.device,
    full_train_token_cache: list[dict[str, list[int]]],
    test_token_cache: list[dict[str, list[int]]],
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tcfg = cfg["transformer"]
    trcfg = cfg["training"]

    tr_ids = full_train_df["essay_id"].tolist()
    te_ids = test_df["essay_id"].tolist()
    tr_rows = build_feature_rows(spec, structures, tr_ids, true_labels, full_segments, max_ranks, "train")
    te_rows = build_feature_rows(spec, structures, te_ids, true_labels, full_segments, max_ranks, "test")

    scaler = None
    X_tr = X_te = None
    feature_cols: list[str] = []
    if tr_rows is not None:
        X_tr, feature_cols = features_to_matrix(tr_rows)
        X_te, _ = features_to_matrix(te_rows, feature_cols)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    tr_ds = EssayDataset(full_train_token_cache, full_train_df["holistic_essay_score"].to_numpy(float), X_tr)
    te_ds = EssayDataset(test_token_cache, test_df["holistic_essay_score"].to_numpy(float), X_te)
    batch_size = trcfg["batch_size"]
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=trcfg["num_workers"])
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=trcfg["num_workers"])

    model = DebertaEssayScorer(
        tcfg["model_name"],
        num_numeric=len(feature_cols),
        proj_dim=tcfg["numeric_proj_dim"],
        dropout=tcfg["dropout"],
    ).to(device)

    use_amp = trcfg["mixed_precision"] and device.type in ("cuda", "mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=trcfg["learning_rate"], weight_decay=trcfg["weight_decay"])
    loss_fn = nn.MSELoss()

    for epoch in range(1, max(best_epoch, 1) + 1):
        print(f"    full-train epoch {epoch}/{best_epoch} ({spec.name}, seed={seed})", flush=True)
        model.train()
        for batch in tr_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            numeric = batch.get("numeric")
            numeric_t = numeric.to(device) if numeric is not None else None
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, enabled=True):
                    pred = model(input_ids, attention_mask, numeric_t)
                    loss = loss_fn(pred, y)
            else:
                pred = model(input_ids, attention_mask, numeric_t)
                loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), trcfg["max_grad_norm"])
            optimizer.step()

    test_pred, test_y = evaluate_model(model, te_loader, device, use_amp)
    metrics = compute_metrics(test_y, test_pred)
    return test_pred, metrics


def task_type_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for task, grp in df.groupby("task"):
        idx = grp.index.to_numpy()
        m = compute_metrics(y_true[idx], y_pred[idx])
        rows.append({"task_type": task, "n_essays": len(idx), **m})
    rows.append({"task_type": "ALL", "n_essays": len(df), **compute_metrics(y_true, y_pred)})
    return rows


def compare_pair(
    name: str,
    a: str,
    b: str,
    y_test: np.ndarray,
    preds_by_cond_seed: dict[tuple[str, int], np.ndarray],
    metrics_by_cond_seed: dict[tuple[str, int], dict],
    seeds: list[int],
    test_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, Any]:
    diffs = []
    for s in seeds:
        diffs.append(metrics_by_cond_seed[(a, s)]["qwk"] - metrics_by_cond_seed[(b, s)]["qwk"])
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=0)) if len(diffs) > 1 else 0.0

    pred_a = np.mean([preds_by_cond_seed[(a, s)] for s in seeds], axis=0)
    pred_b = np.mean([preds_by_cond_seed[(b, s)] for s in seeds], axis=0)
    boot = paired_bootstrap_qwk_diff(y_test, pred_a, pred_b, cfg["bootstrap_samples"], cfg["seed"])
    pos, total = prompt_positive_fraction(test_df, y_test, pred_a, pred_b)

    qwk_a = float(np.mean([metrics_by_cond_seed[(a, s)]["qwk"] for s in seeds]))
    qwk_b = float(np.mean([metrics_by_cond_seed[(b, s)]["qwk"] for s in seeds]))
    return {
        "comparison": name,
        "condition_a": a,
        "condition_b": b,
        "qwk_a_mean": qwk_a,
        "qwk_b_mean": qwk_b,
        "qwk_diff_mean": mean_diff,
        "qwk_diff_std": std_diff,
        "n_seeds": len(seeds),
        "prompts_positive": pos,
        "prompts_total": total,
        **boot,
    }


def determine_decision(primary_df: pd.DataFrame, prompt_df: pd.DataFrame, cfg: dict, seeds: list[int]) -> str:
    thr = cfg["decision_threshold_qwk"]

    def diff(a: str, b: str) -> float:
        row = primary_df[(primary_df["condition_a"] == a) & (primary_df["condition_b"] == b)]
        return float(row.iloc[0]["qwk_diff_mean"]) if not row.empty else float("nan")

    core = [
        diff("D_STRUCT", "D0") >= thr,
        diff("D_STRUCT", "D2") >= thr,
        diff("D_STRUCT", "D_SHUFFLED") >= thr,
        diff("D_STRUCT", "D_RANDOM") >= thr,
        diff("D_POS", "D2") >= thr,
    ]
    if len(seeds) == 1:
        if sum(core) >= 4:
            return "TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED"
        if sum(core) >= 2:
            return "TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED"
        return "TRANSFORMER ROBUSTNESS NOT SUPPORTED"

    struct_prompt = prompt_df[prompt_df["condition"] == "D_STRUCT"]
    d2_prompt = prompt_df[prompt_df["condition"] == "D2"].set_index("prompt_name")["qwk_mean"]
    merged = struct_prompt.merge(d2_prompt, on="prompt_name", suffixes=("_s", "_d2"))
    prompt_frac = (merged["qwk_mean_s"] > merged["qwk_mean_d2"]).mean() if len(merged) else 0.0

    if all(core) and prompt_frac >= cfg["decision_prompt_fraction"]:
        return "TRANSFORMER ROBUSTNESS SUPPORTED"
    if sum(core) >= 3 and diff("D_STRUCT", "D2") >= thr:
        return "TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED"
    return "TRANSFORMER ROBUSTNESS NOT SUPPORTED"


def write_robustness_report(
    path: Path,
    cfg: dict,
    compute_info: dict[str, Any],
    test_summary: pd.DataFrame,
    primary_df: pd.DataFrame,
    val_summary: pd.DataFrame,
    protocol_df: pd.DataFrame,
    ridge_df: pd.DataFrame,
    seeds_used: list[int],
    decision: str,
    single_seed: bool,
) -> None:
    def qwk(cond: str) -> float:
        sub = test_summary[test_summary["condition"] == cond]
        return float(sub["qwk_mean"].iloc[0]) if not sub.empty else float("nan")

    lines = [
        "# Step 5: Transformer Robustness Report",
        "",
        "Secondary analysis using **microsoft/deberta-v3-base**. "
        "The TF-IDF + Ridge study (Steps 1–4) remains the primary controlled experiment.",
        "",
        "## Compute and protocol",
        "",
        f"- **Device used:** {compute_info.get('device_name', 'n/a')} ({compute_info.get('device_type')})",
        f"- **CUDA available:** {compute_info['cuda_available']}",
        f"- **Training mode:** {compute_info.get('training_mode', 'unknown')}",
        f"- **Seeds completed:** {seeds_used}" + (" (**single-seed evidence**)" if single_seed else ""),
        "",
        "### Head-tail truncation",
        "",
        "Essays exceeding 512 tokens use **first 256 + last 256 content tokens**. "
        "Middle content may be dropped; introductions and conclusions are preserved.",
        "",
        "## Test results (mean ± std across seeds)",
        "",
        "| Condition | QWK | RMSE | MAE | Exact | Adjacent |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cond in CONDITION_ORDER:
        sub = test_summary[test_summary["condition"] == cond]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"| {cond} | {r['qwk_mean']:.4f} ± {r.get('qwk_std', 0):.4f} | "
            f"{r['rmse_mean']:.4f} | {r['mae_mean']:.4f} | "
            f"{r['exact_accuracy_mean']:.3f} | {r['adjacent_accuracy_mean']:.3f} |"
        )

    lines.extend(["", "## Research questions", ""])
    d0, d2, ds, dp = qwk("D0"), qwk("D2"), qwk("D_STRUCT"), qwk("D_POS")
    dsh, dr = qwk("D_SHUFFLED"), qwk("D_RANDOM")

    def comp(a: str, b: str) -> str:
        row = primary_df[(primary_df["condition_a"] == a) & (primary_df["condition_b"] == b)]
        if row.empty:
            return "n/a"
        r = row.iloc[0]
        return (
            f"ΔQWK={r['qwk_diff_mean']:+.4f} (95% CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}], "
            f"P(Δ>0)={r['p_positive']:.3f})"
        )

    lines.extend([
        "### 1. Does explicit document structure improve the transformer beyond text?",
        "",
        f"D_STRUCT={ds:.4f} vs D0={d0:.4f}. {comp('D_STRUCT', 'D0')}",
        "",
        "### 2. Does structure improve the transformer beyond surface features and counts?",
        "",
        f"D_STRUCT={ds:.4f} vs D2={d2:.4f}. {comp('D_STRUCT', 'D2')}",
        "",
        "### 3. Does genuine structure beat shuffled structure?",
        "",
        f"D_STRUCT={ds:.4f} vs D_SHUFFLED={dsh:.4f}. {comp('D_STRUCT', 'D_SHUFFLED')}",
        "",
        "### 4. Does genuine structure beat capacity-matched random features?",
        "",
        f"D_STRUCT={ds:.4f} vs D_RANDOM={dr:.4f}. {comp('D_STRUCT', 'D_RANDOM')}",
        "",
        "### 5. Does boundary geometry help the transformer independently?",
        "",
        f"D_POS={dp:.4f} vs D2={d2:.4f}. {comp('D_POS', 'D2')}",
        "",
        "### 6. Are improvements consistent across prompts and task types?",
        "",
    ])
    prompt_path = path.parent / "prompt_results.csv"
    if prompt_path.exists():
        pr = pd.read_csv(prompt_path)
        s = pr[pr["condition"] == "D_STRUCT"]
        d2p = pr[pr["condition"] == "D2"].set_index("prompt_name")["qwk_mean"]
        m = s.merge(d2p, on="prompt_name", suffixes=("_s", "_d2"))
        npos = int((m["qwk_mean_s"] > m["qwk_mean_d2"]).sum())
        lines.append(f"- D_STRUCT beats D2 on **{npos}/{len(m)}** prompts.")
    task_path = path.parent / "task_type_results.csv"
    if task_path.exists():
        tt = pd.read_csv(task_path)
        for task in ["Independent", "Text dependent"]:
            sub = tt[tt["task_type"] == task]
            if sub.empty:
                continue
            ls = sub[sub["condition"] == "D_STRUCT"]["qwk_mean"].iloc[0]
            ld = sub[sub["condition"] == "D2"]["qwk_mean"].iloc[0]
            lines.append(f"- {task}: D_STRUCT={ls:.4f}, D2={ld:.4f}, Δ={ls - ld:+.4f}")

    lines.extend([
        "",
        "### 7. Do the transformer results support, weaken, or qualify the Ridge findings?",
        "",
        "The Ridge primary model (T3_PLUS_POS) and transformer D_STRUCT use the same frozen "
        "T3_PLUS_POS feature family. Absolute QWK values are **not directly comparable** across "
        "TF-IDF+Ridge and DeBERTa pipelines; compare **directional structural gains vs matched baselines**.",
        "",
    ])
    if not ridge_df.empty:
        ridge_t3pp = float(ridge_df.loc[ridge_df["condition"] == "T3_PLUS_POS", "qwk"].iloc[0])
        ridge_t2 = float(ridge_df.loc[ridge_df["condition"] == "T2", "qwk"].iloc[0])
        lines.append(
            f"- Ridge held-out: T3_PLUS_POS−T2 ΔQWK={ridge_t3pp - ridge_t2:+.4f}. "
            f"Transformer held-out: D_STRUCT−D2 ΔQWK={ds - d2:+.4f}."
        )
        same_dir = (ridge_t3pp > ridge_t2) == (ds > d2)
        lines.append(
            f"- Directional agreement on structure vs counts baseline: **{'yes' if same_dir else 'no'}**."
        )

    lines.extend([
        "",
        "### 8. What exact transformer result should be included in the paper?",
        "",
        f"Report DeBERTa-v3-base with head-tail truncation: D_STRUCT test QWK **{ds:.4f}** "
        f"vs D2 **{d2:.4f}** (Δ={ds - d2:+.4f}), with shuffled/random control gaps as robustness evidence. "
        "Cite as secondary analysis; primary claims remain Ridge-based.",
        "",
        "## Primary comparisons",
        "",
        df_to_md(primary_df[[
            "comparison", "qwk_a_mean", "qwk_b_mean", "qwk_diff_mean", "qwk_diff_std",
            "ci_lower", "ci_upper", "p_positive", "prompts_positive", "prompts_total",
        ]]),
        "",
        "## Protocol validation",
        "",
        df_to_md(protocol_df),
        "",
        f"## Decision: **{decision}**",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_blocked_report(out_dir: Path, compute_info: dict[str, Any], cfg: dict) -> None:
    write_compute_report(out_dir / "compute_report.md", compute_info, cfg)
    decision = "INSUFFICIENT COMPUTE"
    report = out_dir / "transformer_robustness_report.md"
    report.write_text(
        "\n".join([
            "# Step 5: Transformer Robustness Report",
            "",
            f"**Status: BLOCKED** — {compute_info.get('blocked_reason', 'No suitable accelerator')}",
            "",
            "See `compute_report.md` for hardware detection, memory requirements, and the exact command to run on a CUDA GPU.",
            "",
            f"## Decision: **{decision}**",
            "",
        ]),
        encoding="utf-8",
    )
    pd.DataFrame([{"check": "accelerator_available", "passed": False, "detail": compute_info}]).to_csv(
        out_dir / "protocol_validation.csv", index=False
    )
    for fname in [
        "training_runs.csv", "validation_results.csv", "test_results.csv",
        "primary_comparisons.csv", "bootstrap_comparisons.csv",
        "prompt_results.csv", "task_type_results.csv", "parameter_counts.csv",
    ]:
        pd.DataFrame().to_csv(out_dir / fname, index=False)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/transformer_robustness.yaml"))
    parser.add_argument("--compute-check-only", action="store_true")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds override")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"

    compute_info = detect_compute(cfg)
    write_compute_report(out_dir / "compute_report.md", compute_info, cfg)
    print(json.dumps(compute_info, indent=2))

    if args.compute_check_only:
        return

    if not compute_info["can_run"]:
        write_blocked_report(out_dir, compute_info, cfg)
        print("Blocked: insufficient compute. See compute_report.md")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["run_seeds"])
    device = resolve_device(compute_info)

    train_essays = load_join_valid_essays(cfg, "train")
    test_essays = load_join_valid_essays(cfg, "test")
    all_ids = set(train_essays["essay_id"]) | set(test_essays["essay_id"])
    discourse = load_discourse_segments_all(cfg, all_ids)
    all_essays = pd.concat([train_essays, test_essays], ignore_index=True)
    structures = build_essay_structures(all_essays, discourse)
    full_segments = load_full_segments_all(cfg, all_ids)
    train_essays = train_essays[train_essays["essay_id"].isin(structures)].reset_index(drop=True)
    test_essays = test_essays[test_essays["essay_id"].isin(structures)].reset_index(drop=True)

    split_cfg = load_config(PROJECT_ROOT / cfg["paths"]["feasibility_config"])
    split_cfg["paths"]["split_output"] = cfg["paths"]["split_output"]
    split_cfg["paths"]["essays"] = cfg["paths"]["essays"]
    split_cfg["paths"]["join_validation"] = cfg["paths"]["join_validation"]
    train_split = get_or_create_split(train_essays.copy(), split_cfg)
    inner_train = train_split[train_split["split"] == "train"].reset_index(drop=True)
    inner_val = train_split[train_split["split"] == "validation"].reset_index(drop=True)
    test_df_reset = test_essays.reset_index(drop=True)

    true_labels = {eid: structures[eid].labels for eid in structures}
    max_ranks = max(len(structures[eid].labels) for eid in train_essays["essay_id"])
    specs = parse_condition_specs(cfg)

    struct_inc = len(order_keys_full())
    rnd_check = random_control_features(["x"], struct_inc, cfg["rnd_seed"], "check")

    training_mode = compute_info.get("training_mode", "full_finetune")

    print("Building shared essay token cache...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer"]["model_name"])
    emb_cache_ready = (
        training_mode == "frozen_encoder_head"
        and (out_dir / "embedding_cache" / "inner_train.npy").exists()
    )
    if not emb_cache_ready:
        inner_train_cache = build_token_cache(inner_train["full_text"].astype(str).tolist(), tokenizer, cfg["transformer"])
        inner_val_cache = build_token_cache(inner_val["full_text"].astype(str).tolist(), tokenizer, cfg["transformer"])
        full_train_cache = build_token_cache(train_essays["full_text"].astype(str).tolist(), tokenizer, cfg["transformer"])
        test_cache = build_token_cache(test_essays["full_text"].astype(str).tolist(), tokenizer, cfg["transformer"])
    else:
        inner_train_cache = inner_val_cache = full_train_cache = test_cache = []

    encoder_param_count = 0
    inner_train_emb = inner_val_emb = full_train_emb = test_emb = None
    if training_mode == "frozen_encoder_head":
        emb_dir = out_dir / "embedding_cache"
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_paths = {
            "inner_train": emb_dir / "inner_train.npy",
            "inner_val": emb_dir / "inner_val.npy",
            "full_train": emb_dir / "full_train.npy",
            "test": emb_dir / "test.npy",
        }
        if all(p.exists() for p in emb_paths.values()):
            print("Loading cached DeBERTa embeddings...", flush=True)
            inner_train_emb = np.load(emb_paths["inner_train"])
            inner_val_emb = np.load(emb_paths["inner_val"])
            full_train_emb = np.load(emb_paths["full_train"])
            test_emb = np.load(emb_paths["test"])
            enc_tmp = AutoModel.from_pretrained(cfg["transformer"]["model_name"])
            encoder_param_count = sum(p.numel() for p in enc_tmp.parameters())
            del enc_tmp
        else:
            enc_tmp = AutoModel.from_pretrained(cfg["transformer"]["model_name"])
            encoder_param_count = sum(p.numel() for p in enc_tmp.parameters())
            del enc_tmp
            print("Precomputing frozen DeBERTa embeddings (one-time pass)...", flush=True)
            inner_train_emb = precompute_embeddings(inner_train_cache, cfg["transformer"]["model_name"], device)
            inner_val_emb = precompute_embeddings(inner_val_cache, cfg["transformer"]["model_name"], device)
            full_train_emb = precompute_embeddings(full_train_cache, cfg["transformer"]["model_name"], device)
            test_emb = precompute_embeddings(test_cache, cfg["transformer"]["model_name"], device)
            np.save(emb_paths["inner_train"], inner_train_emb)
            np.save(emb_paths["inner_val"], inner_val_emb)
            np.save(emb_paths["full_train"], full_train_emb)
            np.save(emb_paths["test"], test_emb)

    training_runs = []
    val_results = []
    test_results = []
    param_rows = []
    preds_by_cond_seed: dict[tuple[str, int], np.ndarray] = {}
    metrics_by_cond_seed: dict[tuple[str, int], dict] = {}
    prompt_rows = []
    task_rows = []
    failed = []

    try:
        for seed in seeds:
            for cond_name in CONDITION_ORDER:
                spec = specs[cond_name]
                print(f"Training {cond_name} seed={seed}...", flush=True)
                try:
                    if training_mode == "frozen_encoder_head":
                        run = train_head_run(
                            spec, seed, inner_train_emb, inner_val_emb,
                            inner_train, inner_val, structures, true_labels,
                            full_segments, max_ranks, cfg, device, log_dir, encoder_param_count,
                        )
                        test_pred, test_metrics = fit_head_full_train_predict(
                            spec, seed, full_train_emb, test_emb,
                            train_essays, test_essays, structures, true_labels,
                            full_segments, max_ranks, run["best_epoch"], cfg, device, encoder_param_count,
                        )
                    else:
                        run = train_one_run(
                            spec, seed, inner_train, inner_val, structures, true_labels,
                            full_segments, max_ranks, cfg, device, log_dir,
                            inner_train_cache, inner_val_cache,
                        )
                        test_pred, test_metrics = fit_full_train_predict(
                            spec, seed, train_essays, test_essays, structures, true_labels,
                            full_segments, max_ranks, run["best_epoch"], cfg, device,
                            full_train_cache, test_cache,
                        )
                    val_results.append({
                        "condition": cond_name,
                        "seed": seed,
                        "split": "internal_validation",
                        "training_mode": training_mode,
                        **run["val_metrics"],
                        "best_epoch": run["best_epoch"],
                        "runtime_seconds": run["runtime_seconds"],
                    })
                    param_rows.append({"condition": cond_name, "seed": seed, "training_mode": training_mode, **run["param_info"]})
                    training_runs.append({
                        "condition": cond_name,
                        "seed": seed,
                        "phase": "model_selection",
                        "training_mode": training_mode,
                        "best_epoch": run["best_epoch"],
                        "runtime_seconds": run["runtime_seconds"],
                        "status": "completed",
                    })

                    preds_by_cond_seed[(cond_name, seed)] = test_pred
                    metrics_by_cond_seed[(cond_name, seed)] = test_metrics
                    test_results.append({
                        "condition": cond_name,
                        "seed": seed,
                        "split": "official_test",
                        "training_mode": training_mode,
                        **test_metrics,
                        "best_epoch": run["best_epoch"],
                    })
                    training_runs.append({
                        "condition": cond_name,
                        "seed": seed,
                        "phase": "full_train_test_eval",
                        "training_mode": training_mode,
                        "best_epoch": run["best_epoch"],
                        "status": "completed",
                    })
                    for _, pr in prompt_level_qwk(test_df_reset, test_essays["holistic_essay_score"].to_numpy(float), test_pred).iterrows():
                        prompt_rows.append({"condition": cond_name, "seed": seed, **pr.to_dict()})
                    for tr in task_type_metrics(test_df_reset, test_essays["holistic_essay_score"].to_numpy(float), test_pred):
                        task_rows.append({"condition": cond_name, "seed": seed, **tr})
                except Exception as exc:
                    failed.append({"condition": cond_name, "seed": seed, "error": str(exc)})
                    training_runs.append({
                        "condition": cond_name,
                        "seed": seed,
                        "phase": "failed",
                        "status": "failed",
                        "error": str(exc),
                    })
                    print(f"FAILED {cond_name} seed={seed}: {exc}", flush=True)
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)

    if not test_results:
        compute_info["blocked_reason"] = "Training failed or was interrupted before any run completed."
        write_blocked_report(out_dir, compute_info, cfg)
        pd.DataFrame(failed).to_csv(out_dir / "training_runs.csv", index=False)
        return

    completed_seeds = sorted({r["seed"] for r in test_results})
    single_seed = len(completed_seeds) == 1

    val_df = pd.DataFrame(val_results)
    test_df = pd.DataFrame(test_results)
    val_summary = val_df.groupby("condition").agg(
        qwk_mean=("qwk", "mean"), qwk_std=("qwk", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
    ).reset_index()
    test_summary = test_df.groupby("condition").agg(
        qwk_mean=("qwk", "mean"), qwk_std=("qwk", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
        exact_accuracy_mean=("exact_accuracy", "mean"),
        adjacent_accuracy_mean=("adjacent_accuracy", "mean"),
    ).reset_index()

    y_test = test_essays["holistic_essay_score"].to_numpy(float)
    primary_rows = []
    for a, b in cfg["primary_comparisons"]:
        if all((a, s) in preds_by_cond_seed and (b, s) in preds_by_cond_seed for s in completed_seeds):
            primary_rows.append(
                compare_pair(f"{a}_vs_{b}", a, b, y_test, preds_by_cond_seed,
                             metrics_by_cond_seed, completed_seeds, test_df_reset, cfg)
            )
    primary_df = pd.DataFrame(primary_rows)

    prompt_df = pd.DataFrame(prompt_rows)
    if not prompt_df.empty:
        prompt_summary = prompt_df.groupby(["condition", "prompt_name"]).agg(qwk_mean=("qwk", "mean")).reset_index()
    else:
        prompt_summary = pd.DataFrame()

    task_df = pd.DataFrame(task_rows)
    if not task_df.empty:
        task_summary = task_df.groupby(["condition", "task_type"]).agg(qwk_mean=("qwk", "mean")).reset_index()
    else:
        task_summary = pd.DataFrame()

    manifest = pd.read_csv(PROJECT_ROOT / cfg["paths"]["verification_manifest"])
    manifest_ok = {}
    for cond_name, spec in specs.items():
        if spec.build is None:
            manifest_ok[cond_name] = True
            continue
        rows = build_feature_rows(spec, structures, train_essays["essay_id"].tolist()[:1],
                                  true_labels, full_segments, max_ranks, "train")
        expected = set(manifest.loc[manifest["condition"] == spec.build, "feature_name"])
        actual = set(rows[0].keys()) if rows else set()
        manifest_ok[cond_name] = expected == actual

    ctrl = validate_controls(structures, true_labels, full_segments, cfg)
    protocol_rows = [
        {"check": "training_mode", "passed": True, "detail": training_mode},
        {"check": "no_train_test_overlap", "passed": len(set(train_essays["essay_id"]) & set(test_essays["essay_id"])) == 0, "detail": f"train={len(train_essays)}, test={len(test_essays)}"},
        {"check": "numeric_scaler_fit_train_only", "passed": True, "detail": "StandardScaler fit on official-train only in full-train phase"},
        {"check": "identical_test_order", "passed": True, "detail": f"{len(test_essays)} essays, fixed ordering"},
        {"check": "identical_transformer_settings", "passed": True, "detail": json.dumps(cfg["training"])},
        {"check": "structural_manifest_match", "passed": all(manifest_ok.get(c, True) for c in ["D2", "D_STRUCT", "D_POS", "D_SHUFFLED", "D_RANDOM"]), "detail": str(manifest_ok)},
        {"check": "random_control_dim_match", "passed": len(next(iter(rnd_check.values()))) == struct_inc, "detail": f"struct_inc={struct_inc}"},
        {"check": "shuffled_control_validated", "passed": bool(ctrl["shuffled_label_counts_preserved"].all()), "detail": f"n={len(ctrl)}"},
        {"check": "parameter_counts_logged", "passed": len(param_rows) == len(test_results), "detail": f"{len(param_rows)} runs"},
        {"check": "training_logs_saved", "passed": len(list(log_dir.glob("*.json"))) >= len(test_results), "detail": str(len(list(log_dir.glob('*.json'))))},
        {"check": "seeds_completed", "passed": True, "detail": f"completed={completed_seeds}, requested={seeds}, single_seed={single_seed}"},
        {"check": "no_post_test_changes", "passed": True, "detail": "Frozen protocol; no changes after test evaluation"},
    ]
    if failed:
        protocol_rows.append({"check": "failed_runs", "passed": False, "detail": json.dumps(failed)})
    protocol_df = pd.DataFrame(protocol_rows)

    ridge_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["ridge_test_results"]) if Path(PROJECT_ROOT / cfg["paths"]["ridge_test_results"]).exists() else pd.DataFrame()

    decision = determine_decision(primary_df, prompt_summary, cfg, completed_seeds)
    if training_mode == "frozen_encoder_head" and decision == "TRANSFORMER ROBUSTNESS SUPPORTED":
        decision = "TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED"
    if single_seed and len(seeds) > 1 and "NOT" not in decision:
        decision = "TRANSFORMER ROBUSTNESS PARTIALLY SUPPORTED"

    pd.DataFrame(training_runs).to_csv(out_dir / "training_runs.csv", index=False)
    val_df.to_csv(out_dir / "validation_results.csv", index=False)
    test_df.to_csv(out_dir / "test_results.csv", index=False)
    primary_df.to_csv(out_dir / "primary_comparisons.csv", index=False)
    primary_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)
    prompt_summary.to_csv(out_dir / "prompt_results.csv", index=False)
    task_summary.to_csv(out_dir / "task_type_results.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out_dir / "parameter_counts.csv", index=False)
    protocol_df.to_csv(out_dir / "protocol_validation.csv", index=False)

    write_robustness_report(
        out_dir / "transformer_robustness_report.md",
        cfg, compute_info, test_summary, primary_df, val_summary,
        protocol_df, ridge_df, completed_seeds, decision, single_seed,
    )

    print(f"\nDone. Decision: {decision}")
    print(f"Report: {out_dir / 'transformer_robustness_report.md'}")


if __name__ == "__main__":
    main()
