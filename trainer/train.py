"""
train_gsa.py

Trains the GSA blocks inserted into the D-BETA ECG encoder while keeping
all pretrained D-BETA weights frozen.

Runs entirely within the D-BETA repo — does not require the ZETA repo or
M3AEModel. Only ECGTransformerModel (the ECG encoder) is loaded, which
avoids pulling in the T5 language model and cross-attention layers that
are irrelevant to GSA training and would waste GPU memory.

Conditions: NORM, LBBB, RBBB, 1dAVB
Data:        PTB-XL (500Hz, filename_hr, wfdb format)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
from dotenv import load_dotenv
from tqdm import tqdm

from models.transformer import ECGTransformerModel
from models.GSA import gsa_dice_loss, Attention_Maps

load_dotenv()
PTBXL_DIR      = os.getenv("PTBXL_DATASET")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

LABEL_TO_IDX = {"NORM": 0, "LBBB": 1, "RBBB": 2, "1dAVB": 3}
# SCP codes in PTB-XL that map to our four classes
SCP_TO_LABEL = {
    "NORM":  "NORM",
    "CLBBB": "LBBB",
    "CRBBB": "RBBB",
    "1AVB":  "1dAVB",
}
# Normal class index — attention loss is skipped for these samples
NORM_IDX = LABEL_TO_IDX["NORM"]


def ptbxl_cond_to_ids():
    """
    Reads ptbxl_database.csv and returns a dict mapping each condition
    to a list of (ecg_id, filename_hr) tuples.
    Only includes records where the condition confidence is 100.0 and
    the record belongs to exactly one of our four classes.
    """
    records = pd.read_csv(os.path.join(PTBXL_DIR, "ptbxl_database.csv"))
    organised_data = {k: [] for k in LABEL_TO_IDX}

    for _, record in records.iterrows():
        raw = record["scp_codes"].removeprefix("{").removesuffix("}")
        assigned = False
        for s in raw.split(","):
            cond, conf = s.split(":")
            cond = cond.replace("'", "").strip()
            conf = conf.strip()
            if conf != "100.0":
                continue
            if cond in SCP_TO_LABEL:
                label = SCP_TO_LABEL[cond]
                organised_data[label].append(
                    (int(record["ecg_id"]), record["filename_hr"])
                )
                assigned = True
                break   # one class per record

    for label, items in organised_data.items():
        logger.info(f"  {label}: {len(items)} records")

    return organised_data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PTBXLGSADataset(Dataset):
    """
    Loads PTB-XL wfdb records and returns:
        ecg            : (12, 5000) float32 tensor  — 12 leads, 500 Hz, 10 s
        attention_mask : (1, 5000) float32 tensor   — binary spatial guidance label
        label_idx      : int                        — class index
        is_normal      : bool                       — True for NORM class
    """

    SAMPLING_FREQ = 500
    TARGET_LENGTH = 5000    # 10 s × 500 Hz

    def __init__(self, samples):
        """
        Args:
            samples: list of (ecg_id, filename_hr, label_str) tuples
        """
        self.samples   = samples
        self.att_maps  = Attention_Maps()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg_id, filename_hr, label_str = self.samples[idx]
        label_idx = LABEL_TO_IDX[label_str]
        is_normal = label_idx == NORM_IDX

        # ── Load wfdb record ──────────────────────────────────────────────
        record_path = os.path.join(PTBXL_DIR, filename_hr)
        record = wfdb.rdrecord(record_path)
        # signal shape from wfdb: (n_samples, n_leads) → transpose to (n_leads, n_samples)
        ecg = record.p_signal.T.astype(np.float32)   # (12, L)

        # ── Crop / pad to TARGET_LENGTH ───────────────────────────────────
        L = ecg.shape[1]
        if L >= self.TARGET_LENGTH:
            ecg = ecg[:, :self.TARGET_LENGTH]
        else:
            ecg = np.pad(ecg, ((0, 0), (0, self.TARGET_LENGTH - L)))

        # ── Generate attention guidance label ─────────────────────────────
        # Use Lead II (index 1) as the reference lead for R-peak detection,
        # consistent with the guided attention paper.
        lead_ii = ecg[1]   # (5000,)

        if is_normal:
            # Normal class: no specific attention region — mask is all zeros.
            # The loss function skips attention loss for normal samples anyway,
            # but we still need a valid tensor of the right shape.
            attention_mask = np.zeros((self.TARGET_LENGTH, 1), dtype=np.float32)
        elif label_str == "1dAVB" or label_str == "1AVB":
            # 1AVB: attention on PR interval
            attention_mask = self.att_maps.generate_pr_mask(lead_ii, self.SAMPLING_FREQ)
        else:
            # LBBB / RBBB: attention on QRS complex
            attention_mask = self.att_maps.generate_qrs_mask(lead_ii, self.SAMPLING_FREQ)

        # Reshape from (L, 1) → (1, L) to match GSA output convention (B, 1, L)
        attention_mask = attention_mask.T.astype(np.float32)   # (1, 5000)

        return {
            "ecg":            torch.from_numpy(ecg),             # (12, 5000)
            "attention_mask": torch.from_numpy(attention_mask),  # (1, 5000)
            "label_idx":      label_idx,
            "is_normal":      is_normal,
        }


def build_datasets(val_fraction=0.15, seed=42):
    """
    Builds train and validation datasets from PTB-XL.
    Stratifies by class so each split has proportional class representation.
    """
    organised = ptbxl_cond_to_ids()

    # Flatten into a single list of (ecg_id, filename_hr, label_str)
    all_samples = []
    for label_str, items in organised.items():
        for ecg_id, filename_hr in items:
            all_samples.append((ecg_id, filename_hr, label_str))

    # Stratified split: split within each class then combine
    rng = np.random.default_rng(seed)
    train_samples, val_samples = [], []

    for label_str, items in organised.items():
        class_samples = [(ecg_id, fn, label_str) for ecg_id, fn in items]
        rng.shuffle(class_samples)
        n_val = max(1, int(len(class_samples) * val_fraction))
        val_samples.extend(class_samples[:n_val])
        train_samples.extend(class_samples[n_val:])

    logger.info(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    return PTBXLGSADataset(train_samples), PTBXLGSADataset(val_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────

def load_model_with_frozen_dbeta(config_path, checkpoint_path):
    """
    1. Instantiates ECGTransformerModel directly from the D-BETA config.
    2. Extracts only the ecg_encoder weights from the full D-BETA checkpoint
       and loads them (strict=False so missing GSA weights don't error —
       they are randomly initialised and will be trained).
    3. Freezes all parameters except the GSA blocks.

    Returns the ECGTransformerModel ready for GSA-only training.
    Using ECGTransformerModel directly rather than M3AEModel means we don't
    load the T5 language model or cross-attention layers, saving significant
    GPU memory since those components are irrelevant to GSA training.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg = SimpleNamespace(**cfg["model"])

    model = ECGTransformerModel(cfg)

    # The full D-BETA checkpoint stores all weights under "model" with keys
    # prefixed by "ecg_encoder." — strip the prefix to match ECGTransformerModel
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    full_state  = checkpoint["model"]

    ecg_state = {}
    for k, v in full_state.items():
        if k.startswith("ecg_encoder."):
            new_key = k[len("ecg_encoder."):]   # strip prefix
            # skip mask_emb — not used in inference/fine-tuning
            if new_key == "mask_emb":
                continue
            ecg_state[new_key] = v

    missing, unexpected = model.load_state_dict(ecg_state, strict=False)
    logger.info(f"Loaded ECG encoder weights.")
    logger.info(f"  Missing keys (expected — these are new GSA params): {len(missing)}")
    logger.info(f"  Unexpected keys: {len(unexpected)}")

    # ── Freeze everything except GSA blocks ───────────────────────────────
    frozen_count, trainable_count = 0, 0
    for name, param in model.named_parameters():
        if "gsa" in name.lower():
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    logger.info(f"Frozen parameters:    {frozen_count}")
    logger.info(f"Trainable parameters: {trainable_count}")

    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Training and validation steps
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, alpha, device):
    model.train()
    # Keep frozen D-BETA layers in eval mode so dropout doesn't interfere
    # with the frozen encoder representations during training
    for name, module in model.named_modules():
        if "gsa" not in name.lower():
            module.eval()

    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        ecg            = batch["ecg"].to(device)              # (B, 12, 5000)
        attention_mask = batch["attention_mask"].to(device)   # (B, 1, 5000)
        is_normal      = batch["is_normal"]                   # list of bools

        optimizer.zero_grad()

        # Call get_embeddings directly on ECGTransformerModel —
        # returns (features, padding_mask, attention_logits)
        _, _, attention_logits = model.get_embeddings(ecg, padding_mask=None)

        # Skip batches that are entirely Normal — no attention loss to compute
        non_normal_indices = [i for i, n in enumerate(is_normal) if not n]
        if len(non_normal_indices) == 0:
            n_batches += 1
            continue

        # Filter to non-normal samples only before computing loss
        filtered_logits = [logit[non_normal_indices] for logit in attention_logits]
        filtered_mask   = attention_mask[non_normal_indices]

        loss = gsa_dice_loss(filtered_logits, filtered_mask)

        loss.backward()
        # Gradient clipping — important since only GSA params are being updated
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        ecg            = batch["ecg"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        is_normal      = batch["is_normal"]

        _, _, attention_logits = model.get_embeddings(ecg, padding_mask=None)

        non_normal_indices = [i for i, n in enumerate(is_normal) if not n]
        if len(non_normal_indices) == 0:
            n_batches += 1
            continue

        filtered_logits = [logit[non_normal_indices] for logit in attention_logits]
        filtered_mask   = attention_mask[non_normal_indices]

        loss = gsa_dice_loss(filtered_logits, filtered_mask)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config_path   = "configs/config.json",
    checkpoint_path = CHECKPOINT_PATH,
    output_dir    = "checkpoints/gsa",
    n_epochs      = 30,
    batch_size    = 32,
    learning_rate = 1e-3,
    alpha         = 0.6,     # weight on classification loss — not used here
                              # since we're only training GSA blocks with Dice loss
    val_fraction  = 0.15,
    num_workers   = 4,
    seed          = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info("Building datasets...")
    train_dataset, val_dataset = build_datasets(val_fraction=val_fraction, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model, cfg = load_model_with_frozen_dbeta(config_path, checkpoint_path)
    model = model.to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    gsa_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(gsa_params, lr=learning_rate)
    # Reduce LR if validation loss plateaus for 5 epochs
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5 
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_ckpt_path = os.path.join(output_dir, "gsa_best.pt")

    logger.info(f"Starting training for {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        logger.info(f"Epoch {epoch}/{n_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, alpha, device)
        val_loss   = validate(model, val_loader, device)
        scheduler.step(val_loss)

        logger.info(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        # ── Save best checkpoint ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only the GSA block weights — the D-BETA weights are
            # unchanged so there's no need to re-save the full model
            gsa_state = {
                name: param
                for name, param in model.state_dict().items()
                if "gsa" in name.lower()
            }
            torch.save(
                {
                    "epoch":     epoch,
                    "gsa_state": gsa_state,
                    "val_loss":  best_val_loss,
                    "cfg":       vars(cfg),
                },
                best_ckpt_path,
            )
            logger.info(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # ── Save latest checkpoint every 5 epochs ────────────────────────
        if epoch % 5 == 0:
            latest_path = os.path.join(output_dir, f"gsa_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch":     epoch,
                    "gsa_state": {
                        name: param
                        for name, param in model.state_dict().items()
                        if "gsa" in name.lower()
                    },
                    "val_loss":  val_loss,
                    "cfg":       vars(cfg),
                },
                latest_path,
            )

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best checkpoint saved to: {best_ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading helper (for inference / ZETA integration)
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_model(config_path, dbeta_checkpoint_path, gsa_checkpoint_path):
    """
    Loads ECGTransformerModel with both D-BETA and trained GSA weights for
    inference or for passing to the ZETA pipeline.

    In the ZETA repo, after calling this function you can assign the returned
    model as the ecg_encoder on M3AEModel:
        zeta_model.ecg_encoder = load_trained_model(...)
    """
    model, cfg = load_model_with_frozen_dbeta(config_path, dbeta_checkpoint_path)

    gsa_checkpoint = torch.load(gsa_checkpoint_path, map_location="cpu")
    gsa_state      = gsa_checkpoint["gsa_state"]

    missing, unexpected = model.load_state_dict(gsa_state, strict=False)
    logger.info(f"Loaded GSA weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(
        config_path    = "configs/config.json",
        checkpoint_path = CHECKPOINT_PATH,
        output_dir     = "checkpoints/gsa",
        n_epochs       = 30,
        batch_size     = 32,
        learning_rate  = 1e-3,
        val_fraction   = 0.15,
        num_workers    = 4,
        seed           = 42,
    )