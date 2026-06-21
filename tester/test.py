"""
test_gsa.py

Evaluates a trained GSA checkpoint on the held-out test set saved during
training (see train_gsa.py — split_manifest.json).

This script never touches train/val data — it reloads the EXACT test split
that was carved out and saved before training began, via
load_split_from_manifest().

Run this AFTER training is complete.
"""

import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm

from trainer.train import (
    load_split_from_manifest,
    load_trained_model,
    LABEL_TO_IDX,
    NORM_IDX,
)
from models.GSA import gsa_dice_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


@torch.no_grad()
def evaluate_test_set(
    model,
    test_loader,
    device,
):
    """
    Computes:
      - Overall mean Dice loss across all non-Normal test samples
      - Per-class mean Dice loss (so you can see if any one condition's
        attention maps are notably worse than the others)

    Returns a dict of results.
    """
    model.eval()

    per_class_losses = {label: [] for label in LABEL_TO_IDX if label != "NORM"}
    all_losses = []

    for batch in tqdm(test_loader, desc="Evaluating test set"):
        ecg            = batch["ecg"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_idx      = batch["label_idx"]
        is_normal      = batch["is_normal"]

        _, _, attention_logits = model.get_embeddings(ecg, padding_mask=None)

        for i in range(len(ecg)):
            if is_normal[i]:
                continue   # no attention loss for Normal class — skip

            single_logits = [logit[i:i+1] for logit in attention_logits]
            single_mask   = attention_mask[i:i+1]

            loss = gsa_dice_loss(single_logits, single_mask).item()
            all_losses.append(loss)

            label_str = IDX_TO_LABEL[label_idx[i].item()]
            per_class_losses[label_str].append(loss)

    results = {
        "overall_mean_dice_loss": float(np.mean(all_losses)) if all_losses else None,
        "overall_std_dice_loss":  float(np.std(all_losses))  if all_losses else None,
        "n_samples_evaluated":    len(all_losses),
        "per_class": {},
    }

    for label, losses in per_class_losses.items():
        if len(losses) > 0:
            results["per_class"][label] = {
                "mean_dice_loss": float(np.mean(losses)),
                "std_dice_loss":  float(np.std(losses)),
                "n_samples":      len(losses),
            }
        else:
            results["per_class"][label] = {
                "mean_dice_loss": None,
                "std_dice_loss":  None,
                "n_samples": 0,
            }

    return results


def print_results(results):
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated: {results['n_samples_evaluated']}")
    print(f"Overall mean Dice loss: {results['overall_mean_dice_loss']:.4f} "
          f"(± {results['overall_std_dice_loss']:.4f})")
    print("\nPer-class breakdown:")
    for label, stats in results["per_class"].items():
        if stats["n_samples"] > 0:
            print(f"  {label:8s}: {stats['mean_dice_loss']:.4f} "
                  f"(± {stats['std_dice_loss']:.4f})  n={stats['n_samples']}")
        else:
            print(f"  {label:8s}: no samples")
    print("=" * 50)


def main(
    config_path          = "configs/config.json",
    dbeta_checkpoint_path = None,   # set via env var or pass explicitly
    gsa_checkpoint_path   = "checkpoints/gsa/gsa_best.pt",
    split_manifest_path   = "checkpoints/gsa/split_manifest.json",
    batch_size            = 32,
    num_workers           = 4,
    results_save_path     = "checkpoints/gsa/test_results.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if dbeta_checkpoint_path is None:
        dbeta_checkpoint_path = os.getenv("CHECKPOINT_PATH")
        assert dbeta_checkpoint_path is not None, (
            "dbeta_checkpoint_path not provided and CHECKPOINT_PATH env var not set"
        )

    # ── Load the exact test split saved during training ────────────────────
    logger.info(f"Loading test split from: {split_manifest_path}")
    _, _, test_dataset = load_split_from_manifest(split_manifest_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Load trained model (frozen D-BETA + trained GSA blocks) ────────────
    logger.info("Loading trained model...")
    model = load_trained_model(
        config_path, dbeta_checkpoint_path, gsa_checkpoint_path
    )
    model = model.to(device)

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = evaluate_test_set(model, test_loader, device)
    print_results(results)

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to: {results_save_path}")

    return results


if __name__ == "__main__":
    main()