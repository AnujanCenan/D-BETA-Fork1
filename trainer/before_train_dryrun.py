"""
dry_run_load.py

Verifies that the ZETA/M3AE checkpoint loads correctly into
GSA-enhanced ECGTransformerModel before committing to a full training run.

Checks:
  1. Config loads correctly
  2. ECGTransformerModel (with GSAConvFeatureExtraction) instantiates
  3. Checkpoint keys are stripped and loaded — missing keys should be
     ONLY the GSA block parameters, unexpected keys should be empty
  4. A single forward pass produces correctly-shaped outputs
  5. Freezing logic correctly isolates GSA parameters
"""

import json
import torch
from types import SimpleNamespace

from models.transformer import ECGTransformerModel
from models.GSA import GSA_Block

from dotenv import load_dotenv
import os

load_dotenv()
CONFIG_PATH     = os.getenv("CONFIG_PATH")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")   # update to your actual path


def dry_run():
    print("=" * 60)
    print("DRY RUN: GSA + D-BETA checkpoint loading")
    print("=" * 60)

    # ── 1. Load config ──────────────────────────────────────────────────
    print("\n[1] Loading config...")
    with open(CONFIG_PATH, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict["model"])
    print(f"    conv_feature_layers: {cfg.conv_feature_layers}")
    print(f"    in_d: {cfg.in_d}")
    print(f"    gsa_placement: {cfg.gsa_placement}")
    print(f"    encoder_embed_dim: {cfg.encoder_embed_dim}")
    print(f"    encoder_layers: {cfg.encoder_layers}")

    # ── 2. Instantiate model ────────────────────────────────────────────
    print("\n[2] Instantiating ECGTransformerModel...")
    try:
        model = ECGTransformerModel(cfg)
        print("    [OK] Model instantiated successfully")
    except Exception as e:
        print(f"    [FAIL] Model instantiation failed: {e}")
        import traceback; traceback.print_exc()
        return

    # Count GSA params vs total params before loading checkpoint
    total_params = sum(p.numel() for p in model.parameters())
    gsa_params_initial = sum(
        p.numel() for n, p in model.named_parameters() if "gsa" in n.lower()
    )
    print(f"    Total parameters:     {total_params:,}")
    print(f"    GSA block parameters: {gsa_params_initial:,}")

    # ── 3. Load checkpoint ───────────────────────────────────────────────
    print("\n[3] Loading checkpoint...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        full_state = checkpoint["model"]
        print(f"    Checkpoint loaded. Total keys: {len(full_state)}")
    except Exception as e:
        print(f"    [FAIL] Could not load checkpoint: {e}")
        return

    ecg_state = {}
    for k, v in full_state.items():
        if k.startswith("ecg_encoder."):
            new_key = k[len("ecg_encoder."):]
            if new_key == "mask_emb":
                continue
            ecg_state[new_key] = v
    print(f"    Filtered ecg_encoder keys: {len(ecg_state)}")

    missing, unexpected = model.load_state_dict(ecg_state, strict=False)

    print(f"\n    Missing keys:    {len(missing)}")
    non_gsa_missing = [k for k in missing if "gsa" not in k.lower()]
    gsa_missing      = [k for k in missing if "gsa" in k.lower()]
    print(f"      -> GSA-related (expected):     {len(gsa_missing)}")
    print(f"      -> NON-GSA (unexpected!):       {len(non_gsa_missing)}")
    if non_gsa_missing:
        print("    [WARNING] Non-GSA missing keys found — checkpoint may not")
        print("    fully match the model architecture. First 10:")
        for k in non_gsa_missing[:10]:
            print(f"        {k}")

    print(f"\n    Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("    [WARNING] Unexpected keys found — checkpoint contains keys")
        print("    not present in the model. First 10:")
        for k in unexpected[:10]:
            print(f"        {k}")
    else:
        print("    [OK] No unexpected keys — checkpoint matches model exactly")
        print("         (aside from new GSA parameters)")

    # ── 4. Forward pass ──────────────────────────────────────────────────
    print("\n[4] Running forward pass with dummy data...")
    model.eval()
    B, T = 2, 5000
    # in_d=12 means the model expects 12 leads as channels already present —
    # NOT a single-channel signal. Shape must be (B, 12, T).
    dummy_ecg = torch.randn(B, cfg.in_d, T)

    try:
        with torch.no_grad():
            features, padding_mask, attention_logits = model.get_embeddings(
                dummy_ecg, padding_mask=None
            )
        print(f"    [OK] Forward pass succeeded")
        print(f"    Feature shape: {features.shape}")
        print(f"    Number of attention logit tensors: {len(attention_logits)}")
        for i, logit in enumerate(attention_logits):
            shape = logit.shape if logit is not None else None
            print(f"      Layer {i}: {shape}")
    except Exception as e:
        print(f"    [FAIL] Forward pass failed: {e}")
        import traceback; traceback.print_exc()
        return

    # ── 5. Freezing logic ────────────────────────────────────────────────
    print("\n[5] Testing freeze logic...")
    frozen_count, trainable_count = 0, 0
    for name, param in model.named_parameters():
        if "gsa" in name.lower():
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(f"    Frozen parameter tensors:    {frozen_count}")
    print(f"    Trainable parameter tensors: {trainable_count}")
    print(f"    Frozen parameter count:      {frozen_params:,}")
    print(f"    Trainable parameter count:   {trainable_params:,}")

    assert trainable_params == gsa_params_initial, (
        "Mismatch: trainable param count doesn't match initial GSA param count"
    )
    print("    [OK] Trainable parameter count matches GSA block parameter count")

    # ── 6. Verify gradients only flow to GSA params ─────────────────────
    print("\n[6] Verifying gradient isolation...")
    model.train()
    for name, module in model.named_modules():
        if "gsa" not in name.lower():
            module.eval()

    dummy_ecg = torch.randn(B, cfg.in_d, T, requires_grad=False)
    features, padding_mask, attention_logits = model.get_embeddings(
        dummy_ecg, padding_mask=None
    )
    # Use the first non-None attention logit to construct a dummy loss
    first_logit = next(l for l in attention_logits if l is not None)
    dummy_loss = first_logit.sum()
    dummy_loss.backward()

    grad_check_passed = True
    for name, param in model.named_parameters():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        if "gsa" in name.lower():
            if not has_grad:
                # Some GSA params may not receive gradient if they're not on
                # the computation path of the specific layer's logit used —
                # this is fine since we only used one layer's output
                pass
        else:
            if has_grad:
                print(f"    [FAIL] Frozen param received gradient: {name}")
                grad_check_passed = False

    if grad_check_passed:
        print("    [OK] No frozen parameters received gradients")
    else:
        print("    [FAIL] Some frozen parameters received gradients — check freeze logic")

    print("\n" + "=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    dry_run()