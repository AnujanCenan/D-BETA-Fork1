"""
predict_single.py

Given a single PTB-XL sample (by ecg_id or filename_hr), runs it through the
trained GSA model and plots:
  1. The ECG signal (Lead II) with the guidance label overlaid (ground truth
     region the model was trained to attend to)
  2. The learned attention weights from each GSA block, overlaid on the same
     signal — so you can visually compare what the model actually learned
     to attend to against what it was told to attend to.

This is the post-training counterpart to sanity_check_attention_labels.py,
which only visualizes the guidance labels (no model involved). This module
visualizes the model's actual learned behaviour.

Usage:
    from predict_single import predict_and_plot

    predict_and_plot(
        ecg_id=12345,
        config_path="configs/config.json",
        dbeta_checkpoint_path="path/to/zeta_checkpoint.pt",
        gsa_checkpoint_path="checkpoints/gsa/gsa_best.pt",
    )

Or from the command line:
    python predict_single.py --ecg_id 12345
"""

import os
import argparse
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from trainer.train_save_datasets import (
    load_trained_model,
    LABEL_TO_IDX,
    SCP_TO_LABEL,
)
from models.GSA import Attention_Maps   # adjust import if renamed differently

load_dotenv()
PTBXL_DIR = os.getenv("PTBXL_DATASET")

SAMPLING_FREQ = 500
TARGET_LENGTH = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Data lookup and loading
# ─────────────────────────────────────────────────────────────────────────────

def lookup_record(ecg_id=None, filename_hr=None):
    """
    Resolves an ecg_id or filename_hr to (ecg_id, filename_hr, label_str).
    Exactly one of ecg_id / filename_hr must be provided.
    """
    assert (ecg_id is not None) ^ (filename_hr is not None), (
        "Provide exactly one of ecg_id or filename_hr"
    )

    records = pd.read_csv(os.path.join(PTBXL_DIR, "ptbxl_database.csv"))

    if ecg_id is not None:
        row = records[records["ecg_id"] == ecg_id]
    else:
        row = records[records["filename_hr"] == filename_hr]

    assert len(row) == 1, f"Expected exactly one matching record, found {len(row)}"
    row = row.iloc[0]

    label_str = None
    raw = row["scp_codes"].removeprefix("{").removesuffix("}")
    for s in raw.split(","):
        cond, conf = s.split(":")
        cond = cond.replace("'", "").strip()
        conf = conf.strip()
        if conf == "100.0" and cond in SCP_TO_LABEL:
            label_str = SCP_TO_LABEL[cond]
            break

    if label_str is None:
        print(
            f"[WARNING] No matching condition found in {{NORM, CLBBB, CRBBB, 1AVB}} "
            f"for ecg_id {row['ecg_id']}. scp_codes: {row['scp_codes']}. "
            f"Proceeding without a guidance label (model attention will still "
            f"be plotted, but there's nothing to compare it against)."
        )

    return int(row["ecg_id"]), row["filename_hr"], label_str


def load_ecg(filename_hr):
    record_path = os.path.join(PTBXL_DIR, filename_hr)
    record = wfdb.rdrecord(record_path)
    ecg = record.p_signal.T.astype(np.float32)   # (12, L)

    L = ecg.shape[1]
    if L >= TARGET_LENGTH:
        ecg = ecg[:, :TARGET_LENGTH]
    else:
        ecg = np.pad(ecg, ((0, 0), (0, TARGET_LENGTH - L)))

    return ecg


def generate_guidance_mask(lead_ii, label_str, att_maps):
    if label_str is None or label_str == "NORM":
        return np.zeros((TARGET_LENGTH, 1), dtype=np.float32)
    elif label_str == "1dAVB":
        return att_maps.generate_pr_mask(lead_ii, SAMPLING_FREQ)
    else:  # LBBB, RBBB
        return att_maps.generate_qrs_mask(lead_ii, SAMPLING_FREQ)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, ecg, device):
    """
    Runs a single ECG through the model and returns the learned attention
    weights from each GSA block, interpolated up to the original signal
    length so they can be plotted against the raw ECG.

    Args:
        model : ECGTransformerModel with trained GSA blocks
        ecg   : (12, TARGET_LENGTH) numpy array
        device: torch device

    Returns:
        List of (TARGET_LENGTH,) numpy arrays — one attention map per
        active GSA block, each rescaled to the original signal length.
    """
    model.eval()
    ecg_tensor = torch.from_numpy(ecg).unsqueeze(0).to(device)   # (1, 12, T)

    _, _, attention_logits = model.get_embeddings(ecg_tensor, padding_mask=None)

    rescaled_maps = []
    for logits in attention_logits:
        if logits is None:
            rescaled_maps.append(None)
            continue
        probs = torch.sigmoid(logits)   # (1, 1, L_i)
        rescaled = F.interpolate(
            probs, size=TARGET_LENGTH, mode="linear", align_corners=False
        )   # (1, 1, TARGET_LENGTH)
        rescaled_maps.append(rescaled.squeeze().cpu().numpy())   # (TARGET_LENGTH,)

    return rescaled_maps


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction(
    ecg,
    guidance_mask,
    learned_attention_maps,
    ecg_id,
    filename_hr,
    label_str,
    lead_idx=1,
    lead_name="Lead II",
    save_path=None,
):
    """
    Produces a stacked plot:
      Row 0: ECG signal with ground-truth guidance region shaded (red)
      Row 1..N: ECG signal with each GSA block's learned attention map
                overlaid as a continuous shading (intensity = attention
                strength, not binary)
    """
    active_maps = [m for m in learned_attention_maps if m is not None]
    n_rows = 1 + len(active_maps)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 2.4 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    time_axis = np.arange(TARGET_LENGTH) / SAMPLING_FREQ
    signal = ecg[lead_idx]

    # ── Row 0: ground truth guidance label ──────────────────────────────
    ax = axes[0]
    ax.plot(time_axis, signal, color='black', linewidth=0.8)
    ax.fill_between(
        time_axis, signal.min(), signal.max(),
        where=(guidance_mask.flatten() > 0), color='red', alpha=0.25,
        label='Guidance label (ground truth)'
    )
    ax.set_title(
        f"Ground truth — {label_str or 'unknown'} — ecg_id {ecg_id} — {filename_hr} — {lead_name}",
        fontsize=10
    )
    ax.set_xlim(0, 5)
    ax.legend(loc='upper right', fontsize=7)

    # ── Rows 1..N: learned attention per GSA block ───────────────────────
    block_idx = 0
    for i, m in enumerate(learned_attention_maps):
        if m is None:
            continue
        block_idx += 1
        ax = axes[block_idx]
        ax.plot(time_axis, signal, color='black', linewidth=0.8)

        # Continuous shading using attention strength as alpha via a
        # twin axis showing the raw attention curve, plus a light fill
        # for regions where attention is above 0.5 (analogous to the
        # binarization used in the GSA paper's TPSr metric)
        ax2 = ax.twinx()
        ax2.plot(time_axis, m, color='tab:blue', linewidth=1.0, alpha=0.7,
                  label='Learned attention')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Attention', fontsize=8)

        ax.fill_between(
            time_axis, signal.min(), signal.max(),
            where=(m > 0.5), color='tab:blue', alpha=0.15,
        )
        ax.set_title(f"GSA block {i} — learned attention", fontsize=10)
        ax.set_xlim(0, 5)

        lines, labels = ax2.get_legend_handles_labels()
        ax.legend(lines, labels, loc='upper right', fontsize=7)

    plt.tight_layout()
    if save_path is None:
        save_path = f"predict_single_{ecg_id}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    plt.show()

    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_plot(
    ecg_id=None,
    filename_hr=None,
    config_path="configs/config.json",
    dbeta_checkpoint_path=None,
    gsa_checkpoint_path="checkpoints/gsa/gsa_best.pt",
    lead_idx=1,
    lead_name="Lead II",
    save_path=None,
):
    """
    Full pipeline: resolves the record, loads the ECG, runs the trained
    model, and produces the comparison plot.

    Args:
        ecg_id / filename_hr  : exactly one must be provided to identify
                                 the PTB-XL record
        config_path            : path to D-BETA/GSA config JSON
        dbeta_checkpoint_path  : path to the full ZETA/D-BETA checkpoint
                                 (defaults to CHECKPOINT_PATH env var)
        gsa_checkpoint_path    : path to the trained GSA-only checkpoint
        lead_idx               : which lead to plot (default 1 = Lead II)
        save_path               : where to save the output plot

    Returns:
        dict with ecg_id, filename_hr, label_str, and the save_path of
        the generated plot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dbeta_checkpoint_path is None:
        dbeta_checkpoint_path = os.getenv("CHECKPOINT_PATH")
        assert dbeta_checkpoint_path is not None, (
            "dbeta_checkpoint_path not provided and CHECKPOINT_PATH env var not set"
        )

    # ── Resolve and load the record ─────────────────────────────────────
    ecg_id, filename_hr, label_str = lookup_record(ecg_id=ecg_id, filename_hr=filename_hr)
    print(f"Resolved record: ecg_id={ecg_id}, filename_hr={filename_hr}, label={label_str}")

    ecg = load_ecg(filename_hr)
    lead_for_mask = ecg[lead_idx]

    att_maps = Attention_Maps()
    guidance_mask = generate_guidance_mask(lead_for_mask, label_str, att_maps)

    # ── Load model and run inference ────────────────────────────────────
    print("Loading model...")
    model = load_trained_model(config_path, dbeta_checkpoint_path, gsa_checkpoint_path)
    model = model.to(device)

    print("Running inference...")
    learned_attention_maps = run_inference(model, ecg, device)

    # ── Plot ─────────────────────────────────────────────────────────────
    out_path = plot_prediction(
        ecg=ecg,
        guidance_mask=guidance_mask,
        learned_attention_maps=learned_attention_maps,
        ecg_id=ecg_id,
        filename_hr=filename_hr,
        label_str=label_str,
        lead_idx=lead_idx,
        lead_name=lead_name,
        save_path=save_path,
    )

    return {
        "ecg_id": ecg_id,
        "filename_hr": filename_hr,
        "label_str": label_str,
        "plot_path": out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and plot GSA attention for a single ECG")
    parser.add_argument("--ecg_id", type=int, default=None, help="PTB-XL ecg_id")
    parser.add_argument("--filename_hr", type=str, default=None, help="PTB-XL filename_hr path")
    parser.add_argument("--config_path", type=str, default="configs/config.json")
    parser.add_argument("--dbeta_checkpoint_path", type=str, default=None)
    parser.add_argument("--gsa_checkpoint_path", type=str, default="checkpoints/gsa/gsa_best.pt")
    parser.add_argument("--lead_idx", type=int, default=1, help="Lead index to plot (default 1 = Lead II)")
    args = parser.parse_args()

    predict_and_plot(
        ecg_id=args.ecg_id,
        filename_hr=args.filename_hr,
        config_path=args.config_path,
        dbeta_checkpoint_path=args.dbeta_checkpoint_path,
        gsa_checkpoint_path=args.gsa_checkpoint_path,
        lead_idx=args.lead_idx,
    )