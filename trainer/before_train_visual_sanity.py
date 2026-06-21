"""
sanity_check_attention_labels.py

Visual sanity check: loads a handful of real PTB-XL samples per condition
and plots the ECG signal (Lead II) with the generated attention guidance
label overlaid, so you can eyeball whether the QRS / PR masks are landing
on the correct regions before trusting them across the full dataset.

Run this BEFORE the full training run.
"""

import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from models.GSA import Attention_Maps  # adjust import if renamed differently

load_dotenv()
PTBXL_DIR = os.getenv("PTBXL_DATASET")

SAMPLING_FREQ = 500
TARGET_LENGTH = 5000

SCP_TO_LABEL = {
    "NORM":  "NORM",
    "CLBBB": "LBBB",
    "CRBBB": "RBBB",
    "1AVB":  "1dAVB",
}

N_SAMPLES_PER_CLASS = 3   # how many examples to plot per condition


def get_n_samples_per_condition(n=N_SAMPLES_PER_CLASS):
    """
    Lightweight version of ptbxl_cond_to_ids() that stops early once it has
    n samples per class — avoids scanning the whole CSV for a quick check.
    """
    records = pd.read_csv(os.path.join(PTBXL_DIR, "ptbxl_database.csv"))
    found = {k: [] for k in ["NORM", "LBBB", "RBBB", "1dAVB"]}

    for _, record in records.iterrows():
        if all(len(v) >= n for v in found.values()):
            break

        raw = record["scp_codes"].removeprefix("{").removesuffix("}")
        for s in raw.split(","):
            cond, conf = s.split(":")
            cond = cond.replace("'", "").strip()
            conf = conf.strip()
            if conf != "100.0":
                continue
            if cond in SCP_TO_LABEL:
                label = SCP_TO_LABEL[cond]
                if len(found[label]) < n:
                    found[label].append((record["ecg_id"], record["filename_hr"]))
                break

    return found


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


def generate_mask(lead_ii, label_str, att_maps):
    if label_str == "NORM":
        return np.zeros((TARGET_LENGTH, 1), dtype=np.float32)
    elif label_str == "1dAVB":
        return att_maps.generate_pr_mask(lead_ii, SAMPLING_FREQ)
    else:  # LBBB, RBBB
        return att_maps.generate_qrs_mask(lead_ii, SAMPLING_FREQ)


def plot_samples():
    att_maps = Attention_Maps()
    samples = get_n_samples_per_condition()

    fig, axes = plt.subplots(
        sum(len(v) for v in samples.values()), 1,
        figsize=(14, 2.2 * sum(len(v) for v in samples.values())),
        squeeze=False
    )
    axes = axes.flatten()

    row = 0
    time_axis = np.arange(TARGET_LENGTH) / SAMPLING_FREQ   # seconds

    for label_str, items in samples.items():
        for ecg_id, filename_hr in items:
            ecg = load_ecg(filename_hr)
            lead_ii = ecg[1]   # Lead II is index 1

            mask = generate_mask(lead_ii, label_str, att_maps).flatten()

            ax = axes[row]
            ax.plot(time_axis, lead_ii, color='black', linewidth=0.8, label='Lead II')

            # Shade attention regions
            ax.fill_between(
                time_axis, lead_ii.min(), lead_ii.max(),
                where=(mask > 0), color='red', alpha=0.25,
                label='Attention region'
            )

            ax.set_title(f"{label_str} — ecg_id {ecg_id} — {filename_hr}", fontsize=10)
            ax.set_xlim(0, 5)   # show first 5 seconds for clarity
            ax.legend(loc='upper right', fontsize=7)
            row += 1

    plt.tight_layout()
    out_path = "attention_label_sanity_check.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity check plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_samples()