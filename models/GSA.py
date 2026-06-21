import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# GSA_Block
# ─────────────────────────────────────────────────────────────────────────────
# Ported from TensorFlow. Key convention change:
#   TF  : (Batch, Length, Channels)  — channels last
#   PyTorch : (Batch, Channels, Length) — channels first
#
# All Conv1d, MaxPool1d, and Upsample operations therefore operate on the
# last dimension (Length). Concatenation is on dim=1 (Channels).
# ─────────────────────────────────────────────────────────────────────────────

class GSA_Block(nn.Module):
    def __init__(self, K: int):
        """
        Args:
            K: Number of input channels coming from the backbone conv layer.
               Must match the `dim` field of the corresponding conv_layers
               config tuple in ConvFeatureExtraction.
        """
        super().__init__()
        self.K = K

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # ── Encoder path ──────────────────────────────────────────────────
        # enc_conv1: top → middle  (K → 32 channels)
        self.enc_conv1  = nn.Sequential(nn.Conv1d(K,  32, kernel_size=3, padding='same'), nn.ReLU())
        # conv_mid1: passes through the middle layer (red arrow in paper fig)
        self.conv_mid1  = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding='same'), nn.ReLU())
        # enc_conv2: middle → bottom  (32 → 64 channels)
        self.enc_conv2  = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding='same'), nn.ReLU())

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck_1 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding='same'), nn.ReLU())
        self.bottleneck_2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding='same'), nn.ReLU())

        # ── Decoder path ──────────────────────────────────────────────────
        # dec_conv2: after upsampling bottom → middle.
        # Input channels = 64 (upsampled) + 32 (skip from middle) = 96
        self.dec_conv2  = nn.Sequential(nn.Conv1d(96, 32, kernel_size=3, padding='same'), nn.ReLU())
        # dec_conv1: after upsampling middle → top.
        # Input channels = 32 (upsampled) + K (skip from input) = 32 + K
        self.dec_conv1  = nn.Sequential(nn.Conv1d(32 + K, 16, kernel_size=3, padding='same'), nn.ReLU())

        # ── Output head ───────────────────────────────────────────────────
        # Produces a single-channel spatial attention map (before sigmoid)
        self.final_conv = nn.Conv1d(16, 1, kernel_size=3, padding='same')

    def forward(self, conv_input: torch.Tensor):
        """
        Args:
            conv_input: (B, K, L) — channel-first feature map from backbone.

        Returns:
            final_output           : (B, K, L) — weighted feature map (same shape as input)
            attention_weights      : (B, 1, L) — sigmoid attention map
            attention_weights_no_sig: (B, 1, L) — pre-sigmoid logits (used for Dice loss)
        """
        # ── Encoder ───────────────────────────────────────────────────────
        middle = self.enc_conv1(conv_input)          # (B, 32, L)
        middle = self.max_pool(middle)               # (B, 32, L/2)
        middle = self.conv_mid1(middle)              # (B, 32, L/2)

        bottom = self.enc_conv2(middle)              # (B, 64, L/2)
        bottom = self.max_pool(bottom)               # (B, 64, L/4)

        # ── Bottleneck ────────────────────────────────────────────────────
        bottom = self.bottleneck_1(bottom)           # (B, 64, L/4)
        bottom = self.bottleneck_2(bottom)           # (B, 64, L/4)

        # ── Decoder: bottom → middle ──────────────────────────────────────
        bottom_up = self.upsample(bottom)            # (B, 64, L/2) approximately

        # Pad if upsampling produces a length one short of the skip connection.
        # In TF this was: tf.pad(x, [[0,0],[0,pad_len],[0,0]])
        # In PyTorch channel-first, Length is dim=2, so F.pad pads the last dim.
        pad_len_mid = middle.shape[2] - bottom_up.shape[2]
        if pad_len_mid > 0:
            bottom_up = F.pad(bottom_up, (0, pad_len_mid))

        middle_cat = torch.cat([bottom_up, middle], dim=1)  # (B, 96, L/2)
        middle = self.dec_conv2(middle_cat)                  # (B, 32, L/2)

        # ── Decoder: middle → top ─────────────────────────────────────────
        middle_up = self.upsample(middle)            # (B, 32, L) approximately

        pad_len_top = conv_input.shape[2] - middle_up.shape[2]
        if pad_len_top > 0:
            middle_up = F.pad(middle_up, (0, pad_len_top))

        top_cat = torch.cat([middle_up, conv_input], dim=1)  # (B, 32+K, L)
        top = self.dec_conv1(top_cat)                         # (B, 16, L)

        # ── Attention output ──────────────────────────────────────────────
        attention_weights_no_sig = self.final_conv(top)       # (B, 1, L)
        attention_weights = torch.sigmoid(attention_weights_no_sig)

        # Residual soft weighting: input * (attn + 1)  — same as TF version
        final_output = conv_input * (attention_weights + 1.0) # (B, K, L)

        return final_output, attention_weights, attention_weights_no_sig


# ─────────────────────────────────────────────────────────────────────────────
# GSA_Conv_Feature_Extraction
# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement for ConvFeatureExtraction that inserts a GSA_Block after
# each conv layer. Designed to slot into ECGTransformerModel in transformer.py.
#
# Usage in ECGTransformerModel.__init__():
#   Replace:
#       self.feature_extractor = ConvFeatureExtraction(...)
#   With:
#       self.feature_extractor = GSA_Conv_Feature_Extraction(...)
#
# The forward() signature is identical to ConvFeatureExtraction so no other
# changes are needed in get_embeddings() except unpacking attention weights.
# ─────────────────────────────────────────────────────────────────────────────

class GSA_Conv_Feature_Extraction(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_d: int = 12,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        gsa_placement: List[bool] = None,
    ):
        """
        Args:
            conv_layers  : Same format as ConvFeatureExtraction — list of
                           (dim, kernel_size, stride) tuples.
            in_d         : Input channel depth (1 for raw ECG).
            dropout      : Dropout applied inside each conv block.
            mode         : "default" or "layer_norm" — same as original.
            conv_bias    : Whether conv layers use bias.
            gsa_placement: Boolean list, one entry per conv layer. True means
                           a GSA block is inserted after that layer.
                           Defaults to True for all layers (equivalent to
                           TTTTT placement in the paper).
        """
        super().__init__()

        assert mode in {"default", "layer_norm"}

        if gsa_placement is None:
            gsa_placement = [True] * len(conv_layers)
        assert len(gsa_placement) == len(conv_layers), (
            "gsa_placement must have one entry per conv layer"
        )

        # ── Reuse the same block builder logic as ConvFeatureExtraction ──
        # from modules.conv_feature_extraction import ConvFeatureExtraction

        # Build conv layers identically to the original class
        self.conv_layers = nn.ModuleList()
        self.gsa_blocks  = nn.ModuleList()   # None entries for unplaced layers
        self.gsa_placement = gsa_placement

        current_in_d = in_d
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            # Build the same conv block as ConvFeatureExtraction
            self.conv_layers.append(
                self._make_block(
                    current_in_d, dim, k, stride,
                    dropout=dropout,
                    is_layer_norm=(mode == "layer_norm"),
                    is_group_norm=(mode == "default" and i == 0),
                    conv_bias=conv_bias,
                )
            )

            # Attach a GSA block if placement flag is True for this layer
            if gsa_placement[i]:
                self.gsa_blocks.append(GSA_Block(K=dim))
            else:
                self.gsa_blocks.append(None)

            current_in_d = dim

    @staticmethod
    def _make_block(n_in, n_out, k, stride, dropout, is_layer_norm,
                    is_group_norm, conv_bias):
        """Mirrors the block() closure inside ConvFeatureExtraction."""
        from models.modules import TransposeLast, Fp32LayerNorm, Fp32GroupNorm

        def make_conv():
            conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            return conv

        assert not (is_layer_norm and is_group_norm), (
            "layer norm and group norm are exclusive"
        )

        if is_layer_norm:
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(),
                    Fp32LayerNorm(n_out, n_out, affine=True),
                    TransposeLast(),
                ),
                nn.GELU(),
            )
        elif is_group_norm:
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                Fp32GroupNorm(n_out, n_out, affine=True),
                nn.GELU(),
            )
        else:
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Raw ECG tensor. Shape (B, T) or (B, 1, T).

        Returns:
            x                    : (B, C, L) — final feature map, same as
                                   ConvFeatureExtraction.forward() output.
            all_attention_weights: List of (B, 1, L_i) tensors, one per GSA
                                   block. None where gsa_placement is False.
            all_attn_no_sig      : List of pre-sigmoid logits — passed to
                                   Dice loss. None where gsa_placement is False.
        """
        if len(x.shape) < 3:
            x = x.unsqueeze(1)                      # (B, T) → (B, 1, T)

        all_attention_weights = []
        all_attn_no_sig       = []

        for conv, gsa in zip(self.conv_layers, self.gsa_blocks):
            x = conv(x)                              # (B, dim, L_i)
            if gsa is not None:
                x, attn, attn_no_sig = gsa(x)
                all_attention_weights.append(attn)
                all_attn_no_sig.append(attn_no_sig)
            else:
                all_attention_weights.append(None)
                all_attn_no_sig.append(None)

        return x, all_attention_weights, all_attn_no_sig


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def gsa_dice_loss(
    attention_logits_list: List[torch.Tensor],
    guidance_label: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss over all active GSA blocks. Mirrors attention_loss() in the TF
    version, with F.interpolate replacing tf.image.resize.

    Args:
        attention_logits_list: List of pre-sigmoid attention tensors, each
                               (B, 1, L_i). None entries (unplaced GSA layers)
                               are skipped automatically.
        guidance_label       : Binary spatial mask (B, 1, L_original).
                               1 = diagnostic region, 0 = elsewhere.
        smooth               : Laplace smoothing constant (default 1.0).

    Returns:
        Scalar Dice loss averaged over active GSA blocks.
    """
    target_length = guidance_label.shape[2]
    total_loss    = torch.tensor(0.0, device=guidance_label.device)
    n_active      = 0

    for logits in attention_logits_list:
        if logits is None:
            continue

        probs = torch.sigmoid(logits)                # (B, 1, L_i)

        # Resize to match the original ECG length
        # F.interpolate expects (B, C, L) — already in that format
        rescaled = F.interpolate(
            probs,
            size=target_length,
            mode='linear',
            align_corners=False
        )                                            # (B, 1, L_original)

        # Dice coefficient — computed per sample then averaged
        intersection = (guidance_label * rescaled).sum(dim=2)          # (B, 1)
        denominator  = guidance_label.sum(dim=2) + rescaled.sum(dim=2) # (B, 1)

        dice_coef  = (2.0 * intersection + smooth) / (denominator + smooth)
        total_loss = total_loss + (1.0 - dice_coef.mean())
        n_active  += 1

    return total_loss / n_active if n_active > 0 else total_loss


def total_loss(
    classification_loss: torch.Tensor,
    attention_logits_list: List[torch.Tensor],
    guidance_label: torch.Tensor,
    alpha: float,
    is_normal: bool = False,
) -> torch.Tensor:
    """
    Weighted sum of classification loss and GSA Dice loss.

    Args:
        classification_loss  : Scalar — the existing D-BETA pretraining loss.
        attention_logits_list: From GSA_Conv_Feature_Extraction.forward().
        guidance_label       : (B, 1, L) binary spatial mask.
        alpha                : Weight for classification loss (0 < alpha < 1).
        is_normal            : If True, skip attention loss (Normal class has
                               no specific diagnostic region). This replaces
                               the string-based check in the original TF code.

    Returns:
        Scalar total loss.
    """
    if is_normal:
        return classification_loss

    att_loss = gsa_dice_loss(attention_logits_list, guidance_label)
    return alpha * classification_loss + (1.0 - alpha) * att_loss


# ─────────────────────────────────────────────────────────────────────────────
# Attention mask generators  (pure numpy — no TF dependency, no changes needed)
# ─────────────────────────────────────────────────────────────────────────────

QRS_DURATION = 0.1
QR_INTERVAL  = 0.04
PR_INTERVAL  = 0.3


class Attention_Maps:
    """
    Generates binary spatial guidance labels for each diagnostic class.
    These are passed as `guidance_label` to gsa_dice_loss().

    Output masks have shape (L, 1) per sample — stack across a batch and
    transpose to (B, 1, L) before passing to the loss.
    """

    def generate_qrs_mask(self, ecg_signal: np.ndarray, sampling_freq: float) -> np.ndarray:
        """
        For LBBB / RBBB: attention region centred on the QRS complex.
        Window = ±(QRS_DURATION/2) around each R-peak.
        """
        from fast_qrs_detector import qrs_detector
        num_samples = ecg_signal.shape[0]
        mask        = np.zeros((num_samples, 1))
        qrs_results = qrs_detector(ecg_signal, sampling_freq)
        half        = int(sampling_freq * (QRS_DURATION / 2))
        for peak in qrs_results:
            mask[max(0, peak - half): min(num_samples, peak + half), 0] = 1
        return mask

    def generate_pr_mask(self, ecg_signal: np.ndarray, sampling_freq: float) -> np.ndarray:
        """
        For 1dAVB: attention region covering the PR interval
        (from PR_INTERVAL before the R-peak to QR_INTERVAL before the R-peak).
        """
        from fast_qrs_detector import qrs_detector
        num_samples = ecg_signal.shape[0]
        mask        = np.zeros((num_samples, 1))
        qrs_results = qrs_detector(ecg_signal, sampling_freq)
        for peak in qrs_results:
            end_idx   = max(0, peak - int(sampling_freq * QR_INTERVAL))
            start_idx = max(0, peak - int(sampling_freq * PR_INTERVAL))
            mask[start_idx:end_idx, 0] = 1
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def gsa_block_smoke_test():
    print("Initializing isolated GSA Block dry-run...")
    B, K, L = 4, 32, 500
    x = torch.randn(B, K, L)
    print(f"-> Input shape: {x.shape}")

    block = GSA_Block(K)
    out, attn, attn_no_sig = block(x)

    print(f"-> Output shape          : {out.shape}")
    print(f"-> Attention map shape   : {attn.shape}")
    print(f"-> Attention logits shape: {attn_no_sig.shape}")

    assert out.shape       == (B, K, L),  f"Feature map shape changed: {out.shape}"
    assert attn.shape      == (B, 1, L),  f"Attention map wrong shape: {attn.shape}"
    assert attn_no_sig.shape == (B, 1, L), f"Logits wrong shape: {attn_no_sig.shape}"
    print("[SUCCESS]: GSA_Block smoke test passed.\n")


def gsa_architecture_smoke_test():
    print("=" * 50)
    print("STARTING GSA_Conv_Feature_Extraction SMOKE TEST")
    print("=" * 50)

    # Mirrors the D-BETA default conv_feature_layers config
    # Format: (out_channels, kernel_size, stride)
    conv_layers = [
        (512, 10, 5),
        (512,  3, 2),
        (512,  3, 2),
        (512,  3, 2),
        (512,  3, 2),
        (512,  2, 2),
        (512,  2, 2),
    ]

    B, T = 4, 5000    # batch=4, 10s @ 500Hz
    x = torch.randn(B, 1, T)   # 12-lead ECG, raw input
    print(f"Input shape: {x.shape}")

    model = GSA_Conv_Feature_Extraction(
        conv_layers=conv_layers,
        in_d=1,
        gsa_placement=[True, True, True, True, True, True, True],
    )

    features, attn_weights, attn_logits = model(x)

    print(f"Output feature map shape : {features.shape}")
    print(f"Number of attention maps : {len(attn_weights)}")
    for i, (aw, al) in enumerate(zip(attn_weights, attn_logits)):
        if aw is not None:
            print(f"  Layer {i}: attn={aw.shape}, logits={al.shape}")

    # Loss test
    L_orig = T  # guidance label is at original signal length
    guidance = torch.randint(0, 2, (B, 1, L_orig)).float()
    alpha    = 0.6
    cls_loss = torch.tensor(0.5)   # mock classification loss

    loss = total_loss(cls_loss, attn_logits, guidance, alpha, is_normal=False)
    print(f"\nTotal loss (alpha={alpha}): {loss.item():.4f}")

    print("\n[SUCCESS]: GSA_Conv_Feature_Extraction smoke test passed.")
    print("=" * 50)


if __name__ == "__main__":
    gsa_block_smoke_test()
    gsa_architecture_smoke_test()
