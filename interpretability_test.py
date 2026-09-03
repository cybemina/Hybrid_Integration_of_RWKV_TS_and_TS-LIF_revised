"""
Tests the paper's interpretability claim:
  "dendritic compartment tracks slow trends; somatic compartment reacts to sudden changes"
Uses the ALREADY-TRAINED Full Hybrid checkpoints (no retraining).
For each test window:
  - split into 12 patches (patch_len=8), matching the model's own patching
  - trend_signal[t]  = level of a long moving-average (slow component) at patch t
  - change_signal[t] = within-patch std of the raw signal (local volatility / suddenness) at patch t
  - rate_d[t] = mean dendritic spike rate at patch t (across neurons)
  - rate_s[t] = mean somatic spike rate at patch t (across neurons)
Then correlate:
  corr(rate_d, trend_signal)   <- paper's claim: should be strong
  corr(rate_s, change_signal)  <- paper's claim: should be strong
  corr(rate_d, change_signal)  <- "wrong pairing", should be weak if claim is true
  corr(rate_s, trend_signal)   <- "wrong pairing", should be weak if claim is true
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import json

from experiment_v2_extended import (
    HybridForecaster, build_dataloaders, CFGS, DEVICE
)


def get_internals(model, x):
    """Replicates HybridForecaster.forward but also returns intermediate spikes."""
    h_seq, h_context = model.backbone(x)
    R_seq = F.relu(model.r_norm(model.coupling(h_seq)))
    s_d, s_s = model.tslif(R_seq, None)
    return h_seq, R_seq, s_d, s_s


def compute_patch_signals(x_np, patch_len=8):
    """
    x_np: (T, C) raw (normalized) input window for ONE channel-averaged or single channel series.
    Returns trend_signal, change_signal each of length n_patches.
    """
    T = x_np.shape[0]
    n_patch = T // patch_len
    # slow trend: moving average with a window twice the patch length, sampled at patch center
    window = patch_len * 3
    kernel = np.ones(window) / window
    padded = np.pad(x_np, (window // 2, window // 2), mode="edge")
    smooth = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]

    trend_signal = np.zeros(n_patch)
    change_signal = np.zeros(n_patch)
    for t in range(n_patch):
        seg = x_np[t * patch_len:(t + 1) * patch_len]
        smooth_seg = smooth[t * patch_len:(t + 1) * patch_len]
        trend_signal[t] = smooth_seg.mean()
        change_signal[t] = seg.std()  # local volatility / suddenness within the patch
    return trend_signal, change_signal


def run_interpretability_test(dataset_key, seed, target_channel_idx=-1, n_windows=300):
    cfg = CFGS[dataset_key]
    dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)

    model = HybridForecaster(cfg, n_channels, use_tslif=True, use_coupling=True, adaptive_threshold=True)
    # materialize pos_embed
    x0, _ = next(iter(dl_test))
    with torch.no_grad():
        model(x0[:1])
    model.load_state_dict(torch.load(f"results/{dataset_key}_s{seed}_full_hybrid.pt"))
    model.to(DEVICE).eval()

    all_rate_d, all_rate_s, all_trend, all_change = [], [], [], []

    collected = 0
    with torch.no_grad():
        for x, _ in dl_test:
            x = x.to(DEVICE)
            h_seq, R_seq, s_d, s_s = get_internals(model, x)
            # mean spike rate per patch, across neurons -> (B, n_patch)
            rate_d = s_d.mean(dim=-1).cpu().numpy()
            rate_s = s_s.mean(dim=-1).cpu().numpy()
            n_patch = rate_d.shape[1]

            x_np = x.cpu().numpy()  # (B, T, C)
            ch = target_channel_idx if target_channel_idx >= 0 else x_np.shape[-1] - 1
            for b in range(x_np.shape[0]):
                series = x_np[b, :, ch]
                trend_signal, change_signal = compute_patch_signals(series, patch_len=cfg["patch_len"])
                if len(trend_signal) != n_patch:
                    m = min(len(trend_signal), n_patch)
                    trend_signal, change_signal = trend_signal[:m], change_signal[:m]
                    rd, rs = rate_d[b, :m], rate_s[b, :m]
                else:
                    rd, rs = rate_d[b], rate_s[b]
                all_rate_d.append(rd)
                all_rate_s.append(rs)
                all_trend.append(trend_signal)
                all_change.append(change_signal)
            collected += x_np.shape[0]
            if collected >= n_windows:
                break

    rate_d = np.concatenate(all_rate_d)
    rate_s = np.concatenate(all_rate_s)
    trend = np.concatenate(all_trend)
    change = np.concatenate(all_change)

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0, 1.0
        r, p = pearsonr(a, b)
        return float(r), float(p)

    r_d_trend, p_d_trend = safe_corr(rate_d, trend)
    r_s_change, p_s_change = safe_corr(rate_s, change)
    r_d_change, p_d_change = safe_corr(rate_d, change)
    r_s_trend, p_s_trend = safe_corr(rate_s, trend)

    result = dict(
        dataset=dataset_key, seed=seed, n_windows=collected,
        predicted_pairing=dict(
            corr_dendrite_trend=r_d_trend, p_dendrite_trend=p_d_trend,
            corr_soma_change=r_s_change, p_soma_change=p_s_change,
        ),
        wrong_pairing=dict(
            corr_dendrite_change=r_d_change, p_dendrite_change=p_d_change,
            corr_soma_trend=r_s_trend, p_soma_trend=p_s_trend,
        ),
        mean_rate_d=float(rate_d.mean()), mean_rate_s=float(rate_s.mean()),
    )
    return result


if __name__ == "__main__":
    import sys
    dataset_key = sys.argv[1]
    seed = int(sys.argv[2])
    res = run_interpretability_test(dataset_key, seed)
    print(json.dumps(res, indent=2))
    with open(f"results/{dataset_key}_s{seed}_interpretability.json", "w") as f:
        json.dump(res, f, indent=2)
