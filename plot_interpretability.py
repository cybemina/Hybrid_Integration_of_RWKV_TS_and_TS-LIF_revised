import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_v2_extended import HybridForecaster, build_dataloaders, CFGS, DEVICE
from interpretability_test import get_internals, compute_patch_signals

cfg = CFGS["ETTh1"]
dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)
model = HybridForecaster(cfg, n_channels, use_tslif=True, use_coupling=True, adaptive_threshold=True)
x0, _ = next(iter(dl_test))
with torch.no_grad():
    model(x0[:1])
model.load_state_dict(torch.load("results/ETTh1_s42_full_hybrid.pt"))
model.to(DEVICE).eval()

x, _ = next(iter(dl_test))
x = x.to(DEVICE)
with torch.no_grad():
    h_seq, R_seq, s_d, s_s = get_internals(model, x)

b = 3  # pick one example window
ch = n_channels - 1
series = x[b, :, ch].cpu().numpy()
trend, change = compute_patch_signals(series, patch_len=cfg["patch_len"])
rate_d = s_d[b].mean(dim=-1).cpu().numpy()
rate_s = s_s[b].mean(dim=-1).cpu().numpy()
n_patch = len(rate_d)
patch_centers = np.arange(n_patch) * cfg["patch_len"] + cfg["patch_len"] / 2

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(series, color="#333333", label="raw signal (OT, normalized)")
axes[0].plot(patch_centers, trend, "o-", color="#2E86AB", label="slow trend (per patch)")
axes[0].set_ylabel("value")
axes[0].legend(loc="upper right")
axes[0].set_title("Input window: raw signal + slow-trend estimate per patch")

axes[1].bar(patch_centers, change, width=cfg["patch_len"] * 0.8, color="#E07A5F", alpha=0.7)
axes[1].set_ylabel("local std (change)")
axes[1].set_title("Sudden-change proxy per patch (within-patch std)")

axes[2].plot(patch_centers, rate_d, "o-", color="#2E86AB", label="dendritic spike rate")
axes[2].plot(patch_centers, rate_s, "s-", color="#E07A5F", label="somatic spike rate")
axes[2].set_ylabel("mean spike rate")
axes[2].set_xlabel("time step (within 96-step window)")
axes[2].legend(loc="upper right")
axes[2].set_title("TS-LIF spike rates per patch (dendrite vs soma)")

plt.tight_layout()
plt.savefig("interpretability_example.png", dpi=130)
print("saved interpretability_example.png")
