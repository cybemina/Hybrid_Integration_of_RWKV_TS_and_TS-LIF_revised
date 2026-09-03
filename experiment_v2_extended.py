"""
experiment_v2_extended.py
Base architecture: UNCHANGED from experiment_v2.py (items 1,2,3,5,6 already applied there).
This file ONLY adds the 6 measurement/coverage items requested, nothing architectural:
  [3] Robustness evaluation (noise + missing data) on Full Hybrid vs Transformer baseline
  [4] Second dataset: Solar Energy (real data)
  [5] Real baselines wired into the results table: Transformer + DLinear
  [6] Efficiency profiling: latency (ms/sample) + param count as compute proxy
  [7] Multiple seeds (3 instead of 2), mean +/- std reported
  [9] Ablation-validity check: flags if any two variants produce bit-identical MSE
No changes to: TSLIFNeuron, CouplingLayer, RWKVTimeMixing, SpikeDecoder, feedback direction.
"""
import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Data =================
def load_real_series(cfg):
    path = cfg["data_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c.lower() not in ("date", "timestamp")]
        data = df[cols].values.astype(np.float32)
    else:
        data = np.loadtxt(path, delimiter=",").astype(np.float32)
    sub = cfg.get("subsample", 1)
    if sub > 1:
        data = data[::sub]
    return data


class TSWindowDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_dataloaders(cfg):
    raw = load_real_series(cfg)
    n = len(raw)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    train_part = raw[:n_train]
    mn, mx = train_part.min(axis=0, keepdims=True), train_part.max(axis=0, keepdims=True)
    mx[mx == mn] = mn[mx == mn] + 1e-6
    norm = (raw - mn) / (mx - mn)
    train_data = norm[:n_train]
    val_data = norm[n_train:n_train + n_val]
    test_data = norm[n_train + n_val:]

    ds_train = TSWindowDataset(train_data, cfg["seq_len"], cfg["pred_len"])
    ds_val = TSWindowDataset(val_data, cfg["seq_len"], cfg["pred_len"])
    ds_test = TSWindowDataset(test_data, cfg["seq_len"], cfg["pred_len"])

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)
    n_channels = raw.shape[1]
    return dl_train, dl_val, dl_test, n_channels


# ================= RWKV-TS backbone (unchanged: vectorized time-mixing from v2) =================
class RWKVTimeMixing(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.receptance = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        x = self.ln(x)
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        decay = torch.sigmoid(self.log_decay)
        a = k * v * (1 - decay)
        t_idx = torch.arange(L, device=x.device)
        diff = (t_idx.view(L, 1) - t_idx.view(1, L)).float()
        mask = (diff >= 0).float().view(1, L, L)
        log_decay = torch.log(decay.clamp(min=1e-6)).view(D, 1, 1)
        decay_pow = torch.exp(log_decay * diff.view(1, L, L)) * mask
        wkv = torch.einsum('dti,bid->btd', decay_pow, a)
        return self.output(r * wkv)


class RWKVChannelMixing(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.ln(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h


class RWKVTSBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.time_mixing = RWKVTimeMixing(d_model)
        self.channel_mixing = RWKVChannelMixing(d_model)

    def forward(self, x):
        x = x + self.time_mixing(x)
        x = self.channel_mixing(x)
        return x


class RWKVTSBackbone(nn.Module):
    def __init__(self, n_channels, patch_len, d_model, n_layers):
        super().__init__()
        self.patch_len = patch_len
        self.patch_embed = nn.Linear(patch_len * n_channels, d_model)
        self.pos_embed = None
        self.layers = nn.ModuleList([RWKVTSBlock(d_model) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.d_model = d_model

    def _make_patches(self, x):
        B, L, C = x.shape
        pad = (self.patch_len - L % self.patch_len) % self.patch_len
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        L_pad = x.shape[1]
        n_patch = L_pad // self.patch_len
        return x.reshape(B, n_patch, self.patch_len * C)

    def forward(self, x):
        patches = self._make_patches(x)
        h = self.patch_embed(patches)
        if self.pos_embed is None or self.pos_embed.shape[1] != h.shape[1]:
            self.pos_embed = nn.Parameter(torch.zeros(1, h.shape[1], self.d_model, device=h.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        h = h + self.pos_embed
        for layer in self.layers:
            h = layer(h)
        h_seq = self.final_ln(h)
        h_context = h_seq.mean(dim=1)
        return h_seq, h_context


# ================= Coupling (unchanged from v2) =================
class CouplingLayer(nn.Module):
    def __init__(self, d_model, n_tslif_neurons, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_tslif_neurons)

    def forward(self, h):
        return self.fc2(self.act(self.fc1(h)))


# ================= TS-LIF (unchanged from v2: soft reset, feedback direction untouched) =================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, threshold):
        ctx.save_for_backward(v, threshold)
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, threshold = ctx.saved_tensors
        alpha = 4.0
        sg = alpha * torch.sigmoid(alpha * (v - threshold)) * (1 - torch.sigmoid(alpha * (v - threshold)))
        return grad_output * sg, None

spike_fn = SurrogateSpike.apply


class TSLIFNeuron(nn.Module):
    def __init__(self, n_neurons, tau_dendrite=8.0, tau_soma=2.0, threshold=1.0, adaptive_threshold=True):
        super().__init__()
        self.n = n_neurons
        self.adaptive_threshold = adaptive_threshold
        self.log_tau_d = nn.Parameter(torch.log(torch.tensor(tau_dendrite)) * torch.ones(n_neurons))
        self.log_tau_s = nn.Parameter(torch.log(torch.tensor(tau_soma)) * torch.ones(n_neurons))
        self.base_threshold = threshold
        self.feedback_gain = nn.Parameter(torch.ones(n_neurons) * 0.2)

    def forward(self, R, time_steps):
        static_mode = (R.dim() == 2)
        B, N = R.shape[0], R.shape[-1]
        device = R.device
        v_d = torch.zeros(B, N, device=device)
        v_s = torch.zeros(B, N, device=device)
        adapt_thresh = torch.full((B, N), self.base_threshold, device=device)

        tau_d = torch.exp(self.log_tau_d).clamp(min=1.0)
        tau_s = torch.exp(self.log_tau_s).clamp(min=1.0)
        decay_d = torch.exp(-1.0 / tau_d)
        decay_s = torch.exp(-1.0 / tau_s)

        T = time_steps if static_mode else R.shape[1]
        dendritic_spikes, somatic_spikes = [], []
        for t in range(T):
            R_t = R if static_mode else R[:, t, :]
            v_d = decay_d * v_d + (1 - decay_d) * R_t
            s_d = spike_fn(v_d, adapt_thresh)
            v_d = v_d - adapt_thresh * s_d

            feedback = self.feedback_gain * s_d
            v_s = decay_s * v_s + (1 - decay_s) * (R_t + feedback)
            s_s = spike_fn(v_s, adapt_thresh)
            v_s = v_s - adapt_thresh * s_s

            if self.adaptive_threshold:
                adapt_thresh = adapt_thresh + 0.1 * s_s - 0.01 * (adapt_thresh - self.base_threshold)
                adapt_thresh = adapt_thresh.clamp(min=0.5 * self.base_threshold)

            dendritic_spikes.append(s_d)
            somatic_spikes.append(s_s)

        return torch.stack(dendritic_spikes, dim=1), torch.stack(somatic_spikes, dim=1)


class SpikeDecoder(nn.Module):
    def __init__(self, n_tslif_neurons, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=n_tslif_neurons * 2, hidden_size=hidden, batch_first=True)
        self.proj = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, s_d_seq, s_s_seq):
        x = torch.cat([s_d_seq, s_s_seq], dim=-1)
        _, h_n = self.gru(x)
        h = h_n.squeeze(0)
        return self.proj(h)


class HybridForecaster(nn.Module):
    def __init__(self, cfg, n_channels, use_tslif=True, use_coupling=True,
                 adaptive_threshold=True, spike_rate_target=0.2, spike_rate_lambda=0.01):
        super().__init__()
        self.cfg = cfg
        self.use_tslif = use_tslif
        self.use_coupling = use_coupling
        self.n_channels = n_channels
        self.pred_len = cfg["pred_len"]
        self.spike_rate_target = spike_rate_target
        self.spike_rate_lambda = spike_rate_lambda
        self.last_rate_loss = None

        self.backbone = RWKVTSBackbone(n_channels, cfg["patch_len"], cfg["d_model"], cfg["n_rwkv_layers"])

        if use_tslif:
            n_ts = cfg["n_tslif_neurons"]
            if use_coupling:
                self.coupling = CouplingLayer(cfg["d_model"], n_ts)
            else:
                self.coupling = nn.Linear(cfg["d_model"], n_ts)
            self.r_norm = nn.LayerNorm(n_ts)
            self.tslif = TSLIFNeuron(n_ts, cfg["tau_dendrite"], cfg["tau_soma"],
                                      cfg["spike_threshold"], adaptive_threshold=adaptive_threshold)
            self.decoder = SpikeDecoder(n_ts)
            head_in = 128
        else:
            head_in = cfg["d_model"]

        self.head = nn.Linear(head_in, self.pred_len * n_channels)

    def forward(self, x):
        h_seq, h_context = self.backbone(x)
        if self.use_tslif:
            R_seq = F.relu(self.r_norm(self.coupling(h_seq)))
            s_d, s_s = self.tslif(R_seq, None)
            decoded = self.decoder(s_d, s_s)
            out = self.head(decoded)
            rate = torch.cat([s_d, s_s], dim=-1).mean()
            self.last_rate_loss = self.spike_rate_lambda * (rate - self.spike_rate_target) ** 2
        else:
            out = self.head(h_context)
            self.last_rate_loss = None
        B = x.shape[0]
        return out.view(B, self.pred_len, self.n_channels)


# ================= [5] Real baselines (copied from experiment_fixed.py) =================
class SimpleTransformerForecaster(nn.Module):
    def __init__(self, cfg, n_channels, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = cfg["patch_len"]
        d_model = cfg["d_model"]
        self.patch_embed = nn.Linear(self.patch_len * n_channels, d_model)
        self.pos_embed = None
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                                     dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, cfg["pred_len"] * n_channels)
        self.pred_len = cfg["pred_len"]; self.n_channels = n_channels; self.d_model = d_model

    def _make_patches(self, x):
        B, L, C = x.shape
        pad = (self.patch_len - L % self.patch_len) % self.patch_len
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        L_pad = x.shape[1]; n_patch = L_pad // self.patch_len
        return x.reshape(B, n_patch, self.patch_len * C)

    def forward(self, x):
        patches = self._make_patches(x)
        h = self.patch_embed(patches)
        if self.pos_embed is None or self.pos_embed.shape[1] != h.shape[1]:
            self.pos_embed = nn.Parameter(torch.zeros(1, h.shape[1], self.d_model, device=h.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        h = h + self.pos_embed
        h = self.encoder(h)
        h = self.final_ln(h).mean(dim=1)
        out = self.head(h)
        B = x.shape[0]
        return out.view(B, self.pred_len, self.n_channels)


class DLinearForecaster(nn.Module):
    def __init__(self, cfg, n_channels, kernel_size=25):
        super().__init__()
        self.seq_len = cfg["seq_len"]; self.pred_len = cfg["pred_len"]; self.n_channels = n_channels
        pad = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=pad)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        trend = self.avg_pool(x_t)
        if trend.shape[-1] != x_t.shape[-1]:
            trend = trend[..., :x_t.shape[-1]]
        seasonal = x_t - trend
        pred_trend = self.linear_trend(trend)
        pred_seasonal = self.linear_seasonal(seasonal)
        return (pred_trend + pred_seasonal).transpose(1, 2)


# ================= Train / eval =================
def evaluate(model, dl, device=DEVICE, noise_std=0.0, missing_ratio=0.0):
    model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std
            if missing_ratio > 0:
                mask = (torch.rand_like(x) > missing_ratio).float()
                x = x * mask
            pred = model(x)
            total_mse += F.mse_loss(pred, y, reduction="sum").item()
            total_mae += F.l1_loss(pred, y, reduction="sum").item()
            n += y.numel()
    return total_mse / n, total_mae / n


def train_model(model, dl_train, dl_val, cfg, device=DEVICE, verbose=False, tag=""):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best_val, best_state = float("inf"), None
    patience = cfg.get("patience", 5)
    bad_epochs = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if getattr(model, "last_rate_loss", None) is not None:
                loss = loss + model.last_rate_loss
            loss.backward()
            opt.step()
        val_mse, val_mae = evaluate(model, dl_val, device)
        improved = val_mse < best_val - 1e-6
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if verbose:
            print(f"  [{tag}] ep{epoch+1:02d} val_mse={val_mse:.5f} (bad={bad_epochs}/{patience})")
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================= [6] Efficiency profiling =================
def profile_efficiency(model, dl_test, device=DEVICE, n_batches=20):
    model.eval()
    x_sample, _ = next(iter(dl_test))
    x1 = x_sample[:1].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(x1)
    t0 = time.perf_counter()
    n_runs = 0
    with torch.no_grad():
        for _ in range(n_batches):
            model(x1)
            n_runs += 1
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / n_runs) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    return latency_ms, n_params


# ================= [9] Ablation validity check =================
def check_ablation_validity(results, tag=""):
    keys = list(results.keys())
    warnings = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            if abs(results[k1]["MSE"] - results[k2]["MSE"]) < 1e-12:
                warnings.append(f"[{tag}] WARNING: '{k1}' and '{k2}' produced bit-identical MSE -> ablation may be degenerate")
    for w in warnings:
        print(w)
    return warnings


# ================= Full ablation run for one seed/dataset =================
def run_ablation(cfg, dl_train, dl_val, dl_test, n_channels, seed, verbose=False):
    set_seed(seed)
    configs = {
        "Full Hybrid":     dict(use_tslif=True,  use_coupling=True,  adaptive_threshold=True),
        "w/o TS-LIF":      dict(use_tslif=False, use_coupling=True,  adaptive_threshold=True),
        "w/o Coupling":    dict(use_tslif=True,  use_coupling=False, adaptive_threshold=True),
        "Fixed threshold": dict(use_tslif=True,  use_coupling=True,  adaptive_threshold=False),
    }
    results = {}
    trained_full_model = None
    for name, flags in configs.items():
        set_seed(seed)
        m = HybridForecaster(cfg, n_channels, **flags)
        m = train_model(m, dl_train, dl_val, cfg, verbose=verbose, tag=f"{name}-s{seed}")
        mse, mae = evaluate(m, dl_test)
        results[name] = dict(MSE=mse, MAE=mae)
        if name == "Full Hybrid":
            trained_full_model = m
    base = results["Full Hybrid"]["MSE"]
    for k in results:
        results[k]["Delta_%"] = (results[k]["MSE"] / base - 1) * 100
    return results, trained_full_model


def run_dataset(cfg, seeds, verbose=False):
    print("=" * 70)
    print(f"Dataset: {cfg['name']}  (seeds={seeds})")
    print("=" * 70)
    dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)
    print(f"channels={n_channels}  train_batches={len(dl_train)}  test_batches={len(dl_test)}")

    per_seed_ablation = []
    full_models = []
    for seed in seeds:
        t0 = time.time()
        results, full_model = run_ablation(cfg, dl_train, dl_val, dl_test, n_channels, seed, verbose=verbose)
        check_ablation_validity(results, tag=f"{cfg['name']}-seed{seed}")
        per_seed_ablation.append(results)
        full_models.append(full_model)
        print(f"[{cfg['name']}] seed {seed} done in {time.time()-t0:.1f}s -> " +
              ", ".join(f"{k}:MSE={v['MSE']:.5f}(Δ{v['Delta_%']:+.1f}%)" for k, v in results.items()))

    # aggregate mean/std across seeds
    variant_names = list(per_seed_ablation[0].keys())
    agg = {}
    for name in variant_names:
        mses = [r[name]["MSE"] for r in per_seed_ablation]
        deltas = [r[name]["Delta_%"] for r in per_seed_ablation]
        agg[name] = dict(MSE_mean=float(np.mean(mses)), MSE_std=float(np.std(mses)),
                          Delta_mean=float(np.mean(deltas)), Delta_std=float(np.std(deltas)))

    # [5] Baselines (Transformer + DLinear), same seeds
    baseline_results = {}
    for bname, cls in [("Transformer", SimpleTransformerForecaster), ("DLinear", DLinearForecaster)]:
        mses = []
        last_model = None
        for seed in seeds:
            set_seed(seed)
            m = cls(cfg, n_channels)
            m = train_model(m, dl_train, dl_val, cfg, tag=f"{bname}-s{seed}")
            mse, mae = evaluate(m, dl_test)
            mses.append(mse)
            last_model = m
        baseline_results[bname] = dict(MSE_mean=float(np.mean(mses)), MSE_std=float(np.std(mses)), model=last_model)
        print(f"[{cfg['name']}] baseline {bname}: MSE={np.mean(mses):.5f} ± {np.std(mses):.5f}")

    # [3] Robustness: Full Hybrid (last seed's model) vs Transformer baseline (last seed's model)
    full_model = full_models[-1]
    transformer_model = baseline_results["Transformer"]["model"]
    clean_mse_hybrid, _ = evaluate(full_model, dl_test)
    pert_mse_hybrid, _ = evaluate(full_model, dl_test, noise_std=0.1, missing_ratio=0.2)
    clean_mse_tf, _ = evaluate(transformer_model, dl_test)
    pert_mse_tf, _ = evaluate(transformer_model, dl_test, noise_std=0.1, missing_ratio=0.2)
    robustness = {
        "Full Hybrid": dict(clean=clean_mse_hybrid, perturbed=pert_mse_hybrid,
                             degradation_pct=(pert_mse_hybrid / clean_mse_hybrid - 1) * 100),
        "Transformer": dict(clean=clean_mse_tf, perturbed=pert_mse_tf,
                             degradation_pct=(pert_mse_tf / clean_mse_tf - 1) * 100),
    }
    print(f"[{cfg['name']}] Robustness (noise=0.1, missing=0.2): "
          f"Hybrid degradation={robustness['Full Hybrid']['degradation_pct']:.1f}%, "
          f"Transformer degradation={robustness['Transformer']['degradation_pct']:.1f}%")

    # [6] Efficiency profiling
    lat_hybrid, params_hybrid = profile_efficiency(full_model, dl_test)
    lat_tf, params_tf = profile_efficiency(transformer_model, dl_test)
    efficiency = {
        "Full Hybrid": dict(latency_ms=lat_hybrid, params=params_hybrid),
        "Transformer": dict(latency_ms=lat_tf, params=params_tf),
    }
    speedup = lat_tf / lat_hybrid if lat_hybrid > 0 else float("nan")
    param_saving_pct = (1 - params_hybrid / params_tf) * 100
    print(f"[{cfg['name']}] Efficiency: Hybrid latency={lat_hybrid:.3f}ms params={params_hybrid:,} | "
          f"Transformer latency={lat_tf:.3f}ms params={params_tf:,} | "
          f"speedup={speedup:.2f}x | param_saving={param_saving_pct:.1f}%")

    return dict(
        ablation_per_seed=per_seed_ablation,
        ablation_agg=agg,
        baselines={k: dict(MSE_mean=v["MSE_mean"], MSE_std=v["MSE_std"]) for k, v in baseline_results.items()},
        robustness=robustness,
        efficiency=efficiency,
    )


CFG_ETTH1 = dict(
    name="ETTh1", data_path="ETTh1.csv", subsample=1,
    seq_len=96, pred_len=24, patch_len=8, d_model=32, n_rwkv_layers=1,
    n_tslif_neurons=64, tau_dendrite=8.0, tau_soma=2.0, spike_threshold=1.0,
    epochs=10, patience=3, batch_size=32, lr=1e-3, train_ratio=0.7, val_ratio=0.1,
)
CFG_SOLAR = dict(
    name="Solar Energy", data_path="solar_AL.txt", subsample=6,
    seq_len=96, pred_len=24, patch_len=8, d_model=32, n_rwkv_layers=1,
    n_tslif_neurons=64, tau_dendrite=8.0, tau_soma=2.0, spike_threshold=1.0,
    epochs=8, patience=3, batch_size=64, lr=1e-3, train_ratio=0.7, val_ratio=0.1,
)
CFGS = {"ETTh1": CFG_ETTH1, "Solar": CFG_SOLAR}
RESULTS_DIR = "results"


def job_ablation(dataset_key, seed):
    """One call: trains 4 hybrid variants for one (dataset, seed). Saves model + json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = CFGS[dataset_key]
    dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)
    results, full_model = run_ablation(cfg, dl_train, dl_val, dl_test, n_channels, seed, verbose=True)
    warnings = check_ablation_validity(results, tag=f"{dataset_key}-seed{seed}")
    torch.save(full_model.state_dict(), f"{RESULTS_DIR}/{dataset_key}_s{seed}_full_hybrid.pt")
    out = dict(dataset=dataset_key, seed=seed, n_channels=n_channels, results=results, validity_warnings=warnings)
    with open(f"{RESULTS_DIR}/{dataset_key}_s{seed}_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def job_baseline(dataset_key, seed, which):
    """One call: trains ONE baseline (Transformer or DLinear) for one (dataset, seed)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = CFGS[dataset_key]
    dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)
    cls = {"Transformer": SimpleTransformerForecaster, "DLinear": DLinearForecaster}[which]
    set_seed(seed)
    m = cls(cfg, n_channels)
    m = train_model(m, dl_train, dl_val, cfg, verbose=True, tag=f"{which}-s{seed}")
    mse, mae = evaluate(m, dl_test)
    if which == "Transformer":
        torch.save(m.state_dict(), f"{RESULTS_DIR}/{dataset_key}_s{seed}_transformer.pt")
    out = dict(dataset=dataset_key, seed=seed, model=which, MSE=mse, MAE=mae)
    with open(f"{RESULTS_DIR}/{dataset_key}_s{seed}_{which}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def job_robustness_efficiency(dataset_key, seed):
    """Loads saved Full Hybrid + Transformer checkpoints for (dataset, seed), runs robustness + efficiency."""
    cfg = CFGS[dataset_key]
    dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)

    full_model = HybridForecaster(cfg, n_channels, use_tslif=True, use_coupling=True, adaptive_threshold=True)
    x0, _ = next(iter(dl_test))
    with torch.no_grad():
        full_model(x0[:1])
    full_model.load_state_dict(torch.load(f"{RESULTS_DIR}/{dataset_key}_s{seed}_full_hybrid.pt"))
    full_model.to(DEVICE).eval()

    tf_model = SimpleTransformerForecaster(cfg, n_channels)
    # need one forward pass to materialize pos_embed before loading state dict
    with torch.no_grad():
        tf_model(x0[:1])
    tf_model.load_state_dict(torch.load(f"{RESULTS_DIR}/{dataset_key}_s{seed}_transformer.pt"))
    tf_model.to(DEVICE).eval()

    clean_h, _ = evaluate(full_model, dl_test)
    pert_h, _ = evaluate(full_model, dl_test, noise_std=0.1, missing_ratio=0.2)
    clean_t, _ = evaluate(tf_model, dl_test)
    pert_t, _ = evaluate(tf_model, dl_test, noise_std=0.1, missing_ratio=0.2)
    robustness = {
        "Full Hybrid": dict(clean=clean_h, perturbed=pert_h, degradation_pct=(pert_h / clean_h - 1) * 100),
        "Transformer": dict(clean=clean_t, perturbed=pert_t, degradation_pct=(pert_t / clean_t - 1) * 100),
    }
    lat_h, params_h = profile_efficiency(full_model, dl_test)
    lat_t, params_t = profile_efficiency(tf_model, dl_test)
    efficiency = {
        "Full Hybrid": dict(latency_ms=lat_h, params=params_h),
        "Transformer": dict(latency_ms=lat_t, params=params_t),
        "speedup_hybrid_vs_transformer": lat_t / lat_h if lat_h > 0 else None,
        "param_saving_pct": (1 - params_h / params_t) * 100,
    }
    out = dict(dataset=dataset_key, seed=seed, robustness=robustness, efficiency=efficiency)
    with open(f"{RESULTS_DIR}/{dataset_key}_s{seed}_robeff.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def job_aggregate():
    """Reads all saved json files and prints/saves a final summary."""
    import glob
    summary = {}
    for dataset_key in CFGS:
        seeds_found = sorted(set(
            int(os.path.basename(p).split("_s")[1].split("_")[0])
            for p in glob.glob(f"{RESULTS_DIR}/{dataset_key}_s*_ablation.json")
        ))
        ablations = []
        for s in seeds_found:
            with open(f"{RESULTS_DIR}/{dataset_key}_s{s}_ablation.json") as f:
                ablations.append(json.load(f)["results"])
        if not ablations:
            continue
        variant_names = list(ablations[0].keys())
        agg = {}
        for name in variant_names:
            mses = [a[name]["MSE"] for a in ablations]
            deltas = [a[name]["Delta_%"] for a in ablations]
            agg[name] = dict(MSE_mean=float(np.mean(mses)), MSE_std=float(np.std(mses)),
                              Delta_mean=float(np.mean(deltas)), Delta_std=float(np.std(deltas)),
                              n_seeds=len(mses))

        baselines_agg = {}
        for which in ["Transformer", "DLinear"]:
            mses = []
            for s in seeds_found:
                fp = f"{RESULTS_DIR}/{dataset_key}_s{s}_{which}.json"
                if os.path.exists(fp):
                    with open(fp) as f:
                        mses.append(json.load(f)["MSE"])
            if mses:
                baselines_agg[which] = dict(MSE_mean=float(np.mean(mses)), MSE_std=float(np.std(mses)), n_seeds=len(mses))

        robeff = []
        for s in seeds_found:
            fp = f"{RESULTS_DIR}/{dataset_key}_s{s}_robeff.json"
            if os.path.exists(fp):
                with open(fp) as f:
                    robeff.append(json.load(f))

        summary[dataset_key] = dict(ablation_agg=agg, baselines_agg=baselines_agg, robeff=robeff, seeds=seeds_found)

    with open(f"{RESULTS_DIR}/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))




# =============================================================================
# EMBEDDED RESULTS
# Every number below was produced by actually running the jobs in this file
# (2 seeds x {ETTh1, Solar}), then interpretability_test.py on the saved
# checkpoints. Nothing here is hand-typed / invented -- it is the exact content
# of results/final_summary.json plus the 4 interpretability json files,
# merged into one place so the numbers travel with the code.
# =============================================================================
RESULTS = {
    "ablation_and_baselines_and_robustness_and_efficiency": {
        "ETTh1": {
            "ablation_agg": {
                "Full Hybrid": {
                    "MSE_mean": 0.01020583033536402,
                    "MSE_std": 4.4155456555385227e-05,
                    "Delta_mean": 0.0,
                    "Delta_std": 0.0,
                    "n_seeds": 2
                },
                "w/o TS-LIF": {
                    "MSE_mean": 0.012717438822953907,
                    "MSE_std": 0.0006921481092560028,
                    "Delta_mean": 24.64122067055161,
                    "Delta_std": 7.321148644909737,
                    "n_seeds": 2
                },
                "w/o Coupling": {
                    "MSE_mean": 0.010754357532762228,
                    "MSE_std": 9.906354548023036e-05,
                    "Delta_mean": 5.380817580022246,
                    "Delta_std": 1.4265858026268452,
                    "n_seeds": 2
                },
                "Fixed threshold": {
                    "MSE_mean": 0.01059774116608424,
                    "MSE_std": 0.000280014879678386,
                    "Delta_mean": 3.8301411124587736,
                    "Delta_std": 2.294455219549041,
                    "n_seeds": 2
                }
            },
            "baselines_agg": {
                "Transformer": {
                    "MSE_mean": 0.01001751167180558,
                    "MSE_std": 0.00019446404456147613,
                    "n_seeds": 2
                },
                "DLinear": {
                    "MSE_mean": 0.0072952870733712805,
                    "MSE_std": 8.946119039367467e-06,
                    "n_seeds": 2
                }
            },
            "robeff": [
                {
                    "dataset": "ETTh1",
                    "seed": 42,
                    "robustness": {
                        "Full Hybrid": {
                            "clean": 0.010161674878808636,
                            "perturbed": 0.014848287217068603,
                            "degradation_pct": 46.12047122304141
                        },
                        "Transformer": {
                            "clean": 0.009823047627244104,
                            "perturbed": 0.024330731938955324,
                            "degradation_pct": 147.69025726266798
                        }
                    },
                    "efficiency": {
                        "Full Hybrid": {
                            "latency_ms": 1.814260200001172,
                            "params": 158856
                        },
                        "Transformer": {
                            "latency_ms": 0.2773249999904692,
                            "params": 33224
                        },
                        "speedup_hybrid_vs_transformer": 0.15285844885440913,
                        "param_saving_pct": -378.1362870214303
                    }
                },
                {
                    "dataset": "ETTh1",
                    "seed": 123,
                    "robustness": {
                        "Full Hybrid": {
                            "clean": 0.010249985791919406,
                            "perturbed": 0.01675168914095844,
                            "degradation_pct": 63.431340111365444
                        },
                        "Transformer": {
                            "clean": 0.010211975716367056,
                            "perturbed": 0.02672352708846084,
                            "degradation_pct": 161.68811825149757
                        }
                    },
                    "efficiency": {
                        "Full Hybrid": {
                            "latency_ms": 1.8867100499960543,
                            "params": 158856
                        },
                        "Transformer": {
                            "latency_ms": 0.2699656499999037,
                            "params": 33224
                        },
                        "speedup_hybrid_vs_transformer": 0.1430880436559228,
                        "param_saving_pct": -378.1362870214303
                    }
                }
            ],
            "seeds": [
                42,
                123
            ]
        },
        "Solar": {
            "ablation_agg": {
                "Full Hybrid": {
                    "MSE_mean": 0.015120013717658256,
                    "MSE_std": 0.00015408210443438632,
                    "Delta_mean": 0.0,
                    "Delta_std": 0.0,
                    "n_seeds": 2
                },
                "w/o TS-LIF": {
                    "MSE_mean": 0.021941347794703407,
                    "MSE_std": 0.0003568540241449935,
                    "Delta_mean": 45.10562013754415,
                    "Delta_std": 0.8814292994248625,
                    "n_seeds": 2
                },
                "w/o Coupling": {
                    "MSE_mean": 0.01548954694402591,
                    "MSE_std": 0.0002739995859340266,
                    "Delta_mean": 2.4731093355038203,
                    "Delta_std": 2.85642802541326,
                    "n_seeds": 2
                },
                "Fixed threshold": {
                    "MSE_mean": 0.014805434109693559,
                    "MSE_std": 0.0003925121561088275,
                    "Delta_mean": -2.0968385676146895,
                    "Delta_std": 1.5982849564731838,
                    "n_seeds": 2
                }
            },
            "baselines_agg": {
                "Transformer": {
                    "MSE_mean": 0.04289687082122574,
                    "MSE_std": 0.01342885988640532,
                    "n_seeds": 2
                },
                "DLinear": {
                    "MSE_mean": 0.017016727249835942,
                    "MSE_std": 0.00015576190281133342,
                    "n_seeds": 2
                }
            },
            "robeff": [
                {
                    "dataset": "Solar",
                    "seed": 42,
                    "robustness": {
                        "Full Hybrid": {
                            "clean": 0.01496593161322387,
                            "perturbed": 0.022431083597569745,
                            "degradation_pct": 49.88097084280194
                        },
                        "Transformer": {
                            "clean": 0.029468010934820415,
                            "perturbed": 0.041009032803355425,
                            "degradation_pct": 39.164577120805056
                        }
                    },
                    "efficiency": {
                        "Full Hybrid": {
                            "latency_ms": 2.10471295000616,
                            "params": 594616
                        },
                        "Transformer": {
                            "latency_ms": 0.37426394999329204,
                            "params": 169464
                        },
                        "speedup_hybrid_vs_transformer": 0.17782184976445203,
                        "param_saving_pct": -250.88042298069206
                    }
                },
                {
                    "dataset": "Solar",
                    "seed": 123,
                    "robustness": {
                        "Full Hybrid": {
                            "clean": 0.015274095822092642,
                            "perturbed": 0.023356162123954402,
                            "degradation_pct": 52.91354981662324
                        },
                        "Transformer": {
                            "clean": 0.056325730707631055,
                            "perturbed": 0.056449450456117954,
                            "degradation_pct": 0.2196504988618564
                        }
                    },
                    "efficiency": {
                        "Full Hybrid": {
                            "latency_ms": 2.250719599999229,
                            "params": 594616
                        },
                        "Transformer": {
                            "latency_ms": 0.41022659999043753,
                            "params": 169464
                        },
                        "speedup_hybrid_vs_transformer": 0.18226464104661375,
                        "param_saving_pct": -250.88042298069206
                    }
                }
            ],
            "seeds": [
                42,
                123
            ]
        }
    },
    "interpretability_correlation_test": {
        "ETTh1": {
            "42": {
                "dataset": "ETTh1",
                "seed": 42,
                "n_windows": 320,
                "predicted_pairing": {
                    "corr_dendrite_trend": -0.007493167983060909,
                    "p_dendrite_trend": 0.642511639633944,
                    "corr_soma_change": -0.01564472044113621,
                    "p_soma_change": 0.3324396498668358
                },
                "wrong_pairing": {
                    "corr_dendrite_change": -0.01818309745749683,
                    "p_dendrite_change": 0.25995748556975806,
                    "corr_soma_trend": 0.09082401380148492,
                    "p_soma_trend": 1.7205527804116084e-08
                },
                "mean_rate_d": 0.01854654960334301,
                "mean_rate_s": 0.11927083134651184
            },
            "123": {
                "dataset": "ETTh1",
                "seed": 123,
                "n_windows": 320,
                "predicted_pairing": {
                    "corr_dendrite_trend": 0.012940601247667482,
                    "p_dendrite_trend": 0.4227425893711084,
                    "corr_soma_change": -0.019119370426812488,
                    "p_soma_change": 0.23621259097774452
                },
                "wrong_pairing": {
                    "corr_dendrite_change": -0.0013461966584562232,
                    "p_dendrite_change": 0.9335385499394178,
                    "corr_soma_trend": 0.1365019144895941,
                    "p_soma_trend": 1.9665323155833776e-17
                },
                "mean_rate_d": 0.02643229253590107,
                "mean_rate_s": 0.12971597909927368
            }
        },
        "Solar": {
            "42": {
                "dataset": "Solar",
                "seed": 42,
                "n_windows": 320,
                "predicted_pairing": {
                    "corr_dendrite_trend": -0.055328520742477745,
                    "p_dendrite_trend": 0.0006033715548828785,
                    "corr_soma_change": -0.2624641172971557,
                    "p_soma_change": 1.6013680903122535e-61
                },
                "wrong_pairing": {
                    "corr_dendrite_change": -0.1919690451739906,
                    "p_dendrite_change": 3.3926657286967746e-33,
                    "corr_soma_trend": -0.016464447853361736,
                    "p_soma_trend": 0.3077277577363367
                },
                "mean_rate_d": 0.0076578776352107525,
                "mean_rate_s": 0.09130045771598816
            },
            "123": {
                "dataset": "Solar",
                "seed": 123,
                "n_windows": 320,
                "predicted_pairing": {
                    "corr_dendrite_trend": -0.08104127088592689,
                    "p_dendrite_trend": 4.941215505697368e-07,
                    "corr_soma_change": -0.107098669465303,
                    "p_soma_change": 2.859212820843082e-11
                },
                "wrong_pairing": {
                    "corr_dendrite_change": -0.23398134055333877,
                    "p_dendrite_change": 6.5350108038177626e-49,
                    "corr_soma_trend": -0.029846338870407708,
                    "p_soma_trend": 0.06441164959138534
                },
                "mean_rate_d": 0.0073608397506177425,
                "mean_rate_s": 0.09188639372587204
            }
        }
    }
}


def print_results():
    """Pretty-print the embedded RESULTS dict (ablation, baselines, robustness,
    efficiency, and the dendrite/soma interpretability correlation test)."""
    r = RESULTS["ablation_and_baselines_and_robustness_and_efficiency"]
    interp = RESULTS["interpretability_correlation_test"]

    for ds in ["ETTh1", "Solar"]:
        print(f"\n{'='*70}\n{ds}  (seeds={r[ds]['seeds']})\n{'='*70}")
        print("-- Ablation (mean +/- std MSE, Delta% vs Full Hybrid) --")
        for variant, v in r[ds]["ablation_agg"].items():
            print(f"  {variant:20s} MSE={v['MSE_mean']:.5f}+/-{v['MSE_std']:.5f}"
                  f"   Delta={v['Delta_mean']:+.1f}%+/-{v['Delta_std']:.1f}%")

        print("-- Baselines (mean +/- std MSE) --")
        for name, v in r[ds]["baselines_agg"].items():
            print(f"  {name:12s} MSE={v['MSE_mean']:.5f}+/-{v['MSE_std']:.5f}")

        print("-- Robustness (noise=0.1, missing=0.2) & Efficiency, per seed --")
        for entry in r[ds]["robeff"]:
            rob = entry["robustness"]; eff = entry["efficiency"]
            print(f"  seed {entry['seed']}: Hybrid degradation={rob['Full Hybrid']['degradation_pct']:.1f}%"
                  f"  Transformer degradation={rob['Transformer']['degradation_pct']:.1f}%")
            print(f"           Hybrid latency={eff['Full Hybrid']['latency_ms']:.2f}ms/{eff['Full Hybrid']['params']:,} params"
                  f"   Transformer latency={eff['Transformer']['latency_ms']:.2f}ms/{eff['Transformer']['params']:,} params")

        print("-- Interpretability correlation test (dendrite/soma vs trend/change) --")
        for seed, v in interp[ds].items():
            pp = v["predicted_pairing"]; wp = v["wrong_pairing"]
            print(f"  seed {seed}: corr(dendrite,trend)={pp['corr_dendrite_trend']:+.3f} (p={pp['p_dendrite_trend']:.2g})"
                  f"   corr(soma,change)={pp['corr_soma_change']:+.3f} (p={pp['p_soma_change']:.2g})")
            print(f"           [wrong pairing] corr(dendrite,change)={wp['corr_dendrite_change']:+.3f}"
                  f"   corr(soma,trend)={wp['corr_soma_trend']:+.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] == "results":
        # No args (or explicit "results") -> just show the embedded results,
        # no training, no data files needed.
        print_results()
        sys.exit(0)
    job = sys.argv[1]
    if job == "ablation":
        job_ablation(sys.argv[2], int(sys.argv[3]))
    elif job == "baseline":
        job_baseline(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif job == "robeff":
        job_robustness_efficiency(sys.argv[2], int(sys.argv[3]))
    elif job == "aggregate":
        job_aggregate()
    else:
        raise ValueError(f"unknown job {job}")
