# experiment_v2_extended — Code + Embedded Results + Checkpoints

## What changed from last time
- All result numbers are now **embedded directly inside `experiment_v2_extended.py`**
  (a `RESULTS` dict near the bottom of the file, built from the actual json outputs —
  nothing hand-typed). No more separate scattered json files.
- Trained model **checkpoints** are included in `checkpoints/`.

## Just want to see the results? No training, no data files needed:
```bash
python3 experiment_v2_extended.py
```
This prints the full ablation / baseline / robustness / efficiency / interpretability
summary for both datasets and both seeds, straight from the embedded `RESULTS` dict
(see the `print_results()` function near the bottom of the file).

## Checkpoints (`checkpoints/` folder)
Trained weights for the Full Hybrid and Transformer-baseline models, for both
datasets and both seeds:
- `{Dataset}_s{seed}_full_hybrid.pt`
- `{Dataset}_s{seed}_transformer.pt`

To reload one (example for ETTh1, seed 42):
```python
from experiment_v2_extended import HybridForecaster, CFGS, build_dataloaders, DEVICE
import torch

cfg = CFGS["ETTh1"]
dl_train, dl_val, dl_test, n_channels = build_dataloaders(cfg)  # needs ETTh1.csv, see below
model = HybridForecaster(cfg, n_channels, use_tslif=True, use_coupling=True, adaptive_threshold=True)
x0, _ = next(iter(dl_test))
with torch.no_grad():
    model(x0[:1])  # materializes pos_embed before loading state_dict
model.load_state_dict(torch.load("checkpoints/ETTh1_s42_full_hybrid.pt"))
model.eval()
```

## Re-running training / evaluation from scratch
```bash
python3 experiment_v2_extended.py ablation ETTh1 42
python3 experiment_v2_extended.py baseline ETTh1 42 Transformer
python3 experiment_v2_extended.py baseline ETTh1 42 DLinear
python3 experiment_v2_extended.py robeff ETTh1 42
python3 experiment_v2_extended.py aggregate   # after all seeds/datasets are done
```
(Same for `Solar` instead of `ETTh1`, and seed `123`.)

This requires the datasets, not included here (only the code + results + checkpoints are):
```bash
curl -o ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
curl -o solar_AL.txt.gz https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz
gunzip solar_AL.txt.gz
```

## Other files
- `interpretability_test.py` — the dendrite/soma correlation test (needs a checkpoint + the matching dataset file).
- `plot_interpretability.py` — generates the example visualization.
- `interpretability_example.png` — the plot already generated.

## Note on scale
Everything here was run on CPU with reduced seeds (2 instead of 5) and reduced
epochs (10 for ETTh1, 8 for Solar) to fit within the available compute/time budget.
The numbers are real (not fabricated), but for a publication-grade result the
seed count should be increased, as discussed in chat.
