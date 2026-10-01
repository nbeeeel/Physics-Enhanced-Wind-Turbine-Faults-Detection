# Wind Turbine Blade Condition Classification with Weighted RFFT Hybrid Learning

This repository implements a vibration-based four-class wind-turbine blade-condition classification pipeline using residual multilayer perceptrons, explicit RFFT spectral representations, trainable frequency weighting, and raw--spectral feature fusion. The experiment distinguishes **Crack, Erosion, Healthy, and Twist** conditions from short nacelle-vibration windows.

The pipeline includes deterministic data preparation, stratified train/test partitioning, five-fold component-wise ablation, held-out evaluation, confidence and misclassification analysis, and frequency-domain interpretation of learned RFFT weights.

## Dataset

The vibration data originate from the experimental wind-turbine blade dataset reported by:

A. A. F. Ogaili, A. A. Jaber, and M. N. Hamzah, *Wind turbine blades fault diagnosis based on vibration dataset analysis*, Data in Brief, 49, 109414, 2023. DOI: `10.1016/j.dib.2023.109414`.

Dataset archive: Mendeley Data Version 2, DOI: `10.17632/5d7vbdp8f7.2`.

Place the four experiment files in `data/`:

```text
data/
├── Crack1.3.csv
├── Erosion1.3.csv
├── H1.3.csv
└── twist1.3.xlsx
```

## Experimental protocol

- Sampling frequency: 1000 Hz
- Window size: 64 samples
- Window stride: 10 samples
- Classes: Crack, Erosion, Healthy, Twist
- Total windows: 176
- Windows per class: 44
- Development/test split: 140/36 using stratified sampling with seed 42
- Final fit/validation/test sizes: 112/28/36
- Ablation: stratified five-fold cross-validation on the development partition
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Maximum epochs: 50
- Early-stopping patience: 8

A 64-sample window produces 33 RFFT bins with a frequency resolution of 15.625 Hz and a Nyquist frequency of 500 Hz.

## Model variants

The ablation evaluates seven configurations:

1. Plain MLP
2. Residual MLP
3. Fixed RFFT + Residual
4. Weighted RFFT + Residual
5. Fixed RFFT + Hybrid
6. Weighted RFFT + Hybrid without residual connection
7. Weighted RFFT + Hybrid

The full model applies a trainable weight to each RFFT magnitude bin, processes the weighted spectrum through a residual branch, processes the raw vibration window through a parallel branch, and fuses both representations before classification.

## Results

| Model | 5-fold accuracy | Macro F1 |
|---|---:|---:|
| Plain MLP | 0.4714 ± 0.1195 | 0.4412 ± 0.1236 |
| Residual MLP | 0.4714 ± 0.0639 | 0.4340 ± 0.0624 |
| Fixed RFFT + Residual | 0.9714 ± 0.0299 | 0.9713 ± 0.0300 |
| Weighted RFFT + Residual | 0.9071 ± 0.0407 | 0.9069 ± 0.0399 |
| Fixed RFFT + Hybrid | 0.9857 ± 0.0196 | 0.9856 ± 0.0197 |
| Weighted RFFT + Hybrid without residual | 0.9571 ± 0.0299 | 0.9561 ± 0.0311 |
| Weighted RFFT + Hybrid | 0.9429 ± 0.0319 | 0.9432 ± 0.0310 |

The final weighted-RFFT hybrid model achieved **97.22% held-out accuracy**, **97.50% macro precision**, **97.22% macro recall**, and **97.21% macro F1**. Thirty-five of 36 test windows were classified correctly. The single error was a Twist window predicted as Erosion with confidence 0.622, compared with mean confidence 0.917 across correct predictions.

The learned frequency-weight magnitudes showed a moderate monotonic association with frequency-wise spectral separability: Spearman `rho = 0.4586`, `p = 0.007277`. Five of the ten highest-weighted bins also appeared among the ten most discriminative spectral bins. The corresponding top-10 Jaccard overlap was 0.3333.

The results show that explicit spectral representation provides the main improvement over time-domain MLP baselines, while raw--spectral fusion provides additional discrimination. Trainable frequency weighting supplies an interpretable frequency profile but does not uniformly improve classification accuracy relative to fixed RFFT variants.

## Reproduction

Use Python 3.10 or 3.11, then create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the complete experiment:

```bash
python run.py --config config.yaml
```

The run creates `outputs/` containing model weights, protocol files, test metrics, ablation tables, prediction tables, spectral analyses, and publication-ready PNG/PDF figures.

## Repository structure

```text
.
├── config.yaml
├── requirements.txt
├── run.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── io_utils.py
│   ├── layers.py
│   ├── models.py
│   ├── plots.py
│   ├── reproducibility.py
│   ├── spectral.py
│   └── training.py
└── results/
    └── reference/
```

`results/reference/` contains the principal outputs from the reported experiment for direct comparison with reproduced runs.
