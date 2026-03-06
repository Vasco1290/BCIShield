# BCIShield 🛡️

**Adversarial Robustness Benchmark for EEG-Based Brain-Computer Interface Systems**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![MNE](https://img.shields.io/badge/MNE-1.4%2B-green)](https://mne.tools)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-BCI%20Competition%20IV%202a-purple)](https://www.bbci.de/competition/iv/)

---

## Overview

EEG-based brain-computer interfaces (BCIs) that control prosthetic limbs and exoskeletons are vulnerable to adversarial attacks — imperceptibly small perturbations to neural signals that cause classifiers to misfire, potentially directing a physical actuator to execute the wrong movement.

**BCIShield** is an open-source adversarial robustness benchmark that:

- Quantifies how dramatically EEG motor imagery classifiers degrade under adversarial attack
- Evaluates two defense strategies: adversarial training and Gaussian input smoothing
- Measures inference latency overhead to confirm real-time control feasibility
- Provides a reproducible experimental framework across 9 subjects and 5 perturbation magnitudes

This project serves as preliminary work for a Master's research proposal on adversarial security in BCI-driven assistive robotics.

---

### Vulnerability — Clean vs. Under Attack (Subject 1)

| Attack | ε | No Defense | Input Smoothing | Adversarial Training |
|--------|---|-----------|-----------------|----------------------|
| Baseline (clean) | — | **63.2%** | — | — |
| FGSM | 0.01 | 14.0% | 15.8% | **52.6%** |
| FGSM | 0.05 | 0.0% | 0.0% | **29.8%** |
| FGSM | 0.10 | 0.0% | 0.0% | **7.0%** |
| PGD | 0.01 | 14.0% | 15.8% | **52.6%** |
| PGD | 0.05 | 0.0% | 0.0% | **29.8%** |
| PGD | 0.10 | 0.0% | 0.0% | **8.8%** |

**Headline finding:** An FGSM adversarial perturbation at ε=0.01 — imperceptible to human observers — reduces EEGNet motor imagery classification accuracy from **63.2% → 14.0%**, a degradation of **49.2 percentage points**. Adversarial training recovers accuracy to **52.6%** under the same attack.

### Latency (Subject 1, CPU)

| Condition | Inference Latency |
|-----------|------------------|
| Baseline (no defense) | 2.28 ms |
| With Input Smoothing | 2.64 ms |
| Overhead | +0.36 ms |

Defense overhead of **0.36ms** is well within real-time BCI control constraints (~50ms window).

---

## Architecture

### Model — EEGNet
Compact CNN for EEG classification (Lawhern et al., 2018, Journal of Neural Engineering).

```
Input: (batch, 1, 22 channels, 1000 samples)
  └─ Block 1: Temporal Convolution + BatchNorm
  └─ Block 2: Depthwise Spatial Convolution + ELU + AvgPool + Dropout
  └─ Block 3: Separable Convolution + ELU + AvgPool + Dropout
  └─ Classifier: Flatten + Linear → 4 classes
```

### Attacks
| Attack | Type | Description |
|--------|------|-------------|
| **FGSM** | Single-step | x_adv = x + ε · sign(∇ₓ L(x, y)) |
| **PGD** | Iterative | FGSM repeated for N steps with α = 2.5ε/N, projected back to ε-ball |

### Defenses
| Defense | Description |
|---------|-------------|
| **Adversarial Training** | Retrain EEGNet on PGD-augmented samples (50% clean / 50% adversarial) |
| **Input Smoothing** | Gaussian kernel (σ=1.0, kernel=5) applied to raw EEG signal pre-inference |

---

## Dataset

**BCI Competition IV Dataset 2a** — Graz University of Technology

- 9 subjects, 4-class motor imagery (left hand, right hand, foot, tongue)
- 22 EEG channels, 250 Hz sampling rate, 288 trials per subject
- Download: [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/)

Place downloaded `.gdf` files in `data/raw/`:
```
data/raw/
├── A01T.gdf
├── A02T.gdf
...
└── A09T.gdf
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Vasco1290/BCIShield.git
cd BCIShield

# Create virtual environment (Windows)
python -m venv bcishield_env
bcishield_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.8+, PyTorch 2.0+, MNE 1.4+, NumPy, SciPy, scikit-learn, pandas, matplotlib, seaborn, tqdm

---

## Usage

### Quick Start — Run Full Benchmark

```bash
python experiments/run_experiment.py
```

Results are saved to `results/tables/experiment_results.csv`. Incremental results are saved after each subject completes, so progress is preserved if interrupted.

### Configure Experiment

Edit `experiments/configs/default_config.yaml`:

```yaml
dataset:
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]  # subjects to run

training:
  epochs: 100         # base model training epochs
  adv_epochs: 50      # adversarial training epochs

attacks:
  fgsm:
    epsilons: [0.01, 0.05, 0.1, 0.2, 0.3]
  pgd:
    epsilons: [0.01, 0.05, 0.1, 0.2, 0.3]
    steps: 10
```

### Run Individual Components

```python
from src.models.eegnet import EEGNet
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defenses.input_smoothing import GaussianSmoothing
import torch
import torch.nn as nn

# Load model
model = EEGNet(num_classes=4, channels=22, samples=1000)

# Run FGSM attack
criterion = nn.CrossEntropyLoss()
x_adv, perturbation = fgsm_attack(model, x, y, epsilon=0.05, criterion=criterion)

# Run PGD attack
x_adv, perturbation = pgd_attack(model, x, y, epsilon=0.05, steps=10)

# Apply input smoothing defense
smoother = GaussianSmoothing(channels=22, kernel_size=5, sigma=1.0)
x_clean = smoother(x_adv)
```

---

## Project Structure

```
BCIShield/
├── data/
│   ├── raw/                    ← Place .gdf files here
│   ├── processed/
│   └── download_dataset.py     ← Dataset download instructions
├── experiments/
│   ├── configs/
│   │   └── default_config.yaml ← All hyperparameters
│   └── run_experiment.py       ← Main experiment pipeline
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_adversarial_attacks.ipynb
│   ├── 04_defense_evaluation.ipynb
│   └── 05_results_visualization.ipynb
├── src/
│   ├── models/
│   │   └── eegnet.py           ← EEGNet implementation (from scratch)
│   ├── attacks/
│   │   ├── fgsm.py             ← FGSM attack
│   │   └── pgd.py              ← PGD attack
│   ├── defenses/
│   │   ├── adversarial_training.py
│   │   └── input_smoothing.py
│   ├── data/
│   │   └── dataset.py          ← MNE loader + preprocessing
│   └── evaluation/
│       └── metrics.py          ← Accuracy, latency, defense effectiveness
├── results/
│   └── tables/
│       └── experiment_results.csv
├── tests/
├── requirements.txt
└── setup.py
```

---

## Reproducing Results

All experiments are fully reproducible with fixed random seeds.

```bash
# Smoke test — 1 subject, 10 epochs (~15 minutes on CPU)
# Edit default_config.yaml: subjects: [1], epochs: 10, adv_epochs: 5
python experiments/run_experiment.py

# Full benchmark — 9 subjects, 100 epochs (~6-8 hours on CPU)
# Edit default_config.yaml: subjects: [1,2,3,4,5,6,7,8,9], epochs: 100
python experiments/run_experiment.py
```

---

## Methodology

### Signal Preprocessing
1. Load raw `.gdf` file using MNE-Python
2. Bandpass filter: 4–40 Hz (FIR, firwin design)
3. Epoch extraction: 0–4 seconds post motor imagery cue
4. Per-channel z-score normalization
5. Train/validation split: 80/20 from training file

### Evaluation Protocol
- Per-subject training and evaluation (no cross-subject transfer)
- Metrics: classification accuracy (%), inference latency (ms), defense recovery (%)
- Attack evaluation: white-box (attacker has full model access)
- Defense evaluation: naive (attacker does not adapt to defense)

### Perturbation Bounds
All attacks operate under L∞ norm constraint: ||x_adv - x||∞ ≤ ε

---

## Motivation

As EEG-BCI systems increasingly control physical devices — prosthetic limbs, exoskeletons, rehabilitation robots — the consequences of adversarial manipulation shift from classification errors to physical safety incidents. A spoofed EEG signal that causes a prosthetic limb to move in the wrong direction is not an accuracy statistic. It is a safety incident.

This benchmark is the first step toward a hybrid defense architecture designed specifically for motor imagery BCI systems operating under real-time control constraints. The full research is being conducted as part of a Master's study proposal at the University of Tokyo under the MEXT Scholarship program.

---

## Citation

If you use BCIShield in your research, please cite:

```bibtex
@misc{rawal2026bcishield,
  author    = {Harshit Kishor Rawal},
  title     = {BCIShield: Adversarial Robustness Benchmark for EEG-Based BCI Systems},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Vasco1290/BCIShield}
}
```

### Key References

- Lawhern, V.J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based BCIs. *Journal of Neural Engineering*, 15(5).
- Goodfellow, I.J., et al. (2015). Explaining and harnessing adversarial examples. *ICLR 2015*.
- Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.
- Brunner, C., et al. (2008). BCI Competition IV Dataset 2a. *Graz University of Technology*.

---

## Roadmap

- [x] EEGNet implementation from scratch (PyTorch)
- [x] FGSM and PGD attack implementations
- [x] Adversarial training defense
- [x] Gaussian input smoothing defense
- [x] Full benchmark pipeline with incremental CSV saving
- [ ] Full 9-subject results (in progress)
- [ ] Jupyter notebooks with visualizations
- [ ] Confidence gating defense (Phase 2)
- [ ] Channel saliency analysis
- [ ] Contribution to IBM Adversarial Robustness Toolbox (ART)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
Built as preliminary research for Master's study at University of Tokyo · MEXT Scholarship 2026
</p>
