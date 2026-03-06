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

A spoofed EEG signal that causes a prosthetic limb to move in the wrong direction is not an accuracy statistic. It is a safety incident.

---

## Key Results

**Full 9-subject benchmark — BCI Competition IV Dataset 2a — 100 training epochs**

### Vulnerability — 9-Subject Averages

| Attack | ε | No Defense | Input Smoothing | Adversarial Training |
|--------|---|-----------|-----------------|----------------------|
| Baseline (clean) | — | **57.9%** | — | — |
| FGSM | 0.01 | 10.1% | 9.9% | **38.8%** |
| FGSM | 0.05 | 0.0% | 0.0% | **18.5%** |
| FGSM | 0.10 | 0.0% | 0.0% | **5.1%** |
| PGD  | 0.01 | 9.4% | 9.9% | **38.8%** |
| PGD  | 0.05 | 0.0% | 0.0% | **18.9%** |

**Headline finding:** An FGSM adversarial perturbation at ε=0.01 — imperceptible to human observers — reduces average EEGNet motor imagery classification accuracy from **57.9% → 10.1%**, a degradation of **47.8 percentage points** across all 9 subjects. Adversarial training recovers accuracy to **38.8%** under the same attack (+28.7pp recovery).

PGD attack produces marginally stronger degradation (9.4% average), confirming the vulnerability persists under iterative attack strategies and is not specific to the attack method.

### Per-Subject Results — FGSM ε=0.01

| Subject | Clean | No Defense | Adv Training | Recovery |
|---------|-------|-----------|--------------|----------|
| S1 | 71.9% | 19.3% | 50.9% | +31.6pp |
| S2 | 38.6% | 1.8% | 22.8% | +21.1pp |
| S3 | 73.7% | 5.3% | 47.4% | +42.1pp |
| S4 | 38.6% | 3.5% | 29.8% | +26.3pp |
| S5 | 42.1% | 5.3% | 26.3% | +21.1pp |
| S6 | 36.8% | 0.0% | 31.6% | +31.6pp |
| S7 | 63.2% | 1.8% | 26.3% | +24.6pp |
| S8 | 79.0% | 19.3% | 49.1% | +29.8pp |
| S9 | 77.2% | 35.1% | **64.9%** | +29.8pp |
| **Avg** | **57.9%** | **10.1%** | **38.8%** | **+28.7pp** |

### Latency (CPU, avg across subjects)

| Condition | Inference Latency |
|-----------|------------------|
| Baseline (no defense) | 5.98 ms |
| With Input Smoothing | ~7.2 ms |

Average baseline inference of **5.98ms** is well within the real-time BCI control constraint (~50ms window).

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

### Run Full Benchmark

```bash
python experiments/run_experiment.py
```

Results are saved to `results/tables/experiment_results.csv`. Incremental results are saved after each subject completes, so progress is preserved if interrupted.

### Configure Experiment

Edit `experiments/configs/default_config.yaml`:

```yaml
dataset:
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]

training:
  epochs: 100
  adv_epochs: 50

attacks:
  fgsm:
    epsilons: [0.01, 0.05, 0.1, 0.2, 0.3]
  pgd:
    epsilons: [0.01, 0.05, 0.1, 0.2, 0.3]
    steps: 10
```

---

## Project Structure

```
BCIShield/
├── data/
│   ├── raw/                    ← Place .gdf files here
│   └── processed/
├── experiments/
│   ├── configs/
│   │   └── default_config.yaml
│   └── run_experiment.py
├── src/
│   ├── models/eegnet.py
│   ├── attacks/fgsm.py
│   ├── attacks/pgd.py
│   ├── defenses/adversarial_training.py
│   ├── defenses/input_smoothing.py
│   ├── data/dataset.py
│   └── evaluation/metrics.py
├── results/tables/
│   └── experiment_results.csv
├── requirements.txt
└── setup.py
```

---

## Methodology

### Signal Preprocessing
1. Load raw `.gdf` file using MNE-Python
2. Bandpass filter: 4–40 Hz (FIR, firwin design)
3. Epoch extraction: 0–4 seconds post motor imagery cue
4. Per-channel z-score normalization
5. Train/validation split: 80/20

### Evaluation Protocol
- Per-subject training and evaluation (no cross-subject transfer)
- White-box attack evaluation (attacker has full model access)
- Naive defense evaluation (attacker does not adapt to defense)
- Perturbation bounds: L∞ norm constraint ||x_adv - x||∞ ≤ ε

---

## Motivation

As EEG-BCI systems increasingly control physical devices — prosthetic limbs, exoskeletons, rehabilitation robots — the consequences of adversarial manipulation shift from classification errors to physical safety incidents. A spoofed EEG signal that causes a prosthetic limb to move in the wrong direction is not an accuracy statistic. It is a safety incident.

This benchmark is the first step toward a hybrid defense architecture designed specifically for motor imagery BCI systems operating under real-time control constraints. The full research is being conducted as part of a Master's study proposal at the University of Tokyo under the MEXT Scholarship program.

---

## Citation

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
- [x] Full 9-subject benchmark (100 epochs, 5 epsilon values)
- [x] Complete results table across all subjects and conditions
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
