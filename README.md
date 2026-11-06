# COSMI: Constrained Order Search with Verification for Mediator Identification

Official implementation and synthetic datasets for the manuscript **"COSMI: Constrained Order Search with Verification for Mediator Identification"**.

## Repository Structure

- `cosmi.py` (`BossM2.py`): Core implementation of the two-stage COSMI framework (Stage I: Constrained Order Search with $T \prec Y$; Stage II: Order-Guided Candidate Selection and BIC-Based Mediator Verification).
- `gst.py`: Grow–Shrink Tree (GST) structure for caching and evaluating decomposable local BIC scores.
- `data.py`: Synthetic structural causal model (SCM) generator for all experimental settings in the manuscript.
- `data/`: Contains all 200 synthetic datasets evaluated in the paper:
  - `data1_1.csv` to `data10_10.csv` (100 datasets): Main $10 \times 10$ confounder–mediator benchmark datasets ($|\mathcal{C}|, |\mathcal{M}| \in \{1, \dots, 10\}$, $n = 1000$, `seed=1`; Sections 3.1, 3.4, and 3.5).
  - `negative_c*_m*_z*.csv` (75 datasets): Post-treatment non-mediator ($Z$) robustness datasets ($|\mathcal{C}|, |\mathcal{M}| \in \{1, 3, 5, 7, 10\}$, $|\mathcal{Z}| \in \{1, 5, 10\}$, $n = 1000$; Section 3.6).
  - `validation_c*_m*.csv` (25 datasets): Independent validation datasets for verification-threshold sensitivity analysis ($|\mathcal{C}|, |\mathcal{M}| \in \{1, 3, 5, 7, 10\}$, $n = 1000$; Section 3.7).

## Requirements

- Python >= 3.8
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `causal-learn` (for baseline comparisons)

## Usage

1. **Generate all synthetic datasets (100 main + 75 non-mediator $Z$ + 25 validation datasets)**:
   ```bash
   python data.py
   ```
2. **Run COSMI mediator identification**:
   ```bash
   python cosmi.py
   ```
