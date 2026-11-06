# COSMI: Constrained Order Search with Verification for Mediator Identification

Official implementation and synthetic benchmark datasets for the manuscript **"COSMI: Constrained Order Search with Verification for Mediator Identification"**.

## Repository Structure

- `cosmi.py` (`BossM2.py`): Core implementation of the two-stage COSMI framework (Stage I: Constrained Order Search with $T \prec Y$; Stage II: Order-Guided Candidate Selection and BIC-Based Mediator Verification).
- `gst.py`: Grow–Shrink Tree (GST) structure for caching and evaluating decomposable local BIC scores.
- `data.py`: Synthetic structural causal model (SCM) generator for the $10 \times 10$ confounder–mediator configuration grid ($|\mathcal{C}|, |\mathcal{M}| \in \{1, \dots, 10\}$, $n = 1000$, `seed=1`).
- `data/`: The 100 synthetic benchmark datasets (`data1_1.csv` to `data10_10.csv`).

## Requirements

- Python >= 3.8
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `statsmodels`

## Usage

1. **Generate synthetic datasets**:
   ```bash
   python data.py
   ```
2. **Run COSMI mediator identification on the 100 benchmark datasets**:
   ```bash
   python cosmi.py
   ```
