import numpy as np
import pandas as pd

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Structural Causal Model (SCM) as functions
def scm_data_generation(n_samples, n_confounder, n_mediator, seed=1, n_nonmediator_z=0):
    rng = np.random.RandomState(seed)

    # Exogenous noise terms
    U_t = rng.normal(0, 1, n_samples)

    # Generate confounders
    confounders = [rng.normal(0, 1, n_samples) for _ in range(n_confounder)]

    # Treatment variable as function of confounders
    treatment_score = sum(confounders) + U_t
    p = sigmoid(treatment_score)
    treatment = rng.binomial(1, p)

    # Generate mediators
    mediators = [2 * treatment + rng.normal(0, 1, n_samples) for _ in range(n_mediator)]

    # Outcome as function of treatment, confounders, mediators
    U_o = rng.normal(0, 1, n_samples)
    outcome = treatment + sum(confounders) + sum(mediators) + U_o

    # Build DataFrame
    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome
    })

    for i, m in enumerate(mediators):
        df[f'mediator{i}'] = m
    for i, c in enumerate(confounders):
        df[f'confounder{i}'] = c
    for k in range(n_nonmediator_z):
        df[f'nonmediator{k}'] = 2 * treatment + rng.normal(0, 1, n_samples)

    return df

if __name__ == '__main__':
    # 1. Generate the 100 main benchmark datasets (Section 3.1)
    for i in range(1, 11):
        for j in range(1, 11):
            df = scm_data_generation(1000, i, j, seed=1)
            df.to_csv(f'./data/data{i}_{j}.csv', index=False)

    # 2. Generate the 75 post-treatment non-mediator (Z) datasets (Section 3.6)
    levels = [1, 3, 5, 7, 10]
    for z in [1, 5, 10]:
        for idx_c, c in enumerate(levels):
            for idx_m, m in enumerate(levels):
                seed = 10001 + 5 * idx_c + idx_m
                df = scm_data_generation(1000, c, m, seed=seed, n_nonmediator_z=z)
                df.to_csv(f'./data/negative_c{c}_m{m}_z{z}.csv', index=False)

    # 3. Generate the 25 independent validation datasets for sensitivity analysis (Section 3.7)
    for idx_c, c in enumerate(levels):
        for idx_m, m in enumerate(levels):
            seed = 10001 + 5 * idx_c + idx_m
            df = scm_data_generation(1000, c, m, seed=seed)
            df.to_csv(f'./data/validation_c{c}_m{m}_z0.csv', index=False)
