import numpy as np
import pandas as pd

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Structural Causal Model (SCM) as functions
def scm_data_generation(n_samples, n_confounder, n_mediator, seed=1):
    np.random.seed(seed)

    # Exogenous noise terms
    U_t = np.random.normal(0, 1, n_samples)

    # Generate confounders
    confounders = [np.random.normal(0, 1, n_samples) for _ in range(n_confounder)]

    # Treatment variable as function of confounders
    treatment_score = sum(confounders) + U_t
    p = sigmoid(treatment_score)
    treatment = np.random.binomial(1, p)

    print("Probability stats:")
    print(f"Mean: {np.mean(p):.4f}")
    print(f"Min: {np.min(p):.4f}")
    print(f"Max: {np.max(p):.4f}")

    # Generate mediators
    mediators = [2 * treatment + np.random.normal(0, 1, n_samples) for _ in range(n_mediator)]

    # Outcome as function of treatment, confounders, mediators
    U_o = np.random.normal(0, 1, n_samples)
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

    return df

for i in range(1, 11):
    for j in range(1, 11):
        df = scm_data_generation(1000, i, j)
        df.to_csv('./data/data'+str(i)+'_'+str(j)+'.csv', index=False)
