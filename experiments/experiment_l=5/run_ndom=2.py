# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: THIRD-PARTY ===
from scipy.stats import beta, expon, lognorm, weibull_min, chi2, gumbel_r, skewnorm

# === IMPORTS: LOCAL ===
from experiments.experiments import run_experiments



# Define info-dict
rvs = [beta(2,3), beta(2,5), chi2(df=4), gumbel_r, lognorm(s=1),
    weibull_min(c=2), expon(scale=0.1), skewnorm(a=6), skewnorm(a=12)]

nsamples_list = [1000,2500,5000,10000,25000]
nexp = 1000
measure = "ks-test"
alpha=0.05
gamma=0.2

model_specs = {
    "nr_doms": 2,
    "joint_idx": [0,1,2,3,4],
    "domain_spec_idx": [[5],[6]],
    "noise_rvs": rvs,
    "sample_sizes":  None,
    "dims": [24,24],
    "graph_density": 0.75,
    "mixing_density": 0.9,
    "mixing_distribution": 'unif',  # unif or normal
    "two_pure_children": True
}

info = {"results": dict(),
        "metadata": dict(
            nsamples_list=nsamples_list,
            nexp=nexp,
            measure=measure,
            model_specs=model_specs,
            alpha=alpha,
            gamma=gamma
            )
        }

# Run experiments
info = run_experiments(info)

# Save
filename = f"experiments/experiment_l=5/results/ndom={model_specs['nr_doms']}.pkl"
with open(filename, "wb") as f:
    pickle.dump(info, f)
