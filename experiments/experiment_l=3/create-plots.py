# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: Local ===
from experiments.utils import get_statistics, plot



# Load
with open("experiments/experiment_l=3/results/ndom=2.pkl", "rb") as f:
    info2 = pickle.load(f)
with open(f"experiments/experiment_l=3/results/ndom=3.pkl", "rb") as f:
    info3 = pickle.load(f)

# Get statistics
nsamples_list = info2["metadata"]["nsamples_list"]
avg_number_shared2, too_many_shared_rate2, mixing_error_rate2, graph_error_median2 = get_statistics(info2)
avg_number_shared3, too_many_shared_rate3, mixing_error_rate3, graph_error_median3 = get_statistics(info3)

# Create plots
plot(nsamples_list, avg_number_shared2, avg_number_shared3, None,
                ylabel="Average of $\hat{\ell}$",
                path="experiments/experiment_l=3/results/avg-shared-nodes.png", ylim=(-0.1,3.1))

plot(nsamples_list, too_many_shared_rate2, too_many_shared_rate3, None,
                ylabel="Fraction with $\hat{\ell} > \ell$",
                path="experiments/experiment_l=3/results/too-many-shared-nodes.png")

plot(nsamples_list, mixing_error_rate2, mixing_error_rate3, None,
                ylabel="Median score$_B$",
                path="experiments/experiment_l=3/results/mixing-error.png")

plot(nsamples_list, graph_error_median2, graph_error_median3, None,
                ylabel="Median score$_A$",
                path="experiments/experiment_l=3/results/graph-error.png")
