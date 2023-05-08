# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: Local ===
from experiments.utils import get_statistics, plot



# Load
with open(f"experiments/experiment_l=3/results/ndom=3.pkl", "rb") as f:
    info_normal = pickle.load(f)
with open(f"experiments/experiment_assump_violated/results/same_errors.pkl", "rb") as f:
    info_same_errors = pickle.load(f)
with open(f"experiments/experiment_assump_violated/results/no_pure_children.pkl", "rb") as f:
    info_no_pure_children = pickle.load(f)

# Get statistics
nsamples_list = info_normal["metadata"]["nsamples_list"]
avg_number_shared1, too_many_shared_rate1, mixing_error_rate1, graph_error_median1 = get_statistics(info_normal)
avg_number_shared2, too_many_shared_rate2, mixing_error_rate2, graph_error_median2 = get_statistics(info_same_errors)
avg_number_shared3, too_many_shared_rate3, mixing_error_rate3, graph_error_median3 = get_statistics(info_no_pure_children)

# Create plots
labels=["Assump. satisfied", "Same errors", "No pure children"]

plot(nsamples_list, avg_number_shared1, avg_number_shared2, avg_number_shared3,
                ylabel="Average of $\hat{\ell}$",
                path="experiments/experiment_assump_violated/results/avg-shared-nodes.png", 
                ylim=(-0.1,3.1), labels=labels, legendfontsize=9)

plot(nsamples_list, too_many_shared_rate1, too_many_shared_rate2, too_many_shared_rate3,
                ylabel="Fraction with $\hat{\ell} > \ell$",
                path="experiments/experiment_assump_violated/results/too-many-shared-nodes.png", 
                labels=labels, legendfontsize=9)

plot(nsamples_list, mixing_error_rate1, mixing_error_rate2, mixing_error_rate3,
                ylabel="Median score$_B$",
                path="experiments/experiment_assump_violated/results/mixing-error.png", 
                labels=labels, legendfontsize=9)

plot(nsamples_list, graph_error_median1, graph_error_median2, graph_error_median3,
                ylabel="Median score$_A$",
                path="experiments/experiment_assump_violated/results/graph-error.png", 
                labels=labels, legendfontsize=9)
