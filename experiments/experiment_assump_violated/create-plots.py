# === IMPORTS: BUILT-IN ===
import json
import pickle

# === IMPORTS: Local ===
from experiments.utils import get_statistics, plot



# Load
with open(f"experiments/experiment_l=3/results_gamma=0.2/ndom=3.pkl", "rb") as f:
    info_normal = pickle.load(f)
with open(f"experiments/experiment_assump_violated/results/same_errors.pkl", "rb") as f:
    info_same_errors = pickle.load(f)
with open(f"experiments/experiment_assump_violated/results/no_pure_children.pkl", "rb") as f:
    info_no_pure_children = pickle.load(f)

# Get statistics
nsamples_list = info_normal["metadata"]["nsamples_list"]
stats2 = get_statistics(info_normal)
stats3 = get_statistics(info_same_errors)
stats4 = get_statistics(info_no_pure_children)

stats = {"2": stats2, "3": stats3, "4": stats4}
with open(f"experiments/experiment_assump_violated/results/stats.json", 'w') as f:
    json.dump(stats, f)

# Create plots
labels=["Assump. satisfied", "Same errors", "No pure children"]

plot(nsamples_list, stats2["number_shared"], stats3["number_shared"], stats4["number_shared"],
                ylabel="Average of $\hat{\ell}$",
                path="experiments/experiment_assump_violated/results/avg-shared-nodes.png", 
                ylim=(-0.1,3.1), labels=labels, legendfontsize=9, error_bars=True)

plot(nsamples_list, stats2["too_many_shared_rate"], stats3["too_many_shared_rate"], stats4["too_many_shared_rate"],
                ylabel="Fraction with $\hat{\ell} > \ell$",
                path="experiments/experiment_assump_violated/results/too-many-shared-nodes.png", 
                labels=labels, legendfontsize=9, error_bars=False)

plot(nsamples_list, stats2["mixing_error"], stats3["mixing_error"], stats4["mixing_error"],
                ylabel="Median score$_B$",
                path="experiments/experiment_assump_violated/results/mixing-error.png", 
                labels=labels, legendfontsize=9, error_bars=True)

plot(nsamples_list, stats2["graph_error"], stats3["graph_error"], stats4["graph_error"],
                ylabel="Median score$_A$",
                path="experiments/experiment_assump_violated/results/graph-error.png", 
                labels=labels, legendfontsize=9, error_bars=True)

# Total time in hours: 
total_time = stats2["total_time"] + stats3["total_time"] + stats4["total_time"]

with open(f"experiments/experiment_assump_violated/results/time.txt", 'w') as f:
    f.write(str(total_time))
    
