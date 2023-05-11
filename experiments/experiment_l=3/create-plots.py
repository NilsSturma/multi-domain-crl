# === IMPORTS: BUILT-IN ===
import json
import pickle

# === IMPORTS: Local ===
from experiments.utils import get_statistics, plot



gamma=0.2

# Load
with open(f"experiments/experiment_l=3/results_gamma={gamma}/ndom=2.pkl", "rb") as f:
    info2 = pickle.load(f)

with open(f"experiments/experiment_l=3/results_gamma={gamma}/ndom=3.pkl", "rb") as f:
    info3 = pickle.load(f)


# Get statistics
nsamples_list = info2["metadata"]["nsamples_list"]
stats2 = get_statistics(info2)
stats3 = get_statistics(info3)

stats = {"2": stats2, "3": stats3}
with open(f"experiments/experiment_l=3/results_gamma={gamma}/stats.json", 'w') as f:
    json.dump(stats, f)

# Create plots
plot(nsamples_list, stats2["number_shared"], stats3["number_shared"], None,
                ylabel="Average of $\hat{\ell}$",
                path=f"experiments/experiment_l=3/results_gamma={gamma}/avg-shared-nodes.png", 
                ylim=(-0.1,3.1), error_bars=True)

plot(nsamples_list, stats2["too_many_shared_rate"], stats3["too_many_shared_rate"], None,
                ylabel="Fraction with $\hat{\ell} > \ell$",
                path=f"experiments/experiment_l=3/results_gamma={gamma}/too-many-shared-nodes.png", 
                error_bars=False)

plot(nsamples_list, stats2["mixing_error"], stats3["mixing_error"], None,
                ylabel="Median score$_B$",
                path=f"experiments/experiment_l=3/results_gamma={gamma}/mixing-error.png", 
                error_bars=True)

plot(nsamples_list, stats2["graph_error"], stats3["graph_error"], None,
                ylabel="Median score$_A$",
                path=f"experiments/experiment_l=3/results_gamma={gamma}/graph-error.png", 
                error_bars=True)

# Total time in hours: 
total_time = stats2["total_time"] + stats3["total_time"]

with open(f"experiments/experiment_l=3/results_gamma={gamma}/time.txt", 'w') as f:
    f.write(str(total_time))
