# === IMPORTS: BUILT-IN ===
import json
import pickle

# === IMPORTS: Local ===
from experiments.utils import get_statistics, plot



# Load
with open("experiments/experiment_l=5/results/ndom=2.pkl", "rb") as f:
    info2 = pickle.load(f)
with open("experiments/experiment_l=5/results/ndom=3.pkl", "rb") as f:
    info3 = pickle.load(f)
with open("experiments/experiment_l=5/results/ndom=4.pkl", "rb") as f:
    info4 = pickle.load(f)

# Get statistics
nsamples_list = info2["metadata"]["nsamples_list"]
stats2 = get_statistics(info2)
stats3 = get_statistics(info3)
stats4 = get_statistics(info4)

stats = {"2": stats2, "3": stats3, "4": stats4}
with open(f"experiments/experiment_l=5/results/stats.json", 'w') as f:
    json.dump(stats, f)

with open(f"experiments/experiment_l=5/results/stats.json", 'r') as f:
    stats = json.load(f)
nsamples_list = [1000, 2500, 5000, 10000, 25000]
stats2 = stats["2"]
stats3 = stats["3"]
stats4 = stats["4"]

# Create plots
plot(nsamples_list, stats2["number_shared"], stats3["number_shared"], stats4["number_shared"],
                ylabel="Average of $\hat{\ell}$",
                path="experiments/experiment_l=5/results/avg-shared-nodes.png", 
                ylim=(-0.1,5.1),  legendfontsize=8, error_bars=True)

plot(nsamples_list, stats2["too_many_shared_rate"], stats3["too_many_shared_rate"], stats4["too_many_shared_rate"],
                ylabel="Fraction with $\hat{\ell} > \ell$",
                path="experiments/experiment_l=5/results/too-many-shared-nodes.png", 
                legendfontsize=8, error_bars=False)

plot(nsamples_list, stats2["mixing_error"], stats3["mixing_error"], stats4["mixing_error"],
                ylabel="Median score$_B$",
                path="experiments/experiment_l=5/results/mixing-error.png", 
                error_bars=True,legendfontsize=8)

plot(nsamples_list, stats2["graph_error"], stats3["graph_error"], stats4["graph_error"],
                ylabel="Median score$_A$",
                path="experiments/experiment_l=5/results/graph-error.png", 
                error_bars=True, legendfontsize=8)

# Total time in hours: 
total_time = stats2["total_time"] + stats3["total_time"] + stats4["total_time"]

with open(f"experiments/experiment_l=5/results/time.txt", 'w') as f:
    f.write(str(total_time))
