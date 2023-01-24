# ===== IMPORTS =====
import pickle
import numpy as np
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform, expon, lognorm, weibull_min, chi2, t, gumbel_r, skewnorm
from utils import get_statistics, plot


# ===== LOAD =====
l = 5
d = 48

with open(f"results-paper/nr_doms=2_l={l}_d={d}.pkl", "rb") as f:
    info2 = pickle.load(f)
with open(f"results-paper/nr_doms=3_l={l}_d={d}.pkl", "rb") as f:
    info3 = pickle.load(f)
with open(f"results-paper/nr_doms=4_l={l}_d={d}.pkl", "rb") as f:
    info4 = pickle.load(f)

# ===== GET STATISTICS =====
nsamples_list = info2["metadata"]["nsamples_list"]
avg_number_shared2, too_many_shared_rate2, mixing_error_rate2, graph_error_median2 = get_statistics(info2)
avg_number_shared3, too_many_shared_rate3, mixing_error_rate3, graph_error_median3 = get_statistics(info3)
avg_number_shared4, too_many_shared_rate4, mixing_error_rate4, graph_error_median4 = get_statistics(info4)

# ===== PLOT =====
plot(nsamples_list, avg_number_shared2, avg_number_shared3, avg_number_shared4,
                ylabel="Average of $\hat{\ell}$",
                path=f"results-paper/avg-shared-nodes_l={l}_d={d}.png", ylim=(-0.1,5.1))

plot(nsamples_list, too_many_shared_rate2, too_many_shared_rate3, too_many_shared_rate4,
                ylabel="Fraction with $\hat{\ell} > \ell$",
                path=f"results-paper/too-many-shared-nodes_l={l}_d={d}.png")

plot(nsamples_list, mixing_error_rate2, mixing_error_rate3, mixing_error_rate4,
                ylabel="score$_B$",
                path=f"results-paper/mixing-error_l={l}_d={d}.png")

plot(nsamples_list, graph_error_median2, graph_error_median3, graph_error_median4,
                ylabel="score$_A$",
                path=f"results-paper/graph-error_l={l}_d={d}.png")
