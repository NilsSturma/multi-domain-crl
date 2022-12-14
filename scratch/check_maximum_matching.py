# === IMPORTS: BUILT-IN ===
from time import time
from functools import partial

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# === IMPORTS: LOCAL ===
from src.integer_program import IntegerProgram
from scratch.rand_weighted_graph import random_clusters, clusters2weights



class ResultCollector:
    def __init__(
        self, 
        nruns: int = 30,
        nlatent_min: int = 3,
        nlatent_max: int = 4,
        num_envs_list: list = [3, 4, 5]
    ):
        self.nruns = nruns
        self.nlatent_min = nlatent_min
        self.nlatent_max = nlatent_max
        self.num_envs_list = num_envs_list

    def run(self, names2algs):
        matches = {name: np.zeros((len(self.num_envs_list), self.nruns)) for name in names2algs}
        times = {name: np.zeros((len(self.num_envs_list), self.nruns)) for name in names2algs}

        for e_ix, num_envs in enumerate(self.num_envs_list):
            for r in trange(self.nruns):
                for name, params in names2algs.items():
                    # STEP 1: generate random true clusters and weights
                    env2dim = {e: np.random.randint(self.nlatent_min, self.nlatent_max) for e in range(num_envs)}
                    true_clusters = random_clusters(env2dim)
                    weights = clusters2weights(env2dim, true_clusters)

                    # === QUADRATIC CONSTRAINT ===
                    # STEP 2: run integer program
                    start = time()
                    ip = IntegerProgram(env2dim, weights, **params)
                    estimated_clusters = ip.solve()
                    elapsed_time = time() - start

                    # STEP 3: check solutions match
                    true_clusters = {frozenset(c) for c in true_clusters}
                    estimated_clusters = {frozenset(c) for c in estimated_clusters}
                    match = true_clusters == estimated_clusters
                    if not match:
                        print('quadratic')
                        raise ValueError
                    matches[name][e_ix, r] = match
                    times[name][e_ix, r] = elapsed_time

        return matches, times

    





 

# env2dim = {e: np.random.randint(2, 4) for e in range(4)}
# true_clusters = random_clusters(env2dim)
# weights = clusters2weights(env2dim, true_clusters)

# model, indicators = names2algs["linear_gurobi"](env2dim, weights)
# model.optimize()
# breakpoint()
# solution = model.getBestSol()
# estimated_clusters = solution2clusters(solution, indicators, env2dim)



col = ResultCollector()
names2algs = {
    "linear_symmetry": dict(linear_constraint=True, symmetry_breaking=True),
    "quadratic_symmetry": dict(linear_constraint=False, symmetry_breaking=True),
    "quadratic_gurobi": dict(linear_constraint=False, symmetry_breaking=True, solver="gurobi"),
    "linear_gurobi": dict(linear_constraint=True, symmetry_breaking=True, solver="gurobi"),
}
matches, times = col.run(names2algs)
plt.clf()
plt.plot(col.num_envs_list, times["linear_symmetry"].mean(axis=1), label="Linear constraint (SCIP)")
plt.plot(col.num_envs_list, times["quadratic_symmetry"].mean(axis=1), label="Quadratic constraint (SCIP)")
plt.plot(col.num_envs_list, times["quadratic_gurobi"].mean(axis=1), label="Quadratic (Gurobi)")
plt.plot(col.num_envs_list, times["linear_gurobi"].mean(axis=1), label="Linear (Gurobi)")
plt.xlabel("Number of environments")
plt.ylabel("Average solution time")
plt.legend()
plt.tight_layout()
plt.savefig(f"scratch/solution_times_quad_vs_lin_p={col.nlatent_min}-{col.nlatent_max}.png")

