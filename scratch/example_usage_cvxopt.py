# === IMPORTS: THIRD-PARTY ===
import numpy as np
import seaborn as sns
sns.set()

# === IMPORTS: LOCAL ===
from src.integer_program import IntegerProgram
from scratch.rand_weighted_graph import random_clusters, clusters2weights


nlatent = 3
num_envs = 3
env2dim = {e: nlatent for e in range(num_envs)}
true_clusters = random_clusters(env2dim)
weights = clusters2weights(env2dim, true_clusters)
ip = IntegerProgram(env2dim, weights, solver="cvxopt")
prob, indicators = ip.create_model_cvxopt()
sol = prob.solve()

estimated_clusters = ip.solve_cvxopt()