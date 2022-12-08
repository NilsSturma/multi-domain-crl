import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import wasserstein_distance
import cvxpy as cp

# === IMPORTS: LOCAL ===
from src.dist import third_moments_distance

from pyscipopt import Model, quicksum, multidict


class PruningStrategy:
    def __init__(self, metric="1-wasserstein", matching="minimum"):
        if metric=="1-wasserstein":
            self.metric = wasserstein_distance
        elif metric=="third-moments":
            self.metric = third_moments_distance
        else:
            raise NotImplementedError("Metric not implemented")

    def match(self, distances):
        pass


def create_model_cvxopt(env2dim: dict, weights):
    p = min(env2dim.values())

    # === CREATE THE VARIABLES
    for k in range(p):
        for e, dim in env2dim.items():
            indicators[(k, e)] = cp.Variable(dim, boolean=True)
    
    # === CREATE THE CONSTRAINTS
    constraints = []
    # each node belongs to at most one cluster
    for e, dim in env2dim.items():
        for j_e in range(dim):
            inds = [indicators[(k, e)][j_e] for k in range(p)]
            constraints.append(sum(inds) <= 1)

    # each cluster has one node from each environment
    for k in range(p):
        for e, dim in env2dim.items():
            inds = [indicators[(k, e)][j_e] for j_e in range(dim)]
            constraints.append(sum(inds) == 1)

    # === CREATE THE OBJECTIVE
    weight = 0
    for k in range(p):
        for e, f in itr.combinations(env2dim, 2):
            for j_e in range(env2dim[e]):
                for j_f in range(env2dim[f]):
                    ind_e = indicators[(k, e)][j_e]
                    ind_f = indicators[(k, f)][j_f]
                    weight += ind_e * ind_f * weights[(e, j_e), (f, j_f)]

    objective = cp.Minimize(weight)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver="SCIP")


def create_model_scip(env2dim: dict, weights):
    model = Model("minimum")
    p = min(env2dim.values())

    indicators = dict()
    for k in range(p):
        for e, dim in env2dim.items():
            for j_e in range(dim):
                indicators[(k, e, j_e)] = model.addVar(vtype="B", name=f"A_{e}{j_e}^{k}")
    
    # === CREATE THE CONSTRAINTS
    # each node belongs to at most one cluster
    for e, dim in env2dim.items():
        for j_e in range(dim):
            inds = [indicators[(k, e, j_e)] for k in range(p)]
            model.addCons(quicksum(inds) <= 1, f"Node_{e}{j_e}")

    # each cluster has one node from each environment
    for k in range(p):
        for e, dim in env2dim.items():
            inds = [indicators[(k, e, j_e)] for j_e in range(dim)]
            model.addCons(quicksum(inds) == 1, f"Cluster_{e}{k}")

    # === CREATE THE OBJECTIVE
    weight_terms = []
    for k in range(p):
        for e, f in itr.combinations(env2dim, 2):
            for j_e in range(env2dim[e]):
                for j_f in range(env2dim[f]):
                    ind_e = indicators[(k, e, j_e)]
                    ind_f = indicators[(k, f, j_e)]
                    joint_ind = model.addVar(vtype="B", name=f"A_{e}{j_e},{f}{j_f}^k")
                    model.addCons(ind_e * ind_f == joint_ind)
                    weight_terms.append(joint_ind * weights[(e, j_e), (f, j_f)])

    model.setObjective(quicksum(weight_terms), "maximize")

    return model, indicators


if __name__ == "__main__":
    env2dim = {0: 3, 1: 4, 2: 4}
    indicators = dict()

    import networkx as nx

    def random_clusters(env2dim: dict):
        p = min(env2dim.values())

        clusters = [[(0, i)] for i in range(p)]
        for env in range(1, len(env2dim)):
            perm = np.random.permutation(list(range(env2dim[env])))
            for i in range(p):
                clusters[i].append((env, perm[i]))
        
        return clusters

    def clusters2weights(env2dim, clusters: list):
        weights = dict()

        # === ASSIGN WEIGHT ZERO FOR ALL VARIABLES IN DIFFERENT CLUSTERS
        for env1, env2 in itr.combinations(env2dim, 2):
            for ix1, ix2 in itr.product(range(env2dim[env1]), range(env2dim[env2])):
                weights[(env1, ix1), (env2, ix2)] = 0

        # === ASSIGN WEIGHT ONE FOR ALL VARIABLES IN SAME CLUSTERS
        for cluster in clusters:
            for (env1, ix1), (env2, ix2) in itr.combinations(cluster, 2):
                weights[(env1, ix1), (env2, ix2)] = 1
        
        return weights

    clusters = random_clusters(env2dim)
    weights = clusters2weights(env2dim, clusters)
    # from scipy.stats import beta, uniform, expon, lognorm, weibull_min, chi2, t, gumbel_r, skewnorm

    # from experiments.rand import rand_model

    # model_specs = {
    #     "nr_doms": 3,
    #     "joint_idx": [0, 1, 2],
    #     "domain_spec_idx": [[3], [4], [5]],
    #     "noise_rvs": [
    #         beta(2,3), 
    #         expon(scale=0.1), 
    #         skewnorm(a=6), 
    #         gumbel_r,
    #         lognorm(s=1), 
    #         weibull_min(c=2), 
    #         chi2(df=6)
    #     ],
    #     "sample_sizes":  [10000, 10000, 10000],
    #     "dims": [10, 10, 10],
    #     "graph_density": 0.75,
    #     "mixing_density": 0.9,
    #     "mixing_distribution": 'unif',  # unif or normal
    #     "indep_domain_spec": True
    # }
    # data, g, B_large = rand_model(model_specs)

    # np.random.seed(121231)
    # weights = dict()
    # for e, f in itr.combinations(env2dim, 2):
    #     for j_e in range(env2dim[e]):
    #         for j_f in range(env2dim[f]):
    #             weights[(e, j_e), (f, j_f)] = np.random.uniform()

    def solution2clusters(sol, indicators, env2dim: dict):
        p = min(env2dim.values())
        g = nx.Graph()

        for k in range(p):
            for e, dim in env2dim.items():
                for j_e in range(dim):
                    indicator = indicators[(k, e, j_e)]
                    if sol[indicator] == 1:
                        g.add_edge((e, j_e), k)

        estimated_clusters = list(nx.connected_components(g))
        
        return estimated_clusters

    model, indicators = create_model_scip(env2dim, weights)
    model.optimize()
    sol = model.getBestSol()

    estimated_clusters = solution2clusters(sol, indicators, env2dim)

    for cluster in estimated_clusters:
        pass