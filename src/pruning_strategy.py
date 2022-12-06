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
                    model.addCons(ind_e + ind_f == joint_ind)
                    weight_terms.append(joint_ind * weights[(e, j_e), (f, j_f)])

    model.setObjective(quicksum(weight_terms), "minimize")

    return model


if __name__ == "__main__":
    env2dim = {0: 3, 1: 4, 2: 4}
    indicators = dict()

    np.random.seed(121231)
    weights = dict()
    for e, f in itr.combinations(env2dim, 2):
        for j_e in range(env2dim[e]):
            for j_f in range(env2dim[f]):
                weights[(e, j_e), (f, j_f)] = np.random.uniform()


    model = create_model_scip(env2dim, weights)
    model.optimize()
    sol = model.getBestSol()