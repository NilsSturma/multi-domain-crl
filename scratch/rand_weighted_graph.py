import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np
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

    # === ASSIGN POSITIVE WEIGHT FOR ALL VARIABLES IN DIFFERENT CLUSTERS
    for env1, env2 in itr.combinations(env2dim, 2):
        for ix1, ix2 in itr.product(range(env2dim[env1]), range(env2dim[env2])):
            weights[(env1, ix1), (env2, ix2)] = np.random.uniform(0.25, 1)

    # === ASSIGN WEIGHT ZERO FOR ALL VARIABLES IN SAME CLUSTERS
    for cluster in clusters:
        for (env1, ix1), (env2, ix2) in itr.combinations(cluster, 2):
            weights[(env1, ix1), (env2, ix2)] = 0
    
    return weights


def solution2clusters(sol, indicators, env2dim: dict):
    p = min(env2dim.values())
    g = nx.Graph()

    for k in range(p):
        for e, dim in env2dim.items():
            for j_e in range(dim):
                indicator = indicators[(k, e, j_e)]
                if np.isclose(sol[indicator], 1):
                    g.add_edge((e, j_e), k)

    estimated_clusters = list(nx.connected_components(g))
    
    return [{elm for elm in c if isinstance(elm, tuple)} for c in estimated_clusters]