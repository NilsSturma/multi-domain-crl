import numpy as np
from itertools import permutations

def score_up_to_signed_perm(mat_hat, mat):
    ncols = mat.shape[1]
    min_error = float('inf')
    for perm in permutations(np.arange(ncols), ncols):
        error = np.linalg.norm(abs(mat_hat[:,perm]) - abs(mat))
        if error < min_error:
            min_error = error
            best_perm = perm   
    mat_hat = mat_hat[:, best_perm]
    for i in range(ncols):
        if np.linalg.norm(-mat_hat[:,i] - mat[:,i]) < np.linalg.norm(mat_hat[:,i] - mat[:,i]):
            mat_hat[:,i] = -mat_hat[:,i]
    return (np.linalg.norm(mat_hat - mat), mat_hat)
# This is a two-step approach: 
# First look for best permutation in terms of comparing absoulte values,
# then check if the column itself or multiplies with -1 is favorable

def get_permutation_matrix(perm: list):
    nodes = list(range(len(perm)))
    mat = np.zeros((len(perm), len(perm)), dtype=int)
    mat[nodes, perm] = 1
    return mat

def permutations_respecting_graph(A):
    l = A.shape[0]
    adj = (A!=0)
    respecting_perms = []
    for perm in permutations(range(l),l):
        is_respecting = True
        for j in range(l):
            for i in range(j+1,l):
                if adj[i,j] and perm[j] > perm[i]:
                    is_respecting = False
                    break
        if is_respecting:
            respecting_perms.append(perm)
    return(respecting_perms)
