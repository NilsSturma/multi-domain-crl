import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import wasserstein_distance


# === IMPORTS: LOCAL ===
from src.dist import third_moments_distance
from src.integer_program import IntegerProgram



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

