import numpy as np
from scipy.stats import moment

def third_moments_distance(X,Y):
    M_X = moment(X, moment=3) # central moment
    M_Y = moment(Y, moment=3)
    dist = abs(M_X - M_Y)
    return dist

def minimum_matching(D, A=10):
    p1, p2 = D.shape
    threshold = A * D.min()
    matching = {}
    for i in range(p1):
        row_argmin = D[i,:].argmin()
        row_min = D[i,:].min()
        if (D[:,row_argmin].argmin() == i) and (row_min < threshold):
            matching[i] = row_argmin
    return matching