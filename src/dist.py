from scipy.stats import moment

def third_moments_distance(X,Y):
    M_X = moment(X, moment=3) # central moment
    M_Y = moment(Y, moment=3)
    dist = abs(M_X - M_Y)
    return dist
