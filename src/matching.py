def minimum_matching(D, A=10):
    p = D.shape[0]
    threshold = A * D.min()
    matching = {}
    for i in range(p):
        row_argmin = D[i,:].argmin()
        row_min = D[i,:].min()
        if (D[:,row_argmin].argmin() == i) and (row_min < threshold):
            matching[i] = row_argmin
    return matching
