import numpy as np
import matplotlib.pyplot as plt

def sample_normed(n, rv):
    return (rv.rvs(size=n) - rv.mean()) / rv.std()

def plot_error_distr(rvs, n=10000, save=False, figsize=(18,5), shape=(2,5)):
    
    samples = np.zeros((n,len(rvs)))
    for i, rv in enumerate(rvs):
        samples[:,i] = sample_normed(n,rv)

    fig, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    fig.delaxes(ax[1,4])

    counter = 0
    for i in range(2):
        for j in range(5):
            if counter < samples.shape[1]:
                ax[i,j].hist(samples[:,counter], bins=50)
                counter = counter+1
    if save:
        plt.savefig("experiments/results-paper/histograms")
    else:
        plt.show()

def plot_hist_noise(indep_comps, save=False, name="plot.png", figsize=(14,9)):
    nr_env = len(indep_comps)
    max_nr_comps = max([eps.shape[1] for eps in indep_comps])
    fig, ax = plt.subplots(nr_env, max_nr_comps, figsize=figsize)
    for i in range(nr_env):
        for j in range(indep_comps[i].shape[1]):
            ax[i,j].hist(indep_comps[i][:,j], bins=50)
    if save:
        fig.savefig(name)
    else:
        plt.show()
