import matplotlib.pyplot as plt

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
