import numpy as np
import seaborn as sns
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

def get_statistics(info):
    metadata = info["metadata"]
    nexp = metadata["nexp"]
    nsamples_list = metadata["nsamples_list"]
    m = metadata["model_specs"]

    mixing_error_median = np.zeros(len(nsamples_list))
    graph_error_median = np.zeros(len(nsamples_list))
    too_many_shared_rate = np.zeros(len(nsamples_list))
    avg_number_shared = np.zeros(len(nsamples_list))
    for s_ix, n in enumerate(nsamples_list):
        mixing_errors = np.full(nexp, np.nan)
        graph_errors = np.full(nexp, np.nan)
        too_many_shared = np.full(nexp, False)
        number_shared = np.full(nexp, 0)
        for exp_ix in range(nexp):
            number_shared[exp_ix] = info["results"][(s_ix, exp_ix)]["nr_joint"]
            mixing_errors[exp_ix] = info["results"][(s_ix, exp_ix)]["mixing_error"] / \
                (min(number_shared[exp_ix], len(m["joint_idx"])) * sum(m["dims"])) ### normalizing!!
            if number_shared[exp_ix] > len(m["joint_idx"]):
                too_many_shared[exp_ix] = True
            if number_shared[exp_ix] == len(m["joint_idx"]):
                graph_errors[exp_ix] = info["results"][(s_ix, exp_ix)]["graph_error"] / \
                    (len(m["joint_idx"])) ### normalizing!!
        too_many_shared_rate[s_ix] = too_many_shared.mean()
        mixing_error_median[s_ix] = np.nanmedian(mixing_errors)
        graph_error_median[s_ix] = np.nanmedian(graph_errors)
        avg_number_shared[s_ix] = number_shared.mean()
    return (avg_number_shared, too_many_shared_rate, mixing_error_median, graph_error_median)

def plot(nsamples_list, stats2, stats3, stats4, ylabel="Score", path="test.png", 
            fontsize=13, ylim=None, legendfontsize=13):
    sns.set()
    plt.figure(figsize=(3.3,2.7))
    plt.clf()
    if ylim is not None:
        plt.ylim(ylim)
    plt.plot(nsamples_list, stats2, "-", color="blue", label="2 domains")
    plt.plot(nsamples_list, stats3, "--", color="red", label="3 domains")
    if stats4 is not None:
        plt.plot(nsamples_list, stats4, "-.", color="green", label="4 domains")
    plt.xscale("log")
    plt.xlabel("Sample size", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=legendfontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
