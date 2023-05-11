# === IMPORTS: THIRD-PARTY ===
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
    mixing_error_lower = np.zeros(len(nsamples_list))
    mixing_error_upper = np.zeros(len(nsamples_list))
    graph_error_median = np.zeros(len(nsamples_list))
    graph_error_lower = np.zeros(len(nsamples_list))
    graph_error_upper = np.zeros(len(nsamples_list))
    too_many_shared_rate = np.zeros(len(nsamples_list))
    too_many_shared_std = np.zeros(len(nsamples_list))
    avg_number_shared = np.zeros(len(nsamples_list))
    std_number_shared = np.zeros(len(nsamples_list))
    #lower_number_shared = np.zeros(len(nsamples_list))
    #upper_number_shared = np.zeros(len(nsamples_list))
    total_time = 0

    for s_ix, n in enumerate(nsamples_list):

        mixing_errors = np.full(nexp, np.nan)
        graph_errors = np.full(nexp, np.nan)
        too_many_shared = np.full(nexp, False)
        number_shared = np.full(nexp, 0)
      
        for exp_ix in range(nexp):
            number_shared[exp_ix] = info["results"][(s_ix, exp_ix)]["nr_joint"]
            mixing_errors[exp_ix] = info["results"][(s_ix, exp_ix)]["mixing_error"] / \
                (min(number_shared[exp_ix], len(m["joint_idx"])) * sum(m["dims"])) 
            if number_shared[exp_ix] > len(m["joint_idx"]):
                too_many_shared[exp_ix] = True
            if number_shared[exp_ix] == len(m["joint_idx"]):
                graph_errors[exp_ix] = info["results"][(s_ix, exp_ix)]["graph_error"] / \
                    len(m["joint_idx"])
            total_time = total_time + info["results"][(s_ix, exp_ix)]["time_spent"]
                
        too_many_shared_rate[s_ix] = too_many_shared.mean()
        too_many_shared_std[s_ix] = too_many_shared.std()
        mixing_error_median[s_ix] = np.nanmedian(mixing_errors)
        mixing_error_lower[s_ix] = np.nanquantile(mixing_errors, 0.25)
        mixing_error_upper[s_ix] = np.nanquantile(mixing_errors, 0.75)
        graph_error_median[s_ix] = np.nanmedian(graph_errors)
        graph_error_lower[s_ix] = np.nanquantile(graph_errors, 0.25)
        graph_error_upper[s_ix] = np.nanquantile(graph_errors, 0.75)
        avg_number_shared[s_ix] = number_shared.mean()  # np.nanmedian(number_shared)
        std_number_shared[s_ix] = number_shared.std()
        #lower_number_shared[s_ix] = np.nanquantile(number_shared, 0.25)
        #upper_number_shared[s_ix] = np.nanquantile(number_shared, 0.75)

    result = {}
    result["too_many_shared_rate"] = {"mean": too_many_shared_rate.tolist(), 
                                      "lower": (too_many_shared_rate-too_many_shared_std).tolist(),
                                      "upper": (too_many_shared_rate+too_many_shared_std).tolist()}
    result["number_shared"] = {"mean": avg_number_shared.tolist(), 
                               "lower": (avg_number_shared-std_number_shared).tolist(),
                               "upper": (avg_number_shared+std_number_shared).tolist()}
    result["mixing_error"] = {"mean": mixing_error_median.tolist(), 
                              "lower": mixing_error_lower.tolist(),
                              "upper": mixing_error_upper.tolist()}
    result["graph_error"] = {"mean": graph_error_median.tolist(), 
                             "lower": graph_error_lower.tolist(),
                             "upper": graph_error_upper.tolist()}
    result["total_time"] = total_time

    return result


def plot(nsamples_list, stats2, stats3, stats4, ylabel="Score", path="test.png", 
            fontsize=13, ylim=None, legendfontsize=13, labels=["2 domains", "3 domains", "4 domains"], error_bars=False):
    
    sns.set()
    plt.figure(figsize=(3.3,2.7))
    plt.clf()
    if ylim is not None:
        plt.ylim(ylim)
    yerr = None


    # Stats2
    if (stats2["lower"] is not None) and error_bars:
        yerr_lower = np.asarray(stats2["mean"]) - np.asarray(stats2["lower"])
        yerr_upper = np.asarray(stats2["upper"]) - np.asarray(stats2["mean"])
        yerr = np.row_stack((yerr_lower, yerr_upper))
    plt.errorbar(x=nsamples_list, y=stats2["mean"], yerr=yerr,
                 linestyle="-", color="blue", label=labels[0], alpha=0.4)
    #plt.plot(nsamples_list, stats2["mean"], "-", color="blue", label=labels[0])
    #if (stats2["lower"] is not None) and error_bars:
    #    plt.fill_between(nsamples_list, stats2["lower"], stats2["upper"], alpha=0.5)

    # Stats3
    if (stats3["lower"] is not None) and error_bars:
        yerr_lower = np.asarray(stats3["mean"]) - np.asarray(stats3["lower"])
        yerr_upper = np.asarray(stats3["upper"]) - np.asarray(stats3["mean"])
        yerr = np.row_stack((yerr_lower, yerr_upper))
    plt.errorbar(x=nsamples_list, y=stats3["mean"], yerr=yerr,
                 linestyle="--", color="red", label=labels[1], alpha=0.4, elinewidth=3)
    #plt.plot(nsamples_list, stats3["mean"], "--", color="red", label=labels[1])
    #if (stats3["lower"] is not None) and error_bars:
    #    plt.fill_between(nsamples_list, stats3["lower"], stats3["upper"], alpha=0.5)

    # Stats4
    if stats4 is not None:
        if (stats4["lower"] is not None) and error_bars:
            yerr_lower = np.asarray(stats4["mean"]) - np.asarray(stats4["lower"])
            yerr_upper = np.asarray(stats4["upper"]) - np.asarray(stats4["mean"])
            yerr = np.row_stack((yerr_lower, yerr_upper))
        plt.errorbar(x=nsamples_list, y=stats4["mean"], yerr=yerr,
                 linestyle="-.", color="green", label=labels[2], alpha=0.4, elinewidth=5)
        #plt.plot(nsamples_list, stats4["mean"], "-.", color="green", label=labels[2])
        #if (stats4["lower"] is not None) and error_bars:
        #    plt.fill_between(nsamples_list, stats4["lower"], stats4["upper"], alpha=0.5)

    plt.xscale("log")
    plt.xlabel("Sample size", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=legendfontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(path)
    #plt.show()
