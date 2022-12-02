import numpy as np
import matplotlib.pyplot as plt

def sample_normed(n, rv):
    return (rv.rvs(size=n) - rv.mean()) / rv.std()

def matrix_from_normal(shape, mu=0, sigma=1):
    A = np.random.normal(loc=mu, scale=sigma, size=shape)
    return A

def create_synthetic_data(rvs, latents, sample_sizes, obs_dims):
    data = []
    for i in range(len(latents)):
        env_rvs = [rvs[j] for j in latents[i]]
        eps = np.zeros((sample_sizes[i],len(env_rvs)))
        G = matrix_from_normal(shape=(obs_dims[i],len(latents[i])))
        for i, rv in enumerate(env_rvs):
            eps[:,i] = sample_normed(sample_sizes[i],rv)
        data.append(np.transpose(np.matmul(G, np.transpose(eps))))
    return data

def plot_hist_noise(indep_comps):
    nr_env = len(indep_comps)
    max_nr_comps = max([eps.shape[1] for eps in indep_comps])
    fig, ax = plt.subplots(nr_env, max_nr_comps, figsize=(9, 9))
    for i in range(nr_env):
        for j in range(indep_comps[i].shape[1]):
            ax[i,j].hist(indep_comps[i][:,j], bins=50)
    fig.savefig('plot.png')