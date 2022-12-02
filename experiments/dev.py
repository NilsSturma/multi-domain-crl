import numpy as np
from src.learner import LinearMDCRL
from experiments.utils import create_synthetic_data, plot_hist_noise
from scipy.stats import uniform, expon, lognorm, weibull_min, chi2



###########################
## Create synthetic data ##
###########################

rvs = [uniform(), expon(scale=0.1), lognorm(s=1), weibull_min(c=2), chi2(df=4)]
nlat = 5
joint_latents = [0,1]
env1_specific = [2]
env2_specific = [3]
env3_specific = [4]
d1 = 10
d2 = 11
d3 = 12
n1 = 1000
n2 = 1000
n3 = 1000

env1_latents = joint_latents + env1_specific
env2_latents = joint_latents + env2_specific
env3_latents = joint_latents + env3_specific

latents = [env1_latents, env2_latents, env3_latents]
sample_sizes = [n1, n2, n3]
obs_dims = [d1,d2,d3]

data = create_synthetic_data(rvs, latents, sample_sizes, obs_dims)

data[2].shape

model = LinearMDCRL()
model.fit(data)

plot_hist_noise(model.indep_comps)

model.joint_factors
model.joint_mixing.shape
model.joint_mixing.round(2)


# TODO: 
# - Solve "signs are not consistent error" (maybe reduce it to a warning?)
# - Implement a routine to check the solution of recovering the joint mixing matrix 
# (check nr_factors and zero_pattern)
# - Add random Erdös Renyi graph in synthetic data creation
# - Implement graph recovery 
# (To find sparse rows: Minimize l_2 norm and simultaneously verify that pure children 
# via low singular value)