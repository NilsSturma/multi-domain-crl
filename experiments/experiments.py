# === IMPORTS: BUILT-IN ===
from time import time

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.learner import LinearMDCRL
from experiments.rand import rand_model



def run_experiments(info):

    metadata = info["metadata"]
    nexp = metadata["nexp"]
    nsamples_list = metadata["nsamples_list"]

    for s_ix, n in enumerate(nsamples_list):
        
        # Info on progress
        print(n)

        for exp_ix in range(nexp):
            
            # Info on progress
            if not exp_ix%50:
                print(exp_ix)
                
            # Sample data
            m = info["metadata"]["model_specs"].copy()
            m["sample_sizes"] = [n for _ in range(m["nr_doms"])]
            data, g, B_true = rand_model(m)
            A_true = np.transpose(g.to_amat())

            # Fit model
            model = LinearMDCRL(measure=metadata["measure"], 
                                alpha=metadata["alpha"], 
                                gamma=metadata["gamma"])
            start_time = time()
            model.fit(data)
            time_spent = time() - start_time

            # Score output
            nr_joint = model.nr_joint
            mixing_error = model.score_shared_columns(B_true, len(m["joint_idx"]))
            if nr_joint == len(m["joint_idx"]):
                graph_error, A_hat = model.score_graph_param_matrix(A_true)
            else:
                graph_error = None

            # Save results
            info["results"][(s_ix, exp_ix)] = dict(
                nr_joint = nr_joint,
                mixing_error = mixing_error,
                graph_error = graph_error,
                time_spent = time_spent
            )
    return info
