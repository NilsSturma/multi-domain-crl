import numpy as np
import causaldag as cd

def sample_normed(n, rv):
    return (rv.rvs(size=n) - rv.mean()) / rv.std()

def binomial(p, size=1):
    return np.random.binomial(1, p, size=size)

def unif_non_zero(low=0.25, high=1, size=1):
    signs = (binomial(0.5, size) - 0.5) * 2
    return signs * np.random.uniform(low, high, size=size)

def rand_generator(size=1, distribution='normal'):
    if distribution=='unif':
        return unif_non_zero(low=0.25, high=1, size=size)
    elif distribution=='normal':
        return np.random.normal(loc=0,scale=1,size=size)
    else:
        raise NotImplementedError("Distribution not implemented.")

def check_model_consistency(m):

    # Check keys
    list(m.keys()) == ['nr_doms', 'joint_idx', 'domain_spec_idx', 'noise_rvs', 
        'sample_sizes', 'dims', 'graph_density', 'mixing_density', 'mixing_distribution', 
        'two_pure_children']

    # Check for correct length of parameter lists
    if (m["nr_doms"] != len(m["domain_spec_idx"])) \
        or (m["nr_doms"] != len(m["sample_sizes"])) \
        or (m["nr_doms"] != len(m["dims"])):
        raise ValueError("Input parameter has different length than nr_doms.")
    
    total_nr_noise = len(m["joint_idx"]) + sum([len(_) for _ in m["domain_spec_idx"]])
    if (total_nr_noise > len(m["noise_rvs"])):
        raise ValueError("Too few noise variables specified.")

#TODO: Add more checks on types, non-Gaussian noise_rvs, different noise_rvs, etc.

def pos_pure_children(G, nr_joints):
    if nr_joints==0:
        return G
    else:
        G_joint = G[:,:nr_joints].copy()
        d, l = G_joint.shape
        for i in range(d):
            m = np.abs(G_joint[i,:]).argmax()
            mask = np.ones(l, dtype=bool)
            mask[m] = False
            if np.sum(np.abs(G_joint[i,mask])) == 0:
                G[i,m] = abs(G[i,m])
        return G

def rand_G(shape, pure_children=0, nr_joints=0, density=0.75, distribution='normal'):
    G = np.zeros(shape)
    if pure_children > 0:
        if (shape[0] < pure_children * nr_joints) or (nr_joints > shape[1]):
            raise ValueError("Inconsistent inputs.")
        for i in range(nr_joints):
            G[(pure_children*i):(pure_children*i+pure_children),i] \
                = rand_generator(size=pure_children, distribution=distribution)
        mask = binomial(density, size=((pure_children*nr_joints),(shape[1]-nr_joints))).astype(bool)
        G[:(pure_children*nr_joints), nr_joints:][mask] \
            = rand_generator(size=mask.sum(), distribution=distribution)
        rem_rows = shape[0] - (pure_children * nr_joints)
    else:
        rem_rows = shape[0]
        nr_joints = 0
    mask = binomial(density, size=(rem_rows,shape[1])).astype(bool)
    G[(pure_children*nr_joints):, :][mask] = rand_generator(size=mask.sum(), distribution=distribution)
    #G = pos_pure_children(G, nr_joints=nr_joints)
    np.random.shuffle(G)
    return G

def rand_model(m):
    # Check 
    check_model_consistency(m)
    nr_joints = len(m["joint_idx"])
    noise_vars = [m["joint_idx"] + dom_specifics for dom_specifics in m["domain_spec_idx"]]
    
    # Create graph among joint latent variables
    g = cd.rand.directed_erdos(nr_joints, density=m["graph_density"], random_order=False)
    g = cd.rand.rand_weights(g)
    A = np.transpose(g.to_amat())
    
    # Generate G, B and random samples for each domain
    data = []
    mixings = []
    for i in range(m["nr_doms"]):
        noise_idx = noise_vars[i]
        nr_rvs = len(noise_idx)
        dom_rvs = [m["noise_rvs"][j] for j in noise_idx]
        n = m["sample_sizes"][i]
        eps = np.zeros((n,len(dom_rvs)))
        for j, rv in enumerate(dom_rvs):
            eps[:,j] = sample_normed(n,rv)
        if m["two_pure_children"] and (i==0):
            if m["nr_doms"]==1:
                pure_children = 2
            else:
                pure_children = 1
        elif m["two_pure_children"] and (i==1):
            pure_children = 1
        else:
            pure_children = 0
        G = rand_G(shape=(m["dims"][i],nr_rvs), 
                    pure_children=pure_children, 
                    nr_joints=nr_joints, 
                    density=m["mixing_density"],
                    distribution=m["mixing_distribution"])
        I = np.identity(nr_rvs)
        A_dom = np.zeros((nr_rvs,nr_rvs))
        A_dom[0:nr_joints,0:nr_joints] = A
        B = np.matmul(G, np.linalg.inv(I-A_dom))
        data.append(np.transpose(np.matmul(B, np.transpose(eps))))
        mixings.append(B)

    # Joint mixing matrix B_large
    total_nr_obs = sum(m["dims"])
    total_nr_noise = nr_joints + sum([len(_) for _ in m["domain_spec_idx"]])
    B_large = np.zeros((total_nr_obs, total_nr_noise))

    # Joint columns
    for i in range(nr_joints):
        current_row = 0
        for dom in range(m["nr_doms"]):
            col = mixings[dom][:,i]
            B_large[np.ix_(np.arange(current_row,(current_row+len(col))),np.array([i]))] \
            = col.reshape((len(col),1))
            current_row = current_row + len(col)
        
    # Domain-specific columns
    current_col = nr_joints
    current_row = 0
    for dom in range(m["nr_doms"]):
        B = mixings[dom]
        nrows, ncols = B.shape
        domain_spec_cols = set(np.arange(ncols)) - set(np.arange(nr_joints))
        for i, col in enumerate(domain_spec_cols):
            B_large[np.ix_(np.arange(current_row,(current_row+nrows)),np.array([current_col+i]))] \
            = B[:,col].reshape((nrows,1))
        current_col = current_col + len(domain_spec_cols)
        current_row = current_row + nrows

    return (data, g, B_large)
