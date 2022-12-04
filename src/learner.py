# === IMPORTS: THIRD-PARTY ===
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import wasserstein_distance

# === IMPORTS: LOCAL ===
from src.dist import third_moments_distance
from src.matching import minimum_matching

class LinearMDCRL:

    def __init__(self, metric="1-wasserstein", matching="minimum"):

        if metric=="1-wasserstein":
            self.metric = wasserstein_distance
        elif metric=="third-moments":
            self.metric = third_moments_distance
        else:
            raise NotImplementedError("Metric not implemented")

        if matching=="minimum":
            self.matching = minimum_matching
        else:
            raise NotImplementedError("Matching not implemented")

    def fit(self, data):
         # Each element in data is a matrix of shape n_e x d_e
        self.data = data
        self.get_sources()
        self.match()
        self.get_joint_mixing()
        #self.get_joint_graph()

    def get_sources(self):
        self.nr_env = len(self.data)
        self.indep_comps = []
        self.mixings = []
        for X in self.data:
            cov = np.cov(np.transpose(X))
            rk = np.linalg.matrix_rank(cov)
            ICA = FastICA(n_components=rk, whiten='unit-variance', max_iter=1000) 
            ICA.fit(X)
            eps = ICA.transform(X)
            scaling = eps.std(axis=0)
            eps = self.rescale_columns(eps)
            mixing = np.matmul(ICA.mixing_, np.diag(scaling))    
            self.indep_comps.append(eps)
            self.mixings.append(mixing)


    def match(self):
        matchings = {}
        for i in range(self.nr_env):
            for j in range(i+1, self.nr_env):
                D = self.signed_distance_matrix(self.indep_comps[i],self.indep_comps[j])
                matchings[str(i)+str(j)] = self.matching(D)
        
        # Define potential factors
        pot_factors = [[i] for i in list(matchings['01'].keys())] 
        for f in pot_factors:
            for i in range(1,self.nr_env):
                if f[0] not in list(matchings['0'+str(i)].keys()):
                    continue
                else:
                    f.append(matchings['0'+str(i)][f[0]])
        pot_factors = [f for f in pot_factors if len(f)==self.nr_env]

        # Only keep consistent factors
        self.joint_factors = [f for f in pot_factors if self.is_consistent(f, matchings)]
        self.joint_signs = [self.add_signs_to_factor(f) for f in self.joint_factors]


    def get_joint_mixing(self):
        self.nr_joint = len(self.joint_factors)
        self.total_nr_obs = sum([M.shape[0] for M in self.mixings])
        self.total_nr_lat = sum([M.shape[1]-self.nr_joint for M in self.mixings]) + self.nr_joint
        M_large = np.zeros((self.total_nr_obs, self.total_nr_lat))
        
        # Joint columns
        for i,f in enumerate(self.joint_factors):
            current_row = 0
            for env in range(self.nr_env):
                col = self.joint_signs[i][env] * self.mixings[env][:,f[env]]
                M_large[np.ix_(np.arange(current_row,(current_row+len(col))),np.array([i]))] \
                = col.reshape((len(col),1))
                current_row = current_row + len(col)
        
        # Domain-specific columns
        current_col = len(self.joint_factors)
        current_row = 0
        for env in range(self.nr_env):
            M = self.mixings[env]
            nrows, ncols = M.shape
            joint_cols = [f[env] for f in self.joint_factors]
            domain_spec_cols = set(np.arange(ncols)) - set(joint_cols)
            for i, col in enumerate(domain_spec_cols):
                M_large[np.ix_(np.arange(current_row,(current_row+nrows)),np.array([current_col+i]))] \
                = M[:,col].reshape((nrows,1))
            current_col = current_col + len(domain_spec_cols)
            current_row = current_row + nrows
        self.joint_mixing = M_large

    def get_joint_graph(self):
        pass

    ########################
    ### helper functions ###
    ########################
    def signed_distance_matrix(self, X1, X2):
        p1 = X1.shape[1]
        p2 = X2.shape[1]
        D_large = np.zeros((p1,2*p2))
        X2_large = np.concatenate((X2,-X2),axis=1)
        for i in range(p1):
            for j in range(2*p2):
                D_large[i,j] = self.metric(X1[:,i],X2_large[:,j]) 
        D = np.zeros((p1,p2))
        for i in range(p2):
            D[:,i] = D_large[:,[i,i+p2]].min(axis=1)
        return D

    def matching_sign(self, distr1, distr2):
        normal_dist = self.metric(distr1,distr2) 
        flipped_dist = self.metric(distr1,-distr2)
        if normal_dist <= flipped_dist:
            return 1
        else:
            return -1 
    
    def add_signs_to_factor(self, f):
        signs = [1]
        for i in range(1,len(f)):
            signs.append(self.matching_sign(self.indep_comps[0][:,f[0]], self.indep_comps[i][:,f[i]]))  
        return signs
    
    @staticmethod
    def is_consistent(f, matchings):
        for i in range(len(f)):
            for j in range(i+1, len(f)):
                if f[i] not in list(matchings[str(i)+str(j)].keys()):
                    return False
                if matchings[str(i)+str(j)][f[i]] != f[j]:
                    return False
        return True


    @staticmethod
    def rescale_columns(eps):
        for i in range(eps.shape[1]):
            eps[:,i] = (eps[:,i] - eps[:,i].mean()) / eps[:,i].std()
        return eps
