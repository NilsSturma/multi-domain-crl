# === IMPORTS: BUILT-IN ===
from itertools import combinations, product

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import wasserstein_distance, ks_2samp

# === IMPORTS: LOCAL ===
from src.scoring import score_up_to_signed_perm, get_permutation_matrix, permutations_respecting_graph



class LinearMDCRL:

    def __init__(self, measure="ks-test", alpha=0.05, gamma=0.1):
        self.measure = measure
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, data):
         # Each element in data is a matrix of shape n_e x d_e
        self.data = data
        self.get_sources()
        self.match()
        self.get_joint_mixing()
        if len(self.joint_factors) > 0:
            self.get_joint_graph()

    def get_sources(self):
        self.nr_env = len(self.data)
        self.indep_comps = []
        self.mixings = []
        for X in self.data:
            cov = np.cov(np.transpose(X))
            rk = np.linalg.matrix_rank(cov)
            ICA = FastICA(n_components=rk, whiten='unit-variance', max_iter=10000, tol=1e-3) 
            ICA.fit(X)
            eps = ICA.transform(X)
            scaling = eps.std(axis=0)
            eps = self.rescale_columns(eps)
            mixing = np.matmul(ICA.mixing_, np.diag(scaling))    
            self.indep_comps.append(eps)
            self.mixings.append(mixing)

    def match(self):
        
        # Adjust alpha
        total_ntests = 0
        for i in range(self.nr_env):
            for j in range(i+1, self.nr_env):
                total_ntests = total_ntests + 2 * self.indep_comps[i].shape[1] * self.indep_comps[j].shape[1]
        self.adj_alpha = self.alpha / total_ntests
        
        # Pairwise matching
        matchings = {}
        for i in range(self.nr_env):
            for j in range(i+1, self.nr_env):
                S = self.signed_similarity_matrix(self.indep_comps[i],self.indep_comps[j])
                matchings[str(i)+str(j)] = self.maximum_matching(S, alpha=self.adj_alpha)
        
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
        self.dom_spec_nr_lat = []
        for env in range(self.nr_env):
            M = self.mixings[env]
            nrows, ncols = M.shape
            joint_cols = [f[env] for f in self.joint_factors]
            domain_spec_cols = set(np.arange(ncols)) - set(joint_cols)
            self.dom_spec_nr_lat.append(len(domain_spec_cols))
            for i, col in enumerate(domain_spec_cols):
                M_large[np.ix_(np.arange(current_row,(current_row+nrows)),np.array([current_col+i]))] \
                = M[:,col].reshape((nrows,1))
            current_col = current_col + len(domain_spec_cols)
            current_row = current_row + nrows
        self.joint_mixing = M_large

    def get_joint_graph(self):

        B = self.joint_mixing[:,:self.nr_joint]

        # Remove zero rows
        B = self.remove_zero_rows(B)

        # Find one pure child per latent node
        B_star = self.get_pure_children(B)

        # Remove permutation indeterminacy
        order_rows, order_cols = self.search_order_noisy(B_star)
        if order_rows is not None:
            B_star = B_star[order_rows, :][:, order_cols]
        # Remove sign indeterminacy from columns
        B_star = np.matmul(B_star, np.diag(np.sign(np.diag(B_star))))

        # Remove scaling indeterminacy from rows  and solve for A
        D = np.diag(B_star)
        if 0 in D:
            D = D + 1e-20
        B_star = np.matmul(np.diag(1/D), B_star)

        # Solve for A
        if np.linalg.matrix_rank(B_star) < self.nr_joint:
            B_star = B_star + np.diag(np.full(self.nr_joint, 1e-20))
        self.A = (np.eye(B_star.shape[0]) - np.linalg.inv(B_star))

    # function as in paper
    def score_shared_columns(self, B_true, true_nr_joint):
        min_error = float('inf')
        if self.nr_joint == true_nr_joint:
            min_error = score_up_to_signed_perm(self.joint_mixing[:,:self.nr_joint], B_true[:,:self.nr_joint])[0]
        elif self.nr_joint < true_nr_joint:
            for comb in combinations(range(true_nr_joint), self.nr_joint):
                res = score_up_to_signed_perm(self.joint_mixing[:,:self.nr_joint], B_true[:,comb])
                if res[0] < min_error:
                    min_error = res[0]
        else:
            for comb in combinations(range(self.nr_joint), true_nr_joint):
                res = score_up_to_signed_perm(self.joint_mixing[:,comb], B_true[:,:true_nr_joint])
                if res[0] < min_error:
                    min_error = res[0]
        return min_error

    # Also scores domain-specific columns but requires \hat(l)=l.
    def score_joint_mixing_complete(self, B_true):
        B_perm = self.joint_mixing.copy()

        # Score joint mixing
        res = score_up_to_signed_perm(self.joint_mixing[:,:self.nr_joint], B_true[:,:self.nr_joint])
        B_perm[:,:self.nr_joint] = res[1]

        # Score domain-specific ones
        current_col = self.nr_joint
        for i in range(self.nr_env):
            nlatents = self.dom_spec_nr_lat[i]
            res = score_up_to_signed_perm(self.joint_mixing[:,current_col:(current_col+nlatents)], 
                                        B_true[:,current_col:(current_col+nlatents)])
            B_perm[:,current_col:(current_col+nlatents)] = res[1]
            current_col = current_col+nlatents
        final_score = np.linalg.norm(B_perm - B_true)

        return (final_score, B_perm)

    # WARNING: this function only works if \hat(l)=l
    def score_graph_param_matrix(self, A_true):
        min_error = float('inf')
        dim = A_true.shape[0]
        for perm in permutations_respecting_graph(A_true):
            P = get_permutation_matrix(perm)
            A_hat_perm = P.T @ self.A @ P
            abs_error = np.linalg.norm(abs(A_hat_perm) - abs(A_true))
            if abs_error < min_error:
                min_error = abs_error
                best_A_hat_perm = A_hat_perm
        min_error = float('inf')
        for binary_seq in product([0,1], repeat=dim):
            binary_seq = np.array(binary_seq).astype(bool)
            D = np.ones(dim)
            D[binary_seq] = -1
            D = np.diag(D)
            A_hat_scaled = D @ best_A_hat_perm @ D
            error = np.linalg.norm(A_hat_scaled - A_true)
            if error < min_error:
                min_error = error
                best_solution = (min_error, A_hat_scaled)
        return best_solution

    #########################################
    ### helper functions for joint mixing ###
    #########################################

    def similarity_measure(self, a, b):
        if self.measure=="1-wasserstein":
            return 1 - wasserstein_distance(a,b)
        elif self.measure=="ks-test":
            return ks_2samp(a, b).pvalue
        else:
            raise NotImplementedError("Metric not implemented")
        
    def signed_similarity_matrix(self, X1, X2):
        p1 = X1.shape[1]
        p2 = X2.shape[1]
        S_large = np.zeros((p1,2*p2))
        X2_large = np.concatenate((X2,-X2),axis=1)
        for i in range(p1):
            for j in range(2*p2):
                S_large[i,j] = self.similarity_measure(X1[:,i],X2_large[:,j]) 
        S = np.zeros((p1,p2))
        for i in range(p2):
            S[:,i] = S_large[:,[i,i+p2]].max(axis=1)
        return S

    @staticmethod
    def maximum_matching(similarity_matrix, alpha):
        p = similarity_matrix.shape[0]
        matching = {}
        for i in range(p):
            row_argmax = similarity_matrix[i,:].argmax()
            row_max = similarity_matrix[i,:].max()
            if (similarity_matrix[:,row_argmax].argmax() == i) and (row_max >= alpha):
                matching[i] = row_argmax
        return matching

    def matching_sign(self, distr1, distr2):
        normal_sim = self.similarity_measure(distr1,distr2) 
        flipped_sim = self.similarity_measure(distr1,-distr2)
        if normal_sim >= flipped_sim:
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

    ###########################################
    ### helper functions for graph recovery ###
    ###########################################

    @staticmethod
    def score_rows(matrix):
        d = matrix.shape[0]
        R = np.full(shape=(d,d),fill_value=.0)
        for i in range(d):
            for j in range(i+1,d):
                u, s, v = np.linalg.svd(matrix[[i,j],:])
                #s2 = np.sort(s) ** 2
                #R[i,j] = s2[-1] / s2.sum()
                R[i,j] = 1/(min(s)+0000000.1)
        return R

    @staticmethod
    def get_duplicates(cand_ids, ord_tup):
        l = len(cand_ids)
        ids0 = ord_tup[0][cand_ids]
        ids1 = ord_tup[1][cand_ids]
        uniques = np.unique(np.concatenate((ids0, ids1), axis=0))
        if len(uniques)==(2*l):
            return None
        else:
            for i in range(l):
                for j in range(i+1,l):
                    if (ids0[i]==ids0[j]) or (ids0[i]==ids1[j]):
                        return cand_ids[j]

    def get_low_rank(self, cand_ids, ord_tup, R):
        l = len(cand_ids)
        ids0 = ord_tup[0][cand_ids]
        ids1 = ord_tup[1][cand_ids]
        for i in range(l):
            for j in range(i+1,l):
                if (R[ids0[i],ids0[j]] > (1/self.gamma)) or (R[ids0[i],ids1[j]] > (1/self.gamma)):
                    return cand_ids[i], cand_ids[j]
        return None 

    @staticmethod
    def update_cand_ids(cand_ids, to_remove):
        current_max = max(cand_ids)
        cand_ids.remove(to_remove)
        cand_ids.append(current_max+1)
        return(cand_ids)

    def get_pure_children(self, B):

        # Score all pairs of rows
        R = self.score_rows(B)
        d = R.shape[0]
        nr_tuples = int(d * (d-1) / 2)
        
        ord_tup = np.unravel_index(R.ravel().argsort(), R.shape)
        ord_tup = (np.flip(ord_tup[0][nr_tuples:]), 
                np.flip(ord_tup[1][nr_tuples:]))
        
        # Make R symmetric now (important)
        for i in range(d):
            for j in range(i+1,d):
                R[j,i]=R[i,j]
                
        # Choose rows that maximize thw "within-tuple" score and at the same time have a low "inter-tuple" score 
        cand_ids = list(np.arange(self.nr_joint))
        while max(cand_ids) < nr_tuples:
            to_remove = self.get_duplicates(cand_ids, ord_tup)
            if to_remove is not None:
                cand_ids = self.update_cand_ids(cand_ids, to_remove)
                continue
            to_remove = self.get_low_rank(cand_ids, ord_tup, R)
            if to_remove is not None:
                temp_cands = cand_ids.copy()
                temp_cands = self.update_cand_ids(temp_cands, to_remove[0])
                if self.get_low_rank(temp_cands, ord_tup, R) is None:
                    cand_ids = self.update_cand_ids(cand_ids, to_remove[0])
                else:
                    cand_ids = self.update_cand_ids(cand_ids, to_remove[1])
                continue
            break   
        pure_children_rows = ord_tup[0][cand_ids]
        pure_children_rows
        
        return B[pure_children_rows,:]  

    @staticmethod
    def l2_without_max(x):
        m = np.abs(x).argmax()
        mask = np.ones(len(x), dtype=bool)
        mask[m] = False
        return np.linalg.norm(x[mask])

    def search_order_noisy(self, matrix):
        d = matrix.shape[0]
        order_rows = []
        order_cols = []
        original_idx_rows = np.arange(d)  
        original_idx_cols = np.arange(d)  
        while 0 < matrix.shape[0]:
            # Find row with lowest l2 norm where all entries but the maximum are considered
            target_row = np.apply_along_axis(self.l2_without_max, 1, matrix).argmin()
            target_col = abs(matrix[target_row,:]).argmax()
            # Append index to order
            order_rows.append(original_idx_rows[target_row])
            order_cols.append(original_idx_cols[target_col])
            original_idx_rows = np.delete(original_idx_rows, target_row)
            original_idx_cols = np.delete(original_idx_cols, target_col)
            # Remove the row and the column from B
            row_mask = np.delete(np.arange(matrix.shape[0]), target_row)
            col_mask = np.delete(np.arange(matrix.shape[1]), target_col)
            matrix = matrix[row_mask,:][:, col_mask]
        if len(order_rows) != d:
                return None, None
        return order_rows, order_cols

    @staticmethod
    def remove_zero_rows(M):
        nrow = M.shape[0]
        to_delete = []
        for i in range(nrow):
            if np.linalg.norm(M[i,:]) < 0.1:
                to_delete.append(i)
        M = np.delete(M, to_delete, axis=0)
        return M
