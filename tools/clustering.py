import numpy as np
from .core import *

def _get_divergence_batch(Z: np.ndarray, alpha: float, P: np.ndarray, Q: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    P = np.maximum(P, eps)
    P = P / P.sum(axis=1, keepdims=True)

    Q = np.maximum(Q, eps)
    Q = Q / Q.sum(axis=1, keepdims=True)

    Z = np.asarray(Z, dtype=float)
    ZP = P @ Z.T    # (M, N) 
    ZQ = Q @ Z.T    # (k, N)

    if alpha == 2:
        pZp = np.sum(ZP * P, axis=1)   # (M,)
        qZq = np.sum(ZQ * Q, axis=1)   # (k,)
        cross = ZP @ Q.T               # (M, k) where entry = p^T Z q
        return pZp[:, None] + qZq[None, :] - 2.0 * cross

    ZP_a1 = ZP ** (alpha - 1.0)         # (M, N)
    ZQ_a1 = ZQ ** (alpha - 1.0)        # (k, N)
    ZQ_a2 = ZQ ** (alpha - 2.0)         # (k, N)

    term1_P = (P * ZP_a1).sum(axis=1)                           # (M,)
    term1_Q = (P[:, None, :] * ZQ_a1[None, :, :]).sum(axis=2)   # (M, k)
    first_term = (1.0 / (alpha-1.)) * (term1_P[:, None] - term1_Q)      # (M, k)

    part1 = (ZP[:, None, :] * (Q[None, :, :] * ZQ_a2[None, :, :])).sum(axis=2)  # (M, k)
    part2 = (Q * (ZQ_a2 * ZQ)).sum(axis=1)                                       # (k,)
    second_term = part1 - part2[None, :]                                         # (M, k)

    return first_term - second_term



class Clustering:
    def __init__(self, vectors:np.ndarray, weights:np.ndarray):
        self.vectors = vectors
        self.weights = weights
        self.global_mean = (vectors * weights[:, np.newaxis]).sum(axis=0)

    @classmethod
    def from_joint(cls, joint:np.ndarray):
        """
        Parameters:
            joint : np.ndarray of size (M, N)
                M is the number of vectors, N is the dimension. joint must sum to 1.
        """
        assert abs(joint.sum() - 1.) < 1e-12, "joint does not sum to 1."
                   
        weights = joint.sum(axis=1)       
        vectors = joint / weights[:, np.newaxis]
        return cls(vectors, weights)

    @classmethod
    def from_vectors(cls, vectors:np.ndarray, weights:np.ndarray|None=None):
        """
        Parameters:
            vectors : np.ndarray of size (M, N)
                M is the number of vectors, N is the dimension. Each row must sum to 1.
            weights : np.ndarray of size (M,) | None, default=None
                Optional, assumed to be the uniform distribution if None
        """
        assert (abs(vectors.sum(axis=1) - 1) < 1e-12).all(), "vectors do not sum to 1."

        if weights is None:
            weights = np.ones(vectors.shape[0]) / vectors.shape[0]
        
        assert weights.shape[0] == vectors.shape[0], "dimensions of weights and vectors do not match."
        assert abs(weights.sum() - 1.) < 1e-12, "weights do not sum to 1."

        return cls(vectors, weights)
    
    def test_centers(self, centers:np.ndarray):
        """
        Parameters:
            centers : np.ndarray of size (k, N)
        
        Returns:
            labels : np.array of length N
            between_cluster_information : float
        """
        labels = self._assign_labels(centers=centers)
        cluster_weights = np.zeros(centers.shape[0])
        for idx in range(centers.shape[0]):
            cluster_weights[idx] = np.where(labels==idx, self.weights, 0.).sum()

        between_cluster_information = 0.
        for (p, w) in zip(centers, cluster_weights):
            between_cluster_information += self.divergence(p, self.global_mean) * w

        return labels, between_cluster_information
        
    def fit(self, k:int, Z:np.ndarray, alpha:float, n_trials:int=50, rel_tol:float=1e-10, max_iter:int=100, random_seed:int|None=None, verbose:bool=True):
        """
        Parameters:
            k : int
                Number of clusters
            Z : np.ndarray of shape (N, N)
                Similarity matrix. Must be positive definite (not checked)
            alpha : float
                Order of divergence. Must be >= 2.
            n_trials : int, default=50
                Number of trials (initialisations).
            rel_tol : float, default=1e-10
                Relative change in loss must decrease this threshold at convergence.
            max_iter : int, default = 100
                Maximum number of iterations per trial.
            random_seed : int | None, default=None
                For reproducibility.
            verbose : bool, default=False
        """
        assert alpha >= 2, "Alpha must be >= 2"
        assert Z.shape[0] == self.vectors.shape[1]
        assert Z.shape[0] == Z.shape[1]

        self.verbose = verbose
        self.divergence = lambda p,q: get_divergence(Z=Z, alpha=alpha, p=p, q=q)
        self.divergence_batch = lambda P, Q: _get_divergence_batch(Z=Z, alpha=alpha, P=P, Q=Q)
        self.rng = np.random.default_rng(random_seed)

        self.total_information = (self.divergence_batch(self.vectors, self.global_mean[None, :]).flatten() * self.weights).sum()

        best_result = (None, None, -np.inf, None) # (Centers, labels, loss, reason)
        for _ in range(n_trials):
            if self.verbose:
                print(f"##### Trial {_+1}/{n_trials} #####")
            result = self._single_trial(k, rel_tol, max_iter)
            if result[2] > best_result[2]:
                best_result = result
            if self.verbose:
                print(f"Trial {_+1}/{n_trials} | {result[-1]} | Loss={np.round(result[2], 5)} | Best = {np.round(best_result[2], 5)}")
                print("###############")

        best_result = list(best_result)
        best_result[0], best_result[1] = self._relabel_by_size(labels=best_result[1], centers=best_result[0])
        return tuple(best_result)
    
    def _relabel_by_size(self, labels, centers):
        """
        Permute labels and change order of centers such that they are in descending order of size.
        Returns new_centers, new_labels
        """
        labels = np.asarray(labels)
        counts = np.bincount(labels)
    
        order = np.argsort(-counts, kind="stable")
        mapping = np.empty_like(order)
        mapping[order] = np.arange(len(order))

        new_labels = mapping[labels]
        new_centers = np.empty_like(centers)
        new_centers[mapping] = centers
        return new_centers, new_labels
   
    def _assign_labels(self, centers:np.ndarray):
        """
        Parameters:
            centers : np.ndarray of size (k, N)
        Returns:
            labels : np.array of length M
                vectors[i] is assigned to centers[labels[i]]
        """
        D = self.divergence_batch(self.vectors, centers)  # (M,k)
        return np.argmin(D, axis=1)

    
    def _single_trial(self, k, rel_tol, max_iter):
        """
        Returns:
            centers : np.ndarray of shape (k, N)
            labels : np.array of length N
            between_cluster_information : float
            reason : str
        """
        between_cluster_information = 1e-16

        # initialise centers by picking data points
        center_idxs = self.rng.choice(a=self.vectors.shape[0], size=k)
        init_centers = self.vectors[center_idxs, :].copy()
        # initalise labels
        old_labels = self._assign_labels(centers=init_centers)

        #### Clustering
        for _ in range(max_iter):
            new_centers, new_labels, new_loss = self._single_iter(k, old_labels)
            if (new_loss / between_cluster_information - 1.0) <= rel_tol: # convergence
                return new_centers, new_labels, new_loss, f'converged in {_+1} iters'

            old_labels = new_labels.copy()
            between_cluster_information = new_loss
        
        return new_centers, new_labels, new_loss, 'max_iter'
  
    def _single_iter(self, k, old_labels):
        """ 
        Returns:
            labels : np.array
            between_cluster_information : float
        """
        # Compute new centers
        centers = []
        list_cluster_weights = []

        for cluster_id in range(k):
            vector_idxs = np.where(old_labels==cluster_id)[0]
            cluster_vectors = self.vectors[vector_idxs, :].copy()
            cluster_weights = self.weights[vector_idxs].copy()

            weight_in_cluster = cluster_weights.sum()
            cluster_weights /= weight_in_cluster

            centers.append((cluster_vectors * cluster_weights[:, np.newaxis]).sum(axis=0))
            list_cluster_weights.append(weight_in_cluster)

        # compute -inertia
        between_cluster_information = 0.
        for (p, w) in zip(centers, list_cluster_weights):
            between_cluster_information += self.divergence(p, self.global_mean) * w

        # compute new labels
        new_labels = self._assign_labels(centers)
        return np.array(centers), np.array(new_labels), between_cluster_information

class Result:
    def __init__(self, vectors:np.ndarray, weights:np.array, Z:np.ndarray, alpha:float, centers:np.ndarray, labels, save_data:bool=True):
        self.centers = centers
        self.labels = labels

        self.vectors = None
        self.weights = None
        self.Z = None
        self.alpha = None
        if save_data:
            self.vectors = vectors
            self.weights = weights
            self.Z = Z
            self.alpha = alpha

        self.k = self.centers.shape[0]
        self.total_information = 0.
        self.between_cluster = 0.
        
        self.list_within_cluster_information = np.zeros(self.k)
        self.list_between_cluster_information = np.zeros(self.k)
        self.cluster_weights = np.zeros(self.k)

        self.list_within_cluster_contribution = []

        self._compute_informations(vectors, weights, Z, alpha)
    
    def _compute_informations(self, vectors, weights, Z, alpha):
        global_mean = (vectors * weights[:, np.newaxis]).sum(axis=0)
        divergence = lambda p,q: get_divergence(Z=Z, alpha=alpha, p=p, q=q)

        for p, w, cluster_idx in zip(vectors, weights, self.labels):
            self.total_information += w * divergence(p, global_mean)
            div_to_center = divergence(p, self.centers[cluster_idx])
            self.list_within_cluster_contribution.append(w * div_to_center)
            self.list_within_cluster_information[cluster_idx] += w * div_to_center
            self.cluster_weights[cluster_idx] += w

        for cluster_idx, q in enumerate(self.centers):
            div_to_global = self.cluster_weights[cluster_idx] * divergence(q, global_mean)
            self.list_between_cluster_information[cluster_idx] = div_to_global
        
        self.between_cluster = self.list_between_cluster_information.sum()
        self.list_within_cluster_contribution = np.array(self.list_within_cluster_contribution)

        self.between_cluster /= self.total_information
        self.list_between_cluster_information /= self.total_information
        self.list_within_cluster_contribution /= self.total_information
        self.list_within_cluster_information /= self.total_information

    def get_stats(self):
        return {
            cluster_idx: [(self.labels==cluster_idx).sum(), self.cluster_weights[cluster_idx], self.list_within_cluster_information[cluster_idx], self.list_between_cluster_information[cluster_idx]] for cluster_idx in range(self.k)
        }

    def describe(self,):
        print("Results of clustering\n-----------------")
        print(f"{len(self.list_within_cluster_contribution)} distributions of length {self.centers.shape[1]} into {self.k} clusters.")
        print(f"Fraction of information captured: {np.round(self.between_cluster, 3)}")
        for cluster_idx in range(self.k):
            print("-------------")
            print(f"Cluster {cluster_idx}:")
            print(f"{(self.labels==cluster_idx).sum()} points with {np.round(self.cluster_weights[cluster_idx], 3)} weight.")
            print(f"Within-cluster information={np.round(self.list_within_cluster_information[cluster_idx], 3)}, Between-cluster contribution={np.round(self.list_between_cluster_information[cluster_idx], 3)}")  