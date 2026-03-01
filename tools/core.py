import numpy as np

def distance_to_similarity(distance_matrix, tau=1.):
    """
    Parameters:
        distance_matrix : np.ndarray
            Must be of negative type to guarantee PD
        tau : float, default=1.0
            Scaling factor
    Returns:
        similarity_matrix
    """
    return np.exp(-1. * tau * distance_matrix)



def get_entropy(Z:np.ndarray, alpha:float, p:np.ndarray):
    """
    Parameters:
        Z : np.ndarray of shape (n, n)
            Similarity matrix; we do not check this.
        alpha : float
            alpha > 0
        p : np.ndarray of length (n,) or (m, n)
            Distribution(s) or number(s) of observations
    
    Returns:
        entropy : scalar if p was 1D else np.ndarray of shape (m,)
    """
    assert alpha > 0, "Error: alpha must be positive"

    eps = 1e-16
    p = np.maximum(p, eps)
    p /= p.sum(axis=-1, keepdims=True)

    Zp = p @ Z

    if alpha == 1.:
        return -1 * (p * np.log(Zp)).sum(axis=-1)

    return (1. / (alpha-1.)) * (1. - np.sum((Zp**(alpha-1))*p, axis=-1))



def get_divergence(Z:np.ndarray, alpha:float, p:np.ndarray, q:np.ndarray):
    """
    Parameters:
        Z : np.ndarray of shape (n, n)
            Similarity matrix
        alpha : float
            alpha >= 2
        p : np.ndarray of length (n,) or (m, n)
            distribution(s)
        q : np.ndarray of length (n,) or (m, n)
            reference distribution(s)
    
    Returns:
        divs : float or np.ndarray of shape (m,)
            float if both p and q are 1D, np.ndarray otherwise
    
    If both p and q are 2D, their first dimension must match and the divergences are of p[i] from q[i]
    If one is 1D, the divergences are using each distribution in the other.
    """
    assert alpha >= 2.0

    eps = 1e-16
    p = np.maximum(p, eps)
    p /= p.sum(axis=-1, keepdims=True)

    q = np.maximum(q, eps)
    q /= q.sum(axis=-1, keepdims=True)

    Zp = p @ Z
    Zq = q @ Z

    first_term = (1.0/(alpha-1.0)) * np.sum(p * (Zp ** (alpha-1.0) - Zq ** (alpha-1.0)), axis=-1)
    second_term = np.sum(q * (Zq ** (alpha - 2.0)) * (Zp - Zq), axis=-1)
    
    return first_term - second_term


def get_bregman_information(Z:np.ndarray, alpha:float, joint:np.ndarray, clustering=None):
    """    
    Parameters:
        Z : np.ndarray
            Similarity matrix of shape (n,n)
        alpha : float
            Order
        joint : np.ndarray of shape (m, n)
            Rows are observations, columns are probability space. Must sum to <= 1.
        clustering : dict | None, default=None
            {cluster_id: [row indices]}. If None, return bregman information of whole joint.
    
    Returns:
        information : float | dict
            Either total bregman informaiton or {cluster_id: information}
    """
    joint = joint.copy()
    joint /= joint.sum()
    
    if clustering is not None:
        information = {}
        for c_id, idxs in clustering.items():
            information[c_id] = get_bregman_information(Z, alpha, joint[idxs, :])
        return information

    weights = joint.sum(axis=1)
    
    mean = joint.sum(axis=0)
    mean /= mean.sum()

    joint /= weights[:, np.newaxis]

    return get_entropy(Z, alpha, mean) - (get_entropy(Z, alpha, joint) * weights).sum()


## Functions specifically for alpha = 2. Use for faster computation.
def get_rao(Z:np.ndarray, p:np.ndarray):
    """
    Parameters:
        Z : np.ndarray of shape (n, n)
            Similarity matrix; we do not check this.
        p : np.ndarray of length (n,) or (m, n)
            Distribution(s) or number(s) of observations
    
    Returns:
        rao_entropy : scalar if p was 1D else np.ndarray of shape (m,)
    """

    eps = 1e-16
    p = np.maximum(p, eps)
    p /= p.sum(axis=-1, keepdims=True)
    Zp = p @ Z
    return 1.0 - np.sum(p * Zp, axis=-1)

def get_sqmahalanobis(Z:np.ndarray, p:np.ndarray, q:np.ndarray):
    """
    Parameters:
        Z : np.ndarray of shape (n, n)
            Similarity matrix
        p : np.ndarray of length (n,) or (m, n)
            distribution(s)
        q : np.ndarray of length (n,) or (m, n)
            reference distribution(s)
    
    Returns:
        divs : float or np.ndarray of shape (m,)
            float if both p and q are 1D, np.ndarray otherwise
    
    If both p and q are 2D, their first dimension must match and the divergences are of p[i] from q[i]
    If one is 1D, the divergences are using each distribution in the other.
    """

    eps = 1e-16
    p = np.maximum(p, eps)
    p /= p.sum(axis=-1, keepdims=True)

    q = np.maximum(q, eps)
    q /= q.sum(axis=-1, keepdims=True)

    pmq = p - q
    Zpmq = pmq @ Z

    return (pmq * Zpmq).sum(axis=-1)