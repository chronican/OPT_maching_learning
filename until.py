import numpy as np
import torch
#code sourse(euclidean_proj_simplex):https://github.com/icdm-extract/extract/blob/18d6e8509f2f35719535e1de6c88874ec533cfb9/python/algo/l1project.py
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

#code sourse(euclidean_proj_l1ball):https://github.com/icdm-extract/extract/blob/18d6e8509f2f35719535e1de6c88874ec533cfb9/python/algo/l1project.py
def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -------
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    #print(u.sum())
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w




def euclidean_proj_simplex_l2ball(v, s=1):
    """
        compute the euclidean projection in L2-ball
        :param v: vector which needed to be projected
        :param s:radius of a unit ball
        :return w:vector after projection into L2-ball
    """
    n, = v.shape  # will raise ValueError if v is not 1-D
    u=np.power(v,2)
    if np.sqrt(u.sum()) == s and np.alltrue(u >= 0):# check if v is l2-ball itself
        return v
    w=np.min([1/np.linalg.norm(v),1])*v
    return w

def euclidean_proj_l2ball(v, s=1):
    """
        check date if in the l2-ball and return the euclidean projection in L2-ball
        :param v: vector which needed to be projected
        :param s:radius of a unit ball
        :return w:vector after projection into L2-ball
    """

    n, = v.shape
    u = v**2
    if np.sqrt(u.sum())  <= s:#check if is in the l2-norm ball
        return v
    w = euclidean_proj_simplex_l2ball(v, s=s)
    return w
