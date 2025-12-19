import sys
import os
sys.path.append(os.path.abspath("../src"))

import numpy as np
from utils.subgroup import in_region


def sample_failure(X, beta):
    if len(X.shape) < 2:
        X = X.reshape(1, len(X))
    scale = np.exp(-X @ beta)
    return np.random.exponential(scale)


def synth_generic_hazard(n, B, hazard_fn, seed):
    rng = np.random.default_rng(seed=seed)
    
    d = B.shape[1]

    X = (B[1, :] - B[0, :]) * rng.random((n, d)) + B[0, :] # X is sampled uniformly from B
    scale = np.exp(-hazard_fn(X))
    T = rng.exponential(scale)

    Y = np.empty(n, dtype=np.dtype([('failure', '?'), ('time', '<f8')]))
    for i in range(n):
        Y[i] = (True, T[i])

    X, Y = map(np.array, zip(*sorted(zip(X, Y), key=lambda x: x[1]['time'])))
    
    return X, Y


def synth_nonlinear(n, B, R, beta_in, beta_out, seed=42, censor_param=0, censor_type='none'):
    """
    Input:
        B: Bounding box specified by a 2 x d matrix. B[0, i] = a_i, B[1, i] = b_i for bounding box \prod_{i=1}^d [a_i, b_i].
        R: Special region specified in the same way as B.

    Returns:
        X: n x d data matrix
        Y: np array of n tuples (1{uncensored}, time of event)
    """
    rng = np.random.default_rng(seed=seed)

    d = B.shape[1]

    X = (B[1, :] - B[0, :]) * rng.random((n, d)) + B[0, :] # X is sampled uniformly from B
    X_in = X[in_region(X, R)]
    X_out = X[~in_region(X, R)]

    X_out_true_feature = X_out.copy()
    X_out_true_feature[:, 0] = 10 * np.sin(100 * X_out_true_feature[:, 0] ** 2)
    # Outside of the correct region, the failure is from a Cox model on a nonlinear
    # transformation of the features.

    T_in = rng.exponential(np.exp(-X_in @ beta_in))
    T_out = rng.exponential(np.exp(-X_out_true_feature @ beta_out))

    T = np.concatenate([T_in, T_out])
    X = np.concatenate([X_in, X_out], axis=0)

    if censor_type == 'uniform':
        C = censor_param * rng.random(n)
    elif censor_type == 'indiv_exp':
        # censor_param = desired fraction of uncensored points
        C = rng.exponential((censor_param / (1 - censor_param)) * np.exp(-(X @ beta_in) * in_region(X, R) - (X @ beta_out) * ~in_region(X, R)))
    elif censor_type == 'indiv_unif':
        # censor_param = desired fraction of uncensored points
        C = rng.uniform(np.zeros(n), 2 * censor_param * np.exp(-(X @ beta_in) * in_region(X, R) - (X @ beta_out) * ~in_region(X, R)))
    elif censor_type == 'none':
        C = np.inf * np.ones(n)
    else:
        assert censor_type == 'exponential', f'censor_type must be "uniform", "exponential", "indiv_exp", "indiv_unif", or "none" (got {censor_type})'
        C = rng.exponential(censor_param, n)

    Y = np.empty(n, dtype=np.dtype([('failure', '?'), ('time', '<f8')]))
    for i in range(n):
        Y[i] = (C[i] > T[i], min(C[i], T[i]))

    X, Y = map(np.array, zip(*sorted(zip(X, Y), key=lambda x: x[1]['time'])))
    return X, Y