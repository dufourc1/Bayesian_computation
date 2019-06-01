'''
Implementation of the approximation functions seen in class of Bayyesian computation:
 -GVA
'''

import autograd.numpy as np
from math import log, pi


def sym(M):
    return 0.5 * (M + M.T)


def exp_matrix(M):
    # return the exponential of the matrix M as defined in the lecture notes
    u, s, vh = np.linalg.svd(M)
    return u @ np.diag(0.5 * np.log(s)) @ vh


def reparametrize(mu, L, etas):
    inter = mu + exp_matrix(L) @ etas.T
    # inter = inter.reshape(inter.shape[1],inter.shape[2])
    return inter


def GVA(phi, grad_phi, H_phi, mu, L, max_iters=20, sample_size=20, sampling_generator=None, lr=0.0001):
    if sampling_generator is None:
        def sampling_generator():
            a = np.array([np.random.normal(loc=0.0, scale=1.0, size=mu.shape[0] - 1) for i in range(sample_size)])
            b = np.array([np.random.uniform(low=1e-8, high=5, size=sample_size)]).reshape(sample_size, 1)
            return np.concatenate((b, a), axis=1)

    d = len(mu)

    trace = []
    trace_mu = []
    trace_l = []

    mu = mu.reshape(len(mu), 1)
    for iter in range(max_iters):
        # generate the sample of unit gaussian
        lr = 1 / (iter + 10) ** 2
        etas = sampling_generator()
        inter = reparametrize(mu, L, etas)
        ELBO = 0

        # the computation below approximate the expectation of the radom quantities
        # involved in the ELB0

        # compute the ELBO
        for elt in inter.T:
            ELBO += phi(elt)
        ELBO /= sample_size
        ELBO += d * log(pi * 2 * np.e) / 2. + np.trace(L)
        trace.append(ELBO)

        # compute the gradient wrt to mu
        grad_mu = np.zeros(len(mu))
        for elt in inter.T:
            grad_mu -= grad_phi(elt)
        grad_mu = grad_mu / sample_size

        # compute the gradient wrt to L
        grad_l = np.zeros_like(L)
        for elt in inter.T:
            grad_l += exp_matrix(L) @ H_phi(elt)
        grad_l /= sample_size
        grad_l = -sym(grad_l)
        grad_l += np.eye(len(grad_l))

        # updating L and mu
        mu = mu - lr * grad_mu.reshape(len(grad_mu), 1)
        L = L - lr * grad_l
        trace_mu.append(mu)
        trace_l.append(L)

    return mu, L, trace, trace_mu, trace_l
