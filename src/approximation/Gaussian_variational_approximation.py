## REVIEW: warning unstable,svd does not converge in high dimensions
"""
unstable version of GVA
"""

import autograd.numpy as np
from math import log, pi
from src.maths import func_stats,mat
from src.helpers import update_progress

def GVA(model,max_steps=10,mu = None,L = None,nb_samples = 100):

    if mu is None:
        mu = np.zeros(model.size)
    if L is None:
        L = 2*np.eye(model.size)

    for i in range(max_steps):
        eta = np.random.randn(model.size, nb_samples)

        nlp_in = mu + np.transpose(mat.exp(L) @ eta)


        elbo_grad_L = 0
        elbo_grad_mu = 0
        for iter in range(nlp_in.shape[0]):
            elbo_grad_mu -= model.neg_log_posterior_grad(nlp_in[iter, :]) / nb_samples
            elbo_grad_L += mat.exp(2 * L) @ model.neg_log_posterior_hessian(nlp_in[iter, :])

        elbo_grad_L = - mat.sym(elbo_grad_L / nb_samples) + np.eye(len(mu))

        mu += 1e-3*elbo_grad_mu / iter
        L += 1e-3*elbo_grad_L / iter

        update_progress((i+1)/max_steps)

    # return values
    return mu, L
