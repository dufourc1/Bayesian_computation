from math import log,exp,isnan,sqrt
import autograd.numpy as np

import time
import datetime

from src.helpers import update_progress
from src.maths import func_stats
from src.models import Model


def importance_sampling(model, proposal_cdf, proposal_simulate,samples_size):

    weights = []
    samples = []
    for i in range(samples_size):
        samples.append(proposal_simulate(model.size))
        weights.append(np.exp(model.log_posterior(samples[-1])-np.log(proposal_cdf(samples[-1]))))
    return [np.array(weights),np.array(samples)]



def rejection_sampling(model,proposal_cdf, proposal_simulate,samples_size, K):

    accepted = []
    while len(accepted) < samples_size:
        samples = proposal_simulate(model.size)
        weights = np.exp(model.log_posterior(samples)-np.log(proposal_cdf(samples)))/K
        threshold = np.random.randn(1)
        if threshold < weights:
            accepted.append(samples)

    return accepted
