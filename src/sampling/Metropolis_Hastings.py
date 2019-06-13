from math import log,exp,isnan,sqrt
import autograd.numpy as np

import time
import datetime

from src.helpers import update_progress
from src.maths import func_stats


## REVIEW: clean and rewrite the code if time, otherwise seems to work

def random_walk_MH(model, verbose = False, verbose_gen = True, RETURN = False,**kwargs):
    """Metropolis Hastings sampling algortihm.

    Parameters
    ----------
    step_size : float
        step_size of the Markov Chain (the default is 0.05).
    max_iter : type
        maximum of iterations of the algortihm (the default is 100).
    verbose: bool
        to print the acceptance rate
    **kwargs : type
        size: float
            size of the beta parameter of the log_posterior_function
        initial: ndarray
            starting point of the algorithm (optional)
        acc: bool
            to return the Acceptance rate of the Markov Chain


    Returns
    -------
    nd_array
        simulated samples from the post of model of size max_iter

    Examples
    -------
    >>> gaussian_model = Model(Gaussian_prior,Gaussian, ... )
    >>> samples = MH_sampling(gaussian_model, max_iter = 300 ,
                                step_size = 0.06, size = 17)

    """


    #get the size of the parameter to simulate
    size = model.size

    #initial point
    if "initial" in kwargs.keys():
        current = np.array(kwargs["initial"])
    else:
        current = np.ones(size) #np.random.randn(size)

    #step size
    if 'step_size' in kwargs.keys():
        step_size = kwargs["step_size"]
    else:
        step_size = 0.04
        print("default step size selected : {}".format(step_size))

    #number of iterations
    if 'max_iter' in kwargs.keys():
        max_iter = kwargs["max_iter"]
    else:
        max_iter = 10


    #create empty containers
    samples = np.zeros([max_iter, size])
    record_acceptance = np.zeros(max_iter)

    #performance measures
    start = time.time()

    #actual MCMC simulation
    for k in range(max_iter):

        #update the current sample
        proposal = current + step_size*np.random.randn(size)

        #compute its acceptance ratio
        ratio = np.exp(model.log_posterior(proposal) \
                        - model.log_posterior(current))

        #check if accepted
        threshold = np.random.random()
        if ratio > threshold:
            current = proposal

        #update samples an acceptance accordingly
        samples[k,:] = current
        record_acceptance[k] = (ratio > threshold)

        if verbose_gen == True:
            if k%5 == 0 or k == max_iter-1:
                update_progress((k+1)/max_iter)

    # saving the data
    #defining the burnin parameter
    if "burning" in kwargs.keys():
        burning = kwargs["burning"]
    else:
        burning = int(max_iter/10)

    #saving the estimates in the model
    ## NOTE: could be generalized if time
    covariance = np.cov(samples[burning:].T)
    model.results["MH_vanilla"] = [np.mean(samples[burning:],axis = 0),covariance]

    end = time.time()
    if verbose:
        print(" Acceptance rate : {:2.1%}  (advised values between 10% and 50%)"\
                    .format(np.mean(record_acceptance)))
        print("  duration: {}".format(str(datetime.timedelta(
                                                    seconds= round(end-start)))))

    model.time["MH_vanilla"] = [end-start,max_iter]
    if RETURN:
        if "acc" in kwargs.keys():
            if kwargs["acc"] == True:
                return samples, np.mean(record_acceptance)
        return samples

def Langevin_MH(model, tau,verbose = True, verbose_gen = True, RETURN = False,**kwargs):

    size = model.size

    if "initial" in kwargs.keys():
        current = np.array(kwargs["initial"])
    else:
        current = np.ones(size)

    if 'max_iter' in kwargs.keys():
        max_iter = kwargs["max_iter"]
    else:
        max_iter = 10
    if "step_size" in kwargs.keys():
        step_size = kwargs["step_size"]
    else:
        step_size = 1



    samples = np.zeros([max_iter, size])
    record_acceptance = np.zeros(max_iter)

    start = time.time()

    def log_qprop(X,Y,tau,grad ):
        # probability to go to X given Y
        R = -1/(4*tau)*np.linalg.norm(X-Y-tau*grad(Y))**2
        return R


    for k in range(max_iter):
        #compute the proposal based on the Langevin update
        proposal = current + step_size*(tau*model.log_posterior_grad(current)+\
                     sqrt(2*tau)*np.random.randn(size))

        #dirty trick to avoid proposition for negative variance
        # REVIEW: code it in distribution, but beware it could break down other computation
        if proposal[0] > 0 or model.name\
                    == "Conditional model : Multilogistic,  Prior : gaussian":
            #comppute acceptance ratio
            ratio = model.log_posterior(proposal) -\
                    model.log_posterior(current)
            ratio = ratio - log_qprop(current,proposal,
                                    tau,model.log_posterior_grad)
            ratio += log_qprop(proposal,current,tau,model.log_posterior_grad)
            ratio = np.exp(ratio)
            threshold = np.random.random()
        else:
            #otherwise since the proposed value is non valid
            #we won't accept it
            ratio = 0
        if ratio > threshold:
            current = proposal

        #record the new samples
        samples[k,:] = current
        record_acceptance[k] = (ratio > threshold)

        if verbose_gen == True:
            if k%5 == 0 or k == max_iter-1:
                update_progress((k+1)/max_iter)
    if "burning" in kwargs.keys():
        burning = kwargs["burning"]
    else:
        burning = int(max_iter/10)
    covariance = np.cov(samples[burning:].T)
    model.results["MH_Langevin"] = [np.mean(samples[burning:],axis = 0),covariance]

    if verbose:
        print(" Acceptance rate : {:2.1%} \
                (advised values between 10% and 50%)"\
                    .format(np.mean(record_acceptance)))

    end = time.time()
    if verbose:
        print("  duration: {}".format(str(datetime.timedelta(
                                                    seconds= round(end-start)))))

    model.time["MH_langevin"] = [end-start,max_iter]
    if RETURN:
        if "acc" in kwargs.keys():
            if kwargs["acc"] == True:
                return samples, np.mean(record_acceptance)
        return samples


def MH_whithin_Gibbs(model, verbose = False, verbose_gen = True, RETURN = False,**kwargs):
    ''' Metropolis within Gibbs, update a batch of parameters at the same time,
    not necessarly the all vector
    use the same input as MH_vanilla
    '''

    #get the size of the parameter to simulate
    size = model.size

    #initial point
    if "initial" in kwargs.keys():
        current = np.array(kwargs["initial"])
    else:
        current = np.ones(size) #np.random.randn(size)

    #step size
    if 'step_size' in kwargs.keys():
        step_size = kwargs["step_size"]
    else:
        step_size = 0.04
        print("default step size selected : {}".format(step_size))

    #number of iterations
    if 'max_iter' in kwargs.keys():
        max_iter = kwargs["max_iter"]
    else:
        max_iter = 10

    if 'batch' in kwargs.keys():
        batch = kwargs["batch"]
    else:
        batch = 5
        print("default batch is {}".format(batch))


    #create empty containers
    samples = np.zeros([max_iter, size])
    record_acceptance = np.zeros(max_iter)

    #performance measures
    start = time.time()

    #actual MCMC simulation
    for k in range(max_iter):

        #choose an index to update
        indices = np.random.randint(len(current),size = batch)
        #update the sample accordingly
        step = np.zeros_like(current)
        for index in indices:
            step[index] = np.random.randn()

        proposal = current + step_size*step

        #compute its acceptance ratio
        ratio = np.exp(model.log_posterior(proposal) \
                        - model.log_posterior(current))

        #check if accepted
        threshold = np.random.random()
        if ratio > threshold:
            current = proposal

        #update samples an acceptance accordingly
        samples[k,:] = current
        record_acceptance[k] = (ratio > threshold)

        if verbose_gen == True:
            if k%5 == 0 or k == max_iter-1:
                update_progress((k+1)/max_iter)

    # saving the data
    #defining the burnin parameter
    if "burning" in kwargs.keys():
        burning = kwargs["burning"]
    else:
        burning = int(max_iter/10)

    #saving the estimates in the model
    ## NOTE: could be generalized if time
    covariance = np.cov(samples[burning:].T)
    model.results["MH_Gibbs_mean"] = [np.mean(samples[burning:],axis = 0),covariance]

    end = time.time()
    if verbose:
        print(" Acceptance rate : {:2.1%}  (advised values between 10% and 50%)"\
                    .format(np.mean(record_acceptance)))
        print("  duration: {}".format(str(datetime.timedelta(
                                                    seconds= round(end-start)))))

    model.time["MH_Gibbs"] = [end-start,max_iter]
    if RETURN:
        if "acc" in kwargs.keys():
            if kwargs["acc"] == True:
                return samples, np.mean(record_acceptance)
        return samples
