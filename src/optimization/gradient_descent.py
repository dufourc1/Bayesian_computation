"""
gradient descent variant

automaticaly find the MAP of the posterior passed in parameter by minimizing the
negative log posterior of the model

"""
import autograd.numpy as np
from autograd.numpy.linalg import inv
from src.helpers import update_progress
import time,datetime


################################################################################
#optimization functions
################################################################################


def vanilla_gd(model, max_iter = 10, step_size = 1e-4, initial = None, trace = False,RETURN = False,save = True):

    if initial is None:
        initial = np.random.randn(model.size)
        #set the variance to a positive parameter
        initial[0]=1

    fun = model.neg_log_posterior
    grad_fun = model.neg_log_posterior_grad
    trace_energy = [fun(initial)]
    trace_theta = [initial]

    start = time.time()
    theta = initial
    for i in range(max_iter):
        theta = theta - step_size*grad_fun(theta)
        trace_theta.append(theta)
        trace_energy.append(fun(theta))

        if i%5 == 0 or i == max_iter-1:
            update_progress((i+1)/max_iter)
    end = time.time()
    print("  duration: {}".format(str(datetime.timedelta(
                                                seconds= round(end-start)))))

    if save:
        model.results["gd"] = theta


    if trace :
        return trace_theta,trace_energy
    elif RETURN:
        return theta


def line_search_gd(model, lambda_, x0, alpha = 0.2, beta = 0.5, max_iter = 20, epsilon = 1e-4,
                    trace = False, RETURN = False):



    f = model.neg_log_posterior
    df = model.neg_log_posterior_grad
    values = [x0]
    energies = [f(x0)]

    old = x0
    for i in range(max_iter):

        gradient = df(old)
        l = lambda_

        candidate = old-l*gradient

        if np.linalg.norm(candidate-old)< epsilon:
            break

        #security measure
        j= 0

        while f(candidate) > f(old)-l*alpha*np.linalg.norm(gradient)**2:
            l *= beta
            candidate = old-l*gradient
            j+= 1
            if j > 100:
                print("more than 100 iterations to adjust the step size")
                break

        values = np.concatenate((values,candidate.reshape(1,len(candidate))))
        energies.append(f(candidate))

        old = candidate
        if i%5 == 0 or i == max_iter-1:
            update_progress((i+1)/max_iter)

    model.results["line_search_gd"] = old

    if trace:
        return values,energies
    if RETURN:
        return old


def Wolfe_cond_gd(model, lambda_0 = None, initial = None, max_iter = 10,
                trace = False, RETURN = False,
                c1= 1e-4, c2= 0.9, beta_C1= 0.9, beta_C2 = 1.1):

    if initial is None:
        initial = np.ones(model.size)
    if lambda_0 is None:
        lambda_0 = 1e-3

    if trace:
        trace_theta = [initial]
        trace_lambdas = [lambda_0]
        trace_energy = [model.neg_log_posterior(initial)]

    theta = initial.copy()
    step_size = lambda_0

    start = time.time()

    for i in range(max_iter):
        proposal = theta-step_size*model.neg_log_posterior_grad(theta)
        checks = check_wolfe_conditions(model.neg_log_posterior,
                                        model.neg_log_posterior_grad,
                                        theta,step_size,
                                        c1,c2)
        if checks[0] and checks[1]:
            if trace:
                trace_energy.append(model.neg_log_posterior(proposal))
                trace_theta.append(proposal)
                trace_lambdas.append(step_size)
            theta = proposal
        elif checks[0]:
            step_size *= beta_C1
        elif checks[1]:
            step_size *= beta_C2
        else:
            raise RuntimeError("Wolfe conditions both wrong, gradient must be wrong")
        if i%5==0 or i == max_iter-1:
            update_progress((i+1)/max_iter)
    end = time.time()
    print("  duration: {}".format(str(datetime.timedelta(
                                                seconds= round(end-start)))))

    model.results["Wolfe_cond_gd"] = theta

    if RETURN:
        return theta
    if trace:
        return trace_theta, trace_energy, trace_lambdas

def newton_gd(model,initial = None, max_iter = 10,
                trace = False, RETURN = False):

    raise RuntimeError("well have to fix it but don't know how")
    fun = model.neg_log_posterior
    grad_fun = model.neg_log_posterior_grad
    hes_fun = model.neg_log_posterior_hessian

    if initial is None:
        initial = np.ones(model.size)
        initial[0] = 2

    if trace :
        trace_thetas = [initial]
        trace_energy = [fun(initial)]


    start = time.time()
    theta = initial.copy()
    for i in range(max_iter):
        v = np.dot(np.linalg.inv(hes_fun(theta)),grad_fun(theta))
        cand = theta - v
        if trace:
            trace_thetas.append(cand)
            trace_energy.append(fun(cand))
        theta = cand
        if i%5 == 0 or i == max_iter-1:
            update_progress((i+1)/max_iter)

    end = time.time()
    print("  duration: {}".format(str(datetime.timedelta(
                                                seconds= round(end-start)))))

    model.results["Newton_gd"] = theta

    if trace:
        return trace_thetas, trace_energy
    if RETURN:
        return theta

def stochastic_gd(model):
    return NotImplemented




################################################################################
#               various helpers
################################################################################

#check Wolfe condition for minimization
def check_wolfe_conditions(fun, grad_fun, theta,lambda_, c1, c2):

    proposal = theta-lambda_*grad_fun(theta)
    norm_grad = np.linalg.norm(grad_fun(theta))**2
    C1 = fun(proposal)-fun(theta) - c1*lambda_*norm_grad
    C2 = -np.dot(grad_fun(theta),grad_fun(proposal)) + c2*norm_grad

    return (C1 <= 0, C2 <= 0)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness
    of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
