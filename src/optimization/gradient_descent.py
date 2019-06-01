"""
gradient descent variant

automaticaly find the MAP of the posterior passed in parameter

"""
import autograd.numpy as np
from autograd.numpy.linalg import inv
from src.helpers import update_progress
import time,datetime


################################################################################
#optimization functions
################################################################################



def vanilla_gd(model, max_iter = 10, step_size = 1e-4, initial = None, trace = False,RETURN = False):

    if initial is None:
        initial = np.random.randn(model.size)
        #set the variance to a positive parameter
        initial[0]=1

    fun = model.log_posterior
    grad_fun = model.log_posterior_grad
    trace_energy = [fun(initial)]
    trace_theta = [initial]

    start = time.time()
    theta = initial
    for i in range(max_iter):
        theta = theta + step_size*model.log_posterior_grad(theta)
        trace_theta.append(theta)
        trace_energy.append(fun(theta))

        if i%5 == 0 or i == max_iter-1:
            update_progress((i+1)/max_iter)
    end = time.time()
    print("  duration: {}".format(str(datetime.timedelta(
                                                seconds= round(end-start)))))

    model.results["gd"] = theta


    if trace :
        return trace_theta,trace_energy
    elif RETURN:
        return theta

def line_search_gd(model):
    return NotImplemented

def Wolfe_cond_gd(model):
    return NotImplemented

def stochastic_gd(model):
    return NotImplemented

def newton_gd():
    return NotImplemented



################################################################################
#
################################################################################

def line_search_gd(f, df, lambda_, x0, alpha = 0.2, beta = 0.5, n = 20, epsilon = 1e-4):
    '''
    return the n first steps of linear search gradient descent (adaptative step-size)

    input:
        f: function to minimize
        df: gradient of the function
        lambda_0 : initial step-size
        x0: initial value for the algorithm
        alpha: parameter to avoid overshooting in [0,1]
        beta: parameter to update the step-size in [0,1]
        n: number of iterations of the algorithm
        epsilon: paramter to check convergence of the algorithm

    output:
        values, energies
        values = array containing the n first coordinate of the gradient descent algortihm
        energies = array containing the value of the function at these points
    '''
    values = [x0]
    energies = [f(x0)]

    old = x0
    for i in range(n):

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


    return values,energies

def newton_gd(f, df, hf, x0, alpha = 0.2, beta = 0.5, n = 20, epsilon = 1e-4):
    '''
    return the n first steps of gradient descent using newton's method (adaptative step-size)

    input:
        f: function to minimize
        df: gradient of the function
        hf: hessian of f
        x0: initial value for the algorithm
        alpha: parameter to avoid overshooting in [0,1]
        beta: parameter to update the step-size in [0,1]
        n: number of iterations of the algorithm
        epsilon: paramter to check convergence of the algorithm

    output:
        values, energies
        values = array containing the n first coordinate of the gradient descent algortihm
        energies = array containing the value of the function at these points
    '''
    values = np.array([x0])
    energies = [f(x0)]

    xn = x0

    for i in range(n):

        gradient = df(xn)
        hessian_inv = inv(hf(xn))
        l = 1

        v = np.dot(hessian_inv,gradient)
        candidate = xn - l*v

        if np.linalg.norm(candidate-xn)< epsilon:
            break

        #securtiy measure
        j = 0
        while f(candidate) > f(xn) - l*alpha*np.dot(gradient.T ,np.dot(hessian_inv, gradient)):
            j+= 1
            l*= beta
            candidate = xn - l*v
            if j > 100:
                print("more than 100 iterations to adjust the step size")
                break
        values = np.concatenate((values,candidate.reshape(1,len(candidate))))
        energies.append(f(candidate))
        xn = candidate

    return values,energies

def stochastic_gd(f, df, x0, batch_size = None, lambdas = None, n = 50, epsilon = 1e-4,*args):
    '''
    return the n first steps of gradient descent using stochastic gradient descent

    input:
        f: function to minimize
        df: !!! gradient of the function for one point only !!! df(theta,**kwargs)
        x0: initial value for the algorithm
        batch_size: size of the batches used to compute the stochastic gradient descent
        lambdas() : function that returns a step size for each n it's given
        n: number of iterations of the algorithm
        epsilon: paramter to check convergence of the algorithm
        *args: arg needed to compute the gradient, should be numpy array for the batching process to work

    output:
        values, energies
        values = array containing the n first coordinate of the gradient descent algortihm
        energies = array containing the value of the function at these points
    '''

    if lambdas == None:
        def lambdas(n):
            return (n+1)**(-2)
    if batch_size == None:
        batch_size = int(len(args[0])/10)


    values = np.array([x0])
    energies = [f(x0)]
    old = x0

    for i in range(n):

        #compute the stochastic_gd
        gradient = 0
        for x,y in batch_iter(*args, batch_size, num_batches=1, shuffle=True):
            gradient += df(old,x,y)


        #update following the usual scheme
        candidate = old - lambdas(i)*gradient
        values = np.concatenate((values,candidate.reshape(1,len(candidate))))
        energies.append(f(candidate))

        if np.linalg.norm(old-candidate)< epsilon:
            break
        old = candidate

    return values,energies

################################################################################
# utilities form ML course
################################################################################


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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
