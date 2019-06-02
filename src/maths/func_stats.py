"""" statistical distribution compatible with autograd computation

    distribution implemented:
        - normal
        - exponential
        - gamma
        -student
"""

import autograd.numpy as np
from autograd import grad
from math import pi
from autograd.scipy.special import gamma
from autograd.scipy.stats import norm as Normal


from scipy.special import factorial

################################################################################
#                      helpers for strictly pos ditribution
################################################################################

def indicator_positive(x):
    test = np.array(x)
    if len(test.shape) > 0:
        return [1 if v >= 0 else 0 for v in x]
    else:
        return 1 if x>=0 else 0


################################################################################
#                               actual distribution
################################################################################

def normal(theta,mean,var, prod = True):
    """ normal distribution with same variance for all the components

    Parameters
    ----------
    theta : float or np.ndarray
    mean : float or np.ndarray
        mean of the distribution
         (if float, assumes same mean for all thetas)
    var : float or np.ndarray
        variance of the distribution
        (if float, assumes same variance for all thetas)
    prod : bool
        If true return the density of the sample
        If False, return the joint distribution

    Returns
    -------
    type
        float if prod
        np.ndarray if not prod
    """
    if np.min(var)<0:
        return 0
        raise ValueError("invalid variance given in normal in func_stats.py")

    individual = np.exp(-(theta-mean)**2 / (2*var**2))/ (np.sqrt(2*pi)*var)

    if prod:
        return np.prod(individual)
    else:
        return individual

def exponential(theta,lambda_, prod = True):
    """exponential distribution, assume a positive input

    Parameters
    ----------
    theta : float or np.ndarray
    lambda_ : float
    prod : bool
        If true return the density of the sample
        If False, return the joint distribution

    Returns
    -------
    type
        float if prod
        np.ndarray if not prod
    """

    x = indicator_positive(theta)
    individual = np.exp(-lambda_*theta)*lambda_*x

    if prod:
        return np.prod(individual)
    else:
        return individual



def gamma_(theta,alpha,beta, prod = True):
    """ gamma distribution with parameter alpha, beta

    Parameters
    ----------
    theta : np.ndarray
    alpha : float
        shape of the gamma distribution, > 0
    beta : float
        rate of the distribution, > 0
    prod : bool
        If true return the density of the sample
        If False, return the joint distribution

    Returns
    -------
    type
        float if prod
        np.ndarray if not prod

    """
    x = indicator_positive(theta)
    individual = beta**alpha*theta**(alpha-1)*np.exp(-beta*theta)/gamma(alpha)*x

    if prod:
        return np.prod(individual)
    else:
        return individual



def student(theta,df,prod = True):
    """Implementation of the student t distribution with df degrees of freedom

    Parameters
    ----------
    theta : type
        Description of parameter `theta`.
    df : type
        Description of parameter `df`.
    prod : bool
        If true return the density of the sample
        If False, return the joint distribution

    Returns
    -------
    type
        float if prod
        np.ndarray if not prod
    """
    individual = gamma((df+1.)/2.)*(1+theta**2 / df)**(-(df+1)/2) \
                /(gamma(df/2.)*np.sqrt(df*pi))
    if prod:
        return np.prod(individual)
    else:
        return individual

def multinomial(theta,p):
    """implementation of the multinomial distribution, using the gamma function
    approximation of the factorial as in scipy.special

    Parameters
    ----------
    theta : np.ndarray of integers of size d
        input following the distribution, should contains the number of examples
        falling into each class
    p : np.ndarray of size d
        probabilities to fall in each classes, shoudl sum up to 1


    Returns
    -------
    float
        probability mass function evaluated in theta

    """

    n = np.sum(theta)
    ratio = gamma(n)/np.prod(gamma(theta))
    return ratio*p**theta
