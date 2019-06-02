import autograd.numpy as np
from autograd import grad
from autograd import jacobian

from src.maths.func_stats import *

class Prior(object):

    def single(self,theta):
        """univariate probability distribution function"""
        return NotImplemented

    def prior(self, theta):
        """likelihood"""
        return NotImplemented

    def log_prior(self, theta):
        """log likelihood"""
        return np.log(self.prior(theta))

    def neg_log_prior(self, theta):
        """negative log likelihood"""
        return - self.log_prior(theta)

    def log_prior_grad(self,theta):
        grad_fun = grad(self.log_prior)
        return grad_fun(theta)

    def log_prior_hes(self, theta):
        """hessian of log likelihood"""
        hessian = jacobian(self.log_prior_grad)
        return hessian(theta)

    def neg_log_prior_grad(self, theta):
        """gradient of negative log likelihood"""
        grad_fun = grad(self.neg_log_prior)
        return grad_fun(theta)

    def neg_log_prior_hes(self, theta):
        """hessian of log likelihood"""
        hessian = jacobian(self.neg_log_prior_grad)
        return hessian(theta)

    def __call__(self, theta):
        return self.prior(theta)


################################################################################
#               priors for the linear parameter + noise parameter
################################################################################
class Gaussian_exp_prior(Prior):
    """ Implementation of the gaussian prior for the parameters beta and the
        exponential prior for the sigma
        beta ~ N(mean,std**2)
        sigma ~ Exp(lamdba)

    Parameters
    ----------
    mean : float
        if the Gaussian_prior is called as a multivariate normal, the mean is a
        vector  equal mean in every coordinate
    std : float > 0
        if the Gaussian_prior is called as a multivariate normal, the variance
        is a diagoonal matrix with std as element on the whole diag
    scale : float
        inverse of the scale for the variance parameters

    Attributes
    ----------
    theta : array like or float
        parameter that follows this prior: [sigma,beta]
    """

    def __init__(self,mean,std, lambda_):

        self.mean = mean
        self.std = std
        self.lambda_ = lambda_
        self.name = "gaussian and exponential"


    def prior(self,theta):

        prior_beta = normal(theta[1:],self.mean, self.std)
        prior_sigma = exponential(theta[0],self.lambda_)

        return prior_beta*prior_sigma


    def log_prior(self,theta):

        log_prior_beta = np.sum(np.log(normal(theta[1:],self.mean,self.std, prod = False)))
        log_prior_sigma = np.sum(np.log(exponential(theta[0], lambda_ = self.lambda_,prod = False)))

        return log_prior_beta + log_prior_sigma


class Gaussian_gamma_prior(Prior):
    """ Implementation of the Student prior for the parameters beta and the
        gamma prior for the df of the student noise
        beta ~ N(mean,std**2)
        df ~ Gamma(alpha,beta)

    Parameters
    ----------
    mean : float
        if the Gaussian_prior is called as a multivariate normal, the mean is a
        vector  equal mean in every coordinate
    std : float > 0
        if the Gaussian_prior is called as a multivariate normal, the variance
        is a diagoonal matrix with std as element on the whole diag
    scale : float
        inverse of the scale for the variance parameters

    Attributes
    ----------
    theta : array like or float
        parameter that follows this prior [sigma,beta]
    """

    def __init__(self,mean,std, alpha, beta):

        self.mean = mean
        self.std = std
        self.alpha = alpha
        self.beta = beta
        self.name = "gaussian and gamma"


    def prior(self,theta):

        prior_beta = normal(theta[1:],self.mean, self.std)
        prior_df = gamma_(theta[0],self.alpha, self.beta)

        return prior_beta*prior_df


    def log_prior(self,theta):

        log_prior_beta = np.sum(np.log(normal(theta[1:],self.mean,self.std,prod = False)))
        log_prior_df = np.sum(np.log(gamma_(theta[0],self.alpha, self.beta,prod = False)))


        return log_prior_beta + log_prior_df

################################################################################
#                priors for the linear parameter only
################################################################################

class Gaussian_prior(Prior):

    """ Implementation of the vanilla gaussian prior
        theta ~ N(mean,std**2)

    Parameters
    ----------
    mean : float
        if the Gaussian_prior is called as a multivariate normal, the mean is a
        vector  equal mean in every coordinate
    std : float > 0
        if the Gaussian_prior is called as a multivariate normal, the variance
        is a diagoonal matrix with std as element on the whole diag

    Attributes
    ----------
    theta : array like or float
        parameter that follows this prior [beta]
    """

    def __init__(self,mean,std):

        self.mean = mean
        self.std = std
        self.name = "gaussian"


    def prior(self,theta):

        return normal(theta,self.mean, self.std)


    def log_prior(self,theta):

        return np.sum(np.log(normal(theta,self.mean,self.std, prod = False)))
