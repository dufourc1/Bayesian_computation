import autograd.numpy as anp
from autograd import grad
from autograd import jacobian

from src.maths.func_stats import *


class Conditional_model(object):

    def __init__(self, X, y, **kwargs):

        self.X = X
        self.y = y
        self.name = "Super Class"


    def l(self, theta):
        # likelihood
        return NotImplemented

    def log_l(self, theta):
        # log likelihood by default
        return np.log(self.l(theta))

    def neg_log_l(self, theta):
        # negative log likelihood
        return - self.log_l(theta)

    def log_l_grad(self, theta):
        # gradient of log likelihood
        grad_fun = grad(self.log_l)
        return grad_fun(theta)

    def log_l_hes(self, theta):
        # hessian of log likelihood
        hessian = jacobian(self.log_l)
        return hessian(theta)

    def neg_log_l_grad(self, theta):
        # gradient of negative log likelihood
        grad_fun = grad(self.neg_log_l)
        return grad_fun(theta)

    def neg_log_l_hes(self, theta):
        # hessian of negative log likelihood
        hessian = jacobian(self.neg_log_l)
        return hessian(theta)

    def prediction(self,X_test,theta):
        return NotImplemented

    def __call__(self, theta):
        return self.l(theta)

    def __repr__(self):
        return "conditional model: "+self.name


class Gaussian(Conditional_model):
    """Implementation of Linear model with Gaussian errors."""

    def __init__(self,X,y,**kwargs):
        super(Gaussian,self).__init__(X,y,**kwargs)
        self.name = "gaussian"


    #likelihood for both new data point and prerecorded data
    def l(self,theta):
        return normal(self.y,np.dot(self.X,theta[1:]),theta[0])

    def l_new(self,theta,y,X):
        return normal(y,np.dot(X,theta[1:]),theta[0])

    def log_l(self,theta):
        return np.sum(np.log(normal(self.y, np.dot(self.X,theta[1:]), theta[0],
                                    prod = False)))

    def log_l_new(self,theta,y,X):
        return np.sum(np.log(normal(y, np.dot(X,theta[1:]), theta[0],
                                    prod = False)))

    def prediction(self,X_test,theta):
        ## REVIEW: wrong !!!!
        return np.dot(X_test,theta)


class Student(Conditional_model):
    """Implementation of Linear model with student errors."""

    def __init__(self,X,y,**kwargs):
        super(Student,self).__init__(X,y,**kwargs)
        self.name = "student"

    def l(self, theta, y = None,X = None):
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        return student(y-np.dot(X,theta[1:]), theta[0])

    def log_l(self,theta, y = None,X = None):
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        return np.sum(np.log(student(y-np.dot(X,theta[1:]),
                                theta[0], prod = False)))

    def prediction(self,X_test,theta):
        ## REVIEW: wrong !!!!
        return np.dot(X_test,theta)


# NOTE: compare with scikit
class Gamma(Conditional_model):
    """Implementatio of the GLM using a gamma distribution and canonical
    link function """

    def __init__(self,X,y,**kwargs):
        super(Gamma,self).__init__(X,y,**kwargs)
        if np.min(self.y)<= 0:
            raise ValueError("response cannot be negative for gamma distr")
        self.name = "Gamma GLM"
        self.nu = 4

    def l(self,theta):
        return gamma_(self.y, alpha= np.exp(self.X,theta), beta= self.nu)

    def log_l(self,theta):
        return np.sum(np.log(gamma_(self.y, alpha= np.exp(self.X,theta),
                                    beta= self.nu,
                                    prod = False)
                            )
                    )


# REVIEW:generalize the Multilogistic model

class Multilogistic(Conditional_model):
    """ predict the 1 class if two classes

    Parameters
    ----------
    X : type
        Description of parameter `X`.
    y : type
        Description of parameter `y`.
    number_classes : type
        Description of parameter `number_classes`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    Attributes
    ----------
    name : type
        Description of attribute `name`.
    self,theta : type
        Description of attribute `self,theta`.
    number_classes

    """

    def __init__(self,X,y, number_classes, **kwargs):
        super(Multilogistic,self).__init__(X,y,**kwargs)
        if number_classes >2:
            raise NotImplementedError("only logistic regression for two classes is implemented for now")
        self.number_classes = number_classes
        self.name = "Multilogistic"

    def extract_proba(self,theta):

        #reshaping data
        dim = self.X.shape[1]

        #computing probabilities
        expo = np.exp(np.dot(self.X,theta))
        P = expo/(1+expo)
        return P


    def l(self,theta):

        #computing probabilities
        P = self.extract_proba(theta)
        #computing the likelihood
        return np.prod(P**self.y)*np.prod((1-P)**(1-self.y))



    def log_l(self,theta):

        #computing probabilities
        P = self.extract_proba(theta)
        #compute the log_likelihood in a satble manner
        return np.sum(self.y*np.log(P)+(1-self.y)*np.log(1-P))



    ## TODO: Implement those if time for Multilogistic regression
    # now only valid for classic logistic regression

    def log_l_grad(self,theta):
        if self.number_classes > 2:
            raise NotImplementedError("only implemented for logistic regression")
        P = self.extract_proba(theta)
        inter = (self.y - P)*self.X.T
        return np.sum(inter,axis = 1)

    def log_l_hes(self, theta):
        if self.number_classes > 2:
            raise NotImplementedError("only implemented for logistic regression")
        P = self.extract_proba(theta)
        return np.sum(self.X[:, :, None] *\
                self.X[:, None, :] *\
                (P / (1 + P) ** 2).reshape(-1, 1, 1),axis=0)

    def neg_log_l_grad(self, theta):
        if self.number_classes > 2:
            raise NotImplementedError("only implemented for logistic regression")
        return -self.log_l_grad(theta)

    def neg_log_l_hes(self, theta):
        if self.number_classes > 2:
            raise NotImplementedError("only implemented for logistic regression")
        return -self.log_l_hes(theta)

    def predict(self,X_test,theta):
        P = self.extract_proba(theta)
        predicted = np.zeros_like(P)
        predicted[P>=0.5]=1
        return predicted


    def __repr__(self):
        return self.name + ", " + str(self.number_classes)+ " classes"


################################################################################
#                           classification helpers
################################################################################

def convert_to_one_hot(labels):

    tmp = np.zeros([len(labels), int(labels.max() + 1)])
    for i,v in enumerate(labels):
        tmp[i,int(v)] = 1
    return tmp
