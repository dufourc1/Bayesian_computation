import autograd.numpy as anp
from autograd import grad
from autograd import jacobian

from src.maths.func_stats import *


class Conditional_model(object):

    def __init__(self, X, y, **kwargs):

        self.X = X
        self.y = y
        self.name = "Super Class"


    def l(self,theta,y = None, X= None):
        #log_likelihood
        return NotImplemented

    def log_l(self, theta,y = None, X= None):
        # log likelihood by default
        return np.log(self.l(theta,y,X))

    def neg_log_l(self, theta,y = None, X= None):
        # negative log likelihood
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        return - self.log_l(theta,y,X)

    def log_l_grad(self, theta,y = None, X= None):
        # gradient of log likelihood
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        grad_fun = grad(self.log_l)
        return grad_fun(theta,y,X)

    def log_l_hes(self, theta,y = None, X= None):
        # hessian of log likelihood
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        hessian = jacobian(self.log_l)
        return hessian(theta,y,X)

    def neg_log_l_grad(self, theta,y = None, X= None):
        # gradient of negative log likelihood
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        grad_fun = grad(self.neg_log_l)
        return grad_fun(theta,y,X)

    def neg_log_l_hes(self, theta,y = None, X= None):
        # hessian of negative log likelihood
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        hessian = jacobian(self.neg_log_l)
        return hessian(theta,y,X)

    def prediction(self,X_test,theta):
        return NotImplemented

    def __call__(self, theta,y = None, X= None):
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        return self.l(theta,y,X)

    def __repr__(self):
        return "conditional model: "+self.name


class Gaussian(Conditional_model):
    """Implementation of Linear model with Gaussian errors."""

    def __init__(self,X,y,**kwargs):
        super(Gaussian,self).__init__(X,y,**kwargs)
        self.name = "gaussian"


    #likelihood
    def l(self,theta,y = None, X= None):
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        return normal(y,np.dot(X,theta[1:]),theta[0])


    def log_l(self,theta, y = None, X = None):
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        return np.sum(np.log(normal(y, np.dot(X,theta[1:]), theta[0],
                                    prod = False)))


    def prediction(self,X_test,theta):
        # only valid under certain specific assumptions
        return np.dot(X_test,theta[1:])


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
        # only valid under certain specific assumptions
        return np.dot(X_test,theta[1:])


# REVIEW: not finished
class Gamma(Conditional_model):
    """Implementation of the GLM using a gamma distribution and canonical
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


#  TODO: :generalize the Multilogistic model

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

    def extract_proba(self,theta, X = None):

        if X is None:
            X = self.X
        #reshaping data
        dim = X.shape[1]

        #computing probabilities
        expo = np.exp(np.dot(X,theta))
        P = expo/(1+expo)
        return P


    def l(self,theta):

        #computing probabilities
        P = self.extract_proba(theta)
        #computing the likelihood
        return np.prod(P**self.y)*np.prod((1-P)**(1-self.y))



    def log_l(self,theta, y = None, X = None):

        if y is None:
            y = self.y

        #computing probabilities
        P = self.extract_proba(theta,X)
        #compute the log_likelihood in a satble manner
        return np.sum(y*np.log(P)+(1-y)*np.log(1-P))



    ## TODO: Implement those if time for Multilogistic regression
    # now only valid for classic logistic regression

    def log_l_grad(self,theta, y = None, X = None):

        if self.number_classes > 2:
            raise NotImplementedError("only implemented for logistic regression")

        if y is None:
            y = self.y
        if X is None:
            X = self.X

        P = self.extract_proba(theta,X)
        inter = (y - P)*X.T
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

    def prediction(self,X_test,theta):
        # only valid under certain specific assumptions
        P = self.extract_proba(theta,X_test)
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
