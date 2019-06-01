import autograd.numpy as anp
from autograd import grad
from autograd import jacobian

from .func_stats import *


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

    def __call__(self, theta):
        return self.l(theta)

    def __repr__(self):
        return "conditional model: "+self.name


class Gaussian(Conditional_model):
    """Implementation of Linear model with Gaussian errors."""

    def __init__(self,X,y,**kwargs):
        super(Gaussian,self).__init__(X,y,**kwargs)
        self.name = "gaussian"


    def l(self,theta):
        return normal(self.y,np.dot(self.X,theta[1:]),theta[0])

    def log_l(self,theta):

        return np.sum(np.log(normal(self.y, np.dot(self.X,theta[1:]), theta[0],
                                    prod = False)))


class Student(Conditional_model):
    """Implementation of Linear model with student errors."""

    def __init__(self,X,y,**kwargs):
        super(Student,self).__init__(X,y,**kwargs)
        self.name = "student"

    def l(self,theta):
        return student(self.y-np.dot(self.X,theta[1:]), theta[0])

    def log_l(self,theta):
        return np.sum(np.log(student(self.y-np.dot(self.X,theta[1:]),
                                theta[0], prod = False)))


## NOTE: not working, maybe check for negative values and compare with scikit
class Gamma(Conditional_model):
    """Implementatio of the GLM using a gamma distribution and canonical
    link function """

    def __init__(self,X,y,**kwargs):
        super(Gamma,self).__init__(X,y,**kwargs)
        self.name = "Gamma GLM"

    def l(self,theta):
        return gamma_(self.y, alpha= 10., beta= np.dot(self.X,theta))

    def log_l(self,theta):
        return np.sum(np.log(gamma_(self.y, alpha= 10., beta= np.dot(self.X,theta),
                                    prod = False)
                            )
                    )

class Multilogistic(Conditional_model):

    def __init__(self,X,y, number_classes, **kwargs):
        super(Multilogistic,self).__init__(X,y,**kwargs)
        self.number_classes = number_classes
        self.name = "Multilogistic"

    def extract_proba(self,theta):

        #reshaping data
        dim = self.X.shape[1]
        Betas = theta.reshape(dim,self.number_classes-1)

        #computing probabilities
        P = np.dot(self.X,Betas)

        #add the last column as the reference one for identifiability
        X0 = 0.*np.zeros((P.shape[0],1))
        P = np.hstack((P,X0))
        #softmax
        P = np.exp(P)
        cste = 1/np.sum(P,axis = 1)
        P = cste[:,None]*P

        return P


    def l(self,theta):

        #reshaping data
        y_temp = convert_to_one_hot(self.y)
        #computing probabilities
        P = self.extract_proba(theta)
        #computing the likelihood
        return np.prod(P**y_temp)



    def log_l(self,theta):

        #reshaping the data
        y_temp = convert_to_one_hot(self.y)
        #computing probabilities
        P = self.extract_proba(theta)
        #compute the log_likelihood in a satble manner
        return np.sum(np.log(P**y_temp))



    ## TODO: Implement those if time

    def log_l_grad(self,theta):
        return NotImplemented

    def log_l_grad(self, theta):
        return NotImplemented

    def log_l_hes(self, theta):
        return NotImplemented

    def neg_log_l_grad(self, theta):
        return NotImplemented

    def neg_log_l_hes(self, theta):
        return NotImplemented



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
