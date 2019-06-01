'''
General abstract class of a model, contains a prior and a conditional distribution
'''

import autograd.numpy as np
from autograd import grad,jacobian




class Model(object):

    def __init__(self, Prior_create, cond_model_create , **kwargs):

        #saving the data
        self.data = kwargs["data"]
        self.response = kwargs["response"]

        #defining the prior distribution
        self.prior = Prior_create(*kwargs["Prior"])

        #defining the Conditional model
        if "cond_model" in kwargs.keys():
            self.cond_model = cond_model_create(self.data, self.response,
                                                    *kwargs["cond_model"])
        else:
            self.cond_model = cond_model_create(self.data,self.response)

        #additional useful information for the function
        self.size = self.data.shape[1]
        if "additional_param" in kwargs.keys():
            self.size += kwargs["additional_param"]

        if self.cond_model.name == "Multilogistic":
            # number of classes * dimension of linear param
            self.size = (self.cond_model.number_classes-1)*self.data.shape[1]

        self.name = "Conditional model : {},  Prior : {}".format(
                            self.cond_model.name,self.prior.name)
        self.results = {}


    def posterior(self,theta):
        return self.prior(theta)*self.cond_model(theta)


    def log_posterior(self,theta):
        return self.prior.log_prior(theta) + self.cond_model.log_l(theta)

    def log_posterior_grad(self,theta):
        return self.prior.log_posterior_grad(theta)\
                +self.cond_model.log_l_grad(theta)
        return grad_fun(theta)

    def log_posterior_hessian(self,theta):
        hes_fun = jacobian(self.log_posterior_grad)
        return hes_fun(theta)


    def __repr__(self):
        return self.name
