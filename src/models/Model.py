'''
General abstract class of a model, contains a prior and a conditional distribution
'''

import autograd.numpy as np
from autograd import grad,jacobian
import pandas as pd




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

        if "predictors_names" in kwargs.keys():
            self.names_pred = kwargs["predictors_name"]
        else:
            self.names_pred = np.array(['ED', 'SOUTH', 'NONWH', 'HISP',
                                        'FE', 'MARR', 'MARRFE', 'EX',
                                        'UNION', 'MANUF', 'CONSTR', 'MANAG',
                                        'SALES', 'CLER', 'SERV', 'PROF'])

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
        return self.prior.log_prior_grad(theta)\
                +self.cond_model.log_l_grad(theta)

    def log_posterior_hessian(self,theta):
        return self.prior.log_prior_hes(theta)\
               +self.cond_model.log_l_hes(theta)


    def neg_log_posterior(self,theta):
        return -self.log_posterior(theta)

    def neg_log_posterior_grad(self,theta):
        return -self.log_posterior_grad(theta)

    def neg_log_posterior_hessian(self,theta):
        return -self.log_posterior_hessian(theta)

    def view(self):
        if len(self.results)== 0:
            return "no results yet"
        inter = pd.DataFrame(self.results).T
        if self.size > len(self.names_pred):
            inter.columns = np.insert(self.names_pred,0,"error prior")
        else:
            inter.columns = self.names_pred
        inter = inter.T
        return inter

    def __call__(self):
        return self.view()


    def __repr__(self):
        return self.name
