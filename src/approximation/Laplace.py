import autograd.numpy as np
import numpy
from math import log, pi

from src.optimization import gradient_descent

def laplace_approx(model, integral_value = False):

    if "gd" not in model.results.keys():
        theta_map = gradient_descent.vanilla_gd(model,
                                    max_iter = 2000,RETURN = True, save = False)
    else:
        theta_map = model.results["gd"]

    curvature = model.neg_log_posterior_hessian(theta_map)
    W,_ = numpy.linalg.eig(curvature)
    det = numpy.prod(W)


    if integral_value:
        return abs(det)**(-0.5)*model.posterior(theta_map)*(2*pi)**(model.size/2)
    else :
        return theta_map,curvature
