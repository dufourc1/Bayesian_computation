import autograd.numpy as np
import numpy
from math import log, pi
from src.maths import mat
import time

from src.optimization import gradient_descent

def laplace_approx(model,name_method = None, integral_value = False, method = None, out = False):

    start = time.time()
    if method is None:
        theta_map = gradient_descent.vanilla_gd(model,
                                    max_iter = 8000,RETURN = True, save = False)
    else:
        theta_map = method(model, max_iter = 8000,RETURN = True, save = False)

    curvature = model.neg_log_posterior_hessian(theta_map)
    #curvature = mat.sym(curvature)
    try:
        if  model.additional_param > 0:
            # the noise parameter are clearly not gaussian, approximation does not make sens
            curvature = curvature[model.additional_param:,model.additional_param:]
    except :
        #no noise parameter (strictly positive found)
        pass
    W,_ = numpy.linalg.eig(curvature)
    det = numpy.prod(W)

    if name_method is None:
        name_method = "Laplace with vanilla gd"
    else:
        name_method = "Laplace with " + name_method
    model.results[name_method] = [theta_map,np.linalg.inv(curvature)]

    stop = time.time()
    model.time[name_method] = [stop-start,8000]
    if integral_value:
        return abs(det)**(-0.5)*model.posterior(theta_map)*(2*pi)**(model.size/2)
    else :
        if out:
            return theta_map,curvature
