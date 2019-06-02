"""
module for matrix manipulation
"""
import autograd.numpy as np

def sym(M):
    return 0.5 * (M + np.transpose(M))

def exp(M):
    u, s, v = np.linalg.svd(M)
    return np.matmul(u, np.matmul(np.diag(np.exp(s)), v))

def log(M):
    u, s, v = np.linalg.svd(M)
    return np.matmul(u, np.matmul(np.diag(np.log(s)), v))

def sqrt(M):
     return np.linalg.cholesky(M)
