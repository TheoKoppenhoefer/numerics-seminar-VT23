
import numpy as np
from numpy.linalg import norm
import scipy.sparse as ss
from fixed_point_schemes import fixed_point_schemes


def problem_loader(problem='CO', **kwargs):
    np.random.seed(457)
    if problem == 'CO':
        m = 500
        n = 300
        c = ss.random(m, n, 1E-2)
        def f(z):
            z = z.reshape((m, n))
            x = prox_norm(z+c) - c
            return np.asarray((z+2*np.mean(x, axis=0)-x-np.mean(z, axis=0)).flatten()).ravel()
        x_0 = np.random.randn(n*m)
        x_0 = x_0 / norm(x_0)
        #kwargs['alpha'] = 1
        kwargs['tol'] = 1E-8
        kwargs['K_max'] = 5E2

    elif problem == 'GD':
        x = np.loadtxt('../Data/madelon_train.data')
        y = np.loadtxt('../Data/madelon_train.labels')
        m = y.size
        n = x.shape[1]
        lam = 1E-2
        L = norm(x)**2/(4*m)
        alpha = 2/(L+lam)
        yx = y@x
        f = lambda theta: theta - alpha*(yx/(m*(1+yx@theta))+lam*theta)
        x_0 = np.random.randn(n)
        x_0 = x_0 / norm(x_0) * 1E-3
        kwargs['alpha']=alpha
    
    else:
        raise Exception('Invalid problem name given.')

    return fixed_point_schemes(x_0, f, **kwargs)

def prox_norm(x):
    tmp = 1-1/norm(x, axis=1)
    tmp[tmp<0] = 0
    return tmp*x