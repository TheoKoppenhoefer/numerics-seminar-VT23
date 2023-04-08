


import numpy as np
from numpy.linalg import norm
from fixed_point_schemes import fixed_point_schemes
from problem_loader import problem_loader
import matplotlib.pyplot as plt
import tikzplotlib




problems = ['CO', 'GD']
methods = ['aa1-prs', 'original']

if True:
    # vary problems and solvers
    for problem in problems[1:]:
        fig, ax = plt.subplots()
        for method in methods:
            solver = problem_loader(problem, method=method)
            norm_g_ks = solver.run()
            ax.semilogy(range(len(norm_g_ks)), norm_g_ks, label=method)
        plt.legend(title='method')
        if True:
            plt.show()
        else:
            ax.set(xlabel=r'iteration number $k$', ylabel=r'residual $\norm{g(x_k)}/\norm{g(x_0)}$')
            tikzplotlib.save(f'../Plots/norm_gk_{problem}.pgf')