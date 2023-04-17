


import numpy as np
from numpy.linalg import norm
from fixed_point_schemes import fixed_point_schemes
from problem_loader import problem_loader
import matplotlib.pyplot as plt
import tikzplotlib
import itertools


# markers = itertools.cycle(('o', '*', 'v', '^', '>', '<', '8', 's', 'p', 'P', 'h', 'H', 'X', 'd')) 
# markers = itertools.cycle(('+', 'x', '|', '_')) 
markers = itertools.cycle(('x'))

def run_problems(problems, methods, export_name='norm_gk', show_plots=True, ms = None, plot_densities=None, withms=None, K_maxs=None):
    # vary problems and solvers

    if not withms:
        withms = len(methods)*[False]
    if not plot_densities:
        plot_densities = len(problems)*[1]
    if not K_maxs:
        K_maxs = len(problems)*[1000]

    for i, problem in enumerate(problems):
        plot_density = int(1/plot_densities[i])
        fig, ax = plt.subplots()
        for j, method in enumerate(methods):
            tmp = ms if withms[j] else [5]
            for m in tmp:
                solver = problem_loader(problem, method=method, m=m, K_max=K_maxs[i])
                norm_g_ks = solver.run()
                labeltext = f'{method}, m={m}' if withms[j] else f'{method}'
                ax.semilogy(range(0, len(norm_g_ks), plot_density), norm_g_ks[::plot_density], label=labeltext, linestyle='', marker=next(markers))
        plt.legend(title='method')
        if show_plots:
            plt.show()
        else:
            ax.set(xlabel=r'iteration number $k$', ylabel=r'residual $\norm{g_k}/\norm{g_0}$')
            tikzplotlib.save(f'../Plots/{export_name}_{problem}.pgf')


if __name__ == '__main__':
    problems = ['CO', 'GD', 'ISTA', 'VI']
    methods = ['aa1-matrix', 'aa1', 'aa1-safe', 'aa2-matrix', 'original']
    ms = [2,5,10,20,50]
    withms = [True, True, False]

    if True:
        run_problems(problems[:-1], methods[1:], 'method_comparison', False)
    if True:
        run_problems(problems[:-1], methods[2:3], 'memory_comparison', False, ms, len(problems)*[0.1], len(methods)*[True])
    if True:
        run_problems(problems[-1:], methods, 'method_comparison', False, K_maxs=[100])
    if True:
        run_problems(problems[-1:], methods[2:3], 'memory_comparison', False, ms, withms=[True], K_maxs=[100])
    