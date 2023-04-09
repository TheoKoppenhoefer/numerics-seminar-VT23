


import numpy as np
from numpy.linalg import norm
from fixed_point_schemes import fixed_point_schemes
from problem_loader import problem_loader
import matplotlib.pyplot as plt
import tikzplotlib
import itertools


# markers = itertools.cycle(('o', '*', 'v', '^', '>', '<', '8', 's', 'p', 'P', 'h', 'H', 'X', 'd')) 
markers = itertools.cycle(('+', 'x', '|', '_')) 

def run_problems(problems, methods, ms, withms, export_name='norm_gk', show_plots=True, plot_density=0.1):
    # vary problems and solvers               
    plot_density = int(1/plot_density)

    for problem in problems:
        fig, ax = plt.subplots()
        for i, method in enumerate(methods):
            tmp = ms if withms[i] else [5]
            for m in tmp:
                solver = problem_loader(problem, method=method, m=m)
                norm_g_ks = solver.run()
                labeltext = f'{method}, m={m}' if withms[i] else f'{method}'
                ax.semilogy(range(0, len(norm_g_ks), plot_density), norm_g_ks[::plot_density], label=labeltext, linestyle='', marker=next(markers))
        plt.legend(title='method')
        if show_plots:
            plt.show()
        else:
            ax.set(xlabel=r'iteration number $k$', ylabel=r'residual $\norm{g(x_k)}/\norm{g(x_0)}$')
            tikzplotlib.save(f'../Plots/{export_name}_{problem}.pgf')


if __name__ == '__main__':
    problems = ['CO', 'GD', 'ISTA']
    methods = ['aa1', 'aa1-safe', 'original']
    ms = [2,5,10,20,50]
    withms = [True, True, False]

    run_problems(problems[2:3], methods[1:2], ms, [True], export_name='aa1_safe_mem', show_plots=False, plot_density=0.1)
    