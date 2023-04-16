import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm

outputs = ['P-value', 'Haircut Sharpe Ratio', 'Haircut pct']
methods = ['Bon', 'Holm', 'BHY']
linestyles = ['-', '--', ':']
var_levels = [0.95, 0.99]

def get_var(alpha: float, sharpe_ratios: np.ndarray):
    z_score = norm.ppf(1-alpha)

    # number of time-series
    N = 240

    # this sigma is the same std of the one used in multivariate normal
    # as specified in sample_random_multests
    sigma = ((0.15/np.sqrt(12)))/np.sqrt(N)

    var = sigma*(sharpe_ratios - z_score)
    return var

def plot_sr(original_sr: np.ndarray, haircut_sr: np.ndarray,
            crosssec_rho: float, autocorrelation: float, num_test: int):
    for output_idx, output in enumerate(outputs):

        fig, ax = plt.subplots()
        for idx, method in enumerate(methods):
            ax.plot(original_sr, haircut_sr[:, output_idx, idx],
                    label=method, linestyle=linestyles[idx])

        if output == 'Haircut pct':
            ax.hlines(0.5, 0, original_sr[-1], color='black', alpha=0.5, label='$50\%$ adjustment')
        elif output == 'Haircut Sharpe Ratio':
            ax.plot([0, original_sr[-1]], [0, original_sr[-1]], color='black', alpha=0.5, label='No adjustment')
            ax.plot([0, original_sr[-1]], [0, original_sr[-1]/2], color='grey', alpha=0.5, label='$50\%$ adjustment')

        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel(output)
        ax.set_title(fr'SR adjustment: {num_test} tests, $\rho_c$ = {crosssec_rho}, $\rho_a$ = {autocorrelation}')
        ax.legend()
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
           
        plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
        plt.savefig(os.path.join(plot_dir, f'SR_{output}_{num_test}.png'))
        plt.show()
        plt.clf()
        

def plot_var(original_sr: np.ndarray, haircut_sr: np.ndarray,
            crosssec_rho: float, autocorrelation: float, num_test: int):
    output_idx = 1
    for alpha in var_levels:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        var = get_var(alpha, original_sr)
        for idx, method in enumerate(methods):
            var_adj = get_var(alpha, haircut_sr[:,output_idx,idx])
            var_hr = 100*(var - var_adj)/var
            
            level = int(alpha*100)
            ax[0].plot(var, var_adj, label=method, linestyle=linestyles[idx])
            ax[0].set_xlabel(fr'var{level}')
            ax[0].set_ylabel(fr'adjusted var{level}')
            
            ax[1].plot(var, var_hr, label=method, linestyle=linestyles[idx])
            ax[1].set_xlabel(fr'var{level}')
            ax[1].set_ylabel(fr'haircut pct')
            ax[1].legend()
            ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())
            fig.suptitle(fr'VaR adjustment: {num_test} tests, $\rho_c$ = {crosssec_rho}, $\rho_a$ = {autocorrelation}', fontsize=14)
            
        plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
        plt.savefig(os.path.join(plot_dir, f'VaR({level})_{num_test}.png'))
        plt.show()
        plt.clf()