import os

import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.stats import norm, t
from sample_random_multests import sample_random_multests

np.random.seed(3)


def get_params(RHO):
    # Parameter input from Harvey, Liu and Zhu (2014)

    # rho, m_tot, p_0, lambd
    para0 = np.array([[0, 1295, 3.9660 * 0.1, 5.4995 * 0.001],
                      [0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001],
                      [0.4, 1476, 4.8604 * 0.1, 5.5413 * 0.001],
                      [0.6, 1773, 5.9902 * 0.1, 5.5512 * 0.001],
                      [0.8, 3109, 8.3901 * 0.1, 5.5956 * 0.001]])

    # Interpolated parameter values based on user specified level of correlation RHO
    if (RHO >= 0) and (RHO < 0.2):
        para_inter = ((0.2 - RHO) / 0.2) * \
            para0[0, :] + ((RHO - 0) / 0.2) * para0[1, :]
    elif (RHO >= 0.2) and (RHO < 0.4):
        para_inter = ((0.4 - RHO) / 0.2) * \
            para0[1, :] + ((RHO - 0.2) / 0.2) * para0[2, :]
    elif (RHO >= 0.4) and (RHO < 0.6):
        para_inter = ((0.6 - RHO) / 0.2) * \
            para0[2, :] + ((RHO - 0.4) / 0.2) * para0[3, :]
    elif (RHO >= 0.6) and (RHO < 0.8):
        para_inter = ((0.8 - RHO) / 0.2) * \
            para0[3, :] + ((RHO - 0.6) / 0.2) * para0[4, :]
    elif (RHO >= 0.8) and (RHO < 1.0):
        para_inter = ((0.8 - RHO) / 0.2) * \
            para0[3, :] + ((RHO - 0.6) / 0.2) * para0[4, :]
    else:
        para_inter = para0[1, :]

    return para_inter


def Haircut_SR(sm_fre, num_obs, SR, ind_an, ind_aut, rho, num_test, RHO, log=False):
    # 'sm_fre': Sampling frequency; [1,2,3,4,5] = [Daily, Weekly, Monthly, Quarterly, Annual];
    # 'num_obs': No. of observations in the frequency specified in the previous step;
    # 'SR': Sharpe ratio; either annualized or in the frequency specified in the previous step;
    # 'ind_an': Indicator; if annulized, 'ind_an' = 1; otherwise = 0;
    # 'ind_aut': Indicator; if adjusted for autocorrelations, 'ind_aut' = 0; otherwise = 1;
    # 'rho': Autocorrelation coefficient at the specified frequency;
    # 'num_test': Number of tests allowed, Harvey, Liu and Zhu (2014) find 315 factors;
    # 'RHO': Average correlation among contemporaneous strategy returns.

    # Calculating the equivalent annualized Sharpe ratio 'sr_annual', after taking autocorrlation into account
    sm_fre_fre_out_dict = {1: 'Daily', 2: 'Weekly',
                           3: 'Monthly', 4: 'Quarterly', 5: 'Annual'}
    sm_fre_days_dict = {1: 360, 2: 52, 3: 12, 4: 4, 5: 1}

    fre_out = sm_fre_fre_out_dict[sm_fre]

    sr_out = 'Yes' if ind_an == 1 else 'No'

    def fn_sr_annual(rho, sr, fre): return sr*(1 + (2*rho/(1-rho))
                                               * (1 - ((1-rho**(fre))/(fre*(1-rho)))))**(-0.5)

    # Calculate annualised sharpe ratio based on input indicators
    if ind_an == 1 and ind_aut == 0:
        sr_annual = SR
    elif ind_an == 1 and ind_aut == 1:
        sr_annual = fn_sr_annual(rho, SR, sm_fre_days_dict[sm_fre])
    elif ind_an == 0 and ind_aut == 0:
        sr_annual = SR * math.sqrt(sm_fre_days_dict[sm_fre])
    elif ind_an == 0 and ind_aut == 1:
        sr_annual = fn_sr_annual(
            rho, SR, sm_fre_days_dict[sm_fre]) * math.sqrt(sm_fre_days_dict[sm_fre])

    # Number of monthly observations 'N'
    N = np.floor(num_obs * 12 / sm_fre_days_dict[sm_fre])

    # Number of tests allowed
    M = num_test

    # Intermediate outputs
    if log:
        print('Inputs:')
        print(f'Frequency = {fre_out};')
        print(f'Number of Observations = {num_obs};')
        print(f'Initial Sharpe Ratio = {SR:.3f};')
        print(f'Sharpe Ratio Annualized = {sr_out};')
        print(f'Autocorrelation = {rho:.3f};')
        print(f'A/C Corrected Annualized Sharpe Ratio = {sr_annual:.3f}')
        print(f'Assumed Number of Tests = {M};')
        print(f'Assumed Average Correlation = {RHO:.3f}.\n')

    # Sharpe ratio adjustment
    m_vec = np.arange(1, M + 2)
    c_const = np.sum(1 / m_vec)

    # Input for Holm and BHY
    para_inter = get_params(RHO)

    WW = 2000  # Number of repetitions

    # Generate a panel of t-ratios (WW*Nsim_tests)
    # make sure Nsim_test >= M
    Nsim_tests = int(
        (np.floor(M / para_inter[1]) + 1) * np.floor(para_inter[1] + 1))
    t_sample = sample_random_multests(
        para_inter[0], Nsim_tests, para_inter[2], para_inter[3], WW)

    # Sharpe ratio, monthly
    sr = sr_annual / np.sqrt(12)
    T = sr * np.sqrt(N)
    p_val = 2 * (1 - t.cdf(T, N - 1))

    # Drawing observations from the underlying p-value distribution; simulate a
    # large number (WW) of p-value samples
    p_holm = np.ones(WW)
    p_bhy = np.ones(WW)

    for ww in range(WW):
        if ww % 100 == 0:
            print(f'Iteration {ww} of {WW}.', end='\r')

        yy = t_sample[ww, :M]
        t_value = yy.reshape(-1, 1)

        p_val_sub = 2 * (1 - norm.cdf(t_value, 0, 1))

        # Holm
        p_val_all = np.concatenate((p_val_sub.T[0], [p_val]))
        p_val_order = np.sort(p_val_all)

        p_holm_vec = np.zeros(M + 1)
        cur_max = 0

        for j in range(1, M + 2):
            cur_max = max(cur_max, p_val_order[j - 1] * (M + 1 - j + 1))
            p_holm_vec[j - 1] = min(cur_max, 1)

        # retreive our original p-value (now adjusted)
        p_sub_holm = p_holm_vec[p_val_order == p_val]
        p_holm[ww] = p_sub_holm[0]

        # BHY
        p_bhy_vec = np.zeros(M + 1)
        p_0 = 1

        for i in range(1, M + 2):
            kk = (M + 1) - (i - 1)
            if kk == (M + 1):
                p_new = p_val_order[-1]
            else:
                p_new = np.min(
                    [((M + 1) * c_const / kk) * p_val_order[kk - 1], p_0])

            p_bhy_vec[kk - 1] = p_new
            p_0 = p_new

        p_sub_bhy = p_bhy_vec[p_val_order == p_val]
        p_bhy[ww] = p_sub_bhy[0]

    print('\n')

    p_BON = min(M * p_val, 1)  # Bonferroni
    p_HOL = np.median(p_holm)  # Holm
    p_BHY = np.median(p_bhy)  # BHY
    p_avg = (p_BON + p_HOL + p_BHY) / 3  # Average

    p_arr = np.array([p_BON, p_HOL, p_BHY, p_avg])
    # Invert to get z-score
    z_arr = t.ppf(1 - p_arr / 2, N - 1)
    # Annualized Sharpe ratio
    sr_arr = z_arr / np.sqrt(N) * np.sqrt(12)
    # Calculate haircut
    haircut_arr = (sr_annual - sr_arr) / sr_annual

    if log:
        for idx, method in enumerate(['Bonferroni', 'Holm', 'BHY', 'Average']):
            print(f'{method} Adjustment:')
            print(f'Adjusted P-value = {p_arr[idx]:.3f};')
            print(f'Haircut Sharpe Ratio = {sr_arr[idx]:.3f};')
            print(f'Percentage Haircut = {haircut_arr[idx]*100:.1f}%.\n')

    return p_arr, sr_arr, haircut_arr


def replicate_paper_plots():

    for num_test in [10, 50, 200]:
        srs = np.linspace(0.2, 1.1, 16)
        results = []

        for idx, sr in enumerate(srs):
            print(f'Running {idx} of {len(srs)}')
            results.append(
                Haircut_SR(sm_fre=3,
                           num_obs=240,
                           SR=sr,
                           ind_an=1,
                           ind_aut=1,
                           rho=0, num_test=num_test,
                           RHO=0)
            )
        results = np.array(results)

        outputs = ['P-value', 'Haircut Sharpe Ratio', 'Haircut pct']
        methods = ['Bon', 'Holm', 'BHY']

        for output_idx, output in enumerate(outputs):

            fig, ax = plt.subplots()
            for idx, method in enumerate(methods):
                ax.plot(srs, results[:, output_idx, idx], label=method)

            if output == 'Haircut pct':
                ax.hlines(0.5, 0, srs[-1], color='black', alpha=0.5)
            elif output == 'Haircut Sharpe Ratio':
                ax.plot([0, srs[-1]],[0,srs[-1]], color='black', alpha=0.5)
                ax.plot([0, srs[-1]],[0,srs[-1]/2], color='black', alpha=0.5)

            
            ax.set_xlabel('Sharpe Ratio')
            ax.set_ylabel(output)
            ax.set_title(f'{num_test} tests.')
            ax.legend()
            ax.set_xlim(xmin=0)
            ax.set_ylim(ymin=0)

            plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

            plt.savefig(os.path.join(plot_dir, f'{output}_{num_test}.png'))
            plt.clf()


if __name__ == '__main__':
    # Haircut_SR(sm_fre=3, num_obs=120,SR=1,ind_an=1,ind_aut=1,rho=0.1, num_test=100,RHO=0.4)
    replicate_paper_plots()
