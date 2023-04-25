from sample_random_multests import sample_random_multests

import numpy as np
from scipy.stats import norm


def Profit_Hurdle(num_tests, num_obs, alpha_sig, vol_annual, RHO):
    # Parameter inputs
    NN = num_tests
    Obs = num_obs
    alpha0 = alpha_sig
    vol_anu = vol_annual

    # Independent test
    B_ind = norm.ppf(1 - alpha0 / 2, 0, 1)

    # Bonferroni
    p0_mat = alpha0 / NN
    t0_mat = norm.ppf(1 - p0_mat / 2, 0, 1)
    BF = t0_mat

    # Input for Holm and BHY
    # Parameter input from Harvey, Liu and Zhu (2014)
    para0 = np.array(
        [
            [0, 1295, 3.9660 * 0.1, 5.4995 * 0.001],
            [0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001],
            [0.4, 1476, 4.8604 * 0.1, 5.5413 * 0.001],
            [0.6, 1773, 5.9902 * 0.1, 5.5512 * 0.001],
            [0.8, 3109, 8.3901 * 0.1, 5.5956 * 0.001],
        ]
    )
    # Interpolated parameter values based on user specified level of correlation RHO
    if 0 <= RHO < 0.2:
        para_inter = ((0.2 - RHO) / 0.2) * para0[0, :] + ((RHO - 0) / 0.2) * para0[1, :]
    elif 0.2 <= RHO < 0.4:
        para_inter = ((0.4 - RHO) / 0.2) * para0[1, :] + ((RHO - 0.2) / 0.2) * para0[2, :]
    elif 0.4 <= RHO < 0.6:
        para_inter = ((0.6 - RHO) / 0.2) * para0[2, :] + ((RHO - 0.4) / 0.2) * para0[3, :]
    elif 0.6 <= RHO < 0.8:
        para_inter = ((0.8 - RHO) / 0.2) * para0[3, :] + ((RHO - 0.6) / 0.2) * para0[4, :]
    elif 0.8 <= RHO < 1.0:
        para_inter = ((0.8 - RHO) / 0.2) * para0[3, :] + ((RHO - 0.6) / 0.2) * para0[4, :]
    else:
        para_inter = para0[1, :]  # Set at the preferred level if RHO is misspecified

    WW = 2000  # Number of repetitions

    # Generate a panel of t-ratios (WW*Nsim_tests)
    Nsim_tests = (int(NN / para_inter[1]) + 1) * int(para_inter[1] + 1)  # make sure Nsim_test >= num_tests
    t_sample = sample_random_multests(para_inter[0], Nsim_tests, para_inter[2], para_inter[3], WW)

    # Holm
    HL_mat = []

    for ww in range(1, WW + 1):
        yy = t_sample[ww - 1, 0:NN]

        p_sub = 2 * (1 - norm.cdf(yy))
        p_new = np.sort(p_sub)

        KK = len(p_new)
        comp_vec = []

        for kk in range(1, KK + 1):
            comp_vec.append(alpha0 / (KK + 1 - kk))

        comp_res = p_new > comp_vec
        comp_new = np.cumsum(comp_res)

        if np.sum(comp_new) == 0:
            HL = 1.96
        else:
            p0 = p_new[comp_new == 1]
            HL = norm.ppf((1 - p0 / 2), 0, 1)[0]

        HL_mat.append(HL)

    # BHY
    BHY_mat = []

    for ww in range(1, WW + 1):
        yy = t_sample[ww - 1, 0:NN]  # Use the ww'th row of t-sample

        p_sub = 2 * (1 - norm.cdf(yy))

        if len(p_new) <= 1:
            BH00 = 1.96
        else:
            p_new11 = np.sort(p_sub)[::-1]

            KK = len(p_new11)
            comp_vec0 = []
            cons_vec = np.arange(1, KK + 1)
            cons_norm = np.sum(1.0 / cons_vec)

            for kk in range(1, KK + 1):
                comp_vec0.append((alpha0 * kk) / (KK * cons_norm))

            comp_vec = np.sort(comp_vec0)[::-1]

            comp_res11 = p_new11 <= comp_vec

            if np.sum(comp_res11) == 0:
                BH00 = 1.96
            else:
                p0 = p_new11[comp_res11]

                b0 = np.argmin(np.abs(p_new11 - p0[0]))

                if b0 == 0:
                    p1 = p0[0]
                else:
                    p1 = p_new11[b0 - 1]

                BH00 = norm.ppf((1 - (p0[0] + p1) / 4), 0, 1)

        BHY_mat.append(BH00)

    tcut_vec = [B_ind, BF, np.median(HL_mat), np.median(BHY_mat)]

    ret_hur = ((vol_anu / np.sqrt(12)) / np.sqrt(Obs)) * np.array(tcut_vec)

    print("Inputs:")
    print("Significance Level = {:.1%};".format(alpha0))
    print("Number of Observations = {};\n".format(num_obs))
    print("Annualized Return Volatility = {:.1%};".format(vol_anu))
    print("Assumed Number of Tests = {};\n".format(NN))
    print("Assumed Average Correlation = {:.3f}.\n".format(RHO))

    print("Outputs:")
    print("Minimum Average Monthly Return:")
    print("Independent = {:.3%};".format(ret_hur[0]))
    print("Bonferroni = {:.3%};".format(ret_hur[1]))
    print("Holm = {:.3%};".format(ret_hur[2]))
    print("BHY = {:.3%};".format(ret_hur[3]))
    print("Average for Multiple Tests = {:.3%}.".format(np.mean(ret_hur[1:])))


if __name__ == "__main__":
    Profit_Hurdle(300, 240, 0.05, 0.1, 0.4)
