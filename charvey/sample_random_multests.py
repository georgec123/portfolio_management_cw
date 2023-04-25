import numpy as np


def get_gamma_params(scale: float, std_mult: float):
    """
    Get the parameters of a gamma distribution with a given scale.
    Variance of the parameters returned will be vat_mult*scale**2.

    k, theta are shape, scale (https://en.wikipedia.org/wiki/Gamma_distribution)
    """

    return 1 / std_mult**2, scale * std_mult**2


def sample_random_multests(rho, m_tot, p_0, lambd, M_simu, autocorrelation=None, dist="exponential", std_scale=1):
    # Parameter input from Harvey, Liu and Zhu (2014)
    # Default: para_vec = [0.2, 1377, 4.4589*0.1, 5.5508*0.001,M_simu]

    # p_0 probability for a random factor to have a zero mean
    # lambd  # average of monthly mean returns for true strategies
    # m_tot  # total number of trials
    # rho  # average cross-correlation among returns
    # M_simu  # number of rows (simulations)

    if dist == "exponential":
        mean_vec = np.random.exponential(lambd, size=(M_simu, m_tot))
    elif dist == "gamma":
        assert std_scale is not None, "std_scale must be specified for gamma distribution"
        k, theta = get_gamma_params(lambd, std_scale)
        mean_vec = np.random.gamma(shape=k, scale=theta, size=(M_simu, m_tot))
    else:
        raise ValueError("dist must be either exponential or gamma")

    sigma = 0.15 / np.sqrt(12)  # assumed level of monthly vol
    N = 240  # number of time-series

    SIGMA = np.full((m_tot, m_tot), rho)
    np.fill_diagonal(SIGMA, 1)
    MU = np.zeros(m_tot)

    shock_mat = np.random.multivariate_normal(MU, SIGMA * (sigma**2 / N), M_simu)
    prob_vec = np.random.uniform(0, 1, size=(M_simu, m_tot))

    m_indi = prob_vec > p_0
    mu_nul = m_indi * mean_vec  # Null-hypothesis
    tstat_mat = np.abs(mu_nul + shock_mat) / (sigma / np.sqrt(N))

    if autocorrelation is not None:
        tstat_mat *= (
            1 + (2 * autocorrelation / (1 - autocorrelation)) * (1 - ((1 - autocorrelation) / (1 - autocorrelation)))
        ) ** (-0.5)

    return tstat_mat


if __name__ == "__main__":
    tstat_mat = sample_random_multests(0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001, 1000)
    print(tstat_mat)
