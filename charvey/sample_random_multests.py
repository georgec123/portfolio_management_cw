import numpy as np

def sample_random_multests(rho, m_tot, p_0, lambd, M_simu):
    # Parameter input from Harvey, Liu and Zhu (2014)
    # Default: para_vec = [0.2, 1377, 4.4589*0.1, 5.5508*0.001,M_simu]
    
    # p_0 probability for a random factor to have a zero mean   
    # lambd  # average of monthly mean returns for true strategies
    # m_tot  # total number of trials
    # rho  # average correlation among returns
    # M_simu  # number of rows (simulations)

    sigma = 0.15/np.sqrt(12)  # assumed level of monthly vol
    N = 240  # number of time-series
    
    sig_vec = np.concatenate(([1], rho*np.ones(m_tot-1)))
    SIGMA = np.vstack([np.roll(sig_vec, i) for i in range(m_tot)])
    MU = np.zeros(m_tot)
    shock_mat = np.random.multivariate_normal(MU, SIGMA*(sigma**2/N), M_simu)
    
    prob_vec = np.random.uniform(0, 1, size=(M_simu, m_tot))
    mean_vec = np.random.exponential(lambd, size=(M_simu, m_tot))
    m_indi = prob_vec > p_0
    mu_nul = m_indi*mean_vec  # Null-hypothesis
    tstat_mat = np.abs(mu_nul + shock_mat)/(sigma/np.sqrt(N))

    return tstat_mat


if __name__ == '__main__':

    tstat_mat = sample_random_multests(0.2, 1377, 4.4589*0.1, 5.5508*0.001, 1000)
    print(tstat_mat)