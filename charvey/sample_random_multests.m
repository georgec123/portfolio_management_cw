%%% Generate empirical p-value distributions based on Harvey, Liu and Zhu (2014) ------ Harvey and Liu
%%% (2014): "Backtesting", Duke University 

function res = sample_random_multests(rho, m_tot, p_0, lambda, M_simu) 

%%%Parameter input from Harvey, Liu and Zhu (2014) %%%%%%%%%%%%
%%%Default: para_vec = [0.2, 1377, 4.4589*0.1, 5.5508*0.001,M_simu]%%%%%%%%%%%

p_0 = p_0 ;  % probability for a random factor to have a zero mean   
lambda = lambda; % average of monthly mean returns for true strategies
m_tot = m_tot; % total number of trials
rho = rho; % average correlation among returns

M_simu = M_simu;  % number of rows (simulations) 

sigma = 0.15/sqrt(12); % assumed level of monthly vol
N = 240;     %number of time-series%

sig_vec = [1, rho*ones(1,m_tot-1)];
SIGMA = toeplitz(sig_vec);
MU = zeros(1,m_tot);
shock_mat = mvnrnd(MU, SIGMA*(sigma^2/N), M_simu);
    
prob_vec = unifrnd(0,1,[M_simu,m_tot]);
mean_vec = exprnd(lambda, [M_simu,m_tot]);
m_indi = prob_vec > p_0;
mu_nul = m_indi.*mean_vec;      %Null-hypothesis%
tstat_mat = abs(mu_nul + shock_mat)/(sigma/sqrt(N));

res = tstat_mat; 