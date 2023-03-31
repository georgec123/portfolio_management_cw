%%% Sharpe ratio adjustment due to testing multiplicity ------ Harvey and Liu
%%% (2014): "Backtesting", Duke University 

function res = Haircut_SR(sm_fre, num_obs, SR, ind_an, ind_aut, rho, num_test, RHO) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Parameter inputs %%%%%%
 
%%% 'sm_fre': Sampling frequency; [1,2,3,4,5] = [Daily, Weekly, Monthly, Quarterly, Annual]; 
%%% 'num_obs': No. of observations in the frequency specified in the previous step; 
%%% 'SR': Sharpe ratio; either annualized or in the frequency specified in the previous step;
%%% 'ind_an': Indicator; if annulized, 'ind_an' = 1; otherwise = 0; 
%%% 'ind_aut': Indicator; if adjusted for autocorrelations, 'ind_aut' = 0; otherwise = 1;
%%% 'rho': Autocorrelation coefficient at the specified frequency ;
%%% 'num_test': Number of tests allowed, Harvey, Liu and Zhu (2014) find 315 factors;
%%% 'RHO': Average correlation among contemporaneous strategy returns.
 
%%% Calculating the equivalent annualized Sharpe ratio 'sr_annual', after 
%%% taking autocorrlation into account 
if sm_fre == 1,
    fre_out = 'Daily';
elseif sm_fre == 2, 
    fre_out = 'Weekly';
elseif sm_fre == 3, 
    fre_out = 'Monthly';
elseif sm_fre == 4, 
    fre_out = 'Quarterly';
else
    fre_out = 'Annual';
end 

if ind_an == 1, 
    sr_out = 'Yes';
else
    sr_out = 'No';
end



if ind_an == 1 && ind_aut == 0, 
    sr_annual = SR;
elseif ind_an ==1 && ind_aut == 1, 
    if sm_fre ==1, 
    sr_annual = SR*[1 + (2*rho/(1-rho))*(1- ((1-rho^(360))/(360*(1-rho))))]^(-0.5);
    elseif sm_fre ==2,
    sr_annual = SR*[1 + (2*rho/(1-rho))*(1- ((1-rho^(52))/(52*(1-rho))))]^(-0.5);
    elseif sm_fre ==3,
    sr_annual = SR*[1 + (2*rho/(1-rho))*(1- ((1-rho^(12))/(12*(1-rho))))]^(-0.5);
    elseif sm_fre ==4,
    sr_annual = SR*[1 + (2*rho/(1-rho))*(1- ((1-rho^(4))/(4*(1-rho))))]^(-0.5);
    elseif sm_fre ==5,
    sr_annual = SR; 
    end
elseif ind_an == 0 && ind_aut == 0,
     if sm_fre ==1, 
    sr_annual = SR*sqrt(360);
    elseif sm_fre ==2,
    sr_annual = SR*sqrt(52);
    elseif sm_fre ==3,
    sr_annual = SR*sqrt(12);
    elseif sm_fre ==4,
    sr_annual = SR*sqrt(4);
    elseif sm_fre ==5,
    sr_annual = SR; 
     end
 elseif ind_an == 0 && ind_aut == 1,
     if sm_fre ==1, 
    sr_annual = sqrt(360)*sr*[1 + (2*rho/(1-rho))*(1- ((1-rho^(360))/(360*(1-rho))))]^(-0.5);
    elseif sm_fre ==2,
    sr_annual = sqrt(52)*sr*[1 + (2*rho/(1-rho))*(1- ((1-rho^(52))/(52*(1-rho))))]^(-0.5);
    elseif sm_fre ==3,
    sr_annual = sqrt(12)*sr*[1 + (2*rho/(1-rho))*(1- ((1-rho^(12))/(12*(1-rho))))]^(-0.5);
    elseif sm_fre ==4,
    sr_annual = sqrt(4)*sr*[1 + (2*rho/(1-rho))*(1- ((1-rho^(4))/(4*(1-rho))))]^(-0.5);
    elseif sm_fre ==5,
    sr_annual = SR; 
     end 
end

%%% Number of monthly observations 'N' %%%

if sm_fre ==1, 
    N = floor(num_obs*12/360);
elseif sm_fre ==2,
    N = floor(num_obs*12/52);
elseif sm_fre == 3,
    N = floor(num_obs*12/12);
elseif sm_fre == 4,
    N = floor(num_obs*12/4);
elseif sm_fre == 5,
    N = floor(num_obs*12/1);
end

%%% Number of tests allowed %%%                              
M = num_test; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Intermediate outputs %%%%%%%%%%
fprintf('Inputs:\n');
fprintf('Frequency = %s;\n', fre_out);
fprintf('Number of Observations = %d;\n', num_obs);
fprintf('Initial Sharpe Ratio = %.3f;\n', SR);
fprintf('Sharpe Ratio Annualized = %s;\n', sr_out);
fprintf('Autocorrelation = %.3f;\n', rho);
fprintf('A/C Corrected Annualized Sharpe Ratio = %.3f\n', sr_annual);
fprintf('Assumed Number of Tests = %d;\n', M);
fprintf('Assumed Average Correlation = %.3f.\n\n', RHO);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Sharpe ratio adjustment %%%%%%%%%

m_vec = 1:(M+1);
c_const = sum(1./m_vec);

%%%Input for Holm and BHY %%%%
%%%Parameter input from Harvey, Liu and Zhu (2014) %%%%%%%
para0 = [0, 1295, 3.9660*0.1, 5.4995*0.001; 
       0.2, 1377, 4.4589*0.1, 5.5508*0.001;
       0.4, 1476, 4.8604*0.1, 5.5413*0.001;
       0.6, 1773, 5.9902*0.1, 5.5512*0.001;
       0.8, 3109, 8.3901*0.1, 5.5956*0.001];
       
%%%Interpolated parameter values based on user specified level of corrlation RHO %%%%%%%%%%   
if (RHO >= 0)&&(RHO < 0.2), 
    para_inter = ((0.2 - RHO)/0.2)*para0(1,:) + ((RHO - 0)/0.2)*para0(2,:); 
elseif (RHO >= 0.2)&&(RHO < 0.4), 
    para_inter = ((0.4 - RHO)/0.2)*para0(2,:) + ((RHO - 0.2)/0.2)*para0(3,:);
elseif (RHO >= 0.4)&&(RHO < 0.6), 
    para_inter = ((0.6 - RHO)/0.2)*para0(3,:) + ((RHO - 0.4)/0.2)*para0(4,:);
elseif (RHO >= 0.6)&&(RHO < 0.8), 
    para_inter = ((0.8 - RHO)/0.2)*para0(4,:) + ((RHO - 0.6)/0.2)*para0(5,:);
elseif (RHO >= 0.8)&&(RHO < 1.0), 
    para_inter = ((0.8 - RHO)/0.2)*para0(4,:) + ((RHO - 0.6)/0.2)*para0(5,:);
else
    para_inter = para0(2,:);   %%% Set at the preferred level if RHO is misspecified 
end

WW = 2000;  %%% Number of repetitions %%%%

%%%Generate a panel of t-ratios (WW*Nsim_tests)%%%%%% 
Nsim_tests = (floor(M/para_inter(2)) + 1)*floor(para_inter(2)+1); % make sure Nsim_test >= M
t_sample = sample_random_multests(para_inter(1), Nsim_tests, para_inter(3), para_inter(4), WW); 

% Sharpe ratio, monthly 
sr = sr_annual/sqrt(12);
T = sr*sqrt(N);
p_val = 2*(1- tcdf(T, N-1));

% Drawing observations from the underlying p-value distribution; simulate a
% large number (WW) of p-value samples 
p_holm = ones(WW,1);
p_bhy = ones(WW,1);

for ww = 1: WW,

  yy =  t_sample(ww, 1:M); 
  t_value = yy'; 
  
  p_val_sub= 2*(1- normcdf(t_value,0,1));
  
  %%% Holm %%%%
  p_val_all =  [p_val_sub', p_val];
  p_val_order = sort(p_val_all);
  p_holm_vec = [];
  
  for i = 1:(M+1),
      p_new = [];
      for j = 1:i,
          p_new = [p_new, (M+1-j+1)*p_val_order(j)];
      end
      p_holm_vec = [p_holm_vec, min(max(p_new),1)];
  end
  
  p_sub_holm = p_holm_vec(p_val_order == p_val);
  p_holm(ww) = p_sub_holm(1);
  
  %%%%% BHY %%%%%%%
  p_bhy_vec = [];

  for i = 1:(M+1),
      kk = (M+1) - (i-1);
      if kk == (M+1),
          p_new = p_val_order(end);
      else
          p_new = min( ((M+1)*c_const/kk)*p_val_order(kk), p_0);
      end    
      p_bhy_vec = [p_new, p_bhy_vec];
      p_0 = p_new;
  end
  
  p_sub_bhy = p_bhy_vec(p_val_order == p_val);
  p_bhy(ww) = p_sub_bhy(1);
  
end

%%% Bonferroni %%%
p_BON = min(M*p_val,1);
%%% Holm %%%
p_HOL = median(p_holm);
%%% BHY %%%
p_BHY = median(p_bhy);
%%% Average %%%
p_avg = (p_BON + p_HOL + p_BHY)/3;

% Invert to get z-score   
z_BON = tinv(1- p_BON/2, N-1);
z_HOL = tinv(1- p_HOL/2, N-1);
z_BHY = tinv(1- p_BHY/2, N-1);
z_avg = tinv(1- p_avg/2, N-1);

% Annualized Sharpe ratio
sr_BON = (z_BON/sqrt(N))*sqrt(12);
sr_HOL = (z_HOL/sqrt(N))*sqrt(12);
sr_BHY = (z_BHY/sqrt(N))*sqrt(12);
sr_avg = (z_avg/sqrt(N))*sqrt(12);

% Calculate haircut
hc_BON = (sr_annual - sr_BON)/sr_annual; 
hc_HOL = (sr_annual - sr_HOL)/sr_annual;
hc_BHY = (sr_annual - sr_BHY)/sr_annual;
hc_avg = (sr_annual - sr_avg)/sr_annual;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Final Output %%%%%%%%%%%
fprintf('Outputs:\n');
fprintf('Bonferroni Adjustment:\n');
fprintf('Adjusted P-value = %.3f;\n', p_BON);
fprintf('Haircut Sharpe Ratio = %.3f;\n', sr_BON);
fprintf('Percentage Haircut = %.1f%%.\n\n', hc_BON*100);

fprintf('Holm Adjustment:\n');
fprintf('Adjusted P-value = %.3f;\n', p_HOL);
fprintf('Haircut Sharpe Ratio = %.3f;\n', sr_HOL);
fprintf('Percentage Haircut = %.1f%%.\n\n', hc_HOL*100);

fprintf('BHY Adjustment:\n');
fprintf('Adjusted P-value = %.3f;\n', p_BHY);
fprintf('Haircut Sharpe Ratio = %.3f;\n', sr_BHY);
fprintf('Percentage Haircut = %.1f%%.\n\n', hc_BHY*100);

fprintf('Average Adjustment:\n');
fprintf('Adjusted P-value = %.3f;\n', p_avg);
fprintf('Haircut Sharpe Ratio = %.3f;\n', sr_avg);
fprintf('Percentage Haircut = %.1f%%.\n\n', hc_avg*100);

