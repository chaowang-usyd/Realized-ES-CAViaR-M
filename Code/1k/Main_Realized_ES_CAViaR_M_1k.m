function Main_Realized_ES_CAViaR_M_1k

% This is the main function to estimate the Realized-ES-CAViaR-M with 1
% realized measure and generate the corresponding VaR and ES forecasts,
% using the S&P 500 data.

parpool('local',16);

load SnPCCReturn;
% load SnP_RkP;
% load SnP_Rv5;
load SnP_BV;

r_data = SnPCCReturn;
Rt = r_data;
Xt = [100*sqrt(SnP_BV)]; % different realized measures can be used, such as RV, BV and RK
RealizedMeasures_all = Xt;
n_fore= 2626; 

bStart = 1; % bStart and bEnd can be changed to have multiple Matlab main files to run in parrallel, e.g., main file 1 to run forecasting steps 1-400, main file 2 to run forecasting steps 401-800
bEnd = n_fore; % in the paper 6 parallel runs have been used

[n_overall,k] = size(Xt);

n_fit= n_overall-n_fore;
return_fore= r_data((n_overall-n_fore+1):n_overall); 

NN = 12;

params_estimates= zeros(n_fore,NN); % save all the parameters estimated by MCMC
VaR_Fore= zeros(n_fore,1); % save all the VaR forecasts
ES_Fore= zeros(n_fore,1); % save all the ES forecasts

quantile_level= 0.025; % alpha= 2.5%

% quantile_level= 0.01; % alpha= 1%

% New blocking

% Block dependent configuration
n_block= 4; 
SigProp0 = cell(n_block, 1);
scale0 = zeros(n_block, 1);
targAcc = zeros(n_block, 1);
accTol = zeros(n_block, 1);

block = cell(n_block,1);
block{1} = [1 2 3 4];
block{2} = [5 8 9];
block{3} = [10 11];
block{4} = [6 7 12];

n_imh= 5000;                                                                                                                                  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nTune = 200; % tuning
nIterAd_1 = 20000;
nDiscardAd = 2000; % discard from covariance calc
minAdapt = 2; % min no. regimes
maxAdapt = 10; % max 
mapcTol = 0.1; % stop adaptive, end of burn-in (variable)
nIter= 10000; % the number of iterations in the final epoch

w = [0.7, 0.15, 0.15];
s = [1, 100, 0.01];

rng(0);

for i = 1:n_block
    nDim = numel(block{i});
    SigProp0{i} = eye(nDim);
    scale0(i) = 2.38 ./ sqrt(nDim);
    targAcc(i) = targaccrate(nDim);
    accTol(i) = 0.075;
end

chainIsBackUp= cell(n_fore,1);

theta0Ad = [0.014,0.98,0.13,0.1,0.16,-0.82,0.95,0.09,0.27,0.23,0.33,0.18]/3;

parfor i_fore = bStart:bEnd   

    start_point = i_fore;
    end_point= i_fore + n_fit - 1;
    data_input= r_data(start_point:end_point)- mean(r_data(start_point:end_point));

    realized_data_input = Xt((start_point:end_point),:);

    kernel = @(theta)LH_1k_int(theta,quantile_level,data_input,realized_data_input);

    [ThetaAd, Scale, SigProp, AcceptAd, mapcAd] = gmrwmetropadapt( ...
        kernel, theta0Ad, block, scale0, SigProp0, w, s, ...
        targAcc, accTol, nTune, nIterAd_1, nDiscardAd, ...
        minAdapt, maxAdapt, mapcTol);

    theta0 = mean(ThetaAd, 1);
    aveScl = mean(Scale, 1);
    for i = 1:n_block
        SigProp{i} = (aveScl(i) .^ 2) .* SigProp{i};
    end
    [Theta, Accept] = gmrwmetrop( ...
        kernel, theta0, block, SigProp, w, s, nIter);

    ChainIs= Theta;
    chainIsBackUp{i_fore} =  ChainIs;

    params_estimates(i_fore,:) = mean(ChainIs);

    [var_fore,es_fore] = VaRES_1k_BaY_int(ChainIs, data_input,realized_data_input,n_imh,n_fit,quantile_level);
    VaR_Fore(i_fore)= mean(var_fore);
    ES_Fore(i_fore)= mean(es_fore);

    disp([i, VaR_Fore(i_fore), ES_Fore(i_fore)]);
end   

save Main_Realized_ES_CAViaR_M_1k_BV.mat

end