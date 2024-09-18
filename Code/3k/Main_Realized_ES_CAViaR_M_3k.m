function Main_Realized_ES_CAViaR_M_3k

% This is the main function to estimate the Realized-ES-CAViaR-M with 3
% realized measures and generate the corresponding VaR and ES forecasts,
% using the S&P 500 data.

parpool('local', 24);

load SnPCCReturn;
load SnP_Rv5;
load SnP_RkP;
load SnP_BV;

r_data = SnPCCReturn;
Rt = r_data;
Xt = [100*sqrt(SnP_Rv5) 100*sqrt(SnP_RkP) 100*sqrt(SnP_BV)];
RealizedMeasures_all = Xt;
n_fore= 2626; 

bStart = 1; % bStart and bEnd can be changed to have multiple Matlab main files to run in parrallel, e.g., main file 1 to run forecasting steps 1-400, main file 2 to run forecasting steps 401-800
bEnd = n_fore; % in the paper 6 parallel runs have been used

[n_overall,k] = size(Xt);

n_fit= n_overall-n_fore;
return_fore= r_data((n_overall-n_fore+1):n_overall); 

NN = 24;

params_estimates= zeros(n_fore,NN); % save all the parameters estimated by MCMC
VaR_Fore= zeros(n_fore,1); % save all the VaR forecasts
ES_Fore= zeros(n_fore,1); % save all the ES forecasts

% quantile_level= 0.01; % alpha= 1%

quantile_level= 0.025; % alpha= 2.5%

% Block dependent configuration
n_block= 8;
SigProp0 = cell(n_block, 1);
scale0 = zeros(n_block, 1);
targAcc = zeros(n_block, 1);
accTol = zeros(n_block, 1);

block = cell(n_block,1);
block{1} = [1 2 3 4];block{2} = [5 6 7 ]; 
block{3} = [8 9 10];block{4} = [11 12 13]; 
block{5} = [14 15 16];block{6} = [17 18 19];
block{7} = [20 21];block{8} = [22 23 24];

n_imh= 5000;                                                                                                                                

nTune = 200; % tuning
nIterAd_1 = 20000;
nDiscardAd = 2000; % discard from covariance calc
minAdapt = 2; % min no. regimes
maxAdapt = 10; % max " "
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

theta0Ad = [0.01,0.98,0.13,0.1,-0.02,0.03,0.2,-0.84,-0.86,-0.99,0.95,0.98,0.98,0.09,0.08,0.13,0.275,0.43,0.19,0.2,0.2,0.15,0.06,0.24]/3;

parfor i_fore = bStart:bEnd   

    start_point = i_fore;
    end_point= i_fore + n_fit - 1;
    data_input= r_data(start_point:end_point)- mean(r_data(start_point:end_point));

    realized_data_input = Xt((start_point:end_point),:);

    kernel = @(theta)LH_3k_int(theta,quantile_level,data_input,realized_data_input);

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

    [var_fore,es_fore] = VaRES_3k_BaY_int(ChainIs, data_input,realized_data_input,n_imh,n_fit,quantile_level);
    VaR_Fore(i_fore)= mean(var_fore);
    ES_Fore(i_fore)= mean(es_fore);

    disp([i, VaR_Fore(i_fore), ES_Fore(i_fore)]);
end   

save Realized_ES_CAViaR_M_SnP_3k.mat

end

