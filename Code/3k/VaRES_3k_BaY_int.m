function [var_fore, es_fore]= VaRES_3k_BaY_int(ChainIs,data_input,realized_data_input, n_imh,n_fit,quantile_level)

data = data_input; 
xt = realized_data_input;
[T,k] = size(xt);

var_fore= zeros(n_imh,1);
es_fore= zeros(n_imh,1);
wt_fore= zeros(n_imh,1);

[T,k] = size(xt); 

rt = data;

qt_caviar= zeros(T,1);
es= zeros(T,1);
wt = zeros(T,1);
rt_sort= sortrows(rt,1); 
qt_caviar(1)= rt_sort(ceil(T*quantile_level));

epsilon_t= zeros(T,1);
ukt = zeros(T,k); 

for ii = 1:n_imh
    
    beta0 = ChainIs(ii,1);
    beta1 = ChainIs(ii,2);
    tau1 = ChainIs(ii,3);
    tau2 = ChainIs(ii,4);
    gamma1 = ChainIs(ii,5); 
    gamma2 = ChainIs(ii,6); 
    gamma3 = ChainIs(ii,7); 
    xi1 = ChainIs(ii,8);  
    xi2 = ChainIs(ii,9); 
    xi3 = ChainIs(ii,10);  
    phi1 = ChainIs(ii,11);  
    phi2 = ChainIs(ii,12); 
    phi3 = ChainIs(ii,13);  
    delta11 = ChainIs(ii,14);  
    delta21 = ChainIs(ii,15); 
    delta31 = ChainIs(ii,16); 
    delta12 = ChainIs(ii,17); 
    delta22 = ChainIs(ii,18);  
    delta32 = ChainIs(ii,19);  
    nu0 = ChainIs(ii,20);
    nu1 = ChainIs(ii,21);
    psi1 = ChainIs(ii,22); 
    psi2 = ChainIs(ii,23); 
    psi3 =ChainIs(ii,24); 


    gamma = [gamma1 gamma2 gamma3];
    xi = [xi1 xi2 xi3];
    phi = [phi1 phi2 phi3];
    deltak1 = [delta11 delta21 delta31];
    deltak2 = [delta12 delta22 delta32];
    psi = [psi1 psi2 psi3];

    epsilon_t(1)= rt(1)/ (qt_caviar(1)+eps);
    wt(1)= -0.2*qt_caviar(1);
    es(1) = qt_caviar(1) - wt(1);
    ukt(1,:) = log(xt(1,:)) - xi - phi .* log(-qt_caviar(1)) - deltak1 .* epsilon_t(1) - deltak2*(epsilon_t(1).^2);%mean(epsilon_t(1).^2)) ;

    for i = 2:T
        qt_caviar(i)= -exp(beta0 + beta1 * log(-qt_caviar(i-1)) + tau1 * epsilon_t(i-1) + tau2 * (epsilon_t(i-1).^2) + gamma * ukt(i-1,:)') ;
        epsilon_t(i)= rt(i) /(qt_caviar(i) +eps);
        wt(i) = nu0 + nu1*wt(i-1) + psi * abs(ukt(i-1,:)');
        es(i) = qt_caviar(i) - wt(i);
        ukt(i,:) = log(xt(i,:)) - xi - phi .* log(-qt_caviar(i)) - deltak1 .* epsilon_t(i) - deltak2*(epsilon_t(i).^2);%mean(epsilon_t(1:i).^2)) ;
    end

    var_fore(ii) =  -exp(beta0 + beta1 * log(-qt_caviar(n_fit)) + tau1 * epsilon_t(n_fit) + tau2 * (epsilon_t(n_fit).^2) + gamma * ukt(n_fit,:)') ;
    wt_fore(ii) = nu0 + nu1*wt(n_fit) + psi * abs(ukt(n_fit,:)');
    es_fore(ii) = var_fore(ii) - wt_fore(ii);
end