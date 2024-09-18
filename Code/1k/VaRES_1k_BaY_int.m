function [var_fore, es_fore]= VaRES_1k_BaY_int(ChainIs,data_input,realized_data_input, n_imh,n_fit,quantile_level)

data = data_input; 

xt = realized_data_input;
[T,k] = size(xt);

var_fore= zeros(n_imh,1);
es_fore= zeros(n_imh,1);
wt_fore= zeros(n_imh,1);

[T,k] = size(xt); 

rt = data;%-m; 

    qt_caviar= zeros(T,1);
    es= zeros(T,1);
    wt = zeros(T,1);
    rt_sort= sortrows(rt,1); 
epsilon_t= zeros(T,1);
    ukt = zeros(T,k); %initialize ut

for ii = 1:n_imh
    
    beta0 = ChainIs(ii,1);
    beta1 = ChainIs(ii,2);
    tau1 = ChainIs(ii,3);
    tau2 = ChainIs(ii,4);
    gamma = ChainIs(ii,5) ;
    xi =  ChainIs(ii,6);
    phi =  ChainIs(ii,7);
    deltaK1 = ChainIs(ii,8);
    deltaK2 = ChainIs(ii,9);
    nu0 = ChainIs(ii,10);
    nu1 = ChainIs(ii,11);
    psi = ChainIs(ii,12); 
        
            
    %m=mean(data);
    
    qt_caviar(1)= rt_sort(ceil(T*quantile_level));

    
    epsilon_t(1)= rt(1)/ (qt_caviar(1)+eps);

    
    wt(1)= -0.2*qt_caviar(1);
    es(1) = qt_caviar(1) - wt(1);
    ukt(1,:) = log(xt(1,:)) - xi - phi .* log(-qt_caviar(1)) - deltaK1 .* epsilon_t(1) - deltaK2*(epsilon_t(1).^2);% - mean(epsilon_t(1).^2)) ;

    for i = 2:T
        qt_caviar(i)= -exp(beta0 + beta1 * log(-qt_caviar(i-1)) + tau1 * epsilon_t(i-1) + tau2 * (epsilon_t(i-1).^2) + gamma * ukt(i-1,:)') ;
        epsilon_t(i)= rt(i) /(qt_caviar(i) +eps);

        wt(i) = nu0 + nu1*wt(i-1) + psi * abs(ukt(i-1,:)');%+ tau1_dstar * epsilon_t(i-1) + tau2_dstar * (epsilon_t(i-1).^2 - mean(epsilon_t(1:i-1).^2)) ;

        es(i) = qt_caviar(i) - wt(i);

        ukt(i,:) = log(xt(i,:)) - xi - phi .* log(-qt_caviar(i)) - deltaK1 .* epsilon_t(i) - deltaK2*(epsilon_t(i).^2);% - mean(epsilon_t(1:i).^2)) ;

    end


    var_fore(ii) =  -exp(beta0 + beta1 * log(-qt_caviar(n_fit)) + tau1 * epsilon_t(n_fit) + tau2 * (epsilon_t(n_fit).^2 - mean(epsilon_t(1:n_fit).^2)) + gamma * ukt(n_fit,:)') ;
    wt_fore(ii) = nu0 + nu1*wt(n_fit) + psi * abs(ukt(n_fit,:)');%+ tau1_dstar * epsilon_t(n_fit) + tau2_dstar * (epsilon_t(n_fit).^2 - mean(epsilon_t(1:n_fit).^2)) ;
    es_fore(ii) = var_fore(ii) - wt_fore(ii);
end
