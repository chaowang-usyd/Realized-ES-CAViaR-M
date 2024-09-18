function loglike= LH_2k_int(mu,quantile_level,data_input,realized_data_input)
% This is the log-likelihood function

data = data_input; 
xt = realized_data_input;
[T,k] = size(xt); 

if any(mu< -3) || any(mu > 3)
    loglike = -inf;
    return
end

beta0 = mu(:,1);
beta1 = mu(:,2);
tau1 = mu(:,3);
tau2 = mu(:,4);
gamma1 = mu(:,5); 
gamma2 = mu(:,6); 
xi1 = mu(:,7);  
xi2 = mu(:,8); 
phi1 = mu(:,9);  
phi2 = mu(:,10);  
delta11 = mu(:,11);  
delta21 = mu(:,12);  
delta12 = mu(:,13);  
delta22 = mu(:,14);  
nu0 = mu(:,15);
nu1 = mu(:,16);
psi1 = mu(:,17); 
psi2 = mu(:,18); 


gamma = [gamma1 gamma2];
xi = [xi1 xi2];
phi = [phi1 phi2];
deltak1 = [delta11 delta21];
deltak2 = [delta12 delta22];
psi = [psi1 psi2];

if( nu0 < 0.01 || nu1 < 0.01 || psi1 < 0 || psi2 < 0) 
    loglike = -inf;
    return
    
end

if abs(beta1) >= 1
    loglike = -inf;
    return
end


rt = data;
qt_caviar= zeros(T,1);
es= zeros(T,1);
wt = zeros(T,1);
rt_sort= sortrows(rt,1); 
qt_caviar(1)= rt_sort(ceil(T*quantile_level));

epsilon_t= zeros(T,1);
epsilon_t(1)= rt(1)/ (qt_caviar(1)+eps);

ukt = zeros(T,k); %initialize ut
wt(1)= -0.2*qt_caviar(1);
es(1) = qt_caviar(1) - wt(1);
ukt(1,:) = log(xt(1,:)) - xi - phi .* log(-qt_caviar(1)) - deltak1 .* epsilon_t(1) - deltak2*(epsilon_t(1).^2);

for i = 2:T

    eps_calc = epsilon_t(i-1);
        
    qt_caviar(i)= -exp(beta0 + beta1 * log(-qt_caviar(i-1)) + tau1 * eps_calc + tau2 * (eps_calc^2) + gamma * ukt(i-1,:)') ;
    epsilon_t(i)= rt(i) /(qt_caviar(i)+eps);
    
    wt(i) = nu0 + nu1*wt(i-1) + psi * abs(ukt(i-1,:))';

    es(i) = qt_caviar(i) - wt(i);
    
    ukt(i,:) = log(xt(i,:)) - xi - phi .* log(-qt_caviar(i)) - deltak1 .* epsilon_t(i) - deltak2*(epsilon_t(i).^2);
      
end

if any(es >= 0)
   loglike = -inf;
   return
end

S = ukt'*ukt;

l_re_escavair_1= log((quantile_level-1)./(es + eps));
l_re_escavair_2= ((rt-qt_caviar) .* (quantile_level-(rt<=qt_caviar)))./(quantile_level*es + eps);

l_measurement= -(1/2) * (T-k-1)*log(det(S));
loglike= ((sum(l_re_escavair_1) + sum(l_re_escavair_2)) + l_measurement); 