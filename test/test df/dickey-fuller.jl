# Function for Dickey–Fuller test for unit root behaviors in AR(p) time series process

#Author: Justin Lee
#Code start date: 5/30/2016

using Statistics, Distributions, Plots

norms = randn(20000,1);
ttestcase1 = zeros(20000,6); #20,000 trial results of each sample size t-statistics for each sample size Case 1;
ttestcase2 = zeros(20000,6); #20,000 trial results of each sample size t-statistics for each sample size Case 2;
ttestcase3 = zeros(20000,6); #20,000 trial results of each sample size t-statistics for each sample size Case 3;
ttestcase4 = zeros(20000,6); #20,000 trial results of each sample size t-statistics for each sample size Case 4;

T = [25 50 100 250 500 10000]; #Sample size for each trial;

for i in 1:length(T) #Each sample size
    Tsize = T[i];
    for j in 1:20000
        #DGP Case 1; start with zero
        eps = randn(Tsize,1); y = cumsum(eps');
        
        #fit estimated model Case 1;
        X = y[1:end-1]; Y = y[2:end];
        rhohat = (X'*X)\X'*Y; Yhat = X.*rhohat; ehat = Y- Yhat;
        
        sigma=ehat'*ehat/(length(y)-1);
        rhohatse = sqrt(sigma.*(X'*X)); 
        estans = (rhohat - 1)/rhohatse;  #convert 1*1 array into 1-element float;
        ttestcase1[j,i] = estans[1,1];

        #Case 2
        # Same DGP as Case 1
        #fit estimated model Case 2;
        X = [ones(length(y)-1,1) y[1:end-1]]; Y = y[2:end];
        rhohat = (X'*X)\X'*Y; Yhat = X*rhohat; ehat = Y- Yhat;

        sigma=ehat'*ehat/(length(y)-2);
        rhohatse = sqrt(complex(sigma.*inv(X'*X))); #in case of negative inverse matrix result
        estans = (rhohat[2] - 1)/rhohatse[2,2]; #convert 1*1 array into 1-element float;
        ttestcase2[j,i] = estans[1,1];        
        
        #DGP Case 3; start with zero
        eps = randn(Tsize,1); y = cumsum(eps'+1);
        #fit estimated model Case 3;
        X = [ones(length(y)-1,1) y[1:end-1]]; Y = y[2:end];
        rhohat = (X'*X)\X'*Y; Yhat = X*rhohat; ehat = Y- Yhat;

        sigma=ehat'*ehat/(length(y)-2);
        rhohatse = sqrt(complex(sigma.*inv(X'*X)));
        estans = (rhohat[2] - 1)/rhohatse[2,2]; #convert 1*1 array into 1-element float;
        ttestcase3[j,i] = estans[1,1];
            
        #Case 4
        # Same DGP as Case 3;
        #fit estimated model Case 3;
        time = 1:1:(length(y)-1); time = time';
        X = [ones(length(y)-1,1) time y[1:end-1]]; Y = y[2:end];
        rhohat = (X'*X)\X'*Y; Yhat = X*rhohat; ehat = Y- Yhat;
        
        sigma=ehat'*ehat/(length(y)-3);
        rhohatse = sqrt(sigma.*inv(X'*X));
        estans = (rhohat[3] - 1)/rhohatse[3,3]; #convert 1*1 array into 1-element float;
        ttestcase4[j,i] = estans[1,1];        
    
    end
end

#plot
#Case 1
plot1 = figure(1)
for j in 1:6
    subplot(4,6,j)
    x = ttestcase1[1:end,j];
    nbin = 100;
    plt[:hist](x,nbin)
end;


cd("$(homedir())/Desktop")

srand(1234); using PyPlot;

Ts = [100:500:20100];
boot_mean = zeros(1,length(Ts)); boot_sdest = zeros(1,length(Ts));
boot_sd = zeros(1,length(Ts));
samp_mean = zeros(1,length(Ts)); samp_sd = zeros(1,length(Ts));

for t = 1:length(Ts) #For each sample size
    T = Ts[t];
    y = randn(T,1); #mean zero variance 5/3, supposed to be t distribution;
    samp_mean[t] = mean(y); samp_sd[t] = sqrt(var(y));
    sampmean = zeros(200,1);sampsd = zeros(200,1);
    for r = 1:200 #bootstrap replication
        index = floor(rand(1,T)*(T))+1;
        boot_y = y[index];
        sampmean[r] = mean(boot_y);
        sampsd[r] = sqrt(var(boot_y));
    end
    boot_mean[t] = mean(sampmean);
    boot_sdest[t] = mean(sampsd);
    # Bootstrap SD of sample mean;
    boot_sd[t] = sqrt(1/(200-1)*(sum((sampmean-boot_mean[t]).^2)));
end

