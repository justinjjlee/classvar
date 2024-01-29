# Functions for the Vector Autoregressive model
import Pkg;
using ProgressMeter;
using LinearAlgebra, Statistics;

eye(n) = Matrix{Float64}(I, n, n)
chol(mat) = convert(Array{Float64}, cholesky(Hermitian(mat)).U');
# For creating diagonal matrix;
# Nuance in julia calculation 
# https://web.eecs.umich.edu/~fessler/course/551/julia/tutor/03-diag.html
diagx(x) = diagm(vec(x))

# (1) Function to calculate impulse response
function func_VAR(data, p)
    # x: partial equilibrium data
    # p: lag order
    T, k = size(data)
    # Set up L.H.S.: set t-by-k array
    y = data';
    Y = y[:, p:T];

    for i = 1:(p-1)
        Y = [Y; y[:, (p-i):(T-i)]];
    end
    # Set up R.H.S.
    X = [ones(1, T-p); Y[:, 1:(T-p)]];
    Y = y[:, ((p+1):T)];

    # Least square estimation
    𝚩 = (Y*X')/(X*X');
    𝞄 = (Y - 𝚩 * X)';
    𝝨 = 𝞄' * 𝞄 ./ (T-p-p*k-1);
    return 𝚩, 𝞄, 𝝨;
end

function func_IRFvar(data, p, h, ℏ)
    T, k = size(data);

    𝚩_ols, 𝞄_ols, 𝝨_ols = func_VAR(data, p);
    V_ols = 𝚩_ols[:, 1];
    𝚩_ols = 𝚩_ols[:, 2:size(𝚩_ols,2)];
    # For standardization of the impulse response,
    ρ = convert(Array{Float64}, cholesky(𝝨_ols).U');

    𝝖 = [𝚩_ols; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
    J = [eye(k) zeros(k, k*(p-1))];

    # Compute impulse response
    ψ = zeros(k, k, (h + 1));

    for i_irf = 0:1:h
        ψ[:,:, (i_irf + 1)] = J * 𝝖^(i_irf) * J';
        # If needed to be standardized, multiply ρ
    end

    # Include recursive-design wild bootstrap calculation of confidence band
    #   Follow Kilian (2009, AER)
    #srand(8301993);
    n_repli = 2000; # Number of iterations

    ψ_boot = zeros(k, k, (h + 1), n_repli);

    y = data';
    Y = y[:, p:T];
    for i = 1:(p-1)
        Y = [Y; y[:, (p-i):(T-i)]];
    end

    Ur = zeros(k*p, T-p+1);
    Yr = zeros(k*p, T-p+1);

    @showprogress 1 "Wild-bootstrap confidence band " for i_iter = 1:n_repli
        pos = Int64(round(rand(1)[]*(T-p+1)));
        if pos == 0 #randomization hits zero, default to 1
            pos = 1
        end
        Yr[:, 1] = Y[:, pos];

        # Recrusive design bootstrap
        η = randn(1, size(𝞄_ols, 1));
        η = kron(η, ones(k, 1));

        Ur[1:k, 2:(T-p+1)] = 𝞄_ols' .* η;

        for ii_iter = 2:(T-p+1)
            Yr[:, ii_iter] = [V_ols; zeros(k*(p-1),1)] +
                                    𝝖 * Yr[:, (ii_iter - 1)] +
                                    Ur[:, ii_iter];
        end

        yr = Yr[1:k, :];
        for i=2:p
    		yr = [Yr[(((i-1)*k)+1):(i*k), 1] yr];
        end;
        yr = yr';

        # Iterated partial equilibrium estimation.;
        𝚩, 𝞄, 𝝨 = func_VAR(yr, p);

        𝚩 = 𝚩[:, 2:size(𝚩,2)];
        # For standardization of the impulse response,
        ρ = cholesky(𝝨).L[:,:]; # = cholesky()';

        𝝖 = [𝚩; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
        J = [eye(k) zeros(k, k*(p-1))];
        # Computaiton of impulse response
        temp_ψ_prev = [];
        for i_irf = 0:1:h
            temp_ψ = J * 𝝖^(i_irf) * J';
            ψ_boot[:,:, (i_irf + 1), i_iter] = temp_ψ ;
            # If needed to be standardized, multiply ρ
        end
    end
    # Need to calculate percentile
    α = 1; # One standard deviaiton confidence band of bootstrap iterations
    # Thus, this measure of confidence band does not carry literal definition
    #   as in classical model.

    # Calculate the standard deviation, across differen iteration (4th dim)
    σ_ψ_boot = std(ψ_boot, dims = 4);

    ψ_ub_1sd = ψ + σ_ψ_boot
    ψ_lb_1sd = ψ - σ_ψ_boot
    ψ_ub_2sd = ψ + σ_ψ_boot * 2;
    ψ_lb_2sd = ψ - σ_ψ_boot * 2;

    ############################################################################
    # FORECAST ERROR VARIANCE DECOMPOSITION
    T, k = size(data);

    𝚩_ols, 𝞄_ols, 𝝨_ols = func_VAR(data, p);
    V_ols = 𝚩_ols[:, 1];
    𝚩_ols = 𝚩_ols[:, 2:size(𝚩_ols,2)];
    # For standardization of the impulse response,
    ρ = convert(Array{Float64}, cholesky(𝝨_ols).U');

    𝝖 = [𝚩_ols; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
    J = [eye(k) zeros(k, k*(p-1))];

    # forecast error variance decomposition
    TH1 = J * 𝝖^(0) *J';
    TH = TH1 * ρ;
    TH = TH';
    TH2 = (TH .* TH);
    TH3 = TH2;

    for i=2:ℏ
        TH = J * 𝝖^(i-1) * J' * ρ;
        TH = TH'; TH2 = (TH.*TH);
        TH3 = TH3 + TH2;
    end;
    TH4 = sum(TH3, dims = 1); VC = zeros(k, k);
    for j=1:k
        VC[j,:] = TH3[j, :] ./ TH4';
    end;

    FEVDC = VC'*100;

    return ψ,
           ψ_lb_1sd, ψ_ub_1sd, ψ_lb_2sd, ψ_ub_2sd,
           FEVDC;
end

function var_hd(y, p)
    h = 15;                            # Horizon - IRF
    ℏ = 12;                            # Horizon - FEVDC

    𝚩, 𝞄, 𝝨 = func_VAR(y, p);          # Coefficient matrix, 
                                   # residuals, 
                                   # and covariance matrix
    ψ,
        ψ_lb_1sd, ψ_ub_1sd,
        ψ_lb_2sd, ψ_ub_2sd,
        FEVDC = func_IRFvar(y, p, h, ℏ);      



    T, K = size(y)
    # already lower triangular
    uhat = 𝞄 * inv(chol(𝝨))
    # Initialize:
    yhat_hd = zeros(T-p, K, K)
    for i_shock = 1:K
        ulast = uhat[:, i_shock]
        uj = zeros(T-p, K);
        for i_horizon = 1:h
            uj = uj .+ (ulast .* ψ[:, i_shock, i_horizon]');
            ulast = vcat(0, ulast[1:(end-1)]);
        end
        yhat_hd[:, :, i_shock] = uj
    end

    return yhat_hd
end

function func_contemp(bmat, 𝝨_ols, k, p)
    Ik = eye(k);
    i_block = 1:k; # initial, block of coefficients for same lag.
    for i = 1:p
        Ik = Ik .- bmat[:, i_block];
        # update the block index.
        i_block = i_block .+ k;
    end
    Ik = inv(Ik);
    
    Bk = eye(k);
    i_block = 1:k; # initial, block of coefficients for same lag.
    for i = 1:p
        Bk = Bk .- bmat[:, i_block]';
        # update the block index.
        i_block = i_block .+ k;
    end
    Bk = inv(Bk);

    # Long run impact matrix
    ψ0 = Ik * 𝝨_ols * Bk;
    #return Ik, Bk, ψ0
    ψ0 = chol(ψ0);

    # Contemporaneous shock identificaiton.
    ρ = Ik\ψ0';
    return ρ
end

function func_IRFvar_LR(data, p, h)
    # Implementation of Blanchard and Quah (1989)
    T, k = size(data);

    𝚩_ols, 𝞄_ols, 𝝨_ols = func_VAR(data, p);
    V_ols = 𝚩_ols[:, 1];
    𝚩_ols = 𝚩_ols[:, 2:size(𝚩_ols,2)];

    𝝖 = [𝚩_ols; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
    J = [eye(k) zeros(k, k*(p-1))];

    # For contemporaneous impact matrix - long run identificaiton
    # Contemporaneous shock identificaiton.
    ρ = func_contemp(𝚩_ols, 𝝨_ols, k, p);

    # Compute impulse response
    ψ = zeros(k, k, (h + 1));

    for i_irf = 0:1:h
        ψ[:,:, (i_irf + 1)] = J * 𝝖^(i_irf) * J' * ρ;
        # If needed to be standardized, multiply ρ
    end

    # Include recursive-design wild bootstrap calculation of confidence band
    #   Follow Kilian (2009, AER)
    #srand(8301993);
    n_repli = 2000; # Number of iterations

    ψ_boot = zeros(k, k, (h + 1), n_repli);

    y = data';
    Y = y[:, p:T];
    for i = 1:(p-1)
        Y = [Y; y[:, (p-i):(T-i)]];
    end

    Ur = zeros(k*p, T-p+1);
    Yr = zeros(k*p, T-p+1);

    @showprogress 1 "Wild-bootstrap confidence band " for i_iter = 1:n_repli
        pos = Int64(round(rand(1)[]*(T-p+1)));
        if pos == 0 #randomization hits zero, default to 1
            pos = 1
        end
        Yr[:, 1] = Y[:, pos];

        # Recrusive design bootstrap
        η = randn(1, size(𝞄_ols, 1));
        η = kron(η, ones(k, 1));

        Ur[1:k, 2:(T-p+1)] = 𝞄_ols' .* η;

        for ii_iter = 2:(T-p+1)
            Yr[:, ii_iter] = [V_ols; zeros(k*(p-1),1)] +
                                    𝝖 * Yr[:, (ii_iter - 1)] +
                                    Ur[:, ii_iter];
        end

        yr = Yr[1:k, :];
        for i=2:p
    		yr = [Yr[(((i-1)*k)+1):(i*k), 1] yr];
        end;
        yr = yr';

        # Iterated partial equilibrium estimation.;
        𝚩, 𝞄, 𝝨 = func_VAR(yr, p);

        𝚩 = 𝚩[:, 2:size(𝚩,2)];
        # For standardization of the impulse response,
        # For contemporaneous impact matrix - long run identificaiton
        # Contemporaneous shock identificaiton.
        ρ = func_contemp(𝚩, 𝝨, k, p);

        𝝖 = [𝚩; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
        J = [eye(k) zeros(k, k*(p-1))];
        # Computaiton of impulse response
        temp_ψ_prev = [];
        for i_irf = 0:1:h
            temp_ψ = J * 𝝖^(i_irf) * J' * ρ;
            ψ_boot[:,:, (i_irf + 1), i_iter] = temp_ψ ;
            # If needed to be standardized, multiply ρ
        end
    end
    # Need to calculate percentile
    α = 1; # One standard deviaiton confidence band of bootstrap iterations
    # Thus, this measure of confidence band does not carry literal definition
    #   as in classical model.

    # Calculate the standard deviation, across differen iteration (4th dim)
    σ_ψ_boot = std(ψ_boot, dims = 4);

    ψ_ub_1sd = ψ + σ_ψ_boot
    ψ_lb_1sd = ψ - σ_ψ_boot
    ψ_ub_2sd = ψ + σ_ψ_boot * 2;
    ψ_lb_2sd = ψ - σ_ψ_boot * 2;

    return ψ,
           ψ_lb_1sd, ψ_ub_1sd, ψ_lb_2sd, ψ_ub_2sd
end


function gIRF(data, p, h)
    # Impulse response functions - point estimate
    #   (a) unit-scale response
    #   (b) orthogonalized response
    #   (c) cumulative shock ("generalized") response
    T, k = size(data);

    𝚩_ols, 𝞄_ols, 𝝨_ols = func_VAR(data, p);
    V_ols = 𝚩_ols[:, 1];
    𝚩_ols = 𝚩_ols[:, 2:size(𝚩_ols,2)];
    # For standardization of the impulse response,
    ρ = convert(Array{Float64}, cholesky(𝝨_ols).U');
    σ = (𝝨_ols * eye(k)).^(-1/2)

    𝝖 = [𝚩_ols; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
    J = [eye(k) zeros(k, k*(p-1))];

    # Compute impulse response
    ψ  = zeros(k, k, (h + 1));
    ψo = zeros(k, k, (h + 1));
    ψg = zeros(k, k, (h + 1));

    for i_irf = 0:1:h
        ψ[:,:, (i_irf + 1)] = J * 𝝖^(i_irf) * J';
        # If needed to be standardized, multiply p
        ψo[:,:, (i_irf + 1)] = ψ[:,:, (i_irf + 1)] * ρ
        ψg[:,:, (i_irf + 1)] = ψ[:,:, (i_irf + 1)] * ρ * σ
    end

    return ψ, ψo, ψg;
end

# Create vectors of lags
function matlag(x, lag)
    # Create matrix with original data in lag structure
    #   also add constant term

    # NOTE: This lag structure only works if the matrix
    #   does not require contemporaneous term.
    #   i.e. OLS regressionlon
    T, N = size(x)
    Y = x[lag:T, :];

    for i = 1:L-1
        Y = hcat(Y, x[(lag-i):(T-i), :]);
    end
    # Add constant vector
    Y = hcat(Y, ones(size(Y)[1], 1))
    return Y
end

# Setting up lag structure of data for
#   Mix-frequency VAR .....................................................
function lagspit(x, p)
    try
        # get size
        global R, C = size(x);
    catch 
        # if error, it means feeding as vector
        global R, C = length(x), 1;
    end
    
    #Take the first R-p rows of matrix x
    x1=x[1:(R-p),:];
    #Preceed them with p rows of zeros and return
    y=vcat(zeros(p,C), x1);
    return y
end

function prepare( data,L )
    X=lagspit(data,1); # First lag
    for j=2:L
        X=hcat(X, lagspit(data,j));
    end
    X=hcat(X, ones(size(X)[1],1));
    return X
end

function create_dummies(lamda,tau,delta,epsilon,p,mu,sigma,n)
    # Creates matrices of dummy observations [...];
    #lamda tightness parameter
    #tau  prior on sum of coefficients
    #delta prior mean for VAR coefficients
    # epsilon tigtness of the prior around constant
    # mu sample mean of the data
    # sigma AR residual variances for the data
    
    
    

    # Initialise output (necessary for final concatenation to work when tau=0):
    x = [];
    y = [];
    yd1 = [];
    yd2 = [];
    xd1 = [];
    xd2 = [];
    
    ## Get dummy matrices in equation (5) of Banbura et al. 2007:
    if lamda>0
        if epsilon >0
            yd1=vcat(diagx(sigma.*delta)./lamda,
                zeros(n*(p-1),n),
                diagx(sigma),
                zeros(1,n));
            
            jp=diagx(1:p);
            
            xd1=vcat(hcat(kron(jp,diagx(sigma)./lamda), zeros((n*p),1)),
                zeros(n,(n*p)+1),
                hcat(zeros(1,n*p), epsilon));

    else
        
        yd1=vcat(diagx(sigma.*delta)./lamda,
             zeros(n*(p-1),n),
             diagx(sigma));
         
        jp=diagx(1:p);
        
        xd1=vcat(kron(jp,diagx(sigma)./lamda),
             zeros(n,(n*p))); 
        end
    end
    ## Get additional dummy matrices - see equation (9) of Banbura et al. 2007:
    if tau>0
        if epsilon>0
            yd2=diagx(delta.*mu)./tau;
            xd2=hcat(kron(ones(1,p),yd2), zeros(n,1));
        else
            yd2=diagx(delta.*mu)./tau;
            xd2=kron(ones(1,p),yd2);  
        end
    end
         
    # return
    y=vcat(yd1, yd2);
    x=vcat(xd1, xd2);
    return y, x
end

function invpd(in)
    temp = eye(size(in,2));
    out = in\temp;
    return out
end

function stability(beta,n,l)

    #coef   (n*l+1)xn matrix with the coef from the VAR
    #l      number of lags
    #n      number of endog variables
    #FF     matrix with all coef
    #S      dummy var: if equal one->stability
    coef=reshape(beta,n*l+1,n);
    #coef
    #coef
    FF=zeros(n*l,n*l);
    FF[n+1:n*l, 1:n*(l-1)]= eye(n*(l-1))#eye(n*(l-1), n*(l-1));
    
    temp = reshape(beta, n*l+1, n);
    temp = temp[1:n*l, 1:n]';
    FF[1:n, 1:n*l] = temp;
    ee=maximum(abs.(eigvals(FF)));
    return ee>1
end

function iwpQ(v,ixpx)
    k=size(ixpx)[1];
    z=zeros(v,k);
    mu=zeros(k,1);
    for i=1:v
        z[i,:]=(chol(ixpx)'*randn(k,1))';
    end
    
    return inv(z'*z)
end

function comp(beta,n,l,ex)

    #coef   (n*l+1)xn matrix with the coef from the VAR
    #l      number of lags
    #n      number of endog variables
    #FF     matrix with all coef
    #S      dummy var: if equal one->instability
    #coef=reshape(beta,n*l+1,n);
    #coef
    #coef
    FF=zeros(n*l,n*l);
    FF[n+1:n*l,1:n*(l-1)]=eye(n*(l-1));
    
    temp=reshape(beta,n*l+ex,n);
    temp=temp[1:n*l,1:n]';
    FF[1:n,1:n*l]=temp;
    temp=reshape(beta,n*l+ex,n);
    mu=zeros(n*l,1);
    mu[1:n]=temp[end,:]';
    
    return FF, mu'
end