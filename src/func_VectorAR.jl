# Functions for the Vector Autoregressive model
import Pkg;
using ProgressMeter;
using LinearAlgebra, Statistics;
using Gadfly, Colors;

eye(n) = Matrix{Float64}(I, n, n)
chol(mat) = convert(Array{Float64}, cholesky(Hermitian(mat)).U');

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