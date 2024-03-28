# Function for VECM
#   Johansen MLE of VECM with unknown cointegrating vector and unrestricted intercept
# Original code provided by Kilian and Lütkepohl
#   Website of Lutz Kilian, translated from MATLAB code

include("func_aux.jl")

function dif_lvl(y)
    return y[2:end,:] .- y[1:end-1,:]
end

function vecm(y, p, r)
    # Set up
    t, q = size(y)
    ydif = dif_lvl(y)

    y    = y'
    ydif = ydif'

    ΔY   = ydif[:, p:t-1]	
    X    = ones(1, t-p)

    for i=1:p-1
        X =[X; ydif[:, p-i:t-1-i]];
    end;
    
    Y = y[:, p:t-1];
    
    R0 = ΔY - (ΔY * X' / (X * X')) * X;
    R1 = Y - (Y * X' / (X * X')) * X;
    
    # Add as StaticArrays object for speedier inverse calculation
    S00 = R0 * R0' / (t-p);
    S11 = R1 * R1' / (t - p);
    S01 = R0 * R1' / (t-p);
    
    # Compute ML estimates
    iS11sq = inv(sqrt(S11));

    egg = eigen(iS11sq * S01' * inv(S00) * S01 * iS11sq)
    B   = egg.vectors
    lam = egg.values

    lamsort = sort(lam, rev=true)
    index   = indexin(lamsort, lam)
    Lam = diagx(lamsort);
    B = B[:, index];

    # Estimation
    β = iS11sq * B[:, 1:r];

    α = S01 * β * inv(β' * S11 * β);

    Γ = (ΔY - α * β' * Y) * X' / (X * X');
    
    U = ΔY - α * β' * Y - Γ * X;
    
    Σ = U * U' / (t-p);

    return α, β, Γ, Lam, Σ, Y, ΔY, X
end

function func_IRFvecm(data, p, r, h)
    # Function to calculate classical impulse response function
    T, k = size(data);

    # Increasing lag prod to match the IRF calculations
    α, β, Γ, Lam, Σ, Y, ΔY, X = vecm(data, p+1, r);

    # Set up the contemporaneous matrix - the coefficient form
    Γ = Γ[:, 2:size(Γ,2)];
    # For standardization of the impulse response,
    ρ = convert(Array{Float64}, cholesky(Σ).U');

    𝝖 = [Γ; [kron(eye(k), eye(p - 1)) zeros(k*(p-1), k)]];
    J = [eye(k) zeros(k, k*(p-1))];

    # Compute impulse response
    ψ = zeros(k, k, (h + 1));

    for i_irf = 0:1:h
        ψ[:,:, (i_irf + 1)] = J * 𝝖^(i_irf) * J';
        # If needed to be standardized, multiply ρ
    end
    
    # No confidence interval, for now
    return ψ;
end