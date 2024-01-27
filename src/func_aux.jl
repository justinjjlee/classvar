# Functions for Auxiliary Operations for this Repository
import Pkg;
using ProgressMeter;
using LinearAlgebra, Statistics;
using Gadfly, Colors;

eye(n) = Matrix{Float64}(I, n, n)
chol(mat) = convert(Array{Float64}, cholesky(Hermitian(mat)).U');
# For creating diagonal matrix;
# Nuance in julia calculation 
# https://web.eecs.umich.edu/~fessler/course/551/julia/tutor/03-diag.html
diagx(x) = diagm(vec(x))

# Auxiliary functions to be used

function recovery_lag(Δy, yfirst, lag)
    # Function to recover level from growth change of 'lag' order
    #   difference in log growth
    #   e.g. Δy ⧋ ln(y_t) - ln(y_{t-5})
    nfin = length(Δy) + (lag - 1) # full list of data
    yfin = zeros(nfin, 1)
    # Initial start
    yfin[1:(lag-1)] = yfirst
    for iter in lag:nfin
        yfin[iter] = exp(Δy[iter-lag+1] + log(yfin[iter-lag+1]))
    end
    return yfin
end

# Function to generate iterative matrix block for regressor
# Reminder that multivariate equation's explanatory and 
#   response variables are interpreted 
#   differently in time series model
function var_data(data, p)
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

    return X, Y
end