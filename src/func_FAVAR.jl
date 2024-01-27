# Function for FAVAR by Bernanke et al. (2015)
import Pkg;
using LinearAlgebra, Distributions, Statistics, MultivariateStats;
using ProgressMeter;
using Gadfly, Colors;

function func_std(mat)
    # Function to standardize data
    T, k = size(mat);
    μ = mean(mat; dims = 1);
    σ = std(mat; dims = 1);
    return (mat .- μ) ./ σ;
end

function func_favar(yₘ, yₚ, p, ζ)
    # yₘ: data - main block
    # yₚ: data - factor block - to be
    # p: lag term for VAR
    # ζ: Number of factors to be extracted

    T, kₘ = size(yₘ);

    # For principal component analysis (PCA), always use standardized data.
    res_pca = fit(PCA, func_std(yₚ); maxoutdim = ζ);
    𝔽ₒ = projection(res_pca);

    Q = yₚ';
    Z = [yₘ 𝔽ₒ]';
    ℾₒ = (Q * Z')/(Z * Z');
    ℾₒ𝑦 = ℾₒ[:, (1:kₘ)];
    x̃ = yₚ - (ℾₒ𝑦 * yₘ')';

    ssrₒ = sum((Q - ℾₒ * Z).^2, dims = 2)

    # Start the iteraction process
    𝔽₁ = 𝔽ₒ;
    ℾ₁ = ℾₒ; ℾ₁𝑦 = ℾₒ𝑦;
    ssr₁ = ssrₒ

    n_ϝ = 0; # Count iterations;
    while sum(abs(ssr₁ .- ssrₒ) > 10^(-6))
        # If no convergence, repeat the process
        # Update the criteria
        𝔽ₒ = 𝔽₁;
        ssrₒ = ssr₁;

        # Redraw component
        res_pca = fit(PCA, func_std(x̃); maxoutdim = ζ);
        𝔽₁ = projection(res_pca);

        Q = yₚ';
        Z = [yₘ 𝔽₁]';
        ℾ₁ = (Q * Z')/(Z * Z');
        ℾ₁𝑦 = ℾ₁[:, (1:kₘ)];
        x̃ = yₚ - (ℾ₁𝑦 * yₘ')';

        ssr₁ = sum((Q - ℾ₁ * Z).^2, dims = 2);

        n_ϝ = n_ϝ + 1;
    end

    # conditional check for factor loading (optional)
    #=
    #Check on elements of factor loading that belongs to first series in the
    # information set matrix - yₚ
    # If an element is negative, multiply whole factor loading with -1 and
    # then respective factor

    for iₚ = (kₘ + 1):size(Gamma_final, 2)
        if ℾ₁[1, iₚ] < 0
            ℾ₁[:, iₚ] = -1 .* ℾ₁[:, iₚ];
            𝔽₁[:, (iₚ - kₘ)] = -1 .* 𝔽₁[:, (iₚ - kₘ)];
        end
    end
    =#

    Q = yₚ';
    Z = [yₘ 𝔽₁]';
    ℾ₁ = (Q * Z')/(Z * Z');
    ℾ₁𝑦 = ℾ₁[:, (1:kₘ)];
    x̃ = yₚ - (ℾ₁𝑦 * yₘ')';

    # compute VAR(p)
    𝔇 = [yₘ 𝔽₁];

    𝚩, 𝞄, 𝝨 = func_VAR(𝔇, p);

    return 𝚩, 𝞄, 𝝨, 𝔽₁, ℾ₁;
end
