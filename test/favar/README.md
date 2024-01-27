# Factor-augmented Vector Autoregressive Model
Implementation of Factor-augmented Vector Autoregressive model (FAVAR(p)) based on [Bernanke, Boivin, and Eliasz (2005)](https://academic.oup.com/qje/article-abstract/120/1/387/1931468)

```julia
import Pkg;
Pkg.update();

using LinearAlgebra, Distributions, Statistics, MultivariateStats;
using ProgressMeter;
using Gadfly, Colors;

include("func_VectorAR.jl");
include("func_FAVAR.jl");
```
Test dataset is downloaded from Federal Reserve Bank of St. Louis - [Federal Reserve Economic Database (FRED) - FAVAR-1](https://research.stlouisfed.org/pdl/763)

```julia
y = open("bernanke_etal_2005_qje.txt");   # Data from Bernanke et al. (2005)
kₘ = 14;                                  # Number of variables in the main - VAR
yₘ = y[:, 1:kₘ];                          # Variables in the macro block
yₚ = y[:, (kₘ + 1):end];                  # Variables in the factor structure

# Other paramters needed.;
p = 12;                                   # Lag order
ζ = 3;                                    # Number of factors to be extracted

# Coefficient matrix, residuals, and covariance matrix.;
# (In order): coefficient, residual, covariance matrix - residual, factors, factor loadings.;
𝚩, 𝞄, 𝝨, 𝔽₁, ℾ₁ = func_favar(yₘ, yₚ, p, ζ);
```

Using the impulse response function,
```julia
ψ,
  ψ_lb_2sd, ψ_lb_1sd,
  ψ_ub_1sd, ψ_ub_2sd,
  FEVDC = func_IRFvar(vcat(y, 𝔽₁), p);       # Results from impulse responses, bootstrap CI band, and FEVDC

# For calculating the responses, multiply factor loadings for those
ψ₁ = ψ[:, (kₘ+1):end] * 𝔽₁';
````
I recommend leveraging bias-corrected bootstrap confidence intervals [Kilian (1998)](https://www.mitpressjournals.org/doi/pdf/10.1162/003465398557465) as in [Bernanke, Boivin, and Eliasz (2005)](https://academic.oup.com/qje/article-abstract/120/1/387/1931468).