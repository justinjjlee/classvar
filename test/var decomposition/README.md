# Vector Autoregressive Model - Decomposition
Julia implementation of vector autoregressive (VAR(p)) model with closed form solution - OLS estimation. 

In this repository, there are methods to calculate decomposition of variations in fitted/projected model.

## Variance Decomposition
Method to compute and explain total variations explained by the fitted model
```julia
# Parameter definition
p = 5 # Horizon defined for model lag term
h = 8 # Horizon measured for the impulse response function
# Horizon decomposition, can be couple unit dates to very large (e.g. 10_000) for equilibrium
ℏ = 8  

FEVDC = var_vd(mat_y, p, ℏ);
```

## Historical Decomposition
Based on the decomposition of variance, the historical decomposition calculates variations point-in-time using estimated impulse response functions.
```julia
# Parameter definition
p = 5 # Horizon defined for model lag term

yhat_hd = var_hd(mat_y, p)
```
