# Multi-variate Time Series Model in julia
Julia implementation for a family of 'classical' estimation approach of multi-variate time series model. Current applications includes vector autoregressive (VAR(p)) model and its relative estimation procedures. See below for a current selection. Sub-folders in this repository includes source function codes and example application using publicly available data.

For the Bayesian implementation of the equivalent models, estimations, and forecasting frameworks, please visit my other repository for [Bayesian Vector Autoregressive model](https://github.com/justinjjlee/bayesianvar) suite.

Section unchecked are currently in development. This repository will be polished and improved as I develop and translate.

## Model Estimation
[x] Vector Autoregressive (VAR(p)) Model
[x] Factor-augmented Vector Autoregressive (FAVAR(p)) Model
[ ] Block-recursive Structural Vector Autoregressive Model
[ ] Vector Error Correction Model (known and unknow co-integration)

## Projections
[x] Short-run impulse response functions (Cholesky estimation)
[x] Long-run impulse response functions
[ ] Generalized impulse response functions
[ ] Block-recursive short-run impulse response functions

## Decompositions
[x] Variance decomposition
[x] Historical decomposition