# Multi-variate Time Series Model in julia
Julia implementation for a family of 'classical' estimation approach of multi-variate time series model. Current applications includes vector autoregressive (VAR(p)) model and its relative estimation procedures. See below for a current selection. Sub-folders in this repository includes source function codes and example application using publicly available data.

For the Bayesian implementation of the equivalent models, estimations, and forecasting frameworks, please visit my other repository for [Bayesian Vector Autoregressive model](https://github.com/justinjjlee/bayesianvar) suite.

Section unchecked are currently in development. This repository will be polished and improved as I develop and translate.

## Model Estimation
- [x] [Vector Autoregressive (VAR(p)) Model](https://github.com/justinjjlee/classvar/tree/master/test/var%20fit)
- [x] [Factor-augmented Vector Autoregressive (FAVAR(p)) Model](https://github.com/justinjjlee/classvar/tree/master/test/favar)
- [ ] Block-recursive Structural Vector Autoregressive Model
- [x] Vector Error Correction Model (known and unknow co-integration)

## Projections
- [x] [Short-run impulse response functions (Cholesky estimation)](https://github.com/justinjjlee/classvar/tree/master/test/var%20irf)
- [x] [Long-run impulse response functions](https://github.com/justinjjlee/classvar/tree/master/test/var%20irf)
- [x] Generalized impulse response functions
- [ ] Block-recursive short-run impulse response functions

## Decompositions
See functions within each models for applications. Based on the model identifications, computing the decomposition methods may vary.
- [x] Variance decomposition
- [x] Historical decomposition

## Example Applications
To test the functions and show application, I use following published or common macroeconometric models and frameworks.
- [Kilian (2009, American Economic Review)](https://www.aeaweb.org/articles?id=10.1257/aer.99.3.1053)
- [Bernanke, Boivin, and Eliasz (2005)](https://academic.oup.com/qje/article-abstract/120/1/387/1931468)
- Forecasting and generalized impulse response function with [US (Treasury)](https://fred.stlouisfed.org/series/T10Y2Y) and [Euro Area (European Central Bank)](https://data.ecb.europa.eu/data/datasets/YC/data-information) 