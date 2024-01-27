# Vector Autoregressive Model
Julia implementation of vector autoregressive (VAR(p)) model with closed form solution - OLS estimation. The code for confidence band primarily follows estimation used by Kilian (2009, American Economic Review).

To run the function, following packages are required and updated to-date.
```julia
import Pkg;
Pkg.update();

using ProgressMeter;
using DataFrames, DelimitedFiles;
using LinearAlgebra, Statistics, Distributions;
using Gadfly, Colors;

# include(pwd() * "//src//func_VectorAR.jl")

# Data from Kilian (2009), into DataFrame
y = readdlm("kilian_2009_aer.txt"); 

# Define parameters
p = 24;                            # Lag order
h = 15;                            # Horizon - IRF
ℏ = 12;                            # Horizon - FEVDC
𝚩, 𝞄, 𝝨 = func_VAR(y, p);          # Coefficient matrix, 
                                   # residuals, 
                                   # and covariance matrix
```
