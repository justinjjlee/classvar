# julia-VectorAR.jl
Julia implementation of vector autoregressive (VAR(p)) model with closed-form solution.
Calculations of the impulse response function calculation with wild-bootstrap estimation, along with forecast error variance decomposition (FEVDC).


To run the function, following packages are required and updated to-date.
```julia
import Pkg;
Pkg.update();

using ProgressMeter;
using LinearAlgebra, Statistics, Distributions;
using Gadfly, Colors;

include("func_VectorAR.jl")
```
To test the function, we can use Kilian (2009, American Economic Review) and replicate the results. The data was publicly available and  downloaded from [American Economic Association](https://www.aeaweb.org/articles?id=10.1257/aer.99.3.1053).

```julia
y = open("kilian_2009_aer.txt");   # Data from Kilian (2009)
p = 12;                            # Lag order

𝚩, 𝞄, 𝝨 = func_VAR(y, p);          # Coefficient matrix, residuals, and covariance matrix
ψ, 
  ψ_lb_2sd, ψ_lb_1sd, 
  ψ_ub_1sd, ψ_ub_2sd,
  FEVDC = func_IRFvar(y, p);       # Results from impulse responses, bootstrap CI band, and FEVDC
```

-Justin J. Lee
