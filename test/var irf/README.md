# Impulse Response Functions
Following examples are variations and examples of computing various identified impulse response functions.

## Short-run Impulse Response Function
To test the function, we use Kilian (2009, American Economic Review) and replicate the results. The data was publicly available and  downloaded from [American Economic Association](https://www.aeaweb.org/articles?id=10.1257/aer.99.3.1053).

```julia
# Data from Kilian (2009), into DataFrame
y = readdlm("kilian_2009_aer.txt"); 

# Define parameters
p = 24;                            # Lag order
h = 15;                            # Horizon - IRF
ℏ = 12;                            # Horizon - FEVDC
𝚩, 𝞄, 𝝨 = func_VAR(y, p);          # Coefficient matrix, 
                                   # residuals, 
                                   # and covariance matrix
ψ,
  ψ_lb_1sd, ψ_ub_1sd,
  ψ_lb_2sd, ψ_ub_2sd = func_IRFvar(y, p, h, ℏ);       
# Results:
# ψ: point estimate of impulse responses
# ↪ Default → unit shock estimate,
# ↪ Standard deviation shock → edit estimation procedure to multiply ρ
# ψ_ub_(#)sd Upper-bound bootstrap confidence interval
# ψ_lb_(#)sd lower-bound bootstrap confidence interval
# ↪ with standard error level of confidence
```
## Long-run Impulse Response Function (Blanchard and Quah (1989))
For a long-run impact measure, we employ Blanchard and Quah [(1989, American Economic Review)](https://www.jstor.org/stable/1827924?seq=1),

```julia
ψ,
  ψ_lb_1sd, ψ_ub_1sd,
  ψ_lb_2sd, ψ_ub_2sd = func_IRFvar_LR(data, p, h);  
```
## Plot Examples

```julia
# Plot impulse response, response of 2nd variable to 3rd shock
i, j = 2,3;

irf_lnt = (ψ[i,j,1:h])
irf_ub = (ψ_ub_1sd[i,j,1:h,1])
irf_lb = (ψ_lb_1sd[i,j,1:h,1])
plt_irf = plot(x = 1:h, y = irf_lnt,
     ymin = irf_lb, ymax = irf_ub,
     yintercept = [0],
     Geom.line, Geom.ribbon,
     Theme(lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.2)),
          Geom.hline(color=["black"], size=[0.5mm]),
     Guide.xlabel("Days since impact"),
     Guide.ylabel("Response (percentage point)")
     );

# Saving the plot
draw(PNG(pwd * "\\results\\irf_$(i)_$(j).png"), plt_irf)
```

## Generalized Impulse Response Function
In the sub-folder `yieldcurve`, I compute generalized impulse response function for yield curve response between United States (Treasury) and estimated Eurozone yield curve (European Central Bank).
