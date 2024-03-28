# Johansen MLE of VECM with unknown cointegrating vector and unrestricted intercept
# Original code provided by Kilian and Lütkepohl
#   Website of Lutz Kilian, translated from MATLAB code

cd(@__DIR__)
str_dir_git_func = splitdir(splitdir(pwd())[1])[1]

using Pkg
using CSV, DataFrames
using StaticArrays, LinearAlgebra
# Import function
include(str_dir_git_func * "/src/func_aux.jl")
include(str_dir_git_func * "/src/func_VECM.jl")

# Import and set data
ffr = DataFrame(CSV.File(pwd() * "/kilian_lutkepohl/fedfunds.txt"), [:yr, :mo, :ffr])
gdf = DataFrame(CSV.File(pwd() * "/kilian_lutkepohl/gnpdeflator.txt"), [:yr, :mo, :gnp])
inf = dif_lvl(log.(gdf[:,3]))*100; 
gnp = DataFrame(CSV.File(pwd() * "/kilian_lutkepohl/realgnp.txt"), [:yr, :mo, :gnp])
gro = dif_lvl(log.(gnp[:,3]))*100; 

# Federal Funds Rate to be period-ending value
irate=[];
for i=1:3:length(ffr[:,3])
    try
        irate=[irate; mean(ffr[i:i+2,3])];
    catch
        # Last couple periods, ignore
        break
        #irate=[irate; mean(ffr[i:end,3])];
    end
end

# Collect DataFrame
y    = convert(Array{Float64}, [gro irate inf])
t, K = size(y)

p = 4;   # VAR lag order
r = 2;   # cointegration rank      

α, β, Γ, Lam, Σ, Y, ΔY, X = vecm(y, p, r)

ν = Γ[:,1]

α

β'

Γ₁ = Γ[: ,2:4]
Γ₂ = Γ[:, 5:7]
Γ₃ = Γ[:, 8:10]

Σ

# Testing fitness
ŷ = α*β' * Y + Γ * X

using Plots

plot(ŷ[1,:])
plot!(ΔY[1,:])

plot(cumsum(ŷ[1,:]))
plot!(cumsum(ΔY[1,:]))

plot(cumsum(ŷ[2,:]))
plot!(cumsum(ΔY[2,:]))

plot(ŷ[3,:])
plot!(ΔY[3,:])

# Horizon sought for impulse response function
h = 20
ψ = func_IRFvecm(y, p, r, h)

plot(cumsum(ψ[2,3,:]))