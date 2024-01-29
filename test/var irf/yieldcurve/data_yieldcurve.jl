# API process to pull US Treasury and Euro Area Yield Curve

using Pkg
using CSV, DataFrames, JSON, FredData
cd(@__DIR__)
# Data location in application
str_dir_git_func = splitdir(splitdir(splitdir(pwd())[1])[1])[1]
str_dir_git = splitdir(str_dir_git_func)[1]
include(str_dir_git_func *"/src/func_VectorAR.jl")

# Download the datasets: 
# API call to ECB: Euro Area (EA) Yield curve
# Key: YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y

# See example: https://data.ecb.europa.eu/help/data-examples
ecb_api_call = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata"
ea_yield = CSV.read(download(ecb_api_call), DataFrame)

ea_yield = ea_yield[:, ["TIME_PERIOD", "OBS_VALUE"]]
rename!(ea_yield, [:date, :ea_yield])

# Import credential locally for FRED API
apikeys = JSON.parsefile(str_dir_git*"/credential.json")
api_key_fred = apikeys["credentials"]["api_key_fred"]

api_fred = Fred(api_key_fred);

us_yield = FredData.get_data(api_fred, "T10Y2Y"; observation_start = "2004-09-06")
us_yield = us_yield.data[:,["date", "value"]]
rename!(us_yield, [:date, :us_yield])

# Join the two data set 
yield = innerjoin(ea_yield, us_yield, on = :date)
# Drop missing value (holiday US)
yield = filter(row -> all(x -> !(x isa Number && isnan(x)), row), yield)

# save the data for copy
CSV.write("yieldcurve_usea.csv", yield)

## ===== Test out the response with the daily data
yield = CSV.read("yieldcurve_usea.csv", DataFrame)
matdf = Matrix(yield[:,2:end])

ψ, ψo, ψg = gIRF(matdf, 12, 24)

# Response of US on EA shock
plot(ψo[2,1,:])
plot(ψg[2,1,:])

# Response of EA on US shock
plot(ψo[1,2,:])
plot(ψg[1,2,:])