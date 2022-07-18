cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")

using LinearAlgebra
using JLD2, FileIO
using Dates
using Plots
using Statistics
using Lux

indicator_names = Dict(
	3 => "rr",
	4 => "wk",
	5 => "pk",
	6 => "si"
)


function EnsemblePrediction(region, seqlen, hdims, indicators...)
	root = datadir("sims", "discrete_model_LSTM", region)

	indicator_name = ""
	for j=1:length(indicators)-1
		indicator_name = indicator_name * indicator_names[indicators[j]] * "-"
	end
	indicator_name = indicator_name * indicator_names[indicators[end]]
	param_name = "hdims=$(hdims)_seqlen=$(seqlen)"
	fname = "$(indicator_name)-$(param_name)"

	filenames = filter(s->rsplit(s, "_", limit=2)[1] == fname, readdir(root))

	f = load(joinpath(root, filenames[1])* "/results.jld2")
	pred = f["prediction"]
	for fname in filenames[2:end]
		f = load(joinpath(root, fname)* "/results.jld2")
		pred = hvncat(3, pred, f["prediction"])
	end

	mean_pred = mean(pred, dims=3)
	sd_pred = std(pred, dims=3)
	return pred
end
