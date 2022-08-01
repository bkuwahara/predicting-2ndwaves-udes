cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using LinearAlgebra
using JLD2, FileIO
using Dates
using Plots
using Statistics
using Lux
using Distributions


indicator_names = Dict(
	3 => "rr",
	4 => "wk",
	5 => "pk",
	6 => "si"
)



function EnsemblePrediction(region::String, hdims::Int, τₘ::AbstractFloat, τᵣ::AbstractFloat, loss_weights::Tuple{Int, Int, Int})
	root = datadir("sims", "udde", region)
	indicator_name = "rr"
	params = (taum = τₘ, taur = τᵣ, hdims=hdims)
	param_name = savename(params)
	weight_name = "weight=$(loss_weights[1])-$(loss_weights[2])-$(loss_weights[3])"

	fname = "$(indicator_name)_$(param_name)_$(weight_name)"
	filenames = filter(s->rsplit(s, "_", limit=2)[1] == fname, readdir(root))

	f = load(joinpath(root, filenames[1])* "/results.jld2")
	pred = f["prediction"]
	for fname in filenames[2:end]
		f = load(joinpath(root, fname)* "/results.jld2")
		pred = hvncat(3, pred, f["prediction"])
	end

	mean_pred = mean(pred, dims=3)
	sd_pred = std(pred, dims=3)
	return mean_pred[:,:,1], sd_pred[:,:,1]
end



function EnsemblePlot(region::String, hdims::Int, τₘ::AbstractFloat, τᵣ::AbstractFloat, loss_weights::Tuple{Int, Int, Int}; 
		CI=0.95, showplot=true, saveplot=true)
	
	μ, sd = EnsemblePrediction(region, hdims, τₘ, τᵣ, loss_weights)

	indicators = [3]
	dataset = load(datadir("exp_pro", "SIMX_7dayavg_roll=false_$(region).jld2"))
	data = dataset["data"][hcat([1 2], indicators), :][1,:,:]
	days = dataset["days"]
	all_tsteps = range(0, step=7, length=size(data,2))
	pl = scatter(all_tsteps, data', layout=(size(data,1), 1), color=:black, label=["Data" nothing nothing nothing nothing nothing],
		ylabel=hcat(["S" "I"], reshape([indicator_names[i] for i in indicators], 1, length(indicators))))

	pred_tsteps = range(max(τᵣ+1, τₘ), length=size(μ,2), stop=all_tsteps[end])
	q = quantile(Normal(), (CI+1)/2)
	ribbon_lower = zeros(size(μ))
	ribbon_lower[1,:] .= q*sd[1,:]
	ribbon_lower[3,:] .= q*sd[3,:]
	ribbon_lower[2,:] .= [μ[2,i] - q*sd[2,i] < 0 ? μ[2,i] : q*sd[2,i] for i in eachindex(μ[2,:])]
	ribbon_upper = q*sd'
	plot!(pl, pred_tsteps, μ', color=:red, ribbon = (ribbon_lower', ribbon_upper), label=["Prediction ($(100*CI)% CI" nothing nothing nothing nothing nothing])

	train_stop = Int(round(0.75*90))
	for i in 1:size(data, 1)-1
		vline!(pl[i], [15+train_stop], color=:black, style=:dash, label=nothing)
	end
	vline!(pl[end], [15+train_stop], color=:black, style=:dash, label="End of training data")

	xlabel!(pl[end], "Time (days since $(days[1]))")
	title!(pl[1], "Ensemble prediction, LSTM model, $(region)")


	if showplot
		display(pl)
	end
	if saveplot
		indicator_name = "rr"
		param_name = "hdims=$(hdims)_taum=$(τₘ)_taur=$(τᵣ)"
		weight_name = "weight=$(loss_weights[1])-$(loss_weights[2])-$(loss_weights[3])"
		fname = "ensemble_$(indicator_name)_$(param_name)_$(weight_name)"
		savefig(pl, datadir("sims", "udde", region, fname*".png"))
	end
	return nothing	
end

mu, sd = EnsemblePrediction("US-NY", 2, 10.0, 14.0, (2, 1, 1))
EnsemblePlot("US-NY", 2, 10.0, 14.0, (2, 1, 1))