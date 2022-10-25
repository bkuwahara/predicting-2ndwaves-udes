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



function EnsembleSummary(sim_name::String, region::String, hdims::Int, τₘ::AbstractFloat, τᵣ::AbstractFloat)
	root = datadir("sims", "udde", sim_name)
	indicator_name = "rr"
	params = (taum = τₘ, taur = τᵣ, hdims=hdims)
	param_name = savename(params)

	fname = "$(region)_$(indicator_name)_$(param_name)"
	filenames = filter(s->rsplit(s, "_", limit=2)[1] == fname, readdir(root))
	f = load(joinpath(root, filenames[1])* "/results.jld2")
	pred = f["prediction"]
	for fn in filenames[2:end]
		f = load(joinpath(root, fn)* "/results.jld2")
		new_pred = f["prediction"]
		if size(new_pred, 2) != size(pred, 2)
			new_pred =  hcat(new_pred, NaN .*zeros(size(new_pred,1), size(pred, 2)-size(new_pred, 2)))
		end
		pred = hvncat(3, pred, new_pred)
	end

	mean_pred = mean(pred, dims=3)
	sd_pred = std(pred, dims=3)
	med_pred = median(pred, dims=3)
	qu = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.75) for i in axes(pred,1), j in axes(pred,2)]
	ql = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.25) for i in axes(pred,1), j in axes(pred,2)]
	
	return mean_pred[:,:,1], med_pred[:,:,1], sd_pred[:,:,1], qu, ql
end



function EnsemblePlot(sim_name::String, region::String, hdims::Int, τₘ::AbstractFloat, τᵣ::AbstractFloat;
	showplot=true, saveplot=true)
	
	μ, med, sd, qu, ql = EnsembleSummary(sim_name, region, hdims, τₘ, τᵣ)

	indicators = [3]
	dataset = load(datadir("exp_pro", "SIMX_7dayavg_roll=false_$(region).jld2"))
	data = dataset["data"][hcat([1 2], indicators), :][1,:,:]
	days = dataset["days"]
	all_tsteps = range(0, step=7, length=size(data,2))
	pl = scatter(all_tsteps, data', layout=(size(data,1), 1), color=:black, label=["Data" nothing nothing nothing nothing nothing],
		ylabel=hcat(["S" "I"], reshape([indicator_names[i] for i in indicators], 1, length(indicators))))

	pred_tsteps = range(max(τᵣ+1, τₘ), length=size(μ,2), stop=all_tsteps[end])

	plot!(pl, pred_tsteps, med', color=:red, ribbon = ((med-ql)', (qu-med)'), label=["Median prediction (IQR)" nothing nothing nothing nothing nothing])
    ylims!(pl[3], (minimum(data[3,:]) - 2, maximum(data[3,:]) + 2))
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
		fname = "ensemble_$(region)_$(indicator_name)_$(param_name)"
		savefig(pl, datadir("sims", "udde", sim_name, fname*".png"))
	end
	return nothing	
end


for region in ["CA" "PA" "NY"]
	EnsemblePlot("test_US", "US-$region", 3, 10.0, 14.0)
end

for region in ["ON" "BC" "QC"]
	EnsemblePlot("test_CA", "CA-$region", 3, 10.0, 14.0)
end

for region in ["NL" "UK"]
	EnsemblePlot("test_EU", region, 3, 10.0, 14.0)
end

EnsemblePlot("ensemble_UK", "UK", 3, 10.0, 14.0)



#==========================================================
Temporary plotting functions
===========================================================#

pl_pred_test = scatter(all_tsteps, all_data', label=["True data" nothing nothing nothing nothing nothing],
color=:black, layout=(2+num_indicators, 1))
plot!(pl_pred_test, pred_test.t, Array(pred_test)', label=["Prediction" nothing nothing nothing nothing nothing],
color=:red, layout=(2+num_indicators, 1))
vline!(pl_pred_test[end], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
label=["Training" "" "" "" ""])
for i = 1:2+num_indicators-1
vline!(pl_pred_test[i], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
	label=["" "" "" "" "" "" ""])
end



pl_pred_lt = plot(pred_lt.t, Array(pred_lt)', label=["Long-term Prediction" nothing nothing nothing nothing nothing],
color=:red, layout=(2+num_indicators, 1))
vline!(pl_pred_lt[end], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
label=["Training" "" "" "" ""])
for i = 1:2+num_indicators-1
vline!(pl_pred_lt[i], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
	label=["" "" "" "" "" "" ""])
end

# beta time series
indicators_predicted = pred_test[3:end,:]
β = zeros(size(pred_test.t))
for i in 1:length(β)
indicator = i <= length(hist_split) ? hist_data[3:end,1] : indicators_predicted[:, i-length(hist_split)]
β[i] = network1(indicator, p_trained.layer1, st1)[1][1]
end
pl_beta_timeseries = plot(range(-length(hist_data), length=length(β), stop=all_tsteps[end]), β,
xlabel="t", ylabel="β", label=nothing, title="Predicted force of infection over time")


# beta dose-response curve
β = [network1([M], p_trained.layer1, st1)[1][1] for M in range(mobility_min, step=0.1, stop=5.0)]
pl_beta_response = plot(range(mobility_min, step=0.1, stop=5.0), β, xlabel="M", ylabel="β", 
label=nothing, title="Force of infection response to mobility")
vline!(pl_beta_response, [mobility_baseline], color=:red, label="Baseline", style=:dot,
legend=:topleft)
vline!(pl_beta_response, [minimum(train_data[3,:]) maximum(train_data[3,:])], color=:black, label=["Training range" nothing], 
style=:dash)

savefig(pl_pred_test, datadir("sims", model_name, sim_name, fname, "test_prediction.png"))
savefig(pl_pred_lt, datadir("sims", model_name, sim_name, fname, "long_term_prediction.png"))
savefig(pl_beta_timeseries, datadir("sims", model_name, sim_name, fname, "beta_timeseries.png"))
savefig(pl_beta_response, datadir("sims", model_name, sim_name, fname, "beta_response.png"))
