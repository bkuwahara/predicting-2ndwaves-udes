cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using LinearAlgebra
using JLD2, FileIO, CSV
using Dates
using Plots
using Statistics
using Distributions
using Lux, Zygote, OptimizationFlux
using Random; rng = Random.default_rng()

const sample_period=7
const train_length = 160
const mobility_min = -1.0
const mobility_baseline = 0.0
const mobility_max = 2.0
const num_indicators = 1
const hidden_dims = 3
const ϵ=0.001


indicator_names = Dict(
	3 => "rr",
	4 => "wk",
	5 => "pk",
	6 => "si"
)

loss_labels = [
	"accuracy"; 
	"beta_mon"; 
	"beta_pos"; 
	"f_lb";
	"f_ub";
	"f_basetend";
	"basetend_mon_M";
	"basetend_mon_R";
	"f_mon_I";
	"f_mon_deltaI";
	 ]

network1 = Lux.Chain(
	Lux.Dense(num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>1))
network2 = Lux.Chain(
	Lux.Dense(3+num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>num_indicators))

p1_temp, st1 = Lux.setup(rng, network1)
p2_temp, st2 = Lux.setup(rng, network2)

function get_inputs(n, yscale)
	I_ = rand(rng, Uniform(0.0, 1.0), 1, n)
	ΔI = rand(rng, 1, n) .+ (I_ .- 1)
	R = rand(rng, 1, n) .* (1 .- I_)
	M = rand(rng, Uniform(mobility_min, 2*mobility_baseline-mobility_min), 1, n)

	return I_/yscale[2], ΔI/yscale[2], M/yscale[3], R/yscale[1]
end


function l_layer1_avg(p, Ms)
	beta_i = network1(Ms, p, st1)[1]
	beta_j = network1(Ms .+ ϵ, p, st1)[1]

	l1 = sum(relu.( -beta_i))
	l5 = sum(relu.( beta_i .- beta_j))

	return [l1, l5]./length(Ms)
end



function l_layer2(p, Is, ΔIs, Ms, Rs)
	n_pts = size(Is,2)
	M_min = mobility_min .* ones(1, n_pts)
	I_baseline = zeros(1, n_pts)
	M_max = 2*(mobility_baseline - mobility_min) .* ones(1,n_pts)
	# Must not decrease when M at M_min
	dM_min = network2([M_min; Is; ΔIs; Rs], p, st2)[1]
	l6 = sum(relu.(-dM_min))

	# Must not increase when M at 2*(mobility_baseline - mobility_min)
	dM_max = network2([M_max; Is; ΔIs; Rs], p, st2)[1]
	l6 = sum(relu.(dM_max))

	# Tending towards M=mobility_baseline when I == ΔI == 0
	dM1_baseline = network2([Ms; I_baseline; I_baseline; Rs], p, st2)[1]
	dM2_baseline = network2([Ms .+ ϵ; I_baseline; I_baseline; Rs], p, st2)[1]
	dM3_baseline = network2([Ms; I_baseline; I_baseline; Rs .+ ϵ], p, st2)[1]
	l2 = sum(relu, relu.(dM1_baseline .* (Ms .- mobility_baseline)))

	# Stabilizing effect is stronger at more extreme M and higher R
	sgn_M = abs.(Ms) .- abs.(Ms .+ ϵ)
	sgn_dM = abs.(dM1_baseline) .- abs.(dM2_baseline)
	l3 = sum(relu.(-1 .* sgn_M .* sgn_dM))
	l7 = sum(relu.(abs.(dM1_baseline) .- abs.(dM3_baseline)))

	## Monotonicity terms
	dM_initial = network2([Ms; Is; ΔIs; Rs], p, st2)[1]

	# f monotonically decreasing in I
	dM_deltaI = network2([Ms; (Is .+ ϵ); ΔIs; Rs], p, st2)[1]
	l3 = sum(relu.(dM_deltaI .- dM_initial))
	
	# f monotonically decreasing in ΔI
	dM2_delta_deltaI = network2([Ms; Is; ΔIs .+ ϵ; Rs], p, st2)[1]
	l4 = sum(relu.(dM2_delta_deltaI .- dM_initial))
	return [l2, l3, l4, l6, l7]./length(Ms)
end





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



function analyze(sim_name, loss_idxs...)
	root = datadir("sims", "udde", sim_name)
	filenames = readdir(root)

	for fname in filenames
		results = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))
		scale = results["scale"]
		losses = results["losses"]
		pred = results["prediction"]
		hist_data = results["hist_data"]
		train_data = results["train_data"]
		test_data = results["test_data"]
		β = results["betas"]


		mobility_baseline = 0.0
		mobility_min = -1.0

		all_data = [train_data test_data]
		data_tsteps = range(0.0, step=sample_period, length=size(all_data,2))
		pred_tsteps = range(0.0, step=1.0, length=size(pred, 2))
		pred_tsteps_short = filter(x -> x <= data_tsteps[end], collect(pred_tsteps))

		pl_pred_test = scatter(data_tsteps, all_data', label=["True data" nothing nothing],
			color=:black, layout=(size(all_data,1), 1))
		plot!(pl_pred_test, pred_tsteps_short, pred[:,1:length(pred_tsteps_short)]', label=["Prediction" nothing nothing],
			color=:red, layout=(size(all_data,1), 1))
		vline!(pl_pred_test[end], [0.0 train_length], color=:black, style=:dash,
		label=["Training" "" "" "" ""])
		for i = 1:size(all_data,1)
			vline!(pl_pred_test[i], [0.0 train_length], color=:black, style=:dash,
			label=["" "" "" "" "" "" ""])
		end

		pl_pred_lt = plot(range(0.0, step=1.0, length=size(pred,2)), pred', label=["Long-term Prediction" nothing nothing nothing nothing nothing],
			color=:red, layout=(size(all_data,1), 1))
		vline!(pl_pred_lt, [0.0 train_length], color=:black, style=:dash,
		label=["Training" "" "" "" ""])
		for i = 1:size(all_data,1)
			vline!(pl_pred_lt[i], [0.0 train_length], color=:black, style=:dash,
			label=["" "" "" "" "" "" ""])
		end


		# beta dose-response curve
		pl_beta_response = plot(range(mobility_min, step=0.1, stop=2*(mobility_baseline - mobility_min)), β', xlabel="M", ylabel="β", 
			label=nothing, title="Force of infection response to mobility")
		vline!(pl_beta_response, [minimum(train_data[3,:]) maximum(train_data[3,:])], color=:black, label=["Training range" nothing], 
			style=:dash)

		# Net loss plot
		l_net = losses[1:1,:] + sum(10*ones(9).*losses[2:end,:], dims=1)
		pl_losses = plot(l_net')
		yaxis!(pl_losses, :log10)

		for idx in loss_idxs
			pl_temp = plot(losses[idx,:])
			# yaxis!(pl_temp, :log10)
			savefig(pl_temp, datadir("sims", "udde", sim_name, fname, "loss_$idx.png"))
		end

		savefig(pl_losses, datadir("sims", "udde", sim_name, fname, "net_losses.png"))
		savefig(pl_pred_test, datadir("sims", "udde", sim_name, fname, "test_prediction.png"))
		savefig(pl_pred_lt, datadir("sims", "udde", sim_name, fname, "long_term_prediction.png"))
		savefig(pl_beta_response, datadir("sims", "udde", sim_name, fname, "beta_response.png"))

	end
	nothing
end




# function loss_analysis(sim_name)
root = datadir("sims", "udde", sim_name)
filenames = readdir(root)

all_losses = zeros(8, length(filenames))

	# for fname in filenames
results = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))
scale = results["scale"]
losses = results["losses"]
pred = results["prediction"]
hist_data = results["hist_data"]
train_data = results["train_data"]
test_data = results["test_data"]
β = results["betas"]
p = results["p"]

acc_loss = minimum(losses[1,:])

Is, ΔIs, Ms, Rs = get_inputs(1000000, size(train_data, 2)*scale)
mean_layer1_loss = l_layer1_avg(p.layer1, Ms)
mean_layer2_loss = l_layer2(p.layer2, Is, ΔIs, Ms, Rs)

mean_loss = [acc_loss; mean_layer1_loss; mean_layer2_loss]
all_losses[:,1] = mean_loss



# end


nothing
end