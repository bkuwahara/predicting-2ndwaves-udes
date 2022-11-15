cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using LinearAlgebra
using JLD2, FileIO, CSV, DataFrames
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
	"f_ub";
	"f_basetend";
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


function l_layer2_avg(p, Is, ΔIs, Ms, Rs)
	n_pts = size(Is,2)
	I_baseline = zeros(1, n_pts)
	M_max = 2*(mobility_baseline - mobility_min) .* ones(1,n_pts)

	# Must not increase when M at 2*(mobility_baseline - mobility_min)
	dM_max = network2([M_max; Is; ΔIs; Rs], p, st2)[1]
	l6 = sum(relu.(dM_max))

	# Tending towards M=mobility_baseline when I == ΔI == 0
	dM1_baseline = network2([Ms; I_baseline; I_baseline; Rs], p, st2)[1]
	dM2_baseline = network2([Ms .+ ϵ; I_baseline; I_baseline; Rs], p, st2)[1]
	dM3_baseline = network2([Ms; I_baseline; I_baseline; Rs .+ ϵ], p, st2)[1]
	l2 = sum(relu, relu.(dM1_baseline .* (Ms .- mobility_baseline)))

	# Stabilizing effect is stronger at more extreme M and higher R
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





function EnsembleSummary(sim_name)
	root = datadir("sims", "udde", sim_name)

	filenames = filter(f -> isdir(root*"\\"*f), readdir(root))
	f = load(joinpath(root, filenames[1])* "/results.jld2")
	pred = f["prediction"]
	betas = f["betas"]
	for fn in filenames[2:end]
		f = load(joinpath(root, fn)* "/results.jld2")
		new_pred = f["prediction"]
		new_betas = f["betas"]
		if size(new_pred, 2) != size(pred, 2)
			new_pred =  hcat(new_pred, NaN .*zeros(size(new_pred,1), size(pred, 2)-size(new_pred, 2)))
		end
		pred = hvncat(3, pred, new_pred)
		betas = hvncat(3, betas, new_betas)
	end

	mean_pred = mean(pred, dims=3)
	med_pred = median(pred, dims=3)[:,:,1]
	qu_pred = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.75) for i in axes(pred,1), j in axes(pred,2)]
	ql_pred = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.25) for i in axes(pred,1), j in axes(pred,2)]

	med_betas = median(betas, dims=3)[:,:,1]
	qu_betas = [quantile(betas[i,j,:], 0.75) for i in axes(betas,1), j in axes(betas,2)]
	ql_betas = [quantile(betas[i,j,:], 0.25) for i in axes(betas,1), j in axes(betas,2)]


	region = split(sim_name, "_")[end]
	indicators=[3]
	dataset = load(datadir("exp_pro", "SIMX_final_7dayavg_roll=false_$(region).jld2"))
	data = dataset["data"][hcat([1 2], indicators), :][1,:,:]
	days = dataset["days"]
	all_tsteps = range(0, step=7, length=size(data,2))
	pl_pred = scatter(all_tsteps, data', layout=(size(data,1), 1), color=:black, label=["Data" nothing nothing nothing nothing nothing],
		ylabel=hcat(["S" "I"], reshape([indicator_names[i] for i in indicators], 1, length(indicators))))



	pred_tsteps = range(14.0, step=1.0, length=size(med_pred,2))
	plot!(pl_pred, pred_tsteps, med_pred', color=:red, ribbon = ((med_pred-ql_pred)', (qu_pred-med_pred)'), label=["Median prediction (IQR)" nothing nothing])
	ylims!(pl[3], (minimum(data[3,:]) - 2, maximum(data[3,:]) + 2))
	xlabel!(pl[end], "Time (days since $(days[1]))")
	title!(pl[1], "Ensemble prediction, $(region)")



	M_test = range(mobility_min, step=0.1, stop=2*(mobility_baseline - mobility_min))
	pl_betas = plot(M_test, med_betas', ribbon=((med_betas-ql_betas)', (qu_betas-med_betas)'), label=nothing)
	xlabel!(pl_betas, "M")
	ylabel!(pl_betas, "β")
	vline!(pl_betas, [minimum(data[3,:]) maximum(data[3,:])], color=:black, label=["Observed range" nothing], 
				style=:dash)

	if showplot
		display(pl_pred)
		display(pl_betas)
	end
	if saveplot
		savefig(pl_pred, datadir("sims", "udde", sim_name, "ensemble_pred.png"))
		savefig(pl_betas, datadir("sims", "udde", sim_name, "ensemble_beta_response.png"))

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
		l_net = losses[1:1,:] + sum(10*ones(7).*losses[2:end,:], dims=1)
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




function loss_analysis(sim_name)
	root = datadir("sims", "udde", sim_name)
	filenames = readdir(root)

	all_losses = zeros(8, length(filenames))
	loss_weight = 0

	for (i, fname) in enumerate(filenames)
		
		results = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))
		scale = results["scale"]
		losses = results["losses"]
		pred = results["prediction"]
		hist_data = results["hist_data"]
		train_data = results["train_data"]
		test_data = results["test_data"]
		β = results["betas"]
		p = results["p"]

		if i == 1
			loss_weight = results["loss_weights"]
		end

		acc_loss = minimum(losses[1,:])

		Is, ΔIs, Ms, Rs = get_inputs(1000000, size(train_data, 2)*scale)
		mean_layer1_loss = l_layer1_avg(p.layer1, Ms)
		mean_layer2_loss = l_layer2_avg(p.layer2, Is, ΔIs, Ms, Rs)

		mean_loss = [acc_loss; mean_layer1_loss; mean_layer2_loss]
		all_losses[:,i] = mean_loss

	end

	med_losses = reshape(median(all_losses, dims=2), size(all_losses,1))
	summary = Dict{String, Any}(zip(["med_" * label for label in loss_labels], med_losses))


	net_losses = all_losses[1,:] .+ transpose(loss_weight .* sum(all_losses[2:end,:], dims=1))
	best_loss, best_ind = findmin(net_losses)
	best_ver = parse(Int64, rsplit(filenames[best_ind[1]], "v")[end])
	summary["best_ver"] = best_ver

	df = DataFrame(summary)
	CSV.write(root*"\\loss_summary.csv", df)

	converged_idxs = reshape(reduce(*, all_losses .<= 0.05, dims=1), length(filenames))
	converged_sims = filenames[converged_idxs]
	save(root * "\\converged_sims.jld2", "sims", converged_sims)

	nothing
end