using DrWatson
@quickactivate("S2022_Project")
using Lux, DiffEqFlux, Zygote
using Optimisers
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using Plots
using Statistics, Distributions
using Random; rng = Random.default_rng()




#=================================
constant hyperparameters
=================================#
sample_period=7
τₘ = 10.0 # 14, 21, 28, 10, 25
τᵣ = 14.0 # 10, 14
const train_length = 140
const maxiters = 2500
const η = 1e-3
const recovery_rate = 1/4
const indicators = [3]
const ϵ=0.01
activation = relu
adtype = Optimization.AutoZygote()

#===============================================
Input hypterparameters
================================================#

function default_setup()
	region="US-NY"
	sim_name = region
	hidden_dims = 3
	loss_weights = (1, 50, 50)
	return sim_name, region, hidden_dims, loss_weights
end
sim_name, region, hidden_dims, loss_weights = default_setup()
if ARGS != []
	sim_name = ARGS[1]
	region = ARGS[2]
	hidden_dims = parse(Int, ARGS[3])
	loss_weights = parse.(Int, ARGS[4:end])
end
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1
verbose = (isempty(ARGS)) ? true : false
model_name = "udde"



#===========================================
Run the model
===========================================#
function run_model()
	println("Starting run: $(region)")
	#===============================================
	Load data
	================================================#
	dataset = load(datadir("exp_pro", "SIMX_7dayavg_roll=false_$(region).jld2"))
	all_data = dataset["data"][hcat([1 2], indicator_idxs), :][1,:,:]
	days = dataset["days"]
	μ_mobility = dataset["mobility_mean"][indicator_idxs .- 2][1]
	sd_mobility = dataset["mobility_std"][indicator_idxs .- 2][1]
	mobility_baseline = -μ_mobility/sd_mobility

	I_domain = (0.0, 1.0)
	ΔI_domain = 10 .*(minimum(all_data[2,2:end] - all_data[2,1:end-1]), maximum(all_data[2,2:end] - all_data[2,1:end-1]))
	M_domain = (-100.0, 100.0)

	# Split the rest into pre-training (history), training and testing
	hist_stop = Int(round(max(τᵣ+1, τₘ)/sample_period))
	hist_split = 1:hist_stop
	train_split = hist_stop+1:hist_stop + div(90, sample_period)
	test_split = hist_stop+div(90, sample_period)+1:size(all_data, 2)


	hist_data = all_data[:, hist_split]
	train_data = all_data[:, train_split]
	test_data = all_data[:, test_split]

	all_tsteps = range(-max((τᵣ+1), τₘ), step=sample_period, length=size(all_data,2))
	hist_tspan = (all_tsteps[1], 0.0)
	t_train = range(0.0, step=sample_period, length=length(train_split))
	t_test = range(t_train[end]+sample_period, step=sample_period, length=length(test_split))

	#===============================================
	Set up model
	================================================#
	# Get scale factors from the data to improve training
	yscale = maximum(train_data, dims=2) .- minimum(train_data, dims=2);
	tscale = t_train[end] - t_train[1];
	scale = yscale/tscale;

	network1 = Lux.Chain(
		Lux.Dense(num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>1))
	network2 = Lux.Chain(
		Lux.Dense(2+num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>num_indicators))
	# network3 = Lux.Chain(
		# Lux.Dense(2=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators))

	# UDDE
	function udde(du, u, h, p, t)
		I_hist = h(p, t-τᵣ)[2]/yscale[2]
		delta_I_hist = (I_hist - h(p, t-(τᵣ+1))[2])/yscale[2]
		du[1] = -u[1]*u[2]*network1(h(p, t-τₘ)[3:end], p.layer1, st1)[1][1]
		du[2] = -du[1] - recovery_rate*u[2]
		du[3] = network2([u[3]; I_hist; delta_I_hist], p.layer2, st2)[1][1] #network2([u[3]], p.layer2, st2)[1][1] - 
		nothing
	end


	p1, st1 = Lux.setup(rng, network1)
	p2, st2 = Lux.setup(rng, network2)
	# p3, st3 = Lux.setup(rng, network3)

	p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2))
	u0 = train_data[:,1]
	h(p,t) = [1.0; 0.0; mobility_baseline]

		
	prob_nn = DDEProblem(udde, u0, h, (0.0, t_train[end]), p_init, constant_lags=[τᵣ τₘ (τᵣ+1)])


	function predict(θ, tspan; u0=u0, saveat=sample_period)
		prob = remake(prob_nn, tspan = tspan, p=θ, u0=u0)
		Array(solve(prob, MethodOfSteps(Tsit5()), saveat=saveat))
	end

	function loss(θ, tspan)
		pred = predict(θ, tspan)
		if size(pred, 2) < abs(tspan[2] - tspan[1])/sample_period
			return Inf
		else
			return sum(abs2, (pred .- train_data[:, 1:size(pred, 2)])./yscale)/size(pred, 2), pred
		end
	end

	function loss_network1(M_samples, p, st)
		loss_negativity = 0
		loss_monotonicity = 0
		for i in eachindex(M_samples)[1:end]
			βi = network1([M_samples[i]], p, st)[1][1]
			βj = network1([M_samples[i]+ϵ], p, st)[1][1]
			sgn = βj - βi
			if sgn < 0
				loss_monotonicity += abs(sgn)
			end
			if βi < 0
				loss_negativity += abs(βi)
			end
		end
		return loss_monotonicity + loss_negativity
	end


	function loss_network2(M_samples, I_samples, ΔI_samples, p, st)
		loss_stability = 0
		loss_monotonicity = 0


		# Encourage monotonicity (decreasing) in both I and ΔI
		for i in eachindex(I_samples)
			# Tending towards M=mobility_baseline when I == ΔI == 0
			# Stabilizing effect is stronger at more extreme M
			dM1_M = network2([M_samples[i]; 0; 0], p, st)[1][1]
			dM2_M = network2([M_samples[i]+ϵ; 0; 0], p, st)[1][1]
			if dM1_M*(M_samples[i]-mobility_baseline) > 0
				loss_stability += dM1_M*(M_samples[i]-mobility_baseline)
			end

			sgn_M = abs(dM1_M) - abs(dM2_M)
			if sgn_M < 0
				loss_monotonicity += abs(sgn_M)
			end

			# Monotonicity in I
			dM1_I = network2([M_samples[i]; I_samples[i]; ΔI_samples[i]], p, st)[1][1]
			dM2_I = network2([M_samples[i]; I_samples[i]+ϵ; ΔI_samples[i]], p, st)[1][1]
			sgn_I = abs(dM1_I) - abs(dM2_I)
			if sgn_I > 0
				loss_monotonicity += sgn_I
			end

			# Monotonicity in ΔI
			dM1_ΔI = network2([M_samples[i]; I_samples[i]; ΔI_samples[i]], p, st)[1][1]
			dM2_ΔI = network2([M_samples[i]; I_samples[i]; ΔI_samples[i]+ϵ], p, st)[1][1]

			sgn_ΔI = (abs(dM1_ΔI) - abs(dM2_ΔI))
			if sgn_ΔI > 0
				loss_monotonicity += sgn_ΔI
			end
		end
		return loss_monotonicity + loss_stability
	end



	function loss_combined(θ, tspan, M_samples, I_samples, ΔI_samples, loss_weights)
		l0, pred = loss(θ, tspan)
		l1 = (loss_weights[2] == 0) ? 0 : loss_network1(M_samples, θ.layer1, st1)
		l2 = (loss_weights[3] == 0) ? 0 : loss_network2(M_samples, I_samples, ΔI_samples, θ.layer2, st2)
		return dot((l0, l1, l2), loss_weights), l1, l2, pred
	end


	function train_combined(p, tspan; maxiters = maxiters, loss_weights=(1, 10, 10), halt_condition=(l, l1, l2)->false, η=η)
		opt_st = Optimisers.setup(Optimisers.Adam(η), p)
		losses = []
		constraint_losses = []
		best_loss = Inf
		best_p = p
		for epoch in 1:maxiters
			M_samples = rand(Uniform(M_domain[1], M_domain[2]), 200)
			I_samples = rand(Uniform(I_domain[1], I_domain[2]), 200)
			ΔI_samples = rand(Uniform(ΔI_domain[1], ΔI_domain[2]), 200)


			(l, l1, l2, pred), back = pullback(θ -> loss_combined(θ, tspan, M_samples, I_samples, ΔI_samples, loss_weights), p)
			push!(losses, l)
			if l < best_loss
				best_loss = l
				best_p = p
			end

			gs = back((one(l), nothing, nothing, nothing))[1]
			# Evaluate gs at different arguments to get other gradients
			# Get rid of one position in loss loss_weights
			# Take it out of function arguments
			# Implement weight updates
			opt_st, p = Optimisers.update(opt_st, p, gs)

			if halt_condition(l, l1, l2)
				break
			end

			if verbose && length(losses) % 50 == 0
				println("Iteration $(length(losses)): $l, constraint loss: $(l1+l2)")
				pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black, 
					label=["Data" nothing nothing], ylabel=["S" "I" "M"])
				plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red,
					label=["Approximation" nothing nothing])
				xlabel!(pl[3], "Time")

				display(pl)
			end
		end
		return best_p, losses	
	end


	p1, losses1 = train_combined(p_init, (t_train[1], t_train[end]/3); loss_weights=loss_weights, maxiters = 2500, η=0.05)
	p2, losses2 = train_combined(p1, (t_train[1], 2*t_train[end]/3); loss_weights=loss_weights, maxiters = 5000)
	
	halt_condition = (l, l1, l2) -> (l1+l2 < 5e-3) && l < 5e-2
	p_trained, losses3 = train_combined(p2, (t_train[1], t_train[end]); loss_weights=loss_weights, maxiters = 10000, η=0.0005, halt_condition=halt_condition)


	#====================================================================
	Analyze the result
	=====================================================================#
	# Network versus test data
	prob_test = remake(prob_nn, p=p_trained, tspan=(t_train[1], t_test[end]))
	pred_test = solve(prob_test, MethodOfSteps(Tsit5()), saveat=1.0)

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

	# Long-term prediction
	prob_lt = remake(prob_nn, p=p_trained, tspan=(t_train[1], 2*t_test[end]))
	pred_lt = solve(prob_lt, MethodOfSteps(Tsit5()), saveat=1.0)

	pl_pred_lt = plot(pred_lt.t, Array(pred_lt)', label=["Long-term Prediction" nothing nothing nothing nothing nothing],
		color=:red, layout=(2+num_indicators, 1))
	vline!(pl_pred_lt[end], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
		label=["Training" "" "" "" ""])
	for i = 1:2+num_indicators-1
		vline!(pl_pred_lt[i], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
			label=["" "" "" "" "" "" ""])
	end

	# Training loss progress
	pl_losses = plot(1:length(losses3), losses3, color=:red, label=nothing, 
		xlabel="Iterations", ylabel="Loss")
	yaxis!(pl_losses, :log10)


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
	β = [network1([M], p_trained.layer1, st1)[1][1] for M in range(-100, step=0.5, stop=100)]
	pl_beta_response = plot(range(-100, step=0.5, stop=100), β, xlabel="M", ylabel="β", 
		label=nothing, title="Force of infection response to mobility")
	vline!(pl_beta_response, [mobility_baseline], color=:red, label="Baseline", style=:dot,
		legend=:topleft)
	vline!(pl_beta_response, [minimum(train_data[3,:]) maximum(train_data[3,:])], color=:black, label=["Training range" nothing], 
		style=:dash)


	## Save the simulation data
	indicator_names = Dict(
		3 => "rr",
		4 => "wk",
		5 => "pk",
		6 => "si"
	)

	indicator_name = ""
	for i=1:length(indicator_idxs)-1
		indicator_name = indicator_name * indicator_names[indicator_idxs[i]] * "-"
	end
	indicator_name = indicator_name * indicator_names[indicator_idxs[end]]

	params = (taum = τₘ, taur = τᵣ, hdims= hidden_dims)
	param_name = savename(params)
	weight_name = "weight=$(loss_weights[1])-$(loss_weights[2])-$(loss_weights[3])"

	fname = "$(region)_$(indicator_name)_$(param_name)_$(weight_name)"

	# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("sims", model_name, sim_name, "$(fname)_v$(model_iteration)"))
		model_iteration += 1
	end
	fname = fname * "_v$model_iteration"
	if !isdir(datadir("sims", model_name, sim_name))
		mkdir(datadir("sims", model_name, sim_name))
	end
	mkdir(datadir("sims", model_name, sim_name, fname))

	savefig(pl_pred_test, datadir("sims", model_name, sim_name, fname, "test_prediction.png"))
	savefig(pl_pred_lt, datadir("sims", model_name, sim_name, fname, "long_term_prediction.png"))
	savefig(pl_losses, datadir("sims", model_name, sim_name, fname, "losses.png"))
	savefig(pl_beta_timeseries, datadir("sims", model_name, sim_name, fname, "beta_timeseries.png"))
	savefig(pl_beta_response, datadir("sims", model_name, sim_name, fname, "beta_response.png"))


	save(datadir("sims", model_name, sim_name, fname, "results.jld2"),
		"p", p_trained, "scale", scale, "losses", losses3, "prediction", Array(pred_test),
		"hist_data", hist_data,	"train_data", train_data, "test_data", test_data, "days", days,
		"taur", τᵣ, "taum", τₘ, "loss_weights", loss_weights, 
		"mobility_mean", μ_mobility, "mobility_std", sd_mobility)
		return nothing
end


run_model()



