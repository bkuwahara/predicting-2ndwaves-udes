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
τₘ = 14.0 # 14, 21, 28, 10, 25
τᵣ = 10.0 # 10, 14
const train_length = 160
const maxiters = 2500
const recovery_rate = 1/4
const indicators = [3]
const ϵ=0.01
activation = relu
adtype = Optimization.AutoZygote()

#===============================================
Utility functions
================================================#
function grad_basis(ind, dims)
	out = zeros(dims)
	out[ind] = 1.0
	return out
end

function weighted_vector_sum(weights, vecs)
	out = zero(vecs[1])
	for i in eachindex(vecs)
		out += weights[i]*vecs[i]
	end
	return out
end



#===============================================
Input hypterparameters
================================================#

function default_setup()
	region="US-NY"
	sim_name = region
	hidden_dims = 3
	return sim_name, region, hidden_dims
end
sim_name, region, hidden_dims = default_setup()
if ARGS != []
	sim_name = ARGS[1]
	region = ARGS[2]
	hidden_dims = parse(Int, ARGS[3])
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
	mobility_min = (-1.0 - μ_mobility)/sd_mobility
	population = dataset["population"]

	# Split the rest into pre-training (history), training and testing
	train_start_ind = 1
	while all_data[2, train_start_ind]*population < 500
		train_start_ind += 1
	end

	hist_stop = train_start_ind - 1
	hist_split = 1:hist_stop
	train_split = train_start_ind:train_start_ind + div(train_length, sample_period) - 1
	test_split = train_start_ind + div(train_length, sample_period):size(all_data, 2)

	hist_data = all_data[:, hist_split]
	train_data = all_data[:, train_split]
	test_data = all_data[:, test_split]

	all_tsteps = range(-hist_stop*sample_period, step=sample_period, length=size(all_data,2))
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
		Lux.Dense(3+num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>num_indicators))
	# network3 = Lux.Chain(
		# Lux.Dense(2=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators))

	# UDDE
	function udde(du, u, h, p, t)
		S, I, M = u
		S_hist, I_hist, M_hist = h(p, t-τₘ)
		delta_I_hist = I - h(p, t-τᵣ)[2]
		du[1] = -S_hist*I_hist*network1(h(p, t-τₘ)[3:end], p.layer1, st1)[1][1]
		du[2] = -du[1] - recovery_rate*u[2]
		du[3] = network2([u[3]; I_hist/yscale[2]; delta_I_hist/yscale[2]; (S-I)/yscale[1]], p.layer2, st2)[1][1] 
		nothing
	end



	p1, st1 = Lux.setup(rng, network1)
	p2, st2 = Lux.setup(rng, network2)
	# p3, st3 = Lux.setup(rng, network3)

	p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2))
	u0 = train_data[:,1]
	h(p,t) = hist_data[:,end]

		
	prob_nn = DDEProblem(udde, u0, h, (0.0, t_train[end]), p_init, constant_lags=[τᵣ τₘ])


	function predict(θ, tspan; u0=u0, saveat=sample_period)
		prob = remake(prob_nn, tspan = tspan, p=θ, u0=u0)
		Array(solve(prob, MethodOfSteps(Tsit5()), saveat=saveat))
	end

	function lr(p, tspan)	
		# Accuracy loss term
		pred = predict(p, tspan)
		l = size(pred, 2) < abs(tspan[2] - tspan[1])/sample_period ? Inf : sum(abs2, (pred .- train_data[:, 1:size(pred, 2)])./yscale)/size(pred, 2)
		return l, pred
	end

	function l_layer1(p, Ms)
		l1 = 0
		l5 = 0
		for M in Ms
			# beta monotonically increasing in M
			βi = network1([M], p, st1)[1][1]
			βj = network1([M + ϵ], p, st1)[1][1]
			sgn = βj - βi
			l1 += relu(-sgn)

			# Nonnegativity of beta
			l5 += relu(-βi)
		end
		return [l1, l5]
	end

	function l_layer2(p, Xs)
		l2 = 0
		l3 = 0
		l4 = 0
		l6 = 0 
		l7 = 0
		for X in Xs
			I, ΔI, M, R = X

			# Must not decrease when M at M_min
			dM_min = network2([mobility_min; I; ΔI; R], p, st2)[1][1]
			l6 += relu(-dM_min)

			# Tending towards M=mobility_baseline when I == ΔI == 0
			dM1_baseline = network2([M; 0; 0; R], p, st2)[1][1]
			dM2_baseline = network2([M+ϵ; 0; 0; R], p, st2)[1][1]
			l2 += relu(dM1_baseline*(M-mobility_baseline))

			# Stabilizing effect is stronger at more extreme M
			sgn_M = abs(dM1_baseline) - abs(dM2_baseline)
			l3 += relu(-sgn_M)


			## Monotonicity terms
			dM_initial = network2([M; I; ΔI; R], p, st2)[1][1]

			# f monotonically decreasing in I
			dM_deltaI = network2([M; (I+ϵ); ΔI; R], p, st2)[1][1]
			l3 += relu(dM_deltaI - dM_initial)
			
			# f monotonically decreasing in ΔI
			dM2_delta_deltaI = network2([M; I; (ΔI+ϵ); R], p, st2)[1][1]
			l4 += relu(dM2_delta_deltaI - dM_initial)

			# Monotonically increasing in R
			dM2_deltaR = network2([M; I; ΔI; R+ϵ], p, st2)[1][1]
			l7 += relu(dM_initial - dM2_deltaR)
		end
		return [l2, l3, l4, l6, l7]
	end


	function random_point(rng)
		I = rand(rng, Uniform(0.0, 1.0))
		ΔI = rand(rng, Uniform(I-1, I))
		R = rand(rng, Uniform(0.0, 1-I))
		M = rand(rng, Uniform(mobility_min, 2*mobility_baseline-mobility_min))
		return [I/yscale[2], ΔI/yscale[2], M, R/yscale[1]]
	end


	function train_combined(p, tspan; maxiters = maxiters, loss_weights = ones(7), halt_condition=l->false, η=1e-3)
		opt_st = Optimisers.setup(Optimisers.Adam(η), p)
		losses = []
		best_loss = Inf
		best_p = p

		for iter in 1:maxiters
			Xs = [random_point(rng) for i = 1:100]
			(l0, pred), back_all = pullback(θ -> lr(θ, tspan), p)
			g_all = back_all((one(l0), nothing))[1]
			if isnothing(g_all)
				println("No gradient found. Loss: $l0")
				g_all = zero(p)
			end

			layer1_losses, back1 = pullback(θ -> l_layer1(θ, [X[3] for X in Xs]), p.layer1)
			g_layer1 = [back1(grad_basis(i, length(layer1_losses)))[1] for i in eachindex(layer1_losses)]

			layer2_losses, back2 = pullback(θ -> l_layer2(θ, Xs), p.layer2)
			g_layer2 = [back2(grad_basis(i, length(layer2_losses)))[1] for i in eachindex(layer2_losses)]
			li = [layer1_losses; layer2_losses]

			# Store best iteration
			l_net = l0 + dot(li, loss_weights)
			push!(losses, li)
			if l_net < best_loss
				best_loss = l_net
				best_p = p
			end
			if verbose && iter % 50 == 0
				display("Total loss: $l_net, constraint losses: $(li)")
				pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black, 
					label=["Data" nothing nothing], ylabel=["S" "I" "M"])
				plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red,
					label=["Approximation" nothing nothing])
				xlabel!(pl[3], "Time")
				display(pl)
			end

			# Update parameters using the gradient
			g_net = Lux.ComponentArray(layer1 = g_all.layer1 + weighted_vector_sum(loss_weights[1:length(g_layer1)], g_layer1), 
				layer2 = g_all.layer2 + weighted_vector_sum(loss_weights[length(g_layer1)+1:end], g_layer2))
			opt_st, p = Optimisers.update(opt_st, p, g_net)


			if halt_condition([l0; li])
				break
			end
		end
		return best_p, losses, loss_weights
	end


	p1, losses1, loss_weights = train_combined(p_init, (t_train[1], t_train[end]/4); maxiters = 2500, loss_weights = 5*ones(7))
	p2, losses2, loss_weights = train_combined(p1, (t_train[1], 2*t_train[end]/2); maxiters = 5000, loss_weights=loss_weights)

	halt_condition = l -> (l[1] < 1e-2) && sum(l[2:end]) < 5e-5
	p_trained, losses3, loss_weights = train_combined(p1, (t_train[1], t_train[end]); maxiters = 10000, η=0.0005, loss_weights=loss_weights, halt_condition=halt_condition)


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
	prob_lt = remake(prob_nn, p=p_trained, tspan=(t_train[1], 3*t_test[end]))
	pred_lt = solve(prob_lt, MethodOfSteps(Tsit5()), saveat=1.0)

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

	fname = "$(region)_$(indicator_name)_$(param_name)"

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
	savefig(pl_beta_timeseries, datadir("sims", model_name, sim_name, fname, "beta_timeseries.png"))
	savefig(pl_beta_response, datadir("sims", model_name, sim_name, fname, "beta_response.png"))


	save(datadir("sims", model_name, sim_name, fname, "results.jld2"),
		"p", p_trained, "scale", scale, "losses", losses3, "prediction", Array(pred_lt),
		"hist_data", hist_data,	"train_data", train_data, "test_data", test_data, "days", days,
		"taur", τᵣ, "taum", τₘ, "loss_weights", loss_weights, 
		"mobility_mean", μ_mobility, "mobility_std", sd_mobility)
		return nothing
end


@time run_model()



