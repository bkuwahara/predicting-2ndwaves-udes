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


# What do we know about dU/dt?
# - S=0, I=0, S=1, R=1 are all invariant sets (soft constraint implemented by standard supervised training loss or hard-coded as dS ∝ SI, conservation law)
# - Should exhibit some form of delayed effect (hard code as DDE or soft code as LSTM, no way to implement via loss)
# - Increased M leads to decreased S and increased I (implement soft constraint using univariate monotonicity loss)
# - Under I ≈ 0, M->0 (unclear level of delay or threshold) (could implement as soft constraint using monotonicity loss)
# - Long-term, I should approach 0 assuming no birth rate (conservation law or long-term loss)

# What might be true about dU/dt?
# - Might exhibit oscillations (multiple repeat waves)
# - Long-term behaviour: does M should tend towards 0 or oscillate about it or another point?
# 



#=================================
constant hyperparameters
=================================#
sample_period=7
τₘ = 10.0/sample_period # 14, 21, 28, 10, 25
τᵣ = 14.0/sample_period # 10, 14
const frac_training = 0.75
const maxiters = 2500
const lr = 0.005
const recovery_rate = 1/4
const indicators = [3]
activation = relu
adtype = Optimization.AutoZygote()

#===============================================
Input hypterparameters
================================================#

function default_setup()
	region="US-NY"
	hidden_dims = 3
	loss_weights = (1, 1, 1)
	return region, hidden_dims, loss_weights
end
region, hidden_dims, loss_weights = default_setup()
if ARGS != []
	region = ARGS[1]
	hidden_dims = parse(Int, ARGS[2])
	loss_weights = parse.(Int, ARGS[3:end])
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

	I_domain = (0.0, 1.0)
	ΔI_domain = 2 .*(minimum(all_data[2,2:end] - all_data[2,1:end-1]), maximum(all_data[2,2:end] - all_data[2,1:end-1]))
	M_domain = 2 .*(minimum(all_data[3,:]), maximum(all_data[3,:]))

	# Split the rest into pre-training (history), training and testing
	hist_stop = Int(round(max(τᵣ+1, τₘ)))
	hist_split = 1:hist_stop
	train_split = hist_stop+1:hist_stop + div(90, sample_period)
	test_split = hist_stop+div(90, sample_period)+1:size(all_data, 2)


	hist_data = all_data[:, hist_split]
	train_data = all_data[:, train_split]
	test_data = all_data[:, test_split]

	all_tsteps = range(-sample_period*max((τᵣ+1), τₘ), step=sample_period, length=size(all_data,2))
	t_hist = all_tsteps[hist_split]
	t_train = all_tsteps[train_split]
	t_test = all_tsteps[test_split]

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
		I_hist = h(p, t-τᵣ)[2]
		delta_I_hist = I_hist - h(p, t-(τᵣ+1))[2]
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
	h(p,t) = hist_data[:,1]

		
	prob_nn = DDEProblem(udde, u0, h, (0.0, t_train[end]), p_init, constant_lags=[τᵣ τₘ (τᵣ+1)])


	function predict(θ, tspan; u0=u0, saveat=sample_period)
		prob = remake(prob_nn, tspan = tspan, p=θ, u0=u0)
		Array(solve(prob, MethodOfSteps(Tsit5()), saveat=saveat))
	end

	function loss(θ, tspan)
		pred = predict(θ, tspan)
		sum(abs2, (pred .- train_data[:, 1:size(pred, 2)])./yscale)/size(pred, 2), pred
	end

	losses = []
	function callback(θ, l, pred)
		push!(losses, l)
		if verbose && (length(losses) % 100 == 0)
			display(l)
			pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black)
			plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red)
			display(pl)
		end
		if l > 1e12
			println("Bad initialization. Aborting...")
			return true
		end
		return false
	end


	function loss_network1(M_samples, p, st)
		loss_negativity = 0
		loss_monotonicity = 0
		for (Mi, Mj) in M_samples
			βi = network1([Mi], p, st)[1][1]
			βj = network1([Mj], p, st)[1][1]
			sgn = (Mj - Mi)*(βj - βi)
			if sgn < 0
				loss_monotonicity += abs(sgn)
			end
			if βi < 0
				loss_negativity += abs(βi)
			end
			if βj < 0
				loss_negativity += abs(βj)
			end
		end
		return loss_monotonicity + loss_negativity
	end


	function loss_network2(M_samples, I_samples, ΔI_samples, p, st)
		loss_stability = 0
		loss_monotonicity = 0

		# Encourage return towards M=0 when I=0, ΔI = 0
		for M in M_samples
			for Mi in M
				dM = network2([Mi; 0; 0], p, st)[1][1]
				if dM*(Mi+μ_mobility/sd_mobility) < 0
					loss_stability += abs(dM*Mi)
				end
			end
		end

		# Encourage monotonicity (decreasing) in both I and ΔI
		for i in eachindex(I_samples)
			for j in [1,2]
				dM1 = network2([M_samples[i][j]; I_samples[i][1]; ΔI_samples[i][j]], p, st)[1][1]
				dM2 = network2([M_samples[i][j]; I_samples[i][2]; ΔI_samples[i][j]], p, st)[1][1]

				sgn = (I_samples[i][1] - I_samples[i][2])*(dM1 - dM2)
				if sgn > 0
					loss_monotonicity += sgn
				end
			end

			for j in [1,2]
				dM1 = network2([M_samples[i][j]; I_samples[i][j]; ΔI_samples[i][1]], p, st)[1][1]
				dM2 = network2([M_samples[i][j]; I_samples[i][j]; ΔI_samples[i][2]], p, st)[1][1]

				sgn = (ΔI_samples[i][1] - ΔI_samples[i][2])*(dM1 - dM2)
				if sgn > 0
					loss_monotonicity += sgn
				end
			end
		end
		return loss_monotonicity + loss_stability
	end



	function loss_combined(θ, tspan, M_samples, I_samples, ΔI_samples, loss_weights)
		l0, pred = loss(θ, tspan)
		l1 = (loss_weights[2] == 0) ? 0 : loss_network1(M_samples, θ.layer1, st1)
		l2 = (loss_weights[3] == 0) ? 0 : loss_network2(M_samples, I_samples, ΔI_samples, θ.layer2, st2)
		return dot((l0, l1, l2), loss_weights), pred
	end


	function train_combined(p, tspan; maxiters = maxiters, loss_weights=(2, 1, 1), halt_condition=l->false)
		opt_st = Optimisers.setup(Optimisers.Adam(0.005), p)
		losses = []
		best_loss = Inf
		best_p = p
		for epoch in 1:maxiters
			M_samples = [rand(Uniform(M_domain[1], M_domain[2]), 2) for j in 1:50]
			I_samples = [rand(Uniform(I_domain[1], I_domain[2]), 2) for j in 1:50]
			ΔI_samples = [rand(Uniform(ΔI_domain[1], ΔI_domain[2]), 2) for j in 1:50]


			(l, pred), back = pullback(θ -> loss_combined(θ, tspan, M_samples, I_samples, ΔI_samples, loss_weights), p)
			push!(losses, l)

			if l < best_loss
				best_loss = l
				best_p = p
			end

			gs = back((one(l), nothing))[1]
			opt_st, p = Optimisers.update(opt_st, p, gs)

			if halt_condition(l)
				break
			end

			if verbose && length(losses) % 50 == 0
				display(l)
				pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black)
				plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red)
				display(pl)
			end
		end
		return best_p, losses	
	end



	function train_fit(p, tspan; maxiters=maxiters, lr = 0.005)
		optf = Optimization.OptimizationFunction((θ, u) -> loss(θ, tspan), adtype)
		optprob = Optimization.OptimizationProblem(optf, p)
		res = Optimization.solve(optprob, ADAM(lr), maxiters=maxiters, callback=callback)
		return res.minimizer
	end


	tspan = (0.0, t_train[end])
	# p1 = train_fit(p_init, tspan, maxiters=1000)
	# p2 = train_fit(p1, (0.0, 50.0), maxiters=1000)
	# p3 = train_fit(p1, tspan, maxiters=1000)
	p1, losses1 = train_combined(p_init, (0.0, 20.0); maxiters = 100)
	p_trained, losses2 = train_combined(p1, (0.0, 50.0); maxiters = 100)


	#====================================================================
	Analyze the result
	=====================================================================#
	# Network prediction
	prob_final = remake(prob_nn, p=p_trained, tspan=(t_train[1], t_test[end]))
	pred_final = solve(prob_final, MethodOfSteps(Tsit5()), saveat=0.1)

	pl_pred = scatter(all_tsteps, all_data', label=["True data" nothing nothing nothing nothing nothing],
		color=:black, layout=(2+num_indicators, 1))
	plot!(pl_pred, pred_final.t, Array(pred_final)', label=["Prediction" nothing nothing nothing nothing nothing],
		color=:red, layout=(2+num_indicators, 1))
	vline!(pl_pred[end], [t_hist[end] t_train[end]], color=:black, style=:dash,
		label=["Training" "" "" "" ""])
	for i = 1:2+num_indicators-1
		vline!(pl_pred[i], [t_hist[end] t_train[end]], color=:black, style=:dash,
			label=["" "" "" "" "" "" ""])
	end

	# Training loss progress
	pl_losses = plot(1:length(losses1), losses1, color=:red, label="Stage 1", 
		xlabel="Iterations", ylabel="Loss")
	plot!(pl_losses, length(losses1)+1:length(losses1)+length(losses2), losses2, color=:blue, label="Stage 2")

	yaxis!(pl_losses, :log10)


	# beta time series
	indicators_predicted = pred_final[3:end,:]
	β = zeros(size(pred_final.t))
	for i in 1:length(β)
		indicator = i <= length(hist_split) ? hist_data[3:end,1] : indicators_predicted[:, i-length(hist_split)]
		β[i] = network1(indicator, p_trained.layer1, st1)[1][1]
	end
	plot(range(-length(hist_data), length=length(β), stop=all_tsteps[end]), β,
		xlabel="t", ylabel="β")

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

	params = (taum = τₘ, taur = τᵣ)
	param_name = savename(params)


	weight_name = "weight=$(loss_weights[1])-$(loss_weights[2])-$(loss_weights[3])"

	fname = "$(indicator_name)_$(param_name)_$(weight_name)"

	# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("sims", model_name, region, "$(fname)-$(model_iteration)"))
		model_iteration += 1
	end
	fname = fname * "_v$(model_iteration)"
	mkdir(datadir("sims", model_name, region, fname))

	savefig(pl_pred, datadir("sims", model_name, region, fname, "final_prediction.png"))
	savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

	save(datadir("sims", model_name, region, fname, "results.jld2"),
		"p", p_trained, "scale", scale, "losses", losses, "prediction", pred_final,
		"train_data", train_data, "test_data", test_data, "days", days,
		"taur", τᵣ, "taum", τₘ, "loss_weights", loss_weights, 
		"mobility_mean", μ_mobility, "mobility_std", sd_mobility)
		return nothing
end


run_model()



