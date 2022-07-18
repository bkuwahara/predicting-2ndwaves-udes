cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using Lux, DiffEqFlux, Zygote, Optimisers
using MLUtils
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using JLD2, FileIO
using Dates
using Plots
using Statistics
using Random; rng = Random.default_rng()

# Input hypterparameters
region = ARGS[1]
indicators = parse.(Int, ARGS[2:end])
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1



# Returns
function normalize_data(data; dims=ndims(data), ϵ=1e-5)
	μ = mean(data, dims=dims)
	sd = std(data, dims=dims)
	data = @. (data - μ)/(sd+ϵ)

	function invnorm(data)
		return @. data*(sd+ϵ) + μ
	end
	return data, invnorm
end

function TimeSeriesSplit(n_splits::Int, test_size::Int)
	@assert (n_splits >= 2) "n_splits must be at least 2"
	n_samples = size(data, 2)
	@assert test_size < n_samples "test size must be "

	test_splits = [i:i+test_size for i in range(n_samples-test_size*(n_splits), stop = n_samples-test_size, step=test_size)]
	train_splits = [1:split[1]-1 for split in test_splits]
	return collect(zip(train_splits, test_splits))
end

# Desired loop: Warmup on the first 14 days by passing in  1 day at a time and looping through
# First perform supervised learning on the two models separately, then train them in conjunction
struct LSTMToDense{L, D} <:
		Lux.AbstractExplicitContainerLayer{(:lstm_cell, :dense)}
	lstm_cell::L
	dense::D
end

function LSTMToDense(input_dims, hidden_dims, output_dims)
	return LSTMToDense(
		Lux.LSTMCell(input_dims => hidden_dims),
		Lux.Dense(hidden_dims => output_dims)
	)
end

# Initialization: creates hidden state and memory state h0 and m0
function (LD::LSTMToDense)(u0::AbstractArray{T, 2}, p, st) where {T}
	(h0, m0), st_lstm = LD.lstm_cell(u0, p.lstm_cell, st.lstm_cell)
	st_new = merge(st, (lstm_cell = st_lstm, dense = st.dense))
	return (h0, m0), st_new
end

# Standard propagation call
function (LD::LSTMToDense)(vhm::Tuple, p, st)
	v, h, m = vhm
	(h_new, m_new), st_lstm = LD.lstm_cell((v, h, m), p.lstm_cell, st.lstm_cell)
	out, st_dense = LD.dense(h_new, p.dense, st.dense)
	st_new = merge(st, (lstm_cell=st_lstm, dense=st_dense))
	return (out, h_new, m_new), st_new
end

function reset!(LD::LSTMToDense, p, st)
	return LD(zeros(num_indicators, 1), p, st)
end


# Declare hyperparameters
τₘ = 14.0 # 14, 21, 28, 10, 25
# τᵣ = 14.0 # 10, 14
const γ = 1/4
const frac_training = 0.75
const maxiters = 2500
const lr = 0.01
hidden_dims = 10


function run_model()

	model_name = "discrete_model_LSTM"
	println("Starting run: $(region)")
	verbose = (isempty(ARGS)) ? true : false


	## Load in data for the region and partition into training, testing
	dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
	days = dataset["days"]
	# Drop data where I=0 (i.e. beta is undefined)
	data = dataset["data"]
	start_idx=1
	while (data[2, start_idx] == 0)
		start_idx += 1
	end
	data = data[:,start_idx:start_idx+119]

	epidemic_data = data[1:2,:]
	indicator_data = data[indicator_idxs,:][1,:,:]
	changes = epidemic_data[:, 2:end] .- epidemic_data[:, 1:end-1]
	β_vals = @. -changes[1,:]/(epidemic_data[1,1:end-1]*epidemic_data[2, 1:end-1])





	## Method 1: custom solver that propagates LSTM hidden state
	β_normalized = log.(β_vals)
	X_vals = [reshape(col, (length(col),1)) for col in eachcol(indicator_data)]
	Y_vals = [val for val in β_normalized]


	beta_network = LSTMToDense(beta_feature_size, hidden_dims, beta_output_size)
	ps, st = Lux.setup(rng, beta_network)


	train_split = 1:90
	test_split = train_split[end]+1:size(changes, 2)

	X_train = X_vals[train_split]
	Y_train = β_normalized[train_split]

	X_test = X_vals[test_split]
	Y_test = β_normalized[test_split]




	function compute_loss_LSTM(X, Y, h, m, p, st)
		(Y_pred, h_new, m_new), st_new  = beta_network((X, h, m), p, st)
		loss = abs2(Y_pred[1] - Y)
		return loss, (h_new, m_new), st_new
	end

	opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps)
	for epoch in 1:100
		(h, m), st = reset!(beta_network, ps, st)
		for i in 1:length(X_train)
			(l, (h, m), st), back = pullback(p->compute_loss_LSTM(X_train[i], Y_train[i], h, m, p, st), ps)
			gs = back((one(l), nothing, nothing))[1]
			opt_state, ps = Optimisers.update(opt_state, ps, gs)
			if epoch % 10 == 0
				println("Epoch [$(epoch)]: Loss = $(l)")
			end
		end
	end


	function get_predictions(model::LSTMToDense, p, st, X::AbstractArray)
		(h, m), st = reset!(model, ps, st)
		preds = zeros(length(X))
		for i = 1:length(X)
			(pred, h, m), st = model((X[i], h, m), p, st)
			preds[i] = pred[1]
		end
		return preds
	end


	# final trained results
	training_preds = get_predictions(beta_network, ps, st, X_train)
	training_losses = @. abs2(training_preds - Y_train)
	mse = mean(training_losses)

	test_preds = get_predictions(beta_network, ps, st, X_test)
	test_losses = @. abs2(exp(test_preds) - exp(Y_test))
	score = mean(test_losses)


	function solve_system(u0, tspan::Tuple{Int, Int}, p, st)
		(h, m), st = reset!(beta_network, p, st)
		Y_out = zeros(length(u0), tspan[end]-tspan[1]+1)

		Y_out[:,1] .= u0
		for t in 2:size(Y_out, 2)
			S, I = Y_out[:,t-1]
			(logβ, h, m), st = beta_network((X_vals[t-1], h, m), p, st)
			ΔS = -exp(logβ[1])*S*I
			ΔI = -ΔS -γ*I
			Y_out[:,t] = Y_out[:,t-1] + [ΔS; ΔI]
		end
		return Y_out
	end

	u0 = epidemic_data[:,1]
	tspan = (0, size(epidemic_data, 2)-1)
	pred = solve_system(u0, tspan, ps, st)


	function loss(u0, tspan, p, st, Y)
		pred = solve_system(u0, tspan, p, st)
		return sum(abs2, pred .- Y[:, tspan[1]+1:tspan[end]+1])
	end

	loss(u0, tspan, ps, st, epidemic_data)
	tsteps = range(0, step=1, length=size(pred,2))
	pl_initial_pred = scatter(tsteps, epidemic_data', layout=(2,1), color=:black, label=["Data" nothing],
		ylabel=["S" "I"])
	plot!(pl_initial_pred, tsteps, pred', color=:red, label=["Prediction" nothing])
	xlabel!(pl_initial_pred[2], "Time (days since $(days[start_idx]))")
	title!(pl_initial_pred[1], "Initial trained prediction: LSTM model, $(region)")


	### Re-train using the initial condition

	function training_func(X, Y, tspan::Tuple{Int, Int}, p, st)
		(h, m), st = reset!(beta_network, p, st)
		U = Y[:,1]
		loss = 0
		for t in 2:tspan[end]
			S, I = U
			(logβ, h, m), st = beta_network((X[t-1], h, m), p, st)
			ΔS = -exp(logβ[1])*S*I
			ΔI = -ΔS -γ*I
			U +=  [ΔS; ΔI]
			loss += sum(abs2, U - Y[:,t])
		end
		return loss
	end

	losses = []
	function callback(p, l)
		push!(losses, l)
		if length(losses) % 50 == 0
			println("Loss after $(length(losses)) iterations: $(l)")
		end
		return false
	end

	X=X_vals[train_split]
	Y = epidemic_data[:, train_split]
	tspan = (0, length(X)-1)


	adtype = Optimization.AutoZygote()
	optf = Optimization.OptimizationFunction((p, x) -> training_func(X, Y, tspan, p, st), adtype)
	optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(ps))
	res = solve(optprob, OptimizationFlux.ADAM(0.01), maxiters=500, callback=callback)

	optprob2 = remake(optprob, p=res.minimizer)
	res_final = solve(optprob2, LBFGS(), maxiters=100, callback=callback)
	p_trained = res_final.minimizer


	tspan = (0, size(epidemic_data, 2)-1)
	u0 = Y[:,1]
	final_pred = solve_system(u0, tspan, p_trained, st)

	tsteps = range(0, step=1, length=size(final_pred,2))
	pl_final_pred = scatter(tsteps, epidemic_data', layout=(2,1), color=:black, label=["Data" nothing],
		ylabel=["S" "I"])
	plot!(pl_final_pred, tsteps, final_pred', color=:red, label=["Prediction" nothing])
	xlabel!(pl_final_pred[2], "Time (days since $(days[start_idx]))")
	title!(pl_final_pred[1], "Final trained prediction: LSTM model, $(region)")



	pl_losses = plot(1:500, losses[1:500], color=:red, xlabel="Iteration", ylabel="Loss", label="ADAM")
	plot!(pl_losses, 501:length(losses), losses[501:end], color=:blue, label="LBFGS")
	yaxis!(pl_losses, :log10)
	title!(pl_losses, "Training losses: LSTM model, $(region)")


	## Save the simulation data
	indicator_names = Dict(
		3 => "rr",
		4 => "wk",
		5 => "pk",
		6 => "si"
	)

	indicator_name = ""
	for j=1:length(indicator_idxs)-1
		indicator_name = indicator_name * indicator_names[indicator_idxs[j]] * "-"
	end
	indicator_name = indicator_name * indicator_names[indicator_idxs[end]]

	params = (hdims = hidden_dims,)
	param_name = savename(params)

	fname = "$(indicator_name)-$(param_name)"

	# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("sims", model_name, region, "$(fname)_v$(model_iteration)"))
		model_iteration += 1
	end
	fname = fname * "_v$(model_iteration)"
	mkdir(datadir("sims", model_name, region, fname))


	savefig(pl_initial_pred, datadir("sims", model_name, region, fname, "initial_prediction.png"))
	savefig(pl_final_pred, datadir("sims", model_name, region, fname, "final_prediction.png"))
	savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

	save(datadir("sims", model_name, region, fname, "results.jld2"),
		"p", p_trained, "prediction", final_pred, "losses", losses,
		"training_data", data[:,train_split], "test_data", data[:,test_split], "days", days)
end

run_model()
