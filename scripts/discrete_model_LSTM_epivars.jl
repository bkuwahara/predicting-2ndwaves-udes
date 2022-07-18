cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")

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
function default_setup()
	region="US-NY"
	series_length=120
	hidden_dims = 3
	indicators=[4, 5]
	return region, series_length, hidden_dims, indicators
end
region, series_length, indicators=default_setup()
if ARGS != []
	region = ARGS[1]
	series_length = parse(Int, ARGS[2])
	hidden_dims = parse(Int, ARGS[3])
	indicators = parse.(Int, ARGS[4:end])
end
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



# Declare hyperparameters
const γ = 1/4
const frac_training = 0.75
const lr = 0.01
hidden_dims = 2


# function run_model()

model_name = "discrete_model_LSTM"
println("Starting run: $(region)")
verbose = (isempty(ARGS)) ? true : false


## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
days = dataset["days"]
data = dataset["data"][hcat([1 2], indicator_idxs),:][1,:,:]
data = reshape(data, size(data)[1], size(data)[2], 1)
start_idx=15 # two weeks used for warmup


warmup_data = data[:, 1:start_idx-1,:]
data = data[:, start_idx:start_idx+series_length-1,:]
function warmup!(LD::LSTMToDense, p, st, warmup_data)
	(h, m), st = LD(warmup_data[:,1,:], p, st)
	for j=2:size(warmup_data,2)
		(out, h, m), st = LD((warmup_data[:,j,:], h, m), p, st)
	end
	return (h, m), st
end


epidemic_data = data[1:2,:,:]
indicator_data = data[3:end,:,:]
changes = data[:, 2:end,:] .- data[:, 1:end-1,:]
β_vals = @. -changes[1,:,:]/(epidemic_data[1,1:end-1,:]*epidemic_data[2, 1:end-1,:])
β_normalized = sqrt.(β_vals)


## Method 1: custom solver that propagates LSTM hidden state


beta_network = LSTMToDense(num_indicators, hidden_dims, beta_output_size)
indicator_network = LSTMToDense(num_indicators+1, hidden_dims, num_indicators)
ps_beta, st_beta = Lux.setup(rng, beta_network)
ps_ind, st_ind = Lux.setup(rng, indicator_network)


train_split = 1:Int(round(frac_training*training_length))
test_split = train_split[end]+1:size(changes, 2)


X_beta = indicator_data
X_ind = vcat(epidemic_data[2,:,:]', indicator_data)
Y_beta = reshape(β_vals, 1, length(β_vals), 1)
Y_ind = changes[3:end,:,:]


function compute_loss_LSTM(model, h, m, p, st, X, Y)
	(Y_pred, h_new, m_new), st_new  = model((X, h, m), p, st)
	loss = sum(abs2, Y_pred - Y)
	return loss, (h_new, m_new), st_new
end


function pre_train(model, p, st, X, Y, warmup)
	opt_state = Optimisers.setup(Optimisers.Adam(0.01), p)
	for epoch in 1:100
		(h, m), st = warmup!(model, p, st, warmup)
		for i in 1:size(X,2)
			(l, (h, m), st), back = pullback(θ->compute_loss_LSTM(model, h, m, θ, st, X[:,i,:], Y[:,i,:]), p)
			gs = back((one(l), nothing, nothing))[1]
			opt_state, p = Optimisers.update(opt_state, p, gs)
			if epoch % 10 == 0 && verbose
				println("Epoch [$(epoch)]: Loss = $(l)")
			end
		end
	end
	return p
end

ps_beta = pre_train(beta_network, ps_beta, st_beta, X_beta[:,train_split,:], Y_beta[:,train_split,:], warmup_beta)



function solve_system(u0, tspan::Tuple{Int, Int}, p, st)
	(h, m), st = warmup!(beta_network, p, st, warmup_data[3:end,:,:])
	Y_out = zeros(length(u0), tspan[end]-tspan[1]+1, 1)

	Y_out[:,1,:] .= u0
	for t in 2:size(Y_out, 2)
		S, I = Y_out[:,t-1]
		(rootβ, h, m), st = beta_network((X_vals[t-1], h, m), p, st)
		ΔS = (rootβ[1]^2)*S*I > S ? -S : -(rootβ[1]^2)*S*I
		ΔI = I + (-ΔS -γ*I) > 1 ? 1-I : -ΔS -γ*I
		Y_out[:,t,:] = Y_out[:,t-1,:] + [ΔS; ΔI]

	end
	return Y_out
end


function plot_prediction(args...; title="Prediction")
	pred, tsteps = solve_system(args...)
	pl = scatter(tsteps, epidemic_data[:,tsteps[1]+1:tsteps[end]+1,1]', layout=(length(u0),1), color=:black, label=["Data" nothing],
		ylabel=["S" "I"])
	plot!(pl, tsteps, pred[:,:,1]', color=:red, label=["Prediction" nothing])
	vline!(pl[1], [train_split[end]], color=:black, style=:dash, label=nothing)
	vline!(pl[end], [[train_split[end]]], color=:black, style=:dash, label="End of training data")

	xlabel!(pl[end], "Time (days since $(days[start_idx]))")
	title!(pl[1], title*", LSTM model, $(region)")
	return pl
end

u0 = epidemic_data[:,1,:]
tspan = (0, size(epidemic_data, 2)-1)
pred = solve_system(u0, tspan, ps, st)


tsteps = range(0, step=1, length=size(pred,2))
pl_initial_pred = scatter(tsteps, epidemic_data', layout=(2,1), color=:black, label=["Data" nothing],
	ylabel=["S" "I"])
plot!(pl_initial_pred, tsteps, pred', color=:red, label=["Prediction" nothing])
xlabel!(pl_initial_pred[2], "Time (days since $(days[start_idx]))")
title!(pl_initial_pred[1], "Initial trained prediction: LSTM model, $(region)")


### Re-train using the initial condition

function solve_loss(X, Y, tspan::Tuple{Int, Int}, p, st; λ=0)
	(h, m), st = warmup!(beta_network, p, st, warmup_data[3:end,:,:])
	U = Y[:,1]
	loss = 0
	for t in 2:tspan[end]
		S, I = U
		(rootβ, h, m), st = beta_network((X[t-1], h, m), p, st)
		ΔS = (rootβ[1]^2)*S*I > S ? -S : -(rootβ[1]^2)*S*I
		ΔI = I + (-ΔS -γ*I) > 1 ? 1-I : -ΔS -γ*I
		U = U + [ΔS; ΔI]
		loss += sum(abs2, U - Y[:,t,:])
	end
	return loss + λ*sum(abs2, p)/length(p)
end

losses = []
function callback(p, l)
	push!(losses, l)
	if length(losses) % 50 == 0 && verbose
		println("Loss after $(length(losses)) iterations: $(l)")
	end
	return false
end

X=X_vals[train_split]
Y = epidemic_data[:, train_split]
tspan = (0, length(X)-1)


adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p, x) -> solve_loss(X, Y, tspan, p, st), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(ps))
res_final = solve(optprob, OptimizationFlux.ADAM(0.01), maxiters=600, callback=callback)

# optprob2 = remake(optprob, p=res.minimizer)
# res_final = solve(optprob2, LBFGS(), maxiters=100, callback=callback)
p_trained = res_final.minimizer



tspan = (0, size(epidemic_data, 2)-1)
u0 = Y[:,1]
final_pred = solve_system(u0, tspan, p_trained, st)

tsteps = range(0, step=1, length=size(final_pred,2))
pl_final_pred = scatter(tsteps, epidemic_data', layout=(2,1), color=:black, label=["Data" nothing],
	ylabel=["S" "I"])
plot!(pl_final_pred, tsteps, final_pred', color=:red, label=["Prediction" nothing])
xlabel!(pl_final_pred[2], "Time (days since $(days[start_idx]))")
title!(pl_final_pred[1], "Final test prediction: LSTM model, $(region)")

pl_training_pred = scatter(tsteps[train_split], epidemic_data[:,train_split]', layout=(2,1), color=:black, label=["Data" nothing],
	ylabel=["S" "I"])
plot!(pl_training_pred, tsteps[train_split], final_pred[:,train_split]', color=:red, label=["Prediction" nothing])
xlabel!(pl_training_pred[2], "Time (days since $(days[start_idx]))")
title!(pl_training_pred[1], "Final training data prediction: LSTM model, $(region)")

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
savefig(pl_final_pred, datadir("sims", model_name, region, fname, "final_test_prediction.png"))
savefig(pl_training_pred, datadir("sims", model_name, region, fname, "training_prediction.png"))
savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

save(datadir("sims", model_name, region, fname, "results.jld2"),
	"p", p_trained, "prediction", final_pred, "losses", losses,
	"training_data", data[:,train_split], "test_data", data[:,test_split], "days", days)
# end

run_model()
