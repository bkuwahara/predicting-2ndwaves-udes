cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using Lux, DiffEqFlux, Zygote, Optimisers
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using JLD2, FileIO
using Dates
using Plots
using Statistics
using Random; rng = Random.default_rng()

## Run


# LSTMCell works on input of length (input_dims, batch_size) or (input_dims, 1)
# If given the former first, then the batch size is fixed and you can only give either that batch size or 1. This is partly because it fixes the size of the hidden state.

# Desired loop: Warmup on the first 14 days by passing in  1 day at a time and looping through


# Declare hyperparameters
τₘ = 14.0 # 14, 21, 28, 10, 25
# τᵣ = 14.0 # 10, 14
const γ = 1/4
const frac_training = 0.75
const maxiters = 2500
const lr = 0.01
hidden_dims = 10


# Input hypterparameters
region = ARGS[1]
indicators = parse.(Int, ARGS[2:end])
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)


model_name = "discrete_model"
println("Starting run: $(region)")
verbose = (isempty(ARGS)) ? true : false


## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))


beta_feature_size = num_indicators # M informs β
beta_output_size = 1


# Drop data where I=0 (i.e. beta is undefined)
data = dataset["data"]
i=1
while (data[2, i] == 0)
	i += 1
end
data = data[:,i:end]

epidemic_data = data[1:2,1:120]
indicator_data = data[indicator_idxs,:][1,:,1:120]
changes = epidemic_data[:, 2:end] .- epidemic_data[:, 1:end-1]
β_vals = @. -changes[1,:]/(epidemic_data[1,1:end-1]*epidemic_data[2, 1:end-1])

## Method 2: Discrete dynamical system using DiffEq API

train_split = 2:90
test_split = train_split[end]+1:size(changes, 2)

X_train = indicator_data[:, train_split]
Y_train = β_vals[train_split]

X_test = indicator_data[:, test_split]
Y_test = β_vals[test_split]

nn = Lux.Chain(Lux.Dense(num_indicators, hidden_dims, relu), Lux.Dense(hidden_dims, beta_output_size))
p_init, st = Lux.setup(rng, nn)
function discrete_model(du, u, p, t)
	X = indicator_data[:, Int(t)]
	S, I = u
	ΔS = -S*I*abs(nn(X, p, st)[1][1])
	du[1] = S + ΔS
	du[2] = I - ΔS - γ*I
	nothing
end


function compute_loss(X, Y, p, st)
	Y_pred = nn(X, p, st)[1][1]
	sum(abs2, Y_pred - Y)
end


function get_optimiser(p)
	return Optimisers.setup(Optimisers.Adam(0.01), p)
end

opt_state = get_optimiser(p_init)
for epoch in 1:25
	for i in 1:size(X_vals, 2)
		l, back = pullback(p -> compute_loss(X_train[:,i], Y_train[i], p, st), p_init)

		gs = back(one(l))[1]
		opt_state, p_init = Optimisers.update(opt_state, p_init, gs)


		println("Epoch [$epoch]: Loss $l")
	end
end


losses_training = [abs2(nn(X_train[:,i], p_init, st)[1][1] - Y_train[i]) for i in 1:size(X_train,2)]
test_scores = [abs2(nn(X_test[:,i], p_init, st)[1][1] - Y_test[i]) for i in 1:size(X_test,2)]
maximum(test_scores)
mean(test_scores)



compute_loss(X_vals[:,1], Y_vals[1], p_init, st)

u0 = epidemic_data[:,1]
tspan = (0.0, size(training_data, 2) - 1)
prob = DiscreteProblem(discrete_model, u0, tspan, p_init)
sol = solve(prob, saveat=1.0)
plot(sol, layout=(2,1))



function predict(p)
	return Array(solve(prob, saveat=1.0, p=p))
end

function loss(p)
	pred = predict(p)
	return sum(abs2, pred .- training_data), pred
end

losses = []
function callback(p, l, pred)
	push!(losses, l)
	if length(losses) % 100 == 0
		display(l)
		pl=scatter(tsteps, training_data', layout=(2,1))
		plot!(pl, tsteps, pred', layout=(2,1))
		display(pl)
	end
	return false
end


adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p, x) -> loss(p), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(p_init))
res = Optimization.solve(optprob, ADAM(0.15), maxiters=10000, callback=callback)

optprob2 = remake(optprob, p=res.minimizer)
res2 = Optimization.solve(optprob2, ADAM(0.15), maxiters=10000, callback=callback)

pred_final = predict(res2.minimizer)

plot(tsteps, pred_final', layout=(2,1))
