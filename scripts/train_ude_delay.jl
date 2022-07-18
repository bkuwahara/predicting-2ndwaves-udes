cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")

using Lux, DiffEqFlux, Zygote
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using JLD2, FileIO
using Dates
using Plots
using DataInterpolations
using Random; rng = Random.default_rng() # Random.seed!(1234)



## Run

# constant hyperparameters
τₘ = 10.0 # 14, 21, 28, 10, 25
τᵣ = 14.0 # 10, 14
const frac_training = 0.75
const maxiters = 2500
const lr = 0.01
const recovery_rate = 1/4
hidden_dims = 4
activation = relu
opt = ADAM(lr)


# Input hypterparameters
region = ARGS[1]
predictors = parse.(Int, ARGS[2:end])
predictor_idxs = reshape(predictors, 1, length(predictors))
num_predictors = length(predictor_idxs)


model_name = "udde"
println("Starting run: $(region)")
verbose = (isempty(ARGS)) ? true : false

## Load in data for the region and partition into training, testing
#= Structured as
	[S(t)'
	 I(t)'
	 M_retail'
	 M_work'
	 M_parks'
	 X']
  where M is Google Mobility data, X is Oxford government stringency index =#

dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
all_data = dataset["data"]
response_vars = all_data[1:2,:]
predictor_vars = all_data[predictor_idxs,:][1,:,:]

all_data = [response_vars; predictor_vars]
days = dataset["days"]


# Take the first 14 days for history function
hist_stop = Int(round(max(τᵣ+1, τₘ)))
hist_data = all_data[:, 1:hist_stop]
h0 = hist_data[:,1]
# Split the rest into training and testing
training_stop = hist_stop + Int(round(frac_training*size(all_data[:,hist_stop+1:end], 2)))
training_data = all_data[:, hist_stop + 1:training_stop]
test_data = all_data[:, training_stop+1:end]


t_train = range(0.0, length = size(training_data, 2), step = 1.0);
t_test = range(t_train[end]+1, length = size(test_data, 2), step = 1.0);

# Get scale factors from the data to improve training
yscale = maximum(training_data, dims=2) .- minimum(training_data, dims=2);
tscale = t_train[end] - t_train[1];
scale = yscale/tscale;



## Set up model
struct Predictor{dS, dM} <:
	Lux.AbstractExplicitContainerLayer{(:predictor, :propagator)}
	predictor::dS
	propagator::dM
end

function Predictor(hidden_dims)
	return Predictor(
		Lux.Chain(
			Lux.Dense(num_predictors=>hidden_dims, activation), Lux.Dense(hidden_dims=>1)),
		Lux.Chain(
			Lux.Dense(2+num_predictors=>hidden_dims, activation), Lux.Dense(hidden_dims=>num_predictors))
	)
end



function (P::Predictor)(du, u, h, p, t)
	S, I = u[1:2]
	indicators = u[3:end]
	I_t1 = h(p, t-(τᵣ +1))[2]
	I_t2 = h(p, t-τᵣ)[2]
	delta_I = (I_t2 -I_t1)
	I_information = [delta_I; I_t2]/scale[2]

	β, st_pred = network.predictor(indicators, p.predictor, st.predictor)
	dX, st_prop = network.propagator([I_information; indicators], p.propagator, st.propagator)
	du[1] = -β[1]*S*I
	du[2] = -du[1] - recovery_rate*I
	du[3:end] .= dX
	nothing
end
##


network = Predictor(hidden_dims)

u0 = training_data[:,1]
h(p, t) = h0
tspan = (t_train[1], t_train[end])
p, st = Lux.setup(rng, network)


function predict(θ)
	Array(solve(prob_nn, MethodOfSteps(Tsit5()),
		p=θ, saveat=1.0, sensealg=ReverseDiffAdjoint()))
end;
function loss(θ)
	pred=predict(θ)
	return (sum(abs2, (training_data[:,2:size(pred,2)] - pred[:,2:end])./yscale)/size(pred,2)), pred
end;
losses = []
training_iters = 0;
function callback(θ, l, pred)
	global training_iters += 1
	if (verbose && training_iters % 100 == 0)
		pl = plot(t_train[1:size(pred,2)], training_data[:,1:size(pred,2)]', layout=(4,1), label=["data" "" "" ""])
		plot!(pl, t_train[1:size(pred,2)], pred', label = ["prediction" "" "" ""])
		display(pl)
		display(l)
	end
	push!(losses, l)
	if l > 1e12 && training_iters < 10
		throw("Bad initialization. Aborting...")
		return true
	elseif l < 1e-5 && training_iters > 1e4
		println("Training converged after $(training_iters) iterations")
		return true
	else
		return false
	end
end;


## Training
prob_nn = DDEProblem(network, u0, h, (0.0, t_train[end]), p; constant_lags=[τᵣ + 1; τᵣ;  τₘ])
prob_nn = remake(prob_nn, tspan = (0.0, 25.0), p=p)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationProblem((θ, x) -> loss(θ), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(p))
res1 = Optimization.solve(optprob, PolyOpt(), maxiters=250)

res1 = DiffEqFlux.sciml_train(loss, Lux.ComponentArray(p), PolyOpt, maxiters=250, cb=callback)

println("Completed training. Final loss: $(losses[end])")


## post-process/analyze
p_trained = p

prob_final = DDEProblem(udde, u0, h, (t_train[1], t_test[end]), p_trained; constant_lags=[τᵣ; τₘ])
pred = Array(solve(prob_final, MethodOfSteps(Tsit5()),
	p=p_trained, saveat=[t_train; t_test]))

# Append a number ot the end of the simulation to allow multiple runs of a
# single set of hyperparameters for ensemble predictions
model_iteration = 1
while isfile(datadir("sims", model_name, "results_$(region)_$(model_iteration).jld2"))
	model_iteration += 1
end

save(datadir("sims", model_name, "results_$(region)_$(model_iteration).jld2"),
	"tau_r", τᵣ, "tau_m", τₘ, "dS_ls", dS_ls, "dM_ls", dM_ls,
	"p_dS", p_trained[1:length(p0_dS)], "p_dM", p_trained[length(p0_dS)+1:end],
	"scale", scale, "losses", losses, "prediction", pred, "hist_data", hist_data,
	"training_data", training_data, "test_data", test_data, "days", days)
nothing
# end

run_model()
