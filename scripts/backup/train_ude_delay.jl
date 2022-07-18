cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim
using DifferentialEquations
using LinearAlgebra
using DiffEqSensitivity
using GalacticOptim
using JLD2, FileIO
using Dates
using Plots
using DataInterpolations



## Run

# Declare variables
τₘ = 10.0 # 14, 21, 28, 10, 25
τᵣ = 14.0 # 10, 14
const recovery_rate = 1/4
const frac_training = 0.75
const maxiters = 2500
const lr = 0.01
const dS_ls = 3
const dM_ls = 4


region = ARGS[1]


activation = gelu
model_name = "udde"

# function run_model()
println("Starting run: $(region)")

verbose = (isempty(ARGS)) ? true : false
opt = ADAM(lr)

## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIM_weekly_avg_2020_$(region).jld2"))
all_data = dataset["data"]
days = dataset["days"]


# Take the first 14 days for history function
hist_stop = Int(round(max(τᵣ+1, τₘ)))
hist_data = all_data[:, 1:hist_stop]
hist_interp = QuadraticInterpolation(hist_data, range(-hist_stop, step=1.0, stop=-1.0))
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
nn_dS = FastChain(FastDense(1, dS_ls, activation), FastDense(dS_ls, dS_ls, activation), FastDense(dS_ls, dS_ls, activation), FastDense(dS_ls, 1))
nn_dM = FastChain(FastDense(3, dM_ls, activation), FastDense(dM_ls, dM_ls, activation), FastDense(dM_ls, dM_ls, activation), FastDense(dM_ls, 1))
p0_dS = initial_params(nn_dS)
p0_dM = initial_params(nn_dM)

p_dS_len = length(p0_dS)
p_dM_len = length(p0_dM)


function udde(du, u, h, p, t)
	S, I, M = u
	I_t1 = h(p, t-(τᵣ +1))[2]
	I_t2 = h(p, t-τᵣ)[2]
	delta_I = (I_t2 -I_t1)
	I_information = [delta_I; I_t2]/scale[2]

	du[1] = -S*I*abs(nn_dS(h(p, t-τₘ)[3], p[1:p_dS_len])[1])*scale[1]
	du[2] = -du[1] - recovery_rate*I
	du[3] = nn_dM([I_information; M], p[p_dS_len+1:end])[1]*scale[3]
	nothing
end

u0 = training_data[:,1]
h(p, t) = hist_interp(t)
tspan = (t_train[1], t_train[end])
p0 = [p0_dS; p0_dM]


function predict(θ)
	Array(solve(prob_nn, MethodOfSteps(Tsit5()),
		p=θ, saveat=1.0, sensealg=ReverseDiffAdjoint()))
end;
function loss(θ)
	pred=predict(θ)
	return (sum(abs2, (training_data[:,1:size(pred,2)] - pred)./yscale)/size(pred,2)), pred
end;
losses = []
training_iters = 0;
function callback(θ, l, pred)
	global training_iters += 1
	if (verbose && training_iters % 100 == 0)
		pl = plot(t_train[1:size(pred,2)], training_data[:,1:size(pred,2)]', layout=(3,1), label=["data" "" ""])
		plot!(pl, t_train[1:size(pred,2)], pred', label = ["prediction" "" ""])
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

function test(idxs, p)
	data = training_data[:, idxs]
	hist_stop = Int(round(max(τᵣ, τₘ)))
	hist_data = data[:, 1:hist_stop]
	hist_interp = QuadraticInterpolation(hist_data, range(-τₘ, step=1.0, stop=-1.0))
	h(p, t) = hist_interp(t)

	data = data[:, hist_stop+1:end]
	times = range(0.0, step=1.0, length=size(data, 2))
	u0 = data[:,1]
	tspan = (times[1], times[end])
	prob_nn = DDEProblem(udde, u0, h, tspan, p)
	sol = Array(solve(prob_nn, MethodOfSteps(Tsit5()), saveat=1.0))

	pl = plot(range(0.0, step=1.0, length=size(sol, 2)), sol', label="prediction", layout=(3,1))
	plot!(pl, range(0.0, step=1.0, length=size(data, 2)), data', layour=(3,1), label="data")
	display(pl)
	err = (sum(abs2, (data - sol)./yscale))/size(sol,2)
	return err
end


## Training
prob_nn = DDEProblem(udde, u0, h, (0.0, t_train[end]), p0; constant_lags=[τᵣ + 1; τᵣ;  τₘ])
p = p0

prob_nn = remake(prob_nn, tspan = (0.0, 25.0), p=p)
res = DiffEqFlux.sciml_train(loss, p, ADAM(lr/2), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(lr), cb=callback, maxiters=5000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, tspan = (0.0, 50.0), p=p)
res = DiffEqFlux.sciml_train(loss, p, ADAM(lr/4), cb=callback, maxiters=100, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(lr/4), cb=callback, maxiters=1500, allow_f_increases=true)
p = res.minimizer

# err = test(50:78, p)

prob_nn = remake(prob_nn, tspan = (0.0, 75.0), p=p)
res = DiffEqFlux.sciml_train(loss, p, ADAM(lr/4), cb=callback, maxiters=5000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, tspan = (0.0, 90.0), p=p)
res = DiffEqFlux.sciml_train(loss, p, ADAM(lr/10), cb=callback, maxiters=1000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, tspan = (0.0, 120.0), p=p)
res = DiffEqFlux.sciml_train(loss, p, ADAM(lr/10), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(lr/5), cb=callback, maxiters=1000, allow_f_increases=true)

p = res.minimizer


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
