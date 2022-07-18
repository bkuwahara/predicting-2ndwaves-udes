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



## Run


# Declare variables
const frac_training = 0.75
const recovery_rate = 1/4

const maxiters = 10000
const lr = 0.05

activation = gelu

# function run(region)
model_name = "ude_time_dependent"
exp_name = "$(region)"

println("Starting run: $(exp_name)")

verbose = (isempty(ARGS)) ? true : false
opt = ADAM(lr)

## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIM_weekly_avg_2020_$(region).jld2"))
data = dataset["data"][1:2,15:end]
days = dataset["days"]


training_stop = 15+Int(round(frac_training*size(data, 2)))
training_data = data[:,1:training_stop]
test_data = data[:, training_stop+1:end]

t_train = range(0.0, length = size(training_data, 2), step = 1.0);
t_test = range(t_train[end]+1, length = size(test_data, 2), step = 1.0);

# Get scale factors from the data to improve training
yscale = maximum(training_data, dims=2) .- minimum(training_data, dims=2);
tscale = t_train[end] - t_train[1];
scale = yscale/tscale;


## Set up model
nn = FastChain(FastDense(1, 6, gelu), FastDense(6, 6, gelu), FastDense(6, 6, gelu), FastDense(6, 1))
p0 = initial_params(nn)
u0 = training_data[:,1]


function ude(du, u, p, t)
	S, I = u
	du[1] = S*I*nn(t, p)[1]*scale[1]
	du[2] = -du[1] - recovery_rate*I
	nothing
end

function predict(θ)
	Array(solve(prob_nn, Tsit5(),
		p=θ, saveat=1.0, sensealg = ForwardDiffSensitivity()))
end;
function loss(θ)
	pred=predict(θ)
	return (sum(abs2, (training_data[:,1:size(pred,2)] - pred)./yscale)/size(pred,2)), pred
end;
losses = []
training_iters = 0;
function callback(θ, l, pred)
	if verbose
		pl = plot(t_train[1:size(pred,2)], training_data[:,1:size(pred,2)]', layout=(2,1), label=["data" ""])
		plot!(pl, t_train[1:size(pred,2)], pred', label = ["prediction" ""])
		display(pl)
		display(l)
	end
	# training_iters += 1
	if training_iters % 100 == 0
		push!(losses, l)
	end
	if l > 1e12 && training_iters < 10
		throw("Bad initialization. Aborting...")
		return true
	elseif l < 1e-3
		println("Training converged after $(training_iters) iterations")
		return true
	else
		return false
	end
end;

stops = range(0.0, stop = t_train[end], step=15)

prob_nn = ODEProblem(ude, u0, (0.0, stops[2]), p0)
res1 = DiffEqFlux.sciml_train(loss, p0, ADAM(lr), cb=callback, maxiters=div(maxiters,4), allow_f_increases=true)

p1 = res1.minimizer
prob_nn = ODEProblem(ude, u0, (0.0, stops[3]), p1)
res2 = DiffEqFlux.sciml_train(loss, p1, ADAM(0.1), cb=callback, maxiters=10, allow_f_increases=true)
res2 = DiffEqFlux.sciml_train(loss, res2.minimizer, LBFGS(), cb=callback, maxiters=100, allow_f_increases=true)


prob_nn = ODEProblem(ude, u0, (0.0, stops[4]), res2.minimizer)
res3 = DiffEqFlux.sciml_train(loss, res2.minimizer, ADAM(lr), cb=callback, maxiters=10, allow_f_increases=false)
res3 = DiffEqFlux.sciml_train(loss, res3.minimizer, LBFGS(), cb=callback, maxiters=100, allow_f_increases=false)


prob_nn = ODEProblem(ude, u0, (0.0, stops[5]), res3.minimizer)
res4 = DiffEqFlux.sciml_train(loss, res3.minimizer, ADAM(lr), cb=callback, maxiters=10, allow_f_increases=true)
res4 = DiffEqFlux.sciml_train(loss, res4.minimizer, LBFGS(), cb=callback, maxiters=100, allow_f_increases=true)

prob_nn = ODEProblem(ude, u0, (0.0, stops[end]), res4.minimizer)
res5 = DiffEqFlux.sciml_train(loss, res4.minimizer, ADAM(lr), cb=callback, maxiters=10, allow_f_increases=true)
res5 = DiffEqFlux.sciml_train(loss, res5.minimizer, LBFGS(), cb=callback, maxiters=100, allow_f_increases=true)


println("Completed training. Final loss: $(losses[end])")

p_trained = res5.minimizer

prob_final = ODEProblem(ude, u0, (t_train[1], t_test[end]), p_trained)
pred = Array(solve(prob_final, Tsit5(),
	p=p_trained, saveat=[t_train; t_test]))


## Analyze and save
pl1 = plot([t_train; t_test], [training_data test_data]', layout=(2,1),
	title=["$(region)" ""],
	label = ["True Data" ""], ylabel=["S" "I"])
plot!(pl1, [t_train; t_test], pred', layout=(3,1),
	label = ["UDE Prediction" ""])
vline!(pl1, [stops[end] stops[end] stops[end]], color = :black,
	label=["End of Training Data" ""])
ylims!(pl1[1], (minimum(data, dims=2)[1], maximum(data, dims=2)[1]))
ylims!(pl1[2], (minimum(data, dims=2)[2], maximum(data, dims=2)[2]))

savefig(pl1, plotsdir("time dependence concept", "final_prediction_$(region)"))

save(datadir("sims", "time dependence concept", "results_$(region).jld2"),
	"p", p_trained, "scale", scale, "losses", losses, "prediction", pred,
	"training_data", training_data, "test_data", test_data, "days", days)
nothing
# end


#
# using Plots
# plot(0.0:1.0:size(pred, 2)-1, pred', layout=(3,1))
