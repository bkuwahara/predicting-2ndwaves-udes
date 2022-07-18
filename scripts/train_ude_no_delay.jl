cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

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
const lr = 0.01

activation = relu

# function run_model(region)
model_name = "ude"
exp_name = "$(region)"

println("Starting run: $(exp_name)")

verbose = (isempty(ARGS)) ? true : false
opt = ADAM(lr)

## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIM_weekly_avg_2020_$(region).jld2"))
data = dataset["data"]
days = dataset["days"]

i_start = 1
while data[2,i_start] <= 1e-5
	i_start += 1
end
data = data[:, i_start:end]
# Take the first 14 days for history function

# Split the rest into training and testing
training_stop = Int(round(frac_training*size(data, 2)))
training_data = data[:, 1:training_stop]
test_data = data[:, training_stop+1:end]


t_train = range(0.0, length = size(training_data, 2), step = 1.0);
t_test = range(t_train[end]+1, length = size(test_data, 2), step = 1.0);

# Get scale factors from the data to improve training
yscale = maximum(training_data, dims=2) .- minimum(training_data, dims=2);
tscale = t_train[end] - t_train[1];
scale = yscale/tscale;


## Set up model
nn_dS = FastChain(FastDense(1, 3, activation), FastDense(3, 3, activation), FastDense(3, 3, activation), FastDense(3, 1))
nn_dM = FastChain(FastDense(2, 3, activation), FastDense(3, 3, activation), FastDense(3, 3, activation), FastDense(3, 1))
p0_dS = initial_params(nn_dS)
p0_dM = initial_params(nn_dM)

p_dS_len = length(p0_dS)
p_dM_len = length(p0_dM)


function ude(du, u, p, t)
	S, I, M = u
	du[1] = -S*I*abs(nn_dS(M, p[1:p_dS_len])[1])*scale[1]
	du[2] = -du[1] - recovery_rate*I
	du[3] = nn_dM([du[2]; M], p[p_dS_len+1:end])[1]*scale[3]
	nothing
end

u0 = training_data[:,1]
tspan = (t_train[1], t_train[end])
p0 = [p0_dS; p0_dM]


function predict(θ)
	Array(solve(prob_nn, Tsit5(),
		p=θ, saveat=1.0, sensealg=ForwardDiffSensitivity()))
end;
function loss(θ)
	pred=predict(θ)
	return (sum(abs2, (training_data[:,1:size(pred,2)] - pred)./yscale)/size(pred,2)), pred
end;
losses = []
training_iters = 0;
function callback(θ, l, pred)
	if verbose && (training_iters % 100 == 0)
		pl = plot(t_train[1:size(pred,2)], training_data[:,1:size(pred,2)]', layout=(3,1), label=["data" "" ""])
		plot!(pl, t_train[1:size(pred,2)], pred', label = ["prediction" "" ""])
		display(pl)
		display(l)
	end
	global training_iters += 1
	if training_iters % 100 == 0
		push!(losses, l)
	end
	if l > 1e12 && training_iters < 10
		throw("Bad initialization. Aborting...")
		return true
	elseif l < 1e-5
		println("Training converged after $(training_iters) iterations")
		return true
	else
		return false
	end
end;

stops = range(0.0, stop = t_train[end], length=6)
prob_nn = ODEProblem(ude, u0, (0.0, t_train[end]), p0)
p = p0
# for (i, stop) in enumerate(stops[2:end])
# 	prob_nn = remake(prob_nn, p=p, tspan=(0.0,stop))
# 	res = DiffEqFlux.sciml_train(loss, p, ADAM(lr), cb=callback, maxiters=1500, allow_f_increases=true)
# 	p = res.minimizer
# end


prob_nn = remake(prob_nn, p=p, tspan=(0.0,10.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.15), cb=callback, maxiters=1500, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.005), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, p=p, tspan=(0.0,20.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.005), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.005), cb=callback, maxiters=3000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.005), cb=callback, maxiters=10000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, p=p, tspan=(0.0,30.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.005), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer


prob_nn = remake(prob_nn, p=p, tspan=(0.0,40.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, p=p, tspan=(0.0,50.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, p=p, tspan=(0.0,70.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, p=p, tspan=(0.0,90.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer

prob_nn = remake(prob_nn, p=p, tspan=(0.0,120.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)

prob_nn = remake(prob_nn, p=p, tspan=(0.0,160.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.01), cb=callback, maxiters=3000, allow_f_increases=true)

prob_nn = remake(prob_nn, p=p, tspan=(0.0,200.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)

prob_nn = remake(prob_nn, p=p, tspan=(0.0,222.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=1000, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.001), cb=callback, maxiters=3000, allow_f_increases=true)


prob_nn = remake(prob_nn, p=p, tspan=(0.0,t_train[end]))
res = DiffEqFlux.sciml_train(loss, p, LBFGS(), cb=callback, maxiters=100, allow_f_increases=true)

println("Completed training. Final loss: $(losses[end])")

p_trained = res.minimizer

prob_final = ODEProblem(ude, u0, (t_train[1], t_test[end]), p_trained)
pred = Array(solve(prob_final, Tsit5(),
	p=p_trained, saveat=[t_train; t_test]))


## Analyze and save
pl1 = plot([t_train; t_test], [training_data test_data]', layout=(3,1),
	title=["$(region)" "" ""],
	label = ["True Data" "" "" ], ylabel=["S" "I" "Mobility"])
plot!(pl1, [t_train; t_test], pred', layout=(3,1),
	label = ["UDE Prediction" "" ""])
vline!(pl1, [stops[end] stops[end] stops[end]], color = :black,
	label=["End of Training Data" "" "" ])


pl2 = plot(range(1.0, step=100.0, length=length(losses)), losses, title="Training Losses: $(exp_name)",
	xlabel = "Iteration", ylabel = "Loss")
yaxis!(pl2, :log10)


savefig(pl1, plotsdir("no delay", "final_prediction_$(region)_tanh_dM.png"))
savefig(pl2, plotsdir("no delay", "losses_$(region)_tanh_dM.png"))

save(datadir("sims", "no delay", "results_$(region)_tanh_dM.jld2"),
	"p_dS", p_trained[1:length(p0_dS)], "p_dM", p_trained[length(p0_dS)+1:end], "p_end", p[end],
	"scale", scale, "losses", losses, "prediction", pred,
	"training_data", training_data, "test_data", test_data, "days", days)
# nothing
# end
p_dM = p_trained[length(p0_dS)+1:end]


M_vals = range(-100, stop = 100, length=200)
I_vals = range(0.0, stop=0.01, length=200)
Z_vals = [nn_dM([I_val/scale[2], M_val], p_dM)[1] for I_val in I_vals, M_val in M_vals].*scale[3]
pl = heatmap(I_vals, M_vals, Z_vals')
xlabel!(pl, "I")
ylabel!(pl, "M")
savefig(pl, plotsdir("no delay", "f_heatmap_$(region)"))



#
# using Plots
# plot(0.0:1.0:size(pred, 2)-1, pred', layout=(3,1))
