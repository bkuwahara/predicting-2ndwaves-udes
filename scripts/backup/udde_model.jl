using DrWatson
@quickactivate("S2022 Project")
using Lux, DiffEqFlux, Zygote
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using Plots
using Random; rng = Random.default_rng()


##  constant hyperparameters
τₘ = 10.0 # 14, 21, 28, 10, 25
τᵣ = 14.0 # 10, 14
const frac_training = 0.75
const maxiters = 2500
const lr = 0.05
const recovery_rate = 1/4
activation = relu
## Utility functions
# https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
function BlockingTimeSeriesSplit(n_splits::Int; margin=1, frac_training=0.8)
	function Splitter(data)
		n_samples = size(data, 2)
		k_fold_size = div(n_samples, n_splits)
        indices = 1:n_samples

		train_splits = []
		test_splits = []
        for i = 1:n_splits
            start = 1 + (i-1) * k_fold_size
            stop = start + k_fold_size - 1
            mid = Int(round((frac_training * (stop - start)) + start))
            push!(train_splits, indices[start: mid])
			push!(test_splits, indices[mid + margin: stop])
		end
		return collect(zip(train_splits, test_splits))
	end
	return Splitter
end


function TimeSeriesSplit(n_splits::Int, test_size::Int)
	@assert (n_splits >= 2) "n_splits must be at least 2"
	function Splitter(data)
		n_samples = size(data, 2)
		test_splits = [i:i+test_size for i in range(n_samples-test_size*(n_splits), stop = n_samples-test_size, step=test_size)]
		train_splits = [1:split[1]-1 for split in test_splits]
		return collect(zip(train_splits, test_splits))
	end
	return Splitter
end

## Model specification
network1 = Lux.Chain(
	Lux.Dense(num_indicators=>hidden_dims, tanh), Lux.Dense(hidden_dims=>1))
network2 = Lux.Chain(
	Lux.Dense(2+num_indicators=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators))
p1, st1 = Lux.setup(rng, network1)
p2, st2 = Lux.setup(rng, network2)

function nde(du, u, h, p, t)
	I_hist = h(p, t-τᵣ)[2]
	delta_I_hist = I_hist - h(p, t-(τᵣ+1))[2]
	du[1] = u[1]*u[2]*network1(h(p, t-τₘ)[3:end], p.layer_1, st1)[1][1]
	du[2] = -du[1] - recovery_rate*u[2]
	du[3:end] .= network2([I_hist; delta_I_hist; u[3:end]], p.layer_2, st2)[1]
	nothing
end
h(p,t) = all_data[:,1]
lags = [τᵣ τₘ (τᵣ+1)]
## Training

function train_model(train_data, t_train)

	# Get scale factors from the data to improve training
	yscale = maximum(train_data, dims=2) .- minimum(train_data, dims=2);
	tscale = t_train[end] - t_train[1];
	scale = yscale/tscale;


	function predict(θ; u0=u0, saveat=1.0)
		Array(solve(prob_nn, MethodOfSteps(Tsit5()), p=θ; u0=u0, saveat=saveat))
	end

	function loss(θ)
		pred = predict(θ)
		sum(abs2, (pred .- train_data)./yscale)/size(pred, 2), pred
	end

	losses = []
	function callback(θ, l, pred)
		push!(losses, l)
		if verbose && (length(losses) % 100 == 0)
			display(l)
			pl = scatter(t_train, train_data', layout=(2+num_indicators,1), color=:black)
			plot!(pl, t_train, pred', layout=(2+num_indicators,1), color=:red)
			display(pl)
		end
		if l > 1e12
			println("Bad initialization. Aborting...")
			return true
		end
		return false
	end


	p_init = Lux.ComponentArray(layer_1 = Lux.ComponentArray(p1), layer_2 = Lux.ComponentArray(p2))
	u0 = train_data[:,1]
	prob_nn = DDEProblem(nde, u0, h, (0.0, t_train[end]), p_init, constant_lags=lags)


	adtype = Optimization.AutoZygote()
	optf = Optimization.OptimizationFunction((p, u) -> loss(p), adtype)
	optprob = Optimization.OptimizationProblem(optf, p_init)
	resfinal = Optimization.solve(optprob, ADAM(0.05), maxiters=5000, callback=callback)
	return resfinal, losses
end

## run model

# Input hypterparameters
region = ARGS[1]
hidden_dims = ARGS[2]
indicators = parse.(Int, ARGS[3:end])
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)


model_name = "udde"
println("Starting run: $(region)")
verbose = (isempty(ARGS)) ? true : false


## Load data
dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
all_data = dataset["data"]
response_vars = all_data[1:2,:]
indicator_vars = all_data[indicator_idxs,:][1,:,:]

all_data = [response_vars; indicator_vars]
days = dataset["days"]
all_tsteps = range(-max(τᵣ+1, τₘ), step=1.0, length=size(all_data,2))


## Split the rest into pre-training (history), training and testing
hist_stop = Int(round(max(τᵣ+1, τₘ)))

hist_data = all_data[:, 1:hist_stop]
hist_tsteps = all_tsteps[1:hist_stop]

firstwave_data = all_data[:, hist_stop+1:hist_stop+120]
firstwave_tsteps = all_tsteps[hist_stop+1:hist_stop+120]

splitter = TimeSeriesSplit(5, 14)

splits = splitter(firstwave_data)

scores = zeros(length(splits))
for (i, split) in enumerate(splits)
	train_split, test_split = split

	train_data = firstwave_data[:, train_split]
	t_train = firstwave_tsteps[train_split]
	res, losses = train_model(train_data, t_train)

	test_data = firstwave_data[:, test_split]
	t_test = firstwave_tsteps[test_split]
	u0 = train_data[:,1]
	prob_test = DDEProblem(nde, u0, h, (t_train[1], t_test[end]), res.minimizer, constant_lags=lags)
	sol = Array(solve(prob_test, MethodOfSteps(Tsit5()), saveat=t_test))
	scores[i] = sum(abs2, sol - test_data)
end





## Analyze the results

# Network prediction
p_trained = resfinal.minimizer
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
pl_losses = plot(1:length(losses), losses, color=:red, label="ADAM")
yaxis!(pl_losses, :log10)


# beta time series
indicators_predicted = pred_final[3:end,:]
β = zeros(size(pred_final.t))
for i in 1:length(β)
	indicator = i <= length(hist_split) ? hist_data[3:end,1] : indicators_predicted[:, i-length(hist_split)]
	β[i] = network1(indicator, p_trained.layer_1, st1)[1][1]
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

fname = "$(indicator_name)-$(param_name)"

# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
model_iteration = 1
while isfile(datadir("sims", model_name, region, "$(fname)-$(model_iteration).jld2"))
	model_iteration += 1
end
fname = fname * "_v$(model_iteration)"

if ! isdir(datadir("sims", model_name, region, fname))
	mkdir(datadir("sims", model_name, region, fname))
end
savefig(pl_pred, datadir("sims", model_name, region, fname, "final_prediction.png"))
savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

save(datadir("sims", model_name, region, fname, "results.jld2"),
	"p", p_trained, "scale", scale, "losses", losses, "prediction", pred_final,
	"train_data", train_data, "test_data", test_data, "days", days,
	"taur", τᵣ, "taum", τₘ)
