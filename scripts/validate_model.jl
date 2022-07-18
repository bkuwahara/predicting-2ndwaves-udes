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
using DataInterpolations
using Statistics



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

## Run

# Declare variables

const recovery_rate = 1/4
const frac_training = 0.75
const maxiters = 8000
const lr = 0.01
const dS_ls = 3
const dM_ls = 4

region = ARGS[1]
τₘ = parse(Float64, ARGS[2]) # 10, 14, 21, 28,
τᵣ = parse(Float64, ARGS[3]) # 10, 14

activation = relu
model_name = "udde_validation"

# function run_model()
println("Starting run: $(region)")

verbose = (isempty(ARGS)) ? true : false
opt = ADAM(lr)

## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIM_weekly_avg_2020_$(region).jld2"))
data = dataset["data"]
days = dataset["days"]
all_times = range(0.0, step=1.0, length=length(days))
# splits = TimeSeriesSplit(10, 28)(data)
splits = BlockingTimeSeriesSplit(5, frac_training=0.66)(data)


nn_dS = FastChain(FastDense(1, dS_ls, activation), FastDense(dS_ls, dS_ls, activation), FastDense(dS_ls, dS_ls, activation), FastDense(dS_ls, 1, relu))
nn_dM = FastChain(FastDense(3, dM_ls, activation), FastDense(dM_ls, dM_ls, activation), FastDense(dM_ls, dM_ls, activation), FastDense(dM_ls, 1))
p0_dS = initial_params(nn_dS)
p0_dM = initial_params(nn_dM)

p_dS_len = length(p0_dS)
p_dM_len = length(p0_dM)


function training_loop(data)
	# Take the first 14 days for history function
	# hist_stop = Int(round(max(τᵣ, τₘ)))
	hist_stop = Int(round(max(τₘ, τᵣ)))
	hist_data = data[:, 1:hist_stop]
	hist_interp = QuadraticInterpolation(hist_data, range(-max(τₘ, τᵣ), step=1.0, stop=-1.0))
	h(p, t) = hist_interp(t)
	data = data[:, hist_stop+1:end]
	times = range(0.0, step=1.0, length=size(data, 2))

	# Get scale factors from the data to improve training
	yscale = maximum(data, dims=2) .- minimum(data, dims=2);
	tscale = times[end] - times[1];
	scale = yscale/tscale;
	u0 = data[:,1]
	tspan = (times[1], times[end])
	p0 = [p0_dS; p0_dM]

	function udde(du, u, h, p, t)
		S, I, M = u
		du[1] = -S*I*nn_dS(h(p, t-τₘ)[3], p[1:p_dS_len])[1]*scale[1]
		du[2] = -du[1] - recovery_rate*I
		du[3] = nn_dM([du[2]; h(p, t-τᵣ)[2]/scale[2]; M], p[p_dS_len+1:end])[1]*scale[3]
		nothing
	end

	function predict(θ)
		Array(solve(prob_nn, MethodOfSteps(Tsit5()),
			p=θ, saveat=1.0, sensealg=ReverseDiffAdjoint()))
	end;

	function loss(θ)
		pred=predict(θ)
		if size(pred) != size(data)
			return Inf
		end
		return (sum(abs2, (data[:,2:end] - pred[:,2:end])./yscale)/size(pred,2)), pred
	end;

	training_iters = 0
	function callback(θ, l, pred)
		if verbose && (training_iters % 50 == 0)
			pl = plot(times[1:size(data,2)], data', layout=(3,1), label=["data" "" ""])
			plot!(pl, times[1:size(pred,2)], pred', label = ["prediction" "" ""])
			display(pl)
			display(l)
		end
		training_iters += 1
		return false
	end;

	u0 = data[:,1]
	tspan = (0.0, times[end])
	prob_nn = DDEProblem(udde, u0, h, tspan, p0; constant_lags=[τₘ; τᵣ])
	res = DiffEqFlux.sciml_train(loss, p0, ADAM(lr), cb=callback, maxiters=10000, allow_f_increases=true)
	return res.minimizer
end


function CV_loop(splits)
	scores = zeros(length(splits))
	trained_params = []
	for (i, split) in enumerate(splits[1:end-1]) # leave off last split for final training/testing
		train_split, test_split = split
		train_data = data[:, train_split]
		CV_data = data[:, test_split]
		p_trained = training_loop(train_data)
		push!(trained_params, p_trained)

		# Evaluate trained network
		hist_stop = Int(round(max(τₘ, τᵣ)))
		hist_data = train_data[:, 1:hist_stop]
		hist_interp = QuadraticInterpolation(hist_data, range(-max(τₘ, τᵣ), step=1.0, stop=-1.0))
		h(p, t) = hist_interp(t)

		# Get scale factors from the data to improve training
		yscale = maximum(train_data, dims=2) .- minimum(train_data, dims=2);
		tscale = 1.0*size(train_data, 2);
		scale = yscale/tscale;
		u0 = train_data[:,1]
		times = range(0.0, step=1.0, length=size(train_data,2)+size(CV_data, 2))
		tspan = (times[1], times[end])
		p0 = [p0_dS; p0_dM]

		function udde(du, u, h, p, t)
			S, I, M = u
			du[1] = -S*I*nn_dS(h(p, t-τₘ)[3], p[1:p_dS_len])[1]*scale[1]
			du[2] = -du[1] - recovery_rate*I
			du[3] = nn_dM([du[2]; h(p, t-τᵣ)[2]/scale[2]; M], p[p_dS_len+1:end])[1]*scale[3]
			nothing
		end

		prob_nn = DDEProblem(udde, u0, h, (0.0, times[end]), p_trained)
		sol = Array(solve(prob_nn, MethodOfSteps(Tsit5()), saveat=all_times[test_split]))
		if size(sol) != size(CV_data)
			display("Error! Mismatched dimensions.")
			scores[i] = Inf
		else
			scores[i] = (sum(abs2, (CV_data - sol)./yscale)/size(sol,2))
		end
	end
	return scores, trained_params
end


scores, params = CV_loop(splits)
total_score = mean(scores)
println("Current model final score: $(total_score)")

params = (dM = dM_ls, dS = dS_ls, tm = τₘ, tr = τᵣ, region=region)
name = savename(params, digits=0)
save(datadir("sims", model_name, region, "$(name).jld2"),
	"scores", scores,
	"tau_r", τᵣ, "tau_m", τₘ, "dM_ls", dM_ls, "dS_ls", dS_ls,
	"region", region, "days", days)




##
