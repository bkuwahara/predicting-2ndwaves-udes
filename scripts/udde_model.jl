using DrWatson
@quickactivate("S2022_Project")
using Lux, DiffEqFlux, Zygote
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using Plots
using Random; rng = Random.default_rng()


# constant hyperparameters
τₘ = 10.0 # 14, 21, 28, 10, 25
τᵣ = 14.0 # 10, 14
const frac_training = 0.75
const maxiters = 2500
const lr = 0.005
const recovery_rate = 1/4
const indicators = [3]
activation = relu

function default_setup()
	region="US-NY"
	sequence_length=200
	hidden_dims = 3
	return region, sequence_length, hidden_dims
end
region, sequence_length, hidden_dims=default_setup()
if ARGS != []
	region = ARGS[1]
	sequence_length = parse(Int, ARGS[2])
	hidden_dims = parse(Int, ARGS[3])
end
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1


## Input hypterparameters

model_name = "udde"
println("Starting run: $(region)")
verbose = (isempty(ARGS)) ? true : false


## Load data
dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
all_data = dataset["data"][hcat([1 2], indicator_idxs), :][1,:,:]
days = dataset["days"]


## Split the rest into pre-training (history), training and testing
hist_stop = Int(round(max(τᵣ+1, τₘ)))
hist_split = 1:hist_stop
train_split = hist_stop+1:hist_stop + 90
test_split = hist_stop+90+1:size(all_data, 2)


hist_data = all_data[:, hist_split]
train_data = all_data[:, train_split]
test_data = all_data[:, test_split]

all_tsteps = range(-max(τᵣ+1, τₘ), step=1.0, length=size(all_data,2))
t_hist = all_tsteps[hist_split]
t_train = all_tsteps[train_split]
t_test = all_tsteps[test_split]

# Get scale factors from the data to improve training
yscale = maximum(train_data, dims=2) .- minimum(train_data, dims=2);
tscale = t_train[end] - t_train[1];
scale = yscale/tscale;


network1 = Lux.Chain(
	Lux.Dense(num_indicators=>hidden_dims, tanh), Lux.Dense(hidden_dims=>1), relu)
network2 = Lux.Chain(
	Lux.Dense(num_indicators=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators))
network3 = Lux.Chain(
	Lux.Dense(2=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators), relu)


function nde(du, u, h, p, t)
	I_hist = h(p, t-τᵣ)[2]
	delta_I_hist = I_hist - h(p, t-(τᵣ+1))[2]
	du[1] = u[1]*u[2]*network1(h(p, t-τₘ)[3:end], p.layer1, st1)[1][1]
	du[2] = -du[1] - recovery_rate*u[2]
	du[3] = -u[3]*network2([u[3]], p.layer2, st2)[1][1] - network3([I_hist; delta_I_hist], p.layer3, st3)[1][1]
	nothing
end


p1, st1 = Lux.setup(rng, network1)
p2, st2 = Lux.setup(rng, network2)
p3, st3 = Lux.setup(rng, network3)

p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2), layer3 = Lux.ComponentArray(p3))
u0 = train_data[:,1]
h(p,t) = hist_data[:,1]

prob_nn = DDEProblem(nde, u0, h, (0.0, t_train[end]), p_init, constant_lags=[τᵣ τₘ (τᵣ+1)])


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


function loss_monotonicity(model, X, p, st)
	loss = 0
	for (xi, xj) in X
		sgn = (xi-xj)*(model([xi], p, st)[1][1] - model([xj], p, st)[1][1])
		if sgn < 0
			loss += abs(sgn)
		else
			loss += 0
		end
	end
	return loss
end



function loss_combined(θ, M_domain_samples, I_domain_samples; monotonicity_weight=100)
	l0, pred = loss(θ)
	l1 = loss_monotonicity(network1, M_domain_samples, θ.layer1, st1)*monotonicity_weight
	l2 = loss_monotonicity(network2, M_domain_samples, θ.layer2, st2)*monotonicity_weight
	return (l0 + l1 + l2), pred
end


function train_combined(p; maxiters = maxiters, callback=false, monotonicity_weight=1)
	opt_st = Optimisers.setup(Optimisers.Adam(0.005), p)
	losses = []
	for epoch in 1:maxiters
		M_samples = [rand(Uniform(M_domain[1], M_domain[2]), 2) for j in 1:100]
		I_samples = [rand(Uniform(I_domain[1], I_domain[2]), 2) for j in 1:100]
	
		(l, pred), back = pullback(θ -> loss_combined(θ, M_samples, I_samples; monotonicity_weight=monotonicity_weight), p)
		
		push!(losses, l)
		gs = back((one(l), nothing))[1]
		opt_st, p = Optimisers.update(opt_st, p, gs)
	
		if callback
			display("Loss after $(length(losses)) epochs: $l")
			pl = scatter(t_train, train_data', layout=(2+num_indicators,1), color=:black, label=["data" nothing nothing])
			plot!(pl, t_train, pred', layout=(2+num_indicators,1), color=:red, label=["prediction" nothing nothing])
			display(pl)
		end
	end
	return p, losses
end


adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p, u) -> loss(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)
res = Optimization.solve(optprob, ADAM(0.05), maxiters=5000, callback=callback)

optprob2 = remake(optprob,u0 = res.minimizer)
res2 = Optimization.solve(optprob2, ADAM(0.05), maxiters=2500, callback=callback)

optprobfinal = remake(optprob,u0 = res2.minimizer)
resfinal = Optimization.solve(optprobfinal, ADAM(0.01), maxiters=200, callback=callback)



## Analyze the result

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
mkdir(datadir("sims", model_name, region, fname))

savefig(pl_pred, datadir("sims", model_name, region, fname, "final_prediction.png"))
savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

save(datadir("sims", model_name, region, fname, "results.jld2"),
	"p", p_trained, "scale", scale, "losses", losses, "prediction", pred_final,
	"train_data", train_data, "test_data", test_data, "days", days,
	"taur", τᵣ, "taum", τₘ)
# end








