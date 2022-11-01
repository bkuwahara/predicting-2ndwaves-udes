using DrWatson
@quickactivate("S2022_Project")
using Lux, DiffEqFlux, Zygote
using Optimisers
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using Plots
using Statistics, Distributions
using Random; rng = Random.default_rng()


#=================================
constant hyperparameters
=================================#
sample_period=7
τₘ = 14.0 # 14, 21, 28, 10, 25
τᵣ = 10.0 # 10, 14
const train_length = 160
const maxiters = 2500
const recovery_rate = 1/4
const indicators = [3]
const ϵ=0.01
const n_pts = 100
activation = relu
indicator_names = Dict(
	3 => "rr",
	4 => "wk",
	5 => "pk",
	6 => "si"
)
#===============================================
Utility functions
================================================#
function grad_basis(ind, dims)
	out = zeros(dims)
	out[ind] = 1.0
	return out
end

function weighted_vector_sum(weights, vecs)
	out = zero(vecs[1])
	for i in eachindex(vecs)
		out += weights[i]*vecs[i]
	end
	return out
end

function monotonicity_loss(f, x, ind, ϵ)
	df = f(x + ϵ*grad_basis(ind, size(x))) - f(x)
	return relu(-df)
end

# function invariant_loss(f, )

#===============================================
Input hypterparameters
================================================#

function default_setup()
	region="US-NY"
	sim_name = region
	hidden_dims = 3
	n_sims = 1
	return sim_name, region, hidden_dims, n_sims
end
sim_name, region, hidden_dims, n_sims = default_setup()
if ARGS != []
	sim_name = ARGS[1]
	region = ARGS[2]
	hidden_dims = parse(Int, ARGS[3])
	n_sims = parse(Int, ARGS[4])
end
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1
verbose = (isempty(ARGS)) ? true : false
model_name = "udde"

if !isdir(datadir("sims", model_name, sim_name)) 
	mkdir(datadir("sims", model_name, sim_name))
end

indicator_name = ""
for i=1:length(indicator_idxs)-1
	global indicator_name = indicator_name * indicator_names[indicator_idxs[i]] * "-"
end
indicator_name = indicator_name * indicator_names[indicator_idxs[end]]

params = (taum = τₘ, taur = τᵣ, hdims= hidden_dims)
param_name = savename(params)
#===============================================
Load data
================================================#
dataset = load(datadir("exp_pro", "SIMX_7dayavg_roll=false_$(region).jld2"))
all_data = dataset["data"][hcat([1 2], indicator_idxs), :][1,:,:]
days = dataset["days"]
μ_mobility = dataset["mobility_mean"][indicator_idxs .- 2][1]
sd_mobility = dataset["mobility_std"][indicator_idxs .- 2][1]
mobility_baseline = -μ_mobility/sd_mobility
mobility_min = (-1.0 - μ_mobility)/sd_mobility
population = dataset["population"]

# Split the rest into pre-training (history), training and testing
train_start_ind = 1
while all_data[2, train_start_ind] == 0
	global train_start_ind += 1
end

hist_stop = train_start_ind - 1
hist_split = 1:hist_stop
train_split = train_start_ind:train_start_ind + div(train_length, sample_period) - 1
test_split = train_start_ind + div(train_length, sample_period):size(all_data, 2)

hist_data = all_data[:, hist_split]
train_data = all_data[:, train_split]
test_data = all_data[:, test_split]

all_tsteps = range(-hist_stop*sample_period, step=sample_period, length=size(all_data,2))
hist_tspan = (all_tsteps[1], 0.0)
t_train = range(0.0, step=sample_period, length=length(train_split))
t_test = range(t_train[end]+sample_period, step=sample_period, length=length(test_split))

u0 = train_data[:,1]
h(p,t) = hist_data[:,end]
tspan = [0.0, t_train[end]]
#===============================================
Set up model
================================================#
# Get scale factors from the data to improve training
yscale = maximum(train_data, dims=2) .- minimum(train_data, dims=2);
tscale = t_train[end] - t_train[1];
scale = yscale/tscale;

network1 = Lux.Chain(
	Lux.Dense(num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>1))
network2 = Lux.Chain(
	Lux.Dense(3+num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>num_indicators))

p1_temp, st1 = Lux.setup(rng, network1)
p2_temp, st2 = Lux.setup(rng, network2)
p_temp = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1_temp), layer2 = Lux.ComponentArray(p2_temp))

# UDDE
function udde(du, u, h, p, t)
	S, I, M = u
	S_hist, I_hist, M_hist = h(p, t-τₘ)
	delta_I_hist = I - h(p, t-τᵣ)[2]
	du[1] = -S*I*network1(h(p, t-τₘ)[3:end], p.layer1, st1)[1][1]
	du[2] = -du[1] - recovery_rate*u[2]
	du[3] = network2([u[3]; I_hist/yscale[2]; delta_I_hist/yscale[2]; (1-S-I)/yscale[1]], p.layer2, st2)[1][1] 
	nothing
end
prob_nn = DDEProblem(udde, u0, h, tspan, p_temp, constant_lags=[τᵣ τₘ])


function predict(θ, tspan; u0=u0, saveat=sample_period)
	prob = remake(prob_nn, tspan = tspan, p=θ, u0=u0)
	Array(solve(prob, MethodOfSteps(Tsit5()), saveat=saveat))
end

function lr(p, tspan)	
	# Accuracy loss term
	pred = predict(p, tspan)
	l = size(pred, 2) < abs(tspan[2] - tspan[1])/sample_period ? Inf : sum(abs2, (pred .- train_data[:, 1:size(pred, 2)])./yscale)/size(pred, 2)
	return l, pred
end

function l_layer1(p, Ms)
	beta_i = network1(Ms, p, st1)[1]
	beta_j = network1(Ms .+ ϵ, p, st1)[1]

	l1 = sum(relu.( -beta_i))
	l5 = sum(relu.( beta_i .- beta_j))

	return [l1, l5]
end


M_min = mobility_min .* ones(1, n_pts)
I_baseline = zeros(1, n_pts)
M_max = 2*(mobility_baseline - mobility_min) .* ones(1,n_pts)
function l_layer2(p, Is, ΔIs, Ms, Rs)
	# Must not decrease when M at M_min
	dM_min = network2([M_min; Is; ΔIs; Rs], p, st2)[1]
	l6 = sum(relu.(-dM_min))

	# Must not increase when M at 2*(mobility_baseline - mobility_min)
	dM_max = network2([M_max; Is; ΔIs; Rs], p, st2)[1]
	l6 = sum(relu.(dM_max))

	# Tending towards M=mobility_baseline when I == ΔI == 0
	dM1_baseline = network2([Ms; I_baseline; I_baseline; Rs], p, st2)[1]
	dM2_baseline = network2([Ms .+ ϵ; I_baseline; I_baseline; Rs], p, st2)[1]
	dM3_baseline = network2([Ms; I_baseline; I_baseline; Rs .+ ϵ], p, st2)[1]
	l2 = sum(relu, relu.(dM1_baseline .* (Ms .- mobility_baseline)))

	# Stabilizing effect is stronger at more extreme M and higher R
	sgn_M = abs.(Ms) .- abs.(Ms .+ ϵ)
	sgn_dM = abs.(dM1_baseline) .- abs.(dM2_baseline)
	l3 = sum(relu.(-1 .* sgn_M .* sgn_dM))
	l7 = sum(relu.(abs.(dM1_baseline) .- abs.(dM3_baseline)))

	## Monotonicity terms
	dM_initial = network2([Ms; Is; ΔIs; Rs], p, st2)[1]

	# f monotonically decreasing in I
	dM_deltaI = network2([Ms; (Is .+ ϵ); ΔIs; Rs], p, st2)[1]
	l3 = sum(relu.(dM_deltaI .- dM_initial))
	
	# f monotonically decreasing in ΔI
	dM2_delta_deltaI = network2([Ms; Is; ΔIs .+ ϵ; Rs], p, st2)[1]
	l4 = sum(relu.(dM2_delta_deltaI .- dM_initial))
	return [l2, l3, l4, l6, l7]
end

function get_inputs(n)
	I_ = rand(rng, Uniform(0.0, 1.0), 1, n)
	ΔI = rand(rng, 1, n) .+ (I_ .- 1)
	R = rand(rng, 1, n) .* (1 .- I_)
	M = rand(rng, Uniform(mobility_min, 2*mobility_baseline-mobility_min), 1, n)

	return I_/yscale[2], ΔI/yscale[2], M, R/yscale[1]
end



function train_combined(p, tspan; maxiters = maxiters, loss_weights = ones(7), halt_condition=l->false, η=1e-3)
	opt_st = Optimisers.setup(Optimisers.Adam(η), p)
	losses = zeros(8)
	best_loss = Inf
	best_p = p

	for iter in 1:maxiters
		Is, ΔIs, Ms, Rs = get_inputs(n_pts)
		(l0, pred), back_all = pullback(θ -> lr(θ, tspan), p)
		g_all = back_all((one(l0), nothing))[1]
		if isnothing(g_all)
			println("No gradient found. Loss: $l0")
			g_all = zero(p)
		end

		layer1_losses, back1 = pullback(θ -> l_layer1(θ, Ms), p.layer1)
		g_layer1 = [back1(grad_basis(i, length(layer1_losses)))[1] for i in eachindex(layer1_losses)]

		layer2_losses, back2 = pullback(θ -> l_layer2(θ, Is, ΔIs, Ms, Rs), p.layer2)
		g_layer2 = [back2(grad_basis(i, length(layer2_losses)))[1] for i in eachindex(layer2_losses)]
		li = [layer1_losses; layer2_losses]

		# Store best iteration
		l_net = l0 + dot(li, loss_weights)
		losses = hcat(losses, [l0; li])
		if l_net < best_loss
			best_loss = l_net
			best_p = p
		end
		if verbose && iter % 50 == 0
			display("Total loss: $l_net, constraint losses: $(li)")
			# pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black, 
			# 	label=["Data" nothing nothing], ylabel=["S" "I" "M"])
			# plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red,
			# 	label=["Approximation" nothing nothing])
			# xlabel!(pl[3], "Time")
			# display(pl)
		end

		# Update parameters using the gradient
		g_net = Lux.ComponentArray(layer1 = g_all.layer1 + weighted_vector_sum(loss_weights[1:length(g_layer1)], g_layer1), 
			layer2 = g_all.layer2 + weighted_vector_sum(loss_weights[length(g_layer1)+1:end], g_layer2))
		opt_st, p = Optimisers.update(opt_st, p, g_net)


		if halt_condition([l0; li])
			break
		end
	end	
	return best_p, losses[:,2:end]#, loss_weights
end

function run_model()
	println("Starting run: $(region) on thread $(Threads.threadid())")

	# Get a new parameter set for each model run
	p1, st1 = Lux.setup(rng, network1)
	p2, st2 = Lux.setup(rng, network2)
	p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2))


	# Make sure to start with an initially stable parameterization
	while lr(p_init, (t_train[1], t_train[end]/4))[1] > 1e4
		println("Unstable initial parameterization. Restarting...")
		p1, st1 = Lux.setup(rng, network1)
		p2, st2 = Lux.setup(rng, network2)
		p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2))
	end

	halt_condition_1 = l -> (l[1] < 0.5) && sum(l[2:end]) < 0.1
	p1, losses1 = train_combined(p_init, tspan/4; maxiters = 10000, loss_weights = ones(7), halt_condition=halt_condition_1)
	println("Finished initial training on thread $(Threads.threadid())")

	p2, losses1 = train_combined(p1, tspan/2; maxiters = 2500, loss_weights=5*ones(7))
	println("Finished stage 2 training on thread $(Threads.threadid())")

	halt_condition_2 = l -> (l[1] < 1e-2) && sum(l[2:end]) < 5e-5
	p_trained, losses_final = train_combined(p1, (t_train[1], t_train[end]); maxiters = 10000, η=0.0005, loss_weights=10*ones(7), halt_condition=halt_condition_2)
	println("Finished final training on thread $(Threads.threadid())")


	#====================================================================
	Final results
	=====================================================================#

	# Long-term prediction
	prob_lt = remake(prob_nn, p=p_trained, tspan=(0.0, 3*365.0), constant_lags=[τᵣ τₘ])
	pred_lt = solve(prob_lt, MethodOfSteps(Tsit5()), saveat=1.0)

	M_test = range(mobility_min, step=0.1, stop=2*(mobility_baseline - mobility_min))
	M_test = Array(reshape(M_test, 1, length(M_test)))
	β = network1(M_test, p_trained.layer1, st1)[1]

	# Save the result
	fname = "$(region)_$(indicator_name)_$(param_name)_t$(Threads.threadid())"

	# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("sims", model_name, sim_name, "$(fname)_v$(model_iteration)"))
		model_iteration += 1
	end
	fname = fname * "_v$model_iteration"

	mkdir(datadir("sims", model_name, sim_name, fname))


	save(datadir("sims", model_name, sim_name, fname, "results.jld2"),
		"p", p_trained, "scale", scale, "losses", losses_final, "prediction", Array(pred_lt), "betas", β, 
		"hist_data", hist_data,	"train_data", train_data, "test_data", test_data, "days", days,
		"taur", τᵣ, "taum", τₘ, "loss_weights", 10*ones(7), 
		"mobility_mean", μ_mobility, "mobility_std", sd_mobility)
	println("Finished run: $(region) on thread $(Threads.threadid())")

	return nothing
end


Threads.@threads for i = 1:n_sims
	run_model()
end


