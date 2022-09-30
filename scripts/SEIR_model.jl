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
const train_length = 140
const maxiters = 2500
const recovery_rate = 1/4
const ϵ=0.01
activation = relu
adtype = Optimization.AutoZygote()

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



#===============================================
Input hypterparameters
================================================#

function default_setup()
	region="US-NY"
	sim_name = region
	hidden_dims = 3
	return sim_name, region, hidden_dims
end
sim_name, region, hidden_dims = default_setup()
if ARGS != []
	sim_name = ARGS[1]
	region = ARGS[2]
	hidden_dims = parse(Int, ARGS[3])
end

verbose = (isempty(ARGS)) ? true : false
model_name = "udde"



#===========================================
Run the model
===========================================#
# function run_model()
println("Starting run: $(region)")
#===============================================
Load data
================================================#
dataset = load(datadir("exp_pro", "IRMX_7dayavg_roll=false_$(region).jld2"))
all_data = dataset["data"][1:3,:]
days = dataset["days"]
μ_mobility = dataset["mobility_mean"][1]
sd_mobility = dataset["mobility_std"][1]
mobility_baseline = -μ_mobility/sd_mobility
mobility_min = (-1.0 - μ_mobility)/sd_mobility
population = dataset["population"]

# Split the rest into pre-training (history), training and testing
train_start_ind = 1
while all_data[2, train_start_ind]*population < 500
    train_start_ind += 1
end

train_split = train_start_ind:train_start_ind + div(train_length, sample_period) - 1
test_split = train_start_ind + div(train_length, sample_period):size(all_data, 2)

train_data = all_data[:, train_split]
test_data = all_data[:, test_split]

tsteps = range(0.0, step=sample_period, length=size(train_data,2) + size(test_data, 2))
t_train = tsteps[1:length(train_split)]
t_test = tsteps[length(train_split)+1:end]

#===============================================
Set up model
================================================#
# Get scale factors from the data to improve training
yscale = maximum(train_data, dims=2) .- minimum(train_data, dims=2);
tscale = t_train[end] - t_train[1];
scale = yscale/tscale;

network1 = Lux.Chain(
    Lux.Dense(1=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>1))
network2 = Lux.Chain(
    Lux.Dense(3=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>1))
# network3 = Lux.Chain(
    # Lux.Dense(2=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators))

# UDDE
function ude(du, u, p, t)
    E, I, R, M = u
    du[1] = S*I*network1([M], p.layer1, st1)[1][1] - p.latency_period*E # dE
    du[2] = p.latency_period*E - recovery_rate*I # dI
    du[3] = recovery_rate*u[2] # dR
    du[4] = network2([I; R; M], p.layer2, st2)[1][1]  # dM
    nothing
end

p1, st1 = Lux.setup(rng, network1)
p2, st2 = Lux.setup(rng, network2)
δ = rand()

p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2), delta=δ)
u0 = [train_data[:,1]]

    
prob_nn = ODEProblem(ude, u0, (0.0, t_train[end]), p_init)


function predict(θ, tspan; u0=u0, saveat=sample_period)
    prob = remake(prob_nn, tspan = tspan, p=θ, u0=u0)
    Array(solve(prob, MethodOfSteps(Tsit5()), saveat=saveat, save_idxs = [2,3,4]))
end

function lr(p, tspan)	
    # Accuracy loss term
    pred = predict(p, tspan)
    if size(pred, 2) < abs(tspan[2] - tspan[1])/sample_period
        return Inf
    else
        return sum(abs2, (pred .- train_data[:,1:size(pred, 2)])./yscale)/size(pred, 2)
    return l, pred
end

function l_layer1(p, Ms)
    l1 = 0
    l5 = 0
    for M in Ms
        # Monotonicity of beta in M
        βi = network1([M], p, st1)[1][1]
        βj = network1([M + ϵ], p, st1)[1][1]
        sgn = βj - βi
        l1 += relu(-sgn)

        # Nonnegativity of beta
        l5 += relu(-βi)
    end
    return [l1, l5]
end

function l_layer2(p, Xs)
    l2 = 0
    l3 = 0
    l4 = 0
    l6 = 0 
    for X in Xs
        I, R, M = X

        # Must not decrease when M at M_min
        dM_min = network2([I; R; mobility_min], p, st2)[1][1]
        l6 += relu(-dM_min)

        # Tending towards M=mobility_baseline when I == ΔI == 0
        dM1_M = network2([0; R; M], p, st2)[1][1]
        dM2_M = network2([0; R; M+ϵ], p, st2)[1][1]
        l2 += relu(dM1_M*(M-mobility_baseline))

        # Stabilizing effect is stronger at more extreme M
        sgn_M = abs(dM1_M) - abs(dM2_M)
        l3 += relu(-sgn_M)

        # f monotonically decreasing in I
        dM1_I = network2([I; R; M], p, st2)[1][1]
        dM2_I = network2([(I+ϵ);R ; M], p, st2)[1][1]
        l3 += relu(dM2_I - dM1_I)
        
        # f monotonically increasing in R
        dM2_R = network2([I; R+ϵ; M], p, st2)[1][1]
        l4 += relu(dM1_I - dM2_R)
    end
    return [l2, l3, l4, l6]
end


function random_point(rng)
    I = rand(rng, Uniform(0.0, 1.0))
    R = rand(rng, Uniform(0.0, I))
    M = rand(rng, Uniform(mobility_baseline, 10.0))
    return [I, R, M]
end


function train_combined(p, tspan; maxiters = maxiters, loss_weights = ones(6), halt_condition=l->false, η=1e-3)
    opt_st = Optimisers.setup(Optimisers.Adam(η), p)
    losses = []
    best_loss = Inf
    best_p = p

    for iter in 1:maxiters
        Xs = [random_point(rng) for i = 1:100]
        (l0, pred), back_all = pullback(θ -> lr(θ, tspan), p)
        g_all = back_all((one(l0), nothing))[1]

        layer1_losses, back1 = pullback(θ -> l_layer1(θ, [X[3] for X in Xs]), p.layer1)
        g_layer1 = [back1(grad_basis(i, length(layer1_losses)))[1] for i in eachindex(layer1_losses)]

        layer2_losses, back2 = pullback(θ -> l_layer2(θ, Xs), p.layer2)
        g_layer2 = [back2(grad_basis(i, length(layer2_losses)))[1] for i in eachindex(layer2_losses)]
        li = [layer1_losses; layer2_losses]

        # Store best iteration
        l_net = l0 + dot(li, loss_weights)
        push!(losses, li)
        if l_net < best_loss
            best_loss = l_net
            best_p = p
        end
        if verbose && iter % 50 == 0
            display("Total loss: $l_net, constraint losses: $(li)")
            pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black, 
                label=["Data" nothing nothing], ylabel=["S" "I" "M"])
            plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red,
                label=["Approximation" nothing nothing])
            xlabel!(pl[3], "Time")
            display(pl)
        end

        # Update parameters using the gradient
        g_net = Lux.ComponentArray(layer1 = g_all.layer1 + weighted_vector_sum(loss_weights[1:length(g_layer1)], g_layer1), 
            layer2 = g_all.layer2 + weighted_vector_sum(loss_weights[length(g_layer1)+1:end], g_layer2))
        opt_st, p = Optimisers.update(opt_st, p, g_net)


        if halt_condition([l0; li])
            break
        end
    end
    return best_p, losses, loss_weights
end


p1, losses1, loss_weights = train_combined(p_init, (t_train[1], t_train[end]/3); maxiters = 2500, loss_weights = ones(6))
p2, losses2, loss_weights = train_combined(p1, (t_train[1], 2*t_train[end]/3); maxiters = 5000, loss_weights=loss_weights)

halt_condition = l -> (l[1] < 1e-2) && sum(l[2:end]) < 5e-5
p_trained, losses3, loss_weights = train_combined(p1, (t_train[1], t_train[end]); maxiters = 10000, η=0.0005, loss_weights=loss_weights, halt_condition=halt_condition)


#====================================================================
Analyze the result
=====================================================================#
# Network versus test data
prob_test = remake(prob_nn, p=p_trained, tspan=(t_train[1], t_test[end]))
pred_test = solve(prob_test, MethodOfSteps(Tsit5()), saveat=1.0)

pl_pred_test = scatter(all_tsteps, all_data', label=["True data" nothing nothing nothing nothing nothing],
    color=:black, layout=(2+num_indicators, 1))
plot!(pl_pred_test, pred_test.t, Array(pred_test)', label=["Prediction" nothing nothing nothing nothing nothing],
    color=:red, layout=(2+num_indicators, 1))
vline!(pl_pred_test[end], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
    label=["Training" "" "" "" ""])
for i = 1:2+num_indicators-1
    vline!(pl_pred_test[i], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
        label=["" "" "" "" "" "" ""])
end

# Long-term prediction
prob_lt = remake(prob_nn, p=p_trained, tspan=(t_train[1], 2*t_test[end]))
pred_lt = solve(prob_lt, MethodOfSteps(Tsit5()), saveat=1.0)

pl_pred_lt = plot(pred_lt.t, Array(pred_lt)', label=["Long-term Prediction" nothing nothing nothing nothing nothing],
    color=:red, layout=(2+num_indicators, 1))
vline!(pl_pred_lt[end], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
    label=["Training" "" "" "" ""])
for i = 1:2+num_indicators-1
    vline!(pl_pred_lt[i], [hist_tspan[end] t_train[end]], color=:black, style=:dash,
        label=["" "" "" "" "" "" ""])
end

# beta time series
indicators_predicted = pred_test[3:end,:]
β = zeros(size(pred_test.t))
for i in 1:length(β)
    indicator = i <= length(hist_split) ? hist_data[3:end,1] : indicators_predicted[:, i-length(hist_split)]
    β[i] = network1(indicator, p_trained.layer1, st1)[1][1]
end
pl_beta_timeseries = plot(range(-length(hist_data), length=length(β), stop=all_tsteps[end]), β,
    xlabel="t", ylabel="β", label=nothing, title="Predicted force of infection over time")


# beta dose-response curve
β = [network1([M], p_trained.layer1, st1)[1][1] for M in range(mobility_min, step=0.1, stop=5.0)]
pl_beta_response = plot(range(mobility_min, step=0.1, stop=5.0), β, xlabel="M", ylabel="β", 
    label=nothing, title="Force of infection response to mobility")
vline!(pl_beta_response, [mobility_baseline], color=:red, label="Baseline", style=:dot,
    legend=:topleft)
vline!(pl_beta_response, [minimum(train_data[3,:]) maximum(train_data[3,:])], color=:black, label=["Training range" nothing], 
    style=:dash)


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

params = (taum = τₘ, taur = τᵣ, hdims= hidden_dims)
param_name = savename(params)

fname = "$(region)_$(indicator_name)_$(param_name)"

# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
model_iteration = 1
while isdir(datadir("sims", model_name, sim_name, "$(fname)_v$(model_iteration)"))
    model_iteration += 1
end
fname = fname * "_v$model_iteration"
if !isdir(datadir("sims", model_name, sim_name))
    mkdir(datadir("sims", model_name, sim_name))
end
mkdir(datadir("sims", model_name, sim_name, fname))

savefig(pl_pred_test, datadir("sims", model_name, sim_name, fname, "test_prediction.png"))
savefig(pl_pred_lt, datadir("sims", model_name, sim_name, fname, "long_term_prediction.png"))
savefig(pl_beta_timeseries, datadir("sims", model_name, sim_name, fname, "beta_timeseries.png"))
savefig(pl_beta_response, datadir("sims", model_name, sim_name, fname, "beta_response.png"))


save(datadir("sims", model_name, sim_name, fname, "results.jld2"),
    "p", p_trained, "scale", scale, "losses", losses3, "prediction", Array(pred_lt),
    "hist_data", hist_data,	"train_data", train_data, "test_data", test_data, "days", days,
    "taur", τᵣ, "taum", τₘ, "loss_weights", loss_weights, 
    "mobility_mean", μ_mobility, "mobility_std", sd_mobility)
    return nothing
# end


run_model()



