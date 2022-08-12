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
τₘ = 10.0 # 14, 21, 28, 10, 25
τᵣ = 14.0 # 10, 14
const frac_training = 0.75
const maxiters = 2500
const lr = 0.005
const recovery_rate = 1/4
const indicators = [3]
const ϵ=0.01
activation = relu
adtype = Optimization.AutoZygote()

#===============================================
Input hypterparameters
================================================#

region="US-NY"
hidden_dims = 3
loss_weights = (1, 10, 10)

indicator_idxs = [3]
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1
model_name = "udde_animation"



#===========================================
Run the model
===========================================#
# function run_model()
println("Starting run: $(region)")
#===============================================
Load data
================================================#
dataset = load(datadir("exp_pro", "SIMX_7dayavg_roll=false_$(region).jld2"))
all_data = dataset["data"][hcat([1 2], indicator_idxs), :][1,:,:]
days = dataset["days"]
μ_mobility = dataset["mobility_mean"][indicator_idxs .- 2][1]
sd_mobility = dataset["mobility_std"][indicator_idxs .- 2][1]
mobility_baseline = -μ_mobility/sd_mobility

I_domain = (0.0, 1.0)
ΔI_domain = 10 .*(minimum(all_data[2,2:end] - all_data[2,1:end-1]), maximum(all_data[2,2:end] - all_data[2,1:end-1]))
M_domain = (-100.0, 100.0)

# Split the rest into pre-training (history), training and testing
hist_stop = Int(round(max(τᵣ+1, τₘ)/sample_period))
hist_split = 1:hist_stop
train_split = hist_stop+1:hist_stop + div(90, sample_period)
test_split = hist_stop+div(90, sample_period)+1:size(all_data, 2)


hist_data = all_data[:, hist_split]
train_data = all_data[:, train_split]
test_data = all_data[:, test_split]

all_tsteps = range(-max((τᵣ+1), τₘ), step=sample_period, length=size(all_data,2))
hist_tspan = (all_tsteps[1], 0.0)
t_train = range(0.0, step=sample_period, length=length(train_split))
t_test = range(t_train[end]+sample_period, step=sample_period, length=length(test_split))

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
    Lux.Dense(2+num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>num_indicators))
# network3 = Lux.Chain(
    # Lux.Dense(2=>hidden_dims, tanh), Lux.Dense(hidden_dims=>num_indicators))

# UDDE
function udde(du, u, h, p, t)
    I_hist = h(p, t-τᵣ)[2]/yscale[2]
    delta_I_hist = (I_hist - h(p, t-(τᵣ+1))[2])/yscale[2]
    du[1] = -u[1]*u[2]*network1(h(p, t-τₘ)[3:end], p.layer1, st1)[1][1]
    du[2] = -du[1] - recovery_rate*u[2]
    du[3] = network2([u[3]; I_hist; delta_I_hist], p.layer2, st2)[1][1] #network2([u[3]], p.layer2, st2)[1][1] - 
    nothing
end


p1, st1 = Lux.setup(rng, network1)
p2, st2 = Lux.setup(rng, network2)
# p3, st3 = Lux.setup(rng, network3)

p_init = Lux.ComponentArray(layer1 = Lux.ComponentArray(p1), layer2 = Lux.ComponentArray(p2))
u0 = train_data[:,1]
h(p,t) = [1.0; 0.0; mobility_baseline]

    
prob_nn = DDEProblem(udde, u0, h, (0.0, t_train[end]), p_init, constant_lags=[τᵣ τₘ (τᵣ+1)])


function predict(θ, tspan; u0=u0, saveat=sample_period)
    prob = remake(prob_nn, tspan = tspan, p=θ, u0=u0)
    Array(solve(prob, MethodOfSteps(Tsit5()), saveat=saveat))
end

function loss(θ, tspan)
    pred = predict(θ, tspan)
    if size(pred, 2) < abs(tspan[2] - tspan[1])/sample_period
        return Inf
    else
        return sum(abs2, (pred .- train_data[:, 1:size(pred, 2)])./yscale)/size(pred, 2), pred
    end
end

function loss_network1(M_samples, p, st)
    loss_negativity = 0
    loss_monotonicity = 0
    for i in eachindex(M_samples)[1:end]
        βi = network1([M_samples[i]], p, st)[1][1]
        βj = network1([M_samples[i]+ϵ], p, st)[1][1]
        sgn = βj - βi
        if sgn < 0
            loss_monotonicity += abs(sgn)
        end
        if βi < 0
            loss_negativity += abs(βi)
        end
    end
    return loss_monotonicity + loss_negativity
end


function loss_network2(M_samples, I_samples, ΔI_samples, p, st)
    loss_stability = 0
    loss_monotonicity = 0


    # Encourage monotonicity (decreasing) in both I and ΔI
    for i in eachindex(I_samples)
        # Tending towards M=mobility_baseline when I == ΔI == 0
        # Stabilizing effect is stronger at more extreme M
        dM1_M = network2([M_samples[i]; 0; 0], p, st)[1][1]
        dM2_M = network2([M_samples[i]+ϵ; 0; 0], p, st)[1][1]
        if dM1_M*(M_samples[i]-mobility_baseline) > 0
            loss_stability += dM1_M*(M_samples[i]-mobility_baseline)
        end

        sgn_M = abs(dM1_M) - abs(dM2_M)
        if sgn_M < 0
            loss_monotonicity += abs(sgn_M)
        end

        # Monotonicity in I
        dM1_I = network2([M_samples[i]; I_samples[i]; ΔI_samples[i]], p, st)[1][1]
        dM2_I = network2([M_samples[i]; I_samples[i]+ϵ; ΔI_samples[i]], p, st)[1][1]
        sgn_I = abs(dM1_I) - abs(dM2_I)
        if sgn_I > 0
            loss_monotonicity += sgn_I
        end

        # Monotonicity in ΔI
        dM1_ΔI = network2([M_samples[i]; I_samples[i]; ΔI_samples[i]], p, st)[1][1]
        dM2_ΔI = network2([M_samples[i]; I_samples[i]; ΔI_samples[i]+ϵ], p, st)[1][1]

        sgn_ΔI = (abs(dM1_ΔI) - abs(dM2_ΔI))
        if sgn_ΔI > 0
            loss_monotonicity += sgn_ΔI
        end
    end
    return loss_monotonicity + loss_stability
end



function loss_combined(θ, tspan, M_samples, I_samples, ΔI_samples, loss_weights)
    l0, pred = loss(θ, tspan)
    l1 = (loss_weights[2] == 0) ? 0 : loss_network1(M_samples, θ.layer1, st1)
    l2 = (loss_weights[3] == 0) ? 0 : loss_network2(M_samples, I_samples, ΔI_samples, θ.layer2, st2)
    return dot((l0, l1, l2), loss_weights), pred
end


function train_combined(p, tspan; maxiters = maxiters, loss_weights=(1, 10, 10), halt_condition=l->false, lr=lr)
    anim = Animation()
    opt_st = Optimisers.setup(Optimisers.Adam(lr), p)
    losses = []
    best_loss = Inf
    best_p = p
    for epoch in 1:maxiters
        M_samples = rand(Uniform(M_domain[1], M_domain[2]), 200)
        I_samples = rand(Uniform(I_domain[1], I_domain[2]), 200)
        ΔI_samples = rand(Uniform(ΔI_domain[1], ΔI_domain[2]), 200)


        (l, pred), back = pullback(θ -> loss_combined(θ, tspan, M_samples, I_samples, ΔI_samples, loss_weights), p)
        push!(losses, l)

        if l < best_loss
            best_loss = l
            best_p = p
        end

        gs = back((one(l), nothing))[1]
        opt_st, p = Optimisers.update(opt_st, p, gs)

        if halt_condition(l)
            break
        end

        if length(losses) % 50 == 0
            display(l)
            pl = scatter(t_train[1:size(pred, 2)], train_data[:,1:size(pred, 2)]', layout=(2+num_indicators,1), color=:black, 
                label=["Data" nothing nothing], ylabel=["S" "I" "M"])
            plot!(pl, t_train[1:size(pred, 2)], pred', layout=(2+num_indicators,1), color=:red,
                label=["Approximation" nothing nothing])
            xlabel!(pl[3], "Time")
            frame(anim)
        end
    end
    gif(anim, fps=15)
    return best_p, losses	
end


function train_fit(p, tspan; maxiters=maxiters, lr = 0.005)
    @assert tspan[1] % sample_period == 0
    optf = Optimization.OptimizationFunction((θ, u) -> loss(θ, tspan), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    res = Optimization.solve(optprob, ADAM(lr), maxiters=maxiters, callback=callback)
    return res.minimizer
end

p1, losses1 = train_combined(p_init, (t_train[1], t_train[end]/3); loss_weights=loss_weights, maxiters = 2500, lr=0.05)
p2, losses2 = train_combined(p1, (t_train[1], 2*t_train[end]/3); loss_weights=loss_weights, maxiters = 5000)
p_trained, losses3 = train_combined(p2, (t_train[1], t_train[end]); loss_weights=loss_weights, maxiters = 10000, lr=0.0005)





