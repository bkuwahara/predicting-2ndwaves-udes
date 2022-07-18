cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")

using Lux, DiffEqFlux, Zygote, Optimisers
using MLUtils
using Optimization, OptimizationOptimJL, OptimizationFlux, OptimizationPolyalgorithms
using DifferentialEquations
using LinearAlgebra
using JLD2, FileIO
using Dates
using Plots
using Statistics
using Random; rng = Random.default_rng()

## Input hypterparameters
function default_setup()
	region="US-NY"
	sequence_length=200
	hidden_dims = 3
	indicators=[4, 5]
	return region, sequence_length, hidden_dims, indicators
end

region, sequence_length, hidden_dims, indicators=default_setup()
if ARGS != []
	region = ARGS[1]
	sequence_length = parse(Int, ARGS[2])
	hidden_dims = parse(Int, ARGS[3])
	indicators = parse.(Int, ARGS[4:end])
end
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1


struct Splitter
    n_splits::Int
    max_train_size::Int
    test_size::Int
    gap::Int
end

function Splitter(; n_splits=5, max_train_size=nothing, test_size=nothing, gap=0)
    @assert(n_splits >=2, "Number of splits must be at least 2")
    if isnothing(max_train_size)
        max_train_size = typemax(Int)
    end
    if isnothing(test_size)
        test_size = 0
    end
    return Splitter(n_splits, max_train_size, test_size, gap)
end

function (S::Splitter)(data::AbstractArray{T}) where{T}
    n_samples = size(data, 2)
    test_size = (S.test_size == 0) ? div(n_samples, S.n_splits+1) - S.gap : S.test_size
    test_splits = [n_samples - j*test_size + 1:n_samples - (j-1)*test_size for j = S.n_splits:-1:1]

    train_splits = [1 + split[1] - S.gap - min(i*div(n_samples, S.n_splits+1)+(n_samples % (S.n_splits + 1)), S.max_train_size)-1:split[1] - S.gap - 1 for (i, split) in enumerate(test_splits)]  
    return collect(zip(train_splits, test_splits))
end

S = Splitter(n_splits=3)

splits = S(X)

# Desired loop: Warmup on the first 14 days by passing in  1 day at a time and looping through
# First perform supervised learning on the two models separately, then train them in conjunction
struct LSTMToDense{L, D} <:
		Lux.AbstractExplicitContainerLayer{(:lstm_cell, :dense)}
	lstm_cell::L
	dense::D
end

function LSTMToDense(input_dims, hidden_dims, output_dims)
	return LSTMToDense(
		Lux.LSTMCell(input_dims => hidden_dims),
		Lux.Dense(hidden_dims => output_dims)
	)
end

# Initialization: creates hidden state and memory state h0 and m0
function (LD::LSTMToDense)(u0::AbstractArray{T, 2}, p, st) where {T}
	(h0, m0), st_lstm = LD.lstm_cell(u0, p.lstm_cell, st.lstm_cell)
	st_new = merge(st, (lstm_cell = st_lstm, dense = st.dense))
	return (h0, m0), st_new
end

# Standard propagation call
function (LD::LSTMToDense)(vhm::Tuple, p, st)
	v, h, m = vhm
	(h_new, m_new), st_lstm = LD.lstm_cell((v, h, m), p.lstm_cell, st.lstm_cell)
	out, st_dense = LD.dense(h_new, p.dense, st.dense)
	st_new = merge(st, (lstm_cell=st_lstm, dense=st_dense))
	return (out, h_new, m_new), st_new
end

indicator_names = Dict(
	3 => "rr",
	4 => "wk",
	5 => "pk",
	6 => "si"
)


## Declare hyperparameters
const γ = 1/4
const frac_training = 0.75
const lr = 0.01
λ = 0

# function run_model()

model_name = "discrete_model_LSTM"
println("Starting run: $(region)")
verbose = (isempty(ARGS)) ? true : false


## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
days = dataset["days"]
data = dataset["data"][hcat([1 2], indicator_idxs),:][1,:,:]
data = reshape(data, size(data)[1], size(data)[2], 1)
start_idx=15 # two weeks used for warmup


warmup_data = data[:, 1:start_idx-1,:]
data = data[:, start_idx:end]
function warmup!(LD::LSTMToDense, p, st, warmup_data)
    (h, m), st = LD(warmup_data[:,1,:], p, st)
    for j=2:size(warmup_data,2)
        (out, h, m), st = LD((warmup_data[:,j,:], h, m), p, st)
    end
    return (h, m), st
end


changes = data[:, 2:end,:] .- data[:, 1:end-1,:]
β_vals = @. -changes[1,:,:]/(data[1,1:end-1,:]*data[2, 1:end-1,:])
β_normalized = sqrt.(β_vals)


beta_network = LSTMToDense(num_indicators, 1, beta_output_size)
indicator_network = LSTMToDense(num_indicators+1, hidden_dims, num_indicators)
ps_beta, st_beta = Lux.setup(rng, beta_network)
ps_ind, st_ind = Lux.setup(rng, indicator_network)


splits = TimeSeriesSplit(...)
### Stage 1: standard supervised learning

function compute_loss_LSTM(model, h, m, p, st, X, Y)
    (Y_pred, h_new, m_new), st_new  = model((X, h, m), p, st)
    loss = sum(abs2, Y_pred - Y)
    return loss, (h_new, m_new), st_new
end


function pre_train(model, p, st, X, Y, warmup; maxiters=100)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), p)
    for epoch in 1:maxiters
        (h, m), st = warmup!(model, p, st, warmup)
        for i in 1:size(X,2)
            (l, (h, m), st), back = pullback(θ->compute_loss_LSTM(model, h, m, θ, st, X[:,i,:], Y[:,i,:]), p)
            gs = back((one(l), nothing, nothing))[1]
            opt_state, p = Optimisers.update(opt_state, p, gs)
            if epoch % 10 == 0 && verbose
                println("Epoch [$(epoch)]: Loss = $(l)")
            end
        end
    end
    return p
end

ps_beta = pre_train(beta_network, ps_beta, st_beta, X_beta[:,train_split,:], Y_beta[:,train_split,:], warmup_data[3:end,:,:])
ps_ind = pre_train(indicator_network, ps_ind, st_ind, X_ind[:,train_split,:], Y_ind[:,train_split,:], warmup_data[2:end,:,:])
ps_stage1 = Lux.ComponentArray(beta = ps_beta, ind=ps_ind)



function solve_system(u0, tspan::Tuple{Int, Int}, p, st_beta, st_ind, warmup_data)
    (h_beta, m_beta), st_beta = warmup!(beta_network, p.beta, st_beta, warmup_data[3:end,:,:])
    (h_ind, m_ind), st_ind = warmup!(indicator_network, p.ind, st_ind, warmup_data[2:end,:,:])

    Y_out = zeros(length(u0), tspan[end]-tspan[1]+1, 1)

    Y_out[:,1,:] .= u0
    for t in 1:size(Y_out, 2)-1
        S, I = Y_out[1:2,t]

        (rootβ, h_beta, m_beta), st_beta = beta_network((Y_out[3:end, t,:], h_beta, m_beta), p.beta, st_beta)
        (ΔX, h_ind, m_ind), st_ind = indicator_network((Y_out[2:end, t, :], h_ind, m_ind), p.ind, st_ind)
        ΔS = (rootβ[1]^2)*S*I > S ? -S : -(rootβ[1]^2)*S*I
        ΔI = I + (-ΔS -γ*I) > 1 ? 1-I : -ΔS -γ*I

        Y_out[:,t+1] = Y_out[:,t] + [ΔS; ΔI; ΔX]
    end
    return Y_out, range(tspan[1], step=1, length=size(Y_out,2))
end


function plot_prediction(args...; title="Prediction")
    pred, tsteps = solve_system(args...)
    pl = scatter(tsteps, data[:,tsteps[1]+1:tsteps[end]+1,1]', layout=(length(u0),1), color=:black, label=["Data" nothing nothing nothing nothing nothing],
        ylabel=hcat(["S" "I"], reshape([indicator_names[i] for i in indicators], 1, num_indicators)))
    plot!(pl, tsteps, pred[:,:,1]', color=:red, label=["Prediction" nothing nothing nothing nothing nothing])
    for i in 1:length(u0)-1
        vline!(pl[i], [train_split[end]], color=:black, style=:dash, label=nothing)
    end
    vline!(pl[end], [[train_split[end]]], color=:black, style=:dash, label="End of training data")

    xlabel!(pl[end], "Time (days since $(days[start_idx]))")
    title!(pl[1], title*", LSTM model, $(region)")
    return pl
end

pl_initial_pred = plot_prediction(u0, tspan_test, ps_stage1, st_beta, st_ind, warmup_data; title="Stage 1 prediction")

### Stage 2 training: fit beta first on true training data, then fit the other network to the first network
function fit_beta(X, Y, p, st_beta; λ=0)
    (h, m), st = warmup!(beta_network, p, st_beta, warmup_data[3:end,:,:])
    U = Y[:,1,:]
    loss = 0
    for t in 1:size(Y, 2)-1
        S, I = U[1:2]
        (rootβ, h, m), st = beta_network((X[:,t,:], h, m), p, st_beta)
        ΔS = (rootβ[1]^2)*S*I > S ? -S : -(rootβ[1]^2)*S*I
        ΔI = I + (-ΔS -γ*I) > 1 ? 1-I : -ΔS -γ*I
        U += [ΔS; ΔI]
        loss += sum(abs2, U - Y[:,t,:])
    end
    return loss + λ*sum(abs2, p)/length(p)  
end


function fit_indicators(Y, p_beta_trained, p_ind, st_beta, st_ind; λ=0)
(h_beta, m_beta), st_beta = warmup!(beta_network, p_beta_trained, st_beta, warmup_data[3:end,:,:])
(h_ind, m_ind), st_ind = warmup!(indicator_network, p_ind, st_ind, warmup_data[2:end,:,:])
U = Y[:,1,:]
loss = 0
for t in 1:size(Y, 2)-1
    S, I = U[1:2]
    (rootβ, h_beta, m_beta), st_beta = beta_network((U[3:end,:], h_beta, m_beta), p_beta_trained, st_beta)
    (ΔX, h_ind, m_ind), st_ind = indicator_network((U[2:end, :], h_ind, m_ind), p_ind, st_ind)

    ΔS = (rootβ[1]^2)*S*I > S ? -S : -(rootβ[1]^2)*S*I
    ΔI = I + (-ΔS -γ*I) > 1 ? 1-I : -ΔS -γ*I
    U += [ΔS; ΔI; ΔX]
    loss += sum(abs2, U - Y[:,t,:])
end
return loss + λ*sum(abs2, p_ind)/length(p_ind)
end

stage2_losses=[]
function stage2_cb(p, l, threshold)
push!(stage2_losses, l)
if length(stage2_losses) % 50 == 0 && verbose
    println("Loss after $(length(stage2_losses)) iterations: $(l)")
end
return l < threshold
end

for split in splits
    pred, back = pullback(...)
    gs = back(...)
    opt_state = Optimisers.update(opt_state, gs) 

end

X = data[3:end, train_split,:]
Y = data[1:2, train_split,:]
adtype = Optimization.AutoZygote()

optf_beta = Optimization.OptimizationFunction((p, x) -> fit_beta(X, Y, p, st_beta, λ=λ), adtype)
optprob_beta = Optimization.OptimizationProblem(optf_beta, ps_stage1.beta)
res_beta = solve(optprob_beta, OptimizationFlux.ADAM(0.02), maxiters=1200, callback = (p, l) -> stage2_cb(p, l, 1e-3))

Y = data[:, train_split,:]
optf_indicators = Optimization.OptimizationFunction((p, x) -> fit_indicators(Y, res_beta.minimizer, p, st_beta, st_ind, λ=λ), adtype)
optprob_indicators = Optimization.OptimizationProblem(optf_indicators, ps_stage1.ind)
res_indicators = solve(optprob_indicators, OptimizationFlux.ADAM(0.05), maxiters=7500, callback = (p, l) -> stage2_cb(p, l, 2.0))

ps_stage2 = Lux.ComponentArray(beta = res_beta.minimizer, ind = res_indicators.minimizer)
pred_stage2 = solve_system(u0, tspan_test, ps_stage2, st_beta, st_ind, warmup_data)
pl_stage2_pred = plot_prediction(u0, tspan_test, ps_stage2, st_beta, st_ind, warmup_data, title="Stage 2 prediction")



    ### Stage 3: train both sets of networks simultaneously
function fit_simul(Y, p, st_beta, st_ind, warmup_data; λ=0)
    (h_beta, m_beta), st_beta = warmup!(beta_network, p.beta, st_beta, warmup_data[3:end,:,:])
    (h_ind, m_ind), st_ind = warmup!(indicator_network, p.ind, st_ind, warmup_data[2:end,:,:])
    U = Y[:,1,:]
    Yscale = maximum(Y, dims=2) - minimum(Y, dims=2)
    loss = 0
    for t in 1:size(Y, 2)-1
        S, I = U[1:2]
        (rootβ, h_beta, m_beta), st_beta = beta_network((U[3:end,:], h_beta, m_beta), p.beta, st_beta)
        (ΔX, h_ind, m_ind), st_ind = indicator_network((U[2:end, :], h_ind, m_ind), p.ind, st_ind)

        ΔS = (rootβ[1]^2)*S*I > S ? -S : -(rootβ[1]^2)*S*I
        ΔI = I + (-ΔS -γ*I) > 1 ? 1-I : -ΔS -γ*I
        U += [ΔS; ΔI; ΔX]
        loss += sum(abs2, (U - Y[:,t,:])./Yscale)
    end
    return loss + λ*sum(abs2, p)/length(p)
end




losses = []
function callback(p, l)
    push!(losses, l)
    if length(losses) % 50 == 0 && verbose
        println("Loss after $(length(losses)) iterations: $(l)")
    end
    return l < 0.01
end
Y = data[:, train_split,:]

optf = Optimization.OptimizationFunction((p, x) -> fit_simul(Y, p, st_beta, st_ind, warmup_data), adtype)
optprob = Optimization.OptimizationProblem(optf, ps_stage2)
res_final = solve(optprob, OptimizationFlux.ADAM(0.001), maxiters=6000, callback=callback)

ps_trained = res_final.minimizer
pred_final, tsteps = solve_system(u0, (0, 300), ps_trained, st_beta, st_ind, warmup_data)
pl_final_pred = plot_prediction(u0, tspan_test, ps_trained, st_beta, st_ind, warmup_data, title="Final prediction")
pl_longterm_pred = plot_prediction(u0, (0, 300), ps_trained, st_beta, st_ind, warmup_data , title="Long term prediction")


pl_losses= plot(1:length(losses), losses, xlabel="Iterations", ylabel = "Loss", title="Final stage training losses, LSTM model, $(region)",
    label=nothing)
yaxis!(pl_losses, :log10)

### Save the simulation data
indicator_name = ""
for j=1:length(indicator_idxs)-1
    indicator_name = indicator_name * indicator_names[indicator_idxs[j]] * "-"
end
indicator_name = indicator_name * indicator_names[indicator_idxs[end]]

params = (hdims = hidden_dims,seqlen=sequence_length)
param_name = savename(params, digits=0)

fname = "$(indicator_name)_$(param_name)"

# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
model_iteration = 1
while isdir(datadir("sims", model_name, region, "$(fname)_v$(model_iteration)"))
    model_iteration += 1
end
fname = fname * "_v$(model_iteration)"
mkdir(datadir("sims", model_name, region, fname))


savefig(pl_initial_pred, datadir("sims", model_name, region, fname, "initial_prediction.png"))
savefig(pl_stage2_pred, datadir("sims", model_name, region, fname, "stage2_prediction.png"))
savefig(pl_final_pred, datadir("sims", model_name, region, fname, "final_test_prediction.png"))
savefig(pl_longterm_pred, datadir("sims", model_name, region, fname, "long_term_prediction.png"))
savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

save(datadir("sims", model_name, region, fname, "results.jld2"),
    "p", ps_trained, "st_beta", st_beta, "st_ind", st_ind,
    "losses", losses, "prediction", pred_final,
    "warmup_data", warmup_data, "data", data, "train_split", train_split, "test_split", test_split, "days", days)


### Extended analysis

new_warmup_data = data[:, train_split[end]+1:train_split[end]+14]
new_u0 = data[:, train_split[end]+15]
new_pred = plot_prediction(new_u0, (train_split[end]+15, train_split[end]+15+100), ps_trained, st_beta, st_ind, new_warmup_data; title="Second wave prediction")
savefig(new_pred, datadir("sims", model_name, region, fname, "novel_prediction.png"))




run_model()
