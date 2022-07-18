cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

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
    hidden_dims = 3
	indicators=[4, 5]
	return region, hidden_dims, indicators
end

region, hidden_dims, indicators=default_setup()
if ARGS != []
	region = ARGS[1]
	hidden_dims = parse(Int, ARGS[2])
	indicators = parse.(Int, ARGS[3:end])
end
indicator_idxs = reshape(indicators, 1, length(indicators))
num_indicators = length(indicator_idxs)
beta_feature_size = num_indicators # M informs β
beta_output_size = 1


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
const warmup_length = 14
const test_length = 28
const firstwave_end = 170
λ = 0


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

function run_model()

    model_name = "LSTM_allvars_splittrain"
    println("Starting run: $(region)")
    verbose = (isempty(ARGS)) ? true : false


    ## Load in data for the region and partition into training, testing
    dataset = load(datadir("exp_pro", "SIMX_7dayavg_2020_$(region).jld2"))
    days = dataset["days"]
    data = dataset["data"][hcat([1 2], indicator_idxs),:][1,:,:]
    data = reshape(data, size(data)[1], size(data)[2], 1)


    function warmup!(LD::LSTMToDense, p, st, warmup_data)
        (h, m), st = LD(warmup_data[:,1,:], p, st)
        for j=2:size(warmup_data,2)
            (out, h, m), st = LD((warmup_data[:,j,:], h, m), p, st)
        end
        return (h, m), st
    end


    changes = data[:, 2:end,:] .- data[:, 1:end-1,:]
    β_vals = @. -changes[1,:,:]/(data[1,1:end-1,:]*data[2, 1:end-1,:])
    β_normalized = reshape(sqrt.(β_vals), 1, length(β_vals), 1)



    beta_network = LSTMToDense(num_indicators, hidden_dims, beta_output_size)
    indicator_network = LSTMToDense(num_indicators+1, hidden_dims, num_indicators)
    ps_beta, st_beta = Lux.setup(rng, beta_network)
    ps_ind, st_ind = Lux.setup(rng, indicator_network)


    firstwave_data = data[:,1:firstwave_end,:]
    secondwave_data = data[:,firstwave_end+1:end,:]

    ### Stage 1: standard supervised learning

    # for split in splits
    split = (1:150, 151:170)
    train_split, CV_split = split
    warmup_data = firstwave_data[:, train_split[1:warmup_length],:]
    X_beta = data[3:end, train_split[warmup_length+1:end],:]
    Y_beta = β_normalized[:, train_split[warmup_length+1:end],:]
    X_ind = data[2:end, train_split[warmup_length+1:end],:]
    Y_ind = changes[3:end, train_split[warmup_length+1:end],:]


    function compute_loss_LSTM(model, h, m, p, st, X, Y)
        (Y_pred, h_new, m_new), st_new  = model((X, h, m), p, st)
        loss = sum(abs2, Y_pred - Y)
        return loss, (h_new, m_new), st_new
    end


    function stage1_train(model, p, st, X, Y, warmup; maxiters=100)
        opt_state = Optimisers.setup(Optimisers.Adam(lr), p)
        overall_losses = []
        for epoch in 1:maxiters
            losses = []
            (h, m), st = warmup!(model, p, st, warmup)
            for i in 1:size(X,2)
                (l, (h, m), st), back = pullback(θ->compute_loss_LSTM(model, h, m, θ, st, X[:,i,:], Y[:,i,:]), p)
                push!(losses, l)
                gs = back((one(l), nothing, nothing))[1]
                opt_state, p = Optimisers.update(opt_state, p, gs)
                
            end
            epoch_loss = mean(losses)
            if epoch % 10 == 0 && verbose
                println("Epoch [$(epoch)]: Loss = $epoch_loss")
            end
            push!(overall_losses, epoch_loss)
        end
        return p, overall_losses
    end

    ps_beta = stage1_train(beta_network, ps_beta, st_beta, X_beta, Y_beta, warmup_data[3:end,:,:])
    ps_ind = stage1_train(indicator_network, ps_ind, st_ind, X_ind, Y_ind, warmup_data[2:end,:,:])
    ps_stage1 = Lux.ComponentArray(beta = ps_beta, ind=ps_ind)

    # Compute the cross-validation loss at this point 

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

    t0 = train_split[warmup_length+1]
    u0 = firstwave_data[:,t0]
    pl_initial_pred = plot_prediction(u0, (t0, size(data,2)-1), ps_stage1, st_beta, st_ind, warmup_data; title="Stage 1 prediction")


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


    function stage2_train(split, p, st_beta, st_ind; maxiters=100)
        opt_state = Optimisers.setup(Optimisers.Adam(0.005), p)
        overall_losses = []
        for epoch in 1:maxiters
            losses = []
            for j in 1:length(split)-warmup_length-test_length 
                warmup_data = firstwave_data[:, train_split[j:j+warmup_length-1], :]
                Y = firstwave_data[:, train_split[j+warmup_length:j+warmup_length+test_length],:]
                loss, back = pullback(θ->fit_simul(Y, θ, st_beta, st_ind, warmup_data), p)
                push!(losses, loss)
                
                gs = back(one(loss))[1]
                opt_state, p = Optimisers.update(opt_state, p, gs)
            end
            epoch_loss = mean(losses)
            if verbose && (epoch % 10) == 0
                display("Loss after $epoch epochs: $epoch_loss")
            end
            push!(overall_losses, epoch_loss)
        end
        return p, overall_losses
    end

    ps_stage2, losses_stage2 = stage2_train(train_split, ps_stage1, st_beta, st_ind, maxiters=2000)



    warmup_test = secondwave_data[:,1:warmup_length,:]
    u0_test = secondwave_data[:, warmup_length+1]
    tspan_test = (firstwave_end, size(data,2)-1)
    pred_final, tsteps = solve_system(u0_test, tspan_test, ps_stage2, st_beta, st_ind, warmup_test)
    pl_final_pred = plot_prediction(u0, tspan_test, ps_stage2, st_beta, st_ind, warmup_data, title="Final prediction")


    pl_losses= plot(1:length(losses_stage2), losses_stage2, xlabel="Iterations", ylabel = "Loss", title="Final stage training losses, LSTM model, $(region)",
        label=nothing)
    yaxis!(pl_losses, :log10)

    ### Save the simulation data
    indicator_name = ""
    for j=1:length(indicator_idxs)-1
        indicator_name = indicator_name * indicator_names[indicator_idxs[j]] * "-"
    end
    indicator_name = indicator_name * indicator_names[indicator_idxs[end]]

    params = (hdims = hidden_dims,)
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
    savefig(pl_final_pred, datadir("sims", model_name, region, fname, "final_test_prediction.png"))
    savefig(pl_losses, datadir("sims", model_name, region, fname, "losses.png"))

    save(datadir("sims", model_name, region, fname, "results.jld2"),
        "p", ps_stage2, "st_beta", st_beta, "st_ind", st_ind,
        "losses", stage2_losses, "prediction", pred_final,
        "firstwave_data", firstwave_data, "secondwave_data", secondwave_data, "days", days)

end



run_model()
