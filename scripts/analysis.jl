cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")
using Plots
using JLD2, FileIO
using Statistics
using DiffEqFlux, Flux
using Dates



function analyze(region, iteration)
	results = load(datadir("sims", "udde", "results_$(region)_$(iteration).jld2"))

	hist_data = results["hist_data"]
	training_data = results["training_data"]
	test_data = results["test_data"]
	data = [hist_data training_data test_data]
	days = results["days"]

	## Output file directory
	outfile ="$(region)_$(iteration)"
	if !isdir(plotsdir("udde", outfile))
		mkdir(plotsdir("udde", outfile))
	end


	p_dM = results["p_dM"]
	p_dS = results["p_dS"]
	scale = results["scale"]
	yscale = scale*size(training_data,2)
	τᵣ = results["tau_r"]
	τₘ = results["tau_m"]
	dS_ls = results["dS_ls"]
	dM_ls = results["dM_ls"]

	#
	nn_dS = FastChain(FastDense(1, dS_ls, gelu), FastDense(dS_ls, dS_ls, gelu), FastDense(dS_ls, dS_ls, gelu), FastDense(dS_ls, 1))
	nn_dM = FastChain(FastDense(2, dM_ls, gelu), FastDense(dM_ls, dM_ls, gelu), FastDense(dM_ls, dM_ls, gelu), FastDense(dM_ls, 1))
	# nn_dS = FastChain(FastDense(1, 3, activation), FastDense(3, 3, activation), FastDense(3, 3, activation), FastDense(3, 3, activation), FastDense(3, 1))
	# nn_dM = FastChain(FastDense(2, 3, activation), FastDense(3, 3, activation), FastDense(3, 3, activation), FastDense(3, 3, activation), FastDense(3, 1))

	pred = results["prediction"]

	total_loss = sum(abs2, (training_data - pred[:,1:size(training_data,2)])./yscale)/size(training_data,2)
	test_loss = sum(abs2, (test_data - pred[:,size(training_data,2)+1:end])./yscale)/size(training_data,2)

	## Prediction
	pl = plot(range(0.0, step=1.0, length=size(pred,2)), pred', layout=(3,1), label=["UDE Prediction" "" ""])
	plot!(pl, range(0.0-max(τᵣ, τₘ), step=1.0, length=size(data,2)), data', label=["True Data" "" ""])
	xlabel!(pl[3], "Time (days)")
	ylabel!(pl[1], "S")
	ylabel!(pl[2], "I")
	ylabel!(pl[3], "M")
	savefig(pl, plotsdir("udde", outfile, "final_prediction_$(region)"))


	## Losses
	losses = results["losses"]
	pl = plot(range(1.0, step=100.0, length=length(losses)), losses, title="Training Losses: $(region)",
		xlabel = "Iteration", ylabel = "Loss")
	yaxis!(pl, :log10)
	savefig(pl, plotsdir("udde", outfile, "losses_$(region).png"))


	## Mobility dose-response
	M_test = -100:0.1:100
	β_response = [-nn_dS(M, p_dS)[1]*scale[1] for M in M_test]
	pl = plot(M_test, β_response, xlabel="Mobility", ylabel="β", title="Mobility Response: $(region)",
		label="β(M)")
	vline!(pl, [minimum(data[3,:]), maximum(data[3,:])], style=:dash, color=:black, label="Observed Range")
	savefig(pl, plotsdir("udde", outfile, "mobility_response_$(region)"))

	## beta timeseries
	M_true = data[3,:]
	times = range(-Int(round(τₘ)), step=1.0, length=length(M_true))

	β_learned = [-nn_dS(M, p_dS)[1]*scale[1] for M in M_true[1:end-Int(round(τₘ))]]

	pl1 = plot(times, M_true, ylabel="M(t)", label=nothing)
	pl2 = plot(times[Int(round(τₘ))+1:end], β_learned, ylabel="β(t)", label=nothing)
	vline!(pl1, [0], style=:dash, color=:black, label="Start training data")
	xlims!(pl2, xlims(pl1)[1], xlims(pl1)[2])
	xlabel!(pl2, "Time (days)")

	lt = @layout([a; b])
	pl = plot(pl1, pl2, layout=lt)

	savefig(pl, plotsdir("udde", outfile, "beta_M_timeseries_$(region)"))



	## Mobility response timeseries
	dM = [nn_dM(data[2:3,i], p_dM)[1] for i = 1:size(data,2)-Int(round(τᵣ))]*scale[3]
	times = range(-Int(round(τᵣ)), step=1.0, length=size(data,2))


	pl1 = plot(times[Int(round(τᵣ))+1:end], dM, ylabel="f(I(t), M(t))", xlabel="Time (days)", label=nothing)
	pl2 = plot(times, data[2,:], ylabel="I(t)", label=nothing)
	pl3 = plot(times, M_true, ylabel="M(t)", label=nothing)
	xlims!(pl1, xlims(pl2)[1], xlims(pl2)[2])
	lt = @layout([a; b; c])
	pl = plot(pl2,pl3, pl1, layout=lt)

	savefig(pl, plotsdir("udde", outfile, "f_timeseries_$(region)"))


	## Mobility dose-responses
	imax, ind = findmax(data[2,:])
	M_test = -100:0.1:100
	f_response_1 = [nn_dM([imax; M], p_dM)[1]*scale[3] for M in M_test]
	pl1 = plot(M_test, f_response_1, xlabel="Mobility", ylabel="f(u, θ)", label="Peak infection")

	f_response_2 = [nn_dM([0.0; M], p_dM)[1]*scale[3] for M in M_test]
	plot!(pl1, M_test, f_response_2, label="No infection")
	vline!(pl1, [minimum(data[3,:]), maximum(data[3,:])], style=:dash, color=:black, label="Observed Range")

	savefig(pl1, plotsdir("udde", outfile, "f_response_to_M_$(region)"))


	I_test = 0.0:0.001:0.10
	f_response_1 = [nn_dM([Ival; 0.0], p_dM)[1]*scale[3] for Ival in I_test]
	pl1 = plot(I_test, f_response_1, xlabel="I", ylabel="f(u, θ)", label="Average mobility")

	M_min = minimum(data[3,:])
	f_response_2 = [nn_dM([Ival; M_min], p_dM)[1]*scale[3] for Ival in I_test]
	plot!(pl1, I_test, f_response_2, label="Minimum mobility")

	savefig(pl1, plotsdir("udde", outfile, "f_response_to_I_$(region)"))

	## Mobility response heatmap
	M_vals = range(-100, stop = 100, length=200)
	I_vals = range(0.0, stop=0.005, length=200)
	Z_vals = [nn_dM([I_val/scale[2], M_val], p_dM)[1] for I_val in I_vals, M_val in M_vals].*scale[3]
	pl = heatmap(I_vals, M_vals, Z_vals')
	xlabel!(pl, "I")
	ylabel!(pl, "M")
	savefig(pl, plotsdir("udde", outfile, "f_heatmap_$(region)"))


	return nothing
end

function analyze(region)
	files = readdir(datadir("sims", "udde"))
	# Only do the most recent one
	iteration = parse(Int64, split(files[end], ".")[1][end])
	analyze(region, iteration)
	return nothing
end

analyze(ARGS[1])
