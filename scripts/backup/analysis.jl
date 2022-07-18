cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")
using Plots
using JLD2, FileIO
using Statistics
using DiffEqFlux, Flux
using Dates


function analyze(region)
	results = load(datadir("sims", "udde", "results_$(region).jld2"))

	hist_data = results["hist_data"]
	training_data = results["training_data"]
	test_data = results["test_data"]
	data = [hist_data training_data test_data]
	days = results["days"]

	nn_dS = FastChain(FastDense(1, 4, gelu), FastDense(4, 4, gelu), FastDense(4, 4, gelu), FastDense(4, 1))
	nn_dM = FastChain(FastDense(3, 4, gelu), FastDense(4, 4, gelu), FastDense(4, 1))

	p_dM = results["p_dM"]
	p_dS = results["p_dS"]
	scale = results["scale"]
	yscale = scale*size(training_data,2)

	pred = results["prediction"]

	total_loss = sum(abs2, (training_data - pred[:,1:size(training_data,2)])./yscale)/size(training_data,2)
	test_loss = sum(abs2, (test_data - pred[:,size(training_data,2)+1:end])./yscale)/size(training_data,2)

	## Prediction
	pl = plot(range(0.0, step=1.0, length=size(pred,2)), pred', layout=(3,1), label=["True Data" "" ""])
	plot!(pl, range(0.0, step=1.0, length=size(data,2)), data', label=["UDE Prediction" "" ""])
	xlabel!(pl[3], "Time (days)")
	ylabel!(pl[1], "S")
	ylabel!(pl[2], "I")
	ylabel!(pl[3], "M")
	# savefig(pl, "US-NY-dS=4by4-dM=3by5-g=0.25-c=5_1\\final_prediction.png")


	## Mobility dose-response
	M_test = -100:0.1:100
	β_response = [-nn_dS(M, p_dS)[1]*scale[1] for M in M_test]
	pl = plot(M_test, β_response, xlabel="Mobility", ylabel="β", title="Mobility Response: $(region)",
		label="β(M)")
	vline!(pl, [minimum(data[3,:]), maximum(data[3,:])], style=:dash, color=:black, label="Observed Range")
	savefig(pl, plotsdir("udde", region, "mobility_response_$(region)"))

	## beta timeseries
	M_true = [hist_data[3,:]; training_data[3,:]; test_data[3,:]]
	times = range(-14.0, step=1.0, length=length(M_true))

	β_learned = [-nn_dS(M, p_dS)[1]*scale[1] for M in M_true[1:end-14]]

	pl1 = plot(times, M_true, ylabel="M(t)", label=nothing)
	pl2 = plot(times[15:end], β_learned, ylabel="β(t)", label=nothing)
	vline!(pl1, [0], style=:dash, color=:black, label="Start training data")
	xlims!(pl2, xlims(pl1)[1], xlims(pl1)[2])
	xlabel!(pl2, "Time (days)")

	lt = @layout([a; b])
	pl = plot(pl1, pl2, layout=lt)

	savefig(pl, plotsdir("udde", region, "beta_M_timeseries_$(region)"))



	## Mobility response timeseries
	dM = [nn_dM(data[:,i], p_dM)[1] for i = 1:size(data,2)-10]*scale[3]
	times = range(-10.0, step=1.0, length=length(dM))

	pl1 = plot(times, dM, ylabel="f(u(t))", xlabel="Time (days)", label=nothing)
	pl2 = plot(-10:1.0:size(data,2)-11, data[2,:], ylabel="I(t)", label=nothing)
	lt = @layout([a; b])
	pl = plot(pl2, pl1, layout=lt)

	savefig(pl, plotsdir("udde", region, "f_timeseries_$(region)"))


	## Mobility dose-responses
	imax, ind = findmax(data[2,:])
	smax = data[1,ind]

	M_test = -100:0.1:100
	f_response_1 = [nn_dM([smax; imax; M], p_dM)[1]*scale[3] for M in M_test]
	pl1 = plot(M_test, f_response_1, xlabel="Mobility", ylabel="f(u, θ)", label="Peak infection")

	f_response_2 = [nn_dM([1.0; 0.0; M], p_dM)[1]*scale[3] for M in M_test]
	plot!(pl1, M_test, f_response_2, label="No infection")
	vline!(pl1, [minimum(data[3,:]), maximum(data[3,:])], style=:dash, color=:black, label="Observed Range")

	savefig(pl, plotsdir("udde", region, "f_response_to_M_$(region)"))


	I_test = 0.0:0.001:0.10
	f_response_1 = [nn_dM([1-Ival; Ival; 0.0], p_dM)[1]*scale[3] for Ival in I_test]
	pl1 = plot(I_test, f_response_1, xlabel="I", ylabel="f(u, θ)", label="Average mobility")

	f_response_2 = [nn_dM([1-Ival; Ival; -75], p_dM)[1]*scale[3] for Ival in I_test]
	plot!(pl1, I_test, f_response_2, label="Minimum mobility")

	savefig(pl, plotsdir("udde", region, "f_response_to_I_$(region)"))
	nothing
end

analyze(ARGS[1])
