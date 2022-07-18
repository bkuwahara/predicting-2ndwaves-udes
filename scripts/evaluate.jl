cd(@__DIR__)
cd("..")
using DrWatson
@quickactivate("S2022_Project")
using Plots
using JLD2, FileIO
using Statistics


#= Plots the prediction of a trained neural network over the full first wave timespan
exp_name: Name of the experiment run
sim: Number of the experiment run
showplot: If true, display the plot.
saveplot: If true, save the figure
=#
function plot_prediction(model_name, exp_name, iteration; showplot = false, saveplot = true)
	if !isfile(datadir("sims", model_name, "$(exp_name)_$(iteration).jld2"))
		println("Experiment not found.")
		return nothing
	end
	results = load(datadir("sims", model_name, "$(exp_name)_$(iteration).jld2"))


	params = results["params"]
	region = params[:region]
	frac_training = params[:frac_training]
	training_data = results["training_data"]
	test_data = results["test_data"]
	data = [training_data test_data]

	times = range(0.0, length = size(data,2), step=1.0)
	pl = plot(times, data', layout=(3,1),
		title=["" "" ""],
		label = ["" "" "True Data"], ylabel=["S" "I" "Mobility"])
	pred = results["prediction"]
	plot!(pl, range(0.0, length= size(pred, 2), step=1.0), pred', layout=(3,1),
		label = ["" "" "UDE Prediction"])
	training_stop = frac_training*times[end]
	vline!(pl, [training_stop training_stop training_stop], color = :black,
		label=["" "" "End of Training Data"])
	if showplot
		display(pl)
	end
	if saveplot
		if !isdir(plotsdir(model_name))
			mkdir(plotsdir(model_name))
		end
		savefig(pl, plotsdir(model_name, "$(exp_name)_$(iteration).png"))
	end
	nothing
end


#=
Plots all runs of an experiment model_name
=#
function plot_all(model_name; redo=false)
	if !isdir(datadir("sims", model_name))
		println("Model not found.")
		return nothing
	end

	for fname in readdir(datadir("sims", model_name))
		output_fname = rsplit(fname, ".", limit=2)[1]*".png"

		if !isfile(plotsdir(model_name, output_fname)) || redo
			results = load(datadir("sims", model_name, fname))
			params = results["params"]
			region = params[:region]
			frac_training = params[:frac_training]
			training_data = results["training_data"]
			test_data = results["test_data"]
			data = [training_data test_data]

			times = range(0.0, length = size(data,2), step=1.0)
			pl = plot(times, data', layout=(3,1),
				title=["" "" ""],
				label = ["" "" "True Data"], ylabel=["S" "I" "Mobility"])
			pred = results["prediction"]
			plot!(pl, range(0.0, length= size(pred, 2), step=1.0), pred', layout=(3,1),
				label = ["" "" "UDE Prediction"])
			training_stop = frac_training*times[end]
			vline!(pl, [training_stop training_stop training_stop], color = :black,
				label=["" "" "End of Training Data"])
			ylims!(pl[1], (0.9, 1.0))
			ylims!(pl[2], (0.0, 0.012))
			ylims!(pl[3], (-100.0, 30))


			if !isdir(plotsdir(model_name))
				mkdir(plotsdir(model_name))
			end
			savefig(pl, plotsdir(model_name, output_fname))
			println(fname)
		end

	end
end



#= Plot the losses over training iterations for an experiment run
exp_name: Name of the experiment run
sim: Number of the experiment run
showplot: If true, display the plot.
saveplot: If true, save the figure
=#
function plot_losses(model_name, exp_name, iteration; showplot = false, saveplot = true)
	results = load(datadir("sims", model_name, "$(exp_name)_$(iteration).jld2"))
	params = results["params"]
	region = params[:region]
	losses = results["losses"]
	pl = plot(1:length(losses), losses, title="Training Losses: $(exp_name)",
		xlabel = "Iteration", ylabel = "Loss")
	yaxis!(pl, :log10)
	if showplot
		display(pl)
	end
	if saveplot
		savefig(pl, plotsdir("exp_name", "$(exp_name)_$(sim)_$(region)_losses"))
	end
	nothing
end




function plot_mean_prediction(exp_name, sims; saveplot=true, showplot=false)
	preds = nothing
	region = nothing
	training_data = nothing
	CV_data = nothing
	test_data = nothing
	frac_training = nothing
	all_data = nothing
	failed_sims = 0
	pred_list = []

	for (i, sim) in enumerate(sims)
		results = load(datadir("sims", exp_name, "$(exp_name)_$(sim).jld2"))
		if i == 1
			region = results["region"]
			training_data = results["training_data"]
			CV_data = results["CV_data"]
			test_data = results["test_data"]
			all_data = [training_data CV_data test_data]

			preds = zeros(length(sims) - failed_sims, size(all_data,1), size(all_data, 2))
			frac_training = results["frac_training"]
		end
		if results["retcode"] == 1
			failed_sims += 1
			continue
		end
		pred = results["prediction"];
		push!(pred_list, pred)
	end


	println("$failed_sims out of $(length(sims)) failed.")
	preds = zeros(length(pred_list), size(pred_list[1],1), size(pred_list[1],2))
	for (i, pred) in enumerate(pred_list)
		preds[i,:,:] = pred
	end
	mean_pred = mean(preds, dims=1)[1,:,:]
	σs = std(preds, dims=1)[1,:,:]
	CI_upper = mean_pred + 1.96*σs;
	CI_lower = mean_pred - 1.96*σs;

	times = range(0.0, length=size(mean_pred, 2), step=1.0)
	i_stop = Int(round(frac_training*length(times)))
	pl = plot(times, [training_data CV_data test_data]', layout=(3,1),
		title=["Prediction and Confidence Interval: $(region)" "" ""],
		ylabel = ["S" "I" "Mobility"], label = ["True Data" "" ""])

	plot!(pl, times[1:i_stop], mean_pred[:, 1:i_stop]', color=:red,
		label=["Prediction" "" ""])
	plot!(pl, times[i_stop+1:end], mean_pred[:,i_stop+1:end]', color=:red,
		linestyle=:dash, label=["Extrapolation" "" ""])
	plot!(pl, times, CI_upper', color=:gray, label=["95% Confidence Interval" "" ""])
	plot!(pl, times, CI_lower', color=:gray, label=["" "" ""])
	if saveplot
		savefig(pl, plotsdir(exp_name, "$(exp_name)_mean_pred"))
	end
	if showplot
		display(pl)
	end
end



function plot_losses_ensemble(exp_name, sim, run)
	results = load(datadir("sims", exp_name, "$(exp_name)_$(sim).jld2"))
	losses = results["losses"]
	pl = plot(1:length(losses[run]), losses[run], title=[run "" ""])
	display(pl)
end
##



plot_all("udde")


results = load(datadir("sims", "udde", "UK-dMstruct=3by5-g=0.25-c=5_3.jld2"))
losses = results["losses"]

println(results["p_dS"])


plot_losses("udde", "CA-ON-dS=5by3-dM=5by3-g=0.25-c=5", 2; showplot=true, saveplot=true)



t = 0.0:0.1:10.0
x = sin.(t)
y = cos.(t)
z = exp.(t)

pl = plot(t, [x y z], layout=(3,1))
ylims!(pl[3], (0.0, 5.0))
