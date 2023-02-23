cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using LinearAlgebra
using JLD2, FileIO, CSV, DataFrames
using Dates
using Plots, StatsPlots
using Statistics
using Distributions
using Lux, Zygote, OptimizationFlux
using Random; rng = Random.default_rng()

const sample_period=7
const train_length = 160
const mobility_min = -1.0
const mobility_baseline = 0.0
const mobility_max = 2.0
const num_indicators = 1
const hidden_dims = 3
const ϵ=0.001
const recovery_rate = 0.25
const d0 = Date(2020, 2, 18)

indicator_names = Dict(
	3 => "rr",
	4 => "wk",
	5 => "pk",
	6 => "si"
)

# Dictionary mapping abbreviated region names to full names
region_code = Dict(
	"CA" => "Canada",
	"US" => "United States",
	"UK" => "United Kingdom",
	"NL" => "Netherlands",
	"AT" => "Austria",
	"AU" => "Australia",
	"DE" => "Germany",
	"BE" => "Belgium",
	"IT" => "Italy",
	"ON" => "Ontario",
	"BC" => "British Columbia",
	"QC" => "Quebec",
	"CA" => "California",
	"PA" => "Pennsylvania",
	"TX" => "Texas",
	"NY" => "New York",
	"FL" => "Florida"
)

loss_labels = [
	"accuracy"; 
	"beta_mon"; 
	"beta_pos"; 
	"f_ub";
	"f_lb";
	"f_basetend";
	"basetend_mon_R";
	"f_mon_I";
	"f_mon_deltaI";
	 ]


all_regions = [
"AT",
"NL",
"BE",
"DE",
"UK",
"IT",
"US-NY",
"US-CA",
"US-PA",
"US-TX",
"CA-ON",
"CA-QC",
"CA-BC"
	]



network1 = Lux.Chain(
	Lux.Dense(num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>1))
network2 = Lux.Chain(
	Lux.Dense(3+num_indicators=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu), Lux.Dense(hidden_dims=>num_indicators))

p1_temp, st1 = Lux.setup(rng, network1)
p2_temp, st2 = Lux.setup(rng, network2)

function get_inputs(n, yscale)
	I_ = rand(rng, Uniform(0.0, 1.0), 1, n)
	ΔI = rand(rng, 1, n) .+ (I_ .- 1)
	R = rand(rng, 1, n) .* (1 .- I_)
	M = rand(rng, Uniform(mobility_min, 2*mobility_baseline-mobility_min), 1, n)

	return I_/yscale[2], ΔI/yscale[2], M/yscale[3], R/yscale[1]
end




function l_layer1_avg(p, Ms)
	beta_i = network1(Ms, p, st1)[1]
	beta_j = network1(Ms .+ ϵ, p, st1)[1]

	l1 = sum(relu.( -beta_i))
	l2 = sum(relu.( beta_i .- beta_j))

	return [l1, l2]./length(Ms)
end

function l_layer2_avg(p, Is, ΔIs, Ms, Rs)
	n_pts = size(Is,2)
	I_baseline = zeros(1, n_pts)
	M_max = 2*(mobility_baseline - mobility_min) .* ones(1,n_pts)
	M_min = mobility_min .* ones(1, n_pts)


	# Must not increase when M at 2*(mobility_baseline - mobility_min)
	dM_max = network2([M_max; Is; ΔIs; Rs], p, st2)[1]
	dM_min = network2([M_min; Is; ΔIs; Rs], p, st2)[1]

	l3 = sum(relu.(dM_max))
	l4 = sum(relu.(-dM_min))

	# Tending towards M=mobility_baseline when I == ΔI == 0
	dM1_baseline = network2([Ms; I_baseline; I_baseline; Rs], p, st2)[1]
	dM3_baseline = network2([Ms; I_baseline; I_baseline; Rs .+ ϵ], p, st2)[1]
	l5 = sum(relu, relu.(dM1_baseline .* (Ms .- mobility_baseline)))

	# Stabilizing effect is stronger at higher R
	l6 = sum(relu.(abs.(dM1_baseline) .- abs.(dM3_baseline)))

	## Monotonicity terms
	dM_initial = network2([Ms; Is; ΔIs; Rs], p, st2)[1]

	# f monotonically decreasing in I
	dM_deltaI = network2([Ms; (Is .+ ϵ); ΔIs; Rs], p, st2)[1]
	l7 = sum(relu.(dM_deltaI .- dM_initial))
	
	# f monotonically decreasing in ΔI
	dM2_delta_deltaI = network2([Ms; Is; ΔIs .+ ϵ; Rs], p, st2)[1]
	l8 = sum(relu.(dM2_delta_deltaI .- dM_initial))
	return [l3, l4, l5, l6, l7, l8]./length(Ms)
end








function EnsembleSummary(sim_name; titles=["a" "b" "c" "a"])
	root = datadir("sims", "udde", sim_name)
	filenames = load(root * "\\converged_sims.jld2")["sims"]
	f = load(joinpath(root, filenames[1])* "/results.jld2")
	pred = f["prediction"]
	betas = f["betas"]
	hist_data = f["hist_data"]
	train_data = f["train_data"]
	test_data = f["test_data"]

	for fn in filenames[2:end]
		f = load(joinpath(root, fn)* "/results.jld2")
		new_pred = f["prediction"]
		new_betas = f["betas"]
		if size(new_pred, 2) != size(pred, 2)
			new_pred =  hcat(new_pred, NaN .*zeros(size(new_pred,1), size(pred, 2)-size(new_pred, 2)))
		end
		pred = hvncat(3, pred, new_pred)
		betas = hvncat(3, betas, new_betas)
	end

	mean_pred = mean(pred, dims=3)
	med_pred = median(pred, dims=3)[:,1:600,1]
	qu_pred = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.75) for i in axes(med_pred,1), j in axes(med_pred,2)]
	ql_pred = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.25) for i in axes(med_pred,1), j in axes(med_pred,2)]

	med_betas = median(betas, dims=3)[:,:,1]
	qu_betas = [quantile(betas[i,j,:], 0.75) for i in axes(betas,1), j in axes(betas,2)]
	ql_betas = [quantile(betas[i,j,:], 0.25) for i in axes(betas,1), j in axes(betas,2)]


	region = split(sim_name, "_")[end]
	indicators=[3]

	train_tsteps = range(0, step=sample_period, length=size(hist_data, 2) + size(train_data,2))
	test_tsteps = range(train_tsteps[end] + sample_period, step = sample_period, length=size(test_data, 2))
	pl_pred = scatter(train_tsteps, [hist_data train_data]', layout=(size(train_data,1), 1), color=:green2, label=["Training Data" nothing nothing nothing],
		ylabel=["S" "I" "M"], title=titles[:,1:3], xtickfontsize=12, ytickfontsize=12, size=(800,600), 
		markershape=:diamond, markersize=5)
	scatter!(pl_pred, test_tsteps, test_data', color=:black, label=["Unseen Data" nothing nothing nothing])
	pred_tsteps = range(train_tsteps[size(hist_data,2)+1], step=1.0, length=size(med_pred,2))
	plot!(pl_pred, pred_tsteps, med_pred', color=:red, ribbon = ((med_pred-ql_pred)', (qu_pred-med_pred)'), label=["Prediction" nothing nothing])
	xlabel!(pl_pred[end], "Time", xguidefontsize=12)



	M_test = range(mobility_min, step=0.1, stop=2*(mobility_baseline - mobility_min))
	pl_betas = plot(M_test, med_betas', ribbon=((med_betas-ql_betas)', (qu_betas-med_betas)'), 
		label=nothing, title=titles[end], xtickfontsize=12, ytickfontsize=12)
	xlabel!(pl_betas, "M", xguidefontsize = 12)
	ylabel!(pl_betas, "β", yguidefontsize = 12)
	vline!(pl_betas, [minimum([hist_data train_data test_data][3,:]) maximum([hist_data train_data test_data][3,:])], color=:black, label=["Observed range" nothing], 
				style=:dash)


	savefig(pl_pred, datadir("sims", "udde", sim_name, "ensemble_pred.png"))
	savefig(pl_betas, datadir("sims", "udde", sim_name, "ensemble_beta_response.png"))

	return pl_pred, pl_betas	

end



function analyze(sim_name, loss_idxs...)
	root = datadir("sims", "udde", sim_name)
	filenames = filter(f -> isdir(joinpath(root, f)), readdir(root))
	for fname in filenames
		results = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))
		losses = results["losses"]
		pred = results["prediction"]
		train_data = results["train_data"]
		test_data = results["test_data"]
		β = results["betas"]


		mobility_baseline = 0.0
		mobility_min = -1.0

		all_data = [train_data test_data]
		data_tsteps = range(0.0, step=sample_period, length=size(all_data,2))

		pl_pred_lt = scatter(data_tsteps, all_data', label=["True data" nothing nothing],
			color=:black, layout=(size(all_data,1), 1))
		plot!(pl_pred_lt, range(0.0, step=1.0, length=size(pred,2)), pred', label=["Prediction" nothing nothing],
			color=:red, layout=(size(all_data,1), 1))
		vline!(pl_pred_lt, [0.0 train_length], color=:black, style=:dash,
		label=["Training" "" "" "" ""])
		for i = 1:size(all_data,1)
			vline!(pl_pred_lt[i], [0.0 train_length], color=:black, style=:dash,
			label=["" "" "" "" "" "" ""])
		end


		# beta dose-response curve
		pl_beta_response = plot(range(mobility_min, step=0.1, stop=2*(mobility_baseline - mobility_min)), β', xlabel="M", ylabel="β", 
			label=nothing)
		vline!(pl_beta_response, [minimum(train_data[3,:]) maximum(train_data[3,:])], color=:black, label=["Training range" nothing], 
			style=:dash)

		# Net loss plot
		l_net = losses[1:1,:] + sum(10 .* losses[2:end,:], dims=1)
		pl_losses = plot(l_net')
		yaxis!(pl_losses, :log10)

		for idx in loss_idxs
			pl_temp = plot(losses[idx,:])
			# yaxis!(pl_temp, :log10)
			savefig(pl_temp, datadir("sims", "udde", sim_name, fname, "loss_$idx.png"))
		end


		# Check for infeasible solution
		lb = -1 .*ones(size(pred,2))
		ub = 2 .*ones(size(pred,2))
		mask = (pred[3,:] .< lb) .|| (pred[3,:] .> ub)
		stable = true
		if sum(mask) >= 1
			stable = false
		end
		results["stable"] = stable

		save(datadir("sims", "udde", sim_name, fname, "results.jld2"), results)

		savefig(pl_losses, datadir("sims", "udde", sim_name, fname, "net_losses.png"))
		savefig(pl_pred_lt, datadir("sims", "udde", sim_name, fname, "long_term_prediction.png"))
		savefig(pl_beta_response, datadir("sims", "udde", sim_name, fname, "beta_response.png"))

	end
	nothing
end




function loss_analysis(sim_name)
	root = datadir("sims", "udde", sim_name)
	filenames = filter(f -> isdir(joinpath(root, f)), readdir(root))

	all_losses = zeros(9, length(filenames))
	loss_weight = 0
	convergence_iters = zeros(length(filenames))
	stable_sims = [true for i in 1:length(filenames)]

	for (i, fname) in enumerate(filenames)
		
		results = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))
		scale = results["scale"]
		losses = results["losses"]
		train_data = results["train_data"]
		p = results["p"]
		stable = results["stable"]

		if i == 1
			loss_weight = results["loss_weights"]
		end

		acc_loss, iter = findmin(losses[1,:])
		convergence_iters[i] = iter

		Is, ΔIs, Ms, Rs = get_inputs(1000000, size(train_data, 2)*scale)
		mean_layer1_loss = l_layer1_avg(p.layer1, Ms)
		mean_layer2_loss = l_layer2_avg(p.layer2, Is, ΔIs, Ms, Rs)

		mean_loss = [acc_loss; mean_layer1_loss; mean_layer2_loss]
		all_losses[:,i] = mean_loss

		stable_sims[i] = stable
	end

	mean_losses = reshape(mean(all_losses, dims=2), size(all_losses,1))
	summary = Dict{String, Any}(zip(["avg_" * label for label in loss_labels], mean_losses))


	net_losses = all_losses[1,:] .+ transpose(loss_weight .* sum(all_losses[2:end,:], dims=1))
	best_loss, best_ind = findmin(net_losses)
	best_ver = parse(Int64, rsplit(filenames[best_ind[1]], "v")[end])
	summary["best_ver"] = best_ver
	summary["med_conv_iter"] = median(convergence_iters)
	summary["mean_conv_iter"] = mean(convergence_iters)

	converged_idxs = reshape(reduce(*, all_losses .<= 0.05, dims=1), length(filenames))
	converged_sims = filenames[stable_sims]

	summary["frac_stable"] = length(converged_sims)/length(filenames)

	df = DataFrame(summary)
	CSV.write(root*"\\loss_summary.csv", df)

	

	save(root * "\\converged_sims.jld2", "sims", converged_sims)

	nothing
end



function wave_count(sim_name)
	root = datadir("sims", "udde", sim_name)
	filenames = load(root * "\\converged_sims.jld2")["sims"]

	waves = ones(length(filenames))
	wave_times = [NaN for i in eachindex(filenames)]
	wave_sizes = zeros(length(filenames))

	true_wave_data = CSV.File(datadir("exp_pro", "true_wave_summary.csv"))
	region_filter = true_wave_data["region"] .== split(sim_name, "_")[2]
	true_wave_times = true_wave_data["peak_times"][region_filter]
	true_wave_sizes = true_wave_data["peak_sizes"][region_filter]
	for (i, fname) in enumerate(filenames)
		results = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))
		pred = results["prediction"]
		days_predicted = range(0.0, step = 1.0, length=size(pred,2)) # Days at which prediction is taken

		I_series = pred[2,:]
		ΔI = I_series[2:end] .- I_series[1:end-1]

		max_filter = [false; [ΔI[j] >=0 && ΔI[j+1] <=0 for j in eachindex(ΔI[1:end-1])]; false]
		magnitude_filter = I_series .>= 1e-3
		combined_filter = max_filter .* magnitude_filter
		peak_times = days_predicted[combined_filter]

		n_waves = sum(combined_filter)
		waves[i] = n_waves
		if length(peak_times) >= 2
			wave_times[i] = peak_times[2]
			wave_sizes[i] = I_series[combined_filter][2]
		end
	end
	med_waves = median(waves)
	max_waves, max_sim = findmax(waves)
	sd_waves = std(waves)
	frac_2ndwaves = sum(waves .>= 2)/length(filenames)
	avg_wave_date = mean(filter(x->!isnan(x), wave_times))
	avg_wave_size = mean(wave_sizes)
	sd_wave_date = std(filter(x->!isnan(x), wave_times))
	sd_wave_size = std(wave_sizes)

	wave_time_err = mean(abs.(filter(x->!isnan(x), wave_times) .- true_wave_times))
	wave_size_err = mean(abs.(wave_sizes .- true_wave_sizes))

	res = DataFrame(Dict{String, Any}(
		"med_waves" => med_waves,
		"max_waves" => max_waves,
		"max_sim" => max_sim,
		"sd_waves" => sd_waves,
		"frac_2ndwaves" => frac_2ndwaves, 
		"avg_wave_date" => avg_wave_date,
		"avg_wave_size" => avg_wave_size,
		"sd_wave_date" => sd_wave_date,
		"sd_wave_size" => sd_wave_size,
		"avg_time_err" => wave_time_err,
		"avg_size_err" => wave_size_err,
		"wave_time_err" => wave_time_err,
		"wave_size_err" => wave_size_err
	))

	CSV.write(root*"\\wave_summary.csv", res)
	nothing
end


function beta_analysis(sim_name)
	root = datadir("sims", "udde", sim_name)
	filenames = load(root * "\\converged_sims.jld2")["sims"]

	normal_beta = zeros(length(filenames))
	crit_beta = zeros(length(filenames))

	for (i, fname) in enumerate(filenames)
		p = load(datadir("sims", "udde", sim_name, fname, "results.jld2"))["p"].layer1
		β(M) = network1([M], p, st1)[1][1]
		normal_beta[i] = β(0)

		# Solve for where β(M) = recovery_rate
		f(M) = β(M)/recovery_rate - 1
		m0 = 0.0
		iters = 0
		while abs(f(m0)) > 1e-5
			gradf = Zygote.gradient(f, m0)[1]
			if iters > 1000 || m0 < -1.0 || gradf == 0
				display("Newton's method failed to converge on $(fname)")
				m0 = NaN
				break
			end
			m0 = m0 - f(m0)/gradf
			iters += 1
		end
		crit_beta[i] = m0
	end

	crit_beta = crit_beta[.!isnan.(crit_beta)]
	frac_crit_valid = length(crit_beta)/length(filenames)

	mean_normal_beta = mean(normal_beta)
	mean_crit_beta = mean(crit_beta)

	sd_normal_beta = std(normal_beta)
	sd_crit_beta = std(crit_beta)

	# Confidence intervals
	CIU_normal = mean_normal_beta + 1.96*sd_normal_beta
	CIL_normal = mean_normal_beta - 1.96*sd_normal_beta

	CIU_crit = mean_crit_beta + 1.96*sd_crit_beta
	CIL_crit = mean_crit_beta - 1.96*sd_crit_beta

	res = DataFrame(mean_normal_beta=mean_normal_beta, mean_crit_beta=mean_crit_beta,
		CIU_normal=CIU_normal, CIL_normal=CIL_normal, 
		CIU_crit=CIU_crit, CIL_crit=CIL_crit, frac_crit_valid = frac_crit_valid)

	pl_beta_normal = histogram(normal_beta, xlabel="β(0)", label=nothing)
	pl_beta_crit = histogram(crit_beta, xlabel="M critical value", label=nothing)
	savefig(pl_beta_normal, joinpath(root, "beta_normal.png"))
	savefig(pl_beta_crit, joinpath(root, "beta_crit.png"))
	CSV.write(joinpath(root, "beta_summary.csv"), res)
	nothing
end


function run_all(sim_name)
	analyze(sim_name)
	loss_analysis(sim_name)
	EnsembleSummary(sim_name)
	wave_count(sim_name)
	beta_analysis(sim_name)
	nothing
end



function overall_summary(sim_list, outname)
	root = datadir("sims", "udde")
	df = DataFrame()


	for sim in sim_list
		if !isfile(joinpath(root, sim, "loss_summary.csv")) || !isfile(joinpath(root, sim, "wave_summary.csv")) || !isfile(joinpath(root, sim, "beta_summary.csv"))
			display("No results found for $(sim)!")
			continue
		end	
		loss_stats = DataFrame(CSV.File(joinpath(root, sim, "loss_summary.csv")))
		wave_stats = DataFrame(CSV.File(joinpath(root, sim, "wave_summary.csv")))
		beta_stats = DataFrame(CSV.File(joinpath(root, sim, "beta_summary.csv")))
		loss_stats[!, "name"] .= sim
		wave_stats[!, "name"] .= sim
		beta_stats[!, "name"] .= sim

		all_stats = innerjoin(loss_stats, wave_stats, on=:name)
		all_stats = innerjoin(all_stats, beta_stats, on=:name)

		append!(df, all_stats)
	end

	CSV.write(joinpath(root, outname*".csv"), df)
	nothing
end



function comparison_plots()
	binn_summ = DataFrame(CSV.File(datadir("sims", "udde", "binn_summ.csv")))
	unbiased_summ = DataFrame(CSV.File(datadir("sims", "udde", "unbiased_summ.csv")))


	x = repeat(all_regions, outer=2)
	pl_stable = groupedbar(x, [binn_summ.frac_stable unbiased_summ.frac_stable], label=nothing,
		xlabel="Region", ylabel="Fraction stable")

	pl_2ndwave = groupedbar(x, [binn_summ.frac_2ndwaves unbiased_summ.frac_2ndwaves], label=nothing,
		xlabel="Region", ylabel="Second wave predictions")

	binn_losses = [mean(binn_summ[!, "avg_"*label]) for label in loss_labels]
	unbiased_losses = [mean(unbiased_summ[!, "avg_"*label]) for label in loss_labels]
	pl_losses = groupedbar(repeat(["acc"; string.(range(1,length(loss_labels)-1))], outer=2), [binn_losses unbiased_losses], label=nothing, 
		xlabel="Loss function", ylabel="Loss", yaxis = :log10)

	pl_conv_iter = groupedbar(x, [binn_summ.mean_conv_iter unbiased_summ.mean_conv_iter], label=nothing,
		xlabel="Region", ylabel="Mean convergence iteration")


	savefig(pl_stable, plotsdir("paper", "stability_comp.png"))
	savefig(pl_2ndwave, plotsdir("paper", "wave_comp.png"))
	savefig(pl_losses, plotsdir("paper", "loss_comp.png"))
	savefig(pl_conv_iter, plotsdir("paper", "iters_comp.png"))
	nothing
end

mean([unbiased_summ[!, "avg_"*l] for l in loss_labels])./mean([binn_summ[!, "avg_"*l] for l in loss_labels])




for region in all_regions
	pred_bias, beta_bias = EnsembleSummary("final_$region", titles=["(a)" "(b)" "(c)" "(a)"])
	pred_unbias, beta_unbias = EnsembleSummary("baseline_$region", titles=["(d)" "(e)" "(f)" "(b)"])
	pl = plot(pred_bias, pred_unbias, layout = (1, 2), size=(1000,800))
	savefig(pl, plotsdir("paper", "appendix", "ensemble_comp_$region.png"))

	pl = plot(beta_bias, beta_unbias, layout = (1, 2), size=(800,500))
	savefig(pl, plotsdir("paper", "appendix", "beta_comp_$region.png"))
end


function all_infected(type)
	preds = []

	for (i,region) in enumerate(all_regions)
		sim_name = type*"_"*region
		root = datadir("sims", "udde", sim_name)
		filenames = load(root * "\\converged_sims.jld2")["sims"]
		f = load(joinpath(root, filenames[1])* "/results.jld2")
		pred = f["prediction"][2,:]
		for fn in filenames[2:end]
			f = load(joinpath(root, fn)* "/results.jld2")
			new_pred = f["prediction"][2,:]
			if size(new_pred, 1) != size(pred, 1)
				new_pred = vcat(new_pred, NaN .*zeros(size(pred, 1)-size(new_pred, 1)))
			end
		pred = hvncat(3, pred, new_pred)
		end
		pred = pred[1:600,:,:]
		mean_pred = mean(pred, dims=3)
		med_pred = median(pred, dims=3)[:,:,1][:]
		qu_pred = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.75) for i in axes(pred,1), j in axes(pred,2)][:]
		ql_pred = [isnan(mean_pred[i,j,1]) ? NaN : quantile(pred[i,j,:], 0.25) for i in axes(pred,1), j in axes(pred,2)][:]

		indicators=[3]
		dataset = load(datadir("exp_pro", "SIMX_final_7dayavg_roll=false_$(region).jld2"))
		data = dataset["data"][hcat([1 2], indicators), :][1,:,:]
		days = dataset["days"]
		all_tsteps = range(0, step=7, length=size(data,2))
		pl_pred = scatter(all_tsteps, data[2,:], color=:black, label=nothing)
		pred_tsteps = range(14.0, step=1.0, length=length(med_pred))
		plot!(pl_pred, pred_tsteps, med_pred, color=:red, ribbon = (med_pred-ql_pred, qu_pred-med_pred), label=nothing)
		if i in [1, 5, 8, 11]
			ylabel!(pl_pred, "I")
		end
		# xlabel!(pl_pred[end], "Time")
		annotate!(10, 0.9*ylims(pl_pred)[2], ("($('a'+i-1))", 16, 0.0, :topleft))
		title!(pl_pred, region_code[split(region, "-")[end]])
		push!(preds, pl_pred)
	end

	lt = @layout [p1 p2 p3 p4; p5 p6 p7; p8 p9 p10; p11 p12 p13]
	pl_final = plot(preds..., size=(1800, 900), layout=lt)

	savefig(pl_final, plotsdir("paper", "$(type)_all_infected.png"))
	nothing
end