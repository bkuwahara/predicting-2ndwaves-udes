cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")


using Statistics
using CSV, Dates
using JLD2, FileIO
using DataInterpolations
using Plots
using DataFrames

function MovingAverage(data, period, rolling=false)
	return_reshape = false
	if length(size(data)) == 1
		data = reshape(data, 1, length(data))
		return_reshape = true
	end
	output_dims = size(data)
	primary_dim = size(data)[end]
	if rolling
		primary_dim = output_dims[end]-(period-1)
	else
		primary_dim = div(output_dims[end], period)
	end
	averaged = zeros((size(data)[1:end-1]..., primary_dim))
	for i = 1:primary_dim
		if rolling
			averaged[:,i] = mean(data[:,i:i+(period-1)], dims=2)
		else
			averaged[:,i] = mean(data[:,1+i*(period-1):i*period], dims=2)
		end
	end
	if return_reshape
		averaged = reshape(averaged, length(averaged))
	end
	return averaged
end


function normalize_data(data; dims=1)
	μ = mean(data, dims=dims)
	sd = std(data, dims=dims)
	data = @. (data - μ)/sd
	return data, μ, sd
end

# Dictionary containing the population of each region considered
population = Dict{String, Float32}(
	"CA-BC" => 5.14e6,
	"CA-AB" => 4.42e6,
	"CA-SK" => 1.18e6,
	"CA-MB" => 1.38e6,
	"CA-ON" => 14.7e6,
	"CA-QC" => 8.57e6,
	"CA-NB" => 7.81e5,
	"CA-NS" => 9.79e5,
	"CA-PE" => 1.60e5,
	"CA-YK" => 4.2e4,
	"CA-NT" => 4.51e4,
	"CA-NU" => 3.9e4,
	"US-CA" => 39.5e6,
	"US-TX" => 29.1e6,
	"US-FL" => 21.5e6,
	"US-NY" => 20.2e6,
	"US-PA" => 13.0e6,
	"UK" => 66.8e6,
	"NL" => 17.6e6
);

# Dictionary mapping abbreviated region names to full names
region_code = Dict(
	"ON" => "Ontario",
	"BC" => "British Columbia",
	"QC" => "Quebec",
	"CA" => "California",
	"PA" => "Pennsylvania",
	"TX" => "Texas",
	"NY" => "New York",
	"FL" => "Florida"
);

country_code = Dict(
	"CA" => "Canada",
	"US" => "United States",
	"UK" => "United Kingdom",
	"NL" => "Netherlands",
)


function get_data(country_region; sample_period=7, rolling=true)
	abbrs = split(country_region, "-")
	country_abbr, region_abbr = (length(abbrs) == 2) ? abbrs : (country_region, nothing)
	country_name = country_code[country_abbr]
	region_name = isnothing(region_abbr) ? nothing : region_code[region_abbr]
	pop = population[country_region]



	## Get mobility data
	datafile = CSV.File(datadir("exp_raw", "2020_$(country_abbr)_Region_Mobility_Report.csv"),
		select= [
			"country_region",
			"sub_region_1",
			"sub_region_2",
			"date",
			"retail_and_recreation_percent_change_from_baseline",
			"workplaces_percent_change_from_baseline",
			"parks_percent_change_from_baseline"])


	datafile = DataFrame(datafile)
	country_df = datafile[datafile.country_region .== country_name,:]
	region_df = nothing
	if isnothing(region_name)
		region_df = country_df[ismissing.(country_df.sub_region_1),:]
	else
		dropmissing!(country_df, :sub_region_1)
		region_subsets = country_df[ismissing.(country_df.sub_region_2),:]
		region_df = region_subsets[region_subsets.sub_region_1 .== region_name,:]
	end

	all_mobility = [region_df.retail_and_recreation_percent_change_from_baseline';
					region_df.workplaces_percent_change_from_baseline';
					region_df.parks_percent_change_from_baseline']./100
	all_days_recorded = region_df.date

	# Interpolate missing data for mobility
	times = Array(range(0.0, length = size(all_mobility,2), step = 1.0))
	interp = QuadraticInterpolation(all_mobility, times)

	for i=1:size(all_mobility,1), j =1:size(all_mobility, 2)
		if ismissing(all_mobility[i, j])
			all_mobility[i, j] = interp(times[j])[i]
		end
	end

	all_mobility = convert(Array{Float64}, all_mobility)
	all_mobility, μ, sd = normalize_data(all_mobility, dims=2)
	mobility_averaged = MovingAverage(all_mobility, sample_period, rolling)

	mobility_days_averaged = all_days_recorded[1+div(sample_period,2):step:end-div(sample_period,2)]
	d0_mobility_averaged = mobility_days_averaged[1]
	df_mobility_averaged = mobility_days_averaged[end]


	## Get case data
	cumulative_cases = nothing
	dates = nothing
	abbrs = split(country_region, "-")

	if country_abbr == "US"
		fname =  "time_series_covid19_confirmed_US.csv"
		datafile = CSV.File(datadir("exp_raw", fname))
		df = DataFrame(datafile)
		region_df = df[df.province_state .== region_name,:]
		cumulative_cases = sum(Array(region_df[:,12:end]), dims=1)
	else
		fname = "time_series_covid19_confirmed_global.csv"
		datafile = CSV.File(datadir("exp_raw", fname))
		df = DataFrame(datafile)
		country_df = df[df.country .== country_name,:]
		region_df = isnothing(region_name) ? country_df[ismissing.(country_df.province_or_state),:] : country_df[country_df.province_or_state .== region_name,:]
		cumulative_cases = Array(region_df[:,5:end])
	end


	# Get the number of cases each day
	daily_cases = [0; cumulative_cases[2:end] .- cumulative_cases[1:end-1]]

	# Infer proportion infected from daily cases
	c = 5 # underreporting_ratio
	γ = 0.25  # Recovery rate per day
	I = zeros(length(daily_cases))
	S = zeros(length(daily_cases))
	I[1] = 0
	S[1] = pop
	for j = 2:length(I)
		I[j] = (1-γ)*I[j-1] + c*daily_cases[j]
		S[j] = S[j-1] - c*daily_cases[j]
	end

	# Get the same reading frame as mobility
	start_idx = 1
	d0_cases = Date(2020, 1, 22) # First date cases are recorded
	while d0_cases < d0_mobility_averaged - Day(div(sample_period, 2))
		start_idx += 1
		d0_cases += Day(1)
	end

	epidemic_data = [S'; I'][:, start_idx:end]
	# Convert to moving average
	epidemic_data = MovingAverage(epidemic_data, sample_period, rolling)./pop
	d0_cases_averaged = d0_cases + Day(div(sample_period, 2))
	step = (rolling ? 1 : sample_period)
	case_days_averaged = range(d0_cases_averaged, step=Day(step), length=size(epidemic_data,2))
	df_cases_averaged = case_days_averaged[end]



	## Stringency index
	datafile = CSV.File(datadir("exp_raw", "strin_index.csv"))
	df = DataFrame(datafile)
	country_df = df[df.country_name .== country_code[country_abbr],:]
	stringency_index = Array(country_df[:,4:end])
	stringency_days = range(Date(2020, 1, 1), step=Day(1), length=length(stringency_index))

	# Filter missing values (marked with "NA")
	for i= 1:length(stringency_index)
		if typeof(stringency_index[i]) != Float64
			try
				stringency_index[i] = parse(Float64, stringency_index[i])
			catch
				@warn "Non-numeric input found: $(stringency_index[i]), index $(i)"
				stringency_index[i] = 0.0
			end
		end
	end
	stringency_index = convert(Array{Float64}, stringency_index)

	# Clip to the same reading frame as cases, mobility
	stringency_start_idx = (d0_mobility_averaged - stringency_days[1]).value+1-div(sample_period, 2)
	stringency_index = stringency_index[1, stringency_start_idx-div(sample_period, 2):end]

	# Convert to moving average
	stringency_averaged = MovingAverage(stringency_index, sample_period, rolling)
	stringency_days_averaged = stringency_days[stringency_start_idx+div(sample_period, 2):step:end-div(sample_period, 2)]
	df_stringency_averaged = stringency_days_averaged[end]

	# Select date range
	d0 = d0_mobility_averaged
	df = min(df_cases_averaged, df_mobility_averaged, df_stringency_averaged)

	# Select mobility from the right time period
	idxs = @. (mobility_days_averaged >= d0)*(mobility_days_averaged <= df)
	mobility = mobility_averaged[:, idxs]
	# Same for cases and stringency index
	idxs = @. (case_days_averaged >= d0)*(case_days_averaged <= df)
	epidemic_data = epidemic_data[:, idxs]

	idxs = @. (stringency_days_averaged >= d0)*(stringency_days_averaged <= df)
	stringency_data = stringency_averaged[idxs]

	days = range(d0, step=Day(step), stop=df)
	data = [epidemic_data; mobility; stringency_data']

	# Save
	output_fname = "SIMX_$(sample_period)dayavg_roll=$(rolling)_$(country_region).jld2"
	save(datadir("exp_pro", output_fname),
		"data", data, "population", population[country_region], "days", days,
		"mobility_mean", μ, "mobility_std", sd)
	nothing
end





function plot_SIM_data(country_region; save=true)
	fname = "SIM_weekly_avg_2020_$(country_region).jld2"
	if !isfile(datadir("exp_pro", fname))
		println("No data exists for this region.")
		return nothing
	end
	dataset = load(datadir("exp_pro", fname))
	data = dataset["data"]
	days = dataset["days"]

	pl = plot(days, data', layout=(3,1),
		title=["$(country_region)" "" ""],
		ylabel=["S" "I" "M"], label=["" "" ""])

	if save
		output_fname = "SIM_weekly_avg_$(country_region).png"
		savefig(pl, plotsdir("datasets", output_fname))
	else
		display(pl)
	end
	nothing
end



function plot_SIMX_data(country_region; save=true)
	fname = "SIMX_7dayavg_2020_$(country_region).jld2"
	if !isfile(datadir("exp_pro", fname))
		println("No data exists for this region.")
		return nothing
	end
	dataset = load(datadir("exp_pro", fname))
	data = dataset["data"]
	days = dataset["days"]

	S = data[1,:]
	I = data[2,:]
	M = data[4:5,:]
	X = data[6,:]

	p1 = plot(days, S, ylabel="S", label="")
	p2 = plot(days, I, ylabel="I", xlabel="Time", label="")
	p3 = plot(days, M', ylabel="M", label=["Workplace" "Parks"])
	p4 = plot(days, X, ylabel="X", xlabel="Time", label="")

	lt = @layout [a b
	 			  c d]
	pl = plot(p1, p3, p2, p4, layout=lt)

	if save
		output_fname = "SIMX_weekly_avg_$(country_region).png"
		savefig(pl, plotsdir("datasets", output_fname))
	else
		display(pl)
	end
	nothing
end




function subsample_data(country_region, sample_period)
	dataset = load(datadir("exp_pro", "SIM_c=5_g=0.25_$(country_region).jld2"))
	data = dataset["data"]
	days = dataset["days"]

	nperiods = div(length(days), sample_period)
	sampled_data = zeros(size(data, 1), nperiods)
	for i = 1:nperiods
		sampled_data[:,i] = mean(data[:, (i-1)*sample_period+1:i*sample_period], dims=2)
	end
	save(datadir("exp_pro", "SIM_sampled_$(country_region)_T=$(sample_period).jld2"),
		"data", sampled_data, "days", days, "population", dataset["population"])
end


function load_data(country_region)
	return load(datadir("exp_pro", "SIMX_7dayavg_2020_$(country_region).jld2"))
end


##

get_data("UK")

plot_SIM_data("UK")

dataset = load_data("CA-ON")
data = dataset["data"]
plot(data[6,:])
histogram(sqrt.(data[6,:]))

deltas = data[:, 2:end] - data[:, 1:end-1]

plot(deltas[6,:])
histogram(deltas[6,:])
