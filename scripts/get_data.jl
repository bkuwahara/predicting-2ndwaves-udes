#=
Scripts to generate all the processed data that will be used as model input
=#
cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")


using Statistics
using CSV, Dates
using JLD2, FileIO
using DataInterpolations
using Plots
using DataFrames

# Calculates the moving average of data, where period is the period over which to average. 
# If rolling is set to true, it will be a rolling average. Otherwise it will sub-sample
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


# Dictionary containing the population of each region considered
population = Dict{String, Float32}(
	"CA-BC" => 5.14e6,
	"CA-ON" => 14.7e6,
	"CA-QC" => 8.57e6,
	"CA-NS" => 9.79e5,
	"US-CA" => 39.5e6,
	"US-TX" => 29.1e6,
	"US-FL" => 21.5e6,
	"US-NY" => 20.2e6,
	"US-PA" => 13.0e6,
	"UK" => 66.8e6,
	"NL" => 17.6e6,
	"AT" => 9.0e6,
	"DE" => 8.31e7,
	"BE" => 11.8e6,
	"IT" => 61.1e6, #61,095,551  
	"KR" => 51.8e6 #51,844,834
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
	"FL" => "Florida",
	"NS" => "Nova Scotia"
);

# Maps abbreviated country names to full names
country_code = Dict(
	"CA" => "Canada",
	"US" => "United States",
	"UK" => "United Kingdom",
	"NL" => "Netherlands",
	"AT" => "Austria",
	"AU" => "Australia",
	"DE" => "Germany",
	"BE" => "Belgium",
	"IT" => "Italy",
	"KR" => "South Korea"
)

# Generates all possible data of interest (covid data, mobility, stringency index) for country_region
function get_SIMX_data(country_region; sample_period=7, rolling=true)
	abbrs = split(country_region, "-")
	country_abbr, region_abbr = (length(abbrs) == 2) ? abbrs : (country_region, nothing)
	country_name = country_code[country_abbr]
	region_name = isnothing(region_abbr) ? nothing : region_code[region_abbr]
	pop = population[country_region]
	step = (rolling ? 1 : sample_period)

	## Get mobility data
	datafile = CSV.File(datadir("exp_raw", "2020_$(country_abbr)_Region_Mobility_Report.csv"),
		select= [
			"country_region",
			"sub_region_1",
			"sub_region_2",
			"metro_area",
			"date",
			"retail_and_recreation_percent_change_from_baseline",
			"workplaces_percent_change_from_baseline",
			"parks_percent_change_from_baseline"])


	datafile = DataFrame(datafile)
	country_df = datafile[datafile.country_region .== country_name,:]
	region_df = nothing
	if isnothing(region_name)
		region_df = country_df[ismissing.(country_df.sub_region_1) .* ismissing.(country_df.metro_area),:]
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
	mobility_averaged = MovingAverage(all_mobility, sample_period, rolling)


	mobility_days_averaged = all_days_recorded[1+div(sample_period,2):step:end-div(sample_period,2)]
	d0_mobility_averaged = mobility_days_averaged[1]
	d0 = d0_mobility_averaged # Mobility has the latest start date, so use it as start date for all datasets
	df_mobility_averaged = mobility_days_averaged[end]


	## Get case data
	cumulative_cases = nothing
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
	case_days_averaged = range(d0_cases_averaged, step=Day(step), length=size(epidemic_data,2))
	df_cases_averaged = case_days_averaged[end]

	@assert(d0_cases_averaged == d0_mobility_averaged)
	@assert((df_cases_averaged - df_mobility_averaged).value % sample_period == 0)

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
	# Clip to the same reading frame as cases, mobility
	stringency_index = convert(Array{Float64}, stringency_index)
	stringency_index_idxs = @. (stringency_days >= d0 - Day(div(sample_period, 2)))
	stringency_index = stringency_index[stringency_index_idxs]
	stringency_days = stringency_days[stringency_index_idxs]


	# Convert to moving average
	stringency_averaged = MovingAverage(stringency_index, sample_period, rolling)
	stringency_days_averaged = stringency_days[div(sample_period, 2)+1:step:end-div(sample_period, 2)]
	df_stringency_averaged = stringency_days_averaged[end]
	@assert((df_stringency_averaged - df_mobility_averaged).value % sample_period == 0)

	# Select date range
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
	output_fname = "SIMX_final_$(sample_period)dayavg_roll=$(rolling)_$(country_region).jld2"
	save(datadir("exp_pro", output_fname),
		"data", data, "population", population[country_region], "days", days)
	nothing
end




#================================================

================================================#
# Plots the data (just S, I, and retail/recreation mobility) for country_region
function plot_SIM_data(country_region, period, rolling; save=true)
	fname = "SIMX_final_$(period)dayavg_roll=$(rolling)_$(country_region).jld2"
	if !isfile(datadir("exp_pro", fname))
		println("No data exists for this region.")
		return nothing
	end
	dataset = load(datadir("exp_pro", fname))
	data = dataset["data"]
	days = dataset["days"]

	S = data[1,:]
	I = data[2,:]
	M = data[3,:]


	pl = scatter(range(0.0, step=period, length=length(days)), [S I M], label=nothing, ylabel=["S" "I" "M"], 
		title=["COVID-19 data, $country_region" "" ""], 
		xlabel=["" "" "Time (days since $(days[1]))"], layout=(3,1))

	if save
		output_fname = "SIM_$(period)dayavg_roll=$(rolling)_$(country_region).png"
		savefig(pl, plotsdir("datasets", output_fname))
	else
		display(pl)
	end
	nothing
end


# Calculates the "true" size and timing of the second wave for all regions
function true_wave_summary()
	peak_times = zeros(length(keys(population)))
	peak_sizes = zeros(length(keys(population)))
	rolling = false
	sample_period = 7
	for (i, country_region) in enumerate(keys(population))
		abbrs = split(country_region, "-")
		country_abbr, region_abbr = (length(abbrs) == 2) ? abbrs : (country_region, nothing)
		country_name = country_code[country_abbr]
		region_name = isnothing(region_abbr) ? nothing : region_code[region_abbr]
		pop = population[country_region]
		step = (rolling ? 1 : sample_period)

		cumulative_cases = nothing
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
		while d0_cases < Date(2020, 2, 18) - Day(div(sample_period, 2))
			start_idx += 1
			d0_cases += Day(1)
		end

		# Convert to moving average
		I_series = MovingAverage(I[start_idx:end], sample_period, rolling)./pop
		times = range(0.0, step=sample_period, length=length(I_series))

		ΔI = I_series[2:end] .- I_series[1:end-1]

		max_filter = [false; [ΔI[j] > 0 && ΔI[j+1] < 0 for j in eachindex(ΔI[1:end-1])]; false]
		firstwave = I_series[max_filter][1]
		magnitude_filter = I_series .>= 0.25*firstwave
		time_filter = times .>= 200
		combined_filter = max_filter .* magnitude_filter .* time_filter
		peaks = I_series[combined_filter]
		peak_time = times[combined_filter]


		peak_sizes[i] = peaks[1]
		peak_times[i] = peak_time[1]

		pl = plot(times, I_series, title=country_region)
		vline!(pl, peak_time)
		display(pl)
	end

	results = DataFrame(region = collect(keys(population)), peak_times = peak_times, peak_sizes = peak_sizes)
	CSV.write(datadir("exp_pro", "true_wave_summary.csv"), results)
end



for region in keys(population)
	get_SIMX_data(region, rolling=false)
	plot_SIM_data(region, 7, false)
end
