cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")
using Plots
using JLD2, FileIO
using Statistics
using DiffEqFlux, Flux
using Dates
using Statistics

base_dir = datadir("sims", "udde_validation")

regions = ["US-NY", "CA-ON", "UK"]
n_sims = 6 # number of variants per region]
model_scores = zeros(length(regions), n_sims) # each region is a row

for (j, region) in enumerate(regions)
    dir = joinpath(base_dir, region)
    files = readdir(dir)

    for (i, fname) in enumerate(files)
        results = load(joinpath(dir, fname))

        scores = results["scores"]

        tau_m = results["tau_m"]
        tau_r = results["tau_r"]
        region = results["region"]

        overall_score = mean(scores)
        model_scores[j, i] = overall_score
    end

    score, index = findmin(model_scores[j,:])
    best_model = files[index]
    println("Best model for $(region): $(best_model) with average score $(score)")
end

mean_scores = mean(model_scores, dims=1)
best_overall, index = findmin(mean_scores)
