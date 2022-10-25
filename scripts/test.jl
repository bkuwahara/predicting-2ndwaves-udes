cd(@__DIR__)
using DrWatson
@quickactivate("S2022_Project")

using DrWatson
@quickactivate("S2022_Project")
using Lux, Zygote
using Optimisers
using DifferentialEquations
using LinearAlgebra
using Statistics, Distributions
using Random; rng = Random.default_rng()

n = parse(Int, ARGS[1])



println("Running on single thread.")
@time for i=1:10
	println("Iteration $i on single thread")
	A = rand(n, n)
	ver = 1
	while isfile("./temp/results_s_v$(ver).jld2")
		ver += 1
	end
	save("./temp/results_s_v$(ver).jld2", "data", A)
end

println("Beginning multi-threaded execution.")
@time Threads.@threads for i=1:10
	println("Running on thread $(Threads.threadid())")
	A = rand(n, n)
	ver = 1
	while isfile("./temp/results_t$(Threads.threadid())_v$(ver).jld2")
		ver += 1
	end
	save("./temp/results_t$(Threads.threadid())_v$(ver).jld2", "data", A)
end

