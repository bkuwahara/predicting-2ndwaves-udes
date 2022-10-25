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
using Plots

n = parse(Int, ARGS[1])

x = LinRange(0.0, 2pi, n)

println("Running on single thread.")
@time for i=1:10
	println("Iteration $i on single thread")
	y = rand()*sin.(x)
	pl = plot(x, y)

	ver = 1
	while isfile("./temp/pl_s_v$(ver).png")
		ver += 1
	end
	savefig(pl, "./temp/pl_s_v$(ver).png")
end

println("Beginning multi-threaded execution.")
@time Threads.@threads for i=1:10
	println("Running on thread $(Threads.threadid())")
	y = rand()*sin.(x)
	pl = plot(x, y)
	ver = 1
	while isfile("./temp/pl_t$(Threads.threadid())_v$(ver).png")
		ver += 1
	end
	savefig(pl, "./temp/pl_t$(Threads.threadid())_v$(ver).png")
end

