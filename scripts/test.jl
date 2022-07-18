cd(@__DIR__)
using DrWatson
@quickactivate("S2022 Project")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim
using DifferentialEquations
using LinearAlgebra
using DiffEqSensitivity
using GalacticOptim
using JLD2, FileIO
using Dates
using Plots
using DataInterpolations


function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the solution
using Plots
plot(sol)

function loss(p)
  sol = solve(prob, Tsit5(), p=p, saveat = tsteps, sensealg=ForwardSensitivity())
  loss = sum(abs2, sol.-1)
  return loss, sol
end

callback = function (p, l, pred)
  display(l)
  plt = plot(pred, ylim = (0, 6))
  display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

result_ode = DiffEqFlux.sciml_train(loss, p,
                                    cb = callback,
                                    maxiters = 100)



dudt(u,p,t) = -tanh(u)
u0 = 10.0
tspan = (0.0,10.0)
prob = ODEProblem(dudt, u0, tspan, 0.0)
sol = solve(prob)
plot(sol)
prob = remake(prob, u0=-10.0)
sol = solve(prob)
plot!(sol)

dudt2(u,p,t) = -(u)
u0 = 10.0
tspan = (0.0,10.0)
prob = ODEProblem(dudt2, u0, tspan, 0.0)
sol = solve(prob)
plot!(sol)
prob = remake(prob, u0=-10.0)
sol = solve(prob)
plot!(sol)

##


# Declare variables
const frac_training = 0.75
const recovery_rate = 1/4

const maxiters = 10000
const lr = 0.01

activation = relu

# function run_model(region)
model_name = "ude"
exp_name = "$(region)"

println("Starting run: $(exp_name)")

verbose = (isempty(ARGS)) ? true : false
opt = ADAM(lr)

## Load in data for the region and partition into training, testing
dataset = load(datadir("exp_pro", "SIM_weekly_avg_2020_$(region).jld2"))
training_data = dataset["data"][3,:]
days = dataset["days"][1:50]


nn = FastChain(FastDense(1, 5, activation), FastDense(5, 5, activation), FastDense(5, 5, activation), FastDense(5, 1))
p0 = initial_params(nn)

function ude(du, u, p, t)
	du[1] = nn(t, p)[1]
	nothing
end

u0 = [training_data[1]]
t_train = range(0.0, step=1.0, length=length(training_data))



function predict(θ)
	Array(solve(prob_nn, Tsit5(),
		p=θ, saveat=1.0, sensealg=ForwardDiffSensitivity()))
end;
function loss(θ)
	pred=predict(θ)
	return sum(abs2, (training_data[1:length(pred)] - pred'))/length(pred), pred
end;
losses = []
training_iters = 0;
function callback(θ, l, pred)
	if verbose && (training_iters % 100 == 0)
		pl = plot(t_train[1:size(pred,2)], training_data[1:size(pred,2)], label="data")
		plot!(pl, t_train[1:size(pred,2)], pred', label="prediction")
		display(pl)
		display(l)
	end
	global training_iters += 1
	if training_iters % 100 == 0
		push!(losses, l)
	end
	if l > 1e12 && training_iters < 10
		throw("Bad initialization. Aborting...")
		return true
	elseif l < 1e-5
		println("Training converged after $(training_iters) iterations")
		return true
	else
		return false
	end
end;

prob_nn = ODEProblem(ude, u0, (0.0, t_train[end]), p0)
p = p0
# for (i, stop) in enumerate(stops[2:end])
# 	prob_nn = remake(prob_nn, p=p, tspan=(0.0,stop))
# 	res = DiffEqFlux.sciml_train(loss, p, ADAM(lr), cb=callback, maxiters=1500, allow_f_increases=true)
# 	p = res.minimizer
# end


prob_nn = remake(prob_nn, p=p, tspan=(0.0,25.0))
res = DiffEqFlux.sciml_train(loss, p, ADAM(0.05), cb=callback, maxiters=1500, allow_f_increases=true)
res = DiffEqFlux.sciml_train(loss, res.minimizer, ADAM(0.005), cb=callback, maxiters=3000, allow_f_increases=true)
p = res.minimizer




m = Chain(RNN(2 => 5), Dense(5 => 1))


function loss(x, y)
  sum(Flux.Losses.mse(m(xi), yi) for (xi, yi) in zip(x, y))
end



seq_init = [rand(Float32, 2)]
seq_1 = [rand(Float32, 2) for i = 1:3]
seq_2 = [rand(Float32, 2) for i = 1:3]

y1 = [rand(Float32, 1) for i = 1:3]
y2 = [rand(Float32, 1) for i = 1:3]

X = [seq_1, seq_2]
Y = [y1, y2]
data = zip(X,Y)

Flux.reset!(m)
[m(x) for x in seq_init]

ps = Flux.params(m)
opt= ADAM(1e-3)
Flux.train!(loss, ps, data, opt)
