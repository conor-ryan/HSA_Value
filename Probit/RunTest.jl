include("ProbitTypes.jl")
include("SimulateData.jl")
include("EvalDemand.jl")
include("log_likelihood.jl")
include("Constructors.jl")
include("Estimate.jl")
include("Halton.jl")


p0 = [0.25;-0.5;
        1.5;2.0;-0.5;0.25;-0.5]

    # 1.0  1.5     2.0
    # 1.5  2.3125  3.0625
    # 2.0  3.0625  4.125

println(p0)
data = simulateData(p0,
                [:V1; :V2],
                4,
                100,
                10000,
                1000)

using Profile
Profile.init(n=10^8,delay=.001)
Profile.clear()
Juno.@profile log_likelihood(data,p0)
#Juno.profiletree()
Juno.profiler()



pars = parDict(p0,data)
individual_shares(data,pars)
println(minimum(pars.s_ij))
ll = log_likelihood(data,p0)
println(ll)

estimation_num = 10
beta_est = Matrix{Float64}(undef,length(pars.β),estimation_num)
sig_est = Array{Float64,3}(undef,size(pars.Σ,1),size(pars.Σ,2),estimation_num)


for i in 1:estimation_num
    println("Estimation number: $i")
    data = simulateData(p0,
                    [:V1; :V2],
                    4,
                    100,
                    1000,
                    1000)
    p_init = vcat((rand(2).-0.5).*4,rand(2)*3,rand(3)*2 .-1)
    res = particle_swarm(25,data,p_init,tol_imp=1e-8,tol_dist=1e-2,verbose=false)
    ret, val, p_est = res
    par_est = parDict(p_est,data)
    beta_est[:,i] = par_est.β[:]
    sig_est[:,:,i] = par_est.Σ[:,:]
    println(val)
end

println(mean(beta_est[:,1:9],dims=2))
println(mean(sig_est[:,:,1:9],dims=3))

res = estimate_ng(data,p_est)


par_est = parDict(p_est,data)

res = estimate_ng(data,p_init)
(flag, val, p_est) = res
res = estimate_ng(data,p_est)

individual_shares(data,pars)

using BenchmarkTools
# @benchmark individual_shares(data,pars)
@benchmark ll = log_likelihood(data,p0)

f_obj(x) = log_likelihood(data,x)
grad = Vector{Float64}(undef,length(p0))
hess = Matrix{Float64}(undef,length(p0),length(p0))

println("Grad")
FiniteDiff.gradient!(grad,f_obj, p0)
