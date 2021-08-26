using FiniteDiff
using BenchmarkTools
using Distributed


include("ProbitTypes.jl")
include("SimulateData.jl")
include("EvalDemand.jl")
include("log_likelihood.jl")
include("Constructors.jl")
include("Estimate.jl")
include("Halton.jl")
include("SpecificationRun.jl")


p0 = [1.0;-2.0;
        1.5;1.7;0.8;0.5;-1.0]

search_bounds = [ [-4.0,4.0],[-4,4],
                 [0,5],[0,5],[-2,2],[-2,2], [-2,2], ]



num_particles = 100
startSpace = permutedims(HaltonSpace(num_particles,length(search_bounds),search_bounds),(2,1))


println(p0)
data = simulateData(p0,
                [:V1; :V2],
                4,
                10000,
                1,
                1000)



pars = parDict(p0,data)
individual_shares(data,pars)
ll = log_likelihood_byshare(data,pars)
println(ll)

ll = log_likelihood(data,p0)
println(ll)



estimation_num = 10
beta_est = Matrix{Float64}(undef,length(pars.β),estimation_num)
sig_est = Array{Float64,3}(undef,size(pars.Ω,1),size(pars.Ω,2),estimation_num)






for i in 1:estimation_num
    println("Estimation number: $i")
    data = simulateData(p0,
                    [:V1; :V2],
                    4,
                    10000,
                    1,
                    10000)
    val, p_est = estimate_Model(data,startSpace,"test",testing_run=true)
    # ret, val, p_est = res
    par_est = parDict(p_est,data)
    beta_est[:,i] = par_est.β[:]
    sig_est[:,:,i] = par_est.Ω[:,:]
    println(val)
end

println(mean(beta_est[:,:],dims=2))
println(mean(sig_est[:,:,:],dims=3))

res = estimate_ng(data,p_est)


par_est = parDict(p_est,data)

res = estimate_ng(data,p_init)
(flag, val, p_est) = res
res = estimate_ng(data,p_est)

individual_shares(data,pars)


using Profile
Profile.init(n=10^8,delay=.001)
Profile.clear()
Juno.@profile individual_shares(data,pars)
#Juno.profiletree()
Juno.profiler()




using BenchmarkTools
# @benchmark individual_shares(data,pars)
@benchmark ll = log_likelihood(data,p0)

f_obj(x) = log_likelihood(data,x)
grad = Vector{Float64}(undef,length(p0))
hess = Matrix{Float64}(undef,length(p0),length(p0))

println("Grad")
FiniteDiff.gradient!(grad,f_obj, p0)
