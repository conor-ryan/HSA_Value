googleDrivePath = "G:/My Drive"
using FiniteDiff
using BenchmarkTools
using Distributed
include("ProbitTypes.jl")
include("Halton.jl")
include("utility.jl")
include("Constructors.jl")
include("EvalDemand.jl")
include("log_likelihood.jl")
include("Estimate.jl")
include("SpecificationRun.jl")
data_file = "choice11_samp5"
haltonDraws = 1000
include("Load.jl")
# spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
#                 :plan2,:plan3,:plan4,
#                 :hra_cost,:hsa_cost,:hmo_cost]
spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
                :plan2,:plan3,:plan4,:plan5,:plan6,:plan7,:plan8,
                :hra_cost,:hsa_cost,:hmo_cost,
                :hra_depend,:hsa_depend,:hmo_depend]
data = ChoiceData(df,product=[:planid],
                spec=spec_vars,
                est_draws=haltonDraws)

p0 = rand(100)

pars = parDict(p0,data)
individual_shares(data,pars)
println(minimum(pars.s_ij))
ll = log_likelihood(data,p0)
println(ll)

# V = calc_Avar(data,p0)

# p0[1]+=1e-6
# ll = log_likelihood(data,p0)
# println(ll)
#
#
# test = iterate(eachperson(data),1)[1]
#
# pars = parDict(p0,test)
# individual_shares(test,pars)
# println(minimum(pars.s_ij))
# ll = log_likelihood(test,p0)
# println(ll)

# variances = ones(length(p0))
search_bounds = [ [-0.05,0.05],[-0.05,0.05],[-0.05,0.05],[-0.05,0.05],
                            [-10,10],[-10,10],[-10,10],[-10,10],[-10,10],[-10,10],[-10,10],
                            [-0.005,0.005],[-0.005,0.005],[-0.005,0.005],
                            [-0.005,0.005],[-0.005,0.005],[-0.005,0.005]]

# Add variance terms
for k in 1:(data.opt_num-2)
        search_bounds = cat(search_bounds,[[-2,2]],dims=1)
end

# Add Covariance terms
for k in 1:( ((data.opt_num-1)^2 - (data.opt_num-1))/2 )
        search_bounds = cat(search_bounds,[[-5,5]],dims=1)
end

num_particles = 100
startSpace = permutedims(HaltonSpace(num_particles,length(search_bounds),search_bounds),(2,1))


println("Estimation Begin")
estimate_Model(data,startSpace,
                        "$googleDrivePath/HSA Probit/Results/Test")

res = particle_swarm(50,data,p0,tol_imp=1e-5,tol_dist=1e-2,verbose=true,variances=variances)
#
#
#
# flag, val, p_est = res
#
#
# par_est = parDict(p_est,data)


f_obj(x) = log_likelihood(data,x)
grad = Vector{Float64}(undef,length(p0))
hess = Matrix{Float64}(undef,length(p0),length(p0))

println("Grad")
FiniteDiff.finite_difference_gradient!(grad,f_obj, p0)
