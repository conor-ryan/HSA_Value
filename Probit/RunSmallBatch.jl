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
data_file = "choice14_samp5"
haltonDraws = 10000
include("Load.jl")
# spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
#                 :plan2,:plan3,:plan4,
#                 :hra_cost,:hsa_cost,:hmo_cost]
spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
                :plan2,:plan3,:plan4,
                :hra_cost,:hsa_cost,:hmo_cost,
                :hra_depend,:hsa_depend,:hmo_depend]
data = ChoiceData(df,
                spec=spec_vars,
                est_draws=haltonDraws)

p0 = [-0.0005;-0.0005;-0.0005;-0.0005;.01;.01;.01;
.0001;.0001;.0001;.01;.01;.01;
        1.5;2.0;-0.5;0.25;-0.5]
# p0 = vcat(ones(length(spec_vars)),[1.5;2.0;-0.5;0.25;-0.5])

pars = parDict(p0,data)
individual_shares(data,pars)
println(minimum(pars.s_ij))
ll = log_likelihood(data,p0)
println(ll)

V = calc_Avar(data,p0)

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
variances = [0.01;0.01;0.01;0.01;1;1;1;
                .01;.01;.01;1;1;1;
                1;1;1;1;1]

println("Estimation Begin")
estimate_Model(data,50,p0,variances,
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
