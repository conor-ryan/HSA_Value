googleDrivePath = "G:/My Drive"
using FiniteDiff
using BenchmarkTools
include("Load.jl")
include("ProbitTypes.jl")
include("Halton.jl")
include("utility.jl")
include("Constructors.jl")
include("EvalDemand.jl")
include("log_likelihood.jl")
include("Estimate.jl")
data = ChoiceData(df,
                spec=[:adjprem,:plan2,:plan3,:plan4,
                :coins,:sclb,:scub,
                :plan2_cost,:plan3_cost,:plan4_cost],
                est_draws=10000)

p0 = [-0.0005;.01;.01;.01;.01;.01;.01;
.0001;.0001;.0001;
        1.5;2.0;-0.5;0.25;-0.5]

pars = parDict(p0,data)
individual_shares(data,pars)
println(minimum(pars.s_ij))
ll = log_likelihood(data,p0)
println(ll)

V = calc_Avar(data,p0)

test = iterate(eachperson(data),1)[1]

pars = parDict(p0,test)
individual_shares(test,pars)
println(minimum(pars.s_ij))
ll = log_likelihood(test,p0)
println(ll)

variances = [0.001;1;1;1;1;1;1;
                .001;.001;.001;
                1;1;1;1;1]
println("Estimation Begin")
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
