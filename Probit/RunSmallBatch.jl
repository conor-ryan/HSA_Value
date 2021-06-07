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
haltonDraws = 50000
include("Load.jl")
# spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
#                 :plan2,:plan3,:plan4,
#                 :hra_cost,:hsa_cost,:hmo_cost]
spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
                :plan2,:plan3,:plan4,#:plan5,:plan6,:plan7,:plan8,
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

p_est = [0.020473887653262697, -0.005726597522972721, -0.0062042829161203066, -0.004818632418367471, -0.8193564280786587, -0.985434515819953, -0.5103057051346926, -4.6300019602959556e-5, -0.0001304257049990771, -0.0001330656093732179, 0.009858338056940787, -0.06524783344203014, 0.05838648016892095, -11.049853755070512, 0.31585400920138873, 19.282075441026336, 0.37566369165701297, 0.6751947168595258, 0.47141542189065044, 0.2576305380647831, 1.1429254820680494, -0.6214535802030503, 1.018765340167648, 0.9616158282257881, 0.9003585177385274, -0.07642815423690802, 17.66062183632715,
-0.27706556130435556, -0.01865041634636931, -0.03327482036923525, -29.737320916342078, -0.0941466599020705, -0.07853947538468303, -0.08361268709870262, 0.27432935642805967, 0.022947568441646613, 0.007394341251062025, 0.032492818166762326, 0.0012026664345907757, -0.10995481879242688]
p0 = p_est
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
