# googleDrivePath = "C:/Users/Stefano/Documents/Research"
googleDrivePath = "G:/My Drive/"
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
include("Elasticity.jl")

include("SpecificationRun.jl")
data_file = "choice11_samp5"
haltonDraws = Int(1e5)
include("Load.jl")
file_out = "$googleDrivePath/HSA Probit/Results/Results-$data_file-draw$haltonDraws"
# spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
#                 :plan2,:plan3,:plan4,
#                 :hra_cost,:hsa_cost,:hmo_cost]
spec_vars = [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
                :plan2,:plan3,:plan4,:plan5,:plan6,:plan7,:plan8,
                :hra_cost,:hsa_cost,:hmo_cost,
                :hra_depend,:hsa_depend,:hmo_depend,
                :hra_over40,:hsa_over40,:hmo_over40]
data = ChoiceData(df,product=[:planid],
                spec=spec_vars,
                est_draws=haltonDraws)


# println(minimum(pars.s_ij))
# ll = log_likelihood(data,p0)
# println(ll)

p_est = [-0.011493121241842607, 0.026548055470485568, 0.018979006308852873, 0.06957477911576672,
 -0.8552334988547332, 9.993118176629617, -2.592927358968003, -0.6321634334419484, -8.17849125674931,
 -4.814669088165273, -8.784343345271429, 0.0005259756492989852, 0.0007317992595092505,
  0.0015435110268290362, 0.0051857655026441365, 0.011579123135279734, -0.003020156072515628,
  0.00012130445239664112, -0.0014534704070951784, -0.0029893311416644176, -1.4884641858479144,
   -2.9489041954662296, -1.6694432452179195, -0.9876708606376372, -5.929595715696769,
   -0.40739298883422936, -4.712646497501815, 0.06228047318934893, 0.6620626753296752,
    -3.023843168262222, 15.744458992718243, 21.610550367661247, -4.726825395584923,
     16.98533046817759, 0.4697380484122662, 42.757328952340934, 0.37442939621732985,
      0.5039696173773289, -3.460168735080324, 1.6419178209346588, -0.6052574002051339,
       -1.015087064577156, 7.50865940663439, -1.1354382607414255, 2.185528870643197,
        0.37477556299788006, 4.457724089523088]
pars = parDict(p_est,data)
println("Evaluate")
individual_shares(data,pars)
println(sum(pars.ll_i))


p0 = p_est


function f_obj(x)
    p = copy(p_est)
    x[1:4] = x[1:4].*2e5
    println(x)
    p = p .+ x[:]
    return log_likelihood(data,p)
end

x = zeros(length(p_est))

grad = Vector{Float64}(undef,length(x))
hess = Matrix{Float64}(undef,length(p0),length(p0))

println("Grad")
FiniteDiff.finite_difference_gradient!(grad,f_obj, x)

pars = parDict(p_est,data)

## Standard Errors
Var, stdErr, t_stat, stars = res_process(data,p_est)

## Output DataFrame
p_print = copy(p_est)
spec_label = String.(data.spec)
for i in length(data.spec):(length(p_est)-1)
    spec_label=vcat(spec_label,"Var_parameter")
end

for i in 1:size(pars.Σ,1), j in 1:i
    p_print = vcat(p_print,pars.Σ[i,j])
    spec_label = vcat(spec_label,"Sigma_$i$j")
    stdErr = vcat(stdErr,0)
    t_stat = vcat(t_stat,0)
    stars = vcat(stars,"")
    if i>j
        continue
    end
end


out = DataFrame(labels=spec_label,pars=p_print,se=stdErr,ts=t_stat,sig=stars)
file = "$file_out.csv"
CSV.write(file,out)


p = 0.0
opt = 1
index = Int.(1:4)

grad = calc_deriv(index,data,p_est)

pars = parDict(p_est,data)
individual_shares(data,pars)



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
