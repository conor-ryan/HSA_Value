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
haltonDraws = Int(1e3)
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

p_est = [-0.008317811448988549, 0.08291237189988446, -0.08161810365871469, -0.015854391549127145,
-0.7179905287160135, 2.0016807189163073, -13.498345372279518, -14.329622842608662, -34.57351200589403,
 -9.228468480801585, -14.308250015210977, 0.000645018931479215, -0.0002849452090445206,
 0.0005808929925723021, -0.013632407307412976, 0.0005476806935812557, -0.03382122287336008,
 0.0006136832983576981, -0.0005740695120465551, 0.00285145275198236, -4.2441805763111695,
 -1.133843278406061, 0.10959386396815697, -2.601244311672703, -4.615884456937003, 1.3509360194330104,
  -1.8205623610290014, 0.8652394416174275, -11.685737996657613, -0.3677134721209504, -7.854084637330585,
   5.8244728742358856, -7.590394294620495, -18.266981781747624, -2.6902284557181955, 0.23042604553220603,
    -0.0026088923121506944, -2.3580041436809696, 0.03732996171315472, 15.732827526405861,
    1.6609760739786084, -2.8783630270600282, 16.36010827486577, -1.435473975968423, 0.7124797160406442,
     8.584436290297862, -12.174693633800633]
pars = parDict(p_est,data)
println("Evaluate")
individual_ll_GHK(data,pars)
println(sum(pars.ll_i))


Var, stdErr, t_stat, stars = res_process(data,p_est)

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
