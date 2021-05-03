googleDrivePath = "G:\My Drive"
include("Load.jl")
include("ProbitTypes.jl")
include("Halton.jl")
include("utility.jl")
include("Constructors.jl")
include("ProbitParams.jl")
include("log_likelihood.jl")
include("Estimate.jl")
data = ChoiceData(df,
                spec=[:adjprem,:plan2,:plan3,:plan4,
                :coins,:sclb,:scub,
                :plan2_cost,:plan3_cost,:plan4_cost],
                est_draws=1000)

p0 = [-0.0005;.01;.01;.01;.01;.01;.01;
.0001;.0001;.0001;
        1.5;2.0;-0.5;0.25;-0.5]

pars = parDict(p0,data)
individual_shares(data,pars)
println(minimum(pars.s_ij))
ll = log_likelihood(data,p0)
println(ll)


variances = [0.001;1;1;1;1;1;1;
                .001;.001;.001;
                1;1;1;1;1]
println("Estimation Begin")
res = particle_swarm(50,data,p0,tol_imp=1e-5,tol_dist=1e-2,verbose=true,variances=variances)



val, p_est = res
res = particle_swarm(50,data,p_est,tol_imp=1e-5,tol_dist=1e-2,verbose=true,variances=variances)

par_est = parDict(p_est,data)
