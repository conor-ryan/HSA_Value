using Distributed

addprocs(3)

@everywhere googleDrivePath = "G:/My Drive"
#@everywhere googleDrivePath = "C:/Users/Stefano/Documents/Research"
@everywhere using FiniteDiff
@everywhere using BenchmarkTools

@everywhere data_file = "choice14_samp5"
@everywhere haltonDraws = 10000


@everywhere include("ProbitTypes.jl")
@everywhere include("Halton.jl")
@everywhere include("utility.jl")
@everywhere include("Constructors.jl")
@everywhere include("EvalDemand.jl")
@everywhere include("log_likelihood.jl")
@everywhere include("Estimate.jl")
@everywhere include("SpecificationRun.jl")

@everywhere include("Load.jl")
@everywhere spec=[:adjprem,:plan2,:plan3,:plan4,
                        :coins,:sclb,:scub,
                        :plan2_cost,:plan3_cost,:plan4_cost]


@everywhere p0 = [-0.0005;.01;.01;.01;.01;.01;.01;
                        .0001;.0001;.0001;
                        1.5;2.0;-0.5;0.25;-0.5]

@everywhere srch_var = [0.001;1;1;1;1;1;1;
                .001;.001;.001;
                1;1;1;1;1]
num_particles = 50



@everywhere data = ChoiceData(df,
                spec=spec,
                est_draws=haltonDraws)
@everywhere df = nothing
@everywhere GC.gc()
estimate_Model(data,num_particles,p0,srch_var,
                        "$googleDrivePath/HSA Probit/Results/Results-$datafile-draw$haltonDraws")
