using Distributed

addprocs(3)

@everywhere googleDrivePath = "G:/My Drive"
#@everywhere googleDrivePath = "C:/Users/Stefano/Documents/Research"
@everywhere using FiniteDiff
@everywhere using BenchmarkTools




@everywhere include("ProbitTypes.jl")
@everywhere include("Halton.jl")
@everywhere include("utility.jl")
@everywhere include("Constructors.jl")
@everywhere include("EvalDemand.jl")
@everywhere include("log_likelihood.jl")
@everywhere include("Estimate.jl")
@everywhere include("SpecificationRun.jl")

data_file_vec = ["choice14_samp20","choice14_samp10","choice14_samp5"]
halton_draw_vec = [10000,50000]

for halton_i in halton_draw_vec, data_i in data_file_vec

    @everywhere data_file = $data_i
    @everywhere haltonDraws = $halton_i

    println("Running Data $data_file with $haltonDraws draws")

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
                            "$googleDrivePath/HSA Probit/Results/Results-$data_file-draw$haltonDraws")
end
