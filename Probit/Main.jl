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

data_file_vec = ["choice14_samp5"]#,"choice14_samp20","choice14_samp10"]
halton_draw_vec = [10000,10001,50000,100000,200000]

for halton_i in halton_draw_vec, data_i in data_file_vec

    @everywhere data_file = $data_i
    @everywhere haltonDraws = $halton_i

    println("Running Data $data_file with $haltonDraws draws")

    @everywhere include("Load.jl")
    @everywhere spec= [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
                    :plan2,:plan3,:plan4,
                    :hra_cost,:hsa_cost,:hmo_cost,
                    :hra_depend,:hsa_depend,:hmo_depend]


    @everywhere p0 = vp0 = [-0.0005;-0.0005;-0.0005;-0.0005;.01;.01;.01;
                            .0001;.0001;.0001;.01;.01;.01;
                                    1.5;2.0;-0.5;0.25;-0.5]

    # @everywhere srch_var = [0.01;0.01;0.01;0.01;1;1;1;
    #                 .01;.01;.01;1;1;1;
    #                 1;1;1;1;1]
    @everywhere search_bounds = [ [-0.05,0.05],[-0.05,0.05],[-0.05,0.05],[-0.05,0.05],
                                [-10,10],[-10,10],[-10,10],
                                [-0.005,0.005],[-0.005,0.005],[-0.005,0.005],
                                [-2,2],[-2,2],[-2,2],
                                [0,10],[0,10],[-5,5],[-5,5],[-5,5]]
    num_particles = 50
    @everywhere startSpace = permutedims(HaltonSpace(num_particles,length(spec),bounds),(2,1))


    @everywhere data = ChoiceData(df,
                    spec=spec,
                    est_draws=haltonDraws)
    @everywhere df = nothing
    @everywhere GC.gc()
    estimate_Model(data,startSpace,
                            "$googleDrivePath/HSA Probit/Results/Results-$data_file-draw$haltonDraws")
end
