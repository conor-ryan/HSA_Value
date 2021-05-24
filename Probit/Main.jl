using Distributed

addprocs(14)

# @everywhere googleDrivePath = "G:/My Drive"
@everywhere googleDrivePath = "C:/Users/Stefano/Documents/Research"
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

data_file_vec = ["choice11_samp10","choice14_samp20"]#,"choice14_samp10"]
halton_draw_vec = [50000,50001,50002,100000,200000]

for halton_i in halton_draw_vec, data_i in data_file_vec

    @everywhere data_file = $data_i
    @everywhere haltonDraws = $halton_i

    println("Running Data $data_file with $haltonDraws draws")

    @everywhere include("Load.jl")
    @everywhere spec= [:logprem,:logprice_family,:logprice_age_40_60,:logprice_age_60plus,
                    :plan2,:plan3,:plan4,:plan5,:plan6,:plan7,:plan8,
                    :hra_cost,:hsa_cost,:hmo_cost,
                    :hra_depend,:hsa_depend,:hmo_depend]


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

    num_particles = 10000
    startSpace = permutedims(HaltonSpace(num_particles,length(search_bounds),search_bounds),(2,1))


    @everywhere data = ChoiceData(df,product=[:planid],
                    spec=spec,
                    est_draws=haltonDraws)
    @everywhere df = nothing
    @everywhere GC.gc()
    estimate_Model(data,startSpace,
                            "$googleDrivePath/HSA Probit/Results/Results-$data_file-draw$haltonDraws")
end
