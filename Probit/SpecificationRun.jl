using LinearAlgebra

function estimate_Model(data::ChoiceData,
                        num_particles::Int,
                        p_init::Vector{Float64},
                        var_init::Vector{Float64},
                        file_out::String;
                        verbose=true)


    println("Estimation Begin")

    ## Estimate
    res = particle_swarm_parallel(num_particles,p_init,data,
            tol_imp=1e-5,tol_dist=1e-2,verbose=verbose,variances=var_init)

    flag, val, p_est = res

    pars = parDict(p_est,data)

    ## Standard Errors
    Var, stdErr, t_stat, stars = res_process(data,p_est)

    ## Output DataFrame
    p_print = copy(p_est)
    spec_label = String.(spec)
    for i in length(spec):(length(p_est)-1)
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


end



function res_process(d::ChoiceData,p::Array{T,1}) where T
    ## Gradient Based Asymptotic Variance
    aVar = calc_Avar(d,p)



    if any(diag(aVar.<0))
        println("Some negative variances")
        stdErr = sqrt.(abs.(diag(aVar)))
    else
        stdErr = sqrt.(diag(aVar))
    end
    t_stat = p./stdErr

    stars = Vector{String}(undef,length(t_stat))
    for i in 1:length(stars)
        if abs(t_stat[i])>2.326
            stars[i] = "***"
        elseif abs(t_stat[i])>1.654
            stars[i] = "**"
        elseif abs(t_stat[i])>1.282
            stars[i] = "*"
        else
            stars[i] = ""
        end
    end

    return aVar, stdErr, t_stat, stars
end
