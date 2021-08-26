function simulateData(x::Vector{Float64},spec::Vector{Symbol},opt_num::Int,N::Int,
                        sim_draw_num::Int,est_draw_num::Int)
    spec_len = length(spec)
    X = MVHaltonNormal(N*opt_num,spec_len).*1

    ## Person Dictionary
    # _perDict = Matrix{Int64}(undef,opt_num,N)
    _perDict = Dict{Int,Array{Int,1}}()
    _optDict = Dict{Int,Array{Int,1}}()
    k = 1
    for i in 1:N #, j in 1:opt_num
        # _perDict[j,i] = k
        # k+=1
        _perDict[i] = k:(k+opt_num-1)
        _optDict[i] = 1:opt_num
        k+=opt_num
    end

    ## Draws
    draws = MVHalton(sim_draw_num,opt_num,scrambled=true)
    draws = permutedims(draws,(2,1))

    Y_1 = rand(N*opt_num)
    # return ChoiceData(X,spec,Y_1,draws,opt_num,N,_perDict)
    c = ChoiceData(X,spec,Y_1,draws,opt_num,N,_perDict,_optDict)
    pars = parDict(x,c)
    simulate_shares(c,pars)
    Y_2 = pars.s_ij[:]
    # draws = MVHaltonNormal(est_draw_num,opt_num-1)
    draws = MVHalton(est_draw_num,opt_num,scrambled=true)
    draws = permutedims(draws,(2,1))

    return ChoiceData(X,spec,Y_2,draws,opt_num,N,_perDict,_optDict)
end
