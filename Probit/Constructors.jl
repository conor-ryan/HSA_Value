using DataFrames
function ChoiceData(data_choice::DataFrame;
    est_draws = 1000,
    person=[:studyid],
    product=[:newpid],
    spec=[:adjprem],
    choice=[:yvar])

    # Get the size of the data
    n, k = size(data_choice)

    # Convert everything to an array once for performance
    i = convert(Matrix{Float64},data_choice[:,person])
    j = convert(Matrix{Float64},data_choice[:,product])
    X = convert(Matrix{Float64},data_choice[:,spec])
    y = convert(Matrix{Float64},data_choice[:,choice])

    # index = Dict{Symbol, Int}()
    # dmat = Matrix{Float64}(undef,n,0)

    options = sort(unique(j))

    ### Create Person and Option Dictionary
    people = sort(Int.(unique(i)))
    _perDict = Dict{Int,Array{Int,1}}()
    _optDict = Dict{Int,Array{Int,1}}()
    for p in people
        idx1 = searchsortedfirst(i[:],p)
        idxJ = searchsortedlast(i[:],p)
        _perDict[p] = idx1:idxJ
        prods = j[idx1:idxJ]
        _optDict[p] = findall(inlist(options,prods))
    end

    opt_num = length(options)
    N = length(people)

    draws = MVHaltonNormal(est_draws,opt_num-1)

    ## Hard-Coded Default:
    # Location normalized product: 1st index
    # Scale normalized product: 2nd index

    # Make the data object
    m = ChoiceData(X,spec,y[:],draws,opt_num,N,_perDict,_optDict)
    return m
end



function parDict(x::Vector{T},data::ChoiceData) where T
    # Parameter Vectors
    β = x[1:length(data.spec)]
    opt_num = data.opt_num
    A = Matrix{T}(undef,opt_num-1,opt_num-1)
    A[:].=0.0
    # A[1,1]=1.0
    k = [length(data.spec)+1]
    # for i in 2:(opt_num-1)
    #     Σ[i,i] = x[k]^2
    #     k+=1
    # end
    for i in 1:(opt_num-1)
        for j in i:(opt_num-1)
            if (i==j) & (i==1)
                A[i,i] = 1.0
            elseif (i==j)
                corr =x[k[1]]
                A[i,j] = corr^2
                k[:] .+= 1
            else
                corr =x[k[1]]
                A[i,j] = corr
                k[:] .+= 1
            end
        end
    end
    # Initialize Idiosyncratic Draws
    Σ = A'*A
    ϵ = Matrix{T}(undef,opt_num,size(data.draws,1))
    ϵ[1,:].=0.0
    ϵ[2:opt_num,:] = permutedims(data.draws*A,(2,1))


    # Initialize individual shares
    s_ij = Vector{T}(undef,size(data.data,1))

    return parDict{T}(β,Σ,ϵ,s_ij)
end
