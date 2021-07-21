using DataFrames
using Distributions
import DataFrames.subset

function ChoiceData(data_choice::DataFrame;
    est_draws = 1000,
    person=[:studyid],
    product=[:newpid],
    spec=[:adjprem],
    choice=[:yvar],
    halton=true)

    # Get the size of the data
    n, k = size(data_choice)

    # Convert everything to an array once for performance
    i = Array(data_choice[!,person])
    j = Array(data_choice[!,product])
    X = Array(data_choice[!,spec])
    y = Array(data_choice[!,choice])

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

    if halton
        draws = MVHaltonNormal(est_draws,opt_num-1,scrambled=false)
    else
        Ω = Normal()
        draws = rand(Ω,(est_draws,opt_num-1))
        # draws = MVNormal(est_draws,opt_num-1,scrambled=false)
    end
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


########## People Iterator ###############
# Define an Iterator Type
mutable struct PersonIterator
    data
    id
end

# Construct an iterator to loop over people
function eachperson(m::ChoiceData)
    #ids = m._personIDs
    ids = sort(Int.(keys(m._perDict)))
    return PersonIterator(m, ids)
end

# Quickly Generate Subsets on People
function subset(d::T, id,idx) where T<:ModelData

    data = d.data[idx,:]
    Y = d.Y[idx]
    _perDict = Dict(id=>1:length(idx))
    _optDict = Dict(id=>d._optDict[id])
    # Don't subset any other fields for now...
    return T(data,d.spec,Y,d.draws,
            d.opt_num,1,_perDict,_optDict)
end


function Base.iterate(iter::PersonIterator, state=1)

    if state> length(iter.id)
        return nothing
    end

    # Get the current market
    id = iter.id[state]

    # Find which indices to use
    idx = iter.data._perDict[id]

    # Subset the data to just look at the current market
    submod = subset(iter.data,id,idx)

    return (submod, state + 1)
end
