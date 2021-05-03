import Base.getindex, Base.setindex!, Base.show
# using NLopt
#using ForwardDiff


function calc_shares(full_value::Matrix{T},bitValue::BitArray{2},
    data::ChoiceData,pars::parDict{T},per::Int) where T
    # ind = data._perDict[:,per]
    ind = data._perDict[per]
    opts = data._optDict[per]
    det_value = data.data[ind,:]*pars.β


    calc_Value!(full_value,det_value,pars.ϵ,opts)

    argMax!(bitValue,full_value,opts)

    choices = sum(bitValue[opts,:],dims=2)[:]
    shares = choices/sum(choices)
    ## Add computational error to prevent 0 shares, smooth estimation.
    shares = shares .+ 1e-20
    shares = shares/sum(shares)
    pars.s_ij[ind] = shares
    return nothing
end

function calc_Value!(X::Matrix{T},val::Vector{T},ϵ::Matrix{T},opts::Vector{Int}) where T
    for j in 1:size(X,2), (k,i) in enumerate(opts)
        X[i,j] = ϵ[i,j] + val[k]
    end
    return nothing
end

function argMax!(bitResult::BitArray{2},X::Matrix{T},opts::Vector{Int}) where T
    max_value = maximum(X[opts,:],dims=1)
    for j in 1:size(X,2), i in opts
        bitResult[i,j] = X[i,j]==max_value[j]
    end
    return nothing
end

function individual_shares(data::ChoiceData,pars::parDict{T}) where T

    full_value = Matrix{T}(undef,size(pars.ϵ))
    bitValue = BitArray(undef,size(pars.ϵ))
    # Store Parameters
    people = sort(Int.(keys(data._perDict)))
    for i in people
        calc_shares(full_value,bitValue,data,pars,i)
    end
    return nothing
end
