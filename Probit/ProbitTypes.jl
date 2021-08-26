
########
# Parameter Structure
#########


mutable struct parDict{T}
    # Parameter Vector
    β::Vector{T}

    # Full Covariance Matrix
    Ω::Matrix{T}

    # Idiosyncratic Draws
    # ϵ::Matrix{T}

    # Predicted Shares (ij pairs)
    s_ij::Vector{T}

    # Log likelihood of the observation
    ll_i::Vector{T}
end

#### Data Structure ####
abstract type
    ModelData
end

struct ChoiceData <: ModelData
    # Matrix of the data (pre-sorted)
    data::Matrix{Float64}

    # Specification
    spec::Vector{Symbol}

    # Vector of the outcome (pre-sorted)
    Y::Vector{Float64}

    # Idiosyncratic Draws
    draws::Matrix{Float64}

    # Data Dimensions
    opt_num::Int
    N::Int

    # Individual Index
    _perDict::Dict{Int,Array{Int,1}}
    _optDict::Dict{Int,Array{Int,1}}
end
