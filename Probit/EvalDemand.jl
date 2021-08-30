import Base.getindex, Base.setindex!, Base.show
using LinearAlgebra
# using NLopt
#using ForwardDiff


# Follow Train Discrete Choice Methods with Simulation (page 139)
function calc_ll_GHK(cum_probs::Matrix{T},
    data::ChoiceData,pars::parDict{T},per::Int,i::Int) where T

    # ind = data._perDict[:,per]
    ind = data._perDict[per]
    opts = data._optDict[per]
    det_value = data.data[ind,:]*pars.β

    choice_ind = findall(data.Y[ind].==1.0)
    choice = opts[data.Y[ind].==1.0]
    opts_tilde = opts[data.Y[ind].==0.0]
    if length(choice)>1
        error("Too many choices for individual $per")
    else
        choice = choice[1]
        choice_ind = choice_ind[1]
    end

    V_tilde =  det_value .- det_value[choice_ind]
    V_tilde = V_tilde[V_tilde.!=0.0]

    A = diagm(-ones(data.opt_num))
    A[:,choice].=1.0
    Ω_tilde = A*pars.Ω*A'
    Ω_tilde = Ω_tilde[opts_tilde,opts_tilde]
    check_symmetry = Ω_tilde - Ω_tilde'
    if maximum(check_symmetry)>1e-12
        error("Symmetry error for individual $per")
    end
    Ω_tilde = Hermitian(Ω_tilde)
    if !isposdef(Ω_tilde)
        println("Matrix not positive definite for choice $choice and person $per")
    end
    decomp = cholesky(Ω_tilde)
    L = decomp.L


    prob_est = calc_share_GHK(V_tilde,opts_tilde,L,cum_probs,data)


    pars.ll_i[i] = log(prob_est)
    return nothing
end

function calc_share_GHK(V_tilde::Vector{Float64},opts_tilde::Vector{Int},
                        L::LowerTriangular{Float64, Matrix{Float64}},
                        cum_probs::Matrix{Float64},data::ChoiceData)

    cum_probs[:].=0.0
    η = Vector{Float64}(undef,length(V_tilde))
    σ_tilde = diag(L)
    L_η = L - diagm(σ_tilde)
    prob_est = 0.0
    for n in 1:size(data.draws,2)
        # μ= data.draws[:,n]
        η[:].=0.0
        prob_n = 1.0
        for j in 1:length(V_tilde)
            dot_product = 0.0
            for k in 1:(j-1)
                dot_product += L_η[j,k]*η[k]
            end
            z1 = -(V_tilde[j] + dot_product)/σ_tilde[j]
            if (z1>8)
                z1 = 8
            elseif (z1<-8)
                z1 = -8
            end
            p = cdf(Normal(),z1)
            @inbounds z2 = data.draws[opts_tilde[j],n]*p
            η_new = norminvcdf(z2)

            η[j] = norminvcdf(z2)
            prob_n = prob_n*p
        end
        if isnan(prob_n)
            println("n: $n, $(isnan(prob_n))")
        end
        if isnan(prob_est)
            println("n: $n, $(isnan(prob_n))")
            break
        end
        prob_est += prob_n
    end
    prob_est = prob_est/size(data.draws,2)


    # prob_est = mean(prod(cum_probs[1:length(V_tilde),:],dims=1))
    return prob_est
end

function calc_shares(ϵ::Matrix{Float64},full_value::Matrix{T},bitValue::BitArray{2},
    data::ChoiceData,pars::parDict{T},per::Int) where T
    # ind = data._perDict[:,per]
    ind = data._perDict[per]
    opts = data._optDict[per]
    det_value = data.data[ind,:]*pars.β


    calc_Value!(full_value,det_value,ϵ,opts)

    argMax!(bitValue,full_value,opts)

    choices = sum(bitValue[opts,:],dims=2)[:]
    shares = choices/sum(choices)
    ## Add computational error to prevent 0 shares, smooth estimation.
    if size(ϵ,2)>1
        shares = shares .+ 1e-20
    end
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

function individual_ll_GHK(data::ChoiceData,pars::parDict{T}) where T

    # full_value = Matrix{T}(undef,size(pars.ϵ))
    # bitValue = BitArray(undef,size(pars.ϵ))
    cum_probs = Matrix{T}(undef,size(data.draws))
    # Store Parameters
    people = sort(Int.(keys(data._perDict)))
    for (i,per) in enumerate(people)
        # calc_shares(full_value,bitValue,data,pars,i)
        calc_ll_GHK(cum_probs,data,pars,per,i)
    end
    return nothing
end

function simulate_shares(data::ChoiceData,pars::parDict{T}) where T
    Σ = pars.Ω[2:data.opt_num,2:data.opt_num]
    decomp = cholesky(Σ)
    L = decomp.L

    full_value = Matrix{T}(undef,data.opt_num,size(data.draws,2))
    bitValue = BitArray(undef,data.opt_num,size(data.draws,2))
    # Store Parameters
    people = sort(Int.(keys(data._perDict)))
    for i in people
        draws = rand(Normal(),(size(data.draws,2),data.opt_num-1))
        ϵ = Matrix{T}(undef,data.opt_num,size(data.draws,2))
        ϵ[1,:].=0.0
        ϵ[2:data.opt_num,:] = permutedims(draws*L,(2,1))
        calc_shares(ϵ,full_value,bitValue,data,pars,i)
    end
    return nothing
end


function individual_shares(data::ChoiceData,pars::parDict{T}) where T
    Σ = pars.Ω[2:data.opt_num,2:data.opt_num]
    decomp = cholesky(Σ)
    L = decomp.L

    ϵ = Matrix{T}(undef,data.opt_num,size(data.draws,2))
    ϵ[1,:].=0.0

    norm_draws = norminvcdf.(data.draws[2:data.opt_num,:])
    ϵ[2:data.opt_num,:] = L*norm_draws


    full_value = Matrix{T}(undef,size(ϵ))
    bitValue = BitArray(undef,size(ϵ))
    # Store Parameters
    people = sort(Int.(keys(data._perDict)))
    for i in people
        calc_shares(ϵ,full_value,bitValue,data,pars,i)
    end
    return nothing
end
