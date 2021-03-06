function log_likelihood_byshare(data::ChoiceData,pars::parDict{T}) where T
    ll = 0
    for i in 1:length(pars.s_ij)
        logshare = max(log(pars.s_ij[i]),-1000)
        ll += data.Y[i]*logshare
    end
    return ll
end

function log_likelihood(data::ChoiceData,pars::parDict{T}) where T
    return sum(pars.ll_i)
end
function log_likelihood_MC(data::ChoiceData,p::Array{T,1}) where T
    pars = parDict(p,data)
    individual_shares(data,pars)
    ll = log_likelihood_byshare(data,pars)
    return ll

end


function log_likelihood(data::ChoiceData,p::Array{T,1}) where T
    pars = parDict(p,data)

    individual_ll_GHK(data,pars)
    return  sum(pars.ll_i)

    # individual_shares(data,pars)
    # ll = log_likelihood_byshare(data,pars)
    # return ll

    # if minimum(pars.s_ij)<1e-12
    #     println("Warning: some values are close to 0")
    # end
    # ll = log_likelihood(data,pars)

end

function avar_obj_func(x::Vector{Float64},app::ChoiceData,p_est::Vector{Float64})
    p = p_est .+ x
    return log_likelihood(app,p)
end


# Calculate Standard Errors
# Hiyashi, p. 491
function calc_Avar(d::ChoiceData,p::Array{T,1}) where T

    Σ = zeros(length(p),length(p))
    Pop = d.N
    grad_obs = Vector{Float64}(undef,length(p))
    dev = zeros(length(p))
    for app in eachperson(d)
        # println(keys(app._perDict))
        f_obj(x) = avar_obj_func(x,app,p)
        grad_obs[:].=0
        FiniteDiff.finite_difference_gradient!(grad_obs,f_obj,dev)
        S_n = grad_obs*grad_obs'
        Σ+= S_n
    end

    Σ = Σ./Pop
    # # This last line is correct
    # E = eigen(Σ)
    # println("Eigenvalues: $(E.values)")
    # println("Eigenvectors: $(E.vectors)")
    Asvar = inv(Σ)
    Beta_var = Asvar./d.N
    return Beta_var
    # return Σ
end
