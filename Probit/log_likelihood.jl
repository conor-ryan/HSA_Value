function log_likelihood(data::ChoiceData,pars::parDict{T}) where T
    ll = 0
    for i in 1:length(pars.s_ij)
        logshare = max(log(pars.s_ij[i]),-1000)
        ll += data.Y[i]*logshare
    end
    return ll
end

function log_likelihood(data::ChoiceData,p::Array{T,1}) where T
    pars = parDict(p,data)
    individual_shares(data,pars)
    # if minimum(pars.s_ij)<1e-12
    #     println("Warning: some values are close to 0")
    # end
    ll = log_likelihood(data,pars)
    return ll
end


# Calculate Standard Errors
# Hiyashi, p. 491
function calc_Avar(d::ChoiceData,p::Array{T,1}) where T

    Σ = zeros(length(p),length(p))
    Pop = d.N
    grad_obs = Vector{Float64}(undef,length(p))

    for app in eachperson(d)
        # println(keys(app._perDict))
        f_obj(x) = log_likelihood(app,x)
        grad_obs[:].=0
        FiniteDiff.finite_difference_gradient!(grad_obs,f_obj, p0)
        S_n = grad_obs*grad_obs'
        Σ+= S_n
    end

    Σ = Σ./Pop
    # This last line is correct
    E = eigen(Σ)
    println("Eigenvalues: $(E.values)")
    println("Eigenvectors: $(E.vectors)")
    Asvar = inv(Σ)
    Beta_var = Asvar./d.N
    return Beta_var
end
