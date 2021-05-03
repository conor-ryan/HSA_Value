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
