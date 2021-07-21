function calc_elas(d::ChoiceData,p::Array{T,1}) where T



    Σ = zeros(length(p),length(p))
    Pop = d.N
    grad_obs = Vector{Float64}(undef,length(p))

    for app in eachperson(d)
        # println(keys(app._perDict))
        f_obj(x) = log_likelihood(app,x)
        grad_obs[:].=0
        FiniteDiff.finite_difference_gradient!(grad_obs,f_obj, p)
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
    # return Σ
end

function calc_deriv(price_indices::Vector{Int},d::ChoiceData,p_est::Vector{Float64})
    X_adj = zeros(d.opt_num,size(d.data,2))
    f_obj(x) = deriv_func(x,price_indices,d,p_est,X_adj)
    p = zeros(d.opt_num)
    grad = Matrix{Float64}(undef,size(d.data,1),1)
    grad = FiniteDiff.finite_difference_jacobian(f_obj, p)
    return grad./1e7
end

function deriv_func(p::Vector{Float64},price_indices::Vector{Int},d::ChoiceData,
    p_est::Vector{Float64},X_adj::Matrix{Float64})
    X_adj[:].=0.0
    for i in price_indices, k in 1:length(p)
        X_adj[k,i]=p[k]*1e7
    end
    println(X_adj)
    println(p)
    pars = parDict(p_est,d)
    individual_shares_deriv(d,pars,X_adj)
    return pars.s_ij
end

function calc_shares_deriv(full_value::Matrix{T},bitValue::BitArray{2},
    data::ChoiceData,pars::parDict{T},per::Int,X_adj::Matrix{T}) where T
    # ind = data._perDict[:,per]
    ind = data._perDict[per]
    opts = data._optDict[per]
    x_add = (data.data[ind,:].>0).*X_adj[opts,:]
    det_value = (data.data[ind,:]+x_add)*pars.β

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

function individual_shares_deriv(data::ChoiceData,pars::parDict{T},X_adj::Matrix{T}) where T

    full_value = Matrix{T}(undef,size(pars.ϵ))
    bitValue = BitArray(undef,size(pars.ϵ))
    # Store Parameters
    people = sort(Int.(keys(data._perDict)))
    for i in people
        calc_shares_deriv(full_value,bitValue,data,pars,i,X_adj)
    end
    return nothing
end

function elasticity(price_indices::Vector{Int},d::ChoiceData,p_est::Vector{Float64},df::DataFrame,pars::ParDict{Float64})
    grad = calc_deriv(price_indices,d,p_est)
    elas_matrix = zeros(d.opt_num,d.opt_num)
    count_matrix = zeros(d.opt_num,d.opt_num)
    people = sort(Int.(keys(d._perDict)))
    adjuster = -1000 ./exp.(df[!,:logprem]./1000)
    price = df[!,:adjprem]
    share = pars.s_ij

    people = unique(df[!,:studyid][(df[!,:logprice_age_40_60].==0.0) .& (df[!,:logprice_age_60plus].==0.0)])
    # Make symmetric the demand derivatives
    for i in people
        ind = data._perDict[i]
        opts = data._optDict[i]
        grad_temp = grad[ind,opts]
        for i in 1:length(opts)
            for j in 1:(i-1)
                new = (grad_temp[i,j] + grad_temp[j,i])/2
                grad_temp[i,j] = new
                grad_temp[j,i] = new
            end
        end
        grad[ind,opts] = grad_temp[:,:]
    end

    for i in people
        ind = data._perDict[i]
        opts = data._optDict[i]
        for opt in opts
            adj = adjuster[ind[opts.==opt]][1]
            prem_ind = price[ind[opts.==opt]][1]
            for (k_short,k_long) in enumerate(ind)
                elas_matrix[opts[k_short],opt] = elas_matrix[opts[k_short],opt] .+ grad[k_long,opt]*adj*prem_ind
                count_matrix[opts[k_short],opt] = count_matrix[opts[k_short],opt] .+ share[k_long]
            end
        end
    end
    elas_matrix = elas_matrix./count_matrix



end
