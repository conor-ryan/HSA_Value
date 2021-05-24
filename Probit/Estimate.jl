using NLopt
#using ForwardDiff
using Statistics
using SharedArrays

function estimate_ng(d::ChoiceData, p0::Vector{Float64};method=:LN_NELDERMEAD)
    # Set up the optimization
    opt = Opt(method, length(p0))
    ftol_rel!(opt,1e-8)
    xtol_rel!(opt,1e-8)
    maxtime!(opt, 500000)

    # unbounded_len = length(data.spec) + (data.opt_num-2)
    # bounded_len = length(p0) - unbounded_len
    # lb = vcat(repeat([-1e5],inner=unbounded_len),repeat([-1.0],inner=bounded_len))
    # ub = vcat(repeat([1e5],inner=unbounded_len),repeat([1.0],inner=bounded_len))
    #
    # lower_bounds!(opt, lb)
    # upper_bounds!(opt, ub)

    ll(x) = log_likelihood(d,x)
    count = 0
    function ll(x, grad)
        count +=1
        x_displ = x[:]
        # println("Iteration $count at $x_displ")
        obj = ll(x)
        # println("Objective equals $obj on iteration $count")

        return obj
    end

    # Set Objective
    max_objective!(opt, ll)

    # Run Optimization
    minf, minx, ret = optimize(opt, p0)
    println("Got $minf at $minx after $count iterations (returned $ret)")

    # Return the object
    return ret, minf, minx
end

# function particle_swarm(N,d::ChoiceData,p0::Vector{Float64};
#     tol_imp=1e-3,tol_dist=1e-3,verbose=true,
#     variances= nothing)
#
#     particles = Matrix{Float64}(undef,length(p0),N)
#     velocities = Matrix{Float64}(undef,length(p0),N)
#     p_best_pos = Matrix{Float64}(undef,length(p0),N)
#     p_best_eval = Vector{Float64}(undef,N)
#
#     func(x) = log_likelihood(d,x)
#     max_eval = [func(p0)]
#     max_pos = Vector{Float64}(undef,length(p0))
#     max_pos[:] = p0[:]
#     ## Parameters
#     ω = 0.8
#     ψ = 1.0
#     ϕ = 1.0
#
#     if variances==nothing
#         init_var = ones(length(p0))
#     else
#         init_var = variances
#     end
#
#     gradient_test = 0
#
#
#     for i in 1:N
#         particles[:,i] = p0[:] + randn(length(p0)).*init_var
#         velocities[:,i] = zeros(length(p0))
#         p_best_pos[:,i] = particles[:,i]
#         p_best_eval[i] = func(particles[:,i])
#         if p_best_eval[i]>max_eval[1]
#             max_eval[1] = p_best_eval[i]
#             max_pos[:] = particles[:,i]
#         end
#     end
#     avg_val = -1e3
#     conv_cnt = 0
#     itr = 0
#     while true
#         itr+=1
#         for i in 1:N
#             for d in 1:length(p0)
#                 rp = rand(1)[1]
#                 rg = rand(1)[1]
#                 velocities[d,i] = ω*velocities[d,i] + ψ*rp*(p_best_pos[d,i]-particles[d,i]) + ϕ*rg*(max_pos[d]-particles[d,i])
#                 particles[d,i]  = particles[d,i] + velocities[d,i]
#             end
#             p_eval = func(particles[:,i])
#             # println("Particle $i at $p_eval")
#             if p_eval>p_best_eval[i]
#                 p_best_eval[i] = p_eval
#                 p_best_pos[:,i] = particles[:,i]
#                 if p_eval>max_eval[1]
#                     max_eval[1] = p_eval
#                     max_pos[:] = particles[:,i]
#                 end
#             end
#         end
#         # Evaluate Convergence
#         MaxDist = abs(maximum(p_best_pos .- max_pos))
#         ImpAv = abs((mean(p_best_eval) - avg_val)/avg_val)
#         avg_val = mean(p_best_eval)
#
#         if (verbose) & (itr%1==0)
#             println("Iteration: $itr")
#             println(p_best_eval)
#             # if (itr%5==0)
#             #     println(p_best_pos)
#             #     println(particles)
#             #     println(velocities)
#             # end
#             println("Best Position: $max_pos")
#             println("Maximum Value: $(max_eval[1])")
#             println("Average Value: $avg_val, $ImpAv")
#         end
#         if itr>5
#             if (ImpAv<tol_imp)
#                 conv_cnt+=1
#             else
#                 conv_cnt = 0
#             end
#         end
#         if (conv_cnt>5)
#             println("Swarm Converged! Local search on best point")
#             res = estimate_ng(d,max_pos)
#             return res
#
#         end
#         #println(particles)
#     end
# end

function evaluate_particles_dist(particles::Matrix{Float64},d::ChoiceData)
    N = size(particles,2)
    p_eval = SharedArray{Float64}(N)
    @sync @distributed for i in 1:N
        y = log_likelihood(d,particles[:,i])
        p_eval[i] = y
    end
    return p_eval
end

function particle_swarm_parallel(p0::Matrix{Float64},d::ChoiceData;
    tol_imp=1e-3,tol_dist=1e-3,verbose=true,itr_max=50,variances=nothing)

    K,N = size(p0)
    particles = Matrix{Float64}(undef,K,N)
    p_best_eval = Vector{Float64}(undef,N)

    ## Parameters
    ω = 0.8
    ψ = 1.0
    ϕ = 1.0

    gradient_test = 0


    for i in 1:N
        particles[:,i] = p0[:,i]
    end

    ## Initialize Best Positions/Values
    p_best_eval[:] .= -1e100
    velocities = zeros(K,N)
    p_best_pos = zeros(K,N)
    max_eval = [-1e100]
    max_pos = zeros(K)


    avg_val = -1e100
    conv_cnt = 0
    itr = 0
    println("Start Estimation")

    while itr<=itr_max
        itr+=1
        #### Evaluate Particles ####
        # evaluate_particles!(p_eval,particles,func)
        p_eval = evaluate_particles_dist(particles,d)

        #### Assign Particle-Specific Best Position and Global Best Position ####
        for i in 1:N
            if (p_eval[i]>p_best_eval[i]) & (p_eval[i]<0)
                p_best_pos[:,i] = particles[:,i]
                p_best_eval[i] = p_eval[i]
                # println("Change: $(p_best_eval[i]),$(p_eval[i])")
                if p_eval[i]>max_eval[1]
                    max_eval[1] = p_eval[i]
                    max_pos[:] = particles[:,i]
                end
            end
        end

        if (itr==1) | (itr==5) | (itr%10==0)
            thresh = median(p_best_eval)
            keep_index = findall(p_best_eval.>=thresh)
            particles = particles[:,keep_index]
            p_eval = p_eval[keep_index]
            p_best_pos = p_best_pos[:,keep_index]
            p_best_eval = p_best_eval[keep_index]
            velocities = velocities[:,keep_index]
            N = size(particles,2)
            println("Reduced swarm to $N")
        end

        #### Update Particle Positions ####
        for i in 1:N
            for d in 1:K
                rp = rand(1)[1]*0.5 + 0.5
                rg = rand(1)[1]*0.5 + 0.5
                velocities[d,i] = ω*velocities[d,i] + ψ*rp*(p_best_pos[d,i]-particles[d,i]) + ϕ*rg*(max_pos[d]-particles[d,i])
                particles[d,i]  = particles[d,i] + velocities[d,i]
            end
        end

        # Evaluate Convergence
        MaxDist = abs(maximum(p_best_pos .- max_pos))
        ImpAv = abs((mean(p_best_eval) - avg_val)/avg_val)
        avg_val = mean(p_best_eval)

        if (verbose)
            println("Iteration: $itr")
            #println(p_best_eval)
            # if (itr%5==0)
            #     println(p_best_pos)
            #     println(particles)
            #     println(velocities)
            # end
            #println("Best Position: $max_pos")
            println("Maximum Value: $(max_eval[1])")
            println("Average Value: $avg_val, $ImpAv")
        end
        if itr>5
            if (ImpAv<tol_imp)
                conv_cnt+=1
            else
                conv_cnt = 0
            end
        end
        if (conv_cnt>5)
            println("Swarm Converged! Local search on best point")
            res = estimate_ng(d,max_pos)
            return res

        end
    end
    println("Iteration Exited. Local search on best point")
    res = estimate_ng(d,max_pos)
    return res
end
