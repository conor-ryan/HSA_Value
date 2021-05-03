using NLopt

function maximize_value_function(State_variable1,Statevariable2, p0::Vector{Float64};method=:LN_NELDERMEAD)
    # Set up the optimization
    # Non-gradient methods :LN_NELDERMEAD, LN_BOBYQA
    opt = Opt(method, length(p0))
    ftol_rel!(opt,1e-8)
    xtol_rel!(opt,1e-8)
    maxtime!(opt, 500000)

    # lb = vcat(repeat([-1e5],inner=unbounded_len),repeat([-1.0],inner=bounded_len))
    # ub = vcat(repeat([1e5],inner=unbounded_len),repeat([1.0],inner=bounded_len))
    # lower_bounds!(opt, lb)
    # upper_bounds!(opt, ub)

    func(x) = value_function(x,State_variable1,State_variable2)
    count = 0
    function func(x, grad)
        count +=1
        # Verbose output

        obj = func(x)
        if count%10==0 # Every ten
            println("Iteration $count at $x")
            println("Objective equals $obj on iteration $count")
        end

        return obj
    end

    # Set Objective
    max_objective!(opt, func)

    # Run Optimization
    value, estimate, flag = optimize(opt, p0)
    println("Got $value at $estimate after $count iterations (returned $flag)")

    # Return the object
    return ret, minf, minx
end

function systemOfEquations(X,n,state_variable1,state_variable2)
    # X is the vector of unknowns
    eq1  = equation1(X,n,state_variable1,state_variable2) # Specified to equal 0.
    eq2  = equation2(X,n,state_variable1,state_variable2) # Specified to equal 0.
    eq3  = equation3(X,n,state_variable1,state_variable2) # Specified to equal 0.

    error = eq1^2 + e2^2 + eq3^2
    return error
end

function SolveSystemOfEquations(n,state_variable1,state_variable2)
    opt = Opt(:LN_NELDERMEAD, length(p0))
    ftol_rel!(opt,1e-10)
    xtol_rel!(opt,1e-8)
    maxtime!(opt, 500000) # Max Iterations

    # lb = vcat(repeat([-1e5],inner=unbounded_len),repeat([-1.0],inner=bounded_len))
    # ub = vcat(repeat([1e5],inner=unbounded_len),repeat([1.0],inner=bounded_len))
    # lower_bounds!(opt, lb)
    # upper_bounds!(opt, ub)

    func(x) = systemOfEquations(x,n,state_variable1,state_variable2)
    count = 0
    function func(x, grad)
        obj = func(x)
        return obj
    end
    # Set Objective
    min_objective!(opt, func)

    # Run Optimization
    value, estimate, flag = optimize(opt, p0)
    return estimate
end


function value_function(n,state_variable1,state_variable2)
    X = SolveSystemOfEquations(n,state_variable1,state_variable2)
    val = value_function_actual(n,X,state_variable1,state_variable2)
    return val
end
