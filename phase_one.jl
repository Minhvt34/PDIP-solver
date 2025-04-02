"""
Phase I to check the feasibility
"""
function is_nonnegative(b::AbstractVector{<:Real})
    return all(x -> x >= 0, b)
end

function phaseone(A, b)

    if is_nonnegative(b)
        println("All components of b are nonnegative.")
    else
        println("Some components of b are negative.")
    end

    m,n = size(A)
    A = [A Matrix{Float64}(I,m,m)]
    c = [zeros(Float64, n);ones(Float64, m)]
    x1,lambda1,s1,flag,iter = solve_standardlp(A,b,c)
    @show dot(c, x1)

    # Set an adaptive tolerance: for instance, proportional to norm(b)
    tol_phaseone = 1e-8 * (1 + norm(b))

    return abs(dot(c, x1)) > tol_phaseone
end
