# Starting point algorithm, from pg. 410 of Nocedal & Wright.
# Related to pg. 224 of Wright.
# Starting point must satisfy KKT conditions:
# A^T\lambda + s = c
# Ax = b
# x >= 0, s >= 0
# For numerical stability, x and s small is good!
# So:
# Solve problem min ||x||_2^2 s.t. Ax = b
# Solve problem min ||s||_2^2 s.t. A^T\lambda + s = c
# (14.40 Nocedal & Wright):
# x = A^T(AA^T)^-1b
# lambda = (AA^T)-1Ac
# s = c - A^Tlambda

using SparseArrays, LinearAlgebra, SuiteSparse.CHOLMOD


function get_starting_point(A, b, c)
    m, n = size(A)
    try
        # Solve AA * y = b  with AA = A Aᵀ (SPD) ---------------------------
        # Regularization parameter
        # Increased regularization
        delta = sqrt(eps(Float64)) * 100 # Start with a larger value than eps()
        # Add regularization: AA = A*A' + delta*I
        AA = A * A'
        
        # Factorize the regularized matrix
        local F # Ensure F is scoped for potential errors
        try
            F = cholesky(AA + delta * I) # Add regularization here
            if !issuccess(F)
                @warn "Cholesky factorization failed even with regularization (delta=$delta). Matrix A*A' might be severely ill-conditioned."
                throw(ErrorException("Cholesky factorization failed")) # Force fallback
            end
        catch e
            @warn "Cholesky factorization failed with delta=$delta." exception=e
            throw(ErrorException("Cholesky factorization failed")) # Force fallback
        end

        y  = F \ b
        if any(!isfinite, y)
            @warn "NaN or Inf detected in intermediate 'y' (from F \\ b)" y=y
            throw(ErrorException("Solve F \\ b resulted in non-finite values")) # Force fallback
        end
        x  = A' * y              # primal that satisfies Ax = b
        if any(!isfinite, x)
            @warn "NaN or Inf detected in initial x" x=x
            throw(ErrorException("Calculation A' * y resulted in non-finite values")) # Force fallback
        end

        Ac = A * c
        λ  = F \ Ac        # dual y that makes rd small
        if any(!isfinite, λ)
            @warn "NaN or Inf detected in initial λ (from F (A*c))" lambda=λ
            throw(ErrorException("Solve F (A*c) resulted in non-finite values")) # Force fallback
        end
        s  = c - A' * λ
        if any(!isfinite, s)
            @warn "NaN or Inf detected in initial s" s=s
             throw(ErrorException("Calculation c - A' * λ resulted in non-finite values")) # Force fallback
        end

        # Shift to interior -----------------------------------------------
        # Ensure x and s are finite before proceeding
        if !all(isfinite, x) || !all(isfinite, s)
             error("Initial x or s contains non-finite values before adjustment. Cannot proceed.")
        end

        dx = max(-1.5 * minimum(x), 0.0)
        ds = max(-1.5 * minimum(s), 0.0)
        x .+= dx
        s .+= ds
        τ   = 0.5 * dot(x, s)
        sum_s = sum(s)
        sum_x = sum(x)
        if sum_s == 0.0
            @warn "Sum of s is zero during starting point adjustment."
            # Handle division by zero, e.g., by adding a small epsilon or skipping adjustment
            # For now, let's just add a small value to avoid NaN/Inf, though this might not be robust
             sum_s = eps(Float64) 
        end
        if sum_x == 0.0
             @warn "Sum of x is zero during starting point adjustment."
             sum_x = eps(Float64)
        end
        x  .+= τ / sum_s
        s  .+= τ / sum_x

        return Vector{Float64}(x), Vector{Float64}(λ), Vector{Float64}(s)
    catch err
        @warn "get_starting_point calculation failed, falling back to trivial init." exception=err
        # Fallback: Use ones for x and s, and a small value for lambda
        fallback_lambda = fill(sqrt(eps(Float64)), m) # Use sqrt(eps) instead of zero
        return ones(n), fallback_lambda, ones(n)
    end
end

function get_starting_point_bk(A, b, c)
    #A = A
    AA = A * A'

    x = A' * ((AA) \ b)
    lambda = (AA) \ (A * c)
    s = c - A' * lambda

    # Adjustment step to ensure x, s have no nonpositive attrs
    dx = max(-3/2 * minimum(x), 0)
    ds = max(-3/2 * minimum(s), 0)
    xhat = x .+ dx
    shat = s .+ ds
    xts = 1/2 * dot(xhat, shat)
    dhatx = xts / (sum(shat))
    dhats = xts / (sum(xhat))

    return xhat .+ dhatx, lambda, shat .+ dhats
end