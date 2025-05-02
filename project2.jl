# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;
using MatrixDepot, SparseArrays

include("problem_def.jl")

# mutable struct IplpSolution
#   x::Vector{Float64} # the solution vector
#   flag::Bool         # a true/false flag indicating convergence or not
#   cs::Vector{Float64} # the objective vector in standard form
#   As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
#   bs::Vector{Float64} # the right hand side (b) in standard form
#   xs::Vector{Float64} # the solution in standard form
#   lam::Vector{Float64} # the solution lambda in standard form
#   s::Vector{Float64} # the solution s in standard form
# end

# mutable struct IplpProblem
#   c::Vector{Float64}
#   A::SparseMatrixCSC{Float64}
#   b::Vector{Float64}
#   lo::Vector{Float64}
#   hi::Vector{Float64}
# end

include("starting_point.jl")
include("presolve_extended.jl")
include("conversions.jl")
include("solve_dev.jl")

include("presolve_simple.jl")
include("starting_point_simple.jl")
include("conversions_simple.jl")
include("solve_simple.jl")

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)
  # key_base = sort(collect(keys(mmmeta)))[1]
  return IplpProblem(
    vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

# argmax alpha {[0, 1] | x + alpha * dx >= 0}
# x + alpha * dx >= 0 -> solve for alpha:
# alpha * dx >= -x -> alpha >= -x / dx (if positive)
# alpha <= -x / dx (if negative)
# Note that we don't care if dx is positive since alpha is [0.0, 1.0]
function calcalpha(x, dx)
    n = length(x)
    alpha = Inf
    for i = 1:n
        if dx[i] < 0
            alpha = min(alpha, -x[i] / dx[i])
        end
    end
    # Ensure alpha is non-negative, guards against cases where all dx[i] >= 0
    # Also handles the Inf case naturally if no dx[i] < 0
    return max(alpha, 0.0) 
end

# Robust IPM Solver Loop (already defined)
# Helper function encapsulating the IPM loop for a *presolved* standard form problem
function solve_presolved_problem(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, c::Vector{Float64}, tol, maxit)
    m, n = size(A)

    # --- Check inputs ---
    if any(!isfinite, A.nzval) # Check non-zero values for sparse matrix
        @error "Matrix A contains non-finite values before starting point calculation!"
        return vec([]), vec([]), vec([]), false # Indicate failure
    end
    if any(!isfinite, b)
        @error "Vector b contains non-finite values before starting point calculation!"
         return vec([]), vec([]), vec([]), false
    end
    if any(!isfinite, c)
        @error "Vector c contains non-finite values before starting point calculation!"
         return vec([]), vec([]), vec([]), false
    end
    # --- End Check ---

    # Get initially feasible x, lambda, s
    x, lambda, s = get_starting_point(A, b, c)

    # --- Check starting point ---
    if any(!isfinite, x) || any(!isfinite, lambda) || any(!isfinite, s)
        @error "Starting point calculation resulted in non-finite values." x=minimum(x) s=minimum(s) lambda=minimum(lambda)
        # Check if initial x or s are non-positive, which get_starting_point should prevent
        if !isempty(x) && minimum(x) <= 0 || !isempty(s) && minimum(s) <= 0 # Added isempty checks
             @warn "Starting point generation resulted in non-positive x or s." minimum_x=isempty(x) ? NaN : minimum(x) minimum_s=isempty(s) ? NaN : minimum(s)
        end
        # Even if non-positive, attempt to proceed? Or fail here? Let's fail early.
         return vec([]), vec([]), vec([]), false
    end
    min_x_start, max_x_start = isempty(x) ? (NaN, NaN) : (minimum(x), maximum(abs.(x)))
    min_s_start, max_s_start = isempty(s) ? (NaN, NaN) : (minimum(s), maximum(abs.(s)))
    @info "Starting point: min(x)=$(min_x_start), max|x|=$(max_x_start), min(s)=$(min_s_start), max|s|=$(max_s_start)"
    # --- End Check ---


    # Implementing Mehrotra's predictor-corrector algorithm (Ch. 10 of Wright)
    for i = 1:maxit
        @debug "Iteration $i" # Changed from @show to @debug for less verbose output unless debugging

        # Check for NaNs/Infs in iterates
        if any(!isfinite, x) || any(!isfinite, lambda) || any(!isfinite, s)
             @warn "Non-finite values detected in iterates at start of iteration $i. Aborting." # x=x lambda=lambda s=s Removed for brevity
             return vec(x), vec(lambda), vec(s), false # Return current state, flag failure
        end

        # Regularization parameters (can be tuned)
        # Increased regularization slightly based on potential KKT issues
        delta_p = 1e-8 # Primal regularization
        delta_d = 1e-8 # Dual regularization

        # Ensure positivity before forming KKT system parts
        x_reg = max.(x, 1e-12) # Prevent zero/negative in diag
        s_reg = max.(s, 1e-12) # Prevent zero/negative in diag


        # Affine direction step (10.1 Wright) - KKT System construction
        # Regularized KKT Matrix M
        M = [
            spdiagm(0 => fill(delta_p, n))  A'              spdiagm(0 => ones(n)); # Use identity instead of Matrix(I, n, n)
            A                              -spdiagm(0 => fill(delta_d, m)) spzeros(m, n);
            spdiagm(0 => s_reg)             spzeros(n, m)   spdiagm(0 => x_reg);
        ]

        # Check condition number if possible/needed (expensive)
        # cond_M = cond(Matrix(M))
        # @debug "Condition number of KKT matrix M: $cond_M"
        # if cond_M > 1e14 # Threshold for ill-conditioning
        #     @warn "KKT matrix is ill-conditioned at iteration $i (cond ≈ $cond_M). Regularizing more?"
        #     # Potentially increase delta_p, delta_d or use iterative refinement?
        # end

        # Factorize KKT system
        local mat # Ensure mat is scoped correctly
        try
            # Use lu for potential singularity, consider LDLt if known positive definite properties hold
            # For general saddle-point systems, lu is often more robust
            mat = lu(M)
            if !issuccess(mat) # Check if LU factorization succeeded numerically
                 @warn "LU factorization of KKT matrix failed numerically at iteration $i. Matrix may be singular or ill-conditioned."
                 # Handle failure: maybe return current state or a specific error status
                 return vec(x), vec(lambda), vec(s), false # Return current state, flag failure
            end
        catch e
             @warn "Error during LU factorization of KKT matrix at iteration $i." exception=(e, catch_backtrace())
             # Handle failure
              return vec(x), vec(lambda), vec(s), false # Return current state, flag failure
        end

        # Affine RHS
        res_p = A' * lambda + s - c # Primal residual (dual feasibility)
        res_d = A * x - b          # Dual residual (primal feasibility)
        res_c = x .* s             # Complementarity residual
        rhs_aff = [ -res_p; -res_d; -res_c ]

        # Solve for affine step direction
        local daff # Scope for error handling
        try
             daff = mat \ rhs_aff # Use \ for potentially slightly better handling of ill-conditioned systems vs \
        catch e
             @warn "Error solving KKT system for affine step at iteration $i." exception=(e, catch_backtrace())
              return vec(x), vec(lambda), vec(s), false # Return current state, flag failure
        end

        # Check for NaN/Inf in affine step
        if any(!isfinite, daff)
             @warn "Affine step contains non-finite values at iteration $i. Aborting." # daff=daff removed for brevity
              return vec(x), vec(lambda), vec(s), false
        end


        dxaff, dlambdaff, dsaff = daff[1:n], daff[n+1:n+m], daff[n+m+1:end] # Use end for robustness

        # Calculate max step lengths (affine)
        alpha_aff_pri = min(1.0, calcalpha(x, dxaff))
        alpha_aff_dual = min(1.0, calcalpha(s, dsaff))

        # Check for very small step sizes
        # if alpha_aff_pri < 1e-12 && alpha_aff_dual < 1e-12
        #      @warn "Affinte step sizes are extremely small at iteration $i. Stalling?"
        # end

        # Calculate mu and centering parameter sigma
        mu = n > 0 ? dot(x, s) / n : 0.0 # Handle n=0 case
        # Prevent division by zero or negative mu
        if mu <= 1e-14 && n > 0 # Add n>0 check
            @debug "Mu is very small or non-positive ($mu) at iteration $i. Checking convergence."
            # If mu is tiny, we might already be converged or very close
             norm_b = 1 + norm(b)
             norm_c = 1 + norm(c)
             primal_feas_norm = norm(res_d) / norm_b
             dual_feas_norm   = norm(res_p) / norm_c
             # current_norm = norm([res_p; res_d; res_c]) / (1 + norm([b;c])) # Use relative norm
             if primal_feas_norm <= tol && dual_feas_norm <= tol # Check only feasibility if gap is tiny
                 @info "Convergence detected due to small mu and feasible residuals at iteration $i."
                 return vec(x), vec(lambda), vec(s), true
             else
                 @warn "Mu is very small ($mu) but residuals (P:$(primal_feas_norm), D:$(dual_feas_norm)) still too large. Potential stall or issue."
                 # Decide whether to proceed with caution or stop? Let's try proceeding one more step carefully.
                 # If mu is non-positive, something is wrong.
                 if mu <= 0 && n > 0
                      @error "Mu became non-positive ($mu) at iteration $i. Aborting."
                      return vec(x), vec(lambda), vec(s), false
                 end
             end
        end

        muaff = n > 0 ? dot(x + alpha_aff_pri * dxaff, s + alpha_aff_dual * dsaff) / n : 0.0
        # Avoid division by zero/very small mu; if mu is tiny, centering isn't the main goal
        sigma = (mu > 1e-12 && n > 0) ? clamp((muaff / mu)^3, 1e-6, 0.5) : 0.1 # Clamp sigma (10.3), ensure it's not too large or small, different default if mu small


        # Centering-Corrector step (Wright 10.7)
        rhs_cc = [
            zeros(n); # No change in feasibility residuals
            zeros(m);
            (n > 0 ? sigma * mu : 0.0) .- (dxaff .* dsaff) # Centering + correction term, handle n=0
        ]

        # Solve for centering-corrector step direction
        local dcc # Scope for error handling
        try
            dcc = mat \ rhs_cc
        catch e
             @warn "Error solving KKT system for centering-corrector step at iteration $i." exception=(e, catch_backtrace())
              return vec(x), vec(lambda), vec(s), false # Return current state, flag failure
        end

        # Check for NaN/Inf in centering-corrector step
         if any(!isfinite, dcc)
             @warn "Centering-corrector step contains non-finite values at iteration $i. Aborting." # dcc=dcc removed
              return vec(x), vec(lambda), vec(s), false
        end

        dxcc, dlambdacc, dscc = dcc[1:n], dcc[n+1:n+m], dcc[n+m+1:end]

        # Combine steps
        dx, dlambda, ds = dxaff + dxcc, dlambdaff + dlambdacc, dsaff + dscc

        # Calculate final step lengths with damping factor (eta)
        eta = 0.99 # Damping factor, common choice is 0.99 or 0.995
        alpha_pri = min(1.0, eta * calcalpha(x, dx))
        alpha_dual = min(1.0, eta * calcalpha(s, ds))

        @debug "Stepsizes: aff_pri=$(alpha_aff_pri), aff_dual=$(alpha_aff_dual), pri=$(alpha_pri), dual=$(alpha_dual)"
        @debug "Mu=$(mu), Mu_aff=$(muaff), Sigma=$(sigma)"


        # --- Step Size Checks ---
        if !isfinite(alpha_pri) || !isfinite(alpha_dual) || alpha_pri < 0 || alpha_dual < 0
            @warn "Calculated step sizes are non-finite or negative at iteration $i. alpha_pri=$alpha_pri, alpha_dual=$alpha_dual. Aborting."
             return vec(x), vec(lambda), vec(s), false
        end
        # Check for excessively large (potentially infinite) steps, indicating unboundedness/infeasibility
         if alpha_pri > 1e50 || alpha_dual > 1e50 # Adjusted threshold
             @warn "Excessively large step size detected (pri=$alpha_pri, dual=$alpha_dual) at iteration $i. Problem might be unbounded or infeasible."
             return vec(x), vec(lambda), vec(s), false # Indicate failure
         end
         # Check for stagnation (very small steps consistently)
         # (Could add a counter for small steps if needed)
         if alpha_pri < 1e-10 && alpha_dual < 1e-10 && n > 0 # Add n>0 check
             @debug "Step sizes are very small at iteration $i. Potential stagnation."
         end
        # --- End Step Size Checks ---

        # Update iterates only if steps are valid
        if n > 0 # Avoid updating empty vectors
            x = x + alpha_pri * dx
            lambda = lambda + alpha_dual * dlambda
            s = s + alpha_dual * ds

            # Ensure positivity after step (due to floating point, eta < 1 might not guarantee)
            x = max.(x, 1e-14) # Project slightly above zero if needed
            s = max.(s, 1e-14)
        end

        # Calculate new residuals and mu
        res_p = A' * lambda + s - c
        res_d = A * x - b
        res_c = x .* s
        mu = n > 0 ? dot(x, s) / n : 0.0


        # --- Check Tolerances ---
        # Use relative norms
        norm_b = 1 + norm(b) # Add 1 to handle b=0 case
        norm_c = 1 + norm(c) # Add 1 to handle c=0 case
        primal_feas_norm = norm(res_d) / norm_b
        dual_feas_norm = norm(res_p) / norm_c
        comp_gap_norm = mu # Use average gap directly

        @debug "Residuals: PrimalFeas=$(primal_feas_norm), DualFeas=$(dual_feas_norm), CompGap=$(comp_gap_norm)"

        # Check convergence conditions
        converged = primal_feas_norm <= tol && dual_feas_norm <= tol && comp_gap_norm <= tol
        
        # Special case: If n=0 (problem fully presolved), check only primal feasibility if m > 0
        if n == 0 && m > 0
             converged = primal_feas_norm <= tol
        elseif n == 0 && m == 0 # Problem is trivially empty
            converged = true
        end

        if converged
            @info "Convergence criteria met at iteration $i."
            return vec(x), vec(lambda), vec(s), true # Converged
        end
        # --- End Check Tolerances ---

        # --- Check for exploding complementarity ---
        if mu > 1e50 && n > 0 # Adjusted threshold
            @warn "Complementarity gap mu is excessively large ($mu) at iteration $i. Problem likely infeasible or unstable."
            return vec(x), vec(lambda), vec(s), false # Indicate failure
        end
         # --- End Check ---

    end # End of main loop

    # Failed to converge within maxit iterations
    @warn "IPM failed to converge within $maxit iterations."
    # Return the final iterates and flag failure
    return vec(x), vec(lambda), vec(s), false
end


# --- Simple Path Attempt ---
function _attempt_simple_path(Problem, tol, maxit)
    @info "Attempting Simple Path..."
    try
        solution = @time simple_iplp(Problem, tol; maxit)

        @show solution.flag

        if solution.flag
            @info "Simple Path Successful."
            return solution
        else
             @info "Simple IPM Failed to Converge."
             return nothing # Indicate simple path failure
        end

    catch e
        @error "Error during Simple Path execution." exception=(e, catch_backtrace())
        return nothing # Indicate simple path failure
    end
end

# --- Extended/Robust Path Attempt ---
function _attempt_extended_path(Problem, tol, maxit)
    @info "Attempting Extended/Robust Path..."
    try
        solution = @time extended_iplp(Problem, tol; maxit)

        return solution

    catch e
        @error "Error during Extended/Robust Path execution." exception=(e, catch_backtrace())
        return nothing # Indicate simple path failure
    end
end


# --- Main iplp Function --- 
function iplp(Problem, tol; maxit=100)
    # Try the simple path first
    simple_result = _attempt_simple_path(Problem, tol, maxit)

    if simple_result !== nothing && simple_result.flag == true # Check if simple path succeeded AND converged
        @info "Using result from Simple Path."
        return simple_result
    else
        if simple_result !== nothing && simple_result.flag == false
            @info "Simple path ran but did not converge."
        elseif simple_result === nothing
             @info "Simple path failed during setup or encountered an error."
        end
        @info "Proceeding to Extended/Robust Path."
        # Simple path failed, try the extended path
        extended_result = _attempt_extended_path(Problem, tol, maxit)
        return extended_result
    end
end

