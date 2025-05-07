# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;
using SparseArrays; # Add SparseArrays for spdiagm

include("starting_point.jl")
include("presolve_extended.jl")
include("conversions.jl")
include("problem_def.jl")


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
    return max(alpha, 0.0)
end

function extended_iplp(Problem, tol; maxit=100)
    # Convert to standard form
    A_std, b_std, c_std, free, bounded_below, bounded_above, bounded = tostandard(Problem)
    A = A_std # Use temporary variables for presolve input
    b = b_std
    c = c_std

    # Presolve step - modify A,b,c to be nicer
    @show size(A)
    m_std, n_std = size(A)

    standard_problem = IplpProblem(c, A, b, zeros(n_std), fill(Inf, n_std)) # Use original std form for revProb context

    presolve_result = presolve(standard_problem)
    status = presolve_result[1]

    if status == :Success
        std_presolved, ind0c_std, dup_main_c_std, ind_dup_c_std, ind_fix_c_std, fix_vals_std, dual_lb, dual_ub, obj_offset, free_singleton_subs, final_col_indices_std = presolve_result[2:end]
        @printf("Original standard form size: (%d, %d), After presolve: (%d, %d) \n",
                m_std, n_std, size(std_presolved.A)...)

        # Directly use presolved results for the IPM
        A = std_presolved.A
        b = std_presolved.b
        c = std_presolved.c
        m_ps, n_ps = size(A) # Dimensions after presolve

        # --- Check Presolve obj_offset ---
        if !isfinite(obj_offset)
            @error "Presolve returned non-finite obj_offset!" obj_offset=obj_offset
        end
        # --- End Check ---

        # --- Check b immediately after presolve ---
        if any(!isfinite, b)
            @error "Vector b contains non-finite values immediately after presolve!" b=b
            # return IplpSolution(...) # Or some error state
        end
        # --- End Check ---

        # --- Rank Check ---
        rank_A = rank(Matrix(A)) # Convert to dense for rank calculation
        @printf("Rank of presolved A: %d (Dimensions: %d x %d)\n", rank_A, m_ps, n_ps)
        if rank_A < m_ps
             @warn "Presolved matrix A has linearly dependent rows (rank $rank_A < $m_ps). KKT matrix will be singular."
             # Optionally: Decide whether to proceed or error out
             # return IplpSolution(..., :PresolveRankDeficient) # Adjust return signature
        end
        # --- End Rank Check ---

    else # Presolve failed or returned non-:Success status
        @error "Presolve failed or returned status: $status. Cannot proceed."
        # Return with original standard form problem data and indicate failure
        return IplpSolution(vec([]), false, vec(c), A, vec(b), vec([]), vec([]), vec([])) # Adjust signature if needed
    end

    # Now A, b, c are the (potentially scaled) presolved matrices
    # m, n now refer to the dimensions of the problem entering the IPM solver
    m, n = size(A) # Should be m_ps, n_ps
    @show size(A)

    # --- Check inputs to get_starting_point ---
    if any(!isfinite, A.nzval)
        @error "Matrix A contains non-finite values before starting point calculation!"
    end
    if any(!isfinite, b)
        @error "Vector b contains non-finite values before starting point calculation!"
    end
    if any(!isfinite, c)
        @error "Vector c contains non-finite values before starting point calculation!"
    end
    # --- End Check ---

    # Get initially feasible x, lambda, s for the (scaled) presolved problem
    x, lambda, s = get_starting_point(A, b, c)

    # @show minimum(x)
    # @show minimum(s)
    # @show maximum(abs.(x))
    # @show maximum(abs.(s))

    # -- Adaptive Regularization Parameters --
    delta_p = 1e-8       # Initial primal regularization
    delta_d = 1e-8       # Initial dual regularization
    min_delta = 1e-10    # Minimum regularization
    max_delta = 1e-2     # Maximum regularization
    delta_increase_factor = 100.0 # Factor to increase delta on LU failure
    delta_decrease_factor = 2.0   # Factor to decrease delta on LU success
    # -------------------------------------

    tmp_mu = 1.0e-2   
    # Implementing Mehrotra's predictor-corrector algorithm (Ch. 10 of Wright)
    for i = 1:maxit
        # -- Adaptive Factorization Attempt --
        factorization_success = false
        local mat # Ensure mat is accessible after the loop
        current_delta_p = delta_p # Use temporary deltas for potential retries
        current_delta_d = delta_d
        max_factorization_retries = 3

        for retry_attempt = 0:max_factorization_retries
            # Construct KKT System with current regularization values
            M = [
                spdiagm(0 => fill(current_delta_p, n))  A'              Matrix(I, n, n);
                A                                     -spdiagm(0 => fill(current_delta_d, m)) spzeros(m, n);
                spdiagm(0 => s)                        spzeros(n, m)  spdiagm(0 => x);
            ]

            try
                mat = lu(M)
                if issuccess(mat)
                    factorization_success = true
                    # Success: Slightly decrease base deltas for *next* outer iteration
                    delta_p = max(min_delta, delta_p / delta_decrease_factor)
                    delta_d = max(min_delta, delta_d / delta_decrease_factor)
                    if retry_attempt > 0
                         @info "LU factorization succeeded at iteration $i after retry $retry_attempt with delta_p=$current_delta_p, delta_d=$current_delta_d"
                    end
                    break # Exit retry loop
                else
                    # LU failed numerically (issuccess=false)
                    @warn "LU factorization failed numerically at iteration $i (retry $retry_attempt). Increasing regularization."
                    # Increase deltas for the next retry attempt (or potentially next outer iter)
                    current_delta_p = min(max_delta, current_delta_p * delta_increase_factor)
                    current_delta_d = min(max_delta, current_delta_d * delta_increase_factor)
                    # Update base deltas immediately for next outer iteration regardless of retry success
                    delta_p = current_delta_p
                    delta_d = current_delta_d
                end
            catch e
                 @warn "Error during LU factorization at iteration $i (retry $retry_attempt)." exception=e
                 # Treat catch block error as instability, increase deltas
                 current_delta_p = min(max_delta, current_delta_p * delta_increase_factor)
                 current_delta_d = min(max_delta, current_delta_d * delta_increase_factor)
                 delta_p = current_delta_p
                 delta_d = current_delta_d
                 # Consider if we should break or continue retrying after a catch?
                 # Let's continue retrying for now.
            end

            if !factorization_success && retry_attempt == max_factorization_retries
                 @warn "LU factorization failed permanently at iteration $i after $max_factorization_retries retries."
                 # Return failure status (using final x, lambda, s from previous iteration)
                  return IplpSolution(vec([]), false, vec(c), A, vec(b), vec(x), vec(lambda), vec(s))
            end
        end # End factorization retry loop
        # -- End Adaptive Factorization Attempt --

        # If we reach here, factorization_success must be true

        # RHS uses presolved c and b
        rhs_aff = [
            - (A' * lambda + s - c);
            - (A * x - b);
            - (x .* s)
        ]

        # Solve for steps using the successful factorization `mat`
        daff = mat \ rhs_aff
        dxaff, dlambdaff, dsaff = daff[1:n], daff[n+1:n+m], daff[n+m+1:n+m+n]

        # Calculate alpha
        alpha_aff_pri = min(1.0, calcalpha(x, dxaff))
        alpha_aff_dual = min(1.0, calcalpha(s, dsaff))

        # Mu calculation uses presolved x, s
        mu = dot(x, s) / n
        muaff = (x + alpha_aff_pri * dxaff)' * (s + alpha_aff_dual * dsaff) / n
        sigma = isapprox(mu, 0.0) ? 0.0 : (muaff / mu)^3 # Avoid division by zero

        tmp_mu = mu < tmp_mu ? mu : tmp_mu
        # Centering-Corrector step
        rhs_cc = [
            zeros(n, 1);
            zeros(m, 1);
            sigma * mu .- (dxaff .* dsaff)
        ]
        dcc = mat \ rhs_cc # Use the same factorization
        dxcc, dlambdacc, dscc = dcc[1:n], dcc[n+1:n+m], dcc[n+m+1:n+m+n]

        dx, dlambda, ds = dxaff + dxcc, dlambdaff + dlambdacc, dsaff + dscc

        # Calculate step lengths
        alpha_pri = min(0.99 * calcalpha(x, dx), 1.0)
        alpha_dual = min(0.99 * calcalpha(s, ds), 1.0)

        # Check for issues
        current_mu = dot(x,s) # Use dot product before update
        if !isfinite(current_mu) || current_mu > 1e300 # Check before potentially large mu
             @warn "Complementarity product is non-finite or excessively large before update at iteration $i." mu=current_mu
             # Return current iterates
             @warn "Complementarity is too large at iteration $i. Returning infeasible solution."
             return IplpSolution(vec([]), false, vec(c), A, vec(b), vec(x), vec(lambda), vec(s))
        end

        if !isfinite(alpha_pri) || !isfinite(alpha_dual) || alpha_pri > 1e300 || alpha_dual > 1e300 # Check for large/infinite alpha
             @warn "Step length alpha is non-finite or excessively large at iteration $i." alpha_pri=alpha_pri alpha_dual=alpha_dual
             # Return current iterates
             @warn "Alpha is too large at iteration $i. Returning infeasible solution."
             return IplpSolution(vec([]), false, vec(c), A, vec(b), vec(x), vec(lambda), vec(s))
        end

        # Update x, lambda, s
        x = x + alpha_pri * dx
        lambda = lambda + alpha_dual * dlambda
        s = s + alpha_dual * ds

        # Check termination criteria (using scaled problem norms)
        # Primal feasibility: norm(A*x - b)
        # Dual feasibility: norm(A'*lambda + s - c)
        # Complementarity: dot(x,s)/n
        # Combine into one norm check
        primal_res = A * x - b
        dual_res = A' * lambda + s - c
        comp_res = x .* s
        mu = dot(x, s) / n # Recompute mu after update

        # Use relative tolerances based on scaled norms
        norm_b = norm(b, Inf)
        norm_c = norm(c, Inf)
        # Use 1.0 if norms are zero to avoid division by zero
        rel_tol_b = max(1.0, norm_b)
        rel_tol_c = max(1.0, norm_c)
        # Combine norms, potentially weighted
        # Using infinity norms for feasibility checks often recommended
        primal_feas = norm(primal_res, Inf) / rel_tol_b
        dual_feas = norm(dual_res, Inf) / rel_tol_c

        # if mu <= tol && primal_feas <= tol && dual_feas <= tol
        if mu <= tol && norm([A'*lambda + s - c; A*x - b; x.*s])/norm([b;c]) <= tol # && primal_feas <= tol && dual_feas <= tol
            @info "Converged at iteration $i."

            # Solution vectors x, lambda, s are the final iterates from the loop
            # They correspond to the UNscaled presolved problem

            # --- Un-Presolve ---
            # Use standard_problem context which holds original standard form dimensions/structure info for revProb
            # Pass the primal solution vector x from the IPM loop
            x_unpresolved = revProb(standard_problem, ind0c_std, dup_main_c_std, ind_dup_c_std, ind_fix_c_std, fix_vals_std, free_singleton_subs, final_col_indices_std, x)

            # --- Check revProb result ---
            if any(!isfinite, x_unpresolved)
                @warn "revProb returned non-finite values in x_unpresolved!" x_unpresolved=x_unpresolved
            end
            # --- End Check ---

            # --- Convert back from standard form ---
            orig_x = fromstandard(Problem, x_unpresolved, free, bounded_below, bounded_above, bounded)

            # --- Check fromstandard result ---
             if any(!isfinite, orig_x)
                @warn "fromstandard returned non-finite values in orig_x!" orig_x=orig_x x_unpresolved=x_unpresolved
            end
            # --- End Check ---

            @show i
            # --- Calculate and check final objective ---
            max_abs_orig_x = isempty(orig_x) ? 0.0 : maximum(abs.(orig_x)) # Handle empty case
            @info "Calculating final objective: dot(Problem.c, orig_x). Max abs(orig_x) = $(max_abs_orig_x)"

            final_obj = isempty(orig_x) ? 0.0 : dot(Problem.c, orig_x) # Use original cost vector
            if !isfinite(final_obj)
                 @warn "Final calculated objective is non-finite!" final_obj=final_obj orig_x=orig_x
            end
            # --- End Check ---

            # Return final solution, original standard form problem, and unscaled *presolved* iterates
            # Return the final x, lambda, s iterates from the IPM loop
            return IplpSolution(vec(orig_x),true,vec(c),A,vec(b),vec(x),vec(lambda),vec(s)) # Use original A_std, b_std, c_std
        end
    end

    # Failed to converge in maxit iterations
    @warn "Solver did not converge within $maxit iterations. mu: $tmp_mu, may adjust tol."
    # Return the final x, lambda, s iterates
    return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s)) # Use original A_std, b_std, c_std
end
