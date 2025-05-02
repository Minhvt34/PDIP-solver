# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;

include("starting_point_simple.jl")
include("presolve_simple.jl")
include("conversions_simple.jl")

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

function simple_iplp(Problem, tol; maxit=100)
    # Convert to standard form
    A_std, b_std, c_std, free, bounded_below, bounded_above, bounded = toStandardSimple(Problem)
    m_std, n_std = size(A_std)

    # Presolve step - modify A,b,c to be nicer
    @show size(A_std)
    orig_n_std = n_std # Keep track of original standard dimension n
    A_ps, b_ps, c_ps, remaining_cols, removed_cols, xpre, feasible = simple_presolve(A_std, b_std, c_std, Problem.hi, Problem.lo)

    if !feasible
        @warn "Problem determined infeasible during simple presolve."
        return IplpSolution(vec([]), false, vec(c_std), A_std, vec(b_std), vec([]), vec([]), vec([]), :PresolveInfeasible)
    end
    m_ps, n_ps = size(A_ps)
    @printf("Original standard form size: (%d, %d), After simple presolve: (%d, %d)\\n",
                m_std, n_std, m_ps, n_ps)

    # --- Conditional Scaling Check ---
    enable_scaling = false # Default to false
    scaling_threshold = 500.0 # <-- Lowered Threshold
    if m_ps > 0 && n_ps > 0 # Only check if matrix is non-empty
        row_norms = vec(maximum(abs.(A_ps), dims=2))
        col_norms = vec(maximum(abs.(A_ps), dims=1))

        # Filter out zero norms before calculating min
        non_zero_row_norms = row_norms[row_norms .> 1e-12]
        non_zero_col_norms = col_norms[col_norms .> 1e-12]

        if !isempty(non_zero_row_norms) && !isempty(non_zero_col_norms)
            min_row_norm = minimum(non_zero_row_norms)
            max_row_norm = maximum(row_norms)
            min_col_norm = minimum(non_zero_col_norms)
            max_col_norm = maximum(col_norms)

            row_ratio = (min_row_norm > 1e-12) ? max_row_norm / min_row_norm : Inf
            col_ratio = (min_col_norm > 1e-12) ? max_col_norm / min_col_norm : Inf

            @info "Scaling Ratios - Row: $(round(row_ratio, digits=2)), Col: $(round(col_ratio, digits=2)) (Threshold: $scaling_threshold)"

            if row_ratio > scaling_threshold || col_ratio > scaling_threshold
                @info "Scaling enabled based on norm ratios."
                enable_scaling = true
            else
                @info "Scaling disabled; matrix norms within threshold."
            end
        else
             @info "Scaling disabled; matrix has zero rows/columns or is empty after filtering."
        end
    else
         @info "Scaling disabled; presolved matrix is empty."
    end
    # --- End Conditional Scaling Check ---

    # --- Variables for scaling factors ---
    Dr = I(m_ps) # Initialize as Identity
    Dc = I(n_ps)
    dr = ones(m_ps)
    dc = ones(n_ps)
    A_ps_orig = A_ps # Keep copy of presolved A before scaling
    b_ps_orig = b_ps
    c_ps_orig = c_ps

    # --- Problem Scaling (Equilibration) --- 
    A = A_ps_orig # Initialize A, b, c with unscaled presolved versions
    b = b_ps_orig
    c = c_ps_orig
    scaled_A = A # Initialize scaled_A for the case the loop doesn't run but scaling enabled?
    if enable_scaling
        @info "--- Starting Scaling Procedure (Simple Path) ---"
        max_attempts = 3
        scale_tol = 1e-2

        prev_max_row_norm = Inf
        prev_max_col_norm = Inf

        for attempt = 1:max_attempts
            # --- Row Scaling (Ruiz) ---
            temp_scaled_A_for_row_norm = Dr * A_ps_orig * Dc
            row_inf_norms = vec(maximum(abs.(temp_scaled_A_for_row_norm), dims=2))
            row_inf_norms[row_inf_norms .< 1e-12] .= 1.0
            dr_update_factor = 1.0 ./ sqrt.(row_inf_norms)
            dr .*= dr_update_factor
            clamp!(dr, 1e-8, 1e8)
            Dr = spdiagm(0 => dr)

            # --- Column Scaling (Ruiz) ---
            temp_scaled_A_for_col_norm = Dr * A_ps_orig * Dc
            col_inf_norms = vec(maximum(abs.(temp_scaled_A_for_col_norm), dims=1))
            col_inf_norms[col_inf_norms .< 1e-12] .= 1.0
            dc_update_factor = 1.0 ./ sqrt.(col_inf_norms)
            dc .*= dc_update_factor
            clamp!(dc, 1e-8, 1e8)
            Dc = spdiagm(0 => dc)

            # --- Convergence Check ---
            scaled_A = Dr * A_ps_orig * Dc
            current_row_norms = maximum(abs.(scaled_A), dims=2)
            current_col_norms = maximum(abs.(scaled_A), dims=1)' # Transpose
            max_row_norm = maximum(current_row_norms)
            max_col_norm = maximum(current_col_norms)
            norm_converged = abs(max_row_norm - 1.0) < scale_tol && abs(max_col_norm - 1.0) < scale_tol
            norm_change_small = abs(max_row_norm - prev_max_row_norm) < scale_tol && abs(max_col_norm - prev_max_col_norm) < scale_tol
            if norm_converged || norm_change_small
                 @info "Scaling converged after $attempt iterations. MaxRowNorm: $max_row_norm, MaxColNorm: $max_col_norm"
                 break
            end
            prev_max_row_norm = max_row_norm
            prev_max_col_norm = max_col_norm
            if attempt == max_attempts
                @warn "Scaling did not fully converge after $max_attempts iterations. MaxRowNorm: $max_row_norm, MaxColNorm: $max_col_norm"
            end
        end

        # Apply final scaling using the last computed factors
        # Recompute scaled_A one last time ensures consistency
        scaled_A = Dr * A_ps_orig * Dc
        A = scaled_A
        b = Dr * b_ps_orig
        c = Dc * c_ps_orig
        @info "Applied equilibration scaling to presolved problem."
        @info "--- Finished Scaling Procedure (Simple Path) ---"
    # else # Scaling skipped, A, b, c remain A_ps, b_ps, c_ps
    #    @info "Scaling skipped."
    end
    # --- End Problem Scaling ---

    # Assign final matrices to be used by IPM (already done above)
    # A = A_ps # Assign presolved matrix (potentially overwritten by scaling later)
    # b = b_ps
    # c = c_ps
    m, n = size(A) # IPM dimensions

    @show size(A)

    # Get initially feasible x, lambda, s
    # Uses the (potentially scaled) A, b, c
    x, lambda, s = get_starting_point_simple(A, b, c)

    # -- Adaptive Regularization Parameters --
    delta_p = 1e-8       # Initial primal regularization
    delta_d = 1e-8       # Initial dual regularization
    min_delta = 1e-10    # Minimum regularization
    max_delta = 1e-2     # Maximum regularization
    delta_increase_factor = 100.0 # Factor to increase delta on LU failure
    delta_decrease_factor = 2.0   # Factor to decrease delta on LU success
    # -------------------------------------

    # Implementing Mehrotra's predictor-corrector algorithm (Ch. 10 of Wright)
    for i = 1:maxit
        # -- Adaptive Factorization Attempt --
        factorization_success = false
        local mat
        current_delta_p = delta_p
        current_delta_d = delta_d
        max_factorization_retries = 3

        for retry_attempt = 0:max_factorization_retries
            M = [
                spdiagm(0 => fill(current_delta_p, n))  A'              Matrix(I, n, n);
                A                                     -spdiagm(0 => fill(current_delta_d, m)) spzeros(m, n);
                spdiagm(0 => s)                        spzeros(n, m)  spdiagm(0 => x);
            ]
            try
                mat = lu(M)
                if issuccess(mat)
                    factorization_success = true
                    delta_p = max(min_delta, delta_p / delta_decrease_factor)
                    delta_d = max(min_delta, delta_d / delta_decrease_factor)
                    if retry_attempt > 0
                         @info "LU factorization succeeded (Simple Path) at iteration $i after retry $retry_attempt with delta_p=$current_delta_p, delta_d=$current_delta_d"
                    end
                    break
                else
                    @warn "LU factorization failed numerically (Simple Path) at iteration $i (retry $retry_attempt). Increasing regularization."
                    current_delta_p = min(max_delta, current_delta_p * delta_increase_factor)
                    current_delta_d = min(max_delta, current_delta_d * delta_increase_factor)
                    delta_p = current_delta_p
                    delta_d = current_delta_d
                end
            catch e
                 @warn "Error during LU factorization (Simple Path) at iteration $i (retry $retry_attempt)." exception=e
                 current_delta_p = min(max_delta, current_delta_p * delta_increase_factor)
                 current_delta_d = min(max_delta, current_delta_d * delta_increase_factor)
                 delta_p = current_delta_p
                 delta_d = current_delta_d
            end

            if !factorization_success && retry_attempt == max_factorization_retries
                 @warn "LU factorization failed permanently (Simple Path) at iteration $i after $max_factorization_retries retries."
                 return IplpSolution(vec([]), false, vec(c_std), A_std, vec(b_std), vec(x), vec(lambda), vec(s), :KKTLUFactorizationFailed)
            end
        end
        # -- End Adaptive Factorization Attempt --

        rhs_aff = [
            - (A' * lambda + s - c);
            - (A * x - b);
            - (x .* s)
        ]
        daff = mat \ rhs_aff
        dxaff, dlambdaff, dsaff = daff[1:n], daff[n+1:n+m], daff[n+m+1:n+m+n]
        alpha_aff_pri = min(1.0, calcalpha(x, dxaff))
        alpha_aff_dual = min(1.0, calcalpha(s, dsaff))
        mu = dot(x, s) / n # 1.11
        muaff = (x + alpha_aff_pri * dxaff)' * (s + alpha_aff_dual * dsaff) / n
        sigma = (muaff / mu)^3 # 10.3

        # Centering-Corrector step from Wright 10.7
        rhs_cc = [
            zeros(n, 1);
            zeros(m, 1);
            sigma * mu .- (dxaff .* dsaff)
        ]
        dcc = mat \ rhs_cc
        dxcc, dlambdacc, dscc = dcc[1:n], dcc[n+1:n+m], dcc[n+m+1:n+m+n]
        dx, dlambda, ds = dxaff + dxcc, dlambdaff + dlambdacc, dsaff + dscc
        alpha_pri = min(0.99 * calcalpha(x, dx), 1.0)
        alpha_dual = min(0.99 * calcalpha(s, ds), 1.0)

        # @show alpha_aff_pri, alpha_aff_dual
        # @show alpha_pri, alpha_dual
        # @show mu, dot(x, s)

        if (dot(x, s) > 1e308)
            # Very large (exploding) complementarity - problem is infeasible
            return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end

        if (alpha_pri > 1e308 || alpha_dual > 1e308)
            # Very large alpha; problem is unbounded or infeasible
            return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end

        x = x + alpha_pri * dx
        lambda = lambda + alpha_dual * dlambda
        s = s + alpha_dual * ds

        # Check if tolerances are satisfied
        # Uses scaled norms if scaling was applied
        primal_res = A * x - b
        dual_res = A' * lambda + s - c
        mu = dot(x, s) / n

        norm_b = norm(b, Inf)
        norm_c = norm(c, Inf)
        rel_tol_b = max(1.0, norm_b)
        rel_tol_c = max(1.0, norm_c)
        primal_feas = norm(primal_res, Inf) / rel_tol_b
        dual_feas = norm(dual_res, Inf) / rel_tol_c

        if mu <= tol && primal_feas <= tol && dual_feas <= tol
            @info "Converged at iteration $i."

            # --- Unscale Solution BEFORE un-presolving if enable_scaling --- 
            x_ps_unscaled, lambda_ps_unscaled, s_ps_unscaled = x, lambda, s # Start with final iterates
            if enable_scaling
                x_ps_unscaled = Dc * x
                lambda_ps_unscaled = Dr * lambda
                s_ps_unscaled = spdiagm(0 => 1.0 ./ dc) * s # Equivalent to Dc \ s
                @info "Unscaled IPM solution vectors."
            end
            # Use x_ps_unscaled, etc. for subsequent steps

            # --- Un-Presolve ---
            x_unpresolved = simple_unpresolve(orig_n_std, x_ps_unscaled, remaining_cols, removed_cols, xpre)

            # --- Convert back from standard form ---
            orig_x = fromStandardSimple(Problem, x_unpresolved, free, bounded_below, bounded_above, bounded)
            @show i

            # Return final solution, original standard form problem, and unscaled *presolved* iterates
            # If scaling was done, x_ps/lambda_ps/s_ps are the unscaled presolved iterates.
            # If not, they are just the final iterates x/lambda/s from the loop.
            return IplpSolution(vec(orig_x), true, vec(c_std), A_std, vec(b_std), vec(x), vec(lambda), vec(s))
        end
    end

    # Failed to converge in maxit iterations
    @warn "Solver did not converge within $maxit iterations."
    # Return final loop iterates

    # Return original standard form problem and (conditionally unscaled) presolved iterates
    return IplpSolution(vec([]), false, vec(c_std), A_std, vec(b_std), vec(x), vec(lambda), vec(s))
end