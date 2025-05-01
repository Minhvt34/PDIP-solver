# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;
using MatrixDepot, SparseArrays

mutable struct IplpSolution
  x::Vector{Float64} # the solution vector
  flag::Bool         # a true/false flag indicating convergence or not
  cs::Vector{Float64} # the objective vector in standard form
  As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
  bs::Vector{Float64} # the right hand side (b) in standard form
  xs::Vector{Float64} # the solution in standard form
  lam::Vector{Float64} # the solution lambda in standard form
  s::Vector{Float64} # the solution s in standard form
end

mutable struct IplpProblem
  c::Vector{Float64}
  A::SparseMatrixCSC{Float64}
  b::Vector{Float64}
  lo::Vector{Float64}
  hi::Vector{Float64}
end

include("starting_point.jl")
#include("presolve.jl")
include("presolve_extended.jl")
include("conversions.jl")

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
    return max(alpha, 0.0)
end

function iplp(Problem, tol; maxit=100)
    # Convert to standard form
    A, b, c, free, bounded_below, bounded_above, bounded = tostandard(Problem)

    # Presolve step - modify A,b,c to be nicer
    @show size(A)
    orig_n = size(A, 2)



    # A, b, c, remaining_cols, removed_cols, xpre, feasible = presolve(A, b, c, Problem.hi, Problem.lo)

    m_std, n_std = size(A)
    lo_std = zeros(n_std)
    hi_std = fill(Inf, n_std)

    standard_problem = IplpProblem(c, A, b, lo_std, hi_std)
    # std_Ps, ind0c, dup_main_c, ind_dup_c = presolve(std_problem)

    # A = std_Ps.A
    # b = std_Ps.b
    # c = std_Ps.c

    presolve_result = presolve(standard_problem)
    status = presolve_result[1]

    if status == :Success
        std_presolved, ind0c_std, dup_main_c_std, ind_dup_c_std, ind_fix_c_std, fix_vals_std, dual_lb, dual_ub, obj_offset, free_singleton_subs, final_col_indices_std = presolve_result[2:end]
        @printf("Original standard form size: (%d, %d), After presolve: (%d, %d)\n", 
                m_std, n_std, size(std_presolved.A)...)
        
        # --- Check Presolve obj_offset ---
        if !isfinite(obj_offset)
            @error "Presolve returned non-finite obj_offset!" obj_offset=obj_offset
        end
        # --- End Check ---

        # --- Solve the presolved problem --- 
        A = std_presolved.A
        b = std_presolved.b
        c = std_presolved.c

        # --- Check b immediately after presolve ---
        if any(!isfinite, b)
            @error "Vector b contains non-finite values immediately after presolve!" b=b
            # Optionally, stop execution or handle the error
            # return IplpSolution(...) # Or some error state
        end
        # --- End Check ---

        # --- Rank Check --- 
        presolved_A = std_presolved.A
        presolved_m, presolved_n = size(presolved_A)
        rank_A = rank(Matrix(presolved_A)) # Convert to dense for rank calculation
        @printf("Rank of presolved A: %d (Dimensions: %d x %d)\n", rank_A, presolved_m, presolved_n)
        if rank_A < presolved_m
             @warn "Presolved matrix A has linearly dependent rows (rank $rank_A < $presolved_m). KKT matrix will be singular."
             # Optionally: Decide whether to proceed or error out
             # return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(y),vec(s), :PresolveRankDeficient)
        end
        # --- End Rank Check -
    end

    # if !feasible
    #     return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
    # end

    m,n = size(A)
    @show size(A)

    # --- Check inputs to get_starting_point ---
    if any(!isfinite, A.nzval) # Check non-zero values for sparse matrix
        @error "Matrix A contains non-finite values before starting point calculation!"
        # Handle error appropriately, e.g., return an error status
    end
    if any(!isfinite, b)
        @error "Vector b contains non-finite values before starting point calculation!"
    end
    if any(!isfinite, c)
        @error "Vector c contains non-finite values before starting point calculation!"
    end
    # --- End Check ---

    # Get initially feasible x, lambda, s
    # TODO
    x, lambda, s = get_starting_point(A, b, c)

    @show minimum(x)
    @show minimum(s)
    @show maximum(abs.(x)) # Check for large values too
    @show maximum(abs.(s))

    # Implementing Mehrotra's predictor-corrector algorithm (Ch. 10 of Wright)
    for i = 1:maxit
        @show i
        # Regularization parameters (can be tuned)
        delta_p = 1e-8 # Primal regularization
        delta_d = 1e-8 # Dual regularization

        # Affine direction step (10.1 Wright)
        M = [
            spdiagm(0 => fill(delta_p, n))  A'              Matrix(I, n, n);
            A                              -spdiagm(0 => fill(delta_d, m)) spzeros(m, n);
            spdiagm(0 => s) spzeros(n, m) spdiagm(0 => x);
        ]

        # Check if M is full-rank
        # Use LU factorization, which is more robust for potentially singular matrices than Cholesky
        local mat # Ensure mat is scoped correctly for potential error handling
        try
            mat = lu(M)
            if !issuccess(mat) # Check if LU factorization succeeded numerically
                 @warn "LU factorization of KKT matrix failed numerically at iteration $i."
                 # Handle failure: maybe return current state or a specific error status
                 return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s), :KKTLUFactorizationFailed)
            end
        catch e
             @warn "Error during LU factorization of KKT matrix at iteration $i." exception=e
             # Handle failure
             return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s), :KKTLUFactorizationError)
        end

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

        @show alpha_aff_pri, alpha_aff_dual
        @show alpha_pri, alpha_dual
        @show mu, dot(x, s)

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
        if dot(x, s) / n <= tol && norm([A'* lambda + s - c; A * x - b; x.*s]) / norm([b;c]) <= tol
            #x_unpresolved = unpresolve(orig_n, x, remaining_cols, removed_cols, xpre)
            #x_unpresolved = revProb(std_problem, ind0c, dup_main_c, ind_dup_c, x)
            x_unpresolved = revProb(standard_problem, ind0c_std, dup_main_c_std, ind_dup_c_std, ind_fix_c_std, fix_vals_std, free_singleton_subs, final_col_indices_std, x)
            
            # --- Check revProb result ---
            if any(!isfinite, x_unpresolved)
                @warn "revProb returned non-finite values in x_unpresolved!" x_unpresolved=x_unpresolved
            end
            # --- End Check ---
            
            orig_x = fromstandard(Problem, x_unpresolved, free, bounded_below, bounded_above, bounded)
            
            # --- Check fromstandard result ---
             if any(!isfinite, orig_x)
                @warn "fromstandard returned non-finite values in orig_x!" orig_x=orig_x x_unpresolved=x_unpresolved
            end
            # --- End Check ---

            @show i
            # --- Calculate and check final objective ---
            # --- Check magnitude of orig_x ---
            max_abs_orig_x = maximum(abs.(orig_x))
            @info "Calculating final objective: dot(Problem.c, orig_x). Max abs(orig_x) = $(max_abs_orig_x)"
            # --- End Check ---
            final_obj = dot(Problem.c, orig_x) # Use original cost vector
            if !isfinite(final_obj)
                 @warn "Final calculated objective is non-finite!" final_obj=final_obj orig_x=orig_x
            end
            # --- End Check ---
            
            return IplpSolution(vec(orig_x),true,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end
    end

    # Failed to converge in maxit iterations
    return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
end

