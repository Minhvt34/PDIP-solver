# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;

include("starting_point.jl")
#include("presolve.jl")
include("presolve_extended.jl")
include("conversions.jl")
include("problem_def.jl")

function solve_processed(Problem, tol; maxit=100)
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
        std_presolved, ind0c_std, dup_main_c_std, ind_dup_c_std, ind_fix_c_std, fix_vals_std, dual_lb, dual_ub, obj_offset, free_singleton_subs = presolve_result[2:end]
        @printf("Original standard form size: (%d, %d), After presolve: (%d, %d)\n", 
                m_std, n_std, size(std_presolved.A)...)
        
        # --- Solve the presolved problem --- 
        A = std_presolved.A
        b = std_presolved.b
        c = std_presolved.c
    end

    # if !feasible
    #     return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
    # end

    m,n = size(A)
    @show size(A)

    # Get initially feasible x, lambda, s
    # TODO
    x, lambda, s = get_starting_point(A, b, c)

    # A is 27x51 (mxn)
    # A' is 51x27 (nxm)
    # 51x51, 51x27, 51x51
    # 27x51, 27x27, 27x51
    # 51x51, 51x27, 51x51

    # Implementing Mehrotra's predictor-corrector algorithm (Ch. 10 of Wright)
    for i = 1:maxit
        # Affine direction step (10.1 Wright)
        mat = lu([
            spzeros(n, n) A' Matrix(I, n, n);
            A spzeros(m, m) spzeros(m, n);
            spdiagm(0 => s) spzeros(n, m) spdiagm(0 => x);
        ])
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
            x_unpresolved = revProb(standard_problem, ind0c_std, dup_main_c_std, ind_dup_c_std, ind_fix_c_std, fix_vals_std, free_singleton_subs, x)
            orig_x = fromstandard(Problem, x_unpresolved, free, bounded_below, bounded_above, bounded)
            @show i
            return IplpSolution(vec(orig_x),true,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end
    end

    # Failed to converge in maxit iterations
    return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
end
