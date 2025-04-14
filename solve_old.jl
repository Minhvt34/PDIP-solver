# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;

include("starting_point.jl")
include("conversions.jl")

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
    m,n = size(Problem.A)

    # Get initially feasible x, lambda, s
    # TODO
    x, lambda, s = get_starting_point(A, b, c)

    # A is 27x51 (mxn)
    # A' is 51x27 (nxm)
    # 51x51, 51x27, 51x51
    # 27x51, 27x27, 27x51
    # 51x51, 51x27, 51x51

    # Implementing Mehotra's predictor-corrector algorithm (Ch. 10 of Wright)
    for i = 1:maxit
        # Affine direction step (10.1 Wright)
        sigma = 0.1
        mu = dot(x, s) / n # 1.11
        mat = lu([
            spzeros(n, n) Problem.A' Matrix(I, n, n);
            Problem.A spzeros(m, m) spzeros(m, n);
            spdiagm(0 => s) spzeros(n, m) spdiagm(0 => x);
        ])
        rhs_aff = [
            - (Problem.A' * lambda + s - Problem.c);
            - (Problem.A * x - Problem.b);
            sigma * mu .- (x .* s)
        ]
        daff = mat \ rhs_aff
        dxaff, dlambdaff, dsaff = daff[1:n], daff[n+1:n+m], daff[n+m+1:n+m+n]
        alpha_pri = min(0.99 * calcalpha(x, dxaff), 1.0)
        alpha_dual = min(0.99 * calcalpha(s, dsaff), 1.0)

        if (alpha_pri > 1e308 || alpha_dual > 1e308)
            # Very large alpha; problem is unbounded or infeasible
            return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end

        x = x + alpha_pri * dxaff
        lambda = lambda + alpha_dual * dlambdaff
        s = s + alpha_dual * dsaff

        # Check if tolerances are satisfied
        if dot(x, s) / n <= tol && norm([A'* lambda + s - c; A * x - b; x.*s]) / norm([b;c]) <= tol
            orig_x = fromstandard(Problem, x, free, bounded_below, bounded_above, bounded)
            @show i
            return IplpSolution(vec(orig_x),true,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end
    end

    # Failed to converge in maxit iterations
    return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
end
