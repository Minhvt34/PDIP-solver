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
        mat = [
            spzeros(n, n) Problem.A' Matrix(I, n, n);
            Problem.A spzeros(m, m) spzeros(m, n);
            spdiagm(0 => s) spzeros(n, m) spdiagm(0 => x);
        ]
        rhs_aff = [
            - (Problem.A' * lambda + s - Problem.c);
            - (Problem.A * x - Problem.b);
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

        if (alpha_pri > 1e308 || alpha_dual > 1e308)
            # Very large alpha; problem is unbounded or infeasible
            return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end

        x = x + alpha_pri * dx
        lambda = lambda + alpha_dual * dlambda
        s = s + alpha_dual * ds

        # Check if tolerances are satisfied
        if dot(x, s) / n <= tol && norm([A'* lambda + s - c; A * x - b; x.*s]) / norm([b;c]) <= tol
            orig_x = fromstandard(Problem, x, free, bounded_below, bounded_above, bounded)
            return IplpSolution(vec(orig_x),true,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
        end
    end

    # Failed to converge in maxit iterations
    return IplpSolution(vec([]),false,vec(c),A,vec(b),vec(x),vec(lambda),vec(s))
end
