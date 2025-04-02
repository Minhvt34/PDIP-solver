using MatrixDepot
using Test
using Printf
using SparseArrays
using LinearAlgebra
using Statistics

include("unreduced_form.jl")
include("alpha.jl")
include("starting.jl")
include("convert.jl")
include("presolve.jl")
include("solve_standardLp.jl")
include("phase_one.jl")

struct IplpSolution
    x::Vector{Float64} # the solution vector 
    flag::Bool         # a true/false flag indicating convergence or not
    cs::Vector{Float64} # the objective vector in standard form
    As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
    bs::Vector{Float64} # the right hand side (b) in standard form
    xs::Vector{Float64} # the solution in standard form
    lam::Vector{Float64} # the solution lambda in standard form
    s::Vector{Float64} # the solution s in standard form
end  

struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64} 
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

function convert_matrixdepot(mmmeta)
    return IplpProblem(
        vec(mmmeta.c),
        mmmeta.A,
        vec(mmmeta.b),
        vec(mmmeta.lo),
        vec(mmmeta.hi))
end

""" 
soln = iplp(Problem,tol) solves the linear program:

minimize c'*x where Ax = b and lo <= x <= hi

where the variables are stored in the following struct:

Problem.A
Problem.c
Problem.b   
Problem.lo
Problem.hi

and the IplpSolution contains fields 

[x,flag,cs,As,bs,xs,lam,s]

which are interpreted as   
a flag indicating whether or not the
solution succeeded (flag = true => success and flag = false => failure),

along with the solution for the problem converted to standard form (xs):

minimize cs'*xs where As*xs = bs and 0 <= xs

and the associated Lagrange multipliers (lam, s).

This solves the problem up to 
the duality measure (xs'*s)/n <= tol and the normalized residual
norm([As'*lam + s - cs; As*xs - bs; xs.*s])/norm([bs;cs]) <= tol
and fails if this takes more than maxit iterations.
"""

function fix_negative_b(A::AbstractMatrix{T}, b::AbstractVector{T}) where T<:Real
    # Create copies so that we don't mutate the original data.
    A_fixed = copy(A)
    b_fixed = copy(b)
    for i in 1:length(b_fixed)
        if b_fixed[i] < 0
            A_fixed[i, :] .= -A_fixed[i, :]
            b_fixed[i] = -b_fixed[i]
        end
    end
    return A_fixed, b_fixed
end

function scale_lp(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, c::Matrix{Float64})
    m, n = size(A)
    # --- Row scaling ---
    # For each row, compute the infinity norm (max absolute value)
    row_scale = [maximum(abs, A[i, :]) for i in 1:m]
    # Avoid division by zero: if a row is all zeros, set its scaling factor to 1.
    row_scale = [rs == 0.0 ? 1.0 : rs for rs in row_scale]
    # Scale rows: divide each row by its scaling factor.
    D = Diagonal(1.0 ./ row_scale)
    A_row_scaled = D * A
    b_scaled = D * b

    # --- Column scaling ---
    # Now compute the infinity norm for each column of the row-scaled A.
    col_scale = [maximum(abs, A_row_scaled[:, j]) for j in 1:n]
    col_scale = [cs == 0.0 ? 1.0 : cs for cs in col_scale]
    C = Diagonal(1.0 ./ col_scale)
    A_scaled = A_row_scaled * C
    # Adjust c accordingly: note that since c multiplies A from the right,
    # we scale c by the same column factors.
    c_scaled = C * c

    return A_scaled, b_scaled, c_scaled, row_scale, col_scale
end

function iplp(Problem, tol; maxit=100)
    ### test input data
    
    @show m0,n0 = size(Problem.A)
    
    if length(Problem.b) != m0 || length(Problem.c) != n0 || length(Problem.lo) != n0 || length(Problem.hi) != n0
        DimensionMismatch("Dimension of matrices A, b, c mismatch. Check your input.")
    end

    @printf("Problem size: %d, %d\n",m0,n0)

    ### presolve stage

    Ps, ind0c, dup_main_c, ind_dup_c = presolve(Problem)

    ### convert to standard form
    @show size(Ps.A)
    @show rank(Array{Float64}(Ps.A))
    
    A,b,c,ind1,ind2,ind3,ind4 = convert2standard(Ps)

    A,b = fix_negative_b(A,b)
    @show typeof(A)
    @show typeof(b)
    @show typeof(c)
    A,b,c,row_scale,col_scale = scale_lp(A,b,c)

    @show size(A)
    @show rank(Array{Float64}(A))
    ### detect infeasibility

    if phaseone(A,b)
        @warn "This problem is infeasible."
        return IplpSolution(vec([0.]),false,vec(c),A,vec(b),vec([0.]),vec([0.]),vec([0.]))
    end

    @printf("\n=============== MPCIP solver ===============\n%3s %6s %11s %9s %9s\n", "ITER", "MU", "RESIDUAL", "ALPHAX", "ALPHAS")

    ### solve the original problem

    x1,lambda1,s1,flag,iter = solve_standardlp(A,b,c,maxit,tol,true)

    @printf("============================================\n")

    # @show iter

    x = get_x(Ps,ind1,ind2,ind3,ind4,x1)

    x = revProb(Problem, ind0c, dup_main_c, ind_dup_c, x)

    if flag == true
        @printf("This problem is solved with optimal value of %.2f.\n\n", dot(Problem.c, x))
    else
        @printf("\nThis problem does not converge in %d steps.", maxit)
    end

    return IplpSolution(vec(x),flag,vec(c),A,vec(b),vec(x1),vec(lambda1),vec(s1))
end
