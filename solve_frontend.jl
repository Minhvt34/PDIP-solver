using Printf

include("project2.jl")

# Get problem to be solved
@printf("Type the name of the LPnetlib problem to be solved (e.g. lp_afiro): ")
name = readline(stdin)

# Load problem
problem_path = "LPnetlib/"*name
@printf("Will solve: %s\n", problem_path)
problem = convert_matrixdepot(mdopen(problem_path))

# Set params and solve problem
tol = 1e-8
maxit = 100
solution = @time iplp(problem, tol; maxit)

@show solution.flag
if solution.flag
    @printf "Objective value: %lf\n" dot(problem.c,solution.x)
else
    @printf "Solver failed to converge. Check if problem is feasible. Try increasing # iterations in solve_frontend.jl or decreasing tolerance."
end