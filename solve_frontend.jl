using Printf

include("project2.jl")

# Get problem to be solved
@printf("Type the name of the LPnetlib problem to be solved (e.g. lp_afiro) or 'ALL_LARGE': ")
name = readline(stdin)

if name == "ALL_LARGE"
    global_logger(ConsoleLogger(stderr, Logging.Warn)) # Set minimum level to Warn

    problems = [
        "lp_cre_a",
        "lp_cre_b",
        "lp_cre_c",
        "lp_cre_d",
        "lp_ken_11",
        "lp_ken_13",
        "lp_ken_18",
        "lp_osa_07",
        "lp_osa_14",
        "lp_osa_30",
        "lp_osa_60",
        "lp_pds_06",
        "lp_pds_10",
        "lp_pds_20",
        "lp_fit2d",
        "lp_pilot",
        "lp_pilot87",
        "greenbea",
        "greenbeb",
    ]
else
    problems = [name]
end

for name in problems
    # Load problem
    problem_path = "LPnetlib/"*name
    @printf("Will solve: %s\n", problem_path)
    problem = convert_matrixdepot(mdopen(problem_path))

    # Set params and solve problem
    tol = 1e-7
    maxit = 100
    solution = @time iplp(problem, tol; maxit)

    @show solution.flag
    if solution.flag
        @printf "Objective value: %lf\n" dot(problem.c,solution.x)
    else
        @printf "Solver failed to converge. Check if problem is feasible. Try increasing # iterations in solve_frontend.jl or decreasing tolerance."
    end
end