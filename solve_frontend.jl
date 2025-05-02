using Printf

include("project2.jl")

# Get problem to be solved
@printf("Type the name of the LPnetlib problem to be solved (e.g. lp_afiro) or 'ALL_LARGE': ")
name = readline(stdin)

if name == "ALL_SMALL"
    global_logger(ConsoleLogger(stderr, Logging.Warn)) # Set minimum level to Warn

    problems = [
        "lp_25fv47",
        "lp_80bau3b",
        "lp_adlittle",
        "lp_afiro",
        "lp_agg",
        "lp_agg2",
        "lp_agg3",
        "lp_bandm",
        "lp_beaconfd",
        "lp_blend",
        "lp_bnl1",
        "lp_bnl2",
        "lp_bore3d",
        "lp_brandy",
        "lp_capri",
        "lp_cycle",
        "lp_czprob",
        "lp_d2q06c",
        "lp_degen2",
        "lp_degen3",
        "lp_e226",
        "lp_etamacro",
        "lp_fffff800",
        "lp_finnis",
        "lp_fit1d",
        "lp_fit2d",
        "lp_ganges",
        "lp_gfrd_pnc",
        "lp_greenbea",
        "lp_greenbeb",
        "lp_grow15",
        "lp_grow22",
        "lp_grow7",
        "lp_israel",
        "lp_kb2",
        "lp_lotfi",
        "lp_maros",
    ]

elseif name == "ALL_LARGE"
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