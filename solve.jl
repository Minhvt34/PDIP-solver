#!/usr/bin/julia

include("lpsolver.jl")

problem_list = ["lp_afiro","lp_brandy","lp_fit1d","lp_adlittle",
"lp_agg","lp_ganges","lp_stocfor1", "lp_25fv47", "lpi_chemcom", "lp_etamacro", "lp_fffff800"]

for i = 1:length(problem_list)
    @printf("%d. %s\n", i, problem_list[i])
end
@printf("0. other\n")
@printf("Which problem to solve? ")

k = parse(Int,readline(stdin))

if k == 0
    @printf("Please enter the problem name (e.g. lp_afiro): ")
    name = "LPnetlib/"*readline(stdin)[1:end-1]
else
    name = "LPnetlib/"*problem_list[k]
end

@printf("Solving %s.", name)

try global P = convert_matrixdepot(mdopen(name))
catch 
    mdopen(name)
    global P = convert_matrixdepot(mdopen(name))
end

tol=1e-8
solution = @time iplp(P, tol; maxit=200)


@show solution.flag
@show dot(P.c,solution.x)
