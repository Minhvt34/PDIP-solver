# Develop a robust Primal-dual Interior-point Solver

This is the final course project for course CS520: Computational Methods for Optimization collaborated with [Vincent Wang](https://github.com/bbworld1). 

Please check this report for more detail.

The complete algorithms and experimental result are in the [report](./report.pdf).

## Features

- Julia implementation

- Presolve stage (basic + extended procedures)

- Mehrotra's predictor-corrector algorithm

## Requirements
You need to install Julia version > v1.1x to load test from MatrixDepot

- [Julia](https://julialang.org/) (v1.11.4)

- [MatrixDepot](https://github.com/JuliaMatrices/MatrixDepot.jl) (v0.8): test matrix collection for Julia

## Usage

To run the solver (test on LPNetLib, problems will be loaded using the MatrixDepot.jl) package.

``` shell
git clone https://github.com/Minhvt34/PDIP-solver.git
cd PDIP-solver
julia solve_frontend.jl
```
The struct and convert matrix depot function are placed in problem_def.jl.
The function to evaluate is placed in project2.jl. For usage:

``` shell
include("project2.jl")

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
```

## References

```bibtex
@article{andersen1995presolving,
  title={Presolving in linear programming},
  author={Andersen, Erling D and Andersen, Knud D},
  journal={Mathematical programming},
  volume={71},
  pages={221--245},
  year={1995},
  publisher={Springer}
}
@book{nocedal1999numerical,
  title={Numerical optimization},
  author={Nocedal, Jorge and Wright, Stephen J},
  year={1999},
  publisher={Springer}
}

@book{doi:10.1137/1.9781611971453,
author = {Wright, Stephen J.},
title = {Primal-Dual Interior-Point Methods},
publisher = {Society for Industrial and Applied Mathematics},
year = {1997},
doi = {10.1137/1.9781611971453},
address = {},
edition   = {},
URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611971453},
eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611971453}
}

@article{Andersen01011995,
author = {Erling D. Andersen and},
title = {Finding all linearly dependent rows in large-scale linear programming},
journal = {Optimization Methods and Software},
volume = {6},
number = {3},
pages = {219--227},
year = {1995},
publisher = {Taylor \& Francis},
doi = {10.1080/10556789508805634},
}

```

