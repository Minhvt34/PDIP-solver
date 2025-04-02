# Develop a robust Primal-dual Interior-point Solver

This is the final course project for course CS520: Computational Methods for Optimization collaborated with [Vincent Wang](github link). 

Please check this report for more detail. We have to make our own weekly report.

The complete algorithms and experimental result are in the [report](./report.pdf).

## Features

- Julia implementation

- Presolve stage

- Mehrotra's predictor-corrector algorithm

## Requirements
You need to install Julia version > v1.1x to load test from MatrixDepot

- [Julia](https://julialang.org/) (v1.11.4)

- [MatrixDepot](https://github.com/JuliaMatrices/MatrixDepot.jl) (v0.8): test matrix collection for Julia

## Usage

The original project from here
``` shell
git clone https://github.com/dlguo/primal-dual-interior-point.git
```

To run the solver

``` shell
git clone https://github.com/Minhvt34/PDIP-solver.git
cd PDIP-solver
julia solve.jl
```

I have mordified the problem by adjusting the tolerance in phase_one to 1e-8.
With the original tol = 1e-9, the solver is not that robust due to the tolerance is too tight. I also made other modification but still cannot improve the robustness of the solver.

I add 11 problems to the interactive interface in which problem lp_etamacro, lp_fffff800 are not included in the posted project.
We also can manually add project by name which should be published in LPnetlib.
So far, I have not tested with other lib because the solver failed to solve problem lp_etamacro. Therefore I anticipate that we have to find ways to improve the current solver.

In the interactive interface, choose 9 preset test problem in `LPnetlib` or use your own dataset.
