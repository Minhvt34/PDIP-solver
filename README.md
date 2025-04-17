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

To run the solver

``` shell
git clone https://github.com/Minhvt34/PDIP-solver.git
cd PDIP-solver
julia solve_frontend.jl
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

```

