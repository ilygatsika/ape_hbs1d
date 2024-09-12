# ape_hbs1d

Julia code for the a posteriori estimation of Hermite basis set errors on a toy model in 1D. This code is used in our paper _A posteriori error estimates for Schrödinger operators discretized with linear combinations of atomic orbital_ available on [arXiv](pending).

## Requirements

Julia 1.8.3 with the libraries:
- Test, BoundaryValueDiffEq for testing;
- LinearAlgebra, Polynomials, SpecialPolynomials, KrylovKit, ForwardDiff for the computations;
- JSON3 for saving results;
- PyPlot, LaTeXStrings for plotting results.

## Install and test

In the root directory, open a Julia shell with `julia --project` and then run
```
using Pkg
Pkg.instantiate()
Pkg.test()
```

## Usage

```
include("run.jl")
```

This program runs all calculations at once, stores results in json format and then produces figures placed in the `img` directory. Running all the computations takes around 20 minutes. To configure your own graphic display options used for plotting, you can modify the `common.jl` file.

## Authors

Mi-Song Dupuy (Sorbonne Université), Geneviève Dusson (Université de Franche Comté, CNRS), Ioanna-Maria Lygatsika (Sorbonne Université).

## Credits

This code heavily uses modified routines originally found in the git repo [gkemlin/1D_basis_optimization](https://github.com/gkemlin/1D_basis_optimization.git) of the author Gaspard Kemlin.


