using LinearAlgebra

"""
    Finite difference solver for
     - source problems
     - eigenvalue problems

    of linear Schrödinger type operator
        H = - (1/2) Δ + V + σ
    where
    Δ     Laplacian in 1D
    V     potential
    σ     shift 
"""

"""
Resolution by finite differences for source problem
"""
function exact_solution(Ng, V, σ, rhs, FD_grid)
    
    # space discretisation
    x_range, δx = FD_grid

    # discrete Hamiltonian on space grid
    A = Hd(δx, Ng, V.(x_range), σ)

    # solve the linear system
    sol = A \ rhs

    sol
end

"""
Resolution by finite differences for eigenvalue problem
with normalisation to 1, yields first nvals eigenpairs
"""
function exact_eigensolver(Ng, V, σ, FD_grid; nvals=17)
    
    # space discretisation
    x_range, δx = FD_grid

    # discrete Hamiltonian on space grid
    A = Hd(δx, Ng, V.(x_range), σ) 

    # solve the generalized eigenvalue problem
    # utiliser eigs au lieu de eigsolve Krylov
    #vals, vecs, info = eigsolve(A, nvals, :SR; issymmetric=true,
    #                            verbosity=1, tol=1e-12, maxiter=1000,
    #                            eager=true)
    F = eigen(Matrix(A))
    # F.values, F.vectors
    
    (F.values[1:nvals],F.vectors[:,1:nvals])
end

