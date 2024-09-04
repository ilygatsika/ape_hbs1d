using Polynomials, SpecialPolynomials
import Polynomials: basis

"""
Gaussian type basis. Return as functions of x.
"""
function simple_gaussian_basis(σ)
    c(a) = (2 * a / π) ^ (1/4)
    # all basis functions
    [x -> c(a) * exp( - a * x^2) for a in σ]
end

"""
Hermite-polynomial Gaussian type basis. Return as functions of x.
"""
function simple_hermite_basis(N)
    X = variable(Polynomial{Rational{Int}})
    c(n) = 1.0 / √(2^(n-1) * factorial(big(n-1)) * √π)

    # All basis functions
    hermite_pol = [basis(Hermite, n)(X) for n in 0:2*N]
    [x->p(x)*exp(-(x)^2/2) * c(n) for (n,p) in enumerate(hermite_pol)][1:N]
end

"""
Translate the simple basis given by the above routine on a.
The result is an Array of size Ng×N.
"""
function centered_hermite_basis(a, N, grid)
    
    x_range, δx = grid
    Ng = length(x_range)
    DB_on_grid = zeros(Ng, N)
    hermite_bfs = simple_hermite_basis(N)
    # Place basis element in a
    for i in 1:N
        DB_on_grid[:,i] .= hermite_bfs[i].(x_range .- a)
    end
    DB_on_grid .* √δx
end

