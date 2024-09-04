using LinearAlgebra
using ForwardDiff
using Crayons.Box
using KrylovKit

"""
n-oder derivative interface with ForwardDiff
reference: https://github.com/jverzani/CalculusWithJulia.jl
"""
function D(f, n::Int=1)
    n < 0 && throw(ArgumentError("n is a non-negative integer"))
    n == 0 && return f
    n == 1 && return t -> ForwardDiff.derivative(f, float(t))
    D(D(f), n-1)
end

"""
Library of radial potentials centered at 0
"""
V_Coulomb    = x -> - 1.0 / abs(x) 
V_Gigi(p)    = x -> - 1.0 / (√(p + x^2)) 
V_smeared(p) = x -> - 1.0 / (1.0 + p * abs(x))
V_harmonic   = x -> 1.0 * x^2

"""
Create potential centered at atom (z,R)
"""
function V_atom(V_rad,z,R)
    x -> z * V_rad(x - R)
end

"""
Discrete Hamiltonian - (1/2) Δ + V + σ
on finite difference grid
"""
Δ(n) = Tridiagonal(ones(n-1), -2.0*ones(n), ones(n-1))
Hd(δx, n, V, σ) = - (1/2) * (1/δx)^2 * Δ(n) + Diagonal(V .+ σ) 

"""
Golden section search for maximiser of f
"""
function maximize(ainit, binit, f; tol=1e-12)

    invphi   = (√(5) - 1) / 2  # 1 / phi
    invphi2  = (3 - √(5)) / 2  # 1 / phi^2
    hinit    = binit - ainit
   
    # calculate required steps to achieve tolerance
    n = trunc(Int, ceil(log(tol / hinit) / log(invphi)) )
    
    a, b, h = ainit, binit, hinit
    c = a + invphi2 * h
    d = a + invphi * h
  
    yc = f(c)
    yd = f(d)

    # divide interval [c,d] until desired tolerance is achieved
    for k in 1:n-1
        if yc > yd
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h     
            yc = f(c)
        else
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
        end
    end
    
    # result
    (c + d)*0.5
end


