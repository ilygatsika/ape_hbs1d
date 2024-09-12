module ape_hbs1d

using PyPlot
using LaTeXStrings

include("basis.jl")
include("utils.jl")
include("finite_diff_solver.jl")

"""
    Collects routines for diatomic molecules. 
    It basically combines other routines of the present code to 
    calculate estimators. Also introduces useful object structures
    to facilitate the usage of this code.

    This code is responsible for:

        - domain decomposition into subdomains
        - partition of unity on subdomains
        - wrap reference solvers using finite differences
        - wrap solvers on Hermite basis
        - spectral decomposition of atomic operators
        - computation of dual norm of residual
"""

struct Molecule
    R::Float64    # atoms at -R (atom 1) and +R (atom 2)
    z1::Float64   # nuclear charge of atom 1
    z2::Float64   # nuclear charge of atom 2
    V::Function   # nuclear potential centered at 0
end

struct Subdomain
    σ::Float64             # shift associated to subdomain
    pfun                   # analytic partition function
    H                      # Hamiltonian on grid
    V::Function            # analytic centered potential
    modes                  # spectral basis
end

# Function p is increasing on [a,b], =0 at x=a, =1 at x=b
h(x) = exp( - 1.0 / x )
p(a,b) = x -> h(x - a) / (h(x - a) + h(b - x))

"""
Partition function supported on (a,d)
equal to 1 on (b,c) ⊂ (a,b)
"""
function p_nuc(x, a, b, c, d)
    val = -1.0
    if (x ≤ a) || (d ≤ x)
        val = 0.0
    elseif a < x < b
        # increasing
        val = p(a,b)(x)
    elseif b ≤ x ≤ c
        val = 1.0
    elseif c < x < d
        # decreasing
        val = 1.0 - p(c,d)(x)
    end
    val
end

"""
Partition function supported at infinity
this is related to the screened Laplacian operator
"""
function p_inf(x, a, b, c, d)
    val = -1.0
    if (x ≤ a) || (d ≤ x)
        val = 1.0
    elseif b ≤ x ≤ c
        val = 0.0
    elseif a < x < b
        # decreasing
        val = 1.0 - p(a,b)(x)
    elseif c < x < d
        # increasing
        val = p(c,d)(x)
    end
    val 
end

# Reflect a across point pt
reflect(a,pt) = 2*pt - a

function subdomain_constant(subdomains, ℓ, cH, grid; with_plot=false, filename=false)

    Ω, Ω1, Ω2, Ω∞ = subdomains
    σ = Ω.σ
    x_range, δx = grid

    # Treat zeros in the denominator
    eps = 1E-14
    p1nz = x -> Ω1.pfun(x) < eps ? 1 : Ω1.pfun(x)
    p2nz = x -> Ω2.pfun(x) < eps ? 1 : Ω2.pfun(x)
    p∞nz = x -> Ω∞.pfun(x) < eps ? 1 : Ω∞.pfun(x)

    # Define function to maximize
    F = x -> max(- (1/2) * D(Ω1.pfun,2)(x) + D(Ω1.pfun,1)(x) ^ 2 / (4.0 * p1nz(x)) + 
                Ω1.V(x) * (Ω1.pfun(x) - 1.0) + (Ω1.σ - σ) * Ω1.pfun(x) - 
                 (1/2) * D(Ω2.pfun,2)(x) + D(Ω2.pfun,1)(x) ^ 2 / (4.0 * p2nz(x)) + 
                Ω2.V(x) * (Ω2.pfun(x) - 1.0) + (Ω2.σ - σ) * Ω2.pfun(x) - 
                 (1/2) * D(Ω∞.pfun,2)(x) + D(Ω∞.pfun,1)(x) ^ 2 / (4.0 * p∞nz(x)) + 
                Ω∞.V(x) * (Ω∞.pfun(x) - 1.0) + (Ω∞.σ - σ) * Ω∞.pfun(x), 
                0.0)

    # Maximize assuming symmetry wrt origin
    a, b = x_range[1], last(x_range) 
    x1 = maximize(a,-ℓ,F)
    x2 = maximize(-ℓ,0,F)
    x3 = maximize(0,ℓ,F)
    x4 = maximize(ℓ,b,F)
    v = max(F(x1), F(x2), F(x3), F(x4))

    if with_plot
        PyPlot.title(L"$\sigma=%$(σ), \sigma_1=\sigma_2=%$(Ω1.σ)$")
        PyPlot.plot(x_range, F.(x_range))
        PyPlot.vlines(-ℓ, 0, v, linestyles ="dotted", colors ="k")
        PyPlot.vlines(ℓ, 0, v, linestyles ="dotted", colors ="k")
        PyPlot.text(-ℓ, 1, L"$x=-ℓ$", rotation=-90)
        PyPlot.text(ℓ, 1, L"$x=+ℓ$", rotation=-90)
        PyPlot.savefig("img/$(filename).png")
        PyPlot.close()
    end

    1.0 + cH^2 * v
end

# Gap constants for first eigenvalue
gap_constant_1(λ2, λ1N) = (1.0 - λ1N / λ2)^2
gap_constant_2(λ2, λ1N) = (1.0 - λ1N / λ2)^2 * λ2

function init_subdomains_omega(mol, σ, K, Ng, grid)

    x_range,δx = grid
    
    # potentials as functions of x
    VΩ1 = V_atom(mol.V, mol.z1, +mol.R)
    VΩ2 = V_atom(mol.V, mol.z2, -mol.R)
    VΩ(x) = VΩ1(x) + VΩ2(x)
   
    # Hamiltonians on grid
    HΩ = SymTridiagonal(Hd(δx, Ng, VΩ.(x_range), σ))

    # Check positive-definiteness
    @assert(isposdef(HΩ))
  
    cH = 1/eigvals(HΩ)[1] # constant of Assumption 3
    Ω  = Subdomain(σ , x_range, HΩ , VΩ , nothing)

    (cH, Ω)

end

function init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, grid)

    x_range,δx = grid
    
    # potentials as functions of x
    VΩ1 = V_atom(mol.V, mol.z1, +mol.R)
    VΩ2 = V_atom(mol.V, mol.z2, -mol.R)
    VΩ(x) = VΩ1(x) + VΩ2(x)
    VΩ∞(x) = 0 
   
    # analytical partition of unity    
    μ̄ = reflect(ℓ, mol.R)
    μ = reflect(-ℓ, mol.R)
    pΩ1 = x -> p_nuc(x, -ℓ, ℓ, μ̄, μ)
    pΩ2 = x -> p_nuc(x, -μ, -μ̄, -ℓ, ℓ)
    pΩ∞ = x -> p_inf(x, -μ, -μ̄, μ̄, μ)

    # Hamiltonians on grid
    HΩ = SymTridiagonal(Hd(δx, Ng, VΩ.(x_range), σ))
    HΩ1 = SymTridiagonal(Hd(δx, Ng, VΩ1.(x_range), σ1))
    HΩ2 = SymTridiagonal(Hd(δx, Ng, VΩ2.(x_range), σ2))
    HΩ∞ = Hd(δx, Ng, VΩ∞.(x_range), σ∞)

    # Check positive-definiteness
    @assert(isposdef(HΩ))
    @assert(isposdef(HΩ1))
    @assert(isposdef(HΩ2))
    @assert(isposdef(HΩ∞))
  
    # Plot partition
    PyPlot.plot(x_range, pΩ1.(x_range), label=L"$p_1$")
    PyPlot.plot(x_range, pΩ2.(x_range), label=L"$p_2$")
    PyPlot.plot(x_range, pΩ∞.(x_range), label=L"$p_∞$")
    PyPlot.legend()
    PyPlot.savefig("img/part.png")
    PyPlot.close()


    # initialize K-spectral basis
    modes1 = spectral_basis(mol, K, VΩ1, σ1, grid)
    modes2 = spectral_basis(mol, K, VΩ2, σ2, grid)
    
    λ,v = modes1
    #println("spectral ε $(λ[1]) $(λ[K])")

    cH = 1/eigvals(HΩ)[1] # constant of Assumption 3
    Ω  = Subdomain(σ , x_range, HΩ , VΩ , nothing)
    Ω1 = Subdomain(σ1, pΩ1    , HΩ1, VΩ1, modes1 )
    Ω2 = Subdomain(σ2, pΩ2    , HΩ2, VΩ2, modes2 )
    Ω∞ = Subdomain(σ∞, pΩ∞    , HΩ∞, VΩ∞, nothing)

    (cH, Ω, Ω1, Ω2, Ω∞)

end

"""
Built test source problem on subdomain
Return exact solution and rhs evaluated on grid
"""
function test_source_pb(mol, Ω, Ω1, Ω2, Ng, grid)

    # build rhs of source problem
    Nb_ref = 1
    HB1 = centered_hermite_basis(+mol.R, Nb_ref, grid)
    HB2 = centered_hermite_basis(-mol.R, Nb_ref, grid)
    f1 = abs.(Ω1.H * HB1)
    f2 = abs.(Ω2.H * HB2)
    rhs = f1[:,1] .+ f2[:,1]
    
    # exact solution for source problem
    sol = exact_solution(Ng, Ω.V, Ω.σ, rhs, grid)
    
    (sol, rhs)

end

"""
Solve eigenproblem on subdomain
Return exact solution evaluated on grid
and first two eigenvalues
"""
function test_eigenpb(mol, Ω, Ng, grid)
    
    # exact solution for eigenvalue problem
    λ, v = exact_eigensolver(Ng, Ω.V, Ω.σ, grid)
    #println("test", λ)
    u_FD = abs.(v[:,1])
    P_FD = u_FD*u_FD'

    # check normalization
    @assert norm(P_FD*P_FD .- P_FD) < 1e-10
   
    (λ[1], λ[2], u_FD)

end

"""
Return normalized spectral basis of size K 
for operator evaluated on grid
"""
function spectral_basis(mol, K, V, σ, grid)

    λ,v = exact_eigensolver(Ng, V, σ, grid, nvals=K)
    x_range, δx = grid
    for i in 1:K
        v[:,i] /= √(δx * v[:,i]'v[:,i])
        v[:,i] *= √δx
        @assert norm(v[:,i]'v[:,i] .- 1.) < 1e-10
    end
    
    (λ,v)

end

"""
Hermite basis solver for source problem
with Nb1 and Nb2 HB functions on atoms 1 and 2.
Return solution evaluated on grid
"""
function hermite_solver(mol, H, Nb1, Nb2, rhs, grid)

    # create Hermite basis
    HB1 = centered_hermite_basis(+mol.R, Nb1, grid)
    HB2 = centered_hermite_basis(-mol.R, Nb2, grid)
    HB = hcat(HB1, HB2)

    # mass matrix (possibly ill-conditioned)
    Mass_HB = Symmetric(HB'*(H*HB))
    
    #println("mass cond ", cond(Mass_HB))

    # Remove lin dependencies with pivoted Cholesky
    PChol = cholesky(Mass_HB, RowMaximum(), check = false)
    r = rank(Mass_HB, atol=1e-8, rtol=1e-8)
    P = PChol.p[1:r]

    # reduce mass and rhs
    Mass_HB = Mass_HB[P,P]
    rhs_HB = (HB'*rhs)[P]

    #println("Cholesky cond ", cond(Mass_HB))
    
    # solve problem with rhs
    c_HB = Mass_HB \ rhs_HB
    c_HB_rdc = zeros(Nb1 + Nb2)
    c_HB_rdc[P] = c_HB[:,1]
    u_HB = abs.(HB * c_HB_rdc)

    u_HB

end

"""
Hermite basis solver for eigenproblem 
with Nb1 and Nb2 HB functions on atoms 1 and 2.
Return first eigenvector evaluated on grid if nv=1
otherwise nv=2 returns first and second eigenvalue only (no vecs).
"""
function hermite_eigensolver(mol, H, Nb1, Nb2, grid; nv=1)
    
    # create Hermite basis
    HB1 = centered_hermite_basis(+mol.R, Nb1, grid)
    HB2 = centered_hermite_basis(-mol.R, Nb2, grid)
    HB = hcat(HB1, HB2)

    # mass matrix (possibly ill-conditioned)
    Mass_HB = Symmetric(HB'*(H*HB))
    
    # overlap matrix
    S_HB = Symmetric(HB'HB)
    
    #println("mass cond ", cond(Mass_HB))

    # Remove lin dependencies with pivoted Cholesky
    PChol = cholesky(Mass_HB, RowMaximum(), check = false)
    r = rank(Mass_HB, atol=1e-8, rtol=1e-8)
    P = PChol.p[1:r]

    # reduce mass and overlap
    S_HB = S_HB[P,P]
    Mass_HB = Mass_HB[P,P]

    #println("Cholesky cond ", cond(Mass_HB))
    
    # solve eigenproblem
    μ_HB, C_HB = eigen(Mass_HB, S_HB)
    C_HB_rdc = zeros(Nb1 + Nb2)
    C_HB_rdc[P] = C_HB[:,1]
    u_1N = abs.(HB * C_HB_rdc)

    if (nv == 1)
        (μ_HB[1], u_1N)
    elseif (nv == 2)
        μ_HB[1:2]
    end

end

"""
Compute A-dual norm of u using K-spectral basis of A
"""
function dual_norm_spectral(modes, u; inf=0)

    λ, v = modes
    K = size(v,2)

    sL,sU = 0,0
    for i in 1:K-1
        s = (u'abs.(v[:,i]))^2
        sU += s
        sL += 1.0/λ[i] * s
    end

    dnorm = sL + 1.0/λ[K] * (u'u - sU)

    if (inf == 1)
        return (sL, dnorm)
    end

    return dnorm

end

"""
Decompose dual norm of w on subdomains
"""
function decompose_dual_norm(w, Ω1, Ω2, Ω∞, Ng, grid; inf=0)
    
    x_range, δx = grid

    # Multiply w by partition functions
    wΩ1 = (Ω1.pfun.(x_range) .^ (1/2)) .* w
    wΩ2 = (Ω2.pfun.(x_range) .^ (1/2)) .* w
    wΩ∞ = (Ω∞.pfun.(x_range) .^ (1/2)) .* w

    # Compute dual norms using spectral basis
    dnorm_Ω1 = dual_norm_spectral(Ω1.modes, wΩ1)
    dnorm_Ω2 = dual_norm_spectral(Ω2.modes, wΩ2)

    # Explicitely invert on Ω∞
    u∞ = exact_solution(Ng, Ω∞.V, Ω∞.σ, wΩ∞, grid)
    dnorm_Ω∞ = wΩ∞'u∞

    if (inf == 1)
        
        dnorm_Ω1_inf, dnorm_Ω1_sup = dual_norm_spectral(Ω1.modes, wΩ1, inf=1)
        # explicitely invert on Ω1
        u1 = exact_solution(Ng, Ω1.V, Ω1.σ, wΩ1, grid)
        dnorm_Ω1 = wΩ1'u1
        return (dnorm_Ω1_inf, dnorm_Ω1, dnorm_Ω1_sup)
    end

    # Return dual norm squared on subdomains
    return (dnorm_Ω1, dnorm_Ω2, dnorm_Ω∞)

end

function estimator_source_pb(c, dnorm_Res)
    r = sum(dnorm_Res)
    √(c * r)
end

function estimator_eigenvector(c, c1, c2, μ1_FD, dnorm_Res) 
    r = sum(dnorm_Res)
    √(c * 1.0/c1 * r + μ1_FD * c^2 * 1.0/c2^2 * r^2)
end

function estimator_eigenvalue(c, c1, dnorm_Res) 
    r = sum(dnorm_Res)
    (c * 1.0/c1 * r)
end

end # module

