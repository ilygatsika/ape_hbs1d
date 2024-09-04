"""
Solve eigenproblem for one or two atoms
"""

using Plots
using Crayons.Box

include("../src/finite_diff_solver.jl")
include("../src/basis.jl")
include("../src/utils.jl")

# PDE parameters
Nb_list      = Array(5:2:60)  # bfs on each atom
z            = 1.0            # nuclear charge
a            = 4.0            # nuclear positions -a and +a
λ1           = 1.6            # local shift
λ            = 2*λ1           # global shift factor
Ng           = 2001           # number of df grid in box
box_size     = 5*a
ℓ            = 1.80           # partition of unity overlap is [-ℓ,ℓ]

println(MAGENTA_FG*BOLD("Nuclei distance "*string(2*a)))
println(RED_FG*BOLD("Overlap size "*string(2*ℓ)))

# Define potential
V0 = V_Gigi(z,0.5)            # centered at 0
V = sum_potential(V0, a)      # centered at - and +a

# Compute prefactor
C = prefactor(a, λ, λ1, ℓ, V0, p; graph=false)
println(CYAN_FG("prefactor "*string(C)))

# Define fine grid for FD reference solution
(x_range, δx) = discretize_space(Ng, box_size)

# Define Hamiltonian on grid
V1 = x -> V0(x-a)                   # centered at +a
H1 = Hd(δx, Ng, V1.(x_range), λ1)   # centered at +a
H  = Hd(δx, Ng, V.(x_range), λ)     # centered at +a and -a

# Compute reference solution first eigenpair
vals, vecs = exact_eigensolver(Ng, V, λ, (x_range, δx))
μ_FD = vals[1]
u_FD = vecs[:,1]
P_FD = u_FD*u_FD'
@assert norm(P_FD*P_FD .- P_FD) < 1e-10

# Define global partition of unity on grid
p_eval = p_global(p, ℓ, x_range)

# Plot exact solution
plot(x_range, u_FD, linewidth=5, color="red", label="solution exacte")

# Test assumption 1 on shifts 
println("----------------------------")
println("Checking Assumption 1 on shifts")
println("L2 norm of Ψ ",   √(u_FD'u_FD)    )
println("H norm of Ψ ",    √(u_FD'H*u_FD)  )
println("Hloc norm of Ψ ", √(u_FD'H1*u_FD) )

nb_tests = length(Nb_list)
errH = zeros(nb_tests)
estH = zeros(nb_tests)
sv_rank = zeros(nb_tests)
sv_min = zeros(nb_tests)

for (i,Nb) in enumerate(Nb_list)

    print("\n")

    # Evaluate Hermite basis on FD grid
    HB = local_hermite_basis(a, Nb, Nb, x_range, δx)
    
    # Solve eigenproblem for Hermite basis
    S_HB = Symmetric(HB'HB)
    Mass_HB = Symmetric(HB'*(H*HB))    
    println("Nb=",2*Nb,"  basis condition ", cond(S_HB))
   
    # Treat lin dependencies: pivoted Cholesky
    PChol = cholesky(S_HB, RowMaximum(), check = false)
    u,s,v = svd(S_HB)
    #r = rank(S_HB, atol=1e-8, rtol=1e-8)
    r = rank(S_HB, atol=1e-6, rtol=1e-6)
    println("singular value ", s[r], " ", s[2*Nb])
    P = PChol.p[1:r]
    println("Cholesky cond ", cond(S_HB[P,P]))
    sv_rank[i] = s[r]
    sv_min[i] = s[2*Nb]

    # select lin independent basis functions
    S_HB = S_HB[P,P]
    Mass_HB = Mass_HB[P,P]

    μ_HB, C_HB = eigen(Mass_HB, S_HB)
    C_HB_rdc = zeros(2*Nb)
    C_HB_rdc[P] = C_HB[:,1]
    u_HB = abs.(HB * C_HB_rdc)
    P_HB = u_HB*u_HB'
    @assert norm(P_HB*P_HB .- P_HB) < 1e-10
    
    # Energy norm of error
    A_H = (u_FD - u_HB)'H*(u_FD - u_HB)
    println(GREEN_FG*BOLD("error H "*string(√A_H)))
    errH[i] = √(A_H)
     
    # plot solution approchée
    plot!(x_range, u_HB, label="$(2*Nb)")
    
    # localisation of residual on fine grid
    Res = μ_HB[1].*u_HB - H*u_HB
    floc = (p_eval .^ (1/2)) .* Res

    # solve local problem using df
    w = exact_solution(Ng, V1, λ1, floc, (x_range, δx))
    println("df residual is ", √((floc - H1*w)'*(floc - H1*w)))
    
    # dual norm squared equal to L2 (discrete) product of w and floc
    dual_norm = floc'w

    # Print estimator
    println(BLUE_FG*BOLD("estimator "*string(C * √(2 * dual_norm))))
    estH[i] = C * √(2 * dual_norm)

end

savefig("img/eig_sol_$(2*a).png")

plot(2 .* Nb_list, errH, xlabel="Nb Hermite", yscale=:log10, label="error H")
plot!(2 .* Nb_list, estH, xlabel="Nb Hermite", yscale=:log10, label="estimator")
savefig("img/eig_estimator_$(2*a).png")

plot(2 .* Nb_list, sv_rank, xlabel="Nb Hermite", yscale=:log10, label="sv_Chol")
savefig("img/eig_sval_$(2*a).png")

