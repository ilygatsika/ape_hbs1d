"""
Diatomic problem with right hand side
rhs = H1u1 + H2u2 for given u1 and u2
"""

using Plots
using Crayons.Box

include("../src/finite_diff_solver.jl")
include("../src/basis.jl")
include("../src/utils.jl")

# PDE parameters
Nb_list     = Array(10:5:60) # bfs on each atom
z           = 1.0 # nuclear charge
a           = 4.0 # nuclear positions -a and +a
λ1          = 1.5 # local shift
λ           = 2*λ1 # global shift factor
Ng          = 5001 # size of DF grid in box
box_size    = 5*a
ℓ           = 1.0 # partition of unity overlap is [-ℓ,ℓ]

# Plot options
plot_partition = false
plot_potential = false
plot_prefactor = false

println(MAGENTA_FG*BOLD("Nuclei distance "*string(2*a)))
println(RED_FG*BOLD("Overlap size "*string(2*ℓ)))

# Define finite difference grid
(x_range, δx) = discretize_space(Ng, box_size)

# Define potential
V0 = V_smeared(z,1.5)     # centered at 0
V = sum_potential(V0, a)  # centered at +a and -a
V1 = x -> V0(x-a)         # centered at +a
V2 = x -> V0(x+a)         # centered at -a

# Define Hamiltonian operator on grid
H1 = Hd(δx, Ng, V1.(x_range), λ1)
H2 = Hd(δx, Ng, V2.(x_range), λ1)
H  = Hd(δx, Ng, V.(x_range), λ) 

# Compute estimator prefactor
C = prefactor(a, λ, λ1, ℓ, V0, p; graph=plot_prefactor)
println(CYAN_FG("prefactor "*string(C)))

# Set right-hand side
Nb_ref = 1
HB1 = centered_hermite_basis(+a, Nb_ref, x_range, δx)
HB2 = centered_hermite_basis(-a, Nb_ref, x_range, δx)
f1 = abs.(H1*HB1)
f2 = abs.(H2*HB2)
rhs = f1[:,1] .+ f2[:,1]

# Compute exact solution to pb with rhs
uref = exact_solution(Ng, V, λ, rhs, (x_range, δx))

# Compute square root of partition of unity on grid
xx_ovlp = x_range[-ℓ .< x_range .<= ℓ]
n1 = length(x_range[x_range .>= ℓ])
n0 = length(x_range[x_range .< -ℓ])
pgrid = vcat(zeros(n0), p(ℓ).(xx_ovlp), ones(n1))

# Affichage graphique
if plot_partition
    plot(x_range, pgrid, label="partition")
end
if plot_potential
    plot(x_range, V.(x_range), label="V")
end
# exact solution
#title("Atomic distance $(2*a)")
plot(x_range, uref, linewidth=3)#, label="solution exacte")

println("----------------------------")
println("Test Assumption 1 on shifts")
println("L2 norm of uref   ", √(uref'uref))
println("H norm of uref    ", √(uref'H*uref))
println("Hloc norm of uref ", √(uref'H1*uref))

nb_tests = length(Nb_list)
errH = zeros(nb_tests)
estH = zeros(nb_tests)

for (i,Nb) in enumerate(Nb_list)

    print("\n")
    println("2*Nb = ", 2*Nb)
    
    # Compute Hermite basis solution for Nb
    HB = local_hermite_basis(a, Nb, Nb, x_range, δx)
    Mass_HB = Symmetric(HB'*(H*HB))
    println("mass cond ", cond(Mass_HB))

    # Remove lin dependencies with pivoted Cholesky
    PChol = cholesky(Mass_HB, RowMaximum(), check = false)
    r = rank(Mass_HB, atol=1e-8, rtol=1e-8)
    P = PChol.p[1:r]
    Mass_HB = Mass_HB[P,P]
    rhs_HB = (HB'*rhs)[P]
    println("Cholesky cond ", cond(Mass_HB))
    
    c_HB = Mass_HB \ rhs_HB
    c_HB_rdc = zeros(2*Nb)
    c_HB_rdc[P] = c_HB[:,1]
    u_HB = abs.(HB * c_HB_rdc)

    # Error in energy norm
    A_H = (uref - u_HB)'H*(uref - u_HB)
    println(GREEN_FG*BOLD("error H "*string(√(A_H))))
    errH[i] = √(A_H)

    # plot solution approchée
    #plot!(x_range, u_HB, label="$(2*Nb)")

    # localisation of residual on fine grid
    Res = rhs - H*u_HB
    pRes = (pgrid .^ (1/2)) .* Res

    # solve Hw=pRes
    w = exact_solution(Ng, V1, λ1, pRes, (x_range, δx))
    
    # dual norm of pRes
    dual_norm = pRes'w

    # final estimator
    estH[i] = C * √(2 * dual_norm)
    
    println(BLUE_FG*BOLD("estimator "*string(C * √(2 * dual_norm))))

end

savefig("img/rhs_sol_2")

plot(2 .* Nb_list, errH, markershape=:circle, 
    xlabel="Nb Hermite", yscale=:log10, label="error H")
plot!(2 .* Nb_list, estH, markershape=:star5,
    xlabel="Nb Hermite", yscale=:log10, label="estimator")
savefig("img/rhs_estimator_2")

