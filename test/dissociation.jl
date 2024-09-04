"""
Solve eigenproblem for one or two atoms
"""

using Plots
using Crayons.Box

include("../src/finite_diff_solver.jl")
include("../src/basis.jl")
include("../src/utils.jl")

# PDE parameters
Nb_list = Array(10:10:60)    # AO basis size is 2*Nb
z = 1.0                    # nuclear charge
a = 2.0                    # nuclear positions -a and +a
λ1 = 1.6                   # local shift
λ = 2*λ1                   # global shift factor
Ng = 2001                  # number of df grid in box
box_size = 5*a
ℓ = 1.80                    # partition of unity overlap is [-ℓ,ℓ]

println(MAGENTA_FG*BOLD("Nuclei distance "*string(2*a)))
println(RED_FG*BOLD("Overlap size "*string(2*ℓ)))

# Define potential
V0 = V_Gigi(z,0.5)            # centered at 0
V = sum_potential(V0, a)      # centered at - and +a
V1 = x -> V0(x-a)             # centered at +a

# Define Hamiltonian on grid
V1 = x -> V0(x-a)                   # centered at +a
H1 = Hd(δx, Ng, V1.(x_range), λ1)   # centered at +a
H  = Hd(δx, Ng, V.(x_range), λ)     # centered at +a and -a

# Compute prefactor
C = prefactor(a, λ, λ1, ℓ, V_rad, p; graph=false)
println(CYAN_FG("prefactor "*string(C)))

# Define fine grid for FD reference solution
(x_range, δx) = discretize_space(Ng, box_size)

# Compute reference solution first eigenpair
vals, vecs = exact_eigensolver(Ng, V, λ, (x_range, δx))
μ_FD = vals[1]
u_FD = vecs[:,1]
P_FD = u_FD*u_FD'
@assert norm(P_FD*P_FD .- P_FD) < 1e-10

# Compute dissociation
V1 = x -> V_rad(x-a)
vals_atm1, vecs_atm1 = exact_eigensolver(Ng, V1, λ1, (x_range,δx))
μ1 = vals_atm1[1]
u1 = vecs_atm1[:,1]
V2 = x -> V_rad(x+a)
vals_atm2, vecs_atm2 = exact_eigensolver(Ng, V2, λ1, (x_range,δx))
μ2 = vals_atm2[1]
u2 = vecs_atm2[:,1]
u_FD_diss = u1 + u2
μ_diss_FD = μ1 + μ2

nb_tests = length(Nb_list)
errH = zeros(nb_tests)
estH = zeros(nb_tests)

for (i,Nb) in enumerate(Nb_list)

    print("\n")

    # Dissociation
    HB1 = centered_hermite_basis(a, Nb, x_range, δx)
    HB2 = centered_hermite_basis(-a, Nb, x_range, δx)
    S_HB1 = Symmetric(HB1'HB1)
    Mass_HB1 = Symmetric(HB1'*(H1*HB1))
    S_HB2 = Symmetric(HB2'HB2)
    Mass_HB2 = Symmetric(HB2'*(H2*HB2))
    μ_HB1, C_HB1 = eigen(Mass_HB1, S_HB1)
    μ_HB2, C_HB2 = eigen(Mass_HB2, S_HB2)
    μ_diss = μ_HB1[1] + μ_HB2[1]
    
    # Evaluate Hermite basis on FD grid
    HB = local_hermite_basis(a, Nb, Nb, x_range, δx)
    
    # Solve eigenproblem for Hermite basis
    S_HB = Symmetric(HB'HB)
    Mass_HB = Symmetric(HB'*(H*HB))
    println("Nb=",2*Nb,"  basis condition ", cond(S_HB))
    
    # Alternative: pivoted Cholesky
    PChol = cholesky(S_HB, RowMaximum(), check = false)
    r = rank(S_HB)
    P = PChol.p[1:r]
    println("Cholesky cond ", cond(S_HB[P,P]))

    # select lin independent basis functions
    S_HB = S_HB[P,P]
    Mass_HB = Mass_HB[P,P]

    μ_HB, C_HB = eigen(Mass_HB, S_HB)
    μ_HB, C_HB = eigen(Mass_HB, S_HB)
    C_HB_rdc = zeros(2*Nb)
    C_HB_rdc[P] = C_HB[:,1]
    u_HB = abs.(HB * C_HB_rdc)
    P_HB = u_HB*u_HB'
    @assert norm(P_HB*P_HB .- P_HB) < 1e-10
    
    # Energy norm of error
    errH[i] = abs((μ_diss_FD - μ_FD) - (μ_diss - μ_HB[1]))
    println(GREEN_FG*BOLD("error energy "*string(errH[i])))    

    # localisation of residual on fine grid
    Res = μ_HB[1].*u_HB - H*u_HB
    floc = (p_eval .^ (1/2)) .* Res

    # solve local problem using df
    w = exact_solution(Ng, V1, λ1, floc, (x_range, δx))
    println("df residual is ", √((floc - H1*w)'*(floc - H1*w)))
    
    # dual norm squared equal to L2 (discrete) product of w and floc
    dual_norm = floc'w

    # Print estimator
    println(BLUE_FG*BOLD("estimator "*string(C^2 * 2 * dual_norm)))
    estH[i] = C^2 * 2 * dual_norm
 
end

plot(2 .* Nb_list, errH, xlabel="Nb Hermite", yscale=:log10, label="error H")
plot!(2 .* Nb_list, estH, xlabel="Nb Hermite", yscale=:log10, label="estimator")
savefig("img/eigval_estimator_$(2*a).png")



