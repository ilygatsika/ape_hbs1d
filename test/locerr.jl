"""
Solve eigenproblem for one or two atoms
"""

using Plots
using Crayons.Box

include("../src/finite_diff_solver.jl")
include("../src/basis.jl")
include("../src/utils.jl")

# PDE parameters
Nb_list      = 50  # bfs on each atom
z            = 1.0            # nuclear charge
a            = 2.0            # nuclear positions -a and +a
λ1           = 1.6            # local shift
λ            = 2*λ1           # global shift factor
Ng           = 2001           # number of df grid in box
box_size     = 5*a
ℓ            = 1.00           # partition of unity overlap is [-ℓ,ℓ]

println(MAGENTA_FG*BOLD("Nuclei distance "*string(2*a)))
println(RED_FG*BOLD("Overlap size "*string(2*ℓ)))

# Define potential
V0 = V_Gigi(z,0.5)            # centered at 0
V = sum_potential(V0, a)      # centered at - and +a
V1 = x -> V0(x-a)             # centered at +a

# Compute prefactor
C = prefactor(a, λ, λ1, ℓ, V0, p; graph=false)
println(CYAN_FG("prefactor "*string(C)))

# Define fine grid for FD reference solution
(x_range, δx) = discretize_space(Ng, box_size)

# Define Hamiltonian on grid
H1 = Hd(δx, Ng, V1.(x_range), λ1)   # centered at +a
H  = Hd(δx, Ng, V.(x_range), λ)     # centered at +a and -a

# Compute reference solution first eigenpair
vals, vecs = exact_eigensolver(Ng, V, λ, (x_range, δx))
μ_FD = vals[1]
u_FD = abs.(vecs[:,1])
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

#nb_tests = length(Nb_list)
#errH = zeros(nb_tests)
#estH = zeros(nb_tests)
#sv_rank = zeros(nb_tests)
#sv_min = zeros(nb_tests)
#Res_loc = zeros(nb_tests)
#Res_tot = zeros(nb_tests)


Nb = 50

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
   
# localisation of residual on fine grid
Res = μ_HB[1].*u_HB - H*u_HB
pRes = (p_eval .^ (1/2)) .* Res

# solve H1w=Res1 using df
w = exact_solution(Ng, V1, λ1, pRes, (x_range, δx))
#println("df residual is ", √((floc - H1*w)'*(floc - H1*w)))
    
Res_tot = H1*pRes

inds = (Res_tot .> 0)
plot(x_range[inds], abs.(Res_tot[inds]), yscale=:log10, xlabel="x", label="H1√pResN", legend = :topleft)

tests = [5,10,15,19]
for nvals in tests

    # spectral decomposition of H1
    dvals, dvecs = exact_eigensolver(Ng, V1, λ1, (x_range, δx), nvals=nvals)
    nn = length(dvecs[:,1])
    vec = zeros(nn)
    for j in 1:nvals
        new = (1.0/dvals[j] * (pRes'abs.(dvecs[:,j]))) * dvecs[:,j]
        for i in 1:nn
            vec[i] += new[i]
        end
    end
    Res_loc = vec

    plot!(x_range, abs.(Res_loc), yscale=:log10, xlabel="x", 
          label="dspectral on K=$(nvals) of H1", legend = :topleft)
end

savefig("img/Res_err_loc.png")

