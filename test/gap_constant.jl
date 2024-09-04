include("../src/diatomic.jl")

"""
    parameter study on the gap constant
"""

# ###################
# System parameters
# ###################
R  = 1.0          # -R and +R atomic positions
z1 = 1.0          # atomic charge at -R
z2 = 1.0          # atomic charge at +R
V = V_Gigi(0.5)   # atomic potential
σ1 = 1.0          # shifts
σ2 = 1.0
σ∞ = 1.0
K  = 17          # size of spectral basis
ℓ  = 0.5

# shift factor
vec_σ = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
nb_σ = length(vec_σ)

# ###############################
# Discretisation basis parameters
# ###############################
Ng = 2001            # number of finite diff points
a  = 5 * R           # size of finite diff box

# #########################
# Main simulation
# #########################

mol = Molecule(R,z1,z2,V)
FD_grid = discretize_space(Ng, a)

# Vary shift
for i in 1:nb_σ

    # compute partitions on domains
    σ = vec_σ[i]
    cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)

    # domain constant
    c = subdomain_constant((Ω, Ω1, Ω2, Ω∞), ℓ, cH, FD_grid)
    
    # reference solution of eigenproblem
    μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)
    
    ratio = μ1_FD/μ2_FD
    
    println("============\nσ=($σ) ratio=($ratio) λ2=($μ2_FD) cH=($cH)")

end

