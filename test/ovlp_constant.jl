include("../src/diatomic.jl")

"""
    Parameter study on overlap constant
"""

# ###################
# System parameters
# ###################
R  = 1.0          # -R and +R atomic positions
z1 = 1.0          # atomic charge at -R
z2 = 1.0          # atomic charge at +R
V = V_Gigi(0.5)   # atomic potential
σ  = 4.0          # shifts
σ∞ = 1.0
K  = 17          # size of spectral basis

# shift factor
vec_σ1 = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
nb_σ1 = length(vec_σ1)

# partition overlap size is 2*ℓ
vec_ℓ = [0.1, 0.3, 0.5, 0.8, 0.9]
nb_ℓ = length(vec_ℓ)

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
for i in 1:nb_σ1

    # Vary size of partition of overlap
    for j in 1:nb_ℓ

        # compute partitions on domains
        σ1 = σ2 = vec_σ1[i]
        ℓ = vec_ℓ[j]
        cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)

        # domain constant
        c = subdomain_constant((Ω, Ω1, Ω2, Ω∞), ℓ, cH, FD_grid)
        println("σ1=($σ1) ℓ=($ℓ) cH=($cH) C=($c)")

    end
end

