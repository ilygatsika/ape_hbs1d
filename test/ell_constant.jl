using PrettyTables

include("../src/diatomic.jl")

"""
    Study partition constant
"""

# Settings
R = 1.0
z1 = z2 = 1.0
V = V_Gigi(0.5) 
σ∞ = 1.0
Ng = 2001
a  = 5 * R
K = 1

# Game of parameters for overlap and shift
vec_ℓ = [0.1, 0.3, 0.5, 0.7, 0.9]
vec_σ = [3.0] # [5.0, 4.0, 2.0]
vec_σ1 = [1.0] # [3.0, 2.0, 1.0]
nb_ℓ = length(vec_ℓ)
nb_σ = length(vec_σ)

# Define molecule
mol = Molecule(R,z1,z2,V)
FD_grid = discretize_space(Ng, a)
vec_c = zeros(nb_ℓ,nb_σ)

for i in 1:nb_ℓ

    ℓ = vec_ℓ[i]

    for j in 1:nb_σ

        σ, σ1 = vec_σ[j], vec_σ1[j]
        σ2 = σ1

        println("$(ℓ), $(σ), $(σ1), $(σ2)")

        subdomains = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)
        f = "partition_$(i)_$(j)"
        #c = subdomain_constant(subdomains, ℓ, FD_grid; with_plot=true, filename=f)
        #vec_c[i,j] = c
    end
end

#data = hcat(vec_ℓ, vec_c)
#pretty_table(data; header=["ℓ", "5,3,3", "4,2,2", "2,1,1"])

