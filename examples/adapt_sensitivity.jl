include("../common.jl")

"""
    Study parameter effect on the performance of the adaptive AO 
    refinement - essentially on the atomic error indicator for fixed Nb.
    Parameters under variation:
        ℓ, σ, σ1, K
"""

# TODO store to file maybe as in sensitiviy.jl

# ###################
# System parameters
# ################### 
z1 = 1.0                # atomic charge at -R
z2 = 1.0                # atomic charge at +R
R  = 1.0                # -R and +R atomic positions
V = V_Gigi(0.5)         # atomic potential
σ = 4.0                 # shifts
σ1 = 3.0
σ2 = 3.0
σ∞ = 1.0
ℓ = 0.3
K = 17                  # size of spectral basis

# ###############################
# Discretisation basis parameters
# ###############################
Ng = 2001                    # number of finite diff points
a  = 5 * R                   # size of finite diff box
vec_Nb = [5,10,15,20,25,30]  # number of AO basis per atom 
nb_Nb = length(vec_Nb)

# ###############################
# Pool of parameter values
# ###############################

vec_ℓ = [0.3,1.0,1.8]   # size of partition overlap
vec_σ = [3.0,5.0,8.0]   # spectral shift for molecule
vec_σ1 = [5.0,2.0,1.0]  # atomic spectral shifts
vec_K = [5,10,17]       # size of spectral basis

nb_ℓ = length(vec_ℓ)
nb_K = length(vec_K)
nb_σ = length(vec_σ)
nb_σ1 = length(vec_σ1)
nb_K = length(vec_K)

# #########################
# Main simulation
# #########################

mol = Molecule(R,z1,z2,V)
FD_grid = discretize_space(Ng, a)
Nb2 = 5 # AO number fixed for second atom

function reset_params()
    σ = 4.0
    σ1 = 3.0
    σ2 = 3.0
    ℓ = 0.3
    K = 17
end

# Function to test for parameter sensitivity
function error_indicator(Nb1,ℓ,σ,σ1,σ2,K)
    # make atomic domains
    cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)
    # solve einproblem using Nb1+Nb2
    λ_1N, λ_2N, u_1N = hermite_eigensolver(mol, Ω.H, Nb1, Nb2, FD_grid)
    # compute residual and its norm
    Res = λ_1N * u_1N - Ω.H * u_1N
    dnorm_Res = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid)
    return (dnorm_Res[1],dnorm_Res[2])
end

# For fixed Nb size, try one pool of parameters
# keeping the other fixed

println("\nParameter sensitivity analysis")
println("When not varied:")
println("fixed ℓ=$ℓ")
println("fixed σ=$σ")
println("fixed σ1=$σ1")
println("fixed σ2=$σ2")
println("fixed K=$K")
println("fixed Nb2=$Nb2")
println("\n")

# Vary ℓ
reset_params()
for i in 1:nb_Nb
    Nb = vec_Nb[i]
    for j in 1:nb_ℓ
        local ℓ = vec_ℓ[j]
        eta1,eta2 = error_indicator(Nb,ℓ,σ,σ1,σ2,K)
        println("Nb1=$Nb ℓ=$ℓ $eta1 $eta2")
    end
end
println("\n")

# Vary σ
reset_params()
for i in 1:nb_Nb
    Nb = vec_Nb[i]
    for j in 1:nb_σ
        local σ = vec_σ[j]
        eta1,eta2 = error_indicator(Nb,ℓ,σ,σ1,σ2,K)
        println("Nb1=$Nb σ=$σ $eta1 $eta2")
    end
end
println("\n")

# Vary σ1=σ2
reset_params()
for i in 1:nb_Nb
    Nb = vec_Nb[i]
    for j in 1:nb_σ1
        local σ1 = vec_σ1[j]
        eta1,eta2 = error_indicator(Nb,ℓ,σ,σ1,σ1,K)
        println("Nb1=$Nb σ1=$σ1 $eta1 $eta2")
    end
end
println("\n")

# Vary K
reset_params()
for i in 1:nb_Nb
    Nb = vec_Nb[i]
    for j in 1:nb_K
        local K = vec_K[j]
        eta1,eta2 = error_indicator(Nb,ℓ,σ,σ1,σ2,K)
        println("Nb1=$Nb K=$K $eta1 $eta2")
    end
end
println("\n")





