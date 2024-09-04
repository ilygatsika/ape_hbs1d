using JSON3

include("../src/diatomic.jl")

"""
    Adaptive Hermite basis sets for diatomic molecule 
    for eigenvalue problem in 1D.
    Reference solution computed using finite differences.
"""

output_dir = "out"

# ###################
# System parameters
# ###################
R  = 1.0          # -R and +R atomic positions
z1 = 1.0          # atomic charge at -R
z2 = 1.0         # atomic charge at +R
V = V_Gigi(0.5)   # atomic potential
σ  = 7.0          # shifts
σ1 = 4.0
σ2 = 4.0
σ∞ = 1.0
K  = 17          # size of spectral basis
ℓ  = 0.3         # size of partition overlap

# ###############################
# Discretisation basis parameters
# ###############################
Ng = 2001            # number of finite diff points
a  = 5 * R           # size of finite diff box
Nb_min, Nb_max = 5, 30

# #########################
# Main simulation
# #########################

mol = Molecule(R,z1,z2,V)
FD_grid = discretize_space(Ng, a)

# partition on subdomains
cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)
c = subdomain_constant((Ω, Ω1, Ω2, Ω∞), ℓ, cH, FD_grid)

# reference solution of eigenproblem
μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)

# Vary Hermite basis size
Nb_list = Array(5:2:Nb_max) # Hermite basis size
nb_tests = length(Nb_list)
Nb1_list, Nb2_list = Vector{Int64}(), Vector{Int64}()
Herr, Hest = Vector{Float64}(), Vector{Float64}()

# init
Nb1, Nb2 = 5, 5
while (Nb1 + Nb2 < 2*Nb_max)

    global Nb1, Nb2

    λ_1N, u_1N = hermite_eigensolver(mol, Ω.H, Nb1, Nb2, FD_grid)
    
    err = u_1N - u_FD
    Herrv = √(err'Ω.H*err)
    @assert( Herrv >= √(err'err) )

    Res = λ_1N * u_1N - Ω.H * u_1N
    dnorm_Res = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid)
    c1 = gap_constant_1(μ2_FD, λ_1N)
    c2 = gap_constant_2(μ2_FD, λ_1N)
    estv = estimator_eigenvector(c, c1, c2, μ1_FD, dnorm_Res)
    
    println("($Nb1) ($Nb2) ($Herrv) ($estv)")

    # increment Hermite basis size
    if dnorm_Res[1] > dnorm_Res[2]
        Nb1 = Nb1 + 2
    else
        Nb2 = Nb2 + 2
    end
    
    # Store results
    push!(Nb1_list, Nb1)
    push!(Nb2_list, Nb2)
    push!(Herr, Herrv)
    push!(Hest, estv)

end

# non adaptive basis
Herr_na, Hest_na = zeros(nb_tests), zeros(nb_tests)
for i in 1:nb_tests
    
    Nb = Nb_list[i]
    λ_1N, u_1N = hermite_eigensolver(mol, Ω.H, Nb, Nb, FD_grid)
    err = u_1N - u_FD
    Herrv = √(err'Ω.H*err)
    @assert( Herrv >= √(err'err) )
    Res = λ_1N * u_1N - Ω.H * u_1N
    dnorm_Res = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid)
    c1 = gap_constant_1(μ2_FD, λ_1N)
    c2 = gap_constant_2(μ2_FD, λ_1N)
    estv = estimator_eigenvector(c, c1, c2, μ1_FD, dnorm_Res)
    println("($Nb) ($Nb) ($Herrv) ($estv)")
    Herr_na[i], Hest_na[i] = Herrv, estv

end
# ##################
# Export results
# ##################

data = Dict{String, Any}()
# adaptive results
data["Nb12_list"] = Nb1_list .+ Nb2_list
data["Herr_eig"] = Herr
data["Hest_eig"] = Hest
# non adaptive results
data["Nb_list"] = 2 .* Nb_list
data["Herr_eig_na"] = Herr_na
data["Hest_eig_na"] = Hest_na

# write to file
open(io -> JSON3.write(io, data, allow_inf=true), "$(output_dir)/res_adapt.json", "w")


