include("../common.jl")

"""
    Parameter sensitivity analysis of
        1- gap constant (γ)
        2- coercivity constant (cA)
    for varying values of global shift (σ), and

        3- partition overlap constant (C)
    for varying values of overlap width (ℓ) and atomic shifts (σ1=σ2).
        
    This code produces a JSON file and/or prints results in the terminal.
"""

if !isfile("$(output_dir)/sensitivity.json")

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

    # Pool of values
    vec_ℓ = [0.1, 0.3, 0.5, 0.8, 0.9]       # overlap width
    vec_σ = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # shift factor
    vec_σ1 = [5.0, 3.0, 2.0, 1.0]           # atomic shift
    nb_ℓ = length(vec_ℓ)
    nb_σ = length(vec_σ)
    nb_σ1 = length(vec_σ1)

    # ###############################
    # Discretisation basis parameters
    # ###############################
    Ng = 2001            # number of finite diff points
    a  = 5 * R           # size of finite diff box
    Nb_list = [5,30,60]  # Hermite basis size

    # #########################
    # Main simulation
    # #########################

    mol = Molecule(R,z1,z2,V)
    FD_grid = discretize_space(Ng, a)
    nb_tests = length(Nb_list)

    # Vary shift
    vec_cA = zeros(nb_σ)
    vec_γ = zeros(nb_σ)
    for i in 1:nb_σ

        # compute partitions on domains
        local ℓ = 0.5 # fix the overlap with
        local σ = vec_σ[i]
        local cA, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)

        # reference solution of eigenproblem
        local μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)
    
        # approximate solution with basis size Nb 
        for j in 1:nb_tests

            Nb = Nb_list[j]
            println("\n============ N=$Nb")
            λ_1N, λ_2N, v_1N = hermite_eigensolver(mol, Ω.H, Nb, Nb, FD_grid)

            println("Verify: |λ2 - λ2N|=$(abs(μ2_FD-λ_2N))")

            # gap constant here = (1.0 - λ1N / μ2_FD)^2
            local γ = gap_constant_1(μ2_FD, λ_1N)

            # not used for analysis here but just for comparison
            cA_prac = 1/sqrt(minimum((Ω.V).(FD_grid[1])) + σ) # practical for H2+
            println("For σ=$σ | γ=$γ exact cA=$(cA) practical cA=$(cA_prac)")

            if ( j == nb_tests ) # saves result for larger Nb
                vec_γ[i] = γ
                vec_cA[i] = cA
            end
        end
    end

    # Vary overlap and atomic shift
    vec_C = zeros(nb_ℓ,nb_σ1)
    for i in 1:nb_ℓ

        local ℓ = vec_ℓ[i]
        local σ = 4

        for j in 1:nb_σ1
    
            local σ1 = vec_σ1[j]
            local σ2 = vec_σ1[j]
    
            local cA, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)
            local C = subdomain_constant((Ω, Ω1, Ω2, Ω∞), ℓ, cA, FD_grid)
    
            vec_C[i,j] = C
            println("For σa=$σ1, ℓ=$ℓ | C=$C\n============")

        end
    end

    # ##################
    # Export results
    # ##################

    data = Dict{String, Any}()
    data["ℓ"] = vec_ℓ
    data["σ"] = vec_σ
    data["σ1"] = vec_σ1
    data["γ_cst"] = vec_γ      # gap constant, varying σ
    data["C_cst"] = vec_C      # partition constant, varying σ1, ℓ 
    data["cA_cst"] = vec_cA    # coercivity constant, varying σ

    # write to file
    open(io -> JSON3.write(io, data, allow_inf=true), "$(output_dir)/sensitivity.json", "w")

end

# read from file
data = open(JSON3.read, "$(output_dir)/sensitivity.json")
ℓ  = Float64.(data["ℓ"])
σ  = Float64.(data["σ"])
σ1 = Float64.(data["σ1"])
γ  = Float64.(data["γ_cst"])
C  = Float64.(data["C_cst"])
cA = Float64.(data["cA_cst"])

println("\nParameter sensitivity analysis")
println("param ℓ=$ℓ")
println("param σ=$σ")
println("param σ1=$σ1")
println(" Vary  | Constant")
println(" ----------------")
println(" ℓ,σ1  |  C=$C")
println("  σ    |  1/γ=$(1 ./ γ)")
println("  σ    |  cA=$cA\n")



