include("../common.jl")

"""
    Main computation of the a posteriori error estimator.

    Hermite basis sets for diatomic molecule
    for source problem and eigenvalue problem in 1D.
    Reference solution computed using finite differences.
    
    This code produces a JSON file and/or plots figure, prints results in the terminal.
"""

if !isfile("$(output_dir)/res_main.json")

    # ###################
    # System parameters
    # ###################
    R  = 1.0          # -R and +R atomic positions
    z1 = 1.0          # atomic charge at -R
    z2 = 1.0          # atomic charge at +R
    V = V_Gigi(0.5)   # atomic potential
    σ  = 4.0          # shifts
    σ1 = 3.0
    σ2 = 3.0
    σ∞ = 1.0
    K  = 17          # size of spectral basis

    # partition overlap size is 2*ℓ
    vec_ℓ = [0.1, 0.3, 0.5, 0.8, 0.9]
    nb_ℓ = length(vec_ℓ)

    # ###############################
    # Discretisation basis parameters
    # ###############################
    Ng = 3001            # number of finite diff points
    a  = 5 * R           # size of finite diff box
    Nb_list = Array(5:2:60) # Hermite basis size

    # #########################
    # Main simulation
    # #########################

    mol = Molecule(R,z1,z2,V)
    FD_grid = discretize_space(Ng, a)

    nb_tests = length(Nb_list)
    Herr_src = zeros(nb_ℓ, nb_tests) 
    Hest_src = zeros(nb_ℓ, nb_tests)
    Herr_eig = zeros(nb_ℓ, nb_tests) 
    Hest_eig = zeros(nb_ℓ, nb_tests)
    eigv_err = zeros(nb_ℓ, nb_tests)
    eigv_est = zeros(nb_ℓ, nb_tests)

    # reference solution of eigenproblem
    local cH, Ω = init_subdomains_omega(mol, σ, K, Ng, FD_grid)
    μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)
    println("gap of eig pb $(μ2_FD - μ1_FD)")

    # Vary size of partition of overlap
    for i in 1:nb_ℓ

        # compute partitions on domains
        local ℓ = vec_ℓ[i]
        local cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)

        # reference solution of source problem
        local sol_FD, rhs = test_source_pb(mol, Ω, Ω1, Ω2, Ng, FD_grid)
    
        # domain constant
        cH = 1/sqrt(- 2 + σ) # use the practical one for H2+
        local c = subdomain_constant((Ω, Ω1, Ω2, Ω∞), ℓ, cH, FD_grid)
        println("ℓ=($ℓ) λ1=($μ1_FD) cH=($cH) C=($c)")

        # Vary Hermite basis size
        for j in 1:nb_tests
    
            # problem with rhs
            Nb = Nb_list[j]
            u_HB = hermite_solver(mol, Ω.H, Nb, Nb, rhs, FD_grid)
            local err = sol_FD - u_HB
            local Herr = √(err'Ω.H*err)
            @assert( Herr >= √(err'err) )
            Res = rhs - Ω.H * u_HB
            dnorm_Res = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid)
            est = estimator_source_pb(c, dnorm_Res)

            # eigenproblem 
            λ_1N, λ_2N, u_1N = hermite_eigensolver(mol, Ω.H, Nb, Nb, FD_grid)
            err = u_1N - u_FD
            Herr_ = √(err'Ω.H*err)
            @assert( Herr_ >= √(err'err) )
            Res = λ_1N * u_1N - Ω.H * u_1N
            dnorm_Res = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid)
            c1 = gap_constant_1(λ_2N, λ_1N)
            c2 = gap_constant_2(λ_2N, λ_1N)
            est_ = estimator_eigenvector(c, c1, c2, μ1_FD, dnorm_Res)

            println("Nb=($Nb), gap const 1 $(c1) 2 $(c2)")
        
            # Store results
            Herr_src[i,j], Hest_src[i,j] = Herr, est
            Herr_eig[i,j], Hest_eig[i,j] = Herr_, est_
            eigv_err[i,j], eigv_est[i,j] = (λ_1N - μ1_FD), estimator_eigenvalue(c, c1, dnorm_Res)
        end
    end

    # ##################
    # Export results
    # ##################

    data = Dict{String, Any}()
    data["Nb_list"] = 2 .* Nb_list
    data["ℓ_list"] = vec_ℓ
    data["Herr_src"] = Herr_src
    data["Herr_eig"] = Herr_eig
    data["Hest_src"] = Hest_src
    data["Hest_eig"] = Hest_eig
    data["eigv_err"] = eigv_err
    data["eigv_est"] = eigv_est
    data["V"] = V.(FD_grid[1])

    # write to file
    open(io -> JSON3.write(io, data, allow_inf=true), "$(output_dir)/res_main.json", "w")

end

# read from file
data = open(JSON3.read, "$(output_dir)/res_main.json")
Nb_list = Int64.(data["Nb_list"])
vec_ℓ = Float64.(data["ℓ_list"])
nb_ℓ = length(vec_ℓ)
nb_tests = length(Nb_list)
Herr_src = reshape(data["Herr_src"], (nb_ℓ, nb_tests))
Hest_src = reshape(data["Hest_src"], (nb_ℓ, nb_tests))
Herr_eig = reshape(data["Herr_eig"], (nb_ℓ, nb_tests))
Hest_eig = reshape(data["Hest_eig"], (nb_ℓ, nb_tests))
eigv_err = reshape(data["eigv_err"], (nb_ℓ, nb_tests))
eigv_est = reshape(data["eigv_est"], (nb_ℓ, nb_tests))

# ##################
# Plot results
# ##################

# Error convergence for source problem wrt Hermite basis size
PyPlot.plot(Nb_list, Herr_src[1,:], marker="x", markevery=3, label=L"$\|u-u_N\|_A$")
for i in 1:nb_ℓ-1
    if (i != 3)
        PyPlot.plot(Nb_list, Hest_src[i,:], marker="^", markevery=3, label=L"est. $\ell=%$(2*vec_ℓ[i])$")
    end
end
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.grid(color="#EEEEEE")
PyPlot.legend(loc="upper right")
PyPlot.savefig("$(figure_dir)/src_pb.pdf")
PyPlot.close()

# Error convergence for eigval problem wrt Hermite basis size
PyPlot.plot(Nb_list, Herr_eig[1,:], marker="x", markevery=3, label=L"$\|\varphi_1 - \varphi_{1N}\|_A$")
for i in 1:nb_ℓ-1
    if ((i == 1) || (i == 4)) 
        PyPlot.plot(Nb_list, Hest_eig[i,:], marker="^", markevery=3, label=L"est. $\ell=%$(2*vec_ℓ[i])$")
    end
end
PyPlot.plot(Nb_list, eigv_err[1,:], marker="s", markevery=3, linestyle=:dashed, label=L"$\lambda_{1N} - \lambda_1$")
i = nb_ℓ - 1
PyPlot.plot(Nb_list, eigv_est[i,:], marker="*", markevery=3, linestyle=:dashed, label=L"est. $\ell=%$(2*vec_ℓ[i])$")
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.grid(color="#EEEEEE")
ax = PyPlot.gca()
handles, labels = ax.get_legend_handles_labels()
first_legend = ax.legend(handles=[handles[1],handles[4]], loc="upper right")
second_legend = plt.legend(handles=[handles[2],handles[3],handles[5]], loc="lower left")
ax.add_artist(first_legend)
PyPlot.savefig("$(figure_dir)/eig_pb.pdf")
PyPlot.close()




