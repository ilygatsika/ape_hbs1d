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
    vec_ℓ = [0.1, 0.3, 0.8]
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
    eigv_err = zeros(nb_ℓ, nb_tests)
    Hest_eig_gua = zeros(nb_ℓ, nb_tests)
    eigv_est_gua = zeros(nb_ℓ, nb_tests)
    Hest_eig_nongua = zeros(nb_ℓ, nb_tests)
    eigv_est_nongua = zeros(nb_ℓ, nb_tests)
    
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
        Vlow = minimum((Ω.V).(FD_grid[1]))
        cH = 1/sqrt(Vlow + σ) # use the practical one for H2+
        local c = subdomain_constant((Ω, Ω1, Ω2, Ω∞), ℓ, cH, FD_grid)
        println("ℓ=($ℓ) λ1=($μ1_FD) cH=($cH) Vlow=($Vlow) C=($c)")
        # Note: if λ1 is close to one it means that the shift σ
        # is optimal
       
        # use guaranteed lb of λ_2 (instead of λ_2N) (Remark 3.4)
        μs_AB, μs_A, μs_B = weyl_lower_bound(V,z1,z2,R,σ,Ng,FD_grid)
        μ2_lb = μs_A[1] + μs_B[2]
        println("Weyl's lower bound (guaranteed): λ2_lb=($μ2_lb) < λ2=($μ2_FD)")
        @assert( μ2_lb <= μ2_FD ) # Assumption 3.2

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
            @assert( Herr_ >= √(err'err) ) # Assumption 2.5
            Res = λ_1N * u_1N - Ω.H * u_1N
            dnorm_Res = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid)
            
            # non-guaranteed estimation
            c1_nongua = gap_constant_1(λ_2N, λ_1N)
            c2_nongua = gap_constant_2(λ_2N, λ_1N)
            println("Practical bound (non-guaranteed): λ2N=($λ_2N) <? λ2=($μ2_FD)")
            println("Nb=($Nb), nongua gap const= $(c1_nongua) $(c2_nongua)")
            
            est_nongua = estimator_eigenvector(c, c1_nongua, c2_nongua, λ_1N, dnorm_Res)
            estλ_nongua = estimator_eigenvalue(c, c1_nongua, dnorm_Res)
            
            # guaranteed estimation
            c1_gua = gap_constant_1(μ2_lb, λ_1N)
            c2_gua = gap_constant_2(μ2_lb, λ_1N)
            @assert( λ_1N <= μ2_lb ) # Assumption 3.2
            println("       guaran gap const= $(c1_gua) $(c2_gua)")

            est_gua = estimator_eigenvector(c, c1_gua, c2_gua, λ_1N, dnorm_Res)
            estλ_gua = estimator_eigenvalue(c, c1_gua, dnorm_Res)
        
            # Store results
            Herr_src[i,j], Hest_src[i,j] = Herr, est
            Herr_eig[i,j] = Herr_
            eigv_err[i,j] = λ_1N - μ1_FD
            Hest_eig_gua[i,j] = est_gua
            eigv_est_gua[i,j] = estλ_gua
            Hest_eig_nongua[i,j] = est_nongua
            eigv_est_nongua[i,j] = estλ_nongua
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
    data["eigv_err"] = eigv_err
    data["Hest_eig_gua"] = Hest_eig_gua
    data["eigv_est_gua"] = eigv_est_gua
    data["Hest_eig_nongua"] = Hest_eig_nongua
    data["eigv_est_nongua"] = eigv_est_nongua
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
eigv_err = reshape(data["eigv_err"], (nb_ℓ, nb_tests))
Hest_eig_gua = reshape(data["Hest_eig_gua"], (nb_ℓ, nb_tests))
eigv_est_gua = reshape(data["eigv_est_gua"], (nb_ℓ, nb_tests))
Hest_eig_nongua = reshape(data["Hest_eig_nongua"], (nb_ℓ, nb_tests))
eigv_est_nongua = reshape(data["eigv_est_nongua"], (nb_ℓ, nb_tests))

# ##################
# Plot results
# ##################

# Error convergence for source problem wrt Hermite basis size
PyPlot.plot(Nb_list, Herr_src[1,:], marker="x", markevery=3, label=L"$\|u-u_N\|_A$")
for i in 1:nb_ℓ
    PyPlot.plot(Nb_list, Hest_src[i,:], marker="^", markevery=3, label=L"est. $\ell=%$(2*vec_ℓ[i])$")
end
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.grid(color="#EEEEEE")
PyPlot.legend(loc="upper right")
PyPlot.savefig("$(figure_dir)/src_pb.pdf")
PyPlot.close()

# Error convergence for eigval problem wrt Hermite basis size
function plot_estimation(Hest_,est_,filename)

    # Figure is splitted in two parts
    fig, (ax1, ax2) = PyPlot.subplots(nrows=2, ncols=1, sharex=true,
                                  figsize=(4.0,4.2), gridspec_kw=["height_ratios"=>[1.5,1.5]])

    nb_size = size(Nb_list,1)
    PyPlot.xticks(1:3:nb_size, Nb_list[1:3:nb_size])
    PyPlot.xlabel(L"N"*" basis functions")

    # Upper part: Eigenvectors
    ax1.plot(Herr_eig[1,:], marker="x", markevery=3, label=L"$\|\varphi_1 - \varphi_{1N}\|_A$")
    for i in 1:nb_ℓ
        ax1.plot(Hest_[i,:], marker="^", markevery=3, label=L"est. $\ell=%$(2*vec_ℓ[i])$")
    end
    ax1.set_yscale("log")
    ax1.grid(color="#EEEEEE")
    ax1.legend()

    # Lower part: Eigenvalues
    ax2.plot(eigv_err[1,:], marker="s", markevery=3, linestyle=:dashed, label=L"$\lambda_{1N} - \lambda_1$")
    for i in 1:nb_ℓ
        ax2.plot(est_[i,:], marker="*", markevery=3, linestyle=:dashed, label=L"est. $\ell=%$(2*vec_ℓ[i])$")
    end
    ax2.set_yscale("log")
    ax2.grid(color="#EEEEEE")
    ax2.legend()

    PyPlot.tight_layout()
    PyPlot.subplots_adjust(wspace=0, hspace=0)
    PyPlot.savefig(filename)
    PyPlot.close()
end

# Plot for guaranteed and non-guaranteed estimation separately
plot_estimation(Hest_eig_gua,eigv_est_gua,"$(figure_dir)/eig_pb_gua.pdf")
plot_estimation(Hest_eig_nongua,eigv_est_nongua,"$(figure_dir)/eig_pb_nongua.pdf")

