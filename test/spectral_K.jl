include("../src/diatomic.jl")

"""
    Sensitivity analysis for spectral basis size
"""

include("../src/plot_conf.jl")

output_dir = "out"

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
ℓ  = 0.8

vec_K = [5,17]
nb_K = length(vec_K)

# ###############################
# Discretisation basis parameters
# ###############################
Ng = 2001            # number of finite diff points
a  = 5 * R           # size of finite diff box
Nb_list = Array(5:2:60) 
nb_tests = length(Nb_list)

# #########################
# Main simulation
# #########################

mol = Molecule(R,z1,z2,V)
FD_grid = discretize_space(Ng, a)


# Vary spectral basis size
ntrue = zeros((nb_K,nb_tests))
ninf,nsup = zeros((nb_K,nb_tests)),zeros((nb_K,nb_tests))
for i in 1:nb_K

    K = vec_K[i]
    println("======= K=$K")
    
    # compute partitions on domains
    cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid) 

    # reference solution of eigenproblem
    μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)
    
    for j in 1:nb_tests

        Nb = Nb_list[j]
        λ_1N, u_1N = hermite_eigensolver(mol, Ω.H, Nb, Nb, FD_grid)
        err = u_1N - u_FD
        Herr_ = √(err'Ω.H*err)
        @assert( Herr_ >= √(err'err) )
        Res = λ_1N * u_1N - Ω.H * u_1N
        inf, real, sup = decompose_dual_norm(Res, Ω1, Ω2, Ω∞, Ng, FD_grid; inf=1)

        println("$(inf) $(real) $(sup)")

        ntrue[i,j] = real
        ninf[i,j] = inf
        nsup[i,j] = sup
    end
end

cls = ["green", "magenta"]
PyPlot.plot(Nb_list, ntrue[1,:], "-x", color="blue", label=L"$\|u_1-u_{1N}\|_{H^{-1}}$")
for i in 1:nb_K
    K = vec_K[i]
    PyPlot.plot(Nb_list, ninf[i,:], "-o", color=cls[i], label=L"K=%$(K)"*" min")
    PyPlot.plot(Nb_list, nsup[i,:], "-^", color=cls[i], label=L"K=%$(K)"*" max")
end
PyPlot.ylabel("error")
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.legend()
PyPlot.savefig("img/spectral.pdf")
PyPlot.close()
    
