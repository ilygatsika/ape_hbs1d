using PyPlot
using LaTeXStrings

include("../src/diatomic.jl")

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
σ2 = 4.5
σ∞ = 1.0
K = 17

# partition overlap size is 2*ℓ
ℓ = 0.5

# ###############################
# Discretisation basis parameters
# ###############################
Ng = 701            # number of finite diff points
a  = 5 * R           # size of finite diff box
Nb_list = Array(3:2:10) # Hermite basis size

# #########################
# Main simulation
# #########################

output_dir = "img"

# Format PyPlot
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 15
rcParams["legend.fontsize"] = "medium"
rcParams["lines.linewidth"] = 0.85
#rcParams["pdf.use14corefonts"] = true
rcParams["lines.markersize"] = 6
rcParams["lines.markerfacecolor"] = "none"
rcParams["legend.edgecolor"] = "k"
rcParams["legend.labelspacing"] = 0.01
rcParams["legend.fancybox"] = false
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelpad"] = 1.4
rcParams["axes.linewidth"] = 1
mol = Molecule(R,z1,z2,V)

FD_grid = discretize_space(Ng, a)

cH, Ω = init_subdomains_omega(mol, σ, K, Ng, FD_grid)
#Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)

# reference solution of eigenproblem
μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)

xx = FD_grid[1]
PyPlot.plot(xx, u_FD, color="blue", linestyle="-", label=L"u_1")

nb_tests = length(Nb_list)

# Vary Hermite basis size
for j in 1:nb_tests
    
    # eigenproblem 
    Nb = Nb_list[j]
    λ_1N, u_1N = hermite_eigensolver(mol, Ω.H, Nb, Nb, FD_grid)
    #PyPlot.plot(xx, u_1N, label=L"$u_{1N}, N=%$(2*Nb)$")
end

#PyPlot.title(L"z_1=1, z_2=1.03")
PyPlot.legend()
PyPlot.ylabel("exact solution")
PyPlot.xlabel(L"x")
PyPlot.vlines(-R, 0, 0.08, linestyles ="dotted", colors ="k")
PyPlot.vlines(R, 0, 0.08, linestyles ="dotted", colors ="k")
PyPlot.text(-R, 0.00, L"$x=-R$", rotation=-90)
PyPlot.text(R, 0.00, L"$x=+R$", rotation=-90)
PyPlot.savefig("$(output_dir)/exact_sol.pdf")
PyPlot.close()

