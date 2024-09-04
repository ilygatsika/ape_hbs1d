using JSON3

include("../src/diatomic.jl")

"""
    Plot sols
"""

output_dir = "out"

# ###################
# System parameters
# ###################
R  = 1.0          # -R and +R atomic positions
z1 = 1.0          # atomic charge at -R
z2 = 1.0          # atomic charge at +R
V = V_Gigi(0.5)   # atomic potential
σ  = 7.0          # shifts
σ1 = 4.0
σ2 = 4.0
σ∞ = 1.0
K  = 17          # size of spectral basis

# partition overlap size is 2*ℓ
ℓ = 0.3

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

# compute partitions on domains
cH, Ω, Ω1, Ω2, Ω∞ = init_subdomains(mol, ℓ, σ, σ1, σ2, σ∞, K, Ng, FD_grid)

# reference solution of source problem
sol_FD, rhs = test_source_pb(mol, Ω, Ω1, Ω2, Ng, FD_grid)

# reference solution of eigenproblem
μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)

VΩ1 = V_atom(V, z1, R)
VΩ2 = V_atom(V, z2, -R)
VΩ(x) = VΩ1(x) + VΩ2(x)

xx, δx = FD_grid
PyPlot.plot(xx, sol_FD, "-", color="magenta", label=L"u")
PyPlot.plot(xx, u_FD, "-", color="blue", label=L"u_1")
PyPlot.vlines(-R, 0, 0.045, color="k", linestyles="dashed")
PyPlot.vlines(+R, 0, 0.045, color="k", linestyles="dashed")
PyPlot.annotate(L"-R", [-R,0])
PyPlot.annotate(L"+R", [+R,0])
PyPlot.xlabel(L"x")
PyPlot.ylabel("exact solution")
PyPlot.legend()
PyPlot.savefig("img/sols.pdf")
PyPlot.close()
"""
PyPlot.plot(xx, VΩ.(xx), "-g")
PyPlot.xlabel(L"x")
PyPlot.savefig("img/pot.pdf")
PyPlot.close()"""


