include("../common.jl")

"""
    Plot potential and reference solutions of model problems.
    
    This code produces a JSON file and/or plots figures.
"""

if !isfile("$(output_dir)/model.json")
    
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
    local sol_FD, rhs = test_source_pb(mol, Ω, Ω1, Ω2, Ng, FD_grid)

    # reference solution of eigenproblem
    μ1_FD, μ2_FD, u_FD = test_eigenpb(mol, Ω, Ng, FD_grid)

    VΩ1 = V_atom(V, z1, R)
    VΩ2 = V_atom(V, z2, -R)
    VΩ(x) = VΩ1(x) + VΩ2(x)

    xx, δx = FD_grid

    # ##################
    # Export results
    # ##################

    data = Dict{String, Any}()
    data["R"] = R
    data["xx"] = xx
    data["sol_FD"] = sol_FD
    data["u_FD"] = u_FD
    data["Vxx"] = VΩ.(xx)

    # write to file
    open(io -> JSON3.write(io, data, allow_inf=true), "$(output_dir)/model.json", "w")

end

# read from file
data = open(JSON3.read, "$(output_dir)/model.json")
R      = Float64.(data["R"])
xx     = Float64.(data["xx"])
sol_FD = Float64.(data["sol_FD"])
u_FD   = Float64.(data["u_FD"])
Vxx    = Float64.(data["Vxx"])

# ##################
# Plot results
# ##################

PyPlot.plot(xx, sol_FD, "-", label=L"u")
PyPlot.plot(xx, u_FD, "-", label=L"\phi_1")
PyPlot.vlines(-R, 0, 0.045, linestyles="dashed")
PyPlot.vlines(+R, 0, 0.045, linestyles="dashed")
PyPlot.annotate(L"-R", [-R,0])
PyPlot.annotate(L"+R", [+R,0])
PyPlot.xlabel(L"x")
PyPlot.ylabel("exact solution")
PyPlot.legend()
PyPlot.savefig("$(figure_dir)/sols.pdf")
PyPlot.close()

PyPlot.plot(xx, Vxx, "-g")
PyPlot.xlabel(L"x")
PyPlot.savefig("$(figure_dir)/pot.pdf")
PyPlot.close()


