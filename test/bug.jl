"""
Bug is that for two atoms RK4 needs simple gaussian rhs 
but for this choice other solvers don't converge.
"""

using PyPlot
include("../src/runge_kutta_solver.jl")
include("../src/finite_diff_solver.jl")
include("../src/utils.jl")

# PDE parameres
L = 15.0 # box size
z = 0.5
a = 2.0 # nuclear positions +a and -a
V_rad = V_smeared(z,1.5)
V = sum_potential(V_rad, a)
Nq = 500 # quadrature points
λ = 0.0
natom = 2

# set rhs
# function must be defined manually otherwise RK4 bugs
rhs(x) = - π^(-1/4)*exp(-(x-a)^2) - π^(-1/4)*exp(-(x+a)^2)

"""
RK4 solver
"""
sol_rk4 = RK4_solver(0.1, rhs, L, V)

"""
Finite difference solver
"""
Ng = 300
x_range, δx = discretize_space(Ng, L)
sol_fd = exact_solution(Ng, V, λ, rhs.(x_range), (x_range, δx))

"""
Plot
"""
#
plot(x_range, sol_fd, label="finite diff")
#plot(x_range,V.(x_range))
plot(sol_rk4.t,[-sol_rk4.u[k][1] for k in eachindex(sol_rk4.t)], label="RK4")
legend()
show()

