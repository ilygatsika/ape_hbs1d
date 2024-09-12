using Test
using ape_hbs1d

include("../src/utils.jl")
include("../src/runge_kutta_solver.jl")
include("../src/finite_diff_solver.jl")

"""
Unit test: 1D harmonic potential

Tests that all exact solvers are solving pb with RHS 
    - (1/2) u'' + (1/2) * x²u = f 
where f(x) = (1/2) exp(-0.5x²)/√0.5
The exact solution is u(x) = exp(-0.5x²)/√0.5.
"""

@info "Running 1d harmonic potential test"

# PDE parameres
L = 5.0 # box size, exp(-25) ≃ 10⁻¹¹
z = 0.5
V = x -> z * V_harmonic(x)
rhs(x) = (1/2) * exp(-0.5*x^2)/√0.5
a = 0.0 # nuclear position
λ = 0.0
natom = 1

"""
RK4 solver
"""
sol_rk4 = RK4_solver(0.1,rhs,L,V;shift = 0.0)

"""
Finite difference solver
"""
Ng = 500
x_range, δx = discretize_space(Ng, L)
sol_fd = exact_solution(Ng, V, λ, rhs.(x_range), (x_range, δx))

"""
Errors
"""
usol(x) = exp(-0.5 * x^2)/√0.5
err_fd = norm(usol.(x_range) - sol_fd)
@test err_fd < 0.0005 # is = 0.0003885877265152854

rk_t = sol_rk4.t
rk_ft = [sol_rk4.u[k][1] for k in eachindex(sol_rk4.t)]
err_rk = norm(usol.(rk_t) - rk_ft) 
@test err_rk < 1.e-5 # is = 9.798947008964184e-6

#plot(x_range, usol.(x_range), color="black", linewidth=3, label="exact solution")
#plot(x_range, sol_fd, marker="o", label="finite diff")
#plot(rk_t, rk_ft, label="RK4")
#legend()
#show()


