using BoundaryValueDiffEq

"""
Wrapper for external RK4 solver
ODE - (1/2)u'' + Vu = f on the line (-L,L) with homogeneous Dirichlet
"""

function bc2a!(resid_a, u_a, p) # u_a is at the beginning of the time span
    resid_a[1] = u_a[1] # the solution at the beginning of the time span should be 0
end
function bc2b!(resid_b, u_b, p) # u_b is at the ending of the time span
    resid_b[1] = u_b[1] # the solution at the end of the time span should be 0
end

"""
Boundary value problem solver 
  - dt : discretisation step
  - rhs : right-hand side
  - L : box size
  - V : Potential (type Function)
"""
function RK4_solver(dt,rhs,L,V;shift = 1.0)
  function oned_toy_pb!(du,u,p,t)
      θ  = u[1]
      dθ = u[2]
      du[1] = dθ
      du[2] = -2*rhs(t)+(V(t) + shift)*2*θ 
      #(-1)^V.harm because RHS is positive for harmonic oscillators but negative for generic V
  end
  println("----------------------------")
  println("RK4 solution for dt=$(dt)")

  tspan = (-L, L)
  bvp = TwoPointBVProblem(oned_toy_pb!, (bc2a!, bc2b!), [0.1,0.1], tspan;
                          bcresid_prototype = (zeros(1), zeros(1)))
  return solve(bvp, MIRK4(), dt=dt)
end

