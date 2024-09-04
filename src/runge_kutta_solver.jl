using BoundaryValueDiffEq

"""
Wrapper for external RK4 solver
ODE - (1/2)u'' + Vu = f on the line (-L,L) with homogeneous Dirichlet
"""

function bc!(residual, u, p, t) # u[1] is the beginning of the time span, and u[end] is the ending
  residual[1] = u[1][1] # the solution at the beginning of the time span should be 0
  residual[2] = u[end][1] # the solution at the end of the time span should be 0
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
  bvp = TwoPointBVProblem(oned_toy_pb!, bc!, [0.1,0.1], (-L,L))
  return solve(bvp, MIRK4(), dt=dt)
end

