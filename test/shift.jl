include("../src/basis.jl")
include("../src/utils.jl")
include("../src/finite_diff_solver.jl")

z         = 1.0
R         = 2.0   # nuclei at -R and R
σ         = 2.3  # global shift
Ng        = 2001  # number of df grid in box
box_size  = 5*R   # big df box
V0        = V_Gigi(0.5) # potential centered at 0

(x_range, δx) = discretize_space(Ng, box_size)

V(x) = V_atom(V0,z,-R)(x) + V_atom(V0,z,R)(x)

vals0, _ = exact_eigensolver(Ng, V, 0, (x_range,δx))
μ = vals0[1] 
#σ = abs(μ) + 0.0001

H  = Hd(δx, Ng, V.(x_range), σ) 

vals, vecs = exact_eigensolver(Ng, V, σ, (x_range,δx))
λ1 = vals[1]
λ2 = vals[2]
u = abs.(vecs[:,1])
P_FD = u*u'
@assert norm(P_FD*P_FD .- P_FD) < 1e-10

println("(H+σ is posdef) ",σ > abs(μ))
A = √(u'H*u)
B = √(u'u)
println("R ",R)
println("σ ",σ)
println("μ ",μ)
println("λ1 ",λ1)

Cshift = B/A
print("σ=",σ," λ1=",λ1," Cshift=",Cshift)

