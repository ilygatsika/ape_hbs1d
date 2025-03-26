include("../common.jl")

# Tests for Weyl inequality

R = 1.0
z1 = 1.0
z2 = 1.0
V0 = V_Gigi(0.5)
σ = 4.0
σ1 = 3.0
σ2 = 3.0
σ∞ = 1.0
K = 17

Ng = 3001
a = 5 * R
FD_grid = discretize_space(Ng, a)
x_range, δx = FD_grid

# Entire molecule
V1 = V_atom(V0,z1,+R)
V2 = V_atom(V0,z2,-R)
V(x) = V1(x) + V2(x)
Lap = (1/δx)^2 * Δ(Ng)

# Entire atom
A = - (1/2) * Lap + Diagonal(V.(x_range) .+ σ) 
res = eigen(Matrix(A))
println("-1/2Delta+V1+V2+shift $(res.values[1]) $(res.values[2])")

# First atom 1/2
B = - (1/2) * Lap + Diagonal(V1.(x_range) .+ σ) 
res = eigen(Matrix(B))
println("-1/2Delta+V1+shift    $(res.values[1]) $(res.values[2])")

B = - (1/2) * Lap + Diagonal(V1.(x_range) .+ σ/2) 
res = eigen(Matrix(B))
println("-1/2Delta+V1+shift/2  $(res.values[1]) $(res.values[2])")

A1 = - (1/2) * Lap + Diagonal(V1.(x_range)) 
res = eigen(Matrix(A1))
println("-1/2Delta+V1          $(res.values[1]) $(res.values[2])")

# First atom 1/4
B = - (1/4) * Lap + Diagonal(V1.(x_range) .+ σ) 
res = eigen(Matrix(B))
println("-1/4Delta+V1+shift    $(res.values[1]) $(res.values[2])")

B = - (1/4) * Lap + Diagonal(V1.(x_range) .+ σ/2) 
res = eigen(Matrix(B))
println("-1/4Delta+V1+shift/2  $(res.values[1]) $(res.values[2])")

B = - (1/4) * Lap + Diagonal(V2.(x_range) .+ σ/2) 
res = eigen(Matrix(B))
println("-1/4Delta+V2+shift/2  $(res.values[1]) $(res.values[2])")

A1 = - (1/4) * Lap + Diagonal(V1.(x_range)) 
res = eigen(Matrix(A1))
println("-1/4Delta+V1          $(res.values[1]) $(res.values[2])")

A1 = - Lap + Diagonal(V1.(x_range)) 
res = eigen(Matrix(A1))
println("-Delta+V1          $(res.values[1]) $(res.values[2])")

# mu_2(mol)
# < psi_2 | (-1/2+eps) Delta + V_1  | psi_2 > 
# < psi_2 | - eps Delta + V_2 + sigma | psi_2 >
