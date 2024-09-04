using Plots

include("../src/basis.jl")
include("../src/utils.jl")
include("../src/finite_diff_solver.jl")

#Nb_list   = Array(5:2:60)  # Nb bfs on each atom
Nb_list      = Array(5:2:20)  # bfs on each atom
R         = 1.0   # nuclei at -R and R
z         = 1.0   # nuclear charge
σk        = 3.0   # local shift
σ∞        = 1     # Laplacian shift
σ         = 4.0  # global shift
ℓ         = 0.9  # partition of unity parameter
Ng        = 2001  # number of df grid in box
box_size  = 5*R   # big df box
V0        = V_Gigi(0.5) # potential centered at 0
K_list    = [5,10,15,20,25,30] # size of eigenbasis
N_target  = 35    # fixed discretisation
K_target  = 17

default(linewidth = 1.5)

C2 = prefactor_2(R,σ,σk, ℓ, V0)
C3 = prefactor_3(R,σ,σk,σ∞, ℓ, V0)
println("Prefactor 2: ",C2)
println("Prefactor 3: ",C3)

(x_range, δx) = discretize_space(Ng, box_size)
ν,μ = domains_symm(R,ℓ)

pneg∞ = 1.0 .- p0x1_global(-ν,-μ,x_range)
ppos∞ = p0x1_global(μ,ν,x_range)
pL = p0x1x0_global(-ν,-μ,-ℓ,ℓ,x_range)
pR = p0x1x0_global(-ℓ,ℓ,μ,ν,x_range)

pL2 = 1.0 .- p0x1_global(-ℓ,ℓ,x_range)
pR2 = p0x1_global(-ℓ,ℓ,x_range)

V = sum_potential(V0,R)
V1(x) = V0(x-R)
V∞(x) = 0
Hk = Hd(δx, Ng, V1.(x_range), σk)
H∞ = Hd(δx, Ng, V∞.(x_range), σ∞)
H  = Hd(δx, Ng, V.(x_range), σ) 

maskR = (1:Ng)[-ℓ .< x_range .< ν]
mask∞ = vcat((1:Ng)[x_range .< -μ],(1:Ng)[μ .< x_range])
Ng_R, x_range_R = size(maskR,1), x_range[maskR]
Ng_∞, x_range_∞ = size(mask∞,1), x_range[mask∞]
Hk_rs = SymTridiagonal(Hd(δx, Ng_R, V1.(x_range_R), σk))
H∞_rs = Hd(δx, Ng_∞, V∞.(x_range_∞), σ∞)
nh = floor(Int,size(H∞_rs,1)/2)
H∞_rs[nh+1:2*nh,1:nh] .= 0.0
H∞_rs[1:nh,nh+1:2*nh] .= 0.0

data_R = eigen(Hk_rs)
dvals_R,dvecs_R = data_R.values, data_R.vectors

vals, vecs = exact_eigensolver(Ng, V, σ, (x_range, δx))
μ_FD = vals[1]
u_FD = abs.(vecs[:,1])
P_FD = u_FD*u_FD'
@assert norm(P_FD*P_FD .- P_FD) < 1e-10

plot(x_range, u_FD, linewidth=3, color="red", label="usol")
plot!(x_range, pneg∞)
plot!(x_range, ppos∞)
plot!(x_range, pL)
plot!(x_range, pR)

savefig("img/partition.png")

# Assumption 1
@assert( √(u_FD'H*u_FD) >= √(u_FD'u_FD) )
@assert( √(u_FD'Hk*u_FD) >= √(u_FD'u_FD) )

nb_tests = length(Nb_list)
errH = zeros(nb_tests)
estm2 = zeros(nb_tests)
estm3 = zeros(nb_tests)
estm3_rs = zeros(nb_tests)
resn_R = zeros(nb_tests)
resn_R_ds_inf,resn_R_ds_sup = zeros(nb_tests),zeros(nb_tests)

for (i,Nb) in enumerate(Nb_list)
   
    println("Nb per atom ",Nb)

    HB = local_hermite_basis(R, Nb, Nb, x_range, δx)
    S_HB = Symmetric(HB'HB)
    Mass_HB = Symmetric(HB'*(H*HB)) 
    PChol = cholesky(S_HB, RowMaximum(), check = false)
    r = rank(S_HB, atol=1e-6, rtol=1e-6)
    P = PChol.p[1:r]
    S_HB = S_HB[P,P]
    Mass_HB = Mass_HB[P,P]

    μ_HB, C_HB = eigen(Mass_HB, S_HB)
    C_HB_rdc = zeros(2*Nb)
    C_HB_rdc[P] = C_HB[:,1]
    u_HB = abs.(HB * C_HB_rdc)
    P_HB = u_HB*u_HB'
    @assert norm(P_HB*P_HB .- P_HB) < 1e-10   
    
    A_H = (u_FD - u_HB)'H*(u_FD - u_HB)
    errH[i] = √(A_H)
    println("error H  \t",errH[i])
    
    Res = μ_HB[1].*u_HB - H*u_HB
    pR_Res = (pR .^ (1/2)) .* Res
    p∞_Res = ((pneg∞ + ppos∞) .^ (1/2)) .* Res
    wR = exact_solution(Ng, V1, σk, pR_Res, (x_range, δx))
    w∞ = exact_solution(Ng, V∞, σ∞, p∞_Res, (x_range, δx))
    dual_norm_R = pR_Res'wR
    dual_norm_∞ = p∞_Res'w∞

    estm3[i] = C3 * √(2 * dual_norm_R + dual_norm_∞)
    println("estimate3\t",estm3[i])

    # old for comparison
    pR2_Res = (pR2 .^ (1/2)) .* Res
    wR2 = exact_solution(Ng, V1, σk, pR2_Res, (x_range, δx))
    dual_norm_R2 = pR2_Res'wR2

    estm2[i] = C2 * √(2 * dual_norm_R2)
    println("estimate2\t",estm2[i])

    # restrict Hk and H∞
    pR_Res_rs = pR_Res[maskR]
    p∞_Res_rs = p∞_Res[mask∞]
    wR_rs = exact_solution(Ng_R, V1, σk, pR_Res_rs, (x_range_R, δx))
    w∞_rs = exact_solution(Ng_∞, V∞, σ∞, p∞_Res_rs, (x_range_∞, δx))
    dual_norm_R_rs = pR_Res_rs'wR_rs
    dual_norm_∞_rs = p∞_Res_rs'w∞_rs

    estm3_rs[i] = C3 * √(2 * dual_norm_R_rs + dual_norm_∞_rs)
    println("estimate3 rs\t",estm3_rs[i])

    sR,sRbis = 0,0
    for j in 1:K_target
        ipR = (pR_Res_rs'abs.(dvecs_R[:,j]))^2
        sRbis += ipR
        sR += 1.0/dvals_R[j] * ipR
    end
    resn_R[i] = dual_norm_R_rs
    resn_R_ds_inf[i] = sR
    resn_R_ds_sup[i] = sR +  1.0/dvals_R[K_target+1] * (pR_Res_rs'pR_Res_rs - sRbis)

    if Nb == N_target
        nK = size(K_list,1)
        resn_K = zeros(nK)
        for (k,Ki) in enumerate(K_list)
            sR = 0
            for j in 1:Ki
                ipR = (pR_Res_rs'abs.(dvecs_R[:,j]))^2
                sR += 1.0/dvals_R[j] * ipR
            end
            resn_K[k] = sR
        end
        plot(K_list, resn_R[i]*ones(nK), labels="exact")
        plot!(K_list, resn_K, labels="spectral projection",marker=:diamond,
              ylabel="Res dual (compact H1)-norm, N=$(Nb*2)", 
          xlabel="K eigenfunctions of H1")
        savefig("img/SD_K_$(2*Nb).png")

    end
end

plot(2 .* Nb_list, errH, yscale=:log10, marker=:hexagon, labels="exact")
plot!(2 .* Nb_list, estm2, yscale=:log10, labels="estim_2")
plot!(2 .* Nb_list, estm3, yscale=:log10, marker=:square, labels="estim_3")
plot!(2 .* Nb_list, estm3_rs, yscale=:log10, marker=:cross,labels="estim_3_rs", 
      ylabel="Discretisation error in H norm", xlabel="N Hermite")
savefig("img/3prt.png")

plot(2 .* Nb_list, resn_R, yscale=:log10, marker=:diamond, labels="exact")
plot!(2 .* Nb_list, resn_R_ds_inf, yscale=:log10, labels="inf bound")
plot!(2 .* Nb_list, resn_R_ds_sup, yscale=:log10, labels="sup bound",
      ylabel="Res_N for compact H1, K=$(K_target)", xlabel="N Hermite")
savefig("img/SD_Nb_Her_$(K_target).png")



