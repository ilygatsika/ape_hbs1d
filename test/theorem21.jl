using Plots
using LaTeXStrings

include("../src/basis.jl")
include("../src/utils.jl")
include("../src/finite_diff_solver.jl")

Nb_list   = Array(5:2:60)  # Nb bfs on each atom
R         = 1.0   # nuclei at -R and R
z         = 1.0   # nuclear charge
σk        = 2.0   # local shift
σ         = 2*σk  # global shift
σ∞        = 1     # Laplacian shift
ℓ         = 0.50  # partition of unity parameter
Ng        = 2001  # number of df grid in box
box_size  = 5*R   # big df box
V0        = V_Gigi(z,0.5) # potential centered at 0
K         = 24
N_target  = 39

default(linewidth = 2.0)

(x_range, δx) = discretize_space(Ng, box_size)
V = sum_potential(V0,R)
H  = Hd(δx, Ng, V.(x_range), σ) 

C2 = prefactor_2(R,σ,σk, ℓ, V0)
C3 = prefactor_3(R,σ,σk,σ∞, ℓ, V0)

# Eigenpb
vals, vecs = exact_eigensolver(Ng, V, σ, (x_range, δx))
μ_FD = vals[1]
u_FD = abs.(vecs[:,1])

# Partitions and restricted partitions
ν,μ = domains_symm(R,ℓ)

pneg∞ = 1.0 .- p0x1_global(-ν,-μ,x_range)
ppos∞ = p0x1_global(μ,ν,x_range)
pL = p0x1x0_global(-ν,-μ,-ℓ,ℓ,x_range)
pR = p0x1x0_global(-ℓ,ℓ,μ,ν,x_range)

pL2 = 1.0 .- p0x1_global(-ℓ,ℓ,x_range)
pR2 = p0x1_global(-ℓ,ℓ,x_range)

V1(x) = V0(x-R)
V∞(x) = 0
Hk = Hd(δx, Ng, V1.(x_range), σk)
H∞ = Hd(δx, Ng, V∞.(x_range), σ∞)

maskR = (1:Ng)[-ℓ .<= x_range .<= ν]
mask∞ = vcat((1:Ng)[x_range .<= -μ],(1:Ng)[μ .<= x_range])
Ng_R, x_range_R = size(maskR,1), x_range[maskR]
Ng_∞, x_range_∞ = size(mask∞,1), x_range[mask∞]
Hk_rs = Hd(δx, Ng_R, V1.(x_range_R), σk)
H∞_rs = Hd(δx, Ng_∞, V∞.(x_range_∞), σ∞)
nh = floor(Int,size(H∞_rs,1)/2)
H∞_rs[nh+1:2*nh,1:nh] .= 0.0
H∞_rs[1:nh,nh+1:2*nh] .= 0.0

# Assumption 1
@assert( √(u_FD'H*u_FD) >= √(u_FD'u_FD) )
@assert( √(u_FD'Hk*u_FD) >= √(u_FD'u_FD) )

# spectral basis of bounded H1
dvals_R, dvecs_R = exact_eigensolver(Ng_R, V1, σk, (x_range_R, δx),nvals=K)
for j in 1:K
    dvecs_R[:,j] /= √(δx * dvecs_R[:,j]'dvecs_R[:,j])
    dvecs_R[:,j] *= √δx
    @assert norm(dvecs_R[:,j]'dvecs_R[:,j] .- 1.) < 1e-10
end

# spectral basis of unbounded H
dvals_H,_ = exact_eigensolver(Ng, V, σ, (x_range, δx),nvals=2)

# spectral basis of unbounded H1
dvals, dvecs = exact_eigensolver(Ng, V1, σk, (x_range, δx),nvals=K)
for j in 1:K
    dvecs[:,j] /= √(δx * dvecs[:,j]'dvecs[:,j])
    dvecs[:,j] *= √δx
    @assert norm(dvecs[:,j]'dvecs[:,j] .- 1.) < 1e-10
end

nb_tests = length(Nb_list)
x = zeros(nb_tests)
x_L2 = zeros(nb_tests)
y = zeros(nb_tests)
y_as = zeros(nb_tests)
z_2 = zeros(nb_tests)
z_3 = zeros(nb_tests)
z_3R = zeros(nb_tests)
z_3R_spec = zeros(nb_tests)

w_exact = zeros(nb_tests)
w_upper = zeros(nb_tests)
w_lower = zeros(nb_tests)

for (i,Nb) in enumerate(Nb_list)
   
    println("Nb per atom ",Nb)
    
    HB = local_hermite_basis(R, Nb, Nb, x_range, δx)
    S_HB = Symmetric(HB'HB)
    Mass_HB = Symmetric(HB'*(H*HB))
    PChol = cholesky(Mass_HB, RowMaximum(), check = false)
    r = rank(Mass_HB, atol=1e-8, rtol=1e-8)
    P = PChol.p[1:r]
    S_HB = S_HB[P,P]
    Mass_HB = Mass_HB[P,P]
    
    # pb eigenvalues
    μ_HB, C_HB = eigen(Mass_HB, S_HB)
    C_HB_rdc = zeros(2*Nb)
    C_HB_rdc[P] = C_HB[:,1]
    u_1N = abs.(HB * C_HB_rdc)
  
    # garanteed bound
    x[i] = (u_1N - u_FD)'H*(u_1N - u_FD)
    x_L2[i] = (u_1N - u_FD)'*(u_1N - u_FD)
    Res = μ_HB[1]*u_1N - H*u_1N
    w_ei = exact_solution(Ng, V, σ, Res, (x_range,δx))
    y[i] = Res'w_ei

    # asymptotically convergent garanteed bound
    Cgap =  1.0/((dvals_H[2] - μ_HB[1])/dvals_H[2])^2
    println("cste gap ", Cgap)
    corr = ((u_1N - u_FD)'*(u_1N - u_FD))^2 * dvals_H[1]/4.0
    y_as[i] = Cgap * Res'w_ei + corr

    @assert( √(Res'H*Res) >= √(Res'Res) )
    @assert( √(Res'Hk*Res) >= √(Res'Res) )

    # two partitions
    pR2_Res = (pR2 .^ (1/2)) .* Res
    wR2 = exact_solution(Ng, V1, σk, pR2_Res, (x_range, δx))
    z_2[i] = Cgap * C2^2 * (2*pR2_Res'wR2)

    # three partitions
    pR_Res = (pR .^ (1/2)) .* Res
    p∞_Res = ((pneg∞ + ppos∞) .^ (1/2)) .* Res
    wR = exact_solution(Ng, V1, σk, pR_Res, (x_range, δx))
    w∞ = exact_solution(Ng, V∞, σ∞, p∞_Res, (x_range, δx))
    z_3[i] = Cgap * C3^2 * (2*pR_Res'wR + p∞_Res'w∞)

    # three restricted partitions
    pR_Res_rs = pR_Res[maskR]
    p∞_Res_rs = p∞_Res[mask∞]
    wR_rs = exact_solution(Ng_R, V1, σk, pR_Res_rs, (x_range_R, δx))
    w∞_rs = exact_solution(Ng_∞, V∞, σ∞, p∞_Res_rs, (x_range_∞, δx))
    z_3R[i] = Cgap * C3^2 * (2*pR_Res_rs'wR_rs + p∞_Res_rs'w∞_rs)

    """
    Spectral approximation for radial part
    """
    # Upper and lower bound
    sR,sRbis = 0,0
    K_target = K-1
    for j in 1:K_target
        ipR = (pR_Res_rs'dvecs_R[:,j])^2
        sRbis += ipR
        sR += 1.0/dvals_R[j] * ipR
    end
    w_exact[i] = pR_Res_rs'wR_rs
    w_lower[i] = sR
    w_upper[i] = (sR +  1.0/dvals_R[K_target] * (pR_Res_rs'pR_Res_rs - sRbis))
    
    # estimator with spectral projection of radial part
    z_3R_spec[i] = Cgap * C3^2 * (2*sR + p∞_Res_rs'w∞_rs)

    # Spectral projection convergence wrt nb of eigenfunctions
    if Nb == N_target
        resn_K = zeros(K)
        resn_unb_K = zeros(K)
        for k in 1:K
            s,ss = 0,0
            for j in 1:k
                s += 1.0/dvals_R[j] * (pR_Res_rs'dvecs_R[:,j])^2
                ss += 1.0/dvals[j] * (pR_Res'dvecs[:,j])^2
            end
            resn_K[k] = s
            resn_unb_K[k] = ss
        end
        resn_R_rs = pR_Res_rs'wR_rs
        resn_R = pR_Res'wR
        plot(1:K, resn_R_rs*ones(K), yscale=:log10, labels="exact bounded")
        #plot(1:K, resn_R*ones(K), yscale=:log10, labels="exact unbounded")
        plot!(1:K, resn_K, yscale=:log10,labels="spectral projection",marker=:diamond,
        #plot!(1:K, resn_unb_K, labels="spectral projection",marker=:diamond,
              ylabel="dual (compact H1)-norm, Hermite N=$(Nb*2)", 
          xlabel="K eigenfunctions of compact H1 operator")
        savefig("img/spectral_$(2*Nb).png")
    end
    
end

plot(2*Nb_list,w_exact,yscale=:log10,xlabel="N",labels="exact dual norm")
plot!(2*Nb_list,w_upper,yscale=:log10,labels="upper bound")
plot!(2*Nb_list,w_lower,yscale=:log10,labels="lower bound",title="K=$(K-1)")
savefig("img/dual_norm_$(K).png")


plot(2*Nb_list,z_2./x, yscale=:log10,labels=L"[C^2 \sum_{k=1}^{K+1}\|√p_k Res u_{1N}\|_{H_k^{-1}}^2]/\|u - u_{1N}\|^2_H",xlabel="N Hermite")
savefig("img/rapport.png")

plot(2*Nb_list,x, marker=:circle,yscale=:log10,labels=L"\|u - u_{1N}\|^2_H")
plot!(2*Nb_list,x_L2,yscale=:log10,labels=L"\|u - u_{1N}\|^2_{L^2}")
plot!(2*Nb_list,y_as, marker=:square,yscale=:log10,labels=L"\|Res u_{1N}\|^2_{H^{-1}} (prop 5.3)",xlabel="N Hermite")
plot!(2*Nb_list,z_2, yscale=:log10,labels="ESTIM 2Pk",xlabel="N Hermite")
plot!(2*Nb_list,z_3, yscale=:log10,labels="ESTIM 3Pk",xlabel="N Hermite")
plot!(2*Nb_list,z_3R, yscale=:log10,labels="ESTIM 3RPk",xlabel="N Hermite")
plot!(2*Nb_list,z_3R_spec, marker=:star, yscale=:log10,labels="ESTIM 3RPk K=$(K-1)",xlabel="N Hermite",legend=:outertopright)
savefig("img/prop41_$(K).png")


