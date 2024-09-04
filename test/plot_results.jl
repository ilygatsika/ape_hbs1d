using JSON3

include("../src/plot_conf.jl")

input_dir = "out"
output_dir = "img"

# read Main results
data = open(JSON3.read, "$(input_dir)/res_main.json")
Nb_list = Int64.(data["Nb_list"])
vec_ℓ = Float64.(data["ℓ_list"])
nb_ℓ = length(vec_ℓ)
nb_tests = length(Nb_list)
Herr_src = reshape(data["Herr_src"], (nb_ℓ, nb_tests))
Hest_src = reshape(data["Hest_src"], (nb_ℓ, nb_tests))
Herr_eig = reshape(data["Herr_eig"], (nb_ℓ, nb_tests))
Hest_eig = reshape(data["Hest_eig"], (nb_ℓ, nb_tests))
eigv_err = reshape(data["eigv_err"], (nb_ℓ, nb_tests))
eigv_est = reshape(data["eigv_est"], (nb_ℓ, nb_tests))

# Error convergence for source problem wrt Hermite basis size
#PyPlot.title("source problem")
PyPlot.plot(Nb_list, Herr_src[1,:], marker="s", color="blue", linewidth=2, label=L"$\|u-u_N\|_H$")
m = ["o", "x", "^", "*","d"]
for i in 1:nb_ℓ
    PyPlot.plot(Nb_list, Hest_src[i,:], marker=m[i],  label=L"$2\ell=%$(2*vec_ℓ[i])$")
end
PyPlot.ylabel("discr. error")
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.legend()
PyPlot.savefig("$(output_dir)/src_pb.pdf")
PyPlot.close()

# Error convergence for eigval problem wrt Hermite basis size
#PyPlot.title("eigenvalue problem")
#PyPlot.title(L"$z_1=z_2=1,\sigma=4,\sigma_1=\sigma_2=3$")
PyPlot.plot(Nb_list, Herr_eig[1,:], marker="s", color="blue", linewidth=2, label=L"$\|u_1 - u_{1N}\|_H$")
m = ["o", "x", "^", "*","d"]
for i in 1:nb_ℓ
    PyPlot.plot(Nb_list, Hest_eig[i,:], marker=m[i], label=L"$2\ell=%$(2*vec_ℓ[i])$")
end
PyPlot.ylabel("discr. error")
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.legend()
PyPlot.savefig("$(output_dir)/eig_pb.pdf")
PyPlot.close()

# Error on eigenvalue
PyPlot.plot(Nb_list, eigv_err[1,:], marker="s", color="blue", linewidth=2, label=L"$\lambda_{1N} - \lambda_1$")
m = ["o", "x", "^", "*","d"]
for i in 1:nb_ℓ
    PyPlot.plot(Nb_list, eigv_est[i,:], marker=m[i], label=L"$2\ell=%$(2*vec_ℓ[i])$")
end
PyPlot.ylabel("discr. error")
PyPlot.xlabel(L"N"*" basis functions")
PyPlot.yscale("log")
PyPlot.legend()
PyPlot.savefig("$(output_dir)/eigval_err.pdf")
PyPlot.close()

# read Adaptivity results
data = open(JSON3.read, "$(input_dir)/res_adapt.json")
Nb12_list = Int64.(data["Nb12_list"])
Nb_list = Int64.(data["Nb_list"])
Herr = Float64.(data["Herr_eig"])
Hest = Float64.(data["Hest_eig"])
Herr_na = Float64.(data["Herr_eig_na"])
Hest_na = Float64.(data["Hest_eig_na"])

m = length(Herr_na)
n = length(Herr)
PyPlot.figure(figsize=(6,4.7))
#PyPlot.title(L"z_1=1, z_2=1.03")
slope1 = (Herr_na[m] - Herr_na[1])/(Nb_list[m] - Nb_list[1])
slope2 = (Herr[n] - Herr[1])/(Nb12_list[n] - Nb12_list[1])
print(slope1, slope2)
PyPlot.plot(Nb_list, Herr_na, marker="^", color="green", label=L"N_1=N_2")
PyPlot.plot(Nb12_list, Herr, marker="o", color="orange", label="adaptive")
PyPlot.ylabel("discr. error")
PyPlot.xlabel(L"N=N_1+N_2"*" discretization basis functions")
PyPlot.yscale("log")
PyPlot.legend()
PyPlot.savefig("$(output_dir)/adapt.pdf")
PyPlot.close()



