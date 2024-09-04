using Plots
gr()

include("../src/quadrature.jl")

"""
Hermite approximation
"""

N_list = 3:3:12
r_list = -5:0.01:5
plt = plot(r_list, x -> exp(-abs(x)), label=L"\exp(-r)")
for N in N_list
  x,w = gausshermite(N)
  plot!(r_list,[dot(w,f.(x;r=r)) for r in r_list],label="N=$N",linestyle=:dot)
end
display(plt)

plt_herm_err = scatter(sqrt.(3:3:300),hermite_error.(3:3:300), xlabel="√N", yscale=:log10,label="Gauss-Hermite quadrature error")
display(plt_herm_err)

"""
Contour integration approximation
"""

#R, ε = 100, 1e-3
#c_R,c_ε = circle_coefficient(R,ε)
#xl,wl = line_coefficients(R,ε;N=1000)
#
#plt2 = plot(r_list, x -> exp(-abs(x)), label=L"\exp(-r)")
#plot!(r_list, r-> contour_approx(r,R,ε,c_R,c_ε,xl,wl),label="Contour approximation")

"""
Boyd integration with change of variables
"""

N = 10
xb,wb = boyd_x_w(N)
plt_boyd = plot(r_list,[boyd_approx(r,xb,wb) for r in r_list],label = "Boyd approximation")
plot!(r_list, x -> exp(-abs(x)), label=L"\exp(-r)")
display(plt_boyd)

plt_boyd_err = scatter(sqrt.(3:3:30),boyd_error_L2.(3:3:30), xlabel="√N",label="L2 Boyd quadrature error",yscale=:log10)
display(plt_boyd_err)

plt_exponent = scatter(1 ./ exp.(2xb) , yscale=:log10, label = "Exponent in Gaussians")

"""
Boyd integration without change of variables
"""

N = 10
xb,wb = boyd_x_w(N)
plt_boyd2 = plot(r_list,[boyd_approx(r,xb,wb;g=h) for r in r_list],label = "Boyd approximation")
plot!(r_list, x -> exp(-abs(x)), label=L"\exp(-r)")
display(plt_boyd2)

plt_boyd_err2 = scatter(sqrt.(3:3:30),boyd_error_L2.(3:3:30;g=h), xlabel="√N",label="Boyd quadrature error",yscale=:log10)
display(plt_boyd_err2)

plt_exponent = scatter(1 ./ (4*xb.^2) , yscale=:log10, label = "Exponent in Gaussians")


"""
Errors of all the methods
"""

error_plot = scatter(sqrt.(3:3:60),hermite_error.(3:3:300), xlabel="√N", yscale=:log10,label="Gauss-Hermite quadrature error")
scatter!(sqrt.(3:3:60),boyd_error_L2.(3:3:60), xlabel="√N",label="Boyd with change of variable quadrature error",yscale=:log10)
scatter!(sqrt.(3:3:60),boyd_error_L2.(3:3:60;g=h), xlabel="√N",label="Boyd vanilla quadrature error",yscale=:log10)
display(error_plot)

"""
Boyd semiline
"""
N = 10
xbs,wbs = boyd_semiline_x_w(N)
plt_boyd_semiline = plot(r_list,[boyd_approx_semiline(r,xbs,wbs) for r in r_list])
plot!(r_list, x -> exp(-abs(x)), label=L"\exp(-r)")
display(plt_boyd_semiline)

"""
Boyd L² vs H¹ error 
"""

scatter(sqrt.(3:3:60),boyd_error_L2.(3:3:60), xlabel="√N",label="Boyd error L^2",yscale=:log10)
scatter!(sqrt.(3:3:60),boyd_error_H1.(3:3:60), xlabel="√N",label="Boyd error H^1",yscale=:log10)
