"""
Test of several approximations of e^{-r} by Gaussians
"""

using FastGaussQuadrature
using LaTeXStrings
using LinearAlgebra
using SpecialFunctions
using QuadGK
using ForwardDiff

#integrand in e^{-r} = 1/√(pi)∫_ℝ exp(-x²)exp(-r^2/(4x^2)) dx
function f(x;r=1.0) 
  return 1/√(pi)*exp(-r^2/(4x^2))
end 

"""
Gauss-Hermite quadrature 
"""

function hermite_error(N;a=10)
  x,w = gausslegendre(1001)
  xh,wh = gausshermite(N)
  error(r) =(exp(-abs(r)) - dot(wh,f.(xh;r=r)) )^2
  return sqrt(a*dot(w,error.(a*x)))
end

"""
Boyd without change of variables
"""

function boyd_semiline_x_w(N;L=1.0)
  w = zeros(N)
  x = Array(1:N)*pi/(N+1)
  for i in 1:N
    w[i] = 2L*sin(x[i])/(1-cos(x[i]))^2*(2/(N+1))*sum(sin(j*x[i])*(1-cos(j*pi))/j for j in 1:N)
  end
  return L*cot.(0.5*x).^2, w
end


function boyd_approx_semiline(r,x,w)
  g(u) = 2/sqrt(pi)*exp(-u^2)*exp(-r^2/(4u^2))
  return dot(w,g.(x))
end


"""
Boyd quadrature after change of variables
"""

#Boyd weights and nodes on ℝ, works for exp(-x^2)
function boyd_x_w(N;L=1.0)
  w = zeros(N)
  x = Array(1:N)*pi/(N+1)
  for i in 1:N
    w[i] = L*pi/((N+1)*sin(x[i])^2)
  end
  return L*cot.(x),w
end

# integrand in 2/sqrt(pi)*∫_0^∞ exp(-x²) exp(-(r²/(4x²))) dx after change of variable x = exp(u).
g(u;r=1.0) = 2/sqrt(pi)*exp(-exp(2u))*exp(u)*exp(-r^2/(4exp(2u)))
h(x;r=1.0) = 1/sqrt(pi)*exp(-x^2)*exp(-r^2/(4x^2))

function boyd_approx(r,x,w;g=g)
  return dot(w,g.(x;r=r))
end

"""
boyd_error_L2: 
  - output: ||exp(-Zr) - Boyd_{N,L,g}(Zr)||²_{L²} computed by Gauss-Legendre quadrature
  - input:
    - N: number of quadrature points in Boyd quadrature
    - L: parameter in Boyd (L=1 by default in the paper)
    - Z: scaling in the variable
    - g: function to approximate by Boyd quadrature
"""
function boyd_error_L2(N;L=1.0,a=10,g=g)
  x,w = gausslegendre(1001)
  xb,wb = boyd_x_w(N;L=L)
  error(r) =(exp(-abs(r)) - dot(wb,g.(xb;r=r)) )^2
  return sqrt(a*dot(w,error.(a*x)))
end

"""
boyd_error_H1: 
  - output: ||exp(-Zr)' - Boyd_{N,L,g}(Zr)'||_{L²} computed by Gauss-Legendre quadrature
  - input:
    - N: number of quadrature points in Boyd quadrature
    - L: parameter in Boyd (L=1 by default in the paper)
    - Z: scaling in the variable
    - g: function to approximate by Boyd quadrature
"""
function boyd_error_H1(N;L=1.0,a=10,g=g)
  x,w = gausslegendre(1000)
  xb,wb = boyd_x_w(N;L=L)
  error(r) =(-sign(r)*exp(-abs(r)) - dot(wb, [ForwardDiff.derivative(y -> g(x;r=y),r) for x in xb]) )^2
  return sqrt(a^2*dot(w,error.(a*x)))
end

"""
Contour integral 
ça ne marche pas !!!!
"""

function integrand_semicircle(a)
  if isapprox(a,0,atol=1e-8)
    return pi
  elseif isapprox(a,1)
    return 1.5pi
  else
    return 0.5*(pi-4atan((a+1)/(a-1)))
  end
end

function integrand_line(x;y=1)
  return x*exp(-x^2)/(x^2+y^2)/pi
end

#plt2 = plot(0.01:0.01:4,integrand_line)
#display(plt2)

function line_weight(R,ε;y=1)
  value,err = quadgk(x -> integrand_line(x;y=y),ε,R)
  return value
end

function circle_coefficient(R,ε)
  g(x,a) = exp(-x^2)*integrand_semicircle(x/a)
  vR,errR = quadgk(x->g(x,R),ε,R)
  vε,errε = quadgk(x->g(x,ε),ε,R)
  return (vR/(2pi),-vε/(2pi))
end

#works
function line_coefficients(R,ε;N=5)
  xl,wl = gausslegendre(N)
  wl = 0.5*(R-ε)*wl
  xl = 0.5*(R-ε)*xl .+ 0.5*(R+ε)
  return xl,wl
end

function contour_approx(r,R,ε,c_R,c_ε,xl,wl)
  #c_R*exp(-r^2/(4R^2))+ c_ε *exp(-r^2/(4ε^2)) + dot(wl,g.(xl))
  g(x) = exp(-r^2/(4x^2))*line_weight(R,ε;y=x)
  return 2/sqrt(pi)*( c_R*exp(-r^2/(4R^2)) + c_ε * exp(-r^2/(4ε^2)) + dot(wl,g.(xl)) )
end