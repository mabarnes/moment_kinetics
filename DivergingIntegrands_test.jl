using FastGaussQuadrature
using SpecialFunctions: ellipk
using Printf

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    function print_matrix(matrix,name,n,m)
        println("\n ",name," \n")
        for i in 1:n
            for j in 1:m
                @printf("%.3f ", matrix[i,j])
            end
            println("")
        end
        println("\n")
    end
    
    function print_vector(vector,name,m)
        println("\n ",name," \n")
        for j in 1:m
            @printf("%.16f ", vector[j])
        end
        println("")
        println("\n")
    end 

    # gauss laguerre test 
    ngrid = 24
    nelement = 1
    x, w = gausslaguerre(ngrid)
    print_vector(x,"Gauss Laguerre x",ngrid)
    print_vector(w,"Gauss Laguerre w",ngrid)
    
    # integrate 1/sqrt(y) from y = 0 to 1
    # use the Gauss-Laguerre quadrature to
    # convert the diverging to a converging integrand
    integrand = Array{Float64,1}(undef,ngrid)
    integrand_sqrty = Array{Float64,1}(undef,ngrid)
    integrand_Kz = Array{Float64,1}(undef,ngrid)
    value_Kz = Array{Float64,1}(undef,ngrid)
    integrand_Kz_sqrt = Array{Float64,1}(undef,ngrid)
    value_Kz_sqrt = Array{Float64,1}(undef,ngrid)
    integrand_Kz_sqrt_diff = Array{Float64,1}(undef,ngrid)
    value_Kz_sqrt_diff = Array{Float64,1}(undef,ngrid)
    value_y = Array{Float64,1}(undef,ngrid)
    value_z = Array{Float64,1}(undef,ngrid)
    L = 1.0
    for i in 1:ngrid
        # change of variables
        y = exp(-x[i]/L)
        # function to integrate in terms of y
        integrand[i] = sqrt(1.0/y)*w[i]
        integrand_sqrty[i] = sqrt(y)*w[i]
        #z = (1.0 - y)
        z = (1.0 - y)/(1.0 + 10^-15)
        ellipk_z = ellipk(z)
        #if isnan(ellipk_z) || isinf(ellipk_z)
        #    ellipk_z = 0.0
        #end
        sqrt_1_z = sqrt(1.0 - z)
        #if isinf(sqrt_1_z) || isinf(sqrt_1_z)
        #    sqrt_1_z = 1.0
        #end
        integrand_Kz[i] = ellipk_z*w[i]
        value_Kz[i] = ellipk_z
        integrand_Kz_sqrt[i] = ellipk_z*w[i]/sqrt_1_z
        value_Kz_sqrt[i] = ellipk_z/sqrt_1_z
        value_Kz_sqrt_diff[i] = (ellipk_z - log(4.0) + log(sqrt_1_z))/sqrt_1_z
        integrand_Kz_sqrt_diff[i] = value_Kz_sqrt_diff[i]*w[i]
        value_z[i] = z
    end
    #@. integrand *= w
    
    print_vector(integrand,"Gauss Laguerre integrand",ngrid)
    
    primitive = sum(integrand)
    primitive_exact = 2.0
    primitive_err = abs(primitive - primitive_exact)
    println("1/sqrt(y): Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
    
    primitive = sum(integrand_sqrty)
    primitive_exact = 2.0/3.0
    primitive_err = abs(primitive - primitive_exact)
    println("sqrt(y): Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
    
    print_vector(integrand_Kz,"Kz integrand",ngrid)
    print_vector(value_Kz,"Kz",ngrid)
    print_vector(value_z,"z",ngrid)
    primitive = sum(integrand_Kz)
    primitive_exact = 2.0
    primitive_err = abs(primitive - primitive_exact)
    println("K(z): Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)

    print_vector(integrand_Kz_sqrt,"Kz/sqrt(1-z) integrand",ngrid)
    print_vector(value_Kz_sqrt,"Kz/sqrt(1-z)",ngrid)
    print_vector(value_z,"z",ngrid)
    primitive = sum(integrand_Kz_sqrt)
    primitive_exact = pi^2/2.0
    primitive_err = abs(primitive - primitive_exact)
    println("K(z)/sqrt(1-z): Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
   
    print_vector(integrand_Kz_sqrt_diff,"(Kz - log 4/sqrt(1-z))/sqrt(1-z) integrand",ngrid)
    print_vector(value_Kz_sqrt_diff,"(Kz - log 4/sqrt(1-z))/sqrt(1-z)",ngrid)
    print_vector(value_z,"z",ngrid)
    primitive = sum(integrand_Kz_sqrt_diff)
    primitive_exact = (pi^2/2.0) - 2.0*(2.0*log(2.0) + 1.0)
    primitive_err = abs(primitive - primitive_exact)
    println("(K(z) - log(4/sqrt(1-z)))/sqrt(1-z): Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
   
    if false   
        # gauss lobatto test 
        ngrid = 10
        nelement = 1
        x, w = gausslegendre(ngrid)
        print_vector(x,"Gauss Legendre x",ngrid)
        print_vector(w,"Gauss Legendre w",ngrid)
        
        b = 1.0 # upper limit on element
        a = 0.0 # lower limit on element
        U = 100.0 # proxy for infinity
        L = 1.0 # scale factor
        z = copy(x)
        y = copy(x)
        wy =copy(x)
        scale = U*(b-a)/L
        @. z = U*(x + 1)
        @. y = exp(-z/L)
        @. wy = w*exp(-z/L)*scale
        # integrate 1/sqrt(y) from y = 0 to 1
        # use the Gauss-Laguerre quadrature to
        # convert the diverging to a converging integrand
        integrand = Array{Float64,1}(undef,ngrid)
        integrand_sqrty = Array{Float64,1}(undef,ngrid)
        L = 1.0
        for i in 1:ngrid
            # function to integrate in terms of y
            integrand[i] = sqrt(1.0/y[i])*wy[i]#*w[i]*exp(-z[i]/L)*scale
            integrand_sqrty[i] = sqrt(y[i])*wy[i]#*w[i]*exp(-z[i]/L)*scale
        end
        #@. integrand *= w
        
        print_vector(w,"Gauss Legendre weights",ngrid)
        print_vector(y,"Gauss Legendre coordinate",ngrid)
        print_vector(integrand,"Gauss Legendre integrand",ngrid)
        
        primitive = sum(integrand)
        primitive_exact = 2.0
        primitive_err = abs(primitive - primitive_exact)
        println("Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
        
        primitive = sum(integrand_sqrty)
        primitive_exact = 2.0/3.0
        primitive_err = abs(primitive - primitive_exact)
        println("Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
    end
end