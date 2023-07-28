using FastGaussQuadrature
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
            @printf("%.3f ", vector[j])
        end
        println("")
        println("\n")
    end 

    # gauss lobatto test 
    ngrid = 100
    nelement = 1
    x, w = gausslaguerre(ngrid)
    print_vector(x,"Gauss Laguerre x",ngrid)
    print_vector(w,"Gauss Laguerre w",ngrid)
    
    # integrate 1/sqrt(y) from y = 0 to 1
    # use the Gauss-Laguerre quadrature to
    # convert the diverging to a converging integrand
    integrand = Array{Float64,1}(undef,ngrid)
    for i in 1:ngrid
        # change of variables
        y = exp(-x[i])
        # function to integrate in terms of y
        integrand[i] = sqrt(1.0/y)
    end
    @. integrand *= w
    
    print_vector(integrand,"Gauss Laguerre integrand",ngrid)
    
    primitive = sum(integrand)
    primitive_exact = 2.0
    primitive_err = abs(primitive - primitive_exact)
    println("Primitive: ",primitive," should be: ",primitive_exact," error: ",primitive_err)
    
end