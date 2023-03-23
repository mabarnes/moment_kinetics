if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral
    import LinearAlgebra
    using LinearAlgebra: mul!
    
    function Djj(x::Array{Float64,1},j::Int64)
        return -0.5*x[j]/( 1.0 - x[j]^2)
    end
    function Djk(x::Array{Float64,1},j::Int64,k::Int64,c_j::Float64,c_k::Float64)
        return  (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
    end
    
    function cheb_derivative_matrix!(D::Array{Float64,2},x::Array{Float64,1},n) 
        D[:,:] .= 0.0
        
        # top left, bottom right
        D[1,1] = (2.0*(n - 1.0)^2 + 1.0)/6.0
        D[n,n] = -(2.0*(n - 1.0)^2 + 1.0)/6.0
        
        # top row 
        j = 1
        c_j = 2.0 
        c_k = 1.0
        for k in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        k = n 
        c_k = 2.0
        D[j,k] = Djk(x,j,k,c_j,c_k)
        
        # bottom row 
        j = n
        c_j = 2.0 
        c_k = 1.0
        for k in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        k = 1
        c_k = 2.0
        D[j,k] = Djk(x,j,k,c_j,c_k)
        
        #left column
        k = 1
        c_j = 1.0 
        c_k = 2.0
        for j in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        
        #right column
        k = n
        c_j = 1.0 
        c_k = 2.0
        for j in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        
        # interior rows and columns
        for j in 2:n-1
            D[j,j] = Djj(x,j)
            #D[j,j] = -0.5*x[j]/( 1.0 - x[j]^2)
            for k in 2:n-1
                if j == k 
                    continue
                end
                c_k = 1.0
                c_j = 1.0
                #D[j,k] = (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
                D[j,k] = Djk(x,j,k,c_j,c_k)
            end
        end
    end 
    
    function cheb_derivative_matrix_reversed!(D::Array{Float64,2},x) 
        D_elementwise = Array{Float64,2}(undef,x.ngrid,x.ngrid)
        cheb_derivative_matrix_elementwise_reversed!(D_elementwise,x.ngrid,x.L,x.nelement_global)
        
        # zero matrix before assignment 
        println("D_elementwise \n ",D_elementwise)
        D[:,:] .= 0.0
        imin = x.imin
        imax = x.imax
        println(imin)
        println(imax)
        # fill in first element 
        j = 1
        if x.bc == "zero"
            D[imin[j],imin[j]:imax[j]] .+= D_elementwise[1,:]./2.0
        else 
            D[imin[j],imin[j]:imax[j]] .+= D_elementwise[1,:]
        end
        for k in 2:imax[j]-imin[j] 
            D[k,imin[j]:imax[j]] .+= D_elementwise[k,:]
        end
        if x.nelement_local > 1 || x.bc == "zero"
            D[imax[j],imin[j]:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0
        else
            D[imax[j],imin[j]:imax[j]] .+= D_elementwise[x.ngrid,:]
        end 
        # remaining elements recalling definitions of imax and imin
        for j in 2:x.nelement_local
            #lower boundary condition on element
            D[imin[j]-1,imin[j]-1:imax[j]] .+= D_elementwise[1,:]./2.0
            for k in 2:imax[j]-imin[j]+1 
                D[k+imin[j]-2,imin[j]-1:imax[j]] .+= D_elementwise[k,:]
            end
            # upper boundary condition on element 
            if j == x.nelement_local && !(x.bc == "zero")
                D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]
            else 
                D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0
            end
        end
        
    end
    
    function cheb_derivative_matrix_elementwise_reversed!(D::Array{Float64,2},n::Int64,L::Float64,nelement::Int64) 
        
        #define Chebyshev points in reversed order x_j = { -1, ... , 1}
        x = Array{Float64,1}(undef,n)
        for j in 1:n
            x[j] = cospi((n-j)/(n-1))
        end
        
        # zero matrix before allocating values
        D[:,:] .= 0.0
        
        # top row 
        j = 1
        c_j = 2.0 
        c_k = 1.0
        for k in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        k = n 
        c_k = 2.0
        D[j,k] = Djk(x,j,k,c_j,c_k)
        
        # bottom row 
        j = n
        c_j = 2.0 
        c_k = 1.0
        for k in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        k = 1
        c_k = 2.0
        D[j,k] = Djk(x,j,k,c_j,c_k)
        
        #left column
        k = 1
        c_j = 1.0 
        c_k = 2.0
        for j in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        
        #right column
        k = n
        c_j = 1.0 
        c_k = 2.0
        for j in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        
        
        # top left, bottom right
        #D[n,n] = (2.0*(n - 1.0)^2 + 1.0)/6.0
        #D[1,1] = -(2.0*(n - 1.0)^2 + 1.0)/6.0        
        # interior rows and columns
        for j in 2:n-1
            #D[j,j] = Djj(x,j)
            for k in 2:n-1
                if j == k 
                    continue
                end
                c_k = 1.0
                c_j = 1.0
                D[j,k] = Djk(x,j,k,c_j,c_k)
            end
        end
        
        # calculate diagonal entries to guarantee that
        # D * (1, 1, ..., 1, 1) = (0, 0, ..., 0, 0)
        for j in 1:n
            D[j,j] = -sum(D[j,:])
        end
        
        #multiply by scale factor for element length
        D .= (2.0*float(nelement)/L).*D
    end 
    
    #using LinearAlgebra.mul
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	etol = 1.0e-15
    outprefix = "derivative_test"
	###################
	## df/dx Nonperiodic (No) BC test
	###################
	
	# define inputs needed for the test
	ngrid = 17 #number of points per element 
	nelement_local = 4 # number of elements per rank
	nelement_global = nelement_local # total number of elements 
	L = 1.0 #physical box size in reference units 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = "fourth_order_centered"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = false
	# create the 'input' struct containing input info needed to create a
	# coordinate
    input = grid_input("coord", ngrid, nelement_global, nelement_local, 
		nrank, irank, L, discretization, fd_option, bc, adv_input,comm)
	# create the coordinate struct 'x'
	println("made inputs")
	x = define_coordinate(input)
	println("made x")
    Dx = Array{Float64,2}(undef, x.n, x.n)
    xchebgrid = Array{Float64,1}(undef, x.n)
    for i in 1:x.n
        xchebgrid[i] = cos(pi*(i - 1)/(x.n - 1))
    end
    println("x",xchebgrid[:])
    cheb_derivative_matrix!(Dx,xchebgrid,x.n)
    println("")
    println("Dx \n")
    for i in 1:x.n
        println(Dx[i,:])
    end
    
    # create array for the function f(x) to be differentiated/integrated
	f = Array{Float64,1}(undef, x.n)
	# create array for the derivative df/dx
	df = Array{Float64,1}(undef, x.n)
    df_exact = Array{Float64,1}(undef, x.n)
    df_err = Array{Float64,1}(undef, x.n)

    for ix in 1:x.n
        f[ix] = sin(pi*xchebgrid[ix])
        df_exact[ix] = (pi)*cos(pi*xchebgrid[ix])
    end
    mul!(df,Dx,f)
    for ix in 1:x.n
        df_err[ix] = df[ix]-df_exact[ix]
    end
    println("df \n",df)
    println("df_exact \n",df_exact)
    println("df_err \n",df_err)
    
    Dxreverse = Array{Float64,2}(undef, x.n, x.n)
    cheb_derivative_matrix_reversed!(Dxreverse,x)
    
    println("x.grid \n",x.grid)
    println("")
    println("Dxreverse \n")
    for i in 1:x.n
        println(Dxreverse[i,:])
    end
    
    for ix in 1:x.n
        f[ix] = sin(2.0*pi*x.grid[ix]/x.L)
        df_exact[ix] = (2.0*pi/x.L)*cos(2.0*pi*x.grid[ix]/x.L)
    end
    mul!(df,Dxreverse,f)
    for ix in 1:x.n
        df_err[ix] = df[ix]-df_exact[ix]
    end
    println("Reversed")
    println("df \n",df)
    println("df_exact \n",df_exact)
    println("df_err \n",df_err)
    
    
end