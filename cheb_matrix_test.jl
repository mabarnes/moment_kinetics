using Printf
using Plots
using LaTeXStrings
using MPI
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral
    #import LinearAlgebra
    using LinearAlgebra: mul!, lu, cond
    using SparseArrays: sparse
    using SpecialFunctions: erf
    zero = 1.0e-10
    
    function print_matrix(matrix,name,n,m)
        println("\n ",name," \n")
        for i in 1:n
            for j in 1:m
                @printf("%.1f ", matrix[i,j])
            end
            println("")
        end
        println("\n")
    end 
    
    function Djj(x::Array{Float64,1},j::Int64)
        return -0.5*x[j]/( 1.0 - x[j]^2)
    end
    function Djk(x::Array{Float64,1},j::Int64,k::Int64,c_j::Float64,c_k::Float64)
        return  (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
    end
    
    """
    The function below is based on the numerical method outlined in 
    Chapter 8.2 from Trefethen 1994 
    https://people.maths.ox.ac.uk/trefethen/8all.pdf
    full list of Chapters may be obtained here 
    https://people.maths.ox.ac.uk/trefethen/pdetext.html
    """
    
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
        if x.ngrid < 8
            println("\n D_elementwise \n")
            for i in 1:x.ngrid
                for j in 1:x.ngrid
                    @printf("%.1f ", D_elementwise[i,j])
                end
                println("")
            end
        end 
        assign_cheb_derivative_matrix!(D,D_elementwise,x)
    end
    
    function cheb_second_derivative_matrix_reversed!(D::Array{Float64,2},x) 
        D_elementwise = Array{Float64,2}(undef,x.ngrid,x.ngrid)
        cheb_derivative_matrix_elementwise_reversed!(D_elementwise,x.ngrid,x.L,x.nelement_global)    
        D2_elementwise = Array{Float64,2}(undef,x.ngrid,x.ngrid)
        mul!(D2_elementwise,D_elementwise,D_elementwise)
        if x.ngrid < 8
            println("\n D2_elementwise \n")
            for i in 1:x.ngrid
                for j in 1:x.ngrid
                    @printf("%.1f ", D2_elementwise[i,j])
                end
                println("")
            end
        end
        assign_cheb_derivative_matrix!(D,D2_elementwise,x)
    end
    
    function assign_cheb_derivative_matrix!(D::Array{Float64,2},D_elementwise::Array{Float64,2},x) 
        
        # zero output matrix before assignment 
        D[:,:] .= 0.0
        imin = x.imin
        imax = x.imax
        
        zero_bc_upper_boundary = x.bc == "zero" || x.bc == "zero_upper"
        zero_bc_lower_boundary = x.bc == "zero" || x.bc == "zero_lower"
        
        # fill in first element 
        j = 1
        if zero_bc_lower_boundary #x.bc == "zero"
            D[imin[j],imin[j]:imax[j]] .+= D_elementwise[1,:]./2.0 #contributions from this element/2
            D[imin[j],imin[j]] += D_elementwise[x.ngrid,x.ngrid]/2.0 #contribution from missing `zero' element/2
        else 
            D[imin[j],imin[j]:imax[j]] .+= D_elementwise[1,:]
        end
        for k in 2:imax[j]-imin[j] 
            D[k,imin[j]:imax[j]] .+= D_elementwise[k,:]
        end
        if zero_bc_upper_boundary && x.nelement_local == 1
            D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0 #contributions from this element/2
            D[imax[j],imax[j]] += D_elementwise[1,1]/2.0              #contribution from missing `zero' element/2
        elseif x.nelement_local > 1 #x.bc == "zero"
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
            if j == x.nelement_local && !(zero_bc_upper_boundary)
                D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]
            elseif j == x.nelement_local && zero_bc_upper_boundary
                D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0 #contributions from this element/2
                D[imax[j],imax[j]] += D_elementwise[1,1]/2.0 #contribution from missing `zero' element/2
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
    
    """
    function integrating d y / d t = f(t)
    """
    function forward_euler_step!(ynew,yold,f,dt,n)
        for i in 1:n
            ynew[i] = yold[i] + dt*f[i]
        end
    end
    """
    function creating lu object for A = I - dt*nu*D2
    """
    function diffusion_matrix(D2,n,dt,nu;return_A=false)
        A = Array{Float64,2}(undef,n,n)
        for i in 1:n
            for j in 1:n
                A[i,j] = - dt*nu*D2[i,j]
            end
            A[i,i] += 1.0
        end
        lu_obj = lu(A)
        if return_A
            return lu_obj, A
        else
            return lu_obj
        end
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
	ngrid = 2 #number of points per element 
	nelement_local = 10 # number of elements per rank
	nelement_global = nelement_local # total number of elements 
	L = 1.0 #physical box size in reference units 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = "fourth_order_centered"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = MPI.COMM_NULL
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
    #println("x",xchebgrid[:])
    cheb_derivative_matrix!(Dx,xchebgrid,x.n)
    #println("")
    #println("Dx \n")
    #for i in 1:x.n
    #    println(Dx[i,:])
    #end
    
     # create array for the function f(x) to be differentiated/integrated
	f = Array{Float64,1}(undef, x.n)
	# create array for the derivative df/dx
	df = Array{Float64,1}(undef, x.n)
	df2 = Array{Float64,1}(undef, x.n)
	df2cheb = Array{Float64,1}(undef, x.n)
    df_exact = Array{Float64,1}(undef, x.n)
    df2_exact = Array{Float64,1}(undef, x.n)
    df_err = Array{Float64,1}(undef, x.n)
    df2_err = Array{Float64,1}(undef, x.n)
    df2cheb_err = Array{Float64,1}(undef, x.n)

    for ix in 1:x.n
        f[ix] = sin(pi*xchebgrid[ix])
        df_exact[ix] = (pi)*cos(pi*xchebgrid[ix])
    end
    mul!(df,Dx,f)
    for ix in 1:x.n
        df_err[ix] = df[ix]-df_exact[ix]
    end
    # test standard cheb D f = df 
    #println("df \n",df)
    #println("df_exact \n",df_exact)
    #println("df_err \n",df_err)
    input = grid_input("coord", ngrid, nelement_global, nelement_local, 
		nrank, irank, L, discretization, fd_option, "zero", adv_input,comm)
	# create the coordinate struct 'x'
	x = define_coordinate(input)
   
    Dxreverse = Array{Float64,2}(undef, x.n, x.n)
    cheb_derivative_matrix_reversed!(Dxreverse,x)
    Dxreverse2 = Array{Float64,2}(undef, x.n, x.n)
    mul!(Dxreverse2,Dxreverse,Dxreverse)
    D2xreverse = Array{Float64,2}(undef, x.n, x.n)
    cheb_second_derivative_matrix_reversed!(D2xreverse,x)
    
    Dxreverse2[1,1] = 2.0*Dxreverse2[1,1]
    Dxreverse2[end,end] = 2.0*Dxreverse2[end,end]
    #println("x.grid \n",x.grid)
    if x.n < 20
        println("\n Dxreverse \n")
        for i in 1:x.n
            for j in 1:x.n
                @printf("%.1f ", Dxreverse[i,j])
            end
            println("")
        end
        println("\n Dxreverse*Dxreverse \n")
        for i in 1:x.n
            for j in 1:x.n
                @printf("%.1f ", Dxreverse2[i,j])
            end
            println("")
        end
        
        println("\n D2xreverse \n")
        for i in 1:x.n
            for j in 1:x.n
                @printf("%.1f ", D2xreverse[i,j])
            end
            println("")
        end
        println("\n")
    end

    alpha = 512.0    
    for ix in 1:x.n
#        f[ix] = sin(2.0*pi*x.grid[ix]/x.L)
#        df_exact[ix] = (2.0*pi/x.L)*cos(2.0*pi*x.grid[ix]/x.L)
#        df2_exact[ix] = -(2.0*pi/x.L)*(2.0*pi/x.L)*sin(2.0*pi*x.grid[ix]/x.L)
 
        f[ix] = exp(-alpha*(x.grid[ix])^2)
        df_exact[ix] = -2.0*alpha*x.grid[ix]*exp(-alpha*(x.grid[ix])^2)
        df2_exact[ix] = ((2.0*alpha*x.grid[ix])^2 - 2.0*alpha)*exp(-alpha*(x.grid[ix])^2)
    end
    #println("test f: \n",f)
    # calculate d f / d x from matrix 
    mul!(df,Dxreverse,f)
    # calculate d^2 f / d x from second application of Dx matrix 
    mul!(df2,Dxreverse2,f)
    # calculate d^2 f / d x from applition of D2x matrix 
    mul!(df2cheb,D2xreverse,f)
    for ix in 1:x.n
        df_err[ix] = df[ix]-df_exact[ix]
        df2_err[ix] = df2[ix]-df2_exact[ix]
        df2cheb_err[ix] = df2cheb[ix]-df2_exact[ix]
    end
    println("Reversed - multiple elements")
    #println("df \n",df)
    #println("df_exact \n",df_exact)
    #println("df_err \n",df_err)
    #println("df2 \n",df2)
    #println("df2_exact \n",df2_exact)
    #println("df2_err \n",df2_err)
    #println("df2cheb_err \n",df2cheb_err)
    
    println("max(df_err) \n",maximum(abs.(df_err)))
    println("max(df2_err) \n",maximum(abs.(df2_err)))
    println("max(df2cheb_err) \n",maximum(abs.(df2cheb_err)))
    
    ### attempt at matrix inversion via LU decomposition
    Dt = 0.1
    Nu = 1.0
    lu_obj, AA = diffusion_matrix(Dxreverse2,x.n,Dt,Nu,return_A=true)
    #AA = Array{Float64,2}(undef,x.n,x.n)
    #for i in 1:x.n
    #    for j in 1:x.n
    #        AA[i,j] = - Dt*Nu*Dxreverse2[i,j]
    #    end
    #    AA[i,i] += 1.0
    #end
    #lu_obj = lu(AA)
    if x.n < 20
        println("L : \n",lu_obj.L)
        println("U : \n",lu_obj.U)
        println("p vector : \n",lu_obj.p)
    end
    LUtest = true
    AA_test_lhs = lu_obj.L*lu_obj.U 
    AA_test_rhs = AA[lu_obj.p,:]
    for i in 1:x.n
        for j in 1:x.n
            if abs.(AA_test_lhs[i,j]-AA_test_rhs[i,j]) > zero
                global LUtest = false
            end
        end
    end
    println("LU == AA : \n",LUtest)
    
    #bb = ones(x.n) try this for bc = "" rather than bc = "zero"
    bb = Array{Float64,1}(undef,x.n)
    yy = Array{Float64,1}(undef,x.n)
    #for i in 1:x.n
    #    bb[i] = f[i]#exp(-(4.0*x.grid[i]/x.L)^2)
    #end
    #yy = lu_obj \ bb # solution to AA yy = bb 
    #println("result", yy)
    #println("check result", AA*yy, bb)
    MMS_test = false 
    evolution_test = false#true 
    elliptic_solve_test = true
    elliptic_2Dsolve_test = false#true
    if MMS_test
        ntest = 5
        MMS_errors = Array{Float64,1}(undef,ntest)
        Dt_list = Array{Float64,1}(undef,ntest)
        fac_list = Array{Int64,1}(undef,ntest)
        fac_list .= [1, 10, 100, 1000, 10000]
        #for itest in [1, 10, 100, 1000, 10000]
        for itest in 1:ntest
            fac = fac_list[itest]
            #println(fac)
            ntime = 1000*fac
            nwrite = 100*fac
            dt = 0.001/fac
            #println(ntime," ",dt)
            nu = 1.0
            LU_obj = diffusion_matrix(Dxreverse2,x.n,dt,nu)
            
            time = Array{Float64,1}(undef,ntime)
            ff = Array{Float64,2}(undef,x.n,ntime)
            ss = Array{Float64,1}(undef,x.n) #source

            time[1] = 0.0
            ff[:,1] .= f[:] #initial condition
            for i in 1:ntime-1
                time[i+1] = (i+1)*dt
                bb .= ff[:,i]
                yy .= LU_obj\bb # implicit backward euler diffusion step
                @. ss = -nu*df2_exact # source term
                 # explicit forward_euler_step with source
                @views forward_euler_step!(ff[:,i+1],yy,ss,dt,x.n)
            end

            ff_error = Array{Float64,1}(undef,x.n)
            ff_error[:] .= abs.(ff[:,end] - ff[:,1])
            maxfferr = maximum(ff_error)
            #println("ff_error \n",ff_error)
            println("max(ff_error) \n",maxfferr)
            #println("t[end]: ",time[end])
            MMS_errors[itest] = maxfferr
            Dt_list[itest] = dt
        end 
        @views plot(Dt_list, [MMS_errors, 100.0*Dt_list], label=[L"max(\epsilon(f))" L"100\Delta t"], 
                     xlabel=L"\Delta t", ylabel="", xscale=:log10, yscale=:log10, shape =:circle)
        outfile = string("ff_err_vs_dt.pdf")
        savefig(outfile)
    end
    
    if evolution_test
        ntime = 100
        nwrite = 1
        dt = 0.001
        nu = 1.0
        LU_obj = diffusion_matrix(Dxreverse2,x.n,dt,nu)
        
        time = Array{Float64,1}(undef,ntime)
        ff = Array{Float64,2}(undef,x.n,ntime)
        ss = Array{Float64,1}(undef,x.n) #source

        time[1] = 0.0
        ff[:,1] .= f[:] #initial condition
        for i in 1:ntime-1
            time[i+1] = (i+1)*dt
            bb .= ff[:,i]
            yy .= LU_obj\bb # implicit backward euler diffusion step
            @. ss = 0.0 # source term
             # explicit forward_euler_step with source
            @views forward_euler_step!(ff[:,i+1],yy,ss,dt,x.n)
        end

        ffmin = minimum(ff)
        ffmax = maximum(ff)
        anim = @animate for i in 1:nwrite:ntime
                @views plot(x.grid, ff[:,i], xlabel="x", ylabel="f", ylims = (ffmin,ffmax))
            end
        outfile = string("ff_vs_x.gif")
        gif(anim, outfile, fps=5)
    end
    
    if elliptic_solve_test
        println("elliptic solve test")
        ngrid = 5
        nelement_local = 50
        L = 25
        nelement_global = nelement_local
        input = grid_input("mu", ngrid, nelement_global, nelement_local, 
		nrank, irank, L, discretization, fd_option, "zero_upper", adv_input,comm)
        y = define_coordinate(input)
        Dy = Array{Float64,2}(undef, y.n, y.n)
        cheb_derivative_matrix_reversed!(Dy,y)
        
        
        yDy = Array{Float64,2}(undef, y.n, y.n)
        for iy in 1:y.n
            @. yDy[iy,:] = y.grid[iy]*Dy[iy,:]
        end
        
        
        Dy_yDy = Array{Float64,2}(undef, y.n, y.n)
        mul!(Dy_yDy,Dy,yDy)
        #Dy_yDy[1,1] = 2.0*Dy_yDy[1,1]
        #Dy_yDy[end,end] = 2.0*Dy_yDy[end,end]
        
        D2y = Array{Float64,2}(undef, y.n, y.n)
        mul!(D2y,Dy,Dy)
        #Dy_yDy[1,1] = 2.0*Dy_yDy[1,1]
        D2y[end,end] = 2.0*D2y[end,end]
        yD2y = Array{Float64,2}(undef, y.n, y.n)
        for iy in 1:y.n
            @. yD2y[iy,:] = y.grid[iy]*D2y[iy,:]
        end
        
        if y.n < 20
            print_matrix(Dy,"Dy",y.n,y.n)
            print_matrix(yDy,"yDy",y.n,y.n)
            print_matrix(Dy_yDy,"Dy_yDy",y.n,y.n)
            print_matrix(yD2y+Dy,"yD2y+Dy",y.n,y.n)
        end 
        Sy = Array{Float64,1}(undef, y.n)
        Fy = Array{Float64,1}(undef, y.n)
        Fy_exact = Array{Float64,1}(undef, y.n)
        Fy_err = Array{Float64,1}(undef, y.n)
        for iy in 1:y.n
            Sy[iy] = (1.0 - y.grid[iy])*exp(-y.grid[iy])
            Fy_exact[iy] = -exp(-y.grid[iy])
        end
        LL = Array{Float64,2}(undef, y.n, y.n)
        @. LL = yD2y + Dy
        println("condition number: ", cond(LL))
        LL_lu_obj = lu(sparse(LL))
        # do elliptic solve 
        Fy = LL_lu_obj\Sy
        @. Fy_err = abs(Fy - Fy_exact)
        
        println("maximum(Fy_err)",maximum(Fy_err))
        #println("Fy_err",Fy_err)
        #println("Fy_exact",Fy_exact)
        #println("Fy",Fy)
        plot([y.grid,y.grid,y.grid], [Fy,Fy_exact,Fy_err], xlabel="y", ylabel="", label=["F" "F_exact" "F_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "1D_elliptic_solve_test.pdf"
        savefig(outfile)
        plot([y.grid], [Fy_err], xlabel="x", ylabel="", label=["F_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "1D_elliptic_solve_test_err.pdf"
        savefig(outfile)

    end
    
    if elliptic_2Dsolve_test
        println("elliptic 2D solve test")
        ngrid = 2
        nelement_local = 5 
        x_L = 1
        y_L = 1
        nelement_global = nelement_local
        
        input = grid_input("vpa", ngrid, nelement_global, nelement_local, 
		nrank, irank, x_L, discretization, fd_option, "zero", adv_input, comm)
        x = define_coordinate(input)
        
        Dx = Array{Float64,2}(undef, x.n, x.n)
        cheb_derivative_matrix_reversed!(Dx,x)
        D2x = Array{Float64,2}(undef, x.n, x.n)
        mul!(D2x,Dx,Dx)
        D2x[1,1] = 2.0*D2x[1,1]
        D2x[end,end] = 2.0*D2x[end,end]
        if x.n < 20
            print_matrix(Dx,"Dx",x.n,x.n)
            print_matrix(D2x,"D2x",x.n,x.n)
        end 
        
        input = grid_input("mu", ngrid, nelement_global, nelement_local, 
		nrank, irank, y_L, discretization, fd_option, "zero_upper", adv_input, comm)
        y = define_coordinate(input)
        
        Dy = Array{Float64,2}(undef, y.n, y.n)
        cheb_derivative_matrix_reversed!(Dy,y)
        
        
        yDy = Array{Float64,2}(undef, y.n, y.n)
        for iy in 1:y.n
            @. yDy[iy,:] = y.grid[iy]*Dy[iy,:]
        end
        
        
        Dy_yDy = Array{Float64,2}(undef, y.n, y.n)
        mul!(Dy_yDy,Dy,yDy)
        #Dy_yDy[1,1] = 2.0*Dy_yDy[1,1]
        #Dy_yDy[end,end] = 2.0*Dy_yDy[end,end]
        
        D2y = Array{Float64,2}(undef, y.n, y.n)
        mul!(D2y,Dy,Dy)
        #Dy_yDy[1,1] = 2.0*Dy_yDy[1,1]
        D2y[end,end] = 2.0*D2y[end,end]
        yD2y = Array{Float64,2}(undef, y.n, y.n)
        for iy in 1:y.n
            @. yD2y[iy,:] = y.grid[iy]*D2y[iy,:]
        end
        
        if y.n < 20
            print_matrix(Dy,"Dy",y.n,y.n)
            print_matrix(yDy,"yDy",y.n,y.n)
            print_matrix(Dy_yDy,"Dy_yDy",y.n,y.n)
            print_matrix(yD2y+Dy,"yD2y+Dy",y.n,y.n)
        end 
        
        ### now form 2D matrix to invert and corresponding sources 
        function dH_Maxwellian_dvpa(Bmag,vpa,mu,ivpa,imu)
            # speed variable
            eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
            zero = 1.0e-10
            if eta < zero
                dHdvpa = -(4.0*vpa.grid[ivpa])/(3.0*sqrt(pi))
            else 
                dHdvpa = (2.0/sqrt(pi))*vpa.grid[ivpa]*((exp(-eta^2)/eta)  - (erf(eta)/(eta^2)))
            end
            return dHdvpa
        end
        # Array in 2D form 
        nx = x.n   
        ny = y.n 
        Sxy = Array{Float64,2}(undef, nx, ny)
        Fxy = Array{Float64,2}(undef, nx, ny)
        Fxy_exact = Array{Float64,2}(undef, nx, ny)
        Fxy_err = Array{Float64,2}(undef, nx, ny)
        LLxy = Array{Float64,4}(undef, nx, ny, nx, ny)
        # Array in compound 1D form 
        # ic = (ix-1) + nx*(iy-1) + 1
        # iy = mod(ic,nx) + 1
        # ix = rem(ic,nx)
        function icfunc(ix,iy,nx)
            return ix + nx*(iy-1)
        end
        function iyfunc(ic,nx)
            #return mod(ic,nx) + 1
            return floor(Int64,(ic-1)/nx) + 1
        end
        function ixfunc(ic,nx)
            ix = ic - nx*(iyfunc(ic,nx) - 1)
            #return rem(ic,nx)
            return ix
        end
        nc = nx*ny
        Fc = Array{Float64,1}(undef, nc)
        Sc = Array{Float64,1}(undef, nc)
        LLc = Array{Float64,2}(undef, nc, nc)
        
        for iy in 1:ny
            for ix in 1:nx
                Sxy[ix,iy] = -4.0*pi*(-2.0*x.grid[ix]*exp(-2.0*y.grid[iy]-x.grid[ix]^2))
                Fxy_exact[ix,iy] = dH_Maxwellian_dvpa(1.0,x,y,ix,iy)
                for iyp in 1:ny
                    for ixp in 1:nx
                        LLxy[ixp,iyp,ix,iy] = D2x[ixp,ix] + yD2y[iyp,iy] + Dy[iyp,iy]
                    end
                end
            end
        end
        for ic in 1:nc
            ix = ixfunc(ic,nx)
            iy = iyfunc(ic,nx)
            Sc[ic] = Sxy[ix,iy]
            for icp in 1:nc
                ixp = ixfunc(icp,nx)
                iyp = iyfunc(icp,nx)
                #println("ic: ",ic," ix: ", ix," iy: ",iy," icp: ",icp," ixp: ", ixp," iyp: ",iyp)
                LLc[icp,ic] = LLxy[ixp,iyp,ix,iy]
            end
        end
        println("condition number(LLc): ", cond(LLc))
        LLc_lu_obj = lu(sparse(LLc))
        # do elliptic solve 
        Fc = LLc_lu_obj\Sc
        #reshape to 2D vector 
        for ic in 1:nc
            ix = ixfunc(ic,nx)
            iy = iyfunc(ic,nx)
            Fxy[ix,iy] = Fc[ic]
        end
        
        @. Fxy_err = abs(Fxy - Fxy_exact)
        
        println("maximum(Fxy_err)",maximum(Fxy_err))
        println("Fxy_err",Fxy_err[1,:])
        println("Fxy_exact",Fxy_exact[1,:])
        println("Fxy",Fxy[1,:])
        

    end
end
