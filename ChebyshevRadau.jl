using Printf
using Plots
using LaTeXStrings
using MPI
using FFTW

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
    import moment_kinetics
    using moment_kinetics.type_definitions: mk_float, mk_int
    using moment_kinetics.array_allocation: allocate_float, allocate_complex
    using moment_kinetics.chebyshev: chebyshev_info, chebyshev_spectral_derivative! #chebyshev_derivative_single_element!, 
    using LinearAlgebra: mul!
    
    struct radau_coordinate
        # n is the total number of local grid points associated with this coordinate
        n::mk_int
        # ngrid is the number of grid points per element in this coordinate
        ngrid::mk_int
        # nelement is the number of elements associated with this coordinate globally
        nelement_global::mk_int
        # nelement_local is the number of elements associated with this coordinate on this rank
        nelement_local::mk_int
        # L is the box length in this coordinate
        L::mk_float
        # grid is the location of the grid points
        grid::Array{mk_float,1}
        # wgts contains the integration weights associated with each grid point
        wgts::Array{mk_float,1}
    end
    
    function setup_chebyshev_radau_pseudospectral(coord)
        # ngrid_fft is the number of grid points in the extended domain
        # in z = cos(theta).  this is necessary to turn a cosine transform on [0,π]
        # into a complex transform on [0,2π], which is more efficient in FFTW
        ngrid_fft = 2*coord.ngrid - 1
        # create array for f on extended [0,2π] domain in theta = ArcCos[z]
        fext = allocate_complex(ngrid_fft)
        # create arrays for storing Chebyshev spectral coefficients of f and f'
        fcheby = allocate_float(coord.ngrid, coord.nelement_local)
        dcheby = allocate_float(coord.ngrid)
        # setup the plans for the forward and backward Fourier transforms
        forward_transform = plan_fft!(fext, flags=FFTW.MEASURE)
        backward_transform = plan_ifft!(fext, flags=FFTW.MEASURE)
        # return a structure containing the information needed to carry out
        # a 1D Chebyshev transform
        return chebyshev_info(fext, fcheby, dcheby, forward_transform, backward_transform)
    end
    
    function chebyshev_radau_forward_transform!(chebyf, fext, ff, transform, n)
        @inbounds begin
            for j ∈ 1:n
                fext[j] = complex(ff[n-j+1],0.0)
            end
            for j ∈ 1:n-1
                fext[n+j] = fext[n-j+1]
            end
        end
        #println("ff",ff)
        #println("fext",fext)
        # perform the forward, complex-to-complex FFT in-place (cheby.fext is overwritten)
        transform*fext
        #println("fext",fext)
        # use reality + evenness of f to eliminate unncessary information
        # and obtain Chebyshev spectral coefficients for this element
        # also sort out normalisation
        @inbounds begin
            nfft = 2*n - 1
            for j ∈ 2:n
                chebyf[j] = 2.0*real(fext[j])/nfft
            end
            chebyf[1] = real(fext[1])/nfft
        end
        return nothing
    end

    """
    """
    function chebyshev_radau_backward_transform!(ff, fext, chebyf, transform, n)
        # chebyf as input contains Chebyshev spectral coefficients
        # need to use reality condition to extend onto negative frequency domain
        @inbounds begin
            # first, fill in values for fext corresponding to positive frequencies
            for j ∈ 2:n
                fext[j] = chebyf[j]*0.5
            end
            # next, fill in values for fext corresponding to negative frequencies
            # using fext(-k) = conjg(fext(k)) = fext(k)
            # usual FFT ordering with j=1 <-> k=0, followed by ascending k up to kmax
            # and then descending from -kmax down to -dk
            for j ∈ 1:n-1
                fext[n+j] = fext[n-j+1]
            end
            # fill in zero frequency mode, which is special in that it does not require
            # the 1/2 scale factor
            fext[1] = chebyf[1]
        end
        #println("chebyf",chebyf)
        #println("fext",fext)
        # perform the backward, complex-to-complex FFT in-place (fext is overwritten)
        transform*fext
        #println("fext",fext)
        
        @inbounds begin
            for j ∈ 1:n
                ff[j] = real(fext[n-j+1])
            end
        end
        return nothing
    end
    function chebyshev_radau_derivative_single_element!(df, ff, cheby_f, cheby_df, cheby_fext, forward, coord)
        # calculate the Chebyshev coefficients of the real-space function ff and return
        # as cheby_f
        chebyshev_radau_forward_transform!(cheby_f, cheby_fext, ff, forward, coord.ngrid)
        # calculate the Chebyshev coefficients of the derivative of ff with respect to coord.grid
        chebyshev_spectral_derivative!(cheby_df, cheby_f)
        # inverse Chebyshev transform to get df/dcoord
        chebyshev_radau_backward_transform!(df, cheby_fext, cheby_df, forward, coord.ngrid)
    end
    
    function calculate_chebyshev_radau_D_matrix_via_FFT!(D::Array{mk_float,2}, coord, spectral)
        ff_buffer = Array{mk_float,1}(undef,coord.ngrid)
        df_buffer = Array{mk_float,1}(undef,coord.ngrid)
        # use response matrix approach to calculate derivative matrix D 
        for j in 1:coord.ngrid 
            ff_buffer .= 0.0 
            ff_buffer[j] = 1.0
            @views chebyshev_radau_derivative_single_element!(df_buffer[:], ff_buffer[:],
                spectral.f[:,1], spectral.df, spectral.fext, spectral.forward, coord)
            @. D[:,j] = df_buffer[:] # assign appropriate column of derivative matrix 
        end
        # correct diagonal elements to gurantee numerical stability
        # gives D*[1.0, 1.0, ... 1.0] = [0.0, 0.0, ... 0.0]
        for j in 1:coord.ngrid
            D[j,j] = 0.0
            D[j,j] = -sum(D[j,:])
        end
    end
    
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
    
    ngrid = 20
    nelement_local = 1
    nelement_global = nelement_local
    npoints = (ngrid - 1)*nelement_local + 1
    radau_points = Array{mk_float,1}(undef,ngrid)
    for j in 1:ngrid 
        radau_points[j] = cospi( (ngrid - j)/(ngrid-0.5))
    end
    println("radau_points",radau_points)
    L = 1.0 
    radau_wgts = Array{mk_float,1}(undef,ngrid)
    coord = radau_coordinate(npoints,ngrid,nelement_global,nelement_local,L,radau_points,radau_wgts)
    spectral = setup_chebyshev_radau_pseudospectral(coord)
    
    ff = Array{mk_float,1}(undef,ngrid)
    df_err = Array{mk_float,1}(undef,ngrid)
    df_exact = Array{mk_float,1}(undef,ngrid)
    df = Array{mk_float,1}(undef,ngrid)
    for j in 1:coord.n
        arg = 2.0*pi*coord.grid[j]/coord.L
        ff[j] = sin(arg)
        df_exact[j] = (2.0*pi/coord.L)*cos(arg)
    end
    
    @views chebyshev_radau_derivative_single_element!(df[:], ff[:],
            spectral.f[:,1], spectral.df, spectral.fext, spectral.forward, coord)
    
    
    @. df_err = df - df_exact
    println("FFT test")
    println("df_err \n",df_err)
    println("df_exact \n",df_exact)
    println("df \n",df)
    
    Dcoord = Array{mk_float,2}(undef,coord.ngrid,coord.ngrid)
    calculate_chebyshev_radau_D_matrix_via_FFT!(Dcoord,coord,spectral)
    print_matrix(Dcoord,"Dcoord",coord.n,coord.n)
    mul!(df,Dcoord,ff)
    @. df_err = df - df_exact
    println("Matrix test sine wave")
    println("df_err \n",df_err)
    println("df_exact \n",df_exact)
    println("df \n",df)
    
    ff .= 1.0
    df_exact .= 0.0
    mul!(df,Dcoord,ff)
    @. df_err = df - df_exact
    println("Matrix test constant")
    println("df_err \n",df_err)
    println("df_exact \n",df_exact)
    println("df \n",df)
end