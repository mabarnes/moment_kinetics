using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
    using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
    using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
    using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral
    using moment_kinetics.type_definitions: mk_float, mk_int
    
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
    
    function print_vector(vector,name,m)
        println("\n ",name," \n")
        for j in 1:m
            @printf("%.3f ", vector[j])
        end
        println("")
        println("\n")
    end 


    # Array in compound 1D form 
    
    function ic_func(ivpa,ivperp,nvpa)
        return ivpa + nvpa*(ivperp-1)
    end
    function ivperp_func(ic,nvpa)
        return floor(Int64,(ic-1)/nvpa) + 1
    end
    function ivpa_func(ic,nvpa)
        ivpa = ic - nvpa*(ivperp_func(ic,nvpa) - 1)
        return ivpa
    end

    function ravel_vpavperp_to_c!(pdf_c,pdf_vpavperp,nvpa,nvperp)
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                ic = ic_func(ivpa,ivperp,nvpa)
                pdf_c[ic] = pdf_vpavperp[ivpa,ivperp]
            end
        end
    return nothing
    end
    
    function ravel_c_to_vpavperp!(pdf_vpavperp,pdf_c,nc,nvpa)
        for ic in 1:nc
            ivpa = ivpa_func(ic,nvpa)
            ivperp = ivperp_func(ic,nvpa)
            pdf_vpavperp[ivpa,ivperp] = pdf_c[ic]
        end
    return nothing
    end
    
    function ivpa_global_func(ivpa_local,ielement_vpa,ngrid_vpa)
        ivpa_global = ivpa_local + (ielement_vpa - 1)*(ngrid_vpa - 1)
        return ivpa_global
    end
    
    # define inputs needed for the test
	ngrid = 2 #number of points per element 
	nelement_local_vpa = 4 # number of elements per rank
	nelement_global_vpa = nelement_local_vpa # total number of elements 
	nelement_local_vperp = 4 # number of elements per rank
	nelement_global_vperp = nelement_local_vperp # total number of elements 
	Lvpa = 6.0 #physical box size in reference units 
	Lvperp = 3.0 #physical box size in reference units 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	#discretization = "chebyshev_pseudospectral"
	discretization = "gausslegendre_pseudospectral"
    fd_option = "fourth_order_centered"
    cheb_option = "matrix"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = MPI.COMM_NULL
	# create the 'input' struct containing input info needed to create a
	# coordinate
    vpa_input = grid_input("vpa", ngrid, nelement_global_vpa, nelement_local_vpa, 
		nrank, irank, Lvpa, discretization, fd_option, cheb_option, bc, adv_input,comm)
	vperp_input = grid_input("vperp", ngrid, nelement_global_vperp, nelement_local_vperp, 
		nrank, irank, Lvperp, discretization, fd_option, cheb_option, bc, adv_input,comm)
	# create the coordinate struct 'x'
	println("made inputs")
	println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
	println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
	vpa = define_coordinate(vpa_input)
	vperp = define_coordinate(vperp_input)
    if vpa.discretization == "chebyshev_pseudospectral" && vpa.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        vpa_spectral = setup_chebyshev_pseudospectral(vpa)
        # obtain the local derivatives of the uniform vpa-grid with respect to the used vpa-grid
        #chebyshev_derivative!(vpa.duniform_dgrid, vpa.uniform_grid, vpa_spectral, vpa)
    elseif vpa.discretization == "gausslegendre_pseudospectral" && vpa.n > 1
        vpa_spectral = setup_gausslegendre_pseudospectral(vpa)
    else
        # create dummy Bool variable to return in place of the above struct
        vpa_spectral = false
        #vpa.duniform_dgrid .= 1.0
    end

    if vperp.discretization == "chebyshev_pseudospectral" && vperp.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vperp
        # and create the plans for the forward and backward fast Chebyshev transforms
        vperp_spectral = setup_chebyshev_pseudospectral(vperp)
        # obtain the local derivatives of the uniform vperp-grid with respect to the used vperp-grid
        #chebyshev_derivative!(vperp.duniform_dgrid, vperp.uniform_grid, vperp_spectral, vperp)
    elseif vperp.discretization == "gausslegendre_pseudospectral" && vperp.n > 1
        vperp_spectral = setup_gausslegendre_pseudospectral(vperp)
    else
        # create dummy Bool variable to return in place of the above struct
        vperp_spectral = false
        #vperp.duniform_dgrid .= 1.0
    end
    
    # Assemble a 2D mass matrix in the global compound coordinate
    nc_global = vpa.n*vperp.n
    nc_local = vpa.ngrid*vperp.ngrid
    MM2D = Array{mk_float,2}(undef,nc_global,nc_global)
    MM2D .= 0.0
    print_matrix(MM2D,"MM2D",nc_global,nc_global)
    
    function get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ic_local,icp_local)
        # indices of vpa, vperp within the element
        ivpa_local = ivpa_func(ic_local,vpa.ngrid)
        ivperp_local = ivperp_func(ic_local,vpa.ngrid)
        ivpap_local = ivpa_func(icp_local,vpa.ngrid)
        ivperpp_local = ivperp_func(icp_local,vpa.ngrid)
        # global indices on the grids
        ivpa_global = vpa.igrid_full[ivpa_local,ielement_vpa]
        ivperp_global = vperp.igrid_full[ivperp_local,ielement_vperp]
        ivpap_global = vpa.igrid_full[ivpap_local,ielement_vpa]
        ivperpp_global = vperp.igrid_full[ivperpp_local,ielement_vperp]
        # global compound indices
        ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
        icp_global = ic_func(ivpap_global,ivperpp_global,vpa.n)
        return ivpa_local, ivperp_local, ivpap_local, ivperpp_local, ic_global, icp_global
    end
    
    for ielement_vperp in 1:1
        for ielement_vpa in 1:vpa.nelement_local
            for icp_local in 1:nc_local
                for ic_local in 1:nc_local
                    ivpa_local, ivperp_local, ivpap_local, ivperpp_local, ic_global, icp_global = get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ic_local,icp_local)
                    # assign mass matrix data
                    MM2D[ic_global,icp_global] += vpa_spectral.lobatto.M0[ivpa_local,ivpap_local]*
                                                    vperp_spectral.radau.M1[ivperp_local,ivperpp_local]
                end
            end
        end
    end
    for ielement_vperp in 2:vperp.nelement_local
        for ielement_vpa in 1:vpa.nelement_local
            for icp_local in 1:nc_local
                for ic_local in 1:nc_local
                    ivpa_local, ivperp_local, ivpap_local, ivperpp_local, ic_global, icp_global = get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ic_local,icp_local)
                    # assign mass matrix data
                    MM2D[ic_global,icp_global] += vpa_spectral.lobatto.M0[ivpa_local,ivpap_local]*
                                                    vperp_spectral.lobatto.M1[ivperp_local,ivperpp_local]
                end
            end
        end
    end
    print_matrix(MM2D,"MM2D",nc_global,nc_global)
end