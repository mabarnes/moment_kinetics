using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using Dates

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
    using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
    using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
    using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral, get_QQ_local!
    using moment_kinetics.type_definitions: mk_float, mk_int
    using moment_kinetics.fokker_planck: F_Maxwellian, H_Maxwellian, G_Maxwellian
    using moment_kinetics.fokker_planck: d2Gdvpa2, d2Gdvperp2, dGdvperp, d2Gdvperpdvpa, dHdvpa, dHdvperp
    using SparseArrays: sparse
    using LinearAlgebra: mul!, lu, cholesky
    
    function print_matrix(matrix,name,n,m)
        println("\n ",name," \n")
        for i in 1:n
            for j in 1:m
                @printf("%.2f ", matrix[i,j])
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

    function plot_test_data(func_exact,func_num,func_err,func_name,vpa,vperp)
        @views heatmap(vperp.grid, vpa.grid, func_num[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                    outfile = string(func_name*"_num.pdf")
                    savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, func_exact[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                    outfile = string(func_name*"_exact.pdf")
                    savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, func_err[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                    outfile = string(func_name*"_err.pdf")
                    savefig(outfile)
        return nothing
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
    
    struct vpa_vperp_boundary_data
        lower_boundary_vpa::Array{mk_float,1}
        upper_boundary_vpa::Array{mk_float,1}
        upper_boundary_vperp::Array{mk_float,1}
    end
    
    struct rosenbluth_potential_boundary_data
        H_data::vpa_vperp_boundary_data
        dHdvpa_data::vpa_vperp_boundary_data
        dHdvperp_data::vpa_vperp_boundary_data
        G_data::vpa_vperp_boundary_data
        dGdvperp_data::vpa_vperp_boundary_data
        d2Gdvperp2_data::vpa_vperp_boundary_data
        d2Gdvperpdvpa_data::vpa_vperp_boundary_data
        d2Gdvpa2_data::vpa_vperp_boundary_data
    end
    
    function allocate_boundary_data(vpa,vperp)
        lower_boundary_vpa = Array{mk_float,1}(undef,vperp.n)
        upper_boundary_vpa = Array{mk_float,1}(undef,vperp.n)
        upper_boundary_vperp = Array{mk_float,1}(undef,vpa.n)
        return vpa_vperp_boundary_data(lower_boundary_vpa,
                upper_boundary_vpa,upper_boundary_vperp)
    end
    
    function assign_exact_boundary_data!(func_data::vpa_vperp_boundary_data,
                                            func_exact,vpa,vperp)
        nvpa = vpa.n
        nvperp = vperp.n
        for ivperp in 1:nvperp
            func_data.lower_boundary_vpa[ivperp] = func_exact[1,ivperp]
            func_data.upper_boundary_vpa[ivperp] = func_exact[nvpa,ivperp]
        end
        for ivpa in 1:nvpa
            func_data.upper_boundary_vperp[ivpa] = func_exact[ivpa,nvperp]
        end
        return nothing
    end
        
    function allocate_rosenbluth_potential_boundary_data(vpa,vperp)
        H_data = allocate_boundary_data(vpa,vperp)
        dHdvpa_data = allocate_boundary_data(vpa,vperp)
        dHdvperp_data = allocate_boundary_data(vpa,vperp)
        G_data = allocate_boundary_data(vpa,vperp)
        dGdvperp_data = allocate_boundary_data(vpa,vperp)
        d2Gdvperp2_data = allocate_boundary_data(vpa,vperp)
        d2Gdvperpdvpa_data = allocate_boundary_data(vpa,vperp)
        d2Gdvpa2_data = allocate_boundary_data(vpa,vperp)
        return rosenbluth_potential_boundary_data(H_data,dHdvpa_data,
            dHdvperp_data,G_data,dGdvperp_data,d2Gdvperp2_data,
            d2Gdvperpdvpa_data,d2Gdvpa2_data)
    end
    
    function calculate_rosenbluth_potential_boundary_data_exact!(rpbd::rosenbluth_potential_boundary_data,
      H_exact,dHdvpa_exact,dHdvperp_exact,G_exact,dGdvperp_exact,
      d2Gdvperp2_exact,d2Gdvperpdvpa_exact,d2Gdvpa2_exact,
      vpa,vperp)
        assign_exact_boundary_data!(rpbd.H_data,H_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.dHdvpa_data,dHdvpa_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.dHdvperp_data,dHdvperp_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.G_data,G_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.dGdvperp_data,dGdvperp_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.d2Gdvperp2_data,d2Gdvperp2_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.d2Gdvperpdvpa_data,d2Gdvperpdvpa_exact,vpa,vperp)
        assign_exact_boundary_data!(rpbd.d2Gdvpa2_data,d2Gdvpa2_exact,vpa,vperp)
        return nothing
    end
    
    # define inputs needed for the test
	plot_test_output = false#true
    ngrid = 9 #number of points per element 
	nelement_local_vpa = 4 # number of elements per rank
	nelement_global_vpa = nelement_local_vpa # total number of elements 
	nelement_local_vperp = 2 # number of elements per rank
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
    Index2D = Array{mk_int,2}(undef,nc_global,nc_global)
    MM2D = Array{mk_float,2}(undef,nc_global,nc_global)
    MM2D .= 0.0
    KKpar2D = Array{mk_float,2}(undef,nc_global,nc_global)
    KKpar2D .= 0.0
    KKperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    KKperp2D .= 0.0
    PUperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    PUperp2D .= 0.0
    PPparPUperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    PPparPUperp2D .= 0.0
    PPpar2D = Array{mk_float,2}(undef,nc_global,nc_global)
    PPpar2D .= 0.0
    MMparMNperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    MMparMNperp2D .= 0.0
    # Laplacian matrix
    LP2D = Array{mk_float,2}(undef,nc_global,nc_global)
    LP2D .= 0.0
    # Modified Laplacian matrix
    LV2D = Array{mk_float,2}(undef,nc_global,nc_global)
    LV2D .= 0.0
    
    #print_matrix(MM2D,"MM2D",nc_global,nc_global)
    # local dummy arrays
    MMpar = Array{mk_float,2}(undef,vpa.ngrid,vpa.ngrid)
    MMperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    MNperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    MRperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    MMperp_p1 = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    KKpar = Array{mk_float,2}(undef,vpa.ngrid,vpa.ngrid)
    KKperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    KJperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    LLperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    PPperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    PUperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    PPpar = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    
    function get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
        # global indices on the grids
        ivpa_global = vpa.igrid_full[ivpa_local,ielement_vpa]
        ivperp_global = vperp.igrid_full[ivperp_local,ielement_vperp]
        # global compound index
        ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
        return ic_global, ivpa_global, ivperp_global
    end
    
    impose_BC_at_zero_vperp = false
    println("begin elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    for ielement_vperp in 1:vperp.nelement_local
        get_QQ_local!(MMperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"M")
        get_QQ_local!(MRperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"R")
        get_QQ_local!(MNperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"N")
        get_QQ_local!(KKperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"K")
        get_QQ_local!(KJperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"J")
        get_QQ_local!(LLperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"L")
        get_QQ_local!(PPperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"P")
        get_QQ_local!(PUperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"U")
        #print_matrix(MMperp,"MMperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(MRperp,"MRperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(MNperp,"MNperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(KKperp,"KKperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(KJperp,"KJperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(LLperp,"LLperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(PPperp,"PPperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(PUperp,"PUperp",vperp.ngrid,vperp.ngrid)
        
        for ielement_vpa in 1:vpa.nelement_local
            get_QQ_local!(MMpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"M")
            get_QQ_local!(KKpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"K")
            get_QQ_local!(PPpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"P")
            #print_matrix(MMpar,"MMpar",vpa.ngrid,vpa.ngrid)
            #print_matrix(KKpar,"KKpar",vpa.ngrid,vpa.ngrid)
            #print_matrix(PPpar,"PPpar",vpa.ngrid,vpa.ngrid)
            
            for ivperpp_local in 1:vperp.ngrid
                for ivperp_local in 1:vperp.ngrid
                    for ivpap_local in 1:vpa.ngrid
                        for ivpa_local in 1:vpa.ngrid
                            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                            icp_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpap_local,ivperpp_local) #get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivpap_local,ivperp_local,ivperpp_local)
                            #println("ielement_vpa: ",ielement_vpa," ielement_vperp: ",ielement_vperp)
                            #println("ivpa_local: ",ivpa_local," ivpap_local: ",ivpap_local)
                            #println("ivperp_local: ",ivperp_local," ivperpp_local: ",ivperpp_local)
                            #println("ic: ",ic_global," icp: ",icp_global)
                            # boundary condition possibilities
                            lower_boundary_row_vpa = (ielement_vpa == 1 && ivpa_local == 1)
                            upper_boundary_row_vpa = (ielement_vpa == vpa.nelement_local && ivpa_local == vpa.ngrid)
                            lower_boundary_row_vperp = (ielement_vperp == 1 && ivperp_local == 1)
                            upper_boundary_row_vperp = (ielement_vperp == vperp.nelement_local && ivperp_local == vperp.ngrid)
                            

                            if lower_boundary_row_vpa
                                if ivpap_local == 1 && ivperp_local == ivperpp_local
                                    MM2D[ic_global,icp_global] = 1.0
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                else 
                                    MM2D[ic_global,icp_global] = 0.0
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_vpa
                                if ivpap_local == vpa.ngrid && ivperp_local == ivperpp_local 
                                    MM2D[ic_global,icp_global] = 1.0
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                else 
                                    MM2D[ic_global,icp_global] = 0.0
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                end
                            elseif lower_boundary_row_vperp && impose_BC_at_zero_vperp
                                if ivperpp_local == 1 && ivpa_local == ivpap_local
                                    MM2D[ic_global,icp_global] = 1.0
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                else 
                                    MM2D[ic_global,icp_global] = 0.0
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_vperp
                                if ivperpp_local == vperp.ngrid && ivpa_local == ivpap_local
                                    MM2D[ic_global,icp_global] = 1.0
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                else 
                                    MM2D[ic_global,icp_global] = 0.0
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                end
                            else
                                # assign mass matrix data
                                #println("MM2D += ", MMpar[ivpa_local,ivpap_local]*MMperp[ivperp_local,ivperpp_local])
                                MM2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                                MMperp[ivperp_local,ivperpp_local]
                                LP2D[ic_global,icp_global] += (KKpar[ivpa_local,ivpap_local]*
                                                                MMperp[ivperp_local,ivperpp_local] +
                                                               MMpar[ivpa_local,ivpap_local]*
                                                                LLperp[ivperp_local,ivperpp_local])
                                LV2D[ic_global,icp_global] += (KKpar[ivpa_local,ivpap_local]*
                                                                MRperp[ivperp_local,ivperpp_local] +
                                                               MMpar[ivpa_local,ivpap_local]*
                                                                (KJperp[ivperp_local,ivperpp_local] -
                                                                 PPperp[ivperp_local,ivperpp_local] - 
                                                                 MNperp[ivperp_local,ivperpp_local]))
                            end
                            
                            # assign K matrices
                            KKpar2D[ic_global,icp_global] += KKpar[ivpa_local,ivpap_local]*
                                                            MMperp[ivperp_local,ivperpp_local]
                            KKperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            KKperp[ivperp_local,ivperpp_local]
                            # assign PU matrix
                            PUperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            PUperp[ivperp_local,ivperpp_local]
                            PPparPUperp2D[ic_global,icp_global] += PPpar[ivpa_local,ivpap_local]*
                                                            PUperp[ivperp_local,ivperpp_local]
                            PPpar2D[ic_global,icp_global] += PPpar[ivpa_local,ivpap_local]*
                                                            MMperp[ivperp_local,ivperpp_local]
                            # assign RHS mass matrix for d2Gdvperp2
                            MMparMNperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            MNperp[ivperp_local,ivperpp_local]
                        end
                    end
                end
            end
        end
    end
    println("finished elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    function enforce_zero_bc!(fc,vpa,vperp)
        # lower vpa boundary
        ielement_vpa = 1
        ivpa_local = 1
        for ielement_vperp in 1:vperp.nelement_local
            for ivperp_local in 1:vperp.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = 0.0
            end
        end
        
        # upper vpa boundary
        ielement_vpa = vpa.nelement_local
        ivpa_local = vpa.ngrid
        for ielement_vperp in 1:vperp.nelement_local
            for ivperp_local in 1:vperp.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = 0.0
            end
        end
        
        # upper vperp boundary
        ielement_vperp = vperp.nelement_local
        ivperp_local = vperp.ngrid
        for ielement_vpa in 1:vpa.nelement_local
            for ivpa_local in 1:vpa.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = 0.0
            end
        end
    end
    
    function enforce_dirichlet_bc!(fc,vpa,vperp,f_bc;dirichlet_vperp_BC=false)
        # lower vpa boundary
        ielement_vpa = 1
        ivpa_local = 1
        for ielement_vperp in 1:vperp.nelement_local
            for ivperp_local in 1:vperp.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc[ivpa_global,ivperp_global]
            end
        end
        
        # upper vpa boundary
        ielement_vpa = vpa.nelement_local
        ivpa_local = vpa.ngrid
        for ielement_vperp in 1:vperp.nelement_local
            for ivperp_local in 1:vperp.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc[ivpa_global,ivperp_global]
            end
        end
        
        if dirichlet_vperp_BC
            # upper vperp boundary
            ielement_vperp = 1
            ivperp_local = 1
            for ielement_vpa in 1:vpa.nelement_local
                for ivpa_local in 1:vpa.ngrid
                    ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                    fc[ic_global] = f_bc[ivpa_global,ivperp_global]
                end
            end
        end
        
        # upper vperp boundary
        ielement_vperp = vperp.nelement_local
        ivperp_local = vperp.ngrid
        for ielement_vpa in 1:vpa.nelement_local
            for ivpa_local in 1:vpa.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc[ivpa_global,ivperp_global]
            end
        end
    end
    
    function enforce_dirichlet_bc!(fc,vpa,vperp,f_bc::vpa_vperp_boundary_data)
        # lower vpa boundary
        ielement_vpa = 1
        ivpa_local = 1
        for ielement_vperp in 1:vperp.nelement_local
            for ivperp_local in 1:vperp.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc.lower_boundary_vpa[ivperp_global]
            end
        end
        
        # upper vpa boundary
        ielement_vpa = vpa.nelement_local
        ivpa_local = vpa.ngrid
        for ielement_vperp in 1:vperp.nelement_local
            for ivperp_local in 1:vperp.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc.upper_boundary_vpa[ivperp_global]
            end
        end
                
        # upper vperp boundary
        ielement_vperp = vperp.nelement_local
        ivperp_local = vperp.ngrid
        for ielement_vpa in 1:vpa.nelement_local
            for ivpa_local in 1:vpa.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc.upper_boundary_vperp[ivpa_global]
            end
        end
        return nothing
    end
    
    if nc_global < 30
        print_matrix(MM2D,"MM2D",nc_global,nc_global)
        print_matrix(KKpar2D,"KKpar2D",nc_global,nc_global)
        print_matrix(KKperp2D,"KKperp2D",nc_global,nc_global)
        print_matrix(LP2D,"LP",nc_global,nc_global)
        print_matrix(LV2D,"LV",nc_global,nc_global)
    end
    # convert these matrices to sparse matrices
    println("begin conversion to sparse matrices   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    MM2D_sparse = sparse(MM2D)
    KKpar2D_sparse = sparse(KKpar2D)
    KKperp2D_sparse = sparse(KKperp2D)
    LP2D_sparse = sparse(LP2D)
    LV2D_sparse = sparse(LV2D)
    
    # create LU decomposition for mass matrix inversion
    println("begin LU decomposition initialisation   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    lu_obj_MM = lu(MM2D_sparse)
    lu_obj_LP = lu(LP2D_sparse)
    lu_obj_LV = lu(LV2D_sparse)
    #cholesky_obj = cholesky(MM2D_sparse)
    println("finish LU decomposition initialisation   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    # define a test function 
    
    fvpavperp = Array{mk_float,2}(undef,vpa.n,vperp.n)
    fvpavperp_test = Array{mk_float,2}(undef,vpa.n,vperp.n)
    fvpavperp_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2fvpavperp_dvpa2_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2fvpavperp_dvpa2_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2fvpavperp_dvpa2_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2fvpavperp_dvperp2_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2fvpavperp_dvperp2_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2fvpavperp_dvperp2_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    fc = Array{mk_float,1}(undef,nc_global)
    dfc = Array{mk_float,1}(undef,nc_global)
    gc = Array{mk_float,1}(undef,nc_global)
    dgc = Array{mk_float,1}(undef,nc_global)
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            fvpavperp[ivpa,ivperp] = exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
            d2fvpavperp_dvpa2_exact[ivpa,ivperp] = (4.0*vpa.grid[ivpa]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
            d2fvpavperp_dvperp2_exact[ivpa,ivperp] = (4.0*vperp.grid[ivperp]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
        end
    end
    
    # boundary conditions
    
    #fvpavperp[vpa.n,:] .= 0.0
    #fvpavperp[1,:] .= 0.0
    #fvpavperp[:,vperp.n] .= 0.0
    
    
    #print_matrix(fvpavperp,"fvpavperp",vpa.n,vperp.n)
    # fill fc with fvpavperp
    ravel_vpavperp_to_c!(fc,fvpavperp,vpa.n,vperp.n)
    ravel_c_to_vpavperp!(fvpavperp_test,fc,nc_global,vpa.n)
    @. fvpavperp_err = abs(fvpavperp - fvpavperp_test)
    println("max(ravel_err)",maximum(fvpavperp_err))
    #print_vector(fc,"fc",nc_global)
    # multiply by KKpar2D and fill dfc
    mul!(dfc,KKpar2D_sparse,fc)
    mul!(dgc,KKperp2D_sparse,fc)
    # enforce zero bc  
    enforce_zero_bc!(fc,vpa,vperp)
    enforce_zero_bc!(gc,vpa,vperp)
    # invert mass matrix and fill fc
    fc = lu_obj_MM \ dfc
    gc = lu_obj_MM \ dgc
    #fc = cholesky_obj \ dfc
    #print_vector(fc,"fc",nc_global)
    # unravel
    ravel_c_to_vpavperp!(d2fvpavperp_dvpa2_num,fc,nc_global,vpa.n)
    ravel_c_to_vpavperp!(d2fvpavperp_dvperp2_num,gc,nc_global,vpa.n)
    if nc_global < 30
        print_matrix(d2fvpavperp_dvpa2_num,"d2fvpavperp_dvpa2_num",vpa.n,vperp.n)
    end
    @. d2fvpavperp_dvpa2_err = abs(d2fvpavperp_dvpa2_num - d2fvpavperp_dvpa2_exact)
    println("maximum(d2fvpavperp_dvpa2_err): ",maximum(d2fvpavperp_dvpa2_err))
    @. d2fvpavperp_dvperp2_err = abs(d2fvpavperp_dvperp2_num - d2fvpavperp_dvperp2_exact)
    println("maximum(d2fvpavperp_dvperp2_err): ",maximum(d2fvpavperp_dvperp2_err))
    if nc_global < 30
        print_matrix(d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2_err",vpa.n,vperp.n)
    end
    if plot_test_output
        plot_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2",vpa,vperp)
        plot_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fvpavperp_dvperp2",vpa,vperp)
    end
    # test the Laplacian solve with a standard F_Maxwellian -> H_Maxwellian test
    
    S_dummy = Array{mk_float,2}(undef,vpa.n,vperp.n)
    F_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
    H_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    H_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    H_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    G_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    G_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    G_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvpa2_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvpa2_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvpa2_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvperp2_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvperp2_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvperp2_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dGdvperp_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dGdvperp_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dGdvperp_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvperpdvpa_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvperpdvpa_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Gdvperpdvpa_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dHdvpa_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dHdvpa_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dHdvpa_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dHdvperp_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dHdvperp_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dHdvperp_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)

    dens = 1.0
    upar = 1.0
    vth = 1.0
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            F_M[ivpa,ivperp] = -(4.0/sqrt(pi))*F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dGdvperp_M_exact[ivpa,ivperp] = dGdvperp(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvpa_M_exact[ivpa,ivperp] = dHdvpa(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvperp_M_exact[ivpa,ivperp] = dHdvperp(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
    end
    # calculate the Rosenbluth potential boundary data (rpbd)
    rpbd = allocate_rosenbluth_potential_boundary_data(vpa,vperp)
    calculate_rosenbluth_potential_boundary_data_exact!(rpbd,
      H_M_exact,dHdvpa_M_exact,dHdvperp_M_exact,G_M_exact,
      dGdvperp_M_exact,d2Gdvperp2_M_exact,
      d2Gdvperpdvpa_M_exact,d2Gdvpa2_M_exact,vpa,vperp)
    
    println("begin H calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    ravel_vpavperp_to_c!(fc,F_M,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,MM2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,H_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.H_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(H_M_num,fc,nc_global,vpa.n)
    @. H_M_err = abs(H_M_num - H_M_exact)
    println("finish H calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(H_M_err): ",maximum(H_M_err))
    
    println("begin dHdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    ravel_vpavperp_to_c!(fc,F_M,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PPpar2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,dHdvpa_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.dHdvpa_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(dHdvpa_M_num,fc,nc_global,vpa.n)
    @. dHdvpa_M_err = abs(dHdvpa_M_num - dHdvpa_M_exact)
    println("finish dHdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(dHdvpa_M_err): ",maximum(dHdvpa_M_err))
    
    println("begin dHdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    ravel_vpavperp_to_c!(fc,F_M,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PUperp2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,dHdvperp_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.dHdvperp_data)
    fc = lu_obj_LV \ dfc
    ravel_c_to_vpavperp!(dHdvperp_M_num,fc,nc_global,vpa.n)
    @. dHdvperp_M_err = abs(dHdvperp_M_num - dHdvperp_M_exact)
    println("finish dHdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(dHdvperp_M_err): ",maximum(dHdvperp_M_err))
    
    println("begin G calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,MM2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,G_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.G_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(G_M_num,fc,nc_global,vpa.n)
    @. G_M_err = abs(G_M_num - G_M_exact)
    println("finish G calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(G_M_err): ",maximum(G_M_err))
                
    println("begin d2Gdvpa2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,KKpar2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,d2Gdvpa2_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.d2Gdvpa2_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(d2Gdvpa2_M_num,fc,nc_global,vpa.n)
    @. d2Gdvpa2_M_err = abs(d2Gdvpa2_M_num - d2Gdvpa2_M_exact)
    println("finish d2Gdvpa2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(d2Gdvpa2_M_err): ",maximum(d2Gdvpa2_M_err))
    
    println("begin dGdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PUperp2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,dGdvperp_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.dGdvperp_data)
    fc = lu_obj_LV \ dfc
    ravel_c_to_vpavperp!(dGdvperp_M_num,fc,nc_global,vpa.n)
    @. dGdvperp_M_err = abs(dGdvperp_M_num - dGdvperp_M_exact)
    println("finish dGdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(dGdvperp_M_err): ",maximum(dGdvperp_M_err))
    
    println("begin d2Gdvperpdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PPparPUperp2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,d2Gdvperpdvpa_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.d2Gdvperpdvpa_data)
    fc = lu_obj_LV \ dfc
    ravel_c_to_vpavperp!(d2Gdvperpdvpa_M_num,fc,nc_global,vpa.n)
    @. d2Gdvperpdvpa_M_err = abs(d2Gdvperpdvpa_M_num - d2Gdvperpdvpa_M_exact)
    println("finish d2Gdvperpdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(d2Gdvperpdvpa_M_err): ",maximum(d2Gdvperpdvpa_M_err))
    
    # use relation 2H = del2 G to compute d2Gdpverp2
    println("begin d2Gdvperp2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    @. S_dummy = -dGdvperp_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,MMparMNperp2D,fc)
    @. S_dummy = 2.0*H_M_num - d2Gdvpa2_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    mul!(dgc,MM2D,fc)
    dfc += dgc
    #enforce_dirichlet_bc!(dfc,vpa,vperp,d2Gdvperp2_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.d2Gdvperp2_data)
    fc = lu_obj_MM \ dfc
    ravel_c_to_vpavperp!(d2Gdvperp2_M_num,fc,nc_global,vpa.n)
    #@. d2Gdvperp2_M_num += 2.0*H_M_num - d2Gdvpa2_M_num
    @. d2Gdvperp2_M_err = abs(d2Gdvperp2_M_num - d2Gdvperp2_M_exact)
    println("finish d2Gdvperp2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    println("maximum(d2Gdvperp2_M_err): ",maximum(d2Gdvperp2_M_err))
    
    if plot_test_output
        plot_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp)
        plot_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp)
        plot_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp)
        plot_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp)
        plot_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp)
        plot_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp)
        plot_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp)
        plot_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp)
    end

end