using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using Dates
import moment_kinetics
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral, get_QQ_local!
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.fokker_planck: F_Maxwellian, H_Maxwellian, G_Maxwellian, Cssp_Maxwellian_inputs
using moment_kinetics.fokker_planck: d2Gdvpa2, d2Gdvperp2, dGdvperp, d2Gdvperpdvpa, dHdvpa, dHdvperp
using moment_kinetics.fokker_planck: init_fokker_planck_collisions, fokkerplanck_arrays_struct, fokkerplanck_boundary_data_arrays_struct
using moment_kinetics.fokker_planck: init_fokker_planck_collisions_new, boundary_integration_weights_struct
using moment_kinetics.fokker_planck: get_element_limit_indices
using moment_kinetics.calculus: derivative!
using moment_kinetics.communication
using moment_kinetics.communication: MPISharedArray
using moment_kinetics.looping
using SparseArrays: sparse
using LinearAlgebra: mul!, lu, cholesky

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    
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
        lower_boundary_vpa::MPISharedArray{mk_float,1}
        upper_boundary_vpa::MPISharedArray{mk_float,1}
        upper_boundary_vperp::MPISharedArray{mk_float,1}
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
    
    
    function calculate_boundary_data!(func_data::vpa_vperp_boundary_data,
                                            weight::MPISharedArray{mk_float,4},func_input,vpa,vperp)
        nvpa = vpa.n
        nvperp = vperp.n
        #for ivperp in 1:nvperp
        begin_vperp_region()
        @loop_vperp ivperp begin
            func_data.lower_boundary_vpa[ivperp] = 0.0
            func_data.upper_boundary_vpa[ivperp] = 0.0
            for ivperpp in 1:nvperp
                for ivpap in 1:nvpa
                    func_data.lower_boundary_vpa[ivperp] += weight[ivpap,ivperpp,1,ivperp]*func_input[ivpap,ivperpp]
                    func_data.upper_boundary_vpa[ivperp] += weight[ivpap,ivperpp,nvpa,ivperp]*func_input[ivpap,ivperpp]
                end
            end
        end
        #for ivpa in 1:nvpa
        begin_vpa_region()
        @loop_vpa ivpa begin
            func_data.upper_boundary_vperp[ivpa] = 0.0
            for ivperpp in 1:nvperp
                for ivpap in 1:nvpa
                    func_data.upper_boundary_vperp[ivpa] += weight[ivpap,ivperpp,ivpa,nvperp]*func_input[ivpap,ivperpp]
                end
            end
        end
        # return to serial parallelisation
        begin_serial_region()
        return nothing
    end
    
    function calculate_boundary_data!(func_data::vpa_vperp_boundary_data,
                                      weight::boundary_integration_weights_struct,
                                      func_input,vpa,vperp)
        nvpa = vpa.n
        nvperp = vperp.n
        #for ivperp in 1:nvperp
        begin_vperp_region()
        @loop_vperp ivperp begin
            func_data.lower_boundary_vpa[ivperp] = 0.0
            func_data.upper_boundary_vpa[ivperp] = 0.0
            for ivperpp in 1:nvperp
                for ivpap in 1:nvpa
                    func_data.lower_boundary_vpa[ivperp] += weight.lower_vpa_boundary[ivpap,ivperpp,ivperp]*func_input[ivpap,ivperpp]
                    func_data.upper_boundary_vpa[ivperp] += weight.upper_vpa_boundary[ivpap,ivperpp,ivperp]*func_input[ivpap,ivperpp]
                end
            end
        end
        #for ivpa in 1:nvpa
        begin_vpa_region()
        @loop_vpa ivpa begin
            func_data.upper_boundary_vperp[ivpa] = 0.0
            for ivperpp in 1:nvperp
                for ivpap in 1:nvpa
                    func_data.upper_boundary_vperp[ivpa] += weight.upper_vperp_boundary[ivpap,ivperpp,ivpa]*func_input[ivpap,ivperpp]
                end
            end
        end
        # return to serial parallelisation
        begin_serial_region()
        return nothing
    end
    
    function calculate_rosenbluth_potential_boundary_data!(rpbd::rosenbluth_potential_boundary_data,
        fkpl::Union{fokkerplanck_arrays_struct,fokkerplanck_boundary_data_arrays_struct},pdf)
        # get derivatives of pdf
        dfdvperp = fkpl.dfdvperp
        dfdvpa = fkpl.dfdvpa
        d2fdvperpdvpa = fkpl.d2fdvperpdvpa
        #for ivpa in 1:vpa.n
        begin_vpa_region()
        @loop_vpa ivpa begin
            @views derivative!(vperp.scratch, pdf[ivpa,:], vperp, vperp_spectral)
            @. dfdvperp[ivpa,:] = vperp.scratch
        end
        begin_vperp_region()
        @loop_vperp ivperp begin
        #for ivperp in 1:vperp.n
            @views derivative!(vpa.scratch, pdf[:,ivperp], vpa, vpa_spectral)
            @. dfdvpa[:,ivperp] = vpa.scratch
            @views derivative!(vpa.scratch, dfdvperp[:,ivperp], vpa, vpa_spectral)
            @. d2fdvperpdvpa[:,ivperp] = vpa.scratch
        end
        # ensure data is synchronized
        begin_serial_region()
        # carry out the numerical integration 
        calculate_boundary_data!(rpbd.H_data,fkpl.H0_weights,pdf,vpa,vperp)
        calculate_boundary_data!(rpbd.dHdvpa_data,fkpl.H0_weights,dfdvpa,vpa,vperp)
        calculate_boundary_data!(rpbd.dHdvperp_data,fkpl.H1_weights,dfdvperp,vpa,vperp)
        calculate_boundary_data!(rpbd.G_data,fkpl.G0_weights,pdf,vpa,vperp)
        calculate_boundary_data!(rpbd.dGdvperp_data,fkpl.G1_weights,dfdvperp,vpa,vperp)
        calculate_boundary_data!(rpbd.d2Gdvperp2_data,fkpl.H2_weights,dfdvperp,vpa,vperp)
        calculate_boundary_data!(rpbd.d2Gdvperpdvpa_data,fkpl.G1_weights,d2fdvperpdvpa,vpa,vperp)
        calculate_boundary_data!(rpbd.d2Gdvpa2_data,fkpl.H3_weights,dfdvpa,vpa,vperp)
        
        return nothing
    end
    
    function test_rosenbluth_potential_boundary_data(rpbd::rosenbluth_potential_boundary_data,
        rpbd_exact::rosenbluth_potential_boundary_data,vpa,vperp)
        
        error_buffer_vpa = Array{mk_float,1}(undef,vpa.n)
        error_buffer_vperp_1 = Array{mk_float,1}(undef,vperp.n)
        error_buffer_vperp_2 = Array{mk_float,1}(undef,vperp.n)
        test_boundary_data(rpbd.H_data,rpbd_exact.H_data,"H",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.dHdvpa_data,rpbd_exact.dHdvpa_data,"dHdvpa",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.dHdvperp_data,rpbd_exact.dHdvperp_data,"dHdvperp",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.G_data,rpbd_exact.G_data,"G",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.dGdvperp_data,rpbd_exact.dGdvperp_data,"dGdvperp",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.d2Gdvperp2_data,rpbd_exact.d2Gdvperp2_data,"d2Gdvperp2",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.d2Gdvperpdvpa_data,rpbd_exact.d2Gdvperpdvpa_data,"d2Gdvperpdvpa",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  
        test_boundary_data(rpbd.d2Gdvpa2_data,rpbd_exact.d2Gdvpa2_data,"d2Gdvpa2",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2)  

        return nothing
    end

    function test_boundary_data(func,func_exact,func_name,vpa,vperp,buffer_vpa,buffer_vperp_1,buffer_vperp_2)
        nvpa = vpa.n
        nvperp = vperp.n
        for ivperp in 1:nvperp
            buffer_vperp_1 = abs(func.lower_boundary_vpa[ivperp] - func_exact.lower_boundary_vpa[ivperp])
            buffer_vperp_2 = abs(func.upper_boundary_vpa[ivperp] - func_exact.upper_boundary_vpa[ivperp])
        end
        for ivpa in 1:nvpa
            buffer_vpa = abs(func.upper_boundary_vperp[ivpa] - func_exact.upper_boundary_vperp[ivpa])
        end
        @serial_region begin
            max_lower_vpa_err = maximum(buffer_vperp_1)
            max_upper_vpa_err = maximum(buffer_vperp_2)
            max_upper_vperp_err = maximum(buffer_vpa)
            println(string(func_name*" boundary data:"))
            println("max(lower_vpa_err) = ",max_lower_vpa_err)
            println("max(upper_vpa_err) = ",max_upper_vpa_err)
            println("max(upper_vperp_err) = ",max_upper_vperp_err)
        end
        return nothing
    end
    
    function get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
        # global indices on the grids
        ivpa_global = vpa.igrid_full[ivpa_local,ielement_vpa]
        ivperp_global = vperp.igrid_full[ivperp_local,ielement_vperp]
        # global compound index
        ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
        return ic_global, ivpa_global, ivperp_global
    end
    function enforce_zero_bc!(fc,vpa,vperp;impose_BC_at_zero_vperp=false)
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
        
        if impose_BC_at_zero_vperp
            # lower vperp boundary
            ielement_vperp = 1
            ivperp_local = 1
            for ielement_vpa in 1:vpa.nelement_local
                for ivpa_local in 1:vpa.ngrid
                    ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                    fc[ic_global] = 0.0
                end
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
    
    
    # define inputs needed for the test
	plot_test_output = true
    impose_zero_gradient_BC = false#true
    test_parallelism = true
    ngrid = 3 #number of points per element 
	nelement_local_vpa = 16 # number of elements per rank
	nelement_global_vpa = nelement_local_vpa # total number of elements 
	nelement_local_vperp = 8 # number of elements per rank
	nelement_global_vperp = nelement_local_vperp # total number of elements 
	Lvpa = 12.0 #physical box size in reference units 
	Lvperp = 6.0 #physical box size in reference units 
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
    # Set up MPI
    initialize_comms!()
    setup_distributed_memory_MPI(1,1,1,1)
    looping.setup_loop_ranges!(block_rank[], block_size[];
                                   s=1, sn=1,
                                   r=1, z=1, vperp=vperp.n, vpa=vpa.n,
                                   vzeta=1, vr=1, vz=1)
    begin_serial_region()
    
    
    # Assemble a 2D mass matrix in the global compound coordinate
    nc_global = vpa.n*vperp.n
    nc_local = vpa.ngrid*vperp.ngrid
    Index2D = Array{mk_int,2}(undef,nc_global,nc_global)
    MM2D = Array{mk_float,2}(undef,nc_global,nc_global)
    MM2D .= 0.0
    MM2DZG = Array{mk_float,2}(undef,nc_global,nc_global)
    MM2DZG .= 0.0
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
        
    impose_BC_at_zero_vperp = false
    @serial_region begin
        println("begin elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
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
    for ielement_vperp in 1:vperp.nelement_local
        get_QQ_local!(MMperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"M")
        for ielement_vpa in 1:vpa.nelement_local
            get_QQ_local!(MMpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"M")
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
                                    MM2DZG[ic_global,icp_global] = 1.0
                                else 
                                    MM2DZG[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_vpa
                                if ivpap_local == vpa.ngrid && ivperp_local == ivperpp_local 
                                    MM2DZG[ic_global,icp_global] = 1.0
                                else 
                                    MM2DZG[ic_global,icp_global] = 0.0
                                end
                            elseif lower_boundary_row_vperp && !lower_boundary_row_vpa && !upper_boundary_row_vperp
                                if ivpa_local == ivpap_local
                                    MM2DZG[ic_global,icp_global] = vperp_spectral.radau.D0[ivperpp_local]
                                else 
                                    MM2DZG[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_vperp
                                if ivperpp_local == vperp.ngrid && ivpa_local == ivpap_local
                                    MM2DZG[ic_global,icp_global] = 1.0
                                else 
                                    MM2DZG[ic_global,icp_global] = 0.0
                                end
                            else
                                # assign mass matrix data
                                #println("MM2D += ", MMpar[ivpa_local,ivpap_local]*MMperp[ivperp_local,ivperpp_local])
                                MM2DZG[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                                MMperp[ivperp_local,ivperpp_local]
                            end
                        end
                    end
                end
            end
        end
    end
    @serial_region begin
        println("finished elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        
        if nc_global < 30
            print_matrix(MM2D,"MM2D",nc_global,nc_global)
            print_matrix(MM2DZG,"MM2DZG",nc_global,nc_global)
            print_matrix(KKpar2D,"KKpar2D",nc_global,nc_global)
            print_matrix(KKperp2D,"KKperp2D",nc_global,nc_global)
            print_matrix(LP2D,"LP",nc_global,nc_global)
            print_matrix(LV2D,"LV",nc_global,nc_global)
        end
        # convert these matrices to sparse matrices
        println("begin conversion to sparse matrices   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    MM2D_sparse = sparse(MM2D)
    MM2DZG_sparse = sparse(MM2DZG)
    KKpar2D_sparse = sparse(KKpar2D)
    KKperp2D_sparse = sparse(KKperp2D)
    LP2D_sparse = sparse(LP2D)
    LV2D_sparse = sparse(LV2D)
    
    @serial_region begin
        # create LU decomposition for mass matrix inversion
        println("begin LU decomposition initialisation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    lu_obj_MM = lu(MM2D_sparse)
    lu_obj_MMZG = lu(MM2DZG_sparse)
    lu_obj_LP = lu(LP2D_sparse)
    lu_obj_LV = lu(LV2D_sparse)
    #cholesky_obj = cholesky(MM2D_sparse)
    @serial_region begin
        println("finished LU decomposition initialisation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
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
    @serial_region begin
        println("max(ravel_err)",maximum(fvpavperp_err))
    end
    #print_vector(fc,"fc",nc_global)
    # multiply by KKpar2D and fill dfc
    mul!(dfc,KKpar2D_sparse,fc)
    mul!(dgc,KKperp2D_sparse,fc)
    if impose_zero_gradient_BC
        # enforce zero bc  
        enforce_zero_bc!(fc,vpa,vperp,impose_BC_at_zero_vperp=true)
        enforce_zero_bc!(gc,vpa,vperp,impose_BC_at_zero_vperp=true)
        # invert mass matrix and fill fc
        fc = lu_obj_MMZG \ dfc
        gc = lu_obj_MMZG \ dgc
    else
        # enforce zero bc  
        enforce_zero_bc!(fc,vpa,vperp,impose_BC_at_zero_vperp=true)
        enforce_zero_bc!(gc,vpa,vperp,impose_BC_at_zero_vperp=true)
        # invert mass matrix and fill fc
        fc = lu_obj_MMZG \ dfc
        gc = lu_obj_MMZG \ dgc
    end
    #fc = cholesky_obj \ dfc
    #print_vector(fc,"fc",nc_global)
    # unravel
    ravel_c_to_vpavperp!(d2fvpavperp_dvpa2_num,fc,nc_global,vpa.n)
    ravel_c_to_vpavperp!(d2fvpavperp_dvperp2_num,gc,nc_global,vpa.n)
    @serial_region begin 
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
    end
    # test the Laplacian solve with a standard F_Maxwellian -> H_Maxwellian test
    
    S_dummy = Array{mk_float,2}(undef,vpa.n,vperp.n)
    Fs_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
    F_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
    C_M_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
    C_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
    C_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dFdvpa_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
    dFdvperp_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
    d2Fdvperpdvpa_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
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

    denss, upars, vths = 1.0, -1.0, 2.0/3.0
    dens = 1.0
    upar = 1.0
    vth = 1.0
    ms = 1.0
    msp = 1.0
    nussp = 1.0
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            Fs_M[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dGdvperp_M_exact[ivpa,ivperp] = dGdvperp(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvpa_M_exact[ivpa,ivperp] = dHdvpa(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvperp_M_exact[ivpa,ivperp] = dHdvperp(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            C_M_exact[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,
                                                            dens,upar,vth,msp,
                                                            nussp,vpa,vperp,ivpa,ivperp)
        end
    end
    # calculate the Rosenbluth potential boundary data (rpbd)
    rpbd_exact = allocate_rosenbluth_potential_boundary_data(vpa,vperp)
    rpbd = allocate_rosenbluth_potential_boundary_data(vpa,vperp)
    # use known test function to provide exact data
    calculate_rosenbluth_potential_boundary_data_exact!(rpbd_exact,
      H_M_exact,dHdvpa_M_exact,dHdvperp_M_exact,G_M_exact,
      dGdvperp_M_exact,d2Gdvperp2_M_exact,
      d2Gdvperpdvpa_M_exact,d2Gdvpa2_M_exact,vpa,vperp)
    # use numerical integration to find the boundary data
    # initialise the weights
    #fkpl_arrays = init_fokker_planck_collisions(vperp,vpa; precompute_weights=true)
    fkpl_arrays = init_fokker_planck_collisions_new(vpa,vperp; precompute_weights=true)
    begin_serial_region()
    # do the numerical integration at the boundaries (N.B. G not supported)
    @serial_region begin 
        println("begin boundary data calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    calculate_rosenbluth_potential_boundary_data!(rpbd,fkpl_arrays,F_M)
    @serial_region begin 
        println("finished boundary data calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    # test the boundary data calculation
    test_rosenbluth_potential_boundary_data(rpbd,rpbd_exact,vpa,vperp)
    #rpbd = rpbd_exact
    @serial_region begin
        println("begin H calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = -(4.0/sqrt(pi))*F_M
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,MM2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,H_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.H_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(H_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. H_M_err = abs(H_M_num - H_M_exact)
        println("finished H calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(H_M_err): ",maximum(H_M_err))
        
        println("begin dHdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = -(4.0/sqrt(pi))*F_M
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PPpar2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,dHdvpa_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.dHdvpa_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(dHdvpa_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. dHdvpa_M_err = abs(dHdvpa_M_num - dHdvpa_M_exact)
        println("finished dHdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(dHdvpa_M_err): ",maximum(dHdvpa_M_err))
        
        println("begin dHdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = -(4.0/sqrt(pi))*F_M
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PUperp2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,dHdvperp_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.dHdvperp_data)
    fc = lu_obj_LV \ dfc
    ravel_c_to_vpavperp!(dHdvperp_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. dHdvperp_M_err = abs(dHdvperp_M_num - dHdvperp_M_exact)
        println("finished dHdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(dHdvperp_M_err): ",maximum(dHdvperp_M_err))
        
        println("begin G calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,MM2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,G_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.G_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(G_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. G_M_err = abs(G_M_num - G_M_exact)
        println("finished G calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(G_M_err): ",maximum(G_M_err))
                    
        println("begin d2Gdvpa2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,KKpar2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,d2Gdvpa2_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.d2Gdvpa2_data)
    fc = lu_obj_LP \ dfc
    ravel_c_to_vpavperp!(d2Gdvpa2_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. d2Gdvpa2_M_err = abs(d2Gdvpa2_M_num - d2Gdvpa2_M_exact)
        println("finished d2Gdvpa2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(d2Gdvpa2_M_err): ",maximum(d2Gdvpa2_M_err))
        
        println("begin dGdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PUperp2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,dGdvperp_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.dGdvperp_data)
    fc = lu_obj_LV \ dfc
    ravel_c_to_vpavperp!(dGdvperp_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. dGdvperp_M_err = abs(dGdvperp_M_num - dGdvperp_M_exact)
        println("finished dGdvperp calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(dGdvperp_M_err): ",maximum(dGdvperp_M_err))
        
        println("begin d2Gdvperpdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    @. S_dummy = 2.0*H_M_num
    ravel_vpavperp_to_c!(fc,S_dummy,vpa.n,vperp.n)
    #enforce_zero_bc!(fc,vpa,vperp)
    mul!(dfc,PPparPUperp2D,fc)
    #enforce_dirichlet_bc!(dfc,vpa,vperp,d2Gdvperpdvpa_M_exact,dirichlet_vperp_BC=impose_BC_at_zero_vperp)
    enforce_dirichlet_bc!(dfc,vpa,vperp,rpbd.d2Gdvperpdvpa_data)
    fc = lu_obj_LV \ dfc
    ravel_c_to_vpavperp!(d2Gdvperpdvpa_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. d2Gdvperpdvpa_M_err = abs(d2Gdvperpdvpa_M_num - d2Gdvperpdvpa_M_exact)
        println("finished d2Gdvperpdvpa calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(d2Gdvperpdvpa_M_err): ",maximum(d2Gdvperpdvpa_M_err))
        
        # use relation 2H = del2 G to compute d2Gdpverp2
        println("begin d2Gdvperp2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
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
    @serial_region begin
        @. d2Gdvperp2_M_err = abs(d2Gdvperp2_M_num - d2Gdvperp2_M_exact)
        println("finished d2Gdvperp2 calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
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
    
    rhsc = Array{mk_float,1}(undef,nc_global)
    rhsc_check = Array{mk_float,1}(undef,nc_global)
    rhsc_err = Array{mk_float,1}(undef,nc_global)
    rhsvpavperp = MPISharedArray{mk_float,2}(undef,vpa.n,vperp.n)
    
    struct YY_collision_operator_arrays
        # let phi_j(vperp) be the jth Lagrange basis function, 
        # and phi'_j(vperp) the first derivative of the Lagrange basis function
        # on the iel^th element. Then, the arrays are defined as follows.
        # YY0perp[i,j,k,iel] = \int phi_i(vperp) phi_j(vperp) phi_k(vperp) vperp d vperp
        YY0perp::Array{mk_float,4}
        # YY1perp[i,j,k,iel] = \int phi_i(vperp) phi_j(vperp) phi'_k(vperp) vperp d vperp
        YY1perp::Array{mk_float,4}
        # YY2perp[i,j,k,iel] = \int phi_i(vperp) phi'_j(vperp) phi'_k(vperp) vperp d vperp
        YY2perp::Array{mk_float,4}
        # YY3perp[i,j,k,iel] = \int phi_i(vperp) phi'_j(vperp) phi_k(vperp) vperp d vperp
        YY3perp::Array{mk_float,4}
        # YY0par[i,j,k,iel] = \int phi_i(vpa) phi_j(vpa) phi_k(vpa) vpa d vpa
        YY0par::Array{mk_float,4}
        # YY1par[i,j,k,iel] = \int phi_i(vpa) phi_j(vpa) phi'_k(vpa) vpa d vpa
        YY1par::Array{mk_float,4}
        # YY2par[i,j,k,iel] = \int phi_i(vpa) phi'_j(vpa) phi'_k(vpa) vpa d vpa
        YY2par::Array{mk_float,4}
        # YY3par[i,j,k,iel] = \int phi_i(vpa) phi'_j(vpa) phi_k(vpa) vpa d vpa
        YY3par::Array{mk_float,4}
    end
    
    function calculate_YY_arrays(vpa,vperp)
        YY0perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
        YY1perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
        YY2perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
        YY3perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
        YY0par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
        YY1par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
        YY2par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
        YY3par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
        
        for ielement_vperp in 1:vperp.nelement_local
            @views get_QQ_local!(YY0perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY0")
            @views get_QQ_local!(YY1perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY1")
            @views get_QQ_local!(YY2perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY2")
            @views get_QQ_local!(YY3perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY3")
         end
         for ielement_vpa in 1:vpa.nelement_local
            @views get_QQ_local!(YY0par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY0")
            @views get_QQ_local!(YY1par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY1")
            @views get_QQ_local!(YY2par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY2")
            @views get_QQ_local!(YY3par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY3")
         end
        
        return YY_collision_operator_arrays(YY0perp,YY1perp,YY2perp,YY3perp,
                                            YY0par,YY1par,YY2par,YY3par)
    end
    
    function assemble_explicit_collision_operator_rhs_serial!(rhsc,pdfs,d2Gspdvpa2,d2Gspdvperpdvpa,
        d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
        vpa,vperp,vpa_spectral,vperp_spectral,
        YY_arrays::YY_collision_operator_arrays)
        # assemble RHS of collision operator
        @. rhsc = 0.0
        
        # loop over elements
        for ielement_vperp in 1:vperp.nelement_local
            YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            
            for ielement_vpa in 1:vpa.nelement_local
                YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                
                # loop over field positions in each element
                for ivperp_local in 1:vperp.ngrid
                    for ivpa_local in 1:vpa.ngrid
                        ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                        # carry out the matrix sum on each 2D element
                        for jvperpp_local in 1:vperp.ngrid
                            jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                            for kvperpp_local in 1:vperp.ngrid
                                kvperpp = vperp.igrid_full[kvperpp_local,ielement_vperp]
                                for jvpap_local in 1:vpa.ngrid
                                    jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                                    pdfjj = pdfs[jvpap,jvperpp]
                                    for kvpap_local in 1:vpa.ngrid
                                        kvpap = vpa.igrid_full[kvpap_local,ielement_vpa]
                                        # first three lines represent parallel flux terms
                                        # second three lines represent perpendicular flux terms
                                        rhsc[ic_global] += (YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY2par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvpa2[kvpap,kvperpp] +
                                                            YY3perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] - 
                                                            2.0*(ms/msp)*YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvpa[kvpap,kvperpp] +
                                                            # end parallel flux, start of perpendicular flux
                                                            YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY3par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] + 
                                                            YY2perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperp2[kvpap,kvperpp] - 
                                                            2.0*(ms/msp)*YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvperp[kvpap,kvperpp])
                                    end
                                end
                            end
                        end
                    end
                end 
            end
        end
        # correct for minus sign due to integration by parts
        # and multiply by the normalised collision frequency
        @. rhsc *= -nussp
        return nothing
    end
    
    function assemble_explicit_collision_operator_rhs_parallel!(rhsc,rhsvpavperp,pdfs,d2Gspdvpa2,d2Gspdvperpdvpa,
        d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
        vpa,vperp,vpa_spectral,vperp_spectral,
        YY_arrays::YY_collision_operator_arrays)
        # assemble RHS of collision operator
        begin_vperp_vpa_region() 
        @loop_vperp_vpa ivperp ivpa begin
            rhsvpavperp[ivpa,ivperp] = 0.0
        end
        # loop over collocation points to benefit from shared-memory parallelism
        ngrid_vpa, ngrid_vperp = vpa.ngrid, vperp.ngrid
        @loop_vperp_vpa ivperp_global ivpa_global begin
            igrid_vpa, ielement_vpax, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperpx, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa_global,ivperp_global,vpa,vperp)
            # loop over elements belonging to this collocation point
            for ielement_vperp in ielement_vperp_low:ielement_vperp_hi
                # correct local ivperp in the case that we on a boundary point
                ivperp_local = igrid_vperp + (ielement_vperp - ielement_vperp_low)*(1-ngrid_vperp)
                YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
                YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
                YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
                YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
                
                for ielement_vpa in ielement_vpa_low:ielement_vpa_hi
                    # correct local ivpa in the case that we on a boundary point
                    ivpa_local = igrid_vpa + (ielement_vpa - ielement_vpa_low)*(1-ngrid_vpa)
                    YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                    YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                    YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                    YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                    
                    # carry out the matrix sum on each 2D element
                    for jvperpp_local in 1:vperp.ngrid
                        jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                        for kvperpp_local in 1:vperp.ngrid
                            kvperpp = vperp.igrid_full[kvperpp_local,ielement_vperp]
                            for jvpap_local in 1:vpa.ngrid
                                jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                                pdfjj = pdfs[jvpap,jvperpp]
                                for kvpap_local in 1:vpa.ngrid
                                    kvpap = vpa.igrid_full[kvpap_local,ielement_vpa]
                                    # first three lines represent parallel flux terms
                                    # second three lines represent perpendicular flux terms
                                    rhsvpavperp[ivpa_global,ivperp_global] += -nussp*(YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY2par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvpa2[kvpap,kvperpp] +
                                                        YY3perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] - 
                                                        2.0*(ms/msp)*YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvpa[kvpap,kvperpp] +
                                                        # end parallel flux, start of perpendicular flux
                                                        YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY3par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] + 
                                                        YY2perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperp2[kvpap,kvperpp] - 
                                                        2.0*(ms/msp)*YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvperp[kvpap,kvperpp])
                                end
                            end
                        end
                    end
                 end
            end
        end
        # ravel to compound index
        begin_serial_region()
        ravel_vpavperp_to_c!(rhsc,rhsvpavperp,vpa.n,vperp.n)
        return nothing
    end
    
    @serial_region begin
        println("begin YY array calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    YY_arrays = calculate_YY_arrays(vpa,vperp)
    @serial_region begin
        println("begin C calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    if test_parallelism
        assemble_explicit_collision_operator_rhs_serial!(rhsc_check,Fs_M,
          d2Gdvpa2_M_num,d2Gdvperpdvpa_M_num,d2Gdvperp2_M_num,
          dHdvpa_M_num,dHdvperp_M_num,ms,msp,nussp,
          vpa,vperp,vpa_spectral,vperp_spectral,YY_arrays)
        @serial_region begin
            println("finished C RHS assembly (serial)   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    assemble_explicit_collision_operator_rhs_parallel!(rhsc,rhsvpavperp,Fs_M,
      d2Gdvpa2_M_num,d2Gdvperpdvpa_M_num,d2Gdvperp2_M_num,
      dHdvpa_M_num,dHdvperp_M_num,ms,msp,nussp,
      vpa,vperp,vpa_spectral,vperp_spectral,YY_arrays)
    @serial_region begin
        println("finished C RHS assembly (parallel)   ", Dates.format(now(), dateformat"H:MM:SS"))
    end    
    if test_parallelism
        @serial_region begin
            @. rhsc_err = abs(rhsc - rhsc_check)
            println("maximum(rhsc_err) (test parallelisation): ",maximum(rhsc_err))
        end    
    end
    if impose_zero_gradient_BC
        enforce_zero_bc!(rhsc,vpa,vperp,impose_BC_at_zero_vperp=true)
        # invert mass matrix and fill fc
        fc = lu_obj_MMZG \ rhsc
    else
        enforce_zero_bc!(rhsc,vpa,vperp)
        # invert mass matrix and fill fc
        fc = lu_obj_MM \ rhsc
    end
    ravel_c_to_vpavperp!(C_M_num,fc,nc_global,vpa.n)
    @serial_region begin
        @. C_M_err = abs(C_M_num - C_M_exact)
        println("finished C calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        println("maximum(C_M_err): ",maximum(C_M_err))
        plot_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp)
    end

end
