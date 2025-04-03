export run_assembly_test
using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using Dates
using LinearAlgebra: lu, ldiv!, mul!
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral, get_QQ_local!
using moment_kinetics.type_definitions: mk_float, mk_int, OptionsDict
using moment_kinetics.fokker_planck: init_fokker_planck_collisions_weak_form
using moment_kinetics.fokker_planck: fokker_planck_collision_operator_weak_form!
using moment_kinetics.fokker_planck: conserving_corrections!, setup_fp_nl_solve
using moment_kinetics.fokker_planck_calculus: enforce_vpavperp_BCs!, calculate_test_particle_preconditioner!
using moment_kinetics.fokker_planck_calculus: calculate_vpavperp_advection_terms!
using moment_kinetics.fokker_planck_test: F_Maxwellian, print_test_data
using moment_kinetics.calculus: derivative!
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.communication
using moment_kinetics.communication: MPISharedArray
using moment_kinetics.looping
using moment_kinetics.input_structs: direct_integration, multipole_expansion
using moment_kinetics.nonlinear_solvers
using moment_kinetics.z_advection: init_z_advection_implicit, z_advection_implicit_advance!

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
    
function print_matrix(matrix,name::String,n::mk_int,m::mk_int)
    println("\n ",name," \n")
    for i in 1:n
        for j in 1:m
            @printf("%.2f ", matrix[i,j])
        end
        println("")
    end
    println("\n")
end

function print_vector(vector,name::String,m::mk_int)
    println("\n ",name," \n")
    for j in 1:m
        @printf("%.3f ", vector[j])
    end
    println("")
    println("\n")
end 

function diagnose_F_Maxwellian(pdf,pdf_exact,pdf_dummy_1,pdf_dummy_2,vpa,vperp,time,mass,it)
    @begin_serial_region()
    @serial_region begin
        dens = get_density(pdf,vpa,vperp)
        upar = get_upar(pdf,vpa,vperp,dens)
        ppar = get_ppar(pdf,vpa,vperp,upar)
        pperp = get_pperp(pdf,vpa,vperp)
        pres = get_pressure(ppar,pperp) 
        vth = sqrt(2.0*pres/(dens*mass))
        @loop_vperp_vpa ivperp ivpa begin
            pdf_exact[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
        println("it = ", it, " time: ", time)
        print_test_data(pdf_exact,pdf,pdf_dummy_1,"F",vpa,vperp,pdf_dummy_2;print_to_screen=true)
        println("dens: ", dens)
        println("upar: ", upar)
        println("vth: ", vth)
    end
end

function diagnose_F_gif(pdf,vpa,vperp,ntime)
    @begin_serial_region()
    @serial_region begin
        anim = @animate for it in 1:ntime
            @views heatmap(vperp.grid, vpa.grid, pdf[:,:,it], xlabel="vperp", ylabel="vpa", c = :deep, interpolation = :cubic)
        end
        outfile = string("implicit_collisions_pdf_vs_vpa_vperp.gif")
        gif(anim, outfile, fps=5)
    end
end

function diagnose_F_gif(pdf,Ez,phi,density,vpa,vperp,z,ntime)
    @begin_serial_region()
    @serial_region begin
        iz = 1
        anim = @animate for it in 1:ntime
            @views heatmap(vperp.grid, vpa.grid, pdf[:,:,iz,it], xlabel="vperp", ylabel="vpa", c = :deep, interpolation = :cubic)
        end
        outfile = string("implicit_collisions_pdf_vs_vpa_vperp_iz_1.gif")
        gif(anim, outfile, fps=5)
        
        ivperp = 1
        anim = @animate for it in 1:ntime
            @views heatmap(z.grid, vpa.grid, pdf[:,ivperp,:,it], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
        end
        outfile = string("implicit_collisions_pdf_vs_vpa_z_ivperp_1.gif")
        gif(anim, outfile, fps=5)

        anim = @animate for it in 1:ntime
            @views plot(z.grid, Ez[:,it], xlabel="z", ylabel=L"E_z", label="")
        end
        outfile = string("implicit_collisions_Ez_z.gif")
        gif(anim, outfile, fps=5)
        
        anim = @animate for it in 1:ntime
            @views plot(z.grid, phi[:,it], xlabel="z", ylabel=L"\phi", label="")
        end
        outfile = string("implicit_collisions_phi_z.gif")
        gif(anim, outfile, fps=5)
        
        anim = @animate for it in 1:ntime
            @views plot(z.grid, density[:,it], xlabel="z", ylabel=L"n_i", label="")
        end
        outfile = string("implicit_collisions_density_z.gif")
        gif(anim, outfile, fps=5)
    end
end

function test_implicit_collisions(; vth0=0.5,vperp0=1.0,vpa0=0.0, ngrid=3,nelement_vpa=8,nelement_vperp=4,
    Lvpa=6.0,Lvperp=3.0,ntime=1,delta_t=1.0,
    restart = 8,
    max_restarts = 1,
    atol = 1.0e-10,
    serial_solve = false,
    anyv_region = true,
    test_particle_preconditioner=false,
    test_linearised_advance=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    plot_test_output=false,
    test_parallelism=false,test_self_operator=true,
    test_dense_construction=false,standalone=false,
    use_Maxwellian_Rosenbluth_coefficients=false,
    use_Maxwellian_field_particle_distribution=false,
    test_numerical_conserving_terms=false,
    algebraic_solve_for_d2Gdvperp2=false,
    boundary_data_option=multipole_expansion)
    
    nelement_local_vpa = nelement_vpa # number of elements per rank
    nelement_global_vpa = nelement_local_vpa # total number of elements 
    nelement_local_vperp = nelement_vperp # number of elements per rank
    nelement_global_vperp = nelement_local_vperp # total number of elements 
    bc = "" #not required to take a particular value, not used 
    # fd_option and adv_input not actually used so given values unimportant
    #discretization = "chebyshev_pseudospectral"
    discretization = "gausslegendre_pseudospectral"
    element_spacing_option = "uniform"
    
    # Set up MPI
    if standalone
        initialize_comms!()
    end
    setup_distributed_memory_MPI(1,1,1,1)
    coords_input = OptionsDict(
        "vperp"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vperp,
                                "nelement_local"=>nelement_local_vperp, "L"=>Lvperp,
                                "discretization"=>discretization,
                                "element_spacing_option"=>element_spacing_option,
                                "bc"=>"none"),
        "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vpa,
                            "nelement_local"=>nelement_local_vpa, "L"=>Lvpa,
                            "discretization"=>discretization,
                            "element_spacing_option"=>element_spacing_option,
                            "bc"=>"none"),
    )
    #println("made inputs")
    #println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
    #println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
    # create the coordinate structs
    vperp, vperp_spectral = define_coordinate(coords_input, "vperp")
    vpa, vpa_spectral = define_coordinate(coords_input, "vpa")
    looping.setup_loop_ranges!(block_rank[], block_size[];
                                    s=1, sn=1,
                                    r=1, z=1, vperp=vperp.n, vpa=vpa.n,
                                    vzeta=1, vr=1, vz=1)
    nc_global = vpa.n*vperp.n
    @begin_serial_region()
    start_init_time = now()
    if boundary_data_option == direct_integration
        precompute_weights = true
    else
        precompute_weights = false
    end
    fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; 
                        precompute_weights=precompute_weights, test_dense_matrix_construction=test_dense_construction)
    finish_init_time = now()
    
    # initial condition
    fvpavperp = allocate_shared_float(vpa.n,vperp.n,ntime+1)
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            fvpavperp[ivpa,ivperp,1] = exp(-((vpa.grid[ivpa]-vpa0)^2 + (vperp.grid[ivperp]-vperp0)^2)/(vth0^2))
        end
        # normalise to unit density
        @views densfac = get_density(fvpavperp[:,:,1],vpa,vperp)
        @loop_vperp_vpa ivperp ivpa begin
            fvpavperp[ivpa,ivperp,1] /= densfac
        end
    end
    # arrays needed for advance
    dummy_vpavperp = Array{mk_float,2}(undef,vpa.n,vperp.n)
    Fold = allocate_shared_float(vpa.n,vperp.n)
    Fnew = allocate_shared_float(vpa.n,vperp.n)
    # dummy arrays
    Fdummy1 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy2 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy3 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy4 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy5 = allocate_shared_float(vpa.n,vperp.n)
    # zero dummy arrays
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
        Fdummy1[ivpa,ivperp] = 0.0
        Fdummy2[ivpa,ivperp] = 0.0
        Fdummy3[ivpa,ivperp] = 0.0
        Fdummy4[ivpa,ivperp] = 0.0
        Fdummy5[ivpa,ivperp] = 0.0
        end
    end
    # physics parameters
    ms = 1.0
    msp = 1.0
    nussp = 1.0

    # initial condition 
    time = 0.0
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            Fold[ivpa,ivperp] = fvpavperp[ivpa,ivperp,1]
            Fnew[ivpa,ivperp] = Fold[ivpa,ivperp]
        end
    end
    diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,0)
    
    implicit_ion_fp_collisions = true
    coords = (vperp=vperp,vpa=vpa)
    nl_solver_params = setup_fp_nl_solve(implicit_ion_fp_collisions, coords)

    #println(nl_solver_params.preconditioners)
    for it in 1:ntime
        fokker_planck_backward_euler_step!(Fnew, Fold, delta_t, ms, msp, nussp, fkpl_arrays, dummy_vpavperp,
            vperp, vpa, vperp_spectral, vpa_spectral, coords,
            Fdummy1, Fdummy2, Fdummy3, Fdummy4, Fdummy5, nl_solver_params,
            test_numerical_conserving_terms=test_numerical_conserving_terms,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
            test_self_operator=test_self_operator,
            test_assembly_serial=test_parallelism,
            use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
            use_Maxwellian_field_particle_distribution=use_Maxwellian_field_particle_distribution,
            algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
            calculate_GG = false, calculate_dGdvperp=false,
            boundary_data_option=boundary_data_option)
        @begin_serial_region()
        # update the pdf
        @serial_region begin
            @loop_vperp_vpa ivperp ivpa begin
                Fold[ivpa,ivperp] = Fnew[ivpa,ivperp]
            end
        end
        # diagnose Fold
        time += delta_t
        diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,it)
        # update outputs
        @serial_region begin
            @loop_vperp_vpa ivperp ivpa begin
                fvpavperp[ivpa,ivperp,it+1] = Fold[ivpa,ivperp] 
            end
        end
    end
    
    #diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,ntime,ms)
    if plot_test_output
        diagnose_F_gif(fvpavperp,vpa,vperp,ntime)
    end    
end

function fokker_planck_backward_euler_step!(Fnew, Fold, delta_t, ms, msp, nussp, fkpl_arrays, dummy_vpavperp,
    vperp, vpa, vperp_spectral, vpa_spectral, coords,
    Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params; dvpadt=0.0,
    test_numerical_conserving_terms=false,
    test_linearised_advance=false,
    test_particle_preconditioner=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    test_self_operator=true,
    test_assembly_serial=false,
    use_Maxwellian_Rosenbluth_coefficients=false,
    use_Maxwellian_field_particle_distribution=false,
    algebraic_solve_for_d2Gdvperp2=false,
    calculate_GG = false, calculate_dGdvperp=false,
    boundary_data_option=multipole_expansion,
    standalone=true,
    upper_wall=false,
    lower_wall=false)
    
    # residual function to be used for Newton-Krylov
    function residual_func!(Fresidual, Fnew; krylov=false)
        #begin_s_r_z_anyv_region()
        fokker_planck_collision_operator_weak_form!(Fnew, Fnew, ms, msp, nussp,
                                                fkpl_arrays,
                                                vperp, vpa, vperp_spectral, vpa_spectral,
                                                test_assembly_serial=test_assembly_serial,
                                                use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
                                                use_Maxwellian_field_particle_distribution=use_Maxwellian_field_particle_distribution,
                                                algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                                                calculate_GG = false, calculate_dGdvperp=false,
                                                boundary_data_option=boundary_data_option)
        # enforce the boundary conditions on CC before it is used for timestepping
        enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral,
                        upper_wall=upper_wall,lower_wall=lower_wall)
        if test_numerical_conserving_terms && test_self_operator
            # make ad-hoc conserving corrections
            conserving_corrections!(fkpl_arrays.CC,Fnew,vpa,vperp,dummy_vpavperp)
        end
        
        calculate_vpavperp_advection_terms!(Fnew,
            dvpadt,fkpl_arrays,vpa,vperp)
        # enforce the boundary conditions on advection terms before it is used for timestepping
        enforce_vpavperp_BCs!(fkpl_arrays.rhs_advection,vpa,vperp,vpa_spectral,vperp_spectral,
                        upper_wall=upper_wall,lower_wall=lower_wall)
        @begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            Fresidual[ivpa,ivperp] = Fnew[ivpa,ivperp] - Fold[ivpa,ivperp] - delta_t * (fkpl_arrays.CC[ivpa,ivperp] + fkpl_arrays.rhs_advection[ivpa,ivperp])
        end
        return nothing
    end
    
    if standalone
        @begin_s_r_z_anyv_region()
    end
    if test_particle_preconditioner
      calculate_test_particle_preconditioner!(Fold,delta_t,ms,msp,nussp,
        vpa,vperp,vpa_spectral,vperp_spectral,
        fkpl_arrays, dvpadt,
        use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
        boundary_data_option=boundary_data_option)
      
      lu_CC = lu(fkpl_arrays.CC2D_sparse) 
      function test_particle_precon!(x)
         pdf = x
         pdf_scratch = fkpl_arrays.rhsvpavperp
         pdf_dummy = fkpl_arrays.S_dummy
         MM2D_sparse = fkpl_arrays.MM2D_sparse
         @begin_anyv_region()
         @anyv_serial_region begin
             @views @. pdf_scratch = pdf
             pdf_c = vec(pdf)
             pdf_scratch_c = vec(pdf_scratch)
             pdf_dummy_c = vec(pdf_dummy)
             mul!(pdf_dummy_c, MM2D_sparse, pdf_scratch_c)
             ldiv!(pdf_c,lu_CC,pdf_dummy_c)
         end
         return nothing
      end 
      right_preconditioner = test_particle_precon!
    else
        right_preconditioner = nothing
    end
    if test_linearised_advance
      test_particle_precon!(Fnew)
    else
      newton_solve!(Fnew, residual_func!, Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params;
                      coords, right_preconditioner=right_preconditioner)
    end
    @_anyv_subblock_synchronize()
    #begin_serial_region()
end    

function field_solve!(Ez, phi, density, Te, Ne, Fold, vpa, vperp, z, z_spectral)
    @begin_z_region()
    @loop_z iz begin
        @views density[iz] = get_density(Fold[:,:,iz],vpa,vperp)
        phi[iz] = Te * log(density[iz]/Ne)
    end
    @begin_serial_region()
    @serial_region begin
        derivative!(Ez, -phi, z, z_spectral)
    end
    return nothing
end

function test_implicit_standard_dke_collisions(; vth0=0.5,vperp0=1.0,vpa0=0.0, ngrid=3,nelement_vpa=8,nelement_vperp=4,ngrid_z=3,nelement_z=2,
    Lvpa=6.0,Lvperp=3.0,Lz=1.0,ntime=1,delta_t=1.0, nu_source = 1.0, nussp = 1.0, Te = 1.0,
    z_element_spacing_option="uniform",
    restart = 8,
    max_restarts = 1,
    atol = 1.0e-10,
    serial_solve = false,
    anyv_region = true,
    test_particle_preconditioner=false,
    test_linearised_advance=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    plot_test_output=false,
    test_parallelism=false,test_self_operator=true,
    test_dense_construction=false,standalone=false,
    use_Maxwellian_Rosenbluth_coefficients=false,
    use_Maxwellian_field_particle_distribution=false,
    test_numerical_conserving_terms=false,
    algebraic_solve_for_d2Gdvperp2=false,
    boundary_data_option=multipole_expansion)
    
    nelement_local_vpa = nelement_vpa # number of elements per rank
    nelement_global_vpa = nelement_local_vpa # total number of elements 
    nelement_local_vperp = nelement_vperp # number of elements per rank
    nelement_global_vperp = nelement_local_vperp # total number of elements 
    bc = "" #not required to take a particular value, not used 
    # fd_option and adv_input not actually used so given values unimportant
    #discretization = "chebyshev_pseudospectral"
    discretization = "gausslegendre_pseudospectral"
    element_spacing_option = "uniform"
    
    # Set up MPI
    if standalone
        initialize_comms!()
    end
    setup_distributed_memory_MPI(1,1,1,1)
    coords_input = OptionsDict(
        "z"=>OptionsDict("ngrid"=>ngrid_z, "nelement"=>nelement_z,
                                "nelement_local"=>nelement_z, "L"=>Lz,
                                "discretization"=>discretization,
                                "element_spacing_option"=>z_element_spacing_option,
                                "bc"=>"none"),
        "vperp"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vperp,
                                "nelement_local"=>nelement_local_vperp, "L"=>Lvperp,
                                "discretization"=>discretization,
                                "element_spacing_option"=>element_spacing_option,
                                "bc"=>"none"),
        "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vpa,
                            "nelement_local"=>nelement_local_vpa, "L"=>Lvpa,
                            "discretization"=>discretization,
                            "element_spacing_option"=>element_spacing_option,
                            "bc"=>"none"),
    )
    #println("made inputs")
    #println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
    #println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
    # create the coordinate structs
    z, z_spectral = define_coordinate(coords_input, "z")
    vperp, vperp_spectral = define_coordinate(coords_input, "vperp")
    vpa, vpa_spectral = define_coordinate(coords_input, "vpa")
    looping.setup_loop_ranges!(block_rank[], block_size[];
                                    s=1, sn=1,
                                    r=1, z=z.n, vperp=vperp.n, vpa=vpa.n,
                                    vzeta=1, vr=1, vz=1)
    @begin_serial_region()
    start_init_time = now()
    if boundary_data_option == direct_integration
        precompute_weights = true
    else
        precompute_weights = false
    end
    fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; 
                        precompute_weights=precompute_weights, test_dense_matrix_construction=test_dense_construction)
    if z.n > 1
        streaming_arrays = init_z_advection_implicit(z,z_spectral,vperp,vpa,delta_t)
    else
        streaming_arrays = nothing
    end
    finish_init_time = now()
    
    # initial condition
    fvpavperpz = allocate_shared_float(vpa.n,vperp.n,z.n,ntime+1)
    Ez_out = allocate_shared_float(z.n,ntime+1)
    phi_out = allocate_shared_float(z.n,ntime+1)
    density_out = allocate_shared_float(z.n,ntime+1)
    Hminus = allocate_float(vpa.n)
    Hplus = allocate_float(vpa.n)
    zerovpa = 1.e-10
    Hminus .= 0.0
    Hplus .= 0.0
    @loop_vpa ivpa begin
        if vpa.grid[ivpa] < -zerovpa  
            Hminus[ivpa] = 1.0
        end
        if vpa.grid[ivpa] > zerovpa
            Hplus[ivpa] = 1.0
        end
    end
    @serial_region begin
        @loop_z_vperp_vpa iz ivperp ivpa begin
            zfac = (((0.5 - z.grid[iz]/z.L)^(0.5))*(vpa.grid[ivpa]^2)*Hminus[ivpa] +
                        ((0.5 + z.grid[iz]/z.L)^(0.5))*(vpa.grid[ivpa]^2)*Hplus[ivpa])
            fvpavperpz[ivpa,ivperp,iz,1] = zfac*exp(-((vpa.grid[ivpa])^2 + (vperp.grid[ivperp])^2)/(vth0^2))/(vth0^3)
        end
    end
    # arrays needed for advance
    dummy_vpavperp = Array{mk_float,2}(undef,vpa.n,vperp.n)
    Fold = allocate_shared_float(vpa.n,vperp.n,z.n)
    Fnew = allocate_shared_float(vpa.n,vperp.n,z.n)
    density = allocate_shared_float(z.n)
    phi = allocate_shared_float(z.n)
    Ez = allocate_shared_float(z.n)
    FSource = allocate_shared_float(vpa.n,vperp.n,z.n)

    # dummy arrays for collision newton_solve!()
    Fdummy1 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy2 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy3 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy4 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy5 = allocate_shared_float(vpa.n,vperp.n)
    # zero dummy arrays
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
        Fdummy1[ivpa,ivperp] = 0.0
        Fdummy2[ivpa,ivperp] = 0.0
        Fdummy3[ivpa,ivperp] = 0.0
        Fdummy4[ivpa,ivperp] = 0.0
        Fdummy5[ivpa,ivperp] = 0.0
        end
    end
    # physics parameters
    ms = 1.0
    msp = 1.0
    Ne = 1.0

    # initial condition 
    time = 0.0
    @serial_region begin
        @loop_z_vperp_vpa iz ivperp ivpa begin
            Fold[ivpa,ivperp,iz] = fvpavperpz[ivpa,ivperp,iz,1]
            Fnew[ivpa,ivperp,iz] = Fold[ivpa,ivperp,iz]
            FSource[ivpa,ivperp,iz] = nu_source * exp(-((vpa.grid[ivpa]-vpa0)^2 + (vperp.grid[ivperp]-vperp0)^2)/(vth0^2))/(vth0^3)
        end
        #@loop_vpa ivpa begin
        #    if vpa.grid[ivpa] > 0.0
        #        FSource[ivpa,:,1] .= 0.0
        #    end
        #    if vpa.grid[ivpa] < -0.0
        #        FSource[ivpa,:,end] .= 0.0
        #    end
        #end
    end
    @views field_solve!(Ez_out[:,1], phi_out[:,1], density_out[:,1], Te, Ne, Fold, vpa, vperp, z, z_spectral)
    #diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,0)
    
    # coords and params for newton_solve!() for collisions only
    coords = (vperp=vperp,vpa=vpa)
    nl_solver_params = setup_nonlinear_solve(
        true,
        OptionsDict("nonlinear_solver" =>
                    OptionsDict("rtol" => 0.0,
                                "atol" => atol,
                                "linear_restart" => restart,
                                "linear_max_restarts" => max_restarts,
                                "nonlinear_max_iterations" => 100)),
        coords; serial_solve=serial_solve, anyv_region=anyv_region,
        preconditioner_type=Val(:lu))
    
    for it in 1:ntime
        if z.n > 1
            # use operator splitting
            # advance due to source and parallel streaming
            @begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                Fold[ivpa,ivperp,iz] += delta_t * FSource[ivpa,ivperp,iz]
            end
            z_advection_implicit_advance!(Fold,z,vpa,vperp,streaming_arrays)
            # compute fields
            field_solve!(Ez, phi, density, Te, Ne, Fold, vpa, vperp, z, z_spectral)
            #println(vec(Fold[:,3,1]))
            #println(vec(Fold[:,3,end]))
        else
            Ez[1] = 1.0
        end
        # advance collisions and velocity advection
        if true
            @begin_s_r_z_anyv_region()
            @loop_z iz begin
                dvpadt = 0.5*Ez[iz]
                @views fokker_planck_backward_euler_step!(Fnew[:,:,iz], Fold[:,:,iz], delta_t, ms, msp, nussp, fkpl_arrays, dummy_vpavperp,
                    vperp, vpa, vperp_spectral, vpa_spectral, coords,
                    Fdummy1, Fdummy2, Fdummy3, Fdummy4, Fdummy5, nl_solver_params,
                    dvpadt=dvpadt,
                    test_numerical_conserving_terms=test_numerical_conserving_terms,
                    test_particle_preconditioner=test_particle_preconditioner,
                    test_linearised_advance=test_linearised_advance,
                    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
                    test_self_operator=test_self_operator,
                    test_assembly_serial=test_parallelism,
                    use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
                    use_Maxwellian_field_particle_distribution=use_Maxwellian_field_particle_distribution,
                    algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                    calculate_GG = false, calculate_dGdvperp=false,
                    boundary_data_option=boundary_data_option,
                    standalone=false,
                    upper_wall=(iz==z.n && z.irank == z.nrank - 1 && z.n > 1),
                    lower_wall=(iz==1 && z.irank == 0 && z.n > 1),)
            end
            @begin_serial_region()
            # update the pdf
            @serial_region begin
                @loop_z_vperp_vpa iz ivperp ivpa begin
                    Fold[ivpa,ivperp,iz] = Fnew[ivpa,ivperp,iz]
                end
            end
        end
        # diagnose Fold
        time += delta_t
        println("it ",it," time ",time)
        #diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,it)
        # update outputs
        @serial_region begin
            @loop_z_vperp_vpa iz ivperp ivpa begin
                fvpavperpz[ivpa,ivperp,iz,it+1] = Fold[ivpa,ivperp,iz]
                Ez_out[iz,it+1] = Ez[iz]
                phi_out[iz,it+1] = phi[iz]
                density_out[iz,it+1] = density[iz]
            end
        end
    end
    
    #diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,ntime,ms)
    if plot_test_output
        diagnose_F_gif(fvpavperpz,Ez_out,phi_out,density_out,vpa,vperp,z,ntime)
    end    
end
    
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
    test_implicit_collisions(ngrid=3,nelement_vpa=8,nelement_vperp=4,ntime=50,delta_t=1.0,
      serial_solve=false,anyv_region=true,plot_test_output=false,
      test_numerical_conserving_terms=true)
end
