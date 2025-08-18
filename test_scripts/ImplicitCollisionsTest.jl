using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using Dates
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int, OptionsDict
using moment_kinetics.fokker_planck: init_fokker_planck_collisions_weak_form
using moment_kinetics.fokker_planck: setup_fp_nl_solve, setup_fkpl_collisions_input,
                                     implicit_ion_fokker_planck_self_collisions!,
                                     fokker_planck_self_collisions_backward_euler_step!
using moment_kinetics.fokker_planck_test: F_Maxwellian, F_Beam, print_test_data
using moment_kinetics.velocity_moments: get_density, get_upar, get_p, get_ppar, get_qpar, get_rmom
using moment_kinetics.communication
using moment_kinetics.communication: MPISharedArray
using moment_kinetics.looping
using moment_kinetics.input_structs: direct_integration, multipole_expansion
using moment_kinetics.nonlinear_solvers
using moment_kinetics.species_input: get_species_input
using moment_kinetics.input_structs: options_to_TOML

function diagnose_F_Maxwellian(pdf,pdf_exact,pdf_dummy_1,pdf_dummy_2,vpa,vperp,time,mass,it)
    @begin_serial_region()
    @serial_region begin
        dens = get_density(pdf,vpa,vperp)
        upar = get_upar(pdf, dens, vpa, vperp, false)
        pressure = get_p(pdf, dens, upar, vpa, vperp, false, false)
        vth = sqrt(2.0*pressure/(dens*mass))
        ppar = get_ppar(dens, upar, pressure, vth, pdf, vpa, vperp, false, false,
                    false)
        qpar = get_qpar(pdf, dens, upar, pressure, vth, vpa, vperp, false, false,
                    false)
        rmom = get_rmom(pdf, upar, vpa, vperp)
        @loop_vperp_vpa ivperp ivpa begin
            pdf_exact[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
        println("it = ", it, " time: ", time)
        print_test_data(pdf_exact,pdf,pdf_dummy_1,"F",vpa,vperp,pdf_dummy_2;print_to_screen=true)
        println("dens: ", dens)
        println("upar: ", upar)
        println("vth: ", vth)
        println("ppar: ", ppar)
        println("qpar: ", qpar)
        println("rmom: ", rmom)
        if vpa.bc == "zero"
            println("test vpa bc: F[1, :]", pdf[1, :])
            println("test vpa bc: F[end, :]", pdf[end, :])
        end
        if vperp.bc == "zero"
            println("test vperp bc: F[:, end]", pdf[:, end])
        end
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

function test_implicit_collisions(; vth0=0.5,vperp0=1.0,vpa0=0.0, ngrid=3,nelement_vpa=8,nelement_vperp=4,
    Lvpa=6.0,Lvperp=3.0,bc_vpa="none",bc_vperp="none",
    ntime=1,delta_t=1.0, zbeam=0.0,
    atol = 1.0e-10,
    rtol = 0.0,
    nonlinear_max_iterations = 20,
    test_particle_preconditioner=true,
    test_linearised_advance=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    plot_test_output=false,
    test_dense_construction=false,standalone=true,
    test_numerical_conserving_terms=false,
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
                                "bc"=>bc_vperp),
        "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vpa,
                            "nelement_local"=>nelement_local_vpa, "L"=>Lvpa,
                            "discretization"=>discretization,
                            "element_spacing_option"=>element_spacing_option,
                            "bc"=>bc_vpa),
    )
    #println("made inputs")
    #println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
    #println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
    # create the coordinate structs
    vperp, vperp_spectral = define_coordinate(coords_input, "vperp")
    vpa, vpa_spectral = define_coordinate(coords_input, "vpa")
    if vperp.bc == "zero-impose-regularity"
        error("vperp.bc = $(vperp.bc) not supported for implicit FP")
    end
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
    fvpavperp = allocate_shared_float(; vpa=vpa, vperp=vperp, t=ntime+1)
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            fvpavperp[ivpa,ivperp,1] = F_Beam(vpa0,vperp0,vth0,vpa,vperp,ivpa,ivperp) +
                                        + zbeam * F_Beam(0.0,vperp0,vth0,vpa,vperp,ivpa,ivperp)
        end
        if vpa.bc == "zero"
            @loop_vperp ivperp begin
                fvpavperp[1,ivperp,1] = 0.0
                fvpavperp[end,ivperp,1] = 0.0
            end
        end
        if vperp.bc == "zero"
            @loop_vpa ivpa begin
                fvpavperp[ivpa,end,1] = 0.0
            end
        end
        # normalise to unit density
        @views densfac = get_density(fvpavperp[:,:,1],vpa,vperp)
        @loop_vperp_vpa ivperp ivpa begin
            fvpavperp[ivpa,ivperp,1] /= densfac
        end
    end
    # arrays needed for advance
    Fold = allocate_shared_float(vpa, vperp)
    # dummy arrays
    Fdummy1 = allocate_shared_float(vpa, vperp)
    Fdummy2 = allocate_shared_float(vpa, vperp)
    Fdummy3 = allocate_shared_float(vpa, vperp)
    # physics parameters
    ms = 1.0
    nuss = 1.0

    # initial condition 
    time = 0.0
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            Fold[ivpa,ivperp] = fvpavperp[ivpa,ivperp,1]
        end
    end
    diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,0)
    
    implicit_ion_fp_collisions = true
    coords = (vperp=vperp,vpa=vpa)
    spectral = (vperp_spectral=vperp_spectral, vpa_spectral=vpa_spectral)
    input_toml = OptionsDict("fokker_planck_collisions_nonlinear_solver" => OptionsDict("rtol" => rtol,
                                                                                        "atol" => atol,
                                                                                        "nonlinear_max_iterations" => nonlinear_max_iterations),)
    fkpl = setup_fkpl_collisions_input(input_toml, true)
    nl_solver_params = setup_fp_nl_solve(implicit_ion_fp_collisions,
                                        input_toml, coords)

    #println(nl_solver_params.preconditioners)
    for it in 1:ntime
        @begin_r_z_anysv_region()
        fokker_planck_self_collisions_backward_euler_step!(Fold, delta_t, ms, nuss, fkpl_arrays,
            coords, spectral,
            nl_solver_params,
            test_numerical_conserving_terms=test_numerical_conserving_terms,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
            boundary_data_option=boundary_data_option)
        @begin_serial_region()
        # update the pdf
        @serial_region begin
            Fnew = fkpl_arrays.Fnew
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
    finish_run_time = now()
    if plot_test_output
        diagnose_F_gif(fvpavperp,vpa,vperp,ntime)
    end
    println("init time (ms): ", Dates.value(finish_init_time - start_init_time))
    println("run time (ms): ", Dates.value(finish_run_time - finish_init_time))
end

function test_implicit_collisions_wrapper(; vth0=0.5,vperp0=1.0,vpa0=0.0, ngrid=3,nelement_vpa=8,nelement_vperp=4,
    Lvpa=6.0,Lvperp=3.0, bc_vpa="none", bc_vperp = "none",
    ntime=1,delta_t=1.0, zbeam = 0.0,
    atol = 1.0e-10,
    rtol = 0.0,
    nonlinear_max_iterations = 20,
    test_particle_preconditioner=true,
    test_linearised_advance=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    plot_test_output=false,
    test_dense_construction=false,standalone=true,
    test_numerical_conserving_terms=false,
    boundary_data_option=multipole_expansion,
    use_end_of_step_corrections=true,
    diagnose=true)
    
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
                                "bc"=>bc_vperp),
        "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vpa,
                            "nelement_local"=>nelement_local_vpa, "L"=>Lvpa,
                            "discretization"=>discretization,
                            "element_spacing_option"=>element_spacing_option,
                            "bc"=>bc_vpa),
        "z"=>OptionsDict("ngrid"=> 1, "nelement"=> 1,
                            "nelement_local"=> 1, "bc"=>"none"),
        "r"=>OptionsDict("ngrid"=> 1, "nelement"=> 1,
                            "nelement_local"=> 1),
    )
    #println("made inputs")
    #println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
    #println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
    # create the coordinate structs
    vperp, vperp_spectral = define_coordinate(coords_input, "vperp")
    vpa, vpa_spectral = define_coordinate(coords_input, "vpa")
    r, r_spectral = define_coordinate(coords_input, "r")
    z, z_spectral = define_coordinate(coords_input, "z")
    composition = get_species_input(OptionsDict("composition" => OptionsDict("n_ion_species" => 1, "n_neutral_species" => 0) ), true)
    if vperp.bc == "zero-impose-regularity"
        error("vperp.bc = $(vperp.bc) not supported for implicit FP")
    end
    looping.setup_loop_ranges!(block_rank[], block_size[];
                                    s=composition.n_ion_species, sn=1,
                                    r=r.n, z=z.n, vperp=vperp.n, vpa=vpa.n,
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
    input_toml = OptionsDict("fokker_planck_collisions" => OptionsDict( "use_fokker_planck" => true,
                                                                        "nuii" => 1.0,
                                                                        "frequency_option" => "manual",
                                                                        "self_collisions" => true,
                                                                        "use_conserving_corrections" => test_numerical_conserving_terms,
                                                                        "boundary_data_option" => boundary_data_option,
                                                                        "use_test_particle_preconditioner" => test_particle_preconditioner,),
                            "fokker_planck_collisions_nonlinear_solver" => OptionsDict("rtol" => rtol,
                                                                                       "atol" => atol,
                                                                                       "nonlinear_max_iterations" => nonlinear_max_iterations),
                            )
    #println(options_to_TOML(input_toml))
    collisions = (fkpl = setup_fkpl_collisions_input(input_toml, true), krook=nothing)

    finish_init_time = now()
    
    # initial condition
    nr = r.n
    nz = z.n
    ns = composition.n_ion_species
    fvpavperpzrst = allocate_shared_float(; vpa=vpa, vperp=vperp, z=z, r=r, ion_species=ns, t=ntime+1)
    fvpavperpzrs_old = allocate_shared_float(; vpa=vpa, vperp=vperp, z=z, r=r, ion_species=ns)
    fvpavperpzrs_new = allocate_shared_float(; vpa=vpa, vperp=vperp, z=z, r=r, ion_species=ns)
    dSdt = allocate_shared_float(; z=z, r=r, ion_species=ns)
    # variables needed to control moment kinetic normalisation factors
    # not the density or vth assocated with the pdf solved for in
    # the fokker_planck.jl functions.
    mk_density = allocate_shared_float(nz,nr,ns)
    mk_vth = allocate_shared_float(nz,nr,ns)
    evolve_p = false
    evolve_density = false
    @serial_region begin
        @loop_s_r_z is ir iz begin
            @loop_vperp_vpa ivperp ivpa begin
                fvpavperpzrst[ivpa,ivperp,iz,ir,is,1] = F_Beam(vpa0,vperp0,vth0,vpa,vperp,ivpa,ivperp) +
                                                  zbeam * F_Beam(0.0,vperp0,vth0,vpa,vperp,ivpa,ivperp)
            end
            if vpa.bc == "zero"
                @loop_vperp ivperp begin
                    fvpavperpzrst[1,ivperp,1] = 0.0
                    fvpavperpzrst[end,ivperp,1] = 0.0
                end
            end
            if vperp.bc == "zero"
                @loop_vpa ivpa begin
                    fvpavperpzrst[ivpa,end,1] = 0.0
                end
            end
            # normalise to unit density
            @views densfac = get_density(fvpavperpzrst[:,:,iz,ir,is,1],vpa,vperp)
            @loop_vperp_vpa ivperp ivpa begin
                fvpavperpzrst[ivpa,ivperp,iz,ir,is,1] /= densfac
            end
            dSdt[iz,ir,is] = 0.0
        end
    end
    # arrays needed for advance
    Fold = allocate_shared_float(vpa, vperp)
    # dummy arrays
    Fdummy1 = allocate_shared_float(vpa, vperp)
    Fdummy2 = allocate_shared_float(vpa, vperp)
    Fdummy3 = allocate_shared_float(vpa, vperp)
    # physics parameters
    ms = 1.0
    nuss = 1.0

    # initial condition 
    time = 0.0
    @serial_region begin
        @loop_s_r_z is ir iz begin
            @loop_vperp_vpa ivperp ivpa begin
                fvpavperpzrs_old[ivpa,ivperp,iz,ir,is] = fvpavperpzrst[ivpa,ivperp,iz,ir,is,1]
            end
        end
    end
    @views diagnose_F_Maxwellian(fvpavperpzrs_old[:,:,1,1,1],Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,0)
    
    implicit_ion_fp_collisions = true
    coords = (vperp=vperp,vpa=vpa)
    spectral_objects = (vperp_spectral=vperp_spectral, vpa_spectral=vpa_spectral)
    nl_solver_params = setup_fp_nl_solve(implicit_ion_fp_collisions,
                                         input_toml, coords)

    #println(nl_solver_params.preconditioners)
    start_run_time = now()
    for it in 1:ntime
        success = implicit_ion_fokker_planck_self_collisions!(fvpavperpzrs_new, fvpavperpzrs_old, dSdt,
                                        mk_density, mk_vth, evolve_p, evolve_density,
                                        composition, collisions, fkpl_arrays, 
                                        vpa, vperp, z, r, delta_t, spectral_objects,
                                        nl_solver_params; diagnose_entropy_production=true,
                                        test_linearised_advance=test_linearised_advance,
                                        use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
                                        use_end_of_step_corrections=use_end_of_step_corrections)
        @begin_s_r_z_vperp_region()
        # update the pdf
        @loop_s_r_z is ir iz begin
            @loop_vperp_vpa ivperp ivpa begin
                fvpavperpzrs_old[ivpa,ivperp,iz,ir,is] = fvpavperpzrs_new[ivpa,ivperp,iz,ir,is]
            end
        end
        # diagnose Fold
        time += delta_t
        if diagnose
            @views diagnose_F_Maxwellian(fvpavperpzrs_old[:,:,1,1,1],Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,it)
            # update outputs
            @serial_region begin
                println("dSdt = ", dSdt[1,1,1])
                @loop_s_r_z is ir iz begin
                    @loop_vperp_vpa ivperp ivpa begin
                        fvpavperpzrst[ivpa,ivperp,iz,ir,is,it+1] = fvpavperpzrs_old[ivpa,ivperp,iz,ir,is]
                    end
                end
            end
        end
    end
    finish_run_time = now()
    if plot_test_output
        @views diagnose_F_gif(fvpavperpzrst[:,:,1,1,1,:],vpa,vperp,ntime)
    end
    println("init time (ms): ", Dates.value(finish_init_time - start_init_time))
    println("run time (ms): ", Dates.value(finish_run_time - start_run_time))
end
    
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
    # test_implicit_collisions(ngrid=3,nelement_vpa=8,nelement_vperp=4,ntime=50,delta_t=1.0,
    #   serial_solve=false,anysv_region=true,plot_test_output=false,
    #   test_numerical_conserving_terms=true)
    test_implicit_collisions_wrapper(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
    vth0=0.5,vperp0=1.0,vpa0=1.0, nelement_vpa=32,nelement_vperp=16,Lvpa=8.0,Lvperp=4.0, bc_vpa="none", bc_vperp="none",
     ntime=2, delta_t = 1.0, ngrid=5, test_linearised_advance=false, use_end_of_step_corrections=true)
end
