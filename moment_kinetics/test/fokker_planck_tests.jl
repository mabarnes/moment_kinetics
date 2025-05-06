module FokkerPlanckTests

include("setup.jl")

export backward_Euler_linearised_collisions_test
export backward_Euler_fokker_planck_self_collisions_test

using MPI
using LinearAlgebra: mul!, ldiv!
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.input_structs: direct_integration, multipole_expansion, delta_f_multipole

using moment_kinetics.fokker_planck: init_fokker_planck_collisions_weak_form, fokker_planck_collision_operator_weak_form!
using moment_kinetics.fokker_planck: conserving_corrections!, init_fokker_planck_collisions_direct_integration
using moment_kinetics.fokker_planck: density_conserving_correction!, fokker_planck_collision_operator_weak_form_Maxwellian_Fsp!
using moment_kinetics.fokker_planck: setup_fp_nl_solve, fokker_planck_self_collisions_backward_euler_step!
using moment_kinetics.fokker_planck: setup_fkpl_collisions_input
using moment_kinetics.fokker_planck_test: print_test_data, fkpl_error_data, allocate_error_data #, plot_test_data
using moment_kinetics.fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian
using moment_kinetics.fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using moment_kinetics.fokker_planck_test: dHdvperp_Maxwellian, dHdvpa_Maxwellian, Cssp_Maxwellian_inputs
using moment_kinetics.fokker_planck_calculus: calculate_rosenbluth_potentials_via_elliptic_solve!, calculate_rosenbluth_potential_boundary_data_exact!
using moment_kinetics.fokker_planck_calculus: test_rosenbluth_potential_boundary_data, allocate_rosenbluth_potential_boundary_data
using moment_kinetics.fokker_planck_calculus: enforce_vpavperp_BCs!, calculate_rosenbluth_potentials_via_direct_integration!
using moment_kinetics.fokker_planck_calculus: interpolate_2D_vspace!, calculate_test_particle_preconditioner!
using moment_kinetics.fokker_planck_calculus: advance_linearised_test_particle_collisions!

function create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=12.0,Lvperp=6.0,bc_vpa="zero",bc_vperp="zero")

        nelement_local_vpa = nelement_vpa # number of elements per rank
        nelement_global_vpa = nelement_local_vpa # total number of elements
        nelement_local_vperp = nelement_vperp # number of elements per rank
        nelement_global_vperp = nelement_local_vperp # total number of elements
        discretization = "gausslegendre_pseudospectral"
        # create the 'input' struct containing input info needed to create a
        # coordinate
        element_spacing_option = "uniform"
        coords_input = OptionsDict(
            "vperp"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vperp,
                                 "nelement_local"=>nelement_local_vperp, "L"=>Lvperp,
                                 "discretization"=>discretization, "bc"=>bc_vperp,
                                 "element_spacing_option"=>element_spacing_option),
            "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vpa,
                               "nelement_local"=>nelement_local_vpa, "L"=>Lvpa,
                               "discretization"=>discretization, "bc"=>bc_vpa,
                               "element_spacing_option"=>element_spacing_option),
        )

        # Set up MPI
        initialize_comms!()
        setup_distributed_memory_MPI(1,1,1,1)
        vperp, vperp_spectral = define_coordinate(coords_input, "vperp")
        vpa, vpa_spectral = define_coordinate(coords_input, "vpa")
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=1, sn=1,
                                       r=1, z=1, vperp=vperp.n, vpa=vpa.n,
                                       vzeta=1, vr=1, vz=1)

        return vpa, vpa_spectral, vperp, vperp_spectral
end

# test of preconditioner matrix for nonlinear implicit solve.
# We use the preconditioner matrix for a time advance of
# dF/dt = C[F,F_M], with F_M a fixed Maxwellian distribution.
# We test that the result F is close to F_M.
function backward_Euler_linearised_collisions_test(;      
                # grid and physics parameters
                ngrid = 5,
                nelement_vpa = 16,
                nelement_vperp = 8,
                bc_vpa="none",
                bc_vperp="none",
                ms = 1.0,
                delta_t = 1.0,
                nuss = 1.0,
                ntime = 100,
                # background Maxwellian
                dens = 1.0,
                upar = 0.0,
                vth = 1.0,
                # initial beam parameters
                vpa0 = 1.0,
                vperp0 = 1.0,
                vth0 = 0.5,
                # options
                boundary_data_option = multipole_expansion,
                use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=true,
                print_to_screen=false,
                # error tolerances
                atol_max = 2.0e-5,
                atol_L2 = 2.0e-6,
                atol_dens = 1.0e-8,
                atol_upar = 1.0e-10,
                atol_vth = 1.0e-7)

    # initialise arrays
    vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                Lvpa=10.0,Lvperp=5.0,
                                                                bc_vperp=bc_vperp,bc_vpa=bc_vpa)
    fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
        precompute_weights=false, print_to_screen=print_to_screen)
    dummy_array = allocate_float(vpa.n,vperp.n)
    FMaxwell = allocate_shared_float(vpa.n,vperp.n)
    FMaxwell_err = allocate_shared_float(vpa.n,vperp.n)
    # make sure to use anyv communicator for any array that is modified in fokker_planck.jl functions
    pdf = allocate_shared_float(vpa.n,vperp.n; comm=comm_anyv_subblock[])
    @begin_serial_region()
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            FMaxwell[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
    end
    @begin_s_r_z_anyv_region()
    @begin_anyv_region()
    @anyv_serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            pdf[ivpa,ivperp] = exp(-((vpa.grid[ivpa]-vpa0)^2 + (vperp.grid[ivperp]-vperp0)^2)/(vth0^2))
        end
        # normalise to unit density
        @views densfac = get_density(pdf,vpa,vperp)
        @loop_vperp_vpa ivperp ivpa begin
            pdf[ivpa,ivperp] /= densfac
        end
    end
    @_anyv_subblock_synchronize()
    # calculate the linearised advance matrix 
    calculate_test_particle_preconditioner!(FMaxwell,delta_t,ms,ms,nuss,
        vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays, 
        use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
        boundary_data_option=boundary_data_option)
    for it in 1:ntime
        advance_linearised_test_particle_collisions!(pdf,fkpl_arrays,
                        vpa,vperp,vpa_spectral,vperp_spectral)
    end
    # now check distribution
    test_F_Maxwellian(FMaxwell,pdf,
            vpa,vperp,
            FMaxwell_err,dummy_array, 
            dens, upar, vth, ms,
            atol_max, atol_L2,
            atol_dens, atol_upar, atol_vth, 
            print_to_screen=print_to_screen)
    finalize_comms!()
    return nothing
end

function test_F_Maxwellian(pdf_Maxwell,pdf,
    vpa,vperp,
    dummy_array_1,dummy_array_2, 
    dens, upar, vth, mass,
    atol_max, atol_L2,
    atol_dens, atol_upar, atol_vth; 
    print_to_screen=false)
    @begin_serial_region()
    @serial_region begin
        F_M_max, F_M_L2 = print_test_data(pdf_Maxwell,pdf,dummy_array_1,"pdf",
          vpa,vperp,dummy_array_2,print_to_screen=print_to_screen)
        dens_num = get_density(pdf,vpa,vperp)
        upar_num = get_upar(pdf,vpa,vperp,dens)
        ppar = get_ppar(pdf,vpa,vperp,upar)
        pperp = get_pperp(pdf,vpa,vperp)
        pres = get_pressure(ppar,pperp) 
        vth_num = sqrt(2.0*pres/(dens_num*mass))
        @test F_M_max < atol_max
        @test F_M_L2 < atol_L2
        @test abs(dens_num - dens) < atol_dens
        @test abs(upar_num - upar) < atol_upar
        @test abs(vth_num - vth) < atol_vth
    end
    return nothing
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
        if vpa.bc == "zero"
            println("test vpa bc: F[1, :]", pdf[1, :])
            println("test vpa bc: F[end, :]", pdf[end, :])
        end
        if vperp.bc == "zero"
            println("test vperp bc: F[:, end]", pdf[:, end])
        end
    end
    return nothing
end

# Test of implementation of backward Euler solve of d F / d t = C[F, F]
# i.e., we solve F^n+1 - F^n = delta_t * C[ F^n+1, F^n+1 ]
# using a Newton-Krylov root-finding method. This test function
# can be used to check the performance of the solver at a single
# velocity space point. We initialise with a beam distribution
# ~ exp ( - ((vpa - vpa0)^2 + (vperp - vperp0)^2) / vth0^2 )
# and timestep for a fixed timestep delta_t to a maximum time
# ntime * delta_t. Errors between F and F_Maxwellian can be printed to screen.
# Different algorithm options can be checked.
function backward_Euler_fokker_planck_self_collisions_test(; 
    # initial beam parameters 
    vth0=0.5,
    vperp0=1.0,
    vpa0=1.0,
    # grid parameters
    ngrid=5,
    nelement_vpa=16,
    nelement_vperp=8,
    Lvpa=10.0,
    Lvperp=5.0,
    bc_vpa="none",
    bc_vperp="none",
    # timestepping parameters
    ntime=100,
    delta_t=1.0,
    # options
    test_particle_preconditioner=true,
    test_linearised_advance=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    test_dense_construction=false,
    test_numerical_conserving_terms=true,
    boundary_data_option=multipole_expansion,
    print_to_screen=true,
    # error tolerances
    atol_max = 2.0e-5,
    atol_L2 = 2.0e-6,
    atol_dens = 1.0e-8,
    atol_upar = 5.0e-9,
    atol_vth = 1.0e-7)
    
    vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=Lvpa,Lvperp=Lvperp,bc_vpa=bc_vpa,bc_vperp=bc_vperp)
    if vperp.bc == "zero-impose-regularity"
        error("vperp.bc = $(vperp.bc) not supported for implicit FP")
    end
    @begin_serial_region()
    if boundary_data_option == direct_integration
        precompute_weights = true
    else
        precompute_weights = false
    end
    fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; 
                        precompute_weights=precompute_weights, test_dense_matrix_construction=test_dense_construction,
                        print_to_screen=print_to_screen)
    
    # initial condition
    Fold = allocate_shared_float(vpa.n,vperp.n)
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            Fold[ivpa,ivperp] = exp(-((vpa.grid[ivpa]-vpa0)^2 + (vperp.grid[ivperp]-vperp0)^2)/(vth0^2))
        end
        if vpa.bc == "zero"
            @loop_vperp ivperp begin
                Fold[1,ivperp] = 0.0
                Fold[end,ivperp] = 0.0
            end
        end
        if vperp.bc == "zero"
            @loop_vpa ivpa begin
                Fold[ivpa,end] = 0.0
            end
        end
        # normalise to unit density
        @views densfac = get_density(Fold[:,:],vpa,vperp)
        @loop_vperp_vpa ivperp ivpa begin
            Fold[ivpa,ivperp] /= densfac
        end
    end
    # dummy arrays
    Fdummy1 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy2 = allocate_shared_float(vpa.n,vperp.n)
    Fdummy3 = allocate_shared_float(vpa.n,vperp.n)
    FMaxwell = allocate_shared_float(vpa.n,vperp.n)
    # physics parameters
    ms = 1.0
    nuss = 1.0
    
    # initial condition 
    time = 0.0
    # Maxwellian and parameters
    dens = get_density(Fold,vpa,vperp)
    upar = get_upar(Fold,vpa,vperp,dens)
    ppar = get_ppar(Fold,vpa,vperp,upar)
    pperp = get_pperp(Fold,vpa,vperp)
    pres = get_pressure(ppar,pperp) 
    vth = sqrt(2.0*pres/(dens*ms))
    @serial_region begin
        @loop_vperp_vpa ivperp ivpa begin
            FMaxwell[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
    end

    if print_to_screen
        diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,0)
    end
    implicit_ion_fp_collisions = true
    coords = (vperp=vperp,vpa=vpa)
    spectral = (vperp_spectral=vperp_spectral, vpa_spectral=vpa_spectral)
    fkpl = setup_fkpl_collisions_input(OptionsDict(), true)
    nl_solver_params = setup_fp_nl_solve(implicit_ion_fp_collisions,
                                        OptionsDict(), coords)

    for it in 1:ntime
        @begin_s_r_z_anyv_region()
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
        if print_to_screen
            diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,it)
        end
    end
    
    # now check distribution
    test_F_Maxwellian(FMaxwell,Fold,
            vpa,vperp,
            Fdummy2,Fdummy3, 
            dens, upar, vth, ms,
            atol_max, atol_L2,
            atol_dens, atol_upar, atol_vth, 
            print_to_screen=print_to_screen)
    finalize_comms!()
    return nothing
end

function runtests()
    print_to_screen = false
    @testset "Fokker Planck tests" verbose=use_verbose begin
        println("Fokker Planck tests")
        
        @testset "backward-Euler nonlinear Fokker-Planck collisions" begin
            println("    - test backward-Euler nonlinear Fokker-Planck collisions")
            @testset "$bc" for bc in ("none", "zero")  
                println("        -  bc=$bc")
                # here test that a Maxwellian initial condition remains Maxwellian,
                # i.e., we check the numerical Maxwellian is close to the analytical one.
                # This is faster and more stable than doing a relaxation from vperp0 /= 0.
                backward_Euler_fokker_planck_self_collisions_test(bc_vperp=bc, bc_vpa=bc,
                   ntime = 10, delta_t = 0.1,
                   vth0 = 1.0, vpa0 = 1.0, vperp0 = 0.0, 
                   print_to_screen=print_to_screen)
            end
        end

        @testset "Lagrange-polynomial 2D interpolation" begin
            println("    - test Lagrange-polynomial 2D interpolation")
            ngrid = 9
            nelement_vpa = 16
            nelement_vperp = 8
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=8.0,Lvperp=4.0)

            # electron pdf on electron grids
            Fe = allocate_shared_float(vpa.n,vperp.n)
            # electron pdf on ion normalised grids
            Fe_interp_ion_units = allocate_shared_float(vpa.n,vperp.n)
            # exact value for comparison
            Fe_exact_ion_units = allocate_shared_float(vpa.n,vperp.n)
            # ion pdf on ion grids
            Fi = allocate_shared_float(vpa.n,vperp.n)
            # ion pdf on electron normalised grids
            Fi_interp_electron_units = allocate_shared_float(vpa.n,vperp.n)
            # exact value for comparison
            Fi_exact_electron_units = allocate_shared_float(vpa.n,vperp.n)
            # test array
            F_err = allocate_float(vpa.n,vperp.n)

            dense = 1.0
            upare = 0.0 # upare in electron reference units
            vthe = 1.0 # vthe in electron reference units
            densi = 1.0
            upari = 0.0 # upari in ion reference units
            vthi = 1.0 # vthi in ion reference units
            # reference speeds for electrons and ions
            cref_electron = 60.0
            cref_ion = 1.0
            # scale factor for change of reference speed
            scalefac = cref_ion/cref_electron

            @begin_serial_region()
            @serial_region begin
                @loop_vperp_vpa ivperp ivpa begin
                    Fe[ivpa,ivperp] = F_Maxwellian(dense,upare,vthe,vpa,vperp,ivpa,ivperp)
                    Fe_exact_ion_units[ivpa,ivperp] = F_Maxwellian(dense,upare/scalefac,vthe/scalefac,vpa,vperp,ivpa,ivperp)/(scalefac^3)
                    Fi[ivpa,ivperp] = F_Maxwellian(densi,upari,vthi,vpa,vperp,ivpa,ivperp)
                    Fi_exact_electron_units[ivpa,ivperp] = (scalefac^3)*F_Maxwellian(densi,upari*scalefac,vthi*scalefac,vpa,vperp,ivpa,ivperp)
                end
            end

            @begin_s_r_z_anyv_region()
            interpolate_2D_vspace!(Fe_interp_ion_units,Fe,vpa,vperp,scalefac)
            #println("Fe",Fe)
            #println("Fe interp",Fe_interp_ion_units)
            #println("Fe exact",Fe_exact_ion_units)
            interpolate_2D_vspace!(Fi_interp_electron_units,Fi,vpa,vperp,1.0/scalefac)
            #println("Fi",Fi)
            #println("Fi interp", Fi_interp_electron_units)
            #println("Fi exact",Fi_exact_electron_units)

            @begin_serial_region()
            # check the result
            @serial_region begin
                # for electron data on ion grids
                @. F_err = abs(Fe_interp_ion_units - Fe_exact_ion_units)
                max_F_err = maximum(F_err)
                max_F = maximum(Fe_exact_ion_units)
                #println(max_F)
                @test max_F_err < 3.0e-8 * max_F
                # for ion data on electron grids
                @. F_err = abs(Fi_interp_electron_units - Fi_exact_electron_units)
                max_F_err = maximum(F_err)
                max_F = maximum(Fi_exact_electron_units)
                #println(max_F)
                @test max_F_err < 3.0e-8 * max_F
            end

        end

        @testset "weak-form 2D differentiation" begin
        # tests the correct definition of mass and stiffness matrices in 2D
            println("    - test weak-form 2D differentiation")

            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=2.0,Lvperp=1.0)
            nc_global = vpa.n*vperp.n
            @begin_serial_region()
            fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
                                                                  precompute_weights=false, print_to_screen=print_to_screen)
            KKpar2D_with_BC_terms_sparse = fkpl_arrays.KKpar2D_with_BC_terms_sparse
            KKperp2D_with_BC_terms_sparse = fkpl_arrays.KKperp2D_with_BC_terms_sparse
            lu_obj_MM = fkpl_arrays.lu_obj_MM

            dummy_array = allocate_float(vpa.n,vperp.n)
            fvpavperp = allocate_float(vpa.n,vperp.n)
            fvpavperp_test = allocate_float(vpa.n,vperp.n)
            fvpavperp_err = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvpa2_exact = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvpa2_err = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvpa2_num = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvperp2_exact = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvperp2_err = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvperp2_num = allocate_float(vpa.n,vperp.n)
            dfc = allocate_float(nc_global)
            dgc = allocate_float(nc_global)
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    fvpavperp[ivpa,ivperp] = exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                    d2fvpavperp_dvpa2_exact[ivpa,ivperp] = (4.0*vpa.grid[ivpa]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                    d2fvpavperp_dvperp2_exact[ivpa,ivperp] = (4.0*vperp.grid[ivperp]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                end
            end

            # Make 1d views
            fc = vec(fvpavperp)
            d2fc_dvpa2 = vec(d2fvpavperp_dvpa2_num)
            d2fc_dvperp2 = vec(d2fvpavperp_dvperp2_num)

            #print_vector(fc,"fc",nc_global)
            # multiply by KKpar2D and fill dfc
            mul!(dfc,KKpar2D_with_BC_terms_sparse,fc)
            mul!(dgc,KKperp2D_with_BC_terms_sparse,fc)
            # invert mass matrix
            ldiv!(d2fc_dvpa2, lu_obj_MM, dfc)
            ldiv!(d2fc_dvperp2, lu_obj_MM, dgc)
            #print_vector(fc,"fc",nc_global)
            @serial_region begin
                d2fvpavperp_dvpa2_max, d2fvpavperp_dvpa2_L2 = print_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fdvpa2",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                @test d2fvpavperp_dvpa2_max < 1.0e-7
                @test d2fvpavperp_dvpa2_L2 < 1.0e-8
                d2fvpavperp_dvperp2_max, d2fvpavperp_dvperp2_L2 = print_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fdvperp2",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                @test d2fvpavperp_dvperp2_max < 1.0e-7
                @test d2fvpavperp_dvperp2_L2 < 1.0e-8
                #if plot_test_output
                #    plot_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2",vpa,vperp)
                #    plot_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fvpavperp_dvperp2",vpa,vperp)
                #end
            end
            finalize_comms!()
        end

        @testset "weak-form Rosenbluth potential calculation: elliptic solve" begin
            println("    - test weak-form Rosenbluth potential calculation: elliptic solve")
            @testset "$boundary_data_option" for boundary_data_option in (direct_integration,multipole_expansion,delta_f_multipole)
                println("        -  boundary_data_option=$boundary_data_option")
                ngrid = 9
                nelement_vpa = 8
                nelement_vperp = 4
                vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                            Lvpa=12.0,Lvperp=6.0)
                @begin_serial_region()
                if boundary_data_option == direct_integration
                    precompute_weights = true
                else
                    precompute_weights = false
                end
                fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
                                                                      precompute_weights=precompute_weights,
                                                                      print_to_screen=print_to_screen)
                dummy_array = allocate_float(vpa.n,vperp.n)
                F_M = allocate_float(vpa.n,vperp.n)
                H_M_exact = allocate_float(vpa.n,vperp.n)
                H_M_num = allocate_shared_float(vpa.n,vperp.n)
                H_M_err = allocate_float(vpa.n,vperp.n)
                G_M_exact = allocate_float(vpa.n,vperp.n)
                G_M_num = allocate_shared_float(vpa.n,vperp.n)
                G_M_err = allocate_float(vpa.n,vperp.n)
                d2Gdvpa2_M_exact = allocate_float(vpa.n,vperp.n)
                d2Gdvpa2_M_num = allocate_shared_float(vpa.n,vperp.n)
                d2Gdvpa2_M_err = allocate_float(vpa.n,vperp.n)
                d2Gdvperp2_M_exact = allocate_float(vpa.n,vperp.n)
                d2Gdvperp2_M_num = allocate_shared_float(vpa.n,vperp.n)
                d2Gdvperp2_M_err = allocate_float(vpa.n,vperp.n)
                dGdvperp_M_exact = allocate_float(vpa.n,vperp.n)
                dGdvperp_M_num = allocate_shared_float(vpa.n,vperp.n)
                dGdvperp_M_err = allocate_float(vpa.n,vperp.n)
                d2Gdvperpdvpa_M_exact = allocate_float(vpa.n,vperp.n)
                d2Gdvperpdvpa_M_num = allocate_shared_float(vpa.n,vperp.n)
                d2Gdvperpdvpa_M_err = allocate_float(vpa.n,vperp.n)
                dHdvpa_M_exact = allocate_float(vpa.n,vperp.n)
                dHdvpa_M_num = allocate_shared_float(vpa.n,vperp.n)
                dHdvpa_M_err = allocate_float(vpa.n,vperp.n)
                dHdvperp_M_exact = allocate_float(vpa.n,vperp.n)
                dHdvperp_M_num = allocate_shared_float(vpa.n,vperp.n)
                dHdvperp_M_err = allocate_float(vpa.n,vperp.n)

                dens, upar, vth = 1.0, 1.0, 1.0
                @begin_serial_region()
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    end
                end
                rpbd_exact = allocate_rosenbluth_potential_boundary_data(vpa,vperp)
                # use known test function to provide exact data
                @begin_s_r_z_anyv_region()
                calculate_rosenbluth_potential_boundary_data_exact!(rpbd_exact,
                      H_M_exact,dHdvpa_M_exact,dHdvperp_M_exact,G_M_exact,
                      dGdvperp_M_exact,d2Gdvperp2_M_exact,
                      d2Gdvperpdvpa_M_exact,d2Gdvpa2_M_exact,vpa,vperp)
                # calculate the potentials numerically
                calculate_rosenbluth_potentials_via_elliptic_solve!(
                     fkpl_arrays.GG, fkpl_arrays.HH, fkpl_arrays.dHdvpa, fkpl_arrays.dHdvperp,
                     fkpl_arrays.d2Gdvpa2, fkpl_arrays.dGdvperp, fkpl_arrays.d2Gdvperpdvpa,
                     fkpl_arrays.d2Gdvperp2, F_M, vpa, vperp, vpa_spectral, vperp_spectral,
                     fkpl_arrays; algebraic_solve_for_d2Gdvperp2=false,
                     calculate_GG=true, calculate_dGdvperp=true,
                     boundary_data_option=boundary_data_option)
                # extract C[Fs,Fs'] result
                # and Rosenbluth potentials for testing
                @begin_s_r_z_anyv_region()
                @begin_anyv_vperp_vpa_region()
                @loop_vperp_vpa ivperp ivpa begin
                    G_M_num[ivpa,ivperp] = fkpl_arrays.GG[ivpa,ivperp]
                    H_M_num[ivpa,ivperp] = fkpl_arrays.HH[ivpa,ivperp]
                    dHdvpa_M_num[ivpa,ivperp] = fkpl_arrays.dHdvpa[ivpa,ivperp]
                    dHdvperp_M_num[ivpa,ivperp] = fkpl_arrays.dHdvperp[ivpa,ivperp]
                    dGdvperp_M_num[ivpa,ivperp] = fkpl_arrays.dGdvperp[ivpa,ivperp]
                    d2Gdvperp2_M_num[ivpa,ivperp] = fkpl_arrays.d2Gdvperp2[ivpa,ivperp]
                    d2Gdvpa2_M_num[ivpa,ivperp] = fkpl_arrays.d2Gdvpa2[ivpa,ivperp]
                    d2Gdvperpdvpa_M_num[ivpa,ivperp] = fkpl_arrays.d2Gdvperpdvpa[ivpa,ivperp]
                end
                @begin_serial_region()
                @serial_region begin
                    # test the boundary data
                    max_H_boundary_data_err, max_dHdvpa_boundary_data_err,
                    max_dHdvperp_boundary_data_err, max_G_boundary_data_err,
                    max_dGdvperp_boundary_data_err, max_d2Gdvperp2_boundary_data_err,
                    max_d2Gdvperpdvpa_boundary_data_err, max_d2Gdvpa2_boundary_data_err = test_rosenbluth_potential_boundary_data(fkpl_arrays.rpbd,rpbd_exact,vpa,vperp,print_to_screen=print_to_screen)
                    if boundary_data_option==multipole_expansion
                        atol_max_H = 5.0e-8
                        atol_max_dHdvpa = 5.0e-8
                        atol_max_dHdvperp = 5.0e-8
                        atol_max_G = 5.0e-7
                        atol_max_dGdvperp = 5.0e-7
                        atol_max_d2Gdvperp2 = 5.0e-7
                        atol_max_d2Gdvperpdvpa = 5.0e-7
                        atol_max_d2Gdvpap2 = 1.0e-6
                    else
                        atol_max_H = 2.0e-12
                        atol_max_dHdvpa = 2.0e-11
                        atol_max_dHdvperp = 6.0e-9
                        atol_max_G = 1.0e-11
                        atol_max_dGdvperp = 2.0e-7
                        atol_max_d2Gdvperp2 = 5.0e-8
                        atol_max_d2Gdvperpdvpa = 2.0e-8
                        atol_max_d2Gdvpap2 = 1.0e-11
                    end
                    @test max_H_boundary_data_err < atol_max_H
                    @test max_dHdvpa_boundary_data_err < atol_max_dHdvpa
                    @test max_dHdvperp_boundary_data_err < atol_max_dHdvperp
                    @test max_G_boundary_data_err < atol_max_G
                    @test max_dGdvperp_boundary_data_err < atol_max_dGdvperp
                    @test max_d2Gdvperp2_boundary_data_err < atol_max_d2Gdvperp2
                    @test max_d2Gdvperpdvpa_boundary_data_err < atol_max_d2Gdvperpdvpa
                    @test max_d2Gdvpa2_boundary_data_err < atol_max_d2Gdvpap2
                    # test the elliptic solvers
                    H_M_max, H_M_L2 = print_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    dHdvpa_M_max, dHdvpa_M_L2 = print_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    dHdvperp_M_max, dHdvperp_M_L2 = print_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    G_M_max, G_M_L2 = print_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    d2Gdvpa2_M_max, d2Gdvpa2_M_L2 = print_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    dGdvperp_M_max, dGdvperp_M_L2 = print_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    d2Gdvperpdvpa_M_max, d2Gdvperpdvpa_M_L2 = print_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    d2Gdvperp2_M_max, d2Gdvperp2_M_L2 = print_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    if boundary_data_option==multipole_expansion
                        atol_max_H = 2.0e-7
                        atol_L2_H = 5.0e-9
                        atol_max_dHdvpa = 2.0e-6
                        atol_L2_dHdvpa = 5.0e-8
                        atol_max_dHdvperp = 2.0e-5
                        atol_L2_dHdvperp = 1.0e-7
                        atol_max_G = 5.0e-7
                        atol_L2_G = 5.0e-8
                        atol_max_d2Gdvpap2 = 1.0e-6
                        atol_L2_d2Gdvpa2 = 5.0e-8
                        atol_max_dGdvperp = 2.0e-6
                        atol_L2_dGdvperp = 2.0e-7
                        atol_max_d2Gdvperpdvpa = 2.0e-6
                        atol_L2_d2Gdvperpdvpa = 5.0e-8
                        atol_max_d2Gdvperp2 = 5.0e-7
                        atol_L2_d2Gdvperp2 = 5.0e-8
                    else
                        atol_max_H = 2.0e-7
                        atol_L2_H = 5.0e-9
                        atol_max_dHdvpa = 2.0e-6
                        atol_L2_dHdvpa = 5.0e-8
                        atol_max_dHdvperp = 2.0e-5
                        atol_L2_dHdvperp = 1.0e-7
                        atol_max_G = 2.0e-8
                        atol_L2_G = 7.0e-10
                        atol_max_d2Gdvpap2 = 2.0e-7
                        atol_L2_d2Gdvpa2 = 4.0e-9
                        atol_max_dGdvperp = 2.0e-6
                        atol_L2_dGdvperp = 2.0e-7
                        atol_max_d2Gdvperpdvpa = 2.0e-6
                        atol_L2_d2Gdvperpdvpa = 2.0e-8
                        atol_max_d2Gdvperp2 = 3.0e-7
                        atol_L2_d2Gdvperp2 = 2.0e-8
                    end
                    @test H_M_max < atol_max_H
                    @test H_M_L2 < atol_L2_H
                    @test dHdvpa_M_max < atol_max_dHdvpa
                    @test dHdvpa_M_L2 < atol_L2_dHdvpa
                    @test dHdvperp_M_max < atol_max_dHdvperp
                    @test dHdvperp_M_L2 < atol_L2_dHdvperp
                    @test G_M_max < atol_max_G
                    @test G_M_L2 < atol_L2_G
                    @test d2Gdvpa2_M_max < atol_max_d2Gdvpap2
                    @test d2Gdvpa2_M_L2 < atol_L2_d2Gdvpa2
                    @test dGdvperp_M_max < atol_max_dGdvperp
                    @test dGdvperp_M_L2 < atol_L2_dGdvperp
                    @test d2Gdvperpdvpa_M_max < atol_max_d2Gdvperpdvpa
                    @test d2Gdvperpdvpa_M_L2 < atol_L2_d2Gdvperpdvpa
                    @test d2Gdvperp2_M_max < atol_max_d2Gdvperp2
                    @test d2Gdvperp2_M_L2 < atol_L2_d2Gdvperp2
                end
                finalize_comms!()
            end
        end

        @testset "weak-form collision operator calculation" begin
            println("    - test weak-form collision operator calculation")
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=12.0,Lvperp=6.0)
            @begin_serial_region()
            fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
                                                                  precompute_weights=true, print_to_screen=print_to_screen)

            @testset "test_self_operator=$test_self_operator test_numerical_conserving_terms=$test_numerical_conserving_terms test_parallelism = $test_parallelism test_dense_construction=$test_dense_construction use_Maxwellian_Rosenbluth_coefficients=$use_Maxwellian_Rosenbluth_coefficients use_Maxwellian_field_particle_distribution=$use_Maxwellian_field_particle_distribution algebraic_solve_for_d2Gdvperp2=$algebraic_solve_for_d2Gdvperp2" for
                    (test_self_operator, test_numerical_conserving_terms, test_parallelism, test_dense_construction,
                     use_Maxwellian_Rosenbluth_coefficients, use_Maxwellian_field_particle_distribution,
                     algebraic_solve_for_d2Gdvperp2) in ((true,false,false,false,false,false,false),(false,false,false,false,false,false,false),
                                                         (true,true,false,false,false,false,false),(true,false,true,false,false,false,false),
                                                         (true,false,false,true,false,false,false),(true,false,false,false,true,false,false),
                                                         (true,false,false,false,false,true,false),(true,false,false,false,false,false,true))

                dummy_array = allocate_float(vpa.n,vperp.n)
                Fs_M = allocate_float(vpa.n,vperp.n)
                F_M = allocate_float(vpa.n,vperp.n)
                C_M_num = allocate_shared_float(vpa.n,vperp.n)
                C_M_exact = allocate_float(vpa.n,vperp.n)
                C_M_err = allocate_float(vpa.n,vperp.n)
                if test_self_operator
                    dens, upar, vth = 1.0, 1.0, 1.0
                    denss, upars, vths = dens, upar, vth
                else
                    denss, upars, vths = 1.0, -1.0, 2.0/3.0
                    dens, upar, vth = 1.0, 1.0, 1.0
                end
                ms = 1.0
                msp = 1.0
                nussp = 1.0
                @begin_serial_region()
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fs_M[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                        F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        C_M_exact[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,
                                                                        dens,upar,vth,msp,
                                                                        nussp,vpa,vperp,ivpa,ivperp)
                    end
                end
                @begin_s_r_z_anyv_region()
                fokker_planck_collision_operator_weak_form!(Fs_M,F_M,ms,msp,nussp,
                                                 fkpl_arrays,
                                                 vperp, vpa, vperp_spectral, vpa_spectral,
                                                 test_assembly_serial=test_parallelism,
                                                 use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
                                                 use_Maxwellian_field_particle_distribution=use_Maxwellian_field_particle_distribution,
                                                 algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                                                 calculate_GG = false, calculate_dGdvperp=false)
                if test_numerical_conserving_terms && test_self_operator
                    # enforce the boundary conditions on CC before it is used for timestepping
                    enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral)
                    # make ad-hoc conserving corrections
                    conserving_corrections!(fkpl_arrays.CC,Fs_M,vpa,vperp,dummy_array)
                end
                # extract C[Fs,Fs'] result
                @begin_s_r_z_anyv_region()
                @begin_anyv_vperp_vpa_region()
                @loop_vperp_vpa ivperp ivpa begin
                    C_M_num[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
                end
                @begin_serial_region()
                @serial_region begin
                    C_M_max, C_M_L2 = print_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    if test_self_operator && !test_numerical_conserving_terms && !use_Maxwellian_Rosenbluth_coefficients && !use_Maxwellian_field_particle_distribution
                        atol_max = 6.0e-4
                        atol_L2 = 7.0e-6
                    elseif test_self_operator && test_numerical_conserving_terms && !use_Maxwellian_Rosenbluth_coefficients && !use_Maxwellian_field_particle_distribution
                        atol_max = 7.0e-4
                        atol_L2 = 7.0e-6
                    elseif test_self_operator && !test_numerical_conserving_terms && use_Maxwellian_Rosenbluth_coefficients && !use_Maxwellian_field_particle_distribution
                        atol_max = 8.0e-4
                        atol_L2 = 8.1e-6
                    elseif test_self_operator && !test_numerical_conserving_terms && !use_Maxwellian_Rosenbluth_coefficients && use_Maxwellian_field_particle_distribution
                        atol_max = 1.1e-3
                        atol_L2 = 9.0e-6
                    else
                        atol_max = 7.0e-2
                        atol_L2 = 6.0e-4
                    end
                    @test C_M_max < atol_max
                    @test C_M_L2 < atol_L2
                    # calculate the entropy production
                    lnfC = fkpl_arrays.rhsvpavperp
                    @loop_vperp_vpa ivperp ivpa begin
                        lnfC[ivpa,ivperp] = Fs_M[ivpa,ivperp]*C_M_num[ivpa,ivperp]
                    end
                    dSdt = - get_density(lnfC,vpa,vperp)
                    if test_self_operator && !test_numerical_conserving_terms
                        if algebraic_solve_for_d2Gdvperp2
                            rtol, atol = 0.0, 1.0e-7
                        else
                            rtol, atol = 0.0, 1.0e-8
                        end
                        @test isapprox(dSdt, rtol ; atol=atol)
                        delta_n = get_density(C_M_num, vpa, vperp)
                        delta_upar = get_upar(C_M_num, vpa, vperp, dens)
                        delta_ppar = msp*get_ppar(C_M_num, vpa, vperp, upar)
                        delta_pperp = msp*get_pperp(C_M_num, vpa, vperp)
                        delta_pressure = get_pressure(delta_ppar,delta_pperp)
                        rtol, atol = 0.0, 1.0e-12
                        @test isapprox(delta_n, rtol ; atol=atol)
                        rtol, atol = 0.0, 1.0e-9
                        @test isapprox(delta_upar, rtol ; atol=atol)
                        if algebraic_solve_for_d2Gdvperp2
                            rtol, atol = 0.0, 1.0e-7
                        else
                            rtol, atol = 0.0, 1.0e-8
                        end
                        @test isapprox(delta_pressure, rtol ; atol=atol)
                        if print_to_screen
                            println("dSdt: $dSdt should be >0.0")
                            println("delta_n: ", delta_n)
                            println("delta_upar: ", delta_upar)
                            println("delta_pressure: ", delta_pressure)
                        end
                    elseif test_self_operator && test_numerical_conserving_terms
                        rtol, atol = 0.0, 6.0e-7
                        @test isapprox(dSdt, rtol ; atol=atol)
                        delta_n = get_density(C_M_num, vpa, vperp)
                        delta_upar = get_upar(C_M_num, vpa, vperp, dens)
                        delta_ppar = msp*get_ppar(C_M_num, vpa, vperp, upar)
                        delta_pperp = msp*get_pperp(C_M_num, vpa, vperp)
                        delta_pressure = get_pressure(delta_ppar,delta_pperp)
                        rtol, atol = 0.0, 1.0e-15
                        @test isapprox(delta_n, rtol ; atol=atol)
                        rtol, atol = 0.0, 1.0e-15
                        @test isapprox(delta_upar, rtol ; atol=atol)
                        rtol, atol = 0.0, 1.0e-15
                        @test isapprox(delta_pressure, rtol ; atol=atol)
                        if print_to_screen
                            println("dSdt: $dSdt should be >0.0")
                            println("delta_n: ", delta_n)
                            println("delta_upar: ", delta_upar)
                            println("delta_pressure: ", delta_pressure)
                        end
                    else
                        atol = 1.0e-4
                        @test isapprox(dSdt, 2.543251178128757 ; atol=atol)
                        delta_n = get_density(C_M_num, vpa, vperp)
                        rtol, atol = 0.0, 1.0e-12
                        @test isapprox(delta_n, rtol ; atol=atol)
                        if print_to_screen
                            println("dSdt: $dSdt")
                            println("delta_n: ", delta_n)
                        end
                    end
                end
            end
            finalize_comms!()
        end

        @testset "weak-form (slowing-down) collision operator calculation" begin
            println("    - test weak-form (slowing-down) collision operator calculation")
            ngrid = 9
            nelement_vpa = 16
            nelement_vperp = 8
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=12.0,Lvperp=6.0)
            @begin_serial_region()
            fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
                                                                  precompute_weights=true, print_to_screen=print_to_screen)

            @testset "slowing_down_test=true test_numerical_conserving_terms=$test_numerical_conserving_terms" for test_numerical_conserving_terms in (true,false)

                dummy_array = allocate_float(vpa.n,vperp.n)
                Fs_M = allocate_float(vpa.n,vperp.n)
                F_M = allocate_float(vpa.n,vperp.n)
                C_M_num = allocate_shared_float(vpa.n,vperp.n)
                C_M_exact = allocate_float(vpa.n,vperp.n)
                C_M_err = allocate_float(vpa.n,vperp.n)

                # pick a set of parameters that represent slowing down
                # on slow ions and faster electrons, but which are close
                # enough to 1 for errors comparable to the self-collision operator
                # increasing or reducing vth, mass increases the errors
                dens, upar, vth = 1.0, 1.0, 1.0
                mref = 1.0
                Zref = 1.0
                msp = [1.0,0.2]#[0.25, 0.25/1836.0]
                Zsp = [0.5,0.5]#[0.5, 0.5]
                denssp = [1.0,1.0]#[1.0, 1.0]
                uparsp = [0.0,0.0]#[0.0, 0.0]
                vthsp = [sqrt(0.5/msp[1]), sqrt(0.5/msp[2])]#[sqrt(0.01/msp[1]), sqrt(0.01/msp[2])]
                nsprime = size(msp,1)
                nuref = 1.0

                @begin_serial_region()
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fs_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        C_M_exact[ivpa,ivperp] = 0.0
                    end
                end
                # sum up contributions to cross-collision operator
                for isp in 1:nsprime
                    zfac = (Zsp[isp]/Zref)^2
                    nussp = nuref*zfac
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                            C_M_exact[ivpa,ivperp] += Cssp_Maxwellian_inputs(dens,upar,vth,mref,
                                                                            denssp[isp],uparsp[isp],vthsp[isp],msp[isp],
                                                                            nussp,vpa,vperp,ivpa,ivperp)
                        end
                    end
                end
                @begin_s_r_z_anyv_region()
                @views fokker_planck_collision_operator_weak_form_Maxwellian_Fsp!(Fs_M[:,:],
                                     nuref,mref,Zref,msp,Zsp,denssp,uparsp,vthsp,
                                     fkpl_arrays,vperp,vpa,vperp_spectral,vpa_spectral)
                if test_numerical_conserving_terms
                    # enforce the boundary conditions on CC before it is used for timestepping
                    enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral)
                    # make ad-hoc conserving corrections
                    density_conserving_correction!(fkpl_arrays.CC,Fs_M,vpa,vperp,dummy_array)
                end
                # extract C[Fs,Fs'] result
                @begin_s_r_z_anyv_region()
                @begin_anyv_vperp_vpa_region()
                @loop_vperp_vpa ivperp ivpa begin
                    C_M_num[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
                end
                @begin_serial_region()
                @serial_region begin
                    C_M_max, C_M_L2 = print_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                    atol_max = 7.0e-2
                    atol_L2 = 6.0e-4
                    @test C_M_max < atol_max
                    @test C_M_L2 < atol_L2
                    if !test_numerical_conserving_terms
                        delta_n = get_density(C_M_num, vpa, vperp)
                        rtol, atol = 0.0, 1.0e-12
                        @test isapprox(delta_n, rtol ; atol=atol)
                        if print_to_screen
                            println("delta_n: ", delta_n)
                        end
                    elseif test_numerical_conserving_terms
                        delta_n = get_density(C_M_num, vpa, vperp)
                        rtol, atol = 0.0, 1.0e-15
                        @test isapprox(delta_n, rtol ; atol=atol)
                        if print_to_screen
                            println("delta_n: ", delta_n)
                        end
                    end
                end
            end
            finalize_comms!()
        end

        @testset "weak-form Rosenbluth potential calculation: direct integration" begin
            println("    - test weak-form Rosenbluth potential calculation: direct integration")
            ngrid = 5 # chosen for a quick test -- direct integration is slow!
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=12.0,Lvperp=6.0)
            @begin_serial_region()
            fkpl_arrays = init_fokker_planck_collisions_direct_integration(vperp,vpa,precompute_weights=true,print_to_screen=print_to_screen)
            dummy_array = allocate_float(vpa.n,vperp.n)
            F_M = allocate_float(vpa.n,vperp.n)
            H_M_exact = allocate_float(vpa.n,vperp.n)
            H_M_num = allocate_shared_float(vpa.n,vperp.n)
            H_M_err = allocate_float(vpa.n,vperp.n)
            G_M_exact = allocate_float(vpa.n,vperp.n)
            G_M_num = allocate_shared_float(vpa.n,vperp.n)
            G_M_err = allocate_float(vpa.n,vperp.n)
            d2Gdvpa2_M_exact = allocate_float(vpa.n,vperp.n)
            d2Gdvpa2_M_num = allocate_shared_float(vpa.n,vperp.n)
            d2Gdvpa2_M_err = allocate_float(vpa.n,vperp.n)
            d2Gdvperp2_M_exact = allocate_float(vpa.n,vperp.n)
            d2Gdvperp2_M_num = allocate_shared_float(vpa.n,vperp.n)
            d2Gdvperp2_M_err = allocate_float(vpa.n,vperp.n)
            dGdvperp_M_exact = allocate_float(vpa.n,vperp.n)
            dGdvperp_M_num = allocate_shared_float(vpa.n,vperp.n)
            dGdvperp_M_err = allocate_float(vpa.n,vperp.n)
            d2Gdvperpdvpa_M_exact = allocate_float(vpa.n,vperp.n)
            d2Gdvperpdvpa_M_num = allocate_shared_float(vpa.n,vperp.n)
            d2Gdvperpdvpa_M_err = allocate_float(vpa.n,vperp.n)
            dHdvpa_M_exact = allocate_float(vpa.n,vperp.n)
            dHdvpa_M_num = allocate_shared_float(vpa.n,vperp.n)
            dHdvpa_M_err = allocate_float(vpa.n,vperp.n)
            dHdvperp_M_exact = allocate_float(vpa.n,vperp.n)
            dHdvperp_M_num = allocate_shared_float(vpa.n,vperp.n)
            dHdvperp_M_err = allocate_float(vpa.n,vperp.n)

            dens, upar, vth = 1.0, 1.0, 1.0
            @begin_serial_region()
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                end
            end
            # calculate the potentials numerically
            @begin_s_r_z_anyv_region()
            calculate_rosenbluth_potentials_via_direct_integration!(G_M_num,H_M_num,dHdvpa_M_num,dHdvperp_M_num,
             d2Gdvpa2_M_num,dGdvperp_M_num,d2Gdvperpdvpa_M_num,d2Gdvperp2_M_num,F_M,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays)
            @begin_serial_region()
            @serial_region begin
                # test the integration
                # to recalculate absolute tolerances atol, set print_to_screen = true
                H_M_max, H_M_L2 = print_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dHdvpa_M_max, dHdvpa_M_L2 = print_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dHdvperp_M_max, dHdvperp_M_L2 = print_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                G_M_max, G_M_L2 = print_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvpa2_M_max, d2Gdvpa2_M_L2 = print_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dGdvperp_M_max, dGdvperp_M_L2 = print_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvperpdvpa_M_max, d2Gdvperpdvpa_M_L2 = print_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvperp2_M_max, d2Gdvperp2_M_L2 = print_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                atol_max = 2.1e-4
                atol_L2 = 6.5e-6
                @test H_M_max < atol_max
                @test H_M_L2 < atol_L2
                atol_max = 1.5e-3
                atol_L2 = 6.5e-5
                @test dHdvpa_M_max < atol_max
                @test dHdvpa_M_L2 < atol_L2
                atol_max = 8.0e-4
                atol_L2 = 4.0e-5
                @test dHdvperp_M_max < atol_max
                @test dHdvperp_M_L2 < atol_L2
                atol_max = 1.1e-4
                atol_L2 = 4.0e-5
                @test G_M_max < atol_max
                @test G_M_L2 < atol_L2
                atol_max = 2.5e-4
                atol_L2 = 1.2e-5
                @test d2Gdvpa2_M_max < atol_max
                @test d2Gdvpa2_M_L2 < atol_L2
                atol_max = 9.0e-5
                atol_L2 = 6.0e-5
                @test dGdvperp_M_max < atol_max
                @test dGdvperp_M_L2 < atol_L2
                atol_max = 1.1e-4
                atol_L2 = 9.0e-6
                @test d2Gdvperpdvpa_M_max < atol_max
                @test d2Gdvperpdvpa_M_L2 < atol_L2
                atol_max = 2.0e-4
                atol_L2 = 1.1e-5
                @test d2Gdvperp2_M_max < atol_max
                @test d2Gdvperp2_M_L2 < atol_L2
            end
            finalize_comms!()
        end
        
        @testset "backward-Euler linearised test particle collisions" begin
            println("    - test backward-Euler linearised test particle collisions")
            @testset "$bc" for bc in ("none", "zero")  
                println("        -  bc=$bc")
                backward_Euler_linearised_collisions_test(bc_vpa=bc,bc_vperp=bc,
                 use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=true)
                backward_Euler_linearised_collisions_test(bc_vpa=bc,bc_vperp=bc,
                 use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
                 atol_vth=3.0e-7)
            end
        end

        
    end
end

end #FokkerPlanckTests

using .FokkerPlanckTests

FokkerPlanckTests.runtests()

