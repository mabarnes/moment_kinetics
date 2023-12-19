module FokkerPlanckTests

include("setup.jl")


using MPI
using moment_kinetics.fokker_planck_calculus: ravel_c_to_vpavperp!, ravel_vpavperp_to_c!, ravel_c_to_vpavperp_parallel!
using LinearAlgebra: mul!
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure

using moment_kinetics.fokker_planck: init_fokker_planck_collisions_weak_form, fokker_planck_collision_operator_weak_form!
using moment_kinetics.fokker_planck: conserving_corrections!, init_fokker_planck_collisions_direct_integration
using moment_kinetics.fokker_planck_test: print_test_data, plot_test_data, fkpl_error_data, allocate_error_data
using moment_kinetics.fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian
using moment_kinetics.fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using moment_kinetics.fokker_planck_test: dHdvperp_Maxwellian, dHdvpa_Maxwellian, Cssp_Maxwellian_inputs
using moment_kinetics.fokker_planck_calculus: calculate_rosenbluth_potentials_via_elliptic_solve!, calculate_rosenbluth_potential_boundary_data_exact!
using moment_kinetics.fokker_planck_calculus: test_rosenbluth_potential_boundary_data, allocate_rosenbluth_potential_boundary_data
using moment_kinetics.fokker_planck_calculus: enforce_vpavperp_BCs!, calculate_rosenbluth_potentials_via_direct_integration!

function create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=12.0,Lvperp=6.0)

        nelement_local_vpa = nelement_vpa # number of elements per rank
        nelement_global_vpa = nelement_local_vpa # total number of elements 
        nelement_local_vperp = nelement_vperp # number of elements per rank
        nelement_global_vperp = nelement_local_vperp # total number of elements 
        bc = "zero" # used only in derivative! functions 
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
        element_spacing_option = "uniform"
        vpa_input = grid_input("vpa", ngrid, nelement_global_vpa, nelement_local_vpa, 
            nrank, irank, Lvpa, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
        vperp_input = grid_input("vperp", ngrid, nelement_global_vperp, nelement_local_vperp, 
            nrank, irank, Lvperp, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
        # create the coordinate struct 'x'
        #println("made inputs")
        #println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
        #println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
        vpa, vpa_spectral = define_coordinate(vpa_input)
        vperp, vperp_spectral = define_coordinate(vperp_input)
        
        # Set up MPI
        initialize_comms!()
        setup_distributed_memory_MPI(1,1,1,1)
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=1, sn=1,
                                       r=1, z=1, vperp=vperp.n, vpa=vpa.n,
                                       vzeta=1, vr=1, vz=1)
        
        return vpa, vpa_spectral, vperp, vperp_spectral
end

function runtests()
    print_to_screen = false
    @testset "Fokker Planck tests" verbose=use_verbose begin
        println("Fokker Planck tests")
        
        @testset " - test weak-form 2D differentiation" begin
        # tests the correct definition of mass and stiffness matrices in 2D
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=2.0,Lvperp=1.0)
            nc_global = vpa.n*vperp.n
            begin_serial_region()
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
            fc = allocate_float(nc_global)
            dfc = allocate_float(nc_global)
            gc = allocate_float(nc_global)
            dgc = allocate_float(nc_global)
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    fvpavperp[ivpa,ivperp] = exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                    d2fvpavperp_dvpa2_exact[ivpa,ivperp] = (4.0*vpa.grid[ivpa]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                    d2fvpavperp_dvperp2_exact[ivpa,ivperp] = (4.0*vperp.grid[ivperp]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                end
            end
            
            # fill fc with fvpavperp
            ravel_vpavperp_to_c!(fc,fvpavperp,vpa.n,vperp.n)
            ravel_c_to_vpavperp!(fvpavperp_test,fc,nc_global,vpa.n)
            @. fvpavperp_err = abs(fvpavperp - fvpavperp_test)
            max_ravel_err = maximum(fvpavperp_err)
            @serial_region begin
                if print_to_screen 
                    println("max(ravel_err)",max_ravel_err)
                end
                @test max_ravel_err < 1.0e-15
            end
            #print_vector(fc,"fc",nc_global)
            # multiply by KKpar2D and fill dfc
            mul!(dfc,KKpar2D_with_BC_terms_sparse,fc)
            mul!(dgc,KKperp2D_with_BC_terms_sparse,fc)
            # invert mass matrix and fill fc
            fc = lu_obj_MM \ dfc
            gc = lu_obj_MM \ dgc
            #print_vector(fc,"fc",nc_global)
            # unravel
            ravel_c_to_vpavperp!(d2fvpavperp_dvpa2_num,fc,nc_global,vpa.n)
            ravel_c_to_vpavperp!(d2fvpavperp_dvperp2_num,gc,nc_global,vpa.n)
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
        
        @testset " - test weak-form Rosenbluth potential calculation: elliptic solve" begin
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=12.0,Lvperp=6.0)
            nc_global = vpa.n*vperp.n
            begin_serial_region()
            fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
                                                                  precompute_weights=true, print_to_screen=print_to_screen)
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
            begin_serial_region()
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
            calculate_rosenbluth_potential_boundary_data_exact!(rpbd_exact,
                  H_M_exact,dHdvpa_M_exact,dHdvperp_M_exact,G_M_exact,
                  dGdvperp_M_exact,d2Gdvperp2_M_exact,
                  d2Gdvperpdvpa_M_exact,d2Gdvpa2_M_exact,vpa,vperp)
            # calculate the potentials numerically
            calculate_rosenbluth_potentials_via_elliptic_solve!(fkpl_arrays.GG,fkpl_arrays.HH,fkpl_arrays.dHdvpa,fkpl_arrays.dHdvperp,
                 fkpl_arrays.d2Gdvpa2,fkpl_arrays.dGdvperp,fkpl_arrays.d2Gdvperpdvpa,fkpl_arrays.d2Gdvperp2,F_M,
                 vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays;
                 algebraic_solve_for_d2Gdvperp2=false,calculate_GG=true,calculate_dGdvperp=true)
            # extract C[Fs,Fs'] result
            # and Rosenbluth potentials for testing
            begin_vperp_vpa_region()
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
            begin_serial_region()
            @serial_region begin
                # test the boundary data
                max_H_boundary_data_err, max_dHdvpa_boundary_data_err, 
                max_dHdvperp_boundary_data_err, max_G_boundary_data_err,
                max_dGdvperp_boundary_data_err, max_d2Gdvperp2_boundary_data_err, 
                max_d2Gdvperpdvpa_boundary_data_err, max_d2Gdvpa2_boundary_data_err = test_rosenbluth_potential_boundary_data(fkpl_arrays.rpbd,rpbd_exact,vpa,vperp,print_to_screen=print_to_screen)
                atol_max = 2.0e-13
                @test max_H_boundary_data_err < atol_max
                atol_max = 2.0e-12
                @test max_dHdvpa_boundary_data_err < atol_max
                atol_max = 3.0e-9
                @test max_dHdvperp_boundary_data_err < atol_max
                atol_max = 7.0e-12
                @test max_G_boundary_data_err < atol_max
                atol_max = 2.0e-7
                @test max_dGdvperp_boundary_data_err < atol_max
                atol_max = 2.0e-8
                @test max_d2Gdvperp2_boundary_data_err < atol_max
                atol_max = 2.0e-8
                @test max_d2Gdvperpdvpa_boundary_data_err < atol_max
                atol_max = 7.0e-12
                @test max_d2Gdvpa2_boundary_data_err < atol_max
                # test the elliptic solvers
                H_M_max, H_M_L2 = print_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dHdvpa_M_max, dHdvpa_M_L2 = print_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dHdvperp_M_max, dHdvperp_M_L2 = print_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                G_M_max, G_M_L2 = print_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvpa2_M_max, d2Gdvpa2_M_L2 = print_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dGdvperp_M_max, dGdvperp_M_L2 = print_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvperpdvpa_M_max, d2Gdvperpdvpa_M_L2 = print_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvperp2_M_max, d2Gdvperp2_M_L2 = print_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                atol_max = 2.0e-7
                atol_L2 = 5.0e-9
                @test H_M_max < atol_max
                @test H_M_L2 < atol_L2
                atol_max = 2.0e-6
                atol_L2 = 5.0e-8
                @test dHdvpa_M_max < atol_max
                @test dHdvpa_M_L2 < atol_L2
                atol_max = 2.0e-5
                atol_L2 = 1.0e-7
                @test dHdvperp_M_max < atol_max
                @test dHdvperp_M_L2 < atol_L2
                atol_max = 2.0e-8
                atol_L2 = 7.0e-10
                @test G_M_max < atol_max
                @test G_M_L2 < atol_L2
                atol_max = 2.0e-7
                atol_L2 = 4.0e-9
                @test d2Gdvpa2_M_max < atol_max
                @test d2Gdvpa2_M_L2 < atol_L2
                atol_max = 2.0e-6
                atol_L2 = 2.0e-7
                @test dGdvperp_M_max < atol_max
                @test dGdvperp_M_L2 < atol_L2
                atol_max = 2.0e-6
                atol_L2 = 2.0e-8
                @test d2Gdvperpdvpa_M_max < atol_max
                @test d2Gdvperpdvpa_M_L2 < atol_L2
                atol_max = 3.0e-7
                atol_L2 = 2.0e-8
                @test d2Gdvperp2_M_max < atol_max
                @test d2Gdvperp2_M_L2 < atol_L2
            end
            finalize_comms!()                                                                  
        end
        
        @testset " - test weak-form collision operator calculation" begin
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=12.0,Lvperp=6.0)
            nc_global = vpa.n*vperp.n
            begin_serial_region()
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
                begin_serial_region()
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fs_M[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                        F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        C_M_exact[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,
                                                                        dens,upar,vth,msp,
                                                                        nussp,vpa,vperp,ivpa,ivperp)
                    end
                end
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
                begin_vperp_vpa_region()
                @loop_vperp_vpa ivperp ivpa begin
                    C_M_num[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
                end
                begin_serial_region()
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
        
        @testset " - test weak-form Rosenbluth potential calculation: direct integration" begin
            ngrid = 5 # chosen for a quick test -- direct integration is slow!
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=12.0,Lvperp=6.0)
            begin_serial_region()
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
            begin_serial_region()
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
            calculate_rosenbluth_potentials_via_direct_integration!(G_M_num,H_M_num,dHdvpa_M_num,dHdvperp_M_num,
             d2Gdvpa2_M_num,dGdvperp_M_num,d2Gdvperpdvpa_M_num,d2Gdvperp2_M_num,F_M,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays)
            begin_serial_region()
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
        
        
    end
end

end #FokkerPlanckTests

using .FokkerPlanckTests

FokkerPlanckTests.runtests()

