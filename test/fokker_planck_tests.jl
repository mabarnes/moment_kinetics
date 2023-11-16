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

using moment_kinetics.fokker_planck: init_fokker_planck_collisions_weak_form
using moment_kinetics.fokker_planck_test: print_test_data, plot_test_data, fkpl_error_data, allocate_error_data

function create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=12.0,Lvperp=6.0)

        nelement_local_vpa = nelement_vpa # number of elements per rank
        nelement_global_vpa = nelement_local_vpa # total number of elements 
        nelement_local_vperp = nelement_vperp # number of elements per rank
        nelement_global_vperp = nelement_local_vperp # total number of elements 
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
    @testset "Fokker Planck tests" verbose=use_verbose begin
        println("Fokker Planck tests")
        
        @testset "weak-form 2D differentiation test" begin
        # tests the correct definition of mass and stiffness matrices in 2D
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vpa_spectral, vperp, vperp_spectral = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                        Lvpa=2.0,Lvperp=1.0)
            nc_global = vpa.n*vperp.n
            begin_serial_region()
            fkpl_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral,
                                                                  precompute_weights=false)
            KKpar2D_with_BC_terms_sparse = fkpl_arrays.KKpar2D_with_BC_terms_sparse
            KKperp2D_with_BC_terms_sparse = fkpl_arrays.KKperp2D_with_BC_terms_sparse
            lu_obj_MM = fkpl_arrays.lu_obj_MM
            
            dummy_array = Array{mk_float,2}(undef,vpa.n,vperp.n)
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
            
            # fill fc with fvpavperp
            ravel_vpavperp_to_c!(fc,fvpavperp,vpa.n,vperp.n)
            ravel_c_to_vpavperp!(fvpavperp_test,fc,nc_global,vpa.n)
            @. fvpavperp_err = abs(fvpavperp - fvpavperp_test)
            max_ravel_err = maximum(fvpavperp_err)
            @serial_region begin
                println("max(ravel_err)",max_ravel_err)
                @test isapprox(max_ravel_err, 1.0e-15 ; atol = 1.0e-15)
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
                d2fvpavperp_dvpa2_max, d2fvpavperp_dvpa2_L2 = print_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fdvpa2",vpa,vperp,dummy_array)
                @test isapprox(d2fvpavperp_dvpa2_max, 1.0e-7 ; atol=1.0e-7)
                @test isapprox(d2fvpavperp_dvpa2_L2, 1.0e-8 ; atol=1.0e-8)
                d2fvpavperp_dvperp2_max, d2fvpavperp_dvperp2_L2 = print_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fdvperp2",vpa,vperp,dummy_array)
                @test isapprox(d2fvpavperp_dvperp2_max, 1.0e-7 ; atol=1.0e-7)
                @test isapprox(d2fvpavperp_dvperp2_L2, 1.0e-8 ; atol=1.0e-8)
                #if plot_test_output
                #    plot_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2",vpa,vperp)
                #    plot_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fvpavperp_dvperp2",vpa,vperp)
                #end
            end
            finalize_comms!()
        end
        
    end
end

end #FokkerPlanckTests

using .FokkerPlanckTests

FokkerPlanckTests.runtests()

