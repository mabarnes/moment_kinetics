export run_assembly_test
using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using Dates
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral, get_QQ_local!
using moment_kinetics.type_definitions: mk_float, mk_int, OptionsDict
using moment_kinetics.fokker_planck: init_fokker_planck_collisions_weak_form
using moment_kinetics.fokker_planck: fokker_planck_collision_operator_weak_form!
using moment_kinetics.fokker_planck: conserving_corrections!
using moment_kinetics.fokker_planck_calculus: enforce_vpavperp_BCs!
using moment_kinetics.fokker_planck_test: F_Maxwellian, print_test_data
using moment_kinetics.calculus: derivative!
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.communication
using moment_kinetics.communication: MPISharedArray
using moment_kinetics.looping
using moment_kinetics.input_structs: direct_integration, multipole_expansion
using moment_kinetics.nonlinear_solvers

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
    begin_serial_region()
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
    begin_serial_region()
    @serial_region begin
        anim = @animate for it in 1:ntime
            @views heatmap(vperp.grid, vpa.grid, pdf[:,:,it], xlabel="vperp", ylabel="vpa", c = :deep, interpolation = :cubic)
        end
        outfile = string("implicit_collisions_pdf_vs_vpa_vperp.gif")
        gif(anim, outfile, fps=5)
    end
end

function test_implicit_collisions(; ngrid=3,nelement_vpa=8,nelement_vperp=4,
    Lvpa=6.0,Lvperp=3.0,ntime=1,delta_t=1.0,
    restart = 8,
    max_restarts = 1,
    atol = 1.0e-10,
    serial_solve = false,
    anyv_region = true,
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
                                "element_spacing_option"=>element_spacing_option),
        "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global_vpa,
                            "nelement_local"=>nelement_local_vpa, "L"=>Lvpa,
                            "discretization"=>discretization,
                            "element_spacing_option"=>element_spacing_option),
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
    begin_serial_region()
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
            fvpavperp[ivpa,ivperp,1] = exp(-vpa.grid[ivpa]^2 - (vperp.grid[ivperp]-1)^2)
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
    
    coords = (vperp=vperp,vpa=vpa)
    nl_solver_params = setup_nonlinear_solve(
        true,
        OptionsDict("nonlinear_solver" =>
                    OptionsDict("rtol" => 0.0,
                                "atol" => atol,
                                "linear_restart" => restart,
                                "linear_max_restarts" => max_restarts,
                                "nonlinear_max_iterations" => 100)),
        coords; serial_solve=serial_solve, anyv_region=anyv_region)
    for it in 1:ntime
        backward_euler_step!(Fnew, Fold, delta_t, ms, msp, nussp, fkpl_arrays, dummy_vpavperp,
            vperp, vpa, vperp_spectral, vpa_spectral, coords,
            Fdummy1, Fdummy2, Fdummy3, Fdummy4, Fdummy5, nl_solver_params,
            test_numerical_conserving_terms=test_numerical_conserving_terms,
            test_self_operator=test_self_operator,
            test_assembly_serial=test_parallelism,
            use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
            use_Maxwellian_field_particle_distribution=use_Maxwellian_field_particle_distribution,
            algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
            calculate_GG = false, calculate_dGdvperp=false,
            boundary_data_option=boundary_data_option)
        begin_serial_region()
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

function backward_euler_step!(Fnew, Fold, delta_t, ms, msp, nussp, fkpl_arrays, dummy_vpavperp,
    vperp, vpa, vperp_spectral, vpa_spectral, coords,
    Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params;
    test_numerical_conserving_terms=false,
    test_self_operator=true,
    test_assembly_serial=false,
    use_Maxwellian_Rosenbluth_coefficients=false,
    use_Maxwellian_field_particle_distribution=false,
    algebraic_solve_for_d2Gdvperp2=false,
    calculate_GG = false, calculate_dGdvperp=false,
    boundary_data_option=multipole_expansion)
    
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
        if test_numerical_conserving_terms && test_self_operator
            # enforce the boundary conditions on CC before it is used for timestepping
            enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral)
            # make ad-hoc conserving corrections
            conserving_corrections!(fkpl_arrays.CC,Fnew,vpa,vperp,dummy_vpavperp)
        end
        begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            Fresidual[ivpa,ivperp] = -Fnew[ivpa,ivperp] + Fold[ivpa,ivperp] + delta_t * fkpl_arrays.CC[ivpa,ivperp]
        end
        return nothing
    end
    begin_s_r_z_anyv_region()
    newton_solve!(Fnew, residual_func!, Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params;
                      coords)
    _anyv_subblock_synchronize()
    begin_serial_region()
end    

    
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
#    println("test_numerical_conserving_terms=true")
    test_implicit_collisions(ngrid=3,nelement_vpa=8,nelement_vperp=4,ntime=50,delta_t=1.0,
      serial_solve=false,anyv_region=true,plot_test_output=false,
      test_numerical_conserving_terms=true)
#    println("test_numerical_conserving_terms=false")
#    test_implicit_collisions(ngrid=5,nelement_vpa=16,nelement_vperp=8,ntime=50,delta_t=1.0,
#      serial_solve=false,anyv_region=true,plot_test_output=false,
#      test_numerical_conserving_terms=false)
end
