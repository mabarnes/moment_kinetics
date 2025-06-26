"""
"""
module time_advance

export setup_time_advance!
export time_advance!
export allocate_advection_structs
export setup_dummy_and_buffer_arrays

using MPI
using OrderedCollections
using Quadmath
using ..type_definitions
using ..array_allocation: allocate_float, allocate_shared_float, allocate_shared_int, allocate_shared_bool
using ..communication
using ..communication: @_block_synchronize
using ..debugging
using ..file_io: write_data_to_ascii, write_all_moments_data_to_binary,
                 write_all_dfns_data_to_binary, setup_electron_io, io_input_struct,
                 setup_dfns_io, write_debug_data_to_binary
using ..initial_conditions: initialize_electrons!
using ..looping
using ..moment_kinetics_structs: scratch_pdf, scratch_electron_pdf
using ..velocity_moments: update_moments!, update_moments_neutral!, reset_moments_status!, update_derived_moments!, update_derived_moments_neutral!
using ..velocity_moments: update_density!, update_upar!, update_ppar!, update_pperp!, update_ion_qpar!, update_vth!, update_derived_ion_moment_time_derivatives!
using ..velocity_moments: update_neutral_density!, update_neutral_qz!
using ..velocity_moments: update_neutral_uzeta!, update_neutral_uz!, update_neutral_ur!
using ..velocity_moments: update_neutral_pzeta!, update_neutral_pz!, update_neutral_pr!, update_derived_neutral_moment_time_derivatives!
using ..velocity_moments: calculate_ion_moment_derivatives!, calculate_neutral_moment_derivatives!
using ..velocity_moments: calculate_electron_moment_derivatives!, update_derived_electron_moment_time_derivatives!
using ..velocity_grid_transforms: vzvrvzeta_to_vpavperp!, vpavperp_to_vzvrvzeta!
using ..boundary_conditions: enforce_boundary_conditions!, get_ion_z_boundary_cutoff_indices
using ..boundary_conditions: enforce_neutral_boundary_conditions!
using ..boundary_conditions: vpagrid_to_dzdt, enforce_v_boundary_condition_local!
using ..input_structs
using ..moment_constraints: hard_force_moment_constraints!,
                            hard_force_moment_constraints_neutral!,
                            moment_constraints_on_residual!
using ..advection: setup_advection
using ..z_advection: update_speed_z!, z_advection!
using ..r_advection: update_speed_r!, r_advection!
using ..neutral_r_advection: update_speed_neutral_r!, neutral_advection_r!
using ..neutral_z_advection: update_speed_neutral_z!, neutral_advection_z!
using ..neutral_vz_advection: update_speed_neutral_vz!, neutral_advection_vz!
using ..vperp_advection: update_speed_vperp!, vperp_advection!
using ..vpa_advection: update_speed_vpa!, vpa_advection!, implicit_vpa_advection!
using ..charge_exchange: ion_charge_exchange_collisions_1V!,
                         neutral_charge_exchange_collisions_1V!,
                         ion_charge_exchange_collisions_3V!,
                         neutral_charge_exchange_collisions_3V!
using ..electron_kinetic_equation: update_electron_pdf!, implicit_electron_advance!,
                                   electron_backward_euler!,
                                   electron_kinetic_equation_euler_update!,
                                   apply_electron_bc_and_constraints_no_r!
using ..electron_vpa_advection: update_electron_speed_vpa!
using ..electron_z_advection: update_electron_speed_z!
using ..ionization: ion_ionization_collisions_1V!, neutral_ionization_collisions_1V!,
                    ion_ionization_collisions_3V!, neutral_ionization_collisions_3V!
using ..krook_collisions: krook_collisions!
using ..maxwell_diffusion: ion_vpa_maxwell_diffusion!, neutral_vz_maxwell_diffusion!
using ..external_sources
using ..nonlinear_solvers
using ..numerical_dissipation: vpa_boundary_buffer_decay!,
                               vpa_boundary_buffer_diffusion!, vpa_dissipation!,
                               z_dissipation!, r_dissipation!, vperp_dissipation!,
                               vz_dissipation_neutral!, z_dissipation_neutral!,
                               r_dissipation_neutral!,
                               vpa_boundary_force_decreasing!, force_minimum_pdf_value!,
                               force_minimum_pdf_value_neutral!
using ..source_terms: source_terms!, source_terms_neutral!, source_terms_manufactured!
using ..continuity: continuity_equation!, neutral_continuity_equation!
using ..force_balance: force_balance!, neutral_force_balance!
using ..energy_equation: energy_equation!, neutral_energy_equation!
using ..em_fields: setup_em_fields, update_phi!
using ..fokker_planck: init_fokker_planck_collisions_weak_form, explicit_fokker_planck_collisions_weak_form!
using ..fokker_planck: explicit_fp_collisions_weak_form_Maxwellian_cross_species!
using ..gyroaverages: init_gyro_operators, gyroaverage_pdf!
using ..manufactured_solns: manufactured_sources
using ..timer_utils
using ..advection: advection_info
using ..runge_kutta: rk_update_evolved_moments!, rk_update_evolved_moments_neutral!,
                     rk_update_variable!, rk_loworder_solution!,
                     setup_runge_kutta_coefficients!, local_error_norm,
                     adaptive_timestep_update_t_params!
using ..utils: to_minutes, get_minimum_CFL_z, get_minimum_CFL_vpa,
               get_minimum_CFL_neutral_z, get_minimum_CFL_neutral_vz
using ..electron_fluid_equations: calculate_electron_moments!
using ..electron_fluid_equations: calculate_electron_density!
using ..electron_fluid_equations: calculate_electron_upar_from_charge_conservation!
using ..electron_fluid_equations: calculate_electron_qpar!, electron_fluid_qpar_boundary_condition!
using ..electron_fluid_equations: calculate_electron_parallel_friction_force!
using ..electron_fluid_equations: electron_energy_equation!, update_electron_vth_temperature!,
                                  electron_braginskii_conduction!,
                                  implicit_braginskii_conduction!
using ..input_structs: braginskii_fluid
using ..derivatives: derivative_z!
@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

using Dates
using ..analysis: steady_state_residuals
#using ..post_processing: draw_v_parallel_zero!

struct scratch_dummy_arrays
    dummy_s::Array{mk_float,1}
    dummy_sr::Array{mk_float,2}
    dummy_vpavperp::Array{mk_float,2}
    dummy_zrs::MPISharedArray{mk_float,3}
    dummy_zrsn::MPISharedArray{mk_float,3}

    #buffer arrays for MPI 
    buffer_z_1::MPISharedArray{mk_float,1}
    buffer_z_2::MPISharedArray{mk_float,1}
    buffer_z_3::MPISharedArray{mk_float,1}
    buffer_z_4::MPISharedArray{mk_float,1}
    buffer_r_1::MPISharedArray{mk_float,1}
    buffer_r_2::MPISharedArray{mk_float,1}
    buffer_r_3::MPISharedArray{mk_float,1}
    buffer_r_4::MPISharedArray{mk_float,1}
    
    buffer_zs_1::MPISharedArray{mk_float,2}
    buffer_zs_2::MPISharedArray{mk_float,2}
    buffer_zs_3::MPISharedArray{mk_float,2}
    buffer_zs_4::MPISharedArray{mk_float,2}
    buffer_zsn_1::MPISharedArray{mk_float,2}
    buffer_zsn_2::MPISharedArray{mk_float,2}
    buffer_zsn_3::MPISharedArray{mk_float,2}
    buffer_zsn_4::MPISharedArray{mk_float,2}

    buffer_rs_1::MPISharedArray{mk_float,2}
    buffer_rs_2::MPISharedArray{mk_float,2}
    buffer_rs_3::MPISharedArray{mk_float,2}
    buffer_rs_4::MPISharedArray{mk_float,2}
    buffer_rs_5::MPISharedArray{mk_float,2}
    buffer_rs_6::MPISharedArray{mk_float,2}
    buffer_rsn_1::MPISharedArray{mk_float,2}
    buffer_rsn_2::MPISharedArray{mk_float,2}
    buffer_rsn_3::MPISharedArray{mk_float,2}
    buffer_rsn_4::MPISharedArray{mk_float,2}
    buffer_rsn_5::MPISharedArray{mk_float,2}
    buffer_rsn_6::MPISharedArray{mk_float,2}

    buffer_zrs_1::MPISharedArray{mk_float,3}
    buffer_zrs_2::MPISharedArray{mk_float,3}
    buffer_zrs_3::MPISharedArray{mk_float,3}

    buffer_vpavperpzs_1::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_2::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_3::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_4::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_5::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_6::MPISharedArray{mk_float,4}

    buffer_vpavperprs_1::MPISharedArray{mk_float,4}
    buffer_vpavperprs_2::MPISharedArray{mk_float,4}
    buffer_vpavperprs_3::MPISharedArray{mk_float,4}
    buffer_vpavperprs_4::MPISharedArray{mk_float,4}
    buffer_vpavperprs_5::MPISharedArray{mk_float,4}
    buffer_vpavperprs_6::MPISharedArray{mk_float,4}

    # buffer to hold derivative after MPI communicates
    # needs to be shared memory
    buffer_vpavperpzrs_1::MPISharedArray{mk_float,5}
    buffer_vpavperpzrs_2::MPISharedArray{mk_float,5}
    # buffers to hold moment quantities for implicit solves
    implicit_buffer_z_1::MPISharedArray{mk_float,1}
    implicit_buffer_z_2::MPISharedArray{mk_float,1}
    implicit_buffer_z_3::MPISharedArray{mk_float,1}
    implicit_buffer_z_4::MPISharedArray{mk_float,1}
    implicit_buffer_z_5::MPISharedArray{mk_float,1}
    implicit_buffer_z_6::MPISharedArray{mk_float,1}
    # buffers to hold electron for implicit solves
    implicit_buffer_vpavperpz_1::MPISharedArray{mk_float,3}
    implicit_buffer_vpavperpz_2::MPISharedArray{mk_float,3}
    implicit_buffer_vpavperpz_3::MPISharedArray{mk_float,3}
    implicit_buffer_vpavperpz_4::MPISharedArray{mk_float,3}
    implicit_buffer_vpavperpz_5::MPISharedArray{mk_float,3}
    implicit_buffer_vpavperpz_6::MPISharedArray{mk_float,3}
    # buffers to hold ion pdf for implicit solves
    implicit_buffer_vpavperpzrs_1::MPISharedArray{mk_float,5}
    implicit_buffer_vpavperpzrs_2::MPISharedArray{mk_float,5}
    implicit_buffer_vpavperpzrs_3::MPISharedArray{mk_float,5}
    implicit_buffer_vpavperpzrs_4::MPISharedArray{mk_float,5}
    implicit_buffer_vpavperpzrs_5::MPISharedArray{mk_float,5}
    implicit_buffer_vpavperpzrs_6::MPISharedArray{mk_float,5}

    buffer_vzvrvzetazsn_1::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_2::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_3::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_4::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_5::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_6::MPISharedArray{mk_float,5}

    buffer_vzvrvzetarsn_1::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_2::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_3::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_4::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_5::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_6::MPISharedArray{mk_float,5}

    # buffer to hold derivative after MPI communicates
    # needs to be shared memory
    buffer_vzvrvzetazrsn_1::MPISharedArray{mk_float,6}
    buffer_vzvrvzetazrsn_2::MPISharedArray{mk_float,6}
    
    buffer_vpavperp_1::MPISharedArray{mk_float,2}
    buffer_vpavperp_2::MPISharedArray{mk_float,2}
    buffer_vpavperp_3::MPISharedArray{mk_float,2}

    buffer_vpavperpzr_1::MPISharedArray{mk_float,4}
    buffer_vpavperpzr_2::MPISharedArray{mk_float,4}
    buffer_vpavperpzr_3::MPISharedArray{mk_float,4}
    buffer_vpavperpzr_4::MPISharedArray{mk_float,4}
    buffer_vpavperpzr_5::MPISharedArray{mk_float,4}
    buffer_vpavperpzr_6::MPISharedArray{mk_float,4}

    buffer_vpavperpr_1::MPISharedArray{mk_float,3}
    buffer_vpavperpr_2::MPISharedArray{mk_float,3}
    buffer_vpavperpr_3::MPISharedArray{mk_float,3}
    buffer_vpavperpr_4::MPISharedArray{mk_float,3}
    buffer_vpavperpr_5::MPISharedArray{mk_float,3}
    buffer_vpavperpr_6::MPISharedArray{mk_float,3}
    int_buffer_rs_1::MPISharedArray{mk_int,2}
    int_buffer_rs_2::MPISharedArray{mk_int,2}
end 

struct advect_object_struct
    vpa_advect::Vector{advection_info{4,5}}
    vperp_advect::Vector{advection_info{4,5}}
    z_advect::Vector{advection_info{4,5}}
    r_advect::Vector{advection_info{4,5}}
    electron_z_advect::Vector{advection_info{4,5}}
    electron_vpa_advect::Vector{advection_info{4,5}}
    neutral_z_advect::Vector{advection_info{5,6}}
    neutral_r_advect::Vector{advection_info{5,6}}
    neutral_vz_advect::Vector{advection_info{5,6}}
end

# consider changing code structure so that
# we can avoid arbitrary types below?
struct spectral_object_struct{Tvz,Tvr,Tvzeta,Tvpa,Tvperp,Tz,Tr}
    vz_spectral::Tvz
    vr_spectral::Tvr
    vzeta_spectral::Tvzeta
    vpa_spectral::Tvpa
    vperp_spectral::Tvperp
    z_spectral::Tz
    r_spectral::Tr
end

function allocate_advection_structs(composition, z, r, vpa, vperp, vz, vr, vzeta)
    # define some local variables for convenience/tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    n_neutral_species_alloc = max(1,composition.n_neutral_species)
    ##                              ##
    # ion particle advection structs #
    ##                              ##
    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the part of the ion kinetic equation dealing
    # with advection in z
    @begin_serial_region()
    z_advect = setup_advection(n_ion_species, z, vpa, vperp, r)
    # create structure r_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the ion kinetic equation dealing
    # with advection in r
    @begin_serial_region()
    r_advect = setup_advection(n_ion_species, r, vpa, vperp, z)
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the ion kinetic equation dealing
    # with advection in vpa
    @begin_serial_region()
    vpa_advect = setup_advection(n_ion_species, vpa, vperp, z, r)
    # create structure vperp_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the ion kinetic equation dealing
    # with advection in vperp
    @begin_serial_region()
    vperp_advect = setup_advection(n_ion_species, vperp, vpa, z, r)
    ##                                   ##
    # electron particle advection structs #
    ##                                   ##
    # create structure electron_z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the part of the electron kinetic equation dealing
    # with advection in z
    @begin_serial_region()
    electron_z_advect = setup_advection(1, z, vpa, vperp, r)
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the part of the electron kinetic equation dealing
    # with advection in vpa
    @begin_serial_region()
    electron_vpa_advect = setup_advection(1, vpa, vperp, z, r)
    ##                                  ##
    # neutral particle advection structs #
    ##                                  ##
    # create structure neutral_z_advect for neutral particle advection
    @begin_serial_region()
    neutral_z_advect = setup_advection(n_neutral_species_alloc, z, vz, vr, vzeta, r)
    # create structure neutral_r_advect for neutral particle advection
    @begin_serial_region()
    neutral_r_advect = setup_advection(n_neutral_species_alloc, r, vz, vr, vzeta, z)
    # create structure neutral_vz_advect for neutral particle advection
    @begin_serial_region()
    neutral_vz_advect = setup_advection(n_neutral_species_alloc, vz, vr, vzeta, z, r)
    ##                                                                 ##
    # construct named list of advection structs to compactify arguments #
    ##                                                                 ##
    advection_structs = advect_object_struct(vpa_advect, vperp_advect, z_advect, r_advect, 
                                             electron_z_advect, electron_vpa_advect,
                                             neutral_z_advect, neutral_r_advect, neutral_vz_advect)
    return advection_structs
end

"""
    setup_time_info(t_input, n_variables, code_time, dt_reload,
                    dt_before_last_fail_reload, composition,
                    manufactured_solns_input, io_input, input_dict; electron=nothing)

Create a [`input_structs.time_info`](@ref) struct using the settings in `t_input`.

If something is passed in `electron`, it is stored in the `electron_t_params` member of
the returned `time_info`.
"""
function setup_time_info(t_input, n_variables, code_time, dt_reload,
                         dt_before_last_fail_reload, composition,
                         manufactured_solns_input, io_input, input_dict; electron=nothing,
                         debug_io=nothing)
    code_time = mk_float(code_time)
    rk_coefs, rk_coefs_implicit, implicit_coefficient_is_zero, n_rk_stages, rk_order,
    adaptive, low_storage, CFL_prefactor =
        setup_runge_kutta_coefficients!(t_input["type"],
                                        mk_float(t_input["CFL_prefactor"]),
                                        t_input["split_operators"])

    if !adaptive
        if electron !== nothing
            # No adaptive timestep, want to use the value from the input file even when we are
            # restarting.
            # Do not want to do this for electrons, because electron_backward_euler!()
            # uses an adaptive timestep (based on nonlinear solver iteration counts) even
            # though it does not use an adaptive RK scheme.
            dt_reload = nothing
        end

        # Makes no sense to use write_error_diagnostics because non-adaptive schemes have
        # no error estimate
        input_dict["timestepping"]["write_error_diagnostics"] = false
    end

    if adaptive && t_input["write_error_diagnostics"] && !t_input["write_after_fixed_step_count"]
        println("WARNING: using adaptive timestepping, so short, random-length timesteps "
                * "before output is written will make diagnostics from "
                * "`write_error_diagnostics=true` hard to interpret. If these "
                * "diagnostics are important, suggest using "
                * "`write_after_fixed_step_count=true`.")
    end

    t = Ref(code_time)
    dt = Ref(dt_reload === nothing ? mk_float(t_input["dt"]) : dt_reload)
    previous_dt = Ref(dt[])
    dt_before_output = Ref(dt[])
    dt_before_last_fail = Ref(dt_before_last_fail_reload === nothing ? mk_float(Inf) : dt_before_last_fail_reload)
    step_to_moments_output = Ref(false)
    step_to_dfns_output = Ref(false)
    write_moments_output = Ref(false)
    write_dfns_output = Ref(false)

    end_time = mk_float(code_time + t_input["dt"] * t_input["nstep"])
    epsilon = 1.e-11
    if adaptive && !t_input["write_after_fixed_step_count"]
        if t_input["nwrite"] == 0
            moments_output_times = [end_time]
        else
            moments_output_times = [code_time + i*t_input["dt"]
                                    for i ∈ t_input["nwrite"]:t_input["nwrite"]:t_input["nstep"]]
        end
        if moments_output_times[end] < end_time - epsilon
            push!(moments_output_times, end_time)
        end
        if t_input["nwrite_dfns"] == 0
            dfns_output_times = [end_time]
        else
            dfns_output_times = [code_time + i*t_input["dt"]
                                 for i ∈ t_input["nwrite_dfns"]:t_input["nwrite_dfns"]:t_input["nstep"]]
        end
        if dfns_output_times[end] < end_time - epsilon
            push!(dfns_output_times, end_time)
        end
    else
        # Use nwrite_moments and nwrite_dfns to determine when to write output
        moments_output_times = mk_float[]
        dfns_output_times = mk_float[]
    end

    if rk_coefs_implicit === nothing
        # Not an IMEX scheme, so cannot have any implicit terms
        t_input["implicit_braginskii_conduction"] = false
        if electron !== nothing && t_input["kinetic_electron_solver"] ∈ (implicit_time_evolving,
                                                                         implicit_p_implicit_pseudotimestep,
                                                                         implicit_steady_state,
                                                                         implicit_p_explicit_pseudotimestep)
            error("kinetic_electron_solver=$(t_input["kinetic_electron_solver"]) "
                  * "not supported when using a fully explicit timestep")
        end
        t_input["implicit_ion_advance"] = false
        t_input["implicit_vpa_advection"] = false
    else
        if composition.electron_physics != braginskii_fluid
            t_input["implicit_braginskii_conduction"] = false
        end
    end

    if electron !== nothing && t_input["implicit_vpa_advection"]
        error("implicit_vpa_advection does not work at the moment. Need to figure out "
              * "what to do with constraints, as explicit and implicit parts would not "
              * "preserve constaints separately.")
    end

    if t_input["high_precision_error_sum"]
        error_sum_zero = Float128(0.0)
    else
        error_sum_zero = 0.0
    end

    if electron === nothing
        # Setting up time_info for electrons.
        # Store io_input as the debug_io variable so we can use it to open the debug
        # output file.
        if t_input["debug_io"] !== false && debug_io === nothing
            if !isa(t_input["debug_io"], mk_int)
                error("`debug_io` input should be an integer, giving the number of steps "
                      * "between writes, if it is passed")
            end
            debug_io = (io_input, input_dict, t_input["debug_io"])
        end

        kinetic_electron_solver = null_kinetic_electrons # This option is only used from the ion time_info struct
        electron_preconditioner_type = nothing
        decrease_dt_iteration_threshold = t_input["decrease_dt_iteration_threshold"]
        increase_dt_iteration_threshold = t_input["increase_dt_iteration_threshold"]
        cap_factor_ion_dt = mk_float(t_input["cap_factor_ion_dt"])
        max_pseudotimesteps = t_input["max_pseudotimesteps"]
        max_pseudotime = t_input["max_pseudotime"]
        include_wall_bc_in_preconditioner = t_input["include_wall_bc_in_preconditioner"]
        electron_t_params = nothing
    elseif electron === false
        kinetic_electron_solver = null_kinetic_electrons
        electron_preconditioner_type = nothing
        decrease_dt_iteration_threshold = -1
        increase_dt_iteration_threshold = typemax(mk_int)
        cap_factor_ion_dt = Inf
        max_pseudotimesteps = -1
        max_pseudotime = Inf
        include_wall_bc_in_preconditioner = false
        electron_t_params = nothing
    else
        kinetic_electron_solver = t_input["kinetic_electron_solver"]
        if kinetic_electron_solver ∈ (implicit_time_evolving,
                                      implicit_p_implicit_pseudotimestep,
                                      implicit_steady_state)
            electron_precon_types = Dict("lu" => :electron_lu, "adi" => :electron_adi)
            if t_input["kinetic_electron_preconditioner"] == "default"
                if block_size[] == 1
                    electron_precon_option = "lu"
                else
                    electron_precon_option = "adi"
                end
            else
                electron_precon_option = t_input["kinetic_electron_preconditioner"]
            end
            if kinetic_electron_solver === implicit_steady_state && electron_precon_option != "lu"
                error("Only LU preconditioner currently supported for "
                      * "kinetic_electron_solver=\"implicit_steady_state\". Got "
                      * "kinetic_electron_preconditioner=$electron_precon_option")
            end
            electron_preconditioner_type = Val(electron_precon_types[electron_precon_option])
        else
            electron_preconditioner_type = Val(:none)
        end

        decrease_dt_iteration_threshold = -1
        increase_dt_iteration_threshold = typemax(mk_int)
        cap_factor_ion_dt = Inf
        max_pseudotimesteps = -1
        max_pseudotime = Inf
        include_wall_bc_in_preconditioner = false
        electron_t_params = electron

        # Check maximum dt for electrons
        if rk_coefs_implicit !== nothing
            electron.dt[] = min(electron.dt[], electron.maximum_dt,
                                electron.cap_factor_ion_dt * dt[] * rk_coefs_implicit[1,1])
        else
            electron.dt[] = min(electron.dt[], electron.maximum_dt)
        end
        electron.previous_dt[] = electron.dt[]
    end
    return time_info(n_variables, t_input["nstep"], end_time, t, dt, previous_dt,
                     dt_before_output, dt_before_last_fail, mk_float(CFL_prefactor),
                     step_to_moments_output, step_to_dfns_output, write_moments_output,
                     write_dfns_output, Ref(0), Ref(0), Ref{mk_float}(0.0), Ref(0),
                     Ref(0), Ref(0), OrderedDict{String,mk_int}(),
                     OrderedDict{String,mk_int}(), t_input["nwrite"],
                     t_input["nwrite_dfns"], moments_output_times, dfns_output_times,
                     t_input["type"], rk_coefs, rk_coefs_implicit,
                     implicit_coefficient_is_zero, n_rk_stages, rk_order,
                     electron !== nothing && t_input["exact_output_times"], adaptive,
                     low_storage, mk_float(t_input["rtol"]), mk_float(t_input["atol"]),
                     mk_float(t_input["atol_upar"]),
                     mk_float(t_input["step_update_prefactor"]),
                     mk_float(t_input["max_increase_factor"]),
                     mk_float(t_input["max_increase_factor_near_last_fail"]),
                     mk_float(t_input["last_fail_proximity_factor"]),
                     mk_float(t_input["minimum_dt"]), mk_float(t_input["maximum_dt"]),
                     electron !== nothing && t_input["implicit_braginskii_conduction"],
                     kinetic_electron_solver, electron_preconditioner_type,
                     electron !== nothing && t_input["implicit_ion_advance"],
                     electron !== nothing && t_input["implicit_vpa_advection"],
                     mk_float(t_input["constraint_forcing_rate"]),
                     decrease_dt_iteration_threshold, increase_dt_iteration_threshold,
                     mk_float(cap_factor_ion_dt), mk_int(max_pseudotimesteps),
                     mk_float(max_pseudotime), include_wall_bc_in_preconditioner,
                     t_input["write_after_fixed_step_count"], error_sum_zero,
                     t_input["split_operators"], t_input["print_nT_live"], 
                     t_input["steady_state_residual"], 
                     mk_float(t_input["converged_residual_value"]),
                     manufactured_solns_input.use_for_advance, t_input["stopfile_name"],
                     debug_io, electron_t_params)
end

"""
create arrays and do other work needed to setup
the main time advance loop.
this includes creating and populating structs
for Chebyshev transforms, velocity space moments,
EM fields, and advection terms
"""
function setup_time_advance!(pdf, fields, vz, vr, vzeta, vpa, vperp, z, r, gyrophase,
                             vz_spectral, vr_spectral, vzeta_spectral, vpa_spectral,
                             vperp_spectral, z_spectral, r_spectral, composition,
                             moments, t_input, code_time, dt_reload,
                             dt_before_last_fail_reload, electron_dt_reload,
                             electron_dt_before_last_fail_reload, collisions, species,
                             geometry, boundaries, external_source_settings,
                             num_diss_params, manufactured_solns_input, advection_structs,
                             io_input, restarting, restart_electron_physics, input_dict;
                             skip_electron_solve=false)
    

    # define some local variables for convenience/tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    ion_mom_diss_coeff = num_diss_params.ion.moment_dissipation_coefficient
    electron_mom_diss_coeff = num_diss_params.electron.moment_dissipation_coefficient
    neutral_mom_diss_coeff = num_diss_params.neutral.moment_dissipation_coefficient

    if composition.electron_physics != restart_electron_physics

        # When restarting from a different electron physics type, and
        # using an adaptive timestep, do not want to keep the `dt` from the previous
        # simulation, in case the new electron physics requires a smaller ion timestep.
        dt_reload = nothing
    end

    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        electron_t_params = setup_time_info(t_input["electron_t_input"], 2, 0.0,
                                            electron_dt_reload,
                                            electron_dt_before_last_fail_reload,
                                            composition, manufactured_solns_input,
                                            io_input, input_dict)
        # Set up entries for counters for which variable caused timestep limits and
        # timestep failures the right length. Do this setup even when not using adaptive
        # timestepping, because it is easier than modifying the file I/O according to
        # whether we are using adaptive timestepping.
        electron_t_params.limit_caused_by["max_increase_factor"] = 0
        electron_t_params.limit_caused_by["max_increase_factor_near_last_fail"] = 0
        electron_t_params.limit_caused_by["minimum_dt"] = 0
        electron_t_params.limit_caused_by["maximum_dt"] = 0
        electron_t_params.limit_caused_by["high_nl_iterations"] = 0

        # electron pdf
        electron_t_params.limit_caused_by["pdf_accuracy"] = 0
        electron_t_params.limit_caused_by["CFL_z"] = 0
        electron_t_params.limit_caused_by["CFL_vpa"] = 0
        electron_t_params.failure_caused_by["pdf_accuracy"] = 0

        # electron p
        electron_t_params.limit_caused_by["p_accuracy"] = 0
        electron_t_params.failure_caused_by["p_accuracy"] = 0
    else
        # Pass `false` rather than `nothing` to `setup_time_info()` call for ions, which
        # indicates that 'debug_io' should never be set up for ions.
        electron_t_params = false
    end
    n_variables = 1 # pdf
    if moments.evolve_density
        # ion density
        n_variables += 1
    end
    if moments.evolve_upar
        # ion flow
        n_variables += 1
    end
    if moments.evolve_p
        # ion pressure
        n_variables += 1
    end
    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        # electron pressure
        n_variables += 1
    end
    if composition.n_neutral_species > 0
        # neutral pdf
        n_variables += 1
        if moments.evolve_density
            # neutral density
            n_variables += 1
        end
        if moments.evolve_upar
            # neutral flow
            n_variables += 1
        end
        if moments.evolve_p
            # neutral pressure
            n_variables += 1
        end
    end
    if t_input["debug_io"] && block_rank[] == 0
        if t_input["nstep"] > 10
            println("You have enabled debug_io while setting a large value for "
                    * "nstep=$(t_input["nstep"]) > 10. Reducing to nstep=10 to avoid "
                    * "excessively large debug output files. If you really need nstep>10 "
                    * "comment out this modification at $(@__FILE__):$(@__LINE__).")
            t_input["nstep"] = 10
        end

        fake_t_params = (failure_caused_by=(), limit_caused_by=(),
                         electron=(failure_caused_by=(), limit_caused_by=(),),)
        # Various diagnostic outputs require information from multiple RK stages. It does
        # not make sense to write these in the debug_io where we write from within
        # individual RK stages, so force these diagnostic outputs to be disabled with
        # `debug_io_input`.
        debug_io_input = io_input_struct(;
                             run_name=io_input.run_name,
                             base_directory=io_input.base_directory,
                             ascii_output=false,
                             binary_format=io_input.binary_format,
                             parallel_io=io_input.parallel_io,
                             run_id=io_input.run_id,
                             output_dir=io_input.output_dir,
                             write_error_diagnostics=false,
                             write_steady_state_diagnostics=false,
                             write_electron_error_diagnostics=false,
                             write_electron_steady_state_diagnostics=false,
                             display_timing_info=false,
                            )
        # Need to exclude internal implementation variable "_section_check_store" from
        # input to be written to debug file.
        fake_input_dict = OptionsDict(k => v for (k,v) ∈ input_dict
                                      if k != "_section_check_store")
        @begin_serial_region()
        debug_io = setup_dfns_io(joinpath(io_input.output_dir, "debug"), debug_io_input,
                                 r, z, vperp, vpa, vzeta, vr, vz, composition, collisions,
                                 moments.evolve_density, moments.evolve_upar,
                                 moments.evolve_p, external_source_settings, nothing,
                                 1, fake_input_dict, comm_inter_block[], nothing, 0.0,
                                 fake_t_params, (); is_debug=true)
    elseif t_input["debug_io"]
        # Need to synchronize shared-memory blocks before/after I/O. Set debug_io=true so
        # we can distinguish from no-debug-IO case on block_rank[]>0 processes.
        @begin_serial_region() # To match one in `t_input["debug_io"] && block_rank[] == 0` branch
        debug_io = true
    else
        debug_io = nothing
    end
    t_params = setup_time_info(t_input, n_variables, code_time, dt_reload,
                               dt_before_last_fail_reload, composition,
                               manufactured_solns_input, io_input, input_dict;
                               electron=electron_t_params, debug_io=debug_io)

    # Set up entries for counters for which variable caused timestep limits and
    # timestep failures the right length. Do this setup even when not using adaptive
    # timestepping, because it is easier than modifying the file I/O according to
    # whether we are using adaptive timestepping.
    t_params.limit_caused_by["max_increase_factor"] = 0
    t_params.limit_caused_by["max_increase_factor_near_last_fail"] = 0
    t_params.limit_caused_by["minimum_dt"] = 0
    t_params.limit_caused_by["maximum_dt"] = 0
    t_params.limit_caused_by["high_nl_iterations"] = 0

    # ion pdf
    t_params.limit_caused_by["pdf_accuracy"] = 0
    if !t_params.implicit_ion_advance
        t_params.limit_caused_by["CFL_z"] = 0
    end
    if !(t_params.implicit_ion_advance || t_params.implicit_vpa_advection)
        t_params.limit_caused_by["CFL_vpa"] = 0
    end
    t_params.failure_caused_by["pdf_accuracy"] = 0

    if moments.evolve_density
        # ion density
        t_params.limit_caused_by["density_accuracy"] = 0
        t_params.failure_caused_by["density_accuracy"] = 0
    end
    if moments.evolve_upar
        # ion flow
        t_params.limit_caused_by["upar_accuracy"] = 0
        t_params.failure_caused_by["upar_accuracy"] = 0
    end
    if moments.evolve_density
        # ion pressure
        t_params.limit_caused_by["p_accuracy"] = 0
        t_params.failure_caused_by["p_accuracy"] = 0
    end

    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        # electron pressure
        t_params.limit_caused_by["electron_p_accuracy"] = 0
        t_params.failure_caused_by["electron_p_accuracy"] = 0
        if t_params.kinetic_electron_solver == implicit_time_evolving
            t_params.limit_caused_by["electron_pdf_accuracy"] = 0
            t_params.failure_caused_by["electron_pdf_accuracy"] = 0
        elseif t_params.kinetic_electron_solver == explicit_time_evolving
            t_params.limit_caused_by["electron_pdf_accuracy"] = 0
            t_params.limit_caused_by["electron_CFL_z"] = 0
            t_params.limit_caused_by["electron_CFL_vpa"] = 0
            t_params.failure_caused_by["electron_pdf_accuracy"] = 0
        end
        if composition.electron_physics ∈ (kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
            t_params.failure_caused_by["kinetic_electron_convergence"] = 0
        end
    end
    if composition.n_neutral_species > 0
        t_params.limit_caused_by["neutral_pdf_accuracy"] = 0
        t_params.limit_caused_by["neutral_CFL_z"] = 0
        t_params.limit_caused_by["neutral_CFL_vpa"] = 0
        t_params.failure_caused_by["neutral_pdf_accuracy"] = 0

        if moments.evolve_density
            # neutral density
            t_params.limit_caused_by["neutral_density_accuracy"] = 0
            t_params.failure_caused_by["neutral_density_accuracy"] = 0
        end
        if moments.evolve_upar
            # neutral flow
            t_params.limit_caused_by["neutral_uz_accuracy"] = 0
            t_params.failure_caused_by["neutral_uz_accuracy"] = 0
        end
        if moments.evolve_density
            # neutral pressure
            t_params.limit_caused_by["neutral_p_accuracy"] = 0
            t_params.failure_caused_by["neutral_p_accuracy"] = 0
        end
    end
    if t_params.rk_coefs_implicit !== nothing
        t_params.failure_caused_by["nonlinear_solver_convergence"] = 0
    end

    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    advance = setup_advance_flags(moments, composition, t_params, collisions,
                                  external_source_settings, num_diss_params,
                                  manufactured_solns_input, r, z, vperp, vpa, vzeta, vr,
                                  vz)
    advance_implicit =
        setup_implicit_advance_flags(moments, composition, t_params, collisions,
                                     external_source_settings, num_diss_params,
                                     manufactured_solns_input, r, z, vperp, vpa, vzeta,
                                     vr, vz)
    # Check that no flags that shouldn't be are set in both advance and advance_implicit
    for field ∈ fieldnames(advance_info)
        if field ∈ (:r_diffusion, :vpa_diffusion, :vperp_diffusion, :vz_diffusion)
            # These are meant to be set in both structs
            continue
        end
        if getfield(advance, field) && getfield(advance_implicit, field)
            error("$field is set to `true` in both `advance` and `advance_implicit`")
        end
    end

    # Set up parameters for Jacobian-free Newton-Krylov solver used for implicit part of
    # timesteps.
    electron_conduction_nl_solve_parameters = setup_nonlinear_solve(t_params.implicit_braginskii_conduction,
                                                                    input_dict, (z=z,);
                                                                    default_rtol=t_params.rtol / 10.0,
                                                                    default_atol=t_params.atol / 10.0)
    nl_solver_electron_advance_params =
        setup_nonlinear_solve(t_params.kinetic_electron_solver ∈ (implicit_time_evolving,
                                                                  implicit_p_implicit_pseudotimestep,
                                                                  implicit_steady_state),
                              input_dict,
                              (z=z, vperp=vperp, vpa=vpa),
                              (r,);
                              default_rtol=t_params.rtol / 10.0,
                              default_atol=t_params.atol / 10.0,
                              electron_p_pdf_solve=true,
                              preconditioner_type=t_params.electron_preconditioner_type)
    nl_solver_ion_advance_params =
        setup_nonlinear_solve(t_params.implicit_ion_advance, input_dict,
                              (s=composition.n_ion_species, r=r, z=z, vperp=vperp,
                               vpa=vpa),
                              ();
                              default_rtol=t_params.rtol / 10.0,
                              default_atol=t_params.atol / 10.0,
                              preconditioner_type=Val(:lu))
    # Implicit solve for vpa_advection term should be done in serial, as it will be called
    # within a parallelised s_r_z_vperp loop.
    nl_solver_vpa_advection_params =
        setup_nonlinear_solve(t_params.implicit_vpa_advection, input_dict, (vpa=vpa,),
                              (composition.n_ion_species, r, z, vperp);
                              default_rtol=t_params.rtol / 10.0,
                              default_atol=t_params.atol / 10.0,
                              serial_solve=true, preconditioner_type=Val(:lu))
    if nl_solver_ion_advance_params !== nothing &&
            nl_solver_vpa_advection_params !== nothing
        error("Cannot use implicit_ion_advance and implicit_vpa_advection at the same "
              * "time")
    end
    nl_solver_params = (electron_conduction=electron_conduction_nl_solve_parameters,
                        electron_advance=nl_solver_electron_advance_params,
                        ion_advance=nl_solver_ion_advance_params,
                        vpa_advection=nl_solver_vpa_advection_params,)

    # Check that no unexpected sections or top-level options were passed (helps to catch
    # typos in input files). Needs to be called after calls to `setup_nonlinear_solve()`
    # because the inputs for nonlinear solvers are only read there, but before electron
    # setup, because `input_dict` needs to be written to the output files, and it cannot
    # be with the `_section_check_store` variable still contained in it (which is used and
    # removed by `check_sections!()`).
    check_sections!(input_dict)

    @begin_serial_region()

    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf, t_params.n_rk_stages + 1,
                                   t_params.kinetic_electron_solver ∈ (implicit_time_evolving, explicit_time_evolving))
    if t_params.rk_coefs_implicit !== nothing
        scratch_implicit = setup_scratch_arrays(moments, pdf, t_params.n_rk_stages,
                                                t_params.kinetic_electron_solver ∈ (implicit_time_evolving, explicit_time_evolving))
    else
        scratch_implicit = nothing
    end
    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        scratch_electron = setup_electron_scratch_arrays(moments, pdf,
                                                         t_params.electron.n_rk_stages+1)
    else
        scratch_electron = nothing
    end
    # setup dummy arrays & buffer arrays for z r MPI
    n_neutral_species_alloc = max(1,composition.n_neutral_species)
    scratch_dummy = setup_dummy_and_buffer_arrays(r.n, z.n, vpa.n, vperp.n, vz.n, vr.n,
                                                  vzeta.n, composition.n_ion_species,
                                                  n_neutral_species_alloc, t_params)
    # create arrays for Fokker-Planck collisions 
    if advance.explicit_weakform_fp_collisions
        if collisions.fkpl.boundary_data_option == direct_integration
            precompute_weights = true
        else
            precompute_weights = false
        end
        fp_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral;
                      precompute_weights=precompute_weights)
    else
        fp_arrays = nothing
    end
    # create gyroaverage matrix arrays
    gyroavs = init_gyro_operators(vperp, z, r, gyrophase, geometry, boundaries,
                                  composition)

    # Now that `t_params` and `scratch` have been created, initialize electrons if
    # necessary
    if restarting &&
            composition.electron_physics ∈ (kinetic_electrons,
                                            kinetic_electrons_with_temperature_equation) &&
            restart_electron_physics ∈ (kinetic_electrons,
                                        kinetic_electrons_with_temperature_equation)
        if t_params.electron.debug_io !== nothing
            # Create *.electron_debug.h5 file so that it can be re-opened in
            # update_electron_pdf!().
            io_electron = setup_electron_io(t_params.electron.debug_io[1], vpa, vperp, z, r,
                                            composition, collisions, moments.evolve_density,
                                            moments.evolve_upar, moments.evolve_p,
                                            external_source_settings, t_params.electron,
                                            t_params.electron.debug_io[2], -1, nothing,
                                            "electron_debug")
        end

        # No need to do electron I/O (apart from possibly debug I/O) any more, so if
        # adaptive timestep is used, it does not need to adjust to output times.
        resize!(t_params.electron.moments_output_times, 0)
        resize!(t_params.electron.dfns_output_times, 0)
        t_params.electron.moments_output_counter[] = 1
        t_params.electron.dfns_output_counter[] = 1
    elseif composition.electron_physics != restart_electron_physics
        @begin_serial_region()
        @serial_region begin
            # zero-initialise phi here, because the boundary points of phi are used as an
            # effective 'cache' for the sheath-boundary cutoff speed for the electrons, so
            # needs to be initialised to something, but phi cannot be calculated properly
            # until after the electrons are initialised.
            fields.phi .= 0.0
        end
        initialize_electrons!(pdf, moments, fields, geometry, composition, r, z,
                              vperp, vpa, vzeta, vr, vz, z_spectral, r_spectral,
                              vperp_spectral, vpa_spectral, collisions, gyroavs,
                              external_source_settings, scratch_dummy, scratch,
                              scratch_electron, nl_solver_params, t_params, t_input,
                              num_diss_params, advection_structs, io_input, input_dict;
                              restart_electron_physics=restart_electron_physics,
                              skip_electron_solve=skip_electron_solve)
    end

    # update the derivatives of the electron moments as these may be needed when
    # computing the electrostatic potential (and components of the electric field)
    calculate_electron_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                           electron_mom_diss_coeff, composition.electron_physics)
    # calculate the electron-ion parallel friction force
    calculate_electron_parallel_friction_force!(moments.electron.parallel_friction, moments.electron.dens,
        moments.electron.upar, moments.ion.upar, moments.electron.dT_dz,
        composition.me_over_mi, collisions.electron_fluid.nu_ei,
        composition.electron_physics)
    # initialize the electrostatic potential
    @begin_serial_region()
    update_phi!(fields, scratch[1], vperp, z, r, composition, collisions, moments,
                geometry, z_spectral, r_spectral, scratch_dummy, gyroavs)
    @serial_region begin
        # save the initial phi(z) for possible use later (e.g., if forcing phi)
        fields.phi0 .= fields.phi
    end

    # Preliminary calculation of moment derivatives, to be used for initial version of
    # 'speed' in advect objects, which are needed for boundary conditions on the
    # distribution function which is then used to (possibly) re-calculate the moments
    # after which the initial values of moment derivatives are re-calculated.
    calculate_ion_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      ion_mom_diss_coeff)
    calculate_neutral_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                          neutral_mom_diss_coeff)

    r_advect = advection_structs.r_advect
    z_advect = advection_structs.z_advect
    vperp_advect = advection_structs.vperp_advect
    vpa_advect = advection_structs.vpa_advect
    neutral_r_advect = advection_structs.neutral_r_advect
    neutral_z_advect = advection_structs.neutral_z_advect
    neutral_vz_advect = advection_structs.neutral_vz_advect
    ##
    # ion particle advection only
    ##

    if r.n > 1
        # initialise the r advection speed
        @begin_s_z_vperp_vpa_region()
        @loop_s is begin
            @views update_speed_r!(r_advect[is], moments.ion.upar[:,:,is],
                                   moments.ion.vth[:,:,is], fields, moments.evolve_upar,
                                   moments.evolve_p, vpa, vperp, z, r, geometry, is)
        end
        # enforce prescribed boundary condition in r on the distribution function f
    end

    if z.n > 1
        # initialise the z advection speed
        @begin_s_r_vperp_vpa_region()
        @loop_s is begin
            @views update_speed_z!(z_advect[is], moments.ion.upar[:,:,is],
                                   moments.ion.vth[:,:,is], moments.evolve_upar,
                                   moments.evolve_p, fields, vpa, vperp, z, r, 0.0,
                                   geometry, is)
        end
    end

    # initialise the vpa advection speed
    @begin_s_r_z_vperp_region()
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, vperp, z, r,
                      composition, collisions, external_source_settings.ion, 0.0,
                      geometry)

    # initialise the vperp advection speed
    # Note that z_advect and r_advect are arguments of update_speed_vperp!
    # This means that z_advect[is].speed and r_advect[is].speed are used to determine
    # vperp_advect[is].speed, so z_advect and r_advect must always be updated before
    # vperp_advect is updated and used.
    if vperp.n > 1
        update_speed_vperp!(vperp_advect, scratch[1], vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    end
    
    ##
    # Neutral particle advection
    ##

    if n_neutral_species > 0 && r.n > 1
        # initialise the r advection speed
        @begin_sn_z_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_r!(neutral_r_advect[isn], r, z, vzeta, vr, vz)
        end
    end

    if n_neutral_species > 0 && z.n > 1
        # initialise the z advection speed
        @begin_sn_r_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_z!(neutral_z_advect[isn],
                                           moments.neutral.uz[:,:,isn],
                                           moments.neutral.vth[:,:,isn],
                                           moments.evolve_upar, moments.evolve_p, vz,
                                           vr, vzeta, z, r, 0.0)
        end
    end

    if n_neutral_species > 0
        # initialise the z advection speed
        @serial_region begin
            # Initialise the vz 'advection speed' in case it does not need updating. It
            # may still be used to decide which boundary is 'incoming' in the vz boundary
            # condition.
            @loop_sn isn begin
                neutral_vz_advect[isn].speed .= 0.0
            end
        end
        @begin_sn_r_z_vzeta_vr_region()
        @views update_speed_neutral_vz!(neutral_vz_advect, fields, scratch[1], moments,
                                        vz, vr, vzeta, z, r, composition, collisions,
                                        external_source_settings.neutral)
    end

    ##
    # construct advect & spectral objects to compactify arguments
    ##

    spectral_objects = spectral_object_struct(vz_spectral, vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral, z_spectral, r_spectral)
    if(advance.manufactured_solns_test)
        manufactured_source_list = manufactured_sources(manufactured_solns_input, r, z,
                                                        vperp, vpa, vzeta, vr, vz,
                                                        boundaries, composition,
                                                        geometry.input, collisions,
                                                        num_diss_params, species)
    else
        manufactured_source_list = nothing
    end

    if !restarting
        @begin_serial_region()
        # ensure initial pdf has no negative values
        force_minimum_pdf_value!(pdf.ion.norm, num_diss_params.ion.force_minimum_pdf_value)
        force_minimum_pdf_value_neutral!(pdf.neutral.norm, num_diss_params.neutral.force_minimum_pdf_value)

        # enforce boundary conditions and moment constraints to ensure a consistent initial
        # condition
        enforce_boundary_conditions!(
            pdf.ion.norm, moments.ion.dens, moments.ion.upar, moments.ion.p, fields.phi,
            boundaries, moments, vpa, vperp, z, r, vpa_spectral, vperp_spectral,
            vpa_advect, vperp_advect, z_advect, r_advect, composition, scratch_dummy,
            advance.r_diffusion, advance.vpa_diffusion, advance.vperp_diffusion)

        # Ensure normalised pdf exactly obeys integral constraints if evolving moments
        if moments.evolve_density && moments.enforce_conservation
            hard_force_moment_constraints!(pdf.ion.norm, moments, vpa, vperp)
        end

        # update moments in case they were affected by applying boundary conditions or
        # constraints to the pdf
        reset_moments_status!(moments)
        update_moments!(moments, pdf.ion.norm, gyroavs, vpa, vperp, z, r, composition,
           r_spectral,geometry,scratch_dummy,z_advect, collisions)
        # enforce boundary conditions in r and z on the neutral particle distribution function
        if n_neutral_species > 0
            # Note, so far vr and vzeta do not need advect objects, so pass `nothing` for
            # those as a placeholder
            enforce_neutral_boundary_conditions!(
                pdf.neutral.norm, pdf.ion.norm, moments.neutral.dens, moments.neutral.uz,
                moments.neutral.p, boundaries, moments, moments.ion.dens,
                moments.ion.upar, fields.Er, vzeta_spectral, vr_spectral, vz_spectral,
                neutral_r_advect, neutral_z_advect, nothing, nothing, neutral_vz_advect,
                r, z, vzeta, vr, vz, composition, geometry, scratch_dummy,
                advance.r_diffusion, advance.vz_diffusion)
            if moments.evolve_density && moments.enforce_conservation
                hard_force_moment_constraints_neutral!(pdf.neutral.norm, moments, vz)
            end
            update_moments_neutral!(moments, pdf.neutral.norm, vz, vr, vzeta, z, r,
                                    composition)
        end

        # Update scratch arrays in case they were affected by applying boundary conditions
        # or constraints to the pdf.
        # Also update scratch[t_params.n_rk_stages+1] as this will be used for the I/O at
        # the initial time.
        @begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
            scratch[1].pdf[:,:,iz,ir,is] .= pdf.ion.norm[:,:,iz,ir,is]
            scratch[1].density[iz,ir,is] = moments.ion.dens[iz,ir,is]
            scratch[1].upar[iz,ir,is] = moments.ion.upar[iz,ir,is]
            scratch[1].p[iz,ir,is] = moments.ion.p[iz,ir,is]
            scratch[t_params.n_rk_stages+1].pdf[:,:,iz,ir,is] .= pdf.ion.norm[:,:,iz,ir,is]
            scratch[t_params.n_rk_stages+1].density[iz,ir,is] = moments.ion.dens[iz,ir,is]
            scratch[t_params.n_rk_stages+1].upar[iz,ir,is] = moments.ion.upar[iz,ir,is]
            scratch[t_params.n_rk_stages+1].p[iz,ir,is] = moments.ion.p[iz,ir,is]
        end

        # update the electron density, parallel flow and parallel pressure (and temperature)
        # in case the corresponding ion quantities have been changed by applying
        # constraints to the ion pdf
        calculate_electron_density!(moments.electron.dens, moments.electron.dens_updated, moments.ion.dens)
        calculate_electron_upar_from_charge_conservation!(moments.electron.upar, moments.electron.upar_updated,
                                                          moments.electron.dens, moments.ion.upar, moments.ion.dens,
                                                          composition.electron_physics, r, z)
        @begin_serial_region()
        # compute the updated electron temperature
        # NB: not currently necessary, as initial vth is not directly dependent on ion quantities
        @serial_region begin
            @. moments.electron.temp = 0.5 * composition.me_over_mi * moments.electron.vth^2
        end
        # as the electron temperature has now been updated, set the appropriate flag
        moments.electron.temp_updated[] = true
        # compute the updated electron parallel pressure
        @serial_region begin
            @. moments.electron.p = moments.electron.dens * moments.electron.temp
        end
        # as the electron p has now been updated, set the appropriate flag
        moments.electron.p_updated[] = true
        # calculate the zed derivative of the initial electron temperature, potentially
        # needed in the following calculation of the electron parallel friction force and
        # parallel heat flux
        @views derivative_z!(moments.electron.dT_dz, moments.electron.temp, 
            scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
            scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
        # calculate the electron parallel heat flux
        calculate_electron_qpar!(moments.electron, pdf.electron, moments.electron.p,
            moments.electron.dens, moments.electron.upar, moments.ion.upar,
            collisions.electron_fluid.nu_ei, composition.me_over_mi,
            composition.electron_physics, vperp, vpa)
        if composition.electron_physics == braginskii_fluid
            electron_fluid_qpar_boundary_condition!(
                moments.electron.p, moments.electron.upar, moments.electron.dens,
                moments.electron, z)
        end
        # Update the electron moment entries in the scratch array.
        # Also update scratch[t_params.n_rk_stages+1] as this will be used for the I/O at
        # the initial time.
        @begin_r_z_region()
        @loop_r_z ir iz begin
            scratch[1].electron_density[iz,ir] = moments.electron.dens[iz,ir]
            scratch[1].electron_upar[iz,ir] = moments.electron.upar[iz,ir]
            scratch[1].electron_p[iz,ir] = moments.electron.p[iz,ir]
            scratch[1].electron_temp[iz,ir] = moments.electron.temp[iz,ir]
            scratch[t_params.n_rk_stages+1].electron_density[iz,ir] = moments.electron.dens[iz,ir]
            scratch[t_params.n_rk_stages+1].electron_upar[iz,ir] = moments.electron.upar[iz,ir]
            scratch[t_params.n_rk_stages+1].electron_p[iz,ir] = moments.electron.p[iz,ir]
            scratch[t_params.n_rk_stages+1].electron_temp[iz,ir] = moments.electron.temp[iz,ir]
        end

        @begin_sn_r_z_region(true)
        @loop_sn_r_z isn ir iz begin
            scratch[1].pdf_neutral[:,:,:,iz,ir,isn] .= pdf.neutral.norm[:,:,:,iz,ir,isn]
            scratch[1].density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            scratch[1].uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            scratch[1].p_neutral[iz,ir,isn] = moments.neutral.p[iz,ir,isn]
            scratch[t_params.n_rk_stages+1].pdf_neutral[:,:,:,iz,ir,isn] .= pdf.neutral.norm[:,:,:,iz,ir,isn]
            scratch[t_params.n_rk_stages+1].density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            scratch[t_params.n_rk_stages+1].uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            scratch[t_params.n_rk_stages+1].p_neutral[iz,ir,isn] = moments.neutral.p[iz,ir,isn]
        end
    end

    # calculate the electron-ion parallel friction force
    calculate_electron_parallel_friction_force!(moments.electron.parallel_friction, moments.electron.dens,
        moments.electron.upar, moments.ion.upar, moments.electron.dT_dz,
        composition.me_over_mi, collisions.electron_fluid.nu_ei,
        composition.electron_physics)

    calculate_ion_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      ion_mom_diss_coeff)
    calculate_electron_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      electron_mom_diss_coeff, composition.electron_physics)
    calculate_neutral_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      neutral_mom_diss_coeff)
    # update the electrostatic potential and components of the electric field, as pdfs and moments
    # may have changed due to enforcing boundary/moment constraints                                      
    update_phi!(fields, scratch[1], vperp, z, r, composition, collisions, moments,
                geometry, z_spectral, r_spectral, scratch_dummy, gyroavs)

    # Ensure all processes are synchronized at the end of the setup
    @_block_synchronize()

    

    return moments, spectral_objects, scratch, scratch_implicit, scratch_electron,
           scratch_dummy, advance, advance_implicit, t_params, fp_arrays, gyroavs,
           manufactured_source_list, nl_solver_params
end

"""
create the 'advance_info' struct to be used in later Euler advance to
indicate which parts of the equations are to be advanced concurrently.
if no splitting of operators, all terms advanced concurrently;
else, will advance one term at a time.
"""
function setup_advance_flags(moments, composition, t_params, collisions,
                             external_source_settings, num_diss_params,
                             manufactured_solns_input, r, z, vperp, vpa, vzeta, vr, vz)
    # default is not to concurrently advance different operators
    advance_vpa_advection = false
    advance_vperp_advection = false
    advance_z_advection = false
    advance_r_advection = false
    advance_ion_cx_1V = false
    advance_neutral_cx_1V = false
    advance_ion_cx = false
    advance_neutral_cx = false
    advance_ion_ionization = false
    advance_neutral_ionization = false
    advance_ion_ionization_1V = false
    advance_neutral_ionization_1V = false
    advance_krook_collisions_ii = false
    advance_maxwell_diffusion_ii = false
    advance_maxwell_diffusion_nn = false
    advance_external_source = false
    advance_ion_numerical_dissipation = false
    advance_neutral_numerical_dissipation = false
    advance_sources = false
    advance_continuity = false
    advance_force_balance = false
    advance_energy = false
    advance_electron_pdf = false
    advance_electron_energy = false
    advance_electron_conduction = false
    advance_neutral_z_advection = false
    advance_neutral_r_advection = false
    advance_neutral_vz_advection = false
    advance_neutral_external_source = false
    advance_neutral_sources = false
    advance_neutral_continuity = false
    advance_neutral_force_balance = false
    advance_neutral_energy = false
    r_diffusion = false
    vpa_diffusion = false
    vperp_diffusion = false
    vz_diffusion = false
    explicit_weakform_fp_collisions = false
    # all advance flags remain false if using operator-splitting
    # otherwise, check to see if the flags need to be set to true
    if !t_params.split_operators
        # default for non-split operators is to include both vpa and z advection together
        # If using an IMEX scheme and implicit vpa advection has been requested, then vpa
        # advection is not included in the explicit part of the timestep.
        advance_vpa_advection = vpa.n > 1 && !(t_params.implicit_ion_advance || t_params.implicit_vpa_advection)
        advance_vperp_advection = vperp.n > 1 && !t_params.implicit_ion_advance
        advance_z_advection = z.n > 1 && !t_params.implicit_ion_advance
        advance_r_advection = r.n > 1 && !t_params.implicit_ion_advance
        if collisions.fkpl.nuii > 0.0 && vperp.n > 1 && !t_params.implicit_ion_advance
            explicit_weakform_fp_collisions = true
        else
            explicit_weakform_fp_collisions = false    
        end
        # if neutrals present, check to see if different ion-neutral
        # collisions are enabled
        if composition.n_neutral_species > 0
            advance_neutral_z_advection = true
            if r.n > 1
                advance_neutral_r_advection = true
            end
            if moments.evolve_upar || moments.evolve_p
                advance_neutral_vz_advection = true
            end
            # if charge exchange collision frequency non-zero,
            # account for charge exchange collisions
            if abs(collisions.reactions.charge_exchange_frequency) > 0.0
                if vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                    advance_ion_cx_1V = !t_params.implicit_ion_advance
                    advance_neutral_cx_1V = true
                elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                    advance_ion_cx = !t_params.implicit_ion_advance
                    advance_neutral_cx = true
                else
                    error("If any perpendicular velocity has length>1 they all must. "
                          * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                          * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
                end
            end
            # if ionization collision frequency non-zero,
            # account for ionization collisions
            if abs(collisions.reactions.ionization_frequency) > 0.0
                if vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                    advance_ion_ionization_1V = !t_params.implicit_ion_advance
                    advance_neutral_ionization_1V = true
                elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                    advance_ion_ionization = !t_params.implicit_ion_advance
                    advance_neutral_ionization = true
                else
                    error("If any perpendicular velocity has length>1 they all must. "
                          * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                          * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
                end
            end
        end
        # set flags for krook and maxwell diffusion collisions, and negative coefficient
        # in both cases (as usual) will mean not employing that operator (flag remains false)
        if collisions.krook.nuii0 > 0.0
            advance_krook_collisions_ii = !t_params.implicit_ion_advance
        end
        if collisions.mxwl_diff.D_ii > 0.0
            advance_maxwell_diffusion_ii = true
        end
        if collisions.mxwl_diff.D_nn > 0.0
            advance_maxwell_diffusion_nn = true
        end
        advance_external_source = any(x -> x.active, external_source_settings.ion) && !t_params.implicit_ion_advance
        advance_neutral_external_source = any(x -> x.active, external_source_settings.neutral)
        advance_ion_numerical_dissipation = !(t_params.implicit_ion_advance || t_params.implicit_vpa_advection)
        advance_neutral_numerical_dissipation = true
        # if evolving the density, must advance the continuity equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_density
            advance_sources = !t_params.implicit_ion_advance
            advance_continuity = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_continuity = true
            end
        end
        # if evolving the parallel flow, must advance the force balance equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_upar
            advance_sources = !t_params.implicit_ion_advance
            advance_force_balance = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_force_balance = true
            end
        end
        # if evolving the parallel pressure, must advance the energy equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_p
            advance_sources = !t_params.implicit_ion_advance
            advance_energy = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_energy = true
            end
        end
        # if treating the electrons as a fluid with Braginskii closure, or
        # moment-kinetically then advance the electron energy equation
        if composition.electron_physics ∈ (kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
            if !(t_params.kinetic_electron_solver ∈ (implicit_time_evolving,
                                                     implicit_p_implicit_pseudotimestep,
                                                     implicit_steady_state,
                                                     explicit_time_evolving,
                                                     implicit_p_explicit_pseudotimestep))
                advance_electron_energy = true
                advance_electron_conduction = true
            end
        elseif composition.electron_physics == braginskii_fluid
            if t_params.implicit_braginskii_conduction
                # if treating the electrons as a fluid with Braginskii closure, and using
                # an IMEX scheme, advance the conduction part of the electron energy
                # equation implicitly.
                advance_electron_energy = true
                advance_electron_conduction = false
            else
                # If not using an IMEX scheme, treat the conduction explicitly.
                advance_electron_energy = true
                advance_electron_conduction = true
            end
        end
        if t_params.kinetic_electron_solver == explicit_time_evolving
            advance_electron_pdf = true
        end

        # *_diffusion flags are set regardless of whether diffusion is included in explicit or
        # implicit part of timestep, because they are used for boundary conditions, not to
        # control which terms are advanced.
        #
        # flag to determine if a d^2/dr^2 operator is present
        r_diffusion = (num_diss_params.ion.r_dissipation_coefficient > 0.0)
        # flag to determine if a d^2/dvpa^2 operator is present
        # When using implicit_vpa_advection, the vpa diffusion is included in the implicit
        # step
        vpa_diffusion = ((num_diss_params.ion.vpa_dissipation_coefficient > 0.0) || (collisions.fkpl.nuii > 0.0 && vperp.n > 1) || advance_maxwell_diffusion_ii)
        vperp_diffusion = ((num_diss_params.ion.vperp_dissipation_coefficient > 0.0) || (collisions.fkpl.nuii > 0.0 && vperp.n > 1))
        vz_diffusion = (num_diss_params.neutral.vz_dissipation_coefficient > 0.0 || advance_maxwell_diffusion_nn)
    end

    manufactured_solns_test = manufactured_solns_input.use_for_advance

    return advance_info(advance_vpa_advection, advance_vperp_advection, advance_z_advection, advance_r_advection,
                        advance_neutral_z_advection, advance_neutral_r_advection,
                        advance_neutral_vz_advection, advance_ion_cx, advance_neutral_cx,
                        advance_ion_cx_1V, advance_neutral_cx_1V, advance_ion_ionization,
                        advance_neutral_ionization, advance_ion_ionization_1V,
                        advance_neutral_ionization_1V, advance_krook_collisions_ii,
                        advance_maxwell_diffusion_ii, advance_maxwell_diffusion_nn,
                        explicit_weakform_fp_collisions,
                        advance_external_source, advance_ion_numerical_dissipation,
                        advance_neutral_numerical_dissipation, advance_sources,
                        advance_continuity, advance_force_balance, advance_energy,
                        advance_electron_pdf, advance_electron_energy,
                        advance_electron_conduction, advance_neutral_external_source,
                        advance_neutral_sources, advance_neutral_continuity,
                        advance_neutral_force_balance, advance_neutral_energy,
                        manufactured_solns_test, r_diffusion, vpa_diffusion,
                        vperp_diffusion, vz_diffusion)
end

"""
create the 'advance_info' struct to be used in the time advance to
indicate which parts of the equations are to be advanced implicitly (using
`backward_euler!()`).
"""
function setup_implicit_advance_flags(moments, composition, t_params, collisions,
                                      external_source_settings, num_diss_params,
                                      manufactured_solns_input, r, z, vperp, vpa, vzeta,
                                      vr, vz)
    # default is not to concurrently advance different operators
    advance_vpa_advection = false
    advance_vperp_advection = false
    advance_z_advection = false
    advance_r_advection = false
    advance_ion_cx_1V = false
    advance_neutral_cx_1V = false
    advance_ion_cx = false
    advance_neutral_cx = false
    advance_ion_ionization = false
    advance_neutral_ionization = false
    advance_ion_ionization_1V = false
    advance_neutral_ionization_1V = false
    advance_krook_collisions_ii = false
    advance_maxwell_diffusion_ii = false
    advance_maxwell_diffusion_nn = false
    advance_external_source = false
    advance_ion_numerical_dissipation = false
    advance_neutral_numerical_dissipation = false
    advance_sources = false
    advance_continuity = false
    advance_force_balance = false
    advance_energy = false
    advance_electron_pdf = false
    advance_electron_energy = false
    advance_electron_conduction = false
    advance_neutral_z_advection = false
    advance_neutral_r_advection = false
    advance_neutral_vz_advection = false
    advance_neutral_external_source = false
    advance_neutral_sources = false
    advance_neutral_continuity = false
    advance_neutral_force_balance = false
    advance_neutral_energy = false
    r_diffusion = false
    vpa_diffusion = false
    vperp_diffusion = false
    vz_diffusion = false
    explicit_weakform_fp_collisions = false
    if t_params.split_operators
        error("Implicit timesteps do not support `t_params.split_operators=true`")
    end
    if t_params.implicit_ion_advance
        advance_vpa_advection = vpa.n > 1 && z.n > 1
        advance_vperp_advection = vperp.n > 1 && z.n > 1
        advance_z_advection = z.n > 1
        advance_r_advection = r.n > 1
        if abs(collisions.reactions.charge_exchange_frequency) > 0.0
            if vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                advance_ion_cx_1V = true
            elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                advance_ion_cx = true
            else
                error("If any perpendicular velocity has length>1 they all must. "
                      * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                      * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
            end
        end
        if abs(collisions.reactions.ionization_frequency) > 0.0
            if vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                advance_ion_ionization_1V = true
            elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                advance_ion_ionization = true
            else
                error("If any perpendicular velocity has length>1 they all must. "
                      * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                      * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
            end
        end
        advance_krook_collisions_ii = collisions.krook.nuii0 > 0.0
        advance_external_source = any(x -> x.active, external_source_settings.ion)
        advance_ion_numerical_dissipation = true
        advance_sources = moments.evolve_density || moments.evolve_upar || moments.evolve_p
        explicit_weakform_fp_collisions = collisions.fkpl.nuii > 0.0 && vperp.n > 1
    elseif t_params.implicit_vpa_advection
        advance_vpa_advection = true
        advance_ion_numerical_dissipation = true
    end
    # *_diffusion flags are set regardless of whether diffusion is included in explicit or
    # implicit part of timestep, because they are used for boundary conditions, not to
    # control which terms are advanced.
    #
    # flag to determine if a d^2/dr^2 operator is present
    r_diffusion = (num_diss_params.ion.r_dissipation_coefficient > 0.0)
    # flag to determine if a d^2/dvpa^2 operator is present
    # When using implicit_vpa_advection, the vpa diffusion is included in the implicit
    # step
    vpa_diffusion = ((num_diss_params.ion.vpa_dissipation_coefficient > 0.0) || (collisions.fkpl.nuii > 0.0 && vperp.n > 1))
    vperp_diffusion = ((num_diss_params.ion.vperp_dissipation_coefficient > 0.0) || (collisions.fkpl.nuii > 0.0 && vperp.n > 1))
    vz_diffusion = (num_diss_params.neutral.vz_dissipation_coefficient > 0.0)

    if t_params.implicit_braginskii_conduction
        # if treating the electrons as a fluid with Braginskii closure, and using an IMEX
        # scheme, advance the conduction part of the electron energy equation implicitly.
        advance_electron_energy = false
        advance_electron_conduction = true
    end

    if (t_params.kinetic_electron_solver ∈ (implicit_time_evolving,
                                            implicit_p_implicit_pseudotimestep,
                                            implicit_steady_state,
                                            implicit_p_explicit_pseudotimestep))
        advance_electron_energy = true
        advance_electron_conduction = true
    end
    if t_params.kinetic_electron_solver == implicit_time_evolving
        advance_electron_pdf = true
    end

    manufactured_solns_test = false

    return advance_info(advance_vpa_advection, advance_vperp_advection, advance_z_advection, advance_r_advection,
                        advance_neutral_z_advection, advance_neutral_r_advection,
                        advance_neutral_vz_advection, advance_ion_cx, advance_neutral_cx,
                        advance_ion_cx_1V, advance_neutral_cx_1V, advance_ion_ionization,
                        advance_neutral_ionization, advance_ion_ionization_1V,
                        advance_neutral_ionization_1V, advance_krook_collisions_ii,
                        advance_maxwell_diffusion_ii, advance_maxwell_diffusion_nn,
                        explicit_weakform_fp_collisions,
                        advance_external_source, advance_ion_numerical_dissipation,
                        advance_neutral_numerical_dissipation, advance_sources,
                        advance_continuity, advance_force_balance, advance_energy,
                        advance_electron_pdf, advance_electron_energy,
                        advance_electron_conduction, advance_neutral_external_source,
                        advance_neutral_sources, advance_neutral_continuity,
                        advance_neutral_force_balance, advance_neutral_energy,
                        manufactured_solns_test, r_diffusion, vpa_diffusion,
                        vperp_diffusion, vz_diffusion)
end

function setup_dummy_and_buffer_arrays(nr, nz, nvpa, nvperp, nvz, nvr, nvzeta,
                                       nspecies_ion, nspecies_neutral, t_params)

    dummy_s = allocate_float(nspecies_ion)
    dummy_sr = allocate_float(nr, nspecies_ion)
    dummy_zrs = allocate_shared_float(nz, nr, nspecies_ion)
    dummy_zrsn = allocate_shared_float(nz, nr, nspecies_neutral)
    dummy_vpavperp = allocate_float(nvpa, nvperp)
    
    buffer_z_1 = allocate_shared_float(nz)
    buffer_z_2 = allocate_shared_float(nz)
    buffer_z_3 = allocate_shared_float(nz)
    buffer_z_4 = allocate_shared_float(nz)
    
    buffer_r_1 = allocate_shared_float(nr)
    buffer_r_2 = allocate_shared_float(nr)
    buffer_r_3 = allocate_shared_float(nr)
    buffer_r_4 = allocate_shared_float(nr)

    buffer_zs_1 = allocate_shared_float(nz,nspecies_ion)
    buffer_zs_2 = allocate_shared_float(nz,nspecies_ion)
    buffer_zs_3 = allocate_shared_float(nz,nspecies_ion)
    buffer_zs_4 = allocate_shared_float(nz,nspecies_ion)
    buffer_zsn_1 = allocate_shared_float(nz,nspecies_neutral)
    buffer_zsn_2 = allocate_shared_float(nz,nspecies_neutral)
    buffer_zsn_3 = allocate_shared_float(nz,nspecies_neutral)
    buffer_zsn_4 = allocate_shared_float(nz,nspecies_neutral)

    buffer_rs_1 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_2 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_3 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_4 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_5 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_6 = allocate_shared_float(nr,nspecies_ion)
    buffer_rsn_1 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_2 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_3 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_4 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_5 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_6 = allocate_shared_float(nr,nspecies_neutral)

    buffer_zrs_1 = allocate_shared_float(nz,nr,nspecies_ion)
    buffer_zrs_2 = allocate_shared_float(nz,nr,nspecies_ion)
    buffer_zrs_3 = allocate_shared_float(nz,nr,nspecies_ion)
    
    buffer_vpavperpzs_1 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_2 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_3 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_4 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_5 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_6 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)

    buffer_vpavperprs_1 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_2 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_3 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_4 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_5 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_6 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)

    buffer_vpavperpzrs_1 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
    buffer_vpavperpzrs_2 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)

    buffer_vpavperpzr_1 = allocate_shared_float(nvpa,nvperp,nz,nr)
    buffer_vpavperpzr_2 = allocate_shared_float(nvpa,nvperp,nz,nr)
    buffer_vpavperpzr_3 = allocate_shared_float(nvpa,nvperp,nz,nr)
    buffer_vpavperpzr_4 = allocate_shared_float(nvpa,nvperp,nz,nr)
    buffer_vpavperpzr_5 = allocate_shared_float(nvpa,nvperp,nz,nr)
    buffer_vpavperpzr_6 = allocate_shared_float(nvpa,nvperp,nz,nr)
    
    buffer_vpavperpr_1 = allocate_shared_float(nvpa,nvperp,nr)
    buffer_vpavperpr_2 = allocate_shared_float(nvpa,nvperp,nr)
    buffer_vpavperpr_3 = allocate_shared_float(nvpa,nvperp,nr)
    buffer_vpavperpr_4 = allocate_shared_float(nvpa,nvperp,nr)
    buffer_vpavperpr_5 = allocate_shared_float(nvpa,nvperp,nr)
    buffer_vpavperpr_6 = allocate_shared_float(nvpa,nvperp,nr)

    if t_params.kinetic_electron_solver ∈ (implicit_time_evolving,
                                           implicit_p_implicit_pseudotimestep,
                                           implicit_steady_state)
        implicit_buffer_z_1 = allocate_shared_float(nz)
        implicit_buffer_z_2 = allocate_shared_float(nz)
        implicit_buffer_z_3 = allocate_shared_float(nz)
        implicit_buffer_z_4 = allocate_shared_float(nz)
        implicit_buffer_z_5 = allocate_shared_float(nz)
        implicit_buffer_z_6 = allocate_shared_float(nz)

        implicit_buffer_vpavperpz_1 = allocate_shared_float(nvpa,nvperp,nz)
        implicit_buffer_vpavperpz_2 = allocate_shared_float(nvpa,nvperp,nz)
        implicit_buffer_vpavperpz_3 = allocate_shared_float(nvpa,nvperp,nz)
        implicit_buffer_vpavperpz_4 = allocate_shared_float(nvpa,nvperp,nz)
        implicit_buffer_vpavperpz_5 = allocate_shared_float(nvpa,nvperp,nz)
        implicit_buffer_vpavperpz_6 = allocate_shared_float(nvpa,nvperp,nz)
    else
        implicit_buffer_z_1 = allocate_shared_float(0)
        implicit_buffer_z_2 = allocate_shared_float(0)
        implicit_buffer_z_3 = allocate_shared_float(0)
        implicit_buffer_z_4 = allocate_shared_float(0)
        implicit_buffer_z_5 = allocate_shared_float(0)
        implicit_buffer_z_6 = allocate_shared_float(0)

        implicit_buffer_vpavperpz_1 = allocate_shared_float(0,0,0)
        implicit_buffer_vpavperpz_2 = allocate_shared_float(0,0,0)
        implicit_buffer_vpavperpz_3 = allocate_shared_float(0,0,0)
        implicit_buffer_vpavperpz_4 = allocate_shared_float(0,0,0)
        implicit_buffer_vpavperpz_5 = allocate_shared_float(0,0,0)
        implicit_buffer_vpavperpz_6 = allocate_shared_float(0,0,0)
    end

    if t_params.implicit_ion_advance
        implicit_buffer_vpavperpzrs_1 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
        implicit_buffer_vpavperpzrs_2 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
        implicit_buffer_vpavperpzrs_3 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
        implicit_buffer_vpavperpzrs_4 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
        implicit_buffer_vpavperpzrs_5 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
        implicit_buffer_vpavperpzrs_6 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
    else
        implicit_buffer_vpavperpzrs_1 = allocate_shared_float(0,0,0,0,0)
        implicit_buffer_vpavperpzrs_2 = allocate_shared_float(0,0,0,0,0)
        implicit_buffer_vpavperpzrs_3 = allocate_shared_float(0,0,0,0,0)
        implicit_buffer_vpavperpzrs_4 = allocate_shared_float(0,0,0,0,0)
        implicit_buffer_vpavperpzrs_5 = allocate_shared_float(0,0,0,0,0)
        implicit_buffer_vpavperpzrs_6 = allocate_shared_float(0,0,0,0,0)
    end

    buffer_vzvrvzetazsn_1 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_2 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_3 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_4 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_5 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_6 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)

    buffer_vzvrvzetarsn_1 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_2 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_3 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_4 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_5 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_6 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)

    buffer_vzvrvzetazrsn_1 = allocate_shared_float(nvz,nvr,nvzeta,nz,nr,nspecies_neutral)
    buffer_vzvrvzetazrsn_2 = allocate_shared_float(nvz,nvr,nvzeta,nz,nr,nspecies_neutral)
    
    buffer_vpavperp_1 = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_3 = allocate_shared_float(nvpa,nvperp)
    
    int_buffer_rs_1 = allocate_shared_int(nr,nspecies_ion)
    int_buffer_rs_2 = allocate_shared_int(nr,nspecies_ion)

    return scratch_dummy_arrays(dummy_s,dummy_sr,dummy_vpavperp,dummy_zrs,dummy_zrsn,
        buffer_z_1,buffer_z_2,buffer_z_3,buffer_z_4,
        buffer_r_1,buffer_r_2,buffer_r_3,buffer_r_4,
        buffer_zs_1,buffer_zs_2,buffer_zs_3,buffer_zs_4,
        buffer_zsn_1,buffer_zsn_2,buffer_zsn_3,buffer_zsn_4,
        buffer_rs_1,buffer_rs_2,buffer_rs_3,buffer_rs_4,buffer_rs_5,buffer_rs_6,
        buffer_rsn_1,buffer_rsn_2,buffer_rsn_3,buffer_rsn_4,buffer_rsn_5,buffer_rsn_6,
        buffer_zrs_1,buffer_zrs_2,buffer_zrs_3,
        buffer_vpavperpzs_1,buffer_vpavperpzs_2,buffer_vpavperpzs_3,buffer_vpavperpzs_4,buffer_vpavperpzs_5,buffer_vpavperpzs_6,
        buffer_vpavperprs_1,buffer_vpavperprs_2,buffer_vpavperprs_3,buffer_vpavperprs_4,buffer_vpavperprs_5,buffer_vpavperprs_6,
        buffer_vpavperpzrs_1,buffer_vpavperpzrs_2,
        implicit_buffer_z_1,implicit_buffer_z_2,implicit_buffer_z_3,implicit_buffer_z_4,implicit_buffer_z_5,implicit_buffer_z_6,
        implicit_buffer_vpavperpz_1,implicit_buffer_vpavperpz_2,implicit_buffer_vpavperpz_3,implicit_buffer_vpavperpz_4,implicit_buffer_vpavperpz_5,implicit_buffer_vpavperpz_6,
        implicit_buffer_vpavperpzrs_1,implicit_buffer_vpavperpzrs_2,implicit_buffer_vpavperpzrs_3,implicit_buffer_vpavperpzrs_4,implicit_buffer_vpavperpzrs_5,implicit_buffer_vpavperpzrs_6,
        buffer_vzvrvzetazsn_1,buffer_vzvrvzetazsn_2,buffer_vzvrvzetazsn_3,buffer_vzvrvzetazsn_4,buffer_vzvrvzetazsn_5,buffer_vzvrvzetazsn_6,
        buffer_vzvrvzetarsn_1,buffer_vzvrvzetarsn_2,buffer_vzvrvzetarsn_3,buffer_vzvrvzetarsn_4,buffer_vzvrvzetarsn_5,buffer_vzvrvzetarsn_6,
        buffer_vzvrvzetazrsn_1, buffer_vzvrvzetazrsn_2,
        buffer_vpavperp_1,buffer_vpavperp_2,buffer_vpavperp_3,
        buffer_vpavperpzr_1, buffer_vpavperpzr_2,buffer_vpavperpzr_3,buffer_vpavperpzr_4,buffer_vpavperpzr_5,buffer_vpavperpzr_6,
        buffer_vpavperpr_1, buffer_vpavperpr_2, buffer_vpavperpr_3, buffer_vpavperpr_4, buffer_vpavperpr_5, buffer_vpavperpr_6,
        int_buffer_rs_1,int_buffer_rs_2)

end

"""
if evolving the density via continuity equation, redefine the normalised f → f/n
if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
'scratch' should be a (nz,nspecies) array
"""
function normalize_pdf!(pdf, moments, scratch)
    error("Function normalise_pdf() has not been updated to be parallelized. Does not "
          * "seem to be used at the moment.")
    if moments.evolve_p
        @. scratch = moments.vth/moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    elseif moments.evolve_density
        @. scratch = 1.0 / moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    end
    return nothing
end

"""
create an array of structs containing scratch arrays for the normalised pdf and low-order moments
that may be evolved separately via fluid equations
"""
function setup_scratch_arrays(moments, pdf, n, time_evolve_electrons)
    # will create n structs, each of which will contain one pdf,
    # density, parallel flow, parallel pressure, and perpendicular pressure array for ions
    # (possibly) the same for electrons, and the same for neutrals. The actual array will
    # be created at the end of the first step of the loop below, once we have a
    # `scratch_pdf` object of the correct type.
    scratch = Vector{scratch_pdf}(undef, n)
    pdf_dims = size(pdf.ion.norm)
    moment_dims = size(moments.ion.dens)

    if time_evolve_electrons
        pdf_electron_dims = size(pdf.electron.norm)
    else
        pdf_electron_dims = (0,0,0,0)
    end
    moment_electron_dims = size(moments.electron.dens)

    pdf_neutral_dims = size(pdf.neutral.norm)
    moment_neutral_dims = size(moments.neutral.dens)
    # populate each of the structs
    for istage ∈ 1:n
        # Allocate arrays in temporary variables so that we can identify them
        # by source line when using @debug_shared_array
        pdf_array = allocate_shared_float(pdf_dims...)
        density_array = allocate_shared_float(moment_dims...)
        upar_array = allocate_shared_float(moment_dims...)
        p_array = allocate_shared_float(moment_dims...)
        temp_array = allocate_shared_float(moment_dims...)

        pdf_electron_array = allocate_shared_float(pdf_electron_dims...)
        density_electron_array = allocate_shared_float(moment_electron_dims...)
        upar_electron_array = allocate_shared_float(moment_electron_dims...)
        p_electron_array = allocate_shared_float(moment_electron_dims...)
        temp_electron_array = allocate_shared_float(moment_electron_dims...)

        pdf_neutral_array = allocate_shared_float(pdf_neutral_dims...)
        density_neutral_array = allocate_shared_float(moment_neutral_dims...)
        uz_neutral_array = allocate_shared_float(moment_neutral_dims...)
        p_neutral_array = allocate_shared_float(moment_neutral_dims...)

        ion_external_source_controller_integral =
            allocate_shared_float(size(moments.ion.external_source_controller_integral)...)
        #electron_external_source_controller_integral =
        #    allocate_shared_float(size(moments.electron.external_source_controller_integral)...)
        neutral_external_source_controller_integral =
            allocate_shared_float(size(moments.neutral.external_source_controller_integral)...)

        scratch[istage] = scratch_pdf(pdf_array, density_array, upar_array, p_array,
                                      ion_external_source_controller_integral,
                                      temp_array, pdf_electron_array,
                                      density_electron_array, upar_electron_array,
                                      p_electron_array, temp_electron_array,
                                      #electron_external_source_controller_integral,
                                      pdf_neutral_array, density_neutral_array,
                                      uz_neutral_array, p_neutral_array,
                                      neutral_external_source_controller_integral)
        @serial_region begin
            scratch[istage].pdf .= pdf.ion.norm
            scratch[istage].density .= moments.ion.dens
            scratch[istage].upar .= moments.ion.upar
            scratch[istage].p .= moments.ion.p
            scratch[istage].ion_external_source_controller_integral .=
                moments.ion.external_source_controller_integral

            if time_evolve_electrons
                scratch[istage].pdf_electron .= pdf.electron.norm
            end
            scratch[istage].electron_density .= moments.electron.dens
            scratch[istage].electron_upar .= moments.electron.upar
            scratch[istage].electron_p .= moments.electron.p
            #scratch[istage].electron_external_source_controller_integral .=
            #    moments.electron.external_source_controller_integral

            scratch[istage].pdf_neutral .= pdf.neutral.norm
            scratch[istage].density_neutral .= moments.neutral.dens
            scratch[istage].uz_neutral .= moments.neutral.uz
            scratch[istage].p_neutral .= moments.neutral.p
            scratch[istage].neutral_external_source_controller_integral .=
                moments.neutral.external_source_controller_integral
        end
    end
    return scratch
end

function setup_electron_scratch_arrays(moments, pdf, n)
    # will create n structs, each of which will contain one pdf, and parallel pressure
    # array for electrons.
    # The actual array will be created at the end of the first step of the loop below,
    # once we have a `scratch_electron_pdf` object of the correct type.
    scratch = Vector{scratch_electron_pdf}(undef, n)
    pdf_dims = size(pdf.electron.norm)
    moment_dims = size(moments.electron.dens)

    # populate each of the structs
    for istage ∈ 1:n
        # Allocate arrays in temporary variables so that we can identify them
        # by source line when using @debug_shared_array
        pdf_array = allocate_shared_float(pdf_dims...)
        p_array = allocate_shared_float(moment_dims...)

        scratch[istage] = scratch_electron_pdf(pdf_array, p_array)
        @serial_region begin
            scratch[istage].pdf_electron .= pdf.electron.norm
            scratch[istage].electron_p .= moments.electron.p
        end
    end
    return scratch
end

"""
solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
define approximate characteristic velocity
v₀(z)=vⁿ(z) and take time derivative along this characteristic
df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
for prudent choice of v₀, expect δv≪v so that explicit
time integrator can be used without severe CFL condition
"""
function time_advance!(pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr,
                       vzeta, vpa, vperp, gyrophase, z, r, moments, fields,
                       spectral_objects, advect_objects, composition, collisions,
                       geometry, gyroavs, boundaries, external_source_settings,
                       num_diss_params, nl_solver_params, advance, advance_implicit,
                       fp_arrays, scratch_dummy, manufactured_source_list, ascii_io,
                       io_moments, io_dfns)

    @debug_detect_redundant_block_synchronize begin
        # Only want to check for redundant _block_synchronize() calls during the
        # time advance loop, so activate these checks here
        debug_detect_redundant_is_active[] = true
    end

    if isfile(t_params.stopfile)
        if filesize(t_params.stopfile) > 0
            error("Found a 'stop file' at $(t_params.stopfile), but it contains some data "
                  * "(file size is greater than zero), so will not delete.")
        end
        if global_rank[] == 0
            rm(t_params.stopfile)
        end
    end
    if isfile(t_params.stopfile * "now") && global_rank[] == 0
        rm(t_params.stopfile * "now")
    end

    @serial_region begin
        if global_rank[] == 0
             println("beginning time advance   ", Dates.format(now(), dateformat"H:MM:SS"))
             flush(stdout)
        end
    end

    start_time = now()

    epsilon = 1.e-11

    # main time advance loop
    finish_now = false
    t_params.step_counter[] = 1
    if t_params.t[] ≥ t_params.end_time - epsilon
        # User must have requested zero output steps, i.e. to just write out the initial
        # profiles
        return nothing
    end
    while true
        @timeit global_timer "time_advance! step" begin
            if t_params.adaptive && !t_params.write_after_fixed_step_count
                maybe_write_moments = t_params.step_to_moments_output[]
                maybe_write_dfns = t_params.step_to_dfns_output[]
            else
                maybe_write_moments = (t_params.step_counter[] % t_params.nwrite_moments == 0
                                       || t_params.step_counter[] >= t_params.nstep)
                maybe_write_dfns = (t_params.step_counter[] % t_params.nwrite_dfns == 0
                                    || t_params.step_counter[] >= t_params.nstep)
            end
            diagnostic_checks = (maybe_write_moments || maybe_write_dfns)

            if t_params.split_operators
                # MRH NOT SUPPORTED
                time_advance_split_operators!(pdf, scratch, scratch_implicit,
                                              scratch_electron, t_params, vpa, z,
                                              vpa_spectral, z_spectral, moments, fields,
                                              vpa_advect, z_advect, composition, collisions,
                                              external_source_settings, num_diss_params,
                                              nl_solver_params, advance, advance_implicit,
                                              t_params.step_counter[])
            else
                time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                                           t_params, vz, vr, vzeta, vpa, vperp, gyrophase,
                                           z, r, moments, fields, spectral_objects,
                                           advect_objects, composition, collisions, geometry,
                                           gyroavs, boundaries, external_source_settings,
                                           num_diss_params, nl_solver_params, advance,
                                           advance_implicit, fp_arrays, scratch_dummy,
                                           manufactured_source_list, diagnostic_checks,
                                           t_params.step_counter[])
            end
            # update the time
            t_params.t[] += t_params.previous_dt[]

            if t_params.t[] ≥ t_params.end_time - epsilon ||
                    (t_params.write_after_fixed_step_count &&
                     t_params.step_counter[] >= t_params.nstep)
                # Ensure all output is written at the final step
                finish_now = true
            elseif t_params.dt[] < 0.0 || isnan(t_params.dt[]) || isinf(t_params.dt[])
                # Negative t_params.dt[] indicates the time stepping has failed, so stop and
                # write output.
                # t_params.dt[] should never be NaN or Inf, so if it is something has gone
                # wrong.
                println("dt=", t_params.dt[], " at t=", t_params.t[], ", terminating run.")
                finish_now = true
            end

            if isfile(t_params.stopfile * "now")
                # Stop cleanly if a file called 'stop' was created
                println("Found 'stopnow' file $(t_params.stopfile * "now"), aborting run")
                finish_now = true
                t_params.dt_before_output[] = t_params.dt[]
            end

            if t_params.adaptive && !t_params.write_after_fixed_step_count
                write_moments = t_params.write_moments_output[] || finish_now
                write_dfns = t_params.write_dfns_output[] || finish_now

                t_params.write_moments_output[] = false
                t_params.write_dfns_output[] = false
            else
                write_moments = (t_params.step_counter[] % t_params.nwrite_moments == 0
                                 || t_params.step_counter[] >= t_params.nstep
                                 || finish_now)
                write_dfns = (t_params.step_counter[] % t_params.nwrite_dfns == 0
                              || t_params.step_counter[] >= t_params.nstep
                              || finish_now)
            end
            if write_moments
                t_params.moments_output_counter[] += 1
                if !t_params.exact_output_times
                    while (t_params.moments_output_counter[] ≤ length(t_params.moments_output_times)
                           && t_params.moments_output_times[t_params.moments_output_counter[]] ≤ t_params.t[])
                        t_params.moments_output_counter[] += 1
                    end
                end
            end
            if write_dfns
                t_params.dfns_output_counter[] += 1
                if !t_params.exact_output_times
                    while (t_params.dfns_output_counter[] ≤ length(t_params.dfns_output_times)
                           && t_params.dfns_output_times[t_params.dfns_output_counter[]] ≤ t_params.t[])
                        t_params.dfns_output_counter[] += 1
                    end
                end
            end

            if write_moments || write_dfns || finish_now
                # Always synchronise here, regardless of if we changed region or not
                @begin_serial_region(true)
                @_block_synchronize()

                if isfile(t_params.stopfile)
                    # Stop cleanly if a file called 'stop' was created
                    println("Found 'stop' file $(t_params.stopfile), aborting run")
                    flush(stdout)
                    finish_now = true
                end

                # If NaNs are present, they should propagate into every field, so only need to
                # check one. Choose phi because it is small (it has no species or velocity
                # dimensions). If a NaN is found, stop the simulation.
                if block_rank[] == 0
                    if any(isnan.(fields.phi))
                        println("Found NaN, stopping simulation")
                        found_nan = 1
                    else
                        found_nan = 0
                    end
                    found_nan = MPI.Allreduce(found_nan, +, comm_inter_block[])
                else
                    found_nan = 0
                end
                found_nan = MPI.Bcast(found_nan, 0, comm_block[])
                if found_nan != 0
                    finish_now = true
                end

                # Do MPI communication to add up counters from different processes, where
                # necessary.
                gather_nonlinear_solver_counters!(nl_solver_params)

                time_for_run = to_minutes(now() - start_time)
            end
            # write moments data to file
            if write_moments || finish_now
                @debug_detect_redundant_block_synchronize begin
                    # Skip check for redundant _block_synchronize() during file I/O because
                    # it only runs infrequently
                    debug_detect_redundant_is_active[] = false
                end
                @begin_serial_region()
                @serial_region begin
                    if global_rank[] == 0
                        print("writing moments output ",
                              rpad(string(t_params.moments_output_counter[]), 4), "  ",
                              "t = ", rpad(string(round(t_params.t[], sigdigits=6)), 7), "  ",
                              "nstep = ", rpad(string(t_params.step_counter[]), 7), "  ")
                        if t_params.print_nT_live
                            midpoint = Int64(round((1+size(moments.ion.dens)[1])/2))
                            print("midpoint density: ", 
                            rpad(string(round(moments.ion.dens[midpoint,1,1], sigdigits = 8)), 7))
                            print("   midpoint temperature: ", 
                            rpad(string(round(moments.ion.temp[midpoint,1,1], sigdigits = 8)), 7), "\n")
                        end
                        if t_params.adaptive
                            print("nfail = ", rpad(string(t_params.failure_counter[]), 7), "  ",
                                  "dt = ", rpad(string(t_params.dt_before_output[]), 7), "  ")
                        end
                        print(Dates.format(now(), dateformat"H:MM:SS"))
                    end
                end
                write_data_to_ascii(pdf, moments, fields, vz, vr, vzeta, vpa, vperp, z, r,
                                    t_params.t[], composition.n_ion_species,
                                    composition.n_neutral_species, ascii_io)
                write_all_moments_data_to_binary(scratch, moments, fields,
                                                 composition.n_ion_species,
                                                 composition.n_neutral_species, io_moments,
                                                 t_params.moments_output_counter[], time_for_run, t_params,
                                                 nl_solver_params, r, z)

                if t_params.steady_state_residual
                    # Calculate some residuals to see how close simulation is to steady state
                    @begin_r_z_region()
                    result_string = ""
                    all_residuals = Vector{mk_float}()
                    @loop_s is begin
                        @views residual_ni =
                            steady_state_residuals(scratch[t_params.n_rk_stages+1].density[:,:,is],
                                                   scratch[1].density[:,:,is], t_params.previous_dt[];
                                                   use_mpi=true, only_max_abs=true)
                        if global_rank[] == 0
                            residual_ni = first(values(residual_ni))[1]
                            push!(all_residuals, residual_ni)
                            result_string *= "  density "
                            result_string *= rpad(string(round(residual_ni; sigdigits=4)), 11)
                        end
                    end
                    if composition.n_neutral_species > 0
                        @loop_sn isn begin
                            residual_nn =
                                steady_state_residuals(scratch[t_params.n_rk_stages+1].density_neutral[:,:,isn],
                                                       scratch[1].density_neutral[:,:,isn],
                                                       t_params.previous_dt[]; use_mpi=true,
                                                       only_max_abs=true)
                            if global_rank[] == 0
                                residual_nn = first(values(residual_nn))[1]
                                push!(all_residuals, residual_nn)
                                result_string *= " density_neutral "
                                result_string *= rpad(string(round(residual_nn; sigdigits=4)), 11)
                            end
                        end
                    end
                    if global_rank[] == 0
                        println("    residuals:", result_string)
                        flush(stdout)
                    end
                    if t_params.converged_residual_value > 0.0
                        if global_rank[] == 0
                            if all(r < t_params.converged_residual_value for r ∈ all_residuals)
                                println("Run converged! All tested residuals less than ",
                                        t_params.converged_residual_value)
                                flush(stdout)
                                finish_now = true
                            end
                        end
                        finish_now = MPI.Bcast(finish_now, 0, comm_world)
                    end
                else
                    if global_rank[] == 0
                        println()
                        flush(stdout)
                    end
                end

                @begin_s_r_z_vperp_region()
                @debug_detect_redundant_block_synchronize begin
                    # Reactivate check for redundant _block_synchronize()
                    debug_detect_redundant_is_active[] = true
                end
            end
            if write_dfns || finish_now
                @debug_detect_redundant_block_synchronize begin
                    # Skip check for redundant _block_synchronize() during file I/O because
                    # it only runs infrequently
                    debug_detect_redundant_is_active[] = false
                end
                @begin_serial_region()
                @serial_region begin
                    if global_rank[] == 0
                        println("writing distribution functions output ",
                                rpad(string(t_params.dfns_output_counter[]), 4), "  ",
                                "t = ", rpad(string(round(t_params.t[], sigdigits=6)), 7), "  ",
                                "nstep = ", rpad(string(t_params.step_counter[]), 7), "  ",
                                Dates.format(now(), dateformat"H:MM:SS"))
                        flush(stdout)
                    end
                end
                write_all_dfns_data_to_binary(scratch, scratch_electron, moments, fields,
                                              composition.n_ion_species,
                                              composition.n_neutral_species, io_dfns,
                                              t_params.dfns_output_counter[], time_for_run,
                                              t_params, nl_solver_params, r, z, vperp, vpa,
                                              vzeta, vr, vz)
                @begin_s_r_z_vperp_region()
                @debug_detect_redundant_block_synchronize begin
                    # Reactivate check for redundant _block_synchronize()
                    debug_detect_redundant_is_active[] = true
                end
            end

            if t_params.previous_dt[] == 0.0
                # Timestep failed, so reset  scratch[t_params.n_rk_stages+1] equal to
                # scratch[1] to start the timestep over.
                scratch_temp = scratch[t_params.n_rk_stages+1]
                scratch[t_params.n_rk_stages+1] = scratch[1]
                scratch[1] = scratch_temp

                # Re-update remaining velocity moments that are calculable from the evolved
                # pdf These need to be re-calculated because `scratch[istage+1]` is now the
                # state at the beginning of the timestep, because the timestep failed
                apply_all_bcs_constraints_update_moments!(
                    scratch[t_params.n_rk_stages+1], pdf, moments, fields, nothing, nothing, vz,
                    vr, vzeta, vpa, vperp, z, r, spectral_objects, advect_objects, composition,
                    collisions, geometry, gyroavs, external_source_settings, num_diss_params,
                    t_params, nl_solver_params, advance, scratch_dummy, false, 0, 0.0;
                    pdf_bc_constraints=false, update_electrons=false)
            end

            if finish_now
                break
            end
            if t_params.adaptive
                if t_params.t[] >= t_params.end_time - epsilon
                    break
                end
            else
                if t_params.step_counter[] >= t_params.nstep
                    break
                end
            end

            t_params.step_counter[] += 1
        end
    end
    return nothing
end

"""
"""
function time_advance_split_operators!(pdf, scratch, scratch_implicit, scratch_electron,
                                       t_params, vpa, z, vpa_spectral, z_spectral,
                                       moments, fields, vpa_advect, z_advect, composition,
                                       collisions, external_source_settings,
                                       num_diss_params, nl_solver_params, advance,
                                       advance_implicit, istep)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    n_rk_stages = t_params.n_rk_stages
    # to ensure 2nd order accuracy in time for operator-split advance,
    # have to reverse order of operations every other time step
    flipflop = (mod(istep,2)==0)
    if flipflop
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for ion species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
            t_params, vpa, z, vpa_spectral, z_spectral, moments, fields, vpa_advect,
            z_advect, composition, collisions, external_source_settings, num_diss_params,
            nl_solver_params, advance, advance_implicit, istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (ion and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
            t_params, vpa, z, vpa_spectral, z_spectral, moments, fields, vpa_advect,
            z_advect, composition, collisions, external_source_settings, num_diss_params,
            nl_solver_params, advance, advance_implicit, istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.reactions.charge_exchange_frequency > 0.0
                advance.ion_cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, vpa, z, vpa_spectral, z_spectral,
                    moments, fields, vpa_advect, z_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.ion_cx_collisions = false
                advance.neutral_cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, vpa, z, vpa_spectral, z_spectral,
                    moments, fields, vpa_advect, z_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.neutral_cx_collisions = false
            end
            if collisions.reactions.ionization_frequency > 0.0
                advance.ion_ionization_collisions = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, z, vpa, z_spectral, vpa_spectral,
                    moments, fields, z_advect, vpa_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.ion_ionization_collisions = false
                advance.neutral_ionization_collisions = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, z, vpa, z_spectral, vpa_spectral,
                    moments, fields, z_advect, vpa_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.neutral_ionization_collisions = false
            end
        end
        if collisions.krook.nuii0  > 0.0
            advance.krook_collisions_ii = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, z, vpa, z_spectral, vpa_spectral, moments, fields, z_advect,
                vpa_advect, z_SL, vpa_SL, composition, collisions, sources,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.krook_collisions_ii = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_p
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.source_terms = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.continuity = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.force_balance = false
        end
        # use the energy equation to update the parallel pressure
        if moments.evolve_p
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.energy = false
        end
    else
        # use the energy equation to update the parallel pressure
        if moments.evolve_p
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.energy = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.force_balance = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.continuity = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_p
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                t_params, vpa, z, vpa_spectral, z_spectral, moments, fields,
                vpa_advect, z_advect, composition, collisions, external_source_settings,
                num_diss_params, nl_solver_params, advance, advance_implicit, istep)
            advance.source_terms = false
        end
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.reactions.ionization_frequency > 0.0
                advance.neutral_ionization = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, z, vpa, z_spectral, vpa_spectral,
                    moments, fields, z_advect, vpa_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.neutral_ionization = false
                advance.ion_ionization = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, z, vpa, z_spectral, vpa_spectral,
                    moments, fields, z_advect, vpa_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.ion_ionization = false
            end
            if collisions.reactions.charge_exchange_frequency > 0.0
                advance.neutral_cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, vpa, z, vpa_spectral, z_spectral,
                    moments, fields, vpa_advect, z_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.neutral_cx_collisions = false
                advance.ion_cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, scratch_implicit,
                    scratch_electron, t_params, vpa, z, vpa_spectral, z_spectral,
                    moments, fields, vpa_advect, z_advect, composition, collisions,
                    external_source_settings, num_diss_params, nl_solver_params, advance,
                    advance_implicit, istep)
                advance.ion_cx_collisions = false
            end
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (ion and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
            t_params, vpa, z, vpa_spectral, z_spectral, moments, fields, vpa_advect,
            z_advect, composition, collisions, external_source_settings, num_diss_params,
            nl_solver_params, advance, advance_implicit, istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for ion species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
            t_params, vpa, z, vpa_spectral, z_spectral, moments, fields, vpa_advect,
            z_advect, composition, collisions, external_source_settings, num_diss_params,
            nl_solver_params, advance, advance_implicit, istep)
        advance.vpa_advection = false
    end
    return nothing
end

"""
"""
function time_advance_no_splitting!(pdf, scratch, scratch_implicit, scratch_electron,
                                    t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
                                    moments, fields, spectral_objects, advect_objects,
                                    composition, collisions, geometry, gyroavs,
                                    boundaries, external_source_settings, num_diss_params,
                                    nl_solver_params, advance, advance_implicit,
                                    fp_arrays, scratch_dummy, manufactured_source_list,
                                    diagnostic_checks, istep)

    ssp_rk!(pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta,
            vpa, vperp, gyrophase, z, r, moments, fields, spectral_objects,
            advect_objects, composition, collisions, geometry, gyroavs,
            boundaries, external_source_settings, num_diss_params, nl_solver_params,
            advance, advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            diagnostic_checks, istep)

    return nothing
end

"""
Use the result of the forward-Euler timestep and the previous Runge-Kutta stages to
compute the updated pdfs, and any evolved moments.
"""
function rk_update!(scratch, scratch_implicit, moments, t_params, istage, composition)
    @begin_s_r_z_region()

    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]
    rk_coefs = t_params.rk_coefs[:,istage]

    ##
    # update the ion distribution and moments
    ##
    # here we seem to have duplicate arrays for storing n, u||, p||, etc, but not for vth
    # 'scratch' is for the multiple stages of time advanced quantities, but 'moments' can be updated directly at each stage
    rk_update_variable!(scratch, scratch_implicit, :pdf, t_params, istage)
    # use Runge Kutta to update any velocity moments evolved separately from the pdf
    rk_update_evolved_moments!(scratch, scratch_implicit, moments, t_params, istage)

    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        rk_update_variable!(scratch, scratch_implicit, :electron_p, t_params, istage)
    end

    if composition.n_neutral_species > 0
        ##
        # update the neutral particle distribution and moments
        ##
        rk_update_variable!(scratch, scratch_implicit, :pdf_neutral, t_params, istage; neutrals=true)
        # use Runge Kutta to update any velocity moments evolved separately from the pdf
        rk_update_evolved_moments_neutral!(scratch, scratch_implicit, moments, t_params, istage)
    end
end

"""
Apply boundary conditions and moment constraints to updated pdfs and calculate derived
moments and moment derivatives
"""
@timeit global_timer apply_all_bcs_constraints_update_moments!(
                         this_scratch, pdf, moments, fields, boundaries, scratch_electron,
                         vz, vr, vzeta, vpa, vperp, z, r, spectral_objects,
                         advect_objects, composition, collisions, geometry, gyroavs,
                         external_source_settings, num_diss_params, t_params,
                         nl_solver_params, advance, scratch_dummy, diagnostic_moments,
                         max_electron_pdf_iterations, max_electron_sim_time;
                         pdf_bc_constraints=true, update_electrons=true) = begin

    @begin_s_r_z_region()

    z_spectral, r_spectral, vpa_spectral, vperp_spectral = spectral_objects.z_spectral, spectral_objects.r_spectral, spectral_objects.vpa_spectral, spectral_objects.vperp_spectral
    vzeta_spectral, vr_spectral, vz_spectral = spectral_objects.vzeta_spectral, spectral_objects.vr_spectral, spectral_objects.vz_spectral
    vpa_advect, vperp_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.vperp_advect, advect_objects.r_advect, advect_objects.z_advect
    electron_z_advect, electron_vpa_advect = advect_objects.electron_z_advect, advect_objects.electron_vpa_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    success = ""

    if pdf_bc_constraints
        # Ensure there are no negative values in the pdf before applying boundary
        # conditions, so that negative deviations do not mess up the integral-constraint
        # corrections in the sheath boundary conditions.
        force_minimum_pdf_value!(this_scratch.pdf, num_diss_params.ion.force_minimum_pdf_value)

        # Enforce boundary conditions in z and vpa on the distribution function.
        # Must be done after Runge Kutta update so that the boundary condition applied to the
        # updated pdf is consistent with the updated moments - otherwise different upar
        # between 'pdf', 'scratch[istage]' and 'scratch[istage+1]' might mean a point that
        # should be set to zero at the sheath boundary according to the final upar has a
        # non-zero contribution from one or more of the terms.  NB: probably need to do the
        # same for the evolved moments
        enforce_boundary_conditions!(this_scratch, moments, fields, boundaries, vpa,
                                     vperp, z, r, vpa_spectral, vperp_spectral,
                                     vpa_advect, vperp_advect, z_advect, r_advect,
                                     composition, scratch_dummy, advance.r_diffusion,
                                     advance.vpa_diffusion, advance.vperp_diffusion)

        if moments.evolve_density && moments.enforce_conservation
            hard_force_moment_constraints!(this_scratch.pdf, moments, vpa, vperp)
        end

        if (composition.electron_physics ∈ (kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
                && length(this_scratch.pdf_electron) > 0)

            for ir ∈ 1:r.n
                @views apply_electron_bc_and_constraints_no_r!(
                           this_scratch.pdf_electron[:,:,:,ir], fields.phi[:,ir], moments,
                           r, z, vperp, vpa, vperp_spectral, vpa_spectral,
                           electron_vpa_advect, num_diss_params, composition, ir,
                           nl_solver_params.electron_advance)
            end
        end
    end

    # update remaining velocity moments that are calculable from the evolved pdf
    # Note these may be needed for the boundary condition on the neutrals, so must be
    # calculated before that is applied. Also may be needed to calculate advection speeds
    # for for CFL stability limit calculations in adaptive_timestep_update!().
    if composition.ion_physics ∈ (drift_kinetic_ions, gyrokinetic_ions)
        update_derived_moments!(this_scratch, moments, vpa, vperp, z, r, composition,
            r_spectral, geometry, gyroavs, scratch_dummy, z_advect, collisions, diagnostic_moments)
    else
        update_derived_moments!(this_scratch, moments, vpa, vperp, z, r, composition,
            r_spectral, geometry, gyroavs, scratch_dummy, z_advect, collisions, false)
    end

    calculate_ion_moment_derivatives!(moments, this_scratch, scratch_dummy, z, z_spectral,
                                      num_diss_params.ion.moment_dissipation_coefficient)

    calculate_electron_moments!(this_scratch, pdf, moments, composition, collisions, r, z,
                                vperp, vpa)
    calculate_electron_moment_derivatives!(moments, this_scratch, scratch_dummy, z,
                                           z_spectral,
                                           num_diss_params.electron.moment_dissipation_coefficient, 
                                           composition.electron_physics)
    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)

        # Copy ion and electron moments from `scratch` into `moments` to be used in
        # electron kinetic equation update
        @begin_r_z_region()
        @loop_s_r_z is ir iz begin
            moments.ion.dens[iz,ir,is] = this_scratch.density[iz,ir,is]
            moments.ion.upar[iz,ir,is] = this_scratch.upar[iz,ir,is]
            moments.ion.p[iz,ir,is] = this_scratch.p[iz,ir,is]
        end
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = this_scratch.density_neutral[iz,ir,isn]
            moments.neutral.uz[iz,ir,isn] = this_scratch.uz_neutral[iz,ir,isn]
            moments.neutral.p[iz,ir,isn] = this_scratch.p_neutral[iz,ir,isn]
        end
        @loop_r_z ir iz begin
            moments.electron.dens[iz,ir] = this_scratch.electron_density[iz,ir]
            moments.electron.upar[iz,ir] = this_scratch.electron_upar[iz,ir]
            moments.electron.p[iz,ir] = this_scratch.electron_p[iz,ir]
        end

        # When we do not need to apply bc's and constraints to the ion/neutral pdf
        # (because this function is being called after a failed timestep, to reset to the
        # state at the beginning of the step), we also do not need to update the
        # electrons.
        # Note that if some solve for the implicit timestep already failed, we will reset
        # to the beginning of the ion/neutral timestep, so the electron solution
        # calculated here would be discarded - we might as well skip calculating it in
        # that case.
        if update_electrons && success == ""
            kinetic_electron_success = update_electron_pdf!(
               scratch_electron, pdf.electron.norm, moments, fields.phi, r, z, vperp, vpa,
               z_spectral, vperp_spectral, vpa_spectral, electron_z_advect,
               electron_vpa_advect, scratch_dummy, t_params.electron, collisions,
               composition, external_source_settings, num_diss_params,
               nl_solver_params.electron_advance, max_electron_pdf_iterations,
               max_electron_sim_time, solution_method="artificial_time_derivative")
            success = kinetic_electron_success
        end
    end
    # update the electron parallel friction force
    calculate_electron_parallel_friction_force!(
        moments.electron.parallel_friction, this_scratch.electron_density,
        this_scratch.electron_upar, this_scratch.upar, moments.electron.dT_dz,
        composition.me_over_mi, collisions.electron_fluid.nu_ei,
        composition.electron_physics)

    # update the electrostatic potential phi
    update_phi!(fields, this_scratch, vperp, z, r, composition, collisions, moments,
                geometry, z_spectral, r_spectral, scratch_dummy, gyroavs)

    if composition.n_neutral_species > 0
        if pdf_bc_constraints
            # Ensure there are no negative values in the pdf before applying boundary
            # conditions, so that negative deviations do not mess up the integral-constraint
            # corrections in the sheath boundary conditions.
            force_minimum_pdf_value_neutral!(this_scratch.pdf_neutral,
                                             num_diss_params.neutral.force_minimum_pdf_value)

            # Enforce boundary conditions in z and vpa on the distribution function.
            # Must be done after Runge Kutta update so that the boundary condition applied to
            # the updated pdf is consistent with the updated moments - otherwise different
            # upar between 'pdf', 'scratch[istage]' and 'scratch[istage+1]' might mean a point
            # that should be set to zero at the sheath boundary according to the final upar
            # has a non-zero contribution from one or more of the terms.  NB: probably need to
            # do the same for the evolved moments Note, so far vr and vzeta do not need advect
            # objects, so pass `nothing` for those as a placeholder
            enforce_neutral_boundary_conditions!(this_scratch.pdf_neutral,
                this_scratch.pdf, this_scratch.density_neutral, this_scratch.uz_neutral,
                this_scratch.p_neutral, boundaries, moments, this_scratch.density,
                this_scratch.upar, fields.Er, vzeta_spectral, vr_spectral, vz_spectral,
                neutral_r_advect, neutral_z_advect, nothing, nothing, neutral_vz_advect,
                r, z, vzeta, vr, vz, composition, geometry, scratch_dummy,
                advance.r_diffusion, advance.vz_diffusion)

            if moments.evolve_density && moments.enforce_conservation
                hard_force_moment_constraints_neutral!(this_scratch.pdf_neutral, moments,
                                                       vz)
            end
        end

        # update remaining velocity moments that are calculable from the evolved pdf
        update_derived_moments_neutral!(this_scratch, moments, vz, vr, vzeta, z, r,
                                        composition)
        # update the thermal speed
        @begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.vth[iz,ir,isn] = sqrt(2.0*this_scratch.p_neutral[iz,ir,isn]/this_scratch.density_neutral[iz,ir,isn])
        end

        # update the parallel heat flux
        update_neutral_qz!(moments.neutral.qz, moments.neutral.qz_updated,
                           this_scratch.density_neutral, this_scratch.uz_neutral,
                           moments.neutral.vth, this_scratch.pdf_neutral, vz, vr, vzeta, z,
                           r, composition, moments.evolve_density, moments.evolve_upar,
                           moments.evolve_p)

        calculate_neutral_moment_derivatives!(moments, this_scratch, scratch_dummy, z,
                                              z_spectral,
                                              num_diss_params.neutral.moment_dissipation_coefficient)
    end

    return success
end

"""
    adaptive_timestep_update!(scratch, scratch_implicit, scratch_electron,
                              t_params, pdf, moments, fields, boundaries, composition,
                              collisions, geometry, external_source_settings,
                              spectral_objects, advect_objects, gyroavs, num_diss_params,
                              nl_solver_params, advance, scratch_dummy, r, z, vperp, vpa,
                              vzeta, vr, vz, success, nl_max_its_fraction,
                              nl_total_its_soft_limit, nl_total_its_soft_limit_reduce_dt)

Check the error estimate for the embedded RK method and adjust the timestep if
appropriate.
"""
@timeit global_timer adaptive_timestep_update!(
                         scratch, scratch_implicit, scratch_electron, t_params, pdf,
                         moments, fields, boundaries, composition, collisions, geometry,
                         external_source_settings, spectral_objects, advect_objects,
                         gyroavs, num_diss_params, nl_solver_params, advance,
                         scratch_dummy, r, z, vperp, vpa, vzeta, vr, vz, success,
                         nl_max_its_fraction, nl_total_its_soft_limit,
                         nl_total_its_soft_limit_reduce_dt) = begin
    #error_norm_method = "Linf"
    error_norm_method = "L2"

    error_coeffs = t_params.rk_coefs[:,end]
    if t_params.n_rk_stages < 3
        # This should never happen as an adaptive RK scheme needs at least 2 RHS evals so
        # (with the pre-timestep data) there must be at least 3 entries in `scratch`.
        error("adaptive timestep needs a buffer scratch array")
    end

    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    vpa_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.r_advect, advect_objects.z_advect
    electron_z_advect, electron_vpa_advect = advect_objects.electron_z_advect, advect_objects.electron_vpa_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect
    evolve_density, evolve_upar, evolve_p = moments.evolve_density, moments.evolve_upar, moments.evolve_p

    CFL_limits = OrderedDict{String,mk_float}()
    error_norm_type = typeof(t_params.error_sum_zero)
    error_norms = OrderedDict{String,error_norm_type}()
    total_points = mk_int[]

    # Test CFL conditions for advection in kinetic equation to give stability limit for
    # timestep
    #
    # ion z-advection
    # No need to synchronize here, as we just called @_block_synchronize()
    # Don't parallelise over species here, because get_minimum_CFL_*() does an MPI
    # reduction over the shared-memory block, so all processes must calculate the same
    # species at the same time.
    @begin_r_vperp_vpa_region(true)
    if !t_params.implicit_ion_advance
        ion_z_CFL = Inf
        @loop_s is begin
            update_speed_z!(z_advect[is], moments.ion.upar, moments.ion.vth, evolve_upar,
                            evolve_p, fields, vpa, vperp, z, r, t_params.t[], geometry,
                            is)
            this_minimum = get_minimum_CFL_z(z_advect[is].speed, z)
            @serial_region begin
                ion_z_CFL = min(ion_z_CFL, this_minimum)
            end
        end
        CFL_limits["CFL_z"] = t_params.CFL_prefactor * ion_z_CFL
    end

    if !(t_params.implicit_ion_advance || t_params.implicit_vpa_advection)
        # ion vpa-advection
        @begin_r_z_vperp_region()
        ion_vpa_CFL = Inf
        update_speed_vpa!(vpa_advect, fields, scratch[t_params.n_rk_stages+1], moments, vpa, vperp, z, r,
                          composition, collisions, external_source_settings.ion, t_params.t[],
                          geometry)
        @loop_s is begin
            this_minimum = get_minimum_CFL_vpa(vpa_advect[is].speed, vpa)
            @serial_region begin
                ion_vpa_CFL = min(ion_vpa_CFL, this_minimum)
            end
        end
        CFL_limits["CFL_vpa"] = t_params.CFL_prefactor * ion_vpa_CFL
    end

    if t_params.kinetic_electron_solver == explicit_time_evolving
        # Need to check electron CFL limits
        @begin_r_vperp_vpa_region()
        update_electron_speed_z!(electron_z_advect[1], moments.electron.upar,
                                 moments.electron.vth, vpa.grid)
        electron_z_CFL = get_minimum_CFL_z(electron_z_advect[1].speed, z)
        if block_rank[] == 0
            CFL_limits["electron_CFL_z"] = t_params.CFL_prefactor * electron_z_CFL
        else
            CFL_limits["electron_CFL_z"] = Inf
        end

        @begin_r_z_vperp_region()
        update_electron_speed_vpa!(electron_vpa_advect[1], moments.electron.dens,
                                   moments.electron.upar,
                                   scratch[t_params.n_rk_stages+1].electron_p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron)
        electron_vpa_CFL = get_minimum_CFL_vpa(electron_vpa_advect[1].speed, vpa)
        if block_rank[] == 0
            CFL_limits["electron_CFL_vpa"] = t_params.CFL_prefactor * electron_vpa_CFL
        else
            CFL_limits["electron_CFL_vpa"] = Inf
        end
    end

    # To avoid double counting points when we use distributed-memory MPI, skip the
    # inner/lower point in r and z if this process is not the first block in that
    # dimension.
    skip_r_inner = r.irank != 0
    skip_z_lower = z.irank != 0

    # Calculate low-order approximations, from which the timestep error can be estimated.
    # Note we store the calculated low-order approxmation in `scratch[2]`.
    rk_loworder_solution!(scratch, scratch_implicit, :pdf, t_params)
    if moments.evolve_density
        @begin_s_r_z_region()
        rk_loworder_solution!(scratch, scratch_implicit, :density, t_params)
    end
    if moments.evolve_upar
        @begin_s_r_z_region()
        rk_loworder_solution!(scratch, scratch_implicit, :upar, t_params)
    end
    if moments.evolve_p
        @begin_s_r_z_region()
        rk_loworder_solution!(scratch, scratch_implicit, :p, t_params)
    end
    if t_params.kinetic_electron_solver ∈ (implicit_time_evolving, explicit_time_evolving)
        @begin_r_z_vperp_vpa_region()
        rk_loworder_solution!(scratch, scratch_implicit, :pdf_electron, t_params)
    end
    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        @begin_r_z_region()
        rk_loworder_solution!(scratch, scratch_implicit, :electron_p, t_params)
    end
    if n_neutral_species > 0
        @begin_sn_r_z_vzeta_vr_region()
        rk_loworder_solution!(scratch, scratch_implicit, :pdf_neutral, t_params; neutrals=true)
        if moments.evolve_density
            @begin_sn_r_z_region()
            rk_loworder_solution!(scratch, scratch_implicit, :density_neutral, t_params; neutrals=true)
        end
        if moments.evolve_upar
            @begin_sn_r_z_region()
            rk_loworder_solution!(scratch, scratch_implicit, :uz_neutral, t_params; neutrals=true)
        end
        if moments.evolve_p
            @begin_sn_r_z_region()
            rk_loworder_solution!(scratch, scratch_implicit, :p_neutral, t_params; neutrals=true)
        end
    end

    # Apply boundary conditions and constraints to the loworder approximation.
    # Need to apply constraints using the high-order moments for consistency, to avoid
    # potential for spurious error estimates at boundary points.
    loworder_constraints_scratch =
        scratch_pdf(scratch[2].pdf, scratch[t_params.n_rk_stages+1].density,
                    scratch[t_params.n_rk_stages+1].upar,
                    scratch[t_params.n_rk_stages+1].p,
                    scratch[t_params.n_rk_stages+1].ion_external_source_controller_integral,
                    scratch[t_params.n_rk_stages+1].temp_z_s,
                    scratch[2].pdf_electron,
                    scratch[t_params.n_rk_stages+1].electron_density,
                    scratch[t_params.n_rk_stages+1].electron_upar,
                    scratch[t_params.n_rk_stages+1].electron_p,
                    scratch[t_params.n_rk_stages+1].electron_temp,
                    #scratch[t_params.n_rk_stages+1].electron_external_source_controller_integral,
                    scratch[2].pdf_neutral,
                    scratch[t_params.n_rk_stages+1].density_neutral,
                    scratch[t_params.n_rk_stages+1].uz_neutral,
                    scratch[t_params.n_rk_stages+1].p_neutral,
                    scratch[t_params.n_rk_stages+1].neutral_external_source_controller_integral)
    apply_all_bcs_constraints_update_moments!(
        loworder_constraints_scratch, pdf, moments, fields, boundaries, scratch_electron,
        vz, vr, vzeta, vpa, vperp, z, r, spectral_objects, advect_objects, composition,
        collisions, geometry, gyroavs, external_source_settings, num_diss_params,
        t_params, nl_solver_params, advance, scratch_dummy, false, 0, 0.0;
        update_electrons=false)

    # Re-calculate moment derivatives in the `moments` struct, in case they were changed
    # by the previous call
    apply_all_bcs_constraints_update_moments!(
        scratch[t_params.n_rk_stages+1], pdf, moments, fields, boundaries,
        scratch_electron, vz, vr, vzeta, vpa, vperp, z, r, spectral_objects,
        advect_objects, composition, collisions, geometry, gyroavs,
        external_source_settings, num_diss_params, t_params, nl_solver_params, advance,
        scratch_dummy, false, 0, 0.0; pdf_bc_constraints=false, update_electrons=false)

    # Calculate the timstep error estimates
    if z.bc == "wall" && (moments.evolve_upar || moments.evolve_p)
        # Set error on last/first non-zero point in ion distribution function to zero, as
        # this this point may cause unhelpful timestep failures when the cutoff moves from
        # one point to another.
        if z.irank == 0 || z.irank == z.nrank - 1
            @begin_s_r_region()
            @loop_s_r is ir begin
                density = @view scratch[t_params.n_rk_stages+1].density[:,ir,is]
                upar = @view scratch[t_params.n_rk_stages+1].upar[:,ir,is]
                p = @view scratch[t_params.n_rk_stages+1].p[:,ir,is]
                phi = fields.phi[:,ir]
                last_negative_vpa_ind, first_positive_vpa_ind =
                    get_ion_z_boundary_cutoff_indices(density, upar,
                                                      moments.ion.vth[1,ir,is],
                                                      moments.ion.vth[end,ir,is],
                                                      moments.evolve_upar,
                                                      moments.evolve_p, z, vpa,
                                                      1.0e-14, phi)
                if z.irank == 0
                    scratch[2].pdf[last_negative_vpa_ind,:,1,ir,is] .=
                        scratch[t_params.n_rk_stages+1].pdf[last_negative_vpa_ind,:,1,ir,is]
                end
                if z.irank == z.nrank - 1
                    scratch[2].pdf[first_positive_vpa_ind,:,end,ir,is] .=
                        scratch[t_params.n_rk_stages+1].pdf[first_positive_vpa_ind,:,end,ir,is]
                end
            end
        end
    end
    ion_pdf_error = local_error_norm(scratch[2].pdf, scratch[t_params.n_rk_stages+1].pdf,
                                     t_params.rtol, t_params.atol;
                                     method=error_norm_method, skip_r_inner=skip_r_inner,
                                     skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
    error_norms["pdf_accuracy"] = ion_pdf_error
    push!(total_points,
          vpa.n_global * vperp.n_global * z.n_global * r.n_global * n_ion_species)

    # Calculate error for ion moments, if necessary
    if moments.evolve_density
        @begin_s_r_z_region()
        ion_n_err = local_error_norm(scratch[2].density,
                                     scratch[t_params.n_rk_stages+1].density,
                                     t_params.rtol, t_params.atol;
                                     method=error_norm_method, skip_r_inner=skip_r_inner,
                                     skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
        error_norms["density_accuracy"] = ion_n_err
        push!(total_points, z.n_global * r.n_global * n_ion_species)
    end
    if moments.evolve_upar
        @begin_s_r_z_region()
        ion_u_err = local_error_norm(scratch[2].upar,
                                     scratch[t_params.n_rk_stages+1].upar, t_params.rtol,
                                     t_params.atol; method=error_norm_method,
                                     skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
        error_norms["upar_accuracy"] = ion_u_err
        push!(total_points, z.n_global * r.n_global * n_ion_species)
    end
    if moments.evolve_p
        @begin_s_r_z_region()
        ion_p_err = local_error_norm(scratch[2].p,
                                     scratch[t_params.n_rk_stages+1].p, t_params.rtol,
                                     t_params.atol; method=error_norm_method,
                                     skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
        error_norms["p_accuracy"] = ion_p_err
        push!(total_points, z.n_global * r.n_global * n_ion_species)
    end

    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        @begin_r_z_region()
        electron_p_err = local_error_norm(scratch[2].electron_p,
                                          scratch[t_params.n_rk_stages+1].electron_p,
                                          t_params.rtol, t_params.atol;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
        error_norms["electron_p_accuracy"] = electron_p_err
        push!(total_points, z.n_global * r.n_global)

        if t_params.kinetic_electron_solver == explicit_time_evolving
            # Need to have an accurate electron_pdf solution to avoid crashing an
            # explicit-timestepping simulation.
            electron_pdf_error = local_error_norm(scratch[2].pdf_electron,
                                                  scratch[t_params.n_rk_stages+1].pdf_electron,
                                                  t_params.rtol, t_params.atol;
                                                  method=error_norm_method,
                                                  skip_r_inner=skip_r_inner,
                                                  skip_z_lower=skip_z_lower,
                                                  error_sum_zero=t_params.error_sum_zero)
            error_norms["electron_pdf_accuracy"] = electron_pdf_error
            push!(total_points,
                  vpa.n_global * vperp.n_global * z.n_global * r.n_global)
        end
    end

    if n_neutral_species > 0
        # neutral z-advection
        # Don't parallelise over species here, because get_minimum_CFL_*() does an MPI
        # reduction over the shared-memory block, so all processes must calculate the same
        # species at the same time.
        @begin_r_vzeta_vr_vz_region()
        neutral_z_CFL = Inf
        @loop_sn isn begin
            update_speed_neutral_z!(neutral_z_advect[isn], moments.neutral.uz,
                                    moments.neutral.vth, evolve_upar, evolve_p, vz, vr,
                                    vzeta, z, r, t_params.t[])
            this_minimum = get_minimum_CFL_neutral_z(neutral_z_advect[isn].speed, z)
            @serial_region begin
                neutral_z_CFL = min(neutral_z_CFL, this_minimum)
            end
        end
        CFL_limits["neutral_CFL_z"] = t_params.CFL_prefactor * neutral_z_CFL

        # neutral vz-advection
        @begin_r_z_vzeta_vr_region()
        neutral_vz_CFL = Inf
        update_speed_neutral_vz!(neutral_vz_advect, fields,
                                 scratch[t_params.n_rk_stages+1], moments, vz, vr, vzeta,
                                 z, r, composition, collisions,
                                 external_source_settings.neutral)
        @loop_sn isn begin
            this_minimum = get_minimum_CFL_neutral_vz(neutral_vz_advect[isn].speed, vz)
            @serial_region begin
                neutral_vz_CFL = min(neutral_vz_CFL, this_minimum)
            end
        end
        CFL_limits["neutral_CFL_vz"] = t_params.CFL_prefactor * neutral_vz_CFL

        # Calculate error for neutral distribution functions
        neut_pdf_error = local_error_norm(scratch[2].pdf_neutral,
                                          scratch[t_params.n_rk_stages+1].pdf_neutral,
                                          t_params.rtol, t_params.atol;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
        error_norms["neutral_pdf_accuracy"] = neut_pdf_error
        push!(total_points,
              vz.n_global * vr.n_global * vzeta.n_global * z.n_global * r.n_global *
              n_neutral_species)

        # Calculate error for neutral moments, if necessary
        if moments.evolve_density
            @begin_sn_r_z_region()
            neut_n_err = local_error_norm(scratch[2].density_neutral,
                                          scratch[t_params.n_rk_stages+1].density_neutral,
                                          t_params.rtol, t_params.atol, true;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
            error_norms["neutral_density_accuracy"] = neut_n_err
            push!(total_points, z.n_global * r.n_global * n_neutral_species)
        end
        if moments.evolve_upar
            @begin_sn_r_z_region()
            neut_u_err = local_error_norm(scratch[2].uz_neutral,
                                          scratch[t_params.n_rk_stages+1].uz_neutral,
                                          t_params.rtol, t_params.atol, true;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
            error_norms["neutral_uz_accuracy"] = neut_u_err
            push!(total_points, z.n_global * r.n_global * n_neutral_species)
        end
        if moments.evolve_p
            @begin_sn_r_z_region()
            neut_p_err = local_error_norm(scratch[2].p_neutral,
                                          scratch[t_params.n_rk_stages+1].p_neutral,
                                          t_params.rtol, t_params.atol, true;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
            error_norms["neutral_p_accuracy"] = neut_p_err
            push!(total_points, z.n_global * r.n_global * n_neutral_species)
        end
    end

    adaptive_timestep_update_t_params!(t_params, CFL_limits, error_norms, total_points,
                                       error_norm_method, success, nl_max_its_fraction,
                                       nl_total_its_soft_limit,
                                       nl_total_its_soft_limit_reduce_dt, composition)

    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        if t_params.previous_dt[] == 0.0
            # Reset electron pdf to its value at the beginning of this step.
            @begin_r_z_vperp_vpa_region()
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf.electron.norm[ivpa,ivperp,iz,ir] =
                    pdf.electron.pdf_before_ion_timestep[ivpa,ivperp,iz,ir]
                scratch_electron[1].pdf_electron[ivpa,ivperp,iz,ir] =
                    pdf.electron.pdf_before_ion_timestep[ivpa,ivperp,iz,ir]
            end
        else
            # Store the current value, which will be the value at the beginning of the
            # next step.
            @begin_r_z_vperp_vpa_region()
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf.electron.pdf_before_ion_timestep[ivpa,ivperp,iz,ir] =
                    pdf.electron.norm[ivpa,ivperp,iz,ir]
            end
        end

        istage = t_params.n_rk_stages+1

        # update the pdf.norm and moments arrays as needed
        @begin_s_r_z_region()
        final_scratch = scratch[istage]
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            pdf.ion.norm[ivpa,ivperp,iz,ir,is] = final_scratch.pdf[ivpa,ivperp,iz,ir,is]
        end
        @loop_s_r_z is ir iz begin
            moments.ion.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
            moments.ion.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
            moments.ion.p[iz,ir,is] = final_scratch.p[iz,ir,is]
        end
        # No need to synchronize here as we only change electron quantities and previous
        # region only changed ion quantities.
        @begin_r_z_region(true)
        @loop_r_z ir iz begin
            moments.electron.dens[iz,ir] = final_scratch.electron_density[iz,ir]
            moments.electron.upar[iz,ir] = final_scratch.electron_upar[iz,ir]
            moments.electron.p[iz,ir] = final_scratch.electron_p[iz,ir]
            moments.electron.temp[iz,ir] = final_scratch.electron_temp[iz,ir]
        end
        if composition.n_neutral_species > 0
            # No need to synchronize here as we only change neutral quantities and previous
            # region only changed plasma quantities.
            @begin_sn_r_z_region(true)
            @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
                pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = final_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
            end
            @loop_sn_r_z isn ir iz begin
                moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
                moments.neutral.uz[iz,ir,isn] = final_scratch.uz_neutral[iz,ir,isn]
                moments.neutral.p[iz,ir,isn] = final_scratch.p_neutral[iz,ir,isn]
            end
            # for now update moments.neutral object directly for diagnostic moments
            # that are not used in Runga-Kutta steps
            update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                               moments.neutral.dens, moments.neutral.vth,
                               pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                               moments.evolve_density, moments.evolve_p)
            update_neutral_uzeta!(moments.neutral.uzeta, moments.neutral.uzeta_updated,
                                  moments.neutral.dens, moments.neutral.vth,
                                  pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                                  moments.evolve_density, moments.evolve_p)
            update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated,
                               moments.neutral.dens, moments.neutral.ur,
                               moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z, r,
                               composition, moments.evolve_density, moments.evolve_upar,
                               moments.evolve_p)
            update_neutral_pzeta!(moments.neutral.pzeta, moments.neutral.pzeta_updated,
                                  moments.neutral.dens, moments.neutral.uzeta,
                                  moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z,
                                  r, composition, moments.evolve_density,
                                  moments.evolve_upar, moments.evolve_p)
            # pz can be calculated from p, pzeta, and pr
            @loop_sn_r_z isn ir iz begin
                moments.neutral.pz[iz,ir,isn] = (3.0 * moments.neutral.p[iz,ir,isn] - moments.neutral.pr[iz,ir,isn] - moments.neutral.pzeta[iz,ir,isn])
            end
            try #below loop can cause DomainError if p < 0 or density < 0, so exit cleanly if possible
                @loop_sn_r_z isn ir iz begin
                    # update density using last density from Runga-Kutta stages
                    moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
                    # get vth for neutrals
                    moments.neutral.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.p[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
                end
            catch e
                if global_size[] > 1
                    println("ERROR: error at line $(@__LINE__) of time_advance.jl")
                    println(e)
                    display(stacktrace(catch_backtrace()))
                    flush(stdout)
                    flush(stderr)
                    MPI.Abort(comm_world, 1)
                end
                rethrow(e)
            end
        end

    end

    return nothing
end

"""
"""
@timeit global_timer ssp_rk!(pdf, scratch, scratch_implicit, scratch_electron, t_params,
                             vz, vr, vzeta, vpa, vperp, gyrophase, z, r, moments, fields,
                             spectral_objects, advect_objects, composition, collisions,
                             geometry, gyroavs, boundaries, external_source_settings,
                             num_diss_params, nl_solver_params, advance, advance_implicit,
                             fp_arrays, scratch_dummy, manufactured_source_list,
                             diagnostic_checks, istep) = begin

    # Convenience wrapper for calls to write debug information within this function.
    function write_debug_IO(this_scratch, istage, label)
        if t_params.debug_io === nothing
            # Allow compiler to optimise away this function if debug IO is not being used.
            return nothing
        end
        write_debug_data_to_binary(this_scratch, moments, fields, composition, t_params,
                                   r, z, vperp, vpa, vzeta, vr, vz, label, istage)
        return nothing
    end

    @begin_s_r_z_region()

    n_rk_stages = t_params.n_rk_stages

    if t_params.electron !== nothing
        max_electron_pdf_iterations = t_params.electron.max_pseudotimesteps
        max_electron_sim_time = t_params.electron.max_pseudotime
    else
        max_electron_pdf_iterations = nothing
        max_electron_sim_time = nothing
    end

    first_scratch = scratch[1]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        first_scratch.pdf[ivpa,ivperp,iz,ir,is] = pdf.ion.norm[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.ion.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.ion.upar[iz,ir,is]
        first_scratch.p[iz,ir,is] = moments.ion.p[iz,ir,is]
    end

    if length(first_scratch.pdf_electron) > 0
        @begin_r_z_vperp_vpa_region()
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            first_scratch.pdf_electron[ivpa,ivperp,iz,ir] = pdf.electron.norm[ivpa,ivperp,iz,ir]
        end
    end
    @begin_r_z_region()
    @loop_r_z ir iz begin
        first_scratch.electron_density[iz,ir] = moments.electron.dens[iz,ir]
        first_scratch.electron_upar[iz,ir] = moments.electron.upar[iz,ir]
        first_scratch.electron_p[iz,ir] = moments.electron.p[iz,ir]
        first_scratch.electron_temp[iz,ir] = moments.electron.temp[iz,ir]
    end

    if composition.n_neutral_species > 0
        @begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            first_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            first_scratch.density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            first_scratch.uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            first_scratch.p_neutral[iz,ir,isn] = moments.neutral.p[iz,ir,isn]
            # other neutral moments here if required
        end
    end
    @begin_serial_region()
    @serial_region begin
        first_scratch.ion_external_source_controller_integral .=
            moments.ion.external_source_controller_integral
        #if length(first_scratch.pdf_electron) > 0
        #    first_scratch.electron_external_source_controller_integral .=
        #        moments.electron.external_source_controller_integral
        #end
        if composition.n_neutral_species > 0
            first_scratch.neutral_external_source_controller_integral .=
                moments.neutral.external_source_controller_integral
        end
    end
    write_debug_IO(first_scratch, 0, "begin ssp_rk!")

    if moments.evolve_upar
        # moments may be read on all ranks, even though loop type is z_s, so need to
        # synchronize here
        @_block_synchronize()
    end

    # success is set to false if an iteration failed to converge in an implicit solve
    success = ""
    for istage ∈ 1:n_rk_stages
        if t_params.rk_coefs_implicit !== nothing
            update_solution_vector!(scratch_implicit[istage], scratch[istage], moments,
                                    composition, vpa, vperp, z, r)
            write_debug_IO(scratch_implicit[istage], istage,
                           "implicit update_solution_vector!")
            if t_params.implicit_coefficient_is_zero[istage]
                # No implicit solve needed at this stage. Do an explicit step of the
                # implicitly-evolved terms so we can store their time-derivative at this
                # stage.
                euler_time_advance!(scratch_implicit[istage], scratch[istage],
                                    pdf, fields, moments, advect_objects, vz, vr, vzeta,
                                    vpa, vperp, gyrophase, z, r, t_params, t_params.dt[],
                                    spectral_objects, composition, collisions, geometry,
                                    scratch_dummy, manufactured_source_list,
                                    external_source_settings, num_diss_params,
                                    advance_implicit, fp_arrays, istage)
                write_debug_IO(scratch_implicit[istage], istage,
                               "implicit_coefficient_is_zero euler_time_advance!")
                # The result of the forward-Euler step is just a hack to store the
                # (explicit) time-derivative of the implicitly advanced terms. The result
                # is not used as input to the explicit part of the IMEX advance.
                old_scratch = scratch[istage]
            else
                # Backward-Euler step for implicitly-evolved terms.
                # Note the timestep for this solve is rk_coefs_implict[istage,istage]*dt.
                # The diagonal elements are equal to the Butcher 'a' coefficients
                # rk_coefs_implicit[istage,istage]=a[istage,istage].
                if scratch_electron === nothing
                    this_scratch_electron = nothing
                elseif t_params.kinetic_electron_solver == implicit_steady_state
                    this_scratch_electron = scratch_electron[t_params.electron.n_rk_stages+1]
                else
                    this_scratch_electron = scratch_electron
                end
                nl_success = backward_euler!(scratch_implicit[istage], scratch[istage],
                                             this_scratch_electron,
                                             pdf, fields, moments, advect_objects, vz, vr,
                                             vzeta, vpa, vperp, gyrophase, z, r,
                                             t_params.dt[] *
                                             t_params.rk_coefs_implicit[istage,istage],
                                             t_params, spectral_objects, composition,
                                             collisions, geometry, scratch_dummy,
                                             manufactured_source_list,
                                             external_source_settings, num_diss_params,
                                             gyroavs, nl_solver_params, advance_implicit,
                                             fp_arrays, istage)
                write_debug_IO(scratch_implicit[istage], istage, "backward_euler!")
                nl_success = MPI.Allreduce(nl_success, &, comm_world)
                if !nl_success
                    success = "nonlinear-solver"
                    # Break out of the istage loop, as passing `success != ""` to the
                    # adaptive timestep update function will signal a failed timestep, so
                    # that we restart this timestep with a smaller `dt`.
                    break
                end
                # The result of the implicit solve gives the state vector at 'istage'
                # which is used as input to the explicit part of the IMEX time step.
                old_scratch = scratch_implicit[istage]
                update_electrons = t_params.kinetic_electron_solver ∉ (implicit_time_evolving,
                                                                       implicit_p_implicit_pseudotimestep,
                                                                       implicit_steady_state,
                                                                       explicit_time_evolving,
                                                                       implicit_p_explicit_pseudotimestep)
                bcs_constraints_success = apply_all_bcs_constraints_update_moments!(
                    scratch_implicit[istage], pdf, moments, fields,
                    boundaries, scratch_electron, vz, vr, vzeta, vpa, vperp, z, r,
                    spectral_objects, advect_objects, composition, collisions, geometry,
                    gyroavs, external_source_settings, num_diss_params, t_params,
                    nl_solver_params, advance, scratch_dummy, false,
                    max_electron_pdf_iterations, max_electron_sim_time;
                    update_electrons=update_electrons)
                write_debug_IO(scratch_implicit[istage], istage,
                               "implicit apply_all_bcs_constraints_update_moments!")
                if bcs_constraints_success != ""
                    success = bcs_constraints_success
                end
                if success != ""
                    # Break out of the istage loop, as passing `success != ""` to the
                    # adaptive timestep update function will signal a failed timestep, so
                    # that we restart this timestep with a smaller `dt`.
                    break
                end
            end
        else
            # Fully explicit method starts the forward-Euler step with the result from the
            # previous stage.
            old_scratch = scratch[istage]
        end
        update_solution_vector!(scratch[istage+1], old_scratch, moments, composition, vpa,
                                vperp, z, r)
        write_debug_IO(scratch[istage+1], istage, "update_solution_vector!")
        # do an Euler time advance, with scratch[istage+1] containing the advanced
        # quantities and scratch[istage] containing quantities at time level n, RK stage
        # istage
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], old_scratch, pdf, fields, moments,
                            advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z,
                            r, t_params, t_params.dt[], spectral_objects, composition,
                            collisions, geometry, scratch_dummy,
                            manufactured_source_list, external_source_settings,
                            num_diss_params, advance, fp_arrays, istage)
        write_debug_IO(scratch[istage+1], istage, "after euler_time_advance!")

        rk_update!(scratch, scratch_implicit, moments, t_params, istage, composition)
        write_debug_IO(scratch[istage+1], istage, "rk_update!")

        # Always apply boundary conditions and constraints here for explicit schemes. For
        # IMEX schemes, only apply boundary conditions and constraints at the final RK
        # stage - for other stages they are imposed after the implicit part of the step.
        # If `implicit_coefficient_is_zero` is true for the next stage, then this step is
        # explicit, so we need the bcs and constraints.
        apply_bc_constraints = (t_params.rk_coefs_implicit === nothing
                                || !t_params.implicit_ion_advance
                                || (istage == n_rk_stages && t_params.implicit_coefficient_is_zero[1])
                                || t_params.implicit_coefficient_is_zero[istage+1])
        update_electrons = ((t_params.rk_coefs_implicit === nothing && t_params.kinetic_electron_solver !== explicit_time_evolving)
                            || t_params.kinetic_electron_solver ∉ (implicit_time_evolving,
                                                                   implicit_p_implicit_pseudotimestep,
                                                                   implicit_steady_state,
                                                                   explicit_time_evolving,
                                                                   implicit_p_explicit_pseudotimestep))
        diagnostic_moments = diagnostic_checks && istage == n_rk_stages
        bcs_constraints_success = apply_all_bcs_constraints_update_moments!(
            scratch[istage+1], pdf, moments, fields, boundaries, scratch_electron, vz, vr,
            vzeta, vpa, vperp, z, r, spectral_objects, advect_objects, composition,
            collisions, geometry, gyroavs, external_source_settings, num_diss_params,
            t_params, nl_solver_params, advance, scratch_dummy, diagnostic_moments,
            max_electron_pdf_iterations, max_electron_sim_time;
            pdf_bc_constraints=apply_bc_constraints, update_electrons=update_electrons)
        write_debug_IO(scratch[istage+1], istage,
                       "apply_all_bcs_constraints_update_moments!")
        if bcs_constraints_success != ""
            success = bcs_constraints_success
        end
        if success != ""
            # Break out of the istage loop, as passing `success != ""` to the
            # adaptive timestep update function will signal a failed timestep, so
            # that we restart this timestep with a smaller `dt`.
            break
        end
    end

    if t_params.adaptive
        nl_max_its_fraction = 0.0
        nl_total_its_soft_limit = false
        nl_total_its_soft_limit_reduce_dt = false
        if t_params.kinetic_electron_solver ∈ (implicit_steady_state, implicit_time_evolving)
            params_to_check = (nl_solver_params.ion_advance,
                               nl_solver_params.vpa_advection,
                               nl_solver_params.electron_conduction,
                               nl_solver_params.electron_advance)
        else
            # nl_solver_params.electron_advance is used for the backward-Euler timestep in
            # electron timestepping, so its iteration count is not relevant here. Instead,
            # check the number of electron pseudo-timesteps or pseudo-time increment
            # compared to their maximum values
            params_to_check = (nl_solver_params.ion_advance,
                               nl_solver_params.vpa_advection,
                               nl_solver_params.electron_conduction)
            if t_params.electron !== nothing
                electron_time_advance_fraction =
                    min(t_params.electron.max_step_count_this_ion_step[] / max_electron_pdf_iterations,
                        t_params.electron.max_t_increment_this_ion_step[] / max_electron_sim_time)
                nl_max_its_fraction = max(electron_time_advance_fraction, nl_max_its_fraction)
            end
        end
        for p ∈ params_to_check
            if p !== nothing
                nl_max_its_fraction =
                    max(p.max_nonlinear_iterations_this_step[] / p.nonlinear_max_iterations,
                        nl_max_its_fraction)
                nl_total_its_soft_limit = p.max_linear_iterations_this_step[] > p.total_its_soft_limit || nl_total_its_soft_limit
                nl_total_its_soft_limit_reduce_dt = p.max_linear_iterations_this_step[] > 1.5 * p.total_its_soft_limit || nl_total_its_soft_limit_reduce_dt
            end
        end
        adaptive_timestep_update!(scratch, scratch_implicit, scratch_electron,
                                  t_params, pdf, moments, fields,
                                  boundaries, composition, collisions, geometry,
                                  external_source_settings, spectral_objects,
                                  advect_objects, gyroavs, num_diss_params,
                                  nl_solver_params, advance, scratch_dummy, r, z, vperp,
                                  vpa, vzeta, vr, vz, success, nl_max_its_fraction,
                                  nl_total_its_soft_limit,
                                  nl_total_its_soft_limit_reduce_dt)
        write_debug_IO(scratch[n_rk_stages+1], 0, "adaptive_timestep_update!")
    elseif success != ""
        error("Implicit part of timestep failed")
    end

    reset_nonlinear_per_stage_counters!(nl_solver_params.ion_advance)
    reset_nonlinear_per_stage_counters!(nl_solver_params.vpa_advection)
    reset_nonlinear_per_stage_counters!(nl_solver_params.electron_conduction)
    if t_params.kinetic_electron_solver != implicit_steady_state && t_params.electron !== nothing
        t_params.electron.max_step_count_this_ion_step[] = 0
        t_params.electron.max_t_increment_this_ion_step[] = 0.0
    end
    if t_params.kinetic_electron_solver == implicit_time_evolving
        reset_nonlinear_per_stage_counters!(nl_solver_params.electron_advance)
    end


    if t_params.previous_dt[] > 0.0
        istage = n_rk_stages+1

        # update the pdf.norm and moments arrays as needed
        @begin_s_r_z_region()
        final_scratch = scratch[istage]
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            pdf.ion.norm[ivpa,ivperp,iz,ir,is] = final_scratch.pdf[ivpa,ivperp,iz,ir,is]
        end
        @loop_s_r_z is ir iz begin
            moments.ion.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
            moments.ion.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
            moments.ion.p[iz,ir,is] = final_scratch.p[iz,ir,is]
        end
        # No need to synchronize here as we only change electron quantities and previous
        # region only changed ion quantities.
        if length(final_scratch.pdf_electron) > 0
            @begin_r_z_vperp_vpa_region()
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf.electron.norm[ivpa,ivperp,iz,ir] = final_scratch.pdf_electron[ivpa,ivperp,iz,ir]
            end
        end
        @begin_r_z_region(true)
        @loop_r_z ir iz begin
            moments.electron.dens[iz,ir] = final_scratch.electron_density[iz,ir]
            moments.electron.upar[iz,ir] = final_scratch.electron_upar[iz,ir]
            moments.electron.p[iz,ir] = final_scratch.electron_p[iz,ir]
            moments.electron.temp[iz,ir] = final_scratch.electron_temp[iz,ir]
        end
        if composition.n_neutral_species > 0
            # No need to synchronize here as we only change neutral quantities and previous
            # region only changed plasma quantities.
            @begin_sn_r_z_region(true)
            @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
                pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = final_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
            end
            @loop_sn_r_z isn ir iz begin
                moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
                moments.neutral.uz[iz,ir,isn] = final_scratch.uz_neutral[iz,ir,isn]
                moments.neutral.p[iz,ir,isn] = final_scratch.p_neutral[iz,ir,isn]
            end
            # for now update moments.neutral object directly for diagnostic moments
            # that are not used in Runga-Kutta steps
            update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                               moments.neutral.dens, moments.neutral.vth,
                               pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                               moments.evolve_density, moments.evolve_p)
            update_neutral_uzeta!(moments.neutral.uzeta, moments.neutral.uzeta_updated,
                                  moments.neutral.dens, moments.neutral.vth,
                                  pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                                  moments.evolve_density, moments.evolve_p)
            update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated,
                               moments.neutral.dens, moments.neutral.ur,
                               moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z, r,
                               composition, moments.evolve_density, moments.evolve_upar,
                               moments.evolve_p)
            update_neutral_pzeta!(moments.neutral.pzeta, moments.neutral.pzeta_updated,
                                  moments.neutral.dens, moments.neutral.uzeta,
                                  moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z,
                                  r, composition, moments.evolve_density,
                                  moments.evolve_upar, moments.evolve_p)
        end
        @begin_serial_region()
        @serial_region begin
            moments.ion.external_source_controller_integral .=
                final_scratch.ion_external_source_controller_integral
            #if length(first_scratch.pdf_electron) > 0
            #    moments.electron.external_source_controller_integral .=
            #        final_scratch.electron_external_source_controller_integral
            #end
            if composition.n_neutral_species > 0
                moments.neutral.external_source_controller_integral .=
                    final_scratch.neutral_external_source_controller_integral
            end
        end
        write_debug_IO(final_scratch, istage, "end ssp_rk!")
    end

    return nothing
end

"""
euler_time_advance! advances the vector equation dfvec/dt = G[f]
that includes the kinetic equation + any evolved moment equations
using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
with fvec_in an input and fvec_out the output.

Note `dt` is passed separately from `t_params` because sometimes (in the IMEX Runge-Kutta
implementation), a call needs to be made with `dt` scaled by some coefficient.
"""
@timeit global_timer euler_time_advance!(
                         fvec_out, fvec_in, pdf, fields, moments, advect_objects, vz, vr,
                         vzeta, vpa, vperp, gyrophase, z, r, t_params, dt,
                         spectral_objects, composition, collisions, geometry,
                         scratch_dummy, manufactured_source_list,
                         external_source_settings, num_diss_params, advance, fp_arrays,
                         istage) = begin

    # Convenience wrapper for calls to write debug information within this function.
    function write_debug_IO(label)
        if t_params.debug_io === nothing
            # Allow compiler to optimise away this function if debug IO is not being used.
            return nothing
        end
        write_debug_data_to_binary(fvec_out, moments, fields, composition, t_params, r, z,
                                   vperp, vpa, vzeta, vr, vz, label, istage)
        return nothing
    end
    write_debug_IO("begin euler_time_advance!")

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    t = t_params.t[]

    vpa_spectral, vperp_spectral, r_spectral, z_spectral = spectral_objects.vpa_spectral, spectral_objects.vperp_spectral, spectral_objects.r_spectral, spectral_objects.z_spectral
    vz_spectral, vr_spectral, vzeta_spectral = spectral_objects.vz_spectral, spectral_objects.vr_spectral, spectral_objects.vzeta_spectral
    vpa_advect, vperp_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.vperp_advect, advect_objects.r_advect, advect_objects.z_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    if advance.external_source
        total_external_ion_source_controllers!(fvec_out.ion_external_source_controller_integral,
                                               fvec_in, moments,
                                               external_source_settings.ion, vperp, dt)
        write_debug_IO("total_external_ion_source_controllers!")
    end
    if advance.neutral_external_source
        total_external_neutral_source_controllers!(fvec_out.neutral_external_source_controller_integral,fvec_in,
                                                   moments,
                                                   external_source_settings.neutral, r,
                                                   z, vzeta, vr, dt)
        write_debug_IO("total_external_neutral_source_controllers!")
    end

    # Start advance for moments
    if advance.continuity
        continuity_equation!(fvec_out.density, fvec_in, moments, composition, dt,
                             z_spectral, collisions.reactions.ionization_frequency,
                             external_source_settings.ion, num_diss_params)
        write_debug_IO("continuity_equation!")
    end
    if advance.force_balance
        force_balance!(fvec_out.upar, fvec_out.density, fvec_in, moments, fields,
                       collisions, dt, z_spectral, composition, geometry,
                       external_source_settings.ion, num_diss_params, z)
        write_debug_IO("force_balance!")
    end
    if advance.energy
        energy_equation!(fvec_out.p, fvec_in, moments, collisions, dt, z_spectral,
                         composition, external_source_settings.ion, num_diss_params)
        write_debug_IO("energy_equation!")
    end
    if advance.neutral_continuity
        neutral_continuity_equation!(fvec_out.density_neutral, fvec_in, moments,
                                     composition, dt, z_spectral,
                                     collisions.reactions.ionization_frequency,
                                     external_source_settings.neutral, num_diss_params)
        write_debug_IO("neutral_continuity_equation!")
    end
    if advance.neutral_force_balance
        neutral_force_balance!(fvec_out.uz_neutral, fvec_out.density_neutral, fvec_in,
                               moments, fields, collisions, dt, z_spectral, composition,
                               geometry, external_source_settings.neutral,
                               num_diss_params)
        write_debug_IO("neutral_force_balance!")
    end
    if advance.neutral_energy
        neutral_energy_equation!(fvec_out.p_neutral, fvec_in, moments, collisions, dt,
                                 z_spectral, composition,
                                 external_source_settings.neutral, num_diss_params)
        write_debug_IO("neutral_energy_equation!")
    end

    if advance.electron_pdf
        for ir ∈ 1:r.n
            if t_params.debug_io !== nothing && ir == 1
                # Probably do not want to write separate debug output for every r-index
                # even in 2D, as this would create very large output files, so pick a
                # single r-index for debug output. This might not be the most useful
                # r-index for 2D simulations!
                this_debug_io = t_params.debug_io
            else
                this_debug_io = nothing
            end
            @views electron_kinetic_equation_euler_update!(
                       fvec_out, fvec_in.pdf_electron[:,:,:,ir],
                       fvec_in.electron_p[:,ir], moments, z, vperp, vpa, z_spectral,
                       vpa_spectral, z_advect, vpa_advect, scratch_dummy, collisions,
                       composition, external_source_settings, num_diss_params,
                       t_params, ir; debug_io=this_debug_io, fields=fields,
                       r=r, vzeta=vzeta, vr=vr, vz=vz, istage=istage)
        end
        write_debug_IO("electron_kinetic_equation_euler_update!")
    end
    if advance.electron_energy
        electron_energy_equation!(fvec_out.electron_p, fvec_out.density,
                                  fvec_in.electron_p, fvec_in.density,
                                  fvec_in.electron_upar, moments.electron.ppar,
                                  fvec_in.density, fvec_in.upar, fvec_in.p,
                                  fvec_in.density_neutral, fvec_in.uz_neutral,
                                  fvec_in.p_neutral, moments.electron, collisions, dt,
                                  composition, external_source_settings.electron,
                                  num_diss_params, r, z;
                                  conduction=advance.electron_conduction)
        update_derived_electron_moment_time_derivatives!(fvec_in.electron_p, moments,
                                                         composition.electron_physics)
        write_debug_IO("electron_energy_equation!")
    elseif advance.electron_conduction
        # Explicit version of the implicit part of the IMEX timestep, need to evaluate
        # only the conduction term.
        for ir ∈ 1:r.n
            @views electron_braginskii_conduction!(
                fvec_out.electron_p[:,ir], fvec_in.electron_p[:,ir],
                fvec_in.electron_density[:,ir], fvec_in.electron_upar[:,ir],
                fvec_in.upar[:,ir], moments.electron, collisions, composition, z,
                z_spectral, scratch_dummy, dt, ir)
        end
        write_debug_IO("electron_braginskii_conduction!")
    end

    update_derived_ion_moment_time_derivatives!(fvec_in, moments)
    update_derived_neutral_moment_time_derivatives!(fvec_in, moments)
    # End advance for moments

    # Start advance for distribution functions
    if composition.ion_physics ∈ (drift_kinetic_ions, gyrokinetic_ions)
        # vpa_advection! advances the 1D advection equation in vpa.
        if advance.vpa_advection
            vpa_advection!(fvec_out.pdf, fvec_in, fields, moments, vpa_advect, vpa, vperp, z, r, dt, t,
                vpa_spectral, composition, collisions, external_source_settings.ion, geometry)
            write_debug_IO("vpa_advection!")
        end

        # z_advection! advances 1D advection equation in z
        # apply z-advection operation to ion species
        if advance.z_advection
            z_advection!(fvec_out.pdf, fvec_in, moments, fields, z_advect, z, vpa, vperp, r,
                        dt, t, z_spectral, composition, geometry, scratch_dummy)
            write_debug_IO("z_advection!")
        end

        # r advection relies on derivatives in z to get ExB
        if advance.r_advection
            r_advection!(fvec_out.pdf, fvec_in, moments, fields, r_advect, r, z, vperp, vpa,
                        dt, r_spectral, composition, geometry, scratch_dummy)
            write_debug_IO("r_advection!")
        end
        # vperp_advection requires information about z and r advection
        # so call vperp_advection! only after z and r advection routines
        if advance.vperp_advection
            vperp_advection!(fvec_out.pdf, fvec_in, vperp_advect, r, z, vperp, vpa,
                        dt, vperp_spectral, composition, z_advect, r_advect, geometry,
                        moments, fields, t)
            write_debug_IO("vperp_advection!")
        end

        if advance.source_terms
            source_terms!(fvec_out.pdf, fvec_in, moments, vpa, vperp, z, r, dt, z_spectral,
                        composition, collisions, external_source_settings.ion)
            write_debug_IO("source_terms!")
        end

        if advance.neutral_z_advection
            neutral_advection_z!(fvec_out.pdf_neutral, fvec_in, moments, neutral_z_advect,
                r, z, vzeta, vr, vz, dt, t, z_spectral, composition, scratch_dummy)
            write_debug_IO("neutral_advection_z!")
        end

        if advance.neutral_r_advection
            neutral_advection_r!(fvec_out.pdf_neutral, fvec_in, neutral_r_advect,
                r, z, vzeta, vr, vz, dt, r_spectral, composition, geometry, scratch_dummy)
            write_debug_IO("neutral_advection_r!")
        end

        # neutral_advection_vz! advances the 1D advection equation in vz.
        # neutral species do not have a force accelerating them in vz;
        # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa
        if advance.neutral_vz_advection
            neutral_advection_vz!(fvec_out.pdf_neutral, fvec_in, fields, moments,
                                neutral_vz_advect, vz, vr, vzeta, z, r, dt, vz_spectral,
                                composition, collisions, external_source_settings.neutral)
            write_debug_IO("neutral_advection_vz!")
        end

        if advance.neutral_source_terms
            source_terms_neutral!(fvec_out.pdf_neutral, fvec_in, moments, vz, z, r, dt, z_spectral,
                        composition, collisions, external_source_settings.neutral)
            write_debug_IO("source_terms_neutral!")
        end

        if advance.manufactured_solns_test
            source_terms_manufactured!(fvec_out.pdf, fvec_out.pdf_neutral, vz, vr, vzeta, vpa, vperp, z, r, t, dt, composition, manufactured_source_list)
            write_debug_IO("source_terms_manufactured!")
        end

        if advance.ion_cx_collisions || advance.ion_ionization_collisions
            # gyroaverage neutral dfn and place it in the ion.buffer array for use in the collisions step
            vzvrvzeta_to_vpavperp!(pdf.ion.buffer, fvec_in.pdf_neutral, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, geometry, composition)
            write_debug_IO("vzvrvzeta_to_vpavperp!")
        end
        if advance.neutral_cx_collisions || advance.neutral_ionization_collisions
            # interpolate ion particle dfn and place it in the neutral.buffer array for use in the collisions step
            vpavperp_to_vzvrvzeta!(pdf.neutral.buffer, fvec_in.pdf, vz, vr, vzeta, vpa, vperp, z, r, geometry, composition)
            write_debug_IO("vpavperp_to_vzvrvzeta!")
        end

        # account for charge exchange collisions between ions and neutrals
        if advance.ion_cx_collisions_1V
            ion_charge_exchange_collisions_1V!(fvec_out.pdf, fvec_in, moments, composition,
                                            vpa, vz, collisions.reactions.charge_exchange_frequency,
                                            vpa_spectral, vz_spectral, dt)
            write_debug_IO("ion_charge_exchange_collisions_1V!")
        elseif advance.ion_cx_collisions
            ion_charge_exchange_collisions_3V!(fvec_out.pdf, pdf.ion.buffer, fvec_in,
                                            composition, vz, vr, vzeta, vpa, vperp, z, r,
                                            collisions.reactions.charge_exchange_frequency, dt)
            write_debug_IO("ion_charge_exchange_collisions_3V!")
        end
        if advance.neutral_cx_collisions_1V
            neutral_charge_exchange_collisions_1V!(fvec_out.pdf_neutral, fvec_in, moments,
                                                composition, vpa, vz,
                                                collisions.reactions.charge_exchange_frequency, vpa_spectral,
                                                vz_spectral, dt)
            write_debug_IO("neutral_charge_exchange_collisions_1V!")
        elseif advance.neutral_cx_collisions
            neutral_charge_exchange_collisions_3V!(fvec_out.pdf_neutral, pdf.neutral.buffer,
                                                fvec_in, composition, vz, vr, vzeta, vpa,
                                                vperp, z, r, collisions.reactions.charge_exchange_frequency,
                                                dt)
            write_debug_IO("neutral_charge_exchange_collisions_3V!")
        end
        # account for ionization collisions between ions and neutrals
        if advance.ion_ionization_collisions_1V
            ion_ionization_collisions_1V!(fvec_out.pdf, fvec_in, vz, vpa, vperp, z, r,
                                        vz_spectral, moments, composition, collisions, dt)
            write_debug_IO("ion_ionization_collisions_1V!")
        elseif advance.ion_ionization_collisions
            ion_ionization_collisions_3V!(fvec_out.pdf, pdf.ion.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions, dt)
            write_debug_IO("ion_ionization_collisions_3V!")
        end
        if advance.neutral_ionization_collisions_1V
            neutral_ionization_collisions_1V!(fvec_out.pdf_neutral, fvec_in, vz, vpa, vperp,
                                            z, r, vz_spectral, moments, composition,
                                            collisions, dt)
            write_debug_IO("neutral_ionization_collisions_1V!")
        elseif advance.neutral_ionization_collisions
            neutral_ionization_collisions_3V!(fvec_out.pdf_neutral, fvec_in, composition, vz,
                                            vr, vzeta, vpa, vperp, z, r, collisions, dt)
            write_debug_IO("neutral_ionization_collisions_3V!")
        end

        # Add Krook collision operator for ions
        if advance.krook_collisions_ii
            krook_collisions!(fvec_out.pdf, fvec_in, moments, composition, collisions,
                            vperp, vpa, dt)
            write_debug_IO("krook_collisions!")
        end
        # Add maxwellian diffusion collision operator for ions
        if advance.mxwl_diff_collisions_ii
            ion_vpa_maxwell_diffusion!(fvec_out.pdf, fvec_in, moments, vpa, vperp, vpa_spectral, 
                                    dt, collisions.mxwl_diff.D_ii)
            write_debug_IO("ion_vpa_maxwell_diffusion!")
        end
        # Add maxwellian diffusion collision operator for neutrals
        if advance.mxwl_diff_collisions_nn
            neutral_vz_maxwell_diffusion!(fvec_out.pdf_neutral, fvec_in, moments, vzeta, vr, vz, vz_spectral, 
                                    dt, collisions.mxwl_diff.D_nn)
            write_debug_IO("neutral_vz_maxwell_diffusion!")
        end

        if advance.external_source
            total_external_ion_sources!(fvec_out.pdf, fvec_in, moments, external_source_settings.ion,
                                vperp, vpa, dt, scratch_dummy)
            write_debug_IO("total_external_ion_sources!")
        end
        if advance.neutral_external_source
            total_external_neutral_sources!(fvec_out.pdf_neutral, fvec_in, moments,
                                    external_source_settings.neutral, vzeta, vr, vz, dt)
            write_debug_IO("total_external_neutral_sources!")
        end

        # add numerical dissipation
        if advance.ion_numerical_dissipation
            vpa_dissipation!(fvec_out.pdf, fvec_in.pdf, vpa, vpa_spectral, dt,
                            num_diss_params.ion.vpa_dissipation_coefficient)
            write_debug_IO("vpa_dissipation!")
            vperp_dissipation!(fvec_out.pdf, fvec_in.pdf, vperp, vperp_spectral, dt,
                            num_diss_params.ion.vperp_dissipation_coefficient)
            write_debug_IO("vperp_dissipation!")
            z_dissipation!(fvec_out.pdf, fvec_in.pdf, z, z_spectral, dt,
                        num_diss_params.ion.z_dissipation_coefficient, scratch_dummy)
            write_debug_IO("z_dissipation!")
            r_dissipation!(fvec_out.pdf, fvec_in.pdf, r, r_spectral, dt,
                        num_diss_params.ion.r_dissipation_coefficient, scratch_dummy)
            write_debug_IO("r_dissipation!")
        end
        if advance.neutral_numerical_dissipation
            vz_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, vz,
                                    vz_spectral, dt, num_diss_params.neutral.vz_dissipation_coefficient)
            write_debug_IO("vz_dissipation_neutral!")
            z_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, z, z_spectral,
                                dt, num_diss_params.neutral.z_dissipation_coefficient, scratch_dummy)
            write_debug_IO("z_dissipation_neutral!")
            r_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, r, r_spectral,
                                dt, num_diss_params.neutral.r_dissipation_coefficient, scratch_dummy)
            write_debug_IO("r_dissipation_neutral!")
        end
        # advance with the Fokker-Planck self-collision operator
        if advance.explicit_weakform_fp_collisions
            update_entropy_diagnostic = (istage == 1)
            if collisions.fkpl.self_collisions
                # self collisions for each species
                explicit_fokker_planck_collisions_weak_form!(fvec_out.pdf,fvec_in.pdf,moments.ion.dSdt,composition,
                                    collisions,dt,fp_arrays,r,z,vperp,vpa,vperp_spectral,vpa_spectral,scratch_dummy,
                                                        diagnose_entropy_production = update_entropy_diagnostic)
                write_debug_IO("explicit_fokker_planck_collisions_weak_form!")
            end
            if collisions.fkpl.slowing_down_test
            # include cross-collsions with fixed Maxwellian backgrounds
                explicit_fp_collisions_weak_form_Maxwellian_cross_species!(fvec_out.pdf,fvec_in.pdf,moments.ion.dSdt,
                                composition,collisions,dt,fp_arrays,r,z,vperp,vpa,vperp_spectral,vpa_spectral,
                                                diagnose_entropy_production = update_entropy_diagnostic)
                write_debug_IO("explicit_fp_collisions_weak_form_Maxwellian_cross_species!")
            end
        end
    end
    # End of advance for distribution function

    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments)
    return nothing
end

@timeit global_timer backward_euler!(
                         fvec_out, fvec_in, scratch_electron, pdf, fields, moments,
                         advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, dt,
                         t_params, spectral_objects, composition, collisions, geometry,
                         scratch_dummy, manufactured_source_list,
                         external_source_settings, num_diss_params, gyroavs,
                         nl_solver_params, advance, fp_arrays, istage) = begin

    vpa_spectral, vperp_spectral, r_spectral, z_spectral = spectral_objects.vpa_spectral, spectral_objects.vperp_spectral, spectral_objects.r_spectral, spectral_objects.z_spectral
    vz_spectral, vr_spectral, vzeta_spectral = spectral_objects.vz_spectral, spectral_objects.vr_spectral, spectral_objects.vzeta_spectral
    vpa_advect, vperp_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.vperp_advect, advect_objects.r_advect, advect_objects.z_advect
    electron_z_advect, electron_vpa_advect = advect_objects.electron_z_advect, advect_objects.electron_vpa_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    success = true
    if t_params.kinetic_electron_solver == implicit_steady_state
        electron_success = implicit_electron_advance!(fvec_out, fvec_in, pdf,
                                                      scratch_electron, moments, fields,
                                                      collisions, composition, geometry,
                                                      external_source_settings,
                                                      num_diss_params, r, z, vperp, vpa,
                                                      r_spectral, z_spectral,
                                                      vperp_spectral, vpa_spectral,
                                                      electron_z_advect,
                                                      electron_vpa_advect, gyroavs,
                                                      scratch_dummy, t_params.electron,
                                                      dt,
                                                      nl_solver_params.electron_advance)

        success = success && (electron_success == "")
    elseif t_params.kinetic_electron_solver == implicit_time_evolving
        t_params.electron.dt[] = dt
        for ir ∈ 1:r.n
            electron_success = electron_backward_euler!(
                          fvec_in, fvec_out, moments, fields.phi, collisions, composition, r,
                          z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
                          vpa_advect, scratch_dummy, t_params.electron,
                          external_source_settings, num_diss_params,
                          nl_solver_params.electron_advance, ir; evolve_p=true)
            success = success && electron_success
        end
    elseif t_params.kinetic_electron_solver == implicit_p_implicit_pseudotimestep
        max_electron_pdf_iterations = t_params.electron.max_pseudotimesteps
        max_electron_sim_time = t_params.electron.max_pseudotime
        electron_success = update_electron_pdf!(scratch_electron, pdf.electron.norm,
                                                moments, fields.phi, r, z, vperp, vpa,
                                                z_spectral, vperp_spectral, vpa_spectral,
                                                electron_z_advect, electron_vpa_advect,
                                                scratch_dummy, t_params.electron,
                                                collisions, composition,
                                                external_source_settings, num_diss_params,
                                                nl_solver_params.electron_advance,
                                                max_electron_pdf_iterations,
                                                max_electron_sim_time; evolve_p=true,
                                                ion_dt=dt)

        # Update `fvec_out.electron_p` with the new electron pressure
        @begin_r_z_region()
        fvec_out_electron_p = fvec_out.electron_p
        moments_electron_p = moments.electron.p
        @loop_r_z ir iz begin
            fvec_out_electron_p[iz,ir] = moments_electron_p[iz,ir]
        end

        success = success && (electron_success == "")

    elseif t_params.kinetic_electron_solver == implicit_p_explicit_pseudotimestep
        max_electron_pdf_iterations = t_params.electron.max_pseudotimesteps
        max_electron_sim_time = t_params.electron.max_pseudotime
        electron_success = update_electron_pdf!(scratch_electron, pdf.electron.norm,
                                                moments, fields.phi, r, z, vperp, vpa,
                                                z_spectral, vperp_spectral, vpa_spectral,
                                                electron_z_advect, electron_vpa_advect,
                                                scratch_dummy, t_params.electron,
                                                collisions, composition,
                                                external_source_settings, num_diss_params,
                                                nl_solver_params.electron_advance,
                                                max_electron_pdf_iterations,
                                                max_electron_sim_time;
                                                solution_method="artificial_time_derivative",
                                                evolve_p=true, ion_dt=dt)

        # Update `fvec_out.electron_p` with the new electron pressure
        @begin_r_z_region()
        fvec_out_electron_p = fvec_out.electron_p
        moments_electron_p = moments.electron.p
        @loop_r_z ir iz begin
            fvec_out_electron_p[iz,ir] = moments_electron_p[iz,ir]
        end

        success = success && (electron_success == "")

    elseif advance.electron_conduction
        success = implicit_braginskii_conduction!(fvec_out, fvec_in, moments, z, r, dt,
                                                  z_spectral, composition, collisions,
                                                  scratch_dummy,
                                                  nl_solver_params.electron_conduction)
    end

    if nl_solver_params.ion_advance !== nothing
        ion_success = implicit_ion_advance!(fvec_out, fvec_in, pdf, fields, moments,
                                            advect_objects, vz, vr, vzeta, vpa, vperp,
                                            gyrophase, z, r, t_params, dt,
                                            spectral_objects, composition, collisions,
                                            geometry, scratch_dummy,
                                            manufactured_source_list,
                                            external_source_settings, num_diss_params,
                                            gyroavs, nl_solver_params.ion_advance,
                                            advance, fp_arrays, istage)
        success = success && ion_success
    elseif advance.vpa_advection
        ion_success = implicit_vpa_advection!(fvec_out.pdf, fvec_in, fields, moments,
                                              z_advect, vpa_advect, vpa, vperp, z, r, dt,
                                              t_params.t[], r_spectral, z_spectral,
                                              vpa_spectral, composition, collisions,
                                              external_source_settings.ion, geometry,
                                              nl_solver_params.vpa_advection,
                                              advance.vpa_diffusion, num_diss_params,
                                              gyroavs, scratch_dummy)
        success = success && ion_success
    end

    return success
end

"""
    implicit_ion_advance!(fvec_out, fvec_in, pdf, fields, moments, advect_objects,
                          vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t_params, dt,
                          spectral_objects, composition, collisions, geometry,
                          scratch_dummy, manufactured_source_list,
                          external_source_settings, num_diss_params,
                          nl_solver_params, advance, fp_arrays, istage)

Do a backward-Euler timestep for all terms in the ion kinetic equation.
"""
@timeit global_timer implicit_ion_advance!(
                         fvec_out, fvec_in, pdf, fields, moments, advect_objects, vz, vr,
                         vzeta, vpa, vperp, gyrophase, z, r, t_params, dt,
                         spectral_objects, composition, collisions, geometry,
                         scratch_dummy, manufactured_source_list,
                         external_source_settings, num_diss_params, gyroavs,
                         nl_solver_params, advance, fp_arrays, istage) = begin

    t = t_params.t[]
    vpa_spectral, vperp_spectral, r_spectral, z_spectral = spectral_objects.vpa_spectral, spectral_objects.vperp_spectral, spectral_objects.r_spectral, spectral_objects.z_spectral
    vpa_advect, vperp_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.vperp_advect, advect_objects.r_advect, advect_objects.z_advect

    # Make a copy of fvec_in.pdf so we can apply boundary conditions at the 'new'
    # timestep, as these are the boundary conditions we need to apply the residual.
    f_old = scratch_dummy.implicit_buffer_vpavperpzrs_1
    @begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        f_old[ivpa,ivperp,iz,ir,is] = fvec_in.pdf[ivpa,ivperp,iz,ir,is]
    end

    coords = (s=composition.n_ion_species, r=r, z=z, vperp=vperp, vpa=vpa)
    icut_lower_z = scratch_dummy.int_buffer_rs_1
    icut_upper_z = scratch_dummy.int_buffer_rs_2
    zero = 1.0e-14

    rtol = nl_solver_params.rtol
    atol = nl_solver_params.atol

    @begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        @views hard_force_moment_constraints!(f_old[:,:,iz,ir,is], moments, vpa, vperp)
    end

    @begin_s_r_region()
    @loop_s_r is ir begin
        if z.irank == 0
            iz = 1
            @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, moments.ion.vth[iz,ir,is],
                                             fvec_in.upar[iz,ir,is],
                                             moments.evolve_p,
                                             moments.evolve_upar)
            icut_lower_z[ir,is] = vpa.n
            for ivpa ∈ vpa.n:-1:1
                # for left boundary in zed (z = -Lz/2), want
                # f(z=-Lz/2, v_parallel > 0) = 0
                if vpa.scratch[ivpa] ≤ zero
                    icut_lower_z[ir,is] = ivpa + 1
                    break
                end
            end
        end
        if z.irank == z.nrank - 1
            iz = z.n
            @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, moments.ion.vth[iz,ir,is],
                                             fvec_in.upar[iz,ir,is],
                                             moments.evolve_p,
                                             moments.evolve_upar)
            icut_upper_z[ir,is] = 0
            for ivpa ∈ 1:vpa.n
                # for right boundary in zed (z = Lz/2), want
                # f(z=Lz/2, v_parallel < 0) = 0
                if vpa.scratch[ivpa] ≥ -zero
                    icut_upper_z[ir,is] = ivpa - 1
                    break
                end
            end
        end
    end

    if vpa.n > 1
        # calculate the vpa advection speed, to ensure it is correct when used to apply the
        # boundary condition
        update_speed_vpa!(vpa_advect, fields, fvec_in, moments, vpa, vperp, z, r, composition,
                          collisions, external_source_settings.ion, t, geometry)
    end
    if z.n > 1
        @loop_s is begin
            # get the updated speed along the z direction using the current f
            @views update_speed_z!(z_advect[is], fvec_in.upar[:,:,is],
                                   moments.ion.vth[:,:,is], moments.evolve_upar,
                                   moments.evolve_p, fields, vpa, vperp, z, r, t,
                                   geometry, is)
        end
    end
    if r.n > 1
        @loop_s is begin
            # get the updated speed along the r direction using the current f
            @views update_speed_r!(r_advect[is], fvec_in.upar[:,:,is],
                                   moments.ion.vth[:,:,is], fields, moments.evolve_upar,
                                   moments.evolve_p, vpa, vperp, z, r, geometry, is)
        end
    end
    if vperp.n > 1
        # calculate the vpa advection speed, to ensure it is correct when used to apply the
        # boundary condition
        @begin_s_r_z_vpa_region()
        @loop_s is begin
            # get the updated speed along the r direction using the current f
            @views update_speed_vperp!(vperp_advect[is], vpa, vperp, z, r, z_advect[is],
                                       r_advect[is], geometry)
        end
    end

    function apply_bc!(x)
        if vpa.n > 1
            @begin_s_r_z_vperp_region()
            @loop_s_r_z_vperp is ir iz ivperp begin
                @views enforce_v_boundary_condition_local!(x[:,ivperp,iz,ir,is], vpa.bc,
                                                           vpa_advect[is].speed[:,ivperp,iz,ir],
                                                           advance.vpa_diffusion, vpa,
                                                           vpa_spectral)
            end
        end
        if vperp.n > 1
            @begin_s_r_z_vpa_region()
            enforce_vperp_boundary_condition!(x, vperp.bc, vperp, vperp_spectral,
                                              vperp_adv, vperp_diffusion)
        end

        if z.bc == "wall" && (z.irank == 0 || z.irank == z.nrank - 1)
            # Wall boundary conditions. Note that as density, upar, p do not
            # change in this implicit step, f_new, f_old, and residual should all
            # be zero at exactly the same set of grid points, so it is reasonable
            # to zero-out `residual` to impose the boundary condition. We impose
            # this after subtracting f_old in case rounding errors, etc. mean that
            # at some point f_old had a different boundary condition cut-off
            # index.
            @begin_s_r_vperp_region()
            if z.irank == 0
                iz = 1
                @loop_s_r_vperp is ir ivperp begin
                    x[icut_lower_z[ir,is]:end,ivperp,iz,ir,is] .= 0.0
                end
            end
            if z.irank == z.nrank - 1
                iz = z.n
                @loop_s_r_vperp is ir ivperp begin
                    x[1:icut_upper_z[ir,is],ivperp,iz,ir,is] .= 0.0
                end
            end
        end

        return nothing
    end

    # Use a forward-Euler step as the initial guess for fvec_out.pdf
    euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments, advect_objects, vz, vr,
                        vzeta, vpa, vperp, gyrophase, z, r, t_params, dt,
                        spectral_objects, composition, collisions, geometry,
                        scratch_dummy, manufactured_source_list, external_source_settings,
                        num_diss_params, advance, fp_arrays, istage)

    # Apply the 'new' boundary conditions to f_old, so it has the same boundary conditions
    # as we will apply to the residual, so that f_new obeys the 'new' boundary conditions.
    apply_bc!(f_old)
    # Also apply the bc to the forward-Euler updated values which are the initial state
    # for 'f_new'.
    apply_bc!(fvec_out.pdf)
    hard_force_moment_constraints!(fvec_out.pdf, moments, vpa, vperp)

    # Define a function whose input is `f_new`, so that when it's output
    # `residual` is zero, f_new is the result of a backward-Euler timestep:
    #   (f_new - f_old) / dt = RHS(f_new)
    # ⇒ f_new - f_old - dt*RHS(f_new) = 0
    function residual_func!(residual, f_new; krylov=false)
        @begin_s_r_z_vperp_vpa_region()
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            residual[ivpa,ivperp,iz,ir,is] = f_old[ivpa,ivperp,iz,ir,is]
        end

        # scratch_pdf struct containing the array passed as f_new
        new_scratch = scratch_pdf(f_new, fvec_out.density, fvec_out.upar, fvec_out.p,
                                  fvec_out.temp_z_s, fvec_out.electron_density,
                                  fvec_out.electron_upar, fvec_out.electron_p,
                                  fvec_out.electron_temp, fvec_out.pdf_neutral,
                                  fvec_out.density_neutral, fvec_out.uz_neutral,
                                  fvec_out.pz_neutral, fvec_out.controller_integrals)
        # scratch_pdf struct containing the array passed as residual
        residual_scratch = scratch_pdf(residual, fvec_out.density, fvec_out.upar,
                                       fvec_out.p, fvec_out.temp_z_s,
                                       fvec_out.electron_density, fvec_out.electron_upar,
                                       fvec_out.electron_p, fvec_out.electron_temp,
                                       fvec_out.pdf_neutral, fvec_out.density_neutral,
                                       fvec_out.uz_neutral, fvec_out.pz_neutral, fvec_out.controller_integrals)

        # Ensure moments are consistent with f_new
        update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition,
                                r_spectral, geometry, gyroavs, scratch_dummy, z_advect,
                                collisions, false)
        calculate_ion_moment_derivatives!(moments, new_scratch, scratch_dummy, z,
                                          z_spectral,
                                          num_diss_params.ion.moment_dissipation_coefficient)

        euler_time_advance!(residual_scratch, new_scratch, pdf, fields, moments,
                            advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z,
                            r, t_params, dt, spectral_objects, composition, collisions,
                            geometry, scratch_dummy, manufactured_source_list,
                            external_source_settings, num_diss_params, advance, fp_arrays,
                            istage)

        # Make sure updated f will not contain negative values
        #@. residual = max(residual, minval)

        # Now
        #   residual = f_old + dt*RHS(f_new)
        # so update to desired residual
        @begin_s_r_z_vperp_vpa_region()
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            residual[ivpa,ivperp,iz,ir,is] = f_new[ivpa,ivperp,iz,ir,is] - residual[ivpa,ivperp,iz,ir,is]
        end

        apply_bc!(residual)

        @begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
            @views moment_constraints_on_residual!(residual[:,:,iz,ir,is],
                                                   f_new[:,:,iz,ir,is], moments, vpa)
        end

        return nothing
    end

    # No preconditioning for now
    left_preconditioner = identity
    right_preconditioner = identity

    # Buffers
    # Note vpa,scratch is used by advance_f!, so we cannot use it here.
    residual = scratch_dummy.implicit_buffer_vpavperpzrs_2
    delta_x = scratch_dummy.implicit_buffer_vpavperpzrs_3
    rhs_delta = scratch_dummy.implicit_buffer_vpavperpzrs_4
    v = scratch_dummy.implicit_buffer_vpavperpzrs_5
    w = scratch_dummy.implicit_buffer_vpavperpzrs_6

    # Using the forward-Euler step seems (in at least one case) to slightly
    # increase the number of iterations, so skip this.
    ## Use forward-Euler step for initial guess
    #residual_func!(residual, this_f_out)
    #this_f_out .+= residual

    success = newton_solve!(fvec_out.pdf, residual_func!, residual, delta_x,
                            rhs_delta, v, w, nl_solver_params, coords=coords,
                            left_preconditioner=left_preconditioner,
                            right_preconditioner=right_preconditioner)

    return success
end

"""
update the vector containing the pdf and any evolved moments of the pdf
for use in the Runge-Kutta time advance
"""
function update_solution_vector!(new_evolved, old_evolved, moments, composition, vpa, vperp, z, r)
    @begin_s_r_z_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_evolved.pdf[ivpa,ivperp,iz,ir,is] = old_evolved.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        new_evolved.density[iz,ir,is] = old_evolved.density[iz,ir,is]
        new_evolved.upar[iz,ir,is] = old_evolved.upar[iz,ir,is]
        new_evolved.p[iz,ir,is] = old_evolved.p[iz,ir,is]
    end
    if length(new_evolved.pdf_electron) > 0
        @begin_r_z_vperp_vpa_region()
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            new_evolved.pdf_electron[ivpa,ivperp,iz,ir] = old_evolved.pdf_electron[ivpa,ivperp,iz,ir]
        end
    end
    @begin_r_z_region()
    @loop_r_z ir iz begin
        new_evolved.electron_density[iz,ir] = old_evolved.electron_density[iz,ir]
        new_evolved.electron_upar[iz,ir] = old_evolved.electron_upar[iz,ir]
        new_evolved.electron_p[iz,ir] = old_evolved.electron_p[iz,ir]
        new_evolved.electron_temp[iz,ir] = old_evolved.electron_temp[iz,ir]
    end
    if composition.n_neutral_species > 0
        @begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            new_evolved.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = old_evolved.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            new_evolved.density_neutral[iz,ir,isn] = old_evolved.density_neutral[iz,ir,isn]
            new_evolved.uz_neutral[iz,ir,isn] = old_evolved.uz_neutral[iz,ir,isn]
            new_evolved.p_neutral[iz,ir,isn] = old_evolved.p_neutral[iz,ir,isn]
        end
    end
    @begin_serial_region
    @serial_region begin
        new_evolved.ion_external_source_controller_integral .= old_evolved.ion_external_source_controller_integral
        #new_evolved.electron_external_source_controller_integral .= electron_evolved.ion_external_source_controller_integral
        if composition.n_neutral_species > 0
            new_evolved.neutral_external_source_controller_integral .= old_evolved.neutral_external_source_controller_integral
        end
    end
    return nothing
end

end
