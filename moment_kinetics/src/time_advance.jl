"""
"""
module time_advance

export setup_time_advance!
export time_advance!
export allocate_advection_structs
export setup_dummy_and_buffer_arrays

using MPI
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float, allocate_shared_bool
using ..communication
using ..communication: _block_synchronize
using ..debugging
using ..file_io: write_data_to_ascii, write_all_moments_data_to_binary, write_all_dfns_data_to_binary, debug_dump
using ..initial_conditions: initialize_electrons!
using ..looping
using ..moment_kinetics_structs: scratch_pdf
using ..velocity_moments: update_moments!, update_moments_neutral!, reset_moments_status!
using ..velocity_moments: update_density!, update_upar!, update_ppar!, update_pperp!, update_qpar!, update_vth!
using ..velocity_moments: update_neutral_density!, update_neutral_qz!
using ..velocity_moments: update_neutral_uzeta!, update_neutral_uz!, update_neutral_ur!
using ..velocity_moments: update_neutral_pzeta!, update_neutral_pz!, update_neutral_pr!
using ..velocity_moments: calculate_ion_moment_derivatives!, calculate_neutral_moment_derivatives!
using ..velocity_moments: calculate_electron_moment_derivatives!
using ..velocity_moments: update_chodura!
using ..velocity_grid_transforms: vzvrvzeta_to_vpavperp!, vpavperp_to_vzvrvzeta!
using ..boundary_conditions: enforce_boundary_conditions!
using ..boundary_conditions: enforce_neutral_boundary_conditions!
using ..input_structs
using ..moment_constraints: hard_force_moment_constraints!,
                            hard_force_moment_constraints_neutral!
using ..advection: setup_advection
using ..z_advection: update_speed_z!, z_advection!
using ..r_advection: update_speed_r!, r_advection!
using ..neutral_r_advection: update_speed_neutral_r!, neutral_advection_r!
using ..neutral_z_advection: update_speed_neutral_z!, neutral_advection_z!
using ..neutral_vz_advection: update_speed_neutral_vz!, neutral_advection_vz!
using ..vperp_advection: update_speed_vperp!, vperp_advection!
using ..vpa_advection: update_speed_vpa!, vpa_advection!
using ..charge_exchange: charge_exchange_collisions_1V!, charge_exchange_collisions_3V!
using ..electron_kinetic_equation: update_electron_pdf!
using ..ionization: ionization_collisions_1V!, ionization_collisions_3V!, constant_ionization_source!
using ..krook_collisions: krook_collisions!
using ..external_sources
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
using ..manufactured_solns: manufactured_sources
using ..advection: advection_info
using ..runge_kutta: rk_update_evolved_moments!, rk_update_evolved_moments_neutral!,
                     rk_update_variable!, rk_error_variable!,
                     setup_runge_kutta_coefficients!, local_error_norm,
                     adaptive_timestep_update_t_params!
using ..utils: to_minutes, get_minimum_CFL_z, get_minimum_CFL_vpa,
               get_minimum_CFL_neutral_z, get_minimum_CFL_neutral_vz
using ..electron_fluid_equations: calculate_electron_density!
using ..electron_fluid_equations: calculate_electron_upar_from_charge_conservation!
using ..electron_fluid_equations: calculate_electron_qpar!
using ..electron_fluid_equations: calculate_electron_parallel_friction_force!
using ..electron_fluid_equations: electron_energy_equation!, update_electron_vth_temperature!
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
    begin_serial_region()
    z_advect = setup_advection(n_ion_species, z, vpa, vperp, r)
    # create structure r_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the ion kinetic equation dealing
    # with advection in r
    begin_serial_region()
    r_advect = setup_advection(n_ion_species, r, vpa, vperp, z)
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the ion kinetic equation dealing
    # with advection in vpa
    begin_serial_region()
    vpa_advect = setup_advection(n_ion_species, vpa, vperp, z, r)
    # create structure vperp_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the ion kinetic equation dealing
    # with advection in vperp
    begin_serial_region()
    vperp_advect = setup_advection(n_ion_species, vperp, vpa, z, r)
    ##                                   ##
    # electron particle advection structs #
    ##                                   ##
    # create structure electron_z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the part of the electron kinetic equation dealing
    # with advection in z
    begin_serial_region()
    electron_z_advect = setup_advection(1, z, vpa, vperp, r)
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the part of the electron kinetic equation dealing
    # with advection in vpa
    begin_serial_region()
    electron_vpa_advect = setup_advection(1, vpa, vperp, z, r)
    ##                                  ##
    # neutral particle advection structs #
    ##                                  ##
    # create structure neutral_z_advect for neutral particle advection
    begin_serial_region()
    neutral_z_advect = setup_advection(n_neutral_species_alloc, z, vz, vr, vzeta, r)
    # create structure neutral_r_advect for neutral particle advection
    begin_serial_region()
    neutral_r_advect = setup_advection(n_neutral_species_alloc, r, vz, vr, vzeta, z)
    # create structure neutral_vz_advect for neutral particle advection
    begin_serial_region()
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
    setup_time_info(t_input; electrons=nothing)

Create a [`input_structs.time_info`](@ref) struct using the settings in `t_input`.

If something is passed in `electron`, it is stored in the `electron_t_params` member of
the returned `time_info`.
"""
function setup_time_info(t_input, code_time, dt_reload, dt_before_last_fail_reload,
                         manufactured_solns_input; electron=nothing)
    dt_shared = allocate_shared_float(1)
    previous_dt_shared = allocate_shared_float(1)
    next_output_time = allocate_shared_float(1)
    dt_before_output = allocate_shared_float(1)
    dt_before_last_fail = allocate_shared_float(1)
    step_to_output = allocate_shared_bool(1)
    if block_rank[] == 0
        dt_shared[] = dt_reload === nothing ? t_input.dt : dt_reload
        previous_dt_shared[] = dt_reload === nothing ? t_input.dt : dt_reload
        next_output_time[] = 0.0
        dt_before_output[] = dt_reload === nothing ? t_input.dt : dt_reload
        dt_before_last_fail[] = dt_before_last_fail_reload === nothing ? Inf : dt_before_last_fail_reload
        step_to_output[] = false
    end
    _block_synchronize()

    end_time = code_time + t_input.dt * t_input.nstep
    epsilon = 1.e-11
    moments_output_times = [code_time + i*t_input.dt
                            for i ∈ t_input.nwrite:t_input.nwrite:t_input.nstep]
    if moments_output_times[end] < end_time - epsilon
        push!(moments_output_times, end_time)
    end
    dfns_output_times = [code_time + i*t_input.dt
                         for i ∈ t_input.nwrite_dfns:t_input.nwrite_dfns:t_input.nstep]
    if dfns_output_times[end] < end_time - epsilon
        push!(dfns_output_times, end_time)
    end
    rk_coefs, n_rk_stages, rk_order, adaptive, low_storage, CFL_prefactor =
        setup_runge_kutta_coefficients!(t_input.type,
                                        t_input.CFL_prefactor,
                                        t_input.split_operators)
    if t_input.high_precision_error_sum
        error_sum_zero = Float128(0.0)
    else
        error_sum_zero = 0.0
    end
    return time_info(t_input.nstep, end_time, dt_shared, previous_dt_shared, next_output_time,
                         dt_before_output, dt_before_last_fail, CFL_prefactor,
                         step_to_output, Ref(0), Ref(0), mk_int[], mk_int[],
                         moments_output_times, dfns_output_times, t_input.type, rk_coefs,
                         n_rk_stages, rk_order, adaptive, low_storage, t_input.rtol,
                         t_input.atol, t_input.atol_upar, t_input.step_update_prefactor,
                         t_input.max_increase_factor,
                         t_input.max_increase_factor_near_last_fail,
                         t_input.last_fail_proximity_factor, t_input.minimum_dt,
                         t_input.maximum_dt, error_sum_zero, t_input.split_operators,
                         t_input.steady_state_residual, t_input.converged_residual_value,
                         manufactured_solns_input.use_for_advance, t_input.stopfile_name,
                         electron)
end

"""
create arrays and do other work needed to setup
the main time advance loop.
this includes creating and populating structs
for Chebyshev transforms, velocity space moments,
EM fields, and advection terms
"""
function setup_time_advance!(pdf, fields, vz, vr, vzeta, vpa, vperp, z, r, vz_spectral,
                             vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral,
                             z_spectral, r_spectral, composition, drive_input, moments,
                             t_input, code_time, dt_reload, dt_before_last_fail_reload,
                             electron_dt_reload, electron_dt_before_last_fail_reload,
                             collisions, species, geometry, boundary_distributions,
                             external_source_settings, num_diss_params,
                             manufactured_solns_input, advection_structs, scratch_dummy,
                             io_input, restarting, restart_had_kinetic_electrons,
                             input_dict)
    # define some local variables for convenience/tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    ion_mom_diss_coeff = num_diss_params.ion.moment_dissipation_coefficient
    electron_mom_diss_coeff = num_diss_params.electron.moment_dissipation_coefficient
    neutral_mom_diss_coeff = num_diss_params.neutral.moment_dissipation_coefficient

    if composition.electron_physics == kinetic_electrons
        electron_t_params = setup_time_info(t_input.electron_t_input, 0.0,
                                            electron_dt_reload,
                                            electron_dt_before_last_fail_reload,
                                            manufactured_solns_input)
        # Make Vectors that count which variable caused timestep limits and timestep failures
        # the right length. Do this setup even when not using adaptive timestepping, because
        # it is easier than modifying the file I/O according to whether we are using adaptive
        # timestepping.
        #
        # Entries for limit by accuracy (which is an average over all variables),
        # max_increase_factor, minimum_dt and maximum_dt
        push!(electron_t_params.limit_caused_by, 0, 0, 0, 0, 0)

        # electron pdf
        push!(electron_t_params.limit_caused_by, 0, 0)
        push!(electron_t_params.failure_caused_by, 0)

        # electron ppar
        push!(electron_t_params.failure_caused_by, 0)
    else
        electron_t_params = nothing
    end
    t_params = setup_time_info(t_input, code_time, dt_reload, dt_before_last_fail_reload,
                               manufactured_solns_input; electron=electron_t_params)

    # Make Vectors that count which variable caused timestep limits and timestep failures
    # the right length. Do this setup even when not using adaptive timestepping, because
    # it is easier than modifying the file I/O according to whether we are using adaptive
    # timestepping.
    #
    # Entries for limit by accuracy (which is an average over all variables),
    # max_increase_factor, minimum_dt and maximum_dt
    push!(t_params.limit_caused_by, 0, 0, 0, 0, 0)

    # ion pdf
    push!(t_params.limit_caused_by, 0, 0)
    push!(t_params.failure_caused_by, 0)
    if moments.evolve_density
        # ion density
        push!(t_params.failure_caused_by, 0)
    end
    if moments.evolve_upar
        # ion flow
        push!(t_params.failure_caused_by, 0)
    end
    if moments.evolve_ppar
        # ion pressure
        push!(t_params.failure_caused_by, 0)
    end
    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons)
        # electron pressure
        push!(t_params.failure_caused_by, 0)
    end
    if composition.n_neutral_species > 0
        # neutral pdf
        push!(t_params.limit_caused_by, 0, 0)
        push!(t_params.failure_caused_by, 0)
        if moments.evolve_density
            # neutral density
            push!(t_params.failure_caused_by, 0)
        end
        if moments.evolve_upar
            # neutral flow
            push!(t_params.failure_caused_by, 0)
        end
        if moments.evolve_ppar
            # neutral pressure
            push!(t_params.failure_caused_by, 0)
        end
    end

    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    advance = setup_advance_flags(moments, composition, t_params, collisions,
                                  external_source_settings, num_diss_params,
                                  manufactured_solns_input, r, z, vperp, vpa, vzeta, vr,
                                  vz)

    begin_serial_region()

    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    n_rk_stages = t_params.n_rk_stages
    if t_params.electron !== nothing
        n_rk_stages = max(n_rk_stages, t_params.electron.n_rk_stages)
    end
    scratch = setup_scratch_arrays(moments, pdf, n_rk_stages)
    # setup dummy arrays & buffer arrays for z r MPI
    n_neutral_species_alloc = max(1,composition.n_neutral_species)
    # create arrays for Fokker-Planck collisions 
    if advance.explicit_weakform_fp_collisions
        fp_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; precompute_weights=true)
    else
        fp_arrays = nothing
    end

    # Now that `t_params` and `scratch` have been created, initialize electrons if
    # necessary
    if !restarting || (composition.electron_physics == kinetic_electrons &&
                       !restart_had_kinetic_electrons)
        initialize_electrons!(pdf, moments, fields, geometry, composition, r, z,
                              vperp, vpa, vzeta, vr, vz, z_spectral, r_spectral,
                              vperp_spectral, vpa_spectral, collisions,
                              external_source_settings, scratch_dummy, scratch, t_params,
                              t_input, num_diss_params, advection_structs, io_input,
                              input_dict; restart_from_Boltzmann_electrons=restarting)
    end

    # update the derivatives of the electron moments as these may be needed when
    # computing the electrostatic potential (and components of the electric field)
    calculate_electron_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                           electron_mom_diss_coeff, composition.electron_physics)
    # initialize the electrostatic potential
    begin_serial_region()
    update_phi!(fields, scratch[1], z, r, composition, collisions, moments, z_spectral, r_spectral, scratch_dummy)
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
        begin_s_z_vperp_vpa_region()
        @loop_s is begin
            @views update_speed_r!(r_advect[is], moments.ion.upar[:,:,is],
                                   moments.ion.vth[:,:,is], fields, moments.evolve_upar,
                                   moments.evolve_ppar, vpa, vperp, z, r, geometry)
        end
        # enforce prescribed boundary condition in r on the distribution function f
    end

    if z.n > 1
        # initialise the z advection speed
        begin_s_r_vperp_vpa_region()
        @loop_s is begin
            @views update_speed_z!(z_advect[is], moments.ion.upar[:,:,is],
                                   moments.ion.vth[:,:,is], moments.evolve_upar,
                                   moments.evolve_ppar, fields, vpa, vperp, z, r, 0.0,
                                   geometry)
        end
    end

    # initialise the vpa advection speed
    begin_s_r_z_vperp_region()
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, vperp, z, r,
                      composition, collisions, external_source_settings.ion, 0.0,
                      geometry)

    # initialise the vperp advection speed
    # Note that z_advect and r_advect are arguments of update_speed_vperp!
    # This means that z_advect[is].speed and r_advect[is].speed are used to determine
    # vperp_advect[is].speed, so z_advect and r_advect must always be updated before
    # vperp_advect is updated and used.
    if vperp.n > 1
        begin_serial_region()
        @serial_region begin
            for is ∈ 1:n_ion_species
                @views update_speed_vperp!(vperp_advect[is], vpa, vperp, z, r, z_advect[is], r_advect[is], geometry)
            end
        end
    end
    
    ##
    # Neutral particle advection
    ##

    if n_neutral_species > 0 && r.n > 1
        # initialise the r advection speed
        begin_sn_z_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_r!(neutral_r_advect[isn], r, z, vzeta, vr, vz)
        end
    end

    if n_neutral_species > 0 && z.n > 1
        # initialise the z advection speed
        begin_sn_r_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_z!(neutral_z_advect[isn],
                                           moments.neutral.uz[:,:,isn],
                                           moments.neutral.vth[:,:,isn],
                                           moments.evolve_upar, moments.evolve_ppar, vz,
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
        begin_sn_r_z_vzeta_vr_region()
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
                                                        composition, geometry.input, collisions,
                                                        num_diss_params, species)
    else
        manufactured_source_list = false # dummy Bool to be passed as argument instead of list
    end

    if !restarting
        begin_serial_region()
        # ensure initial pdf has no negative values
        force_minimum_pdf_value!(pdf.ion.norm, num_diss_params.ion.force_minimum_pdf_value)
        force_minimum_pdf_value_neutral!(pdf.neutral.norm, num_diss_params.neutral.force_minimum_pdf_value)
        # enforce boundary conditions and moment constraints to ensure a consistent initial
        # condition
        enforce_boundary_conditions!(
            pdf.ion.norm, boundary_distributions.pdf_rboundary_ion,
            moments.ion.dens, moments.ion.upar, moments.ion.ppar, moments,
            vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_spectral, vperp_spectral,
            vpa_advect, vperp_advect, z_advect, r_advect,
            composition, scratch_dummy, advance.r_diffusion,
            advance.vpa_diffusion, advance.vperp_diffusion)
        # Ensure normalised pdf exactly obeys integral constraints if evolving moments
        begin_s_r_z_region()
        if moments.evolve_density && moments.enforce_conservation
            A = moments.ion.constraints_A_coefficient
            B = moments.ion.constraints_B_coefficient
            C = moments.ion.constraints_C_coefficient
            @loop_s_r_z is ir iz begin
                (A[iz,ir,is], B[iz,ir,is], C[iz,ir,is]) =
                    @views hard_force_moment_constraints!(pdf.ion.norm[:,:,iz,ir,is],
                                                          moments, vpa)
            end
        end
        # update moments in case they were affected by applying boundary conditions or
        # constraints to the pdf
        reset_moments_status!(moments)
        update_moments!(moments, pdf.ion.norm, vpa, vperp, z, r, composition)
        # update the Chodura diagnostic -- note that the pdf should be the unnormalised one
        # so this will break for the split moments cases
        update_chodura!(moments,pdf.ion.norm,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
        # enforce boundary conditions in r and z on the neutral particle distribution function
        if n_neutral_species > 0
            # Note, so far vr and vzeta do not need advect objects, so pass `nothing` for
            # those as a placeholder
            enforce_neutral_boundary_conditions!(
                pdf.neutral.norm, pdf.ion.norm, boundary_distributions,
                moments.neutral.dens, moments.neutral.uz, moments.neutral.pz, moments,
                moments.ion.dens, moments.ion.upar, fields.Er, vzeta_spectral,
                vr_spectral, vz_spectral, neutral_r_advect, neutral_z_advect, nothing,
                nothing, neutral_vz_advect, r, z, vzeta, vr, vz, composition, geometry,
                scratch_dummy, advance.r_diffusion, advance.vz_diffusion)
            begin_sn_r_z_region()
            if moments.evolve_density && moments.enforce_conservation
                A = moments.neutral.constraints_A_coefficient
                B = moments.neutral.constraints_B_coefficient
                C = moments.neutral.constraints_C_coefficient
                @loop_sn_r_z isn ir iz begin
                    (A[iz,ir,isn], B[iz,ir,isn], C[iz,ir,isn]) =
                        @views hard_force_moment_constraints_neutral!(
                            pdf.neutral.norm[:,:,:,iz,ir,isn], moments, vz)
                end
            end
            update_moments_neutral!(moments, pdf.neutral.norm, vz, vr, vzeta, z, r,
                                    composition)
        end

        # update scratch arrays in case they were affected by applying boundary conditions
        # or constraints to the pdf
        begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
            scratch[1].pdf[:,:,iz,ir,is] .= pdf.ion.norm[:,:,iz,ir,is]
            scratch[1].density[iz,ir,is] = moments.ion.dens[iz,ir,is]
            scratch[1].upar[iz,ir,is] = moments.ion.upar[iz,ir,is]
            scratch[1].ppar[iz,ir,is] = moments.ion.ppar[iz,ir,is]
        end

        # update the electron density, parallel flow and parallel pressure (and temperature)
        # in case the corresponding ion quantities have been changed by applying
        # constraints to the ion pdf
        calculate_electron_density!(moments.electron.dens, moments.electron.dens_updated, moments.ion.dens)
        calculate_electron_upar_from_charge_conservation!(moments.electron.upar, moments.electron.upar_updated,
                                                          moments.electron.dens, moments.ion.upar, moments.ion.dens,
                                                          composition.electron_physics, r, z)
        begin_serial_region()
        # compute the updated electron temperature
        # NB: not currently necessary, as initial vth is not directly dependent on ion quantities
        @serial_region begin
            @. moments.electron.temp = moments.electron.vth^2
        end
        # as the electron temperature has now been updated, set the appropriate flag
        moments.electron.temp_updated[] = true
        # compute the updated electron parallel pressure
        @serial_region begin
            @. moments.electron.ppar = 0.5 * moments.electron.dens * moments.electron.temp
        end
        # as the electron ppar has now been updated, set the appropriate flag
        moments.electron.ppar_updated[] = true
        # calculate the zed derivative of the initial electron temperature, potentially
        # needed in the following calculation of the electron parallel friction force and
        # parallel heat flux
        @views derivative_z!(moments.electron.dT_dz, moments.electron.temp, 
            scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
            scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
        # calculate the electron parallel heat flux
        calculate_electron_qpar!(moments.electron, pdf.electron, moments.electron.ppar,
            moments.electron.upar, moments.ion.upar, collisions.nu_ei,
            composition.me_over_mi, composition.electron_physics, vpa)
        # calculate the electron-ion parallel friction force
        calculate_electron_parallel_friction_force!(moments.electron.parallel_friction, moments.electron.dens,
            moments.electron.upar, moments.ion.upar, moments.electron.dT_dz,
            composition.me_over_mi, collisions.nu_ei, composition.electron_physics)
        # update the electron moment entries in the scratch array
        begin_r_z_region()
        @loop_r_z ir iz begin
            scratch[1].electron_density[iz,ir] = moments.electron.dens[iz,ir]
            scratch[1].electron_upar[iz,ir] = moments.electron.upar[iz,ir]
            scratch[1].electron_ppar[iz,ir] = moments.electron.ppar[iz,ir]
            scratch[1].electron_temp[iz,ir] = moments.electron.temp[iz,ir]
        end

        begin_sn_r_z_region(no_synchronize=true)
        @loop_sn_r_z isn ir iz begin
            scratch[1].pdf_neutral[:,:,:,iz,ir,isn] .= pdf.neutral.norm[:,:,:,iz,ir,isn]
            scratch[1].density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            scratch[1].uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            scratch[1].pz_neutral[iz,ir,isn] = moments.neutral.pz[iz,ir,isn]
        end
    end

    calculate_ion_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      ion_mom_diss_coeff)
    calculate_electron_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      electron_mom_diss_coeff, composition.electron_physics)
    calculate_neutral_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, 
                                      neutral_mom_diss_coeff)
    # update the electrostatic potential and components of the electric field, as pdfs and moments
    # may have changed due to enforcing boundary/moment constraints                                      
    update_phi!(fields, scratch[1], z, r, composition, collisions, moments, z_spectral, r_spectral,
                scratch_dummy)

    # Ensure all processes are synchronized at the end of the setup
    _block_synchronize()

    return moments, fields, spectral_objects, scratch, advance, t_params,
    fp_arrays, scratch_dummy, manufactured_source_list
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
    advance_cx_1V = false
    advance_cx = false
    advance_ionization = false
    advance_ionization_1V = false
    advance_ionization_source = false
    advance_krook_collisions_ii = false
    advance_external_source = false
    advance_numerical_dissipation = false
    advance_sources = false
    advance_continuity = false
    advance_force_balance = false
    advance_energy = false
    advance_electron_energy = false
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
        advance_vpa_advection = vpa.n > 1 && z.n > 1
        advance_vperp_advection = vperp.n > 1 && z.n > 1
        advance_z_advection = z.n > 1
        advance_r_advection = r.n > 1
        if collisions.nuii > 0.0 && vperp.n > 1
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
            if moments.evolve_upar || moments.evolve_ppar
                advance_neutral_vz_advection = true
            end
            # if charge exchange collision frequency non-zero,
            # account for charge exchange collisions
            if abs(collisions.charge_exchange) > 0.0
                if vz.n == vpa.n && vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                    advance_cx_1V = true
                elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                    advance_cx = true
                else
                    error("If any perpendicular velocity has length>1 they all must. "
                          * "If all perpendicular velocities have length=1, then vpa and "
                          * "vz should be the same.\n"
                          * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                          * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
                end
            end
            # if ionization collision frequency non-zero,
            # account for ionization collisions
            if abs(collisions.ionization) > 0.0
                if vz.n == vpa.n && vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                    advance_ionization_1V = true
                elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                    advance_ionization = true
                else
                    error("If any perpendicular velocity has length>1 they all must. "
                          * "If all perpendicular velocities have length=1, then vpa and "
                          * "vz should be the same.\n"
                          * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                          * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
                end
            end
        end
        # exception for the case where ions are evolved alone but sourced by ionization
        if collisions.ionization > 0.0 && collisions.constant_ionization_rate
            advance_ionization_source = true
        end
        if collisions.krook_collision_frequency_prefactor_ii > 0.0
            advance_krook_collisions_ii = true
        end
        advance_external_source = external_source_settings.ion.active
        advance_neutral_external_source = external_source_settings.neutral.active
        advance_numerical_dissipation = true
        # if evolving the density, must advance the continuity equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_density
            advance_sources = true
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
            advance_sources = true
            advance_force_balance = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_force_balance = true
            end
        end
        # if evolving the parallel pressure, must advance the energy equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_ppar
            advance_sources = true
            advance_energy = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_energy = true
            end
        end
        # if treating the electrons as a fluid with Braginskii closure, or
        # moment-kinetically then advance the electron energy equation
        if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons)
            advance_electron_energy = true
        end

        # flag to determine if a d^2/dr^2 operator is present
        r_diffusion = (advance_numerical_dissipation && num_diss_params.ion.r_dissipation_coefficient > 0.0)
        # flag to determine if a d^2/dvpa^2 operator is present
        vpa_diffusion = ((advance_numerical_dissipation && num_diss_params.ion.vpa_dissipation_coefficient > 0.0) || explicit_weakform_fp_collisions)
        vperp_diffusion = ((advance_numerical_dissipation && num_diss_params.ion.vperp_dissipation_coefficient > 0.0) || explicit_weakform_fp_collisions)
        vz_diffusion = (advance_numerical_dissipation && num_diss_params.neutral.vz_dissipation_coefficient > 0.0)
    end

    manufactured_solns_test = manufactured_solns_input.use_for_advance

    return advance_info(advance_vpa_advection, advance_vperp_advection, advance_z_advection, advance_r_advection,
                        advance_neutral_z_advection, advance_neutral_r_advection,
                        advance_neutral_vz_advection, advance_cx, advance_cx_1V,
                        advance_ionization, advance_ionization_1V,
                        advance_ionization_source, advance_krook_collisions_ii,
                        explicit_weakform_fp_collisions,
                        advance_external_source, advance_numerical_dissipation,
                        advance_sources, advance_continuity, advance_force_balance,
                        advance_energy, advance_electron_energy, advance_neutral_external_source,
                        advance_neutral_sources, advance_neutral_continuity,
                        advance_neutral_force_balance, advance_neutral_energy,
                        manufactured_solns_test, r_diffusion, vpa_diffusion, vperp_diffusion, vz_diffusion)
end

function setup_dummy_and_buffer_arrays(nr,nz,nvpa,nvperp,nvz,nvr,nvzeta,nspecies_ion,nspecies_neutral)

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
        buffer_vzvrvzetazsn_1,buffer_vzvrvzetazsn_2,buffer_vzvrvzetazsn_3,buffer_vzvrvzetazsn_4,buffer_vzvrvzetazsn_5,buffer_vzvrvzetazsn_6,
        buffer_vzvrvzetarsn_1,buffer_vzvrvzetarsn_2,buffer_vzvrvzetarsn_3,buffer_vzvrvzetarsn_4,buffer_vzvrvzetarsn_5,buffer_vzvrvzetarsn_6,
        buffer_vzvrvzetazrsn_1, buffer_vzvrvzetazrsn_2,
        buffer_vpavperp_1,buffer_vpavperp_2,buffer_vpavperp_3,
        buffer_vpavperpzr_1, buffer_vpavperpzr_2,buffer_vpavperpzr_3,buffer_vpavperpzr_4,buffer_vpavperpzr_5,buffer_vpavperpzr_6,
        buffer_vpavperpr_1, buffer_vpavperpr_2, buffer_vpavperpr_3, buffer_vpavperpr_4, buffer_vpavperpr_5, buffer_vpavperpr_6)

end

"""
if evolving the density via continuity equation, redefine the normalised f → f/n
if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
'scratch' should be a (nz,nspecies) array
"""
function normalize_pdf!(pdf, moments, scratch)
    error("Function normalise_pdf() has not been updated to be parallelized. Does not "
          * "seem to be used at the moment.")
    if moments.evolve_ppar
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
function setup_scratch_arrays(moments, pdf, n_rk_stages)
    # will create n_rk_stages+1 structs, each of which will contain one pdf,
    # density, parallel flow, parallel pressure, and perpendicular pressure array for ions
    # (possibly) the same for electrons, and the same for neutrals. The actual array will
    # be created at the end of the first step of the loop below, once we have a
    # `scratch_pdf` object of the correct type.
    scratch = Vector{scratch_pdf{5,3,4,2,6,3}}(undef, n_rk_stages+1)
    pdf_dims = size(pdf.ion.norm)
    moment_dims = size(moments.ion.dens)
    if pdf.electron !== nothing
        using_electrons = true
        pdf_electron_dims = size(pdf.electron.norm)
    else
        using_electrons = false
        pdf_electron_dims = (0,0,0,0)
    end
    moment_electron_dims = size(moments.electron.dens)

    pdf_neutral_dims = size(pdf.neutral.norm)
    moment_neutral_dims = size(moments.neutral.dens)
    # populate each of the structs
    for istage ∈ 1:n_rk_stages+1
        # Allocate arrays in temporary variables so that we can identify them
        # by source line when using @debug_shared_array
        pdf_array = allocate_shared_float(pdf_dims...)
        density_array = allocate_shared_float(moment_dims...)
        upar_array = allocate_shared_float(moment_dims...)
        ppar_array = allocate_shared_float(moment_dims...)
        pperp_array = allocate_shared_float(moment_dims...)
        temp_array = allocate_shared_float(moment_dims...)

        pdf_electron_array = allocate_shared_float(pdf_electron_dims...)
        density_electron_array = allocate_shared_float(moment_electron_dims...)
        upar_electron_array = allocate_shared_float(moment_electron_dims...)
        ppar_electron_array = allocate_shared_float(moment_electron_dims...)
        pperp_electron_array = allocate_shared_float(moment_electron_dims...)
        temp_electron_array = allocate_shared_float(moment_electron_dims...)

        pdf_neutral_array = allocate_shared_float(pdf_neutral_dims...)
        density_neutral_array = allocate_shared_float(moment_neutral_dims...)
        uz_neutral_array = allocate_shared_float(moment_neutral_dims...)
        pz_neutral_array = allocate_shared_float(moment_neutral_dims...)


        scratch[istage] = scratch_pdf(pdf_array, density_array, upar_array,
                                      ppar_array, pperp_array, temp_array,
                                      pdf_electron_array, density_electron_array,
                                      upar_electron_array, ppar_electron_array,
                                      pperp_electron_array, temp_electron_array,
                                      pdf_neutral_array, density_neutral_array,
                                      uz_neutral_array, pz_neutral_array)
        @serial_region begin
            scratch[istage].pdf .= pdf.ion.norm
            scratch[istage].density .= moments.ion.dens
            scratch[istage].upar .= moments.ion.upar
            scratch[istage].ppar .= moments.ion.ppar
            scratch[istage].pperp .= moments.ion.pperp

            if using_electrons
                scratch[istage].pdf_electron .= pdf.electron.norm
            end
            scratch[istage].electron_density .= moments.electron.dens
            scratch[istage].electron_upar .= moments.electron.upar
            scratch[istage].electron_ppar .= moments.electron.ppar
            scratch[istage].electron_pperp .= 0.0 #moments.electron.pperp

            scratch[istage].pdf_neutral .= pdf.neutral.norm
            scratch[istage].density_neutral .= moments.neutral.dens
            scratch[istage].uz_neutral .= moments.neutral.uz
            scratch[istage].pz_neutral .= moments.neutral.pz
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
function time_advance!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions,
           external_source_settings, num_diss_params, advance, fp_arrays, scratch_dummy,
           manufactured_source_list, ascii_io, io_moments, io_dfns)

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
    moments_output_counter = 1
    dfns_output_counter = 1
    @serial_region begin
        t_params.next_output_time[] =
            min(t_params.moments_output_times[moments_output_counter],
                t_params.dfns_output_times[dfns_output_counter])
    end
    _block_synchronize()

    # main time advance loop
    iwrite_moments = 2
    iwrite_dfns = 2
    finish_now = false
    t_params.step_counter[] = 1
    while true
        if t_params.split_operators
            # MRH NOT SUPPORTED
            time_advance_split_operators!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, t_params.step_counter[])
        else
            time_advance_no_splitting!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
                moments, fields, spectral_objects, advect_objects,
                composition, collisions, geometry, boundary_distributions,
                external_source_settings, num_diss_params, advance, fp_arrays,  scratch_dummy,
                manufactured_source_list, t_params.step_counter[])
        end
        # update the time
        t += t_params.previous_dt[]

        if t ≥ t_params.end_time - epsilon || t_params.dt[] < 0.0
            # Ensure all output is written at the final step
            # Negative t_params.dt[] indicates the time stepping has failed, so stop and
            # write output.
            finish_now = true
        end

        if isfile(t_params.stopfile * "now")
            # Stop cleanly if a file called 'stop' was created
            println("Found 'stopnow' file $(t_params.stopfile * "now"), aborting run")
            finish_now = true
        end

        if t ≥ t_params.moments_output_times[moments_output_counter] - epsilon
            moments_output_counter += 1
            if moments_output_counter ≤ length(t_params.moments_output_times)
                @serial_region begin
                    t_params.next_output_time[] =
                        min(t_params.moments_output_times[moments_output_counter],
                            t_params.dfns_output_times[dfns_output_counter])
                end
            end
            write_moments = true
        else
            write_moments = false
        end
        if t ≥ t_params.dfns_output_times[dfns_output_counter] - epsilon
            dfns_output_counter += 1
            if dfns_output_counter ≤ length(t_params.dfns_output_times)
                @serial_region begin
                    t_params.next_output_time[] =
                        min(t_params.moments_output_times[moments_output_counter],
                            t_params.dfns_output_times[dfns_output_counter])
                end
            end
            write_dfns = true
        else
            write_dfns = false
        end

        if write_moments || write_dfns || finish_now
            # update the diagnostic chodura condition
            update_chodura!(moments,scratch[t_params.n_rk_stages+1].pdf,vpa,vperp,z,r,spectral_objects.r_spectral,composition,geometry,scratch_dummy,advect_objects.z_advect)

            # Always synchronise here, regardless of if we changed region or not
            begin_serial_region(no_synchronize=true)
            _block_synchronize()

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

            time_for_run = to_minutes(now() - start_time)
        end
        # write moments data to file
        if write_moments || finish_now
            @debug_detect_redundant_block_synchronize begin
                # Skip check for redundant _block_synchronize() during file I/O because
                # it only runs infrequently
                debug_detect_redundant_is_active[] = false
            end
            begin_serial_region()
            @serial_region begin
                if global_rank[] == 0
                    print("writing moments output ",
                          rpad(string(moments_output_counter - 1), 4), "  ",
                          "t = ", rpad(string(round(t, sigdigits=6)), 7), "  ",
                          "nstep = ", rpad(string(t_params.step_counter[]), 7), "  ")
                    if t_params.adaptive
                        print("nfail = ", rpad(string(t_params.failure_counter[]), 7), "  ",
                              "dt = ", rpad(string(t_params.dt_before_output[]), 7), "  ")
                    end
                    print(Dates.format(now(), dateformat"H:MM:SS"))
                end
            end
            write_data_to_ascii(moments, fields, z, r, t, composition.n_ion_species,
                                composition.n_neutral_species, ascii_io)
            write_all_moments_data_to_binary(moments, fields, t,
                                             composition.n_ion_species,
                                             composition.n_neutral_species, io_moments,
                                             iwrite_moments, time_for_run, t_params, r, z)

            if t_params.steady_state_residual
                # Calculate some residuals to see how close simulation is to steady state
                begin_r_z_region()
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

            iwrite_moments += 1
            begin_s_r_z_vperp_region()
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
            begin_serial_region()
            @serial_region begin
                if global_rank[] == 0
                    println("writing distribution functions output ",
                            rpad(string(dfns_output_counter  - 1), 4), "  ",
                            "t = ", rpad(string(round(t, sigdigits=6)), 7), "  ",
                            "nstep = ", rpad(string(t_params.step_counter[]), 7), "  ",
                            Dates.format(now(), dateformat"H:MM:SS"))
                    flush(stdout)
                end
            end
            write_all_dfns_data_to_binary(pdf, moments, fields, t,
                                          composition.n_ion_species,
                                          composition.n_neutral_species, io_dfns,
                                          iwrite_dfns, time_for_run, t_params, r, z, vperp, vpa,
                                          vzeta, vr, vz)
            iwrite_dfns += 1
            begin_s_r_z_vperp_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end

        if finish_now
            break
        end
        if t_params.adaptive
            if t >= t_params.end_time - epsilon
                break
            end
        else
            if t_params.step_counter[] >= t_params.nstep
                break
            end
        end

        t_params.step_counter[] += 1
    end
    return nothing
end

"""
"""
function time_advance_split_operators!(pdf, scratch, t, t_params, vpa, z,
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
    composition, collisions, external_source_settings, num_diss_params, advance, istep)

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
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (ion and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.cx_collisions = false
            end
            if collisions.ionization > 0.0
                advance.ionization_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.ionization_collisions = false
            end
        end
        if collisions.krook_collision_frequency_prefactor  > 0.0
            advance.krook_collisions_ii = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, collisions, sources, num_diss_params,
                advance, istep)
            advance.krook_collisions_ii = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.source_terms = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.continuity = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.force_balance = false
        end
        # use the energy equation to update the parallel pressure
        if moments.evolve_ppar
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.energy = false
        end
    else
        # use the energy equation to update the parallel pressure
        if moments.evolve_ppar
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.energy = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.force_balance = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.continuity = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.source_terms = false
        end
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.ionization > 0.0
                advance.ionization = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.ionization = false
            end
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.cx_collisions = false
            end
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (ion and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for ion species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.vpa_advection = false
    end
    return nothing
end

"""
"""
function time_advance_no_splitting!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions,
           external_source_settings, num_diss_params, advance, fp_arrays, scratch_dummy,
           manufactured_source_list, istep)

    ssp_rk!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
        moments, fields, spectral_objects, advect_objects, composition, collisions,
        geometry, boundary_distributions, external_source_settings, num_diss_params,
        advance, fp_arrays, scratch_dummy, manufactured_source_list, istep)

    return nothing
end

"""
use information obtained from the Runge-Kutta stages to compute the updated pdf;
for the quantities (density, upar, ppar, vth, qpar and phi) that are derived
from the 'true', un-modified pdf, either: update them using info from Runge Kutta
stages, if the quantities are evolved separately from the modified pdf;
or update them by taking the appropriate velocity moment of the evolved pdf
"""
function rk_update!(scratch, pdf, moments, fields, boundary_distributions, vz, vr, vzeta,
                    vpa, vperp, z, r, spectral_objects, advect_objects, t, t_params,
                    istage, composition, collisions, geometry, external_source_settings,
                    num_diss_params, advance, scratch_dummy, istep)
    begin_s_r_z_region()

    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]
    rk_coefs = t_params.rk_coefs[:,istage]

    z_spectral, r_spectral, vpa_spectral, vperp_spectral = spectral_objects.z_spectral, spectral_objects.r_spectral, spectral_objects.vpa_spectral, spectral_objects.vperp_spectral
    vzeta_spectral, vr_spectral, vz_spectral = spectral_objects.vzeta_spectral, spectral_objects.vr_spectral, spectral_objects.vz_spectral
    vpa_advect, vperp_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.vperp_advect, advect_objects.r_advect, advect_objects.z_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    ##
    # update the ion distribution and moments
    ##
    # here we seem to have duplicate arrays for storing n, u||, p||, etc, but not for vth
    # 'scratch' is for the multiple stages of time advanced quantities, but 'moments' can be updated directly at each stage
    rk_update_variable!(scratch, :pdf, t_params, istage)
    # use Runge Kutta to update any velocity moments evolved separately from the pdf
    rk_update_evolved_moments!(scratch, moments, t_params, istage)

    # Ensure there are no negative values in the pdf before applying boundary
    # conditions, so that negative deviations do not mess up the integral-constraint
    # corrections in the sheath boundary conditions.
    force_minimum_pdf_value!(new_scratch.pdf, num_diss_params.ion.force_minimum_pdf_value)

    # Enforce boundary conditions in z and vpa on the distribution function.
    # Must be done after Runge Kutta update so that the boundary condition applied to
    # the updated pdf is consistent with the updated moments - otherwise different upar
    # between 'pdf', 'old_scratch' and 'new_scratch' might mean a point that should be
    # set to zero at the sheath boundary according to the final upar has a non-zero
    # contribution from one or more of the terms.
    # NB: probably need to do the same for the evolved moments
    enforce_boundary_conditions!(new_scratch, moments,
        boundary_distributions.pdf_rboundary_ion, vpa.bc, z.bc, r.bc, vpa, vperp, z,
        r, vpa_spectral, vperp_spectral, 
        vpa_advect, vperp_advect, z_advect, r_advect, composition, scratch_dummy,
        advance.r_diffusion, advance.vpa_diffusion, advance.vperp_diffusion)

    if moments.evolve_density && moments.enforce_conservation
        begin_s_r_z_region()
        A = moments.ion.constraints_A_coefficient
        B = moments.ion.constraints_B_coefficient
        C = moments.ion.constraints_C_coefficient
        @loop_s_r_z is ir iz begin
            (A[iz,ir,is], B[iz,ir,is], C[iz,ir,is]) =
                @views hard_force_moment_constraints!(new_scratch.pdf[:,:,iz,ir,is],
                                                     moments, vpa)
        end
    end

    function update_derived_ion_moments_and_derivatives()
        # update remaining velocity moments that are calculable from the evolved pdf
        # Note these may be needed for the boundary condition on the neutrals, so must be
        # calculated before that is applied. Also may be needed to calculate advection speeds
        # for for CFL stability limit calculations in adaptive_timestep_update!().
        update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition)
        # update the diagnostic chodura condition
        # update_chodura!(moments,new_scratch.pdf,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
        # update the thermal speed
        begin_s_r_z_region()
        try #below block causes DomainError if ppar < 0 or density, so exit cleanly if possible
            update_vth!(moments.ion.vth, new_scratch.ppar, new_scratch.pperp, new_scratch.density, vperp, z, r, composition)
        catch e
            if global_size[] > 1
                println("ERROR: error calculating vth in time_advance.jl")
                println(e)
                display(stacktrace(catch_backtrace()))
                flush(stdout)
                flush(stderr)
                MPI.Abort(comm_world, 1)
            end
            rethrow(e)
        end
        # update the parallel heat flux
        update_qpar!(moments.ion.qpar, moments.ion.qpar_updated, new_scratch.density,
                     new_scratch.upar, moments.ion.vth, new_scratch.pdf, vpa, vperp, z, r,
                     composition, moments.evolve_density, moments.evolve_upar,
                     moments.evolve_ppar)

        calculate_ion_moment_derivatives!(moments, new_scratch, scratch_dummy, z, z_spectral,
                                          num_diss_params.ion.moment_dissipation_coefficient)
    end
    update_derived_ion_moments_and_derivatives()

    function update_derived_electron_moments_and_derivatives()
        # update the lowest three electron moments (density, upar and ppar)
        calculate_electron_density!(new_scratch.electron_density, moments.electron.dens_updated, new_scratch.density)
        calculate_electron_upar_from_charge_conservation!(new_scratch.electron_upar, moments.electron.upar_updated,
            new_scratch.electron_density, new_scratch.upar, new_scratch.density, composition.electron_physics, r, z)
        # if electron model is braginskii_fluid, then ppar is evolved via the energy equation
        # and is already updated;
        # otherwise update assuming electron temperature is fixed in time
        if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons)
            @loop_r_z ir iz begin
                new_scratch.electron_ppar[iz,ir] = (rk_coefs[1]*moments.electron.ppar[iz,ir] 
                    + rk_coefs[2]*old_scratch.electron_ppar[iz,ir] + rk_coefs[3]*new_scratch.electron_ppar[iz,ir])
            end
        else
            @loop_r_z ir iz begin
                new_scratch.electron_ppar[iz,ir] = 0.5 * new_scratch.electron_density[iz,ir] *
                                                   moments.electron.vth[iz,ir]^2
            end
        end
        # regardless of electron model, electron ppar is now updated
        moments.electron.ppar_updated[] = true
        update_electron_vth_temperature!(moments, new_scratch.electron_ppar,
                                         new_scratch.electron_density)
        # calculate the corresponding zed derivatives of the moments
        calculate_electron_moment_derivatives!(moments, new_scratch, scratch_dummy, z, z_spectral,
                                               num_diss_params.electron.moment_dissipation_coefficient, 
                                               composition.electron_physics)
        # update the electron parallel heat flux
        calculate_electron_qpar!(moments.electron, new_scratch.pdf_electron,
            new_scratch.electron_ppar, new_scratch.electron_upar, new_scratch.upar,
            collisions.nu_ei, composition.me_over_mi, composition.electron_physics, vpa)
        if composition.electron_physics == kinetic_electrons
            max_electron_pdf_iterations = 100000

            # Copy ion and electron moments from `scratch` into `moments` to be used in
            # electron kinetic equation update
            begin_r_z_region()
            @loop_s_r_z is ir iz begin
                moments.ion.dens[iz,ir,is] = new_scratch.density[iz,ir,is]
                moments.ion.upar[iz,ir,is] = new_scratch.upar[iz,ir,is]
                moments.ion.ppar[iz,ir,is] = new_scratch.ppar[iz,ir,is]
            end
            @loop_sn_r_z isn ir iz begin
                moments.neutral.dens[iz,ir,isn] = new_scratch.density_neutral[iz,ir,isn]
                moments.neutral.uz[iz,ir,isn] = new_scratch.uz_neutral[iz,ir,isn]
                moments.neutral.pz[iz,ir,isn] = new_scratch.pz_neutral[iz,ir,isn]
            end
            @loop_r_z ir iz begin
                moments.electron.ppar[iz,ir] = new_scratch.electron_ppar[iz,ir]
            end

            update_electron_pdf!(scratch, pdf.electron.norm, moments, fields.phi, r, z, vperp,
                                 vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
                                 vpa_advect, scratch_dummy, t_params.electron, collisions,
                                 composition, external_source_settings, num_diss_params,
                                 max_electron_pdf_iterations)
        end
        # update the electron parallel friction force
        calculate_electron_parallel_friction_force!(moments.electron.parallel_friction, new_scratch.electron_density,
            new_scratch.electron_upar, new_scratch.upar, moments.electron.dT_dz, composition.me_over_mi,
            collisions.nu_ei, composition.electron_physics)
    end
    update_derived_electron_moments_and_derivatives()

    if composition.n_neutral_species > 0
        ##
        # update the neutral particle distribution and moments
        ##
        rk_update_variable!(scratch, :pdf_neutral, t_params, istage; neutrals=true)
        # use Runge Kutta to update any velocity moments evolved separately from the pdf
        rk_update_evolved_moments_neutral!(scratch, moments, t_params, istage)

        # Ensure there are no negative values in the pdf before applying boundary
        # conditions, so that negative deviations do not mess up the integral-constraint
        # corrections in the sheath boundary conditions.
        force_minimum_pdf_value_neutral!(new_scratch.pdf_neutral, num_diss_params.neutral.force_minimum_pdf_value)

        # Enforce boundary conditions in z and vpa on the distribution function.
        # Must be done after Runge Kutta update so that the boundary condition applied to
        # the updated pdf is consistent with the updated moments - otherwise different upar
        # between 'pdf', 'old_scratch' and 'new_scratch' might mean a point that should be
        # set to zero at the sheath boundary according to the final upar has a non-zero
        # contribution from one or more of the terms.
        # NB: probably need to do the same for the evolved moments
        # Note, so far vr and vzeta do not need advect objects, so pass `nothing` for
        # those as a placeholder
        enforce_neutral_boundary_conditions!(new_scratch.pdf_neutral, new_scratch.pdf,
            boundary_distributions, new_scratch.density_neutral, new_scratch.uz_neutral,
            new_scratch.pz_neutral, moments, new_scratch.density, new_scratch.upar,
            fields.Er, vzeta_spectral, vr_spectral, vz_spectral, neutral_r_advect,
            neutral_z_advect, nothing, nothing, neutral_vz_advect, r, z, vzeta, vr, vz,
            composition, geometry, scratch_dummy, advance.r_diffusion,
            advance.vz_diffusion)

        if moments.evolve_density && moments.enforce_conservation
            begin_sn_r_z_region()
            A = moments.neutral.constraints_A_coefficient
            B = moments.neutral.constraints_B_coefficient
            C = moments.neutral.constraints_C_coefficient
            @loop_sn_r_z isn ir iz begin
                (A[iz,ir,isn], B[iz,ir,isn], C[iz,ir,isn]) =
                    @views hard_force_moment_constraints_neutral!(
                        new_scratch.pdf_neutral[:,:,:,iz,ir,isn], moments, vz)
            end
        end

        function update_derived_neutral_moments_and_derivatives()
            # update remaining velocity moments that are calculable from the evolved pdf
            update_derived_moments_neutral!(new_scratch, moments, vz, vr, vzeta, z, r,
                                            composition)
            # update the thermal speed
            begin_sn_r_z_region()
            @loop_sn_r_z isn ir iz begin
                moments.neutral.vth[iz,ir,isn] = sqrt(2.0*new_scratch.pz_neutral[iz,ir,isn]/new_scratch.density_neutral[iz,ir,isn])
            end

            # update the parallel heat flux
            update_neutral_qz!(moments.neutral.qz, moments.neutral.qz_updated,
                               new_scratch.density_neutral, new_scratch.uz_neutral,
                               moments.neutral.vth, new_scratch.pdf_neutral, vz, vr, vzeta, z,
                               r, composition, moments.evolve_density, moments.evolve_upar,
                               moments.evolve_ppar)

            calculate_neutral_moment_derivatives!(moments, new_scratch, scratch_dummy, z, z_spectral, 
                                                  num_diss_params.neutral.moment_dissipation_coefficient)
        end
        update_derived_neutral_moments_and_derivatives()
    end

    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], z, r, composition, collisions, moments,
                z_spectral, r_spectral, scratch_dummy)
    # _block_synchronize() here because phi needs to be read on different ranks than
    # it was written on, even though the loop-type does not change here. However,
    # after the final RK stage can skip if:
    #  * evolving upar or ppar as synchronization will be triggered after moments
    #    updates at the beginning of the next RK step
    _block_synchronize()

    if t_params.adaptive && istage == t_params.n_rk_stages
        # Note the timestep update must be done before calculating derived moments and
        # moment derivatives, because the timstep might need to be re-done with a smaller
        # dt, in which case scratch[t_params.n_rk_stages+1] will be reset to the values
        # from the beginning of the timestep here.
        adaptive_timestep_update!(scratch, t, t_params, moments, fields, composition,
                                  collisions, geometry, external_source_settings,
                                  advect_objects, r, z, vperp, vpa, vzeta, vr, vz)
        # Re-do this in case adaptive_timestep_update re-arranged the `scratch` vector
        new_scratch = scratch[istage+1]
        old_scratch = scratch[istage]

        if t_params.previous_dt[] == 0.0
            # Re-update remaining velocity moments that are calculable from the evolved
            # pdf These need to be re-calculated because `new_scratch` was swapped with
            # the beginning of the timestep, because the timestep failed
            update_derived_ion_moments_and_derivatives()
            update_derived_electron_moments_and_derivatives()
            if composition.n_neutral_species > 0
                update_derived_neutral_moments_and_derivatives()
            end

            # update the electrostatic potential phi
            update_phi!(fields, scratch[istage+1], z, r, composition, collisions, moments,
                        z_spectral, r_spectral, scratch_dummy)
            if !(( moments.evolve_upar || moments.evolve_ppar) &&
                      istage == t_params.n_rk_stages)
                # _block_synchronize() here because phi needs to be read on different ranks than
                # it was written on, even though the loop-type does not change here. However,
                # after the final RK stage can skip if:
                #  * evolving upar or ppar as synchronization will be triggered after moments
                #    updates at the beginning of the next RK step
                _block_synchronize()
            end
        end
    end
end

"""
    adaptive_timestep_update!(scratch, t_params, rk_coefs, moments, n_neutral_species)

Check the error estimate for the embedded RK method and adjust the timestep if
appropriate.
"""
function adaptive_timestep_update!(scratch, t, t_params, moments, fields, composition,
                                   collisions, geometry, external_source_settings,
                                   advect_objects, r, z, vperp, vpa, vzeta, vr, vz)
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
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect
    evolve_density, evolve_upar, evolve_ppar = moments.evolve_density, moments.evolve_upar, moments.evolve_ppar

    CFL_limits = mk_float[]
    error_norm_type = typeof(t_params.error_sum_zero)
    error_norms = error_norm_type[]
    total_points = mk_int[]

    # Read the current dt here, so we only need one _block_synchronize() call for this and
    # the begin_s_r_z_vperp_vpa_region()
    current_dt = t_params.dt[]
    _block_synchronize()

    # Test CFL conditions for advection in kinetic equation to give stability limit for
    # timestep
    #
    # ion z-advection
    # No need to synchronize here, as we just called _block_synchronize()
    # Don't parallelise over species here, because get_minimum_CFL_*() does an MPI
    # reduction over the shared-memory block, so all processes must calculate the same
    # species at the same time.
    begin_r_vperp_vpa_region(; no_synchronize=true)
    ion_z_CFL = Inf
    @loop_s is begin
        update_speed_z!(z_advect[is], moments.ion.upar, moments.ion.vth,
                        evolve_upar, evolve_ppar, fields, vpa, vperp, z, r, t, geometry)
        this_minimum = get_minimum_CFL_z(z_advect[is].speed, z)
        @serial_region begin
            ion_z_CFL = min(ion_z_CFL, this_minimum)
        end
    end
    push!(CFL_limits, t_params.CFL_prefactor * ion_z_CFL)

    # ion vpa-advection
    begin_r_z_vperp_region()
    ion_vpa_CFL = Inf
    update_speed_vpa!(vpa_advect, fields, scratch[t_params.n_rk_stages+1], moments, vpa, vperp, z, r,
                      composition, collisions, external_source_settings.ion, t,
                      geometry)
    @loop_s is begin
        this_minimum = get_minimum_CFL_vpa(vpa_advect[is].speed, vpa)
        @serial_region begin
            ion_vpa_CFL = min(ion_vpa_CFL, this_minimum)
        end
    end
    push!(CFL_limits, t_params.CFL_prefactor * ion_vpa_CFL)

    # To avoid double counting points when we use distributed-memory MPI, skip the
    # inner/lower point in r and z if this process is not the first block in that
    # dimension.
    skip_r_inner = r.irank != 0
    skip_z_lower = z.irank != 0

    # Calculate error for ion distribution functions
    # Note rk_error_variable!() stores the calculated error in `scratch[2]`.
    rk_error_variable!(scratch, :pdf, t_params)
    ion_pdf_error = local_error_norm(scratch[2].pdf, scratch[t_params.n_rk_stages+1].pdf,
                                     t_params.rtol, t_params.atol;
                                     method=error_norm_method, skip_r_inner=skip_r_inner,
                                     skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
    push!(error_norms, ion_pdf_error)
    push!(total_points,
          vpa.n_global * vperp.n_global * z.n_global * r.n_global * n_ion_species)

    # Calculate error for ion moments, if necessary
    if moments.evolve_density
        begin_s_r_z_region()
        rk_error_variable!(scratch, :density, t_params)
        ion_n_err = local_error_norm(scratch[2].density,
                                     scratch[t_params.n_rk_stages+1].density,
                                     t_params.rtol, t_params.atol;
                                     method=error_norm_method, skip_r_inner=skip_r_inner,
                                     skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
        push!(error_norms, ion_n_err)
        push!(total_points, z.n_global * r.n_global * n_ion_species)
    end
    if moments.evolve_upar
        begin_s_r_z_region()
        rk_error_variable!(scratch, :upar, t_params)
        ion_u_err = local_error_norm(scratch[2].upar,
                                     scratch[t_params.n_rk_stages+1].upar, t_params.rtol,
                                     t_params.atol; method=error_norm_method,
                                     skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
        push!(error_norms, ion_u_err)
        push!(total_points, z.n_global * r.n_global * n_ion_species)
    end
    if moments.evolve_ppar
        begin_s_r_z_region()
        rk_error_variable!(scratch, :ppar, t_params)
        ion_p_err = local_error_norm(scratch[2].ppar,
                                     scratch[t_params.n_rk_stages+1].ppar, t_params.rtol,
                                     t_params.atol; method=error_norm_method,
                                     skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                     error_sum_zero=t_params.error_sum_zero)
        push!(error_norms, ion_p_err)
        push!(total_points, z.n_global * r.n_global * n_ion_species)
    end

    if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons)
        begin_r_z_region()
        rk_error_variable!(scratch, :electron_ppar, t_params)
        electron_p_err = local_error_norm(scratch[2].electron_ppar,
                                          scratch[t_params.n_rk_stages+1].electron_ppar,
                                          t_params.rtol, t_params.atol;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
        push!(error_norms, electron_p_err)
        push!(total_points, z.n_global * r.n_global)
    end

    if n_neutral_species > 0
        # neutral z-advection
        # Don't parallelise over species here, because get_minimum_CFL_*() does an MPI
        # reduction over the shared-memory block, so all processes must calculate the same
        # species at the same time.
        begin_r_vzeta_vr_vz_region()
        neutral_z_CFL = Inf
        @loop_sn isn begin
            update_speed_neutral_z!(neutral_z_advect[isn], moments.neutral.uz,
                                    moments.neutral.vth, evolve_upar, evolve_ppar, vz, vr,
                                    vzeta, z, r, t)
            this_minimum = get_minimum_CFL_neutral_z(neutral_z_advect[isn].speed, z)
            @serial_region begin
                neutral_z_CFL = min(neutral_z_CFL, this_minimum)
            end
        end
        push!(CFL_limits, t_params.CFL_prefactor * neutral_z_CFL)

        # neutral vz-advection
        begin_r_z_vzeta_vr_region()
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
        push!(CFL_limits, t_params.CFL_prefactor * neutral_vz_CFL)

        # Calculate error for neutral distribution functions
        rk_error_variable!(scratch, :pdf_neutral, t_params; neutrals=true)
        neut_pdf_error = local_error_norm(scratch[2].pdf_neutral,
                                          scratch[t_params.n_rk_stages+1].pdf_neutral,
                                          t_params.rtol, t_params.atol;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
        push!(error_norms, neut_pdf_error)
        push!(total_points,
              vz.n_global * vr.n_global * vzeta.n_global * z.n_global * r.n_global *
              n_neutral_species)

        # Calculate error for neutral moments, if necessary
        if moments.evolve_density
            begin_sn_r_z_region()
            rk_error_variable!(scratch, :density_neutral, t_params; neutrals=true)
            neut_n_err = local_error_norm(scratch[2].density_neutral,
                                          scratch[t_params.n_rk_stages+1].density_neutral,
                                          t_params.rtol, t_params.atol, true;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
            push!(error_norms, neut_n_err)
            push!(total_points, z.n_global * r.n_global * n_neutral_species)
        end
        if moments.evolve_upar
            begin_sn_r_z_region()
            rk_error_variable!(scratch, :uz_neutral, t_params; neutrals=true)
            neut_u_err = local_error_norm(scratch[2].uz_neutral,
                                          scratch[t_params.n_rk_stages+1].uz_neutral,
                                          t_params.rtol, t_params.atol, true;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
            push!(error_norms, neut_u_err)
            push!(total_points, z.n_global * r.n_global * n_neutral_species)
        end
        if moments.evolve_ppar
            begin_sn_r_z_region()
            rk_error_variable!(scratch, :pz_neutral, t_params; neutrals=true)
            neut_p_err = local_error_norm(scratch[2].pz_neutral,
                                          scratch[t_params.n_rk_stages+1].pz_neutral,
                                          t_params.rtol, t_params.atol, true;
                                          method=error_norm_method,
                                          skip_r_inner=skip_r_inner,
                                          skip_z_lower=skip_z_lower,
                                          error_sum_zero=t_params.error_sum_zero)
            push!(error_norms, neut_p_err)
            push!(total_points, z.n_global * r.n_global * n_neutral_species)
        end
    end

    adaptive_timestep_update_t_params!(t_params, scratch, t, CFL_limits, error_norms,
                                       total_points, current_dt, error_norm_method)

    return nothing
end

"""
update velocity moments that are calculable from the evolved ion pdf
"""
function update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition)
    if !moments.evolve_density
        update_density!(new_scratch.density, moments.ion.dens_updated,
                        new_scratch.pdf, vpa, vperp, z, r, composition)
    end
    if !moments.evolve_upar
        update_upar!(new_scratch.upar, moments.ion.upar_updated, new_scratch.density,
                     new_scratch.ppar, new_scratch.pdf, vpa, vperp, z, r, composition,
                     moments.evolve_density, moments.evolve_ppar)
    end
    if !moments.evolve_ppar
        # update_ppar! calculates (p_parallel/m_s N_e c_s^2) + (n_s/N_e)*(upar_s/c_s)^2 = (1/√π)∫d(vpa/c_s) (vpa/c_s)^2 * (√π f_s c_s / N_e)
        update_ppar!(new_scratch.ppar, moments.ion.ppar_updated, new_scratch.density,
                     new_scratch.upar, new_scratch.pdf, vpa, vperp, z, r, composition,
                     moments.evolve_density, moments.evolve_upar)
    end
    update_pperp!(new_scratch.pperp, new_scratch.pdf, vpa, vperp, z, r, composition)
end

"""
update velocity moments that are calculable from the evolved neutral pdf
"""
function update_derived_moments_neutral!(new_scratch, moments, vz, vr, vzeta, z, r,
                                         composition)
    if !moments.evolve_density
        update_neutral_density!(new_scratch.density_neutral, moments.neutral.dens_updated,
                                new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition)
    end
    if !moments.evolve_upar
        update_neutral_uz!(new_scratch.uz_neutral, moments.neutral.uz_updated,
                           new_scratch.density_neutral, new_scratch.pz_neutral,
                           new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_ppar)
    end
    if !moments.evolve_ppar
        update_neutral_pz!(new_scratch.pz_neutral, moments.neutral.pz_updated,
                           new_scratch.density_neutral, new_scratch.uz_neutral,
                           new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_upar)
    end
end

"""
"""
function ssp_rk!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects, composition, collisions,
           geometry, boundary_distributions, external_source_settings, num_diss_params,
           advance, fp_arrays, scratch_dummy, manufactured_source_list, istep)

    begin_s_r_z_region()

    n_rk_stages = t_params.n_rk_stages

    first_scratch = scratch[1]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        first_scratch.pdf[ivpa,ivperp,iz,ir,is] = pdf.ion.norm[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.ion.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.ion.upar[iz,ir,is]
        first_scratch.ppar[iz,ir,is] = moments.ion.ppar[iz,ir,is]
        first_scratch.pperp[iz,ir,is] = moments.ion.pperp[iz,ir,is]
    end

    begin_r_z_region()
    @loop_r_z ir iz begin
        first_scratch.electron_density[iz,ir] = moments.electron.dens[iz,ir]
        first_scratch.electron_upar[iz,ir] = moments.electron.upar[iz,ir]
        first_scratch.electron_ppar[iz,ir] = moments.electron.ppar[iz,ir]
        first_scratch.electron_temp[iz,ir] = moments.electron.temp[iz,ir]
    end

    if composition.n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            first_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            first_scratch.density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            first_scratch.uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            first_scratch.pz_neutral[iz,ir,isn] = moments.neutral.pz[iz,ir,isn]
            # other neutral moments here if required
        end
    end
    if moments.evolve_upar
        # moments may be read on all ranks, even though loop type is z_s, so need to
        # synchronize here
        _block_synchronize()
    end

    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage, composition, vpa, vperp, z, r)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments,
            advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t,
            t_params.dt[], spectral_objects, composition,
            collisions, geometry, scratch_dummy, manufactured_source_list,
            external_source_settings, num_diss_params, advance, fp_arrays, istage)
        @views rk_update!(scratch, pdf, moments, fields, boundary_distributions, vz, vr,
                          vzeta, vpa, vperp, z, r, spectral_objects, advect_objects,
                          t, t_params, istage, composition, collisions, geometry,
                          external_source_settings, num_diss_params, advance,
                          scratch_dummy, istep)
    end

    istage = n_rk_stages+1

    # update the pdf.norm and moments arrays as needed
    begin_s_r_z_region()
    final_scratch = scratch[istage]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.ion.norm[ivpa,ivperp,iz,ir,is] = final_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        moments.ion.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
        moments.ion.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
        moments.ion.ppar[iz,ir,is] = final_scratch.ppar[iz,ir,is]
        moments.ion.pperp[iz,ir,is] = final_scratch.pperp[iz,ir,is]
    end
    # No need to synchronize here as we only change electron quantities and previous
    # region only changed ion quantities.
    begin_r_z_region(no_synchronize=true)
    @loop_r_z ir iz begin
        moments.electron.dens[iz,ir] = final_scratch.electron_density[iz,ir]
        moments.electron.upar[iz,ir] = final_scratch.electron_upar[iz,ir]
        moments.electron.ppar[iz,ir] = final_scratch.electron_ppar[iz,ir]
        moments.electron.temp[iz,ir] = final_scratch.electron_temp[iz,ir]
    end
    if composition.n_neutral_species > 0
        # No need to synchronize here as we only change neutral quantities and previous
        # region only changed plasma quantities.
        begin_sn_r_z_region(no_synchronize=true)
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = final_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
            moments.neutral.uz[iz,ir,isn] = final_scratch.uz_neutral[iz,ir,isn]
            moments.neutral.pz[iz,ir,isn] = final_scratch.pz_neutral[iz,ir,isn]
        end
        # for now update moments.neutral object directly for diagnostic moments
        # that are not used in Runga-Kutta steps
        update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_pzeta!(moments.neutral.pzeta, moments.neutral.pzeta_updated, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        # Update ptot (isotropic pressure)
        if r.n > 1 #if 2D geometry
            @loop_sn_r_z isn ir iz begin
                moments.neutral.ptot[iz,ir,isn] = (moments.neutral.pz[iz,ir,isn] + moments.neutral.pr[iz,ir,isn] + moments.neutral.pzeta[iz,ir,isn])/3.0
            end
        else # 1D model
            @loop_sn_r_z isn ir iz begin
                moments.neutral.ptot[iz,ir,isn] = moments.neutral.pz[iz,ir,isn]
            end
        end
        # get particle fluxes (n.b. bad naming convention uz -> means -> n uz here)
        update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                           moments.neutral.dens, pdf.neutral.norm, vz, vr, vzeta, z, r,
                           composition)
        update_neutral_uzeta!(moments.neutral.uzeta, moments.neutral.uzeta_updated,
                              moments.neutral.dens, pdf.neutral.norm, vz, vr, vzeta, z,
                              r, composition)
        try #below loop can cause DomainError if ptot < 0 or density < 0, so exit cleanly if possible
            @loop_sn_r_z isn ir iz begin
                # update density using last density from Runga-Kutta stages
                moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
                # get vth for neutrals
                moments.neutral.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.ptot[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
            end
        catch e
            if global_size[] > 1
                println("ERROR: error at line 724 of time_advance.jl")
                println(e)
                display(stacktrace(catch_backtrace()))
                flush(stdout)
                flush(stderr)
                MPI.Abort(comm_world, 1)
            end
            rethrow(e)
        end
    end

    return nothing
end

"""
euler_time_advance! advances the vector equation dfvec/dt = G[f]
that includes the kinetic equation + any evolved moment equations
using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
with fvec_in an input and fvec_out the output
"""
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments,
    advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t, dt,
    spectral_objects, composition, collisions, geometry, scratch_dummy,
    manufactured_source_list, external_source_settings, num_diss_params, advance, fp_arrays, istage)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    # vpa_advection! advances the 1D advection equation in vpa.
    # only ion species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa

    vpa_spectral, vperp_spectral, r_spectral, z_spectral = spectral_objects.vpa_spectral, spectral_objects.vperp_spectral, spectral_objects.r_spectral, spectral_objects.z_spectral
    vz_spectral, vr_spectral, vzeta_spectral = spectral_objects.vz_spectral, spectral_objects.vr_spectral, spectral_objects.vzeta_spectral
    vpa_advect, vperp_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.vperp_advect, advect_objects.r_advect, advect_objects.z_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    if advance.external_source
        external_ion_source_controller!(fvec_in, moments, external_source_settings.ion,
                                        dt)
    end
    if advance.neutral_external_source
        external_neutral_source_controller!(fvec_in, moments,
                                            external_source_settings.neutral, r, z, dt)
    end

    if advance.vpa_advection
        vpa_advection!(fvec_out.pdf, fvec_in, fields, moments, vpa_advect, vpa, vperp, z, r, dt, t,
            vpa_spectral, composition, collisions, external_source_settings.ion, geometry)
    end

    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to ion species

    if advance.z_advection
        z_advection!(fvec_out.pdf, fvec_in, moments, fields, z_advect, z, vpa, vperp, r,
                     dt, t, z_spectral, composition, geometry, scratch_dummy)
    end

    # r advection relies on derivatives in z to get ExB
    if advance.r_advection && r.n > 1
        r_advection!(fvec_out.pdf, fvec_in, moments, fields, r_advect, r, z, vperp, vpa,
                     dt, r_spectral, composition, geometry, scratch_dummy)
    end
    # vperp_advection requires information about z and r advection
    # so call vperp_advection! only after z and r advection routines
    if advance.vperp_advection
        vperp_advection!(fvec_out.pdf, fvec_in, vperp_advect, r, z, vperp, vpa,
                      dt, vperp_spectral, composition, z_advect, r_advect, geometry)
    end

    if advance.source_terms
        source_terms!(fvec_out.pdf, fvec_in, moments, vpa, z, r, dt, z_spectral,
                      composition, collisions, external_source_settings.ion)
    end

    if advance.neutral_z_advection
        neutral_advection_z!(fvec_out.pdf_neutral, fvec_in, moments, neutral_z_advect,
            r, z, vzeta, vr, vz, dt, t, z_spectral, composition, scratch_dummy)
    end

    if advance.neutral_r_advection && r.n > 1
        neutral_advection_r!(fvec_out.pdf_neutral, fvec_in, neutral_r_advect,
            r, z, vzeta, vr, vz, dt, r_spectral, composition, geometry, scratch_dummy)
    end

    if advance.neutral_vz_advection
        neutral_advection_vz!(fvec_out.pdf_neutral, fvec_in, fields, moments,
                              neutral_vz_advect, vz, vr, vzeta, z, r, dt, vz_spectral,
                              composition, collisions, external_source_settings.neutral)
    end

    if advance.neutral_source_terms
        source_terms_neutral!(fvec_out.pdf_neutral, fvec_in, moments, vpa, z, r, dt, z_spectral,
                      composition, collisions, external_source_settings.neutral)
    end

    if advance.manufactured_solns_test
        source_terms_manufactured!(fvec_out.pdf, fvec_out.pdf_neutral, vz, vr, vzeta, vpa, vperp, z, r, t, dt, composition, manufactured_source_list)
    end

    if advance.cx_collisions || advance.ionization_collisions
        # gyroaverage neutral dfn and place it in the ion.buffer array for use in the collisions step
        vzvrvzeta_to_vpavperp!(pdf.ion.buffer, fvec_in.pdf_neutral, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, geometry, composition)
        # interpolate ion particle dfn and place it in the neutral.buffer array for use in the collisions step
        vpavperp_to_vzvrvzeta!(pdf.neutral.buffer, fvec_in.pdf, vz, vr, vzeta, vpa, vperp, z, r, geometry, composition)
    end

    # account for charge exchange collisions between ions and neutrals
    if advance.cx_collisions_1V
        charge_exchange_collisions_1V!(fvec_out.pdf, fvec_out.pdf_neutral, fvec_in,
                                       moments, composition, vpa, vz,
                                       collisions.charge_exchange, vpa_spectral,
                                       vz_spectral, dt)
    elseif advance.cx_collisions
        charge_exchange_collisions_3V!(fvec_out.pdf, fvec_out.pdf_neutral, pdf.ion.buffer, pdf.neutral.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions.charge_exchange, dt)
    end
    # account for ionization collisions between ions and neutrals
    if advance.ionization_collisions_1V
        ionization_collisions_1V!(fvec_out.pdf, fvec_out.pdf_neutral, fvec_in, vz, vpa,
                                  vperp, z, r, vz_spectral, moments, composition,
                                  collisions, dt)
    elseif advance.ionization_collisions
        ionization_collisions_3V!(fvec_out.pdf, fvec_out.pdf_neutral, pdf.ion.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions, dt)
    end
    if advance.ionization_source
        constant_ionization_source!(fvec_out.pdf, fvec_in, vpa, vperp, z, r, moments,
                                    composition, collisions, dt)
    end

    # Add Krook collision operator for ions
    if advance.krook_collisions_ii
        krook_collisions!(fvec_out.pdf, fvec_in, moments, composition, collisions,
                          vperp, vpa, dt)
    end

    if advance.external_source
        external_ion_source!(fvec_out.pdf, fvec_in, moments, external_source_settings.ion,
                            vperp, vpa, dt)
    end
    if advance.neutral_external_source
        external_neutral_source!(fvec_out.pdf_neutral, fvec_in, moments,
                                external_source_settings.neutral, vzeta, vr, vz, dt)
    end

    # add numerical dissipation
    if advance.numerical_dissipation
        vpa_dissipation!(fvec_out.pdf, fvec_in.pdf, vpa, vpa_spectral, dt,
                         num_diss_params.ion.vpa_dissipation_coefficient)
        vperp_dissipation!(fvec_out.pdf, fvec_in.pdf, vperp, vperp_spectral, dt,
                         num_diss_params.ion.vperp_dissipation_coefficient)
        z_dissipation!(fvec_out.pdf, fvec_in.pdf, z, z_spectral, dt,
                       num_diss_params.ion.z_dissipation_coefficient, scratch_dummy)
        r_dissipation!(fvec_out.pdf, fvec_in.pdf, r, r_spectral, dt,
                       num_diss_params.ion.r_dissipation_coefficient, scratch_dummy)
        vz_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, vz,
                                vz_spectral, dt, num_diss_params.neutral.vz_dissipation_coefficient)
        z_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, z, z_spectral,
                               dt, num_diss_params.neutral.z_dissipation_coefficient, scratch_dummy)
        r_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, r, r_spectral,
                               dt, num_diss_params.neutral.r_dissipation_coefficient, scratch_dummy)
    end
    # advance with the Fokker-Planck self-collision operator
    if advance.explicit_weakform_fp_collisions
        update_entropy_diagnostic = (istage == 1)
        explicit_fokker_planck_collisions_weak_form!(fvec_out.pdf,fvec_in.pdf,moments.ion.dSdt,composition,collisions,dt,
                                             fp_arrays,r,z,vperp,vpa,vperp_spectral,vpa_spectral,scratch_dummy,
                                             diagnose_entropy_production = update_entropy_diagnostic)
    end
    
    # End of advance for distribution function

    # Start advancing moments
    if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
        # Only need to change region type if moment evolution equations will be used.
        # Exept when using wall boundary conditions, do not actually need to synchronize
        # here because above we only modify the distribution function and below we only
        # modify the moments, so there is no possibility of race conditions.
        begin_s_r_z_region(no_synchronize=true)
    end
    if advance.continuity
        continuity_equation!(fvec_out.density, fvec_in, moments, composition, dt,
                             z_spectral, collisions.ionization,
                             external_source_settings.ion, num_diss_params)
    end
    if advance.force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        force_balance!(fvec_out.upar, fvec_out.density, fvec_in, moments, fields,
                       collisions, dt, z_spectral, composition, geometry,
                       external_source_settings.ion, num_diss_params)
    end
    if advance.energy
        energy_equation!(fvec_out.ppar, fvec_in, moments, collisions, dt, z_spectral,
                         composition, external_source_settings.ion, num_diss_params)
    end
    if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
        # Only need to change region type if moment evolution equations will be used.
        # Exept when using wall boundary conditions, do not actually need to synchronize
        # here because above we only modify the distribution function and below we only
        # modify the moments, so there is no possibility of race conditions.
        begin_sn_r_z_region(no_synchronize=true)
    end
    if advance.neutral_continuity
        neutral_continuity_equation!(fvec_out.density_neutral, fvec_in, moments,
                                     composition, dt, z_spectral, collisions.ionization,
                                     external_source_settings.neutral, num_diss_params)
    end
    if advance.neutral_force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        neutral_force_balance!(fvec_out.uz_neutral, fvec_out.density_neutral, fvec_in,
                               moments, fields, collisions, dt, z_spectral, composition,
                               geometry, external_source_settings.neutral,
                               num_diss_params)
    end
    if advance.neutral_energy
        neutral_energy_equation!(fvec_out.pz_neutral, fvec_in, moments, collisions, dt,
                                 z_spectral, composition,
                                 external_source_settings.neutral, num_diss_params)
    end
    if advance.electron_energy
        electron_energy_equation!(fvec_out.electron_ppar, fvec_in.electron_ppar,
                                  fvec_in.density, fvec_in.electron_upar, fvec_in.upar,
                                  fvec_in.ppar, fvec_in.density_neutral,
                                  fvec_in.uz_neutral, fvec_in.pz_neutral,
                                  moments.electron, collisions, dt, composition,
                                  external_source_settings.electron, num_diss_params, z)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments)
    return nothing
end

"""
update the vector containing the pdf and any evolved moments of the pdf
for use in the Runge-Kutta time advance
"""
function update_solution_vector!(evolved, moments, istage, composition, vpa, vperp, z, r)
    new_evolved = evolved[istage+1]
    old_evolved = evolved[istage]
    begin_s_r_z_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_evolved.pdf[ivpa,ivperp,iz,ir,is] = old_evolved.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        new_evolved.density[iz,ir,is] = old_evolved.density[iz,ir,is]
        new_evolved.upar[iz,ir,is] = old_evolved.upar[iz,ir,is]
        new_evolved.ppar[iz,ir,is] = old_evolved.ppar[iz,ir,is]
    end
    begin_r_z_region()
    @loop_r_z ir iz begin
        new_evolved.electron_density[iz,ir] = old_evolved.electron_density[iz,ir]
        new_evolved.electron_upar[iz,ir] = old_evolved.electron_upar[iz,ir]
        new_evolved.electron_ppar[iz,ir] = old_evolved.electron_ppar[iz,ir]
        new_evolved.electron_temp[iz,ir] = old_evolved.electron_temp[iz,ir]
    end
    if composition.n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            new_evolved.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = old_evolved.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            new_evolved.density_neutral[iz,ir,isn] = old_evolved.density_neutral[iz,ir,isn]
            new_evolved.uz_neutral[iz,ir,isn] = old_evolved.uz_neutral[iz,ir,isn]
            new_evolved.pz_neutral[iz,ir,isn] = old_evolved.pz_neutral[iz,ir,isn]
        end
    end
    return nothing
end

end
