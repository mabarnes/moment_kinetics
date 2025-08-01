module electron_kinetic_equation

using LinearAlgebra
using MPI
using OrderedCollections
using SparseArrays

export get_electron_critical_velocities

using ..looping
using ..analysis: steady_state_residuals
using ..derivatives: derivative_z!, derivative_z_pdf_vpavperpz!
using ..boundary_conditions: enforce_v_boundary_condition_local!,
                             enforce_vperp_boundary_condition!,
                             skip_f_electron_bc_points_in_Jacobian, vpagrid_to_vpa
using ..calculus: derivative!, second_derivative!, integral,
                  reconcile_element_boundaries_MPI!,
                  reconcile_element_boundaries_MPI_z_pdf_vpavperpz!
using ..communication
using ..gauss_legendre: gausslegendre_info
using ..input_structs
using ..interpolation: interpolate_to_grid_1d!, fill_1d_interpolation_matrix!,
                       interpolate_symmetric!, fill_interpolate_symmetric_matrix!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float
using ..electron_fluid_equations: calculate_electron_moments!,
                                  calculate_electron_moments_no_r!,
                                  update_electron_vth_temperature!,
                                  calculate_electron_qpar_from_pdf!,
                                  calculate_electron_qpar_from_pdf_no_r!,
                                  calculate_electron_parallel_friction_force!
using ..electron_fluid_equations: electron_energy_equation!,
                                  electron_energy_equation_no_r!,
                                  add_electron_energy_equation_to_Jacobian!,
                                  add_electron_energy_equation_to_v_only_Jacobian!,
                                  add_electron_energy_equation_to_z_only_Jacobian!,
                                  electron_energy_residual!
using ..electron_z_advection: electron_z_advection!, update_electron_speed_z!,
                              add_electron_z_advection_to_Jacobian!,
                              add_electron_z_advection_to_v_only_Jacobian!,
                              add_electron_z_advection_to_z_only_Jacobian!
using ..electron_vpa_advection: electron_vpa_advection!, update_electron_speed_vpa!,
                                add_electron_vpa_advection_to_Jacobian!,
                                add_electron_vpa_advection_to_v_only_Jacobian!
using ..em_fields: update_phi!
using ..external_sources: total_external_electron_sources!,
                          add_total_external_electron_source_to_Jacobian!,
                          add_total_external_electron_source_to_v_only_Jacobian!,
                          add_total_external_electron_source_to_z_only_Jacobian!
using ..file_io: get_electron_io_info, write_electron_state, finish_electron_io,
                 write_debug_data_to_binary
using ..collision_frequencies: get_collision_frequency_ee,
                                 get_collision_frequency_ei
using ..krook_collisions: electron_krook_collisions!, add_electron_krook_collisions_to_Jacobian!,
                          add_electron_krook_collisions_to_v_only_Jacobian!,
                          add_electron_krook_collisions_to_z_only_Jacobian!
using ..timer_utils
using ..moment_constraints: hard_force_moment_constraints!,
                            moment_constraints_on_residual!,
                            electron_implicit_constraint_forcing!,
                            add_electron_implicit_constraint_forcing_to_Jacobian!,
                            add_electron_implicit_constraint_forcing_to_v_only_Jacobian!,
                            add_electron_implicit_constraint_forcing_to_z_only_Jacobian!
using ..moment_kinetics_structs: scratch_pdf, scratch_electron_pdf, electron_pdf_substruct
using ..nonlinear_solvers
using ..runge_kutta: rk_update_variable!, rk_loworder_solution!, local_error_norm,
                     adaptive_timestep_update_t_params!
using ..utils: get_minimum_CFL_z, get_minimum_CFL_vpa
using ..velocity_moments: calculate_electron_moment_derivatives!,
                          calculate_electron_moment_derivatives_no_r!,
                          update_derived_electron_moment_time_derivatives!

# Only needed so we can reference it in a docstring
import ..runge_kutta

const integral_correction_sharpness = 4.0

"""
update_electron_pdf is a function that uses the electron kinetic equation 
to solve for the updated electron pdf

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor

    INPUTS:
    scratch = `scratch_pdf` struct used to store Runge-Kutta stages
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    moments = struct containing electron moments
    r = struct containing r-coordinate information
    z = struct containing z-coordinate information
    vperp = struct containing vperp-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vperp_spectral = struct containing spectral information for the vperp-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    z_advect = struct containing information for z-advection
    vpa_advect = struct containing information for vpa-advection
    scratch_dummy = dummy arrays to be used for temporary storage
    t_params = parameters, etc. for time-stepping
    collisions = parameters for charged species collisions and neutral reactions
    composition = parameters describing number and type of species
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
    max_electron_sim_time = maximum amount of de-dimensionalised time to use in the solution of the electron kinetic equation
    io_electorn = if this is passed, write some output from the electron pseudo-timestepping
    initial_time = if this is passed set the initial pseudo-time to this value
    residual_tolerance = if this is passed, it overrides the value of `t_params.converged_residual_value`
    evolve_p = if set to true, update the electron pressure as part of the solve
    ion_dt = if this is passed, the electron pressure is evolved in a form that results in
             a backward-Euler update on the ion timestep (ion_dt) once the electron
             pseudo-timestepping reaches steady state.
    solution_method = if this is passed, choose a non-standard method for the solution
OUTPUT:
    pdf = updated (modified) electron pdf
"""
@timeit global_timer update_electron_pdf!(
                         scratch, pdf, moments, phi, r, z, vperp, vpa, z_spectral,
                         vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, t_params, collisions, composition,
                         external_source_settings, num_diss_params, nl_solver_params,
                         max_electron_pdf_iterations, max_electron_sim_time;
                         io_electron=nothing, initial_time=nothing,
                         residual_tolerance=nothing, evolve_p=false, ion_dt=nothing,
                         solution_method="backward_euler") = begin

    # set the method to use to solve the electron kinetic equation
    #solution_method = "artificial_time_derivative"
    #solution_method = "shooting_method"
    #solution_method = "picard_iteration"
    # solve the electron kinetic equation using the specified method
    if solution_method == "artificial_time_derivative"
        return update_electron_pdf_with_time_advance!(scratch, pdf, moments, phi,
            collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
            vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
            external_source_settings, num_diss_params, max_electron_pdf_iterations,
            max_electron_sim_time; io_electron=io_electron, initial_time=initial_time,
            residual_tolerance=residual_tolerance, evolve_p=evolve_p, ion_dt=ion_dt)
    elseif solution_method == "backward_euler"
        return electron_backward_euler_pseudotimestepping!(scratch, pdf, moments, phi,
            collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
            vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
            external_source_settings, num_diss_params, nl_solver_params,
            max_electron_pdf_iterations, max_electron_sim_time; io_electron=io_electron,
            initial_time=initial_time, residual_tolerance=residual_tolerance,
            evolve_p=evolve_p, ion_dt=ion_dt)
    elseif solution_method == "shooting_method"
        dens = moments.electron.dens
        vthe = moments.electron.vth
        p = moments.electron.p
        qpar = moments.electron.qpar
        qpar_updated = moments.electron.qpar_updated
        ddens_dz = moments.electron.ddens_dz
        dppar_dz = moments.electron.dppar_dz
        dqpar_dz = moments.electron.dqpar_dz
        dvth_dz = moments.electron.dvth_dz
        return update_electron_pdf_with_shooting_method!(pdf, dens, vthe, p, qpar,
            qpar_updated, phi, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, z, vpa,
            vpa_spectral, scratch_dummy, composition)
    elseif solution_method == "picard_iteration"
        dens = moments.electron.dens
        vthe = moments.electron.vth
        p = moments.electron.p
        qpar = moments.electron.qpar
        qpar_updated = moments.electron.qpar_updated
        ddens_dz = moments.electron.ddens_dz
        dppar_dz = moments.electron.dppar_dz
        dqpar_dz = moments.electron.dqpar_dz
        dvth_dz = moments.electron.dvth_dz
        return update_electron_pdf_with_picard_iteration!(pdf, dens, vthe, p, ddens_dz,
            dppar_dz, dqpar_dz, dvth_dz, z, vpa, vpa_spectral, scratch_dummy,
            max_electron_pdf_iterations)
    else
        error("!!! invalid solution method '$solution_method' specified !!!")
    end
    return nothing    
end

"""
update_electron_pdf_with_time_advance is a function that introduces an artifical time derivative to advance 
the electron kinetic equation until a steady-state solution is reached.

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor

    INPUTS:
    scratch = `scratch_pdf` struct used to store Runge-Kutta stages
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    moments = struct containing electron moments
    r = struct containing r-coordinate information
    z = struct containing z-coordinate information
    vperp = struct containing vperp-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vperp_spectral = struct containing spectral information for the vperp-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    z_advect = struct containing information for z-advection
    vpa_advect = struct containing information for vpa-advection
    scratch_dummy = dummy arrays to be used for temporary storage
    t_params = parameters, etc. for time-stepping
    collisions = parameters for charged species collisions and neutral reactions
    composition = parameters describing number and type of species
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
    max_electron_sim_time = maximum amount of de-dimensionalised time to use in the solution of the electron kinetic equation
    io_electorn = if this is passed, write some output from the electron pseudo-timestepping
    initial_time = if this is passed set the initial pseudo-time to this value
    residual_tolerance = if this is passed, it overrides the value of `t_params.converged_residual_value`
    evolve_p = if set to true, update the electron pressure as part of the solve
    ion_dt = if this is passed, the electron pressure is evolved in a form that results in
             a backward-Euler update on the ion timestep (ion_dt) once the electron
             pseudo-timestepping reaches steady state.
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_time_advance!(scratch, pdf, moments, phi, collisions,
        composition, r, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
        vpa_advect, scratch_dummy, t_params, external_source_settings, num_diss_params,
        max_electron_pdf_iterations, max_electron_sim_time; io_electron=nothing,
        initial_time=nothing, residual_tolerance=nothing, evolve_p=false,
        ion_dt=nothing)

    if max_electron_pdf_iterations === nothing && max_electron_sim_time === nothing
        error("Must set one of max_electron_pdf_iterations and max_electron_sim_time")
    end

    @begin_r_z_region()

    # create several (r) dimension dummy arrays for use in taking derivatives
    buffer_r_1 = @view scratch_dummy.buffer_rs_1[:,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[:,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[:,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[:,1]
    buffer_r_5 = @view scratch_dummy.buffer_rs_5[:,1]
    buffer_r_6 = @view scratch_dummy.buffer_rs_6[:,1]

    @begin_r_z_region()
    @loop_r_z ir iz begin
        # update the electron thermal speed using the updated electron pressure
        moments.electron.vth[iz,ir] = sqrt(abs(2.0 * moments.electron.p[iz,ir] /
                                                (moments.electron.dens[iz,ir] *
                                                 composition.me_over_mi)))
        scratch[t_params.n_rk_stages+1].electron_p[iz,ir] = moments.electron.p[iz,ir]
    end
    calculate_electron_moment_derivatives!(moments,
                                           (electron_density=moments.electron.dens,
                                            electron_upar=moments.electron.upar,
                                            electron_p=moments.electron.p),
                                           scratch_dummy, z, z_spectral,
                                           num_diss_params.electron.moment_dissipation_coefficient,
                                           composition.electron_physics)

    if ion_dt !== nothing
        evolve_p = true

        # Use forward-Euler step (with `ion_dt` as the timestep) as initial guess for
        # updated electron_p
        electron_energy_equation!(scratch[t_params.n_rk_stages+1].electron_p,
                                  moments.electron.dens, moments.electron.p,
                                  moments.electron.dens, moments.electron.upar,
                                  moments.electron.p, moments.ion.dens,
                                  moments.ion.upar, moments.ion.p, moments.neutral.dens,
                                  moments.neutral.uz, moments.neutral.p, moments.electron,
                                  collisions, ion_dt, composition,
                                  external_source_settings.electron, num_diss_params, r,
                                  z)
    end

    if !evolve_p
        # p is not updated in the pseudo-timestepping loop below. So that we can read
        # p from the scratch structs, copy moments.electron.p into all of them.
        moments_p = moments.electron.p
        for istage ∈ 1:t_params.n_rk_stages+1
            scratch_p = scratch[istage].electron_p
            @loop_r_z ir iz begin
                scratch_p[iz,ir] = moments_p[iz,ir]
            end
        end
    end

    if initial_time !== nothing
        t_params.t[] = initial_time
        # Make sure that output times are set relative to this initial_time (the values in
        # t_params are set relative to 0.0).
        moments_output_times = t_params.moments_output_times .+ initial_time
        dfns_output_times = t_params.dfns_output_times .+ initial_time
    else
        initial_time = t_params.t[]
    end
    if io_electron === nothing && t_params.debug_io !== nothing
        # Overwrite the debug output file with the output from this call to
        # update_electron_pdf_with_time_advance!().
        io_electron = get_electron_io_info(t_params.debug_io[1], "electron_debug")
        do_debug_io = true
        debug_io_nwrite = t_params.debug_io[3]
    else
        do_debug_io = false
    end

    text_output = false

    epsilon = 1.e-11
    # Store the initial number of iterations in the solution of the electron kinetic
    # equation
    initial_step_counter = t_params.step_counter[]
    t_params.step_counter[] += 1
    # initialise the electron pdf convergence flag to false
    electron_pdf_converged = false

    if text_output
        if n_blocks[] == 1
            text_output_suffix = ""
        else
            text_output_suffix = "$(iblock_index[])"
        end
        @begin_serial_region()
        @serial_region begin
            # open files to write the electron heat flux and pdf to file                                
            io_upar = open("upar$text_output_suffix.txt", "w")
            io_qpar = open("qpar$text_output_suffix.txt", "w")
            io_p = open("p$text_output_suffix.txt", "w")
            io_pdf = open("pdf$text_output_suffix.txt", "w")
            io_vth = open("vth$text_output_suffix.txt", "w")
            if !electron_pdf_converged
                # need to exit or handle this appropriately
                @loop_vpa ivpa begin
                    @loop_z iz begin
                        println(io_pdf, "z: ", z.grid[iz], " wpa: ", vpa.grid[ivpa], " pdf: ", scratch[t_params.n_rk_stages+1].pdf_electron[ivpa, 1, iz, 1], " time: ", t_params.t[], " residual: ", residual[ivpa, 1, iz, 1])
                    end
                    println(io_pdf,"")
                end
                @loop_z iz begin
                    println(io_upar, "z: ", z.grid[iz], " upar: ", moments.electron.upar[iz,1], " dupar_dz: ", moments.electron.dupar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                    println(io_qpar, "z: ", z.grid[iz], " qpar: ", moments.electron.qpar[iz,1], " dqpar_dz: ", moments.electron.dqpar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                    println(io_p, "z: ", z.grid[iz], " p: ", scratch[t_params.n_rk_stages+1].electron_p[iz,1], " dppar_dz: ", moments.electron.dppar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                    println(io_vth, "z: ", z.grid[iz], " vthe: ", moments.electron.vth[iz,1], " dvth_dz: ", moments.electron.dvth_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter, " dens: ", dens[iz,1])
                end
                println(io_upar,"")
                println(io_qpar,"")
                println(io_p,"")
                println(io_vth,"")
            end
            io_pdf_stages = open("pdf_zright$text_output_suffix.txt", "w")
        end
    end

    # Counter for pseudotime just in this pseudotimestepping loop
    local_pseudotime = 0.0
    residual_norm = -1.0

    @begin_serial_region()
    t_params.moments_output_counter[] += 1
    @serial_region begin
        if io_electron !== nothing
            write_electron_state(scratch, moments, t_params, io_electron,
                                 t_params.moments_output_counter[], local_pseudotime, 0.0,
                                 r, z, vperp, vpa)
        end
    end
    # evolve (artificially) in time until the residual is less than the tolerance
    while (!electron_pdf_converged
           && (max_electron_pdf_iterations === nothing || t_params.step_counter[] - initial_step_counter < max_electron_pdf_iterations)
           && (max_electron_sim_time === nothing || t_params.t[] - initial_time < max_electron_sim_time)
           && t_params.dt[] > 0.0 && !isnan(t_params.dt[]))

        # Set the initial values for the next step to the final values from the previous
        # step
        @begin_r_z_vperp_vpa_region()
        new_pdf = scratch[1].pdf_electron
        old_pdf = scratch[t_params.n_rk_stages+1].pdf_electron
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            new_pdf[ivpa,ivperp,iz,ir] = old_pdf[ivpa,ivperp,iz,ir]
        end
        if evolve_p
            @begin_r_z_region()
            new_p = scratch[1].electron_p
            old_p = scratch[t_params.n_rk_stages+1].electron_p
            @loop_r_z ir iz begin
                new_p[iz,ir] = old_p[iz,ir]
            end
        end

        for istage ∈ 1:t_params.n_rk_stages
            # Set the initial values for this stage to the final values from the previous
            # stage
            @begin_r_z_vperp_vpa_region()
            new_pdf = scratch[istage+1].pdf_electron
            old_pdf = scratch[istage].pdf_electron
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                new_pdf[ivpa,ivperp,iz,ir] = old_pdf[ivpa,ivperp,iz,ir]
            end
            if evolve_p
                @begin_r_z_region()
                new_p = scratch[istage+1].electron_p
                old_p = scratch[istage].electron_p
                @loop_r_z ir iz begin
                    new_p[iz,ir] = old_p[iz,ir]
                end
            end
            # Do a forward-Euler update of the electron pdf, and (if evove_p=true) the
            # electron pressure.
            # electron_kinetic_equation_euler_update!() is parallelised over {z,vperp,vpa}
            # (so that it can be used inside a loop over r in the implicit solve), so the
            # outer loop here should _not_ be parallelised (until we have defined an
            # 'anyzv' region to allow parallelising over r as well).
            for ir ∈ 1:r.n
                @views electron_kinetic_equation_euler_update!(
                           scratch[istage+1], scratch[istage].pdf_electron[:,:,:,ir],
                           scratch[istage].electron_p[:,ir], moments, z, vperp, vpa,
                           z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                           collisions, composition, external_source_settings,
                           num_diss_params, t_params, ir; evolve_p=evolve_p,
                           ion_dt=ion_dt)
            end

            rk_update_variable!(scratch, nothing, :pdf_electron, t_params, istage)

            if evolve_p
                rk_update_variable!(scratch, nothing, :electron_p, t_params, istage)

                update_electron_vth_temperature!(moments, scratch[istage+1].electron_p,
                                                 moments.electron.dens, composition)
            end

            apply_electron_bc_and_constraints!(scratch[istage+1], phi, moments, r, z,
                                               vperp, vpa, vperp_spectral, vpa_spectral,
                                               vpa_advect, num_diss_params, composition)

            latest_pdf = scratch[istage+1].pdf_electron
            
            function update_derived_moments_and_derivatives(update_vth=false)
                # update the electron heat flux
                moments.electron.qpar_updated[] = false
                calculate_electron_qpar_from_pdf!(moments.electron.qpar,
                                                  moments.electron.dens,
                                                  moments.electron.vth, latest_pdf, vperp,
                                                  vpa, composition.me_over_mi)

                if evolve_p
                    this_p = scratch[istage+1].electron_p
                    this_dens = moments.electron.dens
                    this_upar = moments.electron.upar
                    if update_vth
                        update_electron_vth_temperature!(moments, this_p, this_dens,
                                                         composition)
                    end
                    calculate_electron_moment_derivatives!(
                        moments,
                        (electron_density=this_dens,
                         electron_upar=this_upar,
                         electron_p=this_p),
                        scratch_dummy, z, z_spectral,
                        num_diss_params.electron.moment_dissipation_coefficient,
                        composition.electron_physics)
                else
                    # compute the z-derivative of the parallel electron heat flux
                    @views derivative_z!(moments.electron.dqpar_dz, moments.electron.qpar,
                                         buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                                         z_spectral, z)
                end
            end
            update_derived_moments_and_derivatives()

            if t_params.adaptive && istage == t_params.n_rk_stages
                if ion_dt === nothing
                    local_max_dt = Inf
                else
                    # Ensure timestep is not too big, so that d(electron_p)/dt 'source
                    # term' is numerically stable.
                    local_max_dt = 0.5 * ion_dt
                end
                electron_adaptive_timestep_update!(scratch, t_params.t[], t_params,
                                                   moments, phi, z_advect, vpa_advect,
                                                   composition, r, z, vperp, vpa,
                                                   vperp_spectral, vpa_spectral,
                                                   external_source_settings,
                                                   num_diss_params; evolve_p=evolve_p,
                                                   local_max_dt=local_max_dt)
                # Re-do this in case electron_adaptive_timestep_update!() re-arranged the
                # `scratch` vector
                new_scratch = scratch[istage+1]
                old_scratch = scratch[istage]

                if t_params.previous_dt[] == 0.0
                    # Re-calculate moments and moment derivatives as the timstep needs to
                    # be re-done with a smaller dt, so scratch[t_params.n_rk_stages+1] has
                    # been reset to the values from the beginning of the timestep here.
                    update_derived_moments_and_derivatives(true)
                end
            end
        end

        # update the time following the pdf update
        t_params.t[] += t_params.previous_dt[]
        local_pseudotime += t_params.previous_dt[]

        residual = -1.0
        if t_params.previous_dt[] > 0.0
            # Calculate residuals to decide if iteration is converged.
            # Might want an option to calculate the residual only after a certain number
            # of iterations (especially during initialization when there are likely to be
            # a large number of iterations required) to avoid the expense, and especially
            # the global MPI.Bcast()?
            @begin_r_z_vperp_vpa_region()
            residual = steady_state_residuals(scratch[t_params.n_rk_stages+1].pdf_electron,
                                              scratch[1].pdf_electron, t_params.previous_dt[];
                                              use_mpi=true, only_max_abs=true)
            if global_rank[] == 0
                residual = first(values(residual))[1]
            end
            if evolve_p
                p_residual =
                    steady_state_residuals(scratch[t_params.n_rk_stages+1].electron_p,
                                           scratch[1].electron_p, t_params.previous_dt[];
                                           use_mpi=true, only_max_abs=true)
                if global_rank[] == 0
                    p_residual = first(values(p_residual))[1]
                    residual = max(residual, p_residual)
                end
            end
            if global_rank[] == 0
                if residual_tolerance === nothing
                    residual_tolerance = t_params.converged_residual_value
                end
                electron_pdf_converged = abs(residual) < residual_tolerance
            end
            @timeit_debug global_timer "MPI.Bcast comm_world" electron_pdf_converged = MPI.Bcast(electron_pdf_converged, 0, comm_world)
        end

        if text_output
            if (mod(t_params.step_counter[] - initial_step_counter, t_params.nwrite_moments)==1)
                @begin_serial_region()
                @serial_region begin
                    @loop_vpa ivpa begin
                        println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", new_pdf[ivpa,1,end,1], " iteration: ", t_params.step_counter[] - initial_step_counter, " flag: ", 1)
                    end
                    println(io_pdf_stages,"")
                end
            end
        end

        if (mod(t_params.step_counter[] - initial_step_counter,100) == 0)
            @begin_serial_region()
            @serial_region begin
                if z.irank == 0 && z.irank == z.nrank - 1
                    println("iteration: ", t_params.step_counter[] - initial_step_counter, " time: ", t_params.t[], " dt_electron: ", t_params.dt[], " phi_boundary: ", phi[[1,end],1], " residual: ", residual)
                elseif z.irank == 0
                    println("iteration: ", t_params.step_counter[] - initial_step_counter, " time: ", t_params.t[], " dt_electron: ", t_params.dt[], " phi_boundary_lower: ", phi[1,1], " residual: ", residual)
                end
            end
        end
        if ((t_params.adaptive && t_params.write_moments_output[])
            || (!t_params.adaptive && t_params.step_counter[] % t_params.nwrite_moments == 0)
            || (do_debug_io && (t_params.step_counter[] % debug_io_nwrite == 0)))

            @begin_serial_region()
            @serial_region begin
                if text_output
                    if (mod(t_params.moments_output_counter[], 100) == 0)
                        @loop_vpa ivpa begin
                            @loop_z iz begin
                                println(io_pdf, "z: ", z.grid[iz], " wpa: ", vpa.grid[ivpa], " pdf: ", new_pdf[ivpa, 1, iz, 1], " time: ", t_params.t[], " residual: ", residual[ivpa, 1, iz, 1])
                            end
                            println(io_pdf,"")
                        end
                        println(io_pdf,"")
                    end
                    @loop_z iz begin
                        println(io_upar, "z: ", z.grid[iz], " upar: ", moments.electron.upar[iz,1], " dupar_dz: ", moments.electron.dupar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                        println(io_qpar, "z: ", z.grid[iz], " qpar: ", moments.electron.qpar[iz,1], " dqpar_dz: ", moments.electron.dqpar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                        println(io_p, "z: ", z.grid[iz], " p: ", scratch[t_params.n_rk_stages+1].electron_p[iz,1], " dppar_dz: ", moments.electron.dppar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                        println(io_vth, "z: ", z.grid[iz], " vthe: ", moments.electron.vth[iz,1], " dvth_dz: ", moments.electron.dvth_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter, " dens: ", dens[iz,1])
                    end
                    println(io_upar,"")
                    println(io_qpar,"")
                    println(io_p,"")
                    println(io_vth,"")
                end
            end
            t_params.moments_output_counter[] += 1
            t_params.write_moments_output[] = false
            @serial_region begin
                if io_electron !== nothing
                    write_electron_state(scratch, moments, t_params, io_electron,
                                         t_params.moments_output_counter[],
                                         local_pseudotime, residual_norm, r, z, vperp,
                                         vpa)
                end
            end
        end

        # check to see if the electron pdf satisfies the electron kinetic equation to within the specified tolerance

        t_params.step_counter[] += 1
        if electron_pdf_converged
            break
        end
    end
    # Update the 'pdf' arrays with the final result
    @begin_r_z_vperp_vpa_region()
    final_scratch_pdf = scratch[t_params.n_rk_stages+1].pdf_electron
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        pdf[ivpa,ivperp,iz,ir] = final_scratch_pdf[ivpa,ivperp,iz,ir]
    end
    if evolve_p
        # Update `moments.electron.p` with the final electron pressure
        @begin_r_z_region()
        scratch_p = scratch[t_params.n_rk_stages+1].electron_p
        moments_p = moments.electron.p
        @loop_r_z ir iz begin
            moments_p[iz,ir] = scratch_p[iz,ir]
        end
    end
    @begin_serial_region()
    @serial_region begin
        if text_output
            if !electron_pdf_converged
                @loop_vpa ivpa begin
                    @loop_z iz begin
                        println(io_pdf, "z: ", z.grid[iz], " wpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " time: ", t_params.t[], " residual: ", residual[ivpa, 1, iz, 1])
                    end
                    println(io_pdf,"")
                end
            end
            close(io_upar)
            close(io_qpar)
            close(io_p)
            close(io_vth)
            close(io_pdf)
            close(io_pdf_stages)
        end
        if !electron_pdf_converged || do_debug_io
            if io_electron !== nothing && io_electron !== true
                t_params.moments_output_counter[] += 1
                write_electron_state(scratch, moments, t_params, io_electron,
                                     t_params.moments_output_counter[], local_pseudotime,
                                     residual_norm, r, z, vperp, vpa)
                finish_electron_io(io_electron)
            end
        end
    end
    if !electron_pdf_converged
        success = "kinetic-electrons"
    else
        success = ""
    end
    return success
end

"""
Update the electron distribution function using backward-Euler for an artifical time
advance of the electron kinetic equation until a steady-state solution is reached.

Note that this function does not use the [`runge_kutta`](@ref) timestep functionality.
`t_params.previous_dt[]` is used to store the (adaptively updated) initial timestep of the
pseudotimestepping loop (initial value of `t_params.dt[]` within
`electron_backward_euler_pseudotimestepping!()`). `t_params.dt[]` is adapted according to
the iteration counts of the Newton solver.
"""
function electron_backward_euler_pseudotimestepping!(scratch, pdf, moments, phi,
        collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
        vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
        external_source_settings, num_diss_params, nl_solver_params,
        max_electron_pdf_iterations, max_electron_sim_time; io_electron=nothing,
        initial_time=nothing, residual_tolerance=nothing, evolve_p=false,
        ion_dt=nothing)

    if max_electron_pdf_iterations === nothing && max_electron_sim_time === nothing
        error("Must set one of max_electron_pdf_iterations and max_electron_sim_time")
    end

    t_params.dt[] = t_params.previous_dt[]

    @begin_r_z_region()
    @loop_r_z ir iz begin
        scratch[t_params.n_rk_stages+1].electron_p[iz,ir] = moments.electron.p[iz,ir]
    end
    calculate_electron_moments!((density=moments.ion.dens, upar=moments.ion.upar,
                                 electron_density=moments.electron.dens,
                                 electron_upar=moments.electron.upar,
                                 electron_p=moments.electron.p,
                                 pdf_electron=scratch[t_params.n_rk_stages+1].pdf_electron),
                                nothing, moments, composition, collisions, r, z, vperp,
                                vpa)
    calculate_electron_moment_derivatives!(moments,
                                           (electron_density=moments.electron.dens,
                                            electron_upar=moments.electron.upar,
                                            electron_p=moments.electron.p),
                                           scratch_dummy, z, z_spectral,
                                           num_diss_params.electron.moment_dissipation_coefficient,
                                           composition.electron_physics)

    reduced_by_ion_dt = false
    if ion_dt !== nothing
        if !evolve_p
            error("evolve_p must be `true` when `ion_dt` is passed. ion_dt=$ion_dt")
        end

        # Use forward-Euler step (with `ion_dt` as the timestep) as initial guess for
        # updated electron_p
        p_guess = scratch[t_params.n_rk_stages+1].electron_p
        electron_energy_equation!(p_guess, moments.electron.dens,
                                  moments.electron.p, moments.electron.dens,
                                  moments.electron.upar, moments.electron.ppar,
                                  moments.ion.dens, moments.ion.upar, moments.ion.p,
                                  moments.neutral.dens, moments.neutral.uz,
                                  moments.neutral.pz, moments.electron, collisions,
                                  ion_dt, composition, external_source_settings.electron,
                                  num_diss_params, r, z)

        calculate_electron_moments!((density=moments.ion.dens, upar=moments.ion.upar,
                                     electron_density=moments.electron.dens,
                                     electron_upar=moments.electron.upar,
                                     electron_p=p_guess,
                                     pdf_electron=scratch[t_params.n_rk_stages+1].pdf_electron),
                                    nothing, moments, composition, collisions, r, z,
                                    vperp, vpa)
        calculate_electron_moment_derivatives!(moments,
                                               (electron_density=moments.electron.dens,
                                                electron_upar=moments.electron.upar,
                                                electron_p=p_guess),
                                               scratch_dummy, z, z_spectral,
                                               num_diss_params.electron.moment_dissipation_coefficient,
                                               composition.electron_physics)
    end

    if !evolve_p
        # p is not updated in the pseudo-timestepping loop below. So that we can read
        # p from the scratch structs, copy moments.electron.p into all of them.
        moments_p = moments.electron.p
        for istage ∈ 1:t_params.n_rk_stages+1
            scratch_p = scratch[istage].electron_p
            @loop_r_z ir iz begin
                scratch_p[iz,ir] = moments_p[iz,ir]
            end
        end
    end

    if initial_time !== nothing
        t_params.t[] = initial_time
        # Make sure that output times are set relative to this initial_time (the values in
        # t_params are set relative to 0.0).
        moments_output_times = t_params.moments_output_times .+ initial_time
        dfns_output_times = t_params.dfns_output_times .+ initial_time
    else
        initial_time = t_params.t[]
    end
    if io_electron === nothing && t_params.debug_io !== nothing
        # Overwrite the debug output file with the output from this call to
        # update_electron_pdf_with_time_advance!().
        io_electron = get_electron_io_info(t_params.debug_io[1], "electron_debug")
        do_debug_io = true
        debug_io_nwrite = t_params.debug_io[3]
    else
        do_debug_io = false
    end

    # Store the initial number of iterations in the solution of the electron kinetic
    # equation
    initial_step_counter = t_params.step_counter[]
    t_params.step_counter[] += 1

    # Pseudotime just in this pseudotimestepping loop
    local_pseudotime = 0.0
    residual_norm = -1.0

    @begin_serial_region()
    t_params.moments_output_counter[] += 1
    @serial_region begin
        if io_electron !== nothing
            write_electron_state(scratch, moments, t_params, io_electron,
                                 t_params.moments_output_counter[], local_pseudotime, 0.0,
                                 r, z, vperp, vpa)
        end
    end
    electron_pdf_converged = Ref(false)
    # No paralleism in r for now - will need to add a specially adapted shared-memory
    # parallelism scheme to allow it for 2D1V or 2D2V simulations.
    for ir ∈ 1:r.n
        # create several 0D dummy arrays for use in taking derivatives
        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # initialise the electron pdf convergence flag to false
        electron_pdf_converged[] = false

        first_step = true
        # evolve (artificially) in time until the residual is less than the tolerance
        while (!electron_pdf_converged[]
               && ((max_electron_pdf_iterations !== nothing && t_params.step_counter[] - initial_step_counter < max_electron_pdf_iterations)
                   || (max_electron_sim_time !== nothing && t_params.t[] - initial_time < max_electron_sim_time))
               && t_params.dt[] > 0.0 && !isnan(t_params.dt[]))

            old_scratch = scratch[1]
            new_scratch = scratch[t_params.n_rk_stages+1]

            # Set the initial values for the next step to the final values from the previous
            # step. The initial guess for f_electron_new and electron_p_new are just the
            # values from the old timestep, so no need to change those.
            @begin_z_vperp_vpa_region()
            f_electron_old = @view old_scratch.pdf_electron[:,:,:,ir]
            f_electron_new = @view new_scratch.pdf_electron[:,:,:,ir]
            @loop_z_vperp_vpa iz ivperp ivpa begin
                f_electron_old[ivpa,ivperp,iz] = f_electron_new[ivpa,ivperp,iz]
            end
            electron_p_old = @view old_scratch.electron_p[:,ir]
            electron_p_new = @view new_scratch.electron_p[:,ir]
            if evolve_p
                @begin_z_region()
                @loop_z iz begin
                    electron_p_old[iz] = electron_p_new[iz]
                end
            end

            step_success = electron_backward_euler!(
                               old_scratch, new_scratch, moments, phi, collisions,
                               composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
                               vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                               t_params, external_source_settings, num_diss_params,
                               nl_solver_params, ir; evolve_p=evolve_p,
                               ion_dt=ion_dt)

            if step_success
                bc_constraints_converged = apply_electron_bc_and_constraints_no_r!(
                                               f_electron_new, phi, moments, r, z, vperp,
                                               vpa, vperp_spectral, vpa_spectral,
                                               vpa_advect, num_diss_params, composition,
                                               ir, nl_solver_params)
                if bc_constraints_converged != ""
                    step_success = false
                end
            end

            if step_success
                if !evolve_p
                    # update the electron heat flux
                    moments.electron.qpar_updated[] = false
                    @views calculate_electron_qpar_from_pdf_no_r!(moments.electron.qpar[:,ir],
                                                                  moments.electron.dens[:,ir],
                                                                  moments.electron.vth[:,ir],
                                                                  f_electron_new, vperp,
                                                                  vpa,
                                                                  composition.me_over_mi,
                                                                  ir)

                    # compute the z-derivative of the parallel electron heat flux
                    @views derivative_z!(moments.electron.dqpar_dz[:,ir],
                                         moments.electron.qpar[:,ir], buffer_1, buffer_2,
                                         buffer_3, buffer_4, z_spectral, z)
                end

                #println("Newton its ", nl_solver_params.max_nonlinear_iterations_this_step[], " ", t_params.dt[])
                # update the time following the pdf update
                t_params.t[] += t_params.dt[]
                local_pseudotime += t_params.dt[]

                if first_step && !reduced_by_ion_dt
                    # Adjust t_params.previous_dt[] which gives the initial timestep for
                    # the electron pseudotimestepping loop.
                    # If ion_dt<t_params.previous_dt[], assume that this is because we are
                    # taking a short ion step to an output time, so we do not want to mess
                    # up t_params.previous_dt[], which should be set sensibly for a
                    # 'normal' timestep.
                    if t_params.dt[] < t_params.previous_dt[]
                        # Had to decrease timestep on the first step to get convergence,
                        # so start next ion timestep with the decreased value.
                        global_rank[] == 0 && print("decreasing previous_dt due to failures ", t_params.previous_dt[])
                        t_params.previous_dt[] = t_params.dt[]
                        global_rank[] == 0 && println(" -> ", t_params.previous_dt[])
                    #elseif nl_solver_params.max_linear_iterations_this_step[] > max(0.4 * nl_solver_params.nonlinear_max_iterations, 5)
                    elseif nl_solver_params.max_linear_iterations_this_step[] > t_params.decrease_dt_iteration_threshold
                        # Step succeeded, but took a lot of iterations so decrease initial
                        # step size.
                        global_rank[] == 0 && print("decreasing previous_dt due to iteration count ", t_params.previous_dt[])
                        t_params.previous_dt[] /= t_params.max_increase_factor
                        global_rank[] == 0 && println(" -> ", t_params.previous_dt[])
                    #elseif nl_solver_params.max_linear_iterations_this_step[] < max(0.1 * nl_solver_params.nonlinear_max_iterations, 2)
                    elseif nl_solver_params.max_linear_iterations_this_step[] < t_params.increase_dt_iteration_threshold && (ion_dt === nothing || t_params.previous_dt[] < t_params.cap_factor_ion_dt * ion_dt)
                        # Only took a few iterations, so increase initial step size.
                        global_rank[] == 0 && print("increasing previous_dt due to iteration count ", t_params.previous_dt[])
                        if ion_dt === nothing
                            t_params.previous_dt[] *= t_params.max_increase_factor
                        else
                            t_params.previous_dt[] = min(t_params.previous_dt[] * t_params.max_increase_factor, t_params.cap_factor_ion_dt * ion_dt)
                        end
                        global_rank[] == 0 && println(" -> ", t_params.previous_dt[])
                    end
                end

                # Adjust the timestep depending on the iteration count.
                # Note nl_solver_params.max_linear_iterations_this_step[] gives the total
                # number of iterations, so is a better measure of the total work done by
                # the solver than the nonlinear iteration count, or the linear iterations
                # per nonlinear iteration
                #if nl_solver_params.max_linear_iterations_this_step[] > max(0.2 * nl_solver_params.nonlinear_max_iterations, 10)
                if nl_solver_params.max_linear_iterations_this_step[] > t_params.decrease_dt_iteration_threshold && t_params.dt[] > t_params.previous_dt[]
                    # Step succeeded, but took a lot of iterations so decrease step size.
                    t_params.dt[] /= t_params.max_increase_factor
                elseif nl_solver_params.max_linear_iterations_this_step[] < t_params.increase_dt_iteration_threshold && (ion_dt === nothing || t_params.dt[] < t_params.cap_factor_ion_dt * ion_dt)
                    # Only took a few iterations, so increase step size.
                    if ion_dt === nothing
                        t_params.dt[] *= t_params.max_increase_factor
                    else
                        t_params.dt[] = min(t_params.dt[] * t_params.max_increase_factor, t_params.cap_factor_ion_dt * ion_dt)
                    end
                    # Ensure dt does not exceed maximum_dt
                    t_params.dt[] = min(t_params.dt[], t_params.maximum_dt)
                end

                first_step = false
            else
                if !t_params.adaptive
                    # Timestep not allowed to change, so errors are fatal
                    error("Electron pseudotimestep failed. Electron pseudotimestep size "
                          * "is fixed, so cannot reduce timestep to re-try.")
                end

                t_params.dt[] *= 0.5

                # Force the preconditioner to be recalculated, because we have just
                # changed `dt` by a fairly large amount.
                nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval

                # Swap old_scratch and new_scratch so that the next step restarts from the
                # same state. Copy values over here rather than just swapping references
                # to arrays, because f_electron_old and electron_p_old are captured by
                # residual_func!() above, so any change in the things they refer to will
                # cause type instability in residual_func!().
                f_electron_new = @view new_scratch.pdf_electron[:,:,:,ir]
                f_electron_old = @view old_scratch.pdf_electron[:,:,:,ir]
                electron_p_new = @view new_scratch.electron_p[:,ir]
                electron_p_old = @view old_scratch.electron_p[:,ir]
                @begin_z_vperp_vpa_region()
                @loop_z_vperp_vpa iz ivperp ivpa begin
                    f_electron_new[ivpa,ivperp,iz] = f_electron_old[ivpa,ivperp,iz]
                end
                @begin_z_region()
                @loop_z iz begin
                    electron_p_new[iz] = electron_p_old[iz]
                end

                new_lowerz_vcut_ind = @view r.scratch_shared_int[ir:ir]
                new_upperz_vcut_ind = @view r.scratch_shared_int2[ir:ir]
                bc_constraints_converged = apply_electron_bc_and_constraints_no_r!(
                                               f_electron_new, phi, moments, r, z, vperp,
                                               vpa, vperp_spectral, vpa_spectral,
                                               vpa_advect, num_diss_params, composition,
                                               ir, nl_solver_params)
                if bc_constraints_converged != ""
                    error("apply_electron_bc_and_constraints_no_r!() failed, but this "
                          * "should not happen here, because we are re-applying the "
                          * "function to a previously successful result while resetting "
                          * "after a failed step.")
                end

                if !evolve_p
                    # update the electron heat flux
                    moments.electron.qpar_updated[] = false
                    @views calculate_electron_qpar_from_pdf_no_r!(moments.electron.qpar[:,ir],
                                                                  moments.electron.dens[:,ir],
                                                                  moments.electron.vth[:,ir],
                                                                  f_electron_new, vperp,
                                                                  vpa,
                                                                  composition.me_over_mi,
                                                                  ir)

                    # compute the z-derivative of the parallel electron heat flux
                    @views derivative_z!(moments.electron.dqpar_dz[:,ir],
                                         moments.electron.qpar[:,ir], buffer_1, buffer_2,
                                         buffer_3, buffer_4, z_spectral, z)
                end
            end

            reset_nonlinear_per_stage_counters!(nl_solver_params)

            residual_norm = -1.0
            if step_success
                # Calculate residuals to decide if iteration is converged.
                # Might want an option to calculate the r_normesidual only after a certain
                # number of iterations (especially during initialization when there are
                # likely to be a large number of iterations required) to avoid the
                # expense, and especially the global MPI.Bcast()?
                @begin_z_vperp_vpa_region()
                if global_rank[] == 0
                    residual_norm = steady_state_residuals(new_scratch.pdf_electron,
                                                           old_scratch.pdf_electron,
                                                           t_params.dt[], true, true)[1]
                else
                    steady_state_residuals(new_scratch.pdf_electron,
                                           old_scratch.pdf_electron, t_params.dt[], true,
                                           true)
                end
                if evolve_p
                    if global_rank[] == 0
                        p_residual =
                            steady_state_residuals(new_scratch.electron_p,
                                                   old_scratch.electron_p,
                                                   t_params.dt[], true, true)[1]
                        residual_norm = max(residual_norm, p_residual)
                    else
                        steady_state_residuals(new_scratch.electron_p,
                                               old_scratch.electron_p,
                                               t_params.dt[], true, true)
                    end
                end
                if global_rank[] == 0
                    if residual_tolerance === nothing
                        residual_tolerance = t_params.converged_residual_value
                    end
                    electron_pdf_converged[] = abs(residual_norm) < residual_tolerance
                end
                @timeit_debug global_timer "MPI.Bcast! comm_world" MPI.Bcast!(electron_pdf_converged, 0, comm_world)
            end

            if (mod(t_params.step_counter[] - initial_step_counter,100) == 0)
                @begin_serial_region()
                @serial_region begin
                    if z.irank == 0 && z.irank == z.nrank - 1
                        println("iteration: ", t_params.step_counter[] - initial_step_counter, " time: ", t_params.t[], " dt_electron: ", t_params.dt[], " phi_boundary: ", phi[[1,end],1], " residual_norm: ", residual_norm)
                    elseif z.irank == 0
                        println("iteration: ", t_params.step_counter[] - initial_step_counter, " time: ", t_params.t[], " dt_electron: ", t_params.dt[], " phi_boundary_lower: ", phi[1,1], " residual_norm: ", residual_norm)
                    end
                end
            end
            if ((t_params.step_counter[] % t_params.nwrite_moments == 0)
                || (do_debug_io && (t_params.step_counter[] % debug_io_nwrite == 0)))

                if r.n == 1
                    # For now can only do I/O within the pseudo-timestepping loop when there
                    # is no r-dimension, because different points in r would take different
                    # number and length of timesteps to converge.

                    # Update the electron heat flux for output. This will be done anyway
                    # before using qpar in the next residual_func!() call, but there is no
                    # convenient way to include that calculation in this debug output.
                    moments.electron.qpar_updated[] = false
                    @views calculate_electron_qpar_from_pdf_no_r!(moments.electron.qpar[:,ir],
                                                                  moments.electron.dens[:,ir],
                                                                  moments.electron.vth[:,ir],
                                                                  f_electron_new, vperp,
                                                                  vpa,
                                                                  composition.me_over_mi,
                                                                  ir)
                    @begin_serial_region()
                    t_params.moments_output_counter[] += 1
                    @serial_region begin
                        if io_electron !== nothing
                            t_params.write_moments_output[] = false
                            write_electron_state(scratch, moments, t_params, io_electron,
                                                 t_params.moments_output_counter[],
                                                 local_pseudotime, residual_norm, r, z,
                                                 vperp, vpa)
                        end
                    end
                end
            end

            t_params.step_counter[] += 1
            if electron_pdf_converged[]
                break
            end
        end
        if !electron_pdf_converged[]
            # If electron solve failed to converge for some `ir`, the failure will be
            # handled by restarting the ion timestep with a smaller dt, so no need to try
            # to solve for further `ir` values.
            break
        end
    end
    # Update the 'pdf' arrays with the final result
    @begin_r_z_vperp_vpa_region()
    final_scratch_pdf = scratch[t_params.n_rk_stages+1].pdf_electron
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        pdf[ivpa,ivperp,iz,ir] = final_scratch_pdf[ivpa,ivperp,iz,ir]
    end
    if evolve_p
        # Update `moments.electron.p` with the final electron pressure
        @begin_r_z_region()
        scratch_p = scratch[t_params.n_rk_stages+1].electron_p
        moments_p = moments.electron.p
        @loop_r_z ir iz begin
            moments_p[iz,ir] = scratch_p[iz,ir]
        end
    end
    @begin_serial_region()
    @serial_region begin
        if !electron_pdf_converged[] || do_debug_io
            if io_electron !== nothing && io_electron !== true
                t_params.moments_output_counter[] += 1
                write_electron_state(scratch, moments, t_params, io_electron,
                                     t_params.moments_output_counter[], local_pseudotime,
                                     residual_norm, r, z, vperp, vpa)
                finish_electron_io(io_electron)
            end
        end
    end

    if r.n > 1
        error("Limits on iteration count and simtime assume 1D simulations. "
              * "Need to fix handling of t_params.t[] and t_params.step_counter[], "
              * "and also t_params.max_step_count_this_ion_step[] and "
              * "t_params.max_t_increment_this_ion_step[]")
    else
        t_params.max_step_count_this_ion_step[] =
            max(t_params.step_counter[] - initial_step_counter,
                t_params.max_step_count_this_ion_step[])
        t_params.max_t_increment_this_ion_step[] =
            max(t_params.t[] - initial_time,
                t_params.max_t_increment_this_ion_step[])
    end

    initial_dt_scale_factor = 0.1
    if t_params.previous_dt[] < initial_dt_scale_factor * t_params.dt[]
        # If dt has increased a lot, we can probably try a larger initial dt for the next
        # solve.
        t_params.previous_dt[] = initial_dt_scale_factor * t_params.dt[]
    end

    if ion_dt !== nothing && t_params.dt[] != t_params.previous_dt[]
        # Reset dt in case it was reduced to be less than 0.5*ion_dt
        t_params.dt[] = t_params.previous_dt[]
    end
    if !electron_pdf_converged[]
        success = "kinetic-electrons"
    else
        success = ""
    end
    return success
end

"""
    electron_backward_euler!(old_scratch, new_scratch, moments, phi,
        collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
        vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
        external_source_settings, num_diss_params, nl_solver_params, ir;
        evolve_p=false, ion_dt=nothing) = begin

Take a single backward euler timestep for the electron shape function \$g_e\$ and parallel
pressure \$p_{e∥}\$.
"""
@timeit global_timer electron_backward_euler!(old_scratch, new_scratch, moments, phi,
            collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
            vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
            external_source_settings, num_diss_params, nl_solver_params, ir;
            evolve_p=false, ion_dt=nothing) = begin

    # create several 0D dummy arrays for use in taking derivatives
    buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

    f_electron_old = @view old_scratch.pdf_electron[:,:,:,ir]
    f_electron_new = @view new_scratch.pdf_electron[:,:,:,ir]
    electron_p_old = @view old_scratch.electron_p[:,ir]
    electron_p_new = @view new_scratch.electron_p[:,ir]

    if isa(old_scratch, scratch_electron_pdf)
        electron_density = @view moments.electron.dens[:,ir]
        electron_upar = @view moments.electron.upar[:,ir]
        ion_density = @view moments.ion.dens[:,ir,:]
        ion_upar = @view moments.ion.upar[:,ir,:]
    else
        electron_density = @view old_scratch.electron_density[:,ir]
        electron_upar = @view old_scratch.electron_upar[:,ir]
        ion_density = @view old_scratch.density[:,ir,:]
        ion_upar = @view old_scratch.upar[:,ir,:]
    end

    # Calculate derived moments and derivatives using updated f_electron
    @views calculate_electron_moments_no_r!(f_electron_new, electron_density,
                                            electron_upar, electron_p_new, ion_density,
                                            ion_upar, moments, composition, collisions, r,
                                            z, vperp, vpa, ir)
    @views calculate_electron_moment_derivatives_no_r!(
               moments, electron_density, electron_upar, electron_p_new, scratch_dummy, z,
               z_spectral, num_diss_params.electron.moment_dissipation_coefficient, ir)

    if nl_solver_params.preconditioner_type === Val(:electron_split_lu)
        if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
            nl_solver_params.solves_since_precon_update[] = 0

            vth = @view moments.electron.vth[:,ir]
            me = composition.me_over_mi
            p = electron_p_new
            ddens_dz = @view moments.electron.ddens_dz[:,ir]
            dupar_dz = @view moments.electron.dupar_dz[:,ir]
            dppar_dz = @view moments.electron.dppar_dz[:,ir]
            dvth_dz = @view moments.electron.dvth_dz[:,ir]
            dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
            source_amplitude = moments.electron.external_source_amplitude
            source_density_amplitude = moments.electron.external_source_density_amplitude
            source_momentum_amplitude = moments.electron.external_source_momentum_amplitude
            source_pressure_amplitude = moments.electron.external_source_pressure_amplitude

            # Note the region(s) used here must be the same as the region(s) used
            # when the matrices are used in `split_precon!()`, so that the
            # parallelisation is the same and each matrix is used on the same
            # process that created it.

            # z-advection preconditioner
            @begin_vperp_vpa_region()
            update_electron_speed_z!(z_advect[1], electron_upar, vth, vpa.grid, ir)
            @loop_vperp_vpa ivperp ivpa begin
                z_matrix, p_matrix = get_electron_split_Jacobians!(
                     ivperp, ivpa, p, moments, collisions, composition, z, vperp, vpa,
                     z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                     scratch_dummy, external_source_settings, num_diss_params, t_params,
                     ion_dt, ir, evolve_p)
                @timeit_debug global_timer "lu" nl_solver_params.preconditioners.z[ivpa,ivperp,ir] = lu(sparse(z_matrix))
                if ivperp == 1 && ivpa == 1
                    @timeit_debug global_timer "lu" nl_solver_params.preconditioners.p[ir] = lu(sparse(p_matrix))
                end
            end
        end

        function split_precon!(x)
            precon_p, precon_f = x

            @begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                z_precon_matrix = nl_solver_params.preconditioners.z[ivpa,ivperp,ir]
                f_slice = @view precon_f[ivpa,ivperp,:]
                @views z.scratch .= f_slice
                @timeit_debug global_timer "ldiv!" ldiv!(z.scratch2, z_precon_matrix, z.scratch)
                f_slice .= z.scratch2
            end

            @begin_z_region()
            p_precon_matrix = nl_solver_params.preconditioners.p[ir]
            @loop_z iz begin
                z.scratch[iz] = precon_p[iz]
            end

            @begin_serial_region()
            @serial_region begin
                @timeit_debug global_timer "ldiv!" ldiv!(precon_p, p_precon_matrix, z.scratch)
            end
        end

        left_preconditioner = identity
        right_preconditioner = split_precon!
    elseif nl_solver_params.preconditioner_type === Val(:electron_lu)

        if t_params.dt[] > 1.5 * nl_solver_params.precon_dt[] ||
                t_params.dt[] < 2.0/3.0 * nl_solver_params.precon_dt[]

            # dt has changed significantly, so update the preconditioner
            nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
        end

        if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
global_rank[] == 0 && println("recalculating precon")
            nl_solver_params.solves_since_precon_update[] = 0
            nl_solver_params.precon_dt[] = t_params.dt[]

            orig_lu, precon_matrix, input_buffer, output_buffer =
                nl_solver_params.preconditioners[ir]

            fill_electron_kinetic_equation_Jacobian!(
                precon_matrix, f_electron_new, electron_p_new, moments, phi,
                collisions, composition, z, vperp, vpa, z_spectral,
                vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params, ion_dt,
                ir, evolve_p)

            @begin_serial_region()
            if block_rank[] == 0
                if size(orig_lu) == (1, 1)
                    # Have not properly created the LU decomposition before, so
                    # cannot reuse it.
                    @timeit_debug global_timer "lu" nl_solver_params.preconditioners[ir] =
                        (lu(sparse(precon_matrix)), precon_matrix, input_buffer,
                         output_buffer)
                else
                    # LU decomposition was previously created. The Jacobian always
                    # has the same sparsity pattern, so by using `lu!()` we can
                    # reuse some setup.
                    try
                        @timeit_debug global_timer "lu!" lu!(orig_lu, sparse(precon_matrix); check=false)
                    catch e
                        if !isa(e, ArgumentError)
                            rethrow(e)
                        end
                        println("Sparsity pattern of matrix changed, rebuilding "
                                * " LU from scratch")
                        @timeit_debug global_timer "lu" orig_lu = lu(sparse(precon_matrix))
                    end
                    nl_solver_params.preconditioners[ir] =
                        (orig_lu, precon_matrix, input_buffer, output_buffer)
                end
            else
                nl_solver_params.preconditioners[ir] =
                    (orig_lu, precon_matrix, input_buffer, output_buffer)
            end
        end


        @timeit_debug global_timer lu_precon!(x) = begin
            precon_p, precon_f = x

            precon_lu, _, this_input_buffer, this_output_buffer =
                nl_solver_params.preconditioners[ir]

            @begin_serial_region()
            counter = 1
            @loop_z_vperp_vpa iz ivperp ivpa begin
                this_input_buffer[counter] = precon_f[ivpa,ivperp,iz]
                counter += 1
            end
            @loop_z iz begin
                this_input_buffer[counter] = precon_p[iz]
                counter += 1
            end

            @begin_serial_region()
            @serial_region begin
                @timeit_debug global_timer "ldiv!" ldiv!(this_output_buffer, precon_lu, this_input_buffer)
            end

            @begin_serial_region()
            counter = 1
            @loop_z_vperp_vpa iz ivperp ivpa begin
                precon_f[ivpa,ivperp,iz] = this_output_buffer[counter]
                counter += 1
            end
            @loop_z iz begin
                precon_p[iz] = this_output_buffer[counter]
                counter += 1
            end

            # Ensure values of precon_f and precon_p are consistent across
            # distributed-MPI block boundaries. For precon_f take the upwind
            # value, and for precon_p take the average.
            f_lower_endpoints = @view scratch_dummy.buffer_vpavperpr_1[:,:,ir]
            f_upper_endpoints = @view scratch_dummy.buffer_vpavperpr_2[:,:,ir]
            receive_buffer1 = @view scratch_dummy.buffer_vpavperpr_3[:,:,ir]
            receive_buffer2 = @view scratch_dummy.buffer_vpavperpr_4[:,:,ir]
            @begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                f_lower_endpoints[ivpa,ivperp] = precon_f[ivpa,ivperp,1]
                f_upper_endpoints[ivpa,ivperp] = precon_f[ivpa,ivperp,end]
            end
            # We upwind the z-derivatives in `electron_z_advection!()`, so would
            # expect that upwinding the results here in z would make sense.
            # However, upwinding here makes convergence much slower (~10x),
            # compared to picking the values from one side or other of the block
            # boundary, or taking the average of the values on either side.
            # Neither direction is special, so taking the average seems most
            # sensible (although in an intial test it does not seem to converge
            # faster than just picking one or the other).
            # Maybe this could indicate that it is more important to have a fully
            # self-consistent Jacobian inversion for the
            # `electron_vpa_advection()` part rather than taking half(ish) of the
            # values from one block and the other half(ish) from the other.
            reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
                precon_f, f_lower_endpoints, f_upper_endpoints, receive_buffer1,
                receive_buffer2, z)

            @begin_serial_region()
            @serial_region begin
                buffer_1[] = precon_p[1]
                buffer_2[] = precon_p[end]
            end
            reconcile_element_boundaries_MPI!(
                precon_p, buffer_1, buffer_2, buffer_3, buffer_4, z)

            return nothing
        end

        left_preconditioner = identity
        right_preconditioner = lu_precon!
    elseif nl_solver_params.preconditioner_type === Val(:electron_adi)

        if t_params.dt[] > 1.5 * nl_solver_params.precon_dt[] ||
                t_params.dt[] < 2.0/3.0 * nl_solver_params.precon_dt[]

            # dt has changed significantly, so update the preconditioner
            nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
        end

        if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
global_rank[] == 0 && println("recalculating precon")
            nl_solver_params.solves_since_precon_update[] = 0
            nl_solver_params.precon_dt[] = t_params.dt[]

            adi_info = nl_solver_params.preconditioners[ir]

            vth = @view moments.electron.vth[:,ir]
            qpar = @view moments.electron.qpar[:,ir]

            # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
            third_moment = scratch_dummy.buffer_z_1
            dthird_moment_dz = scratch_dummy.buffer_z_2
            @begin_z_region()
            @loop_z iz begin
                third_moment[iz] = 0.5 * qpar[iz] / electron_p_new[iz] / vth[iz]
            end
            derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                          buffer_3, buffer_4, z_spectral, z)

            z_speed = @view z_advect[1].speed[:,:,:,ir]

            dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
            @begin_vperp_vpa_region()
            update_electron_speed_z!(z_advect[1], electron_upar, vth, vpa.grid, ir)
            @loop_vperp_vpa ivperp ivpa begin
                @views z_advect[1].adv_fac[:,ivpa,ivperp,ir] = -z_speed[:,ivpa,ivperp]
            end
            #calculate the upwind derivative
            @views derivative_z_pdf_vpavperpz!(dpdf_dz, f_electron_new,
                                               z_advect[1].adv_fac[:,:,:,ir],
                                               scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                               scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                               scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                               scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                               scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                               scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                               z_spectral, z)

            dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
            @begin_z_vperp_region()
            update_electron_speed_vpa!(vpa_advect[1], electron_density, electron_upar,
                                       electron_p_new, moments, composition.me_over_mi,
                                       vpa.grid, external_source_settings.electron, ir)
            @loop_z_vperp iz ivperp begin
                @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
            end
            #calculate the upwind derivative of the electron pdf w.r.t. wpa
            @loop_z_vperp iz ivperp begin
                @views derivative!(dpdf_dvpa[:,ivperp,iz], f_electron_new[:,ivperp,iz], vpa,
                                   vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
            end

            zeroth_moment = z.scratch_shared
            first_moment = z.scratch_shared2
            second_moment = z.scratch_shared3
            @begin_z_region()
            vpa_grid = vpa.grid
            vpa_wgts = vpa.wgts
            @loop_z iz begin
                @views zeroth_moment[iz] = integral(f_electron_new[:,1,iz], vpa_wgts)
                @views first_moment[iz] = integral(f_electron_new[:,1,iz], vpa_grid, vpa_wgts)
                @views second_moment[iz] = integral(f_electron_new[:,1,iz], vpa_grid, 2, vpa_wgts)
            end

            v_size = vperp.n * vpa.n

            # Do setup for 'v solves'
            v_solve_counter = 0
            A = adi_info.v_solve_matrix_buffer
            explicit_J = adi_info.J_buffer
            # Get sparse matrix for explicit, right-hand-side part of the
            # solve.
            if adi_info.n_extra_iterations > 0
                # If we only do one 'iteration' we don't need the 'explicit
                # matrix' for the first solve (the v-solve), because the initial
                # guess is zero,
                fill_electron_kinetic_equation_Jacobian!(
                    explicit_J, f_electron_new, electron_p_new, moments, phi, collisions,
                    composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
                    z_advect, vpa_advect, scratch_dummy, external_source_settings,
                    num_diss_params, t_params, ion_dt, ir, evolve_p, :explicit_z, false)
            end
            @begin_z_region()
            @loop_z iz begin
                v_solve_counter += 1
                # Get LU-factorized matrix for implicit part of the solve
                @views fill_electron_kinetic_equation_v_only_Jacobian!(
                    A, f_electron_new[:,:,iz], electron_p_new[iz], dpdf_dz[:,:,iz],
                    dpdf_dvpa[:,:,iz], z_speed, moments, zeroth_moment[iz],
                    first_moment[iz], second_moment[iz], third_moment[iz],
                    dthird_moment_dz[iz], phi[iz], collisions, composition, z, vperp, vpa,
                    z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                    scratch_dummy, external_source_settings, num_diss_params, t_params,
                    ion_dt, ir, iz, evolve_p)
                A_sparse = sparse(A)
                if !isassigned(adi_info.v_solve_implicit_lus, v_solve_counter)
                    @timeit_debug global_timer "lu" adi_info.v_solve_implicit_lus[v_solve_counter] = lu(A_sparse)
                else
                    # LU decomposition was previously created. The Jacobian always
                    # has the same sparsity pattern, so by using `lu!()` we can
                    # reuse some setup.
                    try
                        @timeit_debug global_timer "lu!" lu!(adi_info.v_solve_implicit_lus[v_solve_counter], A_sparse; check=false)
                    catch e
                        if !isa(e, ArgumentError)
                            rethrow(e)
                        end
                        println("Sparsity pattern of matrix changed, rebuilding "
                                * " LU from scratch ir=$ir, iz=$iz")
                        @timeit_debug global_timer "lu" adi_info.v_solve_implicit_lus[v_solve_counter] = lu(A_sparse)
                    end
                end

                if adi_info.n_extra_iterations > 0
                    # If we only do one 'iteration' we don't need the 'explicit
                    # matrix' for the first solve (the v-solve), because the
                    # initial guess is zero,
                    adi_info.v_solve_explicit_matrices[v_solve_counter] = sparse(@view(explicit_J[adi_info.v_solve_global_inds[v_solve_counter],:]))
                end
            end
            @boundscheck v_solve_counter == adi_info.v_solve_nsolve || error("v_solve_counter($v_solve_counter) != v_solve_nsolve($(adi_info.v_solve_nsolve))")

            # Do setup for 'z solves'
            z_solve_counter = 0
            A = adi_info.z_solve_matrix_buffer
            explicit_J = adi_info.J_buffer
            # Get sparse matrix for explicit, right-hand-side part of the
            # solve.
            fill_electron_kinetic_equation_Jacobian!(
                explicit_J, f_electron_new, electron_p_new, moments, phi, collisions,
                composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
                z_advect, vpa_advect, scratch_dummy, external_source_settings,
                num_diss_params, t_params, ion_dt, ir, evolve_p, :explicit_v, false)
            @begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                z_solve_counter += 1

                # Get LU-factorized matrix for implicit part of the solve
                @views fill_electron_kinetic_equation_z_only_Jacobian_f!(
                    A, f_electron_new[ivpa,ivperp,:], electron_p_new,
                    dpdf_dz[ivpa,ivperp,:], dpdf_dvpa[ivpa,ivperp,:], z_speed, moments,
                    zeroth_moment, first_moment, second_moment, third_moment,
                    dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                    vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                    external_source_settings, num_diss_params, t_params, ion_dt, ir,
                    ivperp, ivpa, evolve_p)

                A_sparse = sparse(A)
                if !isassigned(adi_info.z_solve_implicit_lus, z_solve_counter)
                    @timeit_debug global_timer "lu" adi_info.z_solve_implicit_lus[z_solve_counter] = lu(A_sparse)
                else
                    # LU decomposition was previously created. The Jacobian always
                    # has the same sparsity pattern, so by using `lu!()` we can
                    # reuse some setup.
                    try
                        @timeit_debug global_timer "lu!" lu!(adi_info.z_solve_implicit_lus[z_solve_counter], A_sparse; check=false)
                    catch e
                        if !isa(e, ArgumentError)
                            rethrow(e)
                        end
                        println("Sparsity pattern of matrix changed, rebuilding "
                                * " LU from scratch ir=$ir, ivperp=$ivperp, ivpa=$ivpa")
                        @timeit_debug global_timer "lu" adi_info.z_solve_implicit_lus[z_solve_counter] = lu(A_sparse)
                    end
                end

                adi_info.z_solve_explicit_matrices[z_solve_counter] = sparse(@view(explicit_J[adi_info.z_solve_global_inds[z_solve_counter],:]))
            end
            @begin_serial_region(true)
            @serial_region begin
                # Do the solve for p on the rank-0 process, which has the fewest grid
                # points to handle if there are not an exactly equal number of points for
                # each process.
                z_solve_counter += 1

                # Get LU-factorized matrix for implicit part of the solve
                @views fill_electron_kinetic_equation_z_only_Jacobian_p!(
                    A, electron_p_new, moments, zeroth_moment, first_moment,
                    second_moment, third_moment, dthird_moment_dz, collisions,
                    composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
                    z_advect, vpa_advect, scratch_dummy, external_source_settings,
                    num_diss_params, t_params, ion_dt, ir, evolve_p)

                A_sparse = sparse(A)
                if !isassigned(adi_info.z_solve_implicit_lus, z_solve_counter)
                    @timeit_debug global_timer "lu" adi_info.z_solve_implicit_lus[z_solve_counter] = lu(A_sparse)
                else
                    # LU decomposition was previously created. The Jacobian always
                    # has the same sparsity pattern, so by using `lu!()` we can
                    # reuse some setup.
                    try
                        @timeit_debug global_timer "lu!" lu!(adi_info.z_solve_implicit_lus[z_solve_counter], A_sparse; check=false)
                    catch e
                        if !isa(e, ArgumentError)
                            rethrow(e)
                        end
                        println("Sparsity pattern of matrix changed, rebuilding "
                                * " LU from scratch ir=$ir, p z-solve")
                        @timeit_debug global_timer "lu" adi_info.z_solve_implicit_lus[z_solve_counter] = lu(A_sparse)
                    end
                end

                adi_info.z_solve_explicit_matrices[z_solve_counter] = sparse(@view(explicit_J[adi_info.z_solve_global_inds[z_solve_counter],:]))
            end
            @boundscheck z_solve_counter == adi_info.z_solve_nsolve || error("z_solve_counter($z_solve_counter) != z_solve_nsolve($(adi_info.z_solve_nsolve))")
        end

        @timeit_debug global_timer adi_precon!(x) = begin
            precon_p, precon_f = x

            adi_info = nl_solver_params.preconditioners[ir]
            precon_iterations = nl_solver_params.precon_iterations
            this_input_buffer = adi_info.input_buffer
            this_intermediate_buffer = adi_info.intermediate_buffer
            this_output_buffer = adi_info.output_buffer
            global_index_subrange = adi_info.global_index_subrange
            n_extra_iterations = adi_info.n_extra_iterations

            v_size = vperp.n * vpa.n
            pdf_size = z.n * v_size

            # Use these views to communicate block-boundary points
            output_buffer_pdf_view = reshape(@view(this_output_buffer[1:pdf_size]), size(precon_f))
            output_buffer_p_view = @view(this_output_buffer[pdf_size+1:end])
            f_lower_endpoints = @view scratch_dummy.buffer_vpavperpr_1[:,:,ir]
            f_upper_endpoints = @view scratch_dummy.buffer_vpavperpr_2[:,:,ir]
            receive_buffer1 = @view scratch_dummy.buffer_vpavperpr_3[:,:,ir]
            receive_buffer2 = @view scratch_dummy.buffer_vpavperpr_4[:,:,ir]

            function adi_communicate_boundary_points()
                # Ensure values of precon_f and precon_p are consistent across
                # distributed-MPI block boundaries. For precon_f take the upwind value,
                # and for precon_p take the average.
                @begin_vperp_vpa_region()
                @loop_vperp_vpa ivperp ivpa begin
                    f_lower_endpoints[ivpa,ivperp] = output_buffer_pdf_view[ivpa,ivperp,1]
                    f_upper_endpoints[ivpa,ivperp] = output_buffer_pdf_view[ivpa,ivperp,end]
                end
                # We upwind the z-derivatives in `electron_z_advection!()`, so would
                # expect that upwinding the results here in z would make sense.
                # However, upwinding here makes convergence much slower (~10x),
                # compared to picking the values from one side or other of the block
                # boundary, or taking the average of the values on either side.
                # Neither direction is special, so taking the average seems most
                # sensible (although in an intial test it does not seem to converge
                # faster than just picking one or the other).
                # Maybe this could indicate that it is more important to have a fully
                # self-consistent Jacobian inversion for the
                # `electron_vpa_advection()` part rather than taking half(ish) of the
                # values from one block and the other half(ish) from the other.
                reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
                    output_buffer_pdf_view, f_lower_endpoints, f_upper_endpoints, receive_buffer1,
                    receive_buffer2, z)

                @begin_serial_region()
                @serial_region begin
                    buffer_1[] = output_buffer_p_view[1]
                    buffer_2[] = output_buffer_p_view[end]
                end
                reconcile_element_boundaries_MPI!(
                    output_buffer_p_view, buffer_1, buffer_2, buffer_3, buffer_4, z)

                return nothing
            end

            @begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                row = (iz - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                this_input_buffer[row] = precon_f[ivpa,ivperp,iz]
            end
            @begin_z_region()
            @loop_z iz begin
                row = pdf_size + iz
                this_input_buffer[row] = precon_p[iz]
            end
            @_block_synchronize()

            # Use this to copy current guess from output_buffer to
            # intermediate_buffer, to avoid race conditions as new guess is
            # written into output_buffer.
            function fill_intermediate_buffer!()
                @_block_synchronize()
                for i ∈ global_index_subrange
                    this_intermediate_buffer[i] = this_output_buffer[i]
                end
                @_block_synchronize()
            end

            v_solve_global_inds = adi_info.v_solve_global_inds
            v_solve_nsolve = adi_info.v_solve_nsolve
            v_solve_implicit_lus = adi_info.v_solve_implicit_lus
            v_solve_explicit_matrices = adi_info.v_solve_explicit_matrices
            v_solve_buffer = adi_info.v_solve_buffer
            v_solve_buffer2 = adi_info.v_solve_buffer2
            function first_adi_v_solve!()
                # The initial guess is all-zero, so for the first solve there is
                # no need to multiply by the 'explicit matrix' as x==0, so E.x==0
                for isolve ∈ 1:v_solve_nsolve
                    this_inds = v_solve_global_inds[isolve]
                    v_solve_buffer .= this_input_buffer[this_inds]
                    @timeit_debug global_timer "ldiv!" ldiv!(v_solve_buffer2, v_solve_implicit_lus[isolve], v_solve_buffer)
                    this_output_buffer[this_inds] .= v_solve_buffer2
                end
            end
            function adi_v_solve!()
                for isolve ∈ 1:v_solve_nsolve
                    this_inds = v_solve_global_inds[isolve]
                    v_solve_buffer .= @view this_input_buffer[this_inds]
                    # Need to multiply the 'explicit matrix' by -1, because all
                    # the Jacobian-calculation functions are defined as if the
                    # terms are being added to the left-hand-side preconditioner
                    # matrix, but here the 'explicit matrix' terms are added on
                    # the right-hand-side.
                    @timeit_debug global_timer "mul!" mul!(v_solve_buffer, v_solve_explicit_matrices[isolve],
                         this_intermediate_buffer, -1.0, 1.0)
                    @timeit_debug global_timer "ldiv!" ldiv!(v_solve_buffer2, v_solve_implicit_lus[isolve], v_solve_buffer)
                    this_output_buffer[this_inds] .= v_solve_buffer2
                end
            end

            z_solve_global_inds = adi_info.z_solve_global_inds
            z_solve_nsolve = adi_info.z_solve_nsolve
            z_solve_implicit_lus = adi_info.z_solve_implicit_lus
            z_solve_explicit_matrices = adi_info.z_solve_explicit_matrices
            z_solve_buffer = adi_info.z_solve_buffer
            z_solve_buffer2 = adi_info.z_solve_buffer2
            function adi_z_solve!()
                for isolve ∈ 1:z_solve_nsolve
                    this_inds = z_solve_global_inds[isolve]
                    z_solve_buffer .= @view this_input_buffer[this_inds]
                    # Need to multiply the 'explicit matrix' by -1, because all
                    # the Jacobian-calculation functions are defined as if the
                    # terms are being added to the left-hand-side preconditioner
                    # matrix, but here the 'explicit matrix' terms are added on
                    # the right-hand-side.
                    @timeit_debug global_timer "mul!" mul!(z_solve_buffer, z_solve_explicit_matrices[isolve], this_intermediate_buffer, -1.0, 1.0)
                    @timeit_debug global_timer "ldiv!" ldiv!(z_solve_buffer2, z_solve_implicit_lus[isolve], z_solve_buffer)
                    this_output_buffer[this_inds] .= z_solve_buffer2
                end
            end

            precon_iterations[] += 1
            first_adi_v_solve!()
            fill_intermediate_buffer!()
            adi_z_solve!()
            adi_communicate_boundary_points()

            for n ∈ 1:n_extra_iterations
                precon_iterations[] += 1
                fill_intermediate_buffer!()
                adi_v_solve!()
                fill_intermediate_buffer!()
                adi_z_solve!()
                adi_communicate_boundary_points()
            end

            # Unpack preconditioner solution
            @begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                row = (iz - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                precon_f[ivpa,ivperp,iz] = this_output_buffer[row]
            end
            @begin_z_region()
            @loop_z iz begin
                row = pdf_size + iz
                precon_p[iz] = this_output_buffer[row]
            end

            return nothing
        end

        left_preconditioner = identity
        right_preconditioner = adi_precon!
    elseif nl_solver_params.preconditioner_type === Val(:none)
        left_preconditioner = identity
        right_preconditioner = identity
    else
        error("preconditioner_type=$(nl_solver_params.preconditioner_type) is not "
              * "supported by electron_backward_euler!().")
    end

    # Do a backward-Euler update of the electron pdf, and (if evove_p=true) the electron
    # parallel pressure.
    function residual_func!(this_residual, new_variables; krylov=false)
        electron_p_residual, f_electron_residual = this_residual
        electron_p_newvar, f_electron_newvar = new_variables

        if evolve_p
            this_vth = @view moments.electron.vth[:,ir]
            @begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                this_vth[iz] = sqrt(abs(2.0 * electron_p_newvar[iz] /
                                        (electron_density[iz] * composition.me_over_mi)))
            end
        end

        # enforce the boundary condition(s) on the electron pdf
        @views enforce_boundary_condition_on_electron_pdf!(
                   f_electron_newvar, phi, moments.electron.vth[:,ir],
                   electron_upar, z, vperp, vpa, vperp_spectral, vpa_spectral, vpa_advect,
                   moments, num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                   composition.me_over_mi, ir; bc_constraints=false, update_vcut=!krylov)

        # Calculate derived moments and derivatives using new_variables
        calculate_electron_moments_no_r!(f_electron_newvar, electron_density,
                                         electron_upar, electron_p_newvar, ion_density,
                                         ion_upar, moments, composition, collisions, r, z,
                                         vperp, vpa, ir)
        calculate_electron_moment_derivatives_no_r!(
            moments, electron_density, electron_upar, electron_p_newvar, scratch_dummy, z,
            z_spectral, num_diss_params.electron.moment_dissipation_coefficient, ir)

        if evolve_p
            @begin_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = electron_p_old[iz,ir]
            end
        else
            @begin_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = 0.0
            end
        end

        # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
        # electron_pdf member of the first argument, so if we set the electron_pdf member
        # of the first argument to zero, and pass dt=1, then it will evaluate the time
        # derivative, which is the residual for a steady-state solution.
        @begin_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_electron_residual[ivpa,ivperp,iz] = f_electron_old[ivpa,ivperp,iz]
        end
        electron_kinetic_equation_euler_update!(
            (f_electron_residual, electron_p_residual), f_electron_newvar,
            electron_p_newvar, moments, z, vperp, vpa, z_spectral, vpa_spectral,
            z_advect, vpa_advect, scratch_dummy, collisions, composition,
            external_source_settings, num_diss_params, t_params, ir; evolve_p=evolve_p,
            ion_dt=ion_dt, soft_force_constraints=true)

        # Now
        #   residual = f_electron_old + dt*RHS(f_electron_newvar)
        # so update to desired residual
        @begin_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_electron_residual[ivpa,ivperp,iz] = f_electron_newvar[ivpa,ivperp,iz] - f_electron_residual[ivpa,ivperp,iz]
        end
        if evolve_p
            @begin_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = electron_p_newvar[iz] - electron_p_residual[iz]
            end
        end

        # Set residual to zero where pdf_electron is determined by boundary conditions.
        if vpa.n > 1
            @begin_z_vperp_region()
            @loop_z_vperp iz ivperp begin
                @views enforce_v_boundary_condition_local!(f_electron_residual[:,ivperp,iz], vpa.bc,
                                                           vpa_advect[1].speed[:,ivperp,iz,ir],
                                                           num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                           vpa, vpa_spectral)
            end
        end
        if vperp.n > 1
            @begin_z_vpa_region()
            enforce_vperp_boundary_condition!(f_electron_residual, vperp.bc,
                                              vperp, vperp_spectral, vperp_adv,
                                              vperp_diffusion, ir)
        end
        zero_z_boundary_condition_points(f_electron_residual, z, vpa, moments, ir)

        return nothing
    end

    residual = (scratch_dummy.implicit_buffer_z_1, scratch_dummy.implicit_buffer_vpavperpz_1)
    delta_x = (scratch_dummy.implicit_buffer_z_2,
               scratch_dummy.implicit_buffer_vpavperpz_2)
    rhs_delta = (scratch_dummy.implicit_buffer_z_3,
                 scratch_dummy.implicit_buffer_vpavperpz_3)
    v = (scratch_dummy.implicit_buffer_z_4,
         scratch_dummy.implicit_buffer_vpavperpz_4)
    w = (scratch_dummy.implicit_buffer_z_5,
         scratch_dummy.implicit_buffer_vpavperpz_5)

    newton_success = newton_solve!((electron_p_new, f_electron_new), residual_func!,
                                   residual, delta_x, rhs_delta, v, w, nl_solver_params;
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   coords=(z=z, vperp=vperp, vpa=vpa))

    return newton_success
end

"""
    implicit_electron_advance!()

Do an implicit solve which finds: the steady-state electron shape function \$g_e\$; the
backward-Euler advanced electron pressure which is updated using \$g_e\$ at the new
time-level.

The r-dimension is not parallelised. For 1D runs this makes no difference. In 2D it might
or might not be necessary. If r-dimension parallelisation is needed, it would need some
work. The simplest option would be a non-parallelised outer loop over r, with each
nonlinear solve being parallelised over {z,vperp,vpa}. More efficient might be to add an
equivalent to the 'anysv' parallelisation used for the collision operator (e.g. 'anyzv'?)
to allow the outer r-loop to be parallelised.
"""
@timeit global_timer implicit_electron_advance!(
                         fvec_out, fvec_in, pdf, scratch_electron, moments, fields,
                         collisions, composition, geometry, external_source_settings,
                         num_diss_params, r, z, vperp, vpa, r_spectral, z_spectral,
                         vperp_spectral, vpa_spectral, z_advect, vpa_advect, gyroavs,
                         scratch_dummy, t_params, ion_dt, nl_solver_params) = begin

    electron_p_out = fvec_out.electron_p
    # Store the solved-for pdf in n_rk_stages+1, because this was the version that gets
    # written to output for the explicit-electron-timestepping version.
    pdf_electron_out = scratch_electron.pdf_electron

    # If we just defined the residual for the electron distribution function solve to be
    # 'dg/dt=0', then we would be asking the solver (roughly) to find g such that
    # 'dg/dt<tol' for some tolerance. However dg/dt has some typical timescale, τ, so
    # 'dg/dt∼g/τ'. It would be inconvenient to have to define the tolerances taking τ
    # (normalised to the reference sound crossing time) into account, so instead estimate
    # the relevant timescale as 'sqrt(me/mi)*z.L', i.e. as the electron thermal crossing
    # time at reference parameters. We pass this as 'dt' to
    # electron_kinetic_equation_euler_update!() so that it multiplies the residual.
    pdf_electron_normalisation_factor = sqrt(composition.me_over_mi) * z.L

    # Do a forward-Euler step for electron_p to get the initial guess.
    # No equivalent for f_electron, as f_electron obeys a steady-state equation.
    calculate_electron_moment_derivatives!(moments, fvec_in, scratch_dummy, z, z_spectral,
                                           num_diss_params.electron.moment_dissipation_coefficient,
                                           composition.electron_physics)
    electron_energy_equation!(electron_p_out, fvec_in.density, fvec_in.electron_p,
                              fvec_in.density, fvec_in.electron_upar, fvec_in.density,
                              fvec_in.upar, fvec_in.p, fvec_in.density_neutral,
                              fvec_in.uz_neutral, fvec_in.pz_neutral, moments.electron,
                              collisions, ion_dt, composition,
                              external_source_settings.electron, num_diss_params, r, z)

    newton_success = false
    for ir ∈ 1:r.n
        f_electron = @view pdf_electron_out[:,:,:,ir]
        p = @view electron_p_out[:,ir]
        phi = @view fields.phi[:,ir]

        function recalculate_preconditioner!()
            if nl_solver_params.preconditioner_type === Val(:electron_lu)
global_rank[] == 0 && println("recalculating precon")
                nl_solver_params.solves_since_precon_update[] = 0
                nl_solver_params.precon_dt[] = ion_dt

                orig_lu, precon_matrix, input_buffer, output_buffer =
                    nl_solver_params.preconditioners[ir]

                fill_electron_kinetic_equation_Jacobian!(
                    precon_matrix, f_electron, p, moments, phi, collisions, composition,
                    z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
                    vpa_advect, scratch_dummy, external_source_settings, num_diss_params,
                    t_params, ion_dt, ir, true, :all, true, false)

                @begin_serial_region()
                if block_rank[] == 0
                    if size(orig_lu) == (1, 1)
                        # Have not properly created the LU decomposition before, so
                        # cannot reuse it.
                        @timeit_debug global_timer "lu" nl_solver_params.preconditioners[ir] =
                            (lu(sparse(precon_matrix)), precon_matrix, input_buffer,
                             output_buffer)
                    else
                        # LU decomposition was previously created. The Jacobian always
                        # has the same sparsity pattern, so by using `lu!()` we can
                        # reuse some setup.
                        try
                            @timeit_debug global_timer "lu!" lu!(orig_lu, sparse(precon_matrix); check=false)
                        catch e
                            if !isa(e, ArgumentError)
                                rethrow(e)
                            end
                            println("Sparsity pattern of matrix changed, rebuilding "
                                    * " LU from scratch")
                            @timeit_debug global_timer "lu" orig_lu = lu(sparse(precon_matrix))
                        end
                        nl_solver_params.preconditioners[ir] =
                            (orig_lu, precon_matrix, input_buffer, output_buffer)
                    end
                else
                    nl_solver_params.preconditioners[ir] =
                        (orig_lu, precon_matrix, input_buffer, output_buffer)
                end

                return nothing
            end
        end

        if nl_solver_params.preconditioner_type === Val(:electron_lu)
            if ion_dt > 1.5 * nl_solver_params.precon_dt[] ||
                    ion_dt < 2.0/3.0 * nl_solver_params.precon_dt[]

                # dt has changed significantly, so update the preconditioner
                nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
            end

            if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
                recalculate_preconditioner!()
            end

            @timeit_debug global_timer lu_precon!(x) = begin
                precon_p, precon_f = x

                precon_lu, _, this_input_buffer, this_output_buffer =
                    nl_solver_params.preconditioners[ir]

                @begin_serial_region()
                counter = 1
                @loop_z_vperp_vpa iz ivperp ivpa begin
                    this_input_buffer[counter] = precon_f[ivpa,ivperp,iz]
                    counter += 1
                end
                @loop_z iz begin
                    this_input_buffer[counter] = precon_p[iz]
                    counter += 1
                end

                @begin_serial_region()
                @serial_region begin
                    @timeit_debug global_timer "ldiv!" ldiv!(this_output_buffer, precon_lu, this_input_buffer)
                end

                @begin_serial_region()
                counter = 1
                @loop_z_vperp_vpa iz ivperp ivpa begin
                    precon_f[ivpa,ivperp,iz] = this_output_buffer[counter]
                    counter += 1
                end
                @loop_z iz begin
                    precon_p[iz] = this_output_buffer[counter]
                    counter += 1
                end

                # Ensure values of precon_f and precon_p are consistent across
                # distributed-MPI block boundaries. For precon_f take the upwind value,
                # and for precon_p take the average.
                f_lower_endpoints = @view scratch_dummy.buffer_vpavperpr_1[:,:,ir]
                f_upper_endpoints = @view scratch_dummy.buffer_vpavperpr_2[:,:,ir]
                receive_buffer1 = @view scratch_dummy.buffer_vpavperpr_3[:,:,ir]
                receive_buffer2 = @view scratch_dummy.buffer_vpavperpr_4[:,:,ir]
                @begin_vperp_vpa_region()
                @loop_vperp_vpa ivperp ivpa begin
                    f_lower_endpoints[ivpa,ivperp] = precon_f[ivpa,ivperp,1]
                    f_upper_endpoints[ivpa,ivperp] = precon_f[ivpa,ivperp,end]
                end
                # We upwind the z-derivatives in `electron_z_advection!()`, so would
                # expect that upwinding the results here in z would make sense.
                # However, upwinding here makes convergence much slower (~10x),
                # compared to picking the values from one side or other of the block
                # boundary, or taking the average of the values on either side.
                # Neither direction is special, so taking the average seems most
                # sensible (although in an intial test it does not seem to converge
                # faster than just picking one or the other).
                # Maybe this could indicate that it is more important to have a fully
                # self-consistent Jacobian inversion for the
                # `electron_vpa_advection()` part rather than taking half(ish) of the
                # values from one block and the other half(ish) from the other.
                reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
                    precon_f, f_lower_endpoints, f_upper_endpoints, receive_buffer1,
                    receive_buffer2, z)

                @begin_serial_region()
                @serial_region begin
                    buffer_1[] = precon_p[1]
                    buffer_2[] = precon_p[end]
                end
                reconcile_element_boundaries_MPI!(
                    precon_p, buffer_1, buffer_2, buffer_3, buffer_4, z)

                return nothing
            end

            left_preconditioner = identity
            right_preconditioner = lu_precon!
        elseif nl_solver_params.preconditioner_type === Val(:none)
            left_preconditioner = identity
            right_preconditioner = identity
        else
            error("preconditioner_type=$(nl_solver_params.preconditioner_type) is not "
                  * "supported by electron_backward_euler!().")
        end

        function residual_func!(residual, new_variables; debug=false, krylov=false)
            electron_p_residual, f_electron_residual = residual
            electron_p_new, f_electron_new = new_variables

            this_vth = moments.electron.vth
            @begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                this_vth[iz,ir] = sqrt(abs(2.0 * electron_p_new[iz,ir] /
                                           (electron_density[iz] *
                                            composition.me_over_mi)))
            end

            # enforce the boundary condition(s) on the electron pdf
            @views enforce_boundary_condition_on_electron_pdf!(
                       f_electron_new, phi, moments.electron.vth[:,ir], electron_upar, z,
                       vperp, vpa, vperp_spectral, vpa_spectral, vpa_advect, moments,
                       num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                       composition.me_over_mi, ir; bc_constraints=false,
                       update_vcut=!krylov)

            calculate_electron_moments_no_r!(f_electron_new, electron_density,
                                             electron_upar, electron_p_new, ion_density,
                                             ion_upar, moments, composition, collisions,
                                             r, z, vperp, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments, electron_density, electron_upar, electron_p_new, scratch_dummy,
                z, z_spectral, num_diss_params.electron.moment_dissipation_coefficient,
                ir)

            @begin_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = 0.0
            end

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            @begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                f_electron_residual[ivpa,ivperp,iz] = 0.0
            end
            t_params.dt[] = pdf_electron_normalisation_factor
            electron_kinetic_equation_euler_update!(
                (f_electron_residual, electron_p_residual), f_electron_new,
                electron_p_new, moments, z, vperp, vpa, z_spectral, vpa_spectral,
                z_advect, vpa_advect, scratch_dummy, collisions, composition,
                external_source_settings, num_diss_params, t_params, ir; evolve_p=true,
                ion_dt=ion_dt, soft_force_constraints=true)

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                @begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(f_electron_residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                @begin_z_vpa_region()
                enforce_vperp_boundary_condition!(f_electron_residual, vperp.bc, vperp, vperp_spectral,
                                                  vperp_adv, vperp_diffusion)
            end
            zero_z_boundary_condition_points(f_electron_residual, z, vpa, moments, ir)
            return nothing
        end

        residual = (scratch_dummy.implicit_buffer_z_1,
                    scratch_dummy.implicit_buffer_vpavperpz_1)
        delta_x = (scratch_dummy.implicit_buffer_z_2,
                   scratch_dummy.implicit_buffer_vpavperpz_2)
        rhs_delta = (scratch_dummy.implicit_buffer_z_3,
                     scratch_dummy.implicit_buffer_vpavperpz_3)
        v = (scratch_dummy.implicit_buffer_z_4,
             scratch_dummy.implicit_buffer_vpavperpz_4)
        w = (scratch_dummy.implicit_buffer_z_5,
             scratch_dummy.implicit_buffer_vpavperpz_5)

        newton_success = newton_solve!((p, f_electron), residual_func!, residual, delta_x,
                                       rhs_delta, v, w, nl_solver_params;
                                       left_preconditioner=nothing,
                                       right_preconditioner=nothing,
                                       recalculate_preconditioner=recalculate_preconditioner!,
                                       coords=(z=z, vperp=vperp, vpa=vpa))
        if !newton_success
            break
        end
    end

    # Fill pdf.electron.norm
    non_scratch_pdf = pdf.electron.norm
    @begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        non_scratch_pdf[ivpa,ivperp,iz,ir] = pdf_electron_out[ivpa,ivperp,iz,ir]
    end

    # Update the electron parallel friction force.
    # This does not actually do anything for kinetic electron runs at the moment, but
    # include as a reminder to update this if/when we do include e-i collisions for
    # kinetic electrons.
    calculate_electron_parallel_friction_force!(
        moments.electron.parallel_friction, fvec_out.electron_density,
        fvec_out.electron_upar, fvec_out.upar, moments.electron.dT_dz,
        composition.me_over_mi, collisions.electron_fluid.nu_ei,
        composition.electron_physics)

    # Solve for EM fields now that electrons are updated.
    update_phi!(fields, fvec_out, vperp, z, r, composition, collisions, moments,
                geometry, z_spectral, r_spectral, scratch_dummy, gyroavs)

    if !newton_success
        success = "kinetic-electrons"
    else
        success = ""
    end
    return success
end

function apply_electron_bc_and_constraints!(this_scratch, phi, moments, r, z, vperp, vpa,
                                            vperp_spectral, vpa_spectral, vpa_advect,
                                            num_diss_params, composition;
                                            lowerz_vcut_inds=nothing,
                                            upperz_vcut_inds=nothing)
    latest_pdf = this_scratch.pdf_electron

    @begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        latest_pdf[ivpa,ivperp,iz,ir] = max(latest_pdf[ivpa,ivperp,iz,ir], 0.0)
    end

    for ir ∈ 1:r.n
        # enforce the boundary condition(s) on the electron pdf
        @views enforce_boundary_condition_on_electron_pdf!(
                   latest_pdf[:,:,:,ir], phi[:,ir], moments.electron.vth[:,ir],
                   moments.electron.upar[:,ir], z, vperp, vpa, vperp_spectral,
                   vpa_spectral, vpa_advect, moments,
                   num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                   composition.me_over_mi, ir;
                   lowerz_vcut_ind=(lowerz_vcut_inds === nothing ? nothing : lowerz_vcut_inds[ir]),
                   upperz_vcut_ind=(upperz_vcut_inds === nothing ? nothing : upperz_vcut_inds[ir]))
    end

    @begin_r_z_region()
    A = moments.electron.constraints_A_coefficient
    B = moments.electron.constraints_B_coefficient
    C = moments.electron.constraints_C_coefficient
    skip_first = z.irank == 0 && !z.periodic
    skip_last = z.irank == z.nrank - 1 && !z.periodic
    @loop_r_z ir iz begin
        if (iz == 1 && skip_first) || (iz == z.n && skip_last)
            continue
        end
        (A[iz,ir], B[iz,ir], C[iz,ir]) =
            @views hard_force_moment_constraints!(latest_pdf[:,:,iz,ir],
                                                  (evolve_density=true,
                                                   evolve_upar=true,
                                                   evolve_p=true), vpa, vperp)
    end
end

function apply_electron_bc_and_constraints_no_r!(f_electron, phi, moments, r, z, vperp,
                                                 vpa, vperp_spectral, vpa_spectral,
                                                 vpa_advect, num_diss_params, composition,
                                                 ir, nl_solver_params)
    @begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        f_electron[ivpa,ivperp,iz] = max(f_electron[ivpa,ivperp,iz], 0.0)
    end

    new_lowerz_vcut_ind = @view r.scratch_shared_int[ir:ir]
    new_upperz_vcut_ind = @view r.scratch_shared_int2[ir:ir]

    # enforce the boundary condition(s) on the electron pdf
    bc_converged = Ref(true)
    bc_converged[] = @views enforce_boundary_condition_on_electron_pdf!(
                              f_electron, phi, moments.electron.vth[:,ir],
                              moments.electron.upar[:,ir], z, vperp, vpa, vperp_spectral,
                              vpa_spectral, vpa_advect, moments,
                              num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                              composition.me_over_mi, ir;
                              lowerz_vcut_ind=new_lowerz_vcut_ind,
                              upperz_vcut_ind=new_upperz_vcut_ind)
    @serial_region begin
        # Only the return value from rank-0 of each shared memory block matters
        MPI.Allreduce!(bc_converged, &, comm_inter_block[])
    end
    MPI.Bcast!(bc_converged, comm_block[]; root=0)
    if !bc_converged[]
        return "electron_bc"
    end

    if nl_solver_params !== nothing
        # Check if either vcut_ind has changed - if either has, recalculate the
        # preconditioner because the response at the grid point before the cutoff is
        # sharp, so the preconditioner could be significantly wrong when it was
        # calculated using the wrong vcut_ind.
        lower_vcut_changed = @view z.scratch_shared_int[1:1]
        upper_vcut_changed = @view z.scratch_shared_int[2:2]
        @serial_region begin
            if z.irank == 0
                precon_lowerz_vcut_inds = nl_solver_params.precon_lowerz_vcut_inds
                if new_lowerz_vcut_ind[] == precon_lowerz_vcut_inds[ir]
                    lower_vcut_changed[] = 0
                else
                    lower_vcut_changed[] = 1
                    precon_lowerz_vcut_inds[ir] = new_lowerz_vcut_ind[]
                end
            end
            MPI.Bcast!(lower_vcut_changed, comm_inter_block[]; root=0)
            #req1 = MPI.Ibcast!(lower_vcut_changed, comm_inter_block[]; root=0)

            if z.irank == z.nrank - 1
                precon_upperz_vcut_inds = nl_solver_params.precon_upperz_vcut_inds
                if new_upperz_vcut_ind[] == precon_upperz_vcut_inds[ir]
                    upper_vcut_changed[] = 0
                else
                    upper_vcut_changed[] = 1
                    precon_upperz_vcut_inds[ir] = new_upperz_vcut_ind[]
                end
            end
            MPI.Bcast!(upper_vcut_changed, comm_inter_block[]; root=n_blocks[]-1)
            #req2 = MPI.Ibcast!(upper_vcut_changed, comm_inter_block[]; root=n_blocks[]-1)

            # Eventually can use Ibcast!() to make the two broadcasts run
            # simultaneously, but need the function to be merged into MPI.jl (see
            # https://github.com/JuliaParallel/MPI.jl/pull/882).
            #MPI.Waitall([req1, req2])
        end
        @_block_synchronize()
        if lower_vcut_changed[] == 1 || upper_vcut_changed[] == 1
            # One or both of the indices changed for some `ir`, so force the
            # preconditioner to be recalculated next time.
            nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
        end
    end

    @begin_z_region()
    A = moments.electron.constraints_A_coefficient
    B = moments.electron.constraints_B_coefficient
    C = moments.electron.constraints_C_coefficient
    skip_first = z.irank == 0 && !z.periodic
    skip_last = z.irank == z.nrank - 1 && !z.periodic
    @loop_z iz begin
        if (iz == 1 && skip_first) || (iz == z.n && skip_last)
            continue
        end
        (A[iz,ir], B[iz,ir], C[iz,ir]) =
            @views hard_force_moment_constraints!(f_electron[:,:,iz],
                                                  (evolve_density=true,
                                                   evolve_upar=true,
                                                   evolve_p=true), vpa, vperp)
    end

    return ""
end

function get_cutoff_params_lower(upar, vthe, phi, me_over_mi, vpa)
    u_over_vt = upar / vthe

    # sigma is the location we use for w_∥(v_∥=0) - set to 0 to ignore the 'upar
    # shift'
    sigma = -u_over_vt

    vpa_unnorm = @. vpa.scratch2 = vthe * (vpa.grid - sigma)

    # Initial guess for cut-off velocity is result from previous RK stage (which
    # might be the previous timestep if this is the first stage). Recalculate this
    # value from phi.
    vcut = sqrt(2.0 * phi / me_over_mi)

    # -vcut is between minus_vcut_ind-1 and minus_vcut_ind
    minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)
    if vcut == 0.0
        # Force a non-zero initial guess, as zero makes no sense - that would mean all
        # electrons are absorbed, i.e. there is no sheath.
        minus_vcut_ind -= 1
        vcut = -vpa_unnorm[minus_vcut_ind]
    end
    if minus_vcut_ind < 2
        error("In lower-z electron bc, failed to find vpa=-vcut point, minus_vcut_ind=$minus_vcut_ind")
    end
    if minus_vcut_ind > vpa.n
        error("In lower-z electron bc, failed to find vpa=-vcut point, minus_vcut_ind=$minus_vcut_ind")
    end

    # sigma is between sigma_ind-1 and sigma_ind
    sigma_ind = searchsortedfirst(vpa_unnorm, 0.0)
    if sigma_ind < 2
        error("In lower-z electron bc, failed to find vpa=0 point, sigma_ind=$sigma_ind")
    end
    if sigma_ind > vpa.n
        error("In lower-z electron bc, failed to find vpa=0 point, sigma_ind=$sigma_ind")
    end

    # sigma_fraction is the fraction of the distance between sigma_ind-1 and
    # sigma_ind where sigma is.
    sigma_fraction = -vpa_unnorm[sigma_ind-1] / (vpa_unnorm[sigma_ind] - vpa_unnorm[sigma_ind-1])

    # Want the element that contains the interval on the lower side of sigma_ind. For
    # points on element boundaries, the `ielement` array contains the element on the lower
    # side of the grid point, so just looking up the `ielement` of `sigma_ind` is what we
    # want here.
    element_with_zero = vpa.ielement[sigma_ind]
    element_with_zero_boundary = element_with_zero == 1 ? vpa.imin[element_with_zero] :
                                                          vpa.imin[element_with_zero] - 1
    # This searchsortedlast() call finds the last point ≤ to the negative of v_∥
    # at the lower boundary of the element containing zero.
    last_point_near_zero = searchsortedlast(vpa_unnorm,
                                            -vpa_unnorm[element_with_zero_boundary])

    # Want to construct the w-grid corresponding to -vpa.
    #   wpa(vpa) = (vpa - upar)/vth
    #   ⇒ vpa = vth*wpa(vpa) + upar
    #   wpa(-vpa) = (-vpa - upar)/vth
    #             = (-(vth*wpa(vpa) + upar) - upar)/vth
    #             = (-vth*wpa - 2*upar)/vth
    #             = -wpa - 2*upar/vth
    # [Note that `vpa.grid` is slightly mis-named here - it contains the values of
    #  wpa(+vpa) as we are using a 'moment kinetic' approach.]
    # Need to reverse vpa.grid because the grid passed as the second argument of
    # interpolate_to_grid_1d!() needs to be sorted in increasing order.
    reversed_wpa_of_minus_vpa = @. vpa.scratch3 = -vpa.grid + 2.0 * sigma
    #reversed_wpa_of_minus_vpa = vpa.scratch3 .= .-vpa.grid
    reverse!(reversed_wpa_of_minus_vpa)

    return vpa_unnorm, u_over_vt, vcut, minus_vcut_ind, sigma, sigma_ind, sigma_fraction,
           element_with_zero, element_with_zero_boundary, last_point_near_zero,
           reversed_wpa_of_minus_vpa
end

function get_cutoff_params_upper(upar, vthe, phi, me_over_mi, vpa)
    u_over_vt = upar / vthe

    # sigma is the location we use for w_∥(v_∥=0) - set to 0 to ignore the 'upar
    # shift'
    sigma = -u_over_vt

    # Delete the upar contribution here if ignoring the 'upar shift'
    vpa_unnorm = @. vpa.scratch2 = vthe * (vpa.grid - sigma)

    # Initial guess for cut-off velocity is result from previous RK stage (which
    # might be the previous timestep if this is the first stage). Recalculate this
    # value from phi.
    vcut = sqrt(2.0 * phi / me_over_mi)

    # vcut is between plus_vcut_ind and plus_vcut_ind+1
    plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)
    if vcut == 0.0
        # Force a non-zero initial guess, as zero makes no sense - that would mean all
        # electrons are absorbed, i.e. there is no sheath.
        plus_vcut_ind += 1
        vcut = vpa_unnorm[plus_vcut_ind]
    end
    if plus_vcut_ind < 1
        error("In upper-z electron bc, failed to find vpa=vcut point, plus_vcut_ind=$plus_vcut_ind")
    end
    if plus_vcut_ind > vpa.n - 1
        error("In upper-z electron bc, failed to find vpa=vcut point, plus_vcut_ind=$plus_vcut_ind")
    end

    # sigma is between sigma_ind and sigma_ind+1
    sigma_ind = searchsortedlast(vpa_unnorm, 0.0)
    if sigma_ind < 1
        error("In upper-z electron bc, failed to find vpa=0 point, sigma_ind=$sigma_ind")
    end
    if sigma_ind > vpa.n - 1
        error("In upper-z electron bc, failed to find vpa=0 point, sigma_ind=$sigma_ind")
    end

    # sigma_fraction is the fraction of the distance between sigma_ind+1 and
    # sigma_ind where sigma is.
    sigma_fraction = -vpa_unnorm[sigma_ind+1] / (vpa_unnorm[sigma_ind] - vpa_unnorm[sigma_ind+1])

    # Want the element that contains the interval on the upper side of sigma_ind. For
    # points on element boundaries, the `ielement` array contains the element on the lower
    # side of the grid point, we need the `ielement` of `sigma_ind+1` here.
    element_with_zero = vpa.ielement[sigma_ind+1]
    element_with_zero_boundary = vpa.imax[element_with_zero]
    # This searchsortedfirst() call finds the first point ≥ to the negative of v_∥ at the
    # upper boundary of the element containing zero.
    first_point_near_zero = searchsortedfirst(vpa_unnorm,
                                              -vpa_unnorm[element_with_zero_boundary])

    # Want to construct the w-grid corresponding to -vpa.
    #   wpa(vpa) = (vpa - upar)/vth
    #   ⇒ vpa = vth*wpa(vpa) + upar
    #   wpa(-vpa) = (-vpa - upar)/vth
    #             = (-(vth*wpa(vpa) + upar) - upar)/vth
    #             = (-vth*wpa - 2*upar)/vth
    #             = -wpa - 2*upar/vth
    # [Note that `vpa.grid` is slightly mis-named here - it contains the values of
    #  wpa(+vpa) as we are using a 'moment kinetic' approach.]
    # Need to reverse vpa.grid because the grid passed as the second argument of
    # interpolate_to_grid_1d!() needs to be sorted in increasing order.
    reversed_wpa_of_minus_vpa = @. vpa.scratch3 = -vpa.grid + 2.0 * sigma
    #reversed_wpa_of_minus_vpa = vpa.scratch3 .= .-vpa.grid
    reverse!(reversed_wpa_of_minus_vpa)

    return vpa_unnorm, u_over_vt, vcut, plus_vcut_ind, sigma, sigma_ind, sigma_fraction,
           element_with_zero, element_with_zero_boundary, first_point_near_zero,
           reversed_wpa_of_minus_vpa
end

function get_minus_vcut_fraction(vcut, minus_vcut_ind, vpa_unnorm)
    return (-vcut - vpa_unnorm[minus_vcut_ind-1]) /
           (vpa_unnorm[minus_vcut_ind] - vpa_unnorm[minus_vcut_ind-1])
end

function get_plus_vcut_fraction(vcut, plus_vcut_ind, vpa_unnorm)
    return (vcut - vpa_unnorm[plus_vcut_ind]) /
           (vpa_unnorm[plus_vcut_ind+1] - vpa_unnorm[plus_vcut_ind])
end

function fill_integral_pieces!(
        pdf_part, vthe, vpa, vpa_unnorm, density_integral_pieces, flow_integral_pieces,
        energy_integral_pieces, cubic_integral_pieces, quartic_integral_pieces)

    @. density_integral_pieces = pdf_part * vpa.wgts
    @. flow_integral_pieces = density_integral_pieces * vpa_unnorm / vthe
    @. energy_integral_pieces = flow_integral_pieces * vpa_unnorm / vthe
    @. cubic_integral_pieces = energy_integral_pieces * vpa_unnorm / vthe
    @. quartic_integral_pieces = cubic_integral_pieces * vpa_unnorm / vthe
end

function get_residual_and_coefficients_for_bc(a1, a1prime, a2, a2prime, b1, b1prime,
                                              c1, c1prime, c2, c2prime, d1, d1prime,
                                              e1, e1prime, e2, e2prime, u_over_vt,
                                              bc_constraints)
    if bc_constraints
        alpha = a1 + 2.0 * a2
        alphaprime = a1prime + 2.0 * a2prime
        beta = c1 + 2.0 * c2
        betaprime = c1prime + 2.0 * c2prime
        gamma = u_over_vt^2 * alpha - 2.0 * u_over_vt * b1 + beta
        gammaprime = u_over_vt^2 * alphaprime - 2.0 * u_over_vt * b1prime + betaprime
        delta = u_over_vt^2 * beta - 2.0 * u_over_vt * d1 + e1 + 2.0 * e2
        deltaprime = u_over_vt^2 * betaprime - 2.0 * u_over_vt * d1prime + e1prime + 2.0 * e2prime

        # Although we write v^2 and d^3w in the notes below, the boundary condition
        # implementation currently only supports 1V distribution functions (it uses 1D
        # scratch arrays from the vpa coordinate object everywhere), and for 1V
        # v^2=v_∥^2, and ∫ <.> d^3w = ∫ <.> dw_∥.
        #
        # Set A and C to impose 'density' and 'pressure' moment constraints. When these
        # are satisfied, the definition of vcut will ensure the 'momentum' constraint is
        # satisfied.
        # a1, a2, b1, b2, c1, c2, d1, d2, e1, e2 are defined so that
        #   ∫ Fhat d^3w = a1 + 2*a2 = alpha
        #   ∫ v_∥ / vth * Fhat d^3w = b1
        #   ∫ v^2 / vth^2 * Fhat d^3w = c1 + 2*c2 = beta
        #   ∫ v_∥ v^2 / vth^3 * Fhat d^3w = d1
        #   ∫ v^4 / vth^4 * Fhat d^3w = e1 + 2*e2
        # We correct F as F = (A + C * v^2 / vth^2) Fhat so that the constraints become,
        # noting that due to the symmetry of the boundary condition between -vcut and
        # +vcut, a2=a3, b2=-b3, c2=c3, d2=-d3, e2=e3 and a4=b4=c4=d4=e4=0 (these
        # constraints will be imposed numerically later).
        #   1 = ∫ F d^3w = A*alpha + C*beta
        #   3/2 = ∫ (v - u)^2 / vth^2 * F d^3w = ∫ (u^2 - 2*u*v_∥ + v^2)/vth^2 * F d^3w
        #                                      = A*(u_over_vt^2*alpha - 2*u_over_vt*b1 + beta) + C*(u_over_vt^2*beta - 2*u_over_vt*d1 + (e1+2*e2))
        #                                      = A*gamma + C*delta
        # are satisfied. These equations can be solved as:
        #   C = (1 - alpha*A) / beta
        #   A*gamma = 3/2 - C*delta
        #   A*gamma*beta = 3/2*beta - (1 - alpha*A)*delta
        #   A*(gamma*beta - alpha*delta) = 3/2*beta - delta
        # Primed variables are the derivatives with respect to vcut. The derivatives of A
        # and C can be found by the chain rule.
        A = (1.5 * beta - delta) / (beta * gamma - alpha * delta)
        Aprime = (1.5 * betaprime - deltaprime
                  - (1.5 * beta - delta) * (gamma * betaprime + beta * gammaprime - delta * alphaprime - alpha * deltaprime)
                  / (beta * gamma - alpha * delta)
                 ) / (beta * gamma - alpha * delta)
        C = (1.0 - alpha * A) / beta
        Cprime = -(A * alphaprime + alpha * Aprime) / beta - (1.0 - alpha * A) * betaprime / beta^2
    else
        A = 1.0
        Aprime = 0.0
        C = 0.0
        Cprime = 0.0
    end

    # epsilon is the error on the momentum constraint, which should be zero for a
    # converged solution.
    # epsilon = ∫ w_∥ F d^3w = ∫ v_∥/vth F d^3w - u_over_vt ∫ F d^3w
    #                        = ∫ v_∥/vth F d^3w - u_over_vt
    #                        = A*b1 + C*d1 - u_over_vt
    epsilon = A * b1 + C * d1 - u_over_vt
    epsilonprime = b1 * Aprime + A * b1prime + d1 * Cprime + C * d1prime

    return epsilon, epsilonprime, A, C
end

function get_integrals_and_derivatives_lowerz(
             vcut, minus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm, u_over_vt,
             density_integral_pieces_lowerz, flow_integral_pieces_lowerz,
             energy_integral_pieces_lowerz, cubic_integral_pieces_lowerz,
             quartic_integral_pieces_lowerz, bc_constraints)
    # vcut_fraction is the fraction of the distance between minus_vcut_ind-1 and
    # minus_vcut_ind where -vcut is.
    vcut_fraction = get_minus_vcut_fraction(vcut, minus_vcut_ind, vpa_unnorm)

    function get_for_one_moment(integral_pieces)
        # Integral contributions from the cell containing vcut.
        # Define these as follows to be consistent with the way the cutoff is
        # applied around plus_vcut_ind below.
        # Note that `integral_vcut_cell_part1` and `integral_vcut_cell_part2`
        # include all the contributions from the grid points
        # `minus_vcut_ind-1` and `minus_vcut_ind`, not just those from
        # 'inside' the grid cell.
        if vcut_fraction < 0.5
            integral_vcut_cell_part2 = integral_pieces[minus_vcut_ind-1] * (0.5 - vcut_fraction) +
                                       integral_pieces[minus_vcut_ind]
            integral_vcut_cell_part1 = integral_pieces[minus_vcut_ind-1] * (0.5 + vcut_fraction)

            # part1prime is d(part1)/d(vcut)
            part1prime = -integral_pieces[minus_vcut_ind-1] / (vpa_unnorm[minus_vcut_ind] - vpa_unnorm[minus_vcut_ind-1])
        else
            integral_vcut_cell_part2 = integral_pieces[minus_vcut_ind] * (1.5 - vcut_fraction)
            integral_vcut_cell_part1 = integral_pieces[minus_vcut_ind-1] +
                                       integral_pieces[minus_vcut_ind] * (vcut_fraction - 0.5)

            # part1prime is d(part1)/d(vcut)
            part1prime = -integral_pieces[minus_vcut_ind] / (vpa_unnorm[minus_vcut_ind] - vpa_unnorm[minus_vcut_ind-1])
        end

        part1 = sum(@view integral_pieces[1:minus_vcut_ind-2]) + integral_vcut_cell_part1

        # Integral contribution from the cell containing sigma
        integral_sigma_cell = (0.5 * integral_pieces[sigma_ind-1] + 0.5 * integral_pieces[sigma_ind])

        part2 = sum(@view integral_pieces[minus_vcut_ind+1:sigma_ind-2])
        part2 += integral_vcut_cell_part2 + 0.5 * integral_pieces[sigma_ind-1] + sigma_fraction * integral_sigma_cell
        # part2prime is d(part2)/d(vcut)
        part2prime = -part1prime

        return part1, part1prime, part2, part2prime
    end
    this_a1, this_a1prime, this_a2, this_a2prime = get_for_one_moment(density_integral_pieces_lowerz)
    this_b1, this_b1prime, this_b2, _ = get_for_one_moment(flow_integral_pieces_lowerz)
    this_c1, this_c1prime, this_c2, this_c2prime = get_for_one_moment(energy_integral_pieces_lowerz)
    this_d1, this_d1prime, this_d2, _ = get_for_one_moment(cubic_integral_pieces_lowerz)
    this_e1, this_e1prime, this_e2, this_e2prime = get_for_one_moment(quartic_integral_pieces_lowerz)

    return get_residual_and_coefficients_for_bc(
               this_a1, this_a1prime, this_a2, this_a2prime, this_b1,
               this_b1prime, this_c1, this_c1prime, this_c2, this_c2prime,
               this_d1, this_d1prime, this_e1, this_e1prime, this_e2,
               this_e2prime, u_over_vt, bc_constraints)...,
           this_a2, this_b2, this_c2, this_d2
end

function get_lowerz_integral_correction_components(
             pdf_slice, vthe, vperp, vpa, vpa_unnorm, u_over_vt, sigma_ind,
             sigma_fraction, vcut, minus_vcut_ind, plus_vcut_ind, bc_constraints)

    # Need to recalculate integral pieces with the updated distribution function
    density_integral_pieces_lowerz = vpa.scratch3
    flow_integral_pieces_lowerz = vpa.scratch4
    energy_integral_pieces_lowerz = vpa.scratch5
    cubic_integral_pieces_lowerz = vpa.scratch6
    quartic_integral_pieces_lowerz = vpa.scratch7
    fill_integral_pieces!(
        pdf_slice, vthe, vpa, vpa_unnorm, density_integral_pieces_lowerz,
        flow_integral_pieces_lowerz, energy_integral_pieces_lowerz,
        cubic_integral_pieces_lowerz, quartic_integral_pieces_lowerz)

    # Update the part2 integrals since we've applied the A and C factors
    _, _, _, _, a2, b2, c2, d2 =
        get_integrals_and_derivatives_lowerz(
            vcut, minus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm, u_over_vt,
            density_integral_pieces_lowerz, flow_integral_pieces_lowerz,
            energy_integral_pieces_lowerz, cubic_integral_pieces_lowerz,
            quartic_integral_pieces_lowerz, bc_constraints)

    function get_part3_for_one_moment_lower(integral_pieces)
        # Integral contribution from the cell containing sigma
        integral_sigma_cell = (0.5 * integral_pieces[sigma_ind-1] + 0.5 * integral_pieces[sigma_ind])

        part3 = sum(@view integral_pieces[sigma_ind+1:plus_vcut_ind+1])
        part3 += 0.5 * integral_pieces[sigma_ind] + (1.0 - sigma_fraction) * integral_sigma_cell

        return part3
    end
    a3 = get_part3_for_one_moment_lower(density_integral_pieces_lowerz)
    b3 = get_part3_for_one_moment_lower(flow_integral_pieces_lowerz)
    c3 = get_part3_for_one_moment_lower(energy_integral_pieces_lowerz)
    d3 = get_part3_for_one_moment_lower(cubic_integral_pieces_lowerz)

    # Use scale factor to adjust how sharp the cutoff near vpa_unnorm=0 is.
    if vperp.n == 1
        # When T_⟂=0, for consistency with original version of code, divide
        # integral_correction_sharpness by 3 when smoothing the cutoff to account for the
        # difference between T and T_∥ in the 1V case.
        correction0_integral_pieces = @. vpa.scratch3 = pdf_slice * vpa.wgts * integral_correction_sharpness / 3.0 * vpa_unnorm^2 / vthe^2 / (1.0 + integral_correction_sharpness / 3.0 * vpa_unnorm^2 / vthe^2)
    else
        correction0_integral_pieces = @. vpa.scratch3 = pdf_slice * vpa.wgts * integral_correction_sharpness * vpa_unnorm^2 / vthe^2 / (1.0 + integral_correction_sharpness * vpa_unnorm^2 / vthe^2)
    end
    for ivpa ∈ 1:sigma_ind
        # We only add the corrections to 'part3', so zero them out for negative v_∥.
        # I think this is only actually significant for `sigma_ind-1` and
        # `sigma_ind`. Even though `sigma_ind` is part of the distribution
        # function that we are correcting, for v_∥>0, it affects the integral in
        # the 'sigma_cell' between `sigma_ind-1` and `sigma_ind`, which would
        # affect the numerically calculated integrals for f(v_∥<0), so if we
        # 'corrected' its value, those integrals would change and the constraints
        # would not be exactly satisfied. The difference should be small, as the
        # correction at that point is multiplied by
        # v_∥^2/vth^2/(1+v_∥^2/vth^2)≈v_∥^2/vth^2≈0.
        correction0_integral_pieces[ivpa] = 0.0
    end
    correction1_integral_pieces = @. vpa.scratch4 = correction0_integral_pieces * vpa_unnorm / vthe
    correction2_integral_pieces = @. vpa.scratch5 = correction1_integral_pieces * vpa_unnorm / vthe
    correction3_integral_pieces = @. vpa.scratch6 = correction2_integral_pieces * vpa_unnorm / vthe
    correction4_integral_pieces = @. vpa.scratch7 = correction3_integral_pieces * vpa_unnorm / vthe
    correction5_integral_pieces = @. vpa.scratch8 = correction4_integral_pieces * vpa_unnorm / vthe
    correction6_integral_pieces = @. vpa.scratch9 = correction5_integral_pieces * vpa_unnorm / vthe

    alpha = get_part3_for_one_moment_lower(correction0_integral_pieces)
    beta = get_part3_for_one_moment_lower(correction1_integral_pieces)
    gamma = get_part3_for_one_moment_lower(correction2_integral_pieces)
    delta = get_part3_for_one_moment_lower(correction3_integral_pieces)
    epsilon = get_part3_for_one_moment_lower(correction4_integral_pieces)
    zeta = get_part3_for_one_moment_lower(correction5_integral_pieces)
    eta = get_part3_for_one_moment_lower(correction6_integral_pieces)

    return a2, b2, c2, d2, a3, b3, c3, d3, alpha, beta, gamma, delta, epsilon, zeta, eta
end

function get_integrals_and_derivatives_upperz(
             vcut, plus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm, u_over_vt,
             density_integral_pieces_upperz, flow_integral_pieces_upperz,
             energy_integral_pieces_upperz, cubic_integral_pieces_upperz,
             quartic_integral_pieces_upperz, bc_constraints)
    # vcut_fraction is the fraction of the distance between plus_vcut_ind and
    # plus_vcut_ind+1 where vcut is.
    vcut_fraction = get_plus_vcut_fraction(vcut, plus_vcut_ind, vpa_unnorm)

    function get_for_one_moment(integral_pieces)
        # Integral contribution from the cell containing vcut
        # Define these as follows to be consistent with the way the cutoff is
        # applied around plus_vcut_ind below.
        # Note that `integral_vcut_cell_part1` and `integral_vcut_cell_part2`
        # include all the contributions from the grid points `plus_vcut_ind`
        # and `plus_vcut_ind+1`, not just those from 'inside' the grid cell.
        if vcut_fraction > 0.5
            integral_vcut_cell_part2 = integral_pieces[plus_vcut_ind] +
                                       integral_pieces[plus_vcut_ind+1] * (vcut_fraction - 0.5)
            integral_vcut_cell_part1 = integral_pieces[plus_vcut_ind+1] * (1.5 - vcut_fraction)

            # part1prime is d(part1)/d(vcut)
            part1prime = -integral_pieces[plus_vcut_ind+1] / (vpa_unnorm[plus_vcut_ind+1] - vpa_unnorm[plus_vcut_ind])
        else
            integral_vcut_cell_part2 = integral_pieces[plus_vcut_ind] * (0.5 + vcut_fraction)
            integral_vcut_cell_part1 = integral_pieces[plus_vcut_ind] * (0.5 - vcut_fraction) +
                                       integral_pieces[plus_vcut_ind+1]

            # part1prime is d(part1)/d(vcut)
            part1prime = -integral_pieces[plus_vcut_ind] / (vpa_unnorm[plus_vcut_ind+1] - vpa_unnorm[plus_vcut_ind])
        end

        part1 = sum(@view integral_pieces[plus_vcut_ind+2:end]) + integral_vcut_cell_part1

        # Integral contribution from the cell containing sigma
        integral_sigma_cell = (0.5 * integral_pieces[sigma_ind] + 0.5 * integral_pieces[sigma_ind+1])

        part2 = sum(@view integral_pieces[sigma_ind+2:plus_vcut_ind-1])
        part2 += integral_vcut_cell_part2 + 0.5 * integral_pieces[sigma_ind+1] + sigma_fraction * integral_sigma_cell
        # part2prime is d(part2)/d(vcut)
        part2prime = -part1prime

        return part1, part1prime, part2, part2prime
    end
    this_a1, this_a1prime, this_a2, this_a2prime = get_for_one_moment(density_integral_pieces_upperz)
    this_b1, this_b1prime, this_b2, _ = get_for_one_moment(flow_integral_pieces_upperz)
    this_c1, this_c1prime, this_c2, this_c2prime = get_for_one_moment(energy_integral_pieces_upperz)
    this_d1, this_d1prime, this_d2, _ = get_for_one_moment(cubic_integral_pieces_upperz)
    this_e1, this_e1prime, this_e2, this_e2prime = get_for_one_moment(quartic_integral_pieces_upperz)

    return get_residual_and_coefficients_for_bc(
               this_a1, this_a1prime, this_a2, this_a2prime, this_b1,
               this_b1prime, this_c1, this_c1prime, this_c2, this_c2prime,
               this_d1, this_d1prime, this_e1, this_e1prime, this_e2,
               this_e2prime, u_over_vt, bc_constraints)...,
           this_a2, this_b2, this_c2, this_d2
end

function get_upperz_integral_correction_components(
             pdf_slice, vthe, vperp, vpa, vpa_unnorm, u_over_vt, sigma_ind,
             sigma_fraction, vcut, minus_vcut_ind, plus_vcut_ind, bc_constraints)

    # Need to recalculate integral pieces with the updated distribution function
    density_integral_pieces_upperz = vpa.scratch3
    flow_integral_pieces_upperz = vpa.scratch4
    energy_integral_pieces_upperz = vpa.scratch5
    cubic_integral_pieces_upperz = vpa.scratch6
    quartic_integral_pieces_upperz = vpa.scratch7
    fill_integral_pieces!(
        pdf_slice, vthe, vpa, vpa_unnorm,
        density_integral_pieces_upperz, flow_integral_pieces_upperz,
        energy_integral_pieces_upperz, cubic_integral_pieces_upperz,
        quartic_integral_pieces_upperz)

    # Update the part2 integrals since we've applied the A and C factors
    _, _, _, _, a2, b2, c2, d2 =
        get_integrals_and_derivatives_upperz(
            vcut, plus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm, u_over_vt,
            density_integral_pieces_upperz, flow_integral_pieces_upperz,
            energy_integral_pieces_upperz, cubic_integral_pieces_upperz,
            quartic_integral_pieces_upperz, bc_constraints)

    function get_part3_for_one_moment_upper(integral_pieces)
        # Integral contribution from the cell containing sigma
        integral_sigma_cell = (0.5 * integral_pieces[sigma_ind] + 0.5 * integral_pieces[sigma_ind+1])

        part3 = sum(@view integral_pieces[minus_vcut_ind-1:sigma_ind-1])
        part3 += 0.5 * integral_pieces[sigma_ind] + (1.0 - sigma_fraction) * integral_sigma_cell

        return part3
    end
    a3 = get_part3_for_one_moment_upper(density_integral_pieces_upperz)
    b3 = get_part3_for_one_moment_upper(flow_integral_pieces_upperz)
    c3 = get_part3_for_one_moment_upper(energy_integral_pieces_upperz)
    d3 = get_part3_for_one_moment_upper(cubic_integral_pieces_upperz)

    # Use scale factor to adjust how sharp the cutoff near vpa_unnorm=0 is.
    if vperp.n == 1
        # When T_⟂=0, for consistency with original version of code, divide
        # integral_correction_sharpness by 3 when smoothing the cutoff to account for the
        # difference between T and T_∥ in the 1V case.
        correction0_integral_pieces = @. vpa.scratch3 = pdf_slice * vpa.wgts * integral_correction_sharpness / 3.0 * vpa_unnorm^2 / vthe^2 / (1.0 + integral_correction_sharpness / 3.0 * vpa_unnorm^2 / vthe^2)
    else
        correction0_integral_pieces = @. vpa.scratch3 = pdf_slice * vpa.wgts * integral_correction_sharpness * vpa_unnorm^2 / vthe^2 / (1.0 + integral_correction_sharpness * vpa_unnorm^2 / vthe^2)
    end
    for ivpa ∈ sigma_ind:vpa.n
        # We only add the corrections to 'part3', so zero them out for positive v_∥.
        # I think this is only actually significant for `sigma_ind` and
        # `sigma_ind+1`. Even though `sigma_ind` is part of the distribution
        # function that we are correcting, for v_∥<0, it affects the integral in
        # the 'sigma_cell' between `sigma_ind` and `sigma_ind+1`, which would
        # affect the numerically calculated integrals for f(v_∥>0), so if we
        # 'corrected' its value, those integrals would change and the constraints
        # would not be exactly satisfied. The difference should be small, as the
        # correction at that point is multiplied by
        # v_∥^2/vth^2/(1+v_∥^2/vth^2)≈v_∥^2/vth^2≈0.
        correction0_integral_pieces[ivpa] = 0.0
    end
    correction1_integral_pieces = @. vpa.scratch4 = correction0_integral_pieces * vpa_unnorm / vthe
    correction2_integral_pieces = @. vpa.scratch5 = correction1_integral_pieces * vpa_unnorm / vthe
    correction3_integral_pieces = @. vpa.scratch6 = correction2_integral_pieces * vpa_unnorm / vthe
    correction4_integral_pieces = @. vpa.scratch7 = correction3_integral_pieces * vpa_unnorm / vthe
    correction5_integral_pieces = @. vpa.scratch8 = correction4_integral_pieces * vpa_unnorm / vthe
    correction6_integral_pieces = @. vpa.scratch9 = correction5_integral_pieces * vpa_unnorm / vthe

    alpha = get_part3_for_one_moment_upper(correction0_integral_pieces)
    beta = get_part3_for_one_moment_upper(correction1_integral_pieces)
    gamma = get_part3_for_one_moment_upper(correction2_integral_pieces)
    delta = get_part3_for_one_moment_upper(correction3_integral_pieces)
    epsilon = get_part3_for_one_moment_upper(correction4_integral_pieces)
    zeta = get_part3_for_one_moment_upper(correction5_integral_pieces)
    eta = get_part3_for_one_moment_upper(correction6_integral_pieces)

    return a2, b2, c2, d2, a3, b3, c3, d3, alpha, beta, gamma, delta, epsilon, zeta, eta
end

@timeit global_timer enforce_boundary_condition_on_electron_pdf!(
                         pdf, phi, vthe, upar, z, vperp, vpa, vperp_spectral,
                         vpa_spectral, vpa_adv, moments, vpa_diffusion, me_over_mi, ir;
                         bc_constraints=true, update_vcut=true, lowerz_vcut_ind=nothing,
                         upperz_vcut_ind=nothing, allow_failure=true) = begin

    @boundscheck bc_constraints && !update_vcut && error("update_vcut is not used when bc_constraints=true, but update_vcut has non-default value")

    newton_tol = 1.0e-13

    # Enforce velocity-space boundary conditions
    if vpa.n > 1
        @begin_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            # enforce the vpa BC
            # use that adv.speed independent of vpa
            @views enforce_v_boundary_condition_local!(pdf[:,ivperp,iz], vpa.bc,
                                                       vpa_adv[1].speed[:,ivperp,iz,ir],
                                                       vpa_diffusion, vpa, vpa_spectral)
        end
    end
    if vperp.n > 1
        @begin_z_vpa_region()
        enforce_vperp_boundary_condition!(pdf, vperp.bc, vperp, vperp_spectral, ir)
    end

    if z.periodic
        # Nothing more to do for z-periodic boundary conditions
        return true
    elseif z.bc == "constant"
        @begin_r_vperp_vpa_region()
        density_offset = 1.0
        vwidth = 1.0/sqrt(me_over_mi)
        dens = moments.electron.dens
        if z.irank == 0
            @loop_vperp_vpa ivperp ivpa begin
                u = moments.electron.upar[1]
                vthe = moments.electron.vth[1]
                speed = vpa.grid[ivpa] * vthe + u
                if speed > 0.0
                    pdf[ivpa,ivperp,1] = density_offset / dens[1] * vthe[1] * exp(-(speed^2 + vperp.grid[ivperp]^2)/vwidth^2)
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_vperp_vpa ivperp ivpa begin
                u = moments.electron.upar[end]
                vthe = moments.electron.vth[end]
                speed = vpa.grid[ivpa] * vthe + u
                if speed > 0.0
                    pdf[ivpa,ivperp,end] = density_offset / dens[end] * vthe[end] * exp(-(speed^2 + vperp.grid[ivperp]^2)/vwidth^2)
                end
            end
        end
        return true
    end

    # first enforce the boundary condition at z_min.
    # this involves forcing the pdf to be zero for electrons travelling faster than the max speed
    # they could attain by accelerating in the electric field between the wall and the simulation boundary;
    # for electrons with positive velocities less than this critical value, they must have the same
    # pdf as electrons with negative velocities of the same magnitude.
    # the electrostatic potential at the boundary, which determines the critical speed, is unknown a priori;
    # use the constraint that the first moment of the normalised pdf be zero to choose the potential.

    @begin_serial_region()

    newton_max_its = 100
    if vperp.n == 1
        this_integral_correction_sharpness = integral_correction_sharpness / 3.0
    else
        this_integral_correction_sharpness = integral_correction_sharpness
    end

    @serial_region begin
        if z.irank == 0
            if z.bc != "wall"
                error("Options other than wall, constant or z-periodic bc not implemented yet for electrons")
            end

            # Impose sheath-edge boundary condition, while also imposing moment
            # constraints and determining the cut-off velocity (and therefore the sheath
            # potential).

            vpa_unnorm, u_over_vt, vcut, minus_vcut_ind, sigma, sigma_ind, sigma_fraction,
                element_with_zero, element_with_zero_boundary, last_point_near_zero,
                reversed_wpa_of_minus_vpa =
                    get_cutoff_params_lower(upar[1,ir], vthe[1,ir], phi[1,ir], me_over_mi,
                                            vpa)

            # interpolate the pdf onto this grid
            # 'near zero' means in the range where
            # abs(v_∥)≤abs(lower boundary of element including v_∥=0)
            # 'far from zero' means larger values of v_∥.

            # Interpolate to the 'near zero' points
            @views interpolate_symmetric!(pdf[sigma_ind:last_point_near_zero,1,1],
                                          vpa_unnorm[sigma_ind:last_point_near_zero],
                                          pdf[element_with_zero_boundary:sigma_ind-1,1,1],
                                          vpa_unnorm[element_with_zero_boundary:sigma_ind-1])

            # Interpolate to the 'far from zero' points
            reversed_pdf_far_from_zero = @view vpa.scratch[last_point_near_zero+1:end]
            @views interpolate_to_grid_1d!(reversed_pdf_far_from_zero,
                                           reversed_wpa_of_minus_vpa[1:vpa.n-last_point_near_zero],
                                           pdf[:,1,1], vpa, vpa_spectral)
            reverse!(reversed_pdf_far_from_zero)
            pdf[last_point_near_zero+1:end,1,1] .= reversed_pdf_far_from_zero

            # Per-grid-point contributions to moment integrals
            density_integral_pieces_lowerz = vpa.scratch3
            flow_integral_pieces_lowerz = vpa.scratch4
            energy_integral_pieces_lowerz = vpa.scratch5
            cubic_integral_pieces_lowerz = vpa.scratch6
            quartic_integral_pieces_lowerz = vpa.scratch7
            fill_integral_pieces!(
                @view(pdf[:,1,1]), vthe[1], vpa, vpa_unnorm,
                density_integral_pieces_lowerz, flow_integral_pieces_lowerz,
                energy_integral_pieces_lowerz, cubic_integral_pieces_lowerz,
                quartic_integral_pieces_lowerz)

            counter = 1
            A = 1.0
            C = 0.0
            # Always do at least one update of vcut
            epsilon, epsilonprime, A, C, a2, b2, c2, d2 =
                get_integrals_and_derivatives_lowerz(
                    vcut, minus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm,
                    u_over_vt, density_integral_pieces_lowerz,
                    flow_integral_pieces_lowerz, energy_integral_pieces_lowerz,
                    cubic_integral_pieces_lowerz, quartic_integral_pieces_lowerz,
                    bc_constraints)
            if bc_constraints
                while true
                    # Newton iteration update. Note that primes denote derivatives with
                    # respect to vcut
                    delta_v = - epsilon / epsilonprime

                    if vcut > vthe[1] && epsilonprime < 0.0
                        # epsilon should be increasing with vcut at epsilon=0, so if
                        # epsilonprime is negative, the solution is actually at a lower vcut -
                        # at larger vcut, epsilon will just tend to 0 but never reach it.
                        if vperp.n == 1
                            delta_v = -0.1 * vthe[1] * sqrt(3.0)
                        else
                            delta_v = -0.1 * vthe[1]
                        end
                    end

                    # Prevent the step size from getting too big, to make Newton iteration
                    # more robust.
                    if vperp.n == 1
                        delta_v = min(delta_v, 0.1 * vthe[1] * sqrt(3.0))
                        delta_v = max(delta_v, -0.1 * vthe[1] * sqrt(3.0))
                    else
                        delta_v = min(delta_v, 0.1 * vthe[1])
                        delta_v = max(delta_v, -0.1 * vthe[1])
                    end

                    vcut = vcut + delta_v
                    minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)

                    epsilon, epsilonprime, A, C, a2, b2, c2, d2 =
                        get_integrals_and_derivatives_lowerz(
                            vcut, minus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm,
                            u_over_vt, density_integral_pieces_lowerz,
                            flow_integral_pieces_lowerz, energy_integral_pieces_lowerz,
                            cubic_integral_pieces_lowerz, quartic_integral_pieces_lowerz,
                            bc_constraints)

                    if abs(epsilon) < newton_tol
                        break
                    end

                    if counter ≥ newton_max_its
                        # Newton iteration for electron lower-z boundary failed to
                        # converge after `counter` iterations
                        if allow_failure
                            return false
                        else
                            error("Newton iteration for electron lower-z boundary failed "
                                  * "to converge after $counter iterations")
                        end
                    end
                    counter += 1
                end
            elseif update_vcut
                # When bc_constraints=false, no constraints are applied in
                # get_integrals_and_derivatives_lowerz(), so updating vcut is usually just
                # solving a linear equation, not doing a Newton iteration. The exception
                # is if minus_vcut_ind changes, in which case we have to re-do the update.
                while true
                    vcut = vcut - epsilon / epsilonprime
                    minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)

                    vcut_fraction = get_minus_vcut_fraction(vcut, minus_vcut_ind, vpa_unnorm)

                    if 0.0 ≤ vcut_fraction ≤ 1.0
                        break
                    end

                    epsilon, epsilonprime, _, _, _, _, _, _ =
                        get_integrals_and_derivatives_lowerz(
                            vcut, minus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm,
                            u_over_vt, density_integral_pieces_lowerz,
                            flow_integral_pieces_lowerz, energy_integral_pieces_lowerz,
                            cubic_integral_pieces_lowerz, quartic_integral_pieces_lowerz,
                            bc_constraints)
                end
            end

            # Adjust pdf so that after reflecting and cutting off tail, it will obey the
            # constraints.
            @. pdf[:,1,1] *= A + C * vpa_unnorm^2 / vthe[1]^2

            plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)
            pdf[plus_vcut_ind+2:end,1,1] .= 0.0
            # vcut_fraction is the fraction of the distance between plus_vcut_ind and
            # plus_vcut_ind+1 where vcut is.
            vcut_fraction = get_plus_vcut_fraction(vcut, plus_vcut_ind, vpa_unnorm)
            if vcut_fraction > 0.5
                if lowerz_vcut_ind !== nothing
                    lowerz_vcut_ind[] = plus_vcut_ind+1
                end
                pdf[plus_vcut_ind+1,1,1] *= vcut_fraction - 0.5
            else
                if lowerz_vcut_ind !== nothing
                    lowerz_vcut_ind[] = plus_vcut_ind
                end
                pdf[plus_vcut_ind+1,1,1] = 0.0
                pdf[plus_vcut_ind,1,1] *= vcut_fraction + 0.5
            end

            # update the electrostatic potential at the boundary to be the value corresponding to the updated cutoff velocity
            phi[1] = 0.5 * me_over_mi * vcut^2

            moments.electron.constraints_A_coefficient[1,ir] = A
            moments.electron.constraints_B_coefficient[1,ir] = 0.0
            moments.electron.constraints_C_coefficient[1,ir] = C

            # Ensure the part of f for 0≤v_∥≤vcut has its first 3 moments symmetric with
            # vcut≤v_∥≤0 (i.e. even moments are the same, odd moments are equal but opposite
            # sign). This should be true analytically because of the definition of the
            # boundary condition, but would not be numerically true because of the
            # interpolation.

            a2, b2, c2, d2, a3, b3, c3, d3, alpha, beta, gamma, delta, epsilon, zeta, eta =
                get_lowerz_integral_correction_components(
                    @view(pdf[:,1,1]), vthe[1], vperp, vpa, vpa_unnorm, u_over_vt,
                    sigma_ind, sigma_fraction, vcut, minus_vcut_ind, plus_vcut_ind,
                    bc_constraints)

            # Update the v_∥>0 part of f to correct the moments as
            # f(0<v_∥<vcut) = (1 + (A + B*v/vth + C*v^2/vth^2 + D*v^3/vth^3) * v^2/vth^2 / (1 + v^2/vth^2)) * fhat(0<v_∥<vcut)
            # Constraints:
            # ∫dv_∥ v_∥^n f(0<v_∥<vcut) = (-1)^n ∫dv_∥ v_∥^n f(-vcut<v_∥<0)    for n=0,1,2,3
            # a2 = a3 + alpha*A + beta*B + gamma*C + delta*D
            # -b2 = b3 + beta*A + gamma*B + delta*C + epsilon*D
            # c2 = c3 + gamma*A + delta*B + epsilon*C + zeta*D
            # -d2 = d3 + delta*A + epsilon*B + zeta*C + eta*D
            solution = [alpha beta    gamma   delta   ;
                        beta  gamma   delta   epsilon ;
                        gamma delta   epsilon zeta    ;
                        delta epsilon zeta    eta
                       ] \ [a2-a3, -b2-b3, c2-c3, -d2-d3]
            A, B, C, D = solution
            for ivpa ∈ sigma_ind+1:plus_vcut_ind+1
                v_over_vth = vpa_unnorm[ivpa]/vthe[1]
                pdf[ivpa,1,1] = pdf[ivpa,1,1] +
                                   (A
                                    + B * v_over_vth
                                    + C * v_over_vth^2
                                    + D * v_over_vth^3) *
                                   this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2) *
                                   pdf[ivpa,1,1]
            end
        end

        # next enforce the boundary condition at z_max.
        # this involves forcing the pdf to be zero for electrons travelling faster than the max speed
        # they could attain by accelerating in the electric field between the wall and the simulation boundary;
        # for electrons with negative velocities less than this critical value, they must have the same
        # pdf as electrons with positive velocities of the same magnitude.
        # the electrostatic potential at the boundary, which determines the critical speed, is unknown a priori;
        # use the constraint that the first moment of the normalised pdf be zero to choose the potential.
        
        if z.irank == z.nrank - 1
            if z.bc != "wall"
                error("Options other than wall or z-periodic bc not implemented yet for electrons")
            end

            # Impose sheath-edge boundary condition, while also imposing moment
            # constraints and determining the cut-off velocity (and therefore the sheath
            # potential).

            vpa_unnorm, u_over_vt, vcut, plus_vcut_ind, sigma, sigma_ind, sigma_fraction,
                element_with_zero, element_with_zero_boundary, first_point_near_zero,
                reversed_wpa_of_minus_vpa =
                    get_cutoff_params_upper(upar[end,ir], vthe[end,ir], phi[end,ir],
                                            me_over_mi, vpa)

            # interpolate the pdf onto this grid
            # 'near zero' means in the range where
            # abs(v_∥)≤abs(upper boundary of element including v_∥=0)
            # 'far from zero' means more negative values of v_∥.

            # Interpolate to the 'near zero' points
            @views interpolate_symmetric!(pdf[first_point_near_zero:sigma_ind,1,end],
                                          vpa_unnorm[first_point_near_zero:sigma_ind],
                                          pdf[sigma_ind+1:element_with_zero_boundary,1,end],
                                          vpa_unnorm[sigma_ind+1:element_with_zero_boundary])

            # Interpolate to the 'far from zero' points
            reversed_pdf = @view vpa.scratch[1:first_point_near_zero-1]
            @views interpolate_to_grid_1d!(reversed_pdf,
                                           reversed_wpa_of_minus_vpa[vpa.n-first_point_near_zero+2:end],
                                           pdf[:,1,end], vpa, vpa_spectral)
            reverse!(reversed_pdf)
            pdf[1:first_point_near_zero-1,1,end] .= reversed_pdf

            # Per-grid-point contributions to moment integrals
            density_integral_pieces_upperz = vpa.scratch3
            flow_integral_pieces_upperz = vpa.scratch4
            energy_integral_pieces_upperz = vpa.scratch5
            cubic_integral_pieces_upperz = vpa.scratch6
            quartic_integral_pieces_upperz = vpa.scratch7
            fill_integral_pieces!(
                @view(pdf[:,1,end]), vthe[end], vpa, vpa_unnorm,
                density_integral_pieces_upperz, flow_integral_pieces_upperz,
                energy_integral_pieces_upperz, cubic_integral_pieces_upperz,
                quartic_integral_pieces_upperz)

            counter = 1
            # Always do at least one update of vcut
            epsilon, epsilonprime, A, C, a2, b2, c2, d2 =
                get_integrals_and_derivatives_upperz(
                    vcut, plus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm, u_over_vt,
                    density_integral_pieces_upperz, flow_integral_pieces_upperz,
                    energy_integral_pieces_upperz, cubic_integral_pieces_upperz,
                    quartic_integral_pieces_upperz, bc_constraints)
            if bc_constraints
                while true
                    # Newton iteration update. Note that primes denote derivatives with
                    # respect to vcut
                    delta_v = - epsilon / epsilonprime

                    if vcut > vthe[1] && epsilonprime > 0.0
                        # epsilon should be decreasing with vcut at epsilon=0, so if
                        # epsilonprime is positive, the solution is actually at a lower vcut -
                        # at larger vcut, epsilon will just tend to 0 but never reach it.
                        if vperp.n == 1
                            delta_v = -0.1 * vthe[1] * sqrt(3.0)
                        else
                            delta_v = -0.1 * vthe[1]
                        end
                    end

                    # Prevent the step size from getting too big, to make Newton iteration
                    # more robust.
                    if vperp.n == 1
                        delta_v = min(delta_v, 0.1 * vthe[end] * sqrt(3.0))
                        delta_v = max(delta_v, -0.1 * vthe[end] * sqrt(3.0))
                    else
                        delta_v = min(delta_v, 0.1 * vthe[end])
                        delta_v = max(delta_v, -0.1 * vthe[end])
                    end

                    vcut = vcut + delta_v
                    plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)

                    epsilon, epsilonprime, A, C, a2, b2, c2, d2 =
                        get_integrals_and_derivatives_upperz(
                            vcut, plus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm,
                            u_over_vt, density_integral_pieces_upperz,
                            flow_integral_pieces_upperz, energy_integral_pieces_upperz,
                            cubic_integral_pieces_upperz, quartic_integral_pieces_upperz,
                            bc_constraints)

                    if abs(epsilon) < newton_tol
                        break
                    end

                    if counter ≥ newton_max_its
                        # Newton iteration for electron upper-z boundary failed to
                        # converge after `counter` iterations
                        if allow_failure
                            return false
                        else
                            error("Newton iteration for electron upper-z boundary failed "
                                  * "to converge after $counter iterations")
                        end
                    end
                    counter += 1
                end
            elseif update_vcut
                # When bc_constraints=false, no constraints are applied in
                # get_integrals_and_derivatives_upperz(), so updating vcut is usually just
                # solving a linear equation, not doing a Newton iteration. The exception
                # is if minus_vcut_ind changes, in which case we have to re-do the update.
                while true
                    vcut = vcut - epsilon / epsilonprime
                    plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)

                    vcut_fraction = get_plus_vcut_fraction(vcut, plus_vcut_ind, vpa_unnorm)

                    if 0.0 ≤ vcut_fraction ≤ 1.0
                        break
                    end

                    epsilon, epsilonprime, _, _, _, _, _, _ =
                        get_integrals_and_derivatives_upperz(
                            vcut, plus_vcut_ind, sigma_ind, sigma_fraction, vpa_unnorm,
                            u_over_vt, density_integral_pieces_upperz,
                            flow_integral_pieces_upperz, energy_integral_pieces_upperz,
                            cubic_integral_pieces_upperz, quartic_integral_pieces_upperz,
                            bc_constraints)
                end
            end

            # Adjust pdf so that after reflecting and cutting off tail, it will obey the
            # constraints.
            @. pdf[:,1,end] *= A + C * vpa_unnorm^2 / vthe[end]^2

            minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)
            pdf[1:minus_vcut_ind-2,1,end] .= 0.0
            # vcut_fraction is the fraction of the distance between minus_vcut_ind-1 and
            # minus_vcut_ind where -vcut is.
            vcut_fraction = get_minus_vcut_fraction(vcut, minus_vcut_ind, vpa_unnorm)
            if vcut_fraction < 0.5
                if upperz_vcut_ind !== nothing
                    upperz_vcut_ind[] = minus_vcut_ind-1
                end
                pdf[minus_vcut_ind-1,1,end] *= 0.5 - vcut_fraction
            else
                if upperz_vcut_ind !== nothing
                    upperz_vcut_ind[] = minus_vcut_ind
                end
                pdf[minus_vcut_ind-1,1,end] = 0.0
                pdf[minus_vcut_ind,1,end] *= 1.5 - vcut_fraction
            end

            # update the electrostatic potential at the boundary to be the value corresponding to the updated cutoff velocity
            phi[end] = 0.5 * me_over_mi * vcut^2

            moments.electron.constraints_A_coefficient[end,ir] = A
            moments.electron.constraints_B_coefficient[end,ir] = 0.0
            moments.electron.constraints_C_coefficient[end,ir] = C

            # Ensure the part of f for -vcut≤v_∥≤0 has its first 3 moments symmetric with
            # 0≤v_∥≤vcut  (i.e. even moments are the same, odd moments are equal but opposite
            # sign). This should be true analytically because of the definition of the
            # boundary condition, but would not be numerically true because of the
            # interpolation.

            a2, b2, c2, d2, a3, b3, c3, d3, alpha, beta, gamma, delta, epsilon, zeta, eta =
                get_upperz_integral_correction_components(
                    @view(pdf[:,1,end]), vthe[end], vperp, vpa, vpa_unnorm, u_over_vt,
                    sigma_ind, sigma_fraction, vcut, minus_vcut_ind, plus_vcut_ind,
                    bc_constraints)

            # Update the v_∥>0 part of f to correct the moments as
            # f(0<v_∥<vcut) = (1 + (A + B*v/vth + C*v^2/vth^2 + D*v^3/vth^3) * v^2/vth^2 / (1 + v^2/vth^2)) * fhat(0<v_∥<vcut)
            # Constraints:
            # ∫dv_∥ v_∥^n f(0<v_∥<vcut) = (-1)^n ∫dv_∥ v_∥^n f(-vcut<v_∥<0)    for n=0,1,2,3
            # a2 = a3 + alpha*A + beta*B + gamma*C + delta*D
            # -b2 = b3 + beta*A + gamma*B + delta*C + epsilon*D
            # c2 = c3 + gamma*A + delta*B + epsilon*C + zeta*D
            # -d2 = d3 + delta*A + epsilon*B + zeta*C + eta*D
            solution = [alpha beta    gamma   delta   ;
                        beta  gamma   delta   epsilon ;
                        gamma delta   epsilon zeta    ;
                        delta epsilon zeta    eta
                       ] \ [a2-a3, -b2-b3, c2-c3, -d2-d3]
            A, B, C, D = solution
            for ivpa ∈ minus_vcut_ind-1:sigma_ind-1
                v_over_vth = vpa_unnorm[ivpa]/vthe[end]
                pdf[ivpa,1,end] = pdf[ivpa,1,end] +
                                  (A
                                   + B * v_over_vth
                                   + C * v_over_vth^2
                                   + D * v_over_vth^3) *
                                  this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2) *
                                  pdf[ivpa,1,end]
            end
        end
    end

    return true
end

# In several places it is useful to zero out (in residuals, etc.) the points that would be
# set by the z boundary condition.
function zero_z_boundary_condition_points(residual, z, vpa, moments, ir)
    if z.bc ∈ ("wall", "constant",) && (z.irank == 0 || z.irank == z.nrank - 1)
        # Boundary conditions on incoming part of distribution function. Note
        # that as density, upar, p do not change in this implicit step, f_electron_newvar,
        # f_old, and residual should all be zero at exactly the same set of grid points,
        # so it is reasonable to zero-out `residual` to impose the boundary condition. We
        # impose this after subtracting f_old in case rounding errors, etc. mean that at
        # some point f_old had a different boundary condition cut-off index.
        @begin_vperp_vpa_region()
        v_unnorm = vpa.scratch
        zero = 1.0e-14
        if z.irank == 0
            v_unnorm .= vpagrid_to_vpa(vpa.grid, moments.electron.vth[1,ir],
                                       moments.electron.upar[1,ir], true, true)
            @loop_vperp_vpa ivperp ivpa begin
                if v_unnorm[ivpa] > -zero
                    residual[ivpa,ivperp,1] = 0.0
                end
            end
        end
        if z.irank == z.nrank - 1
            v_unnorm .= vpagrid_to_vpa(vpa.grid, moments.electron.vth[end,ir],
                                       moments.electron.upar[end,ir], true, true)
            @loop_vperp_vpa ivperp ivpa begin
                if v_unnorm[ivpa] < zero
                    residual[ivpa,ivperp,end] = 0.0
                end
            end
        end
    end
    return nothing
end

"""
    add_wall_boundary_condition_to_Jacobian!(jacobian, phi, pdf, p, vthe, upar, z, vperp,
                                             vpa, vperp_spectral, vpa_spectral, vpa_adv,
                                             moments, vpa_diffusion, me_over_mi, ir)

All the contributions that we add in this function have to be added with a -'ve sign so
that they combine with the 1 on the diagonal of the preconditioner matrix to make rows
corresponding to the boundary points which define constraint equations imposing the
boundary condition on those entries of δg (when the right-hand-side is set to zero).
"""
@timeit global_timer add_wall_boundary_condition_to_Jacobian!(
                         jacobian, phi, pdf, p, vthe, upar, z, vperp, vpa, vperp_spectral,
                         vpa_spectral, vpa_adv, moments, vpa_diffusion, me_over_mi, ir,
                         include, iz=nothing; pdf_offset=0, p_offset=0) = begin
    if z.bc != "wall"
        return nothing
    end

    if include ∈ (:all, :explicit_v)
        include_lower = (z.irank == 0)
        include_upper = (z.irank == z.nrank - 1)
        zend = z.n
        phi_lower = phi[1]
        phi_upper = phi[end]
        pdf_lower = @view pdf[:,:,1]
        pdf_upper = @view pdf[:,:,end]
        p_lower = p[1]
        p_upper = p[end]
        vthe_lower = vthe[1]
        vthe_upper = vthe[end]
        upar_lower = upar[1]
        upar_upper = upar[end]
        shared_mem_parallel = true
    elseif include === :implicit_v
        include_lower = (z.irank == 0) && iz == 1
        include_upper = (z.irank == z.nrank - 1) && iz == z.n
        zend = 1
        phi_lower = phi_upper = phi
        pdf_lower = pdf_upper = pdf
        p_lower = p_upper = p
        vthe_lower = vthe_upper = vthe
        upar_lower = upar_upper = upar

        # When using :implicit_v, this function is called inside a loop that is already
        # parallelised over z, so we cannot change the parallel region type.
        shared_mem_parallel = false
    else
        return nothing
    end

    if vperp.n == 1
        this_integral_correction_sharpness = integral_correction_sharpness / 3.0
    else
        this_integral_correction_sharpness = integral_correction_sharpness
    end

    if include_lower
        shared_mem_parallel && @begin_vperp_region()
        @loop_vperp ivperp begin
            # Skip velocity space boundary points.
            if vperp.n > 1 && ivperp == vperp.n
                continue
            end

            # Get matrix entries for the response of the sheath-edge boundary condition.
            # Ignore constraints, as these are non-linear and also should be small
            # corrections which should not matter much for a preconditioner.

            jac_range = pdf_offset+(ivperp-1)*vpa.n+1 : pdf_offset+ivperp*vpa.n
            jacobian_zbegin = @view jacobian[jac_range,jac_range]
            jacobian_zbegin_p = @view jacobian[jac_range,p_offset+1]

            vpa_unnorm, u_over_vt, vcut, minus_vcut_ind, sigma, sigma_ind, sigma_fraction,
                element_with_zero, element_with_zero_boundary, last_point_near_zero,
                reversed_wpa_of_minus_vpa =
                    get_cutoff_params_lower(upar_lower, vthe_lower, phi_lower, me_over_mi,
                                            vpa)

            plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)
            # plus_vcut_fraction is the fraction of the distance between plus_vcut_ind and
            # plus_vcut_ind+1 where vcut is.
            plus_vcut_fraction = get_plus_vcut_fraction(vcut, plus_vcut_ind, vpa_unnorm)
            if plus_vcut_fraction > 0.5
                last_nonzero_ind = plus_vcut_ind + 1
            else
                last_nonzero_ind = plus_vcut_ind
            end

            # Interpolate to the 'near zero' points
            @views fill_interpolate_symmetric_matrix!(
                       jacobian_zbegin[sigma_ind:last_point_near_zero,element_with_zero_boundary:sigma_ind-1],
                       vpa_unnorm[sigma_ind:last_point_near_zero],
                       vpa_unnorm[element_with_zero_boundary:sigma_ind-1])

            # Interpolate to the 'far from zero' points
            @views fill_1d_interpolation_matrix!(
                       jacobian_zbegin[last_nonzero_ind:-1:last_point_near_zero+1,:],
                       reversed_wpa_of_minus_vpa[vpa.n-last_nonzero_ind+1:vpa.n-last_point_near_zero],
                       vpa, vpa_spectral)

            # Reverse the sign of the elements we just filled
            jacobian_zbegin[sigma_ind:last_nonzero_ind,1:sigma_ind-1] .*= -1.0

            if plus_vcut_fraction > 0.5
                jacobian_zbegin[last_nonzero_ind,1:sigma_ind-1] .*= plus_vcut_fraction - 0.5
            else
                jacobian_zbegin[last_nonzero_ind,1:sigma_ind-1] .*= plus_vcut_fraction + 0.5
            end

            # Fill in elements giving response to changes in electron_p
            # A change to p results in a shift in the location of w_∥(v_∥=0).
            # The interpolated values of g_e(w_∥) that are filled by the boundary
            # condition are (dropping _∥ subscripts for the remaining comments in this
            # if-clause)
            #   g(w|v>0) = g(2*u/vth - w)
            # So
            #   δg(w|v>0) = dg/dw(2*u/vth - w) * (-2*u/vth^2) * δvth
            #             = dg/dw(2*u/vth - w) * (-2*u/vth^2) * δ(sqrt(2*p/n)
            #             = dg/dw(2*u/vth - w) * (-2*u/vth^2) * δp * vth / (2*p)
            #             = -u/vth/p * dg/dw(2*u/vth - w) * δp
            #
            # As jacobian_zbegin_p only depends on variations of a single value `p[1]`, it
            # might be more maintainable and computationally cheaper to calculate
            # jacobian_zbegin_p with a finite difference method (applying the boundary
            # condition function with a perturbed pressure `p[1] + ϵ` for some small ϵ)
            # rather than calculating it using the method below. If we used the finite
            # difference approach, we would have to be careful that the ϵ step does not
            # change the cutoff index (otherwise we would pick up non-smooth changes) -
            # one possibility might to be to switch to `-ϵ` instead of `+ϵ` if this
            # happens.

            dpdfdv_near_zero = @view vpa.scratch[sigma_ind:last_point_near_zero]
            @views interpolate_symmetric!(dpdfdv_near_zero,
                                          vpa_unnorm[sigma_ind:last_point_near_zero],
                                          pdf_lower[element_with_zero_boundary:sigma_ind-1,ivperp],
                                          vpa_unnorm[element_with_zero_boundary:sigma_ind-1],
                                          Val(1))
            # The above call to interpolate_symmetric calculates dg/dv rather than dg/dw,
            # so need to multiply by an extra factor of vthe.
            #   δg(w|v>0) = -u/p * dg/dv(2*u/vth - w) * δp
            @. jacobian_zbegin_p[sigma_ind:last_point_near_zero] -=
                   -upar_lower / p_lower * dpdfdv_near_zero

            dpdfdw_far_from_zero = @view vpa.scratch[last_point_near_zero+1:last_nonzero_ind]
            @views interpolate_to_grid_1d!(dpdfdw_far_from_zero,
                                           reversed_wpa_of_minus_vpa[vpa.n-last_nonzero_ind+1:vpa.n-last_point_near_zero],
                                           pdf_lower[:,ivperp], vpa, vpa_spectral, Val(1))
            reverse!(dpdfdw_far_from_zero)
            # Note that because we calculated the derivative of the interpolating
            # function, and then reversed the results, we need to multiply the derivative
            # by -1.
            @. jacobian_zbegin_p[last_point_near_zero+1:last_nonzero_ind] -=
                   upar_lower / vthe_lower / p_lower * dpdfdw_far_from_zero

            # Whatever the variation due to interpolation is at the last nonzero grid
            # point, it will be reduced by the cutoff.
            if plus_vcut_fraction > 0.5
                jacobian_zbegin_p[last_nonzero_ind] *= plus_vcut_fraction - 0.5
            else
                jacobian_zbegin_p[last_nonzero_ind] *= plus_vcut_fraction + 0.5
            end

            # The change in electron_p also changes the position of
            #     wcut = (vcut - upar)/vthe
            # as vcut does not change (within the Krylov iteration where this
            # preconditioner matrix is used), but vthe does because of the change in
            # electron_p. We actually use plus_vcut_fraction calculated from vpa_unnorm,
            # so it is most convenient to consider:
            #     v = vthe * w + upar
            #     δv = δvthe * w
            #        = δvthe * (v - upar)/vthe
            #        = δp * vthe / (2*p) * (v - upar)/vthe
            #        = δp * (v - upar) / 2 / p
            # with vl and vu the values of v at the grid points below and above vcut
            #     plus_vcut_fraction = (vcut - vl) / (vu - vl)
            #     δplus_vcut_fraction = -(vcut - vl) / (vu - vl)^2 * (δvu - δvl) - δvl / (vu - vl)
            #     δplus_vcut_fraction = [-(vcut - vl) / (vu - vl)^2 * (vu - vl) - (vl - upar) / (vu - vl)] * δp / 2 / p
            #     δplus_vcut_fraction = [-(vcut - vl) / (vu - vl) - (vl - upar) / (vu - vl)] * δp / 2 / p
            #     δplus_vcut_fraction = -(vcut - upar) / (vu - vl) / 2 / p * δp
            interpolated_pdf_at_last_nonzero_ind = @view vpa.scratch[1:1]
            reversed_last_nonzero_ind = vpa.n-last_nonzero_ind+1
            @views interpolate_to_grid_1d!(interpolated_pdf_at_last_nonzero_ind,
                                           reversed_wpa_of_minus_vpa[reversed_last_nonzero_ind:reversed_last_nonzero_ind],
                                           pdf_lower[:,ivperp], vpa, vpa_spectral)

            dplus_vcut_fraction_dp = -(vcut - upar_lower) / (vpa_unnorm[plus_vcut_ind+1] - vpa_unnorm[plus_vcut_ind]) / 2.0 / p_lower
            jacobian_zbegin_p[last_nonzero_ind] -= interpolated_pdf_at_last_nonzero_ind[] * dplus_vcut_fraction_dp

            # Calculate some numerical integrals of dpdfdw that we will need later
            function get_part3_for_one_moment_lower(integral_pieces)
                # Integral contribution from the cell containing sigma
                # The contribution from integral_pieces[sigma_ind-1] should be dropped,
                # because that point is on the input grid, and this correction is due to
                # the changes in interpolated values on the output grid due to δp_e∥. The
                # value of g_e at sigma_ind-1 isn't interpolated so does not change in
                # this way.
                integral_sigma_cell = 0.5 * integral_pieces[sigma_ind]

                part3 = sum(@view integral_pieces[sigma_ind+1:plus_vcut_ind+1])
                part3 += 0.5 * integral_pieces[sigma_ind] + (1.0 - sigma_fraction) * integral_sigma_cell

                return part3
            end
            # The contents of jacobian_zbegin_p at this point are the coefficients needed
            # to get the new distribution function due to a change δp, which we can now
            # use to calculate the response of various integrals to the same change.  Note
            # that jacobian_zbegin_p already contains `2.0*dsigma_dp`.
            @. vpa.scratch = -jacobian_zbegin_p * vpa.wgts
            da3_dp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            db3_dp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            dc3_dp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            dd3_dp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch = -jacobian_zbegin_p * vpa.wgts * this_integral_correction_sharpness * vpa_unnorm^2 / vthe_lower^2 / (1.0 + this_integral_correction_sharpness * vpa_unnorm^2 / vthe_lower^2)
            vpa.scratch[sigma_ind-1:sigma_ind] .= 0.0
            dalpha_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            dbeta_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            dgamma_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            ddelta_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            depsilon_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            dzeta_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_lower
            deta_dp_interp = get_part3_for_one_moment_lower(vpa.scratch)

            pdf_lowerz = vpa.scratch10
            @views @. pdf_lowerz[1:sigma_ind-1] = pdf_lower[1:sigma_ind-1,ivperp]

            # interpolate the pdf_lowerz onto this grid
            # 'near zero' means in the range where
            # abs(v_∥)≤abs(lower boundary of element including v_∥=0)
            # 'far from zero' means larger values of v_∥.

            # Interpolate to the 'near zero' points
            @views interpolate_symmetric!(pdf_lowerz[sigma_ind:last_point_near_zero],
                                          vpa_unnorm[sigma_ind:last_point_near_zero],
                                          pdf_lowerz[element_with_zero_boundary:sigma_ind-1],
                                          vpa_unnorm[element_with_zero_boundary:sigma_ind-1])

            # Interpolate to the 'far from zero' points
            reversed_pdf_far_from_zero = @view pdf_lowerz[last_point_near_zero+1:end]
            interpolate_to_grid_1d!(reversed_pdf_far_from_zero,
                                    reversed_wpa_of_minus_vpa[1:vpa.n-last_point_near_zero],
                                    pdf_lowerz, vpa, vpa_spectral)
            reverse!(reversed_pdf_far_from_zero)

            minus_vcut_fraction = get_minus_vcut_fraction(vcut, minus_vcut_ind, vpa_unnorm)
            if minus_vcut_fraction < 0.5
                lower_cutoff_ind = minus_vcut_ind-1
            else
                lower_cutoff_ind = minus_vcut_ind
            end
            if plus_vcut_fraction > 0.5
                upper_cutoff_ind = plus_vcut_ind+1
                upper_cutoff_factor = plus_vcut_fraction - 0.5
            else
                upper_cutoff_ind = plus_vcut_ind
                upper_cutoff_factor = plus_vcut_fraction + 0.5
            end
            pdf_lowerz[upper_cutoff_ind] *= upper_cutoff_factor
            pdf_lowerz[upper_cutoff_ind+1:end] .= 0.0

            a2, b2, c2, d2, a3, b3, c3, d3, alpha, beta, gamma, delta, epsilon, zeta, eta =
                get_lowerz_integral_correction_components(
                    pdf_lowerz, vthe_lower, vperp, vpa, vpa_unnorm, u_over_vt, sigma_ind,
                    sigma_fraction, vcut, minus_vcut_ind, plus_vcut_ind, false)

            output_range = sigma_ind+1:upper_cutoff_ind

            v_over_vth = @views @. vpa.scratch[output_range] = vpa_unnorm[output_range] / vthe_lower

            correction_matrix = [alpha beta    gamma   delta   ;
                                 beta  gamma   delta   epsilon ;
                                 gamma delta   epsilon zeta    ;
                                 delta epsilon zeta    eta
                                ]

            # jacobian_zbegin_p state at this point would generate (when multiplied by
            # some δp) a δf due to the effectively-shifted grid. The unperturbed f
            # required corrections to make its integrals correct. Need to apply the same
            # corrections to δf.
            # This term seems to have a very small contribution, and could probably be
            # skipped.
            A, B, C, D = correction_matrix \ [a2-a3, -b2-b3, c2-c3, -d2-d3]
            @. jacobian_zbegin_p[output_range] *=
                   1.0 +
                   (A
                    + B * v_over_vth
                    + C * v_over_vth^2
                    + D * v_over_vth^3) *
                   this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2)

            # Calculate the changes in the integrals a2, b2, c2, d2, a3, b3, c3, and d3 in
            # response to changes in electron_p. The changes are calculated for the
            # combinations (a2-a3), (-b2-b3), (c2-c3), and (-d2-d3) to take advantage of
            # some cancellations.

            # Need to calculate the variation of various intermediate quantities with
            # δp.
            #
            #   vth = sqrt(2*p/n)
            #   δvth = vth / (2 * p) * δp
            #
            #   sigma = -u / vth
            #   δsigma = u / vth^2 δvth
            #          = u / (2 * vth * p) * δp
            #
            # We could write sigma_fraction as
            #   sigma_fraction = (sigma - vpa[sigma_ind-1]) / (vpa[sigma_ind] - vpa[sigma_ind-1])
            # so that
            #   δsigma_fraction = δsigma / (vpa[sigma_ind] - vpa[sigma_ind-1])
            #                   = u / (2 * vth * p) / (vpa[sigma_ind] - vpa[sigma_ind-1]) * δp
            #
            #   minus_vcut_fraction = ((-vcut - u)/vth - vpa[minus_vcut_ind-1]) / (vpa[minus_vcut_ind] - vpa[minus_vcut_ind-1])
            #   δminus_vcut_fraction = (vcut + u) / vth^2 / (vpa[minus_vcut_ind] - vpa[minus_vcut_ind-1]) * δvth
            #                        = (vcut + u) / (2 * vth * p) / (vpa[minus_vcut_ind] - vpa[minus_vcut_ind-1]) * δp
            #
            #   plus_vcut_fraction = ((vcut - u)/vth - vpa[plus_vcut_ind-1]) / (vpa[plus_vcut_ind+1] - vpa[plus_vcut_ind])
            #   δplus_vcut_fraction = -(vcut - u) / vth^2 / (vpa[plus_vcut_ind+1] - vpa[plus_vcut_ind]) * δvth
            #                       = -(vcut - u) / (2 * vth * p) / (vpa[plus_vcut_ind+1] - vpa[plus_vcut_ind]) * δp
            density_integral_pieces_lowerz = vpa.scratch3
            flow_integral_pieces_lowerz = vpa.scratch4
            energy_integral_pieces_lowerz = vpa.scratch5
            cubic_integral_pieces_lowerz = vpa.scratch6
            quartic_integral_pieces_lowerz = vpa.scratch7
            fill_integral_pieces!(
                pdf_lowerz, vthe_lower, vpa, vpa_unnorm, density_integral_pieces_lowerz,
                flow_integral_pieces_lowerz, energy_integral_pieces_lowerz,
                cubic_integral_pieces_lowerz, quartic_integral_pieces_lowerz)
            vpa_grid = vpa.grid

            dsigma_dp = upar_lower / (2.0 * vthe_lower * p_lower)

            dsigma_fraction_dp = dsigma_dp / (vpa_grid[sigma_ind] - vpa_grid[sigma_ind-1])

            dminus_vcut_fraction_dp = (vcut + upar_lower) / (2.0 * vthe_lower * p_lower) / (vpa_grid[minus_vcut_ind] - vpa_grid[minus_vcut_ind-1])

            dplus_vcut_fraction_dp = -(vcut - upar_lower) / (2.0 * vthe_lower * p_lower) / (vpa_grid[plus_vcut_ind+1] - vpa_grid[plus_vcut_ind])

            density_integral_sigma_cell = 0.5 * (density_integral_pieces_lowerz[sigma_ind-1] +
                                                 density_integral_pieces_lowerz[sigma_ind])
            da2_minus_a3_dp = (
                # Contribution from integral limits at sigma
                2.0 * density_integral_sigma_cell * dsigma_fraction_dp
                # Contribution from integral limits at -wcut-. The contribution from wcut+
                # is included in da3_dp
                - density_integral_pieces_lowerz[lower_cutoff_ind] * dminus_vcut_fraction_dp
                # No contribution from explicit σ factors in w-integral for a2 or a3 as
                # integrand does not contain σ.
                # Change in a3 integral due to different interpolated ̂g values
                - da3_dp
               )

            dminus_b2_minus_b3_dp = (
                # Contribution from integral limits at sigma cancel exactly
                # Contribution from integral limits at -wcut-. The contribution from wcut+
                # is included in db3_dp
                + flow_integral_pieces_lowerz[lower_cutoff_ind] * dminus_vcut_fraction_dp
                # Contribution from w-integral due to variation of integrand with p.
                + (a2 + a3) * dsigma_dp
                # Change in b3 integral due to different interpolated ̂g values
                - db3_dp
               )

            energy_integral_sigma_cell = 0.5 * (energy_integral_pieces_lowerz[sigma_ind-1] +
                                                energy_integral_pieces_lowerz[sigma_ind])
            dc2_minus_c3_dp = (
                # Contribution from integral limits at sigma
                2.0 * energy_integral_sigma_cell * dsigma_fraction_dp
                # Contribution from integral limits at -wcut-. The contribution from wcut+
                # is included in dc3_dp
                - energy_integral_pieces_lowerz[lower_cutoff_ind] * dminus_vcut_fraction_dp
                # Contribution from w-integral due to variation of integrand with p.
                + 2.0 * (-b2 + b3) * dsigma_dp
                # Change in c3 integral due to different interpolated ̂g values
                - dc3_dp
               )

            dminus_d2_minus_d3_dp = (
                # Contribution from integral limits at sigma cancel exactly
                # Contribution from integral limits at -wcut-. The contribution from wcut+
                # is included in dd3_dp
                + cubic_integral_pieces_lowerz[lower_cutoff_ind] * dminus_vcut_fraction_dp
                # Contribution from w-integral due to variation of integrand with p.
                + 3.0 * (c2 + c3) * dsigma_dp
                # Change in d3 integral due to different interpolated ̂g values
                - dd3_dp
               )

            correction0_integral_pieces = @. vpa.scratch3 = pdf_lowerz * vpa.wgts * this_integral_correction_sharpness * vpa_unnorm^2 / vthe_lower^2 / (1.0 + this_integral_correction_sharpness * vpa_unnorm^2 / vthe_lower^2)
            for ivpa ∈ 1:sigma_ind
                correction0_integral_pieces[ivpa] = 0.0
            end

            correctionminus1_integral_pieces = @. vpa.scratch4 = correction0_integral_pieces / vpa_unnorm * vthe_lower
            integralminus1 = sum(@view(correctionminus1_integral_pieces[sigma_ind+1:upper_cutoff_ind]))

            correction0_integral_type2_pieces = @. vpa.scratch4 = pdf_lowerz * vpa.wgts * 2.0 * this_integral_correction_sharpness^2 * vpa_unnorm^3 / vthe_lower^3 / (1.0 + this_integral_correction_sharpness * vpa_unnorm^2 / vthe_lower^2)^2
            integral_type2 = sum(@view(correction0_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            dalpha_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in dalpha_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * integralminus1 + integral_type2) * dsigma_dp
                # Change in alpha integral due to different interpolated ̂g values
                + dalpha_dp_interp
               )

            correction1_integral_pieces = @. vpa.scratch5 = correction0_integral_pieces * vpa_unnorm / vthe_lower
            correction1_integral_type2_pieces = @. vpa.scratch6 = correction0_integral_type2_pieces * vpa_unnorm / vthe_lower
            integral_type2 = sum(@view(correction1_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            dbeta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in dbeta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * alpha + integral_type2) * dsigma_dp
                # Change in beta integral due to different interpolated ̂g values
                + dbeta_dp_interp
               )

            # Here we overwrite the buffers that were used for correction1_integral_pieces
            # and correction1_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction2_integral_pieces = @. vpa.scratch5 = correction1_integral_pieces * vpa_unnorm / vthe_lower
            correction2_integral_type2_pieces = @. vpa.scratch6 = correction1_integral_type2_pieces * vpa_unnorm / vthe_lower
            integral_type2 = sum(@view(correction2_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            dgamma_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in dgamma_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * beta + integral_type2) * dsigma_dp
                # Change in gamma integral due to different interpolated ̂g values
                + dgamma_dp_interp
               )

            # Here we overwrite the buffers that were used for correction2_integral_pieces
            # and correction2_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction3_integral_pieces = @. vpa.scratch5 = correction2_integral_pieces * vpa_unnorm / vthe_lower
            correction3_integral_type2_pieces = @. vpa.scratch6 = correction2_integral_type2_pieces * vpa_unnorm / vthe_lower
            integral_type2 = sum(@view(correction3_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            ddelta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in ddelta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * gamma + integral_type2) * dsigma_dp
                # Change in delta integral due to different interpolated ̂g values
                + ddelta_dp_interp
               )

            # Here we overwrite the buffers that were used for correction3_integral_pieces
            # and correction3_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction4_integral_pieces = @. vpa.scratch5 = correction3_integral_pieces * vpa_unnorm / vthe_lower
            correction4_integral_type2_pieces = @. vpa.scratch6 = correction3_integral_type2_pieces * vpa_unnorm / vthe_lower
            integral_type2 = sum(@view(correction4_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            depsilon_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in depsilon_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * delta + integral_type2) * dsigma_dp
                # Change in epsilon integral due to different interpolated ̂g values
                + depsilon_dp_interp
               )

            # Here we overwrite the buffers that were used for correction4_integral_pieces
            # and correction4_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction5_integral_pieces = @. vpa.scratch5 = correction4_integral_pieces * vpa_unnorm / vthe_lower
            correction5_integral_type2_pieces = @. vpa.scratch6 = correction4_integral_type2_pieces * vpa_unnorm / vthe_lower
            integral_type2 = sum(@view(correction5_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            dzeta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in dzeta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * epsilon + integral_type2) * dsigma_dp
                # Change in zeta integral due to different interpolated ̂g values
                + dzeta_dp_interp
               )

            # Here we overwrite the buffers that were used for correction4_integral_pieces
            # and correction4_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction6_integral_pieces = @. vpa.scratch5 = correction5_integral_pieces * vpa_unnorm / vthe_lower
            correction6_integral_type2_pieces = @. vpa.scratch6 = correction5_integral_type2_pieces * vpa_unnorm / vthe_lower
            integral_type2 = sum(@view(correction6_integral_type2_pieces[sigma_ind+1:upper_cutoff_ind]))
            deta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at wcut+ is included in deta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * zeta + integral_type2) * dsigma_dp
                # Change in eta integral due to different interpolated ̂g values
                + deta_dp_interp
               )

            dA_dp, dB_dp, dC_dp, dD_dp = correction_matrix \ (
                                             [da2_minus_a3_dp, dminus_b2_minus_b3_dp, dc2_minus_c3_dp, dminus_d2_minus_d3_dp]
                                             - [dalpha_dp dbeta_dp    dgamma_dp   ddelta_dp   ;
                                                dbeta_dp  dgamma_dp   ddelta_dp   depsilon_dp ;
                                                dgamma_dp ddelta_dp   depsilon_dp dzeta_dp    ;
                                                ddelta_dp depsilon_dp dzeta_dp    deta_dp
                                               ]
                                               * [A, B, C, D]
                                            )

            @views @. jacobian_zbegin_p[output_range] -=
                          (dA_dp
                           + dB_dp * v_over_vth
                           + dC_dp * v_over_vth^2
                           + dD_dp * v_over_vth^3) *
                          this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2) *
                          pdf_lowerz[output_range]

            # Add variation due to variation of ̃v coordinate with δp.
            # These contributions seem to make almost no difference, and could probably be
            # skipped.
            dv_over_vth_dp = -dsigma_dp
            @views @. jacobian_zbegin_p[output_range] -= (
                          (B * dv_over_vth_dp
                           + 2.0 * C * v_over_vth * dv_over_vth_dp
                           + 3.0 * D * v_over_vth^2 * dv_over_vth_dp) *
                          this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2) *
                          pdf_lowerz[output_range]
                          +
                          (A
                           + B * v_over_vth
                           + C * v_over_vth^2
                           + D * v_over_vth^3) *
                          (2.0 * this_integral_correction_sharpness * v_over_vth * dv_over_vth_dp / (1.0 + this_integral_correction_sharpness * v_over_vth^2)
                          - 2.0 * this_integral_correction_sharpness^2 * v_over_vth^3 * dv_over_vth_dp / (1.0 + this_integral_correction_sharpness * v_over_vth^2)^2) *
                          pdf_lowerz[output_range]
                         )
        end
    end

    if include_upper
        shared_mem_parallel && @begin_vperp_region()
        @loop_vperp ivperp begin
            # Skip vperp boundary points.
            if vperp.n > 1 && ivperp == vperp.n
                continue
            end

            # Get matrix entries for the response of the sheath-edge boundary condition.
            # Ignore constraints, as these are non-linear and also should be small
            # corrections which should not matter much for a preconditioner.

            jac_range = pdf_offset+(zend-1)*vperp.n*vpa.n+(ivperp-1)*vpa.n+1 : pdf_offset+(zend-1)*vperp.n*vpa.n+ivperp*vpa.n
            jacobian_zend = @view jacobian[jac_range,jac_range]
            jacobian_zend_p = @view jacobian[jac_range,p_offset+zend]

            vpa_unnorm, u_over_vt, vcut, plus_vcut_ind, sigma, sigma_ind, sigma_fraction,
                element_with_zero, element_with_zero_boundary, first_point_near_zero,
                reversed_wpa_of_minus_vpa =
                    get_cutoff_params_upper(upar_upper, vthe_upper, phi_upper, me_over_mi,
                                            vpa)

            minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)
            # minus_vcut_fraction is the fraction of the distance between minus_vcut_ind-1 and
            # minus_vcut_ind where -vcut is.
            minus_vcut_fraction = get_minus_vcut_fraction(vcut, minus_vcut_ind, vpa_unnorm)

            if minus_vcut_fraction < 0.5
                first_nonzero_ind = minus_vcut_ind - 1
            else
                first_nonzero_ind = minus_vcut_ind
            end

            # Interpolate to the 'near zero' points
            @views fill_interpolate_symmetric_matrix!(
                       jacobian_zend[first_point_near_zero:sigma_ind,sigma_ind+1:element_with_zero_boundary],
                       vpa_unnorm[first_point_near_zero:sigma_ind],
                       vpa_unnorm[sigma_ind+1:element_with_zero_boundary])

            # Interpolate to the 'far from zero' points
            @views fill_1d_interpolation_matrix!(
                       jacobian_zend[first_point_near_zero-1:-1:first_nonzero_ind,:],
                       reversed_wpa_of_minus_vpa[vpa.n-first_point_near_zero+2:vpa.n-first_nonzero_ind+1],
                       vpa, vpa_spectral)

            # Reverse the sign of the elements we just filled
            jacobian_zend[first_nonzero_ind:sigma_ind,sigma_ind+1:end] .*= -1.0

            if minus_vcut_fraction < 0.5
                jacobian_zend[first_nonzero_ind,sigma_ind+1:end] .*= 0.5 - minus_vcut_fraction
            else
                jacobian_zend[first_nonzero_ind,sigma_ind+1:end] .*= 1.5 - minus_vcut_fraction
            end

            # Fill in elements giving response to changes in electron_p
            # A change to p results in a shift in the location of w_∥(v_∥=0).  The
            # interpolated values of g_e(w_∥) that are filled by the boundary condition
            # are (dropping _∥ subscripts for the remaining comments in this if-clause)
            #   g(w|v<0) = g(2*u/vth - w)
            # So
            #   δg(w|v<0) = dg/dw(2*u/vth - w) * (-2*u/vth^2) * δvth
            #             = dg/dw(2*u/vth - w) * (-2*u/vth^2) * δ(sqrt(2*p/n)
            #             = dg/dw(2*u/vth - w) * (-2*u/vth^2) * δp * vth / (2*p)
            #             = -u/vth/p * dg/dw(2*u/vth - w) * δp
            #
            # As jacobian_zend_p only depends on variations of a single value `p[end]` it
            # might be more maintainable and computationally cheaper to calculate
            # jacobian_zend_p with a finite difference method (applying the boundary
            # condition function with a perturbed pressure `p[end] + ϵ` for some small ϵ)
            # rather than calculating it using the method below. If we used the finite
            # difference approach, we would have to be careful that the ϵ step does not
            # change the cutoff index (otherwise we would pick up non-smooth changes) -
            # one possibility might to be to switch to `-ϵ` instead of `+ϵ` if this
            # happens.

            dpdfdv_near_zero = @view vpa.scratch[first_point_near_zero:sigma_ind]
            @views interpolate_symmetric!(dpdfdv_near_zero,
                                          vpa_unnorm[first_point_near_zero:sigma_ind],
                                          pdf_upper[sigma_ind+1:element_with_zero_boundary,ivperp],
                                          vpa_unnorm[sigma_ind+1:element_with_zero_boundary],
                                          Val(1))
            # The above call to interpolate_symmetric calculates dg/dv rather than dg/dw,
            # so need to multiply by an extra factor of vthe.
            #   δg(w|v<0) = -u/p * dg/dv(2*u/vth - w) * δp
            @. jacobian_zend_p[first_point_near_zero:sigma_ind] -=
                   -upar_upper / p_upper * dpdfdv_near_zero

            dpdfdw_far_from_zero = @view vpa.scratch[first_nonzero_ind:first_point_near_zero-1]
            @views interpolate_to_grid_1d!(dpdfdw_far_from_zero,
                                           reversed_wpa_of_minus_vpa[vpa.n-first_point_near_zero+2:vpa.n-first_nonzero_ind+1],
                                           pdf_upper[:,ivperp], vpa, vpa_spectral, Val(1))
            reverse!(dpdfdw_far_from_zero)
            # Note that because we calculated the derivative of the interpolating
            # function, and then reversed the results, we need to multiply the derivative
            # by -1.
            @. jacobian_zend_p[first_nonzero_ind:first_point_near_zero-1] -=
                   upar_upper / vthe_upper / p_upper * dpdfdw_far_from_zero

            # Whatever the variation due to interpolation is at the last nonzero grid
            # point, it will be reduced by the cutoff.
            if minus_vcut_fraction < 0.5
                jacobian_zend_p[first_nonzero_ind] *= 0.5 - minus_vcut_fraction
            else
                jacobian_zend_p[first_nonzero_ind] *= 1.5 - minus_vcut_fraction
            end

            # The change in electron_p also changes the position of
            #     wcut = (-vcut - upar)/vthe
            # as vcut does not change (within the Krylov iteration where this
            # preconditioner matrix is used), but vthe does because of the change in
            # electron_p. We actually use minus_vcut_fraction calculated from
            # vpa_unnorm, so it is most convenient to consider:
            #     v = vthe * w + upar
            #     δv = δvthe * w
            #        = δvthe * (v - upar)/vthe
            #        = δp * vthe / (2*p) * (v - upar)/vthe
            #        = δp * (v - upar) / 2 / p
            # with vl and vu the values of v at the grid points below and above vcut
            #     minus_vcut_fraction = (-vcut - vl) / (vu - vl)
            #     δminus_vcut_fraction = -(-vcut - vl) / (vu - vl)^2 * (δvu - δvl) - δvl / (vu - vl)
            #     δminus_vcut_fraction = [-(-vcut - vl) / (vu - vl)^2 * (vu - vl) - (vl - upar) / (vu - vl)] * δp / 2 / p
            #     δminus_vcut_fraction = [-(-vcut - vl) / (vu - vl) - (vl - upar) / (vu - vl)] * δp / 2 / p
            #     δminus_vcut_fraction = -(-vcut - upar) / (vu - vl) / 2 / p * δp
            #     δminus_vcut_fraction = (vcut + upar) / (vu - vl) / 2 / p * δp
            interpolated_pdf_at_first_nonzero_ind = @view vpa.scratch[1:1]
            reversed_first_nonzero_ind = vpa.n-first_nonzero_ind+1
            @views interpolate_to_grid_1d!(interpolated_pdf_at_first_nonzero_ind,
                                           reversed_wpa_of_minus_vpa[reversed_first_nonzero_ind:reversed_first_nonzero_ind],
                                           pdf_upper[:,ivperp], vpa, vpa_spectral)

            dminus_vcut_fraction_dp = (vcut + upar_upper) / (vpa_unnorm[minus_vcut_ind] - vpa_unnorm[minus_vcut_ind-1]) / 2.0 / p_upper
            # Note that pdf_upper[first_nonzero_ind,ivperp] depends on -minus_vcut_fraction, so
            # need a -'ve sign in the following line.
            jacobian_zend_p[first_nonzero_ind] -= -interpolated_pdf_at_first_nonzero_ind[] * dminus_vcut_fraction_dp

            # Calculate some numerical integrals of dpdfdw that we will need later
            function get_part3_for_one_moment_upper(integral_pieces)
                # Integral contribution from the cell containing sigma
                # The contribution from integral_pieces[sigma_ind+1] should be dropped,
                # because that point is on the input grid, and this correction is due to
                # the changes in interpolated values on the output grid due to δp_e∥. The
                # value of g_e at sigma_ind+1 isn't interpolated so does not change in
                # this way.
                integral_sigma_cell = 0.5 * integral_pieces[sigma_ind]

                part3 = sum(@view integral_pieces[minus_vcut_ind-1:sigma_ind-1])
                part3 += 0.5 * integral_pieces[sigma_ind] + (1.0 - sigma_fraction) * integral_sigma_cell

                return part3
            end
            # The contents of jacobian_zend_p at this point are the coefficients needed
            # to get the new distribution function due to a change δp, which we can now
            # use to calculate the response of various integrals to the same change.  Note
            # that jacobian_zend_p already contains `2.0*dsigma_dp`.
            @. vpa.scratch = -jacobian_zend_p * vpa.wgts
            da3_dp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            db3_dp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            dc3_dp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            dd3_dp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch = -jacobian_zend_p * vpa.wgts * this_integral_correction_sharpness * vpa_unnorm^2 / vthe_upper^2 / (1.0 + this_integral_correction_sharpness * vpa_unnorm^2 / vthe_upper^2)
            vpa.scratch[sigma_ind:sigma_ind+1] .= 0.0
            dalpha_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            dbeta_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            dgamma_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            ddelta_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            depsilon_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            dzeta_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)
            @. vpa.scratch *= vpa_unnorm / vthe_upper
            deta_dp_interp = get_part3_for_one_moment_upper(vpa.scratch)

            pdf_upperz = vpa.scratch10
            @views @. pdf_upperz[sigma_ind:end] = pdf_upper[sigma_ind:end,ivperp]

            # interpolate the pdf_upperz onto this grid
            # 'near zero' means in the range where
            # abs(v_∥)≤abs(upper boundary of element including v_∥=0)
            # 'far from zero' means more negative values of v_∥.

            # Interpolate to the 'near zero' points
            @views interpolate_symmetric!(pdf_upperz[first_point_near_zero:sigma_ind],
                                          vpa_unnorm[first_point_near_zero:sigma_ind],
                                          pdf_upperz[sigma_ind+1:element_with_zero_boundary],
                                          vpa_unnorm[sigma_ind+1:element_with_zero_boundary])

            # Interpolate to the 'far from zero' points
            reversed_pdf_far_from_zero = @view pdf_upperz[1:first_point_near_zero-1]
            interpolate_to_grid_1d!(reversed_pdf_far_from_zero,
                                    reversed_wpa_of_minus_vpa[vpa.n-first_point_near_zero+2:end],
                                    pdf_upperz, vpa, vpa_spectral)
            reverse!(reversed_pdf_far_from_zero)

            plus_vcut_fraction = get_plus_vcut_fraction(vcut, plus_vcut_ind, vpa_unnorm)
            if plus_vcut_fraction > 0.5
                upper_cutoff_ind = plus_vcut_ind+1
            else
                upper_cutoff_ind = plus_vcut_ind
            end
            if minus_vcut_fraction < 0.5
                lower_cutoff_ind = minus_vcut_ind-1
                lower_cutoff_factor = 0.5 - minus_vcut_fraction
            else
                lower_cutoff_ind = minus_vcut_ind
                lower_cutoff_factor = 1.5 - minus_vcut_fraction
            end
            pdf_upperz[lower_cutoff_ind] *= lower_cutoff_factor
            pdf_upperz[1:lower_cutoff_ind-1] .= 0.0

            a2, b2, c2, d2, a3, b3, c3, d3, alpha, beta, gamma, delta, epsilon, zeta, eta =
                get_upperz_integral_correction_components(
                    pdf_upperz, vthe_upper, vperp, vpa, vpa_unnorm, u_over_vt, sigma_ind,
                    sigma_fraction, vcut, minus_vcut_ind, plus_vcut_ind, false)

            output_range = lower_cutoff_ind:sigma_ind

            v_over_vth = @views @. vpa.scratch[output_range] = vpa_unnorm[output_range] / vthe_upper

            correction_matrix = [alpha beta    gamma   delta   ;
                                 beta  gamma   delta   epsilon ;
                                 gamma delta   epsilon zeta    ;
                                 delta epsilon zeta    eta
                                ]

            # jacobian_zbegin_p state at this point would generate (when multiplied by
            # some δp) a δf due to the effectively-shifted grid. The unperturbed f
            # required corrections to make its integrals correct. Need to apply the same
            # corrections to δf.
            # This term seems to have a very small contribution, and could probably be
            # skipped.
            A, B, C, D = correction_matrix \ [a2-a3, -b2-b3, c2-c3, -d2-d3]
            @. jacobian_zend_p[output_range] *=
                   1.0 +
                   (A
                    + B * v_over_vth
                    + C * v_over_vth^2
                    + D * v_over_vth^3) *
                   this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2)

            # Calculate the changes in the integrals a2, b2, c2, d2, a3, b3, c3, and d3 in
            # response to changes in electron_p. The changes are calculated for the
            # combinations (a2-a3), (-b2-b3), (c2-c3), and (-d2-d3) to take advantage of
            # some cancellations.

            # Need to calculate the variation of various intermediate quantities with
            # δp.
            #
            #   vth = sqrt(2*p/n)
            #   δvth = vth / (2 * p) * δp
            #
            #   sigma = -u / vth
            #   δsigma = u / vth^2 δvth
            #          = u / (2 * vth * p) * δp
            #
            # We could write sigma_fraction as
            #   sigma_fraction = (sigma - vpa[sigma_ind]) / (vpa[sigma_ind+1] - vpa[sigma_ind])
            # so that
            #   δsigma_fraction = δsigma / (vpa[sigma_ind+1] - vpa[sigma_ind])
            #                   = u / (2 * vth * p) / (vpa[sigma_ind+1] - vpa[sigma_ind]) * δp
            #
            #   minus_vcut_fraction = ((-vcut - u)/vth - vpa[minus_vcut_ind-1]) / (vpa[minus_vcut_ind] - vpa[minus_vcut_ind-1])
            #   δminus_vcut_fraction = (vcut + u) / vth^2 / (vpa[minus_vcut_ind] - vpa[minus_vcut_ind-1]) * δvth
            #                        = (vcut + u) / (2 * vth * p) / (vpa[minus_vcut_ind] - vpa[minus_vcut_ind-1]) * δp
            #
            #   plus_vcut_fraction = ((vcut - u)/vth - vpa[plus_vcut_ind-1]) / (vpa[plus_vcut_ind+1] - vpa[plus_vcut_ind])
            #   δplus_vcut_fraction = -(vcut - u) / vth^2 / (vpa[plus_vcut_ind+1] - vpa[plus_vcut_ind]) * δvth
            #                       = -(vcut - u) / (2 * vth * p) / (vpa[plus_vcut_ind+1] - vpa[plus_vcut_ind]) * δp

            density_integral_pieces_upperz = vpa.scratch3
            flow_integral_pieces_upperz = vpa.scratch4
            energy_integral_pieces_upperz = vpa.scratch5
            cubic_integral_pieces_upperz = vpa.scratch6
            quartic_integral_pieces_upperz = vpa.scratch7
            fill_integral_pieces!(
                pdf_upperz, vthe_upper, vpa, vpa_unnorm, density_integral_pieces_upperz,
                flow_integral_pieces_upperz, energy_integral_pieces_upperz,
                cubic_integral_pieces_upperz, quartic_integral_pieces_upperz)
            vpa_grid = vpa.grid

            dsigma_dp = upar_upper / (2.0 * vthe_upper * p_upper)

            dsigma_fraction_dp = dsigma_dp / (vpa_grid[sigma_ind+1] - vpa_grid[sigma_ind])

            dminus_vcut_fraction_dp = (vcut + upar_upper) / (2.0 * vthe_upper * p_upper) / (vpa_grid[minus_vcut_ind] - vpa_grid[minus_vcut_ind-1])

            dplus_vcut_fraction_dp = -(vcut - upar_upper) / (2.0 * vthe_upper * p_upper) / (vpa_grid[plus_vcut_ind+1] - vpa_grid[plus_vcut_ind])


            density_integral_sigma_cell = 0.5 * (density_integral_pieces_upperz[sigma_ind] +
                                                 density_integral_pieces_upperz[sigma_ind+1])
            da2_minus_a3_dp = (
                # Contribution from integral limits at sigma
                -2.0 * density_integral_sigma_cell * dsigma_fraction_dp
                # Contribution from integral limits at wcut+. The contribution from -wcut-
                # is included in da3_dp
                + density_integral_pieces_upperz[upper_cutoff_ind] * dplus_vcut_fraction_dp
                # No contribution from w-integral for a2 or a3 as integrand does not
                # depend on p.
                # Change in a3 integral due to different interpolated ̂g values
                - da3_dp
               )

            dminus_b2_minus_b3_dp = (
                # Contribution from integral limits at sigma cancel exactly
                # Contribution from integral limits at wcut+. The contribution from -wcut-
                # is included in db3_dp
                - flow_integral_pieces_upperz[upper_cutoff_ind] * dplus_vcut_fraction_dp
                # Contribution from w-integral due to variation of integrand with p.
                + (a2 + a3) * dsigma_dp
                # Change in b3 integral due to different interpolated ̂g values
                - db3_dp
               )

            energy_integral_sigma_cell = 0.5 * (energy_integral_pieces_upperz[sigma_ind] +
                                                energy_integral_pieces_upperz[sigma_ind+1])
            dc2_minus_c3_dp = (
                # Contribution from integral limits at sigma
                -2.0 * energy_integral_sigma_cell * dsigma_fraction_dp
                # Contribution from integral limits at wcut+. The contribution from -wcut-
                # is included in dc3_dp
                + energy_integral_pieces_upperz[upper_cutoff_ind] * dplus_vcut_fraction_dp
                # Contribution from w-integral due to variation of integrand with p.
                + 2.0 * (-b2 + b3) * dsigma_dp
                # Change in c3 integral due to different interpolated ̂g values
                - dc3_dp
               )

            dminus_d2_minus_d3_dp = (
                # Contribution from integral limits at sigma cancel exactly
                # Contribution from integral limits at wcut+. The contribution from -wcut-
                # is included in dc3_dp
                - cubic_integral_pieces_upperz[upper_cutoff_ind] * dplus_vcut_fraction_dp
                # Contribution from w-integral due to variation of integrand with p.
                + 3.0 * (c2 + c3) * dsigma_dp
                # Change in d3 integral due to different interpolated ̂g values
                - dd3_dp
               )

            correction0_integral_pieces = @. vpa.scratch3 = pdf_upperz * vpa.wgts * this_integral_correction_sharpness * vpa_unnorm^2 / vthe_upper^2 / (1.0 + this_integral_correction_sharpness * vpa_unnorm^2 / vthe_upper^2)
            for ivpa ∈ sigma_ind:vpa.n
                correction0_integral_pieces[ivpa] = 0.0
            end

            correctionminus1_integral_pieces = @. vpa.scratch4 = correction0_integral_pieces / vpa_unnorm * vthe_upper
            integralminus1 = sum(@view(correctionminus1_integral_pieces[lower_cutoff_ind:sigma_ind-1]))

            correction0_integral_type2_pieces = @. vpa.scratch4 = pdf_upperz * vpa.wgts * 2.0 * this_integral_correction_sharpness^2 * vpa_unnorm^3 / vthe_upper^3 / (1.0 + this_integral_correction_sharpness * vpa_unnorm^2 / vthe_upper^2)^2
            integral_type2 = sum(@view(correction0_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            dalpha_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in dalpha_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                 (-2.0 * integralminus1 + integral_type2) * dsigma_dp
                # Change in alpha integral due to different interpolated ̂g values
                + dalpha_dp_interp
               )

            correction1_integral_pieces = @. vpa.scratch5 = correction0_integral_pieces * vpa_unnorm / vthe_upper
            correction1_integral_type2_pieces = @. vpa.scratch6 = correction0_integral_type2_pieces * vpa_unnorm / vthe_upper
            integral_type2 = sum(@view(correction1_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            dbeta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in dbeta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * alpha + integral_type2) * dsigma_dp
                # Change in beta integral due to different interpolated ̂g values
                + dbeta_dp_interp
               )

            # Here we overwrite the buffers that were used for correction1_integral_pieces
            # and correction1_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction2_integral_pieces = @. vpa.scratch5 = correction1_integral_pieces * vpa_unnorm / vthe_upper
            correction2_integral_type2_pieces = @. vpa.scratch6 = correction1_integral_type2_pieces * vpa_unnorm / vthe_upper
            integral_type2 = sum(@view(correction2_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            dgamma_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in dgamma_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * beta + integral_type2) * dsigma_dp
                # Change in gamma integral due to different interpolated ̂g values
                + dgamma_dp_interp
               )

            # Here we overwrite the buffers that were used for correction2_integral_pieces
            # and correction2_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction3_integral_pieces = @. vpa.scratch5 = correction2_integral_pieces * vpa_unnorm / vthe_upper
            correction3_integral_type2_pieces = @. vpa.scratch6 = correction2_integral_type2_pieces * vpa_unnorm / vthe_upper
            integral_type2 = sum(@view(correction3_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            ddelta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in ddelta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * gamma + integral_type2) * dsigma_dp
                # Change in delta integral due to different interpolated ̂g values
                + ddelta_dp_interp
               )

            # Here we overwrite the buffers that were used for correction3_integral_pieces
            # and correction3_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction4_integral_pieces = @. vpa.scratch5 = correction3_integral_pieces * vpa_unnorm / vthe_upper
            correction4_integral_type2_pieces = @. vpa.scratch6 = correction3_integral_type2_pieces * vpa_unnorm / vthe_upper
            integral_type2 = sum(@view(correction4_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            depsilon_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in depsilon_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * delta + integral_type2) * dsigma_dp
                # Change in epsilon integral due to different interpolated ̂g values
                + depsilon_dp_interp
               )

            # Here we overwrite the buffers that were used for correction4_integral_pieces
            # and correction4_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction5_integral_pieces = @. vpa.scratch5 = correction4_integral_pieces * vpa_unnorm / vthe_upper
            correction5_integral_type2_pieces = @. vpa.scratch6 = correction4_integral_type2_pieces * vpa_unnorm / vthe_upper
            integral_type2 = sum(@view(correction5_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            dzeta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in dzeta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * epsilon + integral_type2) * dsigma_dp
                # Change in zeta integral due to different interpolated ̂g values
                + dzeta_dp_interp
               )

            # Here we overwrite the buffers that were used for correction4_integral_pieces
            # and correction4_integral_type2_pieces, but this is OK as we never need those
            # arrays again.
            correction6_integral_pieces = @. vpa.scratch5 = correction5_integral_pieces * vpa_unnorm / vthe_upper
            correction6_integral_type2_pieces = @. vpa.scratch6 = correction5_integral_type2_pieces * vpa_unnorm / vthe_upper
            integral_type2 = sum(@view(correction6_integral_type2_pieces[lower_cutoff_ind:sigma_ind-1]))
            deta_dp = (
                # The grid points either side of sigma are zero-ed out for these
                # corrections, so this boundary does not contribute.
                # Contribution from integral limit at -wcut- is included in deta_dp_interp
                # Contribution from w-integral due to variation of integrand with p.
                + (-2.0 * zeta + integral_type2) * dsigma_dp
                # Change in eta integral due to different interpolated ̂g values
                + deta_dp_interp
               )

            dA_dp, dB_dp, dC_dp, dD_dp = correction_matrix \ (
                                             [da2_minus_a3_dp, dminus_b2_minus_b3_dp, dc2_minus_c3_dp, dminus_d2_minus_d3_dp]
                                             - [dalpha_dp dbeta_dp    dgamma_dp   ddelta_dp   ;
                                                dbeta_dp  dgamma_dp   ddelta_dp   depsilon_dp ;
                                                dgamma_dp ddelta_dp   depsilon_dp dzeta_dp    ;
                                                ddelta_dp depsilon_dp dzeta_dp    deta_dp
                                               ]
                                               * [A, B, C, D]
                                              )

            output_range = lower_cutoff_ind:sigma_ind-1
            v_over_vth = @views @. vpa.scratch[output_range] = vpa_unnorm[output_range] / vthe_upper
            @views @. jacobian_zend_p[output_range] -=
                          (dA_dp
                           + dB_dp * v_over_vth
                           + dC_dp * v_over_vth^2
                           + dD_dp * v_over_vth^3) *
                          this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2) *
                          pdf_upperz[output_range]

            # Add variation due to variation of ̃v coordinate with δp.
            # These contributions seem to make almost no difference, and could probably be
            # skipped.
            dv_over_vth_dp = -dsigma_dp
            @views @. jacobian_zend_p[output_range] -= (
                          (B * dv_over_vth_dp
                           + 2.0 * C * v_over_vth * dv_over_vth_dp
                           + 3.0 * D * v_over_vth^2 * dv_over_vth_dp) *
                          this_integral_correction_sharpness * v_over_vth^2 / (1.0 + this_integral_correction_sharpness * v_over_vth^2) *
                          pdf_upperz[output_range]
                          +
                          (A
                           + B * v_over_vth
                           + C * v_over_vth^2
                           + D * v_over_vth^3) *
                          (2.0 * this_integral_correction_sharpness * v_over_vth * dv_over_vth_dp / (1.0 + this_integral_correction_sharpness * v_over_vth^2)
                          - 2.0 * this_integral_correction_sharpness^2 * v_over_vth^3 * dv_over_vth_dp / (1.0 + this_integral_correction_sharpness * v_over_vth^2)^2) *
                          pdf_upperz[output_range]
                         )
        end
    end

    return nothing
end

"""
    electron_adaptive_timestep_update!(scratch, t, t_params, moments, phi, z_advect,
                                       vpa_advect, composition, r, z, vperp, vpa,
                                       vperp_spectral, vpa_spectral,
                                       external_source_settings, num_diss_params;
                                       evolve_p=false)

Check the error estimate for the embedded RK method and adjust the timestep if
appropriate.
"""
@timeit global_timer electron_adaptive_timestep_update!(
                         scratch, t, t_params, moments, phi, z_advect, vpa_advect,
                         composition, r, z, vperp, vpa, vperp_spectral, vpa_spectral,
                         external_source_settings, num_diss_params; evolve_p=false,
                         local_max_dt=Inf) = begin
    #error_norm_method = "Linf"
    error_norm_method = "L2"

    error_coeffs = t_params.rk_coefs[:,end]
    if t_params.n_rk_stages < 3
        # This should never happen as an adaptive RK scheme needs at least 2 RHS evals so
        # (with the pre-timestep data) there must be at least 3 entries in `scratch`.
        error("adaptive timestep needs a buffer scratch array")
    end

    CFL_limits = OrderedDict{String,mk_float}()
    error_norm_type = typeof(t_params.error_sum_zero)
    error_norms = OrderedDict{String,error_norm_type}()
    total_points = mk_int[]

    # Test CFL conditions for advection in electron kinetic equation to give stability
    # limit for timestep
    #
    # z-advection
    # No need to synchronize here, as we just called @_block_synchronize()
    @begin_r_vperp_vpa_region(true)
    update_electron_speed_z!(z_advect[1], moments.electron.upar, moments.electron.vth,
                             vpa.grid)
    z_CFL = get_minimum_CFL_z(z_advect[1].speed, z)
    if block_rank[] == 0
        CFL_limits["CFL_z"] = t_params.CFL_prefactor * z_CFL
    else
        CFL_limits["CFL_z"] = Inf
    end

    # vpa-advection
    @begin_r_z_vperp_region()
    update_electron_speed_vpa!(vpa_advect[1], moments.electron.dens,
                               moments.electron.upar,
                               scratch[t_params.n_rk_stages+1].electron_p, moments,
                               composition.me_over_mi, vpa.grid,
                               external_source_settings.electron)
    vpa_CFL = get_minimum_CFL_vpa(vpa_advect[1].speed, vpa)
    if block_rank[] == 0
        CFL_limits["CFL_vpa"] = t_params.CFL_prefactor * vpa_CFL
    else
        CFL_limits["CFL_vpa"] = Inf
    end

    # To avoid double counting points when we use distributed-memory MPI, skip the
    # inner/lower point in r and z if this process is not the first block in that
    # dimension.
    skip_r_inner = r.irank != 0
    skip_z_lower = z.irank != 0

    # Calculate error ion distribution functions
    # Note rk_loworder_solution!() stores the calculated error in `scratch[2]`.
    rk_loworder_solution!(scratch, nothing, :pdf_electron, t_params)
    if evolve_p
        @begin_r_z_region()
        rk_loworder_solution!(scratch, nothing, :electron_p, t_params)

        # Make vth consistent with `scratch[2]`, as it is needed for the electron pdf
        # boundary condition.
        update_electron_vth_temperature!(moments, scratch[2].electron_p,
                                         moments.electron.dens, composition)
    end
    apply_electron_bc_and_constraints!(scratch[t_params.n_rk_stages+1], phi, moments, r,
                                       z, vperp, vpa, vperp_spectral, vpa_spectral,
                                       vpa_advect, num_diss_params, composition)
    if evolve_p
        # Reset vth in the `moments` struct to the result consistent with full-accuracy RK
        # solution.
        @begin_r_z_region()
        update_electron_vth_temperature!(moments,
                                         scratch[t_params.n_rk_stages+1].electron_p,
                                         moments.electron.dens, composition)
    end

    pdf_error = local_error_norm(scratch[2].pdf_electron,
                                 scratch[t_params.n_rk_stages+1].pdf_electron,
                                 t_params.rtol, t_params.atol; method=error_norm_method,
                                 skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                 error_sum_zero=t_params.error_sum_zero)
    error_norms["pdf_accuracy"] = pdf_error
    push!(total_points, vpa.n_global * vperp.n_global * z.n_global * r.n_global)

    # Calculate error for moments, if necessary
    if evolve_p
        @begin_r_z_region()
        p_err = local_error_norm(scratch[2].electron_p,
                                 scratch[t_params.n_rk_stages+1].electron_p,
                                 t_params.rtol, t_params.atol; method=error_norm_method,
                                 skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                 error_sum_zero=t_params.error_sum_zero)
        error_norms["p_accuracy"] = p_err
        push!(total_points, z.n_global * r.n_global)
    end

    adaptive_timestep_update_t_params!(t_params, CFL_limits, error_norms, total_points,
                                       error_norm_method, "", 0.0, false, false,
                                       composition; electron=true,
                                       local_max_dt=local_max_dt)
    if t_params.previous_dt[] == 0.0
        # Timestep failed, so reset  scratch[t_params.n_rk_stages+1] equal to
        # scratch[1] to start the timestep over.
        scratch_temp = scratch[t_params.n_rk_stages+1]
        scratch[t_params.n_rk_stages+1] = scratch[1]
        scratch[1] = scratch_temp
    end

    return nothing
end

"""
update_electron_pdf_with_shooting_method is a function that using a shooting method to solve for the electron pdf

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor
The shooting method is 'explicit' in z, solving
    zdot_i * (pdf_{i+1} - pdf_{i})/dz_{i} + wpadot_{i} * d(pdf_{i})/dwpa = pdf_{i} * pre_factor_{i}

    INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    p = electron pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_shooting_method!(pdf, dens, vthe, p, qpar, qpar_updated, phi, 
    ddens_dz, dppar_dz, dqpar_dz, dvth_dz, z, vpa, vpa_spectral, scratch_dummy, composition)

    # it is convenient to create a pointer to a scratch array to store the RHS of the linear 
    # system that includes all terms evaluated at the previous z location in the shooting method
    rhs = scratch_dummy.buffer_vpavperpzr_1
    # initialise the RHS to zero
    rhs .= 0.0
    # get critical velocities beyond which electrons are lost to the wall
    crit_speed_zmin, crit_speed_zmax = get_electron_critical_velocities(phi, vthe, composition.me_over_mi, z)
    # add the contribution to rhs from the term proportional to the pdf (rather than its derivatives)
    add_contribution_from_pdf_term!(rhs, pdf, p, vthe, dens, ddens_dz, upar, dvth_dz, dqpar_dz, vpa.grid, z, external_source_settings.electron)
    # add the contribution to rhs from the wpa advection term
    add_contribution_from_wpa_advection!(rhs, pdf, vthe, p, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
    # shoot in z from incoming boundary (using sign of zdot to determine direction)
    # pdf_{i+1} = pdf_{i} - dz_{i} * rhs_{i}
    @loop_r_vperp ir ivperp begin
        for ivpa ∈ 1:vpa.n
            # deal with case of positive z-advection speed
            if vpa.grid[ivpa] > eps(mk_float)
                for iz ∈ 1:z.n-1
                    pdf[ivpa, ivperp, iz + 1, ir] = pdf[ivpa, ivperp, iz, ir] - (z.cell_width[iz] * rhs[ivpa, ivperp, iz, ir] 
                                                              / (vpa.grid[ivpa] * vthe[iz, ir]))
                end
            # deal with case of negative z-advection speed
            elseif vpa.grid[ivpa] < -eps(mk_float)
                for iz ∈ z.n:-1:2
                    pdf[ivpa, ivperp, iz - 1, ir] = pdf[ivpa, ivperp, iz, ir] + (z.cell_width[iz-1] * rhs[ivpa, ivperp, iz, ir] 
                                                              / (vpa.grid[ivpa] * vthe[iz, ir]))
                end
            # deal with case of zero z-advection speed
            else
                # hack for now until I figure out how to deal with the singular case
                @. pdf[ivpa, ivperp, :, ir] = 0.5 * (pdf[ivpa+1, ivperp, :, ir] + pdf[ivpa-1, ivperp, :, ir])
            end
        end
        # enforce the boundary condition at the walls
        for ivpa ∈ 1:vpa.n
            # find the ivpa index corresponding to the same magnitude 
            # of vpa but with opposite sign
            ivpamod = vpa.n - ivpa + 1
            # deal with wall at z_min
            if vpa.grid[ivpa] > eps(mk_float)
                iz = 1
                # electrons travelling sufficiently fast in the negative z-direction are lost to the wall
                # the maximum posative z-velocity that can be had is the amount by which electrons
                # can be accelerated by the electric field between the wall and the simulation boundary
                if vpa.grid[ivpa] > crit_speed_zmin
                    pdf[ivpa, ivperp, iz, ir] = 0.0
                else
                    pdf[ivpa, ivperp, iz, ir] = pdf[ivpamod, ivperp, iz, ir]
                end
            # deal with wall at z_max
            elseif vpa.grid[ivpa] < -eps(mk_float)
                iz = z.n
                # electrons travelling sufficiently fast in the positive z-direction are lost to the wall
                # the maximum negative z-velocity that can be had is the amount by which electrons
                # can be accelerated by the electric field between the wall and the simulation boundary
                if vpa.grid[ivpa] < crit_speed_zmax
                    pdf[ivpa, ivperp, iz, ir] = 0.0
                else
                    pdf[ivpa, ivperp, iz, ir] = pdf[ivpamod, ivperp, iz, ir]
                end
            end
        end
    end
    # the electron parallel heat flux is no longer consistent with the electron pdf
    qpar_updated = false
    # calculate the updated electron parallel heat flux
    calculate_electron_qpar_from_pdf!(qpar, density, vthe, pdf, vperp, vpa,
                                      composition.me_over_mi)
    for iz ∈ 1:z.n
        for ivpa ∈ 1:vpa.n
            println("z: ", z.grid[iz], " vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " qpar: ", qpar[iz, 1], 
                " dppardz: ", dppar_dz[iz,1], " vth: ", vthe[iz,1], " p: ", p[iz,1], " dqpar_dz: ", dqpar_dz[iz,1],
                " dvth_dz: ", dvth_dz[iz,1], " ddens_dz: ", ddens_dz[iz,1])
        end
        println()
    end
    return nothing    
end

"""
use Picard iteration to solve the electron kinetic equation

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor
Picard iteration uses the previous iteration of the electron pdf to calculate the next iteration:
    zdot * d(pdf^{i+1})/dz + wpadot^{i} * d(pdf^{i})/dwpa = pdf^{i} * prefactor^{i}

INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    p = electron pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_picard_iteration!(pdf, dens, vthe, p, ddens_dz, dppar_dz, dqpar_dz, dvth_dz,
    z, vpa, vpa_spectral, scratch_dummy, max_electron_pdf_iterations)

    # it is convenient to create a pointer to a scratch array to store the RHS of the linear 
    # system that will be solved iteratively
    rhs = scratch_dummy.buffer_vpavperpzr_1
    # define residual to point to a scratch array;
    # to be filled with the difference between successive iterations of the electron pdf
    residual = scratch_dummy.buffer_vpavperpzr_2
    # define pdf_new to point to a scratch array;
    # to be filled with the updated electron pdf
    pdf_new = scratch_dummy.buffer_vpavperpzr_3
    # initialise the electron pdf convergence flag to false
    electron_pdf_converged = false
    # initialise the number of iterations in the solution of the electron kinetic equation to be 1
    iteration = 1
    while !electron_pdf_converged && (iteration < max_electron_pdf_iterations)
        # calculate the RHS of the linear system, consisting of all terms in which the
        # pdf is evaluated at the previous iteration.
        
        # initialise the RHS to zero
        rhs .= 0.0
        # add the contribution to rhs from the term proportional to the pdf (rather than its derivatives)
        add_contribution_from_pdf_term!(rhs, pdf, p, vthe, dens, ddens_dz, upar, dvth_dz, dqpar_dz, vpa.grid, z, external_source_settings.electron)
        # add the contribution to rhs from the wpa advection term
        #add_contribution_from_wpa_advection!(rhs, pdf, vthe, p, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
        # loop over wpa locations, solving the linear system at each location;
        # care must be taken when wpa = 0, as the linear system is singular in this case
        @loop_r_vperp_vpa ir ivperp ivpa begin
            # modify the rhs vector to account for the pre-factor multiplying d(pdf)/dz
            if abs(vpa.grid[ivpa]) < eps()
                # hack for now until I figure out how to deal with the singular linear system
                rhs[ivpa, ivperp, :, ir] .= 0.0
            else
                @loop_z iz begin
                    #rhs[ivpa, ivperp, iz, ir] /= -(vpa.grid[ivpa] * vthe[iz, ir])
                    rhs[ivpa, ivperp, iz, ir] = -rhs[ivpa, ivperp, iz, ir]
                end
            end
            # solve the linear system at this (wpa, wperp, r) location
            pdf_new[ivpa, ivperp, :, ir] = z.differentiation_matrix \ rhs[ivpa, ivperp, :, ir]
        end
        # calculate the difference between successive iterations of the electron pdf
        @. residual = pdf_new - pdf
        # check to see if the electron pdf has converged to within the specified tolerance
        average_residual, electron_pdf_converged = check_electron_pdf_convergence(residual)
        # prepare for the next iteration by updating the pdf
        @. pdf = pdf_new
        # if converged, exit the loop; otherwise, increment the iteration counter
        if electron_pdf_converged
            break
        else
            iteration +=1
        end
    end
    # if the maximum number of iterations has been exceeded, print a warning
    if !electron_pdf_converged
        # need to exit or handle this appropriately
        println("!!! max number of iterations for electron pdf update exceeded !!!")
        @loop_z iz begin
            println("z: ", z.grid[iz], " pdf: ", pdf[10, 1, iz, 1])
        end
    end
    
    return nothing
end

"""
    electron_kinetic_equation_euler_update!(result_object, f_in, p_in, moments, z, vperp,
                                            vpa, z_spectral, vpa_spectral, z_advect,
                                            vpa_advect, scratch_dummy, collisions,
                                            composition, external_source_settings,
                                            num_diss_params, t_params, ir; evolve_p=false,
                                            ion_dt=nothing, soft_force_constraints=false,
                                            debug_io=nothing, fields=nothing, r=nothing,
                                            vzeta=nothing, vr=nothing, vz=nothing,
                                            istage=0)

Do a forward-Euler update of the electron kinetic equation.

When `evolve_p=true` is passed, also updates the electron parallel pressure.

Note that this function operates on a single point in `r`, given by `ir`, and `f_out`,
`p_out`, `f_in`, and `p_in` should have no r-dimension.

`result_object` should be either a `scratch_pdf` object (which contains data for all
r-indices) or a Tuple containing `(pdf_electron, electron_p)` fields for r-index `ir`
only. This allows `result_object` to be (possibly) passed to
`write_debug_data_to_binary()` when `result_object` is a `scratch_pdf`.

`fields`, `r`, `vzeta`, `vr`, `vz`, and `istage` are only required when a non-`nothing`
`debug_io` is passed.
"""
@timeit global_timer electron_kinetic_equation_euler_update!(
                         result_object, f_in, p_in, moments, z, vperp, vpa, z_spectral,
                         vpa_spectral, z_advect, vpa_advect, scratch_dummy, collisions,
                         composition, external_source_settings, num_diss_params, t_params,
                         ir; evolve_p=false, ion_dt=nothing, soft_force_constraints=false,
                         debug_io=nothing, fields=nothing, r=nothing, vzeta=nothing,
                         vr=nothing, vz=nothing, istage=0) = begin

    if debug_io !== nothing && !isa(result_object, scratch_pdf)
        error("debug_io can only be used when a `scratch_pdf` is passed as result_object")
    end
    if debug_io !== nothing && fields === nothing
        error("`fields` is required when debug_io is passed")
    end

    function write_debug_IO(label)
        if debug_io === nothing
            return nothing
        end
        write_debug_data_to_binary(result_object, moments, fields, composition, t_params,
                                   r, z, vperp, vpa, vzeta, vr, vz, label, istage)
        return nothing
    end

    if isa(result_object, scratch_pdf) || isa(result_object, scratch_electron_pdf)
        # Arrays including r-dimension passed to allow debug output to be written
        f_out = @view result_object.pdf_electron[:,:,:,ir]
        p_out = @view result_object.electron_p[:,ir]
    else
        f_out, p_out = result_object
    end
    dt = t_params.dt[]

    if evolve_p
        @views electron_energy_equation_no_r!(
                   p_out, moments.electron.dens[:,ir], p_in, moments.electron.dens[:,ir],
                   moments.electron.upar[:,ir], moments.electron.ppar[:,ir],
                   moments.ion.dens[:,ir,:], moments.ion.upar[:,ir,:],
                   moments.ion.p[:,ir,:], moments.neutral.dens[:,ir,:],
                   moments.neutral.uz[:,ir,:], moments.neutral.pz[:,ir,:],
                   moments.electron, collisions, dt, composition,
                   external_source_settings.electron, num_diss_params, z, ir;
                   ion_dt=ion_dt)

        update_derived_electron_moment_time_derivatives!(p_in, moments,
                                                         composition.electron_physics)
        write_debug_IO("electron_energy_equation_no_r!")
    end

    # add the contribution from the z advection term
    @views electron_z_advection!(f_out, f_in, moments.electron.upar[:,ir],
                                 moments.electron.vth[:,ir], z_advect, z, vpa.grid,
                                 z_spectral, scratch_dummy, dt, ir)
    write_debug_IO("electron_z_advection!")

    # add the contribution from the wpa advection term
    @views electron_vpa_advection!(f_out, f_in, moments.electron.dens[:,ir],
                                   moments.electron.upar[:,ir], p_in, moments,
                                   composition, vpa_advect, vpa, vpa_spectral,
                                   scratch_dummy, dt, external_source_settings.electron,
                                   ir)
    write_debug_IO("electron_vpa_advection!")

    # add in the contribution to the residual from the term proportional to the pdf
    add_contribution_from_pdf_term!(f_out, f_in, p_in, moments.electron.dens[:,ir],
                                    moments.electron.upar[:,ir], moments, vpa.grid, z, dt,
                                    external_source_settings.electron, ir)
    write_debug_IO("add_contribution_from_pdf_term!")

    # add in numerical dissipation terms
    add_dissipation_term!(f_out, f_in, scratch_dummy, z_spectral, z, vpa, vpa_spectral,
                          num_diss_params, dt)
    write_debug_IO("add_dissipation_term!")

    if collisions.krook.nuee0 > 0.0 || collisions.krook.nuei0 > 0.0
        # Add a Krook collision operator
        # Set dt=-1 as we update the residual here rather than adding an update to
        # 'fvec_out'.
        @views electron_krook_collisions!(f_out, f_in, moments.electron.dens[:,ir],
                                          moments.electron.upar[:,ir],
                                          moments.ion.upar[:,ir],
                                          moments.electron.vth[:,ir], collisions, vperp,
                                          vpa, dt)
        write_debug_IO("electron_krook_collisions!")
    end

    @views total_external_electron_sources!(f_out, f_in, moments.electron.dens[:,ir],
                                            moments.electron.upar[:,ir], moments,
                                            composition, external_source_settings.electron,
                                            vperp, vpa, dt, ir)
    write_debug_IO("total_external_electron_sources!")

    if soft_force_constraints
        electron_implicit_constraint_forcing!(f_out, f_in,
                                              t_params.constraint_forcing_rate, vpa, dt,
                                              ir)
        write_debug_IO("electron_implicit_constraint_forcing!")
    end

    return nothing
end

"""
    fill_electron_kinetic_equation_Jacobian!(jacobian_matrix, f, p, moments, collisions,
                                             composition, z, vperp, vpa, z_spectral,
                                             vperp_specral, vpa_spectral, z_advect,
                                             vpa_advect, scratch_dummy,
                                             external_source_settings, num_diss_params,
                                             t_params, ion_dt, ir, evolve_p, include=:all)

Fill a pre-allocated matrix with the Jacobian matrix for electron kinetic equation and (if
`evolve_p=true`) the electron energy equation.

`add_identity=false` can be passed to skip adding 1's on the diagonal. Doing this would
produce the Jacobian for a steady-state solve, rather than a backward-Euler timestep
solve. [Note that rows representing boundary points still need the 1 added to their
diagonal, as that 1 is used to impose the boundary condition, not to represent the 'new f'
in the time derivative term as it is for the non-boundary points.]
"""
@timeit global_timer fill_electron_kinetic_equation_Jacobian!(
                         jacobian_matrix, f, p, moments, phi_global, collisions,
                         composition, z, vperp, vpa, z_spectral, vperp_spectral,
                         vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                         external_source_settings, num_diss_params, t_params, ion_dt, ir,
                         evolve_p, include=:all, include_qpar_integral_terms=true,
                         add_identity=true) = begin
    dt = t_params.dt[]

    buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

    vth = @view moments.electron.vth[:,ir]
    me = composition.me_over_mi
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    ppar = @view moments.electron.ppar[:,ir]
    qpar = @view moments.electron.qpar[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dupar_dz = @view moments.electron.dupar_dz[:,ir]
    dp_dz = @view moments.electron.dp_dz[:,ir]
    dppar_dz = @view moments.electron.dppar_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
    phi = @view phi_global[:,ir]

    upar_ion = @view moments.ion.upar[:,ir,1]

    # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
    third_moment = scratch_dummy.buffer_z_1
    dthird_moment_dz = scratch_dummy.buffer_z_2
    @begin_z_region()
    @loop_z iz begin
        third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
    end
    derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                  buffer_3, buffer_4, z_spectral, z)

    pdf_size = z.n * vperp.n * vpa.n
    v_size = vperp.n * vpa.n

    z_speed = @view z_advect[1].speed[:,:,:,ir]

    # Initialise jacobian_matrix to the identity
    @begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa

        jacobian_matrix[row,:] .= 0.0
        if (include === :all
            && (add_identity
                || skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa,
                                                         z_speed))
           )
            jacobian_matrix[row,row] += 1.0
        end
    end
    @begin_z_region()
    @loop_z iz begin
        # Rows corresponding to electron_p
        row = pdf_size + iz

        jacobian_matrix[row,:] .= 0.0
        if include === :all && add_identity
            jacobian_matrix[row,row] += 1.0
        end
    end

    if include ∈ (:all, :explicit_v)
        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        @loop_vperp_vpa ivperp ivpa begin
            @views z_advect[1].adv_fac[:,ivpa,ivperp,ir] = -z_speed[:,ivpa,ivperp]
        end
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[1].adv_fac[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)
    else
        dpdf_dz = nothing
    end

    dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
    @begin_z_vperp_region()
    update_electron_speed_vpa!(vpa_advect[1], dens, upar, p, moments,
                               composition.me_over_mi, vpa.grid,
                               external_source_settings.electron, ir)
    @loop_z_vperp iz ivperp begin
        @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
    end
    #calculate the upwind derivative of the electron pdf w.r.t. wpa
    @loop_z_vperp iz ivperp begin
        @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                           vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
    end

    zeroth_moment = z.scratch_shared
    first_moment = z.scratch_shared2
    second_moment = z.scratch_shared3
    @begin_z_region()
    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts
    @loop_z iz begin
        @views zeroth_moment[iz] = integral(f[:,1,iz], vpa_wgts)
        @views first_moment[iz] = integral(f[:,1,iz], vpa_grid, vpa_wgts)
        @views second_moment[iz] = integral(f[:,1,iz], vpa_grid, 2, vpa_wgts)
    end

    add_electron_z_advection_to_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, dpdf_dz, me, z, vperp, vpa, z_spectral,
        z_advect, z_speed, scratch_dummy, dt, ir, include; p_offset=pdf_size)
    add_electron_vpa_advection_to_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, dpdf_dvpa, ddens_dz,
        dp_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp, vpa, z_spectral,
        vpa_spectral, vpa_advect, z_speed, scratch_dummy, external_source_settings, dt,
        ir, include, include_qpar_integral_terms; p_offset=pdf_size)
    add_contribution_from_electron_pdf_term_to_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, ddens_dz, dp_dz, dvth_dz,
        dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z, vperp, vpa,
        z_spectral, z_speed, scratch_dummy, dt, ir, include, include_qpar_integral_terms;
        p_offset=pdf_size)
    add_electron_dissipation_term_to_Jacobian!(
        jacobian_matrix, f, num_diss_params, z, vperp, vpa, vpa_spectral, z_speed, dt, ir,
        include)
    add_electron_krook_collisions_to_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, upar_ion, collisions, z, vperp, vpa,
        z_speed, dt, ir, include; p_offset=pdf_size)
    add_total_external_electron_source_to_Jacobian!(
        jacobian_matrix, f, moments, me, z_speed, external_source_settings.electron, z,
        vperp, vpa, dt, ir, include; p_offset=pdf_size)
    add_electron_implicit_constraint_forcing_to_Jacobian!(
        jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z, vperp,
        vpa, t_params.constraint_forcing_rate, dt, ir, include)
    # Always add the electron energy equation term, even if evolve_p=false, so that the
    # Jacobian matrix always has the same shape, meaning that we can always reuse the LU
    # factorization struct.
    add_electron_energy_equation_to_Jacobian!(
        jacobian_matrix, f, dens, upar, p, ppar, vth, third_moment, ddens_dz, dupar_dz,
        dp_dz, dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
        num_diss_params, dt, ir, include; p_offset=pdf_size)
    if ion_dt !== nothing
        add_ion_dt_forcing_of_electron_p_to_Jacobian!(
            jacobian_matrix, z, dt, ion_dt, ir, include; p_offset=pdf_size)
    end
    if t_params.include_wall_bc_in_preconditioner
        add_wall_boundary_condition_to_Jacobian!(
            jacobian_matrix, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral,
            vpa_spectral, vpa_advect, moments,
            num_diss_params.electron.vpa_dissipation_coefficient, me, ir, include;
            p_offset=pdf_size)
    end

    return nothing
end

"""
    fill_electron_kinetic_equation_v_only_Jacobian!()
        jacobian_matrix, f, p, dpdf_dz, dpdf_dvpa, z_speed, moments, zeroth_moment,
        first_moment, second_moment, third_moment, dthird_moment_dz, phi, collisions,
        composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
        vpa_advect, scratch_dummy, external_source_settings, num_diss_params, t_params,
        ion_dt, ir, iz, evolve_p)

Fill a pre-allocated matrix with the Jacobian matrix for a velocity-space solve part of
the ADI method for electron kinetic equation and (if `evolve_p=true`) the electron energy
equation.
"""
@timeit global_timer fill_electron_kinetic_equation_v_only_Jacobian!(
                         jacobian_matrix, f, p, dpdf_dz, dpdf_dvpa, z_speed, moments,
                         zeroth_moment, first_moment, second_moment, third_moment,
                         dthird_moment_dz, phi, collisions, composition, z, vperp, vpa,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, iz, evolve_p) = begin
    dt = t_params.dt[]

    vth = moments.electron.vth[iz,ir]
    me = composition.me_over_mi
    dens = moments.electron.dens[iz,ir]
    upar = moments.electron.upar[iz,ir]
    ppar = moments.electron.ppar[iz,ir]
    qpar = moments.electron.qpar[iz,ir]
    ddens_dz = moments.electron.ddens_dz[iz,ir]
    dupar_dz = moments.electron.dupar_dz[iz,ir]
    dp_dz = moments.electron.dp_dz[iz,ir]
    dppar_dz = moments.electron.dppar_dz[iz,ir]
    dvth_dz = moments.electron.dvth_dz[iz,ir]
    dqpar_dz = moments.electron.dqpar_dz[iz,ir]

    upar_ion = moments.ion.upar[iz,ir,1]

    v_size = vperp.n * vpa.n

    # Initialise jacobian_matrix to the identity
    for row ∈ 1:size(jacobian_matrix, 1)
        jacobian_matrix[row,:] .= 0.0
        jacobian_matrix[row,row] += 1.0
    end

    add_electron_z_advection_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, dpdf_dz, me, z, vperp, vpa, z_spectral,
        z_advect, z_speed, scratch_dummy, dt, ir, iz)
    add_electron_vpa_advection_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, dpdf_dvpa, ddens_dz,
        dp_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp, vpa, z_spectral,
        vpa_spectral, vpa_advect, z_speed, scratch_dummy, external_source_settings, dt,
        ir, iz)
    add_contribution_from_electron_pdf_term_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, ddens_dz, dp_dz, dvth_dz,
        dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z, vperp, vpa,
        z_spectral, z_speed, scratch_dummy, dt, ir, iz)
    add_electron_dissipation_term_to_v_only_Jacobian!(
        jacobian_matrix, f, num_diss_params, z, vperp, vpa, vpa_spectral, z_speed, dt, ir,
        iz)
    add_electron_krook_collisions_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, upar_ion, collisions, z, vperp, vpa,
        z_speed, dt, ir, iz)
    add_total_external_electron_source_to_v_only_Jacobian!(
        jacobian_matrix, f, moments, me, z_speed, external_source_settings.electron, z,
        vperp, vpa, dt, ir, iz)
    add_electron_implicit_constraint_forcing_to_v_only_Jacobian!(
        jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z, vperp,
        vpa, t_params.constraint_forcing_rate, dt, ir, iz)
    # Always add the electron energy equation term, even if evolve_p=false, so that the
    # Jacobian matrix always has the same shape, meaning that we can always reuse the LU
    # factorization struct.
    add_electron_energy_equation_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, ppar, vth, third_moment, ddens_dz, dupar_dz,
        dp_dz, dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
        num_diss_params, dt, ir, iz)
    if ion_dt !== nothing
        add_ion_dt_forcing_of_electron_p_to_v_only_Jacobian!(
            jacobian_matrix, z, dt, ion_dt, ir, iz)
    end

    if t_params.include_wall_bc_in_preconditioner
        add_wall_boundary_condition_to_Jacobian!(
            jacobian_matrix, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral,
            vpa_spectral, vpa_advect, moments,
            num_diss_params.electron.vpa_dissipation_coefficient, me, ir, :implicit_v, iz;
            p_offset=v_size)
    end

    return nothing
end

"""
    fill_electron_kinetic_equation_z_only_Jacobian_f!(
        jacobian_matrix, f, p, dpdf_dz, dpdf_dvpa, z_speed, moments, zeroth_moment,
        first_moment, second_moment, third_moment, dthird_moment_dz, collisions,
        composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
        vpa_advect, scratch_dummy, external_source_settings, num_diss_params, t_params,
        ion_dt, ir, ivperp, ivpa, evolve_p)

Fill a pre-allocated matrix with the Jacobian matrix for a z-direction solve part of the
ADI method for electron kinetic equation and (if `evolve_p=true`) the electron energy
equation.
"""
@timeit global_timer fill_electron_kinetic_equation_z_only_Jacobian_f!(
                         jacobian_matrix, f, p, dpdf_dz, dpdf_dvpa, z_speed, moments,
                         zeroth_moment, first_moment, second_moment, third_moment,
                         dthird_moment_dz, collisions, composition, z, vperp, vpa,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, ivperp, ivpa, evolve_p) = begin
    dt = t_params.dt[]

    vth = @view moments.electron.vth[:,ir]
    me = composition.me_over_mi
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    qpar = @view moments.electron.qpar[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dupar_dz = @view moments.electron.dupar_dz[:,ir]
    dppar_dz = @view moments.electron.dppar_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]

    upar_ion = @view moments.ion.upar[:,ir,1]

    pdf_size = z.n * vperp.n * vpa.n
    v_size = vperp.n * vpa.n

    # Initialise jacobian_matrix to the identity
    for row ∈ 1:size(jacobian_matrix, 1)
        jacobian_matrix[row,:] .= 0.0
        jacobian_matrix[row,row] += 1.0
    end

    add_electron_z_advection_to_z_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, dpdf_dz, me, z, vperp, vpa, z_spectral,
        z_advect, z_speed, scratch_dummy, dt, ir, ivperp, ivpa)
    add_contribution_from_electron_pdf_term_to_z_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, ddens_dz, dppar_dz, dvth_dz,
        dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z,
        vperp, vpa, z_spectral, z_speed, scratch_dummy, dt, ir, ivperp, ivpa)
    add_electron_krook_collisions_to_z_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, upar_ion, collisions, z, vperp, vpa,
        z_speed, dt, ir, ivperp, ivpa)
    add_total_external_electron_source_to_z_only_Jacobian!(
        jacobian_matrix, f, moments, me, z_speed, external_source_settings.electron, z,
        vperp, vpa, dt, ir, ivperp, ivpa)
    add_electron_implicit_constraint_forcing_to_z_only_Jacobian!(
        jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z, vperp,
        vpa, t_params.constraint_forcing_rate, dt, ir, ivperp, ivpa)

    return nothing
end

"""
    fill_electron_kinetic_equation_z_only_Jacobian_p!(
        jacobian_matrix, p, moments, zeroth_moment, first_moment, second_moment,
        third_moment, dthird_moment_dz, collisions, composition, z, vperp, vpa,
        z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
        external_source_settings, num_diss_params, t_params, ion_dt, ir, evolve_p)

Fill a pre-allocated matrix with the Jacobian matrix for a z-direction solve part of the
ADI method for electron kinetic equation and (if `evolve_p=true`) the electron energy
equation.
"""
@timeit global_timer fill_electron_kinetic_equation_z_only_Jacobian_p!(
                         jacobian_matrix, p, moments, zeroth_moment, first_moment,
                         second_moment, third_moment, dthird_moment_dz, collisions,
                         composition, z, vperp, vpa, z_spectral, vperp_spectral,
                         vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                         external_source_settings, num_diss_params, t_params, ion_dt, ir,
                         evolve_p) = begin
    dt = t_params.dt[]

    vth = @view moments.electron.vth[:,ir]
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    ppar = @view moments.electron.ppar[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dupar_dz = @view moments.electron.dupar_dz[:,ir]
    dp_dz = @view moments.electron.dp_dz[:,ir]

    pdf_size = z.n * vperp.n * vpa.n

    # Initialise jacobian_matrix to the identity
    for row ∈ 1:size(jacobian_matrix, 1)
        jacobian_matrix[row,:] .= 0.0
        jacobian_matrix[row,row] += 1.0
    end

    add_electron_energy_equation_to_z_only_Jacobian!(
        jacobian_matrix, dens, upar, p, ppar, vth, third_moment, ddens_dz, dupar_dz,
        dp_dz, dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
        num_diss_params, dt, ir)
    if ion_dt !== nothing
        add_ion_dt_forcing_of_electron_p_to_z_only_Jacobian!(
            jacobian_matrix, z, dt, ion_dt, ir)
    end

    return nothing
end

"""
    get_electron_split_Jacobians!(ivperp, ivpa, p, moments, collisions, composition, z,
                                  vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
                                  z_advect, vpa_advect, scratch_dummy,
                                  external_source_settings, num_diss_params, t_params,
                                  ion_dt, ir, evolve_p)

Fill a pre-allocated matrix with the Jacobian matrix for electron kinetic equation and (if
`evolve_p=true`) the electron energy equation.
"""
@timeit global_timer get_electron_split_Jacobians!(
                         ivperp, ivpa, p, moments, collisions, composition, z, vperp, vpa,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, evolve_p) = begin

    dt = t_params.dt[]

    z_matrix = allocate_float(z.n, z.n)
    z_matrix .= 0.0

    z_speed = @view z_advect[1].speed[:,ivpa,ivperp,ir]
    for ielement ∈ 1:z.nelement_local
        imin = z.imin[ielement] - (ielement != 1)
        imax = z.imax[ielement]
        if ielement == 1
            z_matrix[imin,imin:imax] .+= z_spectral.lobatto.Dmat[1,:] ./ z.element_scale[ielement]
        else
            if z_speed[imin] < 0.0
                z_matrix[imin,imin:imax] .+= z_spectral.lobatto.Dmat[1,:] ./ z.element_scale[ielement]
            elseif z_speed[imin] > 0.0
                # Do nothing
            else
                z_matrix[imin,imin:imax] .+= 0.5 .* z_spectral.lobatto.Dmat[1,:] ./ z.element_scale[ielement]
            end
        end
        z_matrix[imin+1:imax-1,imin:imax] .+= z_spectral.lobatto.Dmat[2:end-1,:] ./ z.element_scale[ielement]
        if ielement == z.nelement_local
            z_matrix[imax,imin:imax] .+= z_spectral.lobatto.Dmat[end,:] ./ z.element_scale[ielement]
        else
            if z_speed[imax] < 0.0
                # Do nothing
            elseif z_speed[imax] > 0.0
                z_matrix[imax,imin:imax] .+= z_spectral.lobatto.Dmat[end,:] ./ z.element_scale[ielement]
            else
                z_matrix[imax,imin:imax] .+= 0.5 .* z_spectral.lobatto.Dmat[end,:] ./ z.element_scale[ielement]
            end
        end
    end
    # Multiply by advection speed
    for row ∈ 1:z.n
        z_matrix[row,:] .*= dt * z_speed[row]
    end

    # Diagonal entries
    for row ∈ 1:z.n
        z_matrix[row,row] += 1.0

        # Terms from `add_contribution_from_pdf_term!()`
        z_matrix[row,row] += dt * (dqpar_dz[row] / 3.0 / ppar[row]
                                   + vpa.grid[ivpa] * (vth[row] / dens[row] * ddens_dz[row]
                                                       - dvth_dz[row]))
    end
    if external_source_settings.electron.active
        for row ∈ 1:z.n
            # Source terms from `add_contribution_from_pdf_term!()`
            z_matrix[row,row] += dt * (1.5 * source_density_amplitude[row] / dens[row]
                                       - 0.5 * source_pressure_amplitude[row] / p[row])
        end
        if external_source_settings.electron.source_type == "energy"
            for row ∈ 1:z.n
                # Contribution from `external_electron_source!()`
                z_matrix[row,row] += dt * source_amplitude[row]
            end
        end
    end
    if collisions.krook.nuee0 > 0.0 || collisions.krook.nuei0 > 0.0
        for row ∈ 1:z.n
            # Contribution from electron_krook_collisions!()
            nu_ee = get_collision_frequency_ee(collisions, dens[row], vth[row])
            nu_ei = get_collision_frequency_ei(collisions, dens[row], vth[row])
            z_matrix[row,row] += dt * (nu_ee + nu_ei)
        end
    end

    if z.irank == 0 && ivperp == 1 && ivpa == 1
        p_matrix = allocate_float(z.n, z.n)
        p_matrix .= 0.0

        if composition.electron_physics == kinetic_electrons_with_temperature_equation
            error("kinetic_electrons_with_temperature_equation not "
                  * "supported yet in preconditioner")
        elseif composition.electron_physics != kinetic_electrons
            error("Unsupported electron_physics=$(composition.electron_physics) "
                  * "in electron_backward_euler!() preconditioner.")
        end

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        @views third_moment = @. moments.electron.qpar[:,ir] / electron_p_new / vth

        # Note that as
        #   qpar = p * vth * third_moment
        #        = sqrt(2) * p^(3/2) / dens^(1/2) / me^(1/2) * third_moment
        # we have that
        #   d(qpar)/dz = sqrt(2) * p^(3/2) / dens^(1/2) / me^(1/2) * d(third_moment)/dz
        #                - 1/sqrt(2) * p^(3/2) / dens^(3/2) / me^(1/2) * third_moment * d(dens)/dz
        #                + 3/sqrt(2) * p^(1/2) / dens^(1/2) / me^(1/2) * third_moment * d(p)/dz
        # so for the Jacobian
        #   d[d(qpar)/dz)]/d[p]
        #     = 3/sqrt(2) * p^(1/2) / dens^(1/2) / me^(1/2) * d(third_moment)/dz
        #       - 3/2/sqrt(2) * p^(1/2) / dens^(3/2) / me^(1/2) * third_moment * d(dens)/dz
        #       + 3/2/sqrt(2) / p^(1/2) / dens^(1/2) / me^(1/2) * third_moment * d(p)/dz
        #       + 3/sqrt(2) * p^(1/2) / dens^(1/2) / me^(1/2) * third_moment * d(.)/dz
        dthird_moment_dz = z.scratch2
        derivative_z!(z.scratch2, third_moment, buffer_1, buffer_2,
                      buffer_3, buffer_4, z_spectral, z)

        # Diagonal terms
        for row ∈ 1:z.n
            p_matrix[row,row] = 1.0

            # 2/3*ppar*dupar_dz+p*dupar_dz
            # = 2*p*dupar_dz+p*dupar_dz
            # = 3*p*dupar_dz
            p_matrix[row,row] += 3.0 * dt * dupar_dz[row]

            # terms from d(qpar)/dz
            p_matrix[row,row] +=
                dt / sqrt(2.0) * (3.0 * sqrt(electron_p_new[row] / dens[row] / me) * dthird_moment_dz[row]
                                  - 1.5 * sqrt(electron_p_new[row] / me) / dens[row]^1.5 * third_moment[row] * ddens_dz[row]
                                  + 1.5 / sqrt(electron_p_new[row] / dens[row] / me) * third_moment[row] * dp_dz[row])
        end
        if ion_dt !== nothing
            # Backward-Euler forcing term
            for row ∈ 1:z.n
                p_matrix[row,row] += dt / ion_dt
            end
        end

        # d(.)/dz terms
        # Note that the z-derivative matrix is local to this block, and
        # for the preconditioner we do not include any distributed-MPI
        # communication (we rely on the JFNK iteration to sort out the
        # coupling between blocks).
        if !isa(z_spectral, gausslegendre_info)
            error("Only gausslegendre_pseudospectral coordinate type is "
                  * "supported by electron_backward_euler!() "
                  * "preconditioner because we need differentiation"
                  * "matrices.")
        end
        z_deriv_matrix = z_spectral.D_matrix
        for row ∈ 1:z.n
            @. p_matrix[row,:] +=
                dt * (upar[row]
                      + 3.0 / sqrt(2.0) * sqrt(electron_p_new[row] / dens[row] / me) * third_moment[row]) *
                z_deriv_matrix[row,:]
        end

        if num_diss_params.electron.moment_dissipation_coefficient > 0.0
            error("z-diffusion of electron_p not yet supported in "
                  * "preconditioner")
        end
        if collisions.nu_ei > 0.0
            error("electron-ion collision terms for electron_p not yet "
                  * "supported in preconditioner")
        end
        if composition.n_neutral_species > 0 && collisions.charge_exchange_electron > 0.0
            error("electron 'charge exchange' terms for electron_p not yet "
                  * "supported in preconditioner")
        end
        if composition.n_neutral_species > 0 && collisions.ionization_electron > 0.0
            error("electron ionization terms for electron_p not yet "
                  * "supported in preconditioner")
        end
    else
        p_matrix = allocate_float(0, 0)
        p_matrix[] = 1.0
    end

    return z_matrix, p_matrix
end

#"""
#electron_kinetic_equation_residual! calculates the residual of the (time-independent) electron kinetic equation
#INPUTS:
#    residual = dummy array to be filled with the residual of the electron kinetic equation
#OUTPUT:
#    residual = updated residual of the electron kinetic equation
#"""
#function electron_kinetic_equation_residual!(residual, max_term, single_term, pdf, dens, upar, vth, p, upar_ion,
#                                             ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                             z, vperp, vpa, z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
#                                             collisions, external_source_settings,
#                                             num_diss_params, dt_electron)
#
#    # initialise the residual to zero                                             
#    @begin_r_vperp_vpa_region()
#    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
#        residual[ivpa,ivperp,iz,ir] = 0.0
#    end
#    # calculate the contribution to the residual from the z advection term
#    electron_z_advection!(residual, pdf, upar, vth, z_advect, z, vpa.grid, z_spectral, scratch_dummy, -1.0)
#    #dt_max_zadv = simple_z_advection!(residual, pdf, vth, z, vpa.grid, dt_electron)
#    #single_term .= residual
#    #max_term .= abs.(residual)
#    #println("z_adv residual = ", maximum(abs.(single_term)))
#    #println("z_advection: ", sum(residual), " dqpar_dz: ", sum(abs.(dqpar_dz)))
#    #calculate_contribution_from_z_advection!(residual, pdf, vth, z, vpa.grid, z_spectral, scratch_dummy)
#    # add in the contribution to the residual from the wpa advection term
#    electron_vpa_advection!(residual, pdf, p, vth, dppar_dz, dqpar_dz, dvth_dz, 
#                            vpa_advect, vpa, vpa_spectral, scratch_dummy, -1.0,
#                            external_source_settings.electron)
#    #dt_max_vadv = simple_vpa_advection!(residual, pdf, p, vth, dppar_dz, dqpar_dz, dvth_dz, vpa, dt_electron)
#    #@. single_term = residual - single_term
#    #max_term .= max.(max_term, abs.(single_term))
#    #@. single_term = residual
#    #println("v_adv residual = ", maximum(abs.(single_term)))
#    #add_contribution_from_wpa_advection!(residual, pdf, vth, p, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
#    # add in the contribution to the residual from the term proportional to the pdf
#    add_contribution_from_pdf_term!(residual, pdf, p, dens, moments, vpa.grid, z, -1.0,
#                                    external_source_settings.electron)
#    #@. single_term = residual - single_term
#    #max_term .= max.(max_term, abs.(single_term))
#    #@. single_term = residual
#    #println("pdf_term residual = ", maximum(abs.(single_term)))
#    # @loop_vpa ivpa begin
#    #     @loop_z iz begin
#    #         println("LHS: ", residual[ivpa,1,iz,1], " vpa: ", vpa.grid[ivpa], " z: ", z.grid[iz], " dvth_dz: ", dvth_dz[iz,1], " type: ", 1) 
#    #     end
#    #     println("")
#    # end
#    # println("")
#    # add in numerical dissipation terms
#    add_dissipation_term!(residual, pdf, scratch_dummy, z_spectral, z, vpa, vpa_spectral,
#                          num_diss_params, -1.0)
#    #@. single_term = residual - single_term
#    #println("dissipation residual = ", maximum(abs.(single_term)))
#    #max_term .= max.(max_term, abs.(single_term))
#    # add in particle and heat source term(s)
#    #@. single_term = residual
#    #add_source_term!(residual, vpa.grid, z.grid, dvth_dz)
#    #@. single_term = residual - single_term
#    #max_term .= max.(max_term, abs.(single_term))
#    #stop()
#    # @loop_vpa ivpa begin
#    #     @loop_z iz begin
#    #         println("total_residual: ", residual[ivpa,1,iz,1], " vpa: ", vpa.grid[ivpa], " z: ", z.grid[iz], " dvth_dz: ", dvth_dz[iz,1], " type: ", 2) 
#    #     end
#    #     println("")
#    # end
#    # stop()
#    #dt_max = min(dt_max_zadv, dt_max_vadv)
#
#    if collisions.krook_collision_frequency_prefactor_ee > 0.0
#        # Add a Krook collision operator
#        # Set dt=-1 as we update the residual here rather than adding an update to
#        # 'fvec_out'.
#        electron_krook_collisions!(residual, pdf, dens, upar, upar_ion, vth,
#                                   collisions, vperp, vpa, -1.0)
#    end
#
#    dt_max = dt_electron
#    #println("dt_max: ", dt_max, " dt_max_zadv: ", dt_max_zadv, " dt_max_vadv: ", dt_max_vadv)
#    return dt_max
#end

function simple_z_advection!(advection_term, pdf, vth, z, vpa, dt_max_in)
    dt_max = dt_max_in
    # take the z derivative of the input pdf
    @begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        speed = vth[iz,ir] * vpa[ivpa]
        dt_max = min(dt_max, 0.5*z.cell_width[iz]/(max(abs(speed),1e-3)))
        if speed > 0
            if iz == 1
                #dpdf_dz  = pdf[ivpa,ivperp,iz,ir]/z.cell_width[1]
                dpdf_dz = 0.0
            else
                dpdf_dz = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa,ivperp,iz-1,ir])/z.cell_width[iz-1]
            end
        elseif speed < 0
            if iz == z.n
                #dpdf_dz = -pdf[ivpa,ivperp,iz,ir]/z.cell_width[z.n-1]
                dpdf_dz = 0.0
            else
                dpdf_dz = (pdf[ivpa,ivperp,iz+1,ir] - pdf[ivpa,ivperp,iz,ir])/z.cell_width[iz]
            end
        else
            if iz == 1
                dpdf_dz = (pdf[ivpa,ivperp,iz+1,ir] - pdf[ivpa,ivperp,iz,ir])/z.cell_width[1]
            elseif iz == z.n
                dpdf_dz = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa,ivperp,iz-1,ir])/z.cell_width[z.n-1]
            else
                dpdf_dz = (pdf[ivpa,ivperp,iz+1,ir] - pdf[ivpa,ivperp,iz,ir])/z.cell_width[1]
            end
        end
        advection_term[ivpa,ivperp,iz,ir] += speed * dpdf_dz
    end
    return dt_max
end

#################
# NEED TO ADD PARTICLE/HEAT SOURCE INTO THE ELECTRON KINETIC EQUATION TO FIX amplitude
#################


function simple_vpa_advection!(advection_term, pdf, p, vth, dppar_dz, dqpar_dz, dvth_dz, vpa, dt_max_in)
    dt_max = dt_max_in
    # take the vpa derivative in the input pdf
    @begin_r_z_vperp_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        speed = ((0.5 * vth[iz,ir] * dppar_dz[iz,ir] + vpa.grid[ivpa] / 3.0 * dqpar_dz[iz,ir]) 
                / p[iz,ir] - vpa.grid[ivpa]^2 * dvth_dz[iz,ir])
        dt_max = min(dt_max, 0.5*vpa.cell_width[ivpa]/max(abs(speed),1e-3))
        if speed > 0
            if ivpa == 1
                dpdf_dvpa  = pdf[ivpa,ivperp,iz,ir]/vpa.cell_width[1]
            else
                dpdf_dvpa = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa-1,ivperp,iz,ir])/vpa.cell_width[ivpa-1]
            end
        elseif speed < 0
            if ivpa == vpa.n
                dpdf_dvpa = -pdf[ivpa,ivperp,iz,ir]/vpa.cell_width[vpa.n-1]
            else
                dpdf_dvpa = (pdf[ivpa+1,ivperp,iz,ir] - pdf[ivpa,ivperp,iz,ir])/vpa.cell_width[ivpa]
            end
        else
            if ivpa == 1
                dpdf_dvpa = (pdf[ivpa+1,ivperp,iz,ir] - pdf[ivpa,ivperp,iz,ir])/vpa.cell_width[1]
            elseif ivpa == vpa.n
                dpdf_dvpa = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa-1,ivperp,iz,ir])/vpa.cell_width[vpa.n-1]
            else
                dpdf_dvpa = (pdf[ivpa+1,ivperp,iz,ir] - pdf[ivpa,ivperp,iz,ir])/vpa.cell_width[1]
            end
        end
        advection_term[ivpa,ivperp,iz,ir] += speed * dpdf_dvpa
    end
    return dt_max
end

function add_source_term!(source_term, vpa, z, dvth_dz)
    # add in particle and heat source term(s)
    @begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
    #    source_term[ivpa,ivperp,iz,ir] -= 40*exp(-vpa[ivpa]^2)*exp(-z[iz]^2)
    #    source_term[ivpa,ivperp,iz,ir] -= vpa[ivpa]*exp(-vpa[ivpa]^2)*(2*vpa[ivpa]^2-3)*dvth_dz[iz,ir]
    end
    return nothing
end

"""
"""
@timeit global_timer add_dissipation_term!(pdf_out, pdf_in, scratch_dummy, z_spectral, z,
                                           vpa, vpa_spectral, num_diss_params, dt) = begin
    if num_diss_params.electron.vpa_dissipation_coefficient ≤ 0.0
        return nothing
    end

    @begin_z_vperp_region()
    @loop_z_vperp iz ivperp begin
        @views second_derivative!(vpa.scratch, pdf_in[:,ivperp,iz], vpa, vpa_spectral)
        @views @. pdf_out[:,ivperp,iz] += dt * num_diss_params.electron.vpa_dissipation_coefficient * vpa.scratch
    end
    return nothing
end

function add_electron_dissipation_term_to_Jacobian!(jacobian_matrix, f, num_diss_params,
                                                    z, vperp, vpa, vpa_spectral, z_speed,
                                                    dt, ir, include=:all; f_offset=0)
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @boundscheck include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    vpa_dissipation_coefficient = num_diss_params.electron.vpa_dissipation_coefficient

    if vpa_dissipation_coefficient ≤ 0.0
        return nothing
    end

    v_size = vperp.n * vpa.n
    vpa_dense_second_deriv_matrix = vpa_spectral.dense_second_deriv_matrix

    @begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

        # Terms from add_dissipation_term!()
        if include ∈ (:all, :explicit_v)
            for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                col = (iz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                jacobian_matrix[row,col] -= dt * vpa_dissipation_coefficient * vpa_dense_second_deriv_matrix[ivpa,icolvpa]
            end
        end
    end

    return nothing
end

function add_electron_dissipation_term_to_v_only_Jacobian!(
        jacobian_matrix, f, num_diss_params, z, vperp, vpa, vpa_spectral, z_speed, dt, ir,
        iz)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    vpa_dissipation_coefficient = num_diss_params.electron.vpa_dissipation_coefficient

    if vpa_dissipation_coefficient ≤ 0.0
        return nothing
    end

    vpa_dense_second_deriv_matrix = vpa_spectral.dense_second_deriv_matrix

    @loop_vperp_vpa ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (ivperp - 1) * vpa.n + ivpa

        # Terms from add_dissipation_term!()
        for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
            col = (icolvperp - 1) * vpa.n + icolvpa
            jacobian_matrix[row,col] -= dt * vpa_dissipation_coefficient * vpa_dense_second_deriv_matrix[ivpa,icolvpa]
        end
    end

    return nothing
end

"""
update_electron_pdf! iterates to find a solution for the electron pdf
from the electron kinetic equation:
    zdot * d(pdf^{i})/dz + wpadot^{i} * d(pdf^{i})/dwpa + wpedot^{i} * d(pdf^{i})/dwpe  = pdf^{i+1} * pre_factor^{i}
NB: zdot, wpadot, wpedot and pre-factor contain electron fluid quantities, that will have been updated
independently of the electron pdf;
NB: wpadot, wpedot and pre-factor all contain the electron parallel heat flux,
which itself depends on the electron pdf.

INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
OUTPUT:
    pdf = updated (modified) electron pdf
"""
# function update_electron_pdf!(pdf, dens, vthe, p, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                               max_electron_pdf_iterations, z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#     # iterate the electron kinetic equation until the electron pdf is converged
#     # or the specified maximum number of iterations is exceeded
#     electron_pdf_converged = false
#     iteration = 1

#     println("pdf_update_electron_pdf: ", sum(abs.(pdf)))
#     while !electron_pdf_converged && (iteration < max_electron_pdf_iterations)
#         # calculate the contribution to the kinetic equation due to z advection
#         # and store in the dummy array scratch_dummy.buffer_vpavperpzr_1
#         @views calculate_contribution_from_z_advection!(scratch_dummy.buffer_vpavperpzr_1, pdf, vthe, z, vpa.grid, z_spectral, scratch_dummy)
#         # calculate the contribution to the kinetic equation due to w_parallel advection and add to the z advection term
#         @views add_contribution_from_wpa_advection!(scratch_dummy.buffer_vpavperpzr_1, pdf, vthe, p, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
#         # calculate the pre-factor multiplying the modified electron pdf
#         # and store in the dummy array scratch_dummy.buffer_vpavperpzr_2
#         @views calculate_pdf_dot_prefactor!(scratch_dummy.buffer_vpavperpzr_2, p, vthe, dens, ddens_dz, dvth_dz, dqpar_dz, vpa.grid)
#         # update the electron pdf
#         #pdf_new = scratch_dummy.buffer_vpavperpzr_1
#         #println("advection_terms: ", sum(abs.(advection_terms)), "pdf_dot_prefactor: ", sum(abs.(pdf_dot_prefactor)))
#         @. scratch_dummy.buffer_vpavperpzr_1 /= scratch_dummy.buffer_vpavperpzr_2
#         # check to see if the electron pdf has converged to within the specified tolerance
#         check_electron_pdf_convergence!(electron_pdf_converged, scratch_dummy.buffer_vpavperpzr_1, pdf)
#         # prepare for the next iteration
#         @. pdf = scratch_dummy.buffer_vpavperpzr_1
#         iteration += 1
#     end
#     if !electron_pdf_converged
#         # need to exit or handle this appropriately
#         println("!!!max number of iterations for electron pdf update exceeded!!!")
#     end
#     return nothing
# end



# """
# use the biconjugate gradient stabilized method to solve the electron kinetic equation
# """
# function update_electron_pdf_bicgstab!(pdf, dens, vthe, p, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#     max_electron_pdf_iterations, z, vpa, z_spectral, vpa_spectral, scratch_dummy)
 
#     for iz in 1:z.n
#         println("z: ", z.grid[iz], " pdf: ", pdf[10, 1, iz, 1], " iteration: ", 1, " z.n: ", z.n)
#     end

#     # set various arrays to point to a corresponding dummy array scratch_dummy.buffer_vpavperpzr_X
#     # so that it is easier to understand what is going on
#     residual = scratch_dummy.buffer_vpavperpzr_1
#     residual_hat = scratch_dummy.buffer_vpavperpzr_2
#     pvec = scratch_dummy.buffer_vpavperpzr_3
#     vvec = scratch_dummy.buffer_vpavperpzr_4
#     tvec = scratch_dummy.buffer_vpavperpzr_5

#     # calculate the residual of the electron kinetic equation for the initial guess of the electron pdf
#     electron_kinetic_equation_residual!(residual, pdf, dens, vthe, p, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                         z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#     @. residual = -residual

#     for iz in 1:z.n
#         println("z: ", z.grid[iz], " residual: ", residual[10, 1, iz, 1], " iteration: ", -1, " z.n: ", z.n)
#     end

#     residual_hat .= residual + pdf

#     # calculate the inner product of the residual and residual_hat
#     rho = dot(residual_hat, residual)
#     #@views rho = integral(residual_hat[10, 1, :, 1], residual[10,1,:,1], z.wgts)

#     println("rho: ", rho)

#     pvec .= residual

#     # iterate the electron kinetic equation until the electron pdf is converged
#     # or the specified maximum number of iterations is exceeded
#     electron_pdf_converged = false
#     iteration = 1

#     while !electron_pdf_converged && (iteration < max_electron_pdf_iterations)
#         # calculate the residual of the electron kinetic equation with the residual of the previous iteration
#         # as the input electron pdf
#         electron_kinetic_equation_residual!(vvec, pvec, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                             z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#         alpha = rho / dot(residual_hat, vvec)
#         #alpha = rho / integral(residual_hat[10,1,:,1], vvec[10,1,:,1], z.wgts)
#         println("alpha: ", alpha)
#         @. pdf += alpha * pvec
#         @. residual -= alpha * vvec
#         # check to see if the electron pdf has converged to within the specified tolerance
#         check_electron_pdf_convergence!(electron_pdf_converged, residual)
#         if electron_pdf_converged
#             break
#         end
#         electron_kinetic_equation_residual!(tvec, residual, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                             z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#         omega = dot(tvec, residual) / dot(tvec, tvec)
#         #omega = integral(tvec[10,1,:,1],residual[10,1,:,1],z.wgts)/integral(tvec[10,1,:,1],tvec[10,1,:,1],z.wgts)
#         @. pdf += omega * residual
#         @. residual -= omega * tvec
#         check_electron_pdf_convergence!(electron_pdf_converged, residual)
#         if electron_pdf_converged
#             break
#         end
#         #rho_new = integral(residual_hat[10,1,:,1],residual[10,1,:,1],z.wgts)
#         rho_new = dot(residual_hat, residual)
#         beta = (rho_new / rho) * (alpha / omega)
#         println("rho_new: ", rho_new, " beta: ", beta)
#         #@. pvec -= omega * vvec
#         #@. pvec *= beta
#         #@. pvec += residual
#         @. pvec = residual + beta * (pvec - omega * vvec)
#         # prepare for the next iteration
#         rho = rho_new
#         iteration += 1
#         println()
#         if iteration == 2
#             for iz in 1:z.n
#                 println("z: ", z.grid[iz], " pvec: ", pvec[10, 1, iz, 1], " iteration: ", -2, " z.n: ", z.n)
#             end
#         end
#         for iz in 1:z.n
#             println("z: ", z.grid[iz], " pdf: ", pdf[10, 1, iz, 1], " iteration: ", iteration, " z.n: ", z.n)
#         end
#     end
#     if !electron_pdf_converged
#         # need to exit or handle this appropriately
#         println("!!!max number of iterations for electron pdf update exceeded!!!")
#     end
#     return nothing
# end

"""
calculates the contribution to the residual of the electron kinetic equation from the z advection term:
residual = zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa - pdf * pre_factor
INPUTS:
    z_advection_term = dummy array to be filled with the contribution to the residual from the z advection term
    pdf = modified electron pdf used in the kinetic equation = (true electron pdf / dens_e) * vth_e
    vthe = electron thermal speed
    z = z grid
    vpa = v_parallel grid
    z_spectral = spectral representation of the z grid
    scratch_dummy = dummy array to be used for temporary storage
OUTPUT:
    z_advection_term = updated contribution to the residual from the z advection term
"""
function calculate_contribution_from_z_advection!(z_advection_term, pdf, vthe, z, vpa, spectral, scratch_dummy)
    # calculate the z-derivative of the electron pdf and store in the dummy array
    # scratch_dummy.buffer_vpavperpzr_1
    derivative_z!(z_advection_term, pdf,
                  scratch_dummy.buffer_vpavperpr_1,
                  scratch_dummy.buffer_vpavperpr_2, scratch_dummy.buffer_vpavperpr_3,
                  scratch_dummy.buffer_vpavperpr_4, spectral, z)
    # contribution from the z-advection term is wpa * vthe * d(pdf)/dz
    @begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @. z_advection_term[ivpa,ivperp,iz,ir] = z_advection_term[ivpa,ivperp,iz,ir] #* vpa[ivpa] * vthe[:, :]
    end
    return nothing
end

function add_contribution_from_wpa_advection!(residual, pdf, vth, p, dppar_dz, dqpar_dz, dvth_dz,
                                                    vpa, vpa_spectral)
    @begin_r_z_vperp_region()
    @loop_r_z_vperp ir iz ivperp begin
        # calculate the wpa-derivative of the pdf and store in the scratch array vpa.scratch
        @views derivative!(vpa.scratch, pdf[:, ivperp, iz, ir], vpa, vpa_spectral)
        # contribution from the wpa-advection term is ( ) * d(pdf)/dwpa
        @. residual[:,ivperp,iz,ir] += vpa.scratch * (0.5 * vth[iz,ir] / p[iz,ir] * dppar_dz[iz,ir]
                + 1.0/3.0 * vpa.grid / p[iz,ir] * dqpar_dz[iz,ir] - vpa.grid^2 * dvth_dz[iz,ir])
    end
    return nothing
end

# calculate the pre-factor multiplying the modified electron pdf
function calculate_pdf_dot_prefactor!(pdf_dot_prefactor, p, vth, dens, ddens_dz, dvth_dz, dqpar_dz, vpa)
    @begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @. pdf_dot_prefactor[ivpa,ivperp,iz,ir] = 1.0/3.0 * dqpar_dz[iz,ir] / p[iz,ir] - vpa[ivpa] * vth[iz,ir] * 
        (ddens_dz[iz,ir] / dens[iz,ir] + dvth_dz[iz,ir] / vth[iz,ir])
    end
    return nothing
end

"""
add contribution to the kinetic equation coming from the term proportional to the pdf
"""
@timeit global_timer add_contribution_from_pdf_term!(pdf_out, pdf_in, p, dens, upar,
                                                     moments, vpa, z, dt,
                                                     electron_source_settings, ir) = begin
    vth = @view moments.electron.vth[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
    @begin_z_vperp_region()
    @loop_z iz begin
        this_dqpar_dz = dqpar_dz[iz]
        this_p = p[iz]
        this_vth = vth[iz]
        this_ddens_dz = ddens_dz[iz]
        this_dens = dens[iz]
        this_dvth_dz = dvth_dz[iz]
        this_vth = vth[iz]
        @loop_vperp ivperp begin
            @views @. pdf_out[:,ivperp,iz] +=
                dt * (-1.0/3.0 * this_dqpar_dz / this_p +
                      vpa * (this_dvth_dz - this_vth * this_ddens_dz / this_dens)) *
                pdf_in[:,ivperp,iz]
        end
    end

    for index ∈ eachindex(electron_source_settings)
        if electron_source_settings[index].active
            @views source_density_amplitude = moments.electron.external_source_density_amplitude[:, ir, index]
            @views source_momentum_amplitude = moments.electron.external_source_momentum_amplitude[:, ir, index]
            @views source_pressure_amplitude = moments.electron.external_source_pressure_amplitude[:, ir, index]
            @loop_z iz begin
                term = dt * (0.5 * source_pressure_amplitude[iz] / p[iz]
                             - 1.5 * source_density_amplitude[iz] / dens[iz])
                @loop_vperp ivperp begin
                    @views @. pdf_out[:,ivperp,iz] += term * pdf_in[:,ivperp,iz]
                end
            end
        end
    end

    return nothing
end

function add_contribution_from_electron_pdf_term_to_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, ddens_dz, dp_dz,
        dvth_dz, dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z,
        vperp, vpa, z_spectral, z_speed, scratch_dummy, dt, ir, include=:all,
        include_qpar_integral_terms=true; f_offset=0, p_offset=0)

    if f_offset == p_offset
        error("Got f_offset=$f_offset the same as p_offset=$p_offset. f and p "
              * "cannot be in same place in state vector.")
    end
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @boundscheck size(jacobian_matrix, 1) ≥ p_offset + z.n || error("p_offset=$p_offset is too big")
    @boundscheck include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    source_density_amplitude = moments.electron.external_source_density_amplitude
    source_momentum_amplitude = moments.electron.external_source_momentum_amplitude
    source_pressure_amplitude = moments.electron.external_source_pressure_amplitude
    z_deriv_matrix = z_spectral.D_matrix_csr
    v_size = vperp.n * vpa.n

    @begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset
        v_remainder = (ivperp - 1) * vpa.n + ivpa

        # Terms from `add_contribution_from_pdf_term!()`
        # (1/3/p*dq/dz + w_∥*(vth/n*dn/dz - *dvth/dz))*g
        #
        # q = p*vth*∫dw_∥ w_∥^3 g
        #   = p^(3/2)*sqrt(2/n/me)*∫dw_∥ w_∥^3 g
        # dq/dz = 3/2*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #         - 1/2*p^(3/2)*sqrt(2/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #         + p*vth*∫dw_∥ w_∥^3 dg/dz
        # 1/3/p*dq/dz = 0.5*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #               - 1/6*sqrt(2*p/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #               + 1/3*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 dg/dz
        #             = 0.5*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #               - 1/6*sqrt(2*p/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #               + 1/3*vth*∫dw_∥ w_∥^3 dg/dz
        # d(1/3/p*dq/dz[irowz])/d(g[icolvpa,icolvperp,icolz]) =
        #   (0.5*sqrt(2/p/n/me)*dp/dz - 1/6*sqrt(2*p/me)/n^(3/2)*dn/dz) * delta(irowz,icolz) * vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3
        #   + 1/3 * vth * vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3 * z_deriv_matrix[irowz,icolz]
        # d(1/3/p*dq/dz[irowz])/d(p[icolz]) =
        #   (-1/4*sqrt(2/n/me)/p^(3/2)*∫dw_∥ w_∥^3 g * dp/dz - 1/12*sqrt(2/me)/sqrt(p)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz + 1/6*sqrt(2/n/me)/sqrt(p)*∫dw_∥ w_∥^3 dg/dz)[irowz] * delta(irowz,icolz)
        #   + (0.5*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g)[irowz] * z_deriv_matrix[irowz,icolz]
        #
        # dvth/dz = d/dz(sqrt(2*p/n/me))
        #         = 1/n/me/sqrt(2*p/n/me)*dp/dz - p/n^2/me/sqrt(2*p/n/me)*dn/dz
        #         = 1/n/me/vth*dp/dz - p/n^2/me/vth*dn/dz
        #         = 1/n/me/vth*dp/dz - 1/2*vth/n*dn/dz
        # ⇒ (vth/n*dn/dz - dvth/dz)
        #   = (vth/n*dn/dz - 1/n/me/vth*dp/dz + 1/2*vth/n*dn/dz)
        #   = (3/2*vth/n*dn/dz - 1/n/me/vth*dp/dz)
        #   = (3/2*sqrt(2*p/me)/n^(3/2)*dn/dz - 1/sqrt(2*p*n*me)*dp/dz)
        # d((vth/n*dn/dz - dvth/dz)[irowz])/d(p[icolz]) =
        #   (3/4*sqrt(2/me)/p^(1/2)/n^(3/2)*dn/dz + 1/2/sqrt(2*n*me)/p^(3/2)*dp/dz)[irowz] * delta(irowz,icolz)
        #   -1/sqrt(2*p*n*me)[irowz] * z_deriv_matrix[irowz,icolz]
        #
        if include === :all
            jacobian_matrix[row,row] += dt * (1.0 / 3.0 * dqpar_dz[iz] / p[iz]
                                              + vpa.grid[ivpa] * (vth[iz] * ddens_dz[iz] / dens[iz]
                                                                  - dvth_dz[iz]))
        end
        if include ∈ (:all, :explicit_v)
            for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                col = (iz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                jacobian_matrix[row,col] +=
                    dt * f[ivpa,ivperp,iz] *
                    (0.5*sqrt(2.0/p[iz]/dens[iz]/me)*dp_dz[iz] - 1.0/6.0*sqrt(2.0*p[iz]/me)/dens[iz]^1.5*ddens_dz[iz]) *
                    vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3
            end
        end
        z_deriv_row_startind = z_deriv_matrix.rowptr[iz]
        z_deriv_row_endind = z_deriv_matrix.rowptr[iz+1] - 1
        z_deriv_colinds = @view z_deriv_matrix.colval[z_deriv_row_startind:z_deriv_row_endind]
        z_deriv_row_nonzeros = @view z_deriv_matrix.nzval[z_deriv_row_startind:z_deriv_row_endind]
        if include_qpar_integral_terms
            for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros), icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                col = (icolz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                jacobian_matrix[row,col] +=
                    dt * f[ivpa,ivperp,iz] * vth[iz] / 3.0 *
                    vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3 * z_deriv_entry
            end
        end
        if include === :all
            for index ∈ eachindex(external_source_settings.electron)
                electron_source = external_source_settings.electron[index]
                if electron_source.active
                    # Source terms from `add_contribution_from_pdf_term!()`
                    jacobian_matrix[row,row] += dt * (1.5 * source_density_amplitude[iz,ir,index] / dens[iz]
                                                      - 0.5 * source_pressure_amplitude[iz,ir,index] / p[iz])
                end
            end
        end
        if include ∈ (:all, :explicit_v)
            jacobian_matrix[row,p_offset+iz] +=
                dt * f[ivpa,ivperp,iz] *
                (-0.25*sqrt(2.0/dens[iz]/me)/p[iz]^1.5*third_moment[iz]*dp_dz[iz]
                 - 1.0/12.0*sqrt(2.0/p[iz]/me)/dens[iz]^1.5*third_moment[iz]*ddens_dz[iz]
                 + 1.0/6.0*sqrt(2.0/p[iz]/dens[iz]/me)*dthird_moment_dz[iz]
                 + vpa.grid[ivpa] * (0.75*sqrt(2.0/me/p[iz])/dens[iz]^1.5*ddens_dz[iz]
                                     + 0.5/sqrt(2.0*dens[iz]*me)/p[iz]^1.5*dp_dz[iz]))
            for index ∈ eachindex(external_source_settings.electron)
                electron_source = external_source_settings.electron[index]
                if electron_source.active
                    # Source terms from `add_contribution_from_pdf_term!()`
                    jacobian_matrix[row,p_offset+iz] += dt * f[ivpa,ivperp,iz] *
                                                        0.5 * source_pressure_amplitude[iz,ir,index] / p[iz]^2
                end
            end
        end
        for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
            col = p_offset + icolz
            jacobian_matrix[row,col] += dt * f[ivpa,ivperp,iz] *
                (0.5*sqrt(2.0/p[iz]/dens[iz]/me)*third_moment[iz]
                 - vpa.grid[ivpa]/sqrt(2.0*p[iz]*dens[iz]*me)) * z_deriv_entry
        end
    end

    return nothing
end

function add_contribution_from_electron_pdf_term_to_z_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, ddens_dz, dp_dz,
        dvth_dz, dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z,
        vperp, vpa, z_spectral, z_speed, scratch_dummy, dt, ir, ivperp, ivpa)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == z.n || error("Jacobian matrix size is wrong")

    source_density_amplitude = moments.electron.external_source_density_amplitude
    source_momentum_amplitude = moments.electron.external_source_momentum_amplitude
    source_pressure_amplitude = moments.electron.external_source_pressure_amplitude

    @loop_z iz begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = iz

        jacobian_matrix[row,row] += dt * (1.0 / 3.0 * dqpar_dz[iz] / p[iz]
                                          + vpa.grid[ivpa] * (vth[iz] * ddens_dz[iz] / dens[iz]
                                                              - dvth_dz[iz]))
        for index ∈ eachindex(external_source_settings.electron)
            electron_source = external_source_settings.electron[index]
            if electron_source.active
                # Source terms from `add_contribution_from_pdf_term!()`
                jacobian_matrix[row,row] += dt * (1.5 * source_density_amplitude[iz,ir,index] / dens[iz]
                                                  - 0.5 * source_pressure_amplitude[iz,ir,index] / p[iz])
            end
        end
    end

    return nothing
end

function add_contribution_from_electron_pdf_term_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, third_moment, ddens_dz, dp_dz,
        dvth_dz, dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z,
        vperp, vpa, z_spectral, z_speed, scratch_dummy, dt, ir, iz)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    source_density_amplitude = moments.electron.external_source_density_amplitude
    source_momentum_amplitude = moments.electron.external_source_momentum_amplitude
    source_pressure_amplitude = moments.electron.external_source_pressure_amplitude
    z_deriv_matrix = z_spectral.D_matrix_csr
    v_size = vperp.n * vpa.n

    @loop_vperp_vpa ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (ivperp - 1) * vpa.n + ivpa

        jacobian_matrix[row,row] += dt * (1.0 / 3.0 * dqpar_dz / p
                                          + vpa.grid[ivpa] * (vth * ddens_dz / dens
                                                              - dvth_dz))
        for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
            col = (icolvperp - 1) * vpa.n + icolvpa
            jacobian_matrix[row,col] +=
                dt * f[ivpa,ivperp] *
                (0.5*sqrt(2.0/p/dens/me)*dp_dz - 1.0/6.0*sqrt(2.0*p/me)/dens^1.5*ddens_dz) *
                vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3
        end
        for index ∈ eachindex(external_source_settings.electron)
            electron_source = external_source_settings.electron[index]
            if electron_source.active
                # Source terms from `add_contribution_from_pdf_term!()`
                jacobian_matrix[row,row] += dt * (1.5 * source_density_amplitude[iz,ir,index] / dens
                                                  - 0.5 * source_pressure_amplitude[iz,ir,index] / p)
            end
        end
        jacobian_matrix[row,end] +=
            dt * f[ivpa,ivperp] *
            (-0.25*sqrt(2.0/dens/me)/p^1.5*third_moment*dp_dz
             - 1.0/12.0*sqrt(2.0/p/me)/dens^1.5*third_moment*ddens_dz
             + 1.0/6.0*sqrt(2.0/p/dens/me)*dthird_moment_dz
             + vpa.grid[ivpa] * (0.75*sqrt(2.0/me/p)/dens^1.5*ddens_dz
                                 + 0.5/sqrt(2.0*dens*me)/p^1.5*dp_dz))
        for index ∈ eachindex(external_source_settings.electron)
            electron_source = external_source_settings.electron[index]
            if electron_source.active
                # Source terms from `add_contribution_from_pdf_term!()`
                jacobian_matrix[row,end] += dt * f[ivpa,ivperp] *
                                                 0.5 * source_pressure_amplitude[iz,ir,index] / p^2
            end
        end
    end

    return nothing
end

function add_ion_dt_forcing_of_electron_p_to_Jacobian!(jacobian_matrix, z, dt, ion_dt,
                                                       ir, include=:all; p_offset=0)
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) ≥ p_offset + z.n || error("p_offset=$p_offset is too big")
    @boundscheck include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    if include === :all
        @begin_z_region()
        @loop_z iz begin
            # Rows corresponding to electron_p
            row = p_offset + iz

            # Backward-Euler forcing term
            jacobian_matrix[row,row] += dt / ion_dt
        end
    end

    return nothing
end

function add_ion_dt_forcing_of_electron_p_to_z_only_Jacobian!(jacobian_matrix, z, dt,
                                                                 ion_dt, ir)
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == z.n || error("Jacobian matrix size is wrong")

    @loop_z iz begin
        # Rows corresponding to electron_p
        row = iz

        # Backward-Euler forcing term
        jacobian_matrix[row,row] += dt / ion_dt
    end

    return nothing
end

function add_ion_dt_forcing_of_electron_p_to_v_only_Jacobian!(jacobian_matrix, z, dt,
                                                                 ion_dt, ir, iz)
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    #@boundscheck size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    # Backward-Euler forcing term
    jacobian_matrix[end,end] += dt / ion_dt

    return nothing
end

# function check_electron_pdf_convergence!(electron_pdf_converged, pdf_new, pdf)
#     # check to see if the electron pdf has converged to within the specified tolerance
#     # NB: the convergence criterion is based on the average relative difference between the
#     # new and old electron pdfs
#     println("sum(abs(pdf)): ", sum(abs.(pdf)), " sum(abs(pdf_new)): ", sum(abs.(pdf_new)))
#     println("sum(abs(pdfdiff)): ", sum(abs.(pdf_new - pdf)) / sum(abs.(pdf)), " length: ", length(pdf))
#     average_relative_error = sum(abs.(pdf_new - pdf)) / sum(abs.(pdf))
#     electron_pdf_converged = (average_relative_error < 1e-3)
#     println("average_relative_error: ", average_relative_error, " electron_pdf_converged: ", electron_pdf_converged)
#     return nothing
# end

function get_electron_critical_velocities(phi, vthe, me_over_mi, z)
    # get the critical velocities beyond which electrons are lost to the wall
    if z.irank == 0 && block_rank[] == 0
        crit_speed_zmin = sqrt(phi[1, 1] / (me_over_mi * vthe[1, 1]^2))
    else
        crit_speed_zmin = 0.0
    end
    @serial_region begin
        @timeit_debug global_timer "MPI.Bcast z.comm" crit_speed_zmin = MPI.Bcast(crit_speed_zmin, 0, z.comm)
    end
    @timeit_debug global_timer "MPI.Bcast comm_block" crit_speed_zmin = MPI.Bcast(crit_speed_zmin, 0, comm_block[])

    if z.irank == z.nrank - 1 && block_rank[] == 0
        crit_speed_zmax = -sqrt(max(phi[end, 1],0.0) / (me_over_mi * vthe[end, 1]^2))
    else
        crit_speed_zmin = 0.0
    end
    @serial_region begin
        @timeit_debug global_timer "MPI.Bcast z.comm" crit_speed_zmax = MPI.Bcast(crit_speed_zmax, z.nrank-1, z.comm)
    end
    @timeit_debug global_timer "MPI.Bcast comm_block" crit_speed_zmax = MPI.Bcast(crit_speed_zmax, 0, comm_block[])

    return crit_speed_zmin, crit_speed_zmax
end

function check_electron_pdf_convergence(residual, pdf, upar, vthe, z, vpa)
    #average_residual = 0.0
    # @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
    #     if pdf[ivpa, ivperp, iz, ir] > eps(mk_float)
    #         average_residual += abs(residual[ivpa, ivperp, iz, ir]) / pdf[ivpa, ivperp, iz, ir]
    #     else
    #         average_residual += abs(residual[ivpa, ivperp, iz, ir])
    #     end
    #     average_residual /= length(residual)
    # end

    # Only sum residual over points that are not set by the sheath boundary condition, as
    # those that are set by the sheath boundary condition are not used by the time
    # advance, and so might not converge to 0.
    # First, sum the contributions from the bulk of the domain
    @begin_r_z_vperp_region()
    sum_residual = 0.0
    sum_pdf = 0.0
    @loop_r_z ir iz begin
        if z.irank == 0 && iz == 1
            vpa_unnorm_lower = @. vpa.scratch3 = vthe[1,ir] * vpa.grid + upar[1,ir]
            iv0_end = findfirst(x -> x>0.0, vpa_unnorm_lower) - 1
        else
            iv0_end = vpa.n
        end
        if z.irank == z.nrank-1 && iz == z.n
            vpa_unnorm_upper = @. vpa.scratch3 = vthe[end,ir] * vpa.grid + upar[end,ir]
            iv0_start = findlast(x -> x>0.0, vpa_unnorm_upper) + 1
        else
            iv0_start = 1
        end
        @loop_vperp ivperp begin
            sum_residual += sum(abs.(@view residual[iv0_start:iv0_end,ivperp,iz,ir]))
            # account for the fact that we want dg/dt << vthe/L * g, but 
            # residual is normalized by c_ref/L_ref * g
            sum_pdf += sum(abs.(@view pdf[iv0_start:iv0_end,ivperp,iz,ir]) * vthe[iz,ir])
        end
    end
    @timeit_debug global_timer "MPI.Allreduce comm_world" sum_residual, sum_pdf = MPI.Allreduce([sum_residual, sum_pdf], +, comm_world)

    average_residual = sum_residual / sum_pdf

    electron_pdf_converged = (average_residual < 1e-3)

    return average_residual, electron_pdf_converged
end

function check_electron_pdf_convergence(residual)
    @begin_r_z_vperp_region()
    sum_residual = 0.0
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        sum_residual += abs.(residual[ivpa,ivperp,iz,ir])
    end
    @timeit_debug global_timer "MPI.Allreduce comm_world" sum_residual, sum_length = MPI.Allreduce((sum_residual, length(residual) / block_size[]), +, comm_world)
    average_residual = sum_residual / sum_length
    electron_pdf_converged = (average_residual < 1e-3)
    return average_residual, electron_pdf_converged
end

end
