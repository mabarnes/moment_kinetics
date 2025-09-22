module electron_kinetic_equation

using LinearAlgebra
using MPI
using OrderedCollections
using SparseArrays

using ..looping
using ..analysis: steady_state_residuals
using ..derivatives: derivative_z_anyzv!, derivative_z_pdf_vpavperpz!
using ..boundary_conditions: enforce_v_boundary_condition_local!,
                             enforce_vperp_boundary_condition!,
                             get_ADI_boundary_v_solve_z_speed, vpagrid_to_vpa
using ..calculus: derivative!, second_derivative!, integral,
                  reconcile_element_boundaries_MPI_anyzv!,
                  reconcile_element_boundaries_MPI_z_pdf_vpavperpz!
using ..communication
using ..debugging
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
                                  get_electron_energy_equation_term
using ..electron_z_advection: electron_z_advection!, update_electron_speed_z!,
                              get_electron_z_advection_term
using ..electron_vpa_advection: electron_vpa_advection!, update_electron_speed_vpa!,
                                get_electron_vpa_advection_term
using ..em_fields: update_phi!
using ..external_sources: total_external_electron_sources!,
                          get_total_external_electron_source_term
using ..file_io: get_electron_io_info, write_electron_state, finish_electron_io,
                 write_debug_data_to_binary
using ..collision_frequencies: get_collision_frequency_ee,
                               get_collision_frequency_ei
using ..jacobian_matrices
using ..krook_collisions: electron_krook_collisions!, get_electron_krook_collisions_term
using ..timer_utils
using ..moment_constraints: hard_force_moment_constraints!,
                            moment_constraints_on_residual!,
                            electron_implicit_constraint_forcing!,
                            get_electron_implicit_constraint_forcing_term
using ..moment_kinetics_structs
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
                         scratch, pdf, moments, phi::AbstractArray{mk_float,ndim_field},
                         r::coordinate, z::coordinate, vperp::coordinate, vpa::coordinate,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, t_params, collisions, composition,
                         external_source_settings, num_diss_params, nl_solver_params,
                         max_electron_pdf_iterations, max_electron_sim_time;
                         initial_time=nothing, residual_tolerance=nothing, evolve_p=false,
                         ion_dt=nothing, solution_method="backward_euler") = begin

    # solve the electron kinetic equation using the specified method
    if solution_method == "artificial_time_derivative"
        return update_electron_pdf_with_time_advance!(scratch, pdf, moments, phi,
            collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
            vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
            external_source_settings, num_diss_params, max_electron_pdf_iterations,
            max_electron_sim_time; initial_time=initial_time,
            residual_tolerance=residual_tolerance, evolve_p=evolve_p, ion_dt=ion_dt)
    elseif solution_method == "backward_euler"
        return electron_backward_euler_pseudotimestepping!(scratch, pdf, moments, phi,
            collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
            vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
            external_source_settings, num_diss_params, nl_solver_params,
            max_electron_pdf_iterations, max_electron_sim_time; initial_time=initial_time,
            residual_tolerance=residual_tolerance, evolve_p=evolve_p, ion_dt=ion_dt)
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
function update_electron_pdf_with_time_advance!(scratch, pdf, moments,
        phi::AbstractArray{mk_float,ndim_field}, collisions, composition, r::coordinate,
        z::coordinate, vperp::coordinate, vpa::coordinate, z_spectral, vperp_spectral,
        vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
        external_source_settings, num_diss_params, max_electron_pdf_iterations,
        max_electron_sim_time; initial_time=nothing, residual_tolerance=nothing,
        evolve_p=false, ion_dt=nothing)

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
        t_params.t .= initial_time
    else
        initial_time = t_params.t
    end
    if t_params.debug_io !== nothing
        # Overwrite the debug output file with the output from this call to
        # update_electron_pdf_with_time_advance!().
        io_electron = get_electron_io_info(t_params.debug_io[1], "electron_debug")
        debug_io_nwrite = t_params.debug_io[3]
    else
        io_electron = nothing
    end

    epsilon = 1.e-11
    # initialise the electron pdf convergence flag to false
    all_electron_pdf_converged = fill(false, r.n)

    @begin_r_anyzv_region()
    @loop_r ir begin
        if r.n > 1 && !r.periodic && r.irank == 0 && ir == 1
            # No need to solve on point that will be set by boundary conditions
            continue
        end

        step_counter = @view t_params.step_counter[ir:ir]
        t = @view t_params.t[ir:ir]
        dt = @view t_params.dt[ir:ir]
        previous_dt = @view t_params.previous_dt[ir:ir]
        moments_output_counter = @view t_params.moments_output_counter[ir:ir]
        electron_pdf_converged = @view all_electron_pdf_converged[ir:ir]

        # Counter for pseudotime just in this pseudotimestepping loop
        local_pseudotime = 0.0
        residual_norm = -1.0
        # Store the initial number of iterations in the solution of the electron kinetic
        # equation
        initial_step_counter = step_counter[]
        step_counter[] += 1

        this_phi = @view phi[:,ir]

        if io_electron !== nothing
            @begin_anyzv_region()
            moments_output_counter[] += 1
            @anyzv_serial_region begin
                write_electron_state(scratch, moments, phi, t_params, io_electron,
                                     moments_output_counter[], local_pseudotime, 0.0,
                                     r, z, vperp, vpa; ir=ir)
            end
        end

        # evolve (artificially) in time until the residual is less than the tolerance
        while (!electron_pdf_converged[]
               && (max_electron_pdf_iterations === nothing || step_counter[] - initial_step_counter < max_electron_pdf_iterations)
               && (max_electron_sim_time === nothing || t[] - initial_time[ir] < max_electron_sim_time)
               && dt[] > 0.0 && !isnan(dt[]))

            # Set the initial values for the next step to the final values from the previous
            # step
            @begin_anyzv_z_vperp_vpa_region()
            new_pdf = @view scratch[1].pdf_electron[:,:,:,ir]
            old_pdf = @view scratch[t_params.n_rk_stages+1].pdf_electron[:,:,:,ir]
            @loop_z_vperp_vpa iz ivperp ivpa begin
                new_pdf[ivpa,ivperp,iz] = old_pdf[ivpa,ivperp,iz]
            end
            if evolve_p
                @begin_anyzv_z_region()
                new_p = @view scratch[1].electron_p[:,ir]
                old_p = @view scratch[t_params.n_rk_stages+1].electron_p[:,ir]
                @loop_z iz begin
                    new_p[iz] = old_p[iz]
                end
            end

            for istage ∈ 1:t_params.n_rk_stages
                # Set the initial values for this stage to the final values from the previous
                # stage
                @begin_anyzv_z_vperp_vpa_region()
                new_pdf = @view scratch[istage+1].pdf_electron[:,:,:,ir]
                old_pdf = @view scratch[istage].pdf_electron[:,:,:,ir]
                @loop_z_vperp_vpa iz ivperp ivpa begin
                    new_pdf[ivpa,ivperp,iz] = old_pdf[ivpa,ivperp,iz]
                end
                old_p = @view scratch[istage].electron_p[:,ir]
                new_p = @view scratch[istage+1].electron_p[:,ir]
                if evolve_p
                    @begin_anyzv_z_region()
                    @loop_z iz begin
                        new_p[iz] = old_p[iz]
                    end
                end
                # Do a forward-Euler update of the electron pdf, and (if evove_p=true) the
                # electron pressure.
                electron_kinetic_equation_euler_update!(
                           scratch[istage+1], old_pdf, old_p, moments, z, vperp, vpa,
                           z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                           collisions, composition, external_source_settings,
                           num_diss_params, t_params, ir; evolve_p=evolve_p,
                           ion_dt=ion_dt)

                rk_update_variable!(scratch, nothing, :pdf_electron, t_params, istage; ir=ir)

                if evolve_p
                    rk_update_variable!(scratch, nothing, :electron_p, t_params, istage; ir=ir)

                    @views update_electron_vth_temperature!(moments, new_p,
                                                            moments.electron.dens[:,ir],
                                                            composition, ir)
                end

                apply_electron_bc_and_constraints_no_r!(new_pdf, this_phi, moments, r, z,
                                                        vperp, vpa, vperp_spectral,
                                                        vpa_spectral, vpa_advect,
                                                        num_diss_params, composition, ir,
                                                        nothing, scratch_dummy)

                function update_derived_moments_and_derivatives(update_vth=false)
                    # update the electron heat flux
                    moments.electron.qpar_updated[] = false
                    @views calculate_electron_qpar_from_pdf_no_r!(
                               moments.electron.qpar[:,ir], moments.electron.dens[:,ir],
                               moments.electron.vth[:,ir],
                               scratch[istage+1].pdf_electron[:,:,:,ir], vperp, vpa,
                               composition.me_over_mi, ir)

                    if evolve_p
                        this_dens = @view moments.electron.dens[:,ir]
                        this_upar = @view moments.electron.upar[:,ir]
                        this_p = @view scratch[istage+1].electron_p[:,ir]
                        if update_vth
                            update_electron_vth_temperature!(moments, this_p, this_dens,
                                                             composition, ir)
                        end
                        calculate_electron_moment_derivatives_no_r!(
                            moments, this_dens, this_upar, this_p, scratch_dummy, z,
                            z_spectral,
                            num_diss_params.electron.moment_dissipation_coefficient, ir)
                    else
                        # compute the z-derivative of the parallel electron heat flux
                        @views derivative_z_anyzv!(moments.electron.dqpar_dz[:,ir],
                                                   moments.electron.qpar[:,ir],
                                                   buffer_r_1[:,ir], buffer_r_2[:,ir],
                                                   buffer_r_3[:,ir], buffer_r_4[:,ir],
                                                   z_spectral, z)
                    end
                end
                update_derived_moments_and_derivatives()

                if t_params.adaptive && istage == t_params.n_rk_stages
                    electron_adaptive_timestep_update!(scratch, t[], t_params,
                                                       moments, this_phi, z_advect,
                                                       vpa_advect, composition, r, z,
                                                       vperp, vpa, vperp_spectral,
                                                       vpa_spectral,
                                                       external_source_settings,
                                                       num_diss_params, scratch_dummy, ir;
                                                       evolve_p=evolve_p)
                    # Re-do this in case electron_adaptive_timestep_update!() re-arranged the
                    # `scratch` vector
                    new_scratch = scratch[istage+1]
                    old_scratch = scratch[istage]

                    if previous_dt[] == 0.0
                        # Re-calculate moments and moment derivatives as the timstep needs to
                        # be re-done with a smaller dt, so scratch[t_params.n_rk_stages+1] has
                        # been reset to the values from the beginning of the timestep here.
                        update_derived_moments_and_derivatives(true)
                    end
                end
            end

            # update the time following the pdf update
            t[] += previous_dt[]
            local_pseudotime += previous_dt[]

            residual = -1.0
            if previous_dt[] > 0.0
                # Calculate residuals to decide if iteration is converged.
                # Might want an option to calculate the residual only after a certain number
                # of iterations (especially during initialization when there are likely to be
                # a large number of iterations required) to avoid the expense, and especially
                # the global MPI.Bcast()?
                residual = steady_state_residuals(scratch[t_params.n_rk_stages+1].pdf_electron,
                                                  scratch[1].pdf_electron, previous_dt[];
                                                  use_mpi=true, only_max_abs=true, ir=ir,
                                                  comm_local=comm_anyzv_subblock[],
                                                  comm_global=z.comm)
                if block_rank[] == 0 && z.irank == 0
                    residual = first(values(residual))[1]
                end
                if evolve_p
                    p_residual =
                        steady_state_residuals(scratch[t_params.n_rk_stages+1].electron_p,
                                               scratch[1].electron_p, previous_dt[];
                                               use_mpi=true, only_max_abs=true, ir=ir,
                                               comm_local=comm_anyzv_subblock[],
                                               comm_global=z.comm)
                    if block_rank[] == 0 && z.irank == 0
                        p_residual = first(values(p_residual))[1]
                        residual = max(residual, p_residual)
                    end
                end
                if block_rank[] == 0 && z.irank == 0
                    if residual_tolerance === nothing
                        residual_tolerance = t_params.converged_residual_value
                    end
                    electron_pdf_converged[] = abs(residual) < residual_tolerance
                end
                @timeit_debug global_timer "MPI.Bcast comm_world" electron_pdf_converged[] = MPI.Bcast(electron_pdf_converged[], 0, comm_world)
            end

            if (mod(step_counter[] - initial_step_counter,100) == 0)
                @begin_anyzv_region()
                @anyzv_serial_region begin
                    if z.irank == 0 && z.irank == z.nrank - 1
                        println("ir: $ir, iteration: ", step_counter[] - initial_step_counter, " time: ", t[], " dt_electron: ", dt[], " phi_boundary: ", this_phi[[1,end]], " residual: ", residual)
                    elseif z.irank == 0
                        println("ir: $ir, iteration: ", step_counter[] - initial_step_counter, " time: ", t[], " dt_electron: ", dt[], " phi_boundary_lower: ", this_phi[1], " residual: ", residual)
                    end
                end
            end
            if (io_electron !== nothing && (step_counter[] % debug_io_nwrite == 0))

                @begin_anyzv_region()
                moments_output_counter[] += 1
                @anyzv_serial_region begin
                    if io_electron !== nothing
                        write_electron_state(scratch, moments, phi, t_params, io_electron,
                                             moments_output_counter[], local_pseudotime,
                                             residual_norm, r, z, vperp, vpa; ir=ir)
                    end
                end
            end

            step_counter[] += 1
            if electron_pdf_converged[]
                break
            end
        end

        if io_electron !== nothing
            @begin_anyzv_region()
            @anyzv_serial_region begin
                moments_output_counter[] += 1
                write_electron_state(scratch, moments, phi, t_params, io_electron,
                                     moments_output_counter[], local_pseudotime,
                                     residual_norm, r, z, vperp, vpa; ir=ir)
                finish_electron_io(io_electron)
            end
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
    if !all(all_electron_pdf_converged)
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
function electron_backward_euler_pseudotimestepping!(scratch, pdf, moments,
        phi::AbstractArray{mk_float,ndim_field}, collisions, composition, r::coordinate,
        z::coordinate, vperp::coordinate, vpa::coordinate, z_spectral, vperp_spectral,
        vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
        external_source_settings, num_diss_params, nl_solver_params,
        max_electron_pdf_iterations, max_electron_sim_time; initial_time=nothing,
        residual_tolerance=nothing, evolve_p=false, ion_dt=nothing)

    if max_electron_pdf_iterations === nothing && max_electron_sim_time === nothing
        error("Must set one of max_electron_pdf_iterations and max_electron_sim_time")
    end

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
        t_params.t .= initial_time
    else
        initial_time = copy(t_params.t)
    end
    if t_params.debug_io !== nothing
        # Overwrite the debug output file with the output from this call to
        # update_electron_pdf_with_time_advance!().
        io_electron = get_electron_io_info(t_params.debug_io[1], "electron_debug")
        debug_io_nwrite = t_params.debug_io[3]
    else
        io_electron = nothing
    end

    # initialise the electron pdf convergence flag to false
    all_electron_pdf_converged = fill(false, r.n)

    # No paralleism in r for now - will need to add a specially adapted shared-memory
    # parallelism scheme to allow it for 2D1V or 2D2V simulations.
    @begin_r_anyzv_region()
    @loop_r ir begin
        if r.n > 1 && !r.periodic && r.irank == 0 && ir == 1
            # No need to solve on point that will be set by boundary conditions
            continue
        end

        step_counter = @view t_params.step_counter[ir:ir]
        t = @view t_params.t[ir:ir]
        dt = @view t_params.dt[ir:ir]
        previous_dt = @view t_params.previous_dt[ir:ir]
        moments_output_counter = @view t_params.moments_output_counter[ir:ir]
        electron_pdf_converged = @view all_electron_pdf_converged[ir:ir]
        max_step_count_this_ion_step = @view t_params.max_step_count_this_ion_step[ir:ir]
        max_t_increment_this_ion_step = @view t_params.max_t_increment_this_ion_step[ir:ir]

        dt[] = previous_dt[]

        # Pseudotime just in this pseudotimestepping loop
        local_pseudotime = 0.0
        residual_norm = -1.0
        # Store the initial number of iterations in the solution of the electron kinetic
        # equation
        initial_step_counter = step_counter[]
        step_counter[] += 1

        this_phi = @view phi[:,ir]

        # create several 0D dummy arrays for use in taking derivatives
        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        @begin_anyzv_region()
        moments_output_counter[] += 1
        @anyzv_serial_region begin
            if io_electron !== nothing
                write_electron_state(scratch, moments, phi, t_params, io_electron,
                                     moments_output_counter[], local_pseudotime, 0.0, r,
                                     z, vperp, vpa; ir=ir)
            end
        end

        first_step = true
        # evolve (artificially) in time until the residual is less than the tolerance
        while (!electron_pdf_converged[]
               && ((max_electron_pdf_iterations !== nothing && step_counter[] - initial_step_counter < max_electron_pdf_iterations)
                   || (max_electron_sim_time !== nothing && t[] - initial_time[ir] < max_electron_sim_time))
               && dt[] > 0.0 && !isnan(dt[]))

            old_scratch = scratch[1]
            new_scratch = scratch[t_params.n_rk_stages+1]

            # Set the initial values for the next step to the final values from the previous
            # step. The initial guess for f_electron_new and electron_p_new are just the
            # values from the old timestep, so no need to change those.
            @begin_anyzv_z_vperp_vpa_region()
            f_electron_old = @view old_scratch.pdf_electron[:,:,:,ir]
            f_electron_new = @view new_scratch.pdf_electron[:,:,:,ir]
            @loop_z_vperp_vpa iz ivperp ivpa begin
                f_electron_old[ivpa,ivperp,iz] = f_electron_new[ivpa,ivperp,iz]
            end
            electron_p_old = @view old_scratch.electron_p[:,ir]
            electron_p_new = @view new_scratch.electron_p[:,ir]
            if evolve_p
                @begin_anyzv_z_region()
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
                                               f_electron_new, this_phi, moments, r, z,
                                               vperp, vpa, vperp_spectral, vpa_spectral,
                                               vpa_advect, num_diss_params, composition,
                                               ir, nl_solver_params, scratch_dummy)
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
                    @views derivative_z_anyzv!(moments.electron.dqpar_dz[:,ir],
                                               moments.electron.qpar[:,ir], buffer_1,
                                               buffer_2, buffer_3, buffer_4, z_spectral,
                                               z)
                end

                #println("Newton its ", nl_solver_params.max_nonlinear_iterations_this_step[], " ", dt[])
                # update the time following the pdf update
                t[] += dt[]
                local_pseudotime += dt[]

                if first_step && !reduced_by_ion_dt
                    # Adjust previous_dt[] which gives the initial timestep for
                    # the electron pseudotimestepping loop.
                    # If ion_dt<previous_dt[], assume that this is because we are
                    # taking a short ion step to an output time, so we do not want to mess
                    # up previous_dt[], which should be set sensibly for a
                    # 'normal' timestep.
                    if dt[] < previous_dt[]
                        # Had to decrease timestep on the first step to get convergence,
                        # so start next ion timestep with the decreased value.
                        global_rank[] == 0 && print("decreasing previous_dt due to failures ", previous_dt[])
                        previous_dt[] = dt[]
                        global_rank[] == 0 && println(" -> ", previous_dt[])
                    #elseif nl_solver_params.max_linear_iterations_this_step[] > max(0.4 * nl_solver_params.nonlinear_max_iterations, 5)
                    elseif nl_solver_params.max_linear_iterations_this_step[] > t_params.decrease_dt_iteration_threshold
                        # Step succeeded, but took a lot of iterations so decrease initial
                        # step size.
                        global_rank[] == 0 && print("decreasing previous_dt due to iteration count ", previous_dt[])
                        previous_dt[] /= t_params.max_increase_factor
                        global_rank[] == 0 && println(" -> ", previous_dt[])
                    #elseif nl_solver_params.max_linear_iterations_this_step[] < max(0.1 * nl_solver_params.nonlinear_max_iterations, 2)
                    elseif nl_solver_params.max_linear_iterations_this_step[] < t_params.increase_dt_iteration_threshold && (ion_dt === nothing || previous_dt[] < t_params.cap_factor_ion_dt * ion_dt)
                        # Only took a few iterations, so increase initial step size.
                        global_rank[] == 0 && print("increasing previous_dt due to iteration count ", previous_dt[])
                        if ion_dt === nothing
                            previous_dt[] *= t_params.max_increase_factor
                        else
                            previous_dt[] = min(previous_dt[] * t_params.max_increase_factor, t_params.cap_factor_ion_dt * ion_dt)
                        end
                        global_rank[] == 0 && println(" -> ", previous_dt[])
                    end
                end

                # Adjust the timestep depending on the iteration count.
                # Note nl_solver_params.max_linear_iterations_this_step[] gives the total
                # number of iterations, so is a better measure of the total work done by
                # the solver than the nonlinear iteration count, or the linear iterations
                # per nonlinear iteration
                #if nl_solver_params.max_linear_iterations_this_step[] > max(0.2 * nl_solver_params.nonlinear_max_iterations, 10)
                if nl_solver_params.max_linear_iterations_this_step[] > t_params.decrease_dt_iteration_threshold && dt[] > previous_dt[]
                    # Step succeeded, but took a lot of iterations so decrease step size.
                    dt[] /= t_params.max_increase_factor
                elseif nl_solver_params.max_linear_iterations_this_step[] < t_params.increase_dt_iteration_threshold && (ion_dt === nothing || dt[] < t_params.cap_factor_ion_dt * ion_dt)
                    # Only took a few iterations, so increase step size.
                    if ion_dt === nothing
                        dt[] *= t_params.max_increase_factor
                    else
                        dt[] = min(dt[] * t_params.max_increase_factor, t_params.cap_factor_ion_dt * ion_dt)
                    end
                    # Ensure dt does not exceed maximum_dt
                    dt[] = min(dt[], t_params.maximum_dt)
                end

                first_step = false
            else
                if !t_params.adaptive
                    # Timestep not allowed to change, so errors are fatal
                    error("Electron pseudotimestep failed. Electron pseudotimestep size "
                          * "is fixed, so cannot reduce timestep to re-try.")
                end

                dt[] *= 0.5

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
                @begin_anyzv_z_vperp_vpa_region()
                @loop_z_vperp_vpa iz ivperp ivpa begin
                    f_electron_new[ivpa,ivperp,iz] = f_electron_old[ivpa,ivperp,iz]
                end
                @begin_anyzv_z_region()
                @loop_z iz begin
                    electron_p_new[iz] = electron_p_old[iz]
                end

                bc_constraints_converged = apply_electron_bc_and_constraints_no_r!(
                                               f_electron_new, this_phi, moments, r, z, vperp,
                                               vpa, vperp_spectral, vpa_spectral,
                                               vpa_advect, num_diss_params, composition,
                                               ir, nl_solver_params, scratch_dummy)
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
                    @views derivative_z_anyzv!(moments.electron.dqpar_dz[:,ir],
                                               moments.electron.qpar[:,ir], buffer_1,
                                               buffer_2, buffer_3, buffer_4, z_spectral,
                                               z)
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
                @begin_anyzv_z_vperp_vpa_region()
                if global_rank[] == 0
                    residual_norm = steady_state_residuals(new_scratch.pdf_electron,
                                                           old_scratch.pdf_electron,
                                                           dt[]; use_mpi=true,
                                                           only_max_abs=true, ir=ir,
                                                           comm_local=comm_anyzv_subblock[],
                                                           comm_global=z.comm)[1]
                else
                    steady_state_residuals(new_scratch.pdf_electron,
                                           old_scratch.pdf_electron, dt[]; use_mpi=true,
                                           only_max_abs=true, ir=ir,
                                           comm_local=comm_anyzv_subblock[],
                                           comm_global=z.comm)
                end
                if evolve_p
                    if global_rank[] == 0
                        p_residual =
                            steady_state_residuals(new_scratch.electron_p,
                                                   old_scratch.electron_p,
                                                   dt[]; use_mpi=true, only_max_abs=true,
                                                   ir=ir, comm_local=comm_anyzv_subblock[],
                                                   comm_global=z.comm)[1]
                        residual_norm = max(residual_norm, p_residual)
                    else
                        steady_state_residuals(new_scratch.electron_p,
                                               old_scratch.electron_p,
                                               dt[]; use_mpi=true, only_max_abs=true,
                                               ir=ir, comm_local=comm_anyzv_subblock[],
                                               comm_global=z.comm)
                    end
                end
                if block_rank[] == 0 && z.irank == 0
                    if residual_tolerance === nothing
                        residual_tolerance = t_params.converged_residual_value
                    end
                    electron_pdf_converged[] = abs(residual_norm) < residual_tolerance
                end
                @timeit_debug global_timer "MPI.Bcast! comm_world" MPI.Bcast!(electron_pdf_converged, 0, comm_world)
            end

            if (mod(step_counter[] - initial_step_counter,100) == 0)
                @begin_anyzv_region()
                @anyzv_serial_region begin
                    if z.irank == 0 && z.irank == z.nrank - 1
                        println("ir: $ir, iteration: ", step_counter[] - initial_step_counter, " time: ", t[], " dt_electron: ", dt[], " phi_boundary: ", this_phi[[1,end]], " residual_norm: ", residual_norm)
                    elseif z.irank == 0
                        println("ir: $ir, iteration: ", step_counter[] - initial_step_counter, " time: ", t[], " dt_electron: ", dt[], " phi_boundary_lower: ", this_phi[1], " residual_norm: ", residual_norm)
                    end
                end
            end
            if (io_electron !== nothing && (step_counter[] % debug_io_nwrite == 0))
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
                    @begin_anyzv_region()
                    moments_output_counter[] += 1
                    @anyzv_serial_region begin
                        if io_electron !== nothing
                            write_electron_state(scratch, moments, phi, t_params,
                                                 io_electron, moments_output_counter[],
                                                 local_pseudotime, residual_norm, r, z,
                                                 vperp, vpa; ir=ir)
                        end
                    end
                end
            end

            step_counter[] += 1
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

        @begin_anyzv_region()
        @anyzv_serial_region begin
            if io_electron !== nothing
                moments_output_counter[] += 1
                write_electron_state(scratch, moments, phi, t_params, io_electron,
                                     moments_output_counter[], local_pseudotime,
                                     residual_norm, r, z, vperp, vpa; ir=ir)
                finish_electron_io(io_electron)
            end
        end

        max_step_count_this_ion_step[] =
            max(step_counter[] - initial_step_counter, max_step_count_this_ion_step[])
        max_t_increment_this_ion_step[] =
            max(t[] - initial_time[ir], max_t_increment_this_ion_step[])

        initial_dt_scale_factor = 0.1
        if previous_dt[] < initial_dt_scale_factor * dt[]
            # If dt has increased a lot, we can probably try a larger initial dt for the next
            # solve.
            previous_dt[] = initial_dt_scale_factor * dt[]
        end

        if ion_dt !== nothing && dt[] != previous_dt[]
            # Reset dt in case it was reduced to be less than 0.5*ion_dt
            dt[] = previous_dt[]
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

    if !all(all_electron_pdf_converged)
        success = "kinetic-electrons"
    else
        success = ""
    end
    return success
end

function get_electron_lu_preconditioner(nl_solver_params, f_electron_new, electron_p_new,
                                        buffer_1, buffer_2, buffer_3, buffer_4,
                                        electron_density, electron_upar, this_phi,
                                        moments, collisions, composition, z, vperp, vpa,
                                        z_spectral, vperp_spectral, vpa_spectral,
                                        z_advect, vpa_advect, scratch_dummy,
                                        external_source_settings, num_diss_params,
                                        t_params, ion_dt, ir, evolve_p, add_identity=true)
    function recalculate_lu_preconditioner!()
global_rank[] == 0 && println("recalculating precon")
        nl_solver_params.solves_since_precon_update[] = 0
        nl_solver_params.precon_dt[] = t_params.dt[]

        orig_lu, precon, input_buffer, output_buffer =
            nl_solver_params.preconditioners[ir]

        fill_electron_kinetic_equation_Jacobian!(
            precon, f_electron_new, electron_p_new, moments, this_phi, collisions,
            composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
            z_advect, vpa_advect, scratch_dummy, external_source_settings,
            num_diss_params, t_params, ion_dt, ir, evolve_p, :all, true, add_identity)

        @begin_anyzv_region()
        if anyzv_subblock_rank[] == 0
            if size(orig_lu) == (1, 1)
                # Have not properly created the LU decomposition before, so
                # cannot reuse it.
                @timeit_debug global_timer "lu" nl_solver_params.preconditioners[ir] =
                    (lu(sparse(precon.matrix)), precon, input_buffer,
                     output_buffer)
            else
                # LU decomposition was previously created. The Jacobian always
                # has the same sparsity pattern, so by using `lu!()` we can
                # reuse some setup.
                try
                    @timeit_debug global_timer "lu!" lu!(orig_lu, sparse(precon.matrix); check=false)
                catch e
                    if !isa(e, ArgumentError)
                        rethrow(e)
                    end
                    println("Sparsity pattern of matrix changed, rebuilding "
                            * " LU from scratch")
                    @timeit_debug global_timer "lu" orig_lu = lu(sparse(precon.matrix))
                end
                nl_solver_params.preconditioners[ir] =
                    (orig_lu, precon, input_buffer, output_buffer)
            end
        else
            nl_solver_params.preconditioners[ir] =
                (orig_lu, precon, input_buffer, output_buffer)
        end
    end


    @timeit_debug global_timer lu_precon!(x) = begin
        precon_p, precon_f = x

        precon_lu, _, this_input_buffer, this_output_buffer =
            nl_solver_params.preconditioners[ir]

        @begin_anyzv_region()
        counter = 1
        @loop_z_vperp_vpa iz ivperp ivpa begin
            this_input_buffer[counter] = precon_f[ivpa,ivperp,iz]
            counter += 1
        end
        @loop_z iz begin
            this_input_buffer[counter] = precon_p[iz]
            counter += 1
        end

        @begin_anyzv_region()
        @anyzv_serial_region begin
            @timeit_debug global_timer "ldiv!" ldiv!(this_output_buffer, precon_lu, this_input_buffer)
        end

        @begin_anyzv_region()
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
        @begin_anyzv_vperp_vpa_region()
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

        @begin_anyzv_region()
        @anyzv_serial_region begin
            buffer_1[] = precon_p[1]
            buffer_2[] = precon_p[end]
        end
        reconcile_element_boundaries_MPI_anyzv!(
            precon_p, buffer_1, buffer_2, buffer_3, buffer_4, z)

        return nothing
    end

    left_preconditioner = identity
    right_preconditioner = lu_precon!

    return left_preconditioner, right_preconditioner, recalculate_lu_preconditioner!
end

function get_electron_adi_preconditioner(nl_solver_params, f_electron_new, electron_p_new,
                                         buffer_1, buffer_2, buffer_3, buffer_4,
                                         electron_density, electron_upar, this_phi,
                                         moments, collisions, composition, z, vperp, vpa,
                                         z_spectral, vperp_spectral, vpa_spectral,
                                         z_advect, vpa_advect, scratch_dummy,
                                         external_source_settings, num_diss_params,
                                         t_params, ion_dt, ir, evolve_p,
                                         add_identity=true)
    function recalculate_adi_preconditioner!()
global_rank[] == 0 && println("recalculating precon")
        nl_solver_params.solves_since_precon_update[] = 0
        nl_solver_params.precon_dt[] = t_params.dt[]

        adi_info = nl_solver_params.preconditioners[ir]

        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = @view scratch_dummy.buffer_zrs_1[:,ir,1]
        dthird_moment_dz = @view scratch_dummy.buffer_zrs_2[:,ir,1]
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / electron_p_new[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                            buffer_3, buffer_4, z_spectral, z)

        z_speed = @view z_advect[1].speed[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_anyzv_vperp_vpa_region()
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
        @begin_anyzv_z_vperp_region()
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
        vpa_speed = @view vpa_advect[1].speed[:,:,:,ir]

        zeroth_moment = @view scratch_dummy.buffer_zrs_3[:,ir,1]
        first_moment = @view scratch_dummy.buffer_zrs_4[:,ir,1]
        second_moment = @view scratch_dummy.buffer_zrs_5[:,ir,1]
        @begin_anyzv_z_region()
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
        A = adi_info.v_solve_jacobian
        explicit_J = adi_info.explicit_jacobian
        # Get sparse matrix for explicit, right-hand-side part of the
        # solve.
        if adi_info.n_extra_iterations > 0
            # If we only do one 'iteration' we don't need the 'explicit
            # matrix' for the first solve (the v-solve), because the initial
            # guess is zero,
            fill_electron_kinetic_equation_Jacobian!(
                explicit_J, f_electron_new, electron_p_new, moments, this_phi,
                collisions, composition, z, vperp, vpa, z_spectral, vperp_spectral,
                vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params, ion_dt, ir,
                evolve_p, :explicit_z, false, add_identity)

            # This is calculated and stored in scratch_dummy.buffer_vpavperpzr_3 in
            # fill_electron_kinetic_equation_Jacobian!().
            d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        else
            d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
            if num_diss_params.electron.vpa_dissipation_coefficient > 0.0
                @begin_anyzv_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz],
                                              f_electron_new[:,ivperp,iz], vpa,
                                              vpa_spectral)
                end
            end
        end

        @begin_anyzv_z_region()
        @loop_z iz begin
            v_solve_counter += 1
            # Get LU-factorized matrix for implicit part of the solve
            fill_electron_kinetic_equation_v_only_Jacobian!(
                A, @view(f_electron_new[:,:,iz]), @view(electron_p_new[iz]),
                @view(dpdf_dz[:,:,iz]), @view(dpdf_dvpa[:,:,iz]),
                @view(d2pdf_dvpa2[:,:,iz]), @view(z_speed[iz,:,:]),
                @view(vpa_speed[:,:,iz]), moments, @view(zeroth_moment[iz]),
                @view(first_moment[iz]), @view(second_moment[iz]),
                @view(third_moment[iz]), dthird_moment_dz[iz], this_phi[iz],
                collisions, composition, z, vperp, vpa, z_spectral, vperp_spectral,
                vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params, ion_dt, ir, iz,
                evolve_p, add_identity)
            A_sparse = sparse(A.matrix)
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
                adi_info.v_solve_explicit_matrices[v_solve_counter] = sparse(@view(explicit_J.matrix[adi_info.v_solve_global_inds[v_solve_counter],:]))
            end
        end
        @debug_consistency_checks v_solve_counter == adi_info.v_solve_nsolve || error("v_solve_counter($v_solve_counter) != v_solve_nsolve($(adi_info.v_solve_nsolve))")

        # Do setup for 'z solves'
        z_solve_counter = 0
        A = adi_info.z_solve_jacobian
        # Get sparse matrix for explicit, right-hand-side part of the
        # solve.
        fill_electron_kinetic_equation_Jacobian!(
            explicit_J, f_electron_new, electron_p_new, moments, this_phi, collisions,
            composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
            z_advect, vpa_advect, scratch_dummy, external_source_settings,
            num_diss_params, t_params, ion_dt, ir, evolve_p, :explicit_v, false,
            add_identity)
        @begin_anyzv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            z_solve_counter += 1

            # Get LU-factorized matrix for implicit part of the solve
            @views fill_electron_kinetic_equation_z_only_Jacobian_f!(
                A, f_electron_new[ivpa,ivperp,:], electron_p_new,
                dpdf_dz[ivpa,ivperp,:], dpdf_dvpa[ivpa,ivperp,:],
                d2pdf_dvpa2[ivpa,ivperp,:], z_speed[:,ivpa,ivperp], moments,
                zeroth_moment, first_moment, second_moment, third_moment,
                dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params, ion_dt, ir,
                ivperp, ivpa, add_identity)

            A_sparse = sparse(A.matrix)
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

            adi_info.z_solve_explicit_matrices[z_solve_counter] = sparse(@view(explicit_J.matrix[adi_info.z_solve_global_inds[z_solve_counter],:]))
        end

        A_p = adi_info.z_solve_jacobian_p
        @begin_anyzv_region(true)
        @anyzv_serial_region begin
            # Do the solve for p on the rank-0 process, which has the fewest grid
            # points to handle if there are not an exactly equal number of points for
            # each process.
            z_solve_counter += 1

            # Get LU-factorized matrix for implicit part of the solve
            @views fill_electron_kinetic_equation_z_only_Jacobian_p!(
                A_p, electron_p_new, f_electron_new[1,1,:], dpdf_dz[1,1,:],
                dpdf_dvpa[1,1,:], d2pdf_dvpa2[1,1,:], z_speed[:,1,1], moments,
                zeroth_moment, first_moment, second_moment, third_moment,
                dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params, ion_dt, ir,
                evolve_p, add_identity)

            A_sparse = sparse(A_p.matrix)
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

            adi_info.z_solve_explicit_matrices[z_solve_counter] = sparse(@view(explicit_J.matrix[adi_info.z_solve_global_inds[z_solve_counter],:]))
        end
        @debug_consistency_checks z_solve_counter == adi_info.z_solve_nsolve || error("z_solve_counter($z_solve_counter) != z_solve_nsolve($(adi_info.z_solve_nsolve))")
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
            @begin_anyzv_vperp_vpa_region()
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

            @begin_anyzv_region()
            @anyzv_serial_region begin
                buffer_1[] = output_buffer_p_view[1]
                buffer_2[] = output_buffer_p_view[end]
            end
            reconcile_element_boundaries_MPI_anyzv!(
                output_buffer_p_view, buffer_1, buffer_2, buffer_3, buffer_4, z)

            return nothing
        end

        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            row = (iz - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
            this_input_buffer[row] = precon_f[ivpa,ivperp,iz]
        end
        @begin_anyzv_z_region()
        @loop_z iz begin
            row = pdf_size + iz
            this_input_buffer[row] = precon_p[iz]
        end
        @_anyzv_subblock_synchronize()

        # Use this to copy current guess from output_buffer to
        # intermediate_buffer, to avoid race conditions as new guess is
        # written into output_buffer.
        function fill_intermediate_buffer!()
            @_anyzv_subblock_synchronize()
            for i ∈ global_index_subrange
                this_intermediate_buffer[i] = this_output_buffer[i]
            end
            @_anyzv_subblock_synchronize()
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
        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            row = (iz - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
            precon_f[ivpa,ivperp,iz] = this_output_buffer[row]
        end
        @begin_anyzv_z_region()
        @loop_z iz begin
            row = pdf_size + iz
            precon_p[iz] = this_output_buffer[row]
        end

        return nothing
    end

    left_preconditioner = identity
    right_preconditioner = adi_precon!

    return left_preconditioner, right_preconditioner, recalculate_adi_preconditioner!
end

function get_electron_preconditioner(nl_solver_params, f_electron_new, electron_p_new,
                                     buffer_1, buffer_2, buffer_3, buffer_4,
                                     electron_density, electron_upar, this_phi, moments,
                                     collisions, composition, z, vperp, vpa, z_spectral,
                                     vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                                     scratch_dummy, external_source_settings,
                                     num_diss_params, t_params, ion_dt, ir, evolve_p,
                                     add_identity=true)
    if nl_solver_params.preconditioner_type === Val(:electron_lu)
        return get_electron_lu_preconditioner(nl_solver_params, f_electron_new,
                                              electron_p_new, buffer_1, buffer_2,
                                              buffer_3, buffer_4, electron_density,
                                              electron_upar, this_phi, moments,
                                              collisions, composition, z, vperp, vpa,
                                              z_spectral, vperp_spectral, vpa_spectral,
                                              z_advect, vpa_advect, scratch_dummy,
                                              external_source_settings, num_diss_params,
                                              t_params, ion_dt, ir, evolve_p,
                                              add_identity)
    elseif nl_solver_params.preconditioner_type === Val(:electron_adi)
        return get_electron_adi_preconditioner(nl_solver_params, f_electron_new,
                                               electron_p_new, buffer_1, buffer_2,
                                               buffer_3, buffer_4, electron_density,
                                               electron_upar, this_phi, moments,
                                               collisions, composition, z, vperp, vpa,
                                               z_spectral, vperp_spectral, vpa_spectral,
                                               z_advect, vpa_advect, scratch_dummy,
                                               external_source_settings, num_diss_params,
                                               t_params, ion_dt, ir, evolve_p,
                                               add_identity)
    elseif nl_solver_params.preconditioner_type === Val(:none)
        left_preconditioner = identity
        right_preconditioner = identity
        # Nothing to recalculate.
        recalculate_none_preconditioner = ()->nothing

        return left_preconditioner, right_preconditioner, recalculate_none_preconditioner
    else
        error("preconditioner_type=$(nl_solver_params.preconditioner_type) is not "
              * "supported.")
    end
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
@timeit global_timer electron_backward_euler!(old_scratch, new_scratch, moments,
            phi::AbstractArray{mk_float,ndim_field}, collisions, composition,
            r::coordinate, z::coordinate, vperp::coordinate, vpa::coordinate, z_spectral,
            vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
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
    this_phi = @view phi[:,ir]

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

    left_preconditioner, right_preconditioner, recalculate_preconditioner! =
        get_electron_preconditioner(nl_solver_params, f_electron_new, electron_p_new,
                                    buffer_1, buffer_2, buffer_3, buffer_4,
                                    electron_density, electron_upar, this_phi, moments,
                                    collisions, composition, z, vperp, vpa, z_spectral,
                                    vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                                    scratch_dummy, external_source_settings,
                                    num_diss_params, t_params, ion_dt, ir, evolve_p)

    # Does preconditioner need to be recalculated?
    if t_params.dt[] > 1.5 * nl_solver_params.precon_dt[] ||
            t_params.dt[] < 2.0/3.0 * nl_solver_params.precon_dt[]

        # dt has changed significantly, so update the preconditioner
        nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
    end
    if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
        recalculate_preconditioner!()
    end

    # Do a backward-Euler update of the electron pdf, and (if evove_p=true) the electron
    # parallel pressure.
    function residual_func!(this_residual, new_variables; krylov=false)
        electron_p_residual, f_electron_residual = this_residual
        electron_p_newvar, f_electron_newvar = new_variables

        if evolve_p
            this_vth = @view moments.electron.vth[:,ir]
            @begin_anyzv_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                this_vth[iz] = sqrt(abs(2.0 * electron_p_newvar[iz] /
                                        (electron_density[iz] * composition.me_over_mi)))
            end
        end

        # enforce the boundary condition(s) on the electron pdf
        @views enforce_boundary_condition_on_electron_pdf!(
                   f_electron_newvar, this_phi, moments.electron.vth[:,ir], electron_upar,
                   z, vperp, vpa, vperp_spectral, vpa_spectral, vpa_advect, moments,
                   num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
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
            @begin_anyzv_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = electron_p_old[iz,ir]
            end
        else
            @begin_anyzv_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = 0.0
            end
        end

        # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
        # electron_pdf member of the first argument, so if we set the electron_pdf member
        # of the first argument to zero, and pass dt=1, then it will evaluate the time
        # derivative, which is the residual for a steady-state solution.
        @begin_anyzv_z_vperp_vpa_region()
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
        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_electron_residual[ivpa,ivperp,iz] = f_electron_newvar[ivpa,ivperp,iz] - f_electron_residual[ivpa,ivperp,iz]
        end
        if evolve_p
            @begin_anyzv_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = electron_p_newvar[iz] - electron_p_residual[iz]
            end
        end

        # Set residual to zero where pdf_electron is determined by boundary conditions.
        if vpa.n > 1
            @begin_anyzv_z_vperp_region()
            @loop_z_vperp iz ivperp begin
                @views enforce_v_boundary_condition_local!(f_electron_residual[:,ivperp,iz], vpa.bc,
                                                           vpa_advect[1].speed[:,ivperp,iz,ir],
                                                           num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                           vpa, vpa_spectral)
            end
        end
        if vperp.n > 1
            @begin_anyzv_z_vpa_region()
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
                         num_diss_params, r::coordinate, z::coordinate, vperp::coordinate,
                         vpa::coordinate, r_spectral, z_spectral, vperp_spectral,
                         vpa_spectral, z_advect, vpa_advect, gyroavs, scratch_dummy,
                         t_params, ion_dt, nl_solver_params) = begin

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
    @begin_r_anyzv_region()
    @loop_r ir begin
        if r.n > 1 && !r.periodic && r.irank == 0 && ir == 1
            # No need to solve on point that will be set by boundary conditions
            continue
        end

        f_electron = @view pdf_electron_out[:,:,:,ir]
        p = @view electron_p_out[:,ir]
        this_phi = @view fields.phi[:,ir]

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        left_preconditioner, right_preconditioner, recalculate_preconditioner! =
            @views get_electron_preconditioner(nl_solver_params, f_electron, p, buffer_1,
                                               buffer_2, buffer_3, buffer_4,
                                               moments.electron.dens[:,ir],
                                               moments.electron.upar[:,ir], this_phi,
                                               moments, collisions, composition, z, vperp,
                                               vpa, z_spectral, vperp_spectral,
                                               vpa_spectral, z_advect, vpa_advect,
                                               scratch_dummy, external_source_settings,
                                               num_diss_params, t_params, ion_dt, ir,
                                               true, false)

        # Does preconditioner need to be recalculated?
        if ion_dt > 1.5 * nl_solver_params.precon_dt[] ||
                ion_dt < 2.0/3.0 * nl_solver_params.precon_dt[]

            # dt has changed significantly, so update the preconditioner
            nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
        end
        if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
            recalculate_preconditioner!()
        end

        function residual_func!(residual, new_variables; debug=false, krylov=false)
            electron_p_residual, f_electron_residual = residual
            electron_p_new, f_electron_new = new_variables

            this_vth = moments.electron.vth
            @begin_anyzv_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                this_vth[iz,ir] = sqrt(abs(2.0 * electron_p_new[iz,ir] /
                                           (electron_density[iz] *
                                            composition.me_over_mi)))
            end

            # enforce the boundary condition(s) on the electron pdf
            @views enforce_boundary_condition_on_electron_pdf!(
                       f_electron_new, this_phi, moments.electron.vth[:,ir],
                       electron_upar, z, vperp, vpa, vperp_spectral, vpa_spectral,
                       vpa_advect, moments,
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

            @begin_anyzv_z_region()
            @loop_z iz begin
                electron_p_residual[iz] = 0.0
            end

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            @begin_anyzv_z_vperp_vpa_region()
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
                @begin_anyzv_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(f_electron_residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                @begin_anyzv_z_vpa_region()
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

function apply_electron_bc_and_constraints_no_r!(
             f_electron, phi::AbstractVector{mk_float}, moments, r::coordinate,
             z::coordinate, vperp::coordinate, vpa::coordinate, vperp_spectral,
             vpa_spectral, vpa_advect, num_diss_params, composition, ir, nl_solver_params,
             scratch_dummy)
    @begin_anyzv_z_vperp_vpa_region()
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
    @anyzv_serial_region begin
        # Only the return value from rank-0 of each shared memory block matters
        MPI.Allreduce!(bc_converged, &, z.comm)
    end
    MPI.Bcast!(bc_converged, comm_anyzv_subblock[]; root=0)
    if !bc_converged[]
        return "electron_bc"
    end

    if nl_solver_params !== nothing
        # Check if either vcut_ind has changed - if either has, recalculate the
        # preconditioner because the response at the grid point before the cutoff is
        # sharp, so the preconditioner could be significantly wrong when it was
        # calculated using the wrong vcut_ind.
        lower_vcut_changed = @view scratch_dummy.int_buffer_rs_1[ir:ir,1]
        upper_vcut_changed = @view scratch_dummy.int_buffer_rs_2[ir:ir,1]
        @anyzv_serial_region begin
            if z.irank == 0
                precon_lowerz_vcut_inds = nl_solver_params.precon_lowerz_vcut_inds
                if new_lowerz_vcut_ind[] == precon_lowerz_vcut_inds[ir]
                    lower_vcut_changed[] = 0
                else
                    lower_vcut_changed[] = 1
                    precon_lowerz_vcut_inds[ir] = new_lowerz_vcut_ind[]
                end
            end
            MPI.Bcast!(lower_vcut_changed, z.comm; root=0)
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
            MPI.Bcast!(upper_vcut_changed, z.comm; root=z.nrank-1)
            #req2 = MPI.Ibcast!(upper_vcut_changed, comm_inter_block[]; root=n_blocks[]-1)

            # Eventually can use Ibcast!() to make the two broadcasts run
            # simultaneously, but need the function to be merged into MPI.jl (see
            # https://github.com/JuliaParallel/MPI.jl/pull/882).
            #MPI.Waitall([req1, req2])
        end
        @_anyzv_subblock_synchronize()
        if lower_vcut_changed[] == 1 || upper_vcut_changed[] == 1
            # One or both of the indices changed for some `ir`, so force the
            # preconditioner to be recalculated next time.
            nl_solver_params.solves_since_precon_update[] = nl_solver_params.preconditioner_update_interval
        end
    end

    @begin_anyzv_z_region()
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

function get_cutoff_params_lower(upar::mk_float, vthe::mk_float, phi::mk_float,
                                 me_over_mi::mk_float, vpa::coordinate)
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

function get_cutoff_params_upper(upar::mk_float, vthe::mk_float, phi::mk_float,
                                 me_over_mi::mk_float, vpa::coordinate)
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
                         pdf::AbstractArray{mk_float,3}, phi::AbstractVector{mk_float},
                         vthe::AbstractVector{mk_float}, upar::AbstractVector{mk_float},
                         z::coordinate, vperp::coordinate, vpa::coordinate,
                         vperp_spectral, vpa_spectral, vpa_adv, moments, vpa_diffusion,
                         me_over_mi, ir; bc_constraints=true, update_vcut=true,
                         lowerz_vcut_ind=nothing, upperz_vcut_ind=nothing,
                         allow_failure=true) = begin

    @debug_consistency_checks bc_constraints && !update_vcut && error("update_vcut is not used when bc_constraints=true, but update_vcut has non-default value")

    newton_tol = 1.0e-13

    # Enforce velocity-space boundary conditions
    if vpa.n > 1
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            # enforce the vpa BC
            # use that adv.speed independent of vpa
            @views enforce_v_boundary_condition_local!(pdf[:,ivperp,iz], vpa.bc,
                                                       vpa_adv[1].speed[:,ivperp,iz,ir],
                                                       vpa_diffusion, vpa, vpa_spectral)
        end
    end
    if vperp.n > 1
        @begin_anyzv_z_vpa_region()
        enforce_vperp_boundary_condition!(pdf, vperp.bc, vperp, vperp_spectral, ir)
    end

    if z.periodic
        # Nothing more to do for z-periodic boundary conditions
        return true
    elseif z.bc == "constant"
        @begin_anyzv_vperp_vpa_region()
        density_offset = 1.0
        vwidth = 1.0/sqrt(me_over_mi)
        dens = @view moments.electron.dens[:,ir]
        if z.irank == 0
            @loop_vperp_vpa ivperp ivpa begin
                u = upar[1]
                vthe = vth[1]
                speed = vpa.grid[ivpa] * vthe + u
                if speed > 0.0
                    pdf[ivpa,ivperp,1] = density_offset / dens[1] * vthe[1] * exp(-(speed^2 + vperp.grid[ivperp]^2)/vwidth^2)
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_vperp_vpa ivperp ivpa begin
                u = upar[end]
                vthe = vth[end]
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

    @begin_anyzv_region()

    newton_max_its = 100
    if vperp.n == 1
        this_integral_correction_sharpness = integral_correction_sharpness / 3.0
    else
        this_integral_correction_sharpness = integral_correction_sharpness
    end

    @anyzv_serial_region begin
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
                    get_cutoff_params_lower(upar[1], vthe[1], phi[1], me_over_mi, vpa)

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
                    get_cutoff_params_upper(upar[end], vthe[end], phi[end], me_over_mi,
                                            vpa)

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
        @begin_anyzv_vperp_vpa_region()
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
                         jacobian::jacobian_info,
                         phi::Union{mk_float,AbstractVector{mk_float}},
                         pdf::Union{AbstractArray{mk_float,3},AbstractMatrix{mk_float}},
                         p::Union{mk_float,AbstractVector{mk_float}},
                         vthe::Union{mk_float,AbstractVector{mk_float}},
                         upar::Union{mk_float,AbstractVector{mk_float}}, z::coordinate,
                         vperp::coordinate,
                         vpa::coordinate, vperp_spectral, vpa_spectral, vpa_adv, moments,
                         vpa_diffusion, me_over_mi, ir, include, iz=nothing) = begin
    if z.bc != "wall" || include === :explicit_z
        return nothing
    end

    jacobian_matrix = jacobian.matrix
    pdf_offset = jacobian.state_vector_offsets[1]
    p_offset = jacobian.state_vector_offsets[2]

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
        shared_mem_parallel && @begin_anyzv_vperp_region()
        @loop_vperp ivperp begin
            # Skip velocity space boundary points.
            if vperp.n > 1 && ivperp == vperp.n
                continue
            end

            # Get matrix entries for the response of the sheath-edge boundary condition.
            # Ignore constraints, as these are non-linear and also should be small
            # corrections which should not matter much for a preconditioner.

            jac_range = pdf_offset+(ivperp-1)*vpa.n+1 : pdf_offset+ivperp*vpa.n
            jacobian_zbegin = @view jacobian_matrix[jac_range,jac_range]
            jacobian_zbegin_p = @view jacobian_matrix[jac_range,p_offset+1]

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
        shared_mem_parallel && @begin_anyzv_vperp_region()
        @loop_vperp ivperp begin
            # Skip vperp boundary points.
            if vperp.n > 1 && ivperp == vperp.n
                continue
            end

            # Get matrix entries for the response of the sheath-edge boundary condition.
            # Ignore constraints, as these are non-linear and also should be small
            # corrections which should not matter much for a preconditioner.

            jac_range = pdf_offset+(zend-1)*vperp.n*vpa.n+(ivperp-1)*vpa.n+1 : pdf_offset+(zend-1)*vperp.n*vpa.n+ivperp*vpa.n
            jacobian_zend = @view jacobian_matrix[jac_range,jac_range]
            jacobian_zend_p = @view jacobian_matrix[jac_range,p_offset+zend]

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
                                       external_source_settings, num_diss_params, ir;
                                       evolve_p=false)

Check the error estimate for the embedded RK method and adjust the timestep if
appropriate.
"""
@timeit global_timer electron_adaptive_timestep_update!(
                         scratch, t, t_params, moments, phi::AbstractVector{mk_float},
                         z_advect, vpa_advect, composition, r::coordinate, z::coordinate,
                         vperp::coordinate, vpa::coordinate, vperp_spectral, vpa_spectral,
                         external_source_settings, num_diss_params, scratch_dummy, ir;
                         evolve_p=false, local_max_dt=Inf) = begin
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
    @begin_anyzv_vperp_vpa_region(true)
    @views update_electron_speed_z!(z_advect[1], moments.electron.upar[:,ir],
                                    moments.electron.vth[:,ir], vpa.grid, ir)
    z_CFL = get_minimum_CFL_z(z_advect[1].speed, z, ir)
    if block_rank[] == 0
        CFL_limits["CFL_z"] = t_params.CFL_prefactor * z_CFL
    else
        CFL_limits["CFL_z"] = Inf
    end

    # vpa-advection
    @begin_anyzv_z_vperp_region()
    @views update_electron_speed_vpa!(vpa_advect[1], moments.electron.dens[:,ir],
                                      moments.electron.upar[:,ir],
                                      scratch[t_params.n_rk_stages+1].electron_p[:,ir],
                                      moments, composition.me_over_mi, vpa.grid,
                                      external_source_settings.electron, ir)
    vpa_CFL = get_minimum_CFL_vpa(vpa_advect[1].speed, vpa, ir)
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
    rk_loworder_solution!(scratch, nothing, :pdf_electron, t_params; ir=ir)
    if evolve_p
        @begin_anyzv_z_region()
        rk_loworder_solution!(scratch, nothing, :electron_p, t_params; ir=ir)

        # Make vth consistent with `scratch[2]`, as it is needed for the electron pdf
        # boundary condition.
        @views update_electron_vth_temperature!(moments, scratch[2].electron_p[:,ir],
                                                moments.electron.dens[:,ir], composition,
                                                ir)
    end
    @views apply_electron_bc_and_constraints_no_r!(
               scratch[t_params.n_rk_stages+1].pdf_electron[:,:,:,ir], phi, moments, r, z,
               vperp, vpa, vperp_spectral, vpa_spectral, vpa_advect, num_diss_params,
               composition, ir, nothing, scratch_dummy)
    if evolve_p
        # Reset vth in the `moments` struct to the result consistent with full-accuracy RK
        # solution.
        @begin_anyzv_z_region()
        @views update_electron_vth_temperature!(moments,
                                                scratch[t_params.n_rk_stages+1].electron_p[:,ir],
                                                moments.electron.dens, composition, ir)
    end

    pdf_error = local_error_norm(scratch[2].pdf_electron,
                                 scratch[t_params.n_rk_stages+1].pdf_electron,
                                 t_params.rtol, t_params.atol; method=error_norm_method,
                                 skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                 error_sum_zero=t_params.error_sum_zero, ir=ir)
    error_norms["pdf_accuracy"] = pdf_error
    push!(total_points, vpa.n_global * vperp.n_global * z.n_global)

    # Calculate error for moments, if necessary
    if evolve_p
        @begin_anyzv_z_region()
        p_err = local_error_norm(scratch[2].electron_p,
                                 scratch[t_params.n_rk_stages+1].electron_p,
                                 t_params.rtol, t_params.atol; method=error_norm_method,
                                 skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                 error_sum_zero=t_params.error_sum_zero, ir=ir)
        error_norms["p_accuracy"] = p_err
        push!(total_points, z.n_global)
    end

    adaptive_timestep_update_t_params!(t_params, CFL_limits, error_norms, total_points,
                                       error_norm_method, "", 0.0, false, false,
                                       composition, z; electron=true,
                                       local_max_dt=local_max_dt, ir=ir)
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
                         result_object, f_in::AbstractArray{mk_float,3},
                         p_in::AbstractVector{mk_float}, moments, z::coordinate,
                         vperp::coordinate, vpa::coordinate, z_spectral, vpa_spectral,
                         z_advect, vpa_advect, scratch_dummy, collisions, composition,
                         external_source_settings, num_diss_params, t_params, ir;
                         evolve_p=false, ion_dt=nothing, soft_force_constraints=false,
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
                                                         composition.electron_physics, ir)
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
                                              t_params.constraint_forcing_rate, vperp,
                                              vpa, dt, ir)
        write_debug_IO("electron_implicit_constraint_forcing!")
    end

    return nothing
end

"""
"""
function get_electron_sub_terms(
             dens_array::AbstractVector{mk_float},
             ddens_dz_array::AbstractVector{mk_float},
             upar_array::AbstractVector{mk_float},
             dupar_dz_array::AbstractVector{mk_float}, p_array::AbstractVector{mk_float},
             dp_dz_array::AbstractVector{mk_float},
             dvth_dz_array::AbstractVector{mk_float},
             zeroth_moment_array::AbstractVector{mk_float},
             first_moment_array::AbstractVector{mk_float},
             second_moment_array::AbstractVector{mk_float},
             third_moment_array::AbstractVector{mk_float},
             dthird_moment_dz_array::AbstractVector{mk_float},
             dq_dz_array::AbstractVector{mk_float},
             upar_ion_array::AbstractVector{mk_float},
             pdf_array::AbstractArray{mk_float,3},
             dpdf_dz_array::AbstractArray{mk_float,3},
             dpdf_dvpa_array::AbstractArray{mk_float,3},
             d2pdf_dvpa2_array::Union{AbstractArray{mk_float,3},Nothing}, me::mk_float,
             moments, collisions, composition, external_source_settings, num_diss_params,
             t_params, ion_dt, z::coordinate, vperp::coordinate, vpa::coordinate,
             z_speed::AbstractArray{mk_float,3}, vpa_speed::AbstractArray{mk_float,3},
             ir::mk_int, separate_zeroth_moment::Bool, separate_first_moment::Bool,
             separate_second_moment::Bool, separate_third_moment::Bool,
             separate_dp_dz::Bool, separate_dq_dz::Bool, include::Symbol=:all;
             include_qpar_integral_terms::Bool=true)

    if composition.electron_physics == kinetic_electrons_with_temperature_equation
        error("kinetic_electrons_with_temperature_equation not "
              * "supported yet in preconditioner")
    elseif composition.electron_physics != kinetic_electrons
        error("Unsupported electron_physics=$(composition.electron_physics) "
              * "in electron_backward_euler!() preconditioner.")
    end
    if num_diss_params.electron.moment_dissipation_coefficient > 0.0
        error("z-diffusion of electron_p not yet supported in "
              * "preconditioner")
    end
    if collisions.electron_fluid.nu_ei > 0.0
        error("electron-ion collision terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_charge_exchange_frequency > 0.0
        error("electron 'charge exchange' terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_ionization_frequency > 0.0
        error("electron ionization terms for electron_p not yet "
              * "supported in preconditioner")
    end

    vpa_dissipation_coefficient = num_diss_params.electron.vpa_dissipation_coefficient
    constraint_forcing_rate = t_params.constraint_forcing_rate
    n = ConstantTerm(dens_array; z=z)
    dn_dz = ConstantTerm(ddens_dz_array; z=z)
    u = ConstantTerm(upar_array; z=z)
    du_dz = ConstantTerm(dupar_dz_array; z=z)
    if  include === :explicit_z
        p = ConstantTerm(p_array; z=z)
    else
        p = EquationTerm(:electron_p, p_array; z=z)
    end
    dp_dz = EquationTerm(:electron_p, dp_dz_array; derivatives=[:z], z=z)
    if separate_dp_dz
        dp_dz_constraint_rhs = dp_dz
        dp_dz = EquationTerm(:electron_dp_dz, dp_dz_array; z=z)
    else
        dp_dz_constraint_rhs = NullTerm()
    end
    vth = sqrt(2.0 / me) * p^0.5 * n^(-0.5)
    if vperp.n == 1
        ppar = 3.0 * p
        dppar_dz = 3.0 * dp_dz
    else
        error("Support for 2V electron_ppar not implemented yet in "
              * "`get_electron_energy_equation_term()`.")
    end
    u_ion = ConstantTerm(upar_ion_array; z=z)

    if include ∈ (:explicit_z, :explicit_v)
        f = ConstantTerm(pdf_array; vpa=vpa, vperp=vperp, z=z)
    else
        f = EquationTerm(:electron_pdf, pdf_array; vpa=vpa, vperp=vperp, z=z)
    end
    if include === :explicit_v
        df_dz = ConstantTerm(dpdf_dz_array; vpa=vpa, vperp=vperp, z=z)
    else
        df_dz = EquationTerm(:electron_pdf, dpdf_dz_array; derivatives=[:z],
                             upwind_speeds=[z_speed], vpa=vpa, vperp=vperp, z=z)
    end
    if include === :explicit_z
        df_dvpa = ConstantTerm(dpdf_dvpa_array; vpa=vpa, vperp=vperp, z=z)
    else
        df_dvpa = EquationTerm(:electron_pdf, dpdf_dvpa_array; derivatives=[:vpa],
                               upwind_speeds=[vpa_speed], vpa=vpa, vperp=vperp, z=z)
    end
    if d2pdf_dvpa2_array === nothing
        d2f_dvpa2 = NullTerm()
    elseif include === :explicit_z
        d2f_dvpa2 = ConstantTerm(d2pdf_dvpa2_array; vpa=vpa, vperp=vperp, z=z)
    else
        d2f_dvpa2 = EquationTerm(:electron_pdf, d2pdf_dvpa2_array;
                                 second_derivatives=[:vpa], vpa=vpa, vperp=vperp, z=z)
    end

    wpa = ConstantTerm(vpa.grid; vpa=vpa)
    wperp = ConstantTerm(vperp.grid; vperp=vperp)

    @debug_consistency_checks include !== :all && separate_third_moment && error("separate_third_moment is not supported for ADI preconditioner, so must have include=:all. Got include=$include")

    if include_qpar_integral_terms
        third_moment_integrand_prefactor = wpa*(wpa^2 + wperp^2)
        if include === :explicit_z
            zeroth_moment = ConstantTerm(zeroth_moment_array; z=z)
            zeroth_moment_constraint_rhs = NullTerm()
            first_moment = ConstantTerm(first_moment_array; z=z)
            first_moment_constraint_rhs = NullTerm()
            second_moment = ConstantTerm(second_moment_array; z=z)
            second_moment_constraint_rhs = NullTerm()
            third_moment = ConstantTerm(third_moment_array; z=z)
            third_moment_constraint_rhs = NullTerm()
        else
            zeroth_moment = EquationTerm(:electron_pdf, zeroth_moment_array;
                                         integrand_coordinates=[vpa,vperp,z],
                                         integrand_prefactor=ConstantTerm(ones(mk_float)),
                                         z=z)
            if separate_zeroth_moment
                zeroth_moment_constraint_rhs = zeroth_moment
                zeroth_moment = EquationTerm(:zeroth_moment, zeroth_moment_array; z=z)
            else
                zeroth_moment_constraint_rhs = NullTerm()
            end
            first_moment = EquationTerm(:electron_pdf, first_moment_array;
                                        integrand_coordinates=[vpa,vperp,z],
                                        integrand_prefactor=wpa,
                                        z=z)
            if separate_first_moment
                first_moment_constraint_rhs = first_moment
                first_moment = EquationTerm(:first_moment, first_moment_array; z=z)
            else
                first_moment_constraint_rhs = NullTerm()
            end
            second_moment = EquationTerm(:electron_pdf, second_moment_array;
                                         integrand_coordinates=[vpa,vperp,z],
                                         integrand_prefactor=(wpa^2 + wperp^2),
                                         z=z)
            if separate_second_moment
                second_moment_constraint_rhs = second_moment
                second_moment = EquationTerm(:second_moment, second_moment_array; z=z)
            else
                second_moment_constraint_rhs = NullTerm()
            end
            third_moment = EquationTerm(:electron_pdf, third_moment_array;
                                        integrand_coordinates=[vpa,vperp,z],
                                        integrand_prefactor=third_moment_integrand_prefactor,
                                        z=z)
            if separate_third_moment
                third_moment_constraint_rhs = third_moment
                third_moment = EquationTerm(:third_moment, third_moment_array; z=z)
            else
                third_moment_constraint_rhs = NullTerm()
            end
        end
        if separate_third_moment
            dthird_moment_dz = EquationTerm(:third_moment, dthird_moment_dz_array;
                                            derivatives=[:z], z=z)
        else
            dthird_moment_dz = EquationTerm(:electron_pdf, dthird_moment_dz_array;
                                            derivatives=[:z],
                                            integrand_coordinates=[vpa,vperp,z],
                                            integrand_prefactor=third_moment_integrand_prefactor,
                                            z=z)
        end
    else
        zeroth_moment = ConstantTerm(zeroth_moment_array; z=z)
        zeroth_moment_constraint_rhs = NullTerm()
        first_moment = ConstantTerm(first_moment_array; z=z)
        second_moment = ConstantTerm(second_moment_array; z=z)
        third_moment = ConstantTerm(third_moment_array; z=z)
        dthird_moment_dz = ConstantTerm(dthird_moment_dz_array; z=z)
        third_moment_constraint_rhs = NullTerm()
    end

    dvth_dz_expanded = sqrt(2.0 / me) * 0.5 * (p^(-0.5) * n^(-0.5) * dp_dz
                                               - p^(0.5) * n^(-1.5) * dn_dz)
    dvth_dz = CompoundTerm(dvth_dz_expanded, dvth_dz_array; z=z)

    dq_dz_expanded = sqrt(2.0/me) * ((-0.5) * p^1.5 * third_moment * n^(-1.5) * dn_dz +
                                     1.5 * n^(-0.5) * third_moment * p^0.5 * dp_dz +
                                     n^(-0.5) * p^1.5 * dthird_moment_dz
                                    )
    dq_dz = CompoundTerm(dq_dz_expanded, dq_dz_array; z=z)
    if separate_dq_dz
        dq_dz_constraint_rhs = dq_dz
        dq_dz = EquationTerm(:electron_dq_dz, dq_dz_array; z=z)
    else
        dq_dz_constraint_rhs = NullTerm()
    end

    source_type = collect(s.source_type for s ∈ external_source_settings.electron)
    source_amplitude = collect(ConstantTerm(@view(moments.electron.external_source_amplitude[:,ir,index]); z=z)
                               for index ∈ eachindex(external_source_settings.electron))
    source_T_array = collect(ConstantTerm(@view s.source_T_array[:,ir]; z=z)
                             for s ∈ external_source_settings.electron)
    density_source = collect(ConstantTerm(@view(moments.electron.external_source_density_amplitude[:,ir,index]); z=z)
                             for index ∈ eachindex(external_source_settings.electron))
    momentum_source = collect(ConstantTerm(@view(moments.electron.external_source_momentum_amplitude[:,ir,index]); z=z)
                              for index ∈ eachindex(external_source_settings.electron))
    pressure_source = collect(ConstantTerm(@view(moments.electron.external_source_pressure_amplitude[:,ir,index]); z=z)
                              for index ∈ eachindex(external_source_settings.electron))
    if vperp.n == 1
        source_vth_factor = collect(sqrt(me / 2.0) * T^(-0.5) for T ∈ source_T_array)
        source_this_vth_factor = vth
    else
        source_vth_factor = collect((me / 2.0)^1.5 * T^(-1.5) for T ∈ source_T_array)
        source_this_vth_factor = vth^3
    end

    nuee0 = collisions.krook.nuee0
    nuei0 = collisions.krook.nuei0
    if vperp.n == 1
        # 1V case
        krook_adjust_vth_1V = sqrt(3.0)
        krook_adjust_1V = 1.0 / sqrt(3.0)
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        krook_adjust_vth_1V = 1.0
        krook_adjust_1V = 1.0
        Maxwellian_prefactor = 1.0 / π^1.5
    end

    return ElectronSubTerms(; me, vpa_dissipation_coefficient, constraint_forcing_rate,
                            ion_dt, n, dn_dz, u, du_dz, p, dp_dz, dp_dz_constraint_rhs,
                            vth, dvth_dz, ppar, dppar_dz, zeroth_moment,
                            zeroth_moment_constraint_rhs, first_moment,
                            first_moment_constraint_rhs, second_moment,
                            second_moment_constraint_rhs, third_moment, dthird_moment_dz,
                            third_moment_constraint_rhs, dq_dz, dq_dz_constraint_rhs,
                            u_ion, wperp, wpa, f, df_dz, df_dvpa, d2f_dvpa2, source_type,
                            source_amplitude, source_T_array, density_source,
                            momentum_source, pressure_source, source_vth_factor,
                            source_this_vth_factor, collisions, nuee0, nuei0,
                            krook_adjust_vth_1V, krook_adjust_1V, Maxwellian_prefactor)
end

function get_electron_sub_terms_z_only_Jacobian(
             dens_array::AbstractVector{mk_float},
             ddens_dz_array::AbstractVector{mk_float},
             upar_array::AbstractVector{mk_float},
             dupar_dz_array::AbstractVector{mk_float}, p_array::AbstractVector{mk_float},
             dp_dz_array::AbstractVector{mk_float},
             dvth_dz_array::AbstractVector{mk_float},
             zeroth_moment_array::AbstractVector{mk_float},
             first_moment_array::AbstractVector{mk_float},
             second_moment_array::AbstractVector{mk_float},
             third_moment_array::AbstractVector{mk_float},
             dthird_moment_dz_array::AbstractVector{mk_float},
             dq_dz_array::AbstractVector{mk_float},
             upar_ion_array::AbstractVector{mk_float},
             pdf_array::AbstractVector{mk_float},
             dpdf_dz_array::AbstractVector{mk_float},
             dpdf_dvpa_array::AbstractVector{mk_float},
             d2pdf_dvpa2_array::Union{AbstractVector{mk_float},Nothing}, me::mk_float,
             moments, collisions, external_source_settings, num_diss_params, t_params,
             ion_dt, z::coordinate, vperp::coordinate, vpa::coordinate,
             z_speed::AbstractVector{mk_float}, ir::mk_int, ivperp::mk_int, ivpa::mk_int;
             include_qpar_integral_terms::Bool=true)

    # Handle boundary condition skip here so that `add_term_to_Jacobian()` does not need
    # to know about ivperp/ivpa.
    if (vperp.n > 1 && ivperp == vperp.n) || ivpa == 1 || ivpa == vpa.n
        scalar_terms = (:me, :vpa_dissipation_coefficient, :constraint_forcing_rate,
                        :ion_dt, :nuee0, :nuei0, :krook_adjust_vth_1V, :krook_adjust_1V,
                        :Maxwellian_prefactor)
        vector_terms = (:source_amplitude, :source_T_array, :density_source,
                        :momentum_source, :pressure_source, :source_vth_factor)
        return ElectronSubTerms(; (f ∈ scalar_terms ? f=>1.0 :
                                   f ∈ vector_terms ? f=>mk_float[] :
                                   f === :collisions ? f=>nothing :
                                   f === :source_type ? f=>String[] :
                                   f=>NullTerm()
                                   for f ∈ fieldnames(ElectronSubTerms))...)
    end

    vpa_dissipation_coefficient = num_diss_params.electron.vpa_dissipation_coefficient
    constraint_forcing_rate = t_params.constraint_forcing_rate
    n = ConstantTerm(dens_array; z=z)
    dn_dz = ConstantTerm(ddens_dz_array; z=z)
    u = ConstantTerm(upar_array; z=z)
    du_dz = ConstantTerm(dupar_dz_array; z=z)
    p = ConstantTerm(p_array; z=z)
    dp_dz = ConstantTerm(dp_dz_array; z=z)
    vth = sqrt(2.0 / me) * p^0.5 * n^(-0.5)
    if vperp.n == 1
        ppar = 3.0 * p
        dppar_dz = 3.0 * dp_dz
    else
        error("Support for 2V electron_ppar not implemented yet in "
              * "`get_electron_energy_equation_term()`.")
    end
    u_ion = ConstantTerm(upar_ion_array; z=z)

    f = EquationTerm(:electron_pdf, pdf_array; z=z)
    df_dz = EquationTerm(:electron_pdf, dpdf_dz_array; derivatives=[:z],
                         upwind_speeds=[z_speed], z=z)
    df_dvpa = ConstantTerm(dpdf_dvpa_array; z=z)
    if d2pdf_dvpa2_array === nothing
        d2f_dvpa2 = NullTerm()
    else
        d2f_dvpa2 = ConstantTerm(d2pdf_dvpa2_array; z=z)
    end

    wperp = vperp.grid[ivperp]
    wpa = vpa.grid[ivpa]

    zeroth_moment = ConstantTerm(zeroth_moment_array; z=z)
    zeroth_moment_constraint_rhs = NullTerm()
    first_moment = ConstantTerm(first_moment_array; z=z)
    first_moment_constraint_rhs = NullTerm()
    second_moment = ConstantTerm(second_moment_array; z=z)
    second_moment_constraint_rhs = NullTerm()
    third_moment = ConstantTerm(third_moment_array; z=z)
    dthird_moment_dz = ConstantTerm(dthird_moment_dz_array; z=z)
    third_moment_constraint_rhs = NullTerm()
    dp_dz_constraint_rhs = NullTerm()
    dvth_dz_expanded = sqrt(2.0 / me) * 0.5 * (p^(-0.5) * n^(-0.5) * dp_dz
                                               - p^(0.5) * n^(-1.5) * dn_dz)
    dvth_dz = CompoundTerm(dvth_dz_expanded, dvth_dz_array; z=z)
    dq_dz_expanded = sqrt(2.0/me) * ((-0.5) * p^1.5 * third_moment * n^(-1.5) * dn_dz
                                     + 1.5 * n^(-0.5) * third_moment * p^0.5 * dp_dz
                                     + n^(-0.5) * p^1.5 * dthird_moment_dz
                                    )
    dq_dz = CompoundTerm(dq_dz_expanded, dq_dz_array; z=z)
    dq_dz_constraint_rhs = NullTerm()

    source_type = collect(s.source_type for s ∈ external_source_settings.electron)
    source_amplitude = collect(ConstantTerm(@view(moments.electron.external_source_amplitude[:,ir,index]); z=z)
                               for index ∈ eachindex(external_source_settings.electron))
    source_T_array = collect(ConstantTerm(@view s.source_T_array[:,ir]; z=z)
                             for s ∈ external_source_settings.electron)
    density_source = collect(ConstantTerm(@view(moments.electron.external_source_density_amplitude[:,ir,index]); z=z)
                             for index ∈ eachindex(external_source_settings.electron))
    momentum_source = collect(ConstantTerm(@view(moments.electron.external_source_momentum_amplitude[:,ir,index]); z=z)
                              for index ∈ eachindex(external_source_settings.electron))
    pressure_source = collect(ConstantTerm(@view(moments.electron.external_source_pressure_amplitude[:,ir,index]); z=z)
                              for index ∈ eachindex(external_source_settings.electron))
    if vperp.n == 1
        source_vth_factor = collect(sqrt(me / 2.0) * T^(-0.5) for T ∈ source_T_array)
        source_this_vth_factor = vth
    else
        source_vth_factor = collect((me / 2.0)^1.5 * T^(-1.5) for T ∈ source_T_array)
        source_this_vth_factor = vth^3
    end

    nuee0 = collisions.krook.nuee0
    nuei0 = collisions.krook.nuei0
    if vperp.n == 1
        # 1V case
        krook_adjust_vth_1V = sqrt(3.0)
        krook_adjust_1V = 1.0 / sqrt(3.0)
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        krook_adjust_vth_1V = 1.0
        krook_adjust_1V = 1.0
        Maxwellian_prefactor = 1.0 / π^1.5
    end

    return ElectronSubTerms(; me, vpa_dissipation_coefficient, constraint_forcing_rate,
                            ion_dt, n, dn_dz, u, du_dz, p, dp_dz, dp_dz_constraint_rhs,
                            vth, dvth_dz, ppar, dppar_dz, zeroth_moment,
                            zeroth_moment_constraint_rhs, first_moment,
                            first_moment_constraint_rhs, second_moment,
                            second_moment_constraint_rhs, third_moment, dthird_moment_dz,
                            third_moment_constraint_rhs, dq_dz, dq_dz_constraint_rhs,
                            u_ion, wperp, wpa, f, df_dz, df_dvpa, d2f_dvpa2, source_type,
                            source_amplitude, source_T_array, density_source,
                            momentum_source, pressure_source, source_vth_factor,
                            source_this_vth_factor, collisions, nuee0, nuei0,
                            krook_adjust_vth_1V, krook_adjust_1V, Maxwellian_prefactor)
end

function get_electron_sub_terms_v_only_Jacobian(
             n::mk_float, dn_dz::mk_float, u::mk_float, du_dz::mk_float,
             p_array::AbstractArray{mk_float,0}, dp_dz::mk_float,
             dvth_dz_array::AbstractArray{mk_float,0},
             zeroth_moment_array::AbstractArray{mk_float,0},
             first_moment_array::AbstractArray{mk_float,0},
             second_moment_array::AbstractArray{mk_float,0},
             third_moment_array::AbstractArray{mk_float,0},
             dthird_moment_dz::mk_float, dq_dz_array::AbstractArray{mk_float,0},
             u_ion::mk_float, pdf_array::AbstractMatrix{mk_float},
             dpdf_dz_array::AbstractMatrix{mk_float},
             dpdf_dvpa_array::AbstractMatrix{mk_float},
             d2pdf_dvpa2_array::Union{AbstractMatrix{mk_float},Nothing}, me::mk_float,
             moments, collisions, external_source_settings, num_diss_params, t_params,
             ion_dt, z::coordinate, vperp::coordinate, vpa::coordinate,
             z_speed::AbstractMatrix{mk_float}, vpa_speed::AbstractMatrix{mk_float},
             ir::mk_int, iz::mk_int; include_qpar_integral_terms::Bool=true)

    vpa_dissipation_coefficient = num_diss_params.electron.vpa_dissipation_coefficient
    constraint_forcing_rate = t_params.constraint_forcing_rate
    p = EquationTerm(:electron_p, p_array)
    vth = sqrt(2.0 / me) * p^0.5 * n^(-0.5)
    if vperp.n == 1
        ppar = 3.0 * p
        dppar_dz = 3.0 * dp_dz
    else
        error("Support for 2V electron_ppar not implemented yet in "
              * "`get_electron_energy_equation_term()`.")
    end

    f = EquationTerm(:electron_pdf, pdf_array; vpa=vpa, vperp=vperp)
    df_dz = ConstantTerm(dpdf_dz_array; vpa=vpa, vperp=vperp)
    df_dvpa = EquationTerm(:electron_pdf, dpdf_dvpa_array; derivatives=[:vpa],
                           upwind_speeds=[vpa_speed], vpa=vpa, vperp=vperp)
    if d2pdf_dvpa2_array === nothing
        d2f_dvpa2 = NullTerm()
    else
        d2f_dvpa2 = EquationTerm(:electron_pdf, d2pdf_dvpa2_array;
                                 second_derivatives=[:vpa], vpa=vpa, vperp=vperp)
    end

    wpa = ConstantTerm(vpa.grid; vpa=vpa)
    wperp = ConstantTerm(vperp.grid; vperp=vperp)

    third_moment_integrand_prefactor = wpa*(wpa^2 + wperp^2)
    zeroth_moment = EquationTerm(:electron_pdf, zeroth_moment_array;
                                 integrand_coordinates=[vpa,vperp],
                                 integrand_prefactor=ConstantTerm(ones(mk_float)))
    zeroth_moment_constraint_rhs = NullTerm()
    first_moment = EquationTerm(:electron_pdf, first_moment_array;
                                integrand_coordinates=[vpa,vperp],
                                integrand_prefactor=wpa)
    first_moment_constraint_rhs = NullTerm()
    second_moment = EquationTerm(:electron_pdf, second_moment_array;
                                 integrand_coordinates=[vpa,vperp],
                                 integrand_prefactor=(wpa^2 + wperp^2))
    second_moment_constraint_rhs = NullTerm()
    third_moment = EquationTerm(:electron_pdf, third_moment_array;
                                integrand_coordinates=[vpa,vperp],
                                integrand_prefactor=third_moment_integrand_prefactor)
    third_moment_constraint_rhs = NullTerm()
    dp_dz_constraint_rhs = NullTerm()
    dvth_dz_expanded = sqrt(2.0 / me) * 0.5 * (p^(-0.5) * n^(-0.5) * dp_dz
                                               - p^(0.5) * n^(-1.5) * dn_dz)
    dvth_dz = CompoundTerm(dvth_dz_expanded, dvth_dz_array)
    dq_dz_expanded = sqrt(2.0/me) * ((-0.5) * p^1.5 * third_moment * n^(-1.5) * dn_dz
                                     + 1.5 * n^(-0.5) * third_moment * p^0.5 * dp_dz
                                     + n^(-0.5) * p^1.5 * dthird_moment_dz
                                    )
    dq_dz = CompoundTerm(dq_dz_expanded, dq_dz_array)
    dq_dz_constraint_rhs = NullTerm()

    source_type = collect(s.source_type for s ∈ external_source_settings.electron)
    source_amplitude = collect(moments.electron.external_source_amplitude[iz,ir,index]
                               for index ∈ eachindex(external_source_settings.electron))
    source_T_array = collect(s.source_T_array[iz,ir]
                             for s ∈ external_source_settings.electron)
    density_source = collect(moments.electron.external_source_density_amplitude[iz,ir,index]
                             for index ∈ eachindex(external_source_settings.electron))
    momentum_source = collect(moments.electron.external_source_momentum_amplitude[iz,ir,index]
                              for index ∈ eachindex(external_source_settings.electron))
    pressure_source = collect(moments.electron.external_source_pressure_amplitude[iz,ir,index]
                              for index ∈ eachindex(external_source_settings.electron))
    if vperp.n == 1
        source_vth_factor = collect(sqrt(me / 2.0) * T^(-0.5) for T ∈ source_T_array)
        source_this_vth_factor = vth
    else
        source_vth_factor = collect((me / 2.0)^1.5 * T^(-1.5) for T ∈ source_T_array)
        source_this_vth_factor = vth^3
    end


    nuee0 = collisions.krook.nuee0
    nuei0 = collisions.krook.nuei0
    if vperp.n == 1
        # 1V case
        krook_adjust_vth_1V = sqrt(3.0)
        krook_adjust_1V = 1.0 / sqrt(3.0)
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        krook_adjust_vth_1V = 1.0
        krook_adjust_1V = 1.0
        Maxwellian_prefactor = 1.0 / π^1.5
    end

    # Modify z_speed argument to indicate whether this is a z-boundary and which boundary
    # it is, because `add_term_to_Jacobian()` does not know about `iz`.
    z_speed = get_ADI_boundary_v_solve_z_speed(z_speed, z, iz)

    return ElectronSubTerms(; me, vpa_dissipation_coefficient, constraint_forcing_rate,
                            ion_dt, n, dn_dz, u, du_dz, p, dp_dz, dp_dz_constraint_rhs,
                            vth, dvth_dz, ppar, dppar_dz, zeroth_moment,
                            zeroth_moment_constraint_rhs, first_moment,
                            first_moment_constraint_rhs, second_moment,
                            second_moment_constraint_rhs, third_moment, dthird_moment_dz,
                            third_moment_constraint_rhs, dq_dz, dq_dz_constraint_rhs,
                            u_ion, wperp, wpa, f, df_dz, df_dvpa, d2f_dvpa2, source_type,
                            source_amplitude, source_T_array, density_source,
                            momentum_source, pressure_source, source_vth_factor,
                            source_this_vth_factor, collisions, nuee0, nuei0,
                            krook_adjust_vth_1V, krook_adjust_1V,
                            Maxwellian_prefactor), z_speed
end

"""
"""
function get_all_electron_terms(sub_terms::ElectronSubTerms)
    pdf_terms = get_electron_z_advection_term(sub_terms)
    pdf_terms += get_electron_vpa_advection_term(sub_terms)
    pdf_terms += get_contribution_from_electron_pdf_term(sub_terms)
    pdf_terms += get_electron_dissipation_term(sub_terms)
    pdf_terms += get_electron_krook_collisions_term(sub_terms)
    pdf_terms += get_total_external_electron_source_term(sub_terms)
    pdf_terms += get_electron_implicit_constraint_forcing_term(sub_terms)

    p_terms = get_electron_energy_equation_term(sub_terms)
    p_terms += get_ion_dt_forcing_of_electron_p_term(sub_terms)

    zeroth_moment_terms = -sub_terms.zeroth_moment_constraint_rhs
    first_moment_terms = -sub_terms.first_moment_constraint_rhs
    second_moment_terms = -sub_terms.second_moment_constraint_rhs
    third_moment_terms = -sub_terms.third_moment_constraint_rhs
    dp_dz_terms = -sub_terms.dp_dz_constraint_rhs
    dq_dz_terms = -sub_terms.dq_dz_constraint_rhs

    return pdf_terms, p_terms, zeroth_moment_terms, first_moment_terms,
           second_moment_terms, third_moment_terms, dp_dz_terms, dq_dz_terms
end

"""
    fill_electron_kinetic_equation_Jacobian!(jacobian, f, p, moments, phi, collisions,
                                             composition, z, vperp, vpa, z_spectral,
                                             vperp_specral, vpa_spectral, z_advect,
                                             vpa_advect, scratch_dummy,
                                             external_source_settings, num_diss_params,
                                             t_params, ion_dt, ir, evolve_p, include=:all,
                                             include_qpar_integral_terms=true,
                                             add_identity=true)

Fill a `jacobian_info` object with the Jacobian matrix for electron kinetic equation and
(if `evolve_p=true`) the electron energy equation.

`add_identity=false` can be passed to skip adding 1's on the diagonal. Doing this would
produce the Jacobian for a steady-state solve, rather than a backward-Euler timestep
solve. [Note that rows representing boundary points still need the 1 added to their
diagonal, as that 1 is used to impose the boundary condition, not to represent the 'new f'
in the time derivative term as it is for the non-boundary points.]
"""
@timeit global_timer fill_electron_kinetic_equation_Jacobian!(
                         jacobian::jacobian_info, f::AbstractArray{mk_float,3},
                         p::AbstractVector{mk_float}, moments,
                         phi::AbstractVector{mk_float}, collisions, composition,
                         z::coordinate, vperp::coordinate, vpa::coordinate, z_spectral,
                         vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, evolve_p, include::Symbol=:all,
                         include_qpar_integral_terms=true, add_identity=true) = begin
    @debug_consistency_checks t_params.electron === nothing || error("electron t_params should be passed to fill_electron_kinetic_equation!(), but got ion t_params.")
    dt = t_params.dt[]

    buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

    vth = @view moments.electron.vth[:,ir]
    me = composition.me_over_mi
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    qpar = @view moments.electron.qpar[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dupar_dz = @view moments.electron.dupar_dz[:,ir]
    dp_dz = @view moments.electron.dp_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]

    upar_ion = @view moments.ion.upar[:,ir,1]

    # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
    third_moment = @view scratch_dummy.buffer_zrs_1[:,ir,1]
    dthird_moment_dz = @view scratch_dummy.buffer_zrs_2[:,ir,1]
    @begin_anyzv_z_region()
    @loop_z iz begin
        third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
    end
    derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                        buffer_4, z_spectral, z)

    z_speed = @view z_advect[1].speed[:,:,:,ir]

    dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    @begin_anyzv_vperp_vpa_region()
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

    dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
    @begin_anyzv_z_vperp_region()
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
    vpa_speed = @view vpa_advect[1].speed[:,:,:,ir]

    d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
    # If not using electron vpa dissipation, the value of d2pdf_dvpa2 won't actually be
    # needed.
    if num_diss_params.electron.vpa_dissipation_coefficient > 0.0
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end
    end

    zeroth_moment = @view scratch_dummy.buffer_zrs_3[:,ir,1]
    first_moment = @view scratch_dummy.buffer_zrs_4[:,ir,1]
    second_moment = @view scratch_dummy.buffer_zrs_5[:,ir,1]
    @begin_anyzv_z_region()
    @loop_z iz begin
        @views zeroth_moment[iz] = integral(f[:,:,iz], vpa.grid, 0, vpa.wgts, vperp.grid,
                                            0, vperp.wgts)
        @views first_moment[iz] = integral(f[:,:,iz], vpa.grid, 1, vpa.wgts, vperp.grid,
                                           0, vperp.wgts)
        @views second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2), f[:,:,iz],
                                            vperp, vpa)
    end

    if include === :all
        if add_identity
            jacobian_initialize_identity!(jacobian)
        else
            jacobian_initialize_bc_diagonal!(jacobian, z_speed)
        end
    else
        jacobian_initialize_zero!(jacobian)
    end

    separate_zeroth_moment = (:zeroth_moment ∈ jacobian.state_vector_entries)
    separate_first_moment = (:first_moment ∈ jacobian.state_vector_entries)
    separate_second_moment = (:second_moment ∈ jacobian.state_vector_entries)
    separate_third_moment = (:third_moment ∈ jacobian.state_vector_entries)
    separate_dp_dz = (:electron_dp_dz ∈ jacobian.state_vector_entries)
    separate_dq_dz = (:electron_dq_dz ∈ jacobian.state_vector_entries)
    sub_terms = get_electron_sub_terms(dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz,
                                       zeroth_moment, first_moment, second_moment,
                                       third_moment, dthird_moment_dz, dqpar_dz, upar_ion,
                                       f, dpdf_dz, dpdf_dvpa, d2pdf_dvpa2, me, moments,
                                       collisions, composition, external_source_settings,
                                       num_diss_params, t_params, ion_dt, z, vperp, vpa,
                                       z_speed, vpa_speed, ir, separate_zeroth_moment,
                                       separate_first_moment, separate_second_moment,
                                       separate_third_moment, separate_dp_dz,
                                       separate_dq_dz, include;
                                       include_qpar_integral_terms=include_qpar_integral_terms)
    pdf_terms, p_terms, zeroth_moment_terms, first_moment_terms, second_moment_terms,
        third_moment_terms, dp_dz_terms, dq_dz_terms = get_all_electron_terms(sub_terms)

    add_term_to_Jacobian!(jacobian, :electron_pdf, dt, pdf_terms, z_speed)
    if t_params.include_wall_bc_in_preconditioner
        add_wall_boundary_condition_to_Jacobian!(
            jacobian, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral, vpa_spectral,
            vpa_advect, moments, num_diss_params.electron.vpa_dissipation_coefficient, me,
            ir, include)
    end

    add_term_to_Jacobian!(jacobian, :electron_p, dt, p_terms)
    add_term_to_Jacobian!(jacobian, :zeroth_moment, 1.0, zeroth_moment_terms)
    add_term_to_Jacobian!(jacobian, :first_moment, 1.0, first_moment_terms)
    add_term_to_Jacobian!(jacobian, :second_moment, 1.0, second_moment_terms)
    add_term_to_Jacobian!(jacobian, :third_moment, 1.0, third_moment_terms)
    add_term_to_Jacobian!(jacobian, :electron_dp_dz, 1.0, dp_dz_terms)
    add_term_to_Jacobian!(jacobian, :electron_dq_dz, 1.0, dq_dz_terms)

    return nothing
end

"""
    fill_electron_kinetic_equation_v_only_Jacobian!()
        jacobian_matrix, f, p, dpdf_dz, dpdf_dvpa, d2pdf_dvpa2, z_speed, vpa_speed,
        moments, zeroth_moment, first_moment, second_moment, third_moment,
        dthird_moment_dz, phi, collisions, composition, z, vperp, vpa, z_spectral,
        vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
        external_source_settings, num_diss_params, t_params, ion_dt, ir, iz, evolve_p,
        add_idenity=true)

Fill a pre-allocated matrix with the Jacobian matrix for a velocity-space solve part of
the ADI method for electron kinetic equation and (if `evolve_p=true`) the electron energy
equation.
"""
@timeit global_timer fill_electron_kinetic_equation_v_only_Jacobian!(
                         jacobian::jacobian_info, f::AbstractMatrix{mk_float},
                         p::AbstractArray{mk_float,0}, dpdf_dz::AbstractMatrix{mk_float},
                         dpdf_dvpa::AbstractMatrix{mk_float},
                         d2pdf_dvpa2::AbstractMatrix{mk_float},
                         z_speed::AbstractMatrix{mk_float},
                         vpa_speed::AbstractMatrix{mk_float}, moments,
                         zeroth_moment::AbstractArray{mk_float,0},
                         first_moment::AbstractArray{mk_float,0},
                         second_moment::AbstractArray{mk_float,0},
                         third_moment::AbstractArray{mk_float,0},
                         dthird_moment_dz::mk_float, phi::mk_float, collisions,
                         composition, z::coordinate, vperp::coordinate, vpa::coordinate,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, iz, evolve_p, add_identity=true) = begin
    @debug_consistency_checks t_params.electron === nothing || error("electron t_params should be passed to fill_electron_kinetic_equation_v_only_Jacobian!(), but got ion t_params.")
    dt = t_params.dt[]

    vth = moments.electron.vth[iz,ir]
    me = composition.me_over_mi
    dens = moments.electron.dens[iz,ir]
    upar = moments.electron.upar[iz,ir]
    ddens_dz = moments.electron.ddens_dz[iz,ir]
    dupar_dz = moments.electron.dupar_dz[iz,ir]
    dp_dz = moments.electron.dp_dz[iz,ir]
    dvth_dz = @view(moments.electron.dvth_dz[iz,ir])
    dqpar_dz = @view(moments.electron.dqpar_dz[iz,ir])

    upar_ion = moments.ion.upar[iz,ir,1]

    if add_identity
        jacobian_initialize_identity!(jacobian)
    else
        jacobian_initialize_bc_diagonal!(jacobian, z_speed)
    end

    sub_terms, this_z_speed = get_electron_sub_terms_v_only_Jacobian(
        dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz, zeroth_moment, first_moment,
        second_moment, third_moment, dthird_moment_dz, dqpar_dz, upar_ion, f, dpdf_dz,
        dpdf_dvpa, d2pdf_dvpa2, me, moments, collisions, external_source_settings,
        num_diss_params, t_params, ion_dt, z, vperp, vpa, z_speed, vpa_speed, ir, iz)
    pdf_terms, p_terms = get_all_electron_terms(sub_terms)

    add_term_to_Jacobian!(jacobian, :electron_pdf, dt, pdf_terms, this_z_speed)
    if t_params.include_wall_bc_in_preconditioner
        add_wall_boundary_condition_to_Jacobian!(
            jacobian, phi, f, p[], vth, upar, z, vperp, vpa, vperp_spectral, vpa_spectral,
            vpa_advect, moments, num_diss_params.electron.vpa_dissipation_coefficient, me,
            ir, :implicit_v, iz)
    end

    add_term_to_Jacobian!(jacobian, :electron_p, dt, p_terms)

    return nothing
end

"""
    fill_electron_kinetic_equation_z_only_Jacobian_f!(
        jacobian_matrix, f, p, dpdf_dz, dpdf_dvpa, z_speed, moments, zeroth_moment,
        first_moment, second_moment, third_moment, dthird_moment_dz, collisions,
        composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
        vpa_advect, scratch_dummy, external_source_settings, num_diss_params, t_params,
        ion_dt, ir, ivperp, ivpa, add_idenity=true)

Fill a pre-allocated matrix with the Jacobian matrix for a z-direction solve part of the
ADI method for the electron kinetic equation.
"""
@timeit global_timer fill_electron_kinetic_equation_z_only_Jacobian_f!(
                         jacobian::jacobian_info, f::AbstractVector{mk_float},
                         p::AbstractVector{mk_float}, dpdf_dz::AbstractVector{mk_float},
                         dpdf_dvpa::AbstractVector{mk_float},
                         d2pdf_dvpa2::AbstractVector{mk_float},
                         z_speed::AbstractVector{mk_float}, moments,
                         zeroth_moment::AbstractVector{mk_float},
                         first_moment::AbstractVector{mk_float},
                         second_moment::AbstractVector{mk_float},
                         third_moment::AbstractVector{mk_float},
                         dthird_moment_dz::AbstractVector{mk_float}, collisions,
                         composition, z::coordinate, vperp::coordinate, vpa::coordinate,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, ivperp, ivpa, add_identity=true) = begin
    @debug_consistency_checks t_params.electron === nothing || error("electron t_params should be passed to fill_electron_kinetic_equation_z_only_Jacobian!(), but got ion t_params.")
    dt = t_params.dt[]

    me = composition.me_over_mi
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dupar_dz = @view moments.electron.dupar_dz[:,ir]
    dp_dz = @view moments.electron.dp_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]

    upar_ion = @view moments.ion.upar[:,ir,1]

    if add_identity
        jacobian_initialize_identity!(jacobian)
    else
        jacobian_initialize_bc_diagonal!(jacobian, z_speed)
    end

    sub_terms = @views get_electron_sub_terms_z_only_Jacobian(
        dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz, zeroth_moment, first_moment,
        second_moment, third_moment, dthird_moment_dz, dqpar_dz, upar_ion, f, dpdf_dz,
        dpdf_dvpa, d2pdf_dvpa2, me, moments, collisions, external_source_settings,
        num_diss_params, t_params, ion_dt, z, vperp, vpa, z_speed, ir, ivperp, ivpa)
    pdf_terms, p_terms = get_all_electron_terms(sub_terms)

    add_term_to_Jacobian!(jacobian, :electron_pdf, dt, pdf_terms, z_speed)

    return nothing
end

"""
    fill_electron_kinetic_equation_z_only_Jacobian_p!(
        jacobian_matrix, p, moments, zeroth_moment, first_moment, second_moment,
        third_moment, dthird_moment_dz, collisions, composition, z, vperp, vpa,
        z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
        external_source_settings, num_diss_params, t_params, ion_dt, ir, evolve_p,
        add_identity=true)

Fill a pre-allocated matrix with (if `evolve_p=true`) the Jacobian matrix for a
z-direction solve part of the ADI method for the electron energy equation.
"""
@timeit global_timer fill_electron_kinetic_equation_z_only_Jacobian_p!(
                         jacobian::jacobian_info, p::AbstractVector{mk_float},
                         f::AbstractVector{mk_float}, dpdf_dz::AbstractVector{mk_float},
                         dpdf_dvpa::AbstractVector{mk_float},
                         d2pdf_dvpa2::AbstractVector{mk_float},
                         z_speed::AbstractVector{mk_float}, moments,
                         zeroth_moment::AbstractVector{mk_float},
                         first_moment::AbstractVector{mk_float},
                         second_moment::AbstractVector{mk_float},
                         third_moment::AbstractVector{mk_float},
                         dthird_moment_dz::AbstractVector{mk_float}, collisions,
                         composition, z::coordinate, vperp::coordinate, vpa::coordinate,
                         z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                         scratch_dummy, external_source_settings, num_diss_params,
                         t_params, ion_dt, ir, evolve_p, add_identity=true) = begin
    @debug_consistency_checks t_params.electron === nothing || error("electron t_params should be passed to fill_electron_kinetic_equation_z_only_Jacobian_p!(), but got ion t_params.")
    dt = t_params.dt[]

    me = composition.me_over_mi
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    ppar = @view moments.electron.ppar[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dupar_dz = @view moments.electron.dupar_dz[:,ir]
    dp_dz = @view moments.electron.dp_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]

    upar_ion = @view moments.ion.upar[:,ir]

    if add_identity
        jacobian_initialize_identity!(jacobian)
    else
        jacobian_initialize_bc_diagonal!(jacobian, z_speed)
    end

    if !evolve_p
        # Not solving for `p`, so leave jacobian for `p` as identity.
        return nothing
    end

    sub_terms = @views get_electron_sub_terms_z_only_Jacobian(
        dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz, zeroth_moment, first_moment,
        second_moment, third_moment, dthird_moment_dz, dqpar_dz, upar_ion, f, dpdf_dz,
        dpdf_dvpa, d2pdf_dvpa2, me, moments, collisions, external_source_settings,
        num_diss_params, t_params, ion_dt, z, vperp, vpa, z_speed, ir, 1, 1)
    pdf_terms, p_terms = get_all_electron_terms(sub_terms)

    add_term_to_Jacobian!(jacobian, :electron_p, dt, p_terms, z_speed)

    return nothing
end

"""
"""
@timeit global_timer add_dissipation_term!(pdf_out::AbstractArray{mk_float,3},
                                           pdf_in::AbstractArray{mk_float,3},
                                           scratch_dummy, z_spectral, z::coordinate,
                                           vpa::coordinate, vpa_spectral, num_diss_params,
                                           dt) = begin
    if num_diss_params.electron.vpa_dissipation_coefficient ≤ 0.0
        return nothing
    end

    @begin_anyzv_z_vperp_region()
    @loop_z_vperp iz ivperp begin
        @views second_derivative!(vpa.scratch, pdf_in[:,ivperp,iz], vpa, vpa_spectral)
        @views @. pdf_out[:,ivperp,iz] += dt * num_diss_params.electron.vpa_dissipation_coefficient * vpa.scratch
    end
    return nothing
end

function get_electron_dissipation_term(sub_terms::ElectronSubTerms)

    vpa_dissipation_coefficient = sub_terms.vpa_dissipation_coefficient

    if vpa_dissipation_coefficient ≤ 0.0
        return NullTerm()
    end

    d2f_dvpa2 = sub_terms.d2f_dvpa2

    term = -vpa_dissipation_coefficient * d2f_dvpa2

    return term
end

"""
add contribution to the kinetic equation coming from the term proportional to the pdf
"""
@timeit global_timer add_contribution_from_pdf_term!(
                         pdf_out::AbstractArray{mk_float,3},
                         pdf_in::AbstractArray{mk_float,3}, p::AbstractVector{mk_float},
                         dens::AbstractVector{mk_float}, upar::AbstractVector{mk_float},
                         moments, vpa_grid, z::coordinate, dt,
                         electron_source_settings, ir) = begin
    vth = @view moments.electron.vth[:,ir]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
    @begin_anyzv_z_vperp_region()
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
                      vpa_grid * (this_dvth_dz - this_vth * this_ddens_dz / this_dens)) *
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

function get_contribution_from_electron_pdf_term(sub_terms::ElectronSubTerms)

    # Terms from `add_contribution_from_pdf_term!()`
    #   ( 1/3/p*dq/dz + w_∥*(vth/n*dn/dz - dvth/dz) + ∑(-0.5*source_pressure_amplitude/p + 1.5*source_density_amplitude/n) )*g

    me = sub_terms.me
    n = sub_terms.n
    dn_dz = sub_terms.dn_dz
    p = sub_terms.p
    dp_dz = sub_terms.dp_dz
    wpa = sub_terms.wpa
    f = sub_terms.f
    vth = sub_terms.vth
    dvth_dz = sub_terms.dvth_dz
    dq_dz = sub_terms.dq_dz

    term = (
            1.0 / 3.0 * dq_dz * p^(-1)
            + wpa * (vth * n^(-1) * dn_dz - dvth_dz)
           )

    for (density_source, momentum_source, pressure_source) ∈ zip(sub_terms.density_source, sub_terms.momentum_source, sub_terms.pressure_source)
        term += (
                 - 0.5 * pressure_source * p^(-1)
                 + 1.5 * density_source * n^(-1)
                )
    end

    term *= f

    return term
end

function get_ion_dt_forcing_of_electron_p_term(sub_terms::ElectronSubTerms)

    ion_dt = sub_terms.ion_dt

    if ion_dt === nothing
        return NullTerm()
    end

    p = sub_terms.p

    # Note don't need to include constant term involving p_previous_ion_step, as this
    # does not contribute to the Jacobian.
    term = 1.0 / ion_dt * p

    return term
end

end
