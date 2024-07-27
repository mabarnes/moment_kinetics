module electron_kinetic_equation

using LinearAlgebra
using MPI

export get_electron_critical_velocities

using ..looping
using ..analysis: steady_state_residuals
using ..derivatives: derivative_z!
using ..boundary_conditions: enforce_v_boundary_condition_local!,
                             enforce_vperp_boundary_condition!
using ..calculus: derivative!, second_derivative!, integral
using ..communication
using ..interpolation: interpolate_to_grid_1d!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float
using ..electron_fluid_equations: calculate_electron_moments!,
                                  update_electron_vth_temperature!,
                                  calculate_electron_qpar_from_pdf!,
                                  calculate_electron_parallel_friction_force!
using ..electron_fluid_equations: electron_energy_equation!, electron_energy_residual!
using ..electron_z_advection: electron_z_advection!, update_electron_speed_z!
using ..electron_vpa_advection: electron_vpa_advection!, update_electron_speed_vpa!
using ..em_fields: update_phi!
using ..external_sources: external_electron_source!
using ..file_io: get_electron_io_info, write_electron_state, finish_electron_io
using ..krook_collisions: electron_krook_collisions!
using ..moment_constraints: hard_force_moment_constraints!,
                            moment_constraints_on_residual!
using ..moment_kinetics_structs: scratch_pdf, scratch_electron_pdf, electron_pdf_substruct
using ..nonlinear_solvers: newton_solve!
using ..runge_kutta: rk_update_variable!, rk_loworder_solution!, local_error_norm,
                     adaptive_timestep_update_t_params!
using ..utils: get_minimum_CFL_z, get_minimum_CFL_vpa
using ..velocity_moments: integrate_over_vspace, calculate_electron_moment_derivatives!

"""
update_electron_pdf is a function that uses the electron kinetic equation 
to solve for the updated electron pdf

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor

    INPUTS:
    scratch = `scratch_pdf` struct used to store Runge-Kutta stages
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    ppar = electron parallel pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
    dt = time step size
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
    ion_dt = if this is passed, the electron pressure is evolved in a form that results in
             a backward-Euler update on the ion timestep (ion_dt) once the electron
             pseudo-timestepping reaches steady state.
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf!(scratch, pdf, moments, phi, r, z, vperp, vpa, z_spectral,
        vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
        collisions, composition, external_source_settings, num_diss_params,
        max_electron_pdf_iterations; io_electron=nothing, initial_time=nothing,
        residual_tolerance=nothing, evolve_ppar=false, ion_dt=nothing)

    # set the method to use to solve the electron kinetic equation
    solution_method = "artificial_time_derivative"
    #solution_method = "shooting_method"
    #solution_method = "picard_iteration"
    # solve the electron kinetic equation using the specified method
    if solution_method == "artificial_time_derivative"
        return update_electron_pdf_with_time_advance!(scratch, pdf, moments, phi,
            collisions, composition, r, z, vperp, vpa, z_spectral, vperp_spectral,
            vpa_spectral, z_advect, vpa_advect, scratch_dummy, t_params,
            external_source_settings, num_diss_params, max_electron_pdf_iterations;
            io_electron=io_electron, initial_time=initial_time,
            residual_tolerance=residual_tolerance, evolve_ppar=evolve_ppar, ion_dt=ion_dt)
    elseif solution_method == "shooting_method"
        dens = moments.electron.dens
        vthe = moments.electron.vth
        ppar = moments.electron.ppar
        qpar = moments.electron.qpar
        qpar_updated = moments.electron.qpar_updated
        ddens_dz = moments.electron.ddens_dz
        dppar_dz = moments.electron.dppar_dz
        dqpar_dz = moments.electron.dqpar_dz
        dvth_dz = moments.electron.dvth_dz
        return update_electron_pdf_with_shooting_method!(pdf, dens, vthe, ppar, qpar,
            qpar_updated, phi, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, z, vpa,
            vpa_spectral, scratch_dummy, composition)
    elseif solution_method == "picard_iteration"
        dens = moments.electron.dens
        vthe = moments.electron.vth
        ppar = moments.electron.ppar
        qpar = moments.electron.qpar
        qpar_updated = moments.electron.qpar_updated
        ddens_dz = moments.electron.ddens_dz
        dppar_dz = moments.electron.dppar_dz
        dqpar_dz = moments.electron.dqpar_dz
        dvth_dz = moments.electron.dvth_dz
        return update_electron_pdf_with_picard_iteration!(pdf, dens, vthe, ppar, ddens_dz,
            dppar_dz, dqpar_dz, dvth_dz, z, vpa, vpa_spectral, scratch_dummy,
            max_electron_pdf_iterations)
    else
        error("!!! invalid solution method specified !!!")
    end
    return nothing    
end

"""
update_electron_pdf_with_time_advance is a function that introduces an artifical time derivative to advance 
the electron kinetic equation until a steady-state solution is reached.

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor

    INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    ppar = electron parallel pressure
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
    io_electron = info struct for binary file I/O
    initial_time = initial value for the (pseudo-)time
    ion_dt = if this is passed, the electron pressure is evolved in a form that results in
             a backward-Euler update on the ion timestep (ion_dt) once the electron
             pseudo-timestepping reaches steady state.
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_time_advance!(scratch, pdf, moments, phi, collisions,
        composition, r, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
        vpa_advect, scratch_dummy, t_params, external_source_settings, num_diss_params,
        max_electron_pdf_iterations; io_electron=nothing, initial_time=nothing,
        residual_tolerance=nothing, evolve_ppar=false, ion_dt=nothing)

    begin_r_z_region()

    if ion_dt !== nothing
        evolve_ppar = true
    end

    # create several (r) dimension dummy arrays for use in taking derivatives
    buffer_r_1 = @view scratch_dummy.buffer_rs_1[:,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[:,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[:,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[:,1]
    buffer_r_5 = @view scratch_dummy.buffer_rs_5[:,1]
    buffer_r_6 = @view scratch_dummy.buffer_rs_6[:,1]

    begin_r_z_region()
    @loop_r_z ir iz begin
        # update the electron thermal speed using the updated electron parallel pressure
        moments.electron.vth[iz,ir] = sqrt(abs(2.0 * moments.electron.ppar[iz,ir] /
                                                (moments.electron.dens[iz,ir] *
                                                 composition.me_over_mi)))
        scratch[t_params.n_rk_stages+1].electron_ppar[iz,ir] = moments.electron.ppar[iz,ir]
    end
    calculate_electron_moment_derivatives!(moments,
                                           (electron_density=moments.electron.dens,
                                            electron_upar=moments.electron.upar,
                                            electron_ppar=moments.electron.ppar),
                                           scratch_dummy, z, z_spectral,
                                           num_diss_params.electron.moment_dissipation_coefficient,
                                           composition.electron_physics)


    # compute the z-derivative of the input electron parallel heat flux, needed for the electron kinetic equation
    @views derivative_z!(moments.electron.dqpar_dz, moments.electron.qpar, buffer_r_1,
                         buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)

    if !evolve_ppar
        # ppar is not updated in the pseudo-timestepping loop below. So that we can read
        # ppar from the scratch structs, copy moments.electron.ppar into all of them.
        moments_ppar = moments.electron.ppar
        for istage ∈ 1:t_params.n_rk_stages+1
            scratch_ppar = scratch[istage].electron_ppar
            @loop_r_z ir iz begin
                scratch_ppar[iz,ir] = moments_ppar[iz,ir]
            end
        end
    end

    if initial_time !== nothing
        @serial_region begin
            t_params.t[] = initial_time
        end
        _block_synchronize()
        # Make sure that output times are set relative to this initial_time (the values in
        # t_params are set relative to 0.0).
        moments_output_times = t_params.moments_output_times .+ initial_time
        dfns_output_times = t_params.dfns_output_times .+ initial_time
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

    #z_speedup_fac = 20.0
    #z_speedup_fac = 5.0
    z_speedup_fac = 1.0

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
        begin_serial_region()
        @serial_region begin
            # open files to write the electron heat flux and pdf to file                                
            io_upar = open("upar$text_output_suffix.txt", "w")
            io_qpar = open("qpar$text_output_suffix.txt", "w")
            io_ppar = open("ppar$text_output_suffix.txt", "w")
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
                    println(io_ppar, "z: ", z.grid[iz], " ppar: ", scratch[t_params.n_rk_stages+1].electron_ppar[iz,1], " dppar_dz: ", moments.electron.dppar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                    println(io_vth, "z: ", z.grid[iz], " vthe: ", moments.electron.vth[iz,1], " dvth_dz: ", moments.electron.dvth_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter, " dens: ", dens[iz,1])
                end
                println(io_upar,"")
                println(io_qpar,"")
                println(io_ppar,"")
                println(io_vth,"")
            end
            io_pdf_stages = open("pdf_zright$text_output_suffix.txt", "w")
        end
    end

    begin_serial_region()
    t_params.moments_output_counter[] += 1
    @serial_region begin
        if io_electron !== nothing
            write_electron_state(scratch, moments, t_params, io_electron,
                                 t_params.moments_output_counter[], r, z, vperp, vpa)
        end
    end
    # evolve (artificially) in time until the residual is less than the tolerance
    while (!electron_pdf_converged
           && (t_params.step_counter[] - initial_step_counter < max_electron_pdf_iterations)
           && t_params.dt[] > 0.0 && !isnan(t_params.dt[]))

        # Set the initial values for the next step to the final values from the previous
        # step
        begin_r_z_vperp_vpa_region()
        new_pdf = scratch[1].pdf_electron
        old_pdf = scratch[t_params.n_rk_stages+1].pdf_electron
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            new_pdf[ivpa,ivperp,iz,ir] = old_pdf[ivpa,ivperp,iz,ir]
        end
        if evolve_ppar
            begin_r_z_region()
            new_ppar = scratch[1].electron_ppar
            old_ppar = scratch[t_params.n_rk_stages+1].electron_ppar
            @loop_r_z ir iz begin
                new_ppar[iz,ir] = old_ppar[iz,ir]
            end
        end

        for istage ∈ 1:t_params.n_rk_stages
            # Set the initial values for this stage to the final values from the previous
            # stage
            begin_r_z_vperp_vpa_region()
            new_pdf = scratch[istage+1].pdf_electron
            old_pdf = scratch[istage].pdf_electron
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                new_pdf[ivpa,ivperp,iz,ir] = old_pdf[ivpa,ivperp,iz,ir]
            end
            if evolve_ppar
                begin_r_z_region()
                new_ppar = scratch[istage+1].electron_ppar
                old_ppar = scratch[istage].electron_ppar
                @loop_r_z ir iz begin
                    new_ppar[iz,ir] = old_ppar[iz,ir]
                end
            end
            # Do a forward-Euler update of the electron pdf, and (if evove_ppar=true) the
            # electron parallel pressure.
            electron_kinetic_equation_euler_update!(scratch[istage+1], scratch[istage],
                                                    moments, z, vperp, vpa, z_spectral,
                                                    vpa_spectral, z_advect, vpa_advect,
                                                    scratch_dummy, collisions,
                                                    composition, external_source_settings,
                                                    num_diss_params, t_params.dt[];
                                                    evolve_ppar=evolve_ppar,
                                                    ion_dt=ion_dt)
            speedup_hack!(scratch[istage+1], scratch[istage], z_speedup_fac, z, vpa;
                          evolve_ppar=evolve_ppar)

            rk_update_variable!(scratch, nothing, :pdf_electron, t_params, istage)

            if evolve_ppar
                rk_update_variable!(scratch, nothing, :electron_ppar, t_params, istage)

                update_electron_vth_temperature!(moments, scratch[istage+1].electron_ppar,
                                                 moments.electron.dens, composition)
            end

            apply_electron_bc_and_constraints!(scratch[istage+1], phi, moments, z, vperp,
                                               vpa, vperp_spectral, vpa_spectral,
                                               vpa_advect, num_diss_params, composition,
                                               t_params.rtol)

            latest_pdf = scratch[istage+1].pdf_electron
            
            function update_derived_moments_and_derivatives(update_vth=false)
                # update the electron heat flux
                moments.electron.qpar_updated[] = false
                calculate_electron_qpar_from_pdf!(moments.electron.qpar,
                                                  scratch[istage+1].electron_ppar,
                                                  moments.electron.vth, latest_pdf, vpa)

                if evolve_ppar
                    this_ppar = scratch[istage+1].electron_ppar
                    this_dens = moments.electron.dens
                    this_upar = moments.electron.upar
                    if update_vth
                        begin_r_z_region()
                        this_vth = moments.electron.vth
                        @loop_r_z ir iz begin
                            # update the electron thermal speed using the updated electron
                            # parallel pressure
                            this_vth[iz,ir] = sqrt(abs(2.0 * this_ppar[iz,ir] /
                                                       (this_dens[iz,ir] *
                                                        composition.me_over_mi)))
                        end
                    end
                    calculate_electron_moment_derivatives!(
                        moments,
                        (electron_density=this_dens,
                         electron_upar=this_upar,
                         electron_ppar=this_ppar),
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
                electron_adaptive_timestep_update!(scratch, t_params.t[], t_params,
                                                   moments, phi, z_advect, vpa_advect,
                                                   composition, r, z, vperp, vpa,
                                                   vperp_spectral, vpa_spectral,
                                                   external_source_settings,
                                                   num_diss_params;
                                                   evolve_ppar=evolve_ppar)
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
        @serial_region begin
            t_params.t[] += t_params.previous_dt[]
        end
        _block_synchronize()

        residual = -1.0
        if t_params.previous_dt[] > 0.0
            # Calculate residuals to decide if iteration is converged.
            # Might want an option to calculate the residual only after a certain number
            # of iterations (especially during initialization when there are likely to be
            # a large number of iterations required) to avoid the expense, and especially
            # the global MPI.Bcast()?
            begin_r_z_vperp_vpa_region()
            residual = steady_state_residuals(scratch[t_params.n_rk_stages+1].pdf_electron,
                                              scratch[1].pdf_electron, t_params.previous_dt[];
                                              use_mpi=true, only_max_abs=true)
            if global_rank[] == 0
                residual = first(values(residual))[1]
            end
            if evolve_ppar
                ppar_residual =
                    steady_state_residuals(scratch[t_params.n_rk_stages+1].electron_ppar,
                                           scratch[1].electron_ppar, t_params.previous_dt[];
                                           use_mpi=true, only_max_abs=true)
                if global_rank[] == 0
                    ppar_residual = first(values(ppar_residual))[1]
                    residual = max(residual, ppar_residual)
                end
            end
            if global_rank[] == 0
                if residual_tolerance === nothing
                    residual_tolerance = t_params.converged_residual_value
                end
                electron_pdf_converged = abs(residual) < residual_tolerance
            end
            electron_pdf_converged = MPI.Bcast(electron_pdf_converged, 0, comm_world)
        end

        if text_output
            if (mod(t_params.step_counter[] - initial_step_counter, t_params.nwrite_moments)==1)
                begin_serial_region()
                @serial_region begin
                    @loop_vpa ivpa begin
                        println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", new_pdf[ivpa,1,end,1], " iteration: ", t_params.step_counter[] - initial_step_counter, " flag: ", 1)
                    end
                    println(io_pdf_stages,"")
                end
            end
        end

        if (mod(t_params.step_counter[] - initial_step_counter,100) == 0)
            begin_serial_region()
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

            begin_serial_region()
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
                        println(io_ppar, "z: ", z.grid[iz], " ppar: ", scratch[t_params.n_rk_stages+1].electron_ppar[iz,1], " dppar_dz: ", moments.electron.dppar_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter)
                        println(io_vth, "z: ", z.grid[iz], " vthe: ", moments.electron.vth[iz,1], " dvth_dz: ", moments.electron.dvth_dz[iz,1], " time: ", t_params.t[], " iteration: ", t_params.step_counter[] - initial_step_counter, " dens: ", dens[iz,1])
                    end
                    println(io_upar,"")
                    println(io_qpar,"")
                    println(io_ppar,"")
                    println(io_vth,"")
                end
            end
            t_params.moments_output_counter[] += 1
            @serial_region begin
                if io_electron !== nothing
                    t_params.write_moments_output[] = false
                    write_electron_state(scratch, moments, t_params, io_electron,
                                         t_params.moments_output_counter[], r, z, vperp,
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
    begin_r_z_vperp_vpa_region()
    final_scratch_pdf = scratch[t_params.n_rk_stages+1].pdf_electron
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        pdf[ivpa,ivperp,iz,ir] = final_scratch_pdf[ivpa,ivperp,iz,ir]
    end
    if evolve_ppar
        # Update `moments.electron.ppar` with the final electron pressure
        begin_r_z_region()
        scratch_ppar = scratch[t_params.n_rk_stages+1].electron_ppar
        moments_ppar = moments.electron.ppar
        @loop_r_z ir iz begin
            moments_ppar[iz,ir] = scratch_ppar[iz,ir]
        end
    end
    begin_serial_region()
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
            close(io_ppar)
            close(io_vth)
            close(io_pdf)
            close(io_pdf_stages)
        end
        if !electron_pdf_converged || do_debug_io
            if io_electron !== nothing && io_electron !== true
                t_params.moments_output_counter[] += 1
                write_electron_state(scratch, moments, t_params, io_electron,
                                     t_params.moments_output_counter[], r, z, vperp, vpa)
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
    implicit_electron_advance!()

Do an implicit solve which finds: the steady-state electron shape function \$g_e\$; the
backward-Euler advanced electron pressure which is updated using \$g_e\$ at the new
time-level.

Implicit electron solve includes r-dimension. For 1D runs this makes no difference. In 2D
it might or might not be necessary. If the r-dimension is not needed in the implicit
solve, we would need to work on the parallelisation. The simplest option would be a
non-parallelised outer loop over r, with each nonlinear solve being parallelised over
{z,vperp,vpa}. More efficient might be to add an equivalent to the 'anyv' parallelisation
used for the collision operator (e.g. 'anyzv'?) to allow the outer r-loop to be
parallelised.
"""
function implicit_electron_advance!(fvec_out, fvec_in, pdf, scratch_electron, moments,
                                    fields, collisions, composition,
                                    external_source_settings, num_diss_params, r, z,
                                    vperp, vpa, r_spectral, z_spectral, vperp_spectral,
                                    vpa_spectral, z_advect, vpa_advect, gyroavs,
                                    scratch_dummy, dt, nl_solver_params)

    electron_ppar_out = fvec_out.electron_ppar
    # Store the solved-for pdf in n_rk_stages+1, because this was the version that gets
    # written to output for the explicit-electron-timestepping version.
    pdf_electron_out = scratch_electron.pdf_electron

    # Do a forward-Euler step for electron_ppar to get the initial guess.
    # No equivalent for f_electron, as f_electron obeys a steady-state equation.
    calculate_electron_moment_derivatives!(moments, fvec_in, scratch_dummy, z, z_spectral,
                                           num_diss_params.electron.moment_dissipation_coefficient,
                                           composition.electron_physics)
    electron_energy_equation!(electron_ppar_out, fvec_in.electron_ppar,
                              fvec_in.density, fvec_in.electron_upar, fvec_in.upar,
                              fvec_in.ppar, fvec_in.density_neutral,
                              fvec_in.uz_neutral, fvec_in.pz_neutral,
                              moments.electron, collisions, dt, composition,
                              external_source_settings.electron, num_diss_params, z)

    function residual_func!(residual, new_variables)
        electron_ppar_residual, f_electron_residual = residual
        electron_ppar_new, f_electron_new = new_variables

        new_scratch = scratch_pdf(fvec_in.pdf, fvec_in.density, fvec_in.upar, fvec_in.ppar,
                                  fvec_in.pperp, fvec_in.temp_z_s,
                                  fvec_in.electron_density, fvec_in.electron_upar,
                                  electron_ppar_new, fvec_in.electron_pperp,
                                  fvec_in.electron_temp, fvec_in.pdf_neutral,
                                  fvec_in.density_neutral, fvec_in.uz_neutral,
                                  fvec_in.pz_neutral)
        new_scratch_electron = scratch_electron_pdf(f_electron_new, electron_ppar_new)

        apply_electron_bc_and_constraints!(new_scratch_electron, fields.phi, moments, z,
                                           vperp, vpa, vperp_spectral, vpa_spectral,
                                           vpa_advect, num_diss_params, composition,
                                           nl_solver_params.rtol)

        # Only the first entry in the `electron_pdf_substruct` will be used, so does not
        # matter what we put in the second and third except that they have the right type.
        new_pdf = (electron=electron_pdf_substruct(f_electron_new, f_electron_new,
                                                   f_electron_new,),)
        # Calculate heat flux and derivatives using new_variables
        calculate_electron_moments!(new_scratch, new_pdf, moments, composition,
                                    collisions, r, z, vpa)
        calculate_electron_moment_derivatives!(moments, new_scratch, scratch_dummy, z,
                                               z_spectral,
                                               num_diss_params.electron.moment_dissipation_coefficient,
                                               composition.electron_physics)

        electron_energy_residual!(electron_ppar_residual, electron_ppar_new, fvec_in,
                                  moments, collisions, composition,
                                  external_source_settings, num_diss_params, z, dt)

        # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
        # electron_pdf member of the first argument, so if we set the electron_pdf member
        # of the first argument to zero, and pass dt=1, then it will evaluate the time
        # derivative, which is the residual for a steady-state solution.
        begin_r_z_vperp_vpa_region()
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            f_electron_residual[ivpa,ivperp,iz,ir] = 0.0
        end
        residual_scratch_electron = scratch_electron_pdf(f_electron_residual,
                                                         electron_ppar_residual)
        new_scratch_electron = scratch_electron_pdf(f_electron_new, electron_ppar_new)
        electron_kinetic_equation_euler_update!(residual_scratch_electron,
                                                new_scratch_electron, moments, z, vperp,
                                                vpa, z_spectral, vpa_spectral, z_advect,
                                                vpa_advect, scratch_dummy, collisions,
                                                composition, external_source_settings,
                                                num_diss_params, 1.0)

        # Set residual to zero where pdf_electron is determined by boundary conditions.
        if vpa.n > 1
            begin_r_z_vperp_region()
            @loop_r_z_vperp ir iz ivperp begin
                @views enforce_v_boundary_condition_local!(f_electron_residual[:,ivperp,iz,ir], vpa.bc,
                                                           vpa_advect[1].speed[:,ivperp,iz,ir],
                                                           num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                           vpa, vpa_spectral)
            end
        end
        if vperp.n > 1
            begin_r_z_vpa_region()
            enforce_vperp_boundary_condition!(f_electron_residual, vperp.bc, vperp, vperp_spectral,
                                              vperp_adv, vperp_diffusion)
        end
        if z.bc == "wall" && (z.irank == 0 || z.irank == z.nrank - 1)
            # Wall boundary conditions. Note that as density, upar, ppar do not
            # change in this implicit step, f_new, f_old, and residual should all
            # be zero at exactly the same set of grid points, so it is reasonable
            # to zero-out `residual` to impose the boundary condition. We impose
            # this after subtracting f_old in case rounding errors, etc. mean that
            # at some point f_old had a different boundary condition cut-off
            # index.
            begin_r_vperp_vpa_region()
            v_unnorm = vpa.scratch
            zero = 1.0e-14
            if z.irank == 0
                iz = 1
                @loop_r ir begin
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, moments.electron.vth[iz,ir],
                                                fvec_in.electron_upar[iz,ir], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm > -zero
                            f_electron_residual[ivpa,ivperp,iz,ir] .= 0.0
                        end
                    end
                end
            end
            if z.irank == z.nrank - 1
                iz = z.n
                @loop_r ir begin
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, moments.electron.vth[iz,ir],
                                                fvec_in.electron_upar[iz,ir], true, true)
                    @loop_vperp_vpa ivpa ivperp begin
                        if v_unnorm < zero
                            f_electron_residual[ivpa,ivperp,iz,ir] .= 0.0
                        end
                    end
                end
            end
        end
        begin_r_z_region()
        @loop_r_z ir iz begin
            @views moment_constraints_on_residual!(f_electron_residual[:,:,iz,ir], f_electron_new[:,:,iz,ir],
                                                   (evolve_density=true, evolve_upar=true, evolve_ppar=true),
                                                   vpa)
        end
        return nothing
    end

    residual = (scratch_dummy.implicit_buffer_zr_1,
                scratch_dummy.implicit_buffer_vpavperpzr_1)
    delta_x = (scratch_dummy.implicit_buffer_zr_2,
               scratch_dummy.implicit_buffer_vpavperpzr_2)
    rhs_delta = (scratch_dummy.implicit_buffer_zr_3,
                 scratch_dummy.implicit_buffer_vpavperpzr_3)
    v = (scratch_dummy.implicit_buffer_zr_4,
         scratch_dummy.implicit_buffer_vpavperpzr_4)
    w = (scratch_dummy.implicit_buffer_zr_5,
         scratch_dummy.implicit_buffer_vpavperpzr_5)

    newton_success = newton_solve!((electron_ppar_out, pdf_electron_out), residual_func!,
                                   residual, delta_x, rhs_delta, v, w, nl_solver_params;
                                   left_preconditioner=nothing,
                                   right_preconditioner=nothing,
                                   coords=(r=r, z=z, vperp=vperp, vpa=vpa))

    # Fill pdf.electron.norm
    non_scratch_pdf = pdf.electron.norm
    begin_r_z_vperp_vpa_region()
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
        composition.me_over_mi, collisions.nu_ei, composition.electron_physics)

    # Solve for EM fields now that electrons are updated.
    update_phi!(fields, fvec_out, vperp, z, r, composition, collisions, moments,
                z_spectral, r_spectral, scratch_dummy, gyroavs)

    if !newton_success
        success = "kinetic-electrons"
    else
        success = ""
    end
    return success
end

function speedup_hack!(fvec_out, fvec_in, z_speedup_fac, z, vpa; evolve_ppar=false)
    # Divide by wpa to relax CFL condition at large wpa - only looking for steady
    # state here, so does not matter that this makes time evolution incorrect.
    # Also increase the effective timestep for z-values far from the sheath boundary -
    # these have a less-limited timestep so letting them evolve faster speeds up
    # convergence to the steady state.

    # Actually modify so that large wpa does go faster (to allow some phase mixing - maybe
    # this makes things more stable?), but not by so much.
    #vpa_fudge_factor = 1.0
    #vpa_fudge_factor = 0.8
    vpa_fudge_factor = 0.0

    Lz = z.L

    if evolve_ppar
        begin_r_z_region()
        ppar_out = fvec_out.electron_ppar
        ppar_in = fvec_in.electron_ppar
        @loop_r_z ir iz begin
            zval = z.grid[iz]
            znorm = 2.0*zval/Lz
            ppar_out[iz,ir] = ppar_in[iz,ir] +
                (1.0 + z_speedup_fac*(1.0 - znorm^2)) *
                (ppar_out[iz,ir] - ppar_in[iz,ir])
        end
    end

    begin_r_z_vperp_vpa_region()
    pdf_out = fvec_out.pdf_electron
    pdf_in = fvec_in.pdf_electron
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        zval = z.grid[iz]
        znorm = 2.0*zval/Lz
        pdf_out[ivpa,ivperp,iz,ir] = pdf_in[ivpa,ivperp,iz,ir] +
            (1.0 + z_speedup_fac*(1.0 - znorm^2)) /
            sqrt(1.0 + vpa_fudge_factor * vpa.grid[ivpa]^2) *
            (pdf_out[ivpa,ivperp,iz,ir] - pdf_in[ivpa,ivperp,iz,ir])
    end
    return nothing
end

function apply_electron_bc_and_constraints!(this_scratch, phi, moments, z, vperp, vpa,
                                            vperp_spectral, vpa_spectral, vpa_advect,
                                            num_diss_params, composition, rtol)
    latest_pdf = this_scratch.pdf_electron

    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        latest_pdf[ivpa,ivperp,iz,ir] = max(latest_pdf[ivpa,ivperp,iz,ir], 0.0)
    end

    # enforce the boundary condition(s) on the electron pdf
    enforce_boundary_condition_on_electron_pdf!(latest_pdf, phi, moments.electron.vth,
                                                moments.electron.upar, z, vperp, vpa,
                                                vperp_spectral, vpa_spectral, vpa_advect,
                                                moments,
                                                num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                composition.me_over_mi, 0.1 * rtol)

    begin_r_z_region()
    A = moments.electron.constraints_A_coefficient
    B = moments.electron.constraints_B_coefficient
    C = moments.electron.constraints_C_coefficient
    skip_first = z.irank == 0 && z.bc != "periodic"
    skip_last = z.irank == z.nrank - 1 && z.bc != "periodic"
    @loop_r_z ir iz begin
        if (iz == 1 && skip_first) || (iz == z.n && skip_last)
            continue
        end
        (A[iz,ir], B[iz,ir], C[iz,ir]) =
            @views hard_force_moment_constraints!(latest_pdf[:,:,iz,ir],
                                                  (evolve_density=true,
                                                   evolve_upar=true,
                                                   evolve_ppar=true), vpa)
    end
end

function enforce_boundary_condition_on_electron_pdf!(pdf, phi, vthe, upar, z, vperp, vpa,
                                                     vperp_spectral, vpa_spectral,
                                                     vpa_adv, moments, vpa_diffusion,
                                                     me_over_mi, newton_tol)
    # Enforce velocity-space boundary conditions
    if vpa.n > 1
        begin_r_z_vperp_region()
        @loop_r_z_vperp ir iz ivperp begin
            # enforce the vpa BC
            # use that adv.speed independent of vpa
            @views enforce_v_boundary_condition_local!(pdf[:,ivperp,iz,ir], vpa.bc,
                                                       vpa_adv[1].speed[:,ivperp,iz,ir],
                                                       vpa_diffusion, vpa, vpa_spectral)
        end
    end
    if vperp.n > 1
        begin_r_z_vpa_region()
        @views enforce_vperp_boundary_condition!(pdf, vperp.bc, vperp, vperp_spectral)
    end

    if z.bc == "periodic"
        # Nothing more to do for z-periodic boundary conditions
        return nothing
    end

    # first enforce the boundary condition at z_min.
    # this involves forcing the pdf to be zero for electrons travelling faster than the max speed
    # they could attain by accelerating in the electric field between the wall and the simulation boundary;
    # for electrons with positive velocities less than this critical value, they must have the same
    # pdf as electrons with negative velocities of the same magnitude.
    # the electrostatic potential at the boundary, which determines the critical speed, is unknown a priori;
    # use the constraint that the first moment of the normalised pdf be zero to choose the potential.

    begin_r_region()

    newton_max_its = 100
    reversed_pdf = vpa.scratch

    function get_residual_and_coefficients_for_bc(a1, a1prime, a2, a2prime, b1, b1prime,
                                                  c1, c1prime, c2, c2prime, d1, d1prime,
                                                  e1, e1prime, e2, e2prime, u_over_vt)
        alpha = a1 + 2.0 * a2
        alphaprime = a1prime + 2.0 * a2prime
        beta = c1 + 2.0 * c2
        betaprime = c1prime + 2.0 * c2prime
        gamma = u_over_vt^2 * alpha - 2.0 * u_over_vt * b1 + beta
        gammaprime = u_over_vt^2 * alphaprime - 2.0 * u_over_vt * b1prime + betaprime
        delta = u_over_vt^2 * beta - 2.0 * u_over_vt * d1 + e1 + 2.0 * e2
        deltaprime = u_over_vt^2 * betaprime - 2.0 * u_over_vt * d1prime + e1prime + 2.0 * e2prime

        A = (0.5 * beta - delta) / (beta * gamma - alpha * delta)
        Aprime = (0.5 * betaprime - deltaprime
                  - (0.5 * beta - delta) * (gamma * betaprime + beta * gammaprime - delta * alphaprime - alpha * deltaprime)
                    / (beta * gamma - alpha * delta)
                 ) / (beta * gamma - alpha * delta)
        C = (1.0 - alpha * A) / beta
        Cprime = -(A * alphaprime + alpha * Aprime) / beta - (1.0 - alpha * A) * betaprime / beta^2

        epsilon = A * b1 + C * d1 - u_over_vt
        epsilonprime = b1 * Aprime + A * b1prime + d1 * Cprime + C * d1prime

        return epsilon, epsilonprime, A, C
    end

    if z.irank == 0
        if z.bc != "wall"
            error("Options other than wall or z-periodic bc not implemented yet for electrons")
        end
        @loop_r ir begin
            # Impose sheath-edge boundary condition, while also imposing moment
            # constraints and determining the cut-off velocity (and therefore the sheath
            # potential).

            # Delete the upar contribution here if ignoring the 'upar shift'
            vpa_unnorm = @. vpa.scratch2 = vthe[1,ir] * vpa.grid + upar[1,ir]

            u_over_vt = upar[1,ir] / vthe[1,ir]

            # Initial guess for cut-off velocity is result from previous RK stage (which
            # might be the previous timestep if this is the first stage). Recalculate this
            # value from phi.
            vcut = sqrt(phi[1,ir] / me_over_mi)

            # -vcut is between minus_vcut_ind-1 and minus_vcut_ind
            minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)
            if minus_vcut_ind < 2
                error("In lower-z electron bc, failed to find vpa=-vcut point, minus_vcut_ind=$minus_vcut_ind")
            end
            if minus_vcut_ind > vpa.n
                error("In lower-z electron bc, failed to find vpa=-vcut point, minus_vcut_ind=$minus_vcut_ind")
            end

            # sigma is the location we use for w_∥(v_∥=0) - set to 0 to ignore the 'upar
            # shift'
            sigma = -u_over_vt

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
            sigma_fraction = (sigma - vpa_unnorm[sigma_ind-1]) / (vpa_unnorm[sigma_ind] - vpa_unnorm[sigma_ind-1])

            # Per-grid-point contributions to moment integrals
            # Note that we need to include the normalisation factor of 1/sqrt(pi) that
            # would be factored in by integrate_over_vspace(). This will need to
            # change/adapt when we support 2V as well as 1V.
            density_integral_pieces = @views @. vpa.scratch3 = pdf[:,1,1,ir] * vpa.wgts / sqrt(pi)
            flow_integral_pieces = @views @. vpa.scratch4 = density_integral_pieces * vpa_unnorm / vthe[1,ir]
            energy_integral_pieces = @views @. vpa.scratch5 = flow_integral_pieces * vpa_unnorm / vthe[1,ir]
            cubic_integral_pieces = @views @. vpa.scratch6 = energy_integral_pieces * vpa_unnorm / vthe[1,ir]
            quartic_integral_pieces = @views @. vpa.scratch7 = cubic_integral_pieces * vpa_unnorm / vthe[1,ir]

            function get_integrals_and_derivatives(vcut, minus_vcut_ind)
                # vcut_fraction is the fraction of the distance between minus_vcut_ind-1 and
                # minus_vcut_ind where -vcut is.
                vcut_fraction = (-vcut - vpa_unnorm[minus_vcut_ind-1]) / (vpa_unnorm[minus_vcut_ind] - vpa_unnorm[minus_vcut_ind-1])

                function get_for_one_moment(integral_pieces, skip_part2=false)
                    # Integral contribution from the cell containing vcut
                    integral_vcut_cell = (0.5 * integral_pieces[minus_vcut_ind-1] + 0.5 * integral_pieces[minus_vcut_ind])

                    part1 = sum(integral_pieces[1:minus_vcut_ind-2])
                    part1 += 0.5 * integral_pieces[minus_vcut_ind-1] + vcut_fraction * integral_vcut_cell
                    # part1prime is d(part1)/d(vcut)
                    part1prime = -integral_vcut_cell / (vpa_unnorm[minus_vcut_ind] - vpa_unnorm[minus_vcut_ind-1])

                    if skip_part2
                        part2 = nothing
                        part2prime = nothing
                    else
                        # Integral contribution from the cell containing sigma
                        integral_sigma_cell = (0.5 * integral_pieces[sigma_ind-1] + 0.5 * integral_pieces[sigma_ind])

                        part2 = sum(integral_pieces[minus_vcut_ind+1:sigma_ind-2])
                        part2 += (1.0 - vcut_fraction) * integral_vcut_cell + 0.5 * integral_pieces[minus_vcut_ind] + 0.5 * integral_pieces[sigma_ind-1] + sigma_fraction * integral_sigma_cell
                        # part2prime is d(part2)/d(vcut)
                        part2prime = -part1prime
                    end

                    return part1, part1prime, part2, part2prime
                end
                a1, a1prime, a2, a2prime = get_for_one_moment(density_integral_pieces)
                b1, b1prime, _, _ = get_for_one_moment(flow_integral_pieces, true)
                c1, c1prime, c2, c2prime = get_for_one_moment(energy_integral_pieces)
                d1, d1prime, _, _ = get_for_one_moment(cubic_integral_pieces, true)
                e1, e1prime, e2, e2prime = get_for_one_moment(quartic_integral_pieces)

                return get_residual_and_coefficients_for_bc(a1, a1prime, a2, a2prime, b1,
                                                            b1prime, c1, c1prime, c2,
                                                            c2prime, d1, d1prime, e1,
                                                            e1prime, e2, e2prime,
                                                            u_over_vt)
            end

            counter = 1
            A = 1.0
            C = 0.0
            # Always do at least one update of vcut
            epsilon, epsilonprime, A, C = get_integrals_and_derivatives(vcut, minus_vcut_ind)
            while true
                # Newton iteration update. Note that primes denote derivatives with
                # respect to vcut
                delta_v = - epsilon / epsilonprime

                # Prevent the step size from getting too big, to make Newton iteration
                # more robust.
                delta_v = min(delta_v, 0.1 * vthe[1,ir])
                delta_v = max(delta_v, -0.1 * vthe[1,ir])

                vcut = vcut + delta_v
                minus_vcut_ind = searchsortedfirst(vpa_unnorm, -vcut)

                epsilon, epsilonprime, A, C = get_integrals_and_derivatives(vcut, minus_vcut_ind)

                if abs(epsilon) < newton_tol * abs(u_over_vt)
                    break
                end

                if counter ≥ newton_max_its
                    error("Newton iteration for electron lower-z boundary failed to "
                          * "converge after $counter iterations")
                end
                counter += 1
            end

            # Adjust pdf so that after reflecting and cutting off tail, it will obey the
            # constraints.
            @. pdf[:,1,1,ir] *= A + C * vpa_unnorm^2 / vthe[1,ir]^2

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

            # interpolate the pdf onto this grid
            #@views interpolate_to_grid_1d!(interpolated_pdf, wpa_values, pdf[:,1,1,ir], vpa, vpa_spectral)
            @views interpolate_to_grid_1d!(reversed_pdf, reversed_wpa_of_minus_vpa, pdf[:,1,1,ir], vpa, vpa_spectral) # Could make this more efficient by only interpolating to the points needed below, by taking an appropriate view of wpa_of_minus_vpa. Also, in the element containing vpa=0, this interpolation depends on the values that will be replaced by the reflected, interpolated values, which is not ideal (maybe this element should be treated specially first?).
            reverse!(reversed_pdf)

            ivpa_max = searchsortedlast(vpa_unnorm, vcut)
            reversed_pdf[ivpa_max+1:end] .= 0.0
            # vcut_fraction is the fraction of the distance between ivpa_max and
            # ivpa_max+1 where vcut is.
            vcut_fraction = (vcut - vpa_unnorm[ivpa_max]) / (vpa_unnorm[ivpa_max+1] - vpa_unnorm[ivpa_max])
            reversed_pdf[ivpa_max] *= vcut_fraction

            # update the electrostatic potential at the boundary to be the value corresponding to the updated cutoff velocity
            phi[1,ir] = me_over_mi * vcut^2
            pdf[sigma_ind:end,1,1,ir] .= reversed_pdf[sigma_ind:end]

            moments.electron.constraints_A_coefficient[1,ir] = A
            moments.electron.constraints_B_coefficient[1,ir] = 0.0
            moments.electron.constraints_C_coefficient[1,ir] = C
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
        @loop_r ir begin
            # Impose sheath-edge boundary condition, while also imposing moment
            # constraints and determining the cut-off velocity (and therefore the sheath
            # potential).

            # Delete the upar contribution here if ignoring the 'upar shift'
            vpa_unnorm = @. vpa.scratch2 = vthe[end,ir] * vpa.grid + upar[end,ir]

            u_over_vt = upar[end,ir] / vthe[end,ir]

            # Initial guess for cut-off velocity is result from previous RK stage (which
            # might be the previous timestep if this is the first stage). Recalculate this
            # value from phi.
            vcut = sqrt(phi[end,ir] / me_over_mi)

            # vcut is between plus_vcut_ind and plus_vcut_ind+1
            plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)
            if plus_vcut_ind < 1
                error("In upper-z electron bc, failed to find vpa=vcut point, plus_vcut_ind=$plus_vcut_ind")
            end
            if plus_vcut_ind > vpa.n - 1
                error("In upper-z electron bc, failed to find vpa=vcut point, plus_vcut_ind=$plus_vcut_ind")
            end

            # sigma is the location we use for w_∥(v_∥=0) - set to 0 to ignore the 'upar
            # shift'
            sigma = -u_over_vt

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
            sigma_fraction = (sigma - vpa_unnorm[sigma_ind+1]) / (vpa_unnorm[sigma_ind] - vpa_unnorm[sigma_ind+1])

            # Per-grid-point contributions to moment integrals
            # Note that we need to include the normalisation factor of 1/sqrt(pi) that
            # would be factored in by integrate_over_vspace(). This will need to
            # change/adapt when we support 2V as well as 1V.
            density_integral_pieces = @views @. vpa.scratch3 = pdf[:,1,end,ir] * vpa.wgts / sqrt(pi)
            flow_integral_pieces = @views @. vpa.scratch4 = density_integral_pieces * vpa_unnorm / vthe[end,ir]
            energy_integral_pieces = @views @. vpa.scratch5 = flow_integral_pieces * vpa_unnorm / vthe[end,ir]
            cubic_integral_pieces = @views @. vpa.scratch6 = energy_integral_pieces * vpa_unnorm / vthe[end,ir]
            quartic_integral_pieces = @views @. vpa.scratch7 = cubic_integral_pieces * vpa_unnorm / vthe[end,ir]

            function get_integrals_and_derivatives(vcut, plus_vcut_ind)
                # vcut_fraction is the fraction of the distance between plus_vcut_ind and
                # plus_vcut_ind+1 where vcut is.
                vcut_fraction = (vcut - vpa_unnorm[plus_vcut_ind+1]) / (vpa_unnorm[plus_vcut_ind] - vpa_unnorm[plus_vcut_ind+1])

                function get_for_one_moment(integral_pieces, skip_part2=false)
                    # Integral contribution from the cell containing vcut
                    integral_vcut_cell = (0.5 * integral_pieces[plus_vcut_ind] + 0.5 * integral_pieces[plus_vcut_ind+1])

                    part1 = sum(integral_pieces[plus_vcut_ind+2:end])
                    part1 += 0.5 * integral_pieces[plus_vcut_ind+1] + vcut_fraction * integral_vcut_cell
                    # part1prime is d(part1)/d(vcut)
                    part1prime = integral_vcut_cell / (vpa_unnorm[plus_vcut_ind] - vpa_unnorm[plus_vcut_ind+1])

                    if skip_part2
                        part2 = nothing
                        part2prime = nothing
                    else
                        # Integral contribution from the cell containing sigma
                        integral_sigma_cell = (0.5 * integral_pieces[sigma_ind] + 0.5 * integral_pieces[sigma_ind+1])

                        part2 = sum(integral_pieces[sigma_ind+2:plus_vcut_ind-1])
                        part2 += (1.0 - vcut_fraction) * integral_vcut_cell + 0.5 * integral_pieces[plus_vcut_ind] + 0.5 * integral_pieces[sigma_ind+1] + sigma_fraction * integral_sigma_cell
                        # part2prime is d(part2)/d(vcut)
                        part2prime = -part1prime
                    end

                    return part1, part1prime, part2, part2prime
                end
                a1, a1prime, a2, a2prime = get_for_one_moment(density_integral_pieces)
                b1, b1prime, _, _ = get_for_one_moment(flow_integral_pieces, true)
                c1, c1prime, c2, c2prime = get_for_one_moment(energy_integral_pieces)
                d1, d1prime, _, _ = get_for_one_moment(cubic_integral_pieces, true)
                e1, e1prime, e2, e2prime = get_for_one_moment(quartic_integral_pieces)

                return get_residual_and_coefficients_for_bc(a1, a1prime, a2, a2prime, b1,
                                                            b1prime, c1, c1prime, c2,
                                                            c2prime, d1, d1prime, e1,
                                                            e1prime, e2, e2prime,
                                                            u_over_vt)
            end

            counter = 1
            # Always do at least one update of vcut
            epsilon, epsilonprime, A, C = get_integrals_and_derivatives(vcut, plus_vcut_ind)
            while true
                # Newton iteration update. Note that primes denote derivatives with
                # respect to vcut
                delta_v = - epsilon / epsilonprime

                # Prevent the step size from getting too big, to make Newton iteration
                # more robust.
                delta_v = min(delta_v, 0.1 * vthe[1,ir])
                delta_v = max(delta_v, -0.1 * vthe[1,ir])

                vcut = vcut + delta_v
                plus_vcut_ind = searchsortedlast(vpa_unnorm, vcut)

                epsilon, epsilonprime, A, C = get_integrals_and_derivatives(vcut, plus_vcut_ind)

                if abs(epsilon) < newton_tol * abs(u_over_vt)
                    break
                end

                if counter ≥ newton_max_its
                    error("Newton iteration for electron upper-z boundary failed to "
                          * "converge after $counter iterations")
                end
                counter += 1
            end

            # Adjust pdf so that after reflecting and cutting off tail, it will obey the
            # constraints.
            @. pdf[:,1,end,ir] *= A + C * vpa_unnorm^2 / vthe[end,ir]^2

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

            # interpolate the pdf onto this grid
            #@views interpolate_to_grid_1d!(interpolated_pdf, wpa_values, pdf[:,1,1,ir], vpa, vpa_spectral)
            @views interpolate_to_grid_1d!(reversed_pdf, reversed_wpa_of_minus_vpa, pdf[:,1,end,ir], vpa, vpa_spectral) # Could make this more efficient by only interpolating to the points needed below, by taking an appropriate view of wpa_of_minus_vpa. Also, in the element containing vpa=0, this interpolation depends on the values that will be replaced by the reflected, interpolated values, which is not ideal (maybe this element should be treated specially first?).
            reverse!(reversed_pdf)

            ivpa_min = searchsortedfirst(vpa_unnorm, -vcut)
            reversed_pdf[1:ivpa_min-1] .= 0.0
            # vcut_fraction is the fraction of the distance between ivpa_min and
            # ivpa_min-1 where -vcut is.
            vcut_fraction = (-vcut - vpa_unnorm[ivpa_min]) / (vpa_unnorm[ivpa_min-1] - vpa_unnorm[ivpa_min])
            reversed_pdf[ivpa_min] *= vcut_fraction

            # update the electrostatic potential at the boundary to be the value corresponding to the updated cutoff velocity
            phi[end,ir] = me_over_mi * vcut^2
            pdf[1:sigma_ind,1,end,ir] .= reversed_pdf[1:sigma_ind]

            moments.electron.constraints_A_coefficient[end,ir] = A
            moments.electron.constraints_B_coefficient[end,ir] = 0.0
            moments.electron.constraints_C_coefficient[end,ir] = C
        end
    end

    return nothing
end

"""
    electron_adaptive_timestep_update!(scratch, t, t_params, moments, phi, z_advect,
                                       vpa_advect, composition, r, z, vperp, vpa,
                                       vperp_spectral, vpa_spectral,
                                       external_source_settings, num_diss_params;
                                       evolve_ppar=false)

Check the error estimate for the embedded RK method and adjust the timestep if
appropriate.
"""
function electron_adaptive_timestep_update!(scratch, t, t_params, moments, phi, z_advect,
                                            vpa_advect, composition, r, z, vperp, vpa,
                                            vperp_spectral, vpa_spectral,
                                            external_source_settings, num_diss_params;
                                            evolve_ppar=false)
    #error_norm_method = "Linf"
    error_norm_method = "L2"

    error_coeffs = t_params.rk_coefs[:,end]
    if t_params.n_rk_stages < 3
        # This should never happen as an adaptive RK scheme needs at least 2 RHS evals so
        # (with the pre-timestep data) there must be at least 3 entries in `scratch`.
        error("adaptive timestep needs a buffer scratch array")
    end

    CFL_limits = mk_float[]
    error_norm_type = typeof(t_params.error_sum_zero)
    error_norms = error_norm_type[]
    total_points = mk_int[]

    # Read the current dt here, so we only need one _block_synchronize() call for this and
    # the begin_s_r_z_vperp_vpa_region()
    current_dt = t_params.dt[]
    _block_synchronize()

    # Test CFL conditions for advection in electron kinetic equation to give stability
    # limit for timestep
    #
    # z-advection
    # No need to synchronize here, as we just called _block_synchronize()
    begin_r_vperp_vpa_region(; no_synchronize=true)
    update_electron_speed_z!(z_advect[1], moments.electron.upar, moments.electron.vth,
                             vpa.grid)
    z_CFL = get_minimum_CFL_z(z_advect[1].speed, z)
    if block_rank[] == 0
        push!(CFL_limits, t_params.CFL_prefactor * z_CFL)
    else
        push!(CFL_limits, Inf)
    end

    # vpa-advection
    begin_r_z_vperp_region()
    update_electron_speed_vpa!(vpa_advect[1], moments.electron.dens,
                               moments.electron.upar,
                               scratch[t_params.n_rk_stages+1].electron_ppar, moments,
                               vpa.grid, external_source_settings.electron)
    vpa_CFL = get_minimum_CFL_vpa(vpa_advect[1].speed, vpa)
    if block_rank[] == 0
        push!(CFL_limits, t_params.CFL_prefactor * vpa_CFL)
    else
        push!(CFL_limits, Inf)
    end

    # To avoid double counting points when we use distributed-memory MPI, skip the
    # inner/lower point in r and z if this process is not the first block in that
    # dimension.
    skip_r_inner = r.irank != 0
    skip_z_lower = z.irank != 0

    # Calculate error ion distribution functions
    # Note rk_loworder_solution!() stores the calculated error in `scratch[2]`.
    rk_loworder_solution!(scratch, nothing, :pdf_electron, t_params)
    if evolve_ppar
        begin_r_z_region()
        rk_loworder_solution!(scratch, nothing, :electron_ppar, t_params)

        # Make vth consistent with `scratch[2]`, as it is needed for the electron pdf
        # boundary condition.
        update_electron_vth_temperature!(moments, scratch[2].electron_ppar,
                                         moments.electron.dens, composition)
    end
    apply_electron_bc_and_constraints!(scratch[t_params.n_rk_stages+1], phi, moments, z,
                                       vperp, vpa, vperp_spectral, vpa_spectral,
                                       vpa_advect, num_diss_params, composition,
                                       t_params.rtol)
    if evolve_ppar
        # Reset vth in the `moments` struct to the result consistent with full-accuracy RK
        # solution.
        begin_r_z_region()
        update_electron_vth_temperature!(moments,
                                         scratch[t_params.n_rk_stages+1].electron_ppar,
                                         moments.electron.dens, composition)
    end

    pdf_error = local_error_norm(scratch[2].pdf_electron,
                                 scratch[t_params.n_rk_stages+1].pdf_electron,
                                 t_params.rtol, t_params.atol; method=error_norm_method,
                                 skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                 error_sum_zero=t_params.error_sum_zero)
    push!(error_norms, pdf_error)
    push!(total_points, vpa.n_global * vperp.n_global * z.n_global * r.n_global)

    # Calculate error for moments, if necessary
    if evolve_ppar
        begin_r_z_region()
        p_err = local_error_norm(scratch[2].electron_ppar,
                                 scratch[t_params.n_rk_stages+1].electron_ppar,
                                 t_params.rtol, t_params.atol; method=error_norm_method,
                                 skip_r_inner=skip_r_inner, skip_z_lower=skip_z_lower,
                                 error_sum_zero=t_params.error_sum_zero)
        push!(error_norms, p_err)
        push!(total_points, z.n_global * r.n_global)
    end

    adaptive_timestep_update_t_params!(t_params, CFL_limits, error_norms, total_points,
                                       current_dt, error_norm_method, "", 0.0,
                                       composition; electron=true)
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
    ppar = electron parallel pressure
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
function update_electron_pdf_with_shooting_method!(pdf, dens, vthe, ppar, qpar, qpar_updated, phi, 
    ddens_dz, dppar_dz, dqpar_dz, dvth_dz, z, vpa, vpa_spectral, scratch_dummy, composition)

    # it is convenient to create a pointer to a scratch array to store the RHS of the linear 
    # system that includes all terms evaluated at the previous z location in the shooting method
    rhs = scratch_dummy.buffer_vpavperpzr_1
    # initialise the RHS to zero
    rhs .= 0.0
    # get critical velocities beyond which electrons are lost to the wall
    crit_speed_zmin, crit_speed_zmax = get_electron_critical_velocities(phi, vthe, composition.me_over_mi, z)
    # add the contribution to rhs from the term proportional to the pdf (rather than its derivatives)
    add_contribution_from_pdf_term!(rhs, pdf, ppar, vthe, dens, ddens_dz, upar, dvth_dz, dqpar_dz, vpa.grid, z, external_source_settings.electron)
    # add the contribution to rhs from the wpa advection term
    add_contribution_from_wpa_advection!(rhs, pdf, vthe, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
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
    calculate_electron_qpar_from_pdf!(qpar, ppar, vthe, pdf, vpa)
    for iz ∈ 1:z.n
        for ivpa ∈ 1:vpa.n
            println("z: ", z.grid[iz], " vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " qpar: ", qpar[iz, 1], 
                " dppardz: ", dppar_dz[iz,1], " vth: ", vthe[iz,1], " ppar: ", ppar[iz,1], " dqpar_dz: ", dqpar_dz[iz,1],
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
    ppar = electron parallel pressure
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
function update_electron_pdf_with_picard_iteration!(pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz,
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
        add_contribution_from_pdf_term!(rhs, pdf, ppar, vthe, dens, ddens_dz, upar, dvth_dz, dqpar_dz, vpa.grid, z, external_source_settings.electron)
        # add the contribution to rhs from the wpa advection term
        #add_contribution_from_wpa_advection!(rhs, pdf, vthe, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
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
    electron_kinetic_equation_euler_update!(fvec, pdf, moments, z, vperp, vpa,
                                            z_spectral, vpa_spectral, z_advect,
                                            vpa_advect, scratch_dummy, collisions,
                                            num_diss_params, dt; evolve_ppar=false)

Do a forward-Euler update of the electron kinetic equation.

When `evolve_ppar=true` is passed, also updates the electron parallel pressure.
"""
function electron_kinetic_equation_euler_update!(fvec_out, fvec_in, moments, z, vperp,
                                                 vpa, z_spectral, vpa_spectral, z_advect,
                                                 vpa_advect, scratch_dummy, collisions,
                                                 composition, external_source_settings,
                                                 num_diss_params, dt; evolve_ppar=false,
                                                 ion_dt=nothing)
    # add the contribution from the z advection term
    electron_z_advection!(fvec_out.pdf_electron, fvec_in.pdf_electron,
                          moments.electron.upar, moments.electron.vth, z_advect, z,
                          vpa.grid, z_spectral, scratch_dummy, dt)

    # add the contribution from the wpa advection term
    electron_vpa_advection!(fvec_out.pdf_electron, fvec_in.pdf_electron,
                            moments.electron.dens, moments.electron.upar,
                            fvec_in.electron_ppar, moments, vpa_advect, vpa, vpa_spectral,
                            scratch_dummy, dt, external_source_settings.electron)

    # add in the contribution to the residual from the term proportional to the pdf
    add_contribution_from_pdf_term!(fvec_out.pdf_electron, fvec_in.pdf_electron,
                                    fvec_in.electron_ppar, moments.electron.dens,
                                    moments.electron.upar, moments, vpa.grid, z, dt,
                                    external_source_settings.electron)

    # add in numerical dissipation terms
    add_dissipation_term!(fvec_out.pdf_electron, fvec_in.pdf_electron, scratch_dummy,
                          z_spectral, z, vpa, vpa_spectral, num_diss_params, dt)

    if collisions.krook.nuee0 > 0.0 || collisions.krook.nuei0 > 0.0
        # Add a Krook collision operator
        # Set dt=-1 as we update the residual here rather than adding an update to
        # 'fvec_out'.
        electron_krook_collisions!(fvec_out.pdf_electron, fvec_in.pdf_electron,
                                   moments.electron.dens, moments.electron.upar,
                                   moments.ion.upar, moments.electron.vth, collisions,
                                   vperp, vpa, dt)
    end

    if external_source_settings.electron.active
        external_electron_source!(fvec_out.pdf_electron, fvec_in.pdf_electron,
                                  moments.electron.dens, moments.electron.upar, moments,
                                  composition, external_source_settings.electron, vperp,
                                  vpa, dt)
    end

    if evolve_ppar
        electron_energy_equation!(fvec_out.electron_ppar, fvec_in.electron_ppar,
                                  moments.electron.dens, moments.electron.upar,
                                  moments.ion.upar, moments.ion.ppar,
                                  moments.neutral.dens, moments.neutral.uz,
                                  moments.neutral.pz, moments.electron, collisions, dt,
                                  composition, external_source_settings.electron,
                                  num_diss_params, z)

        if ion_dt !== nothing
            # Add source term to turn steady state solution into a backward-Euler update of
            # electron_ppar with the ion timestep `ion_dt`.
            ppar_out = fvec_out.electron_ppar
            ppar_previous_ion_step = moments.electron.ppar
            begin_r_z_region()
            @loop_r_z ir iz begin
                # At this point, ppar_out = ppar_in + dt*RHS(ppar_in). Here we add a
                # source/damping term so that in the steady state of the electron
                # pseudo-timestepping iteration,
                #   RHS(ppar) - (ppar - ppar_previous_ion_step) / ion_dt = 0,
                # resulting in a backward-Euler step (as long as the pseudo-timestepping
                # loop converges).
                ppar_out[iz,ir] += -dt * (ppar_out[iz,ir] - ppar_previous_ion_step[iz,ir]) / ion_dt
            end
        end
    end

    return nothing
end

"""
electron_kinetic_equation_residual! calculates the residual of the (time-independent) electron kinetic equation
INPUTS:
    residual = dummy array to be filled with the residual of the electron kinetic equation
OUTPUT:
    residual = updated residual of the electron kinetic equation
"""
function electron_kinetic_equation_residual!(residual, max_term, single_term, pdf, dens, upar, vth, ppar, upar_ion,
                                             ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
                                             z, vperp, vpa, z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                                             collisions, external_source_settings,
                                             num_diss_params, dt_electron)

    # initialise the residual to zero                                             
    begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        residual[ivpa,ivperp,iz,ir] = 0.0
    end
    # calculate the contribution to the residual from the z advection term
    electron_z_advection!(residual, pdf, upar, vth, z_advect, z, vpa.grid, z_spectral, scratch_dummy, -1.0)
    #dt_max_zadv = simple_z_advection!(residual, pdf, vth, z, vpa.grid, dt_electron)
    #single_term .= residual
    #max_term .= abs.(residual)
    #println("z_adv residual = ", maximum(abs.(single_term)))
    #println("z_advection: ", sum(residual), " dqpar_dz: ", sum(abs.(dqpar_dz)))
    #calculate_contribution_from_z_advection!(residual, pdf, vth, z, vpa.grid, z_spectral, scratch_dummy)
    # add in the contribution to the residual from the wpa advection term
    electron_vpa_advection!(residual, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, 
                            vpa_advect, vpa, vpa_spectral, scratch_dummy, -1.0,
                            external_source_settings.electron)
    #dt_max_vadv = simple_vpa_advection!(residual, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, vpa, dt_electron)
    #@. single_term = residual - single_term
    #max_term .= max.(max_term, abs.(single_term))
    #@. single_term = residual
    #println("v_adv residual = ", maximum(abs.(single_term)))
    #add_contribution_from_wpa_advection!(residual, pdf, vth, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
    # add in the contribution to the residual from the term proportional to the pdf
    add_contribution_from_pdf_term!(residual, pdf, ppar, dens, moments, vpa.grid, z, -1.0,
                                    external_source_settings.electron)
    #@. single_term = residual - single_term
    #max_term .= max.(max_term, abs.(single_term))
    #@. single_term = residual
    #println("pdf_term residual = ", maximum(abs.(single_term)))
    # @loop_vpa ivpa begin
    #     @loop_z iz begin
    #         println("LHS: ", residual[ivpa,1,iz,1], " vpa: ", vpa.grid[ivpa], " z: ", z.grid[iz], " dvth_dz: ", dvth_dz[iz,1], " type: ", 1) 
    #     end
    #     println("")
    # end
    # println("")
    # add in numerical dissipation terms
    add_dissipation_term!(residual, pdf, scratch_dummy, z_spectral, z, vpa, vpa_spectral,
                          num_diss_params, -1.0)
    #@. single_term = residual - single_term
    #println("dissipation residual = ", maximum(abs.(single_term)))
    #max_term .= max.(max_term, abs.(single_term))
    # add in particle and heat source term(s)
    #@. single_term = residual
    #add_source_term!(residual, vpa.grid, z.grid, dvth_dz)
    #@. single_term = residual - single_term
    #max_term .= max.(max_term, abs.(single_term))
    #stop()
    # @loop_vpa ivpa begin
    #     @loop_z iz begin
    #         println("total_residual: ", residual[ivpa,1,iz,1], " vpa: ", vpa.grid[ivpa], " z: ", z.grid[iz], " dvth_dz: ", dvth_dz[iz,1], " type: ", 2) 
    #     end
    #     println("")
    # end
    # stop()
    #dt_max = min(dt_max_zadv, dt_max_vadv)

    if collisions.krook_collision_frequency_prefactor_ee > 0.0
        # Add a Krook collision operator
        # Set dt=-1 as we update the residual here rather than adding an update to
        # 'fvec_out'.
        electron_krook_collisions!(residual, pdf, dens, upar, upar_ion, vth,
                                   collisions, vperp, vpa, -1.0)
    end

    dt_max = dt_electron
    #println("dt_max: ", dt_max, " dt_max_zadv: ", dt_max_zadv, " dt_max_vadv: ", dt_max_vadv)
    return dt_max
end

function simple_z_advection!(advection_term, pdf, vth, z, vpa, dt_max_in)
    dt_max = dt_max_in
    # take the z derivative of the input pdf
    begin_r_vperp_vpa_region()
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


function simple_vpa_advection!(advection_term, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, vpa, dt_max_in)
    dt_max = dt_max_in
    # take the vpa derivative in the input pdf
    begin_r_z_vperp_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        speed = ((vth[iz,ir] * dppar_dz[iz,ir] + vpa.grid[ivpa] * dqpar_dz[iz,ir]) 
                / (2 * ppar[iz,ir]) - vpa.grid[ivpa]^2 * dvth_dz[iz,ir])
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
    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
    #    source_term[ivpa,ivperp,iz,ir] -= 40*exp(-vpa[ivpa]^2)*exp(-z[iz]^2)
    #    source_term[ivpa,ivperp,iz,ir] -= vpa[ivpa]*exp(-vpa[ivpa]^2)*(2*vpa[ivpa]^2-3)*dvth_dz[iz,ir]
    end
    return nothing
end

function add_dissipation_term!(pdf_out, pdf_in, scratch_dummy, z_spectral, z, vpa,
                               vpa_spectral, num_diss_params, dt)
    dummy_zr1 = @view scratch_dummy.dummy_zrs[:,:,1]
    dummy_zr2 = @view scratch_dummy.buffer_vpavperpzr_1[1,1,:,:]
    buffer_r_1 = @view scratch_dummy.buffer_rs_1[:,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[:,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[:,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[:,1]
    # add in numerical dissipation terms
    #@loop_vperp_vpa ivperp ivpa begin
    #    @views derivative_z!(dummy_zr1, pdf_in[ivpa,ivperp,:,:], buffer_r_1, buffer_r_2, buffer_r_3,
    #                         buffer_r_4, z_spectral, z)
    #    @views derivative_z!(dummy_zr2, dummy_zr1, buffer_r_1, buffer_r_2, buffer_r_3,
    #                         buffer_r_4, z_spectral, z)
    #    @. residual[ivpa,ivperp,:,:] -= num_diss_params.electron.z_dissipation_coefficient * dummy_zr2
    #end
    begin_r_z_vperp_region()
    @loop_r_z_vperp ir iz ivperp begin
        #@views derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir], vpa, false)
        #@views derivative!(vpa.scratch2, vpa.scratch, vpa, false)
        #@. residual[:,ivperp,iz,ir] -= num_diss_params.electron.vpa_dissipation_coefficient * vpa.scratch2
        @views second_derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir], vpa, vpa_spectral)
        @. pdf_out[:,ivperp,iz,ir] += dt * num_diss_params.electron.vpa_dissipation_coefficient * vpa.scratch
    end
    #stop()
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
# function update_electron_pdf!(pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
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
#         @views add_contribution_from_wpa_advection!(scratch_dummy.buffer_vpavperpzr_1, pdf, vthe, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
#         # calculate the pre-factor multiplying the modified electron pdf
#         # and store in the dummy array scratch_dummy.buffer_vpavperpzr_2
#         @views calculate_pdf_dot_prefactor!(scratch_dummy.buffer_vpavperpzr_2, ppar, vthe, dens, ddens_dz, dvth_dz, dqpar_dz, vpa.grid)
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
# function update_electron_pdf_bicgstab!(pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
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
#     electron_kinetic_equation_residual!(residual, pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
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
    begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @. z_advection_term[ivpa,ivperp,iz,ir] = z_advection_term[ivpa,ivperp,iz,ir] #* vpa[ivpa] * vthe[:, :]
    end
    return nothing
end

function add_contribution_from_wpa_advection!(residual, pdf, vth, ppar, dppar_dz, dqpar_dz, dvth_dz,
                                                    vpa, vpa_spectral)
    begin_r_z_vperp_region()
    @loop_r_z_vperp ir iz ivperp begin
        # calculate the wpa-derivative of the pdf and store in the scratch array vpa.scratch
        @views derivative!(vpa.scratch, pdf[:, ivperp, iz, ir], vpa, vpa_spectral)
        # contribution from the wpa-advection term is ( ) * d(pdf)/dwpa
        @. residual[:,ivperp,iz,ir] += vpa.scratch * (0.5 * vth[iz,ir] / ppar[iz,ir] * dppar_dz[iz,ir]
                + 0.5 * vpa.grid / ppar[iz,ir] * dqpar_dz[iz,ir] - vpa.grid^2 * dvth_dz[iz,ir])
    end
    return nothing
end

# calculate the pre-factor multiplying the modified electron pdf
function calculate_pdf_dot_prefactor!(pdf_dot_prefactor, ppar, vth, dens, ddens_dz, dvth_dz, dqpar_dz, vpa)
    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @. pdf_dot_prefactor[ivpa,ivperp,iz,ir] = 0.5 * dqpar_dz[iz,ir] / ppar[iz,ir] - vpa[ivpa] * vth[iz,ir] * 
        (ddens_dz[iz,ir] / dens[iz,ir] + dvth_dz[iz,ir] / vth[iz,ir])
    end
    return nothing
end

# add contribution to the residual coming from the term proporational to the pdf
function add_contribution_from_pdf_term!(pdf_out, pdf_in, ppar, dens, upar, moments, vpa,
                                         z, dt, electron_source_settings)
    vth = moments.electron.vth
    ddens_dz = moments.electron.ddens_dz
    dvth_dz = moments.electron.dvth_dz
    dqpar_dz = moments.electron.dqpar_dz
    begin_r_z_vperp_vpa_region()
    @loop_r_z ir iz begin
        this_dqpar_dz = dqpar_dz[iz,ir]
        this_ppar = ppar[iz,ir]
        this_vth = vth[iz,ir]
        this_ddens_dz = ddens_dz[iz,ir]
        this_dens = dens[iz,ir]
        this_dvth_dz = dvth_dz[iz,ir]
        this_vth = vth[iz,ir]
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] +=
                dt * (-0.5 * this_dqpar_dz / this_ppar - vpa[ivpa] * this_vth *
                      (this_ddens_dz / this_dens - this_dvth_dz / this_vth)) *
                pdf_in[ivpa,ivperp,iz,ir]
            #pdf_out[ivpa, ivperp, :, :] -= (-0.5 * dqpar_dz[:, :] / ppar[:, :]) * pdf_in[ivpa, ivperp, :, :]
        end
    end

    if electron_source_settings.active
        source_density_amplitude = moments.electron.external_source_density_amplitude
        source_momentum_amplitude = moments.electron.external_source_momentum_amplitude
        source_pressure_amplitude = moments.electron.external_source_pressure_amplitude
        @loop_r_z ir iz begin
            term = dt * (1.5 * source_density_amplitude[iz,ir] / dens[iz,ir] -
                         (0.5 * source_pressure_amplitude[iz,ir] +
                          source_momentum_amplitude[iz,ir]) / ppar[iz,ir])
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir] -= term * pdf_in[ivpa,ivperp,iz,ir]
            end
        end
    end

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
        crit_speed_zmin = MPI.Bcast(crit_speed_zmin, 0, z.comm)
    end
    crit_speed_zmin = MPI.Bcast(crit_speed_zmin, 0, comm_block[])

    if z.irank == z.nrank - 1 && block_rank[] == 0
        crit_speed_zmax = -sqrt(max(phi[end, 1],0.0) / (me_over_mi * vthe[end, 1]^2))
    else
        crit_speed_zmin = 0.0
    end
    @serial_region begin
        crit_speed_zmax = MPI.Bcast(crit_speed_zmax, z.nrank-1, z.comm)
    end
    crit_speed_zmax = MPI.Bcast(crit_speed_zmax, 0, comm_block[])

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
    begin_r_z_vperp_region()
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
    sum_residual, sum_pdf = MPI.Allreduce([sum_residual, sum_pdf], +, comm_world)

    average_residual = sum_residual / sum_pdf

    electron_pdf_converged = (average_residual < 1e-3)

    return average_residual, electron_pdf_converged
end

function check_electron_pdf_convergence(residual)
    begin_r_z_vperp_region()
    sum_residual = 0.0
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        sum_residual += abs.(residual[ivpa,ivperp,iz,ir])
    end
    sum_residual, sum_length = MPI.Allreduce((sum_residual, length(residual) / block_size[]), +, comm_world)
    average_residual = sum_residual / sum_length
    electron_pdf_converged = (average_residual < 1e-3)
    return average_residual, electron_pdf_converged
end

end
