# Import SparseArrays so that we can use the package after `include()`'ing this script.
using SparseArrays

module MKJacobianUtils

export get_electron_Jacobian_matrix

using moment_kinetics
using moment_kinetics.array_allocation: allocate_shared_float
using moment_kinetics.derivatives: derivative_z!
using moment_kinetics.electron_kinetic_equation: fill_electron_kinetic_equation_Jacobian!,
                                                 add_electron_z_advection_to_Jacobian!,
                                                 add_electron_vpa_advection_to_Jacobian!,
                                                 add_contribution_from_electron_pdf_term_to_Jacobian!,
                                                 add_electron_dissipation_term_to_Jacobian!,
                                                 add_ion_dt_forcing_of_electron_ppar_to_Jacobian!,
                                                 add_total_external_electron_source_to_Jacobian!,
                                                 add_electron_energy_equation_to_Jacobian!,
                                                 add_ion_dt_forcing_of_electron_ppar_to_Jacobian!
using moment_kinetics: setup_moment_kinetics
using moment_kinetics.load_data: open_readonly_output_file, load_input
using moment_kinetics.looping

"""
    get_electron_Jacobian_matrix(run_directory; restart_time_index=1,
                                 restart_index=nothing,
                                 include_z_advection=true,
                                 include_vpa_advection=true,
                                 include_electron_pdf_term=true,
                                 include_dissipation=true,
                                 include_krook=true,
                                 include_external_source=true,
                                 include_constraint_forcing=true,
                                 include_energy_equation=true,
                                 include_ion_dt_forcing=true)

Calculate and return a Jacobian matrix for the kinetic electron solve.

To use this function, first include the script containing it
```julia
include("util/MKJacobianUtils.jl")
```
then you can call the function.

To avoid extreme behaviour due to an arbitrary initial condition, this script expects to
'restart' from an existing simulation. By default it returns the Jacobian matrix that
would be used at the start of the first ion timestep (after the initialisation stage where
electrons are relaxed towards steady state treating the ions as a fixed background).

`run_directory` is the path to the directory where the run to 'restart' from is stored.

`restart_time_index` can be passed an integer value if you want the Jacobian for an output
step other than the initial one. `restart_time_index=-1` would give the final time point
of the simulation in `run_directory`.

If there were multiple restarts of the simulation in `run_directory`, `restart_index` can
be used to select which restart to read from. Reads from the latest one by default.
Numerical values can only be used for restarts before the latest one - e.g. if the run was
not restarted, `restart_index=1` is not valid, only the default `restart_index=nothing`
can be used.

The `include_*` arguments can be used to select which particular terms to include in the
Jacobian. By default all terms are included.
"""
function get_electron_Jacobian_matrix(run_directory; restart_time_index=1,
                                      restart_index=nothing,
                                      include_z_advection=true,
                                      include_vpa_advection=true,
                                      include_electron_pdf_term=true,
                                      include_dissipation=true,
                                      include_krook=true,
                                      include_external_source=true,
                                      include_constraint_forcing=true,
                                      include_energy_equation=true,
                                      include_ion_dt_forcing=true,
                                      include_wall_bc=true)

    # Read the simulation input from the output file, to ensure we are consistent with
    # what was run.
    ##################################################################################
    if isfile(run_directory)
        # run_directory is actually a filename. Assume it is a moment_kinetics output file
        # and infer the directory and the run_name from the filename.

        filename = basename(run_directory)
        run_directory = dirname(run_directory)

        if occursin(".moments.", filename)
            run_name = split(filename, ".moments.")[1]
        elseif occursin(".dfns.", filename)
            run_name = split(filename, ".dfns.")[1]
        elseif occursin(".initial_electron.", filename)
            run_name = split(filename, ".initial_electron.")[1]
        elseif occursin(".electron_debug.", filename)
            run_name = split(filename, ".electron_debug.")[1]
            electron_debug = true
        else
            error("Cannot recognise '$run_directory/$filename' as a moment_kinetics output file")
        end
    elseif isdir(run_directory)
        # Normalise by removing any trailing slash - with a slash basename() would return an
        # empty string
        run_directory = rstrip(run_directory, '/')

        run_name = basename(run_directory)
    else
        error("$run_directory does not exist")
    end
    base_prefix = joinpath(run_directory, run_name)
    if restart_index === nothing
        run_prefix = base_prefix
    elseif restart_index > 0
        run_prefix = base_prefix * "_$restart_index"
    else
        error("Invalid restart_index=$restart_index")
    end
    restart_file = open_readonly_output_file(run_prefix, "dfns", printout=false)
    input = load_input(restart_file)
    close(restart_file)

    # For now, just run in serial. Need to figure out how to bodge this to get partial
    # matrices for domain-decomposed case...
    input["z"]["nelement_local"] = input["z"]["nelement"]

    # Make the output directory a temporary directory to avoid moving around files in the
    # run's directory, which would otherwise happen because we are 'restarting' the run.
    temp_dir = tempname()
    mkpath(temp_dir)
    input["output"]["base_directory"] = temp_dir

    # Get the profiles to use from the output files. For `restart_time_index=1` these are
    # the initial profiles, but after the inital relaxation of electrons on a fixed ion
    # background. The 'inital' ion profiles will usually be the final state of a Boltzmann
    # electron simulation.
    mk_state = setup_moment_kinetics(input; restart=run_prefix * ".dfns.h5",
                                     restart_time_index=restart_time_index,
                                     write_output=false, warn_unexpected_input=true)

    # Extract separate variables from mk_state Tuple
    pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa, vperp,
        gyrophase, z, r, moments, fields, spectral_objects, advect_objects, composition,
        collisions, geometry, gyroavs, boundary_distributions, external_source_settings,
        num_diss_params, nl_solver_params, advance, advance_implicit, fp_arrays,
        scratch_dummy, manufactured_source_list, ascii_io, io_moments, io_dfns = mk_state

    # Allocate an array for the Jacobian matrix
    pdf_size = length(pdf.electron.norm)
    ppar_size = length(moments.electron.ppar)
    jacobian_size = pdf_size + ppar_size
    jacobian_matrix = allocate_shared_float(jacobian_size, jacobian_size)

    # Fill the Jacobian matrix with the chosen terms
    ################################################

    # r-index - assume this is always 1 because we only consider 1D simulations
    ir = 1

    # Ion timestep within the first Runge-Kutta stage (t_params.dt[] is the length of the
    # full Runge-Kutta timestep).
    ion_dt = t_params.dt[] * t_params.rk_coefs_implicit[1,1]

    f = @view pdf.electron.norm[:,:,:,ir]
    ppar = @view moments.electron.ppar[:,ir]
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    vth = @view moments.electron.vth[:,ir]
    qpar = @view moments.electron.qpar[:,ir]
    me = composition.me_over_mi
    dt = t_params.electron.dt[]
    ddens_dz = @view moments.electron.ddens_dz[:,ir]
    dppar_dz = @view moments.electron.dppar_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]

    z_spectral = spectral_objects.z_spectral
    vpa_spectral = spectral_objects.vpa_spectral
    vperp_spectral = spectral_objects.vperp_spectral

    z_advect = advect_objects.z_advect
    vpa_advect = advect_objects.vpa_advect

    # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
    third_moment = scratch_dummy.buffer_z_1
    dthird_moment_dz = scratch_dummy.buffer_z_2
    buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]
    @begin_z_region()
    @loop_z iz begin
        third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
    end
    derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                  buffer_3, buffer_4, z_spectral, z)

    if all([include_z_advection, include_vpa_advection, include_electron_pdf_term,
            include_dissipation, include_krook, include_external_source,
            include_constraint_forcing, include_energy_equation, include_ion_dt_forcing])
        fill_electron_kinetic_equation_Jacobian!(
            jacobian_matrix, f, ppar, moments, fields.phi, collisions, composition, z,
            vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
            scratch_dummy, external_source_settings, num_diss_params, t_params.electron,
            ion_dt, ir, true)
    else
        # Allow any combination of terms, selected by the include_* flags

        z_speed = @view z_advect[1].speed[:,:,:,ir]
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

        if include_z_advection
            add_electron_z_advection_to_Jacobian!(
                jacobian_matrix, f, dens, upar, ppar, vth, dpdf_dz, me, z, vperp, vpa,
                z_spectral, z_advect, z_speed, scratch_dummy, dt, ir;
                ppar_offset=pdf_size)
        end
        if include_vpa_advection
            add_electron_vpa_advection_to_Jacobian!(
                jacobian_matrix, f, dens, upar, ppar, vth, third_moment, dpdf_dz,
                ddens_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp, vpa,
                z_spectral, vpa_spectral, vpa_advect, z_speed, scratch_dummy,
                external_source_settings, dt, ir; ppar_offset=pdf_size)
        end
        if include_electron_pdf_term
            add_contribution_from_electron_pdf_term_to_Jacobian!(
                jacobian_matrix, f, dens, upar, ppar, vth, third_moment, ddens_dz,
                dppar_dz, dvth_dz, dqpar_dz, dthird_moment_dz, moments, me,
                external_source_settings, z, vperp, vpa, z_spectral, z_speed,
                scratch_dummy, dt, ir; ppar_offset=pdf_size)
        end
        if include_dissipation
            add_electron_dissipation_term_to_Jacobian!(
                jacobian_matrix, f, num_diss_params, z, vperp, vpa, vpa_spectral, z_speed,
                dt, ir)
        end
        if include_krook
            add_electron_krook_collisions_to_Jacobian!(
                jacobian_matrix, f, dens, upar, ppar, vth, upar_ion, collisions, z, vperp,
                vpa, z_speed, dt, ir; ppar_offset=pdf_size)
        end
        if include_external_source
            add_total_external_electron_source_to_Jacobian!(
                jacobian_matrix, f, moments, me, z_speed,
                external_source_settings.electron, z, vperp, vpa, dt, ir;
                ppar_offset=pdf_size)
        end
        if include_constraint_forcing
            add_electron_implicit_constraint_forcing_to_Jacobian!(
                jacobian_matrix, f, z_speed, z, vperp, vpa,
                t_params.constraint_forcing_rate, dt, ir)
        end
        if include_energy_equation
            add_electron_energy_equation_to_Jacobian!(
                jacobian_matrix, f, dens, upar, ppar, vth, third_moment, ddens_dz,
                dupar_dz, dppar_dz, dthird_moment_dz, collisions, composition, z, vperp,
                vpa, z_spectral, num_diss_params, dt, ir; ppar_offset=pdf_size)
        end
        if include_ion_dt_forcing && ion_dt !== nothing
            add_ion_dt_forcing_of_electron_ppar_to_Jacobian!(
                jacobian_matrix, z, dt, ion_dt, ir; ppar_offset=pdf_size)
        end
        if include_wall_bc && t_params.electron.include_wall_bc_in_preconditioner
            add_wall_boundary_condition_to_Jacobian!(
                jacobian_matrix, fields.phi, f, ppar, vth, upar, z, vperp, vpa,
                vperp_spectral, vpa_spectral, vpa_advect, moments,
                num_diss_params.electron.vpa_dissipation_coefficient, me, ir;
                ppar_offset=pdf_size)
        end
    end

    return jacobian_matrix
end

end # MKJacobianUtils

using .MKJacobianUtils
