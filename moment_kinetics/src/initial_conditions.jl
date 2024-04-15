"""
"""
module initial_conditions

export allocate_pdf_and_moments
export init_pdf_and_moments!
export initialize_electrons!

# functional testing 
export create_boundary_distributions
export create_pdf

# package
using SpecialFunctions: erfc
# modules
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..bgk: init_bgk_pdf!
using ..boundary_conditions: vpagrid_to_dzdt
using ..communication
using ..external_sources
using ..interpolation: interpolate_to_grid_1d!
using ..looping
using ..em_fields: update_phi!
using ..file_io: setup_electron_io, write_electron_state, finish_electron_io
using ..load_data: reload_electron_data!
using ..moment_kinetics_structs: scratch_pdf, pdf_substruct, electron_pdf_substruct,
                                 pdf_struct, moments_struct, boundary_distributions_struct
using ..velocity_moments: integrate_over_vspace, integrate_over_neutral_vspace
using ..velocity_moments: integrate_over_positive_vz, integrate_over_negative_vz
using ..velocity_moments: create_moments_ion, create_moments_electron, create_moments_neutral
using ..velocity_moments: update_qpar!
using ..velocity_moments: update_neutral_density!, update_neutral_pz!, update_neutral_pr!, update_neutral_pzeta!
using ..velocity_moments: update_neutral_uz!, update_neutral_ur!, update_neutral_uzeta!, update_neutral_qz!
using ..velocity_moments: update_ppar!, update_upar!, update_density!, update_pperp!, update_vth!, reset_moments_status!
using ..electron_fluid_equations: calculate_electron_density!
using ..electron_fluid_equations: calculate_electron_upar_from_charge_conservation!
using ..electron_fluid_equations: calculate_electron_qpar!
using ..electron_fluid_equations: calculate_electron_parallel_friction_force!
using ..electron_kinetic_equation: update_electron_pdf!, enforce_boundary_condition_on_electron_pdf!
using ..input_structs: boltzmann_electron_response_with_simple_sheath, kinetic_electrons
using ..derivatives: derivative_z!
using ..utils: get_default_restart_filename, get_prefix_iblock_and_move_existing_file

using ..manufactured_solns: manufactured_solutions

using MPI

"""
Creates the structs for the pdf and the velocity-space moments
"""
function allocate_pdf_and_moments(composition, r, z, vperp, vpa, vzeta, vr, vz,
                                  evolve_moments, collisions, external_source_settings,
                                  num_diss_params, t_input)
    pdf = create_pdf(composition, r, z, vperp, vpa, vzeta, vr, vz)

    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    # moments arrays have same r and z grids for both ion and neutral species
    # and so are included in the same struct
    ion = create_moments_ion(z.n, r.n, composition.n_ion_species,
        evolve_moments.density, evolve_moments.parallel_flow,
        evolve_moments.parallel_pressure, external_source_settings.ion,
        num_diss_params)
    electron = create_moments_electron(z.n, r.n,
        composition.electron_physics, num_diss_params)
    neutral = create_moments_neutral(z.n, r.n, composition.n_neutral_species,
        evolve_moments.density, evolve_moments.parallel_flow,
        evolve_moments.parallel_pressure, external_source_settings.neutral,
        num_diss_params)

    if abs(collisions.ionization) > 0.0 || z.bc == "wall"
        # if ionization collisions are included or wall BCs are enforced, then particle
        # number is not conserved within each species
        particle_number_conserved = false
    else
        # by default, assumption is that particle number should be conserved for each species
        particle_number_conserved = true
    end

    moments = moments_struct(ion, electron, neutral, evolve_moments.density,
                             particle_number_conserved,
                             evolve_moments.conservation,
                             evolve_moments.parallel_flow,
                             evolve_moments.parallel_pressure)

    boundary_distributions = create_boundary_distributions(vz, vr, vzeta, vpa, vperp, z,
                                                           composition)

    return pdf, moments, boundary_distributions
end

"""
Allocate arrays for pdfs
"""
function create_pdf(composition, r, z, vperp, vpa, vzeta, vr, vz)
    # allocate pdf arrays
    pdf_ion_norm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, composition.n_ion_species)
    pdf_ion_buffer = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, composition.n_neutral_species) # n.b. n_species is n_neutral_species here
    pdf_neutral_norm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, composition.n_neutral_species)
    pdf_neutral_buffer = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, composition.n_ion_species)
    if composition.electron_physics == kinetic_electrons
        pdf_electron_norm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n)
        # MB: not sure if pdf_electron_buffer will ever be needed, but create for now
        # to emulate ion and neutral behaviour
        pdf_electron_buffer = allocate_shared_float(vpa.n, vperp.n, z.n, r.n)
        pdf_before_ion_timestep = allocate_shared_float(vpa.n, vperp.n, z.n, r.n)
        electron_substruct = electron_pdf_substruct(pdf_electron_norm,
                                                    pdf_electron_buffer,
                                                    pdf_before_ion_timestep)
    else
        electron_substruct = nothing
    end

    return pdf_struct(pdf_substruct(pdf_ion_norm, pdf_ion_buffer),
                      electron_substruct,
                      pdf_substruct(pdf_neutral_norm, pdf_neutral_buffer))

end

"""
creates the normalised pdfs and the velocity-space moments and populates them
with a self-consistent initial condition
"""
function init_pdf_and_moments!(pdf, moments, fields, boundary_distributions, geometry, composition, r, z,
                               vperp, vpa, vzeta, vr, vz, z_spectral, r_spectral,
                               vperp_spectral, vpa_spectral, vz_spectral, species,
                               collisions, external_source_settings,
                               manufactured_solns_input, scratch_dummy, t_input,
                               num_diss_params, advection_structs, io_input, input_dict)
    if manufactured_solns_input.use_for_init
        init_pdf_moments_manufactured_solns!(pdf, moments, vz, vr, vzeta, vpa, vperp, z,
                                             r, composition.n_ion_species,
                                             composition.n_neutral_species,
                                             geometry.input, composition, species,
                                             manufactured_solns_input)
    else
        n_ion_species = composition.n_ion_species
        n_neutral_species = composition.n_neutral_species
        @serial_region begin
            # initialise the ion density profile
            init_density!(moments.ion.dens, z, r, species.ion, n_ion_species)
            # initialise the ion parallel flow profile
            init_upar!(moments.ion.upar, z, r, species.ion, n_ion_species)
            # initialise the ion parallel thermal speed profile
            init_vth!(moments.ion.vth, z, r, species.ion, n_ion_species)
            @. moments.ion.ppar = 0.5 * moments.ion.dens * moments.ion.vth^2
            # initialise pressures assuming isotropic distribution
            @. moments.ion.ppar = 0.5 * moments.ion.dens * moments.ion.vth^2
            @. moments.ion.pperp = moments.ion.ppar
            if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
                @. moments.ion.constraints_A_coefficient = 1.0
                @. moments.ion.constraints_B_coefficient = 0.0
                @. moments.ion.constraints_C_coefficient = 0.0
            end
            if n_neutral_species > 0
                # initialise the neutral density profile
                init_density!(moments.neutral.dens, z, r, species.neutral, n_neutral_species)
                # initialise the z-component of the neutral flow
                init_uz!(moments.neutral.uz, z, r, species.neutral, n_neutral_species)
                # initialise the r-component of the neutral flow
                init_ur!(moments.neutral.ur, z, r, species.neutral, n_neutral_species)
                # initialise the zeta-component of the neutral flow
                init_uzeta!(moments.neutral.uzeta, z, r, species.neutral, n_neutral_species)
                # initialise the neutral thermal speed
                init_vth!(moments.neutral.vth, z, r, species.neutral, n_neutral_species)
                # calculate the z-component of the neutral pressure
                @. moments.neutral.pz = 0.5 * moments.neutral.dens * moments.neutral.vth^2
                # calculate the total neutral pressure
                @. moments.neutral.ptot = 1.5 * moments.neutral.dens * moments.neutral.vth^2
                if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
                    @. moments.neutral.constraints_A_coefficient = 1.0
                    @. moments.neutral.constraints_B_coefficient = 0.0
                    @. moments.neutral.constraints_C_coefficient = 0.0
                end
            end
        end
        # reflect the fact that the ion moments have now been updated
        moments.ion.dens_updated .= true
        moments.ion.upar_updated .= true
        moments.ion.ppar_updated .= true
        # account for the fact that the neutral moments have now been updated
        moments.neutral.dens_updated .= true
        moments.neutral.uz_updated .= true
        moments.neutral.pz_updated .= true
        # create and initialise the normalised, ion particle distribution function (pdf)
        # such that ∫dwpa pdf.norm = 1, ∫dwpa wpa * pdf.norm = 0, and ∫dwpa wpa^2 * pdf.norm = 1/2
        # note that wpa = vpa - upar, unless moments.evolve_ppar = true, in which case wpa = (vpa - upar)/vth
        # the definition of pdf.norm changes accordingly from pdf_unnorm / density to pdf_unnorm * vth / density
        # when evolve_ppar = true.
        initialize_pdf!(pdf, moments, boundary_distributions, composition, r, z, vperp,
                        vpa, vzeta, vr, vz, vpa_spectral, vz_spectral, species)

        begin_s_r_z_region()
        # calculate the initial parallel heat flux from the initial un-normalised pdf
        update_qpar!(moments.ion.qpar, moments.ion.qpar_updated,
                     moments.ion.dens, moments.ion.upar, moments.ion.vth,
                     pdf.ion.norm, vpa, vperp, z, r, composition,
                     moments.evolve_density, moments.evolve_upar, moments.evolve_ppar)

        begin_serial_region()
        @serial_region begin
            # If electrons are being used, they will be initialized properly later. Here
            # we only set the values to avoid false positives from the debug checks
            # (when @debug_track_initialized is active).
            moments.electron.dens .= 0.0
            moments.electron.upar .= 0.0
            moments.electron.ppar .= 0.0
            moments.electron.qpar .= 0.0
            moments.electron.temp .= 0.0
            moments.electron.constraints_A_coefficient .= 1.0
            moments.electron.constraints_B_coefficient .= 0.0
            moments.electron.constraints_C_coefficient .= 0.0
            if composition.electron_physics == kinetic_electrons
                pdf.electron.norm .= 0.0
            end
        end

        initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                              vzeta, vr, n_neutral_species)
        initialize_external_source_controller_integral!(moments, external_source_settings,
                                                        n_neutral_species)

        if n_neutral_species > 0
            update_neutral_qz!(moments.neutral.qz, moments.neutral.qz_updated,
                               moments.neutral.dens, moments.neutral.uz,
                               moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z,
                               r, composition, moments.evolve_density,
                               moments.evolve_upar, moments.evolve_ppar)
            update_neutral_pz!(moments.neutral.pz, moments.neutral.pz_updated,
                               moments.neutral.dens, moments.neutral.uz,
                               pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                               moments.evolve_density, moments.evolve_upar)
            update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated,
                               pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
            update_neutral_pzeta!(moments.neutral.pzeta, moments.neutral.pzeta_updated,
                                  pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        end
    end

    # Zero-initialise the dSdt diagnostic to avoid writing uninitialised values, as the
    # collision operator will not be calculated before the initial values are written to
    # file.
    @serial_region begin
        moments.ion.dSdt .= 0.0
    end

    init_boundary_distributions!(boundary_distributions, pdf, vz, vr, vzeta, vpa, vperp,
                                 z, r, composition)

    return nothing
end

function initialize_electrons!(pdf, moments, fields, geometry, composition, r, z,
                               vperp, vpa, vzeta, vr, vz, z_spectral, r_spectral,
                               vperp_spectral, vpa_spectral, collisions,
                               external_source_settings, scratch_dummy, scratch, t_params,
                               t_input, num_diss_params, advection_structs, io_input,
                               input_dict; restart_from_Boltzmann_electrons=false)
    
    moments.electron.dens_updated[] = false
    # initialise the electron density profile
    init_electron_density!(moments.electron.dens, moments.electron.dens_updated, moments.ion.dens)
    # initialise the electron parallel flow profile
    moments.electron.upar_updated[] = false
    init_electron_upar!(moments.electron.upar, moments.electron.upar_updated, moments.electron.dens, 
        moments.ion.upar, moments.ion.dens, composition.electron_physics, r, z)
    # different choices for initialization of electron temperature/pressure/vth depending on whether
    # we are restarting from a previous simulation with Boltzmann electrons or not
    if restart_from_Boltzmann_electrons
        begin_serial_region()
        @serial_region begin
            # if restarting from a simulations where Boltzmann electrons were used, then the assumption is
            # that the electron parallel temperature is constant along the field line and equal to T_e
            moments.electron.temp .= composition.T_e
            # the thermal speed is related to the temperature by vth_e / v_ref = sqrt((T_e/T_ref) / (m_e/m_ref))
            moments.electron.vth .= sqrt(composition.T_e / composition.me_over_mi)
            # ppar = 0.5 * n * T, so we can calculate the parallel pressure from the density and T_e
            moments.electron.ppar .= 0.5 * moments.electron.dens * composition.T_e
        end
    else
        # initialise the electron thermal speed profile
        init_electron_vth!(moments.electron.vth, moments.ion.vth, composition.T_e, composition.me_over_mi, z.grid)
        begin_r_z_region()
        # calculate the electron temperature from the thermal speed
        @loop_r_z ir iz begin
            moments.electron.temp[iz,ir] = composition.me_over_mi * moments.electron.vth[iz,ir]^2
        end
        # calculate the electron parallel pressure from the density and temperature
        @loop_r_z ir iz begin
            moments.electron.ppar[iz,ir] = 0.5 * moments.electron.dens[iz,ir] * moments.electron.temp[iz,ir]
        end
    end
    # the electron temperature has now been updated
    moments.electron.temp_updated[] = true
    # the electron parallel pressure now been updated
    moments.electron.ppar_updated[] = true

    # calculate the zed derivative of the initial electron density
    @views derivative_z!(moments.electron.ddens_dz, moments.electron.dens, 
        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
    # calculate the zed derivative of the initial electron temperature
    @views derivative_z!(moments.electron.dT_dz, moments.electron.temp, 
        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
    # calculate the zed derivative of the initial electron thermal speed
    @views derivative_z!(moments.electron.dvth_dz, moments.electron.vth, 
        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
    # calculate the zed derivative of the initial electron parallel pressure
    @views derivative_z!(moments.electron.dppar_dz, moments.electron.ppar, 
        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
    if composition.electron_physics == kinetic_electrons
        # Initialise the array for the electron pdf
        begin_serial_region()
        speed = @view scratch_dummy.buffer_vpavperpzrs_1[:,:,:,:,1]
        @serial_region begin
            speed .= 0.0
        end
        init_electron_pdf_over_density_and_boundary_phi!(
            pdf.electron.norm, fields.phi, moments.electron.dens, moments.electron.upar,
            moments.electron.vth, z, vpa, vperp, vperp_spectral, vpa_spectral,
            [(speed=speed,)], moments, num_diss_params,
            composition.me_over_mi, scratch_dummy)
    end
    # calculate the initial electron parallel heat flux;
    # if using kinetic electrons, this relies on the electron pdf, which itself relies on the electron heat flux
    moments.electron.qpar_updated[] = false
    calculate_electron_qpar!(moments.electron, pdf.electron, moments.electron.ppar,
        moments.electron.upar, moments.ion.upar, collisions.nu_ei, composition.me_over_mi,
        composition.electron_physics, vpa)
    # calculate the zed derivative of the initial electron parallel heat flux
    @views derivative_z!(moments.electron.dqpar_dz, moments.electron.qpar, 
        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
    # calculate the electron-ion parallel friction force
    calculate_electron_parallel_friction_force!(moments.electron.parallel_friction, moments.electron.dens,
        moments.electron.upar, moments.ion.upar, moments.electron.dT_dz,
        composition.me_over_mi, collisions.nu_ei, composition.electron_physics)
    
    # initialize the scratch arrays containing pdfs and moments for the first RK stage
    # the electron pdf is yet to be initialised but with the current code logic, the scratch
    # arrays need to exist and be otherwise initialised in order to compute the initial
    # electron pdf. The electron arrays will be updated as necessary by
    # initialize_electron_pdf!().
    if !restart_from_Boltzmann_electrons
        begin_serial_region()
        @serial_region begin
            scratch[1].electron_density .= moments.electron.dens
            scratch[1].electron_upar .= moments.electron.upar
            scratch[1].electron_ppar .= moments.electron.ppar
            scratch[1].electron_pperp .= 0.0 #moments.electron.pperp
        end
        # get the initial electrostatic potential and parallel electric field
        update_phi!(fields, scratch[1], z, r, composition, collisions, moments, z_spectral, r_spectral, scratch_dummy)
    end

    # initialize the electron pdf that satisfies the electron kinetic equation
    initialize_electron_pdf!(scratch, pdf, moments, fields.phi, r, z, vpa, vperp,
                             vzeta, vr, vz, z_spectral, vperp_spectral, vpa_spectral,
                             advection_structs.electron_z_advect,
                             advection_structs.electron_vpa_advect, scratch_dummy,
                             collisions, composition, geometry, external_source_settings,
                             num_diss_params, t_params.electron, t_input.electron_t_input,
                             io_input, input_dict)

    return nothing
end

"""
"""
function initialize_pdf!(pdf, moments, boundary_distributions, composition, r, z, vperp,
                         vpa, vzeta, vr, vz, vpa_spectral, vz_spectral, species)
    wall_flux_0 = allocate_float(r.n, composition.n_ion_species)
    wall_flux_L = allocate_float(r.n, composition.n_ion_species)

    @serial_region begin
        for is ∈ 1:composition.n_ion_species, ir ∈ 1:r.n
            # Add ion contributions to wall flux here. Neutral contributions will be
            # added in init_neutral_pdf_over_density!()
            if species.ion[is].z_IC.initialization_option == "bgk" || species.ion[is].vpa_IC.initialization_option == "bgk"
                @views init_bgk_pdf!(pdf.ion.norm[:,1,:,ir,is], 0.0, species.ion[is].initial_temperature, z.grid, z.L, vpa.grid)
            else
                # updates pdf_norm to contain pdf / density, so that ∫dvpa pdf.norm = 1,
                # ∫dwpa wpa * pdf.norm = 0, and ∫dwpa m_s (wpa/vths)^2 pdf.norm = 1/2
                # to machine precision
                @views init_ion_pdf_over_density!(
                    pdf.ion.norm[:,:,:,ir,is], species.ion[is], composition, vpa, vperp,
                    z, vpa_spectral, moments.ion.dens[:,ir,is],
                    moments.ion.upar[:,ir,is], moments.ion.ppar[:,ir,is],
                    moments.ion.vth[:,ir,is],
                    moments.ion.v_norm_fac[:,ir,is], moments.evolve_density,
                    moments.evolve_upar, moments.evolve_ppar)
            end
            @views wall_flux_0[ir,is] = -(moments.ion.dens[1,ir,is] *
                                          moments.ion.upar[1,ir,is])
            @views wall_flux_L[ir,is] = moments.ion.dens[end,ir,is] *
                                        moments.ion.upar[end,ir,is]

            @loop_z iz begin
                if moments.evolve_ppar
                    @. pdf.ion.norm[:,:,iz,ir,is] *= moments.ion.vth[iz,ir,is]
                elseif moments.evolve_density == false
                    @. pdf.ion.norm[:,:,iz,ir,is] *= moments.ion.dens[iz,ir,is]
                end
            end
        end
    end

    @serial_region begin
        @loop_sn_r isn ir begin
            # For now, assume neutral species index `isn` corresponds to the ion
            # species index `is`, to get the `wall_flux_0` and `wall_flux_L` for the
            # initial condition.
            @views init_neutral_pdf_over_density!(
                pdf.neutral.norm[:,:,:,:,ir,isn], boundary_distributions,
                species.neutral[isn], composition, vz, vr, vzeta, z, vz_spectral,
                moments.neutral.dens[:,ir,isn], moments.neutral.uz[:,ir,isn],
                moments.neutral.pz[:,ir,isn], moments.neutral.vth[:,ir,isn],
                moments.neutral.v_norm_fac[:,ir,isn], moments.evolve_density,
                moments.evolve_upar, moments.evolve_ppar,
                wall_flux_0[ir,min(isn,composition.n_ion_species)],
                wall_flux_L[ir,min(isn,composition.n_ion_species)])
            @loop_z iz begin
                if moments.evolve_ppar
                    @. pdf.neutral.norm[:,:,:,iz,ir,isn] *= moments.neutral.vth[iz,ir,isn]
                elseif moments.evolve_density == false
                    @. pdf.neutral.norm[:,:,:,iz,ir,isn] *= moments.neutral.dens[iz,ir,isn]
                end
            end
        end
    end

    # @serial_region begin
    #     @loop_r ir begin
    #         # this is the initial guess for the electron pdf
    #         # it will be iteratively updated to satisfy the time-independent
    #         # electron kinetic equation
    #         @views init_electron_pdf_over_density!(pdf.electron.norm[:,:,:,ir], moments.electron.dens[:,ir],
    #             moments.electron.upar[:,ir], moments.electron.vth[:,ir], z, vpa, vperp)
    #     end
    #     # now that we have our initial guess for the electron pdf, we iterate
    #     # using the time-independent electron kinetic equation to find a self-consistent
    #     # solution for the electron pdf
    #     max_electron_pdf_iterations = 100
    #     @views update_electron_pdf!(pdf.electron.norm, moments.electron.dens, moments.electron.vth, moments.electron.ppar, 
    #                                 moments.electron.ddens_dz, moments.electron.dppar_dz, moments.electron.dqpar_dz, moments.electron.dvth_dz,
    #                                 max_electron_pdf_iterations, z, vpa, z_spectral, vpa_spectral, scratch_dummy)
    # end



    return nothing
end

function initialize_electron_pdf!(scratch, pdf, moments, phi, r, z, vpa, vperp, vzeta, vr,
                                  vz, z_spectral, vperp_spectral, vpa_spectral, z_advect,
                                  vpa_advect, scratch_dummy, collisions, composition,
                                  geometry, external_source_settings, num_diss_params,
                                  t_params, t_input, io_input, input_dict)

    # now that the initial electron pdf is given, the electron parallel heat flux should be updated
    # if using kinetic electrons
    if composition.electron_physics == kinetic_electrons
        begin_serial_region()
        if t_input.no_restart
            restart_filename = nothing
        else
            restart_filename = get_default_restart_filename(io_input, "initial_electron";
                                                            error_if_no_file_found=false)
        end
        if restart_filename === nothing
            # No file to restart from
            previous_runs_info = nothing
            code_time = 0.0
            restart_time_index = -1
        else
            if global_rank[] == 0
                println("Restarting electrons from $restart_filename")
            end
            # Previously-created electron distribution function exists, so use it as
            # the initial guess.
            backup_prefix_iblock =
                get_prefix_iblock_and_move_existing_file(restart_filename,
                                                         io_input.output_dir)

            # Reload pdf and moments from an existing output file
            code_time, previous_runs_info, restart_time_index =
                reload_electron_data!(pdf, moments, t_params, backup_prefix_iblock, -1,
                                      geometry, r, z, vpa, vperp, vzeta, vr, vz)

            # Broadcast code_time from the root process of each shared-memory block (on which it
            # might have been loaded from a restart file).
            code_time = MPI.Bcast(code_time, 0, comm_block[])
        end
        # Set to `true` rather than `nothing` so that processes that are not writing
        # output (i.e. not rank-0 of their shared-memory block) know that 'initial
        # electron output' is being written (so that they know not to activate 'debug
        # I/O').
        io_initial_electron = true
        @serial_region begin
            # Setup I/O for initial electron state
            io_initial_electron = setup_electron_io(io_input, vpa, vperp, z, r,
                                                    composition, collisions,
                                                    moments.evolve_density,
                                                    moments.evolve_upar,
                                                    moments.evolve_ppar,
                                                    external_source_settings, input_dict,
                                                    restart_time_index,
                                                    previous_runs_info,
                                                    "initial_electron")

            # update the electron pdf in the first scratch
            scratch[1].pdf_electron .= pdf.electron.norm
        end

        @loop_r_z ir iz begin
            # update the electron thermal speed using the updated electron parallel pressure
            moments.electron.vth[iz,ir] = sqrt(abs(2.0 * moments.electron.ppar[iz,ir] / (moments.electron.dens[iz,ir] * composition.me_over_mi)))
        end

        moments.electron.qpar_updated[] = false
        calculate_electron_qpar!(moments.electron, pdf.electron, moments.electron.ppar,
                                 moments.electron.upar, moments.ion.upar,
                                 collisions.nu_ei, composition.me_over_mi,
                                 composition.electron_physics, vpa)
        # update dqpar/dz for electrons
        # calculate the zed derivative of the initial electron parallel heat flux
        @views derivative_z!(moments.electron.dqpar_dz, moments.electron.qpar, 
            scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
            scratch_dummy.buffer_rs_4[:,1], z_spectral, z)

        # now that we have our initial guess for the electron pdf, we iterate
        # using the time-independent electron kinetic equation to find a self-consistent
        # solution for the electron pdf.
        # First run with evolve_ppar=true to get electron_ppar close to steady state.
        # electron_ppar does not have to be exactly steady state as it will be
        # time-evolved along with the ions.
        max_electron_pdf_iterations = 2000000
        #max_electron_pdf_iterations = 500000
        #max_electron_pdf_iterations = 10000
        if global_rank[] == 0
            println("Initializing electrons - evolving both pdf_electron and electron_ppar")
        end
        electron_pseudotime, n_debug_outputs =
            @views update_electron_pdf!(scratch, pdf.electron.norm, moments, phi, r, z,
                                        vperp, vpa, z_spectral, vperp_spectral,
                                        vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                                        t_params, collisions, composition,
                                        external_source_settings, num_diss_params,
                                        max_electron_pdf_iterations;
                                        io_electron=io_initial_electron,
                                        initial_time=code_time,
                                        residual_tolerance=t_input.initialization_residual_value,
                                        evolve_ppar=true)

        # Now run without evolve_ppar=true to get pdf_electron fully to steady state,
        # ready for the start of the ion time advance.
        if global_rank[] == 0
            println("Initializing electrons - evolving pdf_electron only to steady state")
        end
        electron_pseudotime, n_debug_outputs =
            @views update_electron_pdf!(scratch, pdf.electron.norm, moments, phi, r, z,
                                        vperp, vpa, z_spectral, vperp_spectral,
                                        vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                                        t_params, collisions, composition,
                                        external_source_settings, num_diss_params,
                                        max_electron_pdf_iterations;
                                        io_electron=io_initial_electron,
                                        initial_time=electron_pseudotime,
                                        initial_output_counter=n_debug_outputs)

        begin_r_z_vperp_vpa_region()
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            pdf.electron.pdf_before_ion_timestep[ivpa,ivperp,iz,ir] =
                pdf.electron.norm[ivpa,ivperp,iz,ir]
        end

        # Write the converged initial state for the electrons to a file so that it can be
        # re-used if the simulation is re-run.
        t_idx = n_debug_outputs + 1
        write_electron_state(pdf.electron.norm, moments, t_params, electron_pseudotime,
                             io_initial_electron, t_idx, r, z, vperp, vpa)
        finish_electron_io(io_initial_electron)

    end
    return nothing
end

"""
for now the only initialisation option for the temperature is constant in z
returns vth0 = sqrt(2Ts/ms) / sqrt(2Te/ms) = sqrt(Ts/Te)
"""
function init_vth!(vth, z, r, spec, n_species)
    for is ∈ 1:n_species
        # Initialise as temperature first, then square root.
        for ir ∈ 1:r.n
            if spec[is].z_IC.initialization_option == "sinusoid"
                # initial condition is sinusoid in z
                @. vth[:,ir,is] =
                    (spec[is].initial_temperature
                     * (1.0 + spec[is].z_IC.temperature_amplitude
                              * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L +
                                    spec[is].z_IC.temperature_phase)))
            elseif spec[is].z_IC.initialization_option == "2D-instability-test"
                background_wavenumber = 1 + round(mk_int,
                                                  spec[is].z_IC.temperature_phase)
                @. vth[:,ir,is] =
                    (spec[is].initial_temperature
                     * (1.0 + spec[is].z_IC.temperature_amplitude
                              * sin(2.0*π*background_wavenumber*z.grid/z.L)))

                # initial perturbation with amplitude set by 'r' initial condition
                # settings, but using the z_IC.wavenumber as the background is always
                # 'wavenumber=1'.
                @. vth[:,ir,is] +=
                (spec[is].initial_temperature
                 * spec[is].r_IC.temperature_amplitude
                 * cos(2.0*π*(spec[is].z_IC.wavenumber*z.grid/z.L +
                              spec[is].r_IC.wavenumber*r.grid[ir]/r.L) +
                       spec[is].r_IC.temperature_phase))
            else
                @. vth[:,ir,is] =  spec[is].initial_temperature
            end
        end
        if r.n > 1
            for iz ∈ 1:z.n
                if spec[is].r_IC.initialization_option == "sinusoid"
                    # initial condition is sinusoid in r
                    @. vth[iz,:,is] +=
                    (spec[is].initial_temperature
                     * spec[is].r_IC.temperature_amplitude
                     * cos(2.0*π*spec[is].r_IC.wavenumber*r.grid/r.L +
                           spec[is].r_IC.temperature_phase))
                elseif spec[is].r_IC.initialization_option == "2D-instability-test"
                    # do nothing here
                end
            end
        end
    end
    @. vth = sqrt(vth)
    return nothing
end

"""
"""
function init_density!(dens, z, r, spec, n_species)
    for is ∈ 1:n_species
        println("init_option: ", spec[is].z_IC.initialization_option)
        for ir ∈ 1:r.n
            if spec[is].z_IC.initialization_option == "gaussian"
                # initial condition is an unshifted Gaussian
                @. dens[:,ir,is] = spec[is].initial_density + exp(-(z.grid/spec[is].z_IC.width)^2)
                println("ion_dens: ", dens[1,1,is], " init: ", spec[is].initial_density, " sum_factor: ", exp(-(z.grid[1]/spec[is].z_IC.width)^2))
            elseif spec[is].z_IC.initialization_option == "sinusoid"
                # initial condition is sinusoid in z
                @. dens[:,ir,is] =
                    (spec[is].initial_density
                     * (1.0 + spec[is].z_IC.density_amplitude
                              * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                                    + spec[is].z_IC.density_phase)))
            elseif spec[is].z_IC.initialization_option == "smoothedsquare"
                # initial condition is first 3 Fourier harmonics of a square wave
                argument = 2.0*π*(spec[is].z_IC.wavenumber*z.grid/z.L +
                                  spec[is].z_IC.density_phase)
                @. dens[:,ir,is] =
                    (spec[is].initial_density
                     * (1.0 + spec[is].z_IC.density_amplitude
                              * cos(argument - sin(2.0*argument)) ))
            elseif spec[is].z_IC.initialization_option == "2D-instability-test"
                if spec[is].z_IC.density_amplitude == 0.0
                    dens[:,ir,is] .= spec[is].initial_density
                else
                    background_wavenumber = 1 + round(mk_int,
                                                      spec[is].z_IC.temperature_phase)
                    eta0 = @. (spec[is].initial_density
                               * (1.0 + spec[is].z_IC.density_amplitude
                                  * sin(2.0*π*background_wavenumber*z.grid/z.L
                                        + spec[is].z_IC.density_phase)))
                    T0 = @. (spec[is].initial_temperature
                             * (1.0 + spec[is].z_IC.temperature_amplitude
                                * sin(2.0*π*background_wavenumber*z.grid/z.L)
                               ))
                    @. dens[:,ir,is] = eta0^((T0/(1+T0)))
                end

                # initial perturbation with amplitude set by 'r' initial condition
                # settings, but using the z_IC.wavenumber as the background is always
                # 'wavenumber=1'.
                @. dens[:,ir,is] +=
                (spec[is].initial_density
                 * spec[is].r_IC.density_amplitude
                 * cos(2.0*π*(spec[is].z_IC.wavenumber*z.grid/z.L +
                              spec[is].r_IC.wavenumber*r.grid[ir]/r.L) +
                       spec[is].r_IC.density_phase))
            elseif spec[is].z_IC.initialization_option == "monomial"
                # linear variation in z, with offset so that
                # function passes through zero at upwind boundary
                @. dens[:,ir,is] = (z.grid + 0.5*z.L)^spec[is].z_IC.monomial_degree
            end
        end
        if r.n > 1
            for iz ∈ 1:z.n
                if spec[is].r_IC.initialization_option == "gaussian"
                    # initial condition is an unshifted Gaussian
                    @. dens[iz,:,is] += exp(-(r.grid/spec[is].r_IC.width)^2)
                elseif spec[is].r_IC.initialization_option == "sinusoid"
                    # initial condition is sinusoid in r
                    @. dens[iz,:,is] +=
                        (spec[is].r_IC.density_amplitude
                         * cos(2.0*π*spec[is].r_IC.wavenumber*r.grid/r.L
                               + spec[is].r_IC.density_phase))
                elseif spec[is].z_IC.initialization_option == "smoothedsquare"
                    # initial condition is first 3 Fourier harmonics of a square wave
                    argument = 2.0*π*(spec[is].r_IC.wavenumber*r.grid/r.L +
                                      spec[is].r_IC.density_phase)
                    @. dens[iz,:,is] +=
                        spec[is].initial_density * spec[is].r_IC.density_amplitude *
                        cos(argument - sin(2.0*argument))
                elseif spec[is].r_IC.initialization_option == "2D-instability-test"
                    # do nothing here
                elseif spec[is].r_IC.initialization_option == "monomial"
                    # linear variation in r, with offset so that
                    # function passes through zero at upwind boundary
                    @. dens[iz,:,is] += (r.grid + 0.5*r.L)^spec[is].r_IC.monomial_degree
                end
            end
        end
    end
    return nothing
end

"""
"""
function init_upar!(upar, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            if spec[is].z_IC.initialization_option == "sinusoid"
                # initial condition is sinusoid in z
                @. upar[:,ir,is] =
                    (spec[is].z_IC.upar_amplitude
                     * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                           + spec[is].z_IC.upar_phase))
            elseif spec[is].z_IC.initialization_option == "gaussian" # "linear"
                # initial condition is linear in z
                # this is designed to give a nonzero J_{||i} at endpoints in z
                # necessary for an electron sheath condition involving J_{||i}
                # option "gaussian" to be consistent with usual init option for now
                @. upar[:,ir,is] = spec[is].z_IC.upar_amplitude * 2.0 * z.grid / z.L
            else
                @. upar[:,ir,is] = 0.0
            end
        end
        if r.n > 1
            for iz ∈ 1:z.n
                if spec[is].r_IC.initialization_option == "sinusoid"
                    # initial condition is sinusoid in r
                    @. upar[iz,:,is] +=
                        (spec[is].r_IC.upar_amplitude
                         * cos(2.0*π*spec[is].r_IC.wavenumber*r.grid/r.L
                               + spec[is].r_IC.upar_phase))
                elseif spec[is].r_IC.initialization_option == "gaussian" # "linear"
                    # initial condition is linear in r
                    # this is designed to give a nonzero J_{||i} at endpoints in r
                    # necessary for an electron sheath condition involving J_{||i}
                    # option "gaussian" to be consistent with usual init option for now
                    @. upar[iz,:,is] +=
                        (spec[is].r_IC.upar_amplitude * 2.0 *
                         (r.grid[:] - r.grid[floor(Int,r.n/2)])/r.L)
                end
            end
        end
    end
    return nothing
end

"""
"""
function init_uz!(uz, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            if spec[is].z_IC.initialization_option == "sinusoid"
                # initial condition is sinusoid in z
                @. uz[:,ir,is] =
                    (spec[is].z_IC.upar_amplitude
                     * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                           + spec[is].z_IC.upar_phase))
            elseif spec[is].z_IC.initialization_option == "gaussian" # "linear"
                # initial condition is linear in z
                # this is designed to give a nonzero J_{||i} at endpoints in z
                # necessary for an electron sheath condition involving J_{||i}
                # option "gaussian" to be consistent with usual init option for now
                @. uz[:,ir,is] = spec[is].z_IC.upar_amplitude * 2.0 * z.grid / z.L
            else
                @. uz[:,ir,is] = 0.0
            end
        end
    end
    return nothing
end

function init_ur!(ur, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            @. ur[:,ir,is] = 0.0
        end
    end
    return nothing
end

function init_uzeta!(uzeta, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            @. uzeta[:,ir,is] = 0.0
        end
    end
    return nothing
end

"""
initialise the electron density
"""
function init_electron_density!(electron_density, updated, ion_density)
    # use quasineutrality to obtain the electron density from the initial
    # densities of the various ion species
    calculate_electron_density!(electron_density, updated, ion_density)
    return nothing
end

"""
initialise the electron parallel flow density
"""
function init_electron_upar!(upar_e, updated, dens_e, upar_i, dens_i, electron_model, r, z)
    calculate_electron_upar_from_charge_conservation!(upar_e, updated, dens_e, upar_i, dens_i, electron_model, r, z)
    return nothing
end

"""
initialise the electron thermal speed profile.
for now the only initialisation option for the temperature is constant in z.
returns vth0 = sqrt(2*Ts/Te)
"""
function init_electron_vth!(vth_e, vth_i, T_e, me_over_mi, z)
    begin_r_z_region()
    # @loop_r_z ir iz begin
    #     vth_e[iz,ir] = sqrt(T_e)
    # end
    @loop_r_z ir iz begin
        vth_e[iz,ir] = vth_i[iz,ir,1] / sqrt(me_over_mi)
        #vth_e[iz,ir] = exp(-5*(z[iz]/z[end])^2)/sqrt(me_over_mi)
    end
end

"""
"""
function init_ion_pdf_over_density!(pdf, spec, composition, vpa, vperp, z,
        vpa_spectral, density, upar, ppar, vth, v_norm_fac, evolve_density, evolve_upar,
        evolve_ppar)

    if spec.vpa_IC.initialization_option == "gaussian"
        # initial condition is a Gaussian in the peculiar velocity
        if z.bc != "wall"
            for iz ∈ 1:z.n
                # obtain (vpa - upar)/vth
                if evolve_ppar
                    # if evolve_upar = true and evolve_ppar = true, then vpa coordinate is (vpa-upar)/vth;
                    if evolve_upar
                        @. vpa.scratch = vpa.grid
                        # if evolve_upar = false and evolve_ppar = true, then vpa coordinate is vpa/vth;
                    else
                        @. vpa.scratch = vpa.grid - upar[iz]/vth[iz]
                    end
                    # if evolve_upar = true and evolve_ppar = false, then vpa coordinate is vpa-upar;
                elseif evolve_upar
                    @. vpa.scratch = vpa.grid/vth[iz]
                    # if evolve_upar = false and evolve_ppar = false, then vpa coordinate is vpa;
                else
                    @. vpa.scratch = (vpa.grid - upar[iz])/vth[iz]
                end

                @. vperp.scratch = vperp.grid/vth[iz]

                @loop_vperp_vpa ivperp ivpa begin
                    pdf[ivpa,ivperp,iz] = exp(-vpa.scratch[ivpa]^2 -
                                              vperp.scratch[ivperp]^2) / vth[iz]
                end
            end

            # Only do this correction for runs without wall bc, because consistency of
            # pdf and moments is taken care of by convert_full_f_ion_to_normalised!()
            # for wall bc cases.
            for iz ∈ 1:z.n
                # densfac = the integral of the pdf over v-space, which should be unity,
                # but may not be exactly unity due to quadrature errors
                densfac = integrate_over_vspace(view(pdf,:,:,iz), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
                # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
                # where w_s = vpa - upar_s;
                # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
                pparfac = integrate_over_vspace(vpa.scratch2, vpa.wgts)
                pparfac = @views (v_norm_fac[iz]/vth[iz])^2 *
                                 integrate_over_vspace(pdf[:,:,iz], vpa.grid, 2, vpa.wgts,
                                                       vperp.grid, 0, vperp.wgts)
                # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
                @views @. vpa.scratch2 = vpa.grid^2 *(vpa.grid^2/pparfac - 1.0/densfac)
                pparfac2 = @views (v_norm_fac[iz]/vth[iz])^4 * integrate_over_vspace(pdf[:,:,iz], vpa.scratch2, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)

                @loop_vperp ivperp begin
                    @views @. pdf[:,ivperp,iz] = pdf[:,ivperp,iz]/densfac +
                                                 (0.5 - pparfac/densfac)/pparfac2 *
                                                 (vpa.grid^2/pparfac - 1.0/densfac) *
                                                 pdf[:,ivperp,iz]*(v_norm_fac[iz]/vth[iz])^2
                end
            end
        else
            # First create distribution functions at the z-boundary points that obey the
            # boundary conditions.
            if z.irank == 0 && z.irank == z.nrank - 1
                zrange = (1,z.n)
            elseif z.irank == 0
                zrange = (1,)
            elseif z.irank == z.nrank - 1
                zrange = (z.n,)
            else
                zrange = ()
            end
            for iz ∈ zrange
                @loop_vperp ivperp begin
                    # Initialise as full-f distribution functions, then
                    # normalise/interpolate (if necessary). This makes it easier to
                    # initialise a normalised pdf consistent with the moments, although it
                    # modifies the moments from the 'input' values.
                    @. pdf[:,ivperp,iz] = density[iz] *
                                          exp(-((vpa.grid - upar[iz])^2 + vperp.grid[ivperp]^2)
                                               / vth[iz]^2) / vth[iz]

                    # Also ensure both species go to zero smoothly at v_parallel=0 at the
                    # wall, where the boundary conditions require that distribution
                    # functions for both ions (where f_ion(v_parallel) = 0 for
                    # ±v_parallel<0) and neutrals (f_neutral=f_Kw, and f_Kw(v_parallel=0)=0)
                    # vanish.
                    #
                    # Implemented by multiplying by a smooth 'notch' function
                    # notch(v,u0,width) = 1 - exp(-(v-u0)^2/width)
                    width = 0.1 * vth[iz]
                    inverse_width = 1.0 / width

                    @. pdf[:,ivperp,iz] *= 1.0 - exp(-vpa.grid^2*inverse_width)
                end
            end
            # Can use non-shared memory here because `init_ion_pdf_over_density!()` is
            # called inside a `@serial_region`
            lower_z_pdf_buffer = allocate_float(vpa.n, vperp.n)
            upper_z_pdf_buffer = allocate_float(vpa.n, vperp.n)
            if z.irank == 0
                lower_z_pdf_buffer .= pdf[:,:,1]
            end
            if z.irank == z.nrank - 1
                upper_z_pdf_buffer .= pdf[:,:,end]
            end
            @views MPI.Bcast!(lower_z_pdf_buffer, 0, z.comm)
            @views MPI.Bcast!(upper_z_pdf_buffer, z.nrank - 1, z.comm)

            zero = 1.e-14
            for ivpa ∈ 1:vpa.n
                if vpa.grid[ivpa] > zero
                    lower_z_pdf_buffer[ivpa,:] .= 0.0
                end
                if vpa.grid[ivpa] < -zero
                    upper_z_pdf_buffer[ivpa,:] .= 0.0
                end
            end

            # Taper boundary distribution functions into each other across the
            # domain to avoid jumps.
            # Add some profile for density by scaling the pdf.
            @. z.scratch = 1.0 + 0.5 * (1.0 - (2.0 * z.grid / z.L)^2)
            for iz ∈ 1:z.n
                # right_weight is 0 on left boundary and 1 on right boundary
                right_weight = z.grid[iz]/z.L + 0.5
                #right_weight = min(max(0.5 + 0.5*(2.0*z.grid[iz]/z.L)^5, 0.0), 1.0)
                #right_weight = min(max(0.5 +
                #                       0.7*(2.0*z.grid[iz]/z.L) -
                #                       0.2*(2.0*z.grid[iz]/z.L)^3, 0.0), 1.0)
                # znorm is 1.0 at the boundary and 0.0 at the midplane
                @views @. pdf[:,:,iz] = z.scratch[iz] * (
                                            (1.0 - right_weight)*lower_z_pdf_buffer +
                                            right_weight*upper_z_pdf_buffer)
            end

            # Get the unnormalised pdf and the moments of the constructed full-f
            # distribution function (which will be modified from the input moments).
            convert_full_f_ion_to_normalised!(pdf, density, upar, ppar, vth, vperp,
                                                  vpa, vpa_spectral, evolve_density,
                                                  evolve_upar, evolve_ppar)

            if !evolve_density
                # Need to divide out density to return pdf/density
                @loop_vperp_vpa ivperp ivpa begin
                    pdf[ivpa,ivperp,:] ./= density
                end
            end
        end
    elseif spec.vpa_IC.initialization_option == "vpagaussian"
        @loop_z_vperp iz ivperp begin
            #@. pdf[:,iz] = vpa.grid^2*exp(-(vpa.grid*(v_norm_fac[iz]/vth[iz]))^2) / vth[iz]
            @. pdf[:,ivperp,iz] = vpa.grid^2*exp(-(vpa.grid)^2 - vperp.grid[ivperp]^2) / vth[iz]
        end
    elseif spec.vpa_IC.initialization_option == "sinusoid"
        # initial condition is sinusoid in vpa
        @loop_z_vperp iz ivperp begin
            @. pdf[:,ivperp,iz] = spec.vpa_IC.amplitude*cospi(2.0*spec.vpa_IC.wavenumber*vpa.grid/vpa.L)
        end
    elseif spec.vpa_IC.initialization_option == "monomial"
        # linear variation in vpa, with offset so that
        # function passes through zero at upwind boundary
        @loop_z_vperp iz ivperp begin
            @. pdf[:,ivperp,iz] = (vpa.grid + 0.5*vpa.L)^spec.vpa_IC.monomial_degree
        end
    end
    return nothing
end

"""
"""
function init_neutral_pdf_over_density!(pdf, boundary_distributions, spec, composition,
        vz, vr, vzeta, z, vz_spectral, density, uz, pz, vth, v_norm_fac, evolve_density,
        evolve_upar, evolve_ppar, wall_flux_0, wall_flux_L)

    # Reduce the ion flux by `recycling_fraction` to account for ions absorbed by the
    # wall.
    wall_flux_0 *= composition.recycling_fraction
    wall_flux_L *= composition.recycling_fraction

    #if spec.vz_IC.initialization_option == "gaussian"
    # For now, continue to use 'vpa' initialization options for neutral species
    if spec.vpa_IC.initialization_option == "gaussian"
        # initial condition is a Gaussian in the peculiar velocity
        if z.bc != "wall"
            for iz ∈ 1:z.n
                # obtain (vz - uz)/vth
                if evolve_ppar
                    # if evolve_upar = true and evolve_ppar = true, then vz coordinate is (vz-uz)/vth;
                    if evolve_upar
                        @. vz.scratch = vz.grid
                        # if evolve_upar = false and evolve_ppar = true, then vz coordinate is vz/vth;
                    else
                        @. vz.scratch = vz.grid - uz[iz]/vth[iz]
                    end
                    # if evolve_upar = true and evolve_ppar = false, then vz coordinate is vz-uz;
                elseif evolve_upar
                    @. vz.scratch = vz.grid/vth[iz]
                    # if evolve_upar = false and evolve_ppar = false, then vz coordinate is vz;
                else
                    @. vz.scratch = (vz.grid - uz[iz])/vth[iz]
                end

                @. vzeta.scratch = vzeta.grid/vth[iz]
                @. vr.scratch = vr.grid/vth[iz]

                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    pdf[ivz,ivr,ivzeta,iz] = exp(-vz.scratch[ivz]^2 - vr.scratch[ivr]^2
                                                 - vzeta.scratch[ivzeta]^2) / vth[iz]
                end
            end

            # Only do this correction for runs without wall bc, because consistency of
            # pdf and moments is taken care of by convert_full_f_neutral_to_normalised!()
            # for wall bc cases.
            for iz ∈ 1:z.n
                # densfac = the integral of the pdf over v-space, which should be unity,
                # but may not be exactly unity due to quadrature errors
                densfac = integrate_over_neutral_vspace(view(pdf,:,:,:,iz), vz.grid, 0,
                                                        vz.wgts, vr.grid, 0, vr.wgts,
                                                        vzeta.grid, 0, vzeta.wgts)
                # pzfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
                # where w_s = vz - uz_s;
                # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
                @views @. vz.scratch = vz.grid^2 * (v_norm_fac[iz]/vth[iz])^2
                pzfac = integrate_over_neutral_vspace(pdf[:,:,:,iz], vz.scratch, 1,
                                                        vz.wgts, vr.grid, 0, vr.wgts,
                                                        vzeta.grid, 0, vzeta.wgts)
                # pzfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
                @views @. vz.scratch = vz.grid^2 *(vz.grid^2/pzfac - 1.0/densfac) *
                                        (v_norm_fac[iz]/vth[iz])^4
                pzfac2 = @views integrate_over_neutral_vspace(pdf[:,:,:,iz], vz.scratch,
                                                                1, vz.wgts, vr.grid, 0,
                                                                vr.wgts, vzeta.grid, 0,
                                                                vzeta.wgts)

                @loop_vzeta_vr ivzeta ivr begin
                    @views @. pdf[:,ivr,ivzeta,iz] = pdf[:,ivr,ivzeta,iz]/densfac + (0.5 - pzfac/densfac)/pzfac2*(vz.grid^2/pzfac - 1.0/densfac)*pdf[:,ivr,ivzeta,iz]*(v_norm_fac[iz]/vth[iz])^2
                end
            end
        else
            # First create distribution functions at the z-boundary points that obey the
            # boundary conditions.
            if z.irank == 0 && z.irank == z.nrank - 1
                zrange = (1,z.n)
            elseif z.irank == 0
                zrange = (1,)
            elseif z.irank == z.nrank - 1
                zrange = (z.n,)
            else
                zrange = ()
            end
            for iz ∈ zrange
                @loop_vzeta_vr ivzeta ivr begin
                    # Initialise as full-f distribution functions, then
                    # normalise/interpolate (if necessary). This makes it easier to
                    # initialise a normalised pdf consistent with the moments, although it
                    # modifies the moments from the 'input' values.
                    @. pdf[:,ivr,ivzeta,iz] = density[iz] *
                                              exp(-((vz.grid - uz[iz])^2 +
                                                    vzeta.grid[ivzeta]^2 + vr.grid[ivr]^2)
                                                  / vth[iz]^2) / vth[iz]

                    # Also ensure both species go to zero smoothly at v_z=0 at the
                    # wall, where the boundary conditions require that distribution
                    # functions for both ions (where f_ion(v_z) = 0 for
                    # ±v_z<0) and neutrals (f_neutral=f_Kw, and f_Kw(v_z=0)=0)
                    # vanish.
                    #
                    # Implemented by multiplying by a smooth 'notch' function
                    # notch(v,u0,width) = 1 - exp(-(v-u0)^2/width)
                    width = 0.1
                    inverse_width = 1.0 / width

                    @. pdf[:,ivr,ivzeta,iz] *= 1.0 - exp(-vz.grid^2*inverse_width)
                end
            end
            # Can use non-shared memory here because `init_ion_pdf_over_density!()` is
            # called inside a `@serial_region`
            lower_z_pdf_buffer = allocate_float(vz.n, vr.n, vzeta.n)
            upper_z_pdf_buffer = allocate_float(vz.n, vr.n, vzeta.n)
            if z.irank == 0
                lower_z_pdf_buffer .= pdf[:,:,:,1]
            end
            if z.irank == z.nrank - 1
                upper_z_pdf_buffer .= pdf[:,:,:,end]
            end

            # Get the boundary pdfs from the processes that have the actual z-boundary
            @views MPI.Bcast!(lower_z_pdf_buffer, 0, z.comm)
            @views MPI.Bcast!(upper_z_pdf_buffer, z.nrank - 1, z.comm)

            # Also need to get the (ion) wall fluxes from the processes that have the
            # actual z-boundary
            temp = Ref(wall_flux_0)
            @views MPI.Bcast!(temp, 0, z.comm)
            wall_flux_0 = temp[]
            temp[] = wall_flux_L
            @views MPI.Bcast!(temp, z.nrank - 1, z.comm)
            wall_flux_L = temp[]

            knudsen_pdf = boundary_distributions.knudsen

            zero = 1.0e-14

            # add this species' contribution to the combined ion/neutral particle flux
            # out of the domain at z=-Lz/2
            @views wall_flux_0 += integrate_over_negative_vz(
                                      abs.(vz.grid) .* lower_z_pdf_buffer, vz.grid, vz.wgts,
                                      vz.scratch3, vr.grid, vr.wgts, vzeta.grid,
                                      vzeta.wgts)
            # for left boundary in zed (z = -Lz/2), want
            # f_n(z=-Lz/2, v_z > 0) = Γ_0 * f_KW(v_z) * pdf_norm_fac(-Lz/2)
            @loop_vz ivz begin
                if vz.grid[ivz] > zero
                    @. lower_z_pdf_buffer[ivz,:,:] = wall_flux_0 * knudsen_pdf[ivz,:,:]
                end
            end

            # add this species' contribution to the combined ion/neutral particle flux
            # out of the domain at z=-Lz/2
            @views wall_flux_L += integrate_over_positive_vz(
                                      abs.(vz.grid) .* upper_z_pdf_buffer, vz.grid, vz.wgts,
                                      vz.scratch3, vr.grid, vr.wgts, vzeta.grid,
                                      vzeta.wgts)
            # for right boundary in zed (z = Lz/2), want
            # f_n(z=Lz/2, v_z < 0) = Γ_Lz * f_KW(v_z) * pdf_norm_fac(Lz/2)
            @loop_vz ivz begin
                if vz.grid[ivz] < -zero
                    @. upper_z_pdf_buffer[ivz,:,:] = wall_flux_L * knudsen_pdf[ivz,:,:]
                end
            end

            # Taper boundary distribution functions into each other across the
            # domain to avoid jumps.
            # Add some profile for density by scaling the pdf.
            @. z.scratch = 1.0 - 0.5 * (1.0 - (2.0 * z.grid / z.L)^2)

            for iz ∈ 1:z.n
                # right_weight is 0 on left boundary and 1 on right boundary
                right_weight = z.grid[iz]/z.L + 0.5
                #right_weight = min(max(0.5 + 0.5*(2.0*z.grid[iz]/z.L)^5, 0.0), 1.0)
                #right_weight = min(max(0.5 +
                #                       0.7*(2.0*z.grid[iz]/z.L) -
                #                       0.2*(2.0*z.grid[iz]/z.L)^3, 0.0), 1.0)
                # znorm is 1.0 at the boundary and 0.0 at the midplane
                @views @. pdf[:,:,:,iz] = z.scratch[iz] * (
                                              (1.0 - right_weight)*lower_z_pdf_buffer +
                                              right_weight*upper_z_pdf_buffer)
            end

            # Get the unnormalised pdf and the moments of the constructed full-f
            # distribution function (which will be modified from the input moments).
            convert_full_f_neutral_to_normalised!(pdf, density, uz, pz, vth, vzeta, vr,
                                                  vz, vz_spectral, evolve_density,
                                                  evolve_upar, evolve_ppar)

            if !evolve_density
                # Need to divide out density to return pdf/density
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    pdf[ivz,ivr,ivzeta,:] ./= density
                end
            end
        end
    #elseif spec.vz_IC.initialization_option == "vzgaussian"
    elseif spec.vpa_IC.initialization_option == "vzgaussian"
        @loop_z_vzeta_vr iz ivzeta ivr begin
            @. pdf[:,ivr,ivzeta,iz] = vz.grid^2*exp(-vz.scratch^2 - vr[ivr]^2 -
                                                    vzeta[ivzeta]^2) / vth[iz]
        end
    #elseif spec.vz_IC.initialization_option == "sinusoid"
    elseif spec.vpa_IC.initialization_option == "sinusoid"
        # initial condition is sinusoid in vz
        @loop_z_vzeta_vr iz ivzeta ivr begin
            @. pdf[:,ivr,ivzeta,iz] =
                spec.vz_IC.amplitude*cospi(2.0*spec.vz_IC.wavenumber*vz.grid/vz.L)
        end
    #elseif spec.vz_IC.initialization_option == "monomial"
    elseif spec.vpa_IC.initialization_option == "monomial"
        # linear variation in vz, with offset so that
        # function passes through zero at upwind boundary
        @loop_z_vzeta_vr iz ivzeta ivr begin
            @. pdf[:,ivr,ivzeta,iz] = (vz.grid + 0.5*vz.L)^spec.vz_IC.monomial_degree
        end
    end
    return nothing
end

"""
init_electron_pdf_over_density_and_boundary_phi initialises the normalised electron pdf = pdf_e *
vth_e / dens_e and the boundary values of the electrostatic potential phi;
care is taken to ensure that the parallel boundary condition is satisfied;
NB: as the electron pdf is obtained via a time-independent equation,
this 'initital' value for the electron will just be the first guess in an iterative solution
"""
function init_electron_pdf_over_density_and_boundary_phi!(pdf, phi, density, upar, vth, z,
        vpa, vperp, vperp_spectral, vpa_spectral, vpa_advect, moments, num_diss_params,
        me_over_mi, scratch_dummy; restart_from_boltzmann=false)

    if z.bc == "wall"
        begin_r_region()
        @loop_r ir begin
            # Initialise an unshifted Maxwellian as a first step
            @loop_z iz begin
                vpa_over_vth = @. vpa.scratch3 = vpa.grid + upar[iz,ir] / vth[iz,ir]
                @loop_vperp ivperp begin
                    @. pdf[:,ivperp,iz,ir] = exp(-vpa_over_vth^2)
                end
            end
        end
        # Apply the sheath boundary condition to get cut-off boundary distribution
        # functions and boundary values of phi
        enforce_boundary_condition_on_electron_pdf!(pdf, phi, vth, upar, z, vperp, vpa,
                                                    vperp_spectral, vpa_spectral,
                                                    vpa_advect, moments,
                                                    num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                    me_over_mi)

        # Distribute the z-boundary pdf values to every process
        begin_serial_region()
        pdf_lower = scratch_dummy.buffer_vpavperprs_1
        pdf_upper = scratch_dummy.buffer_vpavperprs_2
        @serial_region begin
            if z.irank == 0
                pdf_lower .= pdf[:,:,1,:]
            end
            MPI.Bcast!(pdf_lower, z.comm; root=0)
            if z.irank == z.nrank - 1
                pdf_upper .= pdf[:,:,end,:]
            end
            MPI.Bcast!(pdf_upper, z.comm; root=z.nrank-1)
        end

        begin_r_z_region()
        @loop_r ir begin
            # get critical velocities beyond which electrons are lost to the wall
            #vpa_crit_zmin, vpa_crit_zmax = get_electron_critical_velocities(phi, vth, me_over_mi, z)
            #println("vpa_crit_zmin = ", vpa_crit_zmin, " vpa_crit_zmax = ", vpa_crit_zmax)
            # Blend boundary distribution function into bulk of domain to avoid
            # discontinuities (as much as possible)
            blend_fac = 100
            @loop_z_vperp iz ivperp begin
                #@. pdf[:,ivperp,iz] = exp(-30*z.grid[iz]^2)
                #@. pdf[:,ivperp,iz] = (density[iz] / vth[iz]) *
                #@. pdf[:,ivperp,iz] = exp(-vpa.grid[:]^2)
                blend_fac_lower = exp(-blend_fac*(z.grid[iz] + 0.5*z.L)^2)
                blend_fac_upper = exp(-blend_fac*(z.grid[iz] - 0.5*z.L)^2)
                @. pdf[:,ivperp,iz,ir] = (1.0 - blend_fac_lower) * (1.0 - blend_fac_upper) * pdf[:,ivperp,iz,ir] +
                                        blend_fac_lower * pdf_lower[:,ivperp,ir] +
                                        blend_fac_upper * pdf_upper[:,ivperp,ir]
                #@. pdf[:,ivperp,iz,ir] = exp(-vpa.grid^2) * (
                #                         (1 - exp(-blend_fac*(z.grid[iz] - z.grid[1])^2) *
                #                          tanh(sharp_fac*(vpa.grid-vpa_crit_zmin))) *
                #                         (1 - exp(-blend_fac*(z.grid[iz] - z.grid[end])^2) *
                #                          tanh(-sharp_fac*(vpa.grid-vpa_crit_zmax)))) #/
                                        #(1 - exp(-blend_fac*(z.grid[iz] - z.grid[1])^2) * tanh(-sharp_fac*vpa_crit_zmin)) /
                                        #(1 - exp(-blend_fac*(z.grid[iz] - z.grid[end])^2) * tanh(sharp_fac*vpa_crit_zmax)))
                                        #exp(-((vpa.grid[:] - upar[iz])^2) / vth[iz]^2)
                                        #exp(-((vpa.grid - upar[iz])^2 + vperp.grid[ivperp]^2) / vth[iz]^2)

                # ensure that the normalised electron pdf integrates to unity
                norm_factor = integrate_over_vspace(pdf[:,ivperp,iz,ir], vpa.wgts)
                @. pdf[:,ivperp,iz,ir] /= norm_factor
                #println("TMP FOR TESTING -- init electron pdf")
                #@. pdf[:,ivperp,iz] = exp(-2*vpa.grid[:]^2)*exp(-z.grid[iz]^2)
            end
        end
    else
        println("!!! currently, only the wall BC is supported for kinetic electrons !!!")
    end
end

function init_pdf_moments_manufactured_solns!(pdf, moments, vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species, geometry,composition)
    manufactured_solns_list = manufactured_solutions(r.L,z.L,r.bc,z.bc,geometry,composition,r.n)
    dfni_func = manufactured_solns_list.dfni_func
    densi_func = manufactured_solns_list.densi_func
    dfnn_func = manufactured_solns_list.dfnn_func
    densn_func = manufactured_solns_list.densn_func
    #nb manufactured functions not functions of species
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        moments.ion.dens[iz,ir,is] = densi_func(z.grid[iz],r.grid[ir],0.0)
        @loop_vperp_vpa ivperp ivpa begin
            pdf.ion.norm[ivpa,ivperp,iz,ir,is] = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],0.0)
        end
    end
    # update upar, ppar, qpar, vth consistent with manufactured solns
    update_density!(moments.ion.dens, pdf.ion.norm, vpa, vperp, z, r, composition)
    update_qpar!(moments.ion.qpar, pdf.ion.norm, vpa, vperp, z, r, composition)
    update_ppar!(moments.ion.ppar, pdf.ion.norm, vpa, vperp, z, r, composition)
    # get particle flux
    update_upar!(moments.ion.upar, pdf.ion.norm, vpa, vperp, z, r, composition)
    # convert from particle particle flux to parallel flow
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        moments.ion.upar[iz,ir,is] /= moments.ion.dens[iz,ir,is]
    # update the thermal speed
        moments.ion.vth[iz,ir,is] = sqrt(2.0*moments.ion.ppar[iz,ir,is]/moments.ion.dens[iz,ir,is])
    end

    if n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = densn_func(z.grid[iz],r.grid[ir],0.0)
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = dfnn_func(vz.grid[ivz],vr.grid[ivr],vzeta.grid[ivzeta],z.grid[iz],r.grid[ir],0.0)
            end
        end
        # get consistent moments with manufactured solutions
        update_neutral_density!(moments.neutral.dens, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_qz!(moments.neutral.qz, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_pz!(moments.neutral.pz, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_pr!(moments.neutral.pr, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_pzeta!(moments.neutral.pzeta, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        #update ptot (isotropic pressure)
        if r.n > 1 #if 2D geometry
            begin_sn_r_z_region()
            @loop_sn_r_z isn ir iz begin
                moments.neutral.ptot[iz,ir,isn] = (moments.neutral.pz[iz,ir,isn] + moments.neutral.pr[iz,ir,isn] + moments.neutral.pzeta[iz,ir,isn])/3.0
            end
        else #1D model
            moments.neutral.ptot .= moments.neutral.pz
        end
        # nb bad naming convention uz -> n uz below
        update_neutral_uz!(moments.neutral.uz, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_ur!(moments.neutral.ur, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_uzeta!(moments.neutral.uzeta, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        # now convert from particle particle flux to parallel flow
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.uz[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            moments.neutral.ur[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            moments.neutral.uzeta[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            # get vth for neutrals
            moments.ion.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.ptot[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
        end
    end
    return nothing
end

function init_knudsen_cosine!(knudsen_cosine, vz, vr, vzeta, vpa, vperp, composition, zero)

    begin_serial_region()
    @serial_region begin
        integrand = zeros(mk_float, vz.n, vr.n, vzeta.n)

        vtfac = sqrt(composition.T_wall * composition.mn_over_mi)

        if vzeta.n > 1 && vr.n > 1
            # 3V specification of neutral wall emission distribution for boundary condition
            if composition.use_test_neutral_wall_pdf
                # use test distribution that is easy for integration scheme to handle
                for ivzeta in 1:vzeta.n
                    for ivr in 1:vr.n
                        for ivz in 1:vz.n
                            v_transverse = sqrt(vzeta.grid[ivzeta]^2 + vr.grid[ivr]^2)
                            v_normal = abs(vz.grid[ivz])
                            knudsen_cosine[ivz,ivr,ivzeta] = (4.0/vtfac^5)*v_normal*exp( - (v_normal/vtfac)^2 - (v_transverse/vtfac)^2 )
                            integrand[ivz,ivr,ivzeta] = vz.grid[ivz]*knudsen_cosine[ivz,ivr,ivzeta]
                        end
                    end
                end
            else # get the true Knudsen cosine distribution for neutral particle wall emission
                for ivzeta in 1:vzeta.n
                    for ivr in 1:vr.n
                        for ivz in 1:vz.n
                            v_transverse = sqrt(vzeta.grid[ivzeta]^2 + vr.grid[ivr]^2)
                            v_normal = abs(vz.grid[ivz])
                            v_tot = sqrt(v_normal^2 + v_transverse^2)
                            if  v_tot > zero
                                prefac = v_normal/v_tot
                            else
                                prefac = 0.0
                            end
                            knudsen_cosine[ivz,ivr,ivzeta] = (3.0*sqrt(pi)/vtfac^4)*prefac*exp( - (v_normal/vtfac)^2 - (v_transverse/vtfac)^2 )
                            integrand[ivz,ivr,ivzeta] = vz.grid[ivz]*knudsen_cosine[ivz,ivr,ivzeta]
                        end
                    end
                end
            end
            normalisation = integrate_over_positive_vz(integrand, vz.grid, vz.wgts,
                                                       vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            # uncomment this line to test:
            #println("normalisation should be 1, it is = ", normalisation)
            #correct knudsen_cosine to conserve particle fluxes numerically
            @. knudsen_cosine /= normalisation

        elseif vzeta.n == 1 && vr.n == 1
            # get the marginalised Knudsen cosine distribution after integrating over vperp
            # appropriate for 1V model
            @. vz.scratch = (3.0*pi/vtfac^3)*abs(vz.grid)*erfc(abs(vz.grid)/vtfac)
            normalisation = integrate_over_positive_vz(vz.grid .* vz.scratch, vz.grid, vz.wgts, vz.scratch2,
                                                       vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            # uncomment this line to test:
            #println("normalisation should be 1, it is = ", normalisation)
            #correct knudsen_cosine to conserve particle fluxes numerically
            @. vz.scratch /= normalisation
            @. knudsen_cosine[:,1,1] = vz.scratch[:]

        end
    end
    return knudsen_cosine
end

function init_rboundary_pdfs!(rboundary_ion, rboundary_neutral, pdf::pdf_struct, vz,
                              vr, vzeta, vpa, vperp, z, r, composition)
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    n_neutral_species_alloc = max(1, n_neutral_species)

    begin_s_z_region() #do not parallelise r here
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        rboundary_ion[ivpa,ivperp,iz,1,is] = pdf.ion.norm[ivpa,ivperp,iz,1,is]
        rboundary_ion[ivpa,ivperp,iz,end,is] = pdf.ion.norm[ivpa,ivperp,iz,end,is]
    end
    if n_neutral_species > 0
        begin_sn_z_region() #do not parallelise r here
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            rboundary_neutral[ivz,ivr,ivzeta,iz,1,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,1,isn]
            rboundary_neutral[ivz,ivr,ivzeta,iz,end,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,end,isn]
        end
    end
    return rboundary_ion, rboundary_neutral
end

"""
Allocate arrays for distributions to be applied as boundary conditions to the pdf at
various boundaries. Also initialise the Knudsen cosine distribution here so it can be used
when initialising the neutral pdf.
"""
function create_boundary_distributions(vz, vr, vzeta, vpa, vperp, z, composition)
    zero = 1.0e-14

    #initialise knudsen distribution for neutral wall bc
    knudsen_cosine = allocate_shared_float(vz.n, vr.n, vzeta.n)
    #initialise knudsen distribution for neutral wall bc - can be done here as this only
    #depends on T_wall, which has already been set
    init_knudsen_cosine!(knudsen_cosine, vz, vr, vzeta, vpa, vperp, composition, zero)
    #initialise fixed-in-time radial boundary condition based on initial condition values
    pdf_rboundary_ion = allocate_shared_float(vpa.n, vperp.n, z.n, 2,
                                                  composition.n_ion_species)
    pdf_rboundary_neutral =  allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, 2,
                                                   composition.n_neutral_species)

    return boundary_distributions_struct(knudsen_cosine, pdf_rboundary_ion, pdf_rboundary_neutral)
end

function init_boundary_distributions!(boundary_distributions, pdf, vz, vr, vzeta, vpa, vperp, z, r, composition)
    #initialise fixed-in-time radial boundary condition based on initial condition values
    init_rboundary_pdfs!(boundary_distributions.pdf_rboundary_ion,
                         boundary_distributions.pdf_rboundary_neutral, pdf, vz, vr, vzeta,
                         vpa, vperp, z, r, composition)
    return nothing
end

"""
Take the full ion distribution function, calculate the moments, then
normalise and shift to the moment-kinetic grid.

Uses input value of `f` and modifies in place to the normalised distribution functions.
Input `density`, `upar`, `ppar`, and `vth` are not used, the values are overwritten with
the moments of `f`.

Inputs/outputs depend on z, vperp, and vpa (should be inside loops over species, r)
"""
function convert_full_f_ion_to_normalised!(f, density, upar, ppar, vth, vperp, vpa,
        vpa_spectral, evolve_density, evolve_upar, evolve_ppar)

    @loop_z iz begin
        # Calculate moments
        @views density[iz] = integrate_over_vspace(f[:,:,iz], vpa.grid, 0, vpa.wgts,
                                                   vperp.grid, 0, vperp.wgts)
        @views upar[iz] = integrate_over_vspace(f[:,:,iz], vpa.grid, 1, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts) /
                             density[iz]
        @views ppar[iz] = integrate_over_vspace(f[:,:,iz], vpa.grid, 2, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts) -
                             density[iz]*upar[iz]^2
        vth[iz] = sqrt(2.0*ppar[iz]/density[iz])

        # Normalise f
        if evolve_ppar
            f[:,:,iz] .*= vth[iz] / density[iz]
        elseif evolve_density
            f[:,:,iz] ./= density[iz]
        end

        # Interpolate f to moment kinetic grid
        if evolve_ppar || evolve_upar
            # The values to interpolate *to* are the v_parallel values corresponding to
            # the w_parallel grid
            vpa.scratch .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], evolve_ppar,
                                           evolve_upar)
            @loop_vperp ivperp begin
                @views vpa.scratch2 .= f[:,ivperp,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[:,ivperp,iz], vpa.scratch, vpa.scratch2,
                                               vpa, vpa_spectral)
            end
        end
    end

    return nothing
end

"""
Take the full neutral-particle distribution function, calculate the moments, then
normalise and shift to the moment-kinetic grid.

Uses input value of `f` and modifies in place to the normalised distribution functions.
Input `density`, `upar`, `ppar`, and `vth` are not used, the values are overwritten with
the moments of `f`.

Inputs/outputs depend on z, vzeta, vr and vz (should be inside loops over species, r)
"""
function convert_full_f_neutral_to_normalised!(f, density, uz, pz, vth, vzeta, vr, vz,
        vz_spectral, evolve_density, evolve_upar, evolve_ppar)

    @loop_z iz begin
        # Calculate moments
        @views density[iz] = integrate_over_neutral_vspace(f[:,:,:,iz], vz.grid, 0,
                                                           vz.wgts, vr.grid, 0, vr.wgts,
                                                           vzeta.grid, 0, vzeta.wgts)
        @views uz[iz] = integrate_over_neutral_vspace(f[:,:,:,iz], vz.grid, 1, vz.wgts,
                                                      vr.grid, 0, vr.wgts, vzeta.grid, 0,
                                                      vzeta.wgts) / density[iz]
        @views pz[iz] = integrate_over_neutral_vspace(f[:,:,:,iz], vz.grid, 2, vz.wgts,
                                                      vr.grid, 0, vr.wgts, vzeta.grid, 0,
                                                      vzeta.wgts) - density[iz]*uz[iz]^2
        vth[iz] = sqrt(2.0*pz[iz]/density[iz])

        # Normalise f
        if evolve_ppar
            f[:,:,:,iz] .*= vth[iz] / density[iz]
        elseif evolve_density
            f[:,:,:,iz] ./= density[iz]
        end

        # Interpolate f to moment kinetic grid
        if evolve_ppar || evolve_upar
            # The values to interpolate *to* are the v_parallel values corresponding to
            # the w_parallel grid
            vz.scratch .= vpagrid_to_dzdt(vz.grid, vth[iz], uz[iz], evolve_ppar,
                                          evolve_upar)
            @loop_vzeta_vr ivzeta ivr begin
                @views vz.scratch2 .= f[:,ivr,ivzeta,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[:,ivr,ivzeta,iz], vz.scratch, vz.scratch2,
                                               vz, vz_spectral)
            end
        end
    end

    return nothing
end

end
