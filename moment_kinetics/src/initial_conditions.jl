"""
"""
module initial_conditions

export allocate_pdf_and_moments
export init_pdf_and_moments!
export initialize_electrons!

# functional testing 
export create_pdf

# package
using Dates
using SpecialFunctions: erfc, erf
# modules
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..bgk: init_bgk_pdf!
using ..boundary_conditions: vpagrid_to_vpa
using ..calculus: integral
using ..communication
using ..external_sources
using ..interpolation: interpolate_to_grid_1d!
using ..looping
using ..electron_fluid_equations: calculate_electron_moments!
using ..electron_kinetic_equation: implicit_electron_advance!
using ..em_fields: update_phi!
using ..file_io: setup_electron_io, write_electron_state, finish_electron_io
using ..load_data: reload_electron_data!
using ..moment_constraints: hard_force_moment_constraints!
using ..moment_kinetics_structs: scratch_pdf, pdf_substruct, electron_pdf_substruct,
                                 pdf_struct, moments_struct
using ..nonlinear_solvers: nl_solver_info
using ..velocity_moments: integrate_over_positive_vz, integrate_over_negative_vz
using ..velocity_moments: create_moments_ion, create_moments_electron, create_moments_neutral
using ..velocity_moments: get_density, get_upar, get_p, get_neutral_density, get_neutral_uz, get_neutral_p
using ..velocity_moments: update_ion_qpar!
using ..velocity_moments: update_neutral_density!, update_neutral_pz!, update_neutral_pr!, update_neutral_pzeta!, update_neutral_p!
using ..velocity_moments: update_neutral_uz!, update_neutral_ur!, update_neutral_uzeta!, update_neutral_qz!
using ..velocity_moments: update_p!, update_ppar!, update_upar!, update_density!,
                          update_pperp!, update_vth!, reset_moments_status!
using ..electron_fluid_equations: calculate_electron_density!
using ..electron_fluid_equations: calculate_electron_upar_from_charge_conservation!
using ..electron_fluid_equations: calculate_electron_qpar!, electron_fluid_qpar_boundary_condition!
using ..electron_fluid_equations: calculate_electron_parallel_friction_force!
using ..electron_kinetic_equation: update_electron_pdf!, enforce_boundary_condition_on_electron_pdf!
using ..input_structs
using ..derivatives: derivative_z!
using ..utils: get_default_restart_filename, get_prefix_iblock_and_move_existing_file,
               get_backup_filename

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
    ion = create_moments_ion(z, r, composition, evolve_moments.density,
                             evolve_moments.parallel_flow, evolve_moments.pressure,
                             external_source_settings.ion, num_diss_params)
    electron = create_moments_electron(z, r, composition.electron_physics,
                                       num_diss_params,
                                       length(external_source_settings.electron))
    neutral = create_moments_neutral(z, r, composition, evolve_moments.density,
                                     evolve_moments.parallel_flow,
                                     evolve_moments.pressure,
                                     external_source_settings.neutral, num_diss_params)

    if abs(collisions.reactions.ionization_frequency) > 0.0 || z.bc == "wall"
        # if ionization collisions are included or wall BCs are enforced, then particle
        # number is not conserved within each species
        particle_number_conserved = false
    else
        # by default, assumption is that particle number should be conserved for each species
        particle_number_conserved = true
    end

    moments = moments_struct(ion, electron, neutral, evolve_moments.density,
                             particle_number_conserved,
                             evolve_moments.moments_conservation,
                             evolve_moments.parallel_flow,
                             evolve_moments.pressure)

    return pdf, moments
end

"""
Allocate arrays for pdfs
"""
function create_pdf(composition, r, z, vperp, vpa, vzeta, vr, vz)
    # allocate pdf arrays
    pdf_ion_norm = allocate_shared_float(vpa, vperp, z, r, composition.ion_species_coord)
    # buffer array is for ion-neutral collisions, not for storing ion pdf
    pdf_ion_buffer = allocate_shared_float(vpa, vperp, z, r, composition.neutral_species_coord) # n.b. n_species is n_neutral_species here
    pdf_neutral_norm = allocate_shared_float(vz, vr, vzeta, z, r, composition.neutral_species_coord)
    # buffer array is for neutral-ion collisions, not for storing neutral pdf
    pdf_neutral_buffer = allocate_shared_float(vz, vr, vzeta, z, r, composition.ion_species_coord)
    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        pdf_electron_norm = allocate_shared_float(vpa, vperp, z, r)
        # MB: not sure if pdf_electron_buffer will ever be needed, but create for now
        # to emulate ion and neutral behaviour
        pdf_electron_buffer = allocate_shared_float(vpa, vperp, z, r)
        pdf_before_ion_timestep = allocate_shared_float(vpa, vperp, z, r)
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
function init_pdf_and_moments!(pdf, moments, fields, geometry, composition, r, z, vperp,
                               vpa, vzeta, vr, vz, z_spectral, r_spectral, vperp_spectral,
                               vpa_spectral, vzeta_spectral, vr_spectral, vz_spectral,
                               r_bc, species, collisions, external_source_settings,
                               manufactured_solns_input, num_diss_params,
                               advection_structs, io_input, input_dict)
    if manufactured_solns_input.use_for_init
        init_pdf_moments_manufactured_solns!(pdf, moments, vz, vr, vzeta, vpa, vperp, z,
                                             r, r_bc, composition.n_ion_species,
                                             composition.n_neutral_species,
                                             geometry.input, composition, species,
                                             manufactured_solns_input, collisions)
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
            @. moments.ion.p = 0.5 * moments.ion.dens * moments.ion.vth^2
            # initialise pressures assuming isotropic distribution
            @. moments.ion.p = 0.5 * moments.ion.dens * moments.ion.vth^2
            if vperp.n == 1
                @. moments.ion.ppar = 3.0 * moments.ion.p
                @. moments.ion.pperp = 0.0
            else
                @. moments.ion.ppar = moments.ion.p
                @. moments.ion.pperp = moments.ion.p
            end
            if moments.evolve_density || moments.evolve_upar || moments.evolve_p
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
                @. moments.neutral.p = 0.5 * moments.neutral.dens * moments.neutral.vth^2
                if vperp.n == 1
                    @. moments.neutral.pz = 3.0 * moments.neutral.p
                    @. moments.neutral.pr = 0.0
                    @. moments.neutral.pzeta = 0.0
                else
                    @. moments.neutral.pz = moments.neutral.p
                    @. moments.neutral.pr = moments.neutral.p
                    @. moments.neutral.pzeta = moments.neutral.p
                end
                if moments.evolve_density || moments.evolve_upar || moments.evolve_p
                    @. moments.neutral.constraints_A_coefficient = 1.0
                    @. moments.neutral.constraints_B_coefficient = 0.0
                    @. moments.neutral.constraints_C_coefficient = 0.0
                end
            end
        end
        # reflect the fact that the ion moments have now been updated
        moments.ion.dens_updated .= true
        moments.ion.upar_updated .= true
        moments.ion.p_updated .= true
        # account for the fact that the neutral moments have now been updated
        moments.neutral.dens_updated .= true
        moments.neutral.uz_updated .= true
        moments.neutral.p_updated .= true
        # create and initialise the normalised, ion particle distribution function (pdf)
        # such that ∫dwpa pdf.norm = 1, ∫dwpa wpa * pdf.norm = 0, and ∫dwpa wpa^2 * pdf.norm = 1/2
        # note that wpa = vpa - upar, unless moments.evolve_p = true, in which case wpa = (vpa - upar)/vth
        # the definition of pdf.norm changes accordingly from pdf_unnorm / density to pdf_unnorm * vth / density
        # when evolve_p = true.
        initialize_pdf!(pdf, moments, composition, r, z, vperp, vpa, vzeta, vr, vz,
                        vperp_spectral, vpa_spectral, vzeta_spectral, vr_spectral,
                        vz_spectral, species)

        @begin_s_r_z_region()
        # calculate the initial parallel heat flux from the initial un-normalised pdf. Even if coll_krook fluid is being
        # advanced, initialised ion_qpar uses the pdf 
        update_ion_qpar!(moments.ion.qpar, moments.ion.qpar_updated,
                     moments.ion.dens, moments.ion.upar, moments.ion.vth, moments.ion.dT_dz,
                     pdf.ion.norm, vpa, vperp, z, r, composition, drift_kinetic_ions, collisions,
                     moments.evolve_density, moments.evolve_upar, moments.evolve_p)

        @begin_serial_region()
        @serial_region begin
            # If electrons are being used, they will be initialized properly later. Here
            # we only set the values to avoid false positives from the debug checks
            # (when @debug_track_initialized is active).
            moments.electron.dens .= 0.0
            moments.electron.upar .= 0.0
            moments.electron.p .= 0.0
            moments.electron.ppar .= 0.0
            moments.electron.pperp .= 0.0
            moments.electron.qpar .= 0.0
            moments.electron.temp .= 0.0
            moments.electron.constraints_A_coefficient .= 1.0
            moments.electron.constraints_B_coefficient .= 0.0
            moments.electron.constraints_C_coefficient .= 0.0
            if composition.electron_physics ∈ (kinetic_electrons,
                                               kinetic_electrons_with_temperature_equation)
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
                               moments.evolve_upar, moments.evolve_p)
            update_neutral_pz!(moments.neutral.pz, moments.neutral.pz_updated,
                               moments.neutral.dens, moments.neutral.uz,
                               moments.neutral.p, moments.neutral.vth, pdf.neutral.norm,
                               vz, vr, vzeta, z, r, composition, moments.evolve_density,
                               moments.evolve_upar, moments.evolve_p)
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
    end

    # Zero-initialise the dSdt diagnostic to avoid writing uninitialised values, as the
    # collision operator will not be calculated before the initial values are written to
    # file.
    @serial_region begin
        moments.ion.dSdt .= 0.0
    end

    return nothing
end

function initialize_electrons!(pdf, moments, fields, geometry, composition, r, z,
                               vperp, vpa, vzeta, vr, vz, z_spectral, r_spectral,
                               vperp_spectral, vpa_spectral, collisions, gyroavs,
                               external_source_settings, scratch_dummy, scratch,
                               scratch_electron, nl_solver_params, t_params, t_input_tuple,
                               num_diss_params, advection_structs, io_input, input_dict;
                               restart_electron_physics, skip_electron_solve=false)
    
    moments.electron.dens_updated[] = false
    # initialise the electron density profile
    init_electron_density!(moments.electron.dens, moments.electron.dens_updated, moments.ion.dens)
    # initialise the electron parallel flow profile
    moments.electron.upar_updated[] = false
    init_electron_upar!(moments.electron.upar, moments.electron.upar_updated, moments.electron.dens, 
        moments.ion.upar, moments.ion.dens, composition.electron_physics, r, z)
    # different choices for initialization of electron temperature/pressure/vth depending on whether
    # we are restarting from a previous simulation with Boltzmann electrons or not
    if restart_electron_physics === nothing
        # Not restarting, so create initial profiles

        # initialise the electron thermal speed profile
        init_electron_vth!(moments.electron.vth, moments.ion.vth, composition, z.grid)
        @begin_r_z_region()
        # calculate the electron temperature from the thermal speed
        @loop_r_z ir iz begin
            moments.electron.temp[iz,ir] = 0.5 * composition.me_over_mi * moments.electron.vth[iz,ir]^2
        end
        # calculate the electron parallel pressure from the density and temperature
        @loop_r_z ir iz begin
            moments.electron.p[iz,ir] = moments.electron.dens[iz,ir] * moments.electron.temp[iz,ir]
        end
        if vperp.n == 1
            @loop_r_z ir iz begin
                moments.electron.ppar[iz,ir] = 3.0 * moments.electron.p[iz,ir]
                moments.electron.pperp[iz,ir] = 0.0
            end
        else
            @loop_r_z ir iz begin
                moments.electron.ppar[iz,ir] = moments.electron.p[iz,ir]
                moments.electron.pperp[iz,ir] = moments.electron.p[iz,ir]
            end
        end
    elseif restart_electron_physics ∉ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        # Restarting from Boltzmann electron simulation, so start with constant
        # electron temperature
        @begin_serial_region()
        @serial_region begin
            # if restarting from a simulations where Boltzmann electrons were used, then the assumption is
            # that the electron parallel temperature is constant along the field line and equal to T_e
            if vperp.n == 1
                moments.electron.temp .= composition.T_e / 3.0
                # the thermal speed is related to the temperature by vth_e / c_ref = sqrt(2.0 * (T_e/T_ref) / (m_e/m_ref))
                moments.electron.vth .= sqrt(2.0 / 3.0 * composition.T_e / composition.me_over_mi)
                # p = n * T, so we can calculate the pressure from the density and T_e
                moments.electron.p .= moments.electron.dens * composition.T_e / 3.0
                moments.electron.ppar .= 3.0 .* moments.electron.p
                moments.electron.pperp .= 0.0
            else
                moments.electron.temp .= composition.T_e
                # the thermal speed is related to the temperature by vth_e / c_ref = sqrt(2.0 * (T_e/T_ref) / (m_e/m_ref))
                moments.electron.vth .= sqrt(2.0 * composition.T_e / composition.me_over_mi)
                # p = n * T, so we can calculate the pressure from the density and T_e
                moments.electron.p .= moments.electron.dens * composition.T_e
                moments.electron.ppar .= moments.electron.p
                moments.electron.pperp .= moments.electron.p
            end
        end
    end # else, we are restarting from `braginskii_fluid` or `kinetic_electrons`, so keep the reloaded electron pressure/temperature profiles.

    # the electron temperature has now been updated
    moments.electron.temp_updated[] = true
    # the electron parallel pressure now been updated
    moments.electron.p_updated[] = true

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
    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        # Initialise the array for the electron pdf
        @begin_serial_region()
        speed = @view scratch_dummy.buffer_vpavperpzrs_1[:,:,:,:,1]
        @serial_region begin
            speed .= 0.0
        end
        init_electron_pdf_over_density_and_boundary_phi!(
            pdf.electron.norm, fields.phi, moments.electron.dens, moments.electron.upar,
            moments.electron.vth, r, z, vpa, vperp, vperp_spectral, vpa_spectral,
            [(speed=speed,)], moments, num_diss_params,
            composition.me_over_mi, scratch_dummy)
    end
    # calculate the initial electron parallel heat flux;
    # if using kinetic electrons, this relies on the electron pdf, which itself relies on the electron heat flux
    if composition.electron_physics == braginskii_fluid
        electron_fluid_qpar_boundary_condition!(
            moments.electron.ppar, moments.electron.upar, moments.electron.dens,
            moments.electron, z)
        if restart_electron_physics ∉ (nothing, braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
            # Restarting from Boltzmann. If we use an exactly constant T_e profile,
            # qpar for the electrons will be non-zero only at the boundary points,
            # which will crash the code unless the timestep is insanely small. Give
            # T_e a cubic profile with gradients at the boundaries that match the
            # boundary qpar. The boundary values are both the constant 'Boltzmann
            # electron' temperature
            @begin_serial_region()
            @serial_region begin
                # Ignore the 0.71*p_e*(u_i-u_e) term here as it would vanish when u_i=u_e
                # and this is only a rough initial condition anyway
                #
                # q at the boundaries tells us dTe/dz for Braginskii electrons
                nu_ei = collisions.electron_fluid.nu_ei
                dTe_dz_lower = Ref{mk_float}(0.0)
                if z.irank == 0
                    dTe_dz_lower[] = @. -moments.electron.qpar[1,:] / 3.16 /
                                         moments.electron.ppar[1,:] *
                                         composition.me_over_mi * nu_ei
                end
                MPI.Bcast!(dTe_dz_lower, z.comm; root=0)

                dTe_dz_upper = Ref{mk_float}(0.0)
                if z.irank == z.nrank - 1
                    dTe_dz_upper[] = @. -moments.electron.qpar[end,:] / 3.16 /
                                         moments.electron.ppar[end,:] *
                                         composition.me_over_mi * nu_ei
                end
                MPI.Bcast!(dTe_dz_upper, z.comm; root=(z.nrank - 1))

                # The temperature should already be equal to the 'Boltzmann electron'
                # Te, so we just need to add a cubic that vanishes at ±Lz/2
                # δT = A + B*z + C*z^2 + D*z^3
                # ⇒ A - B*Lz/2 + C*Lz^2/4 - D*Lz^3/8 = 0
                #   A + B*Lz/2 + C*Lz^2/4 + D*Lz^3/8 = 0
                #   B - C*Lz + 3*D*Lz^2/4 = dTe/dz_lower
                #   B + C*Lz + 3*D*Lz^2/4 = dTe/dz_upper
                #
                # Adding the first pair together, and subtracting the second pair:
                #   A + C*Lz^2/4 = 0
                #   2*C*Lz = dT/dz_upper - dT/dz_lower
                #
                # Subtracting the first pair and adding the second instead:
                #   B*Lz/2 + D*Lz^3/8 = 0  ⇒  D*Lz^2/2 = -2*B
                #   2*B + 3*D*Lz^2/2 = dTe/dz_upper - dTe/dz_lower
                #
                #   2*B - 3*2*B = -4*B = dTe/dz_upper + dTe/dz_lower
                Lz = z.L
                zg = z.grid
                C = @. (dTe_dz_upper[] - dTe_dz_lower[]) / 2.0 / Lz
                A = @. -C * Lz^2 / 4
                B = @. -(dTe_dz_lower[] + dTe_dz_upper[]) / 4.0
                D = @. -4.0 * B / Lz^2
                @loop_r ir begin
                    @. moments.electron.temp[:,ir] += A[ir] + B[ir]*zg + C[ir]*zg^2 +
                                                      D[ir]*zg^3
                end

                @. moments.electron.vth = sqrt(2.0 * moments.electron.temp /
                                               composition.me_over_mi)
                @. moments.electron.ppar = moments.electron.dens * moments.electron.temp
            end
            @views derivative_z!(moments.electron.dT_dz, moments.electron.temp,
                                 scratch_dummy.buffer_rs_1[:,1],
                                 scratch_dummy.buffer_rs_2[:,1],
                                 scratch_dummy.buffer_rs_3[:,1],
                                 scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
        end
    end
    moments.electron.qpar_updated[] = false
    calculate_electron_qpar!(moments.electron, pdf.electron, moments.electron.p,
        moments.electron.dens, moments.electron.upar, moments.ion.upar,
        collisions.electron_fluid.nu_ei, composition.me_over_mi,
        composition.electron_physics, vperp, vpa)
    if composition.electron_physics == braginskii_fluid
        electron_fluid_qpar_boundary_condition!(
            moments.electron.ppar, moments.electron.upar, moments.electron.dens,
            moments.electron, z)
    end
    # calculate the zed derivative of the initial electron parallel heat flux
    @views derivative_z!(moments.electron.dqpar_dz, moments.electron.qpar, 
        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
    # calculate the electron-ion parallel friction force
    calculate_electron_parallel_friction_force!(moments.electron.parallel_friction, moments.electron.dens,
        moments.electron.upar, moments.ion.upar, moments.electron.dT_dz,
        composition.me_over_mi, collisions.electron_fluid.nu_ei,
        composition.electron_physics)
    
    # initialize the scratch arrays containing pdfs and moments for the first RK stage
    # the electron pdf is yet to be initialised but with the current code logic, the scratch
    # arrays need to exist and be otherwise initialised in order to compute the initial
    # electron pdf. The electron arrays will be updated as necessary by
    # initialize_electron_pdf!().
    @begin_serial_region()
    @serial_region begin
        scratch[1].electron_density .= moments.electron.dens
        scratch[1].electron_upar .= moments.electron.upar
        scratch[1].electron_p .= moments.electron.p
        n_rk_stages = t_params.n_rk_stages
        scratch[n_rk_stages+1].electron_density .= moments.electron.dens
        scratch[n_rk_stages+1].electron_upar .= moments.electron.upar
        scratch[n_rk_stages+1].electron_p .= moments.electron.p
    end
    if scratch_electron !== nothing
        @begin_serial_region()
        @serial_region begin
            scratch_electron[1].electron_p .= moments.electron.p
        end
    end

    # initialize the electron pdf that satisfies the electron kinetic equation
    initialize_electron_pdf!(scratch, scratch_electron, pdf, moments, fields, r, z, vpa,
                             vperp, vzeta, vr, vz, r_spectral, z_spectral, vperp_spectral,
                             vpa_spectral, advection_structs.electron_z_advect,
                             advection_structs.electron_vpa_advect, scratch_dummy,
                             collisions, composition, geometry, external_source_settings,
                             num_diss_params, gyroavs, nl_solver_params, t_params,
                             t_input_tuple.electron_t_input, io_input, input_dict;
                             skip_electron_solve=skip_electron_solve)

    return nothing
end

"""
"""
function initialize_pdf!(pdf, moments, composition, r, z, vperp, vpa, vzeta, vr, vz,
                         vperp_spectral, vpa_spectral, vzeta_spectral, vr_spectral,
                         vz_spectral, species)
    wall_flux_0 = allocate_float(r, composition.ion_species_coord)
    wall_flux_L = allocate_float(r, composition.ion_species_coord)

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
                    z, vperp_spectral, vpa_spectral, moments.ion.dens[:,ir,is],
                    moments.ion.upar[:,ir,is], moments.ion.p[:,ir,is],
                    moments.ion.ppar[:,ir,is], moments.ion.vth[:,ir,is],
                    moments.ion.v_norm_fac[:,ir,is], moments.evolve_density,
                    moments.evolve_upar, moments.evolve_p)
            end
            @views wall_flux_0[ir,is] = -(moments.ion.dens[1,ir,is] *
                                          moments.ion.upar[1,ir,is])
            @views wall_flux_L[ir,is] = moments.ion.dens[end,ir,is] *
                                        moments.ion.upar[end,ir,is]

            @loop_z iz begin
                if moments.evolve_density == false
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
                pdf.neutral.norm[:,:,:,:,ir,isn], species.neutral[isn], composition, vz,
                vr, vzeta, z, vzeta_spectral, vr_spectral, vz_spectral,
                moments.neutral.dens[:,ir,isn], moments.neutral.uz[:,ir,isn],
                moments.neutral.p[:,ir,isn], moments.neutral.vth[:,ir,isn],
                moments.neutral.v_norm_fac[:,ir,isn], moments.evolve_density,
                moments.evolve_upar, moments.evolve_p,
                wall_flux_0[ir,min(isn,composition.n_ion_species)],
                wall_flux_L[ir,min(isn,composition.n_ion_species)])
            @loop_z iz begin
                if moments.evolve_density == false
                    @. pdf.neutral.norm[:,:,:,iz,ir,isn] *= moments.neutral.dens[iz,ir,isn]
                end
            end
        end
    end

    return nothing
end

function initialize_electron_pdf!(scratch, scratch_electron, pdf, moments, fields, r, z,
                                  vpa, vperp, vzeta, vr, vz, r_spectral, z_spectral,
                                  vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                                  scratch_dummy, collisions, composition, geometry,
                                  external_source_settings, num_diss_params, gyroavs,
                                  nl_solver_params, t_params, t_input_tuple, io_input,
                                  input_dict; skip_electron_solve)

    if t_input_tuple.skip_electron_initial_solve
        skip_electron_solve = true
    end

    if composition.electron_physics ∈ (kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        @begin_serial_region()
        if t_input_tuple.no_restart
            restart_filename = nothing
        else
            restart_filename = get_default_restart_filename(io_input, "initial_electron";
                                                            error_if_no_file_found=false)
            # Synchronize to ensure that some processes do not detect the restart file
            # when they should not, because this function gets called after the other
            # processes create the file.
            MPI.Barrier(comm_world)
        end
        if restart_filename === nothing
            # No file to restart from
            previous_runs_info = ()
            code_time = fill(mk_float(0.0), r.n)
            restart_time_index = -1
            pdf_electron_converged = false
        else
            # Previously-created electron distribution function exists, so use it as
            # the initial guess.
            backup_prefix_iblock, initial_electrons_filename,
            backup_initial_electrons_filename =
                get_prefix_iblock_and_move_existing_file(restart_filename,
                                                         io_input.output_dir)

            # Reload pdf and moments from an existing output file
            code_time, pdf_electron_converged, previous_runs_info, restart_time_index =
                reload_electron_data!(pdf, moments, fields.phi, t_params.electron,
                                      backup_prefix_iblock, -1, geometry, r, z, vpa,
                                      vperp, vzeta, vr, vz)

            if pdf_electron_converged
                if global_rank[] == 0
                    println("Reading initial electron state from $restart_filename")
                end
                # Move the *.initial_electron.h5 file back to its original location, as we
                # do not need to create a new output file.
                MPI.Barrier(comm_world)
                if global_rank[] == 0
                    if initial_electrons_filename != backup_initial_electrons_filename
                        mv(backup_initial_electrons_filename, initial_electrons_filename)
                    end
                end
                MPI.Barrier(comm_world)
            else
                if global_rank[] == 0
                    println("Restarting electron initialisation from $restart_filename")
                end
            end
        end

        @begin_serial_region()
        @serial_region begin
            # update the electron pdf in the last scratch_electron (which will be copied
            # to the first entry as part of the pseudo-time-loop in
            # update_electron_pdf!()).
            scratch_electron[t_params.electron.n_rk_stages+1].pdf_electron .= pdf.electron.norm
        end

        @begin_r_z_region()
        @loop_r_z ir iz begin
            # update the electron thermal speed using the updated electron parallel pressure
            moments.electron.vth[iz,ir] = sqrt(abs(2.0 * moments.electron.p[iz,ir] / (moments.electron.dens[iz,ir] * composition.me_over_mi)))
        end

        moments.electron.qpar_updated[] = false
        calculate_electron_qpar!(moments.electron, pdf.electron, moments.electron.p,
                                 moments.electron.dens, moments.electron.upar,
                                 moments.ion.upar, collisions.electron_fluid.nu_ei,
                                 composition.me_over_mi, composition.electron_physics,
                                 vperp, vpa)
        # update dqpar/dz for electrons
        # calculate the zed derivative of the initial electron parallel heat flux
        @views derivative_z!(moments.electron.dqpar_dz, moments.electron.qpar, 
            scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
            scratch_dummy.buffer_rs_4[:,1], z_spectral, z)

        # now that we have our initial guess for the electron pdf, we iterate
        # using the time-independent electron kinetic equation to find a self-consistent
        # solution for the electron pdf.
        # First run with evolve_p=true to get electron_ppar close to steady state.
        # electron_ppar does not have to be exactly steady state as it will be
        # time-evolved along with the ions.
        #max_electron_pdf_iterations = 2000000
        ##max_electron_pdf_iterations = 500000
        ##max_electron_pdf_iterations = 10000
        #max_electron_sim_time = nothing
        max_electron_pdf_iterations = nothing
        max_electron_sim_time = max(2.0, t_params.electron.max_pseudotime)
        if t_params.electron.debug_io !== nothing
            @begin_serial_region
            @serial_region begin
                for ir ∈ 1:r.n
                    setup_electron_io(t_params.electron.debug_io[1], vpa, vperp, z, r,
                                      composition, collisions, moments.evolve_density,
                                      moments.evolve_upar, moments.evolve_p,
                                      external_source_settings, t_params.electron,
                                      t_params.electron.debug_io[2], -1, (),
                                      "electron_debug"; ir=ir)
                end
            end
        end
        if !pdf_electron_converged 
            if global_rank[] == 0
                println("Initializing electrons - evolving both pdf_electron and electron_p")
            end
            # Setup I/O for initial electron state
            io_initial_electron = setup_electron_io(io_input, vpa, vperp, z, r,
                                                    composition, collisions,
                                                    moments.evolve_density,
                                                    moments.evolve_upar,
                                                    moments.evolve_p,
                                                    external_source_settings,
                                                    t_params.electron, input_dict,
                                                    restart_time_index,
                                                    previous_runs_info,
                                                    "initial_electron")

            if !skip_electron_solve
                # Can't let this counter stay set to 0
                t_params.electron.dfns_output_counter[] = max(t_params.electron.dfns_output_counter[], 1)
                implicit_electron_pseudotimestep = (nl_solver_params.electron_advance !== nothing)
                electron_solution_method = Val(implicit_electron_pseudotimestep ? :backward_euler : :artificial_time_derivative)
                success =
                    @views update_electron_pdf!(scratch_electron, pdf.electron.norm,
                                                moments, fields.phi, r, z, vperp, vpa,
                                                z_spectral, vperp_spectral, vpa_spectral,
                                                z_advect, vpa_advect, scratch_dummy,
                                                t_params.electron, collisions,
                                                composition, external_source_settings,
                                                num_diss_params,
                                                nl_solver_params.electron_advance,
                                                max_electron_pdf_iterations,
                                                max_electron_sim_time;
                                                initial_time=code_time,
                                                residual_tolerance=t_input_tuple.initialization_residual_value,
                                                evolve_p=true,
                                                solution_method=electron_solution_method)
                if success != ""
                    error("!!!max number of iterations for electron pdf update exceeded!!!\n"
                          * "Stopping at $(Dates.format(now(), dateformat"H:MM:SS"))")
                end
            end

            # Now run without evolve_p=true to get pdf_electron fully to steady state,
            # ready for the start of the ion time advance.
            if global_rank[] == 0
                println("Initializing electrons - evolving pdf_electron only to steady state")
            end
            if skip_electron_solve
                success = ""
            elseif t_params.kinetic_electron_solver == implicit_steady_state
                # Create new nl_solver_info ojbect with higher maximum iterations for
                # initialisation.
                initialisation_nl_solver_params =
                    nl_solver_info(nl_solver_params.electron_advance.rtol,
                                   nl_solver_params.electron_advance.atol,
                                   100000,
                                   nl_solver_params.electron_advance.linear_rtol,
                                   nl_solver_params.electron_advance.linear_atol,
                                   nl_solver_params.electron_advance.linear_restart,
                                   nl_solver_params.electron_advance.linear_max_restarts,
                                   nl_solver_params.electron_advance.H,
                                   nl_solver_params.electron_advance.c,
                                   nl_solver_params.electron_advance.s,
                                   nl_solver_params.electron_advance.g,
                                   nl_solver_params.electron_advance.V,
                                   nl_solver_params.electron_advance.linear_initial_guess,
                                   nl_solver_params.electron_advance.n_solves,
                                   nl_solver_params.electron_advance.nonlinear_iterations,
                                   nl_solver_params.electron_advance.linear_iterations,
                                   nl_solver_params.electron_advance.precon_iterations,
                                   nl_solver_params.electron_advance.global_n_solves,
                                   nl_solver_params.electron_advance.global_nonlinear_iterations,
                                   nl_solver_params.electron_advance.global_linear_iterations,
                                   nl_solver_params.electron_advance.global_precon_iterations,
                                   nl_solver_params.electron_advance.solves_since_precon_update,
                                   nl_solver_params.electron_advance.precon_dt,
                                   nl_solver_params.electron_advance.precon_lowerz_vcut_inds,
                                   nl_solver_params.electron_advance.precon_upperz_vcut_inds,
                                   nl_solver_params.electron_advance.serial_solve,
                                   nl_solver_params.electron_advance.max_nonlinear_iterations_this_step,
                                   nl_solver_params.electron_advance.max_linear_iterations_this_step,
                                   nl_solver_params.electron_advance.total_its_soft_limit,
                                   nl_solver_params.electron_advance.preconditioner_type,
                                   nl_solver_params.electron_advance.preconditioner_update_interval,
                                   nl_solver_params.electron_advance.preconditioners,
                                  )
                # Run implicit solve with dt=0 so that we don't update electron_ppar here
                success =
                    implicit_electron_advance!(scratch[t_params.n_rk_stages+1],
                                               scratch[1], pdf,
                                               scratch_electron[t_params.electron.n_rk_stages+1],
                                               moments, fields, collisions, composition,
                                               geometry, external_source_settings,
                                               num_diss_params, r, z, vperp, vpa,
                                               r_spectral, z_spectral, vperp_spectral,
                                               vpa_spectral, z_advect, vpa_advect,
                                               gyroavs, scratch_dummy, t_params.electron,
                                               0.0, initialisation_nl_solver_params)
            else
                success =
                    update_electron_pdf!(scratch_electron, pdf.electron.norm, moments,
                                         fields.phi, r, z, vperp, vpa, z_spectral,
                                         vperp_spectral, vpa_spectral, z_advect,
                                         vpa_advect, scratch_dummy, t_params.electron,
                                         collisions, composition,
                                         external_source_settings, num_diss_params,
                                         nl_solver_params.electron_advance,
                                         max_electron_pdf_iterations,
                                         max_electron_sim_time;
                                         evolve_p=true, ion_dt=t_params.dt[],
                                         solution_method=electron_solution_method)
            end
            if success != ""
                error("!!!max number of iterations for electron pdf update exceeded!!!\n"
                      * "Stopping at $(Dates.format(now(), dateformat"H:MM:SS"))")
            end

            # Write the converged initial state for the electrons to a file so that it can be
            # re-used if the simulation is re-run.
            write_electron_state(scratch_electron, moments, fields.phi, t_params.electron,
                                 io_initial_electron, 2, -1.0, 0.0, r, z, vperp, vpa;
                                 pdf_electron_converged=true)
            finish_electron_io(io_initial_electron)
        end

        @begin_r_z_vperp_vpa_region()
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            pdf.electron.pdf_before_ion_timestep[ivpa,ivperp,iz,ir] =
                pdf.electron.norm[ivpa,ivperp,iz,ir]
        end
        if length(scratch[1].pdf_electron) > 0
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                for i ∈ 1:length(scratch)
                    scratch[i].pdf_electron[ivpa,ivperp,iz,ir] =
                        pdf.electron.norm[ivpa,ivperp,iz,ir]
                end
            end
        end
        @begin_r_z_region()
        @loop_r_z ir iz begin
            for i ∈ 1:length(scratch)
                scratch[i].electron_p[iz,ir] = moments.electron.p[iz,ir]
            end
        end
        calculate_electron_moments!(scratch[1], pdf, moments, composition, collisions, r,
                                    z, vperp, vpa)

        # No need to do electron I/O (apart from possibly debug I/O) any more, so if
        # adaptive timestep is used, it does not need to adjust to output times.
        resize!(t_params.electron.moments_output_times, 0)
        resize!(t_params.electron.dfns_output_times, 0)
        t_params.electron.moments_output_counter[] = 1
        t_params.electron.dfns_output_counter[] = 1

    end
    return nothing
end

"""
for now the only initialisation option for the temperature is constant in z
returns vth0 = sqrt(2Ts/ms) / sqrt(T_ref/m_ref) = sqrt(2Ts/T_ref)
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
    @. vth = sqrt(2.0 * vth)
    return nothing
end

"""
"""
function init_density!(dens, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            if spec[is].z_IC.initialization_option == "gaussian"
                # initial condition is an unshifted Gaussian
                @. dens[:,ir,is] = spec[is].initial_density + exp(-(z.grid/spec[is].z_IC.width)^2)
            elseif spec[is].z_IC.initialization_option == "sinusoid"
                # initial condition is sinusoid in z
                @. dens[:,ir,is] =
                    (spec[is].initial_density
                     * (1.0 + spec[is].z_IC.density_amplitude
                              * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                                    + spec[is].z_IC.density_phase)))
            elseif spec[is].z_IC.initialization_option == "sinusoid_sum"
                # initial condition is sum of sinusoids in z
                @. dens[:,ir,is] = 
                    (spec[is].initial_density
                      * (1.0 + spec[is].z_IC.density_amplitude
                              * (cos(2 * 2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                                    + spec[is].z_IC.density_phase) + 
                                 cos(4 * 2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                                    + spec[is].z_IC.density_phase) + 
                                 cos(3 * 2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                                    + spec[is].z_IC.density_phase)
                                )
                        )
                    )
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
For Boltzmann electrons returns vth0 = sqrt(2*Ts/T_ref/me_over_mi)
For Braginskii or kinetic electrons, sets T_e=T_i, so returns vth_i/sqrt(me_over_mi).
"""
function init_electron_vth!(vth_e, vth_i, composition, z)
    @begin_r_z_region()
    if composition.electron_physics ∈ (boltzmann_electron_response,
                                       boltzmann_electron_response_with_simple_sheath)
        @loop_r_z ir iz begin
            vth_e[iz,ir] = sqrt(2.0 * composition.T_e / composition.me_over_mi)
        end
    else
        @loop_r_z ir iz begin
            vth_e[iz,ir] = vth_i[iz,ir,1] / sqrt(composition.me_over_mi)
        end
    end
end

"""
"""
function init_ion_pdf_over_density!(pdf, spec, composition, vpa, vperp, z,
        vperp_spectral, vpa_spectral, density, upar, p, ppar, vth, v_norm_fac,
        evolve_density, evolve_upar, evolve_p)

    # Prefactor for Maxwellian distribution functions
    if vperp.n == 1
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        Maxwellian_prefactor = 1.0 / π^1.5
    end
    if spec.vpa_IC.initialization_option == "gaussian"
        # initial condition is a Gaussian in the peculiar velocity
        if z.bc != "wall"
            for iz ∈ 1:z.n
                # obtain (vpa - upar)/vth
                if evolve_p
                    # if evolve_upar = true and evolve_p = true, then vpa coordinate is (vpa-upar)/vth;
                    if evolve_upar
                        @. vpa.scratch = vpa.grid
                        # if evolve_upar = false and evolve_p = true, then vpa coordinate is vpa/vth;
                    else
                        @. vpa.scratch = vpa.grid - upar[iz]/vth[iz]
                    end
                    # if evolve_upar = true and evolve_p = false, then vpa coordinate is vpa-upar;
                elseif evolve_upar
                    @. vpa.scratch = vpa.grid/vth[iz]
                    # if evolve_upar = false and evolve_p = false, then vpa coordinate is vpa;
                else
                    @. vpa.scratch = (vpa.grid - upar[iz])/vth[iz]
                end

                @. vperp.scratch = vperp.grid/vth[iz]

                if vperp.n == 1
                    # Need to initialise using Maxwellian defined using T_∥ = 3*T as T_⟂=0
                    vth_factor = sqrt(3.0)
                    if !evolve_p
                        vth_factor *= vth[iz]
                    end
                    vpa.scratch ./= sqrt(3.0)
                else
                    if !evolve_p
                        vth_factor = vth[iz]^3
                    else
                        vth_factor = 1.0
                    end
                end
                @loop_vperp_vpa ivperp ivpa begin
                    pdf[ivpa,ivperp,iz] = Maxwellian_prefactor * exp(-vpa.scratch[ivpa]^2 -
                                                                     vperp.scratch[ivperp]^2) / vth_factor
                end
            end

            # Only do this correction for runs without wall bc, because consistency of
            # pdf and moments is taken care of by convert_full_f_ion_to_normalised!()
            # for wall bc cases.
            for iz ∈ 1:z.n
                # densfac = the integral of the pdf over v-space, which should be unity,
                # but may not be exactly unity due to quadrature errors
                densfac = integral(view(pdf,:,:,iz), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
                # Save w_s = vpa - upar_s in vpa.scratch
                if evolve_upar
                    vpa.scratch .= vpa.grid
                else
                    @. vpa.scratch = vpa.grid - upar[iz]
                end
                # pfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
                # where w_s^2 = (vpa - upar_s)^2 + vperp^2;
                # In moment-kinetic case, the velocity grids are already scaled by vths -
                # v_norm_fac takes care of this (it is 1 when velocity grids are not
                # normalised by vths, or vths when velocity grids are normalised by vths).
                # pfac should be equal to 3/2, but may not be exactly 3/2 due to quadrature errors
                pfac = @views (v_norm_fac[iz]/vth[iz])^2 *
                              (integral(pdf[:,:,iz], vpa.scratch, 2, vpa.wgts, vperp.grid,
                                        0, vperp.wgts)
                               + integral(pdf[:,:,iz], vpa.scratch, 0, vpa.wgts, vperp.grid,
                                          2, vperp.wgts))
                # pfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
                if evolve_upar
                    upar_offset = 0.0
                else
                    upar_offset = upar[iz]
                end
                pfac2 = @views (v_norm_fac[iz]/vth[iz])^2 *
                               integral((vperp,vpa)->(((vpa - upar_offset)^2 + vperp^2) * (((vpa - upar_offset)^2 + vperp^2) * (v_norm_fac[iz] / vth[iz])^2 / pfac - 1.0/densfac)),
                                        pdf[:,:,iz], vperp, vpa)

                # The following update ensures the density and pressure moments of pdf
                # have the expected values. The velocity moment is always exactly zero
                # from symmetry, so does not need correcting.
                # The corrected version has the correct moments because
                #   ∫d^3v pdf_before = densfac
                #   ∫d^3v m_s w_s^2 / vths^2 * pdf_before = pfac
                #   ∫d^3v m_s w_s^2 / vths^2 (m_s*w_s^2/vths^2/pfac - 1/densfac) pdf_before = pfac2
                # so if
                #   pdf = ( 1/densfac + (1.5 - pfac/densfac)/pfac2 * (m_s*w_s^2/vths^2/pfac - 1/densfac) ) * pdf_before
                # then
                #   ∫d^3v ( 1/densfac + (1.5 - pfac/densfac)/pfac2 * (m_s*w_s^2/vths^2/pfac - 1/densfac) ) * pdf_before
                #   = 1 + (1.5 - pfac / densfac) / pfac2 * (pfac/pfac - densfac/densfac)
                #   = 1
                # and
                #   ∫d^3v m_s w_s^2 / vths^2 * ( 1/densfac + (1.5 - pfac/densfac)/pfac2 * (m_s*w_s^2/vths^2/pfac - 1/densfac) ) * pdf_before
                #   = pfac/densfac + (1.5 - pfac/densfac)/pfac2 * pfac2
                #   = 1.5
                @loop_vperp ivperp begin
                    @views @. pdf[:,ivperp,iz] = pdf[:,ivperp,iz]/densfac +
                                                 (1.5 - pfac/densfac)/pfac2 *
                                                 ((vperp.grid[ivperp]^2 + vpa.scratch^2)*(v_norm_fac[iz]/vth[iz])^2/pfac - 1.0/densfac) *
                                                 pdf[:,ivperp,iz]
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
            if evolve_p
                # Scale the velocity grid used for initialization in case the
                # temperature changes a lot.
                vgrid_scale_factor = copy(vth)
            else
                vgrid_scale_factor = ones(size(vth))
            end
            for iz ∈ zrange

                @. vpa.scratch = vpa.grid * vgrid_scale_factor[iz]
                @. vperp.scratch = vperp.grid * vgrid_scale_factor[iz]

                @loop_vperp ivperp begin
                    # Initialise as full-f distribution functions, then
                    # normalise/interpolate (if necessary). This makes it easier to
                    # initialise a normalised pdf consistent with the moments, although it
                    # modifies the moments from the 'input' values.
                    if vperp.n == 1
                        # Need to initialise using Maxwellian defined using T_∥ = 3*T as T_⟂=0
                        this_vth = sqrt(3.0) * vth[iz]
                        vth_factor = this_vth
                    else
                        this_vth = vth[iz]
                        vth_factor = vth[iz]^3
                    end
                    @. pdf[:,ivperp,iz] = density[iz] * Maxwellian_prefactor *
                                          exp(-((vpa.scratch - upar[iz])^2 + vperp.scratch[ivperp]^2)
                                               / this_vth^2) / vth_factor

                    # Also ensure both species go to zero smoothly at v_parallel=0 at the
                    # wall, where the boundary conditions require that distribution
                    # functions for both ions (where f_ion(v_parallel) = 0 for
                    # ±v_parallel<0) and neutrals (f_neutral=f_Kw, and f_Kw(v_parallel=0)=0)
                    # vanish.
                    #
                    # Implemented by multiplying by a smooth 'notch' function
                    # notch(v,u0,width) = 1 - exp(-(v-u0)^2/width)
                    # Factor of sqrt(2) included to make this consistent with earlier
                    # version of code - this width is arbitrary anyway.
                    width = sqrt(0.1) * this_vth
                    inverse_width_squared = 1.0 / width^2

                    @. pdf[:,ivperp,iz] *= 1.0 - exp(-vpa.scratch^2*inverse_width_squared)
                end
            end

            # Can use non-shared memory here because `init_ion_pdf_over_density!()` is
            # called inside a `@serial_region`
            lower_z_pdf_buffer = allocate_float(vpa, vperp)
            upper_z_pdf_buffer = allocate_float(vpa, vperp)
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

            # Add a non-flowing Maxwellian (that vanishes at the sheath entrance boundaries) to try to
            # avoid the 'hole' in the distribution function that can drive instabilities.
            @loop_z_vperp iz ivperp begin
                @. vpa.scratch = vpa.grid * vgrid_scale_factor[iz]
                vperp.scratch[ivperp] = vperp.grid[ivperp] * vgrid_scale_factor[iz]
                if vperp.n == 1
                    # Need to initialise using Maxwellian defined using T_∥ = 3*T as T_⟂=0
                    this_vth = sqrt(3.0) * vth[iz]
                    vth_factor = this_vth
                else
                    this_vth = vth[iz]
                    vth_factor = vth[iz]^3
                end
                @. pdf[:,ivperp,iz] += spec.z_IC.density_amplitude * Maxwellian_prefactor *
                                       (1.0 - (2.0 * z.grid[iz] / z.L)^2) *
                                       exp(-(vpa.scratch^2 + vperp.scratch[ivperp]^2)
                                           / this_vth^2) / vth_factor
            end

            # Get the unnormalised pdf and the moments of the constructed full-f
            # distribution function (which will be modified from the input moments).
            convert_full_f_ion_to_normalised!(pdf, density, upar, p, vth, vperp, vpa,
                                              vperp_spectral, vpa_spectral,
                                              evolve_density, evolve_upar, evolve_p,
                                              vgrid_scale_factor)

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
            @. pdf[:,ivperp,iz] = Maxwellian_prefactor*vpa.grid^2*exp(-(vpa.grid)^2 - vperp.grid[ivperp]^2) / vth[iz]
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
    elseif spec.vpa_IC.initialization_option == "isotropic-beam"
        v0 = spec.vpa_IC.v0 #0.5*sqrt(vperp.L^2 + (0.5*vpa.L)^2) # birth speed of beam
        vth0 = spec.vpa_IC.vth0
        v4norm = (v0^2)*(vth0^2) # spread of the beam in speed is vth0
        @loop_z iz begin
            @loop_vperp_vpa ivperp ivpa begin
                v2 = (vpa.grid[ivpa])^2 + vperp.grid[ivperp]^2 - v0^2
                pdf[ivpa,ivperp,iz] = Maxwellian_prefactor * exp(-(v2^2)/v4norm)
            end
            normfac = integral(view(pdf,:,:,iz), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
            @. pdf[:,:,iz] /= normfac
        end
    elseif spec.vpa_IC.initialization_option == "directed-beam"
        vpa0 = spec.vpa_IC.vpa0 #0.25*0.5*abs(vpa.L) # centre of beam in vpa
        vperp0 = spec.vpa_IC.vperp0 #0.5*abs(vperp.L) # centre of beam in vperp
        vth0 = spec.vpa_IC.vth0 #0.05*sqrt(vperp.L^2 + (0.5*vpa.L)^2) # width of beam in v 
        @loop_z iz begin
            # Force the parallel flow moment to be consistent with the directed-beam
            # distribution function, in case we are using moment-kinetic options.
            upar[iz] = vpa0

            # Remembering that the distribution function initialised here will be
            # normalised to unit density at the bottom, to calculate the temperature and
            # pressure we need the density integral and the internal energy integral.
            # Doing the integrals with WolframAlpha,
            #   f = exp(-((vpa-vpa0)^2 - (vperp-vperp0)^2 / vth0^2))
            #   ∫f d^3v = (2π) * (π^0.5 vth0) * (vth0^2 0.5 (π^0.5 vperp0/vth0 (erf(vperp0/vth0) + 1) + exp(-vperp0^2/vth0^2)))
            #           = π^1.5 vth0^3 * (π^0.5 vperp0/vth0 (erf(vperp0/vth0) + 1) + exp(-vperp0^2/vth0^2))
            #   ∫0.5 ((vpa-upar)^2 + vperp^2) f d^3v
            #     = π ∫vperp (0.5 π^0.5 vth0^3 + π^0.5 vth0 vperp^2) exp(-(vperp-vperp0)^2/vth0^2) dvperp
            #     = π^1.5 ∫vperp (0.5 vth0^3 + vth0 vperp^2) exp(-(vperp-vperp0)^2/vth0^2) dvperp
            #     = π^1.5 [0.5 vth0^3 0.5 vth0^2 (π^0.5 vperp0/vth0 (erf(vperp0/vth0) + 1) + exp(-vperp0^2/vth0^2)) + vth0 vth0^4 0.25 (π^0.5 vperp0/vth0 (2 vperp0^2/vth0^2 + 3) (erf(vperp0/vth0) + 1) + 2 exp(-vperp0^2/vth0^2) (vperp0^2/vth0^2 + 1))]
            #     = 0.25 π^1.5 vth0^5 [(π^0.5 vperp0/vth0 (erf(vperp0/vth0) + 1) + exp(-vperp0^2/vth0^2)) + (π^0.5 vperp0/vth0 (2 vperp0^2/vth0^2 + 3) (erf(vperp0/vth0) + 1) + 2 exp(-vperp0^2/vth0^2) (vperp0^2/vth0^2 + 1))]
            w = vperp0 / vth0
            mom0 = π^1.5 * vth0^3 * (sqrt(π) * w * (erf(w) + 1.0) + exp(-w^2))
            mom2 = 0.25 * π^1.5 * vth0^5 * (sqrt(π) * w * ((erf(w) + 1.0) + exp(-w^2)) + (sqrt(π) * w * (2.0 * w^2 + 3.0) * (erf(w) + 1.0) + 2.0 * exp(-w^2) * (w^2  + 1.0)))

            T = 2.0 / 3.0 * mom2 / mom0
            vth[iz] = sqrt(2.0 * T)
            p[iz] = density[iz] * T

            if evolve_p
                vpa_unnorm = @. vpa.grid * vth[iz] + upar[iz]
                vperp_unnorm = @. vperp.grid * vth[iz]
            elseif evolve_upar
                vpa_unnorm = @. vpa.grid + upar[iz]
                vperp_unnorm = vperp.grid
            else
                vpa_unnorm = vpa.grid
                vperp_unnorm = vperp.grid
            end
            @loop_vperp_vpa ivperp ivpa begin
                v2 = (vpa_unnorm[ivpa] - vpa0)^2 + (vperp_unnorm[ivperp] - vperp0)^2
                v2norm = vth0^2
                pdf[ivpa,ivperp,iz] = Maxwellian_prefactor * exp(-v2/v2norm)
            end
            normfac = integral(view(pdf,:,:,iz), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
            @. pdf[:,:,iz] /= normfac
        end
    end
    return nothing
end

"""
"""
function init_neutral_pdf_over_density!(pdf, spec, composition, vz, vr, vzeta, z,
                                        vzeta_spectral, vr_spectral, vz_spectral, density,
                                        uz, p, vth, v_norm_fac, evolve_density,
                                        evolve_upar, evolve_p, wall_flux_0, wall_flux_L)

    zero = 1.0e-14

    # Reduce the ion flux by `recycling_fraction` to account for ions absorbed by the
    # wall.
    wall_flux_0 *= composition.recycling_fraction
    wall_flux_L *= composition.recycling_fraction

    if vzeta.n == 1 && vr.n == 1
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        Maxwellian_prefactor = 1.0 / π^1.5
    end
    if spec.vz_IC.initialization_option == "gaussian"
        # initial condition is a Gaussian in the peculiar velocity
        if z.bc != "wall"
            for iz ∈ 1:z.n
                # obtain (vz - uz)/vth
                if evolve_p
                    # if evolve_upar = true and evolve_p = true, then vz coordinate is (vz-uz)/vth;
                    if evolve_upar
                        @. vz.scratch = vz.grid
                        # if evolve_upar = false and evolve_p = true, then vz coordinate is vz/vth;
                    else
                        @. vz.scratch = vz.grid - uz[iz]/vth[iz]
                    end
                    # if evolve_upar = true and evolve_p = false, then vz coordinate is vz-uz;
                elseif evolve_upar
                    @. vz.scratch = vz.grid/vth[iz]
                    # if evolve_upar = false and evolve_p = false, then vz coordinate is vz;
                else
                    @. vz.scratch = (vz.grid - uz[iz])/vth[iz]
                end

                @. vzeta.scratch = vzeta.grid/vth[iz]
                @. vr.scratch = vr.grid/vth[iz]

                if vzeta.n == 1 && vr.n == 1
                    # Need to initialise using Maxwellian defined using T_∥ = 3*T as T_⟂=0
                    vth_factor = sqrt(3.0)
                    if !evolve_p
                        vth_factor *= vth[iz]
                    end
                    vz.scratch ./= sqrt(3.0)
                else
                    if !evolve_p
                        vth_factor = vth[iz]^3
                    else
                        vth_factor = 1.0
                    end
                end
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    pdf[ivz,ivr,ivzeta,iz] = Maxwellian_prefactor *
                                             exp(-vz.scratch[ivz]^2 - vr.scratch[ivr]^2
                                                 - vzeta.scratch[ivzeta]^2) / vth_factor
                end
            end

            # Only do this correction for runs without wall bc, because consistency of
            # pdf and moments is taken care of by convert_full_f_neutral_to_normalised!()
            # for wall bc cases.
            for iz ∈ 1:z.n
                # densfac = the integral of the pdf over v-space, which should be unity,
                # but may not be exactly unity due to quadrature errors
                densfac = integral(view(pdf,:,:,:,iz), vz.grid, 0, vz.wgts, vr.grid, 0,
                                   vr.wgts, vzeta.grid, 0, vzeta.wgts)
                # Save w_s = vz - upar_s in vz.scratch
                if evolve_upar
                    vz.scratch .= vz.grid
                else
                    @. vz.scratch = vz.grid - uz[iz]
                end
                # pfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
                # where w_s^2 = (vz - uz_s)^2 + vr^2 + vzeta^2;
                # In moment-kinetic case, the velocity grids are already scaled by vths -
                # v_norm_fac takes care of this (it is 1 when velocity grids are not
                # normalised by vths, or vths when velocity grids are normalised by vths).
                # pfac should be equal to 3/2, but may not be exactly 3/2 due to quadrature errors
                pfac = @views (v_norm_fac[iz]/vth[iz])^2 *
                              (integral(pdf[:,:,:,iz], vz.scratch, 2, vz.wgts, vr.grid, 0,
                                        vr.wgts, vzeta.scratch, 0, vzeta.wgts)
                               + integral(pdf[:,:,:,iz], vz.scratch, 0, vz.wgts, vr.grid, 2,
                                          vr.wgts, vzeta.scratch, 0, vzeta.wgts)
                               + integral(pdf[:,:,:,iz], vz.scratch, 0, vz.wgts, vr.grid, 0,
                                          vr.wgts, vzeta.grid, 2, vzeta.wgts))
                # pfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
                if evolve_upar
                    uz_offset = 0.0
                else
                    uz_offset = uz[iz]
                end
                pfac2 = @views (v_norm_fac[iz]/vth[iz])^2 *
                               integral((vzeta,vr,vz)->(((vz - uz_offset)^2 + vzeta^2 + vr^2) * (((vz - uz_offset)^2 + vzeta^2 + vr^2) * (v_norm_fac[iz] / vth[iz])^2 / pfac - 1.0/densfac)),
                                        pdf[:,:,:,iz], vzeta, vr, vz)

                # The following update ensures the density and pressure moments of pdf
                # have the expected values. The velocity moment is always exactly zero
                # from symmetry, so does not need correcting.
                # The corrected version has the correct moments because
                #   ∫d^3v pdf_before = densfac
                #   ∫d^3v m_s w_s^2 / vths^2 * pdf_before = pfac
                #   ∫d^3v m_s w_s^2 (m_s*w_s^2/vths^2/pfac - 1/densfac) / vths^2 pdf_before = pfac2
                # so if
                #   pdf = ( 1/densfac + (1.5 - pfac/densfac)/pfac2 * (m_s*w_s^2/vths^2/pfac - 1/densfac) ) * pdf_before
                # then
                #   ∫d^3v ( 1/densfac + (1.5 - pfac/densfac)/pfac2 * (m_s*w_s^2/vths^2/pfac - 1/densfac) ) * pdf_before
                #   = 1 + (1.5 - pfac / densfac) / pfac2 * (pfac/pfac - densfac/densfac)
                #   = 1
                # and
                #   ∫d^3v m_s w_s^2 / vths^2 * ( 1/densfac + (1.5 - pfac/densfac)/pfac2 * (m_s*w_s^2/vths^2/pfac - 1/densfac) ) * pdf_before
                #   = pfac/densfac + (1.5 - pfac/densfac)/pfac2 * pfac2
                #   = 1.5
                @loop_vzeta_vr ivzeta ivr begin
                    @views @. pdf[:,ivr,ivzeta,iz] = pdf[:,ivr,ivzeta,iz]/densfac +
                                                     (1.5 - pfac/densfac)/pfac2 *
                                                     ((vr.grid[ivr]^2 + vzeta.grid[ivzeta]^2 + vz.scratch^2)*(v_norm_fac[iz]/vth[iz])^2/pfac - 1.0/densfac) *
                                                     pdf[:,ivr,ivzeta,iz]
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
            if evolve_p
                # Scale the velocity grid used for initialization in case the
                # temperature changes a lot.
                vgrid_scale_factor = copy(vth)
            else
                vgrid_scale_factor = ones(size(vth))
            end
            for iz ∈ zrange
                @. vz.scratch = vz.grid * vgrid_scale_factor[iz]
                @. vzeta.scratch = vzeta.grid * vgrid_scale_factor[iz]
                @. vr.scratch = vr.grid * vgrid_scale_factor[iz]
                @loop_vzeta_vr ivzeta ivr begin
                    # Initialise as full-f distribution functions, then
                    # normalise/interpolate (if necessary). This makes it easier to
                    # initialise a normalised pdf consistent with the moments, although it
                    # modifies the moments from the 'input' values.
                    if vzeta.n == 1 && vr.n == 1
                        # Need to initialise using Maxwellian defined using T_∥ = 3*T as T_⟂=0
                        this_vth = sqrt(3.0) * vth[iz]
                        vth_factor = this_vth
                    else
                        this_vth = vth[iz]
                        vth_factor = vth[iz]^3
                    end
                    @. pdf[:,ivr,ivzeta,iz] = density[iz] * Maxwellian_prefactor *
                                              exp(-((vz.scratch - uz[iz])^2 +
                                                    vzeta.scratch[ivzeta]^2 + vr.scratch[ivr]^2)
                                                  / this_vth^2) / vth_factor

                    # Also ensure both species go to zero smoothly at v_z=0 at the
                    # wall, where the boundary conditions require that distribution
                    # functions for both ions (where f_ion(v_z) = 0 for
                    # ±v_z<0) and neutrals (f_neutral=f_Kw, and f_Kw(v_z=0)=0)
                    # vanish.
                    #
                    # Implemented by multiplying by a smooth 'notch' function
                    # notch(v,u0,width) = 1 - exp(-(v-u0)^2/width)
                    # Factor of sqrt(2) included to make this consistent with earlier
                    # version of code - this width is arbitrary anyway.
                    width = sqrt(0.1) * this_vth
                    inverse_width_squared = 1.0 / width^2

                    @. pdf[:,ivr,ivzeta,iz] *= 1.0 - exp(-vz.scratch^2*inverse_width_squared)
                end
            end
            # Can use non-shared memory here because `init_ion_pdf_over_density!()` is
            # called inside a `@serial_region`
            lower_z_pdf_buffer = allocate_float(vz, vr, vzeta)
            upper_z_pdf_buffer = allocate_float(vz, vr, vzeta)
            if z.irank == 0
                lower_z_pdf_buffer .= pdf[:,:,:,1]
            end
            if z.irank == z.nrank - 1
                upper_z_pdf_buffer .= pdf[:,:,:,end]
            end

            # Get the boundary pdfs from the processes that have the actual z-boundary
            MPI.Bcast!(lower_z_pdf_buffer, 0, z.comm)
            MPI.Bcast!(upper_z_pdf_buffer, z.nrank - 1, z.comm)

            # Also need to get the (ion) wall fluxes from the processes that have the
            # actual z-boundary
            temp = Ref(wall_flux_0)
            MPI.Bcast!(temp, 0, z.comm)
            wall_flux_0 = temp[]
            temp[] = wall_flux_L
            MPI.Bcast!(temp, z.nrank - 1, z.comm)
            wall_flux_L = temp[]

            # ...and the vgrid_scale_factor from both z-boundaries
            temp[] = vgrid_scale_factor[1]
            MPI.Bcast!(temp, 0, z.comm)
            vgrid_scale_factor0 = temp[]
            temp[] = vgrid_scale_factor[end]
            MPI.Bcast!(temp, z.nrank - 1, z.comm)
            vgrid_scale_factorL = temp[]

            # Re-calculate Knudsen distribution instead of using
            # `boundary_distributions.knudsen`, so that we can include vgrid_scale_factor
            # here.
            knudsen_pdf_lower = allocate_float(vz, vr, vzeta)
            knudsen_pdf_upper = allocate_float(vz, vr, vzeta)
            T_wall_over_m = composition.T_wall / composition.mn_over_mi
            if vzeta.n > 1 && vr.n > 1
                # 3V specification of neutral wall emission distribution for boundary condition
                # get the true Knudsen cosine distribution for neutral particle wall emission
                @. vz.scratch = vz.grid * vgrid_scale_factor0
                @. vzeta.scratch = vzeta.grid * vgrid_scale_factor0
                @. vr.scratch = vr.grid * vgrid_scale_factor0
                for ivzeta in 1:vzeta.n
                    for ivr in 1:vr.n
                        for ivz in 1:vz.n
                            v_transverse = sqrt(vzeta.scratch[ivzeta]^2 + vr.scratch[ivr]^2)
                            v_normal = abs(vz.scratch[ivz])
                            v_tot = sqrt(v_normal^2 + v_transverse^2)
                            if  v_tot > zero
                                prefac = v_normal/v_tot
                            else
                                prefac = 0.0
                            end
                            knudsen_pdf_lower[ivz,ivr,ivzeta] = 0.75 / π / T_wall_over_m^2 * prefac *
                                                                exp(-0.5 * (v_normal^2 + v_transverse^2) / T_wall_over_m)
                        end
                    end
                end
                @. vz.scratch = vz.grid * vgrid_scale_factorL
                @. vzeta.scratch = vzeta.grid * vgrid_scale_factorL
                @. vr.scratch = vr.grid * vgrid_scale_factorL
                for ivzeta in 1:vzeta.n
                    for ivr in 1:vr.n
                        for ivz in 1:vz.n
                            v_transverse = sqrt(vzeta.scratch[ivzeta]^2 + vr.scratch[ivr]^2)
                            v_normal = abs(vz.scratch[ivz])
                            v_tot = sqrt(v_normal^2 + v_transverse^2)
                            if  v_tot > zero
                                prefac = v_normal/v_tot
                            else
                                prefac = 0.0
                            end
                            knudsen_pdf_upper[ivz,ivr,ivzeta] = 0.75 / π / T_wall_over_m^2 * prefac *
                                                                exp(-0.5 * (v_normal^2 + v_transverse^2) / T_wall_over_m)
                        end
                    end
                end
            elseif vzeta.n == 1 && vr.n == 1
                # get the marginalised Knudsen cosine distribution after integrating over
                # vperp appropriate for 1V model

                # Knudsen cosine distribution does not have separate T_∥ and T_⟂, so is
                # marginalised rather than setting T_⟂=0, therefore no need to convert to
                # a thermal speed defined with the parallel temperature in 1V case.
                @. vz.scratch = vz.grid * vgrid_scale_factor0
                @. knudsen_pdf_lower[:,1,1] = (3.0 * sqrt(π) * (0.5 / T_wall_over_m)^1.5) * abs(vz.scratch) * erfc(sqrt(0.5 / T_wall_over_m) * abs(vz.scratch))

                @. vz.scratch = vz.grid * vgrid_scale_factorL
                @. knudsen_pdf_upper[:,1,1] = (3.0 * sqrt(π) * (0.5 / T_wall_over_m)^1.5) * abs(vz.scratch) * erfc(sqrt(0.5 / T_wall_over_m) * abs(vz.scratch))
            else
                error("If 1V expect both vzeta.n and vr.n to be 1. Got "
                      * "vzeta.n=$(vzeta.n), vr.n=$(vr.n).")
            end

            # add this species' contribution to the combined ion/neutral particle flux
            # out of the domain at z=-Lz/2
            if vzeta.n > 1 || vr.n > 1
                wgts_3V_scale_factor0 = vgrid_scale_factor0
            else
                wgts_3V_scale_factor0 = 1.0
            end
            @views wall_flux_0 += integrate_over_negative_vz(
                                      vgrid_scale_factor0 .* abs.(vz.grid) .* lower_z_pdf_buffer,
                                      vgrid_scale_factor0 .* vz.grid,
                                      vgrid_scale_factor0 .* vz.wgts, vz.scratch3,
                                      vgrid_scale_factor0 .* vr.grid,
                                      wgts_3V_scale_factor0 .* vr.wgts,
                                      vgrid_scale_factor0 .* vzeta.grid,
                                      wgts_3V_scale_factor0 .* vzeta.wgts)
            # for left boundary in zed (z = -Lz/2), want
            # f_n(z=-Lz/2, v_z > 0) = Γ_0 * f_KW(v_z) * pdf_norm_fac(-Lz/2)
            @loop_vz ivz begin
                if vz.scratch[ivz] > zero
                    @. lower_z_pdf_buffer[ivz,:,:] = wall_flux_0 * knudsen_pdf_lower[ivz,:,:]
                end
            end

            # add this species' contribution to the combined ion/neutral particle flux
            # out of the domain at z=-Lz/2
            if vzeta.n > 1 || vr.n > 1
                wgts_3V_scale_factorL = vgrid_scale_factorL
            else
                wgts_3V_scale_factorL = 1.0
            end
            @views wall_flux_L += integrate_over_positive_vz(
                                      vgrid_scale_factorL .* abs.(vz.grid) .* upper_z_pdf_buffer,
                                      vgrid_scale_factorL .* vz.grid,
                                      vgrid_scale_factorL .* vz.wgts, vz.scratch3,
                                      vgrid_scale_factorL .* vr.grid,
                                      wgts_3V_scale_factorL .* vr.wgts,
                                      vgrid_scale_factorL .* vzeta.grid,
                                      wgts_3V_scale_factorL .* vzeta.wgts)
            # for right boundary in zed (z = Lz/2), want
            # f_n(z=Lz/2, v_z < 0) = Γ_Lz * f_KW(v_z) * pdf_norm_fac(Lz/2)
            @loop_vz ivz begin
                if vz.grid[ivz] < -zero
                    @. upper_z_pdf_buffer[ivz,:,:] = wall_flux_L * knudsen_pdf_upper[ivz,:,:]
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
            convert_full_f_neutral_to_normalised!(pdf, density, uz, p, vth, vzeta, vr,
                                                  vz, vzeta_spectral, vr_spectral,
                                                  vz_spectral, evolve_density,
                                                  evolve_upar, evolve_p,
                                                  vgrid_scale_factor)

            if !evolve_density
                # Need to divide out density to return pdf/density
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    pdf[ivz,ivr,ivzeta,:] ./= density
                end
            end
        end
    elseif spec.vz_IC.initialization_option == "vzgaussian"
        @loop_z_vzeta_vr iz ivzeta ivr begin
            @. pdf[:,ivr,ivzeta,iz] = vz.grid^2*Maxwellian_prefactor*exp(-vz.scratch^2 - vr[ivr]^2 -
                                                                       vzeta[ivzeta]^2) / vth[iz]
        end
    elseif spec.vz_IC.initialization_option == "sinusoid"
        # initial condition is sinusoid in vz
        @loop_z_vzeta_vr iz ivzeta ivr begin
            @. pdf[:,ivr,ivzeta,iz] =
                spec.vz_IC.amplitude*cospi(2.0*spec.vz_IC.wavenumber*vz.grid/vz.L)
        end
    elseif spec.vz_IC.initialization_option == "monomial"
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
function init_electron_pdf_over_density_and_boundary_phi!(pdf, phi, density, upar, vth, r,
        z, vpa, vperp, vperp_spectral, vpa_spectral, vpa_advect, moments, num_diss_params,
        me_over_mi, scratch_dummy; restart_from_boltzmann=false)

    if vperp.n == 1
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        Maxwellian_prefactor = 1.0 / π^1.5
    end
    if z.bc == "wall"
        @begin_r_region()
        if vperp.n == 1
            # Initialize with a 1D Maxwellian with temperature T_∥
            @loop_r ir begin
                # Initialise an unshifted Maxwellian as a first step
                @loop_z iz begin
                    vpa_over_vth = @. vpa.scratch3 = (vpa.grid + upar[iz,ir] / vth[iz,ir]) / sqrt(3.0)
                    @loop_vperp ivperp begin
                        @. pdf[:,ivperp,iz,ir] = Maxwellian_prefactor / sqrt(3.0) * exp(-vpa_over_vth^2)
                    end
                end
            end
        else
            vperp_grid = vperp.grid
            @loop_r ir begin
                # Initialise an unshifted Maxwellian as a first step
                @loop_z iz begin
                    vpa_over_vth = @. vpa.scratch3 = vpa.grid + upar[iz,ir] / vth[iz,ir]
                    @loop_vperp ivperp begin
                        @. pdf[:,ivperp,iz,ir] = Maxwellian_prefactor * exp(-vpa_over_vth^2 - vperp_grid[ivperp]^2)
                    end
                end
            end
        end
        # Apply the sheath boundary condition to get cut-off boundary distribution
        # functions and boundary values of phi
        @begin_r_anyzv_region()
        @loop_r ir begin
            @views enforce_boundary_condition_on_electron_pdf!(
                       pdf[:,:,:,ir], phi[:,ir], vth[:,ir], upar[:,ir], z, vperp, vpa,
                       vperp_spectral, vpa_spectral, vpa_advect, moments,
                       num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                       me_over_mi, ir; allow_failure=false)
        end

        # Distribute the z-boundary pdf values to every process
        @begin_serial_region()
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

        @begin_r_z_region()
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
                norm_factor = integral(pdf[:,ivperp,iz,ir], vpa.wgts)
                @. pdf[:,ivperp,iz,ir] /= norm_factor
                #println("TMP FOR TESTING -- init electron pdf")
                #@. pdf[:,ivperp,iz] = exp(-2*vpa.grid[:]^2)*exp(-z.grid[iz]^2)
            end
        end
    else
        @begin_r_z_region()
        @loop_r ir begin
            # Initialise an unshifted Maxwellian as a first step
            @loop_z iz begin
                @loop_vperp ivperp begin
                    @. pdf[:,ivperp,iz,ir] = Maxwellian_prefactor * exp(-vpa.grid^2)
                end
            end
        end
    end

    # Ensure initial electron distribution function obeys constraints
    hard_force_moment_constraints!(pdf, moments, vpa, vperp)

    return nothing
end

function init_pdf_moments_manufactured_solns!(pdf, moments, vz, vr, vzeta, vpa, vperp, z,
                                              r, r_bc, n_ion_species, n_neutral_species,
                                              geometry, composition, species,
                                              manufactured_solns_input, collisions)
    manufactured_solns_list = manufactured_solutions(manufactured_solns_input, r.L, z.L,
                                                     r_bc, z.bc, geometry, composition,
                                                     species, r.n, vperp.n, vzeta.n, vr.n)
    dfni_func = manufactured_solns_list.dfni_func
    densi_func = manufactured_solns_list.densi_func
    dfnn_func = manufactured_solns_list.dfnn_func
    densn_func = manufactured_solns_list.densn_func
    #nb manufactured functions not functions of species
    @begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        moments.ion.dens[iz,ir,is] = densi_func(z.grid[iz],r.grid[ir],0.0)
        @loop_vperp_vpa ivperp ivpa begin
            pdf.ion.norm[ivpa,ivperp,iz,ir,is] = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],0.0)
        end
    end
    # update upar, ppar, qpar, vth consistent with manufactured solns
    update_density!(moments.ion.dens, moments.ion.dens_updated,
                    pdf.ion.norm, vpa, vperp, z, r, composition)
    # get particle flux
    update_upar!(moments.ion.upar, moments.ion.upar_updated,
                 moments.ion.dens, moments.ion.ppar, pdf.ion.norm,
                 vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_p)
    update_p!(moments.ion.p, moments.ion.p_updated, moments.ion.dens, moments.ion.upar,
              pdf.ion.norm, vpa, vperp, z, r, composition, moments.evolve_density,
              moments.evolve_upar)
    update_ppar!(moments.ion.ppar, moments.ion.dens, moments.ion.upar, moments.ion.vth, moments.ion.p,
                 pdf.ion.norm, vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_upar, moments.evolve_p)
    update_pperp!(moments.ion.pperp, moments.ion.p, moments.ion.ppar, vperp, z, r,
                  composition)
    update_ion_qpar!(moments.ion.qpar, moments.ion.qpar_updated,
                 moments.ion.dens, moments.ion.upar,
                 moments.ion.vth, moments.ion.dT_dz, pdf.ion.norm, vpa, vperp, z, r,
                 composition, drift_kinetic_ions, collisions, moments.evolve_density, moments.evolve_upar,
                 moments.evolve_p)
    update_vth!(moments.ion.vth, moments.ion.p, moments.ion.dens, z, r, composition)

    @begin_serial_region()
    @serial_region begin
        # If electrons are being used, they will be initialized properly later. Here
        # we only set the values to avoid false positives from the debug checks
        # (when @debug_track_initialized is active).
        moments.electron.dens .= 0.0
        moments.electron.upar .= 0.0
        moments.electron.p .= 0.0
        moments.electron.ppar .= 0.0
        moments.electron.pperp .= 0.0
        moments.electron.qpar .= 0.0
        moments.electron.temp .= 0.0
        moments.electron.constraints_A_coefficient .= 1.0
        moments.electron.constraints_B_coefficient .= 0.0
        moments.electron.constraints_C_coefficient .= 0.0
        if composition.electron_physics ∈ (kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
            pdf.electron.norm .= 0.0
        end
    end

    if n_neutral_species > 0
        @begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = densn_func(z.grid[iz],r.grid[ir],0.0)
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = dfnn_func(vz.grid[ivz],vr.grid[ivr],vzeta.grid[ivzeta],z.grid[iz],r.grid[ir],0.0)
            end
        end
        # get consistent moments with manufactured solutions
        update_neutral_density!(moments.neutral.dens, moments.neutral.dens_updated,
                                pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_uz!(moments.neutral.uz, moments.neutral.uz_updated,
                           moments.neutral.dens, moments.neutral.vth, pdf.neutral.norm,
                           vz, vr, vzeta, z, r, composition, moments.evolve_density,
                           moments.evolve_p)
        update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                           moments.neutral.dens, moments.neutral.vth, pdf.neutral.norm,
                           vz, vr, vzeta, z, r, composition, moments.evolve_density,
                           moments.evolve_p)
        update_neutral_uzeta!(moments.neutral.uzeta, moments.neutral.uzeta_updated,
                              moments.neutral.dens, moments.neutral.vth, pdf.neutral.norm,
                              vz, vr, vzeta, z, r, composition, moments.evolve_density,
                              moments.evolve_p)
        update_neutral_p!(moments.neutral.p, moments.neutral.p_updated,
                          moments.neutral.dens, moments.neutral.uz, moments.neutral.ur,
                          moments.neutral.uzeta, moments.neutral.vth, pdf.neutral.norm,
                          vz, vr, vzeta, z, r, composition, moments.evolve_density,
                          moments.evolve_upar, moments.evolve_p)
        update_neutral_pz!(moments.neutral.pz, moments.neutral.pz_updated,
                           moments.neutral.dens, moments.neutral.uz, moments.neutral.p,
                           moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z, r,
                           composition, moments.evolve_density, moments.evolve_upar,
                           moments.evolve_p)
        update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated,
                           moments.neutral.dens, moments.neutral.ur, moments.neutral.vth,
                           pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_upar, moments.evolve_p)
        update_neutral_pzeta!(moments.neutral.pzeta, moments.neutral.pzeta_updated,
                              moments.neutral.dens, moments.neutral.uzeta,
                              moments.neutral.vth, pdf.neutral.norm, vz, vr, vzeta, z, r,
                              composition, moments.evolve_density, moments.evolve_upar,
                              moments.evolve_p)
        update_neutral_qz!(moments.neutral.qz, moments.neutral.qz_updated,
                           moments.neutral.dens, moments.neutral.uz, moments.neutral.vth,
                           pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_upar, moments.evolve_p)
    end
    return nothing
end

"""
Take the full ion distribution function, calculate the moments, then
normalise and shift to the moment-kinetic grid.

Uses input value of `f` and modifies in place to the normalised distribution functions.
Input `density`, `upar`, `p`, and `vth` are not used, the values are overwritten with
the moments of `f`.

Inputs/outputs depend on z, vperp, and vpa (should be inside loops over species, r).

The velocity grid that the input `f` is defined on can be scaled by `vgrid_scale_factor`:
`f` is given on a velocity grid `vperp.grid .* vgrid_scale_factor` and
`vpa.grid .* vgrid_scale_factor`.
"""
function convert_full_f_ion_to_normalised!(f, density, upar, p, vth, vperp, vpa,
                                           vperp_spectral, vpa_spectral, evolve_density,
                                           evolve_upar, evolve_p,
                                           vgrid_scale_factor=ones(size(vth)))

    @loop_z iz begin
        vpa_grid_input = vpa.grid .* vgrid_scale_factor[iz]
        vpa_wgts_input = vpa.wgts .* vgrid_scale_factor[iz]
        vperp_grid_input = vperp.grid .* vgrid_scale_factor[iz]
        if vperp.n == 1
            vperp_wgts_input = vperp.wgts
        else
            vperp_wgts_input = vperp.wgts .* vgrid_scale_factor[iz]
        end

        # Calculate moments
        @views density[iz] = integral(f[:,:,iz], vpa_grid_input, 0, vpa_wgts_input,
                                      vperp_grid_input, 0, vperp_wgts_input)
        @views upar[iz] = integral(f[:,:,iz], vpa_grid_input, 1, vpa_wgts_input,
                                   vperp_grid_input, 0, vperp_wgts_input) /
                          density[iz]
        @views p[iz] = (integral(f[:,:,iz], vpa_grid_input .- upar[iz], 2, vpa_wgts_input,
                                 vperp_grid_input, 0, vperp_wgts_input)
                        + integral(f[:,:,iz], vpa_grid_input, 0, vpa_wgts_input,
                                   vperp_grid_input, 2, vperp_wgts_input)) / 3.0
        vth[iz] = sqrt(2.0*p[iz]/density[iz])

        # Normalise f
        if evolve_p && vperp.n == 1
            f[:,:,iz] .*= vth[iz] / density[iz]
        elseif evolve_p
            f[:,:,iz] .*= vth[iz]^3 / density[iz]
        elseif evolve_density
            f[:,:,iz] ./= density[iz]
        end

        # Interpolate f to moment kinetic grid
        if evolve_p || evolve_upar
            # The values to interpolate *to* are the v_parallel/vperp values corresponding
            # to the w_parallel/w_perp grid
            vpa.scratch .= vpagrid_to_vpa(vpa.grid, vth[iz], upar[iz], evolve_p,
                                          evolve_upar)
            if evolve_p
                @. vperp.scratch .= vperp.grid * vth[iz]
            else
                vperp.scratch .= vperp.grid
            end
            # It would be inconvienient to create a coordinate object corresponding to
            # vpa_grid_input just to do this interpolation. Instead we can use vpa, but
            # scale the output grid by 1/vgrid_scale_factor
            vpa.scratch ./= vgrid_scale_factor[iz]
            vperp.scratch ./= vgrid_scale_factor[iz]
            @loop_vperp ivperp begin
                @views vpa.scratch2 .= f[:,ivperp,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[:,ivperp,iz], vpa.scratch, vpa.scratch2,
                                               vpa, vpa_spectral)
            end
            @loop_vpa ivpa begin
                @views vperp.scratch2 .= f[ivpa,:,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[ivpa,:,iz], vperp.scratch,
                                               vperp.scratch2, vperp, vperp_spectral)
            end
        end
    end

    return nothing
end

"""
Take the full neutral-particle distribution function, calculate the moments, then
normalise and shift to the moment-kinetic grid.

Uses input value of `f` and modifies in place to the normalised distribution functions.
Input `density`, `uz`, `p`, and `vth` are not used, the values are overwritten with
the moments of `f`.

Inputs/outputs depend on z, vzeta, vr and vz (should be inside loops over species, r).

The velocity grid that the input `f` is defined on can be scaled by `vgrid_scale_factor`:
`f` is given on a velocity grid `vzeta.grid .* vgrid_scale_factor`,
`vr.grid .* vgrid_scale_factor`, and `vz.grid .* vgrid_scale_factor`.
"""
function convert_full_f_neutral_to_normalised!(f, density, uz, p, vth, vzeta, vr, vz,
                                               vzeta_spectral, vr_spectral, vz_spectral,
                                               evolve_density, evolve_upar, evolve_p,
                                               vgrid_scale_factor=ones(size(vth)))

    @loop_z iz begin
        vz_grid_input = vz.grid .* vgrid_scale_factor[iz]
        vz_wgts_input = vz.wgts .* vgrid_scale_factor[iz]
        vzeta_grid_input = vzeta.grid .* vgrid_scale_factor[iz]
        vr_grid_input = vr.grid .* vgrid_scale_factor[iz]
        if vzeta.n == 1 && vr.n == 1
            vzeta_wgts_input = vzeta.wgts
            vr_wgts_input = vr.wgts
        else
            vzeta_wgts_input = vzeta.wgts .* vgrid_scale_factor[iz]
            vr_wgts_input = vr.wgts .* vgrid_scale_factor[iz]
        end

        # Calculate moments
        @views density[iz] = integral(
                                 f[:,:,:,iz], vz_grid_input, 0, vz_wgts_input,
                                 vr_grid_input, 0, vr_wgts_input, vzeta_grid_input, 0,
                                 vzeta_wgts_input)
        @views uz[iz] = integral(
                            f[:,:,:,iz], vz_grid_input, 1, vz_wgts_input, vr_grid_input,
                            0, vr_wgts_input, vzeta_grid_input, 0, vzeta_wgts_input) /
                        density[iz]
        @views p[iz] = (integral(
                            f[:,:,:,iz], vz_grid_input .- uz[iz], 2, vz_wgts_input,
                            vr_grid_input, 0, vr_wgts_input, vzeta_grid_input, 0,
                            vzeta_wgts_input) +
                        integral(
                            f[:,:,:,iz], vz_grid_input, 0, vz_wgts_input, vr_grid_input,
                            2, vr_wgts_input, vzeta_grid_input, 0, vzeta_wgts_input) +
                        integral(
                            f[:,:,:,iz], vz_grid_input, 0, vz_wgts_input, vr_grid_input,
                            0, vr_wgts_input, vzeta_grid_input, 2,  vzeta_wgts_input)) / 3.0
        vth[iz] = sqrt(2.0*p[iz]/density[iz])

        # Normalise f
        if evolve_p && vzeta.n == 1 && vr.n == 1
            f[:,:,:,iz] .*= vth[iz] / density[iz]
        elseif evolve_p
            f[:,:,:,iz] .*= vth[iz]^3 / density[iz]
        elseif evolve_density
            f[:,:,:,iz] ./= density[iz]
        end

        # Interpolate f to moment kinetic grid
        if evolve_p || evolve_upar
            # The values to interpolate *to* are the vz/vzeta/vr values corresponding to
            # the wz/wzeta/wr grid.
            vz.scratch .= vpagrid_to_vpa(vz.grid, vth[iz], uz[iz], evolve_p, evolve_upar)
            if evolve_p
                @. vzeta.scratch = vzeta.grid * vth[iz]
                @. vr.scratch = vr.grid * vth[iz]
            else
                vzeta.scratch .= vzeta.grid
                vr.scratch .= vr.grid
            end
            # It would be inconvienient to create a coordinate object corresponding to
            # vpa_grid_input just to do this interpolation. Instead we can use vpa, but
            # scale the output grid by 1/vgrid_scale_factor
            vz.scratch ./= vgrid_scale_factor[iz]
            vzeta.scratch ./= vgrid_scale_factor[iz]
            vr.scratch ./= vgrid_scale_factor[iz]
            @loop_vzeta_vr ivzeta ivr begin
                @views vz.scratch2 .= f[:,ivr,ivzeta,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[:,ivr,ivzeta,iz], vz.scratch,
                                               vz.scratch2, vz, vz_spectral)
            end
            @loop_vr_vz ivr ivz begin
                @views vzeta.scratch2 .= f[ivz,ivr,:,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[ivz,ivr,:,iz], vzeta.scratch,
                                               vzeta.scratch2, vzeta, vzeta_spectral)
            end
            @loop_vzeta_vz ivzeta ivz begin
                @views vr.scratch2 .= f[ivz,:,ivzeta,iz] # Copy to use as input to interpolation
                @views interpolate_to_grid_1d!(f[ivz,:,ivzeta,iz], vr.scratch,
                                               vr.scratch2, vr, vr_spectral)
            end
        end
    end

    return nothing
end

end
