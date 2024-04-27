"""
"""
module initial_conditions

export allocate_pdf_and_moments
export init_pdf_and_moments!

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
using ..moment_kinetics_structs: scratch_pdf, pdf_substruct,
                                 pdf_struct, moments_struct, boundary_distributions_struct
using ..velocity_moments: integrate_over_vspace, integrate_over_neutral_vspace
using ..velocity_moments: integrate_over_positive_vz, integrate_over_negative_vz
using ..velocity_moments: create_moments_ion, create_moments_neutral, update_qpar!
using ..velocity_moments: update_neutral_density!, update_neutral_pz!, update_neutral_pr!, update_neutral_pzeta!
using ..velocity_moments: update_neutral_uz!, update_neutral_ur!, update_neutral_uzeta!, update_neutral_qz!
using ..velocity_moments: update_ppar!, update_upar!, update_density!, update_pperp!, update_vth!, reset_moments_status!

using ..manufactured_solns: manufactured_solutions

using MPI

"""
Creates the structs for the pdf and the velocity-space moments
"""
function allocate_pdf_and_moments(composition, r, z, vperp, vpa, vzeta, vr, vz,
                                  evolve_moments, collisions, external_source_settings,
                                  num_diss_params)
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

    moments = moments_struct(ion, neutral, evolve_moments.density,
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
    # buffer array is for ion-neutral collisions, not for storing ion pdf
    pdf_ion_buffer = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, composition.n_neutral_species) # n.b. n_species is n_neutral_species here
    pdf_neutral_norm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, composition.n_neutral_species)
    # buffer array is for neutral-ion collisions, not for storing neutral pdf
    pdf_neutral_buffer = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, composition.n_ion_species)
    return pdf_struct(pdf_substruct(pdf_ion_norm, pdf_ion_buffer),
                      pdf_substruct(pdf_neutral_norm, pdf_neutral_buffer))

end

"""
creates the normalised pdf and the velocity-space moments and populates them
with a self-consistent initial condition
"""
function init_pdf_and_moments!(pdf, moments, boundary_distributions, geometry,
                               composition, r, z, vperp, vpa, vzeta, vr, vz,
                               vpa_spectral, vz_spectral, species,
                               external_source_settings, manufactured_solns_input)
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
        @. vth = sqrt(vth)
    end
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

function init_pdf_moments_manufactured_solns!(pdf, moments, vz, vr, vzeta, vpa, vperp, z,
                                              r, n_ion_species, n_neutral_species,
                                              geometry, composition, species,
                                              manufactured_solns_input)
    manufactured_solns_list = manufactured_solutions(manufactured_solns_input, r.L, z.L,
                                                     r.bc, z.bc, geometry, composition,
                                                     species, r.n, vperp.n)
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
    update_density!(moments.ion.dens, moments.ion.dens_updated,
                    pdf.ion.norm, vpa, vperp, z, r, composition)
    # get particle flux
    update_upar!(moments.ion.upar, moments.ion.upar_updated,
                 moments.ion.dens, moments.ion.ppar, pdf.ion.norm,
                 vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_ppar)
    update_ppar!(moments.ion.ppar, moments.ion.ppar_updated,
                 moments.ion.dens, moments.ion.upar, pdf.ion.norm,
                 vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_upar)
    update_pperp!(moments.ion.pperp, pdf.ion.norm, vpa, vperp, z, r, composition)
    update_qpar!(moments.ion.qpar, moments.ion.qpar_updated,
                 moments.ion.dens, moments.ion.upar,
                 moments.ion.vth, pdf.ion.norm, vpa, vperp, z, r,
                 composition, moments.evolve_density, moments.evolve_upar,
                 moments.evolve_ppar)
    update_vth!(moments.ion.vth, moments.ion.ppar, moments.ion.pperp, moments.ion.dens, vperp, z, r, composition)

    if n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = densn_func(z.grid[iz],r.grid[ir],0.0)
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = dfnn_func(vz.grid[ivz],vr.grid[ivr],vzeta.grid[ivzeta],z.grid[iz],r.grid[ir],0.0)
            end
        end
        # get consistent moments with manufactured solutions
        update_neutral_density!(moments.neutral.dens,
                                moments.neutral.dens_updated, pdf.neutral.norm,
                                vz, vr, vzeta, z, r, composition)
        # nb bad naming convention uz -> n uz below
        update_neutral_uz!(moments.neutral.uz, moments.neutral.uz_updated,
                           moments.neutral.dens, moments.neutral.pz,
                           pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_ppar)
        update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                           moments.neutral.dens, pdf.neutral.norm, vz, vr,
                           vzeta, z, r, composition)
        update_neutral_uzeta!(moments.neutral.uzeta,
                              moments.neutral.uzeta_updated,
                              moments.neutral.dens, pdf.neutral.norm, vz, vr,
                              vzeta, z, r, composition)
        @loop_sn_r_z isn ir iz begin
            moments.neutral.uz[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            moments.neutral.ur[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            moments.neutral.uzeta[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
        end
        update_neutral_pz!(moments.neutral.pz, moments.neutral.pz_updated,
                           moments.neutral.dens, moments.neutral.uz,
                           pdf.neutral.norm, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_upar)
        update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated,
                           pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_pzeta!(moments.neutral.pzeta,
                              moments.neutral.pzeta_updated, pdf.neutral.norm,
                              vz, vr, vzeta, z, r, composition)
        update_neutral_qz!(moments.neutral.qz, moments.neutral.qz_updated,
                           moments.neutral.dens, moments.neutral.uz,
                           moments.neutral.vth, pdf.neutral.norm, vz, vr,
                           vzeta, z, r, composition, moments.evolve_density,
                           moments.evolve_upar, moments.evolve_ppar)
        #update ptot (isotropic pressure)
        if vzeta.n > 1 || vr.n > 1 #if not using marginalised distribution function
            begin_sn_r_z_region()
            @loop_sn_r_z isn ir iz begin
                moments.neutral.ptot[iz,ir,isn] = (moments.neutral.pz[iz,ir,isn] + moments.neutral.pr[iz,ir,isn] + moments.neutral.pzeta[iz,ir,isn])/3.0
            end
        else #1D model
            moments.neutral.ptot .= moments.neutral.pz
        end
        # now convert from particle particle flux to parallel flow
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            # get vth for neutrals
            moments.neutral.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.ptot[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
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
                            if  v_normal > zero
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
