"""
"""
module initial_conditions

export allocate_pdf_and_moments
export init_pdf_and_moments!
export enforce_r_boundary_condition!
export enforce_z_boundary_condition!
export enforce_vpa_boundary_condition!
export enforce_boundary_conditions!
export enforce_neutral_boundary_conditions!
export enforce_neutral_r_boundary_condition!
export enforce_neutral_z_boundary_condition!

# package
using SpecialFunctions: erfc
# modules
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..bgk: init_bgk_pdf!
using ..communication
using ..calculus: reconcile_element_boundaries_MPI!
using ..coordinates: coordinate
using ..external_sources
using ..interpolation: interpolate_to_grid_1d!
using ..looping
using ..moment_kinetics_structs: scratch_pdf
using ..velocity_moments: integrate_over_vspace, integrate_over_neutral_vspace
using ..velocity_moments: integrate_over_positive_vpa, integrate_over_negative_vpa
using ..velocity_moments: integrate_over_positive_vz, integrate_over_negative_vz
using ..velocity_moments: create_moments_charged, create_moments_neutral, update_qpar!
using ..velocity_moments: moments_charged_substruct, moments_neutral_substruct
using ..velocity_moments: update_neutral_density!, update_neutral_pz!, update_neutral_pr!, update_neutral_pzeta!
using ..velocity_moments: update_neutral_uz!, update_neutral_ur!, update_neutral_uzeta!, update_neutral_qz!
using ..velocity_moments: update_ppar!, update_upar!, update_density!, reset_moments_status!

using ..manufactured_solns: manufactured_solutions

using MPI

"""
"""
struct pdf_substruct{n_distribution}
    norm::MPISharedArray{mk_float,n_distribution}
    buffer::MPISharedArray{mk_float,n_distribution} # for collision operator terms when pdfs must be interpolated onto different velocity space grids
end

# struct of structs neatly contains i+n info?
struct pdf_struct
    #charged particles: s + r + z + vperp + vpa
    charged::pdf_substruct{5}
    #neutral particles: s + r + z + vzeta + vr + vz
    neutral::pdf_substruct{6}
end

struct moments_struct
    charged::moments_charged_substruct
    neutral::moments_neutral_substruct
    # flag that indicates if the density should be evolved via continuity equation
    evolve_density::Bool
    # flag that indicates if particle number should be conserved for each species
    # effects like ionisation or net particle flux from the domain would lead to
    # non-conservation
    particle_number_conserved::Bool
    # flag that indicates if exact particle conservation should be enforced
    enforce_conservation::Bool
    # flag that indicates if the parallel flow should be evolved via force balance
    evolve_upar::Bool
    # flag that indicates if the parallel pressure should be evolved via the energy equation
    evolve_ppar::Bool
end

struct boundary_distributions_struct
    # knudsen cosine distribution for imposing the neutral wall boundary condition
    knudsen::MPISharedArray{mk_float,3}
    # charged particle r boundary values (vpa,vperp,z,r,s)
    pdf_rboundary_charged::MPISharedArray{mk_float,5}
    # neutral particle r boundary values (vz,vr,vzeta,z,r,s)
    pdf_rboundary_neutral::MPISharedArray{mk_float,6}
end

"""
Creates the structs for the pdf and the velocity-space moments
"""
function allocate_pdf_and_moments(composition, r, z, vperp, vpa, vzeta, vr, vz,
                                  evolve_moments, collisions, external_source_settings,
                                  numerical_dissipation)
    pdf = create_pdf(composition, r, z, vperp, vpa, vzeta, vr, vz)

    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    # moments arrays have same r and z grids for both ion and neutral species
    # and so are included in the same struct
    charged = create_moments_charged(z.n, r.n, composition.n_ion_species,
        evolve_moments.density, evolve_moments.parallel_flow,
        evolve_moments.parallel_pressure, external_source_settings.ion,
        numerical_dissipation)
    neutral = create_moments_neutral(z.n, r.n, composition.n_neutral_species,
        evolve_moments.density, evolve_moments.parallel_flow,
        evolve_moments.parallel_pressure, external_source_settings.neutral,
        numerical_dissipation)

    if abs(collisions.ionization) > 0.0 || z.bc == "wall"
        # if ionization collisions are included or wall BCs are enforced, then particle
        # number is not conserved within each species
        particle_number_conserved = false
    else
        # by default, assumption is that particle number should be conserved for each species
        particle_number_conserved = true
    end

    moments = moments_struct(charged, neutral, evolve_moments.density,
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
    pdf_charged_norm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, composition.n_ion_species)
    pdf_charged_buffer = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, composition.n_neutral_species) # n.b. n_species is n_neutral_species here
    pdf_neutral_norm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, composition.n_neutral_species)
    pdf_neutral_buffer = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, composition.n_ion_species)

    return pdf_struct(pdf_substruct(pdf_charged_norm, pdf_charged_buffer),
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
                                             geometry, composition, species,
                                             manufactured_solns_input)
    else
        n_ion_species = composition.n_ion_species
        n_neutral_species = composition.n_neutral_species
        @serial_region begin
            # initialise the density profile
            init_density!(moments.charged.dens, z, r, species.charged, n_ion_species)
            # initialise the parallel flow profile
            init_upar!(moments.charged.upar, z, r, species.charged, n_ion_species)
            # initialise the parallel thermal speed profile
            init_vth!(moments.charged.vth, z, r, species.charged, n_ion_species)
            @. moments.charged.ppar = 0.5 * moments.charged.dens * moments.charged.vth^2

            if n_neutral_species > 0
                #neutral particles
                init_density!(moments.neutral.dens, z, r, species.neutral, n_neutral_species)
                init_uz!(moments.neutral.uz, z, r, species.neutral, n_neutral_species)
                init_ur!(moments.neutral.ur, z, r, species.neutral, n_neutral_species)
                init_uzeta!(moments.neutral.uzeta, z, r, species.neutral, n_neutral_species)
                init_vth!(moments.neutral.vth, z, r, species.neutral, n_neutral_species)
                @. moments.neutral.pz = 0.5 * moments.neutral.dens * moments.neutral.vth^2
                @. moments.neutral.ptot = 1.5 * moments.neutral.dens * moments.neutral.vth^2
            end
        end
        moments.charged.dens_updated .= true
        moments.charged.upar_updated .= true
        moments.charged.ppar_updated .= true
        moments.neutral.dens_updated .= true
        moments.neutral.uz_updated .= true
        moments.neutral.pz_updated .= true
        # create and initialise the normalised particle distribution function (pdf)
        # such that ∫dwpa pdf.norm = 1, ∫dwpa wpa * pdf.norm = 0, and ∫dwpa wpa^2 * pdf.norm = 1/2
        # note that wpa = vpa - upar, unless moments.evolve_ppar = true, in which case wpa = (vpa - upar)/vth
        # the definition of pdf.norm changes accordingly from pdf_unnorm / density to pdf_unnorm * vth / density
        # when evolve_ppar = true.
        initialize_pdf!(pdf, moments, boundary_distributions, composition, r, z, vperp,
                        vpa, vzeta, vr, vz, vpa_spectral, vz_spectral, species)
        begin_s_r_z_region()
        # calculate the initial parallel heat flux from the initial un-normalised pdf
        update_qpar!(moments.charged.qpar, moments.charged.qpar_updated,
                     moments.charged.dens, moments.charged.upar, moments.charged.vth,
                     pdf.charged.norm, vpa, vperp, z, r, composition,
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
            if species.charged[is].z_IC.initialization_option == "bgk" || species.charged[is].vpa_IC.initialization_option == "bgk"
                @views init_bgk_pdf!(pdf.charged.norm[:,1,:,ir,is], 0.0, species.charged[is].initial_temperature, z.grid, z.L, vpa.grid)
            else
                # updates pdf_norm to contain pdf / density, so that ∫dvpa pdf.norm = 1,
                # ∫dwpa wpa * pdf.norm = 0, and ∫dwpa m_s (wpa/vths)^2 pdf.norm = 1/2
                # to machine precision
                @views init_charged_pdf_over_density!(
                    pdf.charged.norm[:,:,:,ir,is], species.charged[is], composition, vpa, vperp,
                    z, vpa_spectral, moments.charged.dens[:,ir,is],
                    moments.charged.upar[:,ir,is], moments.charged.ppar[:,ir,is],
                    moments.charged.vth[:,ir,is],
                    moments.charged.v_norm_fac[:,ir,is], moments.evolve_density,
                    moments.evolve_upar, moments.evolve_ppar)
            end
            @views wall_flux_0[ir,is] = -(moments.charged.dens[1,ir,is] *
                                          moments.charged.upar[1,ir,is])
            @views wall_flux_L[ir,is] = moments.charged.dens[end,ir,is] *
                                        moments.charged.upar[end,ir,is]

            @loop_z iz begin
                if moments.evolve_ppar
                    @. pdf.charged.norm[:,:,iz,ir,is] *= moments.charged.vth[iz,ir,is]
                elseif moments.evolve_density == false
                    @. pdf.charged.norm[:,:,iz,ir,is] *= moments.charged.dens[iz,ir,is]
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
function init_charged_pdf_over_density!(pdf, spec, composition, vpa, vperp, z,
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
            # pdf and moments is taken care of by convert_full_f_charged_to_normalised!()
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
            # Can use non-shared memory here because `init_charged_pdf_over_density!()` is
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
            convert_full_f_charged_to_normalised!(pdf, density, upar, ppar, vth, vperp,
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
            # Can use non-shared memory here because `init_charged_pdf_over_density!()` is
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
                                                     species, r.n)
    dfni_func = manufactured_solns_list.dfni_func
    densi_func = manufactured_solns_list.densi_func
    dfnn_func = manufactured_solns_list.dfnn_func
    densn_func = manufactured_solns_list.densn_func
    #nb manufactured functions not functions of species
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        moments.charged.dens[iz,ir,is] = densi_func(z.grid[iz],r.grid[ir],0.0)
        @loop_vperp_vpa ivperp ivpa begin
            pdf.charged.norm[ivpa,ivperp,iz,ir,is] = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],0.0)
        end
    end
    # update upar, ppar, qpar, vth consistent with manufactured solns
    reset_moments_status!(moments)
    update_density!(moments.charged.dens, moments.charged.dens_updated,
                    pdf.charged.norm, vpa, vperp, z, r, composition)
    # get particle flux
    update_upar!(moments.charged.upar, moments.charged.upar_updated,
                 moments.charged.dens, moments.charged.ppar, pdf.charged.norm,
                 vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_ppar)
    update_ppar!(moments.charged.ppar, moments.charged.ppar_updated,
                 moments.charged.dens, moments.charged.upar, pdf.charged.norm,
                 vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_upar)
    update_qpar!(moments.charged.qpar, moments.charged.qpar_updated,
                 moments.charged.dens, moments.charged.upar,
                 moments.charged.vth, pdf.charged.norm, vpa, vperp, z, r,
                 composition, moments.evolve_density, moments.evolve_upar,
                 moments.evolve_ppar)
    # convert from particle particle flux to parallel flow
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        # update the thermal speed
        moments.charged.vth[iz,ir,is] = sqrt(2.0*moments.charged.ppar[iz,ir,is]/moments.charged.dens[iz,ir,is])
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
            moments.charged.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.ptot[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
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

function init_rboundary_pdfs!(rboundary_charged, rboundary_neutral, pdf::pdf_struct, vz,
                              vr, vzeta, vpa, vperp, z, r, composition)
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    n_neutral_species_alloc = max(1, n_neutral_species)

    begin_s_z_region() #do not parallelise r here
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        rboundary_charged[ivpa,ivperp,iz,1,is] = pdf.charged.norm[ivpa,ivperp,iz,1,is]
        rboundary_charged[ivpa,ivperp,iz,end,is] = pdf.charged.norm[ivpa,ivperp,iz,end,is]
    end
    if n_neutral_species > 0
        begin_sn_z_region() #do not parallelise r here
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            rboundary_neutral[ivz,ivr,ivzeta,iz,1,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,1,isn]
            rboundary_neutral[ivz,ivr,ivzeta,iz,end,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,end,isn]
        end
    end
    return rboundary_charged, rboundary_neutral
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
    pdf_rboundary_charged = allocate_shared_float(vpa.n, vperp.n, z.n, 2,
                                                  composition.n_ion_species)
    pdf_rboundary_neutral =  allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, 2,
                                                   composition.n_neutral_species)

    return boundary_distributions_struct(knudsen_cosine, pdf_rboundary_charged, pdf_rboundary_neutral)
end

function init_boundary_distributions!(boundary_distributions, pdf, vz, vr, vzeta, vpa, vperp, z, r, composition)
    #initialise fixed-in-time radial boundary condition based on initial condition values
    init_rboundary_pdfs!(boundary_distributions.pdf_rboundary_charged,
                         boundary_distributions.pdf_rboundary_neutral, pdf, vz, vr, vzeta,
                         vpa, vperp, z, r, composition)
    return nothing
end

"""
enforce boundary conditions in vpa and z on the evolved pdf;
also enforce boundary conditions in z on all separately evolved velocity space moments of the pdf
"""
function enforce_boundary_conditions!(f, f_r_bc, density, upar, ppar, moments, vpa_bc,
        z_bc, r_bc, vpa, vperp, z, r, vpa_adv, z_adv, r_adv, composition, scratch_dummy,
        r_diffusion, vpa_diffusion)

    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        # enforce the vpa BC
        # use that adv.speed independent of vpa
        @views enforce_v_boundary_condition_local!(f[:,ivperp,iz,ir,is], vpa_bc,
                                                   vpa_adv[is].speed[:,ivperp,iz,ir],
                                                   vpa_diffusion)
    end
    begin_s_r_vperp_vpa_region()
    # enforce the z BC on the evolved velocity space moments of the pdf
    @views enforce_z_boundary_condition_moments!(density, moments, z_bc)
    @views enforce_z_boundary_condition!(f, density, upar, ppar, moments, z_bc, z_adv, z,
                                         vperp, vpa, composition)
    if r.n > 1
        begin_s_z_vperp_vpa_region()
        @views enforce_r_boundary_condition!(f, f_r_bc, r_bc, r_adv, vpa, vperp, z, r, composition,
            scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
            scratch_dummy.buffer_vpavperpzs_3, scratch_dummy.buffer_vpavperpzs_4,
            scratch_dummy.buffer_vpavperpzrs_1, r_diffusion)
    end
end

function enforce_boundary_conditions!(fvec_out::scratch_pdf, moments, f_r_bc, vpa_bc,
        z_bc, r_bc, vpa, vperp, z, r, vpa_adv, z_adv, r_adv, composition, scratch_dummy,
        r_diffusion, vpa_diffusion)
    enforce_boundary_conditions!(fvec_out.pdf, f_r_bc, fvec_out.density, fvec_out.upar,
        fvec_out.ppar, moments, vpa_bc, z_bc, r_bc, vpa, vperp, z, r, vpa_adv, z_adv,
        r_adv, composition, scratch_dummy, r_diffusion, vpa_diffusion)
end

"""
enforce boundary conditions on f in r
"""
function enforce_r_boundary_condition!(f::AbstractArray{mk_float,5}, f_r_bc, bc::String,
        adv::T, vpa, vperp, z, r, composition, end1::AbstractArray{mk_float,4},
        end2::AbstractArray{mk_float,4}, buffer1::AbstractArray{mk_float,4},
        buffer2::AbstractArray{mk_float,4}, buffer_dfn::AbstractArray{mk_float,5}, r_diffusion::Bool) where T

    nr = r.n

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            end1[ivpa,ivperp,iz,is] = f[ivpa,ivperp,iz,1,is]
            end2[ivpa,ivperp,iz,is] = f[ivpa,ivperp,iz,nr,is]
        end
        @views reconcile_element_boundaries_MPI!(f,
            end1, end2,	buffer1, buffer2, r)
    end

    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    # enforce the condition if r is local
    if bc == "periodic" && r.nelement_global == r.nelement_local
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            f[ivpa,ivperp,iz,1,is] = 0.5*(f[ivpa,ivperp,iz,nr,is]+f[ivpa,ivperp,iz,1,is])
            f[ivpa,ivperp,iz,nr,is] = f[ivpa,ivperp,iz,1,is]
        end
    end
    if bc == "Dirichlet"
        zero = 1.0e-10
        # use the old distribution to force the new distribution to have
        # consistant-in-time values at the boundary
        # with bc = "Dirichlet" and r_diffusion = false
        # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
        # with bc = "Dirichlet" and r_diffusion = true
        # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            ir = 1 # r = -L/2 -- check that the point is on lowest rank
            if r.irank == 0 && (r_diffusion || adv[is].speed[ir,ivpa,ivperp,iz] > zero)
                f[ivpa,ivperp,iz,ir,is] = f_r_bc[ivpa,ivperp,iz,1,is]
            end
            ir = r.n # r = L/2 -- check that the point is on highest rank
            if r.irank == r.nrank - 1 && (r_diffusion || adv[is].speed[ir,ivpa,ivperp,iz] < -zero)
                f[ivpa,ivperp,iz,ir,is] = f_r_bc[ivpa,ivperp,iz,end,is]
            end
        end
    end
end

"""
enforce boundary conditions on charged particle f in z
"""
function enforce_z_boundary_condition!(pdf, density, upar, ppar, moments, bc::String, adv,
                                       z, vperp, vpa, composition)
    # define a zero that accounts for finite precision
    zero = 1.0e-14
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if bc == "constant"
        begin_s_r_vperp_vpa_region()
        density_offset = 1.0
        vwidth = 1.0
        if z.irank == 0
            @loop_s_r_vperp_vpa is ir ivperp ivpa begin
                if adv[is].speed[ivpa,1,ir] > 0.0
                    pdf[ivpa,ivperp,1,ir,is] = density_offset * exp(-(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)/vwidth^2) / sqrt(pi)
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_s_r_vperp_vpa is ir ivperp ivpa begin
                if adv[is].speed[ivpa,end,ir] > 0.0
                    pdf[ivpa,ivperp,end,ir,is] = density_offset * exp(-(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)/vwidth^2) / sqrt(pi)
                end
            end
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif bc == "periodic" && z.nelement_global == z.nelement_local
        begin_s_r_vperp_vpa_region()
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            pdf[ivpa,ivperp,1,ir,is] = 0.5*(pdf[ivpa,ivperp,z.n,ir,is]+pdf[ivpa,ivperp,1,ir,is])
            pdf[ivpa,ivperp,z.n,ir,is] = pdf[ivpa,ivperp,1,ir,is]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif bc == "wall"
        # Need integrals over vpa at wall boundaries in z, so cannot parallelize over z
        # or vpa.
        begin_s_r_region()
        @loop_s is begin
            # zero incoming BC for ions, as they recombine at the wall
            if moments.evolve_upar
                @loop_r ir begin
                    @views enforce_zero_incoming_bc!(
                        pdf[:,:,:,ir,is], z, vpa, density[:,ir,is], upar[:,ir,is],
                        ppar[:,ir,is], moments.evolve_upar, moments.evolve_ppar, zero)
                end
            else
                @loop_r ir begin
                    @views enforce_zero_incoming_bc!(pdf[:,:,:,ir,is],
                                                     adv[is].speed[:,:,:,ir], z, zero)
                end
            end
        end
    end
end

"""
enforce boundary conditions on neutral particle distribution function
"""
function enforce_neutral_boundary_conditions!(f_neutral, f_charged,
        boundary_distributions, density_neutral, uz_neutral, pz_neutral, moments,
        density_ion, upar_ion, Er, r_adv, z_adv, vzeta_adv, vr_adv, vz_adv, r, z, vzeta,
        vr, vz, composition, geometry, scratch_dummy, r_diffusion, vz_diffusion)

    if vzeta_adv !== nothing && vzeta.n_global > 1 && vzeta.bc != "none"
        begin_sn_r_z_vr_vz_region()
        @loop_sn_r_z_vr_vz isn ir iz ivr ivz begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[ivz,ivr,:,iz,ir,isn],
                                                       vzeta.bc,
                                                       vzeta_adv[isn].speed[ivz,ivr,:,iz,ir],
                                                       false)
        end
    end
    if vr_adv !== nothing && vr.n_global > 1 && vr.bc != "none"
        begin_sn_r_z_vzeta_vz_region()
        @loop_sn_r_z_vzeta_vz isn ir iz ivzeta ivz begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[ivz,:,ivzeta,iz,ir,isn],
                                                       vr.bc,
                                                       vr_adv[isn].speed[ivz,:,ivzeta,iz,ir],
                                                       false)
        end
    end
    if vz_adv !== nothing && vz.n_global > 1 && vz.bc != "none"
        begin_sn_r_z_vzeta_vr_region()
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[:,ivr,ivzeta,iz,ir,isn],
                                                       vz.bc,
                                                       vz_adv[isn].speed[:,ivr,ivzeta,iz,ir],
                                                       vz_diffusion)
        end
    end
    # f_initial contains the initial condition for enforcing a fixed-boundary-value condition
    # no bc on vz vr vzeta required as no advection in these coordinates
    begin_sn_r_vzeta_vr_vz_region()
    @views enforce_neutral_z_boundary_condition!(f_neutral, density_neutral, uz_neutral,
        pz_neutral, moments, density_ion, upar_ion, Er, boundary_distributions,
        z_adv, z, vzeta, vr, vz, composition, geometry)
    if r.n > 1
        begin_sn_z_vzeta_vr_vz_region()
        @views enforce_neutral_r_boundary_condition!(f_neutral, boundary_distributions.pdf_rboundary_neutral,
                                    r_adv, vz, vr, vzeta, z, r, composition,
                                    scratch_dummy.buffer_vzvrvzetazsn_1, scratch_dummy.buffer_vzvrvzetazsn_2,
                                    scratch_dummy.buffer_vzvrvzetazsn_3, scratch_dummy.buffer_vzvrvzetazsn_4,
                                    scratch_dummy.buffer_vzvrvzetazrsn_1, r_diffusion)
    end
end

function enforce_neutral_r_boundary_condition!(f::AbstractArray{mk_float,6},
        f_r_bc::AbstractArray{mk_float,6}, adv::T, vz, vr, vzeta, z, r, composition,
        end1::AbstractArray{mk_float,5}, end2::AbstractArray{mk_float,5},
        buffer1::AbstractArray{mk_float,5}, buffer2::AbstractArray{mk_float,5},
        buffer_dfn::AbstractArray{mk_float,6}, r_diffusion) where T #f_initial,

    bc = r.bc
    nr = r.n

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            end1[ivz,ivr,ivzeta,iz,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
            end2[ivz,ivr,ivzeta,iz,isn] = f[ivz,ivr,ivzeta,iz,nr,isn]
        end
        @views reconcile_element_boundaries_MPI!(f,
            end1, end2,	buffer1, buffer2, r)
    end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    # local case only when no communication required
    if bc == "periodic" && r.nelement_global == r.nelement_local
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            f[ivz,ivr,ivzeta,iz,1,isn] = 0.5*(f[ivz,ivr,ivzeta,iz,1,isn]+f[ivz,ivr,ivzeta,iz,nr,isn])
            f[ivz,ivr,ivzeta,iz,nr,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
        end
    end
    # Dirichlet boundary condition for external endpoints
    if bc == "Dirichlet"
        zero = 1.0e-10
        # use the old distribution to force the new distribution to have
        # consistant-in-time values at the boundary
        # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            ir = 1 # r = -L/2
            # incoming particles and on lowest rank
            if r.irank == 0 && (r_diffusion || adv[isn].speed[ir,ivz,ivr,ivzeta,iz] > zero)
                f[ivz,ivr,ivzeta,iz,ir,isn] = f_r_bc[ivz,ivr,ivzeta,iz,1,isn]
            end
            ir = nr # r = L/2
            # incoming particles and on highest rank
            if r.irank == r.nrank - 1 && (r_diffusion || adv[isn].speed[ir,ivz,ivr,ivzeta,iz] < -zero)
                f[ivz,ivr,ivzeta,iz,ir,isn] = f_r_bc[ivz,ivr,ivzeta,iz,end,isn]
            end
        end
    end
end

"""
enforce boundary conditions on neutral particle f in z
"""
function enforce_neutral_z_boundary_condition!(pdf, density, uz, pz, moments, density_ion,
                                               upar_ion, Er, boundary_distributions, adv,
                                               z, vzeta, vr, vz, composition, geometry)
    zero = 1.0e-14
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if z.bc == "constant"
        begin_sn_r_vzeta_vr_vz_region()
        density_offset = 1.0
        vwidth = 1.0
        if z.irank == 0
            @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
                if adv[isn].speed[ivz,ivr,ivzeta,1,ir] > 0.0
                    pdf[ivz,ivr,ivzeta,1,ir,is] = density_offset *
                        exp(-(vzeta.grid[ivzeta]^2 + vr.grid[ivr] + vz.grid[ivz])/vwidth^2) /
                        sqrt(pi)
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
                if adv[isn].speed[ivz,ivr,ivzeta,end,ir] > 0.0
                    pdf[ivz,ivr,ivzeta,end,ir,is] = density_offset *
                        exp(-(vzeta.grid[ivzeta]^2 + vr.grid[ivr] + vz.grid[ivz])/vwidth^2) /
                        sqrt(pi)
                end
            end
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif z.bc == "periodic" && z.nelement_global == z.nelement_local
        begin_sn_r_vzeta_vr_vz_region()
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            pdf[ivz,ivr,ivzeta,1,ir,isn] = 0.5*(pdf[ivz,ivr,ivzeta,1,ir,isn] +
                                                pdf[ivz,ivr,ivzeta,end,ir,isn])
            pdf[ivz,ivr,ivzeta,end,ir,isn] = pdf[ivz,ivr,ivzeta,1,ir,isn]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif z.bc == "wall"
        # Need integrals over vpa at wall boundaries in z, so cannot parallelize over z
        # or vpa.
        begin_sn_r_region()
        @loop_sn isn begin
            # BC for neutrals
            @loop_r ir begin
                # define vtfac to avoid repeated computation below
                vtfac = sqrt(composition.T_wall * composition.mn_over_mi)
                # Assume for now that the ion species index corresponding to this neutral
                # species is the same as the neutral species index.
                # Note, have already calculated moments of ion distribution function(s),
                # so can use the moments here to get the flux
                if z.irank == 0
                    ion_flux_0 = -density_ion[1,ir,isn] * (upar_ion[1,ir,isn]*geometry.bzed - 0.5*geometry.rhostar*Er[1,ir])
                else
                    ion_flux_0 = NaN
                end
                if z.irank == z.nrank - 1
                    ion_flux_L = density_ion[end,ir,isn] * (upar_ion[end,ir,isn]*geometry.bzed - 0.5*geometry.rhostar*Er[end,ir])
                else
                    ion_flux_L = NaN
                end
                # enforce boundary condition on the neutral pdf that all ions and neutrals
                # that leave the domain re-enter as neutrals
                @views enforce_neutral_wall_bc!(
                    pdf[:,:,:,:,ir,isn], z, vzeta, vr, vz, pz[:,ir,isn], uz[:,ir,isn],
                    density[:,ir,isn], ion_flux_0, ion_flux_L, boundary_distributions,
                    vtfac, composition.recycling_fraction, moments.evolve_ppar,
                    moments.evolve_upar, moments.evolve_density, zero)
            end
        end
    end
end

"""
enforce a zero incoming BC in z for given species pdf at each radial location
"""
function enforce_zero_incoming_bc!(pdf, speed, z, zero)
    nvpa = size(pdf,1)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa
    if z.irank == 0
        @loop_vperp_vpa ivperp ivpa begin
            # for left boundary in zed (z = -Lz/2), want
            # f(z=-Lz/2, v_parallel > 0) = 0
            if speed[1,ivpa,ivperp] > zero
                pdf[ivpa,ivperp,1] = 0.0
            end
        end
    end
    if z.irank == z.nrank - 1
        @loop_vperp_vpa ivperp ivpa begin
            # for right boundary in zed (z = Lz/2), want
            # f(z=Lz/2, v_parallel < 0) = 0
            if speed[end,ivpa,ivperp] < -zero
                pdf[ivpa,ivperp,end] = 0.0
            end
        end
    end
end
function enforce_zero_incoming_bc!(pdf, z::coordinate, vpa::coordinate, density, upar,
                                   ppar, evolve_upar, evolve_ppar, zero)
    if z.irank != 0 && z.irank != z.nrank - 1
        # No z-boundary in this block
        return nothing
    end
    nvpa, nvperp, nz = size(pdf)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    # absolute velocity at left boundary
    if z.irank == 0
        @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, sqrt(2.0*(ppar[1]/density[1])),
                                         upar[1], evolve_ppar, evolve_upar)
        @loop_vpa ivpa begin
            # for left boundary in zed (z = -Lz/2), want
            # f(z=-Lz/2, v_parallel > 0) = 0
            if vpa.scratch[ivpa] > zero
                pdf[ivpa,:,1] .= 0.0
            end
        end
    end
    # absolute velocity at right boundary
    if z.irank == z.nrank - 1
        @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, sqrt(2.0*(ppar[end]/density[end])),
                                          upar[end], evolve_ppar, evolve_upar)
        @loop_vpa ivpa begin
            # for right boundary in zed (z = Lz/2), want
            # f(z=Lz/2, v_parallel < 0) = 0
            if vpa.scratch2[ivpa] < -zero
                pdf[ivpa,:,end] .= 0.0
            end
        end
    end

    # Special constraint-forcing code that tries to keep the modifications smooth at
    # v_parallel=0.
    if z.irank == 0 && z.irank == z.nrank - 1
        # Both z-boundaries in this block
        z_range = (1,nz)
    elseif z.irank == 0
        z_range = (1,)
    elseif z.irank == z.nrank - 1
        z_range = (nz,)
    else
        error("No boundary in this block, should have returned already")
    end
    for iz ∈ z_range
        # moment-kinetic approach only implemented for 1V case so far
        @boundscheck size(pdf,2) == 1

        f = @view pdf[:,1,iz]
        if evolve_ppar && evolve_upar
            I0 = integrate_over_vspace(f, vpa.wgts)
            I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
            I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)

            # Store v_parallel with upar shift removed in vpa.scratch
            vth = sqrt(2.0*ppar[iz]/density[iz])
            @. vpa.scratch = vpa.grid + upar[iz]/vth
            # Introduce factor to ensure corrections go smoothly to zero near
            # v_parallel=0
            @. vpa.scratch2 = f * abs(vpa.scratch) / (1.0 + abs(vpa.scratch))
            J1 = integrate_over_vspace(vpa.scratch2, vpa.grid, vpa.wgts)
            J2 = integrate_over_vspace(vpa.scratch2, vpa.grid, 2, vpa.wgts)
            J3 = integrate_over_vspace(vpa.scratch2, vpa.grid, 3, vpa.wgts)
            J4 = integrate_over_vspace(vpa.scratch2, vpa.grid, 4, vpa.wgts)

            A = (J3^2 - J2*J4 + 0.5*(J2^2 - J1*J3)) /
                (I0*(J3^2 - J2*J4) + I1*(J1*J4 - J2*J3) + I2*(J2^2 - J1*J3))
            B = (0.5*J3 + A*(I1*J4 - I2*J3)) / (J3^2 - J2*J4)
            C = (0.5 - A*I2 -B*J3) / J4

            @. f = A*f + B*vpa.grid*vpa.scratch2 + C*vpa.grid*vpa.grid*vpa.scratch2
        elseif evolve_upar
            I0 = integrate_over_vspace(f, vpa.wgts)
            I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)

            # Store v_parallel with upar shift removed in vpa.scratch
            @. vpa.scratch = vpa.grid + upar[iz]
            # Introduce factor to ensure corrections go smoothly to zero near
            # v_parallel=0
            @. vpa.scratch2 = f * abs(vpa.scratch) / (1.0 + abs(vpa.scratch))
            J1 = integrate_over_vspace(vpa.scratch2, vpa.grid, vpa.wgts)
            J2 = integrate_over_vspace(vpa.scratch2, vpa.grid, 2, vpa.wgts)

            A = 1.0 / (I0 - I1*J1/J2)
            B = -A*I1/J2

            @. f = A*f + B*vpa.grid*vpa.scratch2
        elseif evolve_density
            I0 = integrate_over_vspace(f, vpa.wgts)
            @. f = f / I0
        end
    end
end

"""
Set up an initial condition that tries to be smoothly compatible with the sheath
boundary condition for ions, by setting f(±(v_parallel-u0)<0) where u0=0 at the sheath
boundaries and for z<0 increases linearly to u0=vpa.L at z=0, while for z>0 increases
from u0=-vpa.L at z=0 to zero at the z=z.L/2 sheath.

To be applied to 'full-f' distribution function on v_parallel grid (not w_parallel
grid).
"""
function enforce_initial_tapered_zero_incoming!(pdf, z::coordinate, vpa::coordinate)
    nvpa = size(pdf,1)
    zero = 1.0e-14
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    for iz ∈ 1:z.n
        u0 = (2.0*z.grid[iz]/z.L - sign(z.grid[iz])) * vpa.L / 2.0
        if z.grid[iz] < -zero
            for ivpa ∈ 1:nvpa
                if vpa.grid[ivpa] > u0 + zero
                    pdf[ivpa,iz] = 0.0
                end
            end
        elseif z.grid[iz] > zero
            for ivpa ∈ 1:nvpa
                if vpa.grid[ivpa] < u0 - zero
                    pdf[ivpa,iz] = 0.0
                end
            end
        end
    end
end

"""
enforce the wall boundary condition on neutrals;
i.e., the incoming flux of neutrals equals the sum of the ion/neutral outgoing fluxes
"""
function enforce_neutral_wall_bc!(pdf, z, vzeta, vr, vz, pz, uz, density, wall_flux_0,
                                  wall_flux_L, boundary_distributions, vtfac,
                                  recycling_fraction, evolve_ppar, evolve_upar,
                                  evolve_density, zero)

    # Reduce the ion flux by `recycling_fraction` to account for ions absorbed by the
    # wall.
    wall_flux_0 *= recycling_fraction
    wall_flux_L *= recycling_fraction

    if !evolve_density && !evolve_upar
        knudsen_cosine = boundary_distributions.knudsen

        if z.irank == 0
            ## treat z = -Lz/2 boundary ##
            # populate vz.scratch2 array with dz/dt values at z = -Lz/2
            vth = sqrt(2.0*pz[1]/density[1])
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[1], evolve_ppar, evolve_upar)

            # add the neutral species's contribution to the combined ion/neutral particle
            # flux out of the domain at z=-Lz/2
            @views wall_flux_0 += integrate_over_negative_vz(abs.(vz.scratch2) .* pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # for left boundary in zed (z = -Lz/2), want
            # f_n(z=-Lz/2, v_parallel > 0) = Γ_0 * f_KW(v_parallel)
            @loop_vz ivz begin
                if vz.scratch2[ivz] >= -zero
                    @views @. pdf[ivz,:,:,1] = wall_flux_0 * knudsen_cosine[ivz,:,:]
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##
            # populate vz.scratch2 array with dz/dt values at z = Lz/2
            vth = sqrt(2.0*pz[end]/density[end])
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[end], evolve_ppar, evolve_upar)

            # add the neutral species's contribution to the combined ion/neutral particle
            # flux out of the domain at z=-Lz/2
            @views wall_flux_L += integrate_over_positive_vz(abs.(vz.scratch2) .* pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # for right boundary in zed (z = Lz/2), want
            # f_n(z=Lz/2, v_parallel < 0) = Γ_Lz * f_KW(v_parallel)
            @loop_vz ivz begin
                if vz.scratch2[ivz] <= zero
                    @views @. pdf[ivz,:,:,end] = wall_flux_L * knudsen_cosine[ivz,:,:]
                end
            end
        end
    elseif !evolve_upar
        # Evolving density case
        knudsen_cosine = boundary_distributions.knudsen

        if z.irank == 0
            ## treat z = -Lz/2 boundary ##
            # populate vz.scratch2 array with dz/dt values at z = -Lz/2
            vth = sqrt(2.0*pz[1]/density[1])
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[1], evolve_ppar, evolve_upar)

            # Note the numerical integrol of knudsen_cosine was forced to be 1 (to machine
            # precision) when it was initialised.
            @views pdf_integral_0 = integrate_over_negative_vz(pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_0 = integrate_over_positive_vz(knudsen_cosine, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # Need to add incoming neutral flux to ion flux to get amplitude for Knudsen
            # distribution
            # account for the fact that the pdf here is the normalised pdf,
            # and the integration in wall_flux is defined relative to the un-normalised pdf
            @views wall_flux_0 += density[1] * pdf_integral_0

            # To update the pdf normalised by density, divide density out from the
            # prefactor for the Knudsen distribution
            normalised_knudsen_amplitude = wall_flux_0 / density[1]

            # Calculate normalisation factors N_in for the incoming and N_out for the
            # Knudsen parts of the distirbution so that ∫dwpa F = 1
            norm_fac = 1.0 / (pdf_integral_0 + normalised_knudsen_amplitude * knudsen_integral_0)

            # for left boundary in zed (z = -Lz/2), want
            # f_n(z=-Lz/2, v_parallel > 0) = Γ_0 * f_KW(v_parallel) * pdf_norm_fac(-Lz/2)
            @loop_vz ivz begin
                if vz.scratch2[ivz] >= -zero
                    @views @. pdf[ivz,:,:,1] = norm_fac * normalised_knudsen_amplitude * knudsen_cosine[ivz,:,:]
                else
                    @views @. pdf[ivz,:,:,1] *= norm_fac
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##
            # populate vz.scratch2 array with dz/dt values at z = Lz/2
            vth = sqrt(2.0*pz[end]/density[end])
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[end], evolve_ppar, evolve_upar)

            # Note the numerical integrol of knudsen_cosine was forced to be 1 (to machine
            # precision) when it was initialised.
            @views pdf_integral_0 = integrate_over_positive_vz(pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_0 = integrate_over_negative_vz(knudsen_cosine, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # Need to add incoming neutral flux to ion flux to get amplitude for Knudsen
            # distribution
            # account for the fact that the pdf here is the normalised pdf,
            # and the integration in wall_flux is defined relative to the un-normalised pdf
            @views wall_flux_L += density[end] * pdf_integral_0

            # To update the pdf normalised by density, divide density out from the
            # prefactor for the Knudsen distribution
            normalised_knudsen_amplitude = wall_flux_L / density[end]

            # Calculate normalisation factors N_in for the incoming and N_out for the
            # Knudsen parts of the distirbution so that ∫dwpa F = 1
            norm_fac = 1.0 / (pdf_integral_0 + normalised_knudsen_amplitude * knudsen_integral_0)

            # for right boundary in zed (z = Lz/2), want
            # f_n(z=Lz/2, v_parallel < 0) = Γ_Lz * f_KW(v_parallel) * pdf_norm_fac(Lz/2)
            @loop_vz ivz begin
                if vz.scratch2[ivz] <= zero
                    @views @. pdf[ivz,:,:,end] = norm_fac * normalised_knudsen_amplitude * knudsen_cosine[ivz,:,:]
                else
                    @views @. pdf[ivz,:,:,end] *= norm_fac
                end
            end
        end
    else
        if z.irank == 0
            ## treat z = -Lz/2 boundary ##
            # populate vz.scratch2 array with dz/dt values at z = -Lz/2
            vth = sqrt(2.0*pz[1]/density[1])
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[1], evolve_ppar, evolve_upar)

            # First apply boundary condition that total neutral outflux is equal to ion
            # influx to uz
            uz[1] = wall_flux_0 / density[1]
            #would setting density work better??
            #density[1] = - wall_flux_0 / uz[1]

            # Create normalised Knudsen cosine distribution, to use for positive v_parallel
            # at z = -Lz/2
            # Note this only makes sense for the 1V case with vr.n=vzeta.n=1
            @. vz.scratch = (3.0*pi/vtfac^3)*abs(vz.scratch2)*erfc(abs(vz.scratch2)/vtfac)

            # The v_parallel>0 part of the pdf is replaced by the Knudsen cosine
            # distribution. To ensure the constraints ∫dwpa wpa^m F = 0 are satisfied when
            # necessary, calculate a normalisation factor for the Knudsen distribution (in
            # vz.scratch) and correction terms for the incoming pdf similar to
            # enforce_moment_constraints!().
            #
            # Note that it seems to be important that this boundary condition not be
            # modified by the moment constraints, as that could cause numerical instability.
            # By ensuring that the constraints are satisfied already here,
            # enforce_moment_constraints!() will not change the pdf at the boundary. For
            # ions this is not an issue, because points set to 0 by the bc are not modified
            # from 0 by enforce_moment_constraints!().
            knudsen_integral_0 = integrate_over_positive_vz(vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_1 = integrate_over_positive_vz(vz.grid .* vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            @views pdf_integral_0 = integrate_over_negative_vz(pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views pdf_integral_1 = integrate_over_negative_vz(vz.grid .* pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            if !evolve_ppar
                # Calculate normalisation factors N_in for the incoming and N_out for the
                # Knudsen parts of the distirbution so that ∫dwpa F = 1 and ∫dwpa wpa F = 0
                # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = 0
                N_in = 1.0 / (pdf_integral_0 - pdf_integral_1/knudsen_integral_1*knudsen_integral_0)
                N_out = -N_in * pdf_integral_1 / knudsen_integral_1

                zero_vz_ind = 0
                for ivz ∈ 1:vz.n
                    if vz.scratch2[ivz] <= -zero
                        pdf[ivz,:,:,1] .= N_in*pdf[ivz,:,:,1]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_z = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @. pdf[ivz,:,:,1] = 0.5*(N_in*pdf[ivz,:,:,1] + N_out*vz.scratch[ivz])
                        else
                            pdf[ivz,:,:,1] .= N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ zero_vz_ind+1:vz.n
                    pdf[ivz,:,:,1] .= N_out*vz.scratch[ivz]
                end
            else
                knudsen_integral_2 = integrate_over_positive_vz(vz.grid .* vz.grid .* vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views pdf_integral_2 = integrate_over_negative_vz(vz.grid .* vz.grid .* pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views pdf_integral_3 = integrate_over_negative_vz(vz.grid .* vz.grid .* vz.grid .* pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                # Calculate normalisation factor N_out for the Knudsen part of the
                # distirbution and normalisation factor N_in and correction term C*wpa*F_in
                # for the incoming distribution so that ∫dwpa F = 1, ∫dwpa wpa F = 0, and
                # ∫dwpa wpa^2 F = 1/2
                # ⇒ N_in*pdf_integral_0 + C*pdf_integral_1 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + C*pdf_integral_2 + N_out*knudsen_integral_1 = 0
                #   N_in*pdf_integral_2 + C*pdf_integral_3 + N_out*knudsen_integral_2 = 1/2
                N_in = (0.5*knudsen_integral_0*pdf_integral_2 +
                        knudsen_integral_1*(pdf_integral_3 - 0.5*pdf_integral_1) -
                        knudsen_integral_2*pdf_integral_2) /
                       (knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) +
                        knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) +
                        knudsen_integral_2*(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2))
                N_out = -(N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2^2) + 0.5*pdf_integral_2) /
                         (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                C = (0.5 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)/pdf_integral_3

                zero_vz_ind = 0
                for ivz ∈ 1:vz.n
                    if vz.scratch2[ivz] <= -zero
                        @views @. pdf[ivz,:,:,1] = N_in*pdf[ivz,:,:,1] + C*vz.grid[ivz]*pdf[ivz,:,:,1]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,1] = 0.5*(N_in*pdf[ivz,:,:,1] +
                                                            C*vz.grid[ivz]*pdf[ivz,:,:,1] +
                                                            N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,1] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ zero_vz_ind+1:vz.n
                    @. pdf[ivz,:,:,1] = N_out*vz.scratch[ivz]
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##
            # populate vz.scratch2 array with dz/dt values at z = Lz/2
            vth = sqrt(2.0*pz[end]/density[end])
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[end], evolve_ppar, evolve_upar)

            # First apply boundary condition that total neutral outflux is equal to ion
            # influx to uz
            uz[end] = - wall_flux_L / density[end]
            #would setting density work better??
            #density[end] = - wall_flux_L / upar[end]

            # obtain the Knudsen cosine distribution at z = Lz/2
            # the z-dependence is only introduced if the peculiiar velocity is used as vz
            # Note this only makes sense for the 1V case with vr.n=vzeta.n=1
            @. vz.scratch = (3.0*pi/vtfac^3)*abs(vz.scratch2)*erfc(abs(vz.scratch2)/vtfac)

            # The v_parallel<0 part of the pdf is replaced by the Knudsen cosine
            # distribution. To ensure the constraint ∫dwpa wpa F = 0 is satisfied, multiply
            # the Knudsen distribution (in vz.scratch) by a normalisation factor given by
            # the integral (over negative v_parallel) of the outgoing Knudsen distribution
            # and (over positive v_parallel) of the incoming pdf.
            knudsen_integral_0 = integrate_over_negative_vz(vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_1 = integrate_over_negative_vz(vz.grid .* vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            @views pdf_integral_0 = integrate_over_positive_vz(pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views pdf_integral_1 = integrate_over_positive_vz(vz.grid .* pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            if !evolve_ppar
                # Calculate normalisation factors N_in for the incoming and N_out for the
                # Knudsen parts of the distirbution so that ∫dwpa F = 1 and ∫dwpa wpa F = 0
                # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = 0
                N_in = 1.0 / (pdf_integral_0 - pdf_integral_1/knudsen_integral_1*knudsen_integral_0)
                N_out = -N_in * pdf_integral_1 / knudsen_integral_1

                zero_vz_ind = 0
                for ivz ∈ vz.n:-1:1
                    if vz.scratch2[ivz] >= zero
                        @views @. pdf[ivz,:,:,end] = N_in*pdf[ivz,:,:,end]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,end] = 0.5*(N_in*pdf[ivz,:,:,end] + N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ 1:zero_vz_ind-1
                    @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                end
            else
                knudsen_integral_2 = integrate_over_negative_vz(vz.grid .* vz.grid .* vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views pdf_integral_2 = integrate_over_positive_vz(vz.grid .* vz.grid .* pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views pdf_integral_3 = integrate_over_positive_vz(vz.grid .* vz.grid .* vz.grid .* pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                # Calculate normalisation factor N_out for the Knudsen part of the
                # distirbution and normalisation factor N_in and correction term C*wpa*F_in
                # for the incoming distribution so that ∫dwpa F = 1, ∫dwpa wpa F = 0, and
                # ∫dwpa wpa^2 F = 1/2
                # ⇒ N_in*pdf_integral_0 + C*pdf_integral_1 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + C*pdf_integral_2 + N_out*knudsen_integral_1 = 0
                #   N_in*pdf_integral_2 + C*pdf_integral_3 + N_out*knudsen_integral_2 = 1/2
                N_in = (0.5*knudsen_integral_0*pdf_integral_2 +
                        knudsen_integral_1*(pdf_integral_3 - 0.5*pdf_integral_1) -
                        knudsen_integral_2*pdf_integral_2) /
                       (knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) +
                        knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) +
                        knudsen_integral_2*(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2))
                N_out = -(N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2^2) + 0.5*pdf_integral_2) /
                         (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                C = (0.5 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)/pdf_integral_3

                zero_vz_ind = 0
                for ivz ∈ vz.n:-1:1
                    if vz.scratch2[ivz] >= zero
                        @views @. pdf[ivz,:,:,end] = N_in*pdf[ivz,:,:,end] + C*vz.grid[ivz]*pdf[ivz,:,:,end]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,end] = 0.5*(N_in*pdf[ivz,:,:,end] +
                                                              C*vz.grid[ivz]*pdf[ivz,:,:,end] +
                                                              N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ 1:zero_vz_ind-1
                    @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                end
            end
        end
    end
end

"""
create an array of dz/dt values corresponding to the given vpagrid values
"""
function vpagrid_to_dzdt(vpagrid, vth, upar, evolve_ppar, evolve_upar)
    if evolve_ppar
        if evolve_upar
            return vpagrid .* vth .+ upar
        else
            return vpagrid .* vth
        end
    elseif evolve_upar
        return vpagrid .+ upar
    else
        return vpagrid
    end
end

"""
Take the full charged-particle distribution function, calculate the moments, then
normalise and shift to the moment-kinetic grid.

Uses input value of `f` and modifies in place to the normalised distribution functions.
Input `density`, `upar`, `ppar`, and `vth` are not used, the values are overwritten with
the moments of `f`.

Inputs/outputs depend on z, vperp, and vpa (should be inside loops over species, r)
"""
function convert_full_f_charged_to_normalised!(f, density, upar, ppar, vth, vperp, vpa,
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

"""
enforce the z boundary condition on the evolved velocity space moments of f
"""
function enforce_z_boundary_condition_moments!(density, moments, bc::String)
    ## TODO: parallelise
    #begin_serial_region()
    #@serial_region begin
    #    # enforce z boundary condition on density if it is evolved separately from f
    #	if moments.evolve_density
    #        # TODO: extend to 'periodic' BC case, as this requires further code modifications to be consistent
    #        # with finite difference derivatives (should be fine for Chebyshev)
    #        if bc == "wall"
    #            @loop_s_r is ir begin
    #                density[1,ir,is] = 0.5*(density[1,ir,is] + density[end,ir,is])
    #                density[end,ir,is] = density[1,ir,is]
    #        	end
    #        end
    #    end
    #end
end
"""
impose the prescribed vpa boundary condition on f
at every z grid point
"""
function enforce_vpa_boundary_condition!(f, bc, src, v_diffusion)
    nz = size(f,2)
    nr = size(f,3)
    for ir ∈ 1:nr
        for iz ∈ 1:nz
            enforce_v_boundary_condition_local!(view(f,:,iz,ir), bc, src.speed[:,iz,ir],
                                                v_diffusion)
        end
    end
end

"""
"""
function enforce_v_boundary_condition_local!(f, bc, speed, v_diffusion)
    if bc == "zero"
        if v_diffusion || speed[1] > 0.0
            # 'upwind' boundary
            f[1] = 0.0
        end
        if v_diffusion || speed[end] < 0.0
            # 'upwind' boundary
            f[end] = 0.0
        end
    elseif bc == "both_zero"
        f[1] = 0.0
        f[end] = 0.0
    elseif bc == "periodic"
        f[1] = 0.5*(f[1]+f[end])
        f[end] = f[1]
    end
end

end
