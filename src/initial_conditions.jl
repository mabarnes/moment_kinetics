"""
"""
module initial_conditions

export init_pdf_and_moments
export enforce_z_boundary_condition!
export enforce_vpa_boundary_condition!
export enforce_boundary_conditions!

# package
using SpecialFunctions: erfc
# modules
using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float
using ..bgk: init_bgk_pdf!
using ..communication
using ..coordinates: coordinate
using ..looping
using ..moment_kinetics_structs: scratch_pdf
using ..velocity_moments: integrate_over_vspace
using ..velocity_moments: integrate_over_positive_vpa, integrate_over_negative_vpa
using ..velocity_moments: create_moments, update_qpar!

"""
"""
struct pdf_struct
    norm::MPISharedArray{mk_float,4}
    unnorm::MPISharedArray{mk_float,4}
end

"""
creates the normalised pdf and the velocity-space moments and populates them
with a self-consistent initial condition
"""
function init_pdf_and_moments(vpa, z, r, composition, species, n_rk_stages, evolve_moments, ionization)
    # define the n_species variable for convenience
    n_species = composition.n_species
    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    moments = create_moments(z.n, r.n, n_species, evolve_moments, ionization, z.bc)
    @serial_region begin
        # initialise the density profile
        init_density!(moments.dens, z, r, species, n_species)
        # initialise the parallel flow profile
        init_upar!(moments.upar, z, r, species, n_species)
        # initialise the parallel thermal speed profile
        init_vth!(moments.vth, z, r, species, n_species)
        @. moments.ppar = 0.5 * moments.dens * moments.vth^2
    end
    moments.dens_updated .= true
    moments.upar_updated .= true
    moments.ppar_updated .= true
    # create and initialise the normalised particle distribution function (pdf)
    # such that ∫dwpa pdf.norm = 1, ∫dwpa wpa * pdf.norm = 0, and ∫dwpa wpa^2 * pdf.norm = 1/2
    # note that wpa = vpa - upar, unless moments.evolve_ppar = true, in which case wpa = (vpa - upar)/vth
    # the definition of pdf.norm changes accordingly from pdf.unnorm / density to pdf.unnorm * vth / density
    # when evolve_ppar = true.
    pdf = create_and_init_pdf(moments, vpa, z, r, n_species, species)
    begin_s_r_z_region()
    # calculate the initial parallel heat flux from the initial un-normalised pdf
    update_qpar!(moments.qpar, moments.qpar_updated, moments.dens, moments.upar,
                 moments.vth, pdf.norm, vpa, z, r, composition, moments.evolve_density,
                 moments.evolve_upar, moments.evolve_ppar)
    return pdf, moments
end

"""
"""
function create_and_init_pdf(moments, vpa, z, r, n_species, species)
    pdf_norm = allocate_shared_float(vpa.n, z.n, r.n, n_species)
    pdf_unnorm = allocate_shared_float(vpa.n, z.n, r.n, n_species)
    @serial_region begin
        for is ∈ 1:n_species
            for ir ∈ 1:r.n
                if species[is].z_IC.initialization_option == "bgk" || species[is].vpa_IC.initialization_option == "bgk"
                    @views init_bgk_pdf!(f[:,:,ir,is], 0.0, species[is].initial_temperature, z.grid, z.L, vpa.grid)
                else
                    # updates pdf_norm to contain pdf / density, so that ∫dvpa pdf.norm = 1,
                    # ∫dwpa wpa * pdf.norm = 0, and ∫dwpa m_s (wpa/vths)^2 pdf.norm = 1/2
                    # to machine precision
                    @views init_pdf_over_density!(pdf_norm[:,:,ir,is], species[is], vpa, z, moments.vth[:,ir,is],
                                                  moments.upar[:,ir,is], moments.vpa_norm_fac[:,ir,is],
                                                  moments.evolve_upar, moments.evolve_ppar)
                end
            end
        end
        pdf_unnorm .= pdf_norm
        for ivpa ∈ 1:vpa.n
            @. pdf_unnorm[ivpa,:,:,:] *= moments.dens
            if moments.evolve_ppar
                @. pdf_norm[ivpa,:,:,:] *= moments.vth
            elseif moments.evolve_density == false
                @. pdf_norm[ivpa,:,:,:] = pdf_unnorm[ivpa,:,:,:]
            end
        end
    end
    return pdf_struct(pdf_norm, pdf_unnorm)
end

"""
for now the only initialisation option for the temperature is constant in z
returns vth0 = sqrt(2Ts/ms) / sqrt(2Te/ms) = sqrt(Ts/Te)
"""
function init_vth!(vth, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            if spec[is].z_IC.initialization_option == "sinusoid"
                # initial condition is sinusoid in z
                @. vth[:,ir,is] =
                    sqrt(spec[is].initial_temperature
                         * (1.0 + spec[is].z_IC.temperature_amplitude
                                  * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L +
                                        spec[is].z_IC.temperature_phase)))
            else
                @. vth[:,ir,is] =  sqrt(spec[is].initial_temperature)
            end
        end
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
            elseif spec[is].z_IC.inititalization_option == "monomial"
                # linear variation in z, with offset so that
                # function passes through zero at upwind boundary
                @. dens[:,ir,is] = (z.grid + 0.5*z.L)^spec[is].z_IC.monomial_degree
            end
        end
    end
    return nothing
end

"""
for now the only initialisation option is zero parallel flow
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
                mid_ind = z.n ÷ 2
                if z.n % 2 == 0
                    z_midpoint = 0.5*(z.grid[mid_ind] + z.grid[mid_ind+1])
                else
                    # because ÷ does integer division (which floors the result), the
                    # actual index of the mid-point is mid_ind+1
                    z_midpoint = z.grid[mid_ind+1]
                end
                @. upar[:,ir,is] =
                    (spec[is].z_IC.upar_amplitude * 2.0 *
                           (z.grid[:] - z_midpoint)/z.L)
            else
                @. upar[:,ir,is] = 0.0
            end
        end
    end
    return nothing
end

"""
"""
function init_pdf_over_density!(pdf, spec, vpa, z, vth, upar, vpa_norm_fac, evolve_upar, evolve_ppar)
    if spec.vpa_IC.initialization_option == "gaussian"
        # initial condition is a Gaussian in the peculiar velocity
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
            @. pdf[:,iz] = exp(-vpa.scratch^2) / vth[iz]
        end
        for iz ∈ 1:z.n
            # densfac = the integral of the pdf over v-space, which should be unity,
            # but may not be exactly unity due to quadrature errors
            densfac = integrate_over_vspace(view(pdf,:,iz), vpa.wgts)
            # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
            # where w_s = vpa - upar_s;
            # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
            @views @. vpa.scratch2 = vpa.grid^2 * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^2
            #@views @. vpa.scratch2 = vpa.scratch^2 * pdf[:,iz]
            pparfac = integrate_over_vspace(vpa.scratch2, vpa.wgts)
            # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
            @views @. vpa.scratch2 = vpa.grid^2 *(vpa.grid^2/pparfac - 1.0/densfac) * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^4
            #@views @. vpa.scratch2 = vpa.scratch^2 *(vpa.scratch^2/pparfac - 1.0/densfac) * pdf[:,iz]
            pparfac2 = integrate_over_vspace(vpa.scratch2, vpa.wgts)

            @views @. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - 1.0/densfac)*pdf[:,iz]*(vpa_norm_fac[iz]/vth[iz])^2
            #@views @. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.scratch^2/pparfac - 1.0/densfac)*pdf[:,iz]
        end
    elseif spec.vpa_IC.initialization_option == "vpagaussian"
        for iz ∈ 1:z.n
            #@. pdf[:,iz] = vpa.grid^2*exp(-(vpa.grid*(vpa_norm_fac[iz]/vth[iz]))^2) / vth[iz]
            @. pdf[:,iz] = vpa.grid^2*exp(-(vpa.scratch)^2) / vth[iz]
        end
    elseif spec.vpa_IC.initialization_option == "sinusoid"
        # initial condition is sinusoid in vpa
        for iz ∈ 1:z.n
            @. pdf[:,iz] = spec.vpa_IC.amplitude*cospi(2.0*spec.vpa_IC.wavenumber*vpa.grid/vpa.L)
        end
    elseif spec.vpa_IC.initialization_option == "monomial"
        # linear variation in vpa, with offset so that
        # function passes through zero at upwind boundary
        for iz ∈ 1:z.n
            @. pdf[:,iz] = (vpa.grid + 0.5*vpa.L)^spec.vpa_IC.monomial_degree
        end
    end
    return nothing
end

"""
enforce boundary conditions in vpa and z on the evolved pdf;
also enforce boundary conditions in z on all separately evolved velocity space moments of the pdf
"""
function enforce_boundary_conditions!(f_out, density, upar, ppar, moments, vpa_bc, z_bc, vpa,
        z, r, vpa_adv, z_adv, composition)
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        # enforce the vpa BC
        @views enforce_vpa_boundary_condition_local!(f_out[:,iz,ir,is], vpa_bc, vpa_adv[is].upwind_idx[iz,ir],
                                                     vpa_adv[is].downwind_idx[iz,ir])
    end
    begin_s_r_vpa_region()
    # enforce the z BC on the evolved velocity space moments of the pdf
    @views enforce_z_boundary_condition_moments!(density, moments, z_bc)
    @views enforce_z_boundary_condition!(f_out, density, upar, ppar, moments, z_bc, z_adv, vpa, r, composition)

end
function enforce_boundary_conditions!(fvec_out::scratch_pdf, moments, vpa_bc, z_bc, vpa,
        z, r, vpa_adv, z_adv, composition)
    enforce_boundary_conditions!(fvec_out.pdf, fvec_out.density, fvec_out.upar,
        fvec_out.ppar, moments, vpa_bc, z_bc, vpa, z, r, vpa_adv, z_adv, composition)
end

"""
enforce boundary conditions on f in z
"""
function enforce_z_boundary_condition!(pdf, density, upar, ppar, moments, bc::String, adv::T, vpa, r, composition) where T
    # define n_species variable for convenience
    n_species = composition.n_species
    # define nvpa variable for convenience
    nvpa = vpa.n
    # define a zero that accounts for finite precision
    zero = 1.0e-10
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if bc == "constant"
        @loop_s_r_vpa is ir ivpa begin
            upwind_idx = adv[is].upwind_idx[ivpa,ir]
            pdf[ivpa,upwind_idx,ir,is] = density_offset * exp(-(vpa.grid[ivpa]/vpawidth)^2) / sqrt(pi)
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif bc == "periodic"
        @loop_s_r_vpa is ir ivpa begin
            downwind_idx = adv[is].downwind_idx[ivpa,ir]
            upwind_idx = adv[is].upwind_idx[ivpa,ir]
            pdf[ivpa,downwind_idx,ir,is] = 0.5*(pdf[ivpa,upwind_idx,ir,is]+pdf[ivpa,downwind_idx,ir,is])
            pdf[ivpa,upwind_idx,ir,is] = pdf[ivpa,downwind_idx,ir,is]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif bc == "wall"
        @loop_s is begin
            if is ∈ composition.ion_species_range
                # zero incoming BC for ions, as they recombine at the wall
                if moments.evolve_upar
                    @loop_r ir begin
                        @views enforce_zero_incoming_bc!(
                            pdf[:,:,ir,is], vpa, density[:,ir,is], upar[:,ir,is],
                            ppar[:,ir,is], moments.evolve_upar, moments.evolve_ppar,
                            zero)
                    end
                else
                    @loop_r ir begin
                        @views enforce_zero_incoming_bc!(pdf[:,:,ir,is], vpa, zero)
                    end
                end
            end
        end
        # BC for neutrals
        if composition.n_neutral_species > 0
            begin_serial_region()
            # TODO: parallelise this...
            @serial_region begin
                for ir ∈ 1:r.n
                    # define vtfac to avoid repeated computation below
                    vtfac = sqrt(composition.T_wall * composition.mn_over_mi)
                    # initialise the combined ion/neutral fluxes into the walls to be zero
                    ion_flux_0 = 0.0
                    ion_flux_L = 0.0
                    # if using moment-kinetic approach, will need to weight normalised pdf
                    # appearing in the wall_flux integrals by the particle density
                    pdf_norm_fac = 1.0
                    for is ∈ 1:composition.n_species
                        # include the contribution to the wall fluxes due to species with index 'is'
                        if is ∈ composition.ion_species_range
                            if moments.evolve_upar
                                # Flux to sheath boundary given by the moments
                                ion_flux_0 += density[1,ir,is] * upar[1,ir,is]
                                ion_flux_L += density[end,ir,is] * upar[end,ir,is]
                            else
                                ## treat the boundary at z = -Lz/2 ##
                                # create an array of dz/dt values at z = -Lz/2
                                vth = sqrt(2.0*ppar[1,ir,is]/density[1,ir,is])
                                @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth, upar[1,ir,is],
                                                                  moments.evolve_ppar, moments.evolve_upar)
                                # account for the fact that the pdf here is the normalised pdf,
                                # and the integration in wall_flux is defined relative to the un-normalised pdf
                                if moments.evolve_density
                                    pdf_norm_fac = density[1,ir,is]
                                end
                                # add this species' contribution to the combined ion/neutral particle flux out of the domain at z=-Lz/2
                                @views ion_flux_0 += (sqrt(composition.mn_over_mi) * pdf_norm_fac *
                                                      integrate_over_negative_vpa(abs.(vpa.scratch2) .* pdf[:,1,ir,is], vpa.scratch2, vpa.wgts, vpa.scratch))
                                ## treat the boundary at z = Lz/2 ##
                                # create an array of dz/dt values at z = Lz/2
                                vth = sqrt(2.0*ppar[end,ir,is]/density[end,ir,is])
                                @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth, upar[end,ir,is],
                                                                  moments.evolve_ppar, moments.evolve_upar)
                                # account for the fact that the pdf here is the normalised pdf,
                                # and the integration in wall_flux is defined relative to the un-normalised pdf
                                if moments.evolve_density
                                    pdf_norm_fac = density[end,ir,is]
                                end
                                # add this species' contribution to the combined ion/neutral particle flux out of the domain at z=Lz/2
                                @views ion_flux_L += (sqrt(composition.mn_over_mi) * pdf_norm_fac *
                                                      integrate_over_positive_vpa(abs.(vpa.scratch2) .* pdf[:,end,ir,is], vpa.scratch2, vpa.wgts, vpa.scratch))
                            end
                        end
                    end
                    # enforce boundary condition on the neutral pdf that all ions and neutrals
                    # that leave the domain re-enter as neutrals
                    for is ∈ composition.neutral_species_range
                        @views enforce_neutral_wall_bc!(
                            pdf[:,:,ir,is], vpa, ppar[:,ir,is], upar[:,ir,is],
                            density[:,ir,is], ion_flux_0, ion_flux_L, vtfac,
                            moments.evolve_ppar, moments.evolve_upar,
                            moments.evolve_density, zero)
                    end
                end
            end
        end
    end
end

"""
enforce a zero incoming BC in z for given species pdf at each radial location
"""
function enforce_zero_incoming_bc!(pdf, vpa::coordinate, zero)
    nvpa = size(pdf,1)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa
    @loop_vpa ivpa begin
        # for left boundary in zed (z = -Lz/2), want
        # f(z=-Lz/2, v_parallel > 0) = 0
        if vpa.grid[ivpa] > zero
            pdf[ivpa,1] = 0.0
        end
        # for right boundary in zed (z = Lz/2), want
        # f(z=Lz/2, v_parallel < 0) = 0
        if vpa.grid[ivpa] < -zero
            pdf[ivpa,end] = 0.0
        end
    end
end
function enforce_zero_incoming_bc!(pdf, vpa::coordinate, density, upar, ppar,
                                   evolve_upar, evolve_ppar, zero)
    nvpa = size(pdf,1)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    # absolute velocity at left boundary
    @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, sqrt(2.0*(ppar[1]/density[1])),
                                     upar[1], evolve_ppar, evolve_upar)
    # absolute velocity at right boundary
    @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, sqrt(2.0*(ppar[end]/density[end])),
                                      upar[end], evolve_ppar, evolve_upar)
    for ivpa ∈ 1:nvpa
        # for left boundary in zed (z = -Lz/2), want
        # f(z=-Lz/2, v_parallel > 0) = 0
        if vpa.scratch[ivpa] > zero
            pdf[ivpa,1] = 0.0
        end
        # for right boundary in zed (z = Lz/2), want
        # f(z=Lz/2, v_parallel < 0) = 0
        if vpa.scratch2[ivpa] < -zero
            pdf[ivpa,end] = 0.0
        end
    end
end

"""
enforce the wall boundary condition on neutrals;
i.e., the incoming flux of neutrals equals the sum of the ion/neutral outgoing fluxes
"""
function enforce_neutral_wall_bc!(pdf, vpa, ppar, upar, density, wall_flux_0,
                                  wall_flux_L, vtfac, evolve_ppar, evolve_upar,
                                  evolve_density, zero)
    nvpa = size(pdf,1)
    if !evolve_upar
        ## treat z = -Lz/2 boundary ##
        # populate vpa.scratch2 array with dz/dt values at z = -Lz/2
        vth = sqrt(2.0*ppar[1]/density[1])
        @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth, upar[1], evolve_ppar, evolve_upar)

        # Need to add incoming neutral flux to ion flux to get amplitude for Knudsen
        # distribution
        # account for the fact that the pdf here is the normalised pdf,
        # and the integration in wall_flux is defined relative to the un-normalised pdf
        if evolve_density
            pdf_norm_fac = density[1,ir,is]
        else
            pdf_norm_fac = 1.0
        end
        # add this species' contribution to the combined ion/neutral particle flux out of the domain at z=-Lz/2
        @views wall_flux_0 += pdf_norm_fac * integrate_over_negative_vpa(abs.(vpa.scratch2) .* pdf[:,1], vpa.scratch2, vpa.wgts, vpa.scratch)

        # obtain the Knudsen cosine distribution at z = -Lz/2
        # the z-dependence is only introduced if the peculiar velocity is used as vpa
        if evolve_ppar
            # Need to normalise velocities by vth
            @. vpa.scratch = (3.0*pi/vtfac^3)*abs(vpa.scratch2)*erfc(abs(vpa.scratch2)/vtfac)
        else
            @. vpa.scratch = (3.0*pi/vtfac^3)*abs(vpa.scratch2)*erfc(abs(vpa.scratch2)/vtfac)
        end
        # the integral of -v_parallel*f_{Kw} over positive v_parallel should be one,
        # but may not be exactly this due to quadrature errors;
        # ensure that this is true to machine precision to make sure particle number in/out of wall is conserved
        knudsen_norm_fac = integrate_over_positive_vpa(vpa.scratch2 .* vpa.scratch, vpa.scratch2, vpa.wgts, vpa.scratch3)
        @. vpa.scratch /= knudsen_norm_fac
        # depending on which moments (if any) are evolved, there is a different factor
        # multiplying the neutral pdf in the wall BC
        if evolve_ppar
            pdf_norm_fac = vth / density[1]
        elseif evolve_density
            pdf_norm_fac = 1.0 / density[1]
        else
            pdf_norm_fac = 1.0
        end
        # for left boundary in zed (z = -Lz/2), want
        # f_n(z=-Lz/2, v_parallel > 0) = Γ_0 * f_KW(v_parallel) * pdf_norm_fac(-Lz/2)
        for ivpa ∈ 1:nvpa
            if vpa.scratch2[ivpa] > zero
                pdf[ivpa,1] = wall_flux_0 * vpa.scratch[ivpa] * pdf_norm_fac
            end
        end

        ## treat the right boundary at z = Lz/2 ##
        # populate vpa.scratch2 array with dz/dt values at z = Lz/2
        vth = sqrt(2.0*ppar[end]/density[end])
        @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth, upar[end], evolve_ppar, evolve_upar)

        # Need to add incoming neutral flux to ion flux to get amplitude for Knudsen
        # distribution
        # account for the fact that the pdf here is the normalised pdf,
        # and the integration in wall_flux is defined relative to the un-normalised pdf
        if evolve_density
            pdf_norm_fac = density[end,ir,is]
        else
            pdf_norm_fac = 1.0
        end
        # add this species' contribution to the combined ion/neutral particle flux out of the domain at z=-Lz/2
        @views wall_flux_L += pdf_norm_fac * integrate_over_positive_vpa(abs.(vpa.scratch2) .* pdf[:,end], vpa.scratch2, vpa.wgts, vpa.scratch)

        # obtain the Knudsen cosine distribution at z = Lz/2
        # the z-dependence is only introduced if the peculiiar velocity is used as vpa
        @. vpa.scratch = (3.0*pi/vtfac^3)*abs(vpa.scratch2)*erfc(abs(vpa.scratch2)/vtfac)
        # the integral of -v_parallel*f_{Kw} over negative v_parallel should be one,
        # but may not be exactly this due to quadrature errors;
        # ensure that this is true to machine precision to make sure particle number in/out of wall is conserved
        knudsen_norm_fac = -integrate_over_negative_vpa(vpa.scratch2 .* vpa.scratch, vpa.scratch2, vpa.wgts, vpa.scratch3)
        @. vpa.scratch /= knudsen_norm_fac
        # depending on which moments (if any) are evolved, there is a different factor
        # multiplying the neutral pdf in the wall BC
        if evolve_ppar
            pdf_norm_fac = vth / density[end]
        elseif evolve_density
            pdf_norm_fac = 1.0 / density[end]
        else
            pdf_norm_fac = 1.0
        end
        # for right boundary in zed (z = Lz/2), want
        # f_n(z=Lz/2, v_parallel < 0) = Γ_Lz * f_KW(v_parallel) * pdf_norm_fac(Lz/2)
        for ivpa ∈ 1:nvpa
            if vpa.scratch2[ivpa] < -zero
                pdf[ivpa,end] = wall_flux_L * vpa.scratch[ivpa] * pdf_norm_fac
            end
        end
    else
        ## treat z = -Lz/2 boundary ##
        # populate vpa.scratch2 array with dz/dt values at z = -Lz/2
        vth = sqrt(2.0*ppar[1]/density[1])
        @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth, upar[1], evolve_ppar, evolve_upar)

        # First apply boundary condition that total neutral outflux is equat to ion
        # influx to upar
        upar[1] = - wall_flux_0 / density[1]

        # Create normalised Knudsen cosine distribution, to use for positive v_parallel
        # at z = -Lz/2
        @. vpa.scratch = (3.0*pi/vtfac^3)*abs(vpa.scratch2)*erfc(abs(vpa.scratch2)/vtfac)

        # The v_parallel>0 part of the pdf is replaced by the Knudsen cosine
        # distribution. To ensure the constraint ∫dwpa wpa F = 0 is satisfied, multiply
        # the Knudsen distribution (in vpa.scratch) by a normalisation factor given by
        # the integral (over positive v_parallel) of the outgoing Knudsen distribution
        # and (over negative v_parallel) of the incoming pdf.
        knudsen_integral = integrate_over_positive_vpa(vpa.grid .* vpa.scratch, vpa.scratch2, vpa.wgts, vpa.scratch3)

        @views pdf_integral = integrate_over_negative_vpa(vpa.grid .* pdf[:,1], vpa.scratch2, vpa.wgts, vpa.scratch3)
        knudsen_norm_fac = -pdf_integral / knudsen_integral
        # for left boundary in zed (z = -Lz/2), want
        # f_n(z=-Lz/2, v_parallel > 0) = knudsen_norm_fac * f_KW(v_parallel)
        zero_vpa_ind = 0
        for ivpa ∈ 1:nvpa
            if vpa.scratch2[ivpa] > -zero
                zero_vpa_ind = ivpa
                if abs(vpa.scratch2[ivpa]) < zero
                    # v_parallel = 0 point, half contribution from original pdf and half
                    # from Knudsen cosine distribution, to be consistent with weights
                    # used in
                    # integrate_over_positive_vpa()/integrate_over_negative_vpa().
                    pdf[ivpa,1] = 0.5 * (pdf[ivpa,1] + knudsen_norm_fac * vpa.scratch[ivpa])
                else
                    pdf[ivpa,1] = knudsen_norm_fac * vpa.scratch[ivpa]
                end
                break
            end
        end
        for ivpa ∈ zero_vpa_ind+1:nvpa
            pdf[ivpa,1] = knudsen_norm_fac * vpa.scratch[ivpa]
        end

        ## treat the right boundary at z = Lz/2 ##
        # populate vpa.scratch2 array with dz/dt values at z = Lz/2
        vth = sqrt(2.0*ppar[end]/density[end])
        @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth, upar[end], evolve_ppar, evolve_upar)

        # First apply boundary condition that total neutral outflux is equat to ion
        # influx to upar
        upar[end] = - wall_flux_L / density[end]

        # obtain the Knudsen cosine distribution at z = Lz/2
        # the z-dependence is only introduced if the peculiiar velocity is used as vpa
        @. vpa.scratch = (3.0*pi/vtfac^3)*abs(vpa.scratch2)*erfc(abs(vpa.scratch2)/vtfac)

        # The v_parallel<0 part of the pdf is replaced by the Knudsen cosine
        # distribution. To ensure the constraint ∫dwpa wpa F = 0 is satisfied, multiply
        # the Knudsen distribution (in vpa.scratch) by a normalisation factor given by
        # the integral (over negative v_parallel) of the outgoing Knudsen distribution
        # and (over positive v_parallel) of the incoming pdf.
        knudsen_integral = integrate_over_negative_vpa(vpa.grid .* vpa.scratch, vpa.scratch2, vpa.wgts, vpa.scratch3)

        @views pdf_integral = integrate_over_positive_vpa(vpa.grid .* pdf[:,end], vpa.scratch2, vpa.wgts, vpa.scratch3)
        knudsen_norm_fac = -pdf_integral / knudsen_integral
        # for right boundary in zed (z = Lz/2), want
        # f_n(z=Lz/2, v_parallel < 0) = knudsen_norm_fac * f_KW(v_parallel)
        zero_vpa_ind = 0
        for ivpa ∈ nvpa:-1:1
            if vpa.scratch2[ivpa] < zero
                zero_vpa_ind = ivpa
                if abs(vpa.scratch2[ivpa]) < zero
                    # v_parallel = 0 point, half contribution from original pdf and half
                    # from Knudsen cosine distribution, to be consistent with weights
                    # used in
                    # integrate_over_positive_vpa()/integrate_over_negative_vpa().
                    pdf[ivpa,end] = 0.5 * (pdf[ivpa,end] + knudsen_norm_fac * vpa.scratch[ivpa])
                else
                    pdf[ivpa,end] = knudsen_norm_fac * vpa.scratch[ivpa]
                end
                break
            end
        end
        for ivpa ∈ 1:zero_vpa_ind-1
            pdf[ivpa,end] = knudsen_norm_fac * vpa.scratch[ivpa]
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
enforce the z boundary condition on the evolved velocity space moments of f
"""
function enforce_z_boundary_condition_moments!(density, moments, bc::String)
    ## TODO: parallelise
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
function enforce_vpa_boundary_condition!(f, bc, src::T) where T
    nz = size(f,2)
    nr = size(f,3)
    for ir ∈ 1:nr
        for iz ∈ 1:nz
            enforce_vpa_boundary_condition_local!(view(f,:,iz,ir), bc, src.upwind_idx[iz],
                src.downwind_idx[iz])
        end
    end
end

"""
"""
function enforce_vpa_boundary_condition_local!(f::T, bc, upwind_idx, downwind_idx) where T
    if bc == "zero"
        f[upwind_idx] = 0.0
        #f[downwind_idx] = 0.0
    elseif bc == "periodic"
        f[downwind_idx] = 0.5*(f[upwind_idx]+f[downwind_idx])
        f[upwind_idx] = f[downwind_idx]
    end
end

end
