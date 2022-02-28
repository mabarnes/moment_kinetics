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
using ..looping
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
function init_pdf_and_moments(vpa, z, r, composition, species, n_rk_stages, evolve_moments)
    # define the n_species variable for convenience
    n_species = composition.n_species
    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    
    moments = create_moments(z.n, r.n, n_species, evolve_moments)
    @serial_region begin
        # initialise the density profile
        init_density!(moments.dens, z, r, species, n_species)
        moments.dens_updated .= true
        # initialise the parallel flow profile
        init_upar!(moments.upar, z, r, species, n_species)
        moments.upar_updated .= true
        # initialise the parallel thermal speed profile
        init_vth!(moments.vth, z, r, species, n_species)
        @. moments.ppar = 0.5 * moments.dens * moments.vth^2
        moments.ppar_updated .= true
    end
    # create and initialise the normalised particle distribution function (pdf)
    # such that ∫dwpa pdf.norm = 1, ∫dwpa wpa * pdf.norm = 0, and ∫dwpa wpa^2 * pdf.norm = 1/2
    # note that wpa = vpa - upar, unless moments.evolve_ppar = true, in which case wpa = (vpa - upar)/vth
    # the definition of pdf.norm changes accordingly from pdf.unnorm / density to pdf.unnorm * vth / density
    # when evolve_ppar = true.
    pdf = create_and_init_pdf(moments, vpa, z, r, n_species, species)
    begin_s_r_z_region()
    # calculate the initial parallel heat flux from the initial un-normalised pdf
    update_qpar!(moments.qpar, moments.qpar_updated, pdf.unnorm, vpa, z, r, composition, moments.vpa_norm_fac)
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
                @. upar[:,ir,is] =
                    (spec[is].z_IC.upar_amplitude * 2.0 *       
                           (z.grid[:] - z.grid[floor(Int,z.n/2)])/z.L)
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
        # if evolve_ppar = true, then vpa coordinate is (vpa - upar)/vth;
        # otherwise it is either (vpa-upar) or simply vpa
        for iz ∈ 1:z.n
            # obtain (vpa - upar)/vth
            if evolve_ppar
                @. vpa.scratch = vpa.grid
            elseif evolve_upar
                @. vpa.scratch = vpa.grid/vth[iz]
            else
                @. vpa.scratch = (vpa.grid - upar[iz])/vth[iz]
            end
            #@. pdf[:,iz] = exp(-(vpa.grid*(vpa_norm_fac[iz]/vth[iz]))^2) / vth[iz]
            @. pdf[:,iz] = exp(-vpa.scratch^2) / vth[iz]
        end
        for iz ∈ 1:z.n
            # densfac = the integral of the pdf over v-space, which should be unity,
            # but may not be exactly unity due to quadrature errors
            densfac = integrate_over_vspace(view(pdf,:,iz), vpa.wgts)
            # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
            # where w_s = vpa - upar_s;
            # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
            @views @. vpa.scratch = vpa.grid^2 * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^2
            pparfac = integrate_over_vspace(vpa.scratch, vpa.wgts)
            # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
            #@. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - vth[iz]^2/densfac) * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^4
            @views @. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - 1.0/densfac) * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^4
            pparfac2 = integrate_over_vspace(vpa.scratch, vpa.wgts)

            #@. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - vth[iz]^2/densfac)*pdf[:,iz]*(vpa_norm_fac[iz]/vth[iz])^2
            @views @. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - 1.0/densfac)*pdf[:,iz]*(vpa_norm_fac[iz]/vth[iz])^2
        end
    elseif spec.vpa_IC.initialization_option == "vpagaussian"
        for iz ∈ 1:z.n
            @. pdf[:,iz] = vpa.grid^2*exp(-(vpa.grid*(vpa_norm_fac[iz]/vth[iz]))^2) / vth[iz]
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
    # for iz ∈ 1:z.n
    #     # densfac = the integral of the pdf over v-space, which should be unity,
    #     # but may not be exactly unity due to quadrature errors
    #     densfac = integrate_over_vspace(view(pdf,:,iz), vpa.wgts)
    #     # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
    #     # where w_s = vpa - upar_s;
    #     # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
    #     @. vpa.scratch = vpa.grid^2 * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^2
    #     pparfac = integrate_over_vspace(vpa.scratch, vpa.wgts)
    #     # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
    #     #@. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - vth[iz]^2/densfac) * pdf[iz,:] * (vpa_norm_fac[iz]/vth[iz])^4
    #     @. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - 1.0/densfac) * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^4
    #     pparfac2 = integrate_over_vspace(vpa.scratch, vpa.wgts)
    #
    #     #@. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - vth[iz]^2/densfac)*pdf[:,iz]*(vpa_norm_fac[iz]/vth[iz])^2
    #     @. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - 1.0/densfac)*pdf[:,iz]*(vpa_norm_fac[iz]/vth[iz])^2
    # end
    return nothing
end

"""
"""
function enforce_boundary_conditions!(f, vpa_bc, z_bc, vpa, z, r, vpa_adv::T1, z_adv::T2, composition) where {T1, T2}
    @loop_s_r_z is ir iz begin
        # enforce the vpa BC
        @views enforce_vpa_boundary_condition_local!(f[:,iz,ir,is], vpa_bc, vpa_adv[is].upwind_idx[iz,ir],
                                                     vpa_adv[is].downwind_idx[iz,ir])
    end
    begin_s_r_vpa_region()
    @views enforce_z_boundary_condition!(f, z_bc, z_adv, vpa, r, composition)
end

"""
enforce boundary conditions on f in z
"""
function enforce_z_boundary_condition!(f, bc::String, adv::T, vpa, r, composition) where T
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
            f[ivpa,upwind_idx,ir,is] = density_offset * exp(-(vpa.grid[ivpa]/vpawidth)^2) / sqrt(pi)
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif bc == "periodic"
        @loop_s_r_vpa is ir ivpa begin
            downwind_idx = adv[is].downwind_idx[ivpa,ir]
            upwind_idx = adv[is].upwind_idx[ivpa,ir]
            f[ivpa,downwind_idx,ir,is] = 0.5*(f[ivpa,upwind_idx,ir,is]+f[ivpa,downwind_idx,ir,is])
            f[ivpa,upwind_idx,ir,is] = f[ivpa,downwind_idx,ir,is]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif bc == "wall"
        @loop_s is begin
            # zero incoming BC for ions, as they recombine at the wall
            if is ∈ composition.ion_species_range
                @loop_vpa ivpa begin
                    # no parallel BC should be enforced for vpa = 0
                    if abs(vpa.grid[ivpa]) > zero
                        @loop_r ir begin
                            upwind_idx = adv[is].upwind_idx[ivpa,ir]
                            f[ivpa,upwind_idx,ir,is] = 0.0
                        end
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
                    wall_flux_0 = 0.0
                    wall_flux_L = 0.0
                    # include the contribution to the wall fluxes due to species with index 'is'
                    for is ∈ 1:composition.n_ion_species
                            @views wall_flux_0 += (sqrt(composition.mn_over_mi) *
                                                   integrate_over_negative_vpa(abs.(vpa.grid) .* f[:,1,ir,is], vpa.grid, vpa.wgts, vpa.scratch))
                            @views wall_flux_L += (sqrt(composition.mn_over_mi) *
                                                   integrate_over_positive_vpa(abs.(vpa.grid) .* f[:,end,ir,is], vpa.grid, vpa.wgts, vpa.scratch))
                    end
                    for isn ∈ 1:composition.n_neutral_species
                            is = isn + composition.n_ion_species
                            @views wall_flux_0 += integrate_over_negative_vpa(abs.(vpa.grid) .* f[:,1,ir,is], vpa.grid, vpa.wgts, vpa.scratch)
                            @views wall_flux_L += integrate_over_positive_vpa(abs.(vpa.grid) .* f[:,end,ir,is], vpa.grid, vpa.wgts, vpa.scratch)
                    end
                    # NB: need to generalise to more than one ion species
                    # get the Knudsen cosine distribution
                    # NB: as vtfac is time-independent, can be made more efficient by creating
                    # array for Knudsen cosine distribution and carrying out following four lines
                    # of calculation at initialization
                    @. vpa.scratch = (3*pi/vtfac^3)*abs(vpa.grid)*erfc(abs(vpa.grid)/vtfac)
                    tmparr = copy(vpa.scratch)
                    tmp = integrate_over_positive_vpa(vpa.grid .* vpa.scratch, vpa.grid, vpa.wgts, tmparr)
                    @. vpa.scratch /= tmp
                    for isn ∈ 1:composition.n_neutral_species
                        is = isn + composition.n_ion_species
                        for ivpa ∈ 1:nvpa
                            # no parallel BC should be enforced for vpa = 0
                            if abs(vpa.grid[ivpa]) > zero
                                if adv[is].upwind_idx[ivpa,ir] == 1
                                    f[ivpa,1,ir,is] = wall_flux_0 * vpa.scratch[ivpa]
                                else
                                    f[ivpa,end,ir,is] = wall_flux_L * vpa.scratch[ivpa]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
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
