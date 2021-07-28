module initial_conditions

export init_pdf_and_moments
export enforce_z_boundary_condition!
export enforce_vpa_boundary_condition!
export enforce_boundary_conditions!

using ..type_definitions: mk_float
using ..array_allocation: allocate_float
using ..bgk: init_bgk_pdf!
using ..velocity_moments: integrate_over_vspace
using ..velocity_moments: create_moments, update_qpar!

struct pdf_struct
    norm::Array{mk_float,3}
    unnorm::Array{mk_float,3}
end

# creates the normalised pdf and the velocity-space moments and populates them
# with a self-consistent initial condition
function init_pdf_and_moments(vpa, z, composition, species, n_rk_stages, evolve_moments)
    # define the n_species variable for convenience
    n_species = composition.n_species
    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    moments = create_moments(z.n, n_species, evolve_moments)
    # initialise the density profile
    init_density!(moments.dens, z, species, n_species)
    moments.dens_updated .= true
    # initialise the parallel flow profile
    init_upar!(moments.upar, z, species, n_species)
    moments.upar_updated .= true
    # initialise the parallel thermal speed profile
    init_vth!(moments.vth, z, species, n_species)
    @. moments.ppar = 0.5 * moments.dens * moments.vth^2
    moments.ppar_updated .= true
    # create and initialise the normalised particle distribution function (pdf)
    # such that ∫dwpa pdf.norm = 1, ∫dwpa wpa * pdf.norm = 0, and ∫dwpa wpa^2 * pdf.norm = 1/2
    # note that wpa = vpa - upar, unless moments.evolve_ppar = true, in which case wpa = (vpa - upar)/vth
    # the definition of pdf.norm changes accordingly from pdf.unnorm / density to pdf.unnorm * vth / density
    # when evolve_ppar = true.
    pdf = create_and_init_pdf(moments, vpa, z, n_species, species)
    # calculae the initial parallel heat flux from the initial un-normalised pdf
    update_qpar!(moments.qpar, moments.qpar_updated, pdf.unnorm, vpa, z.n, moments.vpa_norm_fac)
    return pdf, moments
end
function create_and_init_pdf(moments, vpa, z, n_species, species)
    pdf_norm = allocate_float(vpa.n, z.n, n_species)
    for is ∈ 1:n_species
        if species[is].z_IC.initialization_option == "bgk" || species[is].vpa_IC.initialization_option == "bgk"
            @views init_bgk_pdf!(f[:,:,is], 0.0, species[is].initial_temperature, z.grid, z.L, vpa.grid)
        else
            # updates pdf_norm to contain pdf / density, so that ∫dvpa pdf.norm = 1,
            # ∫dwpa wpa * pdf.norm = 0, and ∫dwpa m_s (wpa/vths)^2 pdf.norm = 1/2
            # to machine precision
            @views init_pdf_over_density!(pdf_norm[:,:,is], species[is], vpa, z,
                                          moments.vth[:,is], moments.vpa_norm_fac[:,is])
        end
    end
    pdf_unnorm = copy(pdf_norm)
    for ivpa ∈ 1:vpa.n
        @. pdf_unnorm[ivpa,:,:] *= moments.dens
        if moments.evolve_ppar
            @. pdf_norm[ivpa,:,:] *= moments.vth
        elseif moments.evolve_density == false
            @. pdf_norm[ivpa,:,:] = pdf_unnorm[ivpa,:,:]
        end
    end
    return pdf_struct(pdf_norm, pdf_unnorm)
end
# for now the only initialisation option for the temperature is constant in z
# returns vth0 = sqrt(2Ts/ms) / sqrt(2Te/ms) = sqrt(Ts/Te)
function init_vth!(vth, z, spec, n_species)
    for is ∈ 1:n_species
        if spec[is].z_IC.initialization_option == "sinusoid"
            # initial condition is sinusoid in z
            @. vth[:,is] =
                sqrt(spec[is].initial_temperature
                     * (1.0 + spec[is].z_IC.temperature_amplitude
                              * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L +
                                    spec[is].z_IC.temperature_phase)))
        else
            @. vth[:,is] =  sqrt(spec[is].initial_temperature)
        end
    end
    return nothing
end
function init_density!(dens, z, spec, n_species)
    for is ∈ 1:n_species
        if spec[is].z_IC.initialization_option == "gaussian"
            # initial condition is an unshifted Gaussian
            @. dens[:,is] = spec[is].initial_density + exp(-(z.grid/spec[is].z_IC.width)^2)
        elseif spec[is].z_IC.initialization_option == "sinusoid"
            # initial condition is sinusoid in z
            @. dens[:,is] =
                (spec[is].initial_density
                 * (1.0 + spec[is].z_IC.density_amplitude
                          * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                                + spec[is].z_IC.density_phase)))
        elseif spec[is].z_IC.inititalization_option == "monomial"
            # linear variation in z, with offset so that
            # function passes through zero at upwind boundary
            @. dens[:,is] = (z.grid + 0.5*z.L)^spec[is].z_IC.monomial_degree
        end
    end
    return nothing
end
# for now the only initialisation option is zero parallel flow
function init_upar!(upar, z, spec, n_species)
    for is ∈ 1:n_species
        if spec[is].z_IC.initialization_option == "sinusoid"
            # initial condition is sinusoid in z
            @. upar[:,is] =
                (spec[is].z_IC.upar_amplitude
                 * cos(2.0*π*spec[is].z_IC.wavenumber*z.grid/z.L
                       + spec[is].z_IC.upar_phase))
        else
            @. upar[:,is] = 0.0
        end
    end
    return nothing
end
function init_pdf_over_density!(pdf, spec, vpa, z, vth, vpa_norm_fac)
    if spec.vpa_IC.initialization_option == "gaussian"
        # initial condition is an unshifted Gaussian
        # if evolve_ppar = true, then vpa coordinate is (vpa - upar)/vth;
        # otherwise it is either (vpa-upar) or simply vpa
        for iz ∈ 1:z.n
            @. pdf[:,iz] = exp(-(vpa.grid*(vpa_norm_fac[iz]/vth[iz]))^2) / vth[iz]
        end
    elseif spec.vpa_IC.initialization_option == "sinusoid"
        # initial condition is sinusoid in vpa
        for iz ∈ 1:z.n
            @. pdf[:,iz] =
                (spec.vpa_IC.density_amplitude
                 * cos(2.0*π*spec.vpa_IC.wavenumber*vpa.grid/vpa.L
                       + spec.vpa_IC.density_phase))
        end
    elseif spec.vpa_IC.initialization_option == "monomial"
        # linear variation in vpa, with offset so that
        # function passes through zero at upwind boundary
        for iz ∈ 1:z.n
            @. pdf[:,iz] = (vpa.grid + 0.5*vpa.L)^spec.vpa_IC.monomial_degree
        end
    end
    for iz ∈ 1:z.n
        # densfac = the integral of the pdf over v-space, which should be unity,
        # but may not be exactly unity due to quadrature errors
        densfac = integrate_over_vspace(view(pdf,:,iz), vpa.wgts)
        # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
        # where w_s = vpa - upar_s;
        # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
        @. vpa.scratch = vpa.grid^2 * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^2
        pparfac = integrate_over_vspace(vpa.scratch, vpa.wgts)
        # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
        #@. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - vth[iz]^2/densfac) * pdf[iz,:] * (vpa_norm_fac[iz]/vth[iz])^4
        @. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - 1.0/densfac) * pdf[:,iz] * (vpa_norm_fac[iz]/vth[iz])^4
        pparfac2 = integrate_over_vspace(vpa.scratch, vpa.wgts)

        #@. pdf[iz,:] = pdf[iz,:]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - vth[iz]^2/densfac)*pdf[iz,:]*(vpa_norm_fac[iz]/vth[iz])^2
        @. pdf[:,iz] = pdf[:,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - 1.0/densfac)*pdf[:,iz]*(vpa_norm_fac[iz]/vth[iz])^2
    end
    return nothing
end
function enforce_boundary_conditions!(f, vpa_bc, z_bc, vpa, vpa_adv::T1, z_adv::T2) where {T1, T2}
    for is ∈ 1:size(f,3)
        # enforce the z BC
        for ivpa ∈ 1:size(f,1)
            @views enforce_z_boundary_condition!(f[ivpa,:,is], z_bc, z_adv[ivpa,is].upwind_idx, z_adv[ivpa,is].downwind_idx, vpa[ivpa])
        end
    end
    for is ∈ 1:size(vpa_adv,2)
        # enforce the vpa BC
        for iz ∈ 1:size(f,2)
            @views enforce_vpa_boundary_condition_local!(f[:,iz,is], vpa_bc, vpa_adv[iz,is].upwind_idx, vpa_adv[iz,is].downwind_idx)
        end
    end
end
# impose the prescribed z boundary condition on f
# at every vpa grid point
function enforce_z_boundary_condition!(f, bc::String, vpa, src::T) where T
    for ivpa ∈ 1:size(src,1)
        enforce_z_boundary_condition!(view(f,ivpa,:), bc,
            src[ivpa].upwind_idx, src[ivpa].downwind_idx, vpa.grid[ivpa])
    end
end
# impose the prescribed z boundary conditin on f
# at a single vpa grid point
function enforce_z_boundary_condition!(f, bc, upwind_idx, downwind_idx, v)
    if bc == "constant"
        # BC is time-independent f at upwind boundary
        # and constant f beyond boundary
        f[upwind_idx] = density_offset * exp(-(v/vpawidth)^2) / sqrt(pi)
    elseif bc == "periodic"
        # impose periodicity
        f[downwind_idx] = 0.5*(f[upwind_idx]+f[downwind_idx])
        f[upwind_idx] = f[downwind_idx]
    end
end
# impose the prescribed vpa boundary condition on f
# at every z grid point
function enforce_vpa_boundary_condition!(f, bc, src::T) where T
    nz = size(f,2)
    for iz ∈ 1:nz
        enforce_vpa_boundary_condition_local!(view(f,:,iz), bc, src[iz].upwind_idx,
            src[iz].downwind_idx)
    end
end
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
