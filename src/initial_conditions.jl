"""
"""
module initial_conditions

export init_pdf_and_moments
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
using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float
using ..bgk: init_bgk_pdf!
using ..communication
using ..looping
using ..velocity_moments: integrate_over_vspace
using ..velocity_moments: integrate_over_positive_vpa, integrate_over_negative_vpa
using ..velocity_moments: create_moments_charged, create_moments_neutral, update_qpar!
using ..velocity_moments: moments_charged_substruct, moments_neutral_substruct

using ..manufactured_solns: manufactured_solutions

"""
"""
struct pdf_substruct{n_distribution}
    norm::MPISharedArray{mk_float,n_distribution}
    unnorm::MPISharedArray{mk_float,n_distribution}
end

# struct of structs neatly contains i+n info?
struct pdf_struct
    #charged particles: s + r + z + vperp + vpa
    charged::pdf_substruct{5}
    #neutral particles: s + r + z + vzeta + vr + vz
    neutral::pdf_substruct{6}
end

# need similar struct for moments? 
struct moments_struct
    charged::moments_charged_substruct
    neutral::moments_neutral_substruct
end
"""
creates the normalised pdf and the velocity-space moments and populates them
with a self-consistent initial condition
"""
function init_pdf_and_moments(vz, vr, vzeta, vpa, vperp, z, r, composition, species, n_rk_stages, evolve_moments, use_manufactured_solns)
    
    begin_serial_region()
    
    # define the n_species variable for convenience
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    # moments arrays have same r and z grids for both ion and neutral species 
    # and so are included in the same struct
    charged = create_moments_charged(z.n, r.n, n_ion_species)
    neutral = create_moments_neutral(z.n, r.n, n_neutral_species)
    moments = moments_struct(charged,neutral)
    
    @serial_region begin
        #charged particles
        # initialise the density profile
        init_density!(moments.charged.dens, z, r, species.charged, n_ion_species)
        # initialise the parallel flow profile
        init_upar!(moments.charged.upar, z, r, species.charged, n_ion_species)
        # initialise the parallel thermal speed profile
        init_vth!(moments.charged.vth, z, r, species.charged, n_ion_species)
        @. moments.charged.ppar = 0.5 * moments.charged.dens * moments.charged.vth^2
        if(n_neutral_species > 0)
            #neutral particles
            # initialise the density profile
            init_density!(moments.neutral.dens, z, r, species.neutral, n_neutral_species)
            init_uz!(moments.neutral.uz, z, r, species.neutral, n_neutral_species)
            init_ur!(moments.neutral.ur, z, r, species.neutral, n_neutral_species)
            init_uzeta!(moments.neutral.uzeta, z, r, species.neutral, n_neutral_species)
            init_vth!(moments.neutral.vth, z, r, species.neutral, n_neutral_species)
        end
    end
    # create and initialise the normalised particle distribution function (pdf)
    pdf = create_and_init_pdf(moments, vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species, species)
    
    if(use_manufactured_solns)
        manufactured_solns_list = manufactured_solutions(r.L,z.L,r.bc,z.bc) 
        dfni_func = manufactured_solns_list.dfni_func
        densi_func = manufactured_solns_list.densi_func
        dfnn_func = manufactured_solns_list.dfnn_func
        densn_func = manufactured_solns_list.densn_func
        #nb manufactured functions not functions of species
        for is in 1:n_ion_species
            for ir in 1:r.n
                for iz in 1:z.n
                    moments.charged.dens[iz,ir,is] = densi_func(z.grid[iz],r.grid[ir],0.0)
                    # other moments here 
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                            pdf.charged.unnorm[ivpa,ivperp,iz,ir,is] = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],0.0)
                            pdf.charged.norm[ivpa,ivperp,iz,ir,is] = pdf.charged.unnorm[ivpa,ivperp,iz,ir,is]
                            #same for neutrals here
                        end
                    end
                end
            end
        end
    end 
    #begin_s_r_z_vperp_region()
    # calculate the initial parallel heat flux from the initial un-normalised pdf
    update_qpar!(moments.charged.qpar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    # need neutral version!!! update_qpar!(moments.charged.qpar, moments.charged.qpar_updated, pdf.charged.unnorm, vpa, vperp, z, r, composition, moments.charged.vpa_norm_fac)
    return pdf, moments
end

"""
"""
function create_and_init_pdf(moments, vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species, species)
    # allocate pdf arrays
    pdf_charged_norm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, n_ion_species)
    pdf_charged_unnorm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, n_ion_species)
    #if n_neutral_species > 0
        pdf_neutral_norm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, n_neutral_species)
        pdf_neutral_unnorm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, n_neutral_species)
    #end
    @serial_region begin
        for is ∈ 1:n_ion_species
            for ir ∈ 1:r.n
                # updates pdf_norm to contain pdf / density, so that ∫dvpa pdf.norm = 1,
                # ∫dwpa wpa * pdf.norm = 0, and ∫dwpa m_s (wpa/vths)^2 pdf.norm = 1/2
                # to machine precision
                @views init_pdf_charged_over_density!(pdf_charged_norm[:,:,:,ir,is], species.charged[is], vpa, vperp, z, moments.charged.vth[:,ir,is],
                                              moments.charged.upar[:,ir,is])
            end
        end
        if n_neutral_species > 0
            for is ∈ 1:n_neutral_species
                for ir ∈ 1:r.n
                    @views init_pdf_neutral_over_density!(pdf_neutral_norm[:,:,:,:,ir,is], species.neutral[is], vz, vr, vzeta, z, moments.neutral.uz, moments.neutral.ur, moments.neutral.uzeta, moments.neutral.vth)
                end
            end
        end
        #set unnorm pdf = norm pdf * dens
        
        for ivperp ∈ 1:vperp.n
            for ivpa ∈ 1:vpa.n
                @. pdf_charged_unnorm[ivpa,ivperp,:,:,:] = pdf_charged_norm[ivpa,ivperp,:,:,:] .* moments.charged.dens[:,:,:]
            end
        end
        if n_neutral_species > 0 
            for ivzeta in 1:vzeta.n
                for ivr in 1:vr.n
                    for ivz in 1:vz.n
                        @. pdf_neutral_unnorm[ivz,ivr,ivzeta,:,:,:] = pdf_neutral_norm[ivz,ivr,ivzeta,:,:,:] .* moments.neutral.dens[:,:,:]
                    end
                end
            end
        end
    end
    return pdf_struct(pdf_substruct(pdf_charged_norm, pdf_charged_unnorm),pdf_substruct(pdf_neutral_norm, pdf_neutral_unnorm))
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
function init_uz!(uz, z, r, spec, n_species)
    for is ∈ 1:n_species
        for ir ∈ 1:r.n
            @. uz[:,ir,is] = 0.0
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
function init_pdf_charged_over_density!(pdf, spec, vpa, vperp, z, vth, upar)
    if spec.vpa_IC.initialization_option == "gaussian"
        # initial condition is a Gaussian in the peculiar velocity
        for iz ∈ 1:z.n
            # obtain (vpa - upar)/vth
            @. vpa.scratch = (vpa.grid - upar[iz])/vth[iz]
            
            @. vperp.scratch = vperp.grid/vth[iz]
            
            if vperp.n == 1 # 1D case 
                ivperp = 1 
                for ivpa ∈ 1:vpa.n
                    pdf[ivpa,ivperp,iz] = exp(-vpa.scratch[ivpa]^2 - vperp.scratch[ivperp]^2  ) / vth[iz]
                end
            else # 2D case with vperp
                for ivperp ∈ 1:vperp.n
                    for ivpa ∈ 1:vpa.n
                        pdf[ivpa,ivperp,iz] = exp(-vpa.scratch[ivpa]^2 - vperp.scratch[ivperp]^2  ) / vth[iz]^3
                    end
                end
            end 
        end
        
    elseif spec.vpa_IC.initialization_option == "sinusoid"
        # initial condition is sinusoid in vpa
        for iz ∈ 1:z.n
            for ivperp ∈ 1:vperp.n
                @. pdf[:,ivperp,iz] = spec.vpa_IC.amplitude*cospi(2.0*spec.vpa_IC.wavenumber*vpa.grid/vpa.L)
            end
        end
    elseif spec.vpa_IC.initialization_option == "monomial"
        # linear variation in vpa, with offset so that
        # function passes through zero at upwind boundary
        for iz ∈ 1:z.n
            for ivperp ∈ 1:vperp.n
                @. pdf[:,ivperp,iz] = (vpa.grid + 0.5*vpa.L)^spec.vpa_IC.monomial_degree
            end
        end
    end
    return nothing
end

"""
"""

function init_pdf_neutral_over_density!(pdf, spec, vz, vr, vzeta, z, uz, ur, uzeta, vth)

    # initial condition is a Gaussian in the peculiar velocity
    for iz ∈ 1:z.n
        # obtain (vpa - upar)/vth
        @. vz.scratch = (vz.grid - uz[iz])/vth[iz]
        @. vr.scratch = (vr.grid - ur[iz])/vth[iz]
        @. vzeta.scratch = (vzeta.grid - uzeta[iz])/vth[iz]
        
        if vr.n == 1 && vzeta.n == 1 # 1D case 
            ivr = 1
            ivzeta = 1 
            for ivz ∈ 1:vz.n
                pdf[ivz,ivr,ivzeta,iz] = exp(-vz.scratch[ivz]^2 - vr.scratch[ivr]^2 - vzeta.scratch[ivr]^2  ) / vth[iz]
            end
        else # 3D case with vr & vzeta
            for ivzeta ∈ 1:vzeta.n
                for ivr ∈ 1:vr.n
                    for ivz ∈ 1:vz.n
                        pdf[ivz,ivr,ivzeta,iz] = exp(-vz.scratch[ivz]^2 - vr.scratch[ivr]^2 - vzeta.scratch[ivr]^2  ) / vth[iz]^3
                    end
                end
            end
        end 
    end
    
    return nothing
end


# The "where" syntax here is confusing...
# Can we have an explicit loop or explain what it does?
# Seems to make the fn a operator elementwise in the vector f
# but loop is over a part of mysterious x_adv objects...
function enforce_boundary_conditions!(f, f_old, vpa_bc, z_bc, r_bc, vpa, vperp, z, r, vpa_adv::T1, z_adv::T2, r_adv::T3, composition) where {T1, T2, T3}
    
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        # enforce the vpa BC
        @views enforce_vpa_boundary_condition_local!(f[:,ivperp,iz,ir,is], vpa_bc, vpa_adv[is].upwind_idx[ivperp,iz,ir],
                                                     vpa_adv[is].downwind_idx[ivperp,iz,ir])
    end
    begin_s_r_vperp_vpa_region()
    @views enforce_z_boundary_condition!(f, z_bc, z_adv, vpa, vperp, r, composition)
    begin_s_z_vperp_vpa_region()
    @views enforce_r_boundary_condition!(f, f_old, r_bc, r_adv, vpa, vperp, z, r, composition)
end


"""
enforce boundary conditions on f in r
"""
function enforce_r_boundary_condition!(f, f_old, bc::String, adv::T, vpa, vperp, z, r, composition) where T
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    if bc == "periodic"
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            downwind_idx = adv[is].downwind_idx[ivpa,ivperp,iz] # 1 #
            upwind_idx = adv[is].upwind_idx[ivpa,ivperp,iz] # r.n #
            f[ivpa,ivperp,iz,downwind_idx,is] = 0.5*(f[ivpa,ivperp,iz,upwind_idx,is]+f[ivpa,ivperp,iz,downwind_idx,is])
            f[ivpa,ivperp,iz,upwind_idx,is] = f[ivpa,ivperp,iz,downwind_idx,is]
        end
    elseif bc == "Dirichlet"
        # use the old distribution to force the new distribution to have 
        # consistant-in-time values at the boundary
        # impose bc on upwind boundary only (Hyperbolic PDE)
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            upwind_idx = adv[is].upwind_idx[ivpa,ivperp,iz] # r.n #
            f[ivpa,ivperp,iz,upwind_idx,is] = f_old[ivpa,ivperp,iz,upwind_idx,is]
        end
    end
end
"""
enforce boundary conditions on f in z
"""
function enforce_z_boundary_condition!(f, bc::String, adv::T, vpa, vperp, r, composition) where T
    # define n_species variable for convenience
    n_species = composition.n_species
    # define nvpa variable for convenience
    nvpa = vpa.n
    # define nz variable for convenience
    nz = size(f, 3)
    # define a zero that accounts for finite precision
    zero = 1.0e-10
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if bc == "constant"
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            upwind_idx = adv[is].upwind_idx[ivpa,ivperp,ir]
            f[ivpa,ivperp,upwind_idx,ir,is] = density_offset * exp(-(vpa.grid[ivpa]/vpawidth)^2) / sqrt(pi)
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif bc == "periodic"
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            downwind_idx = adv[is].downwind_idx[ivpa,ivperp,ir]
            upwind_idx = adv[is].upwind_idx[ivpa,ivperp,ir]
            f[ivpa,ivperp,downwind_idx,ir,is] = 0.5*(f[ivpa,ivperp,upwind_idx,ir,is]+f[ivpa,ivperp,downwind_idx,ir,is])
            f[ivpa,ivperp,upwind_idx,ir,is] = f[ivpa,ivperp,downwind_idx,ir,is]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif bc == "wall"
        @loop_s is begin
            # zero incoming BC for ions, as they recombine at the wall
            if is ∈ composition.ion_species_range
                @loop_r_vperp_vpa ir ivperp ivpa begin
                    # no parallel BC should be enforced for vpa = 0
                    # adv.speed is signed 
                    # adv.speed =  vpa*kpar - 0.5 *rhostar*Er
                    
                    iz = 1 # z = -L/2
                    if adv[is].speed[iz,ivpa,ivperp,ir] > zero
                        f[ivpa,ivperp,iz,ir,is] = 0.0
                    end
                    iz = nz # z = L/2
                    if adv[is].speed[iz,ivpa,ivperp,ir] < -zero
                        f[ivpa,ivperp,iz,ir,is] = 0.0
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
                    for ivperp ∈ 1:vperp.n
                        # define vtfac to avoid repeated computation below
                        vtfac = sqrt(composition.T_wall * composition.mn_over_mi)
                        # initialise the combined ion/neutral fluxes into the walls to be zero
                        wall_flux_0 = 0.0
                        wall_flux_L = 0.0
                        # include the contribution to the wall fluxes due to species with index 'is'
                        for is ∈ 1:composition.n_ion_species
                                @views wall_flux_0 += (sqrt(composition.mn_over_mi) *
                                                       integrate_over_negative_vpa(abs.(vpa.grid) .* f[:,ivperp,1,ir,is], vpa.grid, vpa.wgts, vpa.scratch))
                                @views wall_flux_L += (sqrt(composition.mn_over_mi) *
                                                       integrate_over_positive_vpa(abs.(vpa.grid) .* f[:,ivperp,end,ir,is], vpa.grid, vpa.wgts, vpa.scratch))
                        end
                        for isn ∈ 1:composition.n_neutral_species
                                is = isn + composition.n_ion_species
                                @views wall_flux_0 += integrate_over_negative_vpa(abs.(vpa.grid) .* f[:,ivperp,1,ir,is], vpa.grid, vpa.wgts, vpa.scratch)
                                @views wall_flux_L += integrate_over_positive_vpa(abs.(vpa.grid) .* f[:,ivperp,end,ir,is], vpa.grid, vpa.wgts, vpa.scratch)
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
                                        f[ivpa,ivperp,1,ir,is] = wall_flux_0 * vpa.scratch[ivpa]
                                    else
                                        f[ivpa,ivperp,end,ir,is] = wall_flux_L * vpa.scratch[ivpa]
                                    end
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
    nvperp = size(f,2)
    nz = size(f,3)
    nr = size(f,4)
    for ir ∈ 1:nr
        for iz ∈ 1:nz
            for ivperp ∈ 1:nvperp
                enforce_vpa_boundary_condition_local!(view(f,:,ivperp,iz,ir), bc, src.upwind_idx[ivperp,iz,ir],
                src.downwind_idx[ivperp,iz,ir])
            end
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

function enforce_neutral_boundary_conditions!(f, r_adv::T, vz, vr, vzeta, z, r, composition) where T #f_initial,
    
    # f_initial contains the initial condition for enforcing a fixed-boundary-value condition 
    # no bc on vz vr vzeta required as no advection in these coordinates
    begin_sn_r_vzeta_vr_vz_region()
    @views enforce_neutral_z_boundary_condition!(f, vz, vr, vzeta, z, r, composition)
    begin_sn_z_vzeta_vr_vz_region()
    @views enforce_neutral_r_boundary_condition!(f, r_adv, vz, vr, vzeta, z, r, composition) #f_initial, 
end

function enforce_neutral_z_boundary_condition!(f, vz, vr, vzeta, z, r, composition)
    bc = z.bc
    nz = z.n
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    if bc == "periodic"
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            f[ivz,ivr,ivzeta,1,ir,isn] = 0.5*(f[ivz,ivr,ivzeta,1,ir,isn]+f[ivz,ivr,ivzeta,nz,ir,isn])
            f[ivz,ivr,ivzeta,nz,ir,isn] = f[ivz,ivr,ivzeta,1,ir,isn]
        end
    end
end

function enforce_neutral_r_boundary_condition!(f, adv::T, vz, vr, vzeta, z, r, composition) where T #f_initial, 
    bc = r.bc
    nr = r.n
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    if bc == "periodic"
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            f[ivz,ivr,ivzeta,iz,1,isn] = 0.5*(f[ivz,ivr,ivzeta,iz,1,isn]+f[ivz,ivr,ivzeta,iz,nr,isn])
            f[ivz,ivr,ivzeta,iz,nr,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
        end
    #elseif bc == "Dirichlet"
    #    # use the old distribution to force the new distribution to have 
    #    # consistant-in-time values at the boundary
    #    # impose bc on upwind boundary only (Hyperbolic PDE)
    #    @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
    #    #ir = 1 # r = -L/2
    #    #if adv[isn].speed[ir,ivz,ivr,ivzeta,ir] > zero
    #    #    f[ivz,ivr,ivzeta,iz,ir,isn] = f_initial[ivz,ivr,ivzeta,iz,ir,isn]
    #    #end
    #    #ir = nr # r = L/2
    #    #if adv[isn].speed[ir,ivz,ivr,ivzeta,ir] < -zero
    #    #    f[ivz,ivr,ivzeta,iz,ir,isn] = f_initial[ivz,ivr,ivzeta,iz,ir,isn]
    #    #end
    #    end
    end
end

end
