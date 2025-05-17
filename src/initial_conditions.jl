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
using ..calculus: reconcile_element_boundaries_MPI!
using ..looping
using ..velocity_moments: integrate_over_vspace, integrate_over_neutral_vspace
using ..velocity_moments: integrate_over_positive_vpa, integrate_over_negative_vpa
using ..velocity_moments: integrate_over_positive_vz, integrate_over_negative_vz
using ..velocity_moments: create_moments_charged, create_moments_neutral, update_qpar!
using ..velocity_moments: moments_charged_substruct, moments_neutral_substruct
using ..velocity_moments: update_neutral_density!, update_neutral_pz!, update_neutral_pr!, update_neutral_pzeta!
using ..velocity_moments: update_neutral_uz!, update_neutral_ur!, update_neutral_uzeta!, update_neutral_qz!
using ..velocity_moments: update_ppar!, update_upar!, update_density!

using ..manufactured_solns: manufactured_solutions

"""
"""
struct pdf_substruct{n_distribution}
    norm::MPISharedArray{mk_float,n_distribution}
    unnorm::MPISharedArray{mk_float,n_distribution}
    buffer::MPISharedArray{mk_float,n_distribution} # for collision operator terms when pdfs must be interpolated onto different velocity space grids
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

struct boundary_distributions_struct
    # knudsen cosine distribution for imposing the neutral wall boundary condition
    knudsen::MPISharedArray{mk_float,3}
    # charged particle r boundary values (vpa,vperp,z,r,s)
    pdf_rboundary_charged::MPISharedArray{mk_float,5}
    # neutral particle r boundary values (vz,vr,vzeta,z,r,s)
    pdf_rboundary_neutral::MPISharedArray{mk_float,6}
end
"""
creates the normalised pdf and the velocity-space moments and populates them
with a self-consistent initial condition
"""
function init_pdf_and_moments(vz, vr, vzeta, vpa, vperp, z, r, composition, geometry, species, n_rk_stages, evolve_moments, use_manufactured_solns)
    
    begin_serial_region()
    
    # define the n_species variable for convenience
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    if n_neutral_species > 0 
        n_neutral_species_alloc = n_neutral_species
    else
        n_neutral_species_alloc = 1
    end
    # create the 'moments' struct that contains various v-space moments and other
    # information related to these moments.
    # the time-dependent entries are not initialised.
    # moments arrays have same r and z grids for both ion and neutral species 
    # and so are included in the same struct
    charged = create_moments_charged(z.n, r.n, n_ion_species)
    neutral = create_moments_neutral(z.n, r.n, n_neutral_species_alloc)
    moments = moments_struct(charged,neutral)
    pdf = create_pdf(vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species_alloc)
    
    if use_manufactured_solns
        init_pdf_moments_manufactured_solns!(pdf, moments, vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species, geometry, composition)
    else 
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
                init_density!(moments.neutral.dens, z, r, species.neutral, n_neutral_species)
                init_uz!(moments.neutral.uz, z, r, species.neutral, n_neutral_species)
                init_ur!(moments.neutral.ur, z, r, species.neutral, n_neutral_species)
                init_uzeta!(moments.neutral.uzeta, z, r, species.neutral, n_neutral_species)
                init_vth!(moments.neutral.vth, z, r, species.neutral, n_neutral_species)
                @. moments.neutral.ptot = 0.5 * moments.neutral.dens * moments.neutral.vth^2
            end
        end
        # initialise the normalised particle distribution function (pdf)
        init_pdf!(pdf, moments, vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species, species)
            
        # calculate the self-consistent initial parallel heat flux and pressure from the initial un-normalised pdf
        update_qpar!(moments.charged.qpar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
        # need neutral version!!! update_qpar!(moments.charged.qpar, moments.charged.qpar_updated, pdf.charged.unnorm, vpa, vperp, z, r, composition, moments.charged.vpa_norm_fac)
        # calculate self-consistent neutral moments 
        if n_neutral_species > 0
            update_neutral_qz!(moments.neutral.qz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
            update_neutral_pz!(moments.neutral.pz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
            update_neutral_pr!(moments.neutral.pr, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
            update_neutral_pzeta!(moments.neutral.pzeta, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        end
    end 
    
    boundary_distributions = create_and_init_boundary_distributions(pdf, vz, vr, vzeta, vpa, vperp, z, r, composition)
    
    return pdf, moments, boundary_distributions
end

"""
"""
function create_pdf(vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species)
    # allocate pdf arrays
    pdf_charged_norm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, n_ion_species)
    pdf_charged_unnorm = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, n_ion_species)
    pdf_charged_buffer = allocate_shared_float(vpa.n, vperp.n, z.n, r.n, n_neutral_species) # n.b. n_species is n_neutral_species here
    pdf_neutral_norm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, n_neutral_species)
    pdf_neutral_unnorm = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, n_neutral_species)
    pdf_neutral_buffer = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, r.n, n_ion_species)

    return pdf_struct(pdf_substruct(pdf_charged_norm, pdf_charged_unnorm, pdf_charged_buffer), 
                    pdf_substruct(pdf_neutral_norm, pdf_neutral_unnorm, pdf_neutral_buffer))

end

function init_pdf!(pdf, moments, vz, vr, vzeta, vpa, vperp, z, r, n_ion_species, n_neutral_species, species)
    @serial_region begin
        for is ∈ 1:n_ion_species
            for ir ∈ 1:r.n
                # updates pdf_norm to contain pdf / density, so that ∫dvpa pdf.norm = 1,
                # ∫dwpa wpa * pdf.norm = 0, and ∫dwpa m_s (wpa/vths)^2 pdf.norm = 1/2
                # to machine precision
                @views init_pdf_charged_over_density!(pdf.charged.norm[:,:,:,ir,is], species.charged[is], vpa, vperp, z, moments.charged.vth[:,ir,is],
                                              moments.charged.upar[:,ir,is])
            end
        end
        if n_neutral_species > 0
            for is ∈ 1:n_neutral_species
                for ir ∈ 1:r.n
                    @views init_pdf_neutral_over_density!(pdf.neutral.norm[:,:,:,:,ir,is], species.neutral[is], vz, vr, vzeta, z, moments.neutral.uz, moments.neutral.ur, moments.neutral.uzeta, moments.neutral.vth)
                end
            end
        end
        #set unnorm pdf = norm pdf * dens
        
        for ivperp ∈ 1:vperp.n
            for ivpa ∈ 1:vpa.n
                @. pdf.charged.unnorm[ivpa,ivperp,:,:,:] = pdf.charged.norm[ivpa,ivperp,:,:,:] .* moments.charged.dens[:,:,:]
                # No evolving moments, so need to set pdf.norm = pdf.unnorm
                @. pdf.charged.norm[ivpa,ivperp,:,:,:] = pdf.charged.unnorm[ivpa,ivperp,:,:,:]
            end
        end
        if n_neutral_species > 0 
            for ivzeta in 1:vzeta.n
                for ivr in 1:vr.n
                    for ivz in 1:vz.n
                        @. pdf.neutral.unnorm[ivz,ivr,ivzeta,:,:,:] = pdf.neutral.norm[ivz,ivr,ivzeta,:,:,:] .* moments.neutral.dens[:,:,:]
                        # No evolving moments, so need to set pdf.norm = pdf.unnorm
                        @. pdf.neutral.norm[ivz,ivr,ivzeta,:,:,:] = pdf.neutral.unnorm[ivz,ivr,ivzeta,:,:,:]
                    end
                end
            end
        end
    end
    return nothing
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
        moments.charged.dens[iz,ir,is] = densi_func(z.grid[iz],r.grid[ir],0.0)
        @loop_vperp_vpa ivperp ivpa begin
            pdf.charged.unnorm[ivpa,ivperp,iz,ir,is] = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],0.0)
            pdf.charged.norm[ivpa,ivperp,iz,ir,is] = pdf.charged.unnorm[ivpa,ivperp,iz,ir,is]
        end
    end
    # update upar, ppar, qpar, vth consistent with manufactured solns
    update_density!(moments.charged.dens, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    update_qpar!(moments.charged.qpar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    update_ppar!(moments.charged.ppar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    # get particle flux
    update_upar!(moments.charged.upar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    # convert from particle particle flux to parallel flow
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        moments.charged.upar[iz,ir,is] /= moments.charged.dens[iz,ir,is]
    # update the thermal speed
        moments.charged.vth[iz,ir,is] = sqrt(2.0*moments.charged.ppar[iz,ir,is]/moments.charged.dens[iz,ir,is])
    end
    
    if n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = densn_func(z.grid[iz],r.grid[ir],0.0)
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf.neutral.unnorm[ivz,ivr,ivzeta,iz,ir,isn] = dfnn_func(vz.grid[ivz],vr.grid[ivr],vzeta.grid[ivzeta],z.grid[iz],r.grid[ir],0.0)
                pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.unnorm[ivz,ivr,ivzeta,iz,ir,isn]
            end
        end
        # get consistent moments with manufactured solutions 
        update_neutral_density!(moments.neutral.dens, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_qz!(moments.neutral.qz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_pz!(moments.neutral.pz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_pr!(moments.neutral.pr, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_pzeta!(moments.neutral.pzeta, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
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
        update_neutral_uz!(moments.neutral.uz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_ur!(moments.neutral.ur, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_uzeta!(moments.neutral.uzeta, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        # now convert from particle particle flux to parallel flow
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.uz[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            moments.neutral.ur[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            moments.neutral.uzeta[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
            # get vth for neutrals
            moments.charged.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.ptot[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
        end
    end
    return nothing
end

function init_knudsen_cosine(vz, vr, vzeta, vpa, vperp, composition)
    knudsen_cosine = allocate_shared_float(vz.n, vr.n, vzeta.n)

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
                            if  v_tot > 0.0
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

function init_rboundary_pdfs(pdf::pdf_struct, vz, vr, vzeta, vpa, vperp, z, r, composition)
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    n_neutral_species_alloc = max(1, n_neutral_species)
    
    rboundary_charged = allocate_shared_float(vpa.n, vperp.n, z.n, 2, n_ion_species)
    rboundary_neutral = allocate_shared_float(vz.n, vr.n, vzeta.n, z.n, 2, n_neutral_species_alloc)
    
    begin_s_z_region() #do not parallelise r here 
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        rboundary_charged[ivpa,ivperp,iz,1,is] = pdf.charged.unnorm[ivpa,ivperp,iz,1,is]
        rboundary_charged[ivpa,ivperp,iz,end,is] = pdf.charged.unnorm[ivpa,ivperp,iz,end,is]
    end
    if n_neutral_species > 0 
        begin_sn_z_region() #do not parallelise r here
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            rboundary_neutral[ivz,ivr,ivzeta,iz,1,isn] = pdf.neutral.unnorm[ivz,ivr,ivzeta,iz,1,isn]
            rboundary_neutral[ivz,ivr,ivzeta,iz,end,isn] = pdf.neutral.unnorm[ivz,ivr,ivzeta,iz,end,isn]
        end
    end
    return rboundary_charged, rboundary_neutral
end

function create_and_init_boundary_distributions(pdf, vz, vr, vzeta, vpa, vperp, z, r, composition)
    #initialise knudsen distribution for neutral wall bc
    knudsen_cosine = init_knudsen_cosine(vz, vr, vzeta, vpa, vperp, composition)
    #initialise fixed-in-time radial boundary condition based on initial condition values
    pdf_rboundary_charged, pdf_rboundary_neutral = init_rboundary_pdfs(pdf, vz, vr, vzeta, vpa, vperp, z, r, composition)
    
    return boundary_distributions_struct(knudsen_cosine, pdf_rboundary_charged, pdf_rboundary_neutral)
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
        for iz ∈ 1:z.n
            # densfac = the integral of the pdf over v-space, which should be unity,
            # but may not be exactly unity due to quadrature errors
            densfac = integrate_over_vspace(view(pdf,:,:,iz), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
            # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
            # where w_s = vpa - upar_s;
            # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
            @views @. vpa.scratch = vpa.grid^2 / vth[iz]^2
            pparfac = integrate_over_vspace(pdf[:,:,iz], vpa.scratch, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)
            # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
            @views @. vpa.scratch = vpa.grid^2 *(vpa.grid^2/pparfac - 1.0/densfac) / vth[iz]^4
            pparfac2 = @views integrate_over_vspace(pdf[:,:,iz], vpa.scratch, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)

            for ivperp ∈ 1:vperp.n
                @views @. pdf[:,ivperp,iz] = pdf[:,ivperp,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vpa.grid^2/pparfac - 1.0/densfac)*pdf[:,ivperp,iz] / vth[iz]^2
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
                pdf[ivz,ivr,ivzeta,iz] = exp(-vz.scratch[ivz]^2 - vr.scratch[ivr]^2 - vzeta.scratch[ivzeta]^2  ) / vth[iz]
            end
        else # 3D case with vr & vzeta
            for ivzeta ∈ 1:vzeta.n
                for ivr ∈ 1:vr.n
                    for ivz ∈ 1:vz.n
                        pdf[ivz,ivr,ivzeta,iz] = exp(-vz.scratch[ivz]^2 - vr.scratch[ivr]^2 - vzeta.scratch[ivzeta]^2  ) / vth[iz]^3
                    end
                end
            end
        end 
    end

    for iz ∈ 1:z.n
        # densfac = the integral of the pdf over v-space, which should be unity,
        # but may not be exactly unity due to quadrature errors
        densfac = integrate_over_neutral_vspace(view(pdf,:,:,:,iz), vz.grid, 0, vz.wgts, vr.grid, 0, vr.wgts, vzeta.grid, 0, vzeta.wgts)
        # pparfac = the integral of the pdf over v-space, weighted by m_s w_s^2 / vths^2,
        # where w_s = vz - upar_s;
        # should be equal to 1/2, but may not be exactly 1/2 due to quadrature errors
        @views @. vz.scratch = vz.grid^2 / vth[iz]^2
        pparfac = integrate_over_neutral_vspace(pdf[:,:,:,iz], vz.scratch, 1, vz.wgts, vr.grid, 0, vr.wgts, vzeta.grid, 0, vzeta.wgts)
        # pparfac2 = the integral of the pdf over v-space, weighted by m_s w_s^2 (w_s^2 - vths^2 / 2) / vth^4
        @views @. vz.scratch = vz.grid^2 *(vz.grid^2/pparfac - 1.0/densfac) / vth[iz]^4
        pparfac2 = @views integrate_over_neutral_vspace(pdf[:,:,:,iz], vz.scratch, 1, vz.wgts, vr.grid, 0, vr.wgts, vzeta.grid, 0, vzeta.wgts)

        for ivzeta ∈ 1:vzeta.n, ivr ∈ 1:vr.n
            @views @. pdf[:,ivr,ivzeta,iz] = pdf[:,ivr,ivzeta,iz]/densfac + (0.5 - pparfac/densfac)/pparfac2*(vz.grid^2/pparfac - 1.0/densfac)*pdf[:,ivr,ivzeta,iz] / vth[iz]^2
        end
    end
    
    return nothing
end


# The "where" syntax here is confusing...
# Can we have an explicit loop or explain what it does?
# Seems to make the fn a operator elementwise in the vector f
# but loop is over a part of mysterious x_adv objects...
function enforce_boundary_conditions!(f, f_r_bc,
          vpa_bc, z_bc, r_bc, vpa, vperp, z, r,
          vpa_adv::T1, z_adv::T2, r_adv::T3, composition,
          scratch_dummy::T4, advance::T5) where {T1, T2, T3, T4, T5}
    
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        # enforce the vpa BC
        # use that adv.speed independent of vpa 
        @views enforce_vpa_boundary_condition_local!(f[:,ivperp,iz,ir,is], vpa_bc, vpa_adv[is].speed[:,ivperp,iz,ir], advance.vpa_diffusion)
    end
    begin_s_r_vperp_vpa_region()
    @views enforce_z_boundary_condition!(f, z_bc, z_adv, vpa, vperp, z, r, composition,
            scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
            scratch_dummy.buffer_vpavperprs_3, scratch_dummy.buffer_vpavperprs_4,
            scratch_dummy.buffer_vpavperpzrs_1)
    if r.n > 1
        begin_s_z_vperp_vpa_region()
        @views enforce_r_boundary_condition!(f, f_r_bc, r_bc, r_adv, vpa, vperp, z, r, composition,
            scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
            scratch_dummy.buffer_vpavperpzs_3, scratch_dummy.buffer_vpavperpzs_4,
            scratch_dummy.buffer_vpavperpzrs_1, advance.r_diffusion)
    end
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
            if (adv[is].speed[ir,ivpa,ivperp,iz] > zero || r_diffusion) && r.irank == 0
                f[ivpa,ivperp,iz,ir,is] = f_r_bc[ivpa,ivperp,iz,1,is]
            end
            ir = r.n # r = L/2 -- check that the point is on highest rank
            if (adv[is].speed[ir,ivpa,ivperp,iz] < -zero || r_diffusion) && r.irank == r.nrank - 1
                f[ivpa,ivperp,iz,ir,is] = f_r_bc[ivpa,ivperp,iz,end,is]
            end    
        end
    end
end
"""
enforce boundary conditions on f in z
"""
function enforce_z_boundary_condition!(f::AbstractArray{mk_float,5}, bc::String, adv::T,
        vpa, vperp, z, r, composition, end1::AbstractArray{mk_float,4},
        end2::AbstractArray{mk_float,4}, buffer1::AbstractArray{mk_float,4},
        buffer2::AbstractArray{mk_float,4}, buffer_dfn::AbstractArray{mk_float,5}) where T
    # define nz variable for convenience
    nz = z.n
    # define a zero that accounts for finite precision
    zero = 1.0e-10

    if z.nelement_global > z.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            end1[ivpa,ivperp,ir,is] = f[ivpa,ivperp,1,ir,is]
            end2[ivpa,ivperp,ir,is] = f[ivpa,ivperp,nz,ir,is]
        end
        @views reconcile_element_boundaries_MPI!(f,
            end1, end2,	buffer1, buffer2, z)
    end
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    # only supported for z local
    if bc == "constant" && z.nelement_global == z.nelement_local
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            if adv[is].speed[ivpa,ivperp,1,ir] > zero
                f[ivpa,ivperp,1,ir,is] = density_offset * exp(-(vpa.grid[ivpa]/vpawidth)^2) / sqrt(pi)
            end
            if adv[is].speed[ivpa,ivperp,nz,ir] < -zero
                f[ivpa,ivperp,nz,ir,is] = density_offset * exp(-(vpa.grid[ivpa]/vpawidth)^2) / sqrt(pi)
            end
        end
    end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    # enforced here for case when z is entirely local
    if bc == "periodic" && z.nelement_global == z.nelement_local
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            f[ivpa,ivperp,1,ir,is] = 0.5*(f[ivpa,ivperp,nz,ir,is]+f[ivpa,ivperp,1,ir,is])
            f[ivpa,ivperp,nz,ir,is] = f[ivpa,ivperp,1,ir,is]
        end
    end
    # 'wall' BC enforces wall boundary conditions
    if bc == "wall"
        @loop_s is begin
            # zero incoming BC for ions, as they recombine at the wall
            @loop_r_vperp_vpa ir ivperp ivpa begin
                # no parallel BC should be enforced for vpa = 0
                # adv.speed is signed 
                # adv.speed =  vpa*bzed - 0.5 *rhostar*Er
                # check that this rank includes the lower/upper boundary
                iz = 1 # z = -L/2
                if adv[is].speed[iz,ivpa,ivperp,ir] > zero && z.irank == 0
                    f[ivpa,ivperp,iz,ir,is] = 0.0
                end
                iz = nz # z = L/2
                if adv[is].speed[iz,ivpa,ivperp,ir] < -zero && z.irank == z.nrank - 1
                    f[ivpa,ivperp,iz,ir,is] = 0.0
                end
                
            end
        
        end
        
    end
end

"""
impose the prescribed vpa boundary condition on f
at every z grid point
"""
function enforce_vpa_boundary_condition!(f, bc, src::T, vpa_diffusion::Bool) where T
    nvperp = size(f,2)
    nz = size(f,3)
    nr = size(f,4)
    for ir ∈ 1:nr
        for iz ∈ 1:nz
            for ivperp ∈ 1:nvperp
                enforce_vpa_boundary_condition_local!(view(f,:,ivperp,iz,ir), bc, src.speed[:,ivperp,iz,ir], vpa_diffusion)
            end
        end
    end
end

"""
"""
function enforce_vpa_boundary_condition_local!(f::T, bc, adv_speed, vpa_diffusion::Bool) where T
    # define a zero that accounts for finite precision
    zero = 1.0e-10
    dvpadt = adv_speed[1] #use that dvpa/dt is indendent of vpa in the current model 
    nvpa = size(f,1)
    if bc == "zero"
        if dvpadt[1] > -zero || vpa_diffusion
            f[1] = 0.0 # -infty forced to zero
        elseif dvpadt[end] < zero || vpa_diffusion
            f[end] = 0.0 # +infty forced to zero
        end
    elseif bc == "periodic"
        f[1] = 0.5*(f[nvpa]+f[1])
        f[nvpa] = f[1]
    end
end

function enforce_neutral_boundary_conditions!(f_neutral, f_charged, boundary_distributions, r_adv_neutral::T1, z_adv_neutral::T2, z_adv_charged::T3, vz, vr, vzeta, vpa, vperp, z, r, composition, scratch_dummy::T4) where {T1, T2, T3, T4} #f_initial,
    
    # f_initial contains the initial condition for enforcing a fixed-boundary-value condition 
    # no bc on vz vr vzeta required as no advection in these coordinates
    begin_sn_r_vzeta_vr_vz_region()
    @views enforce_neutral_z_boundary_condition!(f_neutral, f_charged, boundary_distributions,
            z_adv_neutral, z_adv_charged, vz, vr, vzeta, vpa, vperp, z, r, composition,
            scratch_dummy.buffer_vzvrvzetarsn_1, scratch_dummy.buffer_vzvrvzetarsn_2,
            scratch_dummy.buffer_vzvrvzetarsn_3, scratch_dummy.buffer_vzvrvzetarsn_4,
            scratch_dummy.buffer_vzvrvzetazrsn)
    if r.n > 1
        begin_sn_z_vzeta_vr_vz_region()
        @views enforce_neutral_r_boundary_condition!(f_neutral, boundary_distributions.pdf_rboundary_neutral,
                                    r_adv_neutral, vz, vr, vzeta, z, r, composition,
                                    scratch_dummy.buffer_vzvrvzetazsn_1, scratch_dummy.buffer_vzvrvzetazsn_2,
                                    scratch_dummy.buffer_vzvrvzetazsn_3, scratch_dummy.buffer_vzvrvzetazsn_4,
                                    scratch_dummy.buffer_vzvrvzetazrsn)
    end
end

function enforce_neutral_z_boundary_condition!(f_neutral::AbstractArray{mk_float,6},
        f_charged::AbstractArray{mk_float,5}, boundary_distributions, z_adv_neutral::T1,
        z_adv_charged::T2, vz, vr, vzeta, vpa, vperp, z, r, composition,
        end1::AbstractArray{mk_float,5}, end2::AbstractArray{mk_float,5},
        buffer1::AbstractArray{mk_float,5}, buffer2::AbstractArray{mk_float,5},
        buffer_dfn::AbstractArray{mk_float,6}) where {T1, T2}
    bc = z.bc
    nz = z.n
    # define a zero that accounts for finite precision
    zero = 1.0e-10
    
    if z.nelement_global > z.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            end1[ivz,ivr,ivzeta,ir,isn] = f_neutral[ivz,ivr,ivzeta,1,ir,isn]
            end2[ivz,ivr,ivzeta,ir,isn] = f_neutral[ivz,ivr,ivzeta,nz,ir,isn]
        end
        @views reconcile_element_boundaries_MPI!(f_neutral,
            end1, end2,	buffer1, buffer2, z)
    end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    # if z data is entirely local, enforce periodic boundary condition on external endpoint
    if bc == "periodic" && z.nelement_global == z.nelement_local
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            f_neutral[ivz,ivr,ivzeta,1,ir,isn] = 0.5*(f_neutral[ivz,ivr,ivzeta,1,ir,isn]+f_neutral[ivz,ivr,ivzeta,nz,ir,isn])
            f_neutral[ivz,ivr,ivzeta,nz,ir,isn] = f_neutral[ivz,ivr,ivzeta,1,ir,isn]
        end
    end
    # If required, enforce wall boundary condition on external endpoints
    if bc == "wall"
    # wall BC for neutrals
        begin_serial_region()
        # TODO: parallelise this...
        @serial_region begin
            for ir ∈ 1:r.n
                # initialise the combined ion/neutral fluxes into the walls to be zero
                wall_flux_0 = 0.0
                wall_flux_L = 0.0
                # include the contribution to the wall fluxes due to species with index 'is'
                for is ∈ 1:composition.n_ion_species
                        # get velocity into the wall at this r at z = -L/2, vz(vpa,vperp)
                        vz_charged = z_adv_charged[is].speed[1,:,:,ir]
                        #n.b. vz_charged independent of vperp in current 2D model so 
                        # vz_charged[:,n] identical for all n in 1:end -> for convenience we pass 
                        # dzdt = vz_charged[:,1] as a 1-D velocity variable for the half-sided integration routines
                        # if vz_charged becomes a fn of vperp then these routines must be generalised
                        @views wall_flux_0 += (sqrt(composition.mn_over_mi) *
                                               integrate_over_negative_vpa(abs.(vz_charged[:,:]) .* f_charged[:,:,1,ir,is], vz_charged[:,1], vpa.wgts, vpa.scratch, vperp.grid, vperp.wgts))
                        # get velocity into the wall at this r at z = L/2, vz(vpa,vperp)
                        vz_charged = z_adv_charged[is].speed[end,:,:,ir]
                        @views wall_flux_L += (sqrt(composition.mn_over_mi) *
                                               integrate_over_positive_vpa(abs.(vz_charged[:,:]) .* f_charged[:,:,end,ir,is], vz_charged[:,1], vpa.wgts, vpa.scratch, vperp.grid, vperp.wgts))
                end
                for isn ∈ 1:composition.n_neutral_species
                        # get velocity into the wall at this r = -L/2 at z = -L/2, vz(vpa,vperp)
                        vz_neutral = z_adv_neutral[isn].speed[1,:,:,:,ir]
                        #n.b. vz_neutral independent of vr, vzeta in current 2D model so 
                        # vz_neutral[:,n,m] = vz.grid[:] is identical for all n, m in 1:end -> for convenience we pass 
                        # dzdt = vz.grid[:] as a 1-D velocity variable for the half-sided integration routines
                        # if vz_neutral becomes a fn of vr vzeta then these routines must be generalised
                        @views wall_flux_0 += integrate_over_negative_vz(abs.(vz_neutral) .* f_neutral[:,:,:,1,ir,isn], vz.grid, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                        vz_neutral = z_adv_neutral[isn].speed[end,:,:,:,ir]
                        @views wall_flux_L += integrate_over_positive_vz(abs.(vz_neutral) .* f_neutral[:,:,:,end,ir,isn], vz.grid, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                end
                # NB: as vtfac is time-independent, can be made more efficient by creating
                # array for Knudsen cosine distribution and carrying out following four lines
                # of calculation at initialization
                
                #@. vz.scratch = (3*pi/vtfac^3)*abs(vz.grid)*erfc(abs(vz.grid)/vtfac)
                #tmparr = copy(vz.scratch)
                #tmp = integrate_over_positive_vpa(vz.grid .* vz.scratch, vz.grid, vz.wgts, tmparr)
                #@. vz.scratch /= tmp
                
                knudsen_cosine = boundary_distributions.knudsen
                for isn ∈ 1:composition.n_neutral_species
                    for ivzeta ∈ 1:vzeta.n
                        for ivr ∈ 1:vr.n
                            for ivz ∈ 1:vz.n
                                # no parallel BC should be enforced for vz = 0
                                iz = 1 # z = -L/2, ensure data is on lowest rank
                                if vz.grid[ivz] > zero && z.irank == 0
                                    f_neutral[ivz,ivr,ivzeta,iz,ir,isn] = wall_flux_0 * knudsen_cosine[ivz,ivr,ivzeta]
                                end
                                iz = nz # z = L/2, ensure data is on highest rank
                                if vz.grid[ivz] < -zero && z.irank == z.nrank - 1
                                    f_neutral[ivz,ivr,ivzeta,iz,ir,isn] = wall_flux_L * knudsen_cosine[ivz,ivr,ivzeta]
                                end
                            end
                        end
                    end
                end
            end
        end
    
    
    end
end

function enforce_neutral_r_boundary_condition!(f::AbstractArray{mk_float,6},
        f_r_bc::AbstractArray{mk_float,6}, adv::T, vz, vr, vzeta, z, r, composition,
        end1::AbstractArray{mk_float,5}, end2::AbstractArray{mk_float,5},
        buffer1::AbstractArray{mk_float,5}, buffer2::AbstractArray{mk_float,5},
        buffer_dfn::AbstractArray{mk_float,6}) where T #f_initial,

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
            if adv[isn].speed[ir,ivz,ivr,ivzeta,iz] > zero && r.irank == 0
                f[ivz,ivr,ivzeta,iz,ir,isn] = f_r_bc[ivz,ivr,ivzeta,iz,1,isn]
            end
            ir = nr # r = L/2
            # incoming particles and on highest rank
            if adv[isn].speed[ir,ivz,ivr,ivzeta,iz] < -zero && r.irank == r.nrank - 1
                f[ivz,ivr,ivzeta,iz,ir,isn] = f_r_bc[ivz,ivr,ivzeta,iz,end,isn]
            end
        end
    end
end

end
