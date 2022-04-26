"""
"""
module time_advance

export setup_time_advance!
export time_advance!

using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: _block_synchronize
using ..debugging
using ..file_io: write_data_to_ascii, write_data_to_binary, debug_dump
using ..looping
using ..moment_kinetics_structs: scratch_pdf
using ..chebyshev: setup_chebyshev_pseudospectral
using ..chebyshev: chebyshev_derivative!
using ..velocity_moments: update_moments!, reset_moments_status!
using ..velocity_moments: enforce_moment_constraints!
using ..velocity_moments: update_density!, update_upar!, update_ppar!, update_qpar!
using ..initial_conditions: enforce_z_boundary_condition!, enforce_boundary_conditions!
using ..initial_conditions: enforce_vpa_boundary_condition!
using ..advection: setup_advection, update_boundary_indices!
using ..z_advection: update_speed_z!, z_advection!
using ..r_advection: update_speed_r!, r_advection!
using ..vperp_advection: update_speed_vperp!, vperp_advection!
using ..vpa_advection: update_speed_vpa!, vpa_advection!
using ..charge_exchange: charge_exchange_collisions!
using ..ionization: ionization_collisions!
using ..source_terms: source_terms!
using ..continuity: continuity_equation!
using ..force_balance: force_balance!
using ..energy_equation: energy_equation!
using ..em_fields: setup_em_fields, update_phi!
using ..semi_lagrange: setup_semi_lagrange

@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

"""
"""
mutable struct advance_info
    vpa_advection::Bool
    z_advection::Bool
    r_advection::Bool
    cx_collisions::Bool
    ionization_collisions::Bool
    source_terms::Bool
    continuity::Bool
    force_balance::Bool
    energy::Bool
    rk_coefs::Array{mk_float,2}
end

mutable struct scratch_dummy_arrays
    dummy_sr::Array{mk_float,2}
    dummy_vpavperp::Array{mk_float,2}
    dummy_zr::Array{mk_float,2}
end 

"""
create arrays and do other work needed to setup
the main time advance loop.
this includes creating and populating structs
for Chebyshev transforms, velocity space moments,
EM fields, semi-Lagrange treatment, and advection terms
"""
function setup_time_advance!(pdf, vpa, vperp, z, r, composition, drive_input, moments,
                             t_input, collisions, species, geometry)
    # define some local variables for convenience/tidiness
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    # create array containing coefficients needed for the Runge Kutta time advance
    rk_coefs = setup_runge_kutta_coefficients(t_input.n_rk_stages)
    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    if t_input.split_operators
        advance = advance_info(false, false, false, false, false, false, false, false, false, rk_coefs)
    else
        if composition.n_neutral_species > 0
            if collisions.charge_exchange > 0.0
                advance_cx = true
            else
                advance_cx = false
            end
            if collisions.ionization > 0.0
                advance_ionization = true
            else
                advance_ionization = false
            end
        else
            advance_cx = false
            advance_ionization = false
        end
        if moments.evolve_density
            advance_sources = true
            advance_continuity = true
            if moments.evolve_upar
                advance_force_balance = true
                if moments.evolve_ppar
                    advance_energy = true
                else
                    advance_energy = false
                end
            else
                advance_force_balance = false
                advance_energy = false
            end
        else
            advance_sources = false
            advance_continuity = false
            advance_force_balance = false
            advance_energy = false
        end
        advance = advance_info(true, true, true, advance_cx, advance_ionization, advance_sources,
                               advance_continuity, advance_force_balance, advance_energy, rk_coefs)
    end
    
    
    if z.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        z_spectral = setup_chebyshev_pseudospectral(z)
        # obtain the local derivatives of the uniform z-grid with respect to the used z-grid
        chebyshev_derivative!(z.duniform_dgrid, z.uniform_grid, z_spectral, z)
    else
        # create dummy Bool variable to return in place of the above struct
        z_spectral = false
        z.duniform_dgrid .= 1.0
    end
    
    if r.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        r_spectral = setup_chebyshev_pseudospectral(r)
        # obtain the local derivatives of the uniform r-grid with respect to the used r-grid
        chebyshev_derivative!(r.duniform_dgrid, r.uniform_grid, r_spectral, r)
    else
        # create dummy Bool variable to return in place of the above struct
        r_spectral = false
        r.duniform_dgrid .= 1.0
    end
    
    if vpa.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        vpa_spectral = setup_chebyshev_pseudospectral(vpa)
        # obtain the local derivatives of the uniform vpa-grid with respect to the used vpa-grid
        chebyshev_derivative!(vpa.duniform_dgrid, vpa.uniform_grid, vpa_spectral, vpa)
    else
        # create dummy Bool variable to return in place of the above struct
        vpa_spectral = false
        vpa.duniform_dgrid .= 1.0
    end
    
    if vperp.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vperp
        # and create the plans for the forward and backward fast Chebyshev transforms
        vperp_spectral = setup_chebyshev_pseudospectral(vperp)
        # obtain the local derivatives of the uniform vperp-grid with respect to the used vperp-grid
        chebyshev_derivative!(vperp.duniform_dgrid, vperp.uniform_grid, vperp_spectral, vperp)
    else
        # create dummy Bool variable to return in place of the above struct
        vperp_spectral = false
        vperp.duniform_dgrid .= 1.0
    end
    
    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf.norm, t_input.n_rk_stages)
    # setup dummy arrays
    dummy_sr = allocate_float(r.n, composition.n_species)
    dummy_zr = allocate_float(z.n, r.n)
    dummy_vpavperp = allocate_float(vpa.n, vperp.n)
    scratch_dummy = scratch_dummy_arrays(dummy_sr,dummy_vpavperp,dummy_zr)
    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, r.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency)
    # initialize the electrostatic potential
    begin_s_r_z_vperp_region()
    update_phi!(fields, scratch[1], z, r, composition, z_spectral, r_spectral)
    begin_serial_region()
    @serial_region begin
        # save the initial phi(z) for possible use later (e.g., if forcing phi)
        fields.phi0 .= fields.phi
    end
    
    # create structure r_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in r
    begin_serial_region()
    r_advect = setup_advection(n_species, r, vpa, vperp, z)
    # initialise the r advection speed
    begin_s_z_vperp_vpa_region()
    @loop_s is begin
        @views update_speed_r!(r_advect[is], fields, moments.upar[:,:,is], moments.vth[:,:,is],
            vpa, vperp, z, r, 0.0, geometry, scratch_dummy, z_spectral)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(r_advect[is], loop_ranges[].vpa, loop_ranges[].vperp, loop_ranges[].z)
    end
    # enforce prescribed boundary condition in r on the distribution function f
    # PLACEHOLDER
    #@views enforce_r_boundary_condition!(pdf.unnorm, r.bc, r_advect, vpa, z, composition)
    
    
    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in z
    begin_serial_region()
    z_advect = setup_advection(n_species, z, vpa, vperp, r)
    # initialise the z advection speed
    begin_s_r_vperp_vpa_region()
    @loop_s is begin
        @views update_speed_z!(z_advect[is], fields, moments.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, vperp, z, r,
                               0.0, geometry, scratch_dummy, r_spectral)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(z_advect[is], loop_ranges[].vpa, loop_ranges[].vperp, loop_ranges[].r)
    end
    # enforce prescribed boundary condition in z on the distribution function f
    @views enforce_z_boundary_condition!(pdf.unnorm, z.bc, z_advect, vpa, vperp, r, composition)
    if z.bc != "wall" || composition.n_neutral_species == 0
        begin_serial_region()
    end
    
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_advect = setup_advection(n_species, vpa, vperp, z, r)
    # initialise the vpa advection speed
    begin_s_r_z_vperp_region()
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, vperp, z, r, composition,
                      collisions.charge_exchange, 0.0, z_spectral, geometry)
    if moments.evolve_upar
        nspec = n_species
    else
        nspec = n_ion_species
    end
    begin_serial_region()
    @serial_region begin
        for is ∈ 1:nspec
            # initialise the upwind/downwind boundary indices in vpa
            update_boundary_indices!(vpa_advect[is], 1:vperp.n, 1:z.n, 1:r.n)
            # enforce prescribed boundary condition in vpa on the distribution function f
            @views enforce_vpa_boundary_condition!(pdf.norm[:,:,:,:,is], vpa.bc, vpa_advect[is])
        end
    end
    # create structure vperp_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vperp
    begin_serial_region()
    vperp_advect = setup_advection(n_species, vperp, vpa, z, r)
    # initialise the vperp advection speed
    if moments.evolve_upar
        nspec = n_species
    else
        nspec = n_ion_species
    end
    begin_serial_region()
    @serial_region begin
        for is ∈ 1:nspec
            @views update_speed_vperp!(vperp_advect[is], vpa, vperp, z, r, 0.0)
            # initialise the upwind/downwind boundary indices in vpa
            update_boundary_indices!(vperp_advect[is], 1:vpa.n, 1:z.n, 1:r.n)
            # enforce prescribed boundary condition in vpa on the distribution function f
            #PLACEHOLDER
            #@views enforce_vperp_boundary_condition!(pdf.norm[:,:,:,:,is], vpa.bc, vpa_advect[is])
        end
    end
    # create an array of structures containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n, vpa.n, vperp.n, r.n)
    vpa_SL = setup_semi_lagrange(vpa.n, vperp.n, z.n, r.n)
    vperp_SL = setup_semi_lagrange(vperp.n, vpa.n, z.n, r.n)
    r_SL = setup_semi_lagrange(r.n, vpa.n, vperp.n, z.n)

    begin_s_r_z_vperp_region()
    return vpa_spectral, vperp_spectral, z_spectral, r_spectral, moments, fields, 
    vpa_advect, vperp_advect, z_advect, r_advect,vpa_SL, vperp_SL, z_SL, r_SL,
    scratch, advance, scratch_dummy
end

"""
if evolving the density via continuity equation, redefine the normalised f → f/n
if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
'scratch' should be a (nz,nspecies) array
"""
function normalize_pdf!(pdf, moments, scratch)
    error("Function normalise_pdf() has not been updated to be parallelized. Does not "
          * "seem to be used at the moment.")
    if moments.evolve_ppar
        @. scratch = moments.vth/moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    elseif moments.evolve_density
        @. scatch = 1.0 / moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    end
    return nothing
end

"""
create an array of structs containing scratch arrays for the normalised pdf and low-order moments
that may be evolved separately via fluid equations
"""
function setup_scratch_arrays(moments, pdf_in, n_rk_stages)
    # create n_rk_stages+1 structs, each of which will contain one pdf,
    # one density, and one parallel flow array
    scratch = Vector{scratch_pdf{5,3}}(undef, n_rk_stages+1)
    pdf_dims = size(pdf_in)
    moment_dims = size(moments.dens)
    # populate each of the structs
    for istage ∈ 1:n_rk_stages+1
        # Allocate arrays in temporary variables so that we can identify them
        # by source line when using @debug_shared_array
        pdf_array = allocate_shared_float(pdf_dims...)
        density_array = allocate_shared_float(moment_dims...)
        upar_array = allocate_shared_float(moment_dims...)
        ppar_array = allocate_shared_float(moment_dims...)
        temp_z_s_array = allocate_shared_float(moment_dims...)
        scratch[istage] = scratch_pdf(pdf_array, density_array, upar_array,
                                      ppar_array, temp_z_s_array)
        @serial_region begin
            scratch[istage].pdf .= pdf_in
            scratch[istage].density .= moments.dens
            scratch[istage].upar .= moments.upar
            scratch[istage].ppar .= moments.ppar
        end
    end
    return scratch
end

"""
given the number of Runge Kutta stages that are requested,
returns the needed Runge Kutta coefficients;
e.g., if f is the function to be updated, then
f^{n+1}[stage+1] = rk_coef[1,stage]*f^{n} + rk_coef[2,stage]*f^{n+1}[stage] + rk_coef[3,stage]*(f^{n}+dt*G[f^{n+1}[stage]]
"""
function setup_runge_kutta_coefficients(n_rk_stages)
    rk_coefs = allocate_float(3,n_rk_stages)
    rk_coefs .= 0.0
    if n_rk_stages == 4
        rk_coefs[1,1] = 0.5
        rk_coefs[3,1] = 0.5
        rk_coefs[2,2] = 0.5
        rk_coefs[3,2] = 0.5
        rk_coefs[1,3] = 2.0/3.0
        rk_coefs[2,3] = 1.0/6.0
        rk_coefs[3,3] = 1.0/6.0
        rk_coefs[2,4] = 0.5
        rk_coefs[3,4] = 0.5
    elseif n_rk_stages == 3
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.75
        rk_coefs[3,2] = 0.25
        rk_coefs[1,3] = 1.0/3.0
        rk_coefs[3,3] = 2.0/3.0
    elseif n_rk_stages == 2
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.5
        rk_coefs[3,2] = 0.5
    else
        rk_coefs[3,1] = 1.0
    end
    return rk_coefs
end

"""
solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
define approximate characteristic velocity
v₀(z)=vⁿ(z) and take time derivative along this characteristic
df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
for prudent choice of v₀, expect δv≪v so that explicit
time integrator can be used without severe CFL condition
"""
function time_advance!(pdf, scratch, t, t_input, vpa, vperp, z, r,
    vpa_spectral, vperp_spectral, z_spectral, r_spectral,
    moments, fields, vpa_advect, vperp_advect, z_advect, r_advect,
    vpa_SL, vperp_SL, z_SL, r_SL, composition,
    collisions, geometry, advance, scratch_dummy, io, cdf)

    @debug_detect_redundant_block_synchronize begin
        # Only want to check for redundant _block_synchronize() calls during the
        # time advance loop, so activate these checks here
        debug_detect_redundant_is_active[] = true
    end

    # main time advance loop
    iwrite = 2
    for i ∈ 1:t_input.nstep
       
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, vperp, z, r,
                vpa_spectral, vperp_spectral, z_spectral, r_spectral,
                moments, fields, vpa_advect, vperp_advect, z_advect, r_advect,
                vpa_SL, vperp_SL, z_SL, r_SL,
                composition, collisions, geometry, advance,  scratch_dummy, i)
        # update the time
        t += t_input.dt
        # write data to file every nwrite time steps
        if mod(i,t_input.nwrite) == 0
            @debug_detect_redundant_block_synchronize begin
                # Skip check for redundant _block_synchronize() during file I/O because
                # it only runs infrequently
                debug_detect_redundant_is_active[] = false
            end
            begin_serial_region()
            @serial_region println("finished time step ", i)
            write_data_to_ascii(pdf.unnorm, moments, fields,
             vpa, vperp, z, r, t, composition.n_species, io)
            # write initial data to binary file (netcdf)
            write_data_to_binary(pdf.unnorm, moments, fields, t, composition.n_species, cdf, iwrite)
            iwrite += 1
            begin_s_r_z_vperp_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end
    end
    return nothing
end

"""
"""
function time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, vperp, z, r, 
    vpa_spectral, vperp_spectral, z_spectral, r_spectral,
    moments, fields, vpa_advect, vperp_advect, z_advect, r_advect,
    vpa_SL, vperp_SL, z_SL, r_SL, 
    composition, collisions, geometry, advance, scratch_dummy, istep)

    if t_input.n_rk_stages > 1
        ssp_rk!(pdf, scratch, t, t_input, vpa, vperp, z, r, 
            vpa_spectral, vperp_spectral, z_spectral, r_spectral,
            moments, fields, vpa_advect, vperp_advect, z_advect, r_advect,
            vpa_SL, vperp_SL, z_SL, r_SL, composition, collisions, geometry, advance,  scratch_dummy, istep)
    else
        euler_time_advance!(scratch, scratch, pdf, fields, moments,
            vpa_SL, vperp_SL, z_SL, r_SL,
            vpa_advect, vperp_advect, z_advect, r_advect, vpa, vperp, z, r, t,
            t_input, vpa_spectral, vperp_spectral, z_spectral, r_spectral, composition,
            collisions, geometry, scratch_dummy, advance, 1)
        # NB: this must be broken -- scratch is updated in euler_time_advance!,
        # but not the pdf or moments.  need to add update to these quantities here
    end
    return nothing
end

"""
"""
function rk_update!(scratch, pdf, moments, fields, vpa, vperp, z, r, rk_coefs, istage, composition, z_spectral, r_spectral)
    begin_s_r_z_vperp_region()
    nvpa = size(pdf.unnorm, 1)
    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_scratch.pdf[ivpa,ivperp,iz,ir,is] = rk_coefs[1]*pdf.norm[ivpa,ivperp,iz,ir,is] + rk_coefs[2]*old_scratch.pdf[ivpa,ivperp,iz,ir,is] + rk_coefs[3]*new_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.unnorm[ivpa,ivperp,iz,ir,is] = new_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    update_density!(new_scratch.density, moments.dens_updated, pdf.unnorm, vpa, vperp, z, r, composition)
    
    update_upar!(new_scratch.upar, moments.upar_updated, pdf.unnorm, vpa, vperp, z, r, composition)
    # convert from particle particle flux to parallel flow
    @loop_s_r_z is ir iz begin
        new_scratch.upar[iz,ir,is] /= new_scratch.density[iz,ir,is]
    end
    
    update_ppar!(new_scratch.ppar, moments.ppar_updated, pdf.unnorm, vpa, vperp, z, r, composition)
    # update the thermal speed
    @loop_s_r_z is ir iz begin
        moments.vth[iz,ir,is] = sqrt(2.0*new_scratch.ppar[iz,ir,is]/new_scratch.density[iz,ir,is])
    end
    
    # update the parallel heat flux
    update_qpar!(moments.qpar, moments.qpar_updated, pdf.unnorm, vpa, vperp, z, r, composition, moments.vpa_norm_fac)
    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], z, r, composition, z_spectral, r_spectral)
    # _block_synchronize() here because phi needs to be read on different ranks than it
    # was written on, even though the loop-type does not change here
    _block_synchronize()
end

"""
"""
function ssp_rk!(pdf, scratch, t, t_input, vpa, vperp, z, r, 
    vpa_spectral, vperp_spectral, z_spectral, r_spectral,
    moments, fields, vpa_advect, vperp_advect, z_advect, r_advect,
    vpa_SL, vperp_SL, z_SL, r_SL, composition, collisions, geometry, advance, scratch_dummy, istep)

    n_rk_stages = t_input.n_rk_stages

    first_scratch = scratch[1]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        first_scratch.pdf[ivpa,ivperp,iz,ir,is] = pdf.norm[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.upar[iz,ir,is]
        first_scratch.ppar[iz,ir,is] = moments.ppar[iz,ir,is]
    end
    
    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage, composition, vpa, vperp, z, r)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments, vpa_SL, vperp_SL, z_SL, r_SL,
            vpa_advect, vperp_advect, z_advect, r_advect, vpa, vperp, z, r, t,
            t_input, vpa_spectral, vperp_spectral, z_spectral, r_spectral, composition,
            collisions, geometry, scratch_dummy, advance, istage)
        @views rk_update!(scratch, pdf, moments, fields, vpa, vperp, z, r, advance.rk_coefs[:,istage], istage, composition, z_spectral, r_spectral)
    end

    istage = n_rk_stages+1
    
    # update the pdf.norm and moments arrays as needed
    final_scratch = scratch[istage]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.norm[ivpa,ivperp,iz,ir,is] = final_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        moments.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
        moments.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
        moments.ppar[iz,ir,is] = final_scratch.ppar[iz,ir,is]
    end
    update_pdf_unnorm!(pdf, moments, scratch[istage].temp_z_s, composition, vpa, vperp)
    return nothing
end

"""
euler_time_advance! advances the vector equation dfvec/dt = G[f]
that includes the kinetic equation + any evolved moment equations
using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
with fvec_in an input and fvec_out the output
"""
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments, vpa_SL, vperp_SL, z_SL, r_SL,
    vpa_advect, vperp_advect, z_advect, r_advect, vpa, vperp, z, r, t, t_input,
    vpa_spectral, vperp_spectral, z_spectral, r_spectral, composition, collisions, geometry,
    scratch_dummy, advance, istage)
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa
    
    if advance.vpa_advection
        vpa_advection!(fvec_out.pdf, fvec_in, pdf.norm, fields, moments,
            vpa_SL, vpa_advect, vpa, vperp, z, r, use_semi_lagrange, dt, t,
            vpa_spectral, z_spectral, composition, collisions.charge_exchange,
            geometry, istage)
    end
    
    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to all species (charged and neutral)
    
    
    # MRH UNSURE ABOUT PARALLELISATION HERE
    # z advection relies on derivatives in r to get ExB
    if advance.z_advection
        begin_s_r_vperp_vpa_region()
        z_advection!(fvec_out.pdf, fvec_in, pdf.norm, fields, moments, z_SL, z_advect, z, vpa, vperp, r, 
            use_semi_lagrange, dt, t, z_spectral, r_spectral, composition, geometry, scratch_dummy, istage)
        begin_s_r_z_vperp_region()
        #  SHOULD NOT CHANGE TO VPA STRUCTURE YET
    end
    # MRH UNSURE ABOUT PARALLELISATION HERE
    
    # MRH UNSURE ABOUT PARALLELISATION HERE  
    # r advection relies on derivatives in z to get ExB
    if advance.r_advection && r.n > 1
        r_advection!(fvec_out.pdf, fvec_in, pdf.norm, fields, moments, r_SL, r_advect, r, z, vperp, vpa, 
            use_semi_lagrange, dt, t, r_spectral, z_spectral, composition, geometry, scratch_dummy, istage)
    end 
    # MRH UNSURE ABOUT PARALLELISATION HERE
    
    #if advance.vperp_advection
    # PLACEHOLDER 
    #end 
    
    # account for charge exchange collisions between ions and neutrals
    if advance.cx_collisions
        charge_exchange_collisions!(fvec_out.pdf, fvec_in, moments, composition, vpa, vperp, z, r,
                                    collisions.charge_exchange, dt)
    end
    # account for ionization collisions between ions and neutrals
    if advance.ionization_collisions
        ionization_collisions!(fvec_out.pdf, fvec_in, moments, n_ion_species,
            composition.n_neutral_species, vpa, vperp, z, r, composition, collisions, z.n, dt)
    end
    
    # enforce boundary conditions in z and vpa on the distribution function
    enforce_boundary_conditions!(fvec_out.pdf, vpa.bc, z.bc, r.bc, vpa, vperp, z, r,
     vpa_advect, z_advect, r_advect, composition)
    # End of advance for distribution function

    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments, composition, z)
    return nothing
end

"""
update the vector containing the pdf and any evolved moments of the pdf
for use in the Runge-Kutta time advance
"""
function update_solution_vector!(evolved, moments, istage, composition, vpa, vperp, z, r)
    new_evolved = evolved[istage+1]
    old_evolved = evolved[istage]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_evolved.pdf[ivpa,ivperp,iz,ir,is] = old_evolved.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        new_evolved.density[iz,ir,is] = old_evolved.density[iz,ir,is]
        new_evolved.upar[iz,ir,is] = old_evolved.upar[iz,ir,is]
        new_evolved.ppar[iz,ir,is] = old_evolved.ppar[iz,ir,is]
    end
    return nothing
end

"""
if separately evolving the density via the continuity equation,
the evolved pdf has been normalised by the particle density
undo this normalisation to get the true particle distribution function

scratch should be a (nz,nspecies) array
"""
function update_pdf_unnorm!(pdf, moments, scratch, composition, vpa, vperp)
    nvpa = size(pdf.unnorm, 1)
    if moments.evolve_ppar
        @loop_s_r_z is ir iz begin
            scratch[iz,ir,is] = moments.dens[iz,ir,is]/moments.vth[iz,ir,is]
        end
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            pdf.unnorm[ivpa,ivperp,iz,ir,is] = pdf.norm[ivpa,ivperp,iz,ir,is]*scratch[iz,ir,is]
        end
    elseif moments.evolve_density
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            pdf.unnorm[ivpa,ivperp,iz,ir,is] = pdf.norm[ivpa,ivperp,iz,ir,is] * moments.dens[iz,ir,is]
        end
    else
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            pdf.unnorm[ivpa,ivperp,iz,ir,is] = pdf.norm[ivpa,ivperp,iz,ir,is]
        end
    end
end

end
