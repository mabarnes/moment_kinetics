module time_advance

export setup_time_advance!
export time_advance!

using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication
using ..communication: _block_synchronize
using ..debugging
using ..file_io: write_data_to_ascii, write_data_to_binary
using ..looping
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

struct scratch_pdf{n_distribution, n_moment}
    pdf::MPISharedArray{mk_float, n_distribution}
    density::MPISharedArray{mk_float, n_moment}
    upar::MPISharedArray{mk_float, n_moment}
    ppar::MPISharedArray{mk_float, n_moment}
    temp_z_s::MPISharedArray{mk_float, n_moment}
end
mutable struct advance_info
    vpa_advection::Bool
    z_advection::Bool
    cx_collisions::Bool
    ionization_collisions::Bool
    source_terms::Bool
    continuity::Bool
    force_balance::Bool
    energy::Bool
    rk_coefs::Array{mk_float,2}
end

# create arrays and do other work needed to setup
# the main time advance loop.
# this includes creating and populating structs
# for Chebyshev transforms, velocity space moments,
# EM fields, semi-Lagrange treatment, and advection terms
function setup_time_advance!(pdf, vpa, z, r, composition, drive_input, moments,
                             t_input, collisions, species)
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
        advance = advance_info(false, false, false, false, false, false, false, false, rk_coefs)
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
        advance = advance_info(true, true, advance_cx, advance_ionization, advance_sources,
                               advance_continuity, advance_force_balance, advance_energy, rk_coefs)
    end
    
    # create structure r_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in r
    begin_serial_region()
    r_advect = setup_advection(n_species, r, vpa, z)
    # initialise the r advection speed
    begin_s_z_vpa_region()
    @s_z_vpa_loop_s is begin
        @views update_speed_r!(r_advect[is], moments.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(r_advect[is], loop_ranges[].s_z_vpa_range_vpa,
         loop_ranges[].s_z_vpa_range_z)
    end
    # enforce prescribed boundary condition in r on the distribution function f
    # PLACEHOLDER
    #@views enforce_r_boundary_condition!(pdf.unnorm, r.bc, r_advect, vpa, z, composition)
    
    
    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in z
    begin_serial_region()
    z_advect = setup_advection(n_species, z, vpa, r)
    # initialise the z advection speed
    begin_s_r_vpa_region()
    @s_r_vpa_loop_s is begin
        @views update_speed_z!(z_advect[is], moments.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(z_advect[is], loop_ranges[].s_r_vpa_range_vpa,
         loop_ranges[].s_r_vpa_range_r)
    end
    # enforce prescribed boundary condition in z on the distribution function f
    @views enforce_z_boundary_condition!(pdf.unnorm, z.bc, z_advect, vpa, r, composition)
    if z.bc != "wall" || composition.n_neutral_species == 0
        begin_serial_region()
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
    
    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf.norm, t_input.n_rk_stages)
    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, r.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency)
    # initialize the electrostatic potential
    begin_s_r_z_region()
    update_phi!(fields, scratch[1], z, r, composition)
    begin_serial_region()
    @serial_region begin
        # save the initial phi(z) for possible use later (e.g., if forcing phi)
        fields.phi0 .= fields.phi
    end
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_advect = setup_advection(n_species, vpa, z, r)
    # initialise the vpa advection speed
    begin_s_r_z_region()
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, z, r, composition,
                      collisions.charge_exchange, 0.0, z_spectral)
    if moments.evolve_upar
        nspec = n_species
    else
        nspec = n_ion_species
    end
    begin_serial_region()
    @serial_region begin
        for is ∈ 1:nspec
            # initialise the upwind/downwind boundary indices in vpa
            update_boundary_indices!(vpa_advect[is], 1:z.n, 1:r.n)
            # enforce prescribed boundary condition in vpa on the distribution function f
            @views enforce_vpa_boundary_condition!(pdf.norm[:,:,:,is], vpa.bc, vpa_advect[is])
        end
    end
    # create an array of structures containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n, vpa.n, r.n)
    vpa_SL = setup_semi_lagrange(vpa.n, z.n, r.n)
    r_SL = setup_semi_lagrange(r.n, vpa.n, z.n)

    begin_s_z_region()
    return vpa_spectral, z_spectral, r_spectral, moments, fields, vpa_advect, z_advect, r_advect,
        vpa_SL, z_SL, r_SL, scratch, advance
end

   
# if evolving the density via continuity equation, redefine the normalised f → f/n
# if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
# 'scratch' should be a (nz,nspecies) array
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
# create an array of structs containing scratch arrays for the normalised pdf and low-order moments
# that may be evolved separately via fluid equations
function setup_scratch_arrays(moments, pdf_in, n_rk_stages)
    # create n_rk_stages+1 structs, each of which will contain one pdf,
    # one density, and one parallel flow array
    scratch = Vector{scratch_pdf{4,3}}(undef, n_rk_stages+1)
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
# given the number of Runge Kutta stages that are requested,
# returns the needed Runge Kutta coefficients;
# e.g., if f is the function to be updated, then
# f^{n+1}[stage+1] = rk_coef[1,stage]*f^{n} + rk_coef[2,stage]*f^{n+1}[stage] + rk_coef[3,stage]*(f^{n}+dt*G[f^{n+1}[stage]]
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
# solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
# define approximate characteristic velocity
# v₀(z)=vⁿ(z) and take time derivative along this characteristic
# df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
# for prudent choice of v₀, expect δv≪v so that explicit
# time integrator can be used without severe CFL condition
function time_advance!(pdf, scratch, t, t_input, vpa, z, vpa_spectral, z_spectral,
    moments, fields, vpa_advect, z_advect, vpa_SL, z_SL, composition,
    collisions, advance, io, cdf)

    @debug_detect_redundant_block_synchronize begin
        # Only want to check for redundant _block_synchronize() calls during the
        # time advance loop, so activate these checks here
        debug_detect_redundant_is_active[] = true
    end

    # main time advance loop
    iwrite = 2
    for i ∈ 1:t_input.nstep
        if t_input.split_operators
            time_advance_split_operators!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, i)
        else
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, i)
        end
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
            write_data_to_ascii(pdf.unnorm, moments, fields, vpa, z, t, composition.n_species, io)
            # write initial data to binary file (netcdf)
            write_data_to_binary(pdf.unnorm, moments, fields, t, composition.n_species, cdf, iwrite)
            iwrite += 1
            begin_s_z_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end
    end
    return nothing
end
function time_advance_split_operators!(pdf, scratch, t, t_input, vpa, z,
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
    vpa_SL, z_SL, composition, collisions, advance, istep)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    dt = t_input.dt
    n_rk_stages = t_input.n_rk_stages
    use_semi_lagrange = t_input.use_semi_lagrange
    # to ensure 2nd order accuracy in time for operator-split advance,
    # have to reverse order of operations every other time step
    flipflop = (mod(istep,2)==0)
    if flipflop
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, advance, istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, advance, istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, advance, istep)
                advance.cx_collisions = false
            end
            if collisions.ionization > 0.0
                advance.ionization_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, collisions, advance, istep)
                advance.ionization_collisions = false
            end
        end
        # use the continuity equation to update the density
        # and add the source terms associated with redefining g = pdf/density to the kinetic equation
        if moments.evolve_density
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, istep)
            advance.source_terms = false
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, istep)
            advance.continuity = false
            if moments.evolve_upar
                advance.force_balance = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, advance, istep)
                advance.force_balance = false
                if moments.evolve_ppar
                    advance.energy = true
                    time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                        vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                        vpa_SL, z_SL, composition, collisions, advance, istep)
                    advance.energy = false
                end
            end
        end
    else
        if moments.evolve_upar
            if moments.evolve_ppar
                advance.energy = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, advance, istep)
                advance.energy = false
            end
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, istep)
            advance.force_balance = false
        end
        # use the continuity equation to update the density
        # and add the source terms associated with redefining g = pdf/density to the kinetic equation
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, istep)
            advance.continuity = false
            # use force balance to update the parallel flow
            # and subsequently add the source terms associated with using the peculiar velocity as a variable
            if moments.evolve_parallel_flow
                advance.force_balance = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, advance, istep)
                advance.force_balance = false
            end
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, advance, istep)
            advance.source_terms = false
        end
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, advance, istep)
                advance.cx_collisions = false
            end
            if collisions.ionization > 0.0
                advance.ionization = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, collisions, advance, istep)
                advance.ionization = false
            end
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, advance, istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, advance, istep)
        advance.vpa_advection = false
    end
    return nothing
end
function time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
    vpa_SL, z_SL, composition, collisions, advance, istep)

    if t_input.n_rk_stages > 1
        ssp_rk!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, advance, istep)
    else
        euler_time_advance!(scratch, scratch, pdf, fields, moments, vpa_SL, z_SL,
            vpa_advect, z_advect, vpa, z, t,
            t_input, vpa_spectral, z_spectral, composition,
            collisions, advance, 1)
        # NB: this must be broken -- scratch is updated in euler_time_advance!,
        # but not the pdf or moments.  need to add update to these quantities here
    end
    return nothing
end
function rk_update!(scratch, pdf, moments, fields, vpa, z, rk_coefs, istage, composition)
    begin_s_z_region()
    nvpa = size(pdf.unnorm, 1)
    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]
    @s_z_loop is iz begin
        for ivpa ∈ 1:nvpa
            new_scratch.pdf[ivpa,iz,is] = rk_coefs[1]*pdf.norm[ivpa,iz,is] + rk_coefs[2]*old_scratch.pdf[ivpa,iz,is] + rk_coefs[3]*new_scratch.pdf[ivpa,iz,is]
        end
    end
    if moments.evolve_density
        @s_z_loop is iz begin
            new_scratch.density[iz,is] = rk_coefs[1]*moments.dens[iz,is] + rk_coefs[2]*old_scratch.density[iz,is] + rk_coefs[3]*new_scratch.density[iz,is]
        end
        @s_z_loop is iz begin
            for ivpa ∈ 1:nvpa
                pdf.unnorm[ivpa,iz,is] = new_scratch.pdf[ivpa,iz,is] * new_scratch.density[iz,is]
            end
        end
    else
        @s_z_loop is iz begin
            for ivpa ∈ 1:nvpa
                pdf.unnorm[ivpa,iz,is] = new_scratch.pdf[ivpa,iz,is]
            end
        end
        update_density!(new_scratch.density, moments.dens_updated, pdf.unnorm, vpa, z, composition)
    end
    # NB: if moments.evolve_upar = true, then moments.evolve_density = true
    if moments.evolve_upar
        @s_z_loop is iz begin
            new_scratch.upar[iz,is] = rk_coefs[1]*moments.upar[iz,is] + rk_coefs[2]*old_scratch.upar[iz,is] + rk_coefs[3]*new_scratch.upar[iz,is]
        end
    else
        update_upar!(new_scratch.upar, moments.upar_updated, pdf.unnorm, vpa, z, composition)
        # convert from particle particle flux to parallel flow
        @s_z_loop is iz begin
            new_scratch.upar[iz,is] /= new_scratch.density[iz,is]
        end
    end
    if moments.evolve_ppar
        @s_z_loop is iz begin
            new_scratch.ppar[iz,is] = rk_coefs[1]*moments.ppar[iz,is] + rk_coefs[2]*old_scratch.ppar[iz,is] + rk_coefs[3]*new_scratch.ppar[iz,is]
        end
    else
        update_ppar!(new_scratch.ppar, moments.ppar_updated, pdf.unnorm, vpa, z, composition)
    end
    # update the thermal speed
    @s_z_loop is iz begin
        moments.vth[iz,is] = sqrt(2.0*new_scratch.ppar[iz,is]/new_scratch.density[iz,is])
    end
    if moments.evolve_ppar
        @s_z_loop is iz begin
            old_scratch.temp_z_s[iz,is] = 1.0 / moments.vth[iz,is]
        end
        @s_z_loop is iz begin
            for ivpa ∈ 1:vpa.n
                pdf.unnorm[ivpa,iz,is] *= old_scratch.temp_z_s[iz,is]
            end
        end
    end
    # update the parallel heat flux
    update_qpar!(moments.qpar, moments.qpar_updated, pdf.unnorm, vpa, z, composition, moments.vpa_norm_fac)
    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], z, composition)
    # _block_synchronize() here because phi needs to be read on different ranks than it
    # was written on, even though the loop-type does not change here
    _block_synchronize()
end
function ssp_rk!(pdf, scratch, t, t_input, vpa, z,
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
    vpa_SL, z_SL, composition, collisions, advance, istep)

    n_rk_stages = t_input.n_rk_stages

    first_scratch = scratch[1]
    @s_z_loop is iz begin
        for ivpa ∈ 1:vpa.n
            first_scratch.pdf[ivpa,iz,is] = pdf.norm[ivpa,iz,is]
        end
    end
    @s_z_loop is iz begin
        first_scratch.density[iz,is] = moments.dens[iz,is]
        first_scratch.upar[iz,is] = moments.upar[iz,is]
        first_scratch.ppar[iz,is] = moments.ppar[iz,is]
    end
    if moments.evolve_upar
        # moments may be read on all ranks, even though loop type is z_s, so need to
        # synchronize here
        _block_synchronize()
    end

    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage, composition, vpa, z)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments, vpa_SL, z_SL, vpa_advect, z_advect, vpa, z, t,
            t_input, vpa_spectral, z_spectral, composition,
            collisions, advance, istage)
        @views rk_update!(scratch, pdf, moments, fields, vpa, z, advance.rk_coefs[:,istage], istage, composition)
    end

    istage = n_rk_stages+1
    if moments.evolve_density && moments.enforce_conservation
        enforce_moment_constraints!(scratch[istage], scratch[1], vpa, z, composition, moments)
    end

    # update the pdf.norm and moments arrays as needed
    final_scratch = scratch[istage]
    @s_z_loop is iz begin
        for ivpa ∈ 1:vpa.n
            pdf.norm[ivpa,iz,is] = final_scratch.pdf[ivpa,iz,is]
        end
    end
    @s_z_loop is iz begin
        moments.dens[iz,is] = final_scratch.density[iz,is]
        moments.upar[iz,is] = final_scratch.upar[iz,is]
        moments.ppar[iz,is] = final_scratch.ppar[iz,is]
    end
    update_pdf_unnorm!(pdf, moments, scratch[istage].temp_z_s, composition, vpa)
    return nothing
end
# euler_time_advance! advances the vector equation dfvec/dt = G[f]
# that includes the kinetic equation + any evolved moment equations
# using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
# with fvec_in an input and fvec_out the output
# MRH entering in s_z_region (?)
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments, vpa_SL, z_SL,
    vpa_advect, z_advect, vpa, z, t, t_input, vpa_spectral, z_spectral,
    composition, collisions, advance, istage)
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa
    if advance.vpa_advection
        vpa_advection!(fvec_out.pdf, fvec_in, pdf.norm, fields, moments,
            vpa_SL, vpa_advect, vpa, z, use_semi_lagrange, dt, t,
            vpa_spectral, z_spectral, composition, collisions.charge_exchange, istage)
    end
    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to all species (charged and neutral)
    if advance.z_advection
        begin_s_vpa_region()
        z_advection!(fvec_out.pdf, fvec_in, pdf.norm, moments, z_SL, z_advect, z, vpa,
            use_semi_lagrange, dt, t, z_spectral, composition, istage)
        begin_s_z_region()
    end
    if advance.source_terms
        source_terms!(fvec_out.pdf, fvec_in, moments, vpa, z, dt, z_spectral,
                      composition, collisions.charge_exchange)
    end
    # account for charge exchange collisions between ions and neutrals
    if advance.cx_collisions
        charge_exchange_collisions!(fvec_out.pdf, fvec_in, moments, composition, vpa, z,
                                    collisions.charge_exchange, dt)
    end
    # account for ionization collisions between ions and neutrals
    if advance.ionization_collisions
        ionization_collisions!(fvec_out.pdf, fvec_in, moments, n_ion_species,
            composition.n_neutral_species, vpa, z, composition, collisions, z.n, dt)
    end
    if advance.continuity
        continuity_equation!(fvec_out.density, fvec_in, moments, composition, vpa, z,
                             dt, z_spectral)
    end
    if advance.force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        force_balance!(fvec_out.upar, fvec_in, fields, collisions, vpa, z, dt, z_spectral, composition)
        # convert from the particle flux to the parallel flow
        @s_z_loop_s is begin
            if 1 ∈ loop_ranges[].s_z_range_z
                @views @. fvec_out.upar[:,is] /= fvec_out.density[:,is]
            end
        end
    end
    if advance.energy
        energy_equation!(fvec_out.ppar, fvec_in, moments, collisions, z, dt, z_spectral, composition)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments, composition, z)
    # enforce boundary conditions in z and vpa on the distribution function
    # NB: probably need to do the same for the evolved moments
    enforce_boundary_conditions!(fvec_out.pdf, vpa.bc, z.bc, vpa, z, vpa_advect, z_advect, composition)
    return nothing
end
# update the vector containing the pdf and any evolved moments of the pdf
# for use in the Runge-Kutta time advance
function update_solution_vector!(evolved, moments, istage, composition, vpa, z)
    new_evolved = evolved[istage+1]
    old_evolved = evolved[istage]
    @s_z_loop is iz begin
        for ivpa ∈ 1:vpa.n
            new_evolved.pdf[ivpa,iz,is] = old_evolved.pdf[ivpa,iz,is]
        end
    end
    @s_z_loop is iz begin
        new_evolved.density[iz,is] = old_evolved.density[iz,is]
        new_evolved.upar[iz,is] = old_evolved.upar[iz,is]
        new_evolved.ppar[iz,is] = old_evolved.ppar[iz,is]
    end
    return nothing
end

# scratch should be a (nz,nspecies) array
function update_pdf_unnorm!(pdf, moments, scratch, composition, vpa)
    # if separately evolving the density via the continuity equation,
    # the evolved pdf has been normalised by the particle density
    # undo this normalisation to get the true particle distribution function
    nvpa = size(pdf.unnorm, 1)
    if moments.evolve_ppar
        @s_z_loop is iz begin
            scratch[iz,is] = moments.dens[iz,is]/moments.vth[iz,is]
        end
        @s_z_loop is iz begin
            for ivpa ∈ 1:vpa.n
                pdf.unnorm[ivpa,iz,is] = pdf.norm[ivpa,iz,is]*scratch[iz,is]
            end
        end
    elseif moments.evolve_density
        @s_z_loop is iz begin
            for ivpa ∈ 1:vpa.n
                pdf.unnorm[ivpa,iz,is] = pdf.norm[ivpa,iz,is] * moments.dens[iz,is]
            end
        end
    else
        @s_z_loop is iz begin
            for ivpa ∈ 1:vpa.n
                pdf.unnorm[ivpa,iz,is] = pdf.norm[ivpa,iz,is]
            end
        end
    end
end

end
