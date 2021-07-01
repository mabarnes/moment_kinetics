struct scratch_pdf{T1,T2}
    pdf::T1
    density::T2
    upar::T2
    ppar::T2
end
mutable struct advance_info
    vpa_advection::Bool
    z_advection::Bool
    cx_collisions::Bool
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
function setup_time_advance!(pdf, z, vpa, composition, drive_input, evolve_moments,
                             t_input, cx_frequency, species)
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
        advance = advance_info(false, false, false, false, false, false, false, rk_coefs)
    else
        if composition.n_neutral_species > 0 && cx_frequency > 0.0
            advance_cx = true
        else
            advance_cx = false
        end
        if evolve_moments.density
            advance_sources = true
            advance_continuity = true
            if evolve_moments.parallel_flow
                advance_force_balance = true
                if evolve_moments.parallel_pressure
                    advance_energy = true
                else
                    advance_energy = false
                end
            else
                advance_force_balance = false
            end
        else
            advance_sources = false
            advance_continuity = false
            advance_force_balance = false
            advance_energy = false
        end
        advance = advance_info(true, true, advance_cx, advance_sources, advance_continuity,
                               advance_force_balance, advance_energy, rk_coefs)
    end
    # pass the distribution function pdf.unnorm (defined such that ∫dvpa pdf = density)
    # and allocate/initialize the velocity space moments needed for advancing
    # the kinetic equation coupled to fluid equations
    # the resulting moments are returned in the structure "moments"
    moments = setup_moments(pdf.unnorm, vpa, z.n, evolve_moments, species)
    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in z
    z_advect = setup_advection(z, vpa, n_species)
    # initialise the z advection speed
    for is ∈ 1:n_species
        @views update_speed_z!(z_advect[:,is], moments.upar[:,is], moments.vth[:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(view(z_advect,:,is))
        # enforce prescribed boundary condition in z on the distribution function f
        @views enforce_z_boundary_condition!(pdf.unnorm[:,:,is], z.bc, vpa, z_advect[:,is])
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
    # redefine the distrubtion function pdf.norm = pdf.unnorm(z,vpa)/n(z) if this option is chosen
    normalize_pdf!(pdf.norm, moments)
    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf.norm, z.n, vpa.n, n_species, t_input.n_rk_stages)
    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency)
    # initialize the electrostatic potential
    #update_phi!(fields, moments, scratch[1], vpa, z.n, composition, 0.0)
    update_phi!(fields, scratch[1], vpa, z.n, composition)
    # save the initial phi(z) for possible use later (e.g., if forcing phi)
    fields.phi0 .= fields.phi
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_advect = setup_advection(vpa, z, n_species)
    # initialise the vpa advection speed
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, z, composition, cx_frequency, 0.0, z_spectral)
    if moments.evolve_upar
        nspec = n_species
    else
        nspec = n_ion_species
    end
    for is ∈ 1:nspec
        # initialise the upwind/downwind boundary indices in vpa
        update_boundary_indices!(view(vpa_advect,:,is))
        # enforce prescribed boundary condition in vpa on the distribution function f
        @views enforce_vpa_boundary_condition!(pdf.norm[:,:,is], vpa.bc, vpa_advect[:,is])
    end
    # create an array of structures containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n, vpa.n)
    vpa_SL = setup_semi_lagrange(vpa.n, z.n)
    return z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect, z_SL,
        vpa_SL, scratch, advance
end
# if evolving the density via continuity equation, redefine the normalised f → f/n
# if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
function normalize_pdf!(pdf, moments)
    nvpa = size(pdf,2)
    if moments.evolve_ppar
        for ivpa ∈ 1:nvpa
            @. pdf[:,ivpa,:] *= moments.vth/moments.dens
        end
    elseif moments.evolve_density
        for ivpa ∈ 1:nvpa
            @. pdf[:,ivpa,:] /= moments.dens
        end
    end
    return nothing
end
# create an array of structs containing scratch arrays for the normalised pdf and low-order moments
# that may be evolved separately via fluid equations
function setup_scratch_arrays(moments, pdf_in, nz, nvpa, nspec, n_rk_stages)
    # create n_rk_stages+1 pdf-sized scratch arrays
    pdf = allocate_float(nz, nvpa, nspec, n_rk_stages+1)
    # create n_rk_stages+1 density-sized scratch arrays
    dens = allocate_float(nz, nspec, n_rk_stages+1)
    # create n_rk_stages+1 parallel-flow-sized scratch arrays
    upar = allocate_float(nz, nspec, n_rk_stages+1)
    # create n_rk_stages+1 parallel-pressure-sized scratch arrays
    ppar = allocate_float(nz, nspec, n_rk_stages+1)
    # create n_rk_stages+1 structs, each of which will contain one pdf,
    # one density, and one parallel flow array
    scratch = Vector{scratch_pdf}(undef, n_rk_stages+1)
    # populate each of the structs
    # NB: all of the array members of the scratch struct will point to
    # the appropriate slice of the pdf, dens and upar arrays created above
    for istage ∈ 1:n_rk_stages+1
        @views scratch[istage] = scratch_pdf(pdf[:,:,:,istage], dens[:,:,istage], upar[:,:,istage], ppar[:,:,istage])
        scratch[istage].pdf .= pdf_in
        scratch[istage].density .= moments.dens
        scratch[istage].upar .= moments.upar
        scratch[istage].ppar .= moments.ppar
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
function time_advance!(pdf, scratch, t, t_input, z, vpa, z_spectral, vpa_spectral,
    moments, fields, z_advect, vpa_advect, z_SL, vpa_SL, composition,
    charge_exchange_frequency, advance, io, cdf)

    # main time advance loop
    iwrite = 2
    for i ∈ 1:t_input.nstep
        if t_input.split_operators
            time_advance_split_operators!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, i)
        else
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, i)
        end
        # update the time
        t += t_input.dt
        # write data to file every nwrite time steps
        if mod(i,t_input.nwrite) == 0
            println("finished time step ", i)
            write_data_to_ascii(pdf.unnorm, moments, fields, z, vpa, t, composition.n_species, io)
            # write initial data to binary file (netcdf)
            write_data_to_binary(pdf.unnorm, moments, fields, t, composition.n_species, cdf, iwrite)
            iwrite += 1
        end
    end
    return nothing
end
function time_advance_split_operators!(pdf, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

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
        time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            advance.cx_collisions = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.cx_collisions = false
        end
        # use the continuity equation to update the density
        # and add the source terms associated with redefining g = pdf/density to the kinetic equation
        if moments.evolve_density
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.source_terms = false
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.continuity = false
            if moments.evolve_upar
                advance.force_balance = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
                advance.force_balance = false
                if moments.evolve_ppar
                    advance.energy = true
                    time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                        z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                        z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
                    advance.energy = false
                end
            end
        end
    else
        if moments.evolve_upar
            if moments.evolve_ppar
                advance.energy = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
                advance.energy = false
            end
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.force_balance = false
        end
        # use the continuity equation to update the density
        # and add the source terms associated with redefining g = pdf/density to the kinetic equation
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.continuity = false
            # use force balance to update the parallel flow
            # and subsequently add the source terms associated with using the peculiar velocity as a variable
            if moments.evolve_parallel_flow
                advance.force_balance = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
                advance.force_balance = false
            end
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.source_terms = false
        end
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            advance.cx_collisions = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
            advance.cx_collisions = false
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.vpa_advection = false
    end
    return nothing
end
function time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

    if t_input.n_rk_stages > 1
        ssp_rk!(pdf, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
    else
        euler_time_advance!(scratch, scratch, pdf, fields, moments, z_SL, vpa_SL,
            z_advect, vpa_advect, z, vpa, t,
            t_input, z_spectral, vpa_spectral, composition,
            charge_exchange_frequency, advance, 1)
        # NB: this must be broken -- scratch is updated in euler_time_advance!,
        # but not the pdf or moments.  need to add update to these quantities here
    end
    return nothing
end
function rk_update!(scratch, pdf, moments, fields, vpa, nz, rk_coefs, istage, composition)
    @. scratch[istage+1].pdf = rk_coefs[1]*pdf.norm + rk_coefs[2]*scratch[istage].pdf + rk_coefs[3]*scratch[istage+1].pdf
    if moments.evolve_density
        @. scratch[istage+1].density = rk_coefs[1]*moments.dens + rk_coefs[2]*scratch[istage].density + rk_coefs[3]*scratch[istage+1].density
        for ivpa ∈ 1:vpa.n
            @. pdf.unnorm[:,ivpa,:] = scratch[istage+1].pdf[:,ivpa,:]*scratch[istage+1].density
        end
    else
        for ivpa ∈ 1:vpa.n
            @. pdf.unnorm[:,ivpa,:] = scratch[istage+1].pdf[:,ivpa,:]
        end
        update_density!(scratch[istage+1].density, moments.dens_updated, pdf.unnorm, vpa, nz)
    end
    # NB: if moments.evolve_upar = true, then moments.evolve_density = true
    if moments.evolve_upar
        @. scratch[istage+1].upar = rk_coefs[1]*moments.upar + rk_coefs[2]*scratch[istage].upar + rk_coefs[3]*scratch[istage+1].upar
    else
        update_upar!(scratch[istage+1].upar, moments.upar_updated, pdf.unnorm, vpa, nz)
        # convert from particle particle flux to parallel flow
        @. scratch[istage+1].upar /= scratch[istage+1].density
    end
    if moments.evolve_ppar
        @. scratch[istage+1].ppar = rk_coefs[1]*moments.ppar + rk_coefs[2]*scratch[istage].ppar + rk_coefs[3]*scratch[istage+1].ppar
    else
        update_ppar!(scratch[istage+1].ppar, moments.ppar_updated, pdf.unnorm, vpa, nz)
    end
    # update the thermal speed
    @. moments.vth = sqrt(2*moments.ppar/moments.dens)
    # update the parallel heat flux
    update_qpar!(moments.qpar, moments.qpar_updated, pdf.unnorm, vpa, nz, moments.vpa_norm_fac)
    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], vpa, nz, composition)
end
function ssp_rk!(pdf, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

    n_rk_stages = t_input.n_rk_stages

    scratch[1].pdf .= pdf.norm
    scratch[1].density .= moments.dens
    scratch[1].upar .= moments.upar
    scratch[1].ppar .= moments.ppar

    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        @views euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments, z_SL, vpa_SL, z_advect, vpa_advect, z, vpa, t,
            t_input, z_spectral, vpa_spectral, composition,
            charge_exchange_frequency, advance, istage)
        @views rk_update!(scratch, pdf, moments, fields, vpa, z.n, advance.rk_coefs[:,istage], istage, composition)
    end

    istage = n_rk_stages+1
    if moments.evolve_density && moments.enforce_conservation
        enforce_moment_constraints!(scratch[istage], scratch[1], z, vpa, moments)
    end

    # update the pdf.norm and moments arrays as needed
    pdf.norm .= scratch[istage].pdf
    moments.dens .= scratch[istage].density
    moments.upar .= scratch[istage].upar
    moments.ppar .= scratch[istage].ppar
    update_pdf_unnorm!(pdf, moments)
    return nothing
end
# euler_time_advance! advances the vector equation dfvec/dt = G[f]
# that includes the kinetic equation + any evolved moment equations
# using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
# with fvec_in an input and fvec_out the output
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments, z_SL, vpa_SL,
    z_advect, vpa_advect, z, vpa, t, t_input, z_spectral, vpa_spectral,
    composition, charge_exchange_frequency, advance, istage)
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa
    if advance.vpa_advection
        @views vpa_advection!(fvec_out.pdf, fvec_in, pdf.norm, fields, moments,
            vpa_SL, vpa_advect, vpa, z, use_semi_lagrange, dt, t,
            vpa_spectral, z_spectral, composition, charge_exchange_frequency, istage)
    end
    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to all species (charged and neutral)
    if advance.z_advection
        @views z_advection!(fvec_out.pdf, fvec_in, pdf.norm, moments, z_SL, z_advect, z, vpa,
            use_semi_lagrange, dt, t, z_spectral, composition.n_species, istage)
    end
    if advance.source_terms
        source_terms!(fvec_out.pdf, fvec_in, moments, z, vpa, dt, z_spectral,
                      composition, charge_exchange_frequency)
    end
    # account for charge exchange collisions between ions and neutrals
    if advance.cx_collisions
        charge_exchange_collisions!(fvec_out.pdf, fvec_in, moments, n_ion_species,
            composition.n_neutral_species, vpa, charge_exchange_frequency, z.n, dt)
    end
    if advance.continuity
        continuity_equation!(fvec_out.density, fvec_in, moments, z, vpa, dt, z_spectral)
    end
    if advance.force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        force_balance!(fvec_out.upar, fvec_in, fields, charge_exchange_frequency, z, vpa, dt, z_spectral, composition)
        # convert from the particle flux to the parallel flow
        @. fvec_out.upar /= fvec_out.density
    end
    if advance.energy
        energy_equation!(fvec_out.ppar, fvec_in, moments, charge_exchange_frequency, z, dt, z_spectral, composition)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments)
    # enforce boundary conditions in z and vpa on the distribution function
    # NB: probably need to do the same for the evolved moments
    enforce_boundary_conditions!(fvec_out.pdf, z.bc, vpa.bc, vpa.grid, z_advect, vpa_advect)
    return nothing
end
# update the vector containing the pdf and any evolved moments of the pdf
# for use in the Runge-Kutta time advance
function update_solution_vector!(evolved, moments, istage)
    evolved[istage+1].pdf .= evolved[istage].pdf
    evolved[istage+1].density .= evolved[istage].density
    evolved[istage+1].upar .= evolved[istage].upar
    evolved[istage+1].ppar .= evolved[istage].ppar
    return nothing
end
function update_pdf_unnorm!(pdf, moments)
    # if separately evolving the density via the continuity equation,
    # the evolved pdf has been normalised by the particle density
    # undo this normalisation to get the true particle distribution function
    if moments.evolve_ppar
        for ivpa ∈ 1:size(pdf.norm,2)
            @. pdf.unnorm[:,ivpa,:] = pdf.norm[:,ivpa,:]*moments.dens/moments.vth
        end
    elseif moments.evolve_density
        for ivpa ∈ 1:size(pdf.norm,2)
            @. pdf.unnorm[:,ivpa,:] = pdf.norm[:,ivpa,:]*moments.dens
        end
    else
        @. pdf.unnorm = pdf.norm
    end
end
