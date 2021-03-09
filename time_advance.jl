module time_advance

export setup_time_advance!
export time_advance!

using type_definitions: mk_float
using array_allocation: allocate_float
using file_io: write_data_to_ascii, write_data_to_binary
using chebyshev: setup_chebyshev_pseudospectral
using chebyshev: chebyshev_derivative!
using velocity_moments: setup_moments, update_moments!, reset_moments_status!
using initial_conditions: enforce_z_boundary_condition!
using initial_conditions: enforce_vpa_boundary_condition!
using advection: setup_source, update_boundary_indices!
using z_advection: update_speed_z!, z_advection!
using vpa_advection: update_speed_vpa!, vpa_advection!
using charge_exchange: charge_exchange_collisions!
using em_fields: setup_em_fields, update_phi!
using semi_lagrange: setup_semi_lagrange

struct scratch_pdf{T}
    pdf::T
end
struct scratch_pdf_dens{T1,T2}
    pdf::T1
    density::T2
end
mutable struct advance_flags
    vpa_advection::Bool
    z_advection::Bool
    cx_collisions::Bool
end

# create arrays and do other work needed to setup
# the main time advance loop.
# this includes creating and populating structs
# for Chebyshev transforms, velocity space moments,
# EM fields, semi-Lagrange treatment, and source terms
function setup_time_advance!(ff, z, vpa, composition, drive_input, evolve_moments, t_input)
    # define some local variables for convenience/tidiness
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    if t_input.split_operators
        advance = advance_flags(false, false, false)
    else
        advance = advance_flags(true, true, true)
    end
    # create structure z_source whose members are the arrays needed to compute
    # the source(s) appearing in the split part of the GK equation dealing
    # with advection in z
    z_source = setup_source(z, vpa, n_species)
    # initialise the z advection speed
    for is ∈ 1:n_species
        update_speed_z!(view(z_source,:,is), vpa, z, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(view(z_source,:,is))
        # enforce prescribed boundary condition in z on the distribution function f
        @views enforce_z_boundary_condition!(ff[:,:,is], z.bc, vpa, z_source[:,is])
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
    # pass the distribution function ff (defined such that ∫dvpa ff = density)
    # and allocate/initialize the velocity space moments needed for advancing
    # the kinetic equation coupled to fluid equations
    # the resulting moments are returned in the structure "moments"
    moments = setup_moments(ff, vpa, z.n, evolve_moments)
    # redefine the distrubtion function ff = f(z,vpa)/n(z) if this option is chosen
    normalize_pdf!(ff, moments)
    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency)
    # initialize the electrostatic potential
    update_phi!(fields, moments, ff, vpa, z.n, composition, 0.0)
    # save the initial phi(z) for possible use later (e.g., if forcing phi)
    fields.phi0 .= fields.phi
    # create structure vpa_source whose members are the arrays needed to compute
    # the source(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_source = setup_source(vpa, z, n_ion_species)
    # initialise the vpa advection speed
    update_speed_vpa!(vpa_source, fields, moments, ff, vpa, z, composition, 0.0, z_spectral)
    for is ∈ 1:n_ion_species
        # initialise the upwind/downwind boundary indices in vpa
        update_boundary_indices!(view(vpa_source,:,is))
        # enforce prescribed boundary condition in vpa on the distribution function f
        @views enforce_vpa_boundary_condition!(ff[:,:,is], vpa.bc, vpa_source[:,is])
    end
    # create an array of structures containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n, vpa.n)
    vpa_SL = setup_semi_lagrange(vpa.n, z.n)
    # create an array of structs containing scratch arrays for the pdf and any moments
    # that are separately evolved via fluid equations
    scratch = setup_scratch_arrays(moments, z.n, vpa.n, n_species, t_input.n_rk_stages)
    return z_spectral, vpa_spectral, moments, fields, z_source, vpa_source, z_SL,
        vpa_SL, scratch, advance
end
# if evolving the density via continuity equation, redefine f → f/n
function normalize_pdf!(pdf, moments)
    nvpa = size(pdf,2)
    if moments.evolve_density
        for ivpa ∈ 1:nvpa
            @. pdf[:,ivpa,:] /= moments.dens
        end
    end
    return nothing
end
# create an array of structs containing scratch arrays for the pdf and any moments
# that are separately evolved via fluid equations
function setup_scratch_arrays(moments, nz, nvpa, nspec, n_rk_stages)
    # create n_rk_stages+1 pdf-sized scratch arrays
    pdf = allocate_float(nz, nvpa, nspec, n_rk_stages+1)
    if moments.evolve_density
        # create n_rk_stages+1 density-sized scratch arrays
        dens = allocate_float(nz, nvpa, nspec, n_rk_stages+1)
        # create n_rk_stages+1 structs, each of which will contain one pdf and one density array
        scratch = Vector{scratch_pdf_dens}(undef, n_rk_stages+1)
        # populate each of the structs
        # NB: all of the array members of the scratch struct will point to
        # the appropriate slice of the pdf and dens arrays created above
        for istage ∈ 1:n_rk_stages+1
            @views scratch[istage] = scratch_pdf_dens(pdf[:,:,:,istage], dens[:,:,istage])
        end
    else
        # create n_rk_stages+1 structs, each of which will contain one pdf array
        scratch = Vector{scratch_pdf}(undef, n_rk_stages+1)
        # populate each of the structs
        # NB: all of the array members of the scratch struct will point to
        # the appropriate slice of the pdf array created above
        for istage ∈ 1:n_rk_stages+1
            @views scratch[istage] = scratch_pdf(pdf[:,:,:,istage])
        end
    end
    return scratch
end
# solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
# define approximate characteristic velocity
# v₀(z)=vⁿ(z) and take time derivative along this characteristic
# df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
# for prudent choice of v₀, expect δv≪v so that explicit
# time integrator can be used without severe CFL condition
function time_advance!(ff, scratch, t, t_input, z, vpa, z_spectral, vpa_spectral,
    moments, fields, z_source, vpa_source, z_SL, vpa_SL, composition,
    charge_exchange_frequency, advance, io, cdf)
    # main time advance loop
    iwrite = 2
    for i ∈ 1:t_input.nstep
        if t_input.split_operators
            time_advance_split_operators!(ff, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, i)
        else
            time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
                z_SL, vpa_SL, composition, charge_exchange_frequency, advance, i)
        end
        # update the time
        t += t_input.dt
        # write data to file every nwrite time steps
        if mod(i,t_input.nwrite) == 0
            println("finished time step ", i)
            write_data_to_ascii(ff, moments, fields, z, vpa, t, composition.n_species, io)
            # write initial data to binary file (netcdf) -- after updating velocity-space moments
            update_moments!(moments, ff, vpa, z.n)
            write_data_to_binary(ff, moments, fields, t, composition.n_species, cdf, iwrite)
            iwrite += 1
        end
    end
    return nothing
end
function time_advance_split_operators!(ff, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
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
        time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        advance.cx_collisions = true
        time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.cx_collisions = false
    else
        # account for charge exchange collisions between ions and neutrals
        advance.cx_collisions = true
        time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.cx_collisions = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
        advance.vpa_advection = false
    end
    return nothing
end
function time_advance_no_splitting!(ff, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

    # define abbreviated variable for tidiness
    n_rk_stages = t_input.n_rk_stages

    if n_rk_stages == 4
        ssp_rk3_4stage!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
    elseif n_rk_stages == 3
        ssp_rk3!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
    elseif n_rk_stages == 2
        ssp_rk2!(ff, scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)
    else
        euler_time_advance!(scratch, scratch, ff, fields, moments, z_SL, vpa_SL,
            z_source, vpa_source, z, vpa, t,
            t_input, z_spectral, vpa_spectral, composition,
            charge_exchange_frequency, advance, 1)
    end
end
function ssp_rk3_4stage!(ff, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

    scratch[1].pdf .= ff
    if moments.evolve_density
        scratch[1].density .= moments.dens
    end

    istage = 1
    # do an Euler time advance, with scratch[2] containing the advanced quantities
    # and scratch[1] containing quantities at time level n
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)
    @. scratch[istage+1].pdf = 0.5*(ff + scratch[istage+1].pdf)
    if moments.evolve_density
        @. scratch[istage+1].density = 0.5*(moments.dens + scratch[istage+1].density)
    end

    istage = 2
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(2)} = f^{(1)} + Δt*G[f^{(1)}] = scratch[3].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)
    # redefinte scratch[3].pdf = g^{(2)} = 1/2 f^{(1)} + 1/2 f^{(2)}
    @. scratch[istage+1].pdf = 0.5*(scratch[istage].pdf + scratch[istage+1].pdf)
    if moments.evolve_density
        @. scratch[istage+1].density = 0.5*(scratch[istage].density + scratch[istage+1].density)
    end

    istage = 3
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(3)} = g^{(2)} + Δt*G[g^{(2)}] = scratch[4].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)
    # redefine scratch[4].pdf = g^{(3)} = 2/3 fⁿ + 1/6 g^{(2)} + 1/6 f^{(3)}
    @. scratch[istage+1].pdf = (4.0*ff + scratch[istage].pdf + scratch[istage+1].pdf)/6.0
    if moments.evolve_density
        @. scratch[istage+1].density = (4.0*moments.dens + scratch[istage].density
            + scratch[istage+1].density)/6.0
    end

    istage = 4
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(4)} = g^{(3)} + Δt*G[g^{(3)}] = scratch[5].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)

    # obtain f^{n+1} = 1/2 g^{(3)} + 1/2 f^{(4)}
    @. ff = 0.5*(scratch[istage].pdf + scratch[istage+1].pdf)
    if moments.evolve_density
        @. moments.dens = 0.5*(scratch[istage].density + scratch[istage+1].density)
    end
end
function ssp_rk3!(ff, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

    scratch[1].pdf .= ff
    if moments.evolve_density
        scratch[1].density .= moments.dens
    end

    istage = 1
    # do an Euler time advance, with scratch[2] containing the advanced quantities
    # and scratch[1] containing quantities at time level n
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)

    istage = 2
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(2)} = f^{(1)} + Δt*G[f^{(1)}] = scratch[3].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)
    # redefinte scratch[3].pdf = g^{(2)} = 3/4 fⁿ + 1/4 f^{(2)}
    @. scratch[istage+1].pdf = 0.75*ff + 0.25*scratch[istage+1].pdf
    if moments.evolve_density
        @. scratch[istage+1].density = 0.75*moments.dens + 0.25*scratch[istage+1].density
    end

    istage = 3
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(3)} = g^{(2)} + Δt*G[g^{(2)}] = scratch[4].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)

    # obtain f^{n+1} = 1/3 fⁿ + 2/3 f^{(3)}
    @. ff = (ff + 2.0*scratch[istage+1].pdf)/3.0
    if moments.evolve_density
        @. moments.dens = (moments.dens + 2.0*scratch[istage+1].density)/3.0
    end
end
function ssp_rk2!(ff, scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, advance, istep)

    scratch[1].pdf .= ff
    if moments.evolve_density
        scratch[1].density .= moments.dens
    end

    istage = 1
    # do an Euler time advance, with scratch[2] containing the advanced quantities
    # and scratch[1] containing quantities at time level n
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)

    istage = 2
    update_solution_vector!(scratch, moments, istage)
    # calculate f^{(2)} = f^{(1)} + Δt*G[f^{(1)}] = scratch[3].pdf
    @views euler_time_advance!(scratch[istage+1], scratch[istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, advance, istage)

    @. ff = 0.5*(ff + scratch[istage+1].pdf)
    if moments.evolve_density
        @. moments.dens = 0.5*(moments.dens + scratch[istage+1].density)
    end
end
# euler_time_advance! advances the vector equation dfvec/dt = G[f]
# that includes the kinetic equation + any evolved moment equations
# using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
# with fvec_in an input and fvec_out the output
function euler_time_advance!(fvec_out, fvec_in, ff, fields, moments, z_SL, vpa_SL,
    z_source, vpa_source, z, vpa, t, t_input, z_spectral, vpa_spectral,
    composition, charge_exchange_frequency, advance, istage)
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa
    if advance.vpa_advection
        @views vpa_advection!(fvec_out.pdf[:,:,1:n_ion_species],
            fvec_in.pdf[:,:,1:n_ion_species], ff[:,:,1:n_ion_species], fields,
            moments, vpa_SL, vpa_source, vpa, z, use_semi_lagrange, dt, t,
            vpa_spectral, z_spectral, composition, istage)
    end
    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to all species (charged and neutral)
    if advance.z_advection
        for is ∈ 1:composition.n_species
            @views z_advection!(fvec_out.pdf[:,:,is], fvec_in.pdf[:,:,is],
                ff[:,:,is], z_SL, z_source[:,is], z, vpa, use_semi_lagrange, dt, t,
                z_spectral, istage)
        end
    end
    if advance.cx_collisions && composition.n_neutral_species > 0
        # account for charge exchange collisions between ions and neutrals
        charge_exchange_collisions!(fvec_out.pdf, fvec_in.pdf, ff, moments, n_ion_species,
            composition.n_neutral_species, vpa, charge_exchange_frequency, z.n, dt)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments)
end
# update the vector containing the pdf and any evolved moments of the pdf
# for use in the Runge-Kutta time advance
function update_solution_vector!(evolved, moments, istage)
    evolved[istage+1].pdf .= evolved[istage].pdf
    if moments.evolve_density
        evolved[istage+1].density .= evolved[istage].density
    end
end

end
