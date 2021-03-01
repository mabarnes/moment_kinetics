module time_advance

export rk_update_f!
export setup_time_advance!

using file_io: write_data_to_ascii, write_data_to_binary
using chebyshev: setup_chebyshev_pseudospectral
using chebyshev: chebyshev_derivative!
using velocity_moments: setup_moments, update_moments!, reset_moments_status!
using initial_conditions: enforce_z_boundary_condition!
using initial_conditions: enforce_vpa_boundary_condition!
using advection: setup_source, update_boundary_indices!
using z_advection: update_speed_z!, z_advection!, z_advection_single_stage!
using vpa_advection: update_speed_vpa!, vpa_advection!, vpa_advection_single_stage!
using charge_exchange: charge_exchange_collisions!, charge_exchange_single_stage!
using em_fields: setup_em_fields, update_phi!
using semi_lagrange: setup_semi_lagrange

# create arrays and do other work needed to setup
# the main time advance loop.
# this includes creating and populating structs
# for Chebyshev transforms, velocity space moments,
# EM fields, semi-Lagrange treatment, and source terms
function setup_time_advance!(ff, z, vpa, composition, drive_input)
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
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
    # pass ff and allocate/initialize the velocity space moments needed for advancing
    # the kinetic equation coupled to fluid equations
    # the resulting moments are returned in the structure "moments"
    moments = setup_moments(ff, vpa, z.n)
    # pass a subarray of ff (its value at the previous time level)
    # and create the "fields" structure that contains arrays
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
    return z_spectral, vpa_spectral, moments, fields, z_source, vpa_source, z_SL, vpa_SL
end
# solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
# define approximate characteristic velocity
# v₀(z)=vⁿ(z) and take time derivative along this characteristic
# df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
# for prudent choice of v₀, expect δv≪v so that explicit
# time integrator can be used without severe CFL condition
function time_advance!(ff, ff_scratch, t, t_input, z, vpa, z_spectral, vpa_spectral,
    moments, fields, z_source, vpa_source, z_SL, vpa_SL, composition,
    charge_exchange_frequency, io, cdf)
    # main time advance loop
    iwrite = 2
    for i ∈ 1:t_input.nstep
        if t_input.split_operators
            time_advance_split_operators!(ff, ff_scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
                z_SL, vpa_SL, composition, charge_exchange_frequency, i)
        else
            time_advance_no_splitting!(ff, ff_scratch, t, t_input, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
                z_SL, vpa_SL, composition, charge_exchange_frequency, i)
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
function time_advance_split_operators!(ff, ff_scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, istep)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    dt = t_input.dt
    n_rk_stages = t_input.n_rk_stages
    use_semi_lagrange = t_input.use_semi_lagrange
    # to ensure 2nd order accuracy in time for operator-split advance,
    # have to reverse order of operations every other time step
    flipflop = (mod(istep,2)==0)
    #NB: following line only for testing
    #flipflop = false
    if flipflop
        # vpa_advection! advances the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        @views vpa_advection!(ff[:,:,1:n_ion_species], ff_scratch[:,:,1:n_ion_species,:],
            fields, moments, vpa_SL, vpa_source, vpa, z, n_rk_stages,
            use_semi_lagrange, dt, t, vpa_spectral, z_spectral, composition)
        for is ∈ 1:composition.n_ion_species
        	@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        for is ∈ 1:composition.n_species
            @views z_advection!(ff[:,:,is], ff_scratch[:,:,is,:], z_SL, z_source[:,is],
                z, vpa, n_rk_stages, use_semi_lagrange, dt, t, z_spectral)
            @views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
        end
        # reset "xx.updated" flags to false since ff has been updated
        # and the corresponding moments have not
        reset_moments_status!(moments)
        if composition.n_neutral_species > 0
            # account for charge exchange collisions between ions and neutrals
            @views charge_exchange_collisions!(ff, ff_scratch, moments, composition,
                vpa, charge_exchange_frequency, z.n, dt, n_rk_stages)
            for is ∈ 1:n_ion_species+n_neutral_species
            	@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
            end
        end
    else
        if composition.n_neutral_species > 0
            # account for charge exchange collisions between ions and neutrals
            @views charge_exchange_collisions!(ff, ff_scratch, moments, composition,
                vpa, charge_exchange_frequency, z.n, dt, n_rk_stages)
            for is ∈ 1:n_ion_species+n_neutral_species
            	@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
            end
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        for is ∈ 1:composition.n_species
            @views z_advection!(ff[:,:,is], ff_scratch[:,:,is,:], z_SL, z_source[:,is],
                z, vpa, n_rk_stages, use_semi_lagrange, dt, t, z_spectral)
            @views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
        end
        # reset "moments.xx_updated" flags to false since ff has been updated
        # and the corresponding moments have not
        reset_moments_status!(moments)
        # vpa_advection! advances the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        @views vpa_advection!(ff[:,:,1:n_ion_species], ff_scratch[:,:,1:n_ion_species,:],
            fields, moments, vpa_SL, vpa_source, vpa, z, n_rk_stages,
            use_semi_lagrange, dt, t, vpa_spectral, z_spectral, composition)
        for is ∈ 1:composition.n_ion_species
        	@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
        end
    end
    return nothing
end
function time_advance_no_splitting!(ff, ff_scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, istep)

    # define abbreviated variable for tidiness
    n_rk_stages = t_input.n_rk_stages

    if n_rk_stages == 3
        ssp_rk3!(ff, ff_scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, istep)
    elseif n_rk_stages == 2
        ssp_rk2!(ff, ff_scratch, t, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
            z_SL, vpa_SL, composition, charge_exchange_frequency, istep)
    else
        euler_time_advance!(ff, ff, ff, fields, moments, z_SL, vpa_SL,
            z_source, vpa_source, z, vpa, t,
            t_input, z_spectral, vpa_spectral, composition,
            charge_exchange_frequency, 1)
    end
#=
    # initialize ff_scratch to the distribution function value at the current time level
    for istage ∈ 1:n_rk_stages + 1
        ff_scratch[:,:,:,istage] .= ff
    end
    for istage ∈ 1:n_rk_stages
        # for SSP RK3, need to redefine ff_scratch[3]
        if istage == 3
            @. ff_scratch[:,:,:,istage] = 0.25*(ff_scratch[:,:,:,istage] +
                ff_scratch[:,:,:,istage-1] + 2.0*ff)
        end
        # do an Euler time advance, with ff_scratch[istage+1] the advanced function
        # and ff_scratch[istage] the function at level n
        @views euler_time_advance!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
            ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
            t_input, z_spectral, vpa_spectral, composition,
            charge_exchange_frequency, istage)
    end
    for is ∈ 1:composition.n_species
		@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
    end
=#
end
function ssp_rk3!(ff, ff_scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, istep)

    ff_scratch[:,:,:,1] .= ff

    istage = 1
    # do an Euler time advance, with ff_scratch[2] the advanced function
    # and ff_scratch[1] the function at time level n
    ff_scratch[:,:,:,istage+1] .= ff_scratch[:,:,:,istage]
    # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = ff_scratch[2]
    @views euler_time_advance!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, istage)

    istage = 2
    ff_scratch[:,:,:,istage+1] .= ff_scratch[:,:,:,istage]
    # calculate f^{(2)} = f^{(1)} + Δt*G[f^{(1)}] = ff_scratch[3]
    @views euler_time_advance!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, istage)
    # redefinte ff_scratch[3] = g^{(2)} = 3/4 fⁿ + 1/4 f^{(2)}
    @. ff_scratch[:,:,:,istage+1] = 0.75*ff + 0.25*ff_scratch[:,:,:,istage+1]

    istage = 3
    ff_scratch[:,:,:,istage+1] .= ff_scratch[:,:,:,istage]
    # calculate f^{(3)} = g^{(2)} + Δt*G[g^{(2)}] = ff_scratch[4]
    @views euler_time_advance!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, istage)

    # obtain f^{n+1} = 1/3 fⁿ + 2/3 f^{(3)}
    @. ff = (ff + 2.0*ff_scratch[:,:,:,istage+1])/3.0
end
function ssp_rk2!(ff, ff_scratch, t, t_input, z, vpa,
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
    z_SL, vpa_SL, composition, charge_exchange_frequency, istep)

    # define abbreviated variable for tidiness
    n_rk_stages = t_input.n_rk_stages

    ff_scratch[:,:,:,1] .= ff

    istage = 1
    # do an Euler time advance, with ff_scratch[2] the advanced function
    # and ff_scratch[1] the function at time level n
    ff_scratch[:,:,:,istage+1] .= ff_scratch[:,:,:,istage]
    @views euler_time_advance!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, istage)

    istage = 2
    ff_scratch[:,:,:,istage+1] .= ff_scratch[:,:,:,istage]
    @views euler_time_advance!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
        ff, fields, moments, z_SL, vpa_SL, z_source, vpa_source, z, vpa, t,
        t_input, z_spectral, vpa_spectral, composition,
        charge_exchange_frequency, istage)

    @. ff = 0.5*(ff + ff_scratch[:,:,:,istage+1])
end
# euler_time_advance! advances the equation df/dt = G[f]
# using the forward Euler method: f_out = f_in + dt*f_in,
# with f_in an input and f_out the output
function euler_time_advance!(f_out, f_in, ff, fields, moments, z_SL, vpa_SL,
    z_source, vpa_source, z, vpa, t, t_input, z_spectral, vpa_spectral,
    composition, charge_exchange_frequency, istage)
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection_single_stage! advances the 1D advection equation in vpa
    # only charged species have a force accelerating them in vpa
    @views vpa_advection_single_stage!(f_out[:,:,1:n_ion_species],
        f_in[:,:,1:n_ion_species], ff[:,:,1:n_ion_species], fields,
        moments, vpa_SL, vpa_source, vpa, z, use_semi_lagrange, dt, t,
        vpa_spectral, z_spectral, composition, istage)
    # z_advection_single_stage! advances 1D advection equation in z
    # apply z-advection operation to all species (charged and neutral)
    for is ∈ 1:composition.n_species
        @views z_advection_single_stage!(f_out[:,:,is], f_in[:,:,is], ff[:,:,is], z_SL,
            z_source[:,is], z, vpa, use_semi_lagrange, dt, t, z_spectral, istage)
    end
    if composition.n_neutral_species > 0
        # account for charge exchange collisions between ions and neutrals
        charge_exchange_single_stage!(f_out, f_in, ff, moments, n_ion_species,
            composition.n_neutral_species, vpa, charge_exchange_frequency, z.n, dt)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments)
end
# rk_update_f! combines the results of the various Runge Kutta stages
# to obtain the updated distribution function
function rk_update_f!(ff, ff_rk, nz, nvpa, n_rk_stages)
    @boundscheck nz == size(ff_rk,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff_rk,2) || throw(BoundsError(ff_rk))
    @boundscheck n_rk_stages+1 == size(ff_rk,3) || throw(BoundsError(ff_rk))
    @boundscheck nz == size(ff,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff,2) || throw(BoundsError(ff_rk))
    if n_rk_stages == 1
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = ff_rk[iz,ivpa,2]
                end
            end
        end
    elseif n_rk_stages == 2
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = 0.5*(ff_rk[iz,ivpa,2] + ff_rk[iz,ivpa,3])
                end
            end
        end
    elseif n_rk_stages == 3
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = (2.0*(ff_rk[iz,ivpa,3] + ff_rk[iz,ivpa,4])-ff_rk[iz,ivpa,1])/3.0
                end
            end
        end
    end
end

end
