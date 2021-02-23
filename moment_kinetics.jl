# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

using TimerOutputs

using file_io: setup_file_io, finish_file_io
using file_io: write_data_to_ascii, write_data_to_binary
using chebyshev: setup_chebyshev_pseudospectral
using chebyshev: chebyshev_derivative!
using coordinates: define_coordinate
using advection: setup_source, update_boundary_indices!
using semi_lagrange: setup_semi_lagrange
using vpa_advection: vpa_advection!, vpa_advection_single_stage!, update_speed_vpa!
using z_advection: z_advection!, z_advection_single_stage!, update_speed_z!
using velocity_moments: setup_moments, update_moments!, reset_moments_status!
using em_fields: setup_em_fields, update_phi!
using initial_conditions: init_f
using initial_conditions: enforce_z_boundary_condition!
using initial_conditions: enforce_vpa_boundary_condition!
using moment_kinetics_input: mk_input
using moment_kinetics_input: performance_test
using charge_exchange: charge_exchange_collisions!, charge_exchange_single_stage!
using time_advance: rk_update_f!

to1 = TimerOutput()
to2 = TimerOutput()

# main function that contains all of the content of the program
function moment_kinetics(to)
    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    run_name, output_dir, t_input, z_input, vpa_input, composition, species,
        charge_exchange_frequency, drive_input = mk_input()
    # initialize z grid and write grid point locations to file
    z = define_coordinate(z_input)
    # initialize vpa grid and write grid point locations to file
    vpa = define_coordinate(vpa_input)
    # initialize f(z)
    ff, ff_scratch = init_f(z, vpa, composition, species, t_input.n_rk_stages)
    # initialize time variable
    code_time = 0.
    # create arrays and do other work needed to setup
    # the main time advance loop
    z_spectral, vpa_spectral, moments, fields, z_source, vpa_source,
        z_SL, vpa_SL = setup_time_advance!(ff, z, vpa, composition, drive_input)
    # setup i/o
    io, cdf = setup_file_io(output_dir, run_name, z, vpa, composition)
    # write initial data to ascii files
    write_data_to_ascii(ff, moments, fields, z, vpa, code_time, composition.n_species, io)
    # write initial data to binary file (netcdf) -- after updating velocity-space moments
    update_moments!(moments, ff, vpa, z.n)
    write_data_to_binary(ff, moments, fields, code_time, composition.n_species, cdf, 1)
    # solve the 1+1D kinetic equation to advance f in time by nstep time steps
    if performance_test
        @timeit to "time_advance" time_advance!(ff, ff_scratch, code_time, t_input,
            z, vpa, z_spectral, vpa_spectral, moments, fields,
            z_source, vpa_source, z_SL, vpa_SL, composition, charge_exchange_frequency,
            io, cdf)
    else
        time_advance!(ff, ff_scratch, code_time, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields,
            z_source, vpa_source, z_SL, vpa_SL, composition, charge_exchange_frequency,
            io, cdf)
    end
    # finish i/o
    finish_file_io(io, cdf)
    return nothing
end
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

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    n_rk_stages = t_input.n_rk_stages
    use_semi_lagrange = t_input.use_semi_lagrange

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
        # vpa_advection_single_stage! advances the 1D advection equation in vpa
        # only charged species have a force accelerating them in vpa
        @views vpa_advection_single_stage!(ff_scratch[:,:,1:n_ion_species,istage:istage+1],
            ff[:,:,1:n_ion_species], fields,
            moments, vpa_SL, vpa_source, vpa, z, use_semi_lagrange, dt, t,
            vpa_spectral, z_spectral, composition, istage)
        # z_advection_single_stage! advances 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        for is ∈ 1:composition.n_species
            @views z_advection_single_stage!(ff_scratch[:,:,is,:], ff[:,:,is], z_SL,
                z_source[:,is], z, vpa, use_semi_lagrange, dt, t, z_spectral, istage)
        end
        if composition.n_neutral_species > 0
            # account for charge exchange collisions between ions and neutrals
            charge_exchange_single_stage!(ff_scratch, ff, moments, n_ion_species,
                composition.n_neutral_species, vpa, charge_exchange_frequency, z.n,
                dt, istage)
        end
        # reset "xx.updated" flags to false since ff has been updated
        # and the corresponding moments have not
        reset_moments_status!(moments)
    end
    for is ∈ 1:composition.n_species
		@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
    end
end
if performance_test
    @timeit to1 "first call to moment_kinetics" moment_kinetics(to1)
    show(to1)
    println()
    @timeit to2 "second call to moment_kinetics" moment_kinetics(to2)
    show(to2)
    println()
else
    moment_kinetics(to1)
end
