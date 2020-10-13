# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

#using TimerOutputs

#to = TimerOutput()

using array_allocation: allocate_float
using file_io: setup_file_io, finish_file_io
using file_io: write_f, write_moments
using chebyshev: setup_chebyshev_pseudospectral
using coordinates: define_coordinate, write_coordinate
using source_terms: setup_source
using source_terms: setup_advection_speed_z
using semi_lagrange: setup_semi_lagrange
using advection: advection_1d!
using velocity_moments: setup_moments

using moment_kinetics_input: run_name
using moment_kinetics_input: z_input, vpa_input
using moment_kinetics_input: zwidth, initialization_option, monomial_degree, vpawidth
using moment_kinetics_input: nstep, dt, nwrite, use_semi_lagrange
using moment_kinetics_input: check_input

function moment_kinetics()
    # check input options to catch errors
    check_input()
    # initialize z grid and write grid point locations to file
    z = define_coordinate(z_input)
    write_coordinate(z, run_name, "zgrid")
    # initialize vpa grid and write grid point locations to file
    vpa = define_coordinate(vpa_input)
    write_coordinate(vpa, run_name, "vpa")
    # initialize f(z)
    ff = init_f(z, vpa)
    # initialize time variable
    code_time = 0.
    # setup i/o
    io = setup_file_io(run_name)
    # write initial condition to file
    write_f(ff, z, vpa, code_time, io.ff)
    # solve the advection equation to advance u in time by nstep time steps
    time_advance!(ff, z, vpa, code_time, io)
    # finish i/o
    finish_file_io(io)
    return nothing
end
# creates ff and specifies its initial condition
function init_f(z, vpa)
    f = allocate_float(z.n, vpa.n, 3)
    @inbounds begin
        if initialization_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    f[i,j,:] .= (exp(-0.5*(z.grid[i]/zwidth)^2)
                     * exp(-0.5*(vpa.grid[j]/vpawidth)^2))
                end
            end
        elseif initialization_option == "monomial"
            # linear variation in z, with offset so that
            # function passes through zero at upwind boundary
            for i ∈ 1:z.n
                f[i,j,:] .= ((z.grid[i] + 0.5*z.L)^monomial_degree
                    .* (vpa.grid[j] + 0.5*vpa.L)^monomial_degree)
            end
        end
        if z.bc == "zero"
            # impose zero incoming BC
            f[1,:,:] .= 0
            #f[nz,:] .= 0
        elseif z.bc == "periodic"
            # impose periodicity
            f[1,:,:] .= f[z.n,:,:]
        end
        if vpa.bc == "zero"
            f[:,1,:] .= 0
        end
    end
    return f
end
# solve ∂f/∂t + v(z,t)⋅∂f/∂z = 0
# define approximate characteristic velocity
# v₀(z)=vⁿ(z) and take time derivative along this characteristic
# df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
# for prudent choice of v₀, expect δv≪v so that explicit
# time integrator can be used without severe CFL condition
function time_advance!(ff, z, vpa, t, io)
    # create arrays needed for explicit Chebyshev pseudospectral treatment
    # and create the plans for the forward and backward fast Chebyshev transforms
    if z.discretization == "chebyshev_pseudospectral"
        # will only chebyshev transform within a given element
        # so send in the correctly-sized array to setup the chebyshev plan
        z_chebyshev = setup_chebyshev_pseudospectral(z)
    end
    if vpa.discretization == "chebyshev_pseudospectral"
        vpa_chebyshev = setup_chebyshev_pseudospectral(vpa)
    end
    # pass a subarray of ff (its value at the previous time level)
    # and allocate/initialize the velocity space moments needed for advancing
    # the kinetic equation coupled to fluid equations
    # the resulting moments are returned in the structure "moments"
    moments = setup_moments(view(ff,:,:,1), vpa, z.n)
#=
    # create structures z_advection and vpa_advection that contain all of the
    # information necessary to update the advection speed in the z and vpa
    # directions
    z_advection_info = setup_advection_speed_z(vpa.grid)
    if z.discretization == "chebyshev_pseudospectral"
        vpa_advection_info = setup_advection_speed_vpa(ff, z_chebyshev.f)
    else
        vpa_advection_info = setup_advection_speed_vpa(ff)
    end
=#
    # create structure z_source whose members are the arrays needed to compute
    # the source(s) appearing in the split part of the GK equation dealing
    # with advection in z
    #z_source = setup_source(z, z_advection_info)
    #vpa_source = setup_source(vpa, vpa_advection_info)
    z_source = setup_source(z)
    vpa_source = setup_source(vpa)
    # create a structure containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n)
    vpa_SL = setup_semi_lagrange(vpa.n)
    # main time advance loop
    for i ∈ 1:nstep
        # advection_1d! advances the operator-split 1D advection equation
        if z.discretization == "chebyshev_pseudospectral"
            for ivpa ∈ 1:vpa.n
                advection_1d!(view(ff,:,ivpa,:), z_SL, z_source, z, use_semi_lagrange, dt, z_chebyshev)
            end
        elseif z.discretization == "finite_difference"
            for ivpa ∈ 1:vpa.n
                advection_1d!(view(ff,:,ivpa,:), z_SL, z_source, z, use_semi_lagrange, dt)
            end
        end
        if vpa.discretization == "chebyshev_pseudospectral"
            for iz ∈ 1:z.n
                advection_1d!(view(ff,iz,:,:), vpa_SL, vpa_source, vpa, use_semi_lagrange, dt, vpa_chebyshev)
            end
        elseif vpa.discretization == "finite_difference"
            for iz ∈ 1:z.n
                advection_1d!(view(ff,iz,:,:), vpa_SL, vpa_source, vpa, use_semi_lagrange, dt)
            end
        end
        # update the time
        t += dt
        # write ff to file every nwrite time steps
        if mod(i,nwrite) == 0
            write_f(ff, z, vpa, t, io.ff)
            write_moments(moments, z, t, io.moments)
        end
    end
    return nothing
end

moment_kinetics()

#show(to)
println()
