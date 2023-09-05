# The code in this file is adapted from the `cvode!()` function in Sundials.jl:
# https://github.com/SciML/Sundials.jl/blob/2f936e77bcbb6ea460f818864ca5afe953af65ff/src/simple.jl#L130
# Here we use an unpreconditioned, matrix-free solver (the default Newton for nonlinear
# solve, GMRES for linear solve). The example used a dense matrix solver, which is
# unsuutable for problames as large as the onse we are solving.

using ..communication

using Sundials

function cvode_solve!(f::Function,
    y0::Vector{Float64},
    t::AbstractVector,
    userdata::Any = nothing;
    integrator = :BDF,
    reltol::Float64 = 1e-3,
    abstol::Float64 = 1e-6,
    callback = (x, y, z) -> true)

    if integrator == :BDF
        mem_ptr = Sundials.CVodeCreate(Sundials.CV_BDF)
    elseif integrator == :Adams
        mem_ptr = Sundials.CVodeCreate(Sundials.CV_ADAMS)
    end

    (mem_ptr == C_NULL) && error("Failed to allocate CVODE solver object")
    mem = Sundials.Handle(mem_ptr)

    c = 1

    userfun = Sundials.UserFunctionAndData(f, userdata)
    y0nv = Sundials.NVector(y0)

    function getcfun(userfun::T) where {T}
        @cfunction(Sundials.cvodefun, Cint, (Sundials.realtype, Sundials.N_Vector,
                                             Sundials.N_Vector, Ref{T}))
    end
    flag = Sundials.@checkflag Sundials.CVodeInit(mem, getcfun(userfun), t[1], convert(Sundials.NVector, y0nv)) true

    flag = Sundials.@checkflag Sundials.CVodeSetUserData(mem, userfun) true

    # Settingns
    flag = Sundials.@checkflag Sundials.CVodeSStolerances(mem, reltol, abstol) true
    # Passing -1 disables the test for maximum number of steps. This is 'not recommended'
    # according to the CVODE manual.
    flag = Sundials.@checkflag Sundials.CVodeSetMaxNumSteps(mem, -1) true
    ## The following needs sundials-6.2, which is not supported yet by Sundials.jl (see
    ## https://github.com/SciML/Sundials.jl/pull/415)
    ###################################################################################
    ## Make CVODE increase the timestep less agressively
    ##flag = Sundials.@checkflag Sundials.CVodeSetEtaFixedStepBounds(mem, 0.0, 1.1) true # Default was (0,0, 1.5), which means minimum increase of step size is a factor 1.5. Decrease this to 1.1.
    ##flag = Sundials.@checkflag Sundials.CVodeSetEtaMax(mem, 1.2) true # Default was 10, which means increase in step size can be up to factor 10. Limit this to 1.2.
    #etastepbound_lower = 0.0
    #etastepbound_upper = 1.1
    #flag = Sundials.@checkflag ccall((:CVodeSetEtaFixedStepBounds,
    #                                  Sundials.libsundials_cvodes), Cint,
    #                                 (Sundials.CVODEMemPtr, Sundials.realtype,
    #                                  Sundials.realtype), mem, etastepbound_lower,
    #                                 etastepbound_upper)
    #etamax = 1.2
    #flag = Sundials.@checkflag ccall((:CVodeSetEtaMax, Sundials.libsundials_cvodes), Cint,
    #                                 (Sundials.CVODEMemPtr, Sundials.realtype), mem,
    #                                 etamax)

    # Set linear solver
    LS = Sundials.SUNLinSol_SPGMR(y0nv, Sundials.PREC_NONE, -1)
    flag = Sundials.@checkflag Sundials.CVodeSetLinearSolver(mem, LS, C_NULL) true

    ynv = Sundials.NVector(copy(y0))
    tout = [0.0]
    for k in 2:length(t)
        flag = Sundials.@checkflag Sundials.CVode(mem, t[k], ynv, tout, Sundials.CV_NORMAL) true
        if !callback(mem, t[k], ynv)
            break
        end
        c = c + 1
    end

    Sundials.SUNLinSolFree_SPGMR(LS)

    return c
end

function time_solve_with_cvode(mk_ddt_state...; ascii_io, io_moments, io_dfns,
                               reltol=1e-3, abstol=1e-6)
    if n_blocks[] != 1
        error("SUNDIALS.jl does not support MPI yet, so cannot use distributed memory.")
    end

    start_time = now()

    if block_rank[] == 0
        dfvec_dt, fvec, pdf, fields, moments, boundary_distributions, advect_objects, vz,
        vr, vzeta, vpa, vperp, gyrophase, z, r, t, t_input, spectral_objects, composition,
        collisions, geometry, scratch_dummy, manufactured_source_list,
        external_source_settings, num_diss_params, advance = mk_ddt_state

        # y0 holds the initial state in a single vector. Need to allocate, then fill it
        # with the initial state from the moment_kinetics variables.
        y0_length = get_cvode_state_size(fvec, moments, composition.n_neutral_species)
        y0 = Vector{Float64}(undef, y0_length)
        pack_cvode_data!(y0, fvec, moments, composition.n_neutral_species)

        # Get time points where we want output
        moments_output_inds = collect(1:t_input.nwrite_moments:t_input.nstep)
        dfns_output_inds = collect(1:t_input.nwrite_dfns:t_input.nstep)
        all_output_inds = sort(unique(vcat(moments_output_inds, dfns_output_inds)))
        moments_times = @. t + t_input.dt * moments_output_inds
        dfns_times = @. t + t_input.dt * dfns_output_inds
        all_time_points = @. t + t_input.dt * all_output_inds

        # simtime is the simulation time.
        # y is the current state, to calculate time derivatives from.
        # dydt is the vector to put output (i.e. the time derivatives) into.
        rhs_counter = 0
        function cvode_rhs_call!(simtime, y, dydt)
            rhs_counter += 1
            rhs_counter % 100 == 0 && println("$rhs_counter t=$simtime")
            unpack_cvode_data!(y, fvec, moments, composition.n_neutral_species)

            # Tell other processes to keep going.
            # Also synchronizes other processes so that they can use the unpacked data.
            finished = MPI.Bcast(0, 0, comm_block[])

            calculate_ddt!(mk_ddt_state...)

            # Synchronize other processes so we can pack data
            _block_synchronize()

            pack_cvode_data!(dydt, dfvec_dt, moments, composition.n_neutral_species)

            return Sundials.CV_SUCCESS
        end

        # p is something we don't need (maybe a pointer to the CVODE 'context'?)
        # y_nvector is the state vector, as an NVector
        iwrite_moments = 2
        iwrite_dfns = 2
        function cvode_output_callback(p, simtime, y_nvector)
            println("t=", simtime, ", ", rhs_counter, " rhs evaluations, ", Dates.format(now(), dateformat"H:MM:SS"))
            flush(stdout)
            rhs_counter = 0

            finish_now = false

            if isfile(t_input.stopfile)
                # Stop cleanly if a file called 'stop' was created
                println("Found 'stop' file $(t_input.stopfile), aborting run")
                finish_now = true
            end

            y = convert(Vector, y_nvector)
            unpack_cvode_data!(y, fvec, moments, composition.n_neutral_species)

            # Tell other processes to keep going.
            # Also synchronizes other processes so that they can use the unpacked data.
            finished = MPI.Bcast(0, 0, comm_block[])

            # Run calculate_ddt!() just to set the boundary conditions, etc. Slightly
            # wasteful, but easy to implement for now.
            calculate_ddt!(mk_ddt_state...)

            # Synchronize other processes so we can pack data
            _block_synchronize()

            # Copy data from fvec into pdf and moments structs for output
            pdf.charged.norm .= fvec.pdf
            moments.charged.dens .= fvec.density
            moments.charged.upar .= fvec.upar
            moments.charged.ppar .= fvec.ppar
            if composition.n_neutral_species > 0
                pdf.neutral.norm .= fvec.pdf_neutral
                moments.neutral.dens .= fvec.density_neutral
                moments.neutral.uz .= fvec.uz_neutral
                moments.neutral.pz .= fvec.pz_neutral
            end

            if any(isapprox.(simtime, all_time_points)) || finish_now
                time_for_run = to_minutes(now() - start_time)
            end

            if any(isapprox.(simtime, moments_times)) || finish_now
                finish_now = do_moments_output!(ascii_io, io_moments, pdf, nothing,
                                                simtime, t_input, vz, vr, vzeta, vpa,
                                                vperp, gyrophase, z, r, moments, fields,
                                                composition, iwrite_moments,
                                                iwrite_moments, time_for_run, finish_now)
                iwrite_moments += 1
            end
            if any(isapprox.(simtime, dfns_times)) || finish_now
                finish_now = do_dfns_output!(io_dfns, pdf, nothing, simtime, t_input, vz,
                                             vr, vzeta, vpa, vperp, gyrophase, z, r,
                                             moments, fields, composition, iwrite_dfns,
                                             iwrite_dfns, time_for_run, finish_now)
                iwrite_dfns += 1
            end

            return !finish_now
        end

        cvode_solve!(cvode_rhs_call!, y0, all_time_points; reltol=reltol, abstol=abstol,
                     callback=cvode_output_callback)

        # Tell other processes to stop
        finished = MPI.Bcast(1, 0, comm_block[])
    else
        while true
            # Check if run has finished
            finished = MPI.Bcast(0, 0, comm_block[])
            if finished != 0
                break
            end

            calculate_ddt!(mk_ddt_state...)

            # Synchronize so root process can pack data
            _block_synchronize()
        end
    end

    return nothing
end

function get_cvode_state_size(fvec, moments, n_neutral_species)
    y0_size = 0

    # Add ion pdf
    y0_size += length(fvec.pdf)

    if moments.evolve_density
        # Add ion density
        y0_size += length(fvec.density)
    end

    if moments.evolve_upar
        # Add ion parallel flow
        y0_size += length(fvec.upar)
    end

    if moments.evolve_ppar
        # Add ion parallel pressure
        y0_size += length(fvec.ppar)
    end

    if n_neutral_species > 0
        # Add neutral pdf
        y0_size += length(fvec.pdf_neutral)

        if moments.evolve_density
            # Add neutral density
            y0_size += length(fvec.density_neutral)
        end

        if moments.evolve_upar
            # Add neutral parallel flow
            y0_size += length(fvec.uz_neutral)
        end

        if moments.evolve_ppar
            # Add neutral parallel pressure
            y0_size += length(fvec.pz_neutral)
        end
    end

    return y0_size
end

function pack_cvode_data!(y, fvec, moments, n_neutral_species)
    start_ind = 1

    # Add ion pdf
    n = length(fvec.pdf)
    end_ind = start_ind + n - 1
    y[start_ind:end_ind] .= reshape(fvec.pdf, n)
    start_ind = end_ind + 1

    if moments.evolve_density
        # Add ion density
        n = length(fvec.density)
        end_ind = start_ind + n - 1
        y[start_ind:end_ind] .= reshape(fvec.density, n)
        start_ind = end_ind + 1
    end

    if moments.evolve_upar
        # Add ion parallel flow
        n = length(fvec.upar)
        end_ind = start_ind + n - 1
        y[start_ind:end_ind] .= reshape(fvec.upar, n)
        start_ind = end_ind + 1
    end

    if moments.evolve_ppar
        # Add ion parallel pressure
        n = length(fvec.ppar)
        end_ind = start_ind + n - 1
        y[start_ind:end_ind] .= reshape(fvec.ppar, n)
        start_ind = end_ind + 1
    end

    if n_neutral_species > 0
        # Add neutral pdf
        n = length(fvec.pdf_neutral)
        end_ind = start_ind + n - 1
        y[start_ind:end_ind] .= reshape(fvec.pdf_neutral, n)
        start_ind = end_ind + 1

        if moments.evolve_density
            # Add neutral density
            n = length(fvec.density_neutral)
            end_ind = start_ind + n - 1
            y[start_ind:end_ind] .= reshape(fvec.density_neutral, n)
            start_ind = end_ind + 1
        end

        if moments.evolve_upar
            # Add neutral parallel flow
            n = length(fvec.uz_neutral)
            end_ind = start_ind + n - 1
            y[start_ind:end_ind] .= reshape(fvec.uz_neutral, n)
            start_ind = end_ind + 1
        end

        if moments.evolve_ppar
            # Add neutral parallel pressure
            n = length(fvec.pz_neutral)
            end_ind = start_ind + n - 1
            y[start_ind:end_ind] .= reshape(fvec.pz_neutral, n)
            start_ind = end_ind + 1
        end
    end

    return nothing
end

function unpack_cvode_data!(y, fvec, moments, n_neutral_species)
    start_ind = 1

    # Add ion pdf
    n = length(fvec.pdf)
    end_ind = start_ind + n - 1
    reshape(fvec.pdf, n) .= y[start_ind:end_ind]
    start_ind = end_ind + 1

    if moments.evolve_density
        # Add ion density
        n = length(fvec.density)
        end_ind = start_ind + n - 1
        reshape(fvec.density, n) .= y[start_ind:end_ind]
        start_ind = end_ind + 1
    end

    if moments.evolve_upar
        # Add ion parallel flow
        n = length(fvec.upar)
        end_ind = start_ind + n - 1
        reshape(fvec.upar, n) .= y[start_ind:end_ind]
        start_ind = end_ind + 1
    end

    if moments.evolve_ppar
        # Add ion parallel pressure
        n = length(fvec.ppar)
        end_ind = start_ind + n - 1
        reshape(fvec.ppar, n) .= y[start_ind:end_ind]
        start_ind = end_ind + 1
    end

    if n_neutral_species > 0
        # Add neutral pdf
        n = length(fvec.pdf_neutral)
        end_ind = start_ind + n - 1
        reshape(fvec.pdf_neutral, n) .= y[start_ind:end_ind]
        start_ind = end_ind + 1

        if moments.evolve_density
            # Add neutral density
            n = length(fvec.density_neutral)
            end_ind = start_ind + n - 1
            reshape(fvec.density_neutral, n) .= y[start_ind:end_ind]
            start_ind = end_ind + 1
        end

        if moments.evolve_upar
            # Add neutral parallel flow
            n = length(fvec.uz_neutral)
            end_ind = start_ind + n - 1
            reshape(fvec.uz_neutral, n) .= y[start_ind:end_ind]
            start_ind = end_ind + 1
        end

        if moments.evolve_ppar
            # Add neutral parallel pressure
            n = length(fvec.pz_neutral)
            end_ind = start_ind + n - 1
            reshape(fvec.pz_neutral, n) .= y[start_ind:end_ind]
            start_ind = end_ind + 1
        end
    end

    return nothing
end
