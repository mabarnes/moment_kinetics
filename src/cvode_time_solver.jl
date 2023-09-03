# The code in this file is adapted from the `cvode!()` function in Sundials.jl:
# https://github.com/SciML/Sundials.jl/blob/2f936e77bcbb6ea460f818864ca5afe953af65ff/src/simple.jl#L130

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
        mem_ptr = CVodeCreate(CV_BDF)
    elseif integrator == :Adams
        mem_ptr = CVodeCreate(CV_ADAMS)
    end

    (mem_ptr == C_NULL) && error("Failed to allocate CVODE solver object")
    mem = Handle(mem_ptr)

    c = 1

    userfun = UserFunctionAndData(f, userdata)
    y0nv = NVector(y0)

    function getcfun(userfun::T) where {T}
        @cfunction(cvodefun, Cint, (realtype, N_Vector, N_Vector, Ref{T}))
    end
    flag = @checkflag CVodeInit(mem, getcfun(userfun), t[1], convert(NVector, y0nv)) true

    flag = @checkflag CVodeSetUserData(mem, userfun) true
    flag = @checkflag CVodeSStolerances(mem, reltol, abstol) true
    A = Sundials.SUNDenseMatrix(length(y0), length(y0))
    LS = Sundials.SUNLinSol_Dense(y0nv, A)
    flag = Sundials.@checkflag Sundials.CVDlsSetLinearSolver(mem, LS, A) true

    ynv = NVector(copy(y0))
    tout = [0.0]
    for k in 2:length(t)
        flag = @checkflag CVode(mem, t[k], ynv, tout, CV_NORMAL) true
        if !callback(mem, t[k], ynv)
            break
        end
        c = c + 1
    end

    Sundials.SUNLinSolFree_Dense(LS)
    Sundials.SUNMatDestroy_Dense(A)

    return c
end

function time_solve_with_cvode(mk_ddt_state...)
    if n_blocks[] != 1
        error("SUNDIALS.jl does not support MPI yet, so cannot use distributed memory.")
    end

    if block_rank[] == 0
        dfvec_dt, fvec, pdf, fields, moments, advect_objects, vz, vr, vzeta, vpa,
        vperp, gyrophase, z, r, t, t_input, spectral_objects, composition, collisions,
        geometry, scratch_dummy, manufactured_source_list, external_source_settings,
        num_diss_params, advance = mk_ddt_state

        # y0 holds the initial state in a single vector. Need to allocate, then fill it
        # with the initial state from the moment_kinetics variables.
        y0_length = get_cvode_state_size(fvec, moments, composition.n_neutral_species)
        y0 = Vector{Float64}(undef, y0_length)
        pack_cvode_data!(y0, fvec, moments, composition.n_neutral_species)

        # Get time points where we want output
        moments_output_inds = collect(1:t_input.nwrite_moments:t_input.nstep)
        dfns_output_inds = collect(1:t_input.nwrite_dfns:t_input.nstep)
        all_output_inds = sort(unique(vcat(moments_output_inds, dfns_output_inds)))
        moments_times = @. t + dt * moments_output_inds
        dfns_times = @. t + dt * dfns_output_inds
        all_time_points = @. t + dt * all_output_inds

        # dydt is the vector to put output (i.e. the time derivatives) into.
        # y is the current state, to calculate time derivatives from.
        # p is something we don't need (maybe a pointer to the CVODE 'context'?)
        # t is (probably) the simulation time.
        function cvode_rhs_call!(dydt, y, p, simtime)
            unpack_cvode_data!(y, fvec, moments, composition.n_neutral_species)

            # Tell other processes to keep going.
            # Also synchronizes other processes so that they can use the unpacked data.
            finished = MPI.Broadcast(0, 0, comm_block[])

            calculate_ddt!(mk_ddt_state...)

            _block_synchronize()

            pack_cvode_data!(dydt, dfvec_dt, moments, composition.n_neutral_species)

            return 0
        end

        # p is something we don't need (maybe a pointer to the CVODE 'context'?)
        # y_nvector is the state vector, as an NVector
        function cvode_output_callback(p, simtime, y_nvector)
            y = convert(Vector, y_nvector)
            unpack_cvode_data!(y, fvec, moments, composition.n_neutral_species)

            # Run calculate_ddt!() just to set the boundary conditions, etc. Slightly
            # wasteful, but easy to implement for now.
            calculate_ddt!(mk_ddt_state...)

            finish_now = false
            if any(isapprox.(simtime, moments_times)) || finish_now
                finish_now = do_moments_output!(pdf, nothing, t, t_input, vz, vr, vzeta, vpa,
                                                vperp, gyrophase, z, r, moments, fields,
                                                composition, i, finish_now)
            end
            if any(isapprox.(simtime, dfns_times)) || finish_now
                finish_now = do_dfns_output!(pdf, nothing, t, t_input, vz, vr, vzeta, vpa,
                                             vperp, gyrophase, z, r, moments, fields,
                                             composition, i, finish_now)
            end

            return Int64(finish_now)
        end

        cvode_solve!(cvode_rhs_call!, y0, all_time_points; reltol=1.e-6, abstol=1.e-12,
                     callback=cvode_output_callback)

        # Tell other processes to stop
        finished = MPI.Broadcast(1, 0, comm_block[])
    else
        while true
            # Check if run has finished
            finished = MPI.Broadcast(0, 0, comm_block[])
            if finished != 0
                break
            end

            calculate_ddt!(mk_ddt_state...)
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
        n = length(fvec.pdf_neutral)
        end_ind = start_ind + n - 1
        y0[start_ind:end_ind] .= reshape(fvec.pdf_neutral, n)
        start_ind = end_ind + 1

        if moments.evolve_density
            # Add neutral density
            n = length(fvec.density_neutral)
            end_ind = start_ind + n - 1
            y0[start_ind:end_ind] .= reshape(fvec.density_neutral, n)
            start_ind = end_ind + 1
        end

        if moments.evolve_upar
            # Add neutral parallel flow
            n = length(fvec.uz_neutral)
            end_ind = start_ind + n - 1
            y0[start_ind:end_ind] .= reshape(fvec.uz_neutral, n)
            start_ind = end_ind + 1
        end

        if moments.evolve_ppar
            # Add neutral parallel pressure
            n = length(fvec.pz_neutral)
            end_ind = start_ind + n - 1
            y0[start_ind:end_ind] .= reshape(fvec.pz_neutral, n)
            start_ind = end_ind + 1
        end
    end

    return nothing
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
