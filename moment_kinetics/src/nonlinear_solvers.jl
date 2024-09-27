"""
Nonlinear solvers, using Jacobian-free Newton-Krylov methods.

These solvers use an outer Newton iteration. Each step of the Newton iteration requires a
linear solve of the Jacobian. An 'inexact Jacobian' method is used, and the GMRES method
(GMRES is a type of Krylov solver) is used to (approximately) solve the (approximate)
linear system.

!!! warning "parallelisation"
    This module uses shared- and distributed-memory parallelism, so the functions in it
    should not be called inside any kind of parallelised loop. This restriction should be
    lifted somehow in future...

`parallel_map()` is used to apply elementwise functions to arbitrary numbers of arguments
using shared-memory parallelism. We do this rather than writing the loops out explicitly
so that `newton_solve!()` and `linear_solve!()` can work for arrays with any combination
of dimensions.

Useful references:
[1] V.A. Mousseau and D.A. Knoll, "Fully Implicit Kinetic Solution of Collisional Plasmas", Journal of Computational Physics 136, 308–323 (1997), https://doi.org/10.1006/jcph.1997.5736.
[2] V.A. Mousseau, "Fully Implicit Kinetic Modelling of Collisional Plasmas", PhD thesis, Idaho National Engineering Laboratory (1996), https://inis.iaea.org/collection/NCLCollectionStore/_Public/27/067/27067141.pdf.
[3] https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
[4] https://www.rikvoorhaar.com/blog/gmres
[5] E. Carson , J. Liesen, Z. Strakoš, "Towards understanding CG and GMRES through examples", Linear Algebra and its Applications 692, 241–291 (2024), https://doi.org/10.1016/j.laa.2024.04.003. 
[6] Q. Zou, "GMRES algorithms over 35 years", Applied Mathematics and Computation 445, 127869 (2023), https://doi.org/10.1016/j.amc.2023.127869
"""
module nonlinear_solvers

export setup_nonlinear_solve, gather_nonlinear_solver_counters!,
       reset_nonlinear_per_stage_counters!, newton_solve!

using ..array_allocation: allocate_float, allocate_shared_float
using ..communication
using ..coordinates: coordinate
using ..input_structs
using ..looping
using ..type_definitions: mk_float, mk_int

using LinearAlgebra
using MPI
using SparseArrays
using StatsBase: mean

struct nl_solver_info{TH,TV,Tcsg,Tlig,Tprecon}
    rtol::mk_float
    atol::mk_float
    nonlinear_max_iterations::mk_int
    linear_rtol::mk_float
    linear_atol::mk_float
    linear_restart::mk_int
    linear_max_restarts::mk_int
    H::TH
    c::Tcsg
    s::Tcsg
    g::Tcsg
    V::TV
    linear_initial_guess::Tlig
    n_solves::Ref{mk_int}
    nonlinear_iterations::Ref{mk_int}
    linear_iterations::Ref{mk_int}
    global_n_solves::Ref{mk_int}
    global_nonlinear_iterations::Ref{mk_int}
    global_linear_iterations::Ref{mk_int}
    solves_since_precon_update::Ref{mk_int}
    precon_dt::Ref{mk_float}
    serial_solve::Bool
    max_nonlinear_iterations_this_step::Ref{mk_int}
    max_linear_iterations_this_step::Ref{mk_int}
    preconditioner_type::String
    preconditioner_update_interval::mk_int
    preconditioners::Tprecon
end

"""

`coords` is a NamedTuple of coordinates corresponding to the dimensions of the variable
that will be solved. The entries in `coords` should be ordered the same as the memory
layout of the variable to be solved (i.e. fastest-varying first).

The nonlinear solver will be called inside a loop over `outer_coords`, so we might need
for example a preconditioner object for each point in that outer loop.
"""
function setup_nonlinear_solve(active, input_dict, coords, outer_coords=(); default_rtol=1.0e-5,
                               default_atol=1.0e-12, serial_solve=false,
                               electron_ppar_pdf_solve=false, preconditioner_type="none")
    nl_solver_section = set_defaults_and_check_section!(
        input_dict, "nonlinear_solver";
        rtol=default_rtol,
        atol=default_atol,
        nonlinear_max_iterations=20,
        linear_rtol=1.0e-3,
        linear_atol=1.0,
        linear_restart=10,
        linear_max_restarts=0,
        preconditioner_update_interval=300,
       )

    if !active
        # This solver will not be used. Return here, after reading the options, so that we
        # can always check that input file sections are supposed to exist.
        return nothing
    end

    nl_solver_input = Dict_to_NamedTuple(nl_solver_section)

    coord_sizes = Tuple(isa(c, coordinate) ? c.n : c for c ∈ coords)
    total_size_coords = prod(coord_sizes)
    outer_coord_sizes = Tuple(isa(c, coordinate) ? c.n : c for c ∈ outer_coords)

    linear_restart = nl_solver_input.linear_restart

    if serial_solve
        H = allocate_float(linear_restart + 1, linear_restart)
        c = allocate_float(linear_restart + 1)
        s = allocate_float(linear_restart + 1)
        g = allocate_float(linear_restart + 1)
        V = allocate_float(reverse(coord_sizes)..., linear_restart+1)
        H .= 0.0
        c .= 0.0
        s .= 0.0
        g .= 0.0
        V .= 0.0
    elseif electron_ppar_pdf_solve
        H = allocate_shared_float(linear_restart + 1, linear_restart)
        c = allocate_shared_float(linear_restart + 1)
        s = allocate_shared_float(linear_restart + 1)
        g = allocate_shared_float(linear_restart + 1)
        V_ppar = allocate_shared_float(coords.z.n, linear_restart+1)
        V_pdf = allocate_shared_float(reverse(coord_sizes)..., linear_restart+1)

        begin_serial_region()
        @serial_region begin
            H .= 0.0
            c .= 0.0
            s .= 0.0
            g .= 0.0
            V_ppar .= 0.0
            V_pdf .= 0.0
        end

        V = (V_ppar, V_pdf)
    else
        H = allocate_shared_float(linear_restart + 1, linear_restart)
        c = allocate_shared_float(linear_restart + 1)
        s = allocate_shared_float(linear_restart + 1)
        g = allocate_shared_float(linear_restart + 1)
        V = allocate_shared_float(reverse(coord_sizes)..., linear_restart+1)

        begin_serial_region()
        @serial_region begin
            H .= 0.0
            c .= 0.0
            s .= 0.0
            g .= 0.0
            V .= 0.0
        end
    end

    if preconditioner_type == "lu"
        # Create dummy LU solver objects so we can create an array for preconditioners.
        # These will be calculated properly within the time loop.
        preconditioners = fill(lu(sparse(1.0*I, total_size_coords, total_size_coords)),
                               reverse(outer_coord_sizes))
    elseif preconditioner_type == "electron_split_lu"
        preconditioners = (z=fill(lu(sparse(1.0*I, coords.z.n, coords.z.n)),
                                  tuple(coords.vpa.n, reverse(outer_coord_sizes)...)),
                           vpa=fill(lu(sparse(1.0*I, coords.vpa.n, coords.vpa.n)),
                                    tuple(coords.z.n, reverse(outer_coord_sizes)...)),
                           ppar=fill(lu(sparse(1.0*I, coords.z.n, coords.z.n)),
                                     reverse(outer_coord_sizes)),
                          )
    elseif preconditioner_type == "electron_lu"
        pdf_plus_ppar_size = total_size_coords + coords.z.n
        preconditioners = fill((lu(sparse(1.0*I, 1, 1)),
                                allocate_shared_float(pdf_plus_ppar_size, pdf_plus_ppar_size),
                                allocate_shared_float(pdf_plus_ppar_size),
                                allocate_shared_float(pdf_plus_ppar_size),
                               ),
                               reverse(outer_coord_sizes))
    elseif preconditioner_type == "none"
        preconditioners = nothing
    else
        error("Unrecognised preconditioner_type=$preconditioner_type")
    end

    linear_initial_guess = zeros(linear_restart)

    return nl_solver_info(nl_solver_input.rtol, nl_solver_input.atol,
                          nl_solver_input.nonlinear_max_iterations,
                          nl_solver_input.linear_rtol, nl_solver_input.linear_atol,
                          linear_restart, nl_solver_input.linear_max_restarts, H, c, s, g,
                          V, linear_initial_guess, Ref(0), Ref(0), Ref(0), Ref(0), Ref(0),
                          Ref(0), Ref(nl_solver_input.preconditioner_update_interval),
                          Ref(0.0), serial_solve, Ref(0), Ref(0), preconditioner_type,
                          nl_solver_input.preconditioner_update_interval, preconditioners)
end

"""
    reset_nonlinear_per_stage_counters!(nl_solver_params::Union{nl_solver_info,Nothing})

Reset the counters that hold per-step totals or maximums in `nl_solver_params`.

Also increment `nl_solver_params.stage_counter[]`.
"""
function reset_nonlinear_per_stage_counters!(nl_solver_params::Union{nl_solver_info,Nothing})
    if nl_solver_params === nothing
        return nothing
    end

    nl_solver_params.max_nonlinear_iterations_this_step[] = 0
    nl_solver_params.max_linear_iterations_this_step[] = 0

    # Also increment the stage counter
    nl_solver_params.solves_since_precon_update[] += 1

    return nothing
end

"""
    gather_nonlinear_solver_counters!(nl_solver_params)

Where necessary, gather the iteration counters for the nonlinear solvers.

Where each solve runs in parallel using all processes, this is unnecessary as the count on
each process already represents the global count. Where each solve uses only a subset of
processes, the counters from different solves need to be added together to get the global
total.
"""
function gather_nonlinear_solver_counters!(nl_solver_params)
    if nl_solver_params.ion_advance !== nothing
        # Solve runs in parallel on all processes, so no need to collect here
        nl_solver_params.ion_advance.global_n_solves[] = nl_solver_params.ion_advance.n_solves[]
        nl_solver_params.ion_advance.global_nonlinear_iterations[] = nl_solver_params.ion_advance.nonlinear_iterations[]
        nl_solver_params.ion_advance.global_linear_iterations[] = nl_solver_params.ion_advance.linear_iterations[]
    end
    if nl_solver_params.vpa_advection !== nothing
        # Solves are run in serial on separate processes, so need a global Allreduce
        nl_solver_params.vpa_advection.global_n_solves[] = MPI.Allreduce(nl_solver_params.vpa_advection.n_solves[], +, comm_world)
        nl_solver_params.vpa_advection.global_nonlinear_iterations[] = MPI.Allreduce(nl_solver_params.vpa_advection.nonlinear_iterations[], +, comm_world)
        nl_solver_params.vpa_advection.global_linear_iterations[] = MPI.Allreduce(nl_solver_params.vpa_advection.linear_iterations[], +, comm_world)
    end
end

"""
    newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, w, nl_solver_params;
                  left_preconditioner=nothing, right_preconditioner=nothing, coords)

`x` is the initial guess at the solution, and is overwritten by the result of the Newton
solve.

`rhs_func!(residual, x)` is the function we are trying to find a solution of. It calculates
```math
\\mathtt{residual} = F(\\mathtt{x})
```
where we are trying to solve \$F(x)=0\$.

`residual`, `delta_x`, `rhs_delta` and `w` are buffer arrays, with the same size as `x`,
used internally.

`left_preconditioner` or `right_preconditioner` apply preconditioning. They should be
passed a function that solves \$P.x = b\$ where \$P\$ is the preconditioner matrix, \$b\$
is given by the values passed to the function as the argument, and the result \$x\$ is
returned by overwriting the argument.

`coords` is a NamedTuple containing the `coordinate` structs corresponding to each
dimension in `x`.


Tolerances
----------

Note that the meaning of the relative tolerance `rtol` and absolute tolerance `atol` is
very different for the outer Newton iteration and the inner GMRES iteration.

For the outer Newton iteration the residual \$R(x^n)\$ measures the departure of the
system from the solution (at each grid point). Its size can be compared to the size of the
solution `x`, so it makes sense to define an `error norm' for \$R(x^n)\$ as
```math
E(x^n) = \\left\\lVert \\frac{R(x^n)}{\\mathtt{rtol} x^n \\mathtt{atol}} \\right\\rVert_2
```
where \$\\left\\lVert \\cdot \\right\\rVert\$ is the 'L2 norm' (square-root of sum of
squares). We can further try to define a grid-size independent error norm by dividing out
the number of grid points to get a root-mean-square (RMS) error rather than an L2 norm.
```math
E_{\\mathrm{RMS}}(x^n) = \\sqrt{ \\frac{1}{N} \\sum_i \\frac{R(x^n)_i}{\\mathtt{rtol} x^n_i \\mathtt{atol}} }
```
where \$N\$ is the total number of grid points.

In contrast, GMRES is constructed to minimise the L2 norm of \$r_k = b - A\\cdot x_k\$
where GMRES is solving the linear system \$A\\cdot x = b\$, \$x_k\$ is the approximation
to the solution \$x\$ at the \$k\$'th iteration and \$r_k\$ is the residual at the
\$k\$'th iteration. There is no flexibility to measure error relative to \$x\$ in any
sense. For GMRES, a `relative tolerance' is relative to the residual of the
right-hand-side \$b\$, which is the first iterate \$x_0\$ (when no initial guess is
given). [Where a non-zero initial guess is given it might be better to use a different
stopping criterion, see Carson et al. section 3.8.]. The stopping criterion for the GMRES
iteration is therefore
```
\\left\\lVert r_k \\right\\rVert < \\max(\\mathtt{linear\\_rtol} \\left\\lVert r_0 \\right\\rVert, \\mathtt{linear\\_atol}) = \\max(\\mathtt{linear\\_rtol} \\left\\lVert b \\right\\rVert, \\mathtt{linear\\_atol})
```
As the GMRES solve is only used to get the right `direction' for the next Newton step, it
is not necessary to have a very tight `linear_rtol` for the GMRES solve.
"""
function newton_solve!(x, residual_func!, residual, delta_x, rhs_delta, v, w,
                       nl_solver_params; left_preconditioner=nothing,
                       right_preconditioner=nothing, coords)

    rtol = nl_solver_params.rtol
    atol = nl_solver_params.atol

    distributed_norm = get_distributed_norm(coords, rtol, atol, x)
    distributed_dot = get_distributed_dot(coords, rtol, atol, x)
    parallel_map = get_parallel_map(coords)
    parallel_delta_x_calc = get_parallel_delta_x_calc(coords)

    if left_preconditioner === nothing
        left_preconditioner = identity
    end
    if right_preconditioner === nothing
        right_preconditioner = identity
    end

    residual_func!(residual, x)
    residual_norm = distributed_norm(residual)
    counter = 0
    linear_counter = 0

    parallel_map(()->0.0, delta_x)

    close_counter = -1
    close_linear_counter = -1
    success = true
    previous_residual_norm = residual_norm
    while (counter < 1 && residual_norm > 1.0e-8) || residual_norm > 1.0
        counter += 1
        #println("\nNewton ", counter)

        # Solve (approximately?):
        #   J δx = -RHS(x)
        parallel_map(()->0.0, delta_x)
        linear_its = linear_solve!(x, residual_func!, residual, delta_x, v, w;
                                   coords=coords, rtol=nl_solver_params.linear_rtol,
                                   atol=nl_solver_params.linear_atol,
                                   restart=nl_solver_params.linear_restart,
                                   max_restarts=nl_solver_params.linear_max_restarts,
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   H=nl_solver_params.H, c=nl_solver_params.c,
                                   s=nl_solver_params.s, g=nl_solver_params.g,
                                   V=nl_solver_params.V, rhs_delta=rhs_delta,
                                   initial_guess=nl_solver_params.linear_initial_guess,
                                   distributed_norm=distributed_norm,
                                   distributed_dot=distributed_dot,
                                   parallel_map=parallel_map,
                                   parallel_delta_x_calc=parallel_delta_x_calc,
                                   serial_solve=nl_solver_params.serial_solve)
        linear_counter += linear_its

        # If the residual does not decrease, we will do a line search to find an update
        # that does decrease the residual. The value of `x` is used to define the
        # normalisation value with rtol that is used to calculate the residual, so do not
        # want to update it until the line search is completed (otherwise the norm changes
        # during the line search, which might make it fail to converge). So calculate the
        # updated value in the buffer `w` until the line search is completed, and only
        # then copy it into `x`.
        parallel_map((x) -> x, w, x)
        parallel_map((x,delta_x) -> x + delta_x, w, x, delta_x)
        residual_func!(residual, w)

        # For the Newton iteration, we want the norm divided by the (sqrt of the) number
        # of grid points, so we can use a tolerance that is independent of the size of the
        # grid. This is unlike the norms needed in `linear_solve!()`.
        residual_norm = distributed_norm(residual)
        if isnan(residual_norm)
            error("NaN in Newton iteration at iteration $counter")
        end
        if residual_norm > previous_residual_norm
            # Do a line search between x and x+delta_x to try to find an update that does
            # decrease residual_norm
            s = 0.5
            while s > 1.0e-2
                parallel_map((x,delta_x) -> x + s * delta_x, w, x, delta_x)
                residual_func!(residual, x)
                residual_norm = distributed_norm(residual)
                if residual_norm ≤ previous_residual_norm
                    break
                end
                s *= 0.5
            end

            #if residual_norm > previous_residual_norm
            #    # Failed to find a point that decreases the residual, so try a negative
            #    # step
            #    s = -1.0e-5
            #    parallel_map((x,delta_x) -> x + s * delta_x, w, x, delta_x)
            #    residual_func!(residual, x)
            #    residual_norm = distributed_norm(residual)
            #    if residual_norm > previous_residual_norm
            #        # That didn't work either, so just take the full step and hope for
            #        # convergence later
            #        parallel_map((x,delta_x) -> x + s * delta_x, w, x, delta_x)
            #        residual_func!(residual, x)
            #        residual_norm = distributed_norm(residual)
            #    end
            #end
            if residual_norm > previous_residual_norm
                # Line search didn't work, so just take the full step and hope for
                # convergence later
                parallel_map((x,delta_x) -> x + s * delta_x, w, x, delta_x)
                residual_func!(residual, x)
                residual_norm = distributed_norm(residual)
            end
        end
        parallel_map((w) -> w, x, w)
        previous_residual_norm = residual_norm

        #println("Newton residual ", residual_norm, " ", linear_its, " $rtol $atol")

        if residual_norm < 0.1/rtol && close_counter < 0 && close_linear_counter < 0
            close_counter = counter
            close_linear_counter = linear_counter
        end

        if counter > nl_solver_params.nonlinear_max_iterations
            println("maximum iteration limit reached")
            success = false
            break
        end
    end
    nl_solver_params.n_solves[] += 1
    nl_solver_params.nonlinear_iterations[] += counter
    nl_solver_params.linear_iterations[] += linear_counter
    nl_solver_params.max_nonlinear_iterations_this_step[] =
        max(counter, nl_solver_params.max_nonlinear_iterations_this_step[])
    nl_solver_params.max_linear_iterations_this_step[] =
        max(linear_counter, nl_solver_params.max_linear_iterations_this_step[])
#    println("Newton iterations: ", counter)
#    println("Final residual: ", residual_norm)
#    println("Total linear iterations: ", linear_counter)
#    println("Linear iterations per Newton: ", linear_counter / counter)
#
#    println("Newton iterations after close: ", counter - close_counter)
#    println("Total linear iterations after close: ", linear_counter - close_linear_counter)
#    println("Linear iterations per Newton after close: ", (linear_counter - close_linear_counter) / (counter - close_counter))
#    println()

    return success
end

"""
    get_distributed_norm(coords, rtol, atol, x)

Get a 'distributed_norm' function that acts on arrays with dimensions given by the
entries in `coords`.
"""
function get_distributed_norm(coords, rtol, atol, x)
    dims = keys(coords)
    if dims == (:z,)
        this_norm = distributed_norm_z
    elseif dims == (:vpa,)
        this_norm = distributed_norm_vpa
    elseif dims == (:z, :vperp, :vpa)
        # Intended for implicit solve combining electron_ppar and pdf_electron, so will
        # not work for a single variable.
        this_norm = distributed_norm_z_vperp_vpa
    elseif dims == (:s, :r, :z, :vperp, :vpa)
        this_norm = distributed_norm_s_r_z_vperp_vpa
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`distributed_norm_*()` function in nonlinear_solvers.jl")
    end

    wrapped_norm = (args...; kwargs...) -> this_norm(args...; rtol=rtol, atol=atol, x=x,
                                                     coords=coords, kwargs...)

    return wrapped_norm
end

function distributed_norm_z(residual::AbstractArray{mk_float, 1}; coords, rtol, atol, x)
    z = coords.z

    begin_z_region()

    local_norm = 0.0
    if z.irank < z.nrank - 1
        zend = z.n
        @loop_z iz begin
            if iz == zend
                continue
            end
            local_norm += (residual[iz] / (rtol * abs(x[iz]) + atol))^2
        end
    else
        @loop_z iz begin
            local_norm += (residual[iz] / (rtol * abs(x[iz]) + atol))^2
        end
    end

    _block_synchronize()
    block_norm = MPI.Reduce(local_norm, +, comm_block[])

    if block_rank[] == 0
        global_norm = MPI.Allreduce(block_norm, +, comm_inter_block[])
        global_norm = sqrt(global_norm / z.n_global)
    else
        global_norm = nothing
    end
    global_norm = MPI.bcast(global_norm, comm_block[]; root=0)

    return global_norm
end

function distributed_norm_vpa(residual::AbstractArray{mk_float, 1}; coords, rtol, atol, x)
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    residual_norm = 0.0
    for i ∈ eachindex(residual, x)
        residual_norm += (residual[i] / (rtol * abs(x[i]) + atol))^2
    end

    residual_norm = sqrt(residual_norm / length(residual))

    return residual_norm
end

function distributed_norm_z_vperp_vpa(residual::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}};
                                      coords, rtol, atol, x)
    ppar_residual, pdf_residual = residual
    x_ppar, x_pdf = x
    z = coords.z
    vperp = coords.vperp
    vpa = coords.vpa

    if z.irank < z.nrank - 1
        zend = z.n
    else
        zend = z.n + 1
    end

    begin_z_region()

    ppar_local_norm_square = 0.0
    @loop_z iz begin
        if iz == zend
            continue
        end
        ppar_local_norm_square += (ppar_residual[iz] / (rtol * abs(x_ppar[iz]) + atol))^2
    end

    _block_synchronize()
    ppar_block_norm_square = MPI.Reduce(ppar_local_norm_square, +, comm_block[])

    if block_rank[] == 0
        ppar_global_norm_square = MPI.Allreduce(ppar_block_norm_square, +, comm_inter_block[])
        ppar_global_norm_square = ppar_global_norm_square / z.n_global
    else
        ppar_global_norm_square = nothing
    end

    begin_z_vperp_vpa_region()

    pdf_local_norm_square = 0.0
    @loop_z iz begin
        if iz == zend
            continue
        end
        @loop_vperp_vpa ivperp ivpa begin
            pdf_local_norm_square += (pdf_residual[ivpa,ivperp,iz] / (rtol * abs(x_pdf[ivpa,ivperp,iz]) + atol))^2
        end
    end

    _block_synchronize()
    pdf_block_norm_square = MPI.Reduce(pdf_local_norm_square, +, comm_block[])

    if block_rank[] == 0
        pdf_global_norm_square = MPI.Allreduce(pdf_block_norm_square, +, comm_inter_block[])
        pdf_global_norm_square = pdf_global_norm_square / (z.n_global * vperp.n_global * vpa.n_global)

        global_norm = sqrt(mean((ppar_global_norm_square, pdf_global_norm_square)))
    else
        global_norm = nothing
    end

    global_norm = MPI.bcast(global_norm, comm_block[]; root=0)

    return global_norm
end

function distributed_norm_s_r_z_vperp_vpa(residual::AbstractArray{mk_float, 5};
                                          coords, rtol, atol, x)
    n_ion_species = coords.s
    r = coords.r
    z = coords.z
    vperp = coords.vperp
    vpa = coords.vpa

    begin_s_r_z_vperp_vpa_region()

    local_norm = 0.0
    if r.irank < r.nrank - 1
        rend = r.n
    else
        rend = r.n + 1
    end
    if z.irank < z.nrank - 1
        zend = z.n
    else
        zend = z.n + 1
    end
    @loop_s_r_z is ir iz begin
        if ir == rend || iz == zend
            continue
        end
        @loop_vperp_vpa ivperp ivpa begin
            local_norm += (residual[ivpa,ivperp,iz,ir,is] / (rtol * abs(x[ivpa,ivperp,iz,ir,is]) + atol))^2
        end
    end

    _block_synchronize()
    block_norm = MPI.Reduce(local_norm, +, comm_block[])

    if block_rank[] == 0
        global_norm = MPI.Allreduce(block_norm, +, comm_inter_block[])
        global_norm = sqrt(global_norm / (n_ion_species * r.n_global * z.n_global * vperp.n_global * vpa.n_global))
    else
        global_norm = nothing
    end
    global_norm = MPI.bcast(global_norm, comm_block[]; root=0)

    return global_norm
end

"""
    get_distributed_dot(coords, rtol, atol, x)

Get a 'distributed_dot' function that acts on arrays with dimensions given by the entries
in `coords`.
"""
function get_distributed_dot(coords, rtol, atol, x)
    dims = keys(coords)
    if dims == (:z,)
        this_dot = distributed_dot_z
    elseif dims == (:vpa,)
        this_dot = distributed_dot_vpa
    elseif dims == (:z, :vperp, :vpa)
        # Intended for implicit solve combining electron_ppar and pdf_electron, so will
        # not work for a single variable.
        this_dot = distributed_dot_z_vperp_vpa
    elseif dims == (:s, :r, :z, :vperp, :vpa)
        this_dot = distributed_dot_s_r_z_vperp_vpa
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`distributed_dot_*()` function in nonlinear_solvers.jl")
    end

    wrapped_dot = (args...; kwargs...) -> this_dot(args...; rtol=rtol, atol=atol, x=x,
                                                   coords=coords, kwargs...)

end

function distributed_dot_z(v::AbstractArray{mk_float, 1}, w::AbstractArray{mk_float, 1};
                           coords, atol, rtol, x)

    z = coords.z

    begin_z_region()

    z = coords.z

    local_dot = 0.0
    if z.irank < z.nrank - 1
        zend = z.n
        @loop_z iz begin
            if iz == zend
                continue
            end
            local_dot += v[iz] * w[iz] / (rtol * abs(x[iz]) + atol)^2
        end
    else
        @loop_z iz begin
            local_dot += v[iz] * w[iz] / (rtol * abs(x[iz]) + atol)^2
        end
    end

    _block_synchronize()
    block_dot = MPI.Reduce(local_dot, +, comm_block[])

    if block_rank[] == 0
        global_dot = MPI.Allreduce(block_dot, +, comm_inter_block[])
        global_dot = global_dot / z.n_global
    else
        global_dot = nothing
    end

    return global_dot
end

function distributed_dot_vpa(v::AbstractArray{mk_float, 1}, w::AbstractArray{mk_float, 1};
                             coords, atol, rtol, x)
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    local_dot = 0.0
    for i ∈ eachindex(v,w)
        local_dot += v[i] * w[i] / (rtol * abs(x[i]) + atol)^2
    end
    local_dot = local_dot / length(v)
    return local_dot
end

function distributed_dot_z_vperp_vpa(v::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}},
                                     w::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}};
                                     coords, atol, rtol, x)
    v_ppar, v_pdf = v
    w_ppar, w_pdf = w
    x_ppar, x_pdf = x

    z = coords.z
    vperp = coords.vperp
    vpa = coords.vpa

    if z.irank < z.nrank - 1
        zend = z.n
    else
        zend = z.n + 1
    end

    begin_z_region()

    ppar_local_dot = 0.0
    @loop_z iz begin
        if iz == zend
            continue
        end
        ppar_local_dot += v_ppar[iz] * w_ppar[iz] / (rtol * abs(x_ppar[iz]) + atol)^2
    end

    _block_synchronize()
    ppar_block_dot = MPI.Reduce(ppar_local_dot, +, comm_block[])

    if block_rank[] == 0
        ppar_global_dot = MPI.Allreduce(ppar_block_dot, +, comm_inter_block[])
        ppar_global_dot = ppar_global_dot / z.n_global
    else
        ppar_global_dot = nothing
    end

    begin_z_vperp_vpa_region()

    pdf_local_dot = 0.0
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if iz == zend
            continue
        end
        pdf_local_dot += v_pdf[ivpa,ivperp,iz] * w_pdf[ivpa,ivperp,iz] / (rtol * abs(x_pdf[ivpa,ivperp,iz]) + atol)^2
    end

    _block_synchronize()
    pdf_block_dot = MPI.Reduce(pdf_local_dot, +, comm_block[])

    if block_rank[] == 0
        pdf_global_dot = MPI.Allreduce(pdf_block_dot, +, comm_inter_block[])
        pdf_global_dot = pdf_global_dot / (z.n_global * vperp.n_global * vpa.n_global)

        global_dot = mean((ppar_global_dot, pdf_global_dot))
    else
        global_dot = nothing
    end

    return global_dot
end

function distributed_dot_s_r_z_vperp_vpa(v::AbstractArray{mk_float, 5},
                                         w::AbstractArray{mk_float, 5};
                                         coords, atol, rtol, x)
    n_ion_species = coords.s
    r = coords.r
    z = coords.z
    vperp = coords.vperp
    vpa = coords.vpa

    begin_s_r_z_vperp_vpa_region()

    local_dot = 0.0
    if r.irank < r.nrank - 1
        rend = r.n
    else
        rend = r.n + 1
    end
    if z.irank < z.nrank - 1
        zend = z.n
    else
        zend = z.n + 1
    end

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        if ir == rend || iz == zend
            continue
        end
        local_dot += v[ivpa,ivperp,iz,ir,is] * w[ivpa,ivperp,iz,ir,is] / (rtol * abs(x[ivpa,ivperp,iz,ir,is]) + atol)^2
    end

    _block_synchronize()
    block_dot = MPI.Reduce(local_dot, +, comm_block[])

    if block_rank[] == 0
        global_dot = MPI.Allreduce(block_dot, +, comm_inter_block[])
        global_dot = global_dot / (n_ion_species * r.n_global * z.n_global * vperp.n_global * vpa.n_global)
    else
        global_dot = nothing
    end

    return global_dot
end

"""
    get_parallel_map(coords)

Get a 'parallel_map' function that acts on arrays with dimensions given by the entries in
`coords`.
"""
function get_parallel_map(coords)
    dims = keys(coords)
    if dims == (:z,)
        return parallel_map_z
    elseif dims == (:vpa,)
        return parallel_map_vpa
    elseif dims == (:z, :vperp, :vpa)
        # Intended for implicit solve combining electron_ppar and pdf_electron, so will
        # not work for a single variable.
        return parallel_map_z_vperp_vpa
    elseif dims == (:s, :r, :z, :vperp, :vpa)
        return parallel_map_s_r_z_vperp_vpa
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`parallel_map_*()` function in nonlinear_solvers.jl")
    end
end

# Separate versions for different numbers of arguments as generator expressions result in
# slow code

function parallel_map_z(func, result::AbstractArray{mk_float, 1})

    begin_z_region()

    @loop_z iz begin
        result[iz] = func()
    end

    return nothing
end
function parallel_map_z(func, result::AbstractArray{mk_float, 1}, x1)

    begin_z_region()

    @loop_z iz begin
        result[iz] = func(x1[iz])
    end

    return nothing
end
function parallel_map_z(func, result::AbstractArray{mk_float, 1}, x1, x2)

    begin_z_region()

    @loop_z iz begin
        result[iz] = func(x1[iz], x2[iz])
    end

    return nothing
end

function parallel_map_vpa(func, result::AbstractArray{mk_float, 1})
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    for i ∈ eachindex(result)
        result[i] = func()
    end
    return nothing
end
function parallel_map_vpa(func, result::AbstractArray{mk_float, 1}, x1)
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    for i ∈ eachindex(result)
        result[i] = func(x1[i])
    end
    return nothing
end
function parallel_map_vpa(func, result::AbstractArray{mk_float, 1}, x1, x2)
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    for i ∈ eachindex(result)
        result[i] = func(x1[i], x2[i])
    end
    return nothing
end

function parallel_map_z_vperp_vpa(func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}})

    result_ppar, result_pdf = result

    begin_z_region()

    @loop_z iz begin
        result_ppar[iz] = func()
    end

    begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        result_pdf[ivpa,ivperp,iz] = func()
    end

    return nothing
end
function parallel_map_z_vperp_vpa(func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}}, x1)

    result_ppar, result_pdf = result
    x1_ppar, x1_pdf = x1

    begin_z_region()

    @loop_z iz begin
        result_ppar[iz] = func(x1_ppar[iz])
    end

    begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz])
    end

    return nothing
end
function parallel_map_z_vperp_vpa(func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}}, x1, x2)

    result_ppar, result_pdf = result
    x1_ppar, x1_pdf = x1
    x2_ppar, x2_pdf = x2

    begin_z_region()

    @loop_z iz begin
        result_ppar[iz] = func(x1_ppar[iz], x2_ppar[iz])
    end

    begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz], x2_pdf[ivpa,ivperp,iz])
    end

    return nothing
end

function parallel_map_s_r_z_vperp_vpa(func, result::AbstractArray{mk_float, 5})

    begin_s_r_z_vperp_vpa_region()

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        result[ivpa,ivperp,iz,ir,is] = func()
    end

    return nothing
end
function parallel_map_s_r_z_vperp_vpa(func, result::AbstractArray{mk_float, 5}, x1)

    begin_s_r_z_vperp_vpa_region()

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is])
    end

    return nothing
end
function parallel_map_s_r_z_vperp_vpa(func, result::AbstractArray{mk_float, 5}, x1, x2)

    begin_s_r_z_vperp_vpa_region()

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is], x2[ivpa,ivperp,iz,ir,is])
    end

    return nothing
end

"""
    get_parallel_delta_x_calc(coords)

Get a parallelised function that calculates the update `delta_x` from the `V` matrix and
the minimum residual coefficients `y`.
"""
function get_parallel_delta_x_calc(coords)
    dims = keys(coords)
    if dims == (:z,)
        return parallel_delta_x_calc_z
    elseif dims == (:vpa,)
        return parallel_delta_x_calc_vpa
    elseif dims == (:z, :vperp, :vpa)
        # Intended for implicit solve combining electron_ppar and pdf_electron, so will
        # not work for a single variable.
        return parallel_delta_x_calc_z_vperp_vpa
    elseif dims == (:s, :r, :z, :vperp, :vpa)
        return parallel_delta_x_calc_s_r_z_vperp_vpa
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`parallel_delta_x_calc_*()` function in nonlinear_solvers.jl")
    end
end

function parallel_delta_x_calc_z(delta_x::AbstractArray{mk_float, 1}, V, y)

    begin_z_region()

    ny = length(y)
    @loop_z iz begin
        for iy ∈ 1:ny
            delta_x[iz] += y[iy] * V[iz,iy]
        end
    end

    return nothing
end

function parallel_delta_x_calc_vpa(delta_x::AbstractArray{mk_float, 1}, V, y)
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    ny = length(y)
    for ivpa ∈ eachindex(delta_x)
        for iy ∈ 1:ny
            delta_x[ivpa] += y[iy] * V[ivpa,iy]
        end
    end
    return nothing
end

function parallel_delta_x_calc_z_vperp_vpa(delta_x::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}}, V, y)

    delta_x_ppar, delta_x_pdf = delta_x
    V_ppar, V_pdf = V

    ny = length(y)

    begin_z_region()

    @loop_z iz begin
        for iy ∈ 1:ny
            delta_x_ppar[iz] += y[iy] * V_ppar[iz,iy]
        end
    end

    begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        for iy ∈ 1:ny
            delta_x_pdf[ivpa,ivperp,iz] += y[iy] * V_pdf[ivpa,ivperp,iz,iy]
        end
    end

    return nothing
end

function parallel_delta_x_calc_s_r_z_vperp_vpa(delta_x::AbstractArray{mk_float, 5}, V, y)

    begin_s_r_z_vperp_vpa_region()

    ny = length(y)
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        for iy ∈ 1:ny
            delta_x[ivpa,ivperp,iz,ir,is] += y[iy] * V[ivpa,ivperp,iz,ir,is,iy]
        end
    end

    return nothing
end

# Utility function for neatness handling that V may be an array or a Tuple of arrays
function select_from_V(V::Tuple, i)
    return Tuple(selectdim(Vpart,ndims(Vpart),i) for Vpart ∈ V)
end
function select_from_V(V, i)
    return selectdim(V,ndims(V),i)
end

"""
Apply the GMRES algorithm to solve the 'linear problem' J.δx^n = R(x^n), which is needed
at each step of the outer Newton iteration (in `newton_solve!()`).

Uses Givens rotations to reduce the upper Hessenberg matrix to an upper triangular form,
which allows conveniently finding the residual at each step, and computing the final
solution, without calculating a least-squares minimisation at each step. See 'algorithm 2
MGS-GMRES' in Zou (2023) [https://doi.org/10.1016/j.amc.2023.127869].
"""
function linear_solve!(x, residual_func!, residual0, delta_x, v, w; coords, rtol, atol,
                       restart, max_restarts, left_preconditioner, right_preconditioner,
                       H, c, s, g, V, rhs_delta, initial_guess, distributed_norm,
                       distributed_dot, parallel_map, parallel_delta_x_calc, serial_solve)
    # Solve (approximately?):
    #   J δx = residual0

    Jv_scale_factor = 1.0e3
    inv_Jv_scale_factor = 1.0 / Jv_scale_factor

    # The vectors `v` that are passed to this function will be normalised so that
    # `distributed_norm(v) == 1.0`. `distributed_norm()` is defined - including the
    # relative and absolute tolerances from the Newton iteration - so that a vector with a
    # norm of 1.0 is 'small' in the sense that a vector with a norm of 1.0 is small enough
    # relative to `x` to consider the iteration converged. This means that `x+v` would be
    # very close to `x`, so R(x+v)-R(x) would be likely to be badly affected by rounding
    # errors, because `v` is so small, relative to `x`. We actually want to multiply `v`
    # by a large number `Jv_scale_factor` (in constrast to the small `epsilon` in the
    # 'usual' case where the norm does not include either reative or absolute tolerance)
    # to ensure that we get a reasonable estimate of J.v.
    function approximate_Jacobian_vector_product!(v)
        right_preconditioner(v)

        parallel_map((x,v) -> x + Jv_scale_factor * v, v, x, v)
        residual_func!(rhs_delta, v)
        parallel_map((rhs_delta, residual0) -> (rhs_delta - residual0) * inv_Jv_scale_factor,
                     v, rhs_delta, residual0)
        left_preconditioner(v)
        return v
    end

    # To start with we use 'w' as a buffer to make a copy of residual0 to which we can apply
    # the left-preconditioner.
    parallel_map((delta_x) -> delta_x, v, delta_x)
    left_preconditioner(residual0)
    # This function transforms the data stored in 'v' from δx to ≈J.δx
    approximate_Jacobian_vector_product!(v)
    # Now we actually set 'w' as the first Krylov vector, and normalise it.
    parallel_map((residual0, v) -> -residual0 - v, w, residual0, v)
    beta = distributed_norm(w)
    parallel_map((w) -> w/beta, select_from_V(V, 1), w)
    if serial_solve
        g[1] = beta
    else
        begin_serial_region()
        @serial_region begin
            g[1] = beta
        end
    end

    # Set tolerance for GMRES iteration to rtol times the initial residual, unless this is
    # so small that it is smaller than atol, in which case use atol instead.
    tol = max(rtol * beta, atol)

    lsq_result = nothing
    residual = Inf
    counter = 0
    restart_counter = 1
    while true
        i = 0
        while i < restart
            i += 1
            counter += 1
            #println("Linear ", counter)

            # Compute next Krylov vector
            parallel_map((V) -> V, w, select_from_V(V, i))
            approximate_Jacobian_vector_product!(w)

            # Gram-Schmidt orthogonalization
            for j ∈ 1:i
                parallel_map((V) -> V, v, select_from_V(V, j))
                w_dot_Vj = distributed_dot(w, v)
                if serial_solve
                    H[j,i] = w_dot_Vj
                else
                    begin_serial_region()
                    @serial_region begin
                        H[j,i] = w_dot_Vj
                    end
                end
                parallel_map((w, V) -> w - H[j,i] * V, w, w, select_from_V(V, j))
            end
            norm_w = distributed_norm(w)
            if serial_solve
                H[i+1,i] = norm_w
            else
                begin_serial_region()
                @serial_region begin
                    H[i+1,i] = norm_w
                end
            end
            parallel_map((w) -> w / H[i+1,i], select_from_V(V, i+1), w)

            if serial_solve
                for j ∈ 1:i-1
                    gamma = c[j] * H[j,i] + s[j] * H[j+1,i]
                    H[j+1,i] = -s[j] * H[j,i] + c[j] * H[j+1,i]
                    H[j,i] = gamma
                end
                delta = sqrt(H[i,i]^2 + H[i+1,i]^2)
                s[i] = H[i+1,i] / delta
                c[i] = H[i,i] / delta
                H[i,i] = c[i] * H[i,i] + s[i] * H[i+1,i]
                H[i+1,i] = 0
                g[i+1] = -s[i] * g[i]
                g[i] = c[i] * g[i]
            else
                begin_serial_region()
                @serial_region begin
                    for j ∈ 1:i-1
                        gamma = c[j] * H[j,i] + s[j] * H[j+1,i]
                        H[j+1,i] = -s[j] * H[j,i] + c[j] * H[j+1,i]
                        H[j,i] = gamma
                    end
                    delta = sqrt(H[i,i]^2 + H[i+1,i]^2)
                    s[i] = H[i+1,i] / delta
                    c[i] = H[i,i] / delta
                    H[i,i] = c[i] * H[i,i] + s[i] * H[i+1,i]
                    H[i+1,i] = 0
                    g[i+1] = -s[i] * g[i]
                    g[i] = c[i] * g[i]
                end
                _block_synchronize()
            end
            residual = abs(g[i+1])

            if residual < tol
                break
            end
        end

        # Update initial guess to restart
        #################################

        @views y = H[1:i,1:i] \ g[1:i]

        # The following calculates
        #    delta_x .= delta_x .+ sum(y[i] .* V[:,i] for i ∈ 1:length(y))
        parallel_delta_x_calc(delta_x, V, y)
        right_preconditioner(delta_x)

        if residual < tol || restart_counter > max_restarts
            break
        end

        restart_counter += 1

        # Store J.delta_x in the variable delta_x, to use it to calculate the new first
        # Krylov vector v/beta.
        parallel_map((delta_x) -> delta_x, v, delta_x)
        approximate_Jacobian_vector_product!(v)

        # Note residual0 has already had the left_preconditioner!() applied to it.
        parallel_map((residual0, v) -> -residual0 - v, v, residual0, v)
        beta = distributed_norm(v)
        for i ∈ 2:length(y)
            parallel_map(() -> 0.0, select_from_V(V, i))
        end
        parallel_map((v) -> v/beta, select_from_V(V, 1), v)
    end

    return counter
end

end
