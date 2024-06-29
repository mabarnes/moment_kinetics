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
"""
module nonlinear_solvers

export setup_nonlinear_solve, gather_nonlinear_solver_counters!,
       reset_nonlinear_per_stage_counters, newton_solve!

using ..array_allocation: allocate_float, allocate_shared_float
using ..communication
using ..coordinates: coordinate
using ..input_structs
using ..looping
using ..type_definitions: mk_float, mk_int

using LinearAlgebra
using MINPACK
using MPI
using SparseArrays

struct nl_solver_info{TH,TV,Tlig,Tprecon}
    rtol::mk_float
    atol::mk_float
    nonlinear_max_iterations::mk_int
    linear_rtol::mk_float
    linear_atol::mk_float
    linear_restart::mk_int
    linear_max_restarts::mk_int
    H::TH
    V::TV
    linear_initial_guess::Tlig
    n_solves::Ref{mk_int}
    nonlinear_iterations::Ref{mk_int}
    linear_iterations::Ref{mk_int}
    global_n_solves::Ref{mk_int}
    global_nonlinear_iterations::Ref{mk_int}
    global_linear_iterations::Ref{mk_int}
    stage_counter::Ref{mk_int}
    serial_solve::Bool
    max_nonlinear_iterations_this_step::Ref{mk_int}
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
function setup_nonlinear_solve(input_dict, coords, outer_coords=(); default_rtol=1.0e-5,
                               default_atol=1.0e-12, serial_solve=false,
                               preconditioner_type=nothing)
    nl_solver_section = set_defaults_and_check_section!(
        input_dict, "nonlinear_solver";
        rtol=default_rtol,
        atol=default_atol,
        nonlinear_max_iterations=20,
        linear_rtol=1.0e-3,
        linear_atol=1.0e-15,
        linear_restart=10,
        linear_max_restarts=0,
        preconditioner_update_interval=300,
       )
    nl_solver_input = Dict_to_NamedTuple(nl_solver_section)

    coord_sizes = Tuple(isa(c, coordinate) ? c.n : c for c ∈ coords)
    total_size_coords = prod(coord_sizes)
    outer_coord_sizes = Tuple(isa(c, coordinate) ? c.n : c for c ∈ outer_coords)

    linear_restart = nl_solver_input.linear_restart

    if serial_solve
        H = allocate_float(linear_restart + 1, linear_restart)
        V = allocate_float(reverse(coord_sizes)..., linear_restart+1)
        H .= 0.0
        V .= 0.0
    else
        H = allocate_shared_float(linear_restart + 1, linear_restart)
        V = allocate_shared_float(reverse(coord_sizes)..., linear_restart+1)

        begin_serial_region()
        @serial_region begin
            H .= 0.0
            V .= 0.0
        end
    end

    if preconditioner_type == "lu"
        # Create dummy LU solver objects so we can create an array for preconditioners.
        # These will be calculated properly within the time loop.
        preconditioners = fill(lu(sparse(1.0*I, total_size_coords, total_size_coords)),
                               reverse(outer_coord_sizes))
    else
        preconditioners = nothing
    end

    linear_initial_guess = zeros(linear_restart)

    return nl_solver_info(nl_solver_input.rtol, nl_solver_input.atol,
                          nl_solver_input.nonlinear_max_iterations,
                          nl_solver_input.linear_rtol, nl_solver_input.linear_atol,
                          linear_restart, nl_solver_input.linear_max_restarts, H, V,
                          linear_initial_guess, Ref(0), Ref(0), Ref(0), Ref(0), Ref(0),
                          Ref(0), Ref(0), serial_solve, Ref(0),
                          nl_solver_input.preconditioner_update_interval, preconditioners)
end

"""
    reset_nonlinear_per_stage_counters(nl_solver_params::Union{nl_solver_info,Nothing})

Reset the counters that hold per-step totals or maximums in `nl_solver_params`.
"""
function reset_nonlinear_per_stage_counters(nl_solver_params::Union{nl_solver_info,Nothing})
    if nl_solver_params === nothing
        return nothing
    end

    nl_solver_params.max_nonlinear_iterations_this_step[] = 0

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

        if left_preconditioner === nothing
            left_preconditioner = identity
        end
        if right_preconditioner === nothing
            right_preconditioner = identity
        end

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
                                   H=nl_solver_params.H, V=nl_solver_params.V,
                                   rhs_delta=rhs_delta,
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

            if residual_norm > previous_residual_norm
                # Failed to find a point that decreases the residual, so try a negative
                # step
                s = -1.0e-5
                parallel_map((x,delta_x) -> x + s * delta_x, w, x, delta_x)
                residual_func!(residual, x)
                residual_norm = distributed_norm(residual)
                if residual_norm > previous_residual_norm
                    # That didn't work either, so just take the full step and hope for
                    # convergence later
                    parallel_map((x,delta_x) -> x + s * delta_x, w, x, delta_x)
                    residual_func!(residual, x)
                    residual_norm = distributed_norm(residual)
                end
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

"""
Apply the GMRES algorithm to solve the 'linear problem' J.δx^n = R(x^n), which is needed
at each step of the outer Newton iteration (in `newton_solve!()`).
"""
function linear_solve!(x, residual_func!, residual0, delta_x, v, w; coords, rtol, atol,
                       restart, max_restarts, left_preconditioner, right_preconditioner,
                       H, V, rhs_delta, initial_guess, distributed_norm, distributed_dot,
                       parallel_map, parallel_delta_x_calc, serial_solve)
    # Solve (approximately?):
    #   J δx = residual0

    tol = max(rtol, atol)
    epsilon = 1.0e-6 / tol
    inv_epsilon = 1.0 / epsilon

    function approximate_Jacobian_vector_product!(v)
        right_preconditioner(v)

        parallel_map((x,v) -> x + epsilon * v, v, x, v)
        residual_func!(rhs_delta, v)
        parallel_map((rhs_delta, residual0) -> (rhs_delta - residual0) * inv_epsilon,
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
    parallel_map((w) -> w/beta, selectdim(V,ndims(V),1), w)

    # Set tolerance for GMRES iteration to rtol times the initial residual, unless this is
    # so small that it is smaller than atol, in which case use atol instead.
    tol = max(rtol * beta, atol)

    lsq_result = nothing
    residual = Inf
    counter = 0
    restart_counter = 1
    while true
        for i ∈ 1:restart
            counter += 1
            #println("Linear ", counter)

            # Compute next Krylov vector
            parallel_map((V) -> V, w, selectdim(V,ndims(V),i))
            approximate_Jacobian_vector_product!(w)

            # Gram-Schmidt orthogonalization
            for j ∈ 1:i
                parallel_map((V) -> V, v, selectdim(V,ndims(V),j))
                w_dot_Vj = distributed_dot(w, v)
                if serial_solve
                    H[j,i] = w_dot_Vj
                else
                    begin_serial_region()
                    @serial_region begin
                        H[j,i] = w_dot_Vj
                    end
                end
                parallel_map((w, V) -> w - H[j,i] * V, w, w, selectdim(V,ndims(V),j))
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
            parallel_map((w) -> w / H[i+1,i], selectdim(V,ndims(V),i+1), w)

            function temporary_residual!(result, guess)
                #println("temporary residual ", size(result), " ", size(@view(H[1:i+1,1:i])), " ", size(guess))
                result .= @view(H[1:i+1,1:i]) * guess
                result[1] -= beta
            end

            # Second argument to fsolve needs to be a Vector{Float64}
            if serial_solve
                resize!(initial_guess, i)
                initial_guess[1] = beta
                initial_guess[2:i] .= 0.0
                lsq_result = fsolve(temporary_residual!, initial_guess, i+1; method=:lm)
                residual = norm(lsq_result.f)
            else
                begin_serial_region()
                if global_rank[] == 0
                    resize!(initial_guess, i)
                    initial_guess[1] = beta
                    initial_guess[2:i] .= 0.0
                    lsq_result = fsolve(temporary_residual!, initial_guess, i+1; method=:lm)
                    residual = norm(lsq_result.f)
                else
                    residual = nothing
                end
                residual = MPI.bcast(residual, comm_world; root=0)
            end
            if residual < tol
                break
            end
        end

        # Update initial guess fo restart
        if serial_solve
            y = lsq_result.x
        else
            if global_rank[] == 0
                y = lsq_result.x
            else
                y = nothing
            end
            y = MPI.bcast(y, comm_world; root=0)
        end

        # The following is the `parallel_map()` version of
        #    delta_x .= delta_x .+ sum(y[i] .* V[:,i] for i ∈ 1:length(y))
        # slightly abusing splatting to get the sum into a lambda-function.
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
            parallel_map(() -> 0.0, selectdim(V,ndims(V),i))
        end
        parallel_map((v) -> v/beta, selectdim(V,ndims(V),1), v)
    end

    return counter
end

end
