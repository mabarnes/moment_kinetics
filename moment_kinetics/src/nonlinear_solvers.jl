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
"""
module nonlinear_solvers

export setup_nonlinear_solve, newton_solve!

using ..array_allocation: allocate_shared_float
using ..communication
using ..input_structs
using ..looping
using ..type_definitions: mk_float, mk_int

using LinearAlgebra
using MINPACK
using MPI

struct nl_solver_info{TH,TV,Tlig}
    atol::mk_float
    linear_restart::mk_int
    linear_max_restarts::mk_int
    H::TH
    V::TV
    linear_initial_guess::Tlig
end

"""

`coords` is a NamedTuple of coordinates corresponding to the dimensions of the variable
that will be solved. The entries in `coords` should be ordered the same as the memory
layout of the variable to be solved (i.e. fastest-varying first).
"""
function setup_nonlinear_solve(input_dict, coords; default_atol=1.0e-6)
    nl_solver_section = set_defaults_and_check_section!(
        input_dict, "nonlinear_solver";
        atol=default_atol,
        linear_restart=10,
        linear_max_restarts=0,
       )
    nl_solver_input = Dict_to_NamedTuple(nl_solver_section)

    linear_restart = nl_solver_input.linear_restart

    H = allocate_shared_float(linear_restart + 1, linear_restart)
    V = allocate_shared_float((c.n for c ∈ values(coords))..., linear_restart+1)
    linear_initial_guess = zeros(linear_restart)

    begin_serial_region()
    @serial_region begin
        H .= 0.0
        V .= 0.0
    end

    return nl_solver_info(nl_solver_input.atol, linear_restart,
                          nl_solver_input.linear_max_restarts, H, V, linear_initial_guess)
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
"""
function newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, v, w, nl_solver_params;
                       left_preconditioner=nothing, right_preconditioner=nothing,
                       coords)

    atol = nl_solver_params.atol

    distributed_norm = get_distributed_norm(coords)
    distributed_dot = get_distributed_dot(coords)
    parallel_map = get_parallel_map(coords)

    rhs_func!(residual, x)
    residual_norm = distributed_norm(residual, coords)
    counter = 0
    linear_counter = 0
    linear_atol_max = 1.0e-3
    linear_atol_min = atol
    d_min = 0.1

    parallel_map(()->0.0, delta_x)

    close_counter = -1
    close_linear_counter = -1
    while residual_norm > atol
        counter += 1
        #println("\nNewton ", counter)

        # Damping coefficient used to make Newton iteration more stable
        d = (1.0 - d_min) * exp(-residual_norm / (100.0 * atol)) + d_min
        #println("d=$d")

        if left_preconditioner === nothing
            left_preconditioner = identity
        end
        if right_preconditioner === nothing
            right_preconditioner = identity
        end

        # Solve (approximately?):
        #   J δx = -RHS(x)
        linear_atol = exp((log(linear_atol_max) - log(linear_atol_min))
                          * (1.0 - exp(-residual_norm / (100.0 * atol)))
                          + log(linear_atol_min))
        #println("linear_atol=$linear_atol")
        parallel_map((delta_x)->((1.0 - d) * delta_x), delta_x, delta_x)
        linear_its = linear_solve!(x, rhs_func!, residual, delta_x, v, w; coords=coords,
                                   atol=linear_atol,
                                   restart=nl_solver_params.linear_restart,
                                   max_restarts=nl_solver_params.linear_max_restarts,
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   H=nl_solver_params.H, V=nl_solver_params.V,
                                   rhs_delta=rhs_delta,
                                   initial_guess=nl_solver_params.linear_initial_guess,
                                   distributed_norm=distributed_norm,
                                   distributed_dot=distributed_dot,
                                   parallel_map=parallel_map)
        linear_counter += linear_its

        parallel_map((x,delta_x) -> x + d * delta_x, x, x, delta_x)
        rhs_func!(residual, x)

        # For the Newton iteration, we want the norm divided by the (sqrt of the) number
        # of grid points, so we can use a tolerance that is independent of the size of the
        # grid. This is unlike the norms needed in `linear_solve!()`.
        residual_norm = distributed_norm(residual, coords; per_grid_point=true)

        #println("Newton residual ", residual_norm, " ", linear_its, " $atol")

        if residual_norm < 0.1 && close_counter < 0 && close_linear_counter < 0
            close_counter = counter
            close_linear_counter = linear_counter
        end

        if counter > 100000
            println("maximum iteration limit reached")
            break
        end
    end
#    println("Newton iterations: ", counter)
#    println("Final residual: ", residual_norm)
#    println("Total linear iterations: ", linear_counter)
#    println("Linear iterations per Newton: ", linear_counter / counter)
#
#    println("Newton iterations after close: ", counter - close_counter)
#    println("Total linear iterations after close: ", linear_counter - close_linear_counter)
#    println("Linear iterations per Newton after close: ", (linear_counter - close_linear_counter) / (counter - close_counter))
#    println()
end

"""
    get_distributed_norm(coords)

Get a 'distributed_norm' function that acts on arrays with dimensions given by the entries
in `coords`.
"""
function get_distributed_norm(coords)
    dims = keys(coords)
    if dims == (:z,)
        return distributed_norm_z
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`distributed_norm_*()` function in nonlinear_solvers.jl")
    end
end

function distributed_norm_z(residual::AbstractArray{mk_float, 1}, coords; per_grid_point=false)
    z = coords.z

    begin_z_region()

    local_norm = 0.0
    if z.irank < z.nrank - 1
        zend = z.n - 1
        @loop_z iz begin
            if iz == zend
                continue
            end
            local_norm += residual[iz]^2
        end
    else
        @loop_z iz begin
            local_norm += residual[iz]^2
        end
    end

    _block_synchronize()
    block_norm = MPI.Reduce(local_norm, +, comm_block[])

    if block_rank[] == 0
        global_norm = MPI.Allreduce(block_norm, +, comm_inter_block[])
        if per_grid_point
            global_norm = sqrt(global_norm / z.n_global)
        else
            global_norm = sqrt(global_norm)
        end
    else
        global_norm = nothing
    end
    global_norm = MPI.bcast(global_norm, comm_block[]; root=0)

    return global_norm
end

"""
    get_distributed_dot(coords)

Get a 'distributed_dot' function that acts on arrays with dimensions given by the entries
in `coords`.
"""
function get_distributed_dot(coords)
    dims = keys(coords)
    if dims == (:z,)
        return distributed_dot_z
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`distributed_dot_*()` function in nonlinear_solvers.jl")
    end
end

function distributed_dot_z(x::AbstractArray{mk_float, 1}, y::AbstractArray{mk_float, 1})

    begin_z_region()

    local_dot = 0.0
    @loop_z iz begin
        local_dot += x[iz] * y[iz]
    end

    _block_synchronize()
    block_dot = MPI.Reduce(local_dot, +, comm_block[])

    if block_rank[] == 0
        global_dot = MPI.Allreduce(block_dot, +, comm_inter_block[])
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
    else
        error("dims=$dims is not supported yet. Need to write another "
              * "`parallel_map_*()` function in nonlinear_solvers.jl")
    end
end

function parallel_map_z(func, result::AbstractArray{mk_float, 1},
                        args::AbstractArray{mk_float, 1}...)

    begin_z_region()

    @loop_z iz begin
        result[iz] = func((x[iz] for x ∈ args)...)
    end

    return nothing
end

function linear_solve!(x, rhs_func!, rhs0, delta_x, v, w; coords, atol, restart,
                       max_restarts, left_preconditioner, right_preconditioner,
                       H, V, rhs_delta, initial_guess, distributed_norm, distributed_dot,
                       parallel_map)
    # Solve (approximately?):
    #   J δx = rhs0

    epsilon = atol / 10.0
    inv_epsilon = 1.0 / epsilon

    function approximate_Jacobian_vector_product!(v)
        right_preconditioner(v)

        parallel_map((x,v) -> x + epsilon * v, v, x, v)
        rhs_func!(rhs_delta, v)
        parallel_map((rhs_delta, rhs0) -> (rhs_delta - rhs0) * inv_epsilon,
                     v, rhs_delta, rhs0)
        left_preconditioner(v)
        return v
    end

    # To start with we use 'w' as a buffer to make a copy of rhs0 to which we can apply
    # the left-preconditioner.
    parallel_map((delta_x) -> delta_x, v, delta_x)
    left_preconditioner(rhs0)
    # This function transforms the data stored in 'v' from δx to ≈J.δx
    approximate_Jacobian_vector_product!(v)
    # Now we actually set 'w' as the first Krylov vector, and normalise it.
    parallel_map((rhs0, v) -> -rhs0 - v, w, rhs0, v)
    beta = distributed_norm(w, coords)
    parallel_map((w) -> w/beta, @view(V[:,1]), w)

    lsq_result = nothing
    residual = Inf
    counter = 0
    restart_counter = 1
    while true
        for i ∈ 1:restart
            counter += 1
            #println("Linear ", counter)

            # Compute next Krylov vector
            parallel_map((V) -> V, w, @view(V[:,i]))
            approximate_Jacobian_vector_product!(w)

            # Gram-Schmidt orthogonalization
            for j ∈ 1:i
                parallel_map((V) -> V, v, @view(V[:,j]))
                w_dot_Vj = distributed_dot(w, v)
                begin_serial_region()
                @serial_region begin
                    H[j,i] = w_dot_Vj
                end
                parallel_map((w, V) -> w - H[j,i] * V, w, w, @view(V[:,j]))
            end
            norm_w = distributed_norm(w, coords)
            begin_serial_region()
            @serial_region begin
                H[i+1,i] = norm_w
            end
            parallel_map((w) -> w / H[i+1,i], @view(V[:,i+1]), w)

            function temporary_residual!(result, guess)
                #println("temporary residual ", size(result), " ", size(@view(H[1:i+1,1:i])), " ", size(guess))
                result .= @view(H[1:i+1,1:i]) * guess
                result[1] -= beta
            end

            # Second argument to fsolve needs to be a Vector{Float64}
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
            if residual < atol
                break
            end
        end

        # Update initial guess fo restart
        if global_rank[] == 0
            y = lsq_result.x
        else
            y = nothing
        end
        y = MPI.bcast(y, comm_world; root=0)

        # The following is the `parallel_map()` version of
        #    delta_x .= delta_x .+ sum(y[i] .* V[:,i] for i ∈ 1:length(y))
        # slightly abusing splatting to get the sum into a lambda-function.
        parallel_map((delta_x, V...) -> delta_x + sum(this_y * this_V for (this_y, this_V) ∈ zip(y, V)),
                     delta_x, delta_x, (@view(V[:,i]) for i ∈ 1:length(y))...)
        right_preconditioner(delta_x)

        if residual < atol || restart_counter > max_restarts
            break
        end

        restart_counter += 1

        # Store J.delta_x in the variable delta_x, to use it to calculate the new first
        # Krylov vector v/beta.
        parallel_map((delta_x) -> delta_x, v, delta_x)
        approximate_Jacobian_vector_product!(v)

        # Note rhs0 has already had the left_preconditioner!() applied to it.
        parallel_map((rhs0, v) -> -rhs0 - v, v, rhs0, v)
        beta = distributed_norm(v, coords)
        for i ∈ 2:length(y)
            parallel_map(() -> 0.0, @view(V[:,i]))
        end
        parallel_map((v) -> v/beta, @view(V[:,1]), v)
    end

    return counter
end

end
