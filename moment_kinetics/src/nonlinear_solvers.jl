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
using ..timer_utils
using ..type_definitions: mk_float, mk_int

using LinearAlgebra
using MPI
using SparseArrays
using StatsBase: mean

struct nl_solver_info{TH,TV,Tcsg,Tlig,Tprecon,Tpretype}
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
    n_solves::Base.RefValue{mk_int}
    nonlinear_iterations::Base.RefValue{mk_int}
    linear_iterations::Base.RefValue{mk_int}
    precon_iterations::Base.RefValue{mk_int}
    global_n_solves::Base.RefValue{mk_int}
    global_nonlinear_iterations::Base.RefValue{mk_int}
    global_linear_iterations::Base.RefValue{mk_int}
    global_precon_iterations::Base.RefValue{mk_int}
    solves_since_precon_update::Base.RefValue{mk_int}
    precon_dt::Base.RefValue{mk_float}
    precon_lowerz_vcut_inds::Vector{mk_int}
    precon_upperz_vcut_inds::Vector{mk_int}
    serial_solve::Bool
    max_nonlinear_iterations_this_step::Base.RefValue{mk_int}
    max_linear_iterations_this_step::Base.RefValue{mk_int}
    total_its_soft_limit::mk_int
    preconditioner_type::Tpretype
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
                               electron_ppar_pdf_solve=false,
                               preconditioner_type=Val(:none), warn_unexpected=false)
    nl_solver_section = set_defaults_and_check_section!(
        input_dict, "nonlinear_solver", warn_unexpected;
        rtol=default_rtol,
        atol=default_atol,
        nonlinear_max_iterations=20,
        linear_rtol=1.0e-3,
        linear_atol=1.0,
        linear_restart=10,
        linear_max_restarts=0,
        preconditioner_update_interval=300,
        total_its_soft_limit=50,
        adi_precon_iterations=1,
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

    n_vcut_inds = 0
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

        @begin_serial_region()
        @serial_region begin
            H .= 0.0
            c .= 0.0
            s .= 0.0
            g .= 0.0
            V_ppar .= 0.0
            V_pdf .= 0.0
        end

        V = (V_ppar, V_pdf)

        n_vcut_inds = prod(outer_coord_sizes)
    else
        H = allocate_shared_float(linear_restart + 1, linear_restart)
        c = allocate_shared_float(linear_restart + 1)
        s = allocate_shared_float(linear_restart + 1)
        g = allocate_shared_float(linear_restart + 1)
        V = allocate_shared_float(reverse(coord_sizes)..., linear_restart+1)

        @begin_serial_region()
        @serial_region begin
            H .= 0.0
            c .= 0.0
            s .= 0.0
            g .= 0.0
            V .= 0.0
        end
    end

    if preconditioner_type === Val(:lu)
        # Create dummy LU solver objects so we can create an array for preconditioners.
        # These will be calculated properly within the time loop.
        preconditioners = fill(lu(sparse(1.0*I, total_size_coords, total_size_coords)),
                               reverse(outer_coord_sizes))
    elseif preconditioner_type === Val(:electron_split_lu)
        preconditioners = (z=fill(lu(sparse(1.0*I, coords.z.n, coords.z.n)),
                                  tuple(coords.vpa.n, reverse(outer_coord_sizes)...)),
                           vpa=fill(lu(sparse(1.0*I, coords.vpa.n, coords.vpa.n)),
                                    tuple(coords.z.n, reverse(outer_coord_sizes)...)),
                           ppar=fill(lu(sparse(1.0*I, coords.z.n, coords.z.n)),
                                     reverse(outer_coord_sizes)),
                          )
    elseif preconditioner_type === Val(:electron_lu)
        pdf_plus_moments_size = total_size_coords + 5 * coords.z.n
        preconditioners = fill((lu(sparse(1.0*I, 1, 1)),
                                allocate_shared_float(pdf_plus_moments_size, pdf_plus_moments_size),
                                allocate_shared_float(pdf_plus_moments_size),
                                allocate_shared_float(pdf_plus_moments_size),
                               ),
                               reverse(outer_coord_sizes))
        @serial_region begin
            for p ∈ preconditioners
                # Zero the input buffer so that RHS entries corresponding to the
                # 'zeroth_moment', 'first_moments', 'second_moment' and 'third_moment'
                # lines are always zero.
                p[3] .= 0.0
            end
        end
    elseif preconditioner_type === Val(:electron_adi)
        nz = coords.z.n
        pdf_plus_ppar_size = total_size_coords + nz
        nvperp = coords.vperp.n
        nvpa = coords.vpa.n
        v_size = nvperp * nvpa

        function get_adi_precon_buffers()
            v_solve_z_range = looping.loop_ranges_store[(:z,)].z
            v_solve_global_inds = [[((iz - 1)*v_size+1 : iz*v_size)..., total_size_coords+iz] for iz ∈ v_solve_z_range]
            v_solve_nsolve = length(v_solve_z_range)
            # Plus one for the one point of ppar that is included in the 'v solve'.
            v_solve_n = nvperp * nvpa + 1
            v_solve_implicit_lus = Vector{SparseArrays.UMFPACK.UmfpackLU{mk_float, mk_int}}(undef, v_solve_nsolve)
            v_solve_explicit_matrices = Vector{SparseMatrixCSC{mk_float, mk_int}}(undef, v_solve_nsolve)
            # This buffer is not shared-memory, because it will be used for a serial LU solve.
            v_solve_buffer = allocate_float(v_solve_n)
            v_solve_buffer2 = allocate_float(v_solve_n)
            v_solve_matrix_buffer = allocate_float(v_solve_n, v_solve_n)

            z_solve_vperp_range = looping.loop_ranges_store[(:vperp,:vpa)].vperp
            z_solve_vpa_range = looping.loop_ranges_store[(:vperp,:vpa)].vpa
            z_solve_global_inds = vec([(ivperp-1)*nvpa+ivpa:v_size:(nz-1)*v_size+(ivperp-1)*nvpa+ivpa for ivperp ∈ z_solve_vperp_range, ivpa ∈ z_solve_vpa_range])
            z_solve_nsolve = length(z_solve_vperp_range) * length(z_solve_vpa_range)
            @serial_region begin
                # Do the solve for ppar on the rank-0 process, which has the fewest grid
                # points to handle if there are not an exactly equal number of points for each
                # process.
                push!(z_solve_global_inds, total_size_coords+1 : total_size_coords+nz)
                z_solve_nsolve += 1
            end
            z_solve_n = nz
            z_solve_implicit_lus = Vector{SparseArrays.UMFPACK.UmfpackLU{mk_float, mk_int}}(undef, z_solve_nsolve)
            z_solve_explicit_matrices = Vector{SparseMatrixCSC{mk_float, mk_int}}(undef, z_solve_nsolve)
            # This buffer is not shared-memory, because it will be used for a serial LU solve.
            z_solve_buffer = allocate_float(z_solve_n)
            z_solve_buffer2 = allocate_float(z_solve_n)
            z_solve_matrix_buffer = allocate_float(z_solve_n, z_solve_n)

            J_buffer = allocate_shared_float(pdf_plus_ppar_size, pdf_plus_ppar_size)
            input_buffer = allocate_shared_float(pdf_plus_ppar_size)
            intermediate_buffer = allocate_shared_float(pdf_plus_ppar_size)
            output_buffer = allocate_shared_float(pdf_plus_ppar_size)
            error_buffer = allocate_shared_float(pdf_plus_ppar_size)

            chunk_size = (pdf_plus_ppar_size + block_size[] - 1) ÷ block_size[]
            # Set up so root process has fewest points, as root may have other work to do.
            global_index_subrange = max(1, pdf_plus_ppar_size - (block_size[] - block_rank[]) * chunk_size + 1):(pdf_plus_ppar_size - (block_size[] - block_rank[] - 1) * chunk_size)

            if nl_solver_input.adi_precon_iterations < 1
                error("Setting adi_precon_iterations=$(nl_solver_input.adi_precon_iterations) "
                      * "would mean the preconditioner does nothing.")
            end
            n_extra_iterations = nl_solver_input.adi_precon_iterations - 1

            return (v_solve_global_inds=v_solve_global_inds,
                    v_solve_nsolve=v_solve_nsolve,
                    v_solve_implicit_lus=v_solve_implicit_lus,
                    v_solve_explicit_matrices=v_solve_explicit_matrices,
                    v_solve_buffer=v_solve_buffer, v_solve_buffer2=v_solve_buffer2,
                    v_solve_matrix_buffer=v_solve_matrix_buffer,
                    z_solve_global_inds=z_solve_global_inds,
                    z_solve_nsolve=z_solve_nsolve,
                    z_solve_implicit_lus=z_solve_implicit_lus,
                    z_solve_explicit_matrices=z_solve_explicit_matrices,
                    z_solve_buffer=z_solve_buffer, z_solve_buffer2=z_solve_buffer2,
                    z_solve_matrix_buffer=z_solve_matrix_buffer, J_buffer=J_buffer,
                    input_buffer=input_buffer, intermediate_buffer=intermediate_buffer,
                    output_buffer=output_buffer,
                    global_index_subrange=global_index_subrange,
                    n_extra_iterations=n_extra_iterations)
        end

        preconditioners = fill(get_adi_precon_buffers(), reverse(outer_coord_sizes))
    elseif preconditioner_type === Val(:none)
        preconditioners = nothing
    else
        error("Unrecognised preconditioner_type=$preconditioner_type")
    end

    linear_initial_guess = zeros(linear_restart)

    return nl_solver_info(mk_float(nl_solver_input.rtol), mk_float(nl_solver_input.atol),
                          nl_solver_input.nonlinear_max_iterations,
                          mk_float(nl_solver_input.linear_rtol),
                          mk_float(nl_solver_input.linear_atol), linear_restart,
                          nl_solver_input.linear_max_restarts, H, c, s, g, V,
                          linear_initial_guess, Ref(0), Ref(0), Ref(0), Ref(0), Ref(0),
                          Ref(0), Ref(0), Ref(0),
                          Ref(nl_solver_input.preconditioner_update_interval),
                          Ref(mk_float(0.0)), zeros(mk_int, n_vcut_inds),
                          zeros(mk_int, n_vcut_inds), serial_solve, Ref(0), Ref(0),
                          nl_solver_input.total_its_soft_limit, preconditioner_type,
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
@timeit_debug global_timer gather_nonlinear_solver_counters!(nl_solver_params) = begin
    if nl_solver_params.ion_advance !== nothing
        # Solve runs in parallel on all processes, so no need to collect here
        nl_solver_params.ion_advance.global_n_solves[] = nl_solver_params.ion_advance.n_solves[]
        nl_solver_params.ion_advance.global_nonlinear_iterations[] = nl_solver_params.ion_advance.nonlinear_iterations[]
        nl_solver_params.ion_advance.global_linear_iterations[] = nl_solver_params.ion_advance.linear_iterations[]
        nl_solver_params.ion_advance.global_precon_iterations[] = nl_solver_params.ion_advance.precon_iterations[]
    end
    if nl_solver_params.vpa_advection !== nothing
        # Solves are run in serial on separate processes, so need a global Allreduce
        @timeit_debug global_timer "MPI.Allreduce! comm_world" MPI.Allreduce!(nl_solver_params.vpa_advection.n_solves[], +, comm_world)
        @timeit_debug global_timer "MPI.Allreduce! comm_world" MPI.Allreduce!(nl_solver_params.vpa_advection.nonlinear_iterations[], +, comm_world)
        @timeit_debug global_timer "MPI.Allreduce! comm_world" MPI.Allreduce!(nl_solver_params.vpa_advection.linear_iterations[], +, comm_world)
        @timeit_debug global_timer "MPI.Allreduce! comm_world" MPI.Allreduce!(nl_solver_params.vpa_advection.precon_iterations[], +, comm_world)
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
@timeit global_timer newton_solve!(
                         x, residual_func!, residual, delta_x, rhs_delta, v, w,
                         nl_solver_params; left_preconditioner=nothing,
                         right_preconditioner=nothing, recalculate_preconditioner=nothing,
                         coords) = begin
    # This wrapper function constructs the `solver_type` from coords, so that the body of
    # the inner `newton_solve!()` can be fully type-stable
    solver_type = Val(Symbol((c for c ∈ keys(coords))...))
    return newton_solve!(x, residual_func!, residual, delta_x, rhs_delta, v, w,
                         nl_solver_params, solver_type; left_preconditioner=left_preconditioner,
                         right_preconditioner=right_preconditioner,
                         recalculate_preconditioner=recalculate_preconditioner,
                         coords=coords)
end
function newton_solve!(x, residual_func!, residual, delta_x, rhs_delta, v, w,
                       nl_solver_params, solver_type::Val; left_preconditioner=nothing,
                       right_preconditioner=nothing, recalculate_preconditioner=nothing,
                       coords)

    rtol = nl_solver_params.rtol
    atol = nl_solver_params.atol

    if left_preconditioner === nothing
        left_preconditioner = identity
    end
    if right_preconditioner === nothing
        right_preconditioner = identity
    end

    norm_params = (coords, nl_solver_params.rtol, nl_solver_params.atol, x)

    residual_func!(residual, x)
    residual_norm = distributed_norm(solver_type, residual, norm_params...)
    counter = 0
    linear_counter = 0

    # Would need this if delta_x was not set to zero within the Newton iteration loop
    # below.
    #parallel_map(solver_type, ()->0.0, delta_x)

    close_counter = -1
    close_linear_counter = -1
    success = true
    previous_residual_norm = residual_norm
old_precon_iterations = nl_solver_params.precon_iterations[]
    while (counter < 1 && residual_norm > 1.0e-8) || residual_norm > 1.0
        counter += 1
        #println("\nNewton ", counter)

        # Solve (approximately?):
        #   J δx = -RHS(x)
        parallel_map(solver_type, ()->0.0, delta_x)
        linear_its = linear_solve!(x, residual_func!, residual, delta_x, v, w,
                                   solver_type, norm_params; coords=coords,
                                   rtol=nl_solver_params.linear_rtol,
                                   atol=nl_solver_params.linear_atol,
                                   restart=nl_solver_params.linear_restart,
                                   max_restarts=nl_solver_params.linear_max_restarts,
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   H=nl_solver_params.H, c=nl_solver_params.c,
                                   s=nl_solver_params.s, g=nl_solver_params.g,
                                   V=nl_solver_params.V, rhs_delta=rhs_delta,
                                   initial_guess=nl_solver_params.linear_initial_guess,
                                   serial_solve=nl_solver_params.serial_solve,
                                   initial_delta_x_is_zero=true)
        linear_counter += linear_its

        # If the residual does not decrease, we will do a line search to find an update
        # that does decrease the residual. The value of `x` is used to define the
        # normalisation value with rtol that is used to calculate the residual, so do not
        # want to update it until the line search is completed (otherwise the norm changes
        # during the line search, which might make it fail to converge). So calculate the
        # updated value in the buffer `w` until the line search is completed, and only
        # then copy it into `x`.
        parallel_map(solver_type, (x) -> x, w, x)
        parallel_map(solver_type, (x,delta_x) -> x + delta_x, w, x, delta_x)
        residual_func!(residual, w)

        # For the Newton iteration, we want the norm divided by the (sqrt of the) number
        # of grid points, so we can use a tolerance that is independent of the size of the
        # grid. This is unlike the norms needed in `linear_solve!()`.
        residual_norm = distributed_norm(solver_type, residual, norm_params...)
        if isnan(residual_norm)
            error("NaN in Newton iteration at iteration $counter")
        end
#        if residual_norm > previous_residual_norm
#            # Do a line search between x and x+delta_x to try to find an update that does
#            # decrease residual_norm
#            s = 0.5
#            while s > 1.0e-2
#                parallel_map(solver_type, (x,delta_x,s) -> x + s * delta_x, w, x, delta_x, s)
#                residual_func!(residual, x)
#                residual_norm = distributed_norm(solver_type, residual, norm_params...)
#                if residual_norm ≤ previous_residual_norm
#                    break
#                end
#                s *= 0.5
#            end
#            println("line search s ", s)
#
#            #if residual_norm > previous_residual_norm
#            #    # Failed to find a point that decreases the residual, so try a negative
#            #    # step
#            #    s = -1.0e-5
#            #    parallel_map(solver_type, (x,delta_x,s) -> x + s * delta_x, w, x, delta_x, s)
#            #    residual_func!(residual, x)
#            #    residual_norm = distributed_norm(solver_type, residual, norm_params...)
#            #    if residual_norm > previous_residual_norm
#            #        # That didn't work either, so just take the full step and hope for
#            #        # convergence later
#            #        parallel_map(solver_type, (x,delta_x,s) -> x + s * delta_x, w, x, delta_x, s)
#            #        residual_func!(residual, x)
#            #        residual_norm = distributed_norm(solver_type, residual, norm_params...)
#            #    end
#            #end
#            if residual_norm > previous_residual_norm
#                # Line search didn't work, so just take the full step and hope for
#                # convergence later
#                parallel_map(solver_type, (x,delta_x,s) -> x + s * delta_x, w, x, delta_x, s)
#                residual_func!(residual, x)
#                residual_norm = distributed_norm(solver_type, residual, norm_params...)
#            end
#        end
        parallel_map(solver_type, (w) -> w, x, w)
        previous_residual_norm = residual_norm

        if recalculate_preconditioner !== nothing && counter % nl_solver_params.preconditioner_update_interval == 0
            # Have taken a large number of Newton iterations already - convergence must be
            # slow, so try updating the preconditioner.
            recalculate_preconditioner()
        end

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
#    precon_count = nl_solver_params.precon_iterations[] - old_precon_iterations
#    println("Total precon iterations: ", precon_count)
#    println("Precon iterations per linear: ", precon_count / linear_counter)
#
#    println("Newton iterations after close: ", counter - close_counter)
#    println("Total linear iterations after close: ", linear_counter - close_linear_counter)
#    println("Linear iterations per Newton after close: ", (linear_counter - close_linear_counter) / (counter - close_counter))
#    println()

    return success
end

@timeit_debug global_timer distributed_norm(
                               ::Val{:z}, residual::AbstractArray{mk_float, 1}, coords,
                               rtol, atol, x) = begin
    z = coords.z

    @begin_z_region()

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

    @_block_synchronize()
    global_norm = Ref(local_norm)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_norm, +, comm_block[]) # global_norm is the norm_square for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_norm, +, comm_inter_block[]) # global_norm is the norm_square for the whole grid
        global_norm[] = sqrt(global_norm[] / z.n_global)
    end
    @_block_synchronize()
    @timeit_debug global_timer "MPI.Bcast! comm_block" MPI.Bcast!(global_norm, comm_block[]; root=0)

    return global_norm[]
end

@timeit_debug global_timer distributed_norm(
                               ::Val{:vpa}, residual::AbstractArray{mk_float, 1}, coords,
                               rtol, atol, x) = begin
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    residual_norm = 0.0
    for i ∈ eachindex(residual, x)
        residual_norm += (residual[i] / (rtol * abs(x[i]) + atol))^2
    end

    residual_norm = sqrt(residual_norm / length(residual))

    return residual_norm
end

@timeit_debug global_timer distributed_norm(
                               ::Val{:zvperpvpa},
                               residual::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}},
                               coords, rtol, atol, x) = begin
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

    @begin_z_region()

    ppar_local_norm_square = 0.0
    @loop_z iz begin
        if iz == zend
            continue
        end
        ppar_local_norm_square += (ppar_residual[iz] / (rtol * abs(x_ppar[iz]) + atol))^2
    end

    @_block_synchronize()
    global_norm_ppar = Ref(ppar_local_norm_square) # global_norm_ppar is the norm_square for ppar in the block
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_norm_ppar, +, comm_block[])

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_norm_ppar, +, comm_inter_block[]) # global_norm_ppar is the norm_square for ppar in the whole grid
        global_norm_ppar[] = global_norm_ppar[] / z.n_global
    end

    @begin_z_vperp_vpa_region()

    pdf_local_norm_square = 0.0
    @loop_z iz begin
        if iz == zend
            continue
        end
        @loop_vperp_vpa ivperp ivpa begin
            pdf_local_norm_square += (pdf_residual[ivpa,ivperp,iz] / (rtol * abs(x_pdf[ivpa,ivperp,iz]) + atol))^2
        end
    end

    @_block_synchronize()
    global_norm = Ref(pdf_local_norm_square)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_norm, +, comm_block[]) # global_norm is the norm_square for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_norm, +, comm_inter_block[]) # global_norm is the norm_square for the whole grid
        global_norm[] = global_norm[] / (z.n_global * vperp.n_global * vpa.n_global)

        global_norm[] = sqrt(mean((global_norm_ppar[], global_norm[])))
    end
    @_block_synchronize()

    @timeit_debug global_timer "MPI.Bcast! comm_block" MPI.Bcast!(global_norm, comm_block[]; root=0)

    return global_norm[]
end

@timeit_debug global_timer distributed_norm(
                  ::Val{:srzvperpvpa}, residual::AbstractArray{mk_float, 5}, coords, rtol,
                  atol, x) = begin
    n_ion_species = coords.s
    r = coords.r
    z = coords.z
    vperp = coords.vperp
    vpa = coords.vpa

    @begin_s_r_z_vperp_vpa_region()

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

    @_block_synchronize()
    global_norm = Ref(local_norm)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_norm, +, comm_block[]) # global_norm is the norm_square for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_norm, +, comm_inter_block[]) # global_norm is the norm_square for the whole grid
        global_norm[] = sqrt(global_norm[] / (n_ion_species * r.n_global * z.n_global * vperp.n_global * vpa.n_global))
    end
    @_block_synchronize()
    @timeit_debug global_timer "MPI.Bcast! comm_block" MPI.Bcast!(global_norm, comm_block[]; root=0)

    return global_norm[]
end

@timeit_debug global_timer distributed_dot(
                  ::Val{:z}, v::AbstractArray{mk_float, 1}, w::AbstractArray{mk_float, 1},
                  coords, rtol, atol, x) = begin

    z = coords.z

    @begin_z_region()

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

    @_block_synchronize()
    global_dot = Ref(local_dot)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_dot, +, comm_block[]) # global_dot is the dot for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_dot, +, comm_inter_block[]) # global_dot is the dot for the whole grid
        global_dot[] = global_dot[] / z.n_global
    end

    return global_dot[]
end

@timeit_debug global_timer distributed_dot(
                  ::Val{:vpa}, v::AbstractArray{mk_float, 1}, w::AbstractArray{mk_float, 1}, coords,
                  rtol, atol, x) = begin
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    local_dot = 0.0
    for i ∈ eachindex(v,w)
        local_dot += v[i] * w[i] / (rtol * abs(x[i]) + atol)^2
    end
    local_dot = local_dot / length(v)
    return local_dot
end

@timeit_debug global_timer distributed_dot(
                  ::Val{:zvperpvpa}, v::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}},
                  w::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}}, coords,
                  rtol, atol, x) = begin
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

    @begin_z_region()

    ppar_local_dot = 0.0
    @loop_z iz begin
        if iz == zend
            continue
        end
        ppar_local_dot += v_ppar[iz] * w_ppar[iz] / (rtol * abs(x_ppar[iz]) + atol)^2
    end

    @_block_synchronize()
    ppar_global_dot = Ref(ppar_local_dot)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(ppar_global_dot, +, comm_block[]) # ppar_global_dot is the ppar_dot for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(ppar_global_dot, +, comm_inter_block[]) # ppar_global_dot is the ppar_dot for the whole grid
        ppar_global_dot[] = ppar_global_dot[] / z.n_global
    end

    @begin_z_vperp_vpa_region()

    pdf_local_dot = 0.0
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if iz == zend
            continue
        end
        pdf_local_dot += v_pdf[ivpa,ivperp,iz] * w_pdf[ivpa,ivperp,iz] / (rtol * abs(x_pdf[ivpa,ivperp,iz]) + atol)^2
    end

    @_block_synchronize()
    global_dot = Ref(pdf_local_dot)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_dot, +, comm_block[]) # global_dot is the dot for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_dot, +, comm_inter_block[]) # global_dot is the dot for the whole grid
        global_dot[] = global_dot[] / (z.n_global * vperp.n_global * vpa.n_global)

        global_dot[] = mean((ppar_global_dot[], global_dot[]))
    end

    return global_dot[]
end

@timeit_debug global_timer distributed_dot(
                  ::Val{:srzvperpvpa}, v::AbstractArray{mk_float, 5},
                  w::AbstractArray{mk_float, 5}, coords, rtol, atol, x) = begin
    n_ion_species = coords.s
    r = coords.r
    z = coords.z
    vperp = coords.vperp
    vpa = coords.vpa

    @begin_s_r_z_vperp_vpa_region()

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

    @_block_synchronize()
    global_dot = Ref(local_dot)
    @timeit_debug global_timer "MPI.Reduce! comm_block" MPI.Reduce!(global_dot, +, comm_block[]) # global_dot is the dot for the block

    if block_rank[] == 0
        @timeit_debug global_timer "MPI.Allreduce! comm_inter_block" MPI.Allreduce!(global_dot, +, comm_inter_block[]) # global_dot is the dot for the whole grid
        global_dot[] = global_dot[] / (n_ion_species * r.n_global * z.n_global * vperp.n_global * vpa.n_global)
    end

    return global_dot[]
end

# Separate versions for different numbers of arguments as generator expressions result in
# slow code

@timeit_debug global_timer parallel_map(
                  ::Val{:z}, func, result::AbstractArray{mk_float, 1}) = begin

    @begin_z_region()

    @loop_z iz begin
        result[iz] = func()
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:z}, func, result::AbstractArray{mk_float, 1}, x1) = begin

    @begin_z_region()

    @loop_z iz begin
        result[iz] = func(x1[iz])
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:z}, func, result::AbstractArray{mk_float, 1}, x1, x2) = begin

    @begin_z_region()

    if isa(x2, AbstractArray)
        @loop_z iz begin
            result[iz] = func(x1[iz], x2[iz])
        end
    else
        @loop_z iz begin
            result[iz] = func(x1[iz], x2)
        end
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:z}, func, result::AbstractArray{mk_float, 1}, x1, x2, x3) = begin

    @begin_z_region()

    if isa(x3, AbstractArray)
        @loop_z iz begin
            result[iz] = func(x1[iz], x2[iz], x3[iz])
        end
    else
        @loop_z iz begin
            result[iz] = func(x1[iz], x2[iz], x3)
        end
    end

    return nothing
end

@timeit_debug global_timer parallel_map(
                  ::Val{:vpa}, func, result::AbstractArray{mk_float, 1}) = begin
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    for i ∈ eachindex(result)
        result[i] = func()
    end
    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:vpa}, func, result::AbstractArray{mk_float, 1}, x1) = begin
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    for i ∈ eachindex(result)
        result[i] = func(x1[i])
    end
    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:vpa}, func, result::AbstractArray{mk_float, 1}, x1, x2) = begin
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    if isa(x2, AbstractArray)
        for i ∈ eachindex(result)
            result[i] = func(x1[i], x2[i])
        end
    else
        for i ∈ eachindex(result)
            result[i] = func(x1[i], x2)
        end
    end
    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:vpa}, func, result::AbstractArray{mk_float, 1}, x1, x2, x3) = begin
    # No parallelism needed when the implicit solve is over vpa - assume that this will be
    # called inside a parallelised s_r_z_vperp loop.
    if isa(x3, AbstractArray)
        for i ∈ eachindex(result)
            result[i] = func(x1[i], x2[i], x3[i])
        end
    else
        for i ∈ eachindex(result)
            result[i] = func(x1[i], x2[i], x3)
        end
    end
    return nothing
end

@timeit_debug global_timer parallel_map(
                  ::Val{:zvperpvpa}, func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}}) = begin

    result_ppar, result_pdf = result

    @begin_z_region()

    @loop_z iz begin
        result_ppar[iz] = func()
    end

    @begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        result_pdf[ivpa,ivperp,iz] = func()
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:zvperpvpa}, func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}},
                  x1) = begin

    result_ppar, result_pdf = result
    x1_ppar, x1_pdf = x1

    @begin_z_region()

    @loop_z iz begin
        result_ppar[iz] = func(x1_ppar[iz])
    end

    @begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz])
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:zvperpvpa}, func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}},
                  x1, x2) = begin

    result_ppar, result_pdf = result
    x1_ppar, x1_pdf = x1

    if isa(x2, Tuple)
        x2_ppar, x2_pdf = x2
        @begin_z_region()

        @loop_z iz begin
            result_ppar[iz] = func(x1_ppar[iz], x2_ppar[iz])
        end

        @begin_z_vperp_vpa_region()

        @loop_z_vperp_vpa iz ivperp ivpa begin
            result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz], x2_pdf[ivpa,ivperp,iz])
        end
    else
        @begin_z_region()

        @loop_z iz begin
            result_ppar[iz] = func(x1_ppar[iz], x2)
        end

        @begin_z_vperp_vpa_region()

        @loop_z_vperp_vpa iz ivperp ivpa begin
            result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz], x2)
        end
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:zvperpvpa}, func, result::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}},
                  x1, x2, x3) = begin

    result_ppar, result_pdf = result
    x1_ppar, x1_pdf = x1
    x2_ppar, x2_pdf = x2

    if isa(x3, Tuple)
        x3_ppar, x3_pdf = x3
        @begin_z_region()

        @loop_z iz begin
            result_ppar[iz] = func(x1_ppar[iz], x2_ppar[iz], x3_ppar[iz])
        end

        @begin_z_vperp_vpa_region()

        @loop_z_vperp_vpa iz ivperp ivpa begin
            result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz], x2_pdf[ivpa,ivperp,iz], x3_pdf[ivpa,ivperp,iz])
        end
    else
        @begin_z_region()

        @loop_z iz begin
            result_ppar[iz] = func(x1_ppar[iz], x2_ppar[iz], x3)
        end

        @begin_z_vperp_vpa_region()

        @loop_z_vperp_vpa iz ivperp ivpa begin
            result_pdf[ivpa,ivperp,iz] = func(x1_pdf[ivpa,ivperp,iz], x2_pdf[ivpa,ivperp,iz], x3)
        end
    end

    return nothing
end

@timeit_debug global_timer parallel_map(
                  ::Val{:srzvperpvpa}, func, result::AbstractArray{mk_float, 5}) = begin

    @begin_s_r_z_vperp_vpa_region()

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        result[ivpa,ivperp,iz,ir,is] = func()
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:srzvperpvpa}, func, result::AbstractArray{mk_float, 5}, x1) = begin

    @begin_s_r_z_vperp_vpa_region()

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is])
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:srzvperpvpa}, func, result::AbstractArray{mk_float, 5}, x1, x2) = begin

    @begin_s_r_z_vperp_vpa_region()

    if isa(x2, AbstractArray)
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is], x2[ivpa,ivperp,iz,ir,is])
        end
    else
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is], x2)
        end
    end

    return nothing
end
@timeit_debug global_timer parallel_map(
                  ::Val{:srzvperpvpa}, func, result::AbstractArray{mk_float, 5}, x1, x2,
                  x3) = begin

    @begin_s_r_z_vperp_vpa_region()

    if isa(x3, AbstractArray)
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is], x2[ivpa,ivperp,iz,ir,is], x3[ivpa,ivperp,iz,ir,is])
        end
    else
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            result[ivpa,ivperp,iz,ir,is] = func(x1[ivpa,ivperp,iz,ir,is], x2[ivpa,ivperp,iz,ir,is], x3)
        end
    end

    return nothing
end

@timeit_debug global_timer parallel_delta_x_calc(
                  ::Val{:z}, delta_x::AbstractArray{mk_float, 1}, V, y) = begin

    @begin_z_region()

    ny = length(y)
    @loop_z iz begin
        for iy ∈ 1:ny
            delta_x[iz] += y[iy] * V[iz,iy]
        end
    end

    return nothing
end

@timeit_debug global_timer parallel_delta_x_calc(
                  ::Val{:vpa}, delta_x::AbstractArray{mk_float, 1}, V, y) = begin
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

@timeit_debug global_timer parallel_delta_x_calc(
                  ::Val{:zvperpvpa}, delta_x::Tuple{AbstractArray{mk_float, 1},AbstractArray{mk_float, 3}}, V,
                  y) = begin

    delta_x_ppar, delta_x_pdf = delta_x
    V_ppar, V_pdf = V

    ny = length(y)

    @begin_z_region()

    @loop_z iz begin
        for iy ∈ 1:ny
            delta_x_ppar[iz] += y[iy] * V_ppar[iz,iy]
        end
    end

    @begin_z_vperp_vpa_region()

    @loop_z_vperp_vpa iz ivperp ivpa begin
        for iy ∈ 1:ny
            delta_x_pdf[ivpa,ivperp,iz] += y[iy] * V_pdf[ivpa,ivperp,iz,iy]
        end
    end

    return nothing
end

@timeit_debug global_timer parallel_delta_x_calc(
                  ::Val{:srzvperpvpa}, delta_x::AbstractArray{mk_float, 5}, V, y) = begin

    @begin_s_r_z_vperp_vpa_region()

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
@timeit global_timer linear_solve!(
                         x, residual_func!, residual0, delta_x, v, w, solver_type::Val,
                         norm_params; coords, rtol, atol, restart, max_restarts,
                         left_preconditioner, right_preconditioner, H, c, s, g, V,
                         rhs_delta, initial_guess, serial_solve,
                         initial_delta_x_is_zero) = begin
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
    function approximate_Jacobian_vector_product!(v, skip_first_precon::Bool=false)
        if !skip_first_precon
            right_preconditioner(v)
        end

        parallel_map(solver_type, (x,v) -> x + Jv_scale_factor * v, v, x, v)
        residual_func!(rhs_delta, v; krylov=true)
        parallel_map(solver_type, (rhs_delta, residual0) -> (rhs_delta - residual0) * inv_Jv_scale_factor,
                     v, rhs_delta, residual0)
        left_preconditioner(v)
        return v
    end

    # To start with we use 'w' as a buffer to make a copy of residual0 to which we can apply
    # the left-preconditioner.
    parallel_map(solver_type, (delta_x) -> delta_x, v, delta_x)
    left_preconditioner(residual0)

    # This function transforms the data stored in 'v' from δx to ≈J.δx
    # If initial δx is all-zero, we can skip a right-preconditioner evaluation because it
    # would just transform all-zero to all-zero.
    approximate_Jacobian_vector_product!(v, initial_delta_x_is_zero)

    # Now we actually set 'w' as the first Krylov vector, and normalise it.
    parallel_map(solver_type, (residual0, v) -> -residual0 - v, w, residual0, v)
    beta = distributed_norm(solver_type, w, norm_params...)
    parallel_map(solver_type, (w,beta) -> w/beta, select_from_V(V, 1), w, beta)
    if serial_solve
        g[1] = beta
    else
        @begin_serial_region()
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
        inner_counter = 0
        for i ∈ 1:restart
            inner_counter = i
            counter += 1
            #println("Linear ", counter)

            # Compute next Krylov vector
            parallel_map(solver_type, (V) -> V, w, select_from_V(V, i))
            approximate_Jacobian_vector_product!(w)

            # Gram-Schmidt orthogonalization
            for j ∈ 1:i
                parallel_map(solver_type, (V) -> V, v, select_from_V(V, j))
                w_dot_Vj = distributed_dot(solver_type, w, v, norm_params...)
                if serial_solve
                    H[j,i] = w_dot_Vj
                else
                    @begin_serial_region()
                    @serial_region begin
                        H[j,i] = w_dot_Vj
                    end
                end
                parallel_map(solver_type, (w, V) -> w - H[j,i] * V, w, w, select_from_V(V, j))
            end
            norm_w = distributed_norm(solver_type, w, norm_params...)
            if serial_solve
                H[i+1,i] = norm_w
            else
                @begin_serial_region()
                @serial_region begin
                    H[i+1,i] = norm_w
                end
            end
            parallel_map(solver_type, (w) -> w / H[i+1,i], select_from_V(V, i+1), w)

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
                @begin_serial_region()
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
                @_block_synchronize()
            end
            residual = abs(g[i+1])

            if residual < tol
                break
            end
        end
        i = inner_counter

        # Update initial guess to restart
        #################################

        @views y = H[1:i,1:i] \ g[1:i]

        # The following calculates
        #    delta_x .= delta_x .+ sum(y[i] .* V[:,i] for i ∈ 1:length(y))
        parallel_delta_x_calc(solver_type, delta_x, V, y)
        right_preconditioner(delta_x)

        if residual < tol || restart_counter > max_restarts
            break
        end

        restart_counter += 1

        # Store J.delta_x in the variable delta_x, to use it to calculate the new first
        # Krylov vector v/beta.
        parallel_map(solver_type, (delta_x) -> delta_x, v, delta_x)
        approximate_Jacobian_vector_product!(v)

        # Note residual0 has already had the left_preconditioner!() applied to it.
        parallel_map(solver_type, (residual0, v) -> -residual0 - v, v, residual0, v)
        beta = distributed_norm(solver_type, v, norm_params...)
        for i ∈ 2:length(y)
            parallel_map(solver_type, () -> 0.0, select_from_V(V, i))
        end
        parallel_map(solver_type, (v,beta) -> v/beta, select_from_V(V, 1), v, beta)
    end

    return counter
end

end
