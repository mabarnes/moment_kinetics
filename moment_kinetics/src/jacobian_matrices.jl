"""
Generic utilities to simplify calculation of (contributions to) Jacobian matrices
"""
module JacobianMatrices

using ..array_allocation: allocate_shared_float
using ..communication
using ..moment_kinetic_structs: coordinate
using ..type_definitions

"""
Jacobian matrix and some associated information.

`matrix` is a (non-sparse) array containing the Jacobian matrix.

`state_vector_entries` are Symbols giving the names of the variables in the state vector.

`state_vector_sizes` are the total lengths of each variable in the state vector.

`state_vector_offsets` gives the offset from indices in the corresponding variable to
indices in the full state vector. The first entry of `state_vector_offsets` is 0.
`state_vector_offsets` has a length one greater than `state_vector_entries` - the final
element is the total size of the state vector. 
"""
struct jacobian_info{NTerms,NOffsets}
    matrix::MPISharedArray{mk_float,2}
    state_vector_entries::NTuple{NTerms, Symbol}
    state_vector_coords::NTuple{NTerms, Vector{coordinate}}
    state_vector_sizes::NTuple{NTerms, Symbol}
    state_vector_offsets::NTuple{NOffsets, mk_int}
end

"""
    create_jacobian_info(; kwargs...)

Create a [`jacobian_info`](@ref) struct.

`kwargs` describes the state vector. The keys are the variable names, and the arguments
are a Vector or Tuple of the coordinates for the dimensions of each variable.
"""
function create_jacobian_info(; comm=nothing, kwargs...)
    if comm === nothing
        comm = comm_block[]
    end
    state_vector_entries = Tuple(keys(kwargs))
    state_vector_coords = Tuple([v...] for v ∈ values(kwargs))
    state_vector_sizes = Tuple(prod(c.n for c ∈ coords) for coords ∈ state_vector_coords)
    state_vector_offsets = Tuple(cumsum([0, state_vector_sizes...]))
    jacobian_matrix = allocate_shared_float(:jacobian_size=>state_vector_offsets[end],
                                            :jacobian_size=>state_vector_offsets[end])

    return jacobian_matrix_info(jacobian_matrix, state_vector_entries,
                                state_vector_coords, state_vector_sizes,
                                state_vector_offsets)
end

# Get linear index corresponding to the set of indices for each dimension.
# Use the standard Julia ordering, so first (leftmost) dimension is fastest-varying.
function get_column_index(jacobian::jacobian_info, variable_index, indices...)
    column_index = indices[end] - 1
    coords = jacobian.state_vector_coords[variable_index]
    for (d, i) ∈ reverse(enumerate(indices[1:end-1]))
        column_index = coords[d].n * column_index + i - 1
    end
    column_index += 1
    return column_index
end

"""
Add the contribution of a single term to the Jacobian matrix.

The term is a product of the form
```
prefactor1 * prefactor2 * ... * factor1^p1 * factor2^p2 * ... * d_dx1(derivative_factor1) * d_dx2(derivative_factor2) * ...
```

`rows_variable` is the name of the variable whose time derivative this 'term' contributes
to - the term contributes to rows of the Jacobian corresponding to this
evolution/constraint equation.

`prefactors` is a Tuple of Tuples `(prefactor_array, coords)` where `prefactor_array`
is one of the `prefactor1`, etc. and `coords` is a Tuple of coordinates corresponding to
each dimension of `prefactor_array`.

`factors` is a Tuple of Tuples `(name::Symbol, factor::AbstractArray{mk_float,N}, p)`
where `name` is one of the names in `jacobian.state_vector_entries`, `factor` is one of
`factor1`, etc., and `p` is the corresponding power `p1`, etc.

`derivative_factors` is a Tuple of Tuples
`(variable, derivative, upwind_speed, derivative_coord, coords)` where `variable` is
`derivative_factor1`, etc., `derivative` is `d_dx1(derivative_factor1)`, etc.,
`upwind_speed` is `nothing` for a centred derivative or the speed to upwind by for an
upwind derivative, `derivative_coord` is the coordinate corresponding to the derivative
`x1`, etc. and `coords` is a Tuple of coordinates corresponding to each dimension of
`variable`/`derivative`.
"""
function add_term_to_Jacobian(jacobian::jacobian_info, rows_variable::Symbol,
                              prefactors::Tuple, factors::Tuple,
                              derivative_factors::Tuple)

    jacobian_matrix = jacobian.matrix

    rows_variable_index = findfirst((v) -> v===rows_variable,
                                    jacobian.state_vector_entries)
    rows_variable_coords = jacobian.state_vector_coords[rows_variable_index]
    row_offset = jacobian.state_vector_offsets[rows_variable_index]

    # Find which dimensions in the 'rows_variable' correspond to the dimensions of each
    # prefactor.
    prefactor_dims = [Tuple(findfirst(c.name, rows_variable_dim_names)
                            for c ∈ coords)
                      for (_, coords) ∈ prefactors]

    # Find which dimensions in the 'rows_variable' correspond to the dimensions of each
    # factor.
    rows_variable_dim_names = Tuple(c.name for c ∈ rows_variable_coords)
    factor_dims = [Tuple(findfirst(c.name, rows_variable_dim_names)
                         for c ∈ jacobian.state_vector_coords[findfirst(name, jacobian.state_vector_entries)])
                   for (name, _, _) ∈ factors]

    # Find which dimensions in the 'rows_variable' correspond to the dimensions of each
    # derivative_factor.
    derivative_factor_dims = [Tuple(findfirst(c.name, rows_variable_dim_names)
                                    for c ∈ coords)
                              for (_, _, _, _, coords) ∈ derivative_factors]

    @inbounds for (i, indices) ∈ enumerate(CartesianIndices(Tuple(c.n for c ∈ rows_variable_coords)))
        row_index = i + row_offset

        # Add the variation with each of the non-derivative factors.
        for (i, (var_name, var_array, power)) ∈ enumerate(factors)
            factor_indices = indices[factor_dims[i]]
            column_index = get_column_index(jacobian, i, factor_indices)

            jacobian_matrix[row_index, column_index] += (
                prod(pre[indices[prefactor_dims[j]]] for (j, (pre, _)) ∈ enumerate(prefactors))
                * power * var_array[factor_indices]^(power - 1)
                * prod(f[indices[factor_dims[j]]]^p for (j, (_, f, p)) ∈ enumerate(factors) if j ≠ i)
                * prod(d[indices[derivative_factor_dims[j]]] for (j, (_, d, _, _, _)) ∈ enumerate(derivative_factors))
               )
        end
    end

    return nothing
end

end # JacobianMatrices
