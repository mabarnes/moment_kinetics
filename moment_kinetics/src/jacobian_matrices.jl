"""
Generic utilities to simplify calculation of (contributions to) Jacobian matrices
"""
module jacobian_matrices

export jacobian_info, create_jacobian_info, jacobian_initialize_identity!,
       jacobian_initialize_zero!, jacobian_initialize_bc_diagonal!, get_joined_array,
       get_sparse_joined_array, get_joined_array_adi, EquationTerm, ConstantTerm,
       CompoundTerm, NullTerm, add_term_to_Jacobian!,
       add_periodicity_constraint_to_jacobian!

using ..array_allocation: allocate_shared_float, allocate_float
using ..communication
using ..debugging
using ..looping
using ..moment_kinetics_structs: coordinate, discretization_info
using ..timer_utils
using ..type_definitions

using BlockArrays
using BlockBandedMatrices
using MPI
using SparseArrays

# Stores some information from `coordinate` struct, but has no type parameters so can be
# stored in a Vector or looked up from a Tuple without type instability.
struct mini_coordinate
    name::Symbol
    n::mk_int
    ngrid::mk_int
    nelement_local::mk_int
    nelement_global::mk_int
    ielement::Vector{mk_int}
    igrid::Vector{mk_int}
    imin::Vector{mk_int}
    element_scale::Vector{mk_float}
    wgts::Vector{mk_float}
    irank::mk_int
    nrank::mk_int
    periodic::Bool
    radau_first_element::Bool

    function mini_coordinate(c::coordinate)
        return new((f === :name ? Symbol(getfield(c, f)) : getfield(c,f)
                    for f ∈ fieldnames(mini_coordinate))...)
    end
end

# Stores some information from `discretization_info` structs, but without subtypes or type
# parameters, so can be stored in a Vector or looked up from a Tuple without type
# instability.
struct mini_discretization_info
    radau_Dmat::Matrix{mk_float}
    lobatto_Dmat::Matrix{mk_float}
    dense_second_deriv_matrix::Matrix{mk_float}

    function mini_discretization_info(d::discretization_info)
        if hasfield(typeof(d), :radau)
            radau_Dmat = d.radau.Dmat
        else
            radau_Dmat = zeros(mk_float, 0, 0)
        end
        if hasfield(typeof(d), :lobatto)
            lobatto_Dmat = d.lobatto.Dmat
        else
            lobatto_Dmat = zeros(mk_float, 0, 0)
        end
        if hasfield(typeof(d), :dense_second_deriv_matrix)
            dense_second_deriv_matrix = d.dense_second_deriv_matrix
        else
            dense_second_deriv_matrix = zeros(mk_float, 0, 0)
        end
        return new(radau_Dmat, lobatto_Dmat, dense_second_deriv_matrix)
    end
end

"""
Jacobian matrix and some associated information.

`matrix` is a block-structured Jacobian matrix. Each block is a (non-sparse) array
containing the Jacobian which is the response of the 'row variable' to the 'column
variable'. The blocks are assembled into rows (a Tuple of arrays for each block in the
row). `matrix` is then a Tuple of rows (Tuple of Tuples of arrays).

`n_entries` is the number of variables in `state_vector_entries`.

`state_vector_entries` are Symbols giving the names of the variables in the state vector.

`state_vector_numbers` is a NamedTuple providing a lookup table for the position of each
name in `state_vector_entries` from the name.

`state_vector_dims` is a Tuple of Tuples (one for each state vector variable) giving the
dimensions of that variable.

`state_vector_coords` are the `mini_coordinate` objects corresponding to
`state_vector_dims`.

`state_vector_sizes` are the total number of points in each state vector variable.

`state_vector_dim_sizes` are the total numbers of points in each dimension in
`state_vector_dims`.

`state_vector_sizes` are the total lengths of each variable in the state vector.

`state_vector_dim_steps` gives the step needed in the flattened state vector to move one
point in each dimension in `state_vector_dims`.

`state_vector_local_ranges` gives the index range in each dimension of `state_vector_dims`
that is locally owned by this process - used for parallelised loops over rows (or
columns) when dealing with a single variable at a time from the state vector.

`state_vector_local_flattened_ranges[i]` gives an index range for the flattened index
within the range `1:state_vector_sizes[i]` that is locally owned by this process - used
for parallelised loops over rows (or columns) when dealing with a single variable at a
time from the state vector.

`coords` is a NamedTuple containing `mini_coordinate` objects for all the dimensions
involved in the `jacobian_info`.

`spectral` is a NamedTuple containing `mini_discretization_info` objects for all the
dimensions involved in the `jacobian_info`.

`boundary_skip_funcs` is a NamedTuple that gives functions (for each variable in
`state_vector_entries`) that indicate when a grid point should be skipped in the Jacobian
because it is set by boundary conditions.

`handle_overlaps` is `Val(true)` or `Val(false)`, indicating whether the Jacobian matrix is
to be solved coupled across all distributed-MPI subdomains, or restricted to a single
distributed-MPI subdomain (for a more approximate but more parallelisable preconditioner).

`synchronize` is the function to be called to synchronize all processes in the
shared-memory group that is working with this `jacobian_info` object.
"""
struct jacobian_info{Tmat,N2,NTerms,Tvecinds,Tvecdims,Tveccoords,Tdimsizes,Tdimsteps,Tranges,Tfranges,Tlbranges,Tcoords,Tspectral,Tbskip,Tho,Tsync}
    matrix::Tmat
    n_entries::mk_int
    n_entries_squared_val::Val{N2}
    state_vector_entries::NTuple{NTerms, Symbol}
    state_vector_numbers::Tvecinds
    state_vector_dims::Tvecdims
    state_vector_coords::Tveccoords
    state_vector_is_constraint::NTuple{NTerms, Bool}
    state_vector_sizes::NTuple{NTerms, mk_int}
    state_vector_dim_sizes::Tdimsizes
    state_vector_dim_steps::Tdimsteps
    state_vector_local_ranges::Tranges
    state_vector_local_flattened_ranges::Tfranges
    local_block_ranges::Tlbranges
    coords::Tcoords
    spectral::Tspectral
    boundary_skip_funcs::Tbskip
    handle_overlaps::Tho
    synchronize::Tsync
end

"""
    create_jacobian_info(coords::NamedTuple, spectral::NamedTuple; comm=comm_block[],
                         synchronize::Union{Function,Nothing}=_block_synchronize,
                         boundary_skip_funcs::BSF=nothing,
                         handle_overlaps::Val=Val(true), block_banded=true,
                         kwargs...) where {BSF}

Create a [`jacobian_info`](@ref) struct.

`kwargs` describes the state vector. The keys are the variable names, and the arguments
are 3 element Tuples: the first element is the 'region type' for parallel loops over that
variable (or `nothing` if `comm = nothing` for serial operation); the second element
Vector or Tuple of Symbols giving the dimensions of the variable; the third element is a
`Bool` indicating whether the variable is determined by a constraint (`true`) or by a time
evolution equation (`false`).

`coords` is a NamedTuple giving all the needed coordinates, `spectral` is a NamedTuple
with all the needed 'spectral' objects.

`comm` is the communicator to use to allocate shared memory arrays. `synchronize` is the
function used to synchronize between shared-memory operations.

`boundary_skip_funcs` is a NamedTuple whose keys are the variable names and whose values
are functions to use to skip points that would be set by boundary conditions (or `nothing`
if no function is needed for the variable).

`handle_overlaps` is `Val(true)` or `Val(false)`, indicating whether the Jacobian matrix is
to be solved coupled across all distributed-MPI subdomains, or restricted to a single
distributed-MPI subdomain (for a more approximate but more parallelisable preconditioner).

`block_banded=false` can be passed to store the Jacobian matrix in a dense
`MPISharedArray`, rather than the default block-banded matrix type (BlockSkylineMatrix,
with data stored in a shared-memory array).
"""
function create_jacobian_info(coords::NamedTuple, spectral::NamedTuple; comm=comm_block[],
                              synchronize::Union{Function,Nothing}=_block_synchronize,
                              boundary_skip_funcs::BSF=nothing,
                              handle_overlaps::Val=Val(true), block_banded=true,
                              kwargs...) where {BSF}

    @debug_consistency_checks all(all(d ∈ keys(coords) for d ∈ v[2]) for v ∈ values(kwargs)) || error("Some coordinate required by the state variables were not included in `coords`.")

    mini_coords = (; (k=>mini_coordinate(v) for (k,v) ∈ pairs(coords))...)
    mini_spectral = (; (k=>mini_discretization_info(v) for (k,v) ∈ pairs(spectral))...)

    state_vector_entries = Tuple(keys(kwargs))
    n_entries = length(state_vector_entries)
    state_vector_numbers = (; (name => i for (i,name) ∈ enumerate(state_vector_entries))...)
    state_vector_region_types = Tuple(v[1] for v ∈ values(kwargs))
    state_vector_dims = Tuple(Tuple(v[2]) for v ∈ values(kwargs))
    state_vector_coords = Tuple(Tuple(mini_coords[d] for d ∈ v) for v ∈ state_vector_dims)
    state_vector_is_constraint = Tuple(v[3] for v ∈ values(kwargs))
    state_vector_sizes = Tuple(prod(mini_coords[d].n for d ∈ dims; init=1) for dims ∈ state_vector_dims)
    state_vector_dim_sizes = Tuple(Tuple(coords[d].n for d ∈ v) for v ∈ state_vector_dims)
    state_vector_dim_steps = Tuple(Tuple(prod(s[1:i-1]; init=mk_int(1)) for i ∈ 1:length(s))
                                   for s ∈ state_vector_dim_sizes)

    if state_vector_coords[1][end].nelement_local == 1
        # When there is only one element in the outer-most coordinate, cannot make a
        # block-structured matrix in the way that we do it below, and probably more
        # efficient anyway to stick with a simple dense matrix, since the block-structured
        # one would not have any zero blocks even if we were to handle this special case.
        block_banded = false
    end
    if (state_vector_coords[1][end].periodic && state_vector_coords[1][end].irank == 0
            && state_vector_coords[1][end].nrank == 1)
        # Periodic, but this cannot be handled by the structure of BlockSkylineMatrix at
        # the moment, so revert to dense matrices.
        println("WARNING: periodic bcs not compatible with block-banded matrix "
                * "structure, using dense matrix storage for Jacobian matrix.")
        block_banded = false
    end

    # For each variable, get the corresponding 'region type' from
    # `state_vector_region_types`. Use that to get the corresponding `LoopRanges` object,
    # and from that get the local index ranges for each dimension in the variable.
    # If the 'region type' is `nothing`, do a serial set up so the 'local' range includes
    # every point in the dimension.
    state_vector_local_ranges = Tuple(rt === nothing ? Tuple(1:mini_coords[d].n for d ∈ v) :
                                      Tuple(getfield(looping.loop_ranges_store[rt], d) for d ∈ v)
                                      for (rt,v) ∈ zip(state_vector_region_types,state_vector_dims))
    # Flattened ranges for shared-memory-parallel loops over each variable.
    function get_local_flattened_range(n)
        if comm === nothing
            local_range = 1:n
        else
            n_blocks = MPI.Comm_size(comm)
            rank = MPI.Comm_rank(comm)
            # Ranges for shared-memory-parallel loop over flattened row indices.
            # For consistency with choices elsewhere the total set of points is split into groups
            # of size [m,m,...,m,(m+1),...,(m+1),(m+1)]
            m = n ÷ n_blocks
            n_extra_point_blocks = n - n_blocks*m
            # Note `rank` is a zero-based index
            if rank < n_blocks - n_extra_point_blocks
                # This rank has `m` points.
                startind = rank * m + 1
                stopind = (rank + 1) * m
            else
                # How far is rank into the set of ranks with (m+1) points per rank?
                rank_extra_points = rank - (n_blocks - n_extra_point_blocks)
                startind = (n_blocks - n_extra_point_blocks) * m + rank_extra_points * (m + 1) + 1
                stopind = (n_blocks - n_extra_point_blocks) * m + (rank_extra_points + 1) * (m + 1)
            end
            local_range = startind:stopind
        end
        return local_range
    end
    state_vector_local_flattened_ranges =
        Tuple(get_local_flattened_range(state_vector_sizes[i]) for i ∈ 1:n_entries)


    if boundary_skip_funcs === nothing
        boundary_skip_funcs = (; (name=>nothing for name ∈ state_vector_entries)...)
    else
        if any(v ∉ keys(boundary_skip_funcs) for v ∈ state_vector_entries)
            error("When passing boundary_skip_funcs, must have an entry for every state "
                  * "vector variable ($state_vector_entries). Only got "
                  * "$(keys(boundary_skip_funcs)).")
        end
    end

    function get_block_sizes(outer_nelement, outer_ngrid, inner_row_dims_length,
                             inner_column_dims_length)
        # Can represent any block of the jacobian in a block-structured way, as long as
        # both variables share the same outer-most dimension so that the ngrid and
        # nelement are the same. The matrix entries where both row and column are in an
        # element interior are 'a', where both are an element boundary are 'd', and the
        # rest are 'b' and 'c'.
        # a  a  a  │  b  │  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  ⋅
        # a  a  a  │  b  │  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  ⋅
        # a  a  a  │  b  │  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  ⋅
        # ─────────┼─────┼────────┼─────┼─────────
        # c  c  c  │  d  │  c  c  │  d  │  ⋅  ⋅  ⋅
        # ─────────┼─────┼────────┼─────┼─────────
        # ⋅  ⋅  ⋅  │  b  │  a  a  │  b  │  ⋅  ⋅  ⋅
        # ⋅  ⋅  ⋅  │  b  │  a  a  │  b  │  ⋅  ⋅  ⋅
        # ─────────┼─────┼────────┼─────┼─────────
        # ⋅  ⋅  ⋅  │  d  │  c  c  │  d  │  c  c  c
        # ─────────┼─────┼────────┼─────┼─────────
        # ⋅  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  │  b  │  a  a  a
        # ⋅  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  │  b  │  a  a  a
        # ⋅  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  │  b  │  a  a  a

        outer_block_sizes = [outer_ngrid - 1]
        push!(outer_block_sizes, 1)
        for ielement ∈ 2:outer_nelement-1
            push!(outer_block_sizes, outer_ngrid - 2)
            push!(outer_block_sizes, 1)
        end
        push!(outer_block_sizes, outer_ngrid - 1)

        row_block_sizes = outer_block_sizes .* inner_row_dims_length
        column_block_sizes = outer_block_sizes .* inner_column_dims_length

        # Need one 'off diagonal' block on either side inside elements, but two 'off
        # diagonal' blocks for element boundaries.
        off_diagonals = mk_int[(i - 1) % 2 + 1 for i ∈ 1:2*outer_nelement-1]
        return row_block_sizes, column_block_sizes, off_diagonals
    end
    if comm === nothing
        # No shared memory needed
        function get_sub_matrix(i, j)
            if block_banded
                if state_vector_coords[i][end].name != state_vector_coords[j][end].name
                    error("Cannot create block-banded matrix for Jacobian block for "
                          * "block ($i,$j) when outer coordinates "
                          * "($(state_vector_coords[i][end].name) and "
                          * "$(state_vector_coords[j][end].name)) are not the same.")
                end
                outer_coord = state_vector_coords[i][end]
                row_block_sizes, column_block_sizes, off_diagonals =
                    get_block_sizes(outer_coord.nelement_local, outer_coord.ngrid,
                                    state_vector_dim_steps[i][end],
                                    state_vector_dim_steps[j][end])
                return BlockSkylineMatrix{mk_float}(BlockBandedMatrices.Zeros(sum(row_block_sizes),
                                                                              sum(column_block_sizes)),
                                                    row_block_sizes, column_block_sizes,
                                                    (off_diagonals,off_diagonals))
            else
                # Use dense matrices.
                return allocate_float(Symbol(:jacobian_size, i)=>state_vector_sizes[i],
                                      Symbol(:jacobian_size, j)=>state_vector_sizes[j])
            end
        end
        jacobian_matrix = Tuple(Tuple(get_sub_matrix(i,j) for j ∈ 1:n_entries)
                                for i ∈ 1:n_entries)
    else
        function get_shared_sub_matrix(i, j)
            if block_banded
                if state_vector_coords[i][end].name != state_vector_coords[j][end].name
                    error("Cannot create block-banded matrix for Jacobian block for "
                          * "block ($i,$j) when outer coordinates "
                          * "($(state_vector_coords[i][end].name) and "
                          * "$(state_vector_coords[j][end].name)) are not the same.")
                end
                # Need to use lower level interface so that the memory backing the
                # BlockSkylineMatrix is a shared-memory array allocated with
                # allocate_shared_float().
                outer_coord = state_vector_coords[i][end]
                row_block_sizes, column_block_sizes, off_diagonals =
                    get_block_sizes(outer_coord.nelement_local, outer_coord.ngrid,
                                    state_vector_dim_steps[i][end],
                                    state_vector_dim_steps[j][end])
                skyline_sizes = BlockBandedMatrices.BlockSkylineSizes(row_block_sizes,
                                                                      column_block_sizes,
                                                                      off_diagonals,
                                                                      off_diagonals)
                data_length = BlockBandedMatrices.bb_numentries(skyline_sizes)
                data = allocate_shared_float(:jacobian_data=>data_length, comm=comm)
                return BlockBandedMatrices._BlockSkylineMatrix(data, skyline_sizes)
            else
                return allocate_shared_float(Symbol(:jacobian_size, i)=>state_vector_sizes[i],
                                             Symbol(:jacobian_size, j)=>state_vector_sizes[j];
                                             comm=comm)
            end
        end
        jacobian_matrix = Tuple(Tuple(get_shared_sub_matrix(i,j) for j ∈ 1:n_entries)
                                for i ∈ 1:n_entries)
    end

    if block_banded
        function get_row_local_ranges(row)
            return Tuple(get_local_flattened_range(length(block.data)) for block ∈ row)
        end
        local_block_ranges = Tuple(get_row_local_ranges(r) for r ∈ jacobian_matrix)

        function get_block_diagonal_inds(block, block_local_range)
            all_diagonal_inds = Int64[]
            n_blocks = length(blockaxes(block)[1])
            col_lengths = blocklengths(block.block_sizes.axes[1])
            for block_ind ∈ 1:n_blocks
                block_start = BlockBandedMatrices.blockstart(block, block_ind, block_ind)
                block_stride = BlockBandedMatrices.blockstride(block, block_ind)
                block_ncols = col_lengths[block_ind]
                for i ∈ 0:block_ncols-1
                    push!(all_diagonal_inds, block_start + i * block_stride + i)
                end
            end
            return intersect(all_diagonal_inds, block_local_range)
        end
    else
        local_block_ranges = nothing
    end

    if synchronize === nothing
        if comm !== nothing
            error("`synchronize` argument is required when `comm !== nothing`.")
        end
        # Pass a dummy function that does nothing
        synchronize = (call_site)->nothing
    end

    return jacobian_info(jacobian_matrix, n_entries, Val(n_entries^2),
                         state_vector_entries, state_vector_numbers, state_vector_dims,
                         state_vector_coords, state_vector_is_constraint,
                         state_vector_sizes, state_vector_dim_sizes,
                         state_vector_dim_steps, state_vector_local_ranges,
                         state_vector_local_flattened_ranges, local_block_ranges,
                         mini_coords, mini_spectral, boundary_skip_funcs,
                         handle_overlaps, synchronize)
end

"""
    jacobian_initialize_identity!(jacobian::jacobian_info)

Initialize `jacobian.matrix` with the identity.
"""
@timeit_debug global_timer jacobian_initialize_identity!(jacobian::jacobian_info) = begin
    jacobian_matrix = jacobian.matrix
    n_entries = jacobian.n_entries
    if isa(jacobian_matrix[1][1], BlockSkylineMatrix)
        state_vector_local_flattened_ranges = jacobian.state_vector_local_flattened_ranges
        for col_variable ∈ 1:n_entries
            col_range = state_vector_local_flattened_ranges[col_variable]
            for row_variable ∈ 1:n_entries
                this_block = jacobian_matrix[row_variable][col_variable]
                if col_variable == row_variable
                    initialize_identity_diagonal_block!(jacobian, this_block,
                                                        jacobian.state_vector_local_ranges[row_variable],
                                                        jacobian.state_vector_dim_sizes[row_variable],
                                                        jacobian.state_vector_coords[row_variable])
                else
                    for col ∈ col_range
                        this_block[:,col] .= 0.0
                    end
                end
            end
        end
    else
        state_vector_local_flattened_ranges = jacobian.state_vector_local_flattened_ranges
        for col_variable ∈ 1:n_entries
            col_range = state_vector_local_flattened_ranges[col_variable]
            for row_variable ∈ 1:n_entries
                this_block = jacobian_matrix[row_variable][col_variable]
                if col_variable == row_variable
                    initialize_identity_diagonal_block!(jacobian, this_block,
                                                        jacobian.state_vector_local_ranges[row_variable],
                                                        jacobian.state_vector_dim_sizes[row_variable],
                                                        jacobian.state_vector_coords[row_variable])
                else
                    for col ∈ col_range
                        this_block[:,col] .= 0.0
                    end
                end
            end
        end
    end

    # Because we store the `synchronize` function in the `jacobian_info` object, we cannot
    # use the corresponding macro that would automatically pass `id_hash`, so need to get
    # and pass `id_hash` explicity.
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    jacobian.synchronize(id_hash)
    return nothing
end

function initialize_identity_diagonal_block!(jacobian, this_block,
                                             rows_variable_local_ranges,
                                             rows_variable_dim_sizes,
                                             rows_variable_coords)
    for indices_CartesianIndex ∈ CartesianIndices(rows_variable_local_ranges)
        indices = Tuple(indices_CartesianIndex)
        overlap_factor = get_overlap_factor(indices, rows_variable_coords,
                                            jacobian.handle_overlaps)
        # We only put a non-zero entry on the diagonal here, and this is a diagonal block,
        # so we can operate by columns instead of by rows for efficiency.
        col_index = get_flattened_index(rows_variable_dim_sizes, indices)
        this_block[:,col_index] .= 0
        this_block[col_index,col_index] = overlap_factor
    end
    return nothing
end

"""
    jacobian_initialize_zero!(jacobian::jacobian_info)

Initialize `jacobian.matrix` to zero.
"""
@timeit_debug global_timer jacobian_initialize_zero!(jacobian::jacobian_info) = begin
    jacobian_matrix = jacobian.matrix
    n_entries = jacobian.n_entries
    if isa(jacobian_matrix[1][1], BlockSkylineMatrix)
        local_block_ranges = jacobian.local_block_ranges
        for col_variable ∈ 1:n_entries
            for row_variable ∈ 1:n_entries
                this_block = jacobian_matrix[row_variable][col_variable]
                local_range = local_block_ranges[row_variable][col_variable]
                this_block.data[local_range] .= 0.0
            end
        end
    else
        state_vector_local_flattened_ranges = jacobian.state_vector_local_flattened_ranges
        for col_variable ∈ 1:n_entries
            col_range = state_vector_local_flattened_ranges[col_variable]
            for row_variable ∈ 1:n_entries
                this_block = jacobian_matrix[row_variable][col_variable]
                for col ∈ col_range
                    this_block[:,col] .= 0.0
                end
            end
        end
    end

    # Because we store the `synchronize` function in the `jacobian_info` object, we cannot
    # use the corresponding macro that would automatically pass `id_hash`, so need to get
    # and pass `id_hash` explicity.
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    jacobian.synchronize(id_hash)
    return nothing
end

"""
    jacobian_initialize_bc_diagonal!(jacobian::jacobian_info)

Initialize `jacobian.matrix` to zero, but with ones on the diagonal for rows that
correspond to boundary condition points (i.e. those skipped by
`jacobian.boundary_skip_funcs`).
"""
@timeit_debug global_timer jacobian_initialize_bc_diagonal!(jacobian::jacobian_info,
                                                            boundary_speed) = begin
    jacobian_matrix = jacobian.matrix
    for col_variable ∈ 1:length(jacobian.state_vector_entries)
        if isa(jacobian_matrix[1][1], BlockSkylineMatrix)
            col_local_block_ranges = jacobian.local_block_ranges[col_variable][col_variable]
        else
            col_local_block_ranges = nothing
        end
        col_variable_local_ranges = jacobian.state_vector_local_ranges[col_variable]
        col_variable_coords = jacobian.state_vector_coords[col_variable]
        if jacobian_state_vector_is_constraint[col_variable]
            boundary_skip = (args...)->true
        else
            boundary_skip = jacobian.boundary_skip_funcs[col_variable]
        end
        diagonal_block = jacobian_matrix[col_variable][col_variable]
        jacobian_initialize_bc_digonal_single_variable!(
            diagonal_block, col_variable_local_ranges, col_variable_coords, boundary_skip,
            boundary_speed, jacobian.handle_overlaps, jacobian.synchronize_shared)

        if isa(jacobian_matrix[1][1], BlockSkylineMatrix)
            local_block_ranges = jacobian.local_block_ranges
            for col_variable ∈ 1:n_entries
                for row_variable ∈ vcat(1:col_variable-1, col_variable+1:n_entries)
                    this_block = jacobian_matrix[row_variable][col_variable]
                    local_range = local_block_ranges[row_variable][col_variable]
                    this_block.data[local_range] .= 0.0
                end
            end
        else
            col_flattened_range = jacobian.state_vector_local_flattened_ranges[col_variable]
            for row_variable ∈ vcat(1:col_variable-1, col_variable+1:n_entries)
                this_block = jacobian_matrix[row_variable][col_variable]
                for col ∈ col_flattened_range
                    this_block[:,col] .= 0.0
                end
            end
        end
    end

    # Because we store the `synchronize` function in the `jacobian_info` object, we cannot
    # use the corresponding macro that would automatically pass `id_hash`, so need to get
    # and pass `id_hash` explicity.
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    jacobian.synchronize(id_hash)
    return nothing
end
function jacobian_initialize_bc_digonal_single_variable!(
             jacobian_block::BlockSkylineMatrix, col_variable_local_ranges,
             col_local_block_ranges, col_variable_coords, boundary_skip::F,
             boundary_speed, handle_ovelaps::Val, synchronize_shared) where {F}
    this_block = jacobian_matrix[row_variable][col_variable]
    this_block.data[col_local_block_ranges] .= 0.0

    # Need to synchronize here as `col_variable_local_ranges` and `col_local_block_ranges`
    # do not necessarily contain exactly the same points.
    synchronize_shared()

    for (col, indices_CartesianIndex) ∈ enumerate(CartesianIndices(col_variable_local_ranges))
        indices = Tuple(indices_CartesianIndex)
        if boundary_skip !== nothing && boundary_skip(boundary_speed, handle_overlaps,
                                                      indices..., col_variable_coords...)
            jacobian_block[col,col] .= 1.0
        end
    end
    return nothing
end
function jacobian_initialize_bc_digonal_single_variable!(
             jacobian_block, col_variable_local_ranges, col_local_block_ranges,
             col_variable_coords, boundary_skip::F, boundary_speed, handle_overlaps::Val,
             synchronize_shared) where {F}
    for (col, indices_CartesianIndex) ∈ enumerate(CartesianIndices(col_variable_local_ranges))
        indices = Tuple(indices_CartesianIndex)
        jacobian_block[:,col] .= 0.0
        if boundary_skip !== nothing && boundary_skip(boundary_speed, handle_overlaps,
                                                      indices..., col_variable_coords...)
            jacobian_block[col,col] .= 1.0
        end
    end
    return nothing
end

@timeit_debug global_timer get_joined_array(jacobian::jacobian_info) = begin
    n_entries = jacobian.n_entries
    array_of_arrays = [jacobian.matrix[i][j] for i ∈ 1:n_entries, j ∈ 1:n_entries]
    joined_array = mortar(array_of_arrays)
    return joined_array
end

@timeit_debug global_timer get_sparse_joined_array(jacobian::jacobian_info) = begin
    n_entries = length(jacobian.state_vector_entries)
    array_of_arrays = ntuple(i->sparse(jacobian.matrix[(i-1)÷n_entries+1][(i-1)%n_entries+1]), jacobian.n_entries_squared_val)
    joined_array = sparse_hvcat(ntuple(i->n_entries, n_entries), array_of_arrays...)::SparseMatrixCSC{mk_float,Int64}
    return joined_array
end

@timeit_debug global_timer get_joined_array_adi(jacobian::jacobian_info, f_slice,
                                                p_slice) = begin
    n_entries = jacobian.n_entries
    array_of_arrays = @views [jacobian.matrix[1][1][f_slice,f_slice] jacobian.matrix[1][2][f_slice,p_slice] ;;
                              jacobian.matrix[2][1][p_slice,f_slice] jacobian.matrix[2][2][p_slice,p_slice]]
    joined_array = mortar(array_of_arrays)
    return joined_array
end

# Optimise conversion of BlockSkylineMatrix to SparseMatrixCSC, because the default goes
# through the AbstractArray interface and is horribly slow, but we sometimes want a
# SparseMatrixCSC to pass to UMFPACK.jl (via LinearAlgebra/SparseArrays).
import SparseArrays: sparse
function sparse(m::BlockSkylineMatrix)
    s = spzeros(size(m))
    s_nzvals = SparseArrays.nonzeros(s)
    s_rowvals = SparseArrays.rowvals(s)
    s_colptr = SparseArrays.getcolptr(s)

    # Number of blocks
    N, M = blocksize(m)
    l, u = BlockBandedMatrices.colblockbandwidths(m)
    m_block_sizes = blocksizes(m)
    column_widths = [bs[2] for bs ∈ m_block_sizes[1,:]]
    column_starts = [0]
    append!(column_starts, cumsum(@view(column_widths[1:end-1])))
    @views column_starts .+= 1
    row_sizes = [bs[1] for bs ∈ m_block_sizes]
    row_offsets = [0]
    append!(row_offsets, cumsum(row_sizes[1:end-1]))
    for J ∈ 1:M
        KR = max(1,J-u[J]):min(J+l[J],N)
        if !isempty(KR)
            # Don't use BlockArrays.jl interface for extracting blocks, because that would
            # not be type-stable. We only select blocks that are a contiguous, dense array
            # so can construct a `Matrix` using the block start and block size.
            n_cols = column_widths[J]
            n_rows = sum(row_sizes[KR,J])
            n_block = n_cols * n_rows
            this_start = BlockBandedMatrices.blockstart(m, KR[1], J)
            this_block = reshape(@view(m.data[this_start:this_start+n_block-1]), n_rows, n_cols)
            sparse_block = sparse(this_block)

            this_nzvals = SparseArrays.nonzeros(sparse_block)
            append!(s_nzvals, this_nzvals)

            # The rows in the full matrix actually start at row_offsets[KR[1]]+1, so add
            # the offset to all the rowvals.
            this_rowvals = SparseArrays.rowvals(sparse_block) .+ row_offsets[KR[1]]
            append!(s_rowvals, this_rowvals)

            this_col_start = column_starts[J]
            this_col_end = this_col_start + n_cols - 1
            this_colptr = SparseArrays.getcolptr(sparse_block)
            @. @views s_colptr[this_col_start+1:this_col_end+1] += this_colptr[2:end] - 1
            @. @views s_colptr[this_col_end+2:end] += this_colptr[end] - 1
        end
    end

    return s
end

# Lookup tables for function and derivatives that can be used with EquationTerms.
# The entry in `func_derivative_lookup` should be the analytical derivative of the
# corresponding entry in `func_lookup`.
const func_lookup = (; exp=exp)
const func_derivative_lookup = (; exp=exp)

# Track different kinds of EquationTerm.
# `is_constant` is tracked separately because all these 'kinds' can also be constant.
@enum EquationTermKind ETsimple ETsum ETproduct ETpower ETfunction ETcompound ETnull

const EquationTermDataVector = @debug_shared_array_ifelse(Union{AbstractVector{mk_float},Vector{mk_float}},Vector{mk_float})

"""
Represents a term in an evolution equation, which can be composed of multiple sub-terms.
The whole right hand side of an evolution equation can be represented as an EquationTerm,
as the sum of all the RHS terms.

If the EquationTerm is not a sum, product, or constant then it is a leaf node of the
EquationTerm tree that gives a contribution to the Jacobian matrix.

Some slightly clusmy handling of array data (only storing a flattened view of the arrays)
is done to ensure that `EquationTerm` has no type parameters so that the
`sub_terms::Vector{EquationTerm}` has a concrete type.

Note that not all fields can be used for every `kind`, but all are included (rather than,
for example, having multiple subtypes of an abstract `EquationTerm` type) so that we can
make a type-stable tree of `EquationTerm` objects.
"""
struct EquationTerm
    # Sub terms, for when this term represents a sum, product, etc.
    sub_terms::Vector{EquationTerm}
    # Label for what kind of term this is - determines how it is evaluated and added to a
    # row of the Jacobian matrix.
    kind::EquationTermKind
    # If this term is constant, it only contributes to prefactors of other terms - it has
    # no variation that contributes to the Jacobian itself.
    is_constant::Bool
    # Which variable in the state vector does this term depend on? Controls which columns
    # in the Jacobian Matrix it contributes to. Will be `Symbol("")` unless `kind =
    # ETsimple`.
    state_variable::Symbol
    # The dimensions of the array of values of this term.
    dimensions::Vector{Symbol}
    # When filling the Jacobian, for each state vector variable, we loop over all the
    # dimensions of that variable. A given term may have all of those dimensions, or only
    # a subset. `dimension_numbers` pick out the indices from the loop that belong to the
    # dimensions of this term.
    dimension_numbers::Vector{mk_int}
    # The sizes of the dimensions in `dimensions`.
    dimension_sizes::Vector{mk_int}
    # If `kind = ETpower`, this term is `sub_terms[1]^exponent`. Otherwise `exponent=1`.
    exponent::Union{mk_int,mk_float}
    # If `kind = ETfunction`, gives the name of the function to look up in `func_lookup`
    # and `func_derivative_lookup`.
    func_name::Symbol
    # If `kind = ETsimple`, this term may represent a derivative, e.g. `∂f/∂z`. This field
    # lists the derivatives. At the moment only one derivative is supported.
    derivatives::Vector{Symbol}
    # The positions in `dimensions` of the entries in `derivatives`.
    derivative_dim_numbers::Vector{mk_int}
    # Buffer used when finding the column indices in `matrix` where entries from the 1d
    # derivative matrix need to be inserted.
    current_derivative_indices::Vector{mk_int}
    # Upwind speeds corresponding to each derivative in `derivatives`. If the derivative
    # is not upwinded, set the entry to a zero-size Vector{mk_int}.
    upwind_speeds::Vector{EquationTermDataVector}
    # If `kind = ETsimple`, this term may represent a second derivative, e.g. `∂²f/∂z²`. This field
    # lists the second derivatives. At the moment only one second derivative is supported.
    second_derivatives::Vector{Symbol}
    # The positions in `dimensions` of the entries in `second_derivatives`.
    second_derivative_dim_numbers::Vector{mk_int}
    # Buffer used when finding the column indices in `matrix` where entries from the 1d
    # second derivative matrix need to be inserted.
    current_second_derivative_indices::Vector{mk_int}
    # If `kind = ETsimple`, this term may represent an integral, e.g. `∫f dv⟂ dv∥`, (or an
    # integral of a derivative e.g. `∫ ∂f/∂z dv⟂ dv∥`). This field lists the dimensions
    # that are integrated over.
    integrals::Vector{Symbol}
    # `EquationTerm` giving the prefactor in the integral, e.g. `∫ prefactor f dv⟂ dv∥`.
    integrand_prefactor::Union{Nothing,EquationTerm}
    # Buffer used to store the dimension indices corresponding to the column when adding
    # an integral to the row.
    integrand_current_indices::Vector{mk_int}
    # Positions within `integrand_current_indices` of the indices corresponding to
    # `dimensions` (i.e. the dimensions that are not being integrated over).
    outer_dim_integrand_current_indices_slice::Vector{mk_int}
    # Positions within `integrand_current_indices` of the indices corresponding to
    # `integrals` (i.e. the dimensions that are being integrated over).
    integral_dim_integrand_current_indices_slice::Vector{mk_int}
    # When this term is an integral of a derivative, an EquationTerm representing the
    # derivative (e.g. `∂f/∂z`).
    integral_derivative_term::Union{Nothing,EquationTerm}
    # The integral weights corresponding to each dimension in `integrals`.
    integral_wgts::Vector{Vector{mk_float}}
    # Sizes of each dimension in `integrals`.
    integral_dim_sizes::Vector{mk_int}
    # Array giving the flattened representation of the (possibly multidimensional) array
    # corresponding to this term. A flattened representation is used so that the data has
    # the same type for any `EquationTerm`, regardless of the number of dimensions of the
    # term, allowing `EquationTerm` to be a concrete type without type parameters, so that
    # the tree structure of EquationTerm.sub_terms is straightforwardly concretely typed.
    flattened_data::EquationTermDataVector
    # The current loop indices corresponding to `dimensions`.
    current_indices::Vector{mk_int}
    # The current value of this term, at `current_indices`, looked up from
    # `flattened_data`.
    current_value::Array{mk_float,0}
end

# Need a wrapper to deal with debug features. `vec()` converts to `Array`, which is not
# allowed for `MPIDebugSharedArray`, which would be used if @debug_shared_array is active.
function get_flattened_array(array)
    return @debug_shared_array_ifelse(
               reshape(array, length(array)),
               vec(array)
              )
end

"""
    ConstantTerm(array::AbstractArray; dims_coords...)

Create a constant `EquationTerm`, which does not depend on any of the state variables. The
values of this term are passed as `array`, with the coordinates corresponding to the
dimensions of `array` being the keyword arguments `dims_coords...`.
"""
function ConstantTerm(array::AbstractArray; dims_coords...)
    @debug_consistency_checks size(array) == Tuple(c.n for c ∈ values(dims_coords)) || error("Size of array $(size(array)) does not match coordinates $((; (d=>c.n for (d,c) ∈ pairs(dims_coords))...))")

    dimensions = Symbol[keys(dims_coords)...]
    dimension_sizes = mk_int[c.n for c ∈ values(dims_coords)]

    return EquationTerm(EquationTerm[], ETsimple, true, Symbol(""), dimensions, mk_int[],
                        dimension_sizes, 1, Symbol(""), Symbol[], mk_int[], mk_int[],
                        Vector{mk_float}[], Symbol[], mk_int[], mk_int[], Symbol[],
                        nothing, mk_int[], mk_int[], mk_int[], nothing,
                        Vector{mk_float}[], mk_int[], get_flattened_array(array),
                        fill(mk_int(0), length(dimensions)), fill(mk_float(NaN)))
end

"""
    NullTerm()

Create a null `EquationTerm`, to use for example when a term is switched off by some
option. Null terms are dropped when summed with non-null terms, or make the whole result
null when multiplied by other terms.
"""
function NullTerm()
    return EquationTerm(EquationTerm[], ETnull, true, Symbol(""), Symbol[], mk_int[],
                        mk_int[], 1, Symbol(""), Symbol[], mk_int[], mk_int[],
                        Vector{mk_float}[], Symbol[], mk_int[], mk_int[], Symbol[],
                        nothing, mk_int[], mk_int[], mk_int[], nothing,
                        Vector{mk_float}[], mk_int[], fill(mk_float(NaN), 1), mk_int[],
                        fill(mk_float(NaN)))
end

const unit_term = ConstantTerm(ones(mk_float))

"""
    EquationTerm(state_variable::Symbol, array::AbstractArray{mk_float};
                 derivatives::Vector{Symbol}=Symbol[],
                 upwind_speeds::Vector{T} where T <: Union{Nothing,AbstractArray{mk_float}}=Nothing[],
                 second_derivatives::Vector{Symbol}=Symbol[],
                 integrand_coordinates::Union{Vector{Tc},Nothing} where Tc <: coordinate=nothing,
                 integrand_prefactor::EquationTerm=unit_term,
                 dims_coords...)

Construct an EquationTerm object, used to fill Jacobian matrices.

`state_variable` is the name of the variable in the `jacobian_info` that this term represents.

`array` gives the values of this term. `nothing` may be passed for `array`, but this is
only intended for internal use.

`derivatives` is a list of any derivatives of `state_variable` included in this term,
which are (optionally) upwinded by `upwind_speeds`.

`second_derivatives` is a list of any second derivatives of `state_variable` included in this term.

`integrand_coordinates` is a list of coordinates corresponding to dimensions that
`state_variable` is integrated over in this term. May be combined with `derivatives`. The
prefactor of `state_variable` in the integral is `integrand_prefactor`.

`dims_coords...` are `name=c::coordinate` keyword arguments giving the coordinates
corresponding to the dimensions of `state_variable`.
"""
function EquationTerm(state_variable::Symbol,
                      array::Union{AbstractArray{mk_float},Nothing};
                      derivatives::Vector{Symbol}=Symbol[],
                      upwind_speeds::Vector{T} where T <: Union{Nothing,AbstractArray{mk_float}}=Nothing[],
                      second_derivatives::Vector{Symbol}=Symbol[],
                      integrand_coordinates::Union{Vector{Tc},Nothing} where Tc <: coordinate=nothing,
                      integrand_prefactor::EquationTerm=unit_term, dims_coords...)
    @debug_consistency_checks array === nothing || size(array) == Tuple(c.n for c ∈ values(dims_coords)) || error("Size of array $(size(array)) does not match coordinates $((; (d=>c.n for (d,c) ∈ pairs(dims_coords))...))")
    @debug_consistency_checks begin
        for (i,(u,du)) ∈ enumerate(zip(upwind_speeds,derivatives))
            if u !== nothing && size(u) != Tuple(c.n for c ∈ values(dims_coords))
                error("Size of upwind speed $i $(size(u)) does not match coordinates $(Tuple(c.n for c ∈ values(dims_coords)))")
            end
        end
    end

    if array === nothing
        array = zeros(mk_float)
    end

    dimensions = Symbol[keys(dims_coords)...]
    if integrand_coordinates === nothing
        integrand_dimensions = Symbol[]
        integral_wgts = Vector{mk_float}[]
        integral_dim_sizes = Vector{mk_int}[]
    else
        integrand_dimensions = Symbol[Symbol(c.name) for c ∈ integrand_coordinates]
        integral_coordinates = [c for (d,c) ∈ zip(integrand_dimensions, integrand_coordinates)
                                if d ∉ dimensions]
        integral_wgts = Vector{mk_float}[c.wgts for c in integral_coordinates]
        integral_dim_sizes = mk_int[c.n for c in integral_coordinates]
    end
    @debug_consistency_checks begin
        integral_dims = setdiff(Set(integrand_dimensions), Set(dimensions))
        if any(dim ∈ integral_dims for dim ∈ derivatives)
            error("Cannot take derivative ($derivatives) in the same dimension as an integral ($integral_dims).")
        end
    end
    dimension_sizes = mk_int[c.n for c ∈ values(dims_coords)]
    derivative_dim_numbers = mk_int[findfirst((d)->d===derivative_dim, dimensions)
                                    for derivative_dim ∈ derivatives]
    second_derivative_dim_numbers = mk_int[findfirst((d)->d===second_derivative_dim, dimensions)
                                           for second_derivative_dim ∈ second_derivatives]

    integrals = Symbol[d for d ∈ integrand_dimensions if d ∉ dimensions]
    if length(integrals) > 0
        # deepcopy the `integrand_prefactor` in case it or some sub-terms that were used
        # in it were also used in other terms. When adding the integral terms to the
        # Jacobian matrix, the `current_value[]` fields of `integrand_prefactor` and its
        # sub-terms will be reset to their values at positions within the integral, not
        # just left at the row position, so might make other terms incorrect if they were
        # shared.
        integrand_prefactor = deepcopy(integrand_prefactor)
    end
    if length(integrals) > 0 && length(derivatives) > 0
        derivative_term = EquationTerm(state_variable, nothing;
                                       derivatives=derivatives,
                                       upwind_speeds=upwind_speeds,
                                       (Symbol(c.name)=>c for c ∈ integrand_coordinates)...)
        current_derivative_indices = zeros(mk_int, length(integrand_coordinates))
    else
        derivative_term = nothing
        current_derivative_indices = zeros(mk_int, length(dimensions))
    end
    if length(integrals) > 0 && length(second_derivatives) > 0
        second_derivative_term = EquationTerm(state_variable, nothing;
                                              second_derivatives=second_derivatives,
                                              upwind_speeds=upwind_speeds,
                                              (Symbol(c.name)=>c for c ∈ integrand_coordinates)...)
        current_second_derivative_indices = zeros(mk_int, length(integrand_coordinates))
    else
        second_derivative_term = nothing
        current_second_derivative_indices = zeros(mk_int, length(dimensions))
    end

    if length(upwind_speeds) == 0
        upwind_speeds = [zeros(mk_float, 0) for _ ∈ derivatives]
    else
        @debug_consistency_checks length(derivatives) == length(upwind_speeds) || error("`upwind_speeds` was passed, but length is not the same as length of `derivatives`.")

        upwind_speeds = [u === nothing ? zeros(mk_float, 0) : get_flattened_array(u)
                         for u ∈ upwind_speeds]
    end

    return EquationTerm(EquationTerm[], ETsimple, false, state_variable, dimensions,
                        mk_int[], dimension_sizes, 1, Symbol(""), derivatives,
                        derivative_dim_numbers, current_derivative_indices,
                        upwind_speeds, second_derivatives, second_derivative_dim_numbers,
                        current_second_derivative_indices, integrals, integrand_prefactor,
                        mk_int[], mk_int[], mk_int[], derivative_term, integral_wgts,
                        integral_dim_sizes, get_flattened_array(array),
                        fill(mk_int(0), length(dimensions)), fill(mk_float(NaN)))
end

"""
    CompoundTerm(compound_term_expanded::EquationTerm,
                 array::AbstractArray{mk_float}; dims_coords...)

Create a 'compound' `EquationTerm`. Useful when the discrete version of the term is not
exactly equal to the combination of its parts, for example a derivative of a product
\$\\partial(ab) = a\\partial b + b \\partial a\$ where the numerical derivative of \$ab\$
is not exactly equal to the expanded form. The expanded form, passed in as
`compound_term_expanded`, is needed to calculate the functional derivatives (terminology?)
that give the Jacobian. The numerical values of this term are passed as `array`, with the
coordinates corresponding to the dimensions of `array` being the keyword arguments
`dims_coords...`.
"""
function CompoundTerm(compound_term_expanded::EquationTerm,
                      array::AbstractArray{mk_float}; dims_coords...)
    @debug_consistency_checks size(array) == Tuple(c.n for c ∈ values(dims_coords)) || error("Size of array $(size(array)) does not match coordinates $((; (d=>c.n for (d,c) ∈ pairs(dims_coords))...))")

    dimensions = Symbol[keys(dims_coords)...]
    dimension_sizes = mk_int[c.n for c ∈ values(dims_coords)]

    return EquationTerm([compound_term_expanded], ETcompound,
                        compound_term_expanded.is_constant, Symbol(""), dimensions,
                        mk_int[], dimension_sizes, 1, Symbol(""), Symbol[], mk_int[],
                        mk_int[], Vector{mk_float}[], Symbol[], mk_int[], mk_int[],
                        Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                        Vector{mk_float}[], mk_int[], get_flattened_array(array),
                        fill(mk_int(0), length(dimensions)), fill(mk_float(NaN)))
end

# Make EquationTerm compatible with broadcasting syntax (e.g. `c .* term`), by having it
# be treated as a scaler (see
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting), so
# that the operations fall back to our definitions below.
Base.broadcastable(o::EquationTerm) = Ref(o)

function Base.:^(x::EquationTerm, exponent)
    if x.kind === ETnull
        return x
    elseif x.kind === ETpower
        return EquationTerm((f === :exponent ? exponent * x.exponent : getfield(x, f)
                             for f ∈ fieldnames(EquationTerm))...)
    else
        return EquationTerm([x], ETpower, x.is_constant, Symbol(""), Symbol[], mk_int[],
                            mk_int[], exponent, Symbol(""), Symbol[], mk_int[], mk_int[],
                            Vector{mk_float}[], Symbol[], mk_int[], mk_int[], Symbol[],
                            nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    end
end

# Needed because (sometimes?) `x^(-1)` is converted to `inv(x)`.
function Base.inv(x::EquationTerm)
    if x.kind === ETnull
        return x
    elseif x.kind === ETpower
        return EquationTerm((f === :exponent ? - x.exponent : getfield(x, f)
                             for f ∈ fieldnames(EquationTerm))...)
    else
        return EquationTerm([x], ETpower, x.is_constant, Symbol(""), Symbol[], mk_int[],
                            mk_int[], -1, Symbol(""), Symbol[], mk_int[], mk_int[],
                            Vector{mk_float}[], Symbol[], mk_int[], mk_int[], Symbol[],
                            nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    end
end

# Only function provided so far is `exp`. Could be extended to other functions by adding a
# similar method here, and adding the functions to `func_lookup` and
# `func_derivative_lookup`.
function Base.exp(x::EquationTerm)
    if x.kind === ETnull
        return x
    end
    return EquationTerm([x], ETfunction, x.is_constant, Symbol(""), Symbol[], mk_int[],
                        mk_int[], 1, :exp, Symbol[], mk_int[], mk_int[],
                        Vector{mk_float}[], Symbol[], mk_int[], mk_int[], Symbol[],
                        nothing, mk_int[], mk_int[], mk_int[], nothing,
                        Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                        fill(mk_float(NaN)))
end

function Base.:*(x::EquationTerm, y::EquationTerm)
    is_constant = x.is_constant && y.is_constant
    if x.kind === ETnull
        # Product of a 'null' term with anything is null.
        return x
    elseif y.kind === ETnull
        # Product of a 'null' term with anything is null.
        return y
    elseif x.kind === ETproduct && y.kind === ETproduct
        return EquationTerm(vcat(x.sub_terms, y.sub_terms), ETproduct, is_constant,
                            Symbol(""), Symbol[], mk_int[], mk_int[], 1, Symbol(""),
                            Symbol[], mk_int[], mk_int[], Vector{mk_float}[],
                            Symbol[], mk_int[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
    elseif x.kind === ETproduct
        return EquationTerm(vcat(x.sub_terms, y), ETproduct, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Symbol[], mk_int[],
                            mk_int[], Symbol[], nothing, mk_int[], mk_int[], mk_int[],
                            nothing, Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    elseif y.kind === ETproduct
        return EquationTerm(vcat(x, y.sub_terms), ETproduct, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Symbol[], mk_int[],
                            mk_int[], Symbol[], nothing, mk_int[], mk_int[], mk_int[],
                            nothing, Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    else
        return EquationTerm([x, y], ETproduct, is_constant, Symbol(""), Symbol[],
                            mk_int[], mk_int[], 1, Symbol(""), Symbol[], mk_int[],
                            mk_int[], Vector{mk_float}[], Symbol[], mk_int[], mk_int[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    end
end

function Base.:*(x::Number, y::EquationTerm)
    x_term = ConstantTerm(fill(mk_float(x)))
    if y.kind === ETnull
        # Product of a 'null' term with anything is null.
        return y
    elseif y.kind === ETproduct
        return EquationTerm(vcat(x_term, y.sub_terms), ETproduct, y.is_constant,
                            Symbol(""), Symbol[], mk_int[], mk_int[], 1, Symbol(""),
                            Symbol[], mk_int[], mk_int[], Vector{mk_float}[],
                            Symbol[], mk_int[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
    else
        return EquationTerm([x_term, y], ETproduct, y.is_constant, Symbol(""), Symbol[],
                            mk_int[], mk_int[], 1, Symbol(""), Symbol[], mk_int[],
                            mk_int[], Vector{mk_float}[], Symbol[], mk_int[], mk_int[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    end
end
Base.:*(x::EquationTerm, y::Number) = Base.:*(y, x)

function Base.:+(x::EquationTerm, y::EquationTerm)
    is_constant = x.is_constant && y.is_constant
    if x.kind === ETsum && y.kind === ETsum
        return EquationTerm(vcat(x.sub_terms, y.sub_terms), ETsum, is_constant,
                            Symbol(""), Symbol[], mk_int[], mk_int[], 1, Symbol(""),
                            Symbol[], mk_int[], mk_int[], Vector{mk_float}[],
                            Symbol[], mk_int[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
    elseif x.kind === ETnull
        # Drop 'null' terms from sums.
        return y
    elseif y.kind === ETnull
        # Drop 'null' terms from sums.
        return x
    elseif x.kind === ETsum
        return EquationTerm(vcat(x.sub_terms, y), ETsum, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Symbol[], mk_int[],
                            mk_int[], Symbol[], nothing, mk_int[], mk_int[], mk_int[],
                            nothing, Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    elseif y.kind === ETsum
        return EquationTerm(vcat(x, y.sub_terms), ETsum, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Symbol[], mk_int[],
                            mk_int[], Symbol[], nothing, mk_int[], mk_int[], mk_int[],
                            nothing, Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    else
        return EquationTerm([x, y], ETsum, is_constant, Symbol(""), Symbol[], mk_int[],
                            mk_int[], 1, Symbol(""), Symbol[], mk_int[], mk_int[],
                            Symbol[], mk_int[], mk_int[], Vector{mk_float}[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    end
end

function Base.:+(x::Number, y::EquationTerm)
    x_term = ConstantTerm(fill(mk_float(x)))
    if y.kind === ETnull
        # Drop 'null' terms from sums.
        return ConstantTerm(fill(mk_float(x)))
    elseif y.kind === ETsum
        return EquationTerm(vcat(x_term, y.sub_terms), ETsum, y.is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Symbol[], mk_int[],
                            mk_int[], Symbol[], nothing, mk_int[], mk_int[], mk_int[],
                            nothing, Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    else
        return EquationTerm([x_term, y], ETsum, y.is_constant, Symbol(""), Symbol[],
                            mk_int[], mk_int[], 1, Symbol(""), Symbol[], mk_int[],
                            mk_int[], Vector{mk_float}[], Symbol[], mk_int[], mk_int[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    end
end
Base.:+(x::EquationTerm, y::Number) = Base.:+(y, x)

function Base.:-(x::EquationTerm)
    return (-1.0) * x
end

function Base.:-(x::EquationTerm, y::EquationTerm)
    return x + (-y)
end

function Base.:-(x::Number, y::EquationTerm)
    return x + (-y)
end

function Base.:-(x::EquationTerm, y::Number)
    return x + (-y)
end

# Called before looping through all the rows of `jacobian.matrix`, sets the
# `dimension_numbers` field, plus a few other things, for each sub-term in the tree of
# `EquationTerm` objects.
function set_dimension_numbers!(jacobian::jacobian_info, row_variable_number,
                                term::EquationTerm)
    @inbounds begin
        for st ∈ term.sub_terms
            set_dimension_numbers!(jacobian, row_variable_number, st)
        end

        empty!(term.dimension_numbers)
        row_variable_dims = jacobian.state_vector_dims[row_variable_number]
        for d ∈ term.dimensions
            push!(term.dimension_numbers, findfirst((x)->x===d, row_variable_dims))
        end

        if length(term.integrals) > 0
            empty!(term.integrand_current_indices)
            empty!(term.outer_dim_integrand_current_indices_slice)
            empty!(term.integral_dim_integrand_current_indices_slice)

            outer_dims = term.dimensions
            outer_ndim = length(outer_dims)
            integral_dims = term.integrals
            integral_ndim = length(integral_dims)
            integrand_dims = jacobian.state_vector_dims[jacobian.state_vector_numbers[term.state_variable]]
            @debug_consistency_checks union(Set(outer_dims), Set(integral_dims)) == Set(integrand_dims) || error("dimensions ($outer_dims) and integral dimensions ($integral_dims) are not consistent with variable $(term.state_variable) which has dimensions $integrand_dims")

            # Vector in which indices that access the integrand will be set.
            push!(term.integrand_current_indices, zeros(mk_int, outer_ndim + integral_ndim)...)

            # Numbers that slice `integrand_current_indices` to just the entries corresponding
            # to the outer dimensions.
            for d ∈ outer_dims
                push!(term.outer_dim_integrand_current_indices_slice, findfirst((x)->x===d, integrand_dims))
            end

            # Numbers that slice `integrand_current_indices` to just the entries corresponding
            # to the integral dimensions.
            for d ∈ integral_dims
                push!(term.integral_dim_integrand_current_indices_slice, findfirst((x)->x===d, integrand_dims))
            end

            # Set dimension numbers for the integrand prefactor term, which would not be
            # caught by the recursion of `set_dimension_numbers!()`.
            integrand_variable_number = jacobian.state_vector_numbers[term.state_variable]
            set_dimension_numbers!(jacobian, integrand_variable_number, term.integrand_prefactor)

            if length(term.derivatives) > 0
                # Also set up the EquationTerm representing the derivative within the
                # integral-of-derivative.
                set_dimension_numbers!(jacobian, integrand_variable_number,
                                       term.integral_derivative_term)
            end
        end

        return nothing
    end
end

# Use the standard Julia ordering, so first (leftmost) dimension is fastest-varying.
function get_flattened_index(sizes, indices)
    @inbounds begin
        @debug_consistency_checks length(sizes) == length(indices) || error("sizes=$sizes and indices=$indices different lengths")
        ndim = length(indices)
        if ndim == 0
            # Need to handle the case when there are no indices specially to avoid incorrect
            # indexing.
            return 1
        end

        flattened_index = indices[end] - 1
        for d ∈ ndim-1:-1:1
            flattened_index = sizes[d] * flattened_index + indices[d] - 1
        end
        flattened_index += 1
        return flattened_index
    end
end

# Traverse a tree of `EquationTerm` objects, looking up or calculating the current value
# at `indices` for each one.
function set_current_values!(jacobian::jacobian_info, x::EquationTerm,
                             indices::Union{Tuple,Vector})
    @inbounds begin
        if x.kind === ETsum
            for sub_term ∈ x.sub_terms
                set_current_values!(jacobian, sub_term, indices)
            end
            x.current_value[] = sum(sub_term.current_value[] for sub_term ∈ x.sub_terms)
        elseif x.kind === ETproduct
            for sub_term ∈ x.sub_terms
                set_current_values!(jacobian, sub_term, indices)
            end
            x.current_value[] = prod(sub_term.current_value[] for sub_term ∈ x.sub_terms)
        elseif x.kind === ETpower
            @debug_consistency_checks length(x.sub_terms) == 1 || error("'power' EquationTerm must have exactly one `sub_term` element")
            sub_term = x.sub_terms[1]
            set_current_values!(jacobian, sub_term, indices)
            x.current_value[] = sub_term.current_value[]^x.exponent
        elseif x.kind === ETfunction
            @debug_consistency_checks length(x.sub_terms) == 1 || error("'function' EquationTerm must have exactly one `sub_term` element")
            sub_term = x.sub_terms[1]
            set_current_values!(jacobian, sub_term, indices)
            x.current_value[] = func_lookup[x.func_name](sub_term.current_value[])
        else
            for (i,j) ∈ enumerate(x.dimension_numbers)
                x.current_indices[i] = indices[j]
            end
            x.current_value[] = x.flattened_data[get_flattened_index(x.dimension_sizes, x.current_indices)]

            if x.kind === ETcompound
                @debug_consistency_checks length(x.sub_terms) == 1 || error("'compound' EquationTerm must have exactly one `sub_term` element")
                set_current_values!(jacobian, x.sub_terms[1], indices)
            end
        end

        return x
    end
end

# Get linear index corresponding to the set of indices for each dimension.
@inline function get_column_index(jacobian::jacobian_info, state_variable::Symbol, indices)
    @inbounds begin
        variable_index = jacobian.state_vector_numbers[state_variable]
        return variable_index, get_column_index(jacobian, variable_index, indices)
    end
end
@inline function get_column_index(jacobian::jacobian_info, variable_index, indices)
    return @inbounds get_flattened_index(jacobian.state_vector_dim_sizes[variable_index], indices)
end

# Get the global column indices corresponding to the entire element in 'dim'  that
# contains the point specified by `indices`.
function get_element_inds(jacobian::jacobian_info, indices, derivative_indices,
                          variable_index, variable_ndims, dim_number, dim_coord, ielement,
                          igrid)
    @inbounds begin
        imin = dim_coord.imin[ielement] - (ielement != 1)
        ngrid = dim_coord.ngrid

        derivative_indices .= indices
        derivative_indices[dim_number] = imin
        min_ind = get_column_index(jacobian, variable_index, derivative_indices)
        step = jacobian.state_vector_dim_steps[variable_index][dim_number]

        return min_ind:step:min_ind-1+step*ngrid
    end
end

# Get the global column indices corresponding to the entire dimension 'dim' that contains
# the point specified by `indices`.
function get_dimension_inds(jacobian::jacobian_info, indices, second_derivative_indices,
                            variable_index, variable_ndims, dim_number, dim_coord)
    @inbounds begin
        n = dim_coord.n

        second_derivative_indices .= indices
        second_derivative_indices[dim_number] = 1
        min_ind = get_column_index(jacobian, variable_index, second_derivative_indices)
        step = jacobian.state_vector_dim_steps[variable_index][dim_number]

        return min_ind:step:min_ind-1+step*n
    end
end

# Extract a slice from the derivative array to insert into the Jacobian matrix.
function get_derivative_matrix_slice(derivative_coord, derivative_spectral, ielement,
                                     igrid)
    @inbounds begin
        # Upwind derivative version.
        if derivative_coord.radau_first_element && ielement == 1
            Dmat = derivative_spectral.radau_Dmat
        else
            Dmat = derivative_spectral.lobatto_Dmat
        end

        return @view(Dmat[igrid, :]), derivative_coord.element_scale[ielement]
    end
end

# When we are using a distributed-memory domain decomposition, the grid points on the
# sub-domain boundaries are shared by two subdomains (or more at corners), and so are
# 'overlapping' as the same entry exists on multiple sub-domains. To deal with this
# overlap, we choose to require that when all the overlapping entries are added together,
# we get the true matrix entry. To achieve this, derivative terms are just added normally
# as the contributions from the elements on either side of the subdomain boundary (after
# accounting for upwinding) need to be added anyway. All other entries are 'diagonal'
# (i.e. non-zero only where the column index is equal to the row index) - any dimensions
# that are integrated over in integral terms are forbidden from being distributed over
# multiple subdomains, so do not need to be handled. For the diagonal terms, we add an
# equal contribution from every subdomain, so the contribution has to be reduced by the
# `overlap_factor` (i.e. 0.5 if the point is shared by two subdomains, 0.25 if it is
# shared by four, etc.).
function get_overlap_factor(indices::NTuple{N,mk_int}, coords::NTuple{N,mini_coordinate},
                            handle_overlaps::Val{true},
                            derivative_dim_number=nothing) where {N}
    overlap_factor = 1.0
    if derivative_dim_number === nothing
        for (i, c) ∈ zip(indices, coords)
            if i == 1 && (c.periodic || c.irank > 0)
                # Is part of the subdomain lower boundary in coordinate `c`.
                overlap_factor *= 0.5
            elseif i == c.n && (c.periodic || c.irank < c.nrank - 1)
                # Is part of the subdomain upper boundary in coordinate `c`.
                overlap_factor *= 0.5
            end
        end
    else
        for (idim, (i, c)) ∈ enumerate(zip(indices, coords))
            if idim != derivative_dim_number
                if i == 1 && (c.periodic || c.irank > 0)
                    # Is part of the subdomain lower boundary in coordinate `c`.
                    overlap_factor *= 0.5
                elseif i == c.n && (c.periodic || c.irank < c.nrank - 1)
                    # Is part of the subdomain upper boundary in coordinate `c`.
                    overlap_factor *= 0.5
                end
            end
        end
    end
    return overlap_factor
end

# Sometimes we want to construct/invert separate matrices on each subdomain (as an
# approximate preconditioner), in which case 'overlapping' entries should have their full
# value on every subdomain, because the contributions will never be added together.
function get_overlap_factor(indices::NTuple{N,mk_int}, coords::NTuple{N,mini_coordinate},
                            handle_overlaps::Val{false},
                            derivative_dim_number=nothing) where {N}
    return 1.0
end

"""
    add_term_to_Jacobian_row!(jacobian::jacobian_info,
                              jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
                              rows_variable::Symbol, prefactor::mk_float,
                              terms::EquationTerm, indices::Tuple,
                              boundary_speed, overlap_factor::mk_float,
                              derivative_overlap_factors::NamedTuple)

Traverse the tree of EquationTerms, accumulating the prefactors until reaching a leaf term
(`kind = ETsimple`) that represents a state vector variable, which makes a contribution to
the Jacobian.
"""
function add_term_to_Jacobian_row!(jacobian::jacobian_info,
                                   jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
                                   rows_variable::Symbol, prefactor::mk_float,
                                   terms::EquationTerm, indices::Tuple, boundary_speed,
                                   overlap_factor::mk_float,
                                   derivative_overlap_factors::NamedTuple)

    @inbounds begin
        if terms.is_constant || terms.kind === ETnull
            # Does not contribute to Jacobian.
        elseif terms.kind === ETproduct
            for (i, sub_term) ∈ enumerate(terms.sub_terms)
                if !sub_term.is_constant
                    other_terms = prod(t.current_value[]
                                       for (j,t) ∈ enumerate(terms.sub_terms) if j ≠ i)
                    add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                              prefactor*other_terms, sub_term, indices,
                                              boundary_speed, overlap_factor,
                                              derivative_overlap_factors)
                end
            end
        elseif terms.kind === ETsum
            for sub_term ∈ terms.sub_terms
                if !sub_term.is_constant
                    add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                              prefactor, sub_term, indices,
                                              boundary_speed, overlap_factor,
                                              derivative_overlap_factors)
                end
            end
        elseif terms.kind === ETpower
            @debug_consistency_checks length(terms.sub_terms) == 1 || error("'power' EquationTerm must have exactly one `sub_term` element")
            sub_term = terms.sub_terms[1]
            exponent = terms.exponent
            power_prefactor = prefactor * exponent * sub_term.current_value[]^(exponent - 1)
            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                      power_prefactor, sub_term, indices, boundary_speed,
                                      overlap_factor, derivative_overlap_factors)
        elseif terms.kind === ETfunction
            @debug_consistency_checks length(terms.sub_terms) == 1 || error("'function' EquationTerm must have exactly one `sub_term` element")
            sub_term = terms.sub_terms[1]
            func_derivative_prefactor = prefactor * func_derivative_lookup[terms.func_name](sub_term.current_value[])
            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                      func_derivative_prefactor, sub_term, indices,
                                      boundary_speed, overlap_factor,
                                      derivative_overlap_factors)
        elseif terms.kind === ETcompound
            @debug_consistency_checks length(terms.sub_terms) == 1 || error("'compound' EquationTerm must have exactly one `sub_term` element")
            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable, prefactor,
                                      terms.sub_terms[1], indices, boundary_speed,
                                      overlap_factor, derivative_overlap_factors)
        elseif length(terms.integrals) > 0
            # Derivative-of-integral terms are also handled by
            # add_integral_term_to_Jacobian_row!().
            add_integral_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                               prefactor, terms, indices, overlap_factor,
                                               derivative_overlap_factors)
        elseif length(terms.derivatives) > 0
            add_derivative_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                                 prefactor, terms, indices,
                                                 derivative_overlap_factors)
        elseif length(terms.second_derivatives) > 0
            add_second_derivative_term_to_Jacobian_row!(jacobian, jacobian_row,
                                                        rows_variable, prefactor, terms,
                                                        indices,
                                                        derivative_overlap_factors)
        else
            add_simple_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                             prefactor, terms, indices, overlap_factor)
        end

        return nothing
    end
end

# `term` is just one of the state vector variables, so add its prefactor to the
# corresponding column.
function add_simple_term_to_Jacobian_row!(jacobian::jacobian_info,
                                          jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
                                          rows_variable::Symbol, prefactor::mk_float,
                                          term::EquationTerm, indices::Tuple,
                                          overlap_factor::mk_float)

    @inbounds begin
        current_indices = term.current_indices
        for (i, j) ∈ enumerate(term.dimension_numbers)
            current_indices[i] = indices[j]
        end
        block_index, column_index = get_column_index(jacobian, term.state_variable, current_indices)

        jacobian_row[block_index][column_index] += overlap_factor * prefactor

        return nothing
    end
end

# `term` is a derivative of a state vector variable, so find the all the columns
# corresponding to points in the element in the derivative direction that contains the
# point corresponding to the row (or possibly the elements on either side if the point
# corresponding to the row is an element boundary, depending on upwinding), and insert
# `prefactor` times a column of the derivative matrix into those points.
function add_derivative_term_to_Jacobian_row!(jacobian::jacobian_info,
                                              jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
                                              rows_variable::Symbol,
                                              prefactor::mk_float, term::EquationTerm,
                                              indices::Union{Tuple,Vector},
                                              derivative_overlap_factors::NamedTuple)
    @inbounds begin
        @debug_consistency_checks length(term.derivatives) > 1 && error("More than one derivative not supported yet")
        current_indices = term.current_indices
        for (i, j) ∈ enumerate(term.dimension_numbers)
            current_indices[i] = indices[j]
        end
        term_ndims = length(term.dimensions)
        term_variable_ndims = length(term.dimensions)
        derivative_dim = term.derivatives[1]
        upwind_speed = term.upwind_speeds[1]
        derivative_coord = jacobian.coords[derivative_dim]
        derivative_spectral = jacobian.spectral[derivative_dim]
        term_dims = term.dimensions

        derivative_variable_index = jacobian.state_vector_numbers[term.state_variable]
        derivative_dim_number = term.derivative_dim_numbers[1]
        derivative_dim_index = current_indices[derivative_dim_number]
        ielement = derivative_coord.ielement[derivative_dim_index]
        igrid = derivative_coord.igrid[derivative_dim_index]

        prefactor *= derivative_overlap_factors[derivative_dim]

        jacobian_row_block = jacobian_row[derivative_variable_index]

        if length(term.upwind_speeds) > 0
            if length(term.upwind_speeds[1]) == 0
                upwind_speed = 0.0
            else
                upwind_speed = term.upwind_speeds[1][get_flattened_index(term.dimension_sizes, current_indices)]
            end
        else
            upwind_speed = 0.0
        end

        # Note `igrid == 1` only occurs in the first element (where `ielement == 1`), as
        # all the other element boundary points are included as the upper boundary of the
        # elements.
        if igrid == 1 && derivative_coord.nrank == 1 && derivative_coord.periodic && jacobian.handle_overlaps === Val(false)
            if upwind_speed == 0.0
                # Add derivative contribution from the 'previous' element on the
                # other side of the periodic boundary, index `nelement_global`.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord,
                                     derivative_coord.nelement_global,
                                     derivative_coord.ngrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                derivative_coord.nelement_global,
                                                derivative_coord.ngrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
                end

                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord,
                                     ielement, igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
                end
            elseif upwind_speed > 0.0
                # Add derivative contribution from the 'previous' element on the
                # other side of the periodic boundary, index `nelement_global`.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord,
                                     derivative_coord.nelement_global,
                                     derivative_coord.ngrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                derivative_coord.nelement_global,
                                                derivative_coord.ngrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
                end
            else # upwind_speed < 0.0
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord,
                                     ielement, igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
                end
            end
        elseif igrid == 1
            if (jacobian.handle_overlaps === Val(false)
                    || (derivative_coord.irank == 0 && !derivative_coord.periodic)
                    || upwind_speed < 0.0)
                # Is a non-periodic domain boundary, or a subdomain boundary where we are
                # neglecting coupling between sub-domains, or an upwind boundary where the
                # upwind speed is towards the boundary, so just use the one-sided
                # derivative.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord, ielement,
                                     igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
                end
            elseif upwind_speed == 0.0
                # One half of the 'centred' derivative.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord, ielement,
                                     igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
                end
            else # upwind_speed > 0.0
                # No contribution to add from this side.
            end
        elseif jacobian.handle_overlaps === Val(false) &&
                ielement == derivative_coord.nelement_local &&
                igrid == derivative_coord.ngrid && derivative_coord.periodic
            # In serial, periodic case, only contribution to the upper boundary rows is
            # the periodicity constraint, so no need to do anything here.
        elseif ielement == derivative_coord.nelement_local && igrid == derivative_coord.ngrid
            if (jacobian.handle_overlaps === Val(false)
                    || (derivative_coord.irank == derivative_coord.nrank - 1 && !derivative_coord.periodic)
                    || upwind_speed > 0.0)
                # Is a non-periodic domain boundary, or a subdomain boundary where we are
                # neglecting coupling between sub-domains, or an upwind boundary where the
                # upwind speed is towards the boundary, so just use the one-sided
                # derivative.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord, ielement,
                                     igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
                end
            elseif upwind_speed == 0.0
                # One half of the 'centred' derivative.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord, ielement,
                                     igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
                end
            else # upwind_speed < 0.0
                # No contribution to add from this side.
            end
        elseif igrid == derivative_coord.ngrid
            # Element boundary, not a block boundary
            if upwind_speed == 0.0
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord, ielement,
                                     igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
                end

                # Add derivative contribution from the 'next' element index
                # `ielement+1`.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord,
                                     ielement + 1, 1)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement + 1, 1)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
                end
            elseif upwind_speed > 0.0
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord, ielement,
                                     igrid)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement, igrid)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
                end
            else # upwind_speed < 0.0
                # Add derivative contribution from the 'next' element index
                # `ielement+1`.
                column_inds =
                    get_element_inds(jacobian, current_indices,
                                     term.current_derivative_indices,
                                     derivative_variable_index, term_variable_ndims,
                                     derivative_dim_number, derivative_coord,
                                     ielement + 1, 1)
                Dmat_slice, scale =
                    get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                                ielement + 1, 1)
                for (i, ci) ∈ enumerate(column_inds)
                    jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
                end
            end
        else
            # Point within an element
            column_inds =
                get_element_inds(jacobian, current_indices,
                                 term.current_derivative_indices,
                                 derivative_variable_index, term_variable_ndims,
                                 derivative_dim_number, derivative_coord, ielement, igrid)
            Dmat_slice, scale =
                get_derivative_matrix_slice(derivative_coord, derivative_spectral,
                                            ielement, igrid)
            for (i, ci) ∈ enumerate(column_inds)
                jacobian_row_block[ci] += prefactor * Dmat_slice[i] / scale
            end
        end

        return nothing
    end
end

# Similar to `add_derivative_term_to_Jacobian_row!()` but for a second derivative. At the
# moment we do not handle any mass matrices separately, so the derivative matrix is
# `M^-1 . K` where `M` is the mass matrix and `K` is the stiffness matrix for the second
# derivative. The `M^-1` makes this matrix dense, so it contributes to the whole
# derivative dimension, not just a single element - this is not optimal, and could be
# improved by special handling of the mass matrix (or matrices, e.g. for each velocity
# dimension) in future.
function add_second_derivative_term_to_Jacobian_row!(
             jacobian::jacobian_info,
             jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
             rows_variable::Symbol, prefactor::mk_float, term::EquationTerm,
             indices::Union{Tuple,Vector}, derivative_overlap_factors::NamedTuple)
    @inbounds begin
        @debug_consistency_checks length(term.derivatives) > 1 && error("More than one derivative not supported yet")
        current_indices = term.current_indices
        for (i, j) ∈ enumerate(term.dimension_numbers)
            current_indices[i] = indices[j]
        end
        term_variable_ndims = length(term.dimensions)
        derivative_dim = term.second_derivatives[1]
        derivative_coord = jacobian.coords[derivative_dim]
        derivative_spectral = jacobian.spectral[derivative_dim]

        derivative_variable_index = jacobian.state_vector_numbers[term.state_variable]
        derivative_dim_number = term.second_derivative_dim_numbers[1]
        derivative_dim_index = current_indices[derivative_dim_number]

        prefactor *= derivative_overlap_factors[derivative_dim]

        jacobian_row_block = jacobian_row[derivative_variable_index]

        # Point within an element
        column_inds =
            get_dimension_inds(jacobian, current_indices,
                               term.current_second_derivative_indices,
                               derivative_variable_index, term_variable_ndims,
                               derivative_dim_number, derivative_coord)
        Dmat_slice = @view derivative_spectral.dense_second_deriv_matrix[derivative_dim_index,:]
        for (i, ci) ∈ enumerate(column_inds)
            jacobian_row_block[ci] += prefactor * Dmat_slice[i]
        end

        return nothing
    end
end

# `term` represents an integral of a state vector variable (or possibly an integral of a
# derivative of a state vector variable). Loops over the columns corresponding to every
# point in the dimensions `term.integrals`.
function add_integral_term_to_Jacobian_row!(jacobian::jacobian_info,
                                            jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
                                            rows_variable::Symbol, prefactor::mk_float,
                                            term::EquationTerm, row_indices::Tuple,
                                            overlap_factor::mk_float,
                                            derivative_overlap_factors::NamedTuple)

    @inbounds begin
        integral_dim_sizes = Tuple(term.integral_dim_sizes)

        integrand_prefactor = term.integrand_prefactor

        for (i,j) ∈ zip(term.outer_dim_integrand_current_indices_slice, term.dimension_numbers)
            term.integrand_current_indices[i] = row_indices[j]
        end

        add_integral_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable, prefactor,
                                           term, row_indices, term.state_variable,
                                           integral_dim_sizes, integrand_prefactor,
                                           term.integral_wgts, overlap_factor,
                                           derivative_overlap_factors)

        return nothing
    end
end

# Separate internal function so that `integral_dim_sizes` can be a Tuple, allowing
# `CartesianIndices(integral_dim_sizes)` to be type stable as the length of
# `integral_dim_sizes` is known at compile time.
function add_integral_term_to_Jacobian_row!(jacobian::jacobian_info,
                                            jacobian_row::NTuple{N,<:AbstractVector{mk_float}} where N,
                                            rows_variable::Symbol, prefactor::mk_float,
                                            term::EquationTerm, row_indices::Tuple,
                                            integrand_variable, integral_dim_sizes,
                                            integrand_prefactor, wgts, overlap_factor,
                                            derivative_overlap_factors::NamedTuple)

    @inbounds begin
        current_indices = term.integrand_current_indices
        integral_dim_integrand_current_indices_slice = term.integral_dim_integrand_current_indices_slice

        integral_variable_index = jacobian.state_vector_numbers[term.state_variable]
        jacobian_row_block = jacobian_row[integral_variable_index]

        for indices ∈ CartesianIndices(integral_dim_sizes)
            for (i,j) ∈ enumerate(integral_dim_integrand_current_indices_slice)
                current_indices[j] = indices[i]
            end

            this_wgt = prod(w[ci] for (w, ci) ∈ zip(wgts, current_indices))
            set_current_values!(jacobian, integrand_prefactor, current_indices)

            if length(term.derivatives) == 0
                column_index = get_column_index(jacobian, integral_variable_index, current_indices)
                jacobian_row_block[column_index] += overlap_factor * prefactor *
                                                    integrand_prefactor.current_value[] *
                                                    this_wgt
            elseif length(term.derivatives) == 1
                # Can re-use `add_derivative_term_to_Jacobian_row!()` here, with
                # `current_indices` taking the place of the row index.
                add_derivative_term_to_Jacobian_row!(jacobian, jacobian_row, integrand_variable,
                                                     prefactor * integrand_prefactor.current_value[] * this_wgt,
                                                     term.integral_derivative_term,
                                                     current_indices,
                                                     derivative_overlap_factors)
            else
                error("Multiple derivatives not supported in integral-of-derivative term yet.")
            end
        end

        return nothing
    end
end

"""
    add_term_to_Jacobian!(jacobian::jacobian_info, rows_variable::Symbol,
                          prefactor::mk_float, terms::EquationTerm,
                          boundary_speed=nothing)

Add the contribution of `terms` to `jacobian`. The terms should be all or part of the
evolution equation for `rows_variable`.

`prefactor` multiplies all the terms. Usually it will be the timestep.

`boundary_speed` is the speed needed by `jacobian.boundary_skip_funcs` to determine
whether a grid point is set by the boundary conditions. This will usually be the speed in
the z-direction.
"""
@timeit global_timer add_term_to_Jacobian!(jacobian::jacobian_info, rows_variable::Symbol,
                                           prefactor::mk_float, terms::EquationTerm,
                                           boundary_speed=nothing) = begin
    if rows_variable ∉ jacobian.state_vector_entries
        # This term not being used at the moment.
        return nothing
    end

    @inbounds begin
        jacobian_matrix = jacobian.matrix
        rows_variable_number = jacobian.state_vector_numbers[rows_variable]
        @debug_consistency_checks jacobian.state_vector_is_constraint[rows_variable_number] && prefactor != 1.0 && error("Prefactor should be 1 for constraint equation, got $prefactor for $rows_variable equation")
        rows_variable_dims = jacobian.state_vector_dims[rows_variable_number]
        rows_variable_dim_sizes = jacobian.state_vector_dim_sizes[rows_variable_number]
        rows_variable_local_ranges = jacobian.state_vector_local_ranges[rows_variable_number]
        rows_variable_coords = jacobian.state_vector_coords[rows_variable_number]

        boundary_skip = jacobian.boundary_skip_funcs[rows_variable]

        set_dimension_numbers!(jacobian, rows_variable_number, terms)

        block_row = jacobian_matrix[rows_variable_number]

        add_term_to_Jacobian!(jacobian, rows_variable, prefactor, terms, boundary_speed,
                              block_row, rows_variable_number, rows_variable_dims,
                              rows_variable_dim_sizes, rows_variable_local_ranges,
                              rows_variable_coords, boundary_skip)
    end

    return nothing
end

# Internal version of the function, which ensures that rows_variable_local_ranges is a
# Tuple so that the `CartesianIndices` call is type-stable - a Tuple is required so that
# `CartesianIndices` knows at compile time how many indices there are. The lookup from
# `jacobian.state_vector_local_ranges` has to happen before this function is called to
# ensure type stability.
function add_term_to_Jacobian!(jacobian::jacobian_info, rows_variable::Symbol,
                               prefactor::mk_float, terms::EquationTerm, boundary_speed,
                               block_row::NTuple{N,AbstractMatrix{mk_float}} where N,
                               rows_variable_number::mk_int, rows_variable_dims::Tuple,
                               rows_variable_dim_sizes::Tuple,
                               rows_variable_local_ranges::Tuple,
                               rows_variable_coords::Tuple, boundary_skip::F) where {F}
    @inbounds begin
        for indices_CartesianIndex ∈ CartesianIndices(rows_variable_local_ranges)
            indices = Tuple(indices_CartesianIndex)
            if boundary_skip !== nothing && boundary_skip(boundary_speed,
                                                          jacobian.handle_overlaps,
                                                          indices...,
                                                          rows_variable_coords...)
                continue
            end

            overlap_factor = get_overlap_factor(indices, rows_variable_coords,
                                                jacobian.handle_overlaps)
            derivative_overlap_factors =
                NamedTuple(d => get_overlap_factor(indices, rows_variable_coords,
                                                   jacobian.handle_overlaps, idim)
                           for (idim, d) ∈ enumerate(rows_variable_dims))

            row_index = get_flattened_index(rows_variable_dim_sizes, indices)
            jacobian_row = Tuple(@view block[row_index,:] for block ∈ block_row)

            set_current_values!(jacobian, terms, indices)

            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable, prefactor, terms,
                                      indices, boundary_speed, overlap_factor,
                                      derivative_overlap_factors)
        end

        id_hash = @debug_block_synchronize_quick_ifelse(
                       hash(string(@__FILE__, @__LINE__)),
                       nothing
                      )
        jacobian.synchronize(id_hash)
        return nothing
    end
end

"""
    add_periodicity_constraint_to_jacobian!(jacobian::jacobian_info)

When using a preconditioner solver that does not have support for overlapping rows/columns
(e.g. the serial LU solver), and one or more dimensions is periodic, we need to add a
constraint to force the repeated points (that are perioidic copies of each other) to be
equal. Apart from this function that adds the constraint, the other contributions to the
Jacobian matrix will be identical for the rows that correspond to these repeated points.
We can therefore leave the row corresponding to the 'lower' point unmodified, and add the
constraint equation to the row corresponding to the 'upper' point, while leaving both RHS
vector values unchanged. The solution is equivalent to the one that is given by the matrix
where the 'lower' row is subtracted from the 'upper' row to leave just <constraint>=0, but
this way minimises the number of places in the code where the `handle_overlaps=Val(false)`
but periodic case needs to be handled specially.
"""
function add_periodicity_constraint_to_jacobian!(jacobian::jacobian_info)
    if jacobian.handle_overlaps === Val(true)
        # No need to add extra constraint, as perioidicity is already handled as part of
        # the 'overlaps'.
        return nothing
    end

    @timeit global_timer "add_periodicity_constraint_to_jacobian!" begin
        @inbounds begin
            jacobian_matrix = jacobian.matrix
            for rows_variable ∈ jacobian.state_vector_entries
                rows_variable_number = jacobian.state_vector_numbers[rows_variable]
                if jacobian.state_vector_is_constraint[rows_variable_number]
                    # No need to add extra constraint for constraint rows.
                    continue
                end
                rows_variable_dims = jacobian.state_vector_dims[rows_variable_number]
                rows_variable_dim_sizes = jacobian.state_vector_dim_sizes[rows_variable_number]
                rows_variable_local_ranges = jacobian.state_vector_local_ranges[rows_variable_number]
                rows_variable_coords = jacobian.state_vector_coords[rows_variable_number]

                # Only need diagonal block.
                block = jacobian_matrix[rows_variable_number][rows_variable_number]

                add_periodicity_constraint_to_jacobian!(jacobian, block,
                                                        rows_variable_number,
                                                        rows_variable_dims,
                                                        rows_variable_dim_sizes,
                                                        rows_variable_local_ranges,
                                                        rows_variable_coords)
            end
        end
    end

    return nothing
end

# Internal version of the function, which ensures that rows_variable_local_ranges is a
# Tuple so that the `CartesianIndices` call is type-stable - a Tuple is required so that
# `CartesianIndices` knows at compile time how many indices there are. The lookup from
# `jacobian.state_vector_local_ranges` has to happen before this function is called to
# ensure type stability.
function add_periodicity_constraint_to_jacobian!(jacobian::jacobian_info,
                                                 block::AbstractMatrix{mk_float},
                                                 rows_variable_number::mk_int,
                                                 rows_variable_dims::Tuple,
                                                 rows_variable_dim_sizes::Tuple,
                                                 rows_variable_local_ranges::Tuple,
                                                 rows_variable_coords::Tuple)
    @inbounds begin
        for (ic, coord) ∈ enumerate(rows_variable_coords)
            if !coord.periodic
                continue
            end
            if coord.nrank > 1
                error("cannot handle periodicity like this with MPI-distributed "
                      * "coordinate.")
            end
            n = coord.n
            for indices_CartesianIndex ∈ CartesianIndices(rows_variable_local_ranges)
                indices = Tuple(indices_CartesianIndex)

                if indices[ic] == n
                    this_row_index = get_flattened_index(rows_variable_dim_sizes, indices)
                    lower_row_indices = ntuple(d -> d==ic ? 1 : indices[d], length(indices))
                    lower_row_index = get_flattened_index(rows_variable_dim_sizes,
                                                          lower_row_indices)
                    # Add periodicity constraint constraint:
                    #    x_upper - xlower = 0
                    #    x[coord_index=n] - x[coord_index=1] = 0
                    # Before this function is called, the corresponding matrix rows are
                    # the identity.
                    block[this_row_index,lower_row_index] -= 1.0
                end
            end
        end

        id_hash = @debug_block_synchronize_quick_ifelse(
                       hash(string(@__FILE__, @__LINE__)),
                       nothing
                      )
        jacobian.synchronize(id_hash)
        return nothing
    end
end

end # jacobian_matrices
