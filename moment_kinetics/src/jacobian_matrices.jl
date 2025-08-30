"""
Generic utilities to simplify calculation of (contributions to) Jacobian matrices
"""
module jacobian_matrices

export jacobian_info, create_jacobian_info, jacobian_initialize_identity!,
       jacobian_initialize_zero!, jacobian_initialize_bc_diagonal!, EquationTerm,
       ConstantTerm, CompoundTerm, NullTerm, add_term_to_Jacobian!

using ..array_allocation: allocate_shared_float, allocate_float
using ..communication
using ..debugging
using ..looping
using ..moment_kinetics_structs: coordinate, discretization_info
using ..timer_utils
using ..type_definitions

using MPI

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

const JacobianMatrixType = @debug_shared_array_ifelse(Union{MPISharedArray{mk_float,2},Matrix{mk_float}},Matrix{mk_float})

"""
Jacobian matrix and some associated information.

`matrix` is a (non-sparse) array containing the Jacobian matrix.

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

`state_vector_offsets` gives the offset from indices in the corresponding variable to
indices in the full state vector. The first entry of `state_vector_offsets` is 0.
`state_vector_offsets` has a length one greater than `state_vector_entries` - the final
element is the total size of the state vector. 

`state_vector_local_ranges` gives the index range in each dimension of `state_vector_dims`
that is locally owned by this process - used for parallelised loops over rows (or
columns) when dealing with a single variable at a time from the state vector.

`row_local_ranges` gives an index range of locally owned rows of `matrix` - used for
parallelised loops over rows (or columns) of the full `matrix` when it is not necessary to
know which variable a row corresponds to.

`coords` is a NamedTuple containing `mini_coordinate` objects for all the dimensions
involved in the `jacobian_info`.

`spectral` is a NamedTuple containing `mini_discretization_info` objects for all the
dimensions involved in the `jacobian_info`.

`boundary_skip_funcs` is a NamedTuple that gives functions (for each variable in
`state_vector_entries`) that indicate when a grid point should be skipped in the Jacobian
because it is set by boundary conditions.

`synchronize` is the function to be called to synchronize all processes in the
shared-memory group that is working with this `jacobian_info` object.
"""
struct jacobian_info{NTerms,NOffsets,Tvecinds,Tvecdims,Tveccoords,Tdimsizes,Tdimsteps,Tranges,Tcoords,Tspectral,Tbskip,Tsync}
    matrix::JacobianMatrixType
    state_vector_entries::NTuple{NTerms, Symbol}
    state_vector_numbers::Tvecinds
    state_vector_dims::Tvecdims
    state_vector_coords::Tveccoords
    state_vector_sizes::NTuple{NTerms, mk_int}
    state_vector_dim_sizes::Tdimsizes
    state_vector_dim_steps::Tdimsteps
    state_vector_offsets::NTuple{NOffsets, mk_int}
    state_vector_local_ranges::Tranges
    row_local_ranges::UnitRange{mk_int}
    coords::Tcoords
    spectral::Tspectral
    boundary_skip_funcs::Tbskip
    synchronize::Tsync
end

"""
    create_jacobian_info(coords::NamedTuple, spectral::NamedTuple; comm=comm_block[],
                         synchronize::Union{Function,Nothing}=_block_synchronize,
                         boundary_skip_funcs=nothing, kwargs...)

Create a [`jacobian_info`](@ref) struct.

`kwargs` describes the state vector. The keys are the variable names, and the arguments
are 2 element Tuples whose first element is the 'region type' for parallel loops over that
variable (or `nothing` if `comm = nothing` for serial operation) and whose second element
Vector or Tuple of Symbols giving the dimensions of the variable.

`coords` is a NamedTuple giving all the needed coordinates, `spectral` is a NamedTuple
with all the needed 'spectral' objects.

`comm` is the communicator to use to allocate shared memory arrays. `synchronize` is the
function used to synchronize between shared-memory operations.

`boundary_skip_funcs` is a NamedTuple whose keys are the variable names and whose values
are functions to use to skip points that would be set by boundary conditions (or `nothing`
if no function is needed for the variable).
"""
function create_jacobian_info(coords::NamedTuple, spectral::NamedTuple; comm=comm_block[],
                              synchronize::Union{Function,Nothing}=_block_synchronize,
                              boundary_skip_funcs=nothing, kwargs...)

    @debug_consistency_checks all(all(d ∈ keys(coords) for d ∈ v[2]) for v ∈ values(kwargs)) || error("Some coordinate required by the state variables were not included in `coords`.")

    mini_coords = (; (k=>mini_coordinate(v) for (k,v) ∈ pairs(coords))...)
    mini_spectral = (; (k=>mini_discretization_info(v) for (k,v) ∈ pairs(spectral))...)

    state_vector_entries = Tuple(keys(kwargs))
    state_vector_numbers = (; (name => i for (i,name) ∈ enumerate(state_vector_entries))...)
    state_vector_region_types = Tuple(v[1] for v ∈ values(kwargs))
    state_vector_dims = Tuple(Tuple(v[2]) for v ∈ values(kwargs))
    state_vector_coords = Tuple(Tuple(mini_coords[d] for d ∈ v) for v ∈ state_vector_dims)
    state_vector_sizes = Tuple(prod(mini_coords[d].n for d ∈ dims; init=1) for dims ∈ state_vector_dims)
    state_vector_dim_sizes = Tuple(Tuple(coords[d].n for d ∈ v) for v ∈ state_vector_dims)
    state_vector_dim_steps = Tuple(Tuple(prod(s[1:i-1]; init=mk_int(1)) for i ∈ 1:length(s))
                                   for s ∈ state_vector_dim_sizes)
    state_vector_offsets = Tuple(cumsum([0, state_vector_sizes...]))

    # For each variable, get the corresponding 'region type' from
    # `state_vector_region_types`. Use that to get the corresponding `LoopRanges` object,
    # and from that get the local index ranges for each dimension in the variable.
    # If the 'region type' is `nothing`, do a serial set up so the 'local' range includes
    # every point in the dimension.
    state_vector_local_ranges = Tuple(rt === nothing ? Tuple(1:mini_coords[d].n for d ∈ v) :
                                      Tuple(getfield(looping.loop_ranges_store[rt], d) for d ∈ v)
                                      for (rt,v) ∈ zip(state_vector_region_types,state_vector_dims))

    n = state_vector_offsets[end] # `jacobian_matrix` is n x n
    if comm === nothing
        row_local_ranges = 1:n
    else
        # Ranges for shared-memory-parallel loop over flattened row indices.
        # For consistency with choices elsewhere the total set of points is split into groups
        # of size [m,m,...,m,(m+1),...,(m+1),(m+1)]
        n_blocks = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
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
        row_local_ranges = startind:stopind
    end

    if boundary_skip_funcs === nothing
        boundary_skip_funcs = (; (name=>nothing for name ∈ state_vector_entries)...)
    else
        if any(v ∉ keys(boundary_skip_funcs) for v ∈ state_vector_entries)
            error("When passing boundary_skip_funcs, must have an entry for every state "
                  * "vector variable ($state_vector_entries). Only got "
                  * "$(keys(boundary_skip_funcs)).")
        end
    end

    if comm === nothing
        # No shared memory needed
        jacobian_matrix = allocate_float(:jacobian_size=>n, :jacobian_size=>n)
    else
        jacobian_matrix = allocate_shared_float(:jacobian_size=>n, :jacobian_size=>n;
                                                comm=comm)
    end

    if synchronize === nothing
        if comm !== nothing
            error("`synchronize` argument is required when `comm !== nothing`.")
        end
        # Pass a dummy function that does nothing
        synchronize = (call_site)->nothing
    end

    return jacobian_info(jacobian_matrix, state_vector_entries, state_vector_numbers,
                         state_vector_dims, state_vector_coords, state_vector_sizes,
                         state_vector_dim_sizes, state_vector_dim_steps,
                         state_vector_offsets, state_vector_local_ranges,
                         row_local_ranges, mini_coords, mini_spectral,
                         boundary_skip_funcs, synchronize)
end

"""
    jacobian_initialize_identity!(jacobian::jacobian_info)

Initialize `jacobian.matrix` with the identity.
"""
function jacobian_initialize_identity!(jacobian::jacobian_info)
    jacobian_matrix = jacobian.matrix
    ncols = size(jacobian_matrix, 1)
    for col ∈ jacobian.row_local_ranges
        jacobian_matrix[:,col] .= 0.0
        jacobian_matrix[col,col] = 1.0
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
    jacobian_initialize_identity!(jacobian::jacobian_info)

Initialize `jacobian.matrix` to zero.
"""
function jacobian_initialize_zero!(jacobian::jacobian_info)
    jacobian_matrix = jacobian.matrix
    ncols = size(jacobian_matrix, 1)
    for col ∈ jacobian.row_local_ranges
        jacobian_matrix[:,col] .= 0.0
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
function jacobian_initialize_bc_diagonal!(jacobian::jacobian_info, boundary_speed)
    jacobian_matrix = jacobian.matrix
    for col_variable ∈ 1:length(jacobian.state_vector_entries)
        col_variable_local_ranges = jacobian.state_vector_local_ranges[col_variable]
        col_variable_coords = jacobian.state_vector_coords[col_variable]
        offset = jacobian.state_vector_offsets[col_variable]
        boundary_skip = jacobian.boundary_skip_funcs[col_variable]
        jacobian_initialize_bc_digonal_single_variable!(
            jacobian_matrix, col_variable_local_ranges, col_variable_coords, offset,
            boundary_skip, boundary_speed)
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
             jacobian_matrix, col_variable_local_ranges, col_variable_coords, offset,
             boundary_skip, boundary_speed)
    for (i, indices_CartesianIndex) ∈ enumerate(CartesianIndices(col_variable_local_ranges))
        indices = Tuple(indices_CartesianIndex)
        col = i + offset
        jacobian_matrix[:,col] .= 0.0
        if boundary_skip !== nothing && boundary_skip(boundary_speed, indices...,
                                                      rows_variable_coords...)
            jacobian_matrix[col,col] .= 1.0
        end
    end
    return nothing
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
    # Size of the dimensions for each entry of `upwind_speeds`. Note that upwind speed
    # arrays have an unusual dimension order - the dimension corresponding to the
    # derivative is the left-most dimension, followed by the remaining dimensions (in the
    # usual order), so this is not equivalent to `dimension_sizes`.
    speed_dim_sizes::Vector{Vector{mk_int}}
    # The current loop indices corresponding to the dimensions of an upwind speed.
    current_speed_indices::Vector{mk_int}
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
                        Vector{mk_float}[], Vector{mk_int}[],
                        fill(mk_int(0), length(dimensions)), Symbol[], mk_int[], mk_int[],
                        Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
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
                        Vector{mk_float}[], Vector{mk_int}[], mk_int[], Symbol[],
                        mk_int[], mk_int[], Symbol[], nothing, mk_int[], mk_int[],
                        mk_int[], nothing, Vector{mk_float}[], mk_int[],
                        fill(mk_float(NaN), 1), mk_int[], fill(mk_float(NaN)))
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
            upwind_coords = [dims_coords[du], (c for (d,c) ∈ pairs(dims_coords) if d ≠ du)...]
            if u !== nothing && size(u) != Tuple(c.n for c ∈ upwind_coords)
                error("Size of upwind speed $i $(size(u)) does not match coordinates $(Tuple(c.n for c ∈ upwind_coords))")
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
        speed_dim_sizes = [zeros(mk_int, 0) for _ ∈ derivatives]
        upwind_speeds = [zeros(mk_float, 0) for _ ∈ derivatives]
    else
        @debug_consistency_checks length(derivatives) == length(upwind_speeds) || error("`upwind_speeds` was passed, but length is not the same as length of `derivatives`.")

        speed_dim_sizes = Vector{mk_int}[[size(u)...] for u ∈ upwind_speeds]
        upwind_speeds = [u === nothing ? zeros(mk_float, 0) : get_flattened_array(u)
                         for u ∈ upwind_speeds]
    end

    return EquationTerm(EquationTerm[], ETsimple, false, state_variable, dimensions,
                        mk_int[], dimension_sizes, 1, Symbol(""), derivatives,
                        derivative_dim_numbers, current_derivative_indices,
                        upwind_speeds, speed_dim_sizes,
                        fill(mk_int(0), length(dimensions)), second_derivatives,
                        second_derivative_dim_numbers, current_second_derivative_indices,
                        integrals, integrand_prefactor, mk_int[], mk_int[], mk_int[],
                        derivative_term, integral_wgts, integral_dim_sizes,
                        get_flattened_array(array), fill(mk_int(0), length(dimensions)),
                        fill(mk_float(NaN)))
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
                        mk_int[], Vector{mk_float}[], Vector{mk_int}[],
                        fill(mk_int(0), length(dimensions)), Symbol[], mk_int[], mk_int[],
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
                            Vector{mk_float}[], Vector{mk_int}[], mk_int[], Symbol[],
                            mk_int[], mk_int[], Symbol[], nothing, mk_int[], mk_int[],
                            mk_int[], nothing, Vector{mk_float}[], mk_int[], mk_float[],
                            mk_int[], fill(mk_float(NaN)))
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
                            Vector{mk_float}[], Vector{mk_int}[], mk_int[], Symbol[],
                            mk_int[], mk_int[], Symbol[], nothing, mk_int[], mk_int[],
                            mk_int[], nothing, Vector{mk_float}[], mk_int[], mk_float[],
                            mk_int[], fill(mk_float(NaN)))
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
                        Vector{mk_float}[], Vector{mk_int}[], mk_int[], Symbol[],
                        mk_int[], mk_int[], Symbol[], nothing, mk_int[], mk_int[],
                        mk_int[], nothing, Vector{mk_float}[], mk_int[], mk_float[],
                        mk_int[], fill(mk_float(NaN)))
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
                            Vector{mk_int}[], mk_int[], Symbol[], mk_int[], mk_int[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    elseif x.kind === ETproduct
        return EquationTerm(vcat(x.sub_terms, y), ETproduct, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Vector{mk_int}[],
                            mk_int[], Symbol[], mk_int[], mk_int[], Symbol[], nothing,
                            mk_int[], mk_int[], mk_int[], nothing, Vector{mk_float}[],
                            mk_int[], mk_float[], mk_int[], fill(mk_float(NaN)))
    elseif y.kind === ETproduct
        return EquationTerm(vcat(x, y.sub_terms), ETproduct, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Vector{mk_int}[],
                            mk_int[], Symbol[], mk_int[], mk_int[], Symbol[], nothing,
                            mk_int[], mk_int[], mk_int[], nothing, Vector{mk_float}[],
                            mk_int[], mk_float[], mk_int[], fill(mk_float(NaN)))
    else
        return EquationTerm([x, y], ETproduct, is_constant, Symbol(""), Symbol[],
                            mk_int[], mk_int[], 1, Symbol(""), Symbol[], mk_int[],
                            mk_int[], Vector{mk_float}[], Vector{mk_int}[], mk_int[],
                            Symbol[], mk_int[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
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
                            Vector{mk_int}[], mk_int[], Symbol[], mk_int[], mk_int[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    else
        return EquationTerm([x_term, y], ETproduct, y.is_constant, Symbol(""), Symbol[],
                            mk_int[], mk_int[], 1, Symbol(""), Symbol[], mk_int[],
                            mk_int[], Vector{mk_float}[], Vector{mk_int}[], mk_int[],
                            Symbol[], mk_int[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
    end
end
Base.:*(x::EquationTerm, y::Number) = Base.:*(y, x)

function Base.:+(x::EquationTerm, y::EquationTerm)
    is_constant = x.is_constant && y.is_constant
    if x.kind === ETsum && y.kind === ETsum
        return EquationTerm(vcat(x.sub_terms, y.sub_terms), ETsum, is_constant,
                            Symbol(""), Symbol[], mk_int[], mk_int[], 1, Symbol(""),
                            Symbol[], mk_int[], mk_int[], Vector{mk_float}[],
                            Vector{mk_int}[], mk_int[], Symbol[], mk_int[], mk_int[],
                            Symbol[], nothing, mk_int[], mk_int[], mk_int[], nothing,
                            Vector{mk_float}[], mk_int[], mk_float[], mk_int[],
                            fill(mk_float(NaN)))
    elseif x.kind === ETnull
        # Drop 'null' terms from sums.
        return y
    elseif y.kind === ETnull
        # Drop 'null' terms from sums.
        return x
    elseif x.kind === ETsum
        return EquationTerm(vcat(x.sub_terms, y), ETsum, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Vector{mk_int}[],
                            mk_int[], Symbol[], mk_int[], mk_int[], Symbol[], nothing,
                            mk_int[], mk_int[], mk_int[], nothing, Vector{mk_float}[],
                            mk_int[], mk_float[], mk_int[], fill(mk_float(NaN)))
    elseif y.kind === ETsum
        return EquationTerm(vcat(x, y.sub_terms), ETsum, is_constant, Symbol(""),
                            Symbol[], mk_int[], mk_int[], 1, Symbol(""), Symbol[],
                            mk_int[], mk_int[], Vector{mk_float}[], Vector{mk_int}[],
                            mk_int[], Symbol[], mk_int[], mk_int[], Symbol[], nothing,
                            mk_int[], mk_int[], mk_int[], nothing, Vector{mk_float}[],
                            mk_int[], mk_float[], mk_int[], fill(mk_float(NaN)))
    else
        return EquationTerm([x, y], ETsum, is_constant, Symbol(""), Symbol[], mk_int[],
                            mk_int[], 1, Symbol(""), Symbol[], mk_int[], mk_int[],
                            Symbol[], mk_int[], mk_int[], Vector{mk_float}[],
                            Vector{mk_int}[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
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
                            mk_int[], mk_int[], Vector{mk_float}[], Vector{mk_int}[],
                            mk_int[], Symbol[], mk_int[], mk_int[], Symbol[], nothing,
                            mk_int[], mk_int[], mk_int[], nothing, Vector{mk_float}[],
                            mk_int[], mk_float[], mk_int[], fill(mk_float(NaN)))
    else
        return EquationTerm([x_term, y], ETsum, y.is_constant, Symbol(""), Symbol[],
                            mk_int[], mk_int[], 1, Symbol(""), Symbol[], mk_int[],
                            mk_int[], Vector{mk_float}[], Vector{mk_int}[], mk_int[],
                            Symbol[], mk_int[], mk_int[], Symbol[], nothing, mk_int[],
                            mk_int[], mk_int[], nothing, Vector{mk_float}[], mk_int[],
                            mk_float[], mk_int[], fill(mk_float(NaN)))
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
        return get_column_index(jacobian, variable_index, indices)
    end
end
@inline function get_column_index(jacobian::jacobian_info, variable_index, indices)
    return @inbounds get_flattened_index(jacobian.state_vector_dim_sizes[variable_index], indices) + jacobian.state_vector_offsets[variable_index]
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

# Traverse the tree of EquationTerms, accumulating the prefactors until reaching a leaf
# term (`kind = ETsimple`) that represents a state vector variable, which makes a
# contribution to the Jacobian.
function add_term_to_Jacobian_row!(jacobian::jacobian_info,
                                   jacobian_row::AbstractVector{mk_float},
                                   rows_variable::Symbol, prefactor::mk_float,
                                   terms::EquationTerm, indices::Tuple,
                                   boundary_speed)

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
                                              boundary_speed)
                end
            end
        elseif terms.kind === ETsum
            for sub_term ∈ terms.sub_terms
                if !sub_term.is_constant
                    add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                              prefactor, sub_term, indices, boundary_speed)
                end
            end
        elseif terms.kind === ETpower
            @debug_consistency_checks length(terms.sub_terms) == 1 || error("'power' EquationTerm must have exactly one `sub_term` element")
            sub_term = terms.sub_terms[1]
            exponent = terms.exponent
            power_prefactor = prefactor * exponent * sub_term.current_value[]^(exponent - 1)
            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                      power_prefactor, sub_term, indices, boundary_speed)
        elseif terms.kind === ETfunction
            @debug_consistency_checks length(terms.sub_terms) == 1 || error("'function' EquationTerm must have exactly one `sub_term` element")
            sub_term = terms.sub_terms[1]
            func_derivative_prefactor = prefactor * func_derivative_lookup[terms.func_name](sub_term.current_value[])
            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                      func_derivative_prefactor, sub_term, indices,
                                      boundary_speed)
        elseif terms.kind === ETcompound
            @debug_consistency_checks length(terms.sub_terms) == 1 || error("'compound' EquationTerm must have exactly one `sub_term` element")
            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable, prefactor,
                                      terms.sub_terms[1], indices, boundary_speed)
        elseif length(terms.integrals) > 0
            # Derivative-of-integral terms are also handled by
            # add_integral_term_to_Jacobian_row!().
            add_integral_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                               prefactor, terms, indices)
        elseif length(terms.derivatives) > 0
            add_derivative_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                                 prefactor, terms, indices)
        elseif length(terms.second_derivatives) > 0
            add_second_derivative_term_to_Jacobian_row!(jacobian, jacobian_row,
                                                        rows_variable, prefactor, terms,
                                                        indices)
        else
            add_simple_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable,
                                             prefactor, terms, indices)
        end

        return nothing
    end
end

# `term` is just one of the state vector variables, so add its prefactor to the
# corresponding column.
function add_simple_term_to_Jacobian_row!(jacobian::jacobian_info,
                                          jacobian_row::AbstractVector{mk_float},
                                          rows_variable::Symbol, prefactor::mk_float,
                                          term::EquationTerm, indices::Tuple)

    @inbounds begin
        current_indices = term.current_indices
        for (i, j) ∈ enumerate(term.dimension_numbers)
            current_indices[i] = indices[j]
        end
        column_index = get_column_index(jacobian, term.state_variable, current_indices)

        jacobian_row[column_index] += prefactor

        return nothing
    end
end

# `term` is a derivative of a state vector variable, so find the all the columns
# corresponding to points in the element in the derivative direction that contains the
# point corresponding to the row (or possibly the elements on either side if the point
# corresponding to the row is an element boundary, depending on upwinding), and insert
# `prefactor` times a column of the derivative matrix into those points.
function add_derivative_term_to_Jacobian_row!(jacobian::jacobian_info,
                                              jacobian_row::AbstractVector{mk_float},
                                              rows_variable::Symbol, prefactor::mk_float,
                                              term::EquationTerm, indices::Union{Tuple,Vector})
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

        if length(term.upwind_speeds) > 0
            if length(term.upwind_speeds[1]) == 0
                upwind_speed = 0.0
            else
                current_speed_indices = term.current_speed_indices
                current_speed_indices[1] = derivative_dim_index
                counter = 2
                for i ∈ 1:term_ndims
                    if i ≠ derivative_dim_number
                        current_speed_indices[counter] = current_indices[i]
                        counter += 1
                    end
                end
                derivative_variable_sizes = jacobian.state_vector_dim_sizes[derivative_variable_index]
                upwind_speed = term.upwind_speeds[1][get_flattened_index(term.speed_dim_sizes[1], current_speed_indices)]
            end
        else
            upwind_speed = 0.0
        end

        if igrid == 1 && derivative_coord.periodic
            if derivative_coord.nrank > 1
                error("Distributed MPI not supported in Jacobian matrix construction "
                      * "for periodic dimension yet.")
            else
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
                        jacobian_row[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
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
                        jacobian_row[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
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
                        jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
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
                        jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
                    end
                end
            end
        elseif igrid == 1
            if derivative_coord.irank == 0
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
                    jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
                end
            else
                # For now, we do not allow the Jacobian matrix to couple different
                # shared-memory blocks, so just use the one-sided derivative
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
                    jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
                end
            end
        elseif ielement == derivative_coord.nelement_local && igrid == derivative_coord.ngrid
            if derivative_coord.irank == derivative_coord.nrank - 1
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
                    jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
                end
            else
                # For now, we do not allow the Jacobian matrix to couple different
                # shared-memory blocks, so just use the one-sided derivative
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
                    jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
                end
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
                    jacobian_row[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
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
                    jacobian_row[ci] += 0.5 * prefactor * Dmat_slice[i] / scale
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
                    jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
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
                    jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
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
                jacobian_row[ci] += prefactor * Dmat_slice[i] / scale
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
             jacobian::jacobian_info, jacobian_row::AbstractVector{mk_float},
             rows_variable::Symbol, prefactor::mk_float, term::EquationTerm,
             indices::Union{Tuple,Vector})
    @inbounds begin
        @debug_consistency_checks length(term.derivatives) > 1 && error("More than one derivative not supported yet")
        current_indices = term.current_indices
        for (i, j) ∈ enumerate(term.dimension_numbers)
            current_indices[i] = indices[j]
        end
        term_ndims = length(term.dimensions)
        term_variable_ndims = length(term.dimensions)
        derivative_dim = term.second_derivatives[1]
        derivative_coord = jacobian.coords[derivative_dim]
        derivative_spectral = jacobian.spectral[derivative_dim]
        term_dims = term.dimensions

        derivative_variable_index = jacobian.state_vector_numbers[term.state_variable]
        derivative_dim_number = term.second_derivative_dim_numbers[1]
        derivative_dim_index = current_indices[derivative_dim_number]

        # Point within an element
        column_inds =
            get_dimension_inds(jacobian, current_indices,
                               term.current_second_derivative_indices,
                               derivative_variable_index, term_variable_ndims,
                               derivative_dim_number, derivative_coord)
        Dmat_slice = @view derivative_spectral.dense_second_deriv_matrix[derivative_dim_index,:]
        for (i, ci) ∈ enumerate(column_inds)
            jacobian_row[ci] += prefactor * Dmat_slice[i]
        end

        return nothing
    end
end

# `term` represents an integral of a state vector variable (or possibly an integral of a
# derivative of a state vector variable). Loops over the columns corresponding to every
# point in the dimensions `term.integrals`.
function add_integral_term_to_Jacobian_row!(jacobian::jacobian_info,
                                            jacobian_row::AbstractVector{mk_float},
                                            rows_variable::Symbol, prefactor::mk_float,
                                            term::EquationTerm, row_indices::Tuple)

    @inbounds begin
        integral_dim_sizes = Tuple(term.integral_dim_sizes)

        integrand_prefactor = term.integrand_prefactor

        for (i,j) ∈ zip(term.outer_dim_integrand_current_indices_slice, term.dimension_numbers)
            term.integrand_current_indices[i] = row_indices[j]
        end

        add_integral_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable, prefactor,
                                           term, row_indices, term.state_variable,
                                           integral_dim_sizes, integrand_prefactor,
                                           term.integral_wgts)

        return nothing
    end
end

# Separate internal function so that `integral_dim_sizes` can be a Tuple, allowing
# `CartesianIndices(integral_dim_sizes)` to be type stable as the length of
# `integral_dim_sizes` is known at compile time.
function add_integral_term_to_Jacobian_row!(jacobian::jacobian_info,
                                            jacobian_row::AbstractVector{mk_float},
                                            rows_variable::Symbol, prefactor::mk_float,
                                            term::EquationTerm, row_indices::Tuple,
                                            integrand_variable, integral_dim_sizes,
                                            integrand_prefactor, wgts)

    @inbounds begin
        current_indices = term.integrand_current_indices
        integral_dim_integrand_current_indices_slice = term.integral_dim_integrand_current_indices_slice

        for indices ∈ CartesianIndices(integral_dim_sizes)
            for (i,j) ∈ enumerate(integral_dim_integrand_current_indices_slice)
                current_indices[j] = indices[i]
            end

            this_wgt = prod(w[ci] for (w, ci) ∈ zip(wgts, current_indices))
            set_current_values!(jacobian, integrand_prefactor, current_indices)

            if length(term.derivatives) == 0
                column_index = get_column_index(jacobian, term.state_variable, current_indices)
                jacobian_row[column_index] += prefactor * integrand_prefactor.current_value[] * this_wgt
            elseif length(term.derivatives) == 1
                # Can re-use `add_derivative_term_to_Jacobian_row!()` here, with
                # `current_indices` taking the place of the row index.
                add_derivative_term_to_Jacobian_row!(jacobian, jacobian_row, integrand_variable,
                                                     prefactor * integrand_prefactor.current_value[] * this_wgt,
                                                     term.integral_derivative_term,
                                                     current_indices)
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
    @inbounds begin
        jacobian_matrix = jacobian.matrix
        rows_variable_number = jacobian.state_vector_numbers[rows_variable]
        rows_variable_dims = jacobian.state_vector_dims[rows_variable_number]
        rows_variable_dim_sizes = jacobian.state_vector_dim_sizes[rows_variable_number]
        rows_variable_local_ranges = jacobian.state_vector_local_ranges[rows_variable_number]
        rows_variable_coords = jacobian.state_vector_coords[rows_variable_number]
        row_offset = jacobian.state_vector_offsets[rows_variable_number]

        boundary_skip = jacobian.boundary_skip_funcs[rows_variable]

        set_dimension_numbers!(jacobian, rows_variable_number, terms)

        add_term_to_Jacobian!(jacobian, rows_variable, prefactor, terms, boundary_speed,
                              jacobian_matrix, rows_variable_number, rows_variable_dims,
                              rows_variable_dim_sizes, rows_variable_local_ranges,
                              rows_variable_coords, row_offset, boundary_skip)
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
                               jacobian_matrix::AbstractMatrix{mk_float},
                               rows_variable_number::mk_int, rows_variable_dims::Tuple,
                               rows_variable_dim_sizes::Tuple,
                               rows_variable_local_ranges::Tuple,
                               rows_variable_coords::Tuple, row_offset::mk_int,
                               boundary_skip)
    @inbounds begin
        for indices_CartesianIndex ∈ CartesianIndices(rows_variable_local_ranges)
            indices = Tuple(indices_CartesianIndex)
            if boundary_skip !== nothing && boundary_skip(boundary_speed, indices...,
                                                          rows_variable_coords...)
                continue
            end

            row_index = get_flattened_index(rows_variable_dim_sizes, indices) + row_offset
            jacobian_row = @view jacobian_matrix[row_index,:]

            set_current_values!(jacobian, terms, indices)

            add_term_to_Jacobian_row!(jacobian, jacobian_row, rows_variable, prefactor, terms,
                                      indices, boundary_speed)
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
