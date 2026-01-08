# No separate module defined here as this file is included within the file_io module

using HDF5
using MPI

# To create unlimited-length arrays of strings, pick a fixed width to pad a Vector{UInt8}
# version of the Strings to, as HDF5.jl does not support arrays of variable-length strings
# when using parallel I/O (as of version 0.17.2).
const string_array_size = 256

function io_has_implementation(::Val{hdf5})
    return true
end

function io_has_parallel(::Val{hdf5})
    return HDF5.has_parallel()
end

function open_output_file_implementation(::Val{hdf5}, prefix, io_input, io_comm, mode="cw")
    # the hdf5 file will be given by output_dir/run_name with .h5 appended
    filename = string(prefix, ".h5")

    # JTO thought the maximum filename length should be 255 according to HDF5.jl, but can
    # no longer find the HDF5.jl check for length. However on one test a filename of
    # length 245 caused a crash, while 243 was OK, so limit to 243.
    if length(filename) > 243
        error("Length of filename '$filename' is too long ($(length(filename)) "
              * "characters), which will cause an error in HDF5.")
    end
    # create the new HDF5 file
    if io_input.parallel_io
        # if a file with the requested name already exists, remove it
        if mode == "cw" && MPI.Comm_rank(io_comm) == 0 && isfile(filename)
            rm(filename)
        end
        MPI.Barrier(io_comm)

        fid = h5open(filename, mode, io_comm)
    else
        # if a file with the requested name already exists, remove it
        mode == "cw" && isfile(filename) && rm(filename)

        # Not doing parallel I/O, so do not need to pass communicator
        fid = h5open(filename, mode)
    end

    return fid, (filename, io_input, io_comm)
end

# HDF5.H5DataStore is the supertype for HDF5.File and HDF5.Group
function create_io_group(parent::HDF5.H5DataStore, name; description=nothing)
    group = create_group(parent, name)

    if description !== nothing
        add_attribute!(group, "description", description)
    end

    return group
end

# HDF5.H5DataStore is the supertype for HDF5.File and HDF5.Group
function add_attribute!(file_or_group::HDF5.H5DataStore, name, value)
    attributes(file_or_group)[name] = value
end
function add_attribute!(var::HDF5.Dataset, name, value)
    attributes(var)[name] = value
end

function get_variable(file_or_group::HDF5.H5DataStore, name::String)
    return file_or_group[name]
end

function get_group(file_or_group::HDF5.H5DataStore, name::String)
    # This overload deals with cases where fid is an HDF5 `File` or `Group` (`H5DataStore`
    # is the abstract super-type for both
    try
        return file_or_group[name]
    catch
        println("An error occured while opening the $name group")
        rethrow()
    end
end

function is_group(file_or_group::HDF5.H5DataStore, name::String)
    return isa(file_or_group[name], HDF5.H5DataStore)
end

function get_subgroup_keys(file_or_group::HDF5.H5DataStore)
    return collect(k for k ∈ keys(file_or_group) if is_group(file_or_group, k))
end

function get_variable_keys(file_or_group::HDF5.H5DataStore)
    return collect(k for k ∈ keys(file_or_group) if !is_group(file_or_group, k))
end

# HDF5.H5DataStore is the supertype for HDF5.File and HDF5.Group
function write_single_value!(file_or_group::HDF5.H5DataStore, name,
                             data::Union{Number, AbstractString, AbstractArray{T,N}},
                             coords::Union{coordinate,mk_int,NamedTuple}...; parallel_io,
                             description=nothing, units=nothing,
                             overwrite=false) where {T,N}
    if isa(data, Union{Number, AbstractString})
        # When we write a scalar, and parallel_io=true, we need to create the variable on
        # every process in `comm_inter_block[]` but we only want to actually write the
        # data from one process (we choose `global_rank[]==0`) to avoid corruption due to
        # the same data being written at the same time from different processes (HDF5
        # might protect against this, but it must be at best inefficient).
        # HDF5.jl's `create_dataset()` for a scalar `data` returns both the 'I/O variable'
        # (the handle to the HDF5 'Dataset' for the variable) and the data type. For
        # String data, the 'data type' is important, because it actually also contains the
        # length of the string, so cannot be easily created by hand. Note that a String of
        # the correct length must be passed from every process in `comm_inter_block[]`,
        # but only the contents of the string on `global_rank[]==0` are actually written.
        if !(overwrite && name ∈ keys(file_or_group))
            create_dataset(file_or_group, name, data)
            if description !== nothing
                add_attribute!(file_or_group[name], "description", description)
            end
            if units !== nothing
                add_attribute!(file_or_group[name], "units", units)
            end
        end
        if !parallel_io || global_rank[] == 0
            write(file_or_group[name], data)
        end
        return nothing
    end

    if any(isa(c, mk_int) ? c < 0 : c.n < 0 for c ∈ coords)
        error("Got a negative `n` in $coords")
    end
    if any(isa(c, mk_int) ? c == 0 : c.n == 0 for c ∈ coords)
        # No data to write
        return nothing
    end

    dim_sizes, chunk_sizes = hdf5_get_fixed_dim_sizes(coords, parallel_io)
    if overwrite && name ∈ keys(file_or_group)
        io_var = file_or_group(name)
    else
        io_var = create_dataset(file_or_group, name, T, dim_sizes, chunk=chunk_sizes)
    end
    local_ranges = Tuple(isa(c, mk_int) ? (1:c) : isa(c, coordinate) ? c.local_io_range : c.n for c ∈ coords)
    global_ranges = Tuple(isa(c, mk_int) ? (1:c) : isa(c, coordinate) ? c.global_io_range : c.n for c ∈ coords)

    io_var[global_ranges...] = @view data[local_ranges...]

    if description !== nothing
        add_attribute!(file_or_group[name], "description", description)
    end

    return nothing
end

"""
Get sizes of fixed dimensions and chunks (i.e. everything but time) for I/O

`coords` should be a tuple whose elements are coordinate structs or integers (e.g. number
of species).
"""
function hdf5_get_fixed_dim_sizes(coords, parallel_io)
    if parallel_io
        dim_sizes = Tuple(isa(c, mk_int) ? c : (isa(c, coordinate) ? c.n_global : c.n) for c in coords)
    else
        dim_sizes = Tuple(isa(c, mk_int) ? c : c.n for c in coords)
    end
    function get_chunk_sizes(block_sizes)
        # ARCHER2 docs suggest that on that system the best size for a block of I/O is
        # 1MiB (2^20 bytes) (https://docs.archer2.ac.uk/user-guide/io/), which corresponds
        # to 2^20/8=2^17 double precision floats.
        target_exponent = 17
        target_size = 2^target_exponent

        if prod(block_sizes) > 8 * target_size
            # block size is large, so break into 'many' smaller chunks. Some chunks
            # probably overlap (see comment in 'else'), but as there are many chunks, the
            # chances of 2 procesess trying to write the same chunk at the same time is
            # small, so no need to worry about this.
            # When we read data, we often want to read only a slice, not the full array.
            # When HDF5 loads data, it loads a chunk at a time, then takes the slice from
            # that chunk, so the optimal thing for reading is if the dimension(s) being
            # sliced is split into small chunks. However, we might want to slice any
            # dimension - there is not an obvious way to pick 'special' dimensions.
            # Therefore try to make the chunk size in each dimension the same, in such a
            # way that the product of all the chunk sizes is `target_size`. As 17 is
            # prime, unless nd==1 we will have to have two different chunk sizes
            # (differing by a factor of 2) to make the product exactly `target_size`, but
            # we also need to make sure that the chunk size is never bigger than the
            # corresponding block size, as that would be inefficient.
            nd = length(block_sizes)
            is_max_size = fill(false, length(block_sizes))
            for (i,s) ∈ enumerate(block_sizes)
                if s == 1
                    # Dimension is size one, so ignore it when chunking.
                    nd -= 1
                    is_max_size[i] = true
                end
            end
            dim_chunk_exponent = target_exponent ÷ nd
            # Work out how many chunks should use dim_chunk_exponent, and how many should
            # use (dim_chunk_exponent+1). Assume we want to slice spatial dimensions
            # slightly more often, while loading full velocity dimensions, so first
            # dimensions should use (dim_chunk_exponent+1) and last dimensions should use
            # dim_chunk_exponent.
            # m*(dim_chunk_exponent + 1) + (nd - m)*dim_chunk_exponent = target_exponent
            # m*dim_chunk_exponent + m + nd*dim_chunk_exponent - m*dim_chunk_exponent = target_exponent
            # m + nd*dim_chunk_exponent = target_exponent
            m = target_exponent - nd * dim_chunk_exponent
            first_guess_sizes_partial = [(i ≤ m ? 2^(dim_chunk_exponent+1) : 2^dim_chunk_exponent) for i ∈ 1:nd]
            # sizes_full includes the length==1 dimensions.
            counter = 1
            sizes_full = mk_int[]
            for s ∈ block_sizes
                if s == 1
                    push!(sizes_full, 1)
                else
                    push!(sizes_full, first_guess_sizes_partial[counter])
                    counter += 1
                end
            end

            # Ensure that no chunk size is greater than a block size.
            extra_two_factors = 0
            for i ∈ eachindex(sizes_full)
                while sizes_full[i] > block_sizes[i]
                    sizes_full[i] = sizes_full[i] ÷ 2
                    extra_two_factors += 1
                end
            end

            # Now need to put the extra factors back into larger block_sizes.
            # The following should always terminate, but just in case make sure the while
            # loop does not continue infinitely.
            max_iterations = extra_two_factors
            for _ ∈ 1:max_iterations
                if extra_two_factors == 0
                    break
                end
                # Iterate in reverse as the last chunk sizes are possibly bigger than the
                # first chunk sizes.
                for i ∈ reverse(eachindex(sizes_full))
                    if extra_two_factors > 0 && block_sizes[i] ≥ 2 * sizes_full[i]
                        sizes_full[i] *= 2
                        extra_two_factors -= 1
                    end
                end
            end

            chunk_sizes = ntuple(i->sizes_full[i], length(block_sizes))
        else
            # block size is smaller than target_size, or not much larger, so if we split
            # the block into chunks there would be few chunks, and so possibly a high
            # chance of overlap/contention between different ranks - as the block is
            # probably not exactly divisible (generically our dimensions are not nice
            # powers of 2 or similar, due to the odd final element boundary point) into
            # chunks, some chunks would overlap block boundaries and need to be written to
            # by multiple processes. To avoid this, just write a single chunk per block.
            chunk_sizes = block_sizes
        end
        return chunk_sizes
    end
    if parallel_io
        block_sizes = Tuple(isa(c, mk_int) ? c : # species index, not a coordinate, never distributed
                            c.nrank == 1 ? c.n : # coordinate is not distributed, so no overlap with other writing MPI processes
                            max(c.n-1,1) for c in coords) # coordinate is distributed, this block size would mean no overlap with other processes
        chunk_sizes = get_chunk_sizes(block_sizes)
    else
        chunk_sizes = get_chunk_sizes(dim_sizes)
    end

    return dim_sizes, chunk_sizes
end

"""
given a tuple, fixed_coords, containing all dimensions except the time dimension,
get the dimension sizes and chunk sizes
"""
function hdf5_get_dynamic_dim_sizes(fixed_coords, parallel_io)
    fixed_dim_sizes, fixed_chunk_sizes =
        hdf5_get_fixed_dim_sizes(fixed_coords, parallel_io)

    initial_dim_sizes = tuple(fixed_dim_sizes..., 1)

    # 'maximum size' of -1 for time dimension indicates that the time dimension has an
    # unlimited length (it can be extended)
    max_dim_sizes = tuple(fixed_dim_sizes..., -1)

    chunk_size = tuple(fixed_chunk_sizes..., 1)

    return initial_dim_sizes, max_dim_sizes, chunk_size
end

function create_dynamic_variable!(file_or_group::HDF5.H5DataStore, name, type,
                                  coords::Union{coordinate,NamedTuple}...; parallel_io,
                                  description=nothing, units=nothing)

    if type === String
        # To create unlimited-length arrays of strings, use a fixed-width array of UInt8
        # as HDF5.jl does not support arrays of variable-length strings when using
        # parallel I/O (as of version 0.17.2).
        var = create_dataset(file_or_group, name, UInt8,
                             ((string_array_size, 1), (string_array_size, -1));
                             chunk=(string_array_size, 1))
        add_attribute!(var, "dims", "string")
        return var
    end

    if any(isa(c, mk_int) ? c < 0 : c.n < 0 for c ∈ coords)
        error("Got a negative `n` in $coords")
    end
    if any(isa(c, mk_int) ? c == 0 : c.n == 0 for c ∈ coords)
        # No data to write
        return nothing
    end

    initial_dim_sizes, max_dim_sizes, chunk_size =
        hdf5_get_dynamic_dim_sizes(coords, parallel_io)
    var = create_dataset(file_or_group, name, type, (initial_dim_sizes, max_dim_sizes),
                         chunk=chunk_size)

    # Add attribute listing the dimensions belonging to this variable
    dim_names = Tuple(c.name for c ∈ coords)
    add_attribute!(var, "dims", join(dim_names, ","))

    if description !== nothing
        add_attribute!(var, "description", description)
    end
    if units !== nothing
        add_attribute!(var, "units", units)
    end

    return var
end

function extend_time_index!(h5, t_idx)
    for var in h5.fid["dynamic_data"]
        dims = size(var)
        nd = ndims(var)
        dims_mod = (first(dims,nd-1)..., t_idx)
        HDF5.set_extent_dims(var, dims_mod)
    end
    return nothing
end

function append_to_dynamic_var(io_var::HDF5.Dataset,
                               data::Union{Nothing,Number,String,AbstractArray{T,N}}, t_idx,
                               parallel_io::Bool,
                               coords::Union{coordinate,NamedTuple,Integer}...;
                               only_root=false, write_from_this_rank=nothing) where {T,N}
    # Extend time dimension for this variable
    dims = size(io_var)
    dims_mod = (dims[1:end-1]..., t_idx)
    HDF5.set_extent_dims(io_var, dims_mod)
    local_ranges = Tuple(isa(c, Integer) ? (1:c) : c.local_io_range for c ∈ coords)
    global_ranges = Tuple(isa(c, Integer) ? (1:c) : c.global_io_range for c ∈ coords)

    if only_root === false
        # Continue
    elseif only_root === true
        if parallel_io && global_rank[] != 0
            # Variable should only be written from root, and this process is not root for the
            # output file
            return nothing
        end
    elseif MPI.Comm_rank(only_root) == 0
        # Continue
    elseif MPI.Comm_rank(only_root) != 0
        # Workaround - if we want the behaviour of `only_root = true`, but on a
        # sub-communicator rather than the global communicator, pass the communicator as
        # `only_root` and then stop here if this process is not the root of that
        # communicator.
        return nothing
    else
        error("Unexpected type '$(typeof(only_root))' for `only_root`.")
    end
    if write_from_this_rank === false
        # The variable is only written from another rank, not this one.
        return nothing
    end

    if isa(data, Number)
        if !parallel_io || write_from_this_rank === true ||
                (write_from_this_rank === nothing && global_rank[] == 0)
            # A scalar value is required to be the same on all processes, so when using
            # parallel I/O, only write from one process to avoid overwriting (which would
            # mean processes having to wait, and make which process wrote the final value
            # random). An exception is if `write_from_this_rank=true` was passed - this
            # should only be passed on one process.
            io_var[t_idx] = data
        end
    elseif isa(data, String)
        if !parallel_io || write_from_this_rank === true ||
                (write_from_this_rank === nothing && global_rank[] == 0)
            # A scalar value is required to be the same on all processes, so when using
            # parallel I/O, only write from one process to avoid overwriting (which would
            # mean processes having to wait, and make which process wrote the final value
            # random). An exception is if `write_from_this_rank=true` was passed - this
            # should only be passed on one process.
            #
            # HDF5.jl does not currently (v0.17.2) support writing variable-length Strings
            # when using parallel I/O, so instead we convert to fixed-length UInt8 arrays,
            # and pad to the fixed size.
            temparray = zeros(UInt8, string_array_size)
            data_vec = Vector{UInt8}(data)
            if length(data_vec) > string_array_size
                data_vec = data_vec[1:string_array_size]
            end
            temparray[1:length(data_vec)] .= data_vec
            io_var[:,t_idx] = temparray
        end
    else
        io_var[global_ranges..., t_idx] = @view data[local_ranges...]
    end

    return nothing
end
