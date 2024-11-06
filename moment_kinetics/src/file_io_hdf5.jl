# No separate module defined here as this file is included within the file_io module

using HDF5
using MPI

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

# HDF5.H5DataStore is the supertype for HDF5.File and HDF5.Group
function modify_attribute!(file_or_group_or_var::Union{HDF5.H5DataStore,HDF5.Dataset}, name, value)
    attrs(file_or_group_or_var)[name] = value
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

    if N == 1
        io_var[global_ranges[1]] = @view data[local_ranges[1]]
    elseif N == 2
        io_var[global_ranges[1], global_ranges[2]] =
            @view data[local_ranges[1], local_ranges[2]]
    elseif N == 3
        io_var[global_ranges[1], global_ranges[2], global_ranges[3]] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3]]
    elseif N == 4
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4]] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4]]
    elseif N == 5
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5]] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5]]
    elseif N == 6
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], global_ranges[6]] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5], local_ranges[6]]
    elseif N == 7
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], global_ranges[6], global_ranges[6], global_ranges[7]] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5], local_ranges[6], local_ranges[7]]
    elseif N == 8
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], global_ranges[6], global_ranges[7], global_ranges[8]] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5], local_ranges[6], local_ranges[7], local_ranges[8]]
    else
        error("data of dimension $N not supported yet")
    end

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
    if parallel_io
        chunk_sizes = Tuple(isa(c, mk_int) ? c : max(c.n-1,1) for c in coords)
    else
        chunk_sizes = dim_sizes
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
                               data::Union{Nothing,Number,AbstractArray{T,N}}, t_idx,
                               parallel_io::Bool,
                               coords::Union{coordinate,Integer}...;
                               only_root=false, write_from_this_rank=nothing) where {T,N}
    # Extend time dimension for this variable
    dims = size(io_var)
    dims_mod = (dims[1:end-1]..., t_idx)
    HDF5.set_extent_dims(io_var, dims_mod)
    local_ranges = Tuple(isa(c, coordinate) ? c.local_io_range : 1:c for c ∈ coords)
    global_ranges = Tuple(isa(c, coordinate) ? c.global_io_range : 1:c for c ∈ coords)

    if only_root && parallel_io && global_rank[] != 0
        # Variable should only be written from root, and this process is not root for the
        # output file
        return nothing
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
    elseif N == 1
        io_var[global_ranges[1], t_idx] = @view data[local_ranges[1]]
    elseif N == 2
        io_var[global_ranges[1], global_ranges[2], t_idx] =
            @view data[local_ranges[1], local_ranges[2]]
    elseif N == 3
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], t_idx] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3]]
    elseif N == 4
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               t_idx] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4]]
    elseif N == 5
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], t_idx] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5]]
    elseif N == 6
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], global_ranges[6], t_idx] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5], local_ranges[6]]
    elseif N == 7
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], global_ranges[6], global_ranges[6], global_ranges[7],
               t_idx] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5], local_ranges[6], local_ranges[7]]
    elseif N == 8
        io_var[global_ranges[1], global_ranges[2], global_ranges[3], global_ranges[4],
               global_ranges[5], global_ranges[6], global_ranges[7], global_ranges[8],
               t_idx] =
            @view data[local_ranges[1], local_ranges[2], local_ranges[3], local_ranges[4],
                       local_ranges[5], local_ranges[6], local_ranges[7], local_ranges[8]]
    else
        error("data of dimension $N not supported yet")
    end

    return nothing
end
