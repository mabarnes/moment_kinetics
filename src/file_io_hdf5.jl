# No separate module defined here as this file is included within the file_io module

using HDF5

function open_output_file_hdf5(prefix)
    # the hdf5 file will be given by output_dir/run_name with .h5 appended
    filename = string(prefix, ".h5")
    # if a file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new HDF5 file
    fid = h5open(filename,"cw")

    return fid
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
function write_single_value!(file_or_group::HDF5.H5DataStore, name, value,
                             coords::coordinate...; description=nothing)
    file_or_group[name] = value

    if description !== nothing
        add_attribute!(file_or_group[name], "description", description)
    end

    return nothing
end

"""
given a tuple, reduced_dims, containing all dimensions except the time dimension,
return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
and the dims tuple which also contains the max size of the dataset, accounting for multiple
time slices
"""
function hdf5_dynamic_dims(reduced_dims)
    # chunk_dims is a tuple indicating the data chunk size to be written each step
    chunk_dims = tuple(reduced_dims..., 1)
    # dims contains the initial allocated data size in chunk_dims and the maximum
    # data size in the second argument; the -1 indicates that the time index is
    # effectively unlimited (as large as the largest unsigned integer value).
    # the time index will be dynamically extended as more data is written to file
    dims = (chunk_dims, tuple(reduced_dims..., -1))

    return chunk_dims, dims
end

function create_dynamic_variable!(file_or_group::HDF5.H5DataStore, name, type,
                                  coords::coordinate...;
                                  n_ion_species=0, n_neutral_species=0,
                                  description=nothing, units=nothing)

    if n_ion_species != 0 && n_neutral_species != 0
        error("Variable should not contain both ion and neutral species dimensions. "
              * "Got n_ion_species=$n_ion_species and "
              * "n_neutral_species=$n_neutral_species")
    end
    n_ion_species < 0 && error("n_ion_species must be non-negative, got $n_ion_species")
    n_neutral_species < 0 && error("n_neutral_species must be non-negative, got $n_neutral_species")

    # create the variable so it can be expanded indefinitely (up to the largest unsigned
    # integer in size) in the time dimension
    coord_dims = Tuple(c.n for c ∈ coords)
    if n_ion_species > 0
        fixed_dims = tuple(coord_dims..., n_ion_species)
    elseif n_neutral_species > 0
        fixed_dims = tuple(coord_dims..., n_neutral_species)
    else
        fixed_dims = coord_dims
    end
    chunk, dims = hdf5_dynamic_dims(fixed_dims)
    var = create_dataset(file_or_group, name, type, dims, chunk=chunk)

    # Add attribute listing the dimensions belonging to this variable
    dim_names = Tuple(c.name for c ∈ coords)
    if n_ion_species > 0
        dim_names = tuple(dim_names..., "ion_species")
    elseif n_neutral_species > 0
        dim_names = tuple(dim_names..., "neutral_species")
    end
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
                               data::Union{Number,AbstractArray{T,N}}, t_idx) where {T,N}
    # Extend time dimension for this variable
    dims = size(io_var)
    dims_mod = (dims[1:end-1]..., t_idx)
    HDF5.set_extent_dims(io_var, dims_mod)

    if isa(data, Number)
        io_var[t_idx] = data
    elseif N == 1
        io_var[:,t_idx] = data
    elseif N == 2
        io_var[:,:,t_idx] = data
    elseif N == 3
        io_var[:,:,:,t_idx] = data
    elseif N == 4
        io_var[:,:,:,:,t_idx] = data
    elseif N == 5
        io_var[:,:,:,:,:,t_idx] = data
    elseif N == 6
        io_var[:,:,:,:,:,:,t_idx] = data
    elseif N == 7
        io_var[:,:,:,:,:,:,:,t_idx] = data
    elseif N == 8
        io_var[:,:,:,:,:,:,:,:,t_idx] = data
    else
        error("data of dimension $N not supported yet")
    end

    return nothing
end
