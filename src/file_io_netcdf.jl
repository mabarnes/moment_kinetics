# No separate module defined here as this file is included within the file_io module

using NCDatasets

function io_has_parallel(::Val{netcdf})
    # NCDatasets.jl does not support parallel I/O yet
    return false
end

function open_output_file_netcdf(prefix)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix, ".cdf")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename,"c")

    return fid
end

function create_io_group(parent::NCDataset, name; description=nothing)
    if description !== nothing
        attributes = Dict("description" => description)
    else
        attributes = ()
    end

    return defGroup(parent, name, attrib=attributes)
end

function add_attribute!(file_or_group::NCDataset, name, value)
    file_or_group.attrib[name] = value
end
function add_attribute!(var::NCDatasets.CFVariable, name, value)
    var.attrib[name] = value
end

function maybe_create_netcdf_dim(file_or_group::NCDataset, name, size)
    if !(name ∈ keys(file_or_group.dim))
        defDim(file_or_group, name, size)
    end
    return nothing
end
function maybe_create_netcdf_dim(file_or_group::NCDataset, coord::coordinate)
    return maybe_create_netcdf_dim(file_or_group, coord.name, coord.n)
end

function write_single_value!(file_or_group::NCDataset, name,
                             value::Union{Number, AbstractArray{T,N}},
                             coords::coordinate...; description=nothing) where {T,N}
    if description !== nothing
        attributes = Dict("description" => description)
    else
        attributes = ()
    end

    if isa(value, Number)
        coords !== () && error("cannot pass coordinates with a scalar")
        type = typeof(value)
        dims = ()
    else
        type = T
        for c ∈ coords
            maybe_create_netcdf_dim(file_or_group, c)
        end
        dims = Tuple(c.name for c in coords)
    end
    var = defVar(file_or_group, name, type, dims, attrib=attributes)
    var[:] = value

    return nothing
end

function create_dynamic_variable!(file_or_group::NCDataset, name, type,
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

    # Create time dimension if necessary
    maybe_create_netcdf_dim(file_or_group, "time", Inf)

    # Create species dimension if necessary
    if n_ion_species > 0
        maybe_create_netcdf_dim(file_or_group, "ion_species", n_ion_species)
    end
    if n_neutral_species > 0
        maybe_create_netcdf_dim(file_or_group, "neutral_species", n_neutral_species)
    end

    # Create other dimensions if necessary
    for c ∈ coords
        maybe_create_netcdf_dim(file_or_group, c)
    end

    # create the variable so it can be expanded indefinitely (up to the largest unsigned
    # integer in size) in the time dimension
    coord_dims = Tuple(c.name for c ∈ coords)
    if n_ion_species > 0
        fixed_dims = tuple(coord_dims..., "ion_species")
    elseif n_neutral_species > 0
        fixed_dims = tuple(coord_dims..., "neutral_species")
    else
        fixed_dims = coord_dims
    end
    dims = tuple(fixed_dims..., "time")

    # create the variable so it can be expanded indefinitely (up to the largest unsigned
    # integer in size) in the time dimension
    var = defVar(file_or_group, name, type, dims)

    if description !== nothing
        add_attribute!(var, "description", description)
    end
    if units !== nothing
        add_attribute!(var, "units", units)
    end

    return var
end

function append_to_dynamic_var(io_var::NCDatasets.CFVariable,
                               data::Union{Number,AbstractArray{T,N}}, t_idx) where {T,N}

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
