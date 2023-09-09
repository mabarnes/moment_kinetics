# No separate module defined here as this file is included within the file_io module

using NCDatasets

function io_has_parallel(::Val{netcdf})
    # NCDatasets.jl does not support parallel I/O yet
    return false
end

function open_output_file_netcdf(prefix, parallel_io, io_comm, mode="c")
    parallel_io && error("NetCDF interface does not support parallel I/O")

    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix, ".cdf")
    # if a netcdf file with the requested name already exists, remove it
    mode == "c" && isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename, mode)

    return fid, (filename, parallel_io, io_comm)
end

function create_io_group(parent::NCDataset, name; description=nothing)
    if description !== nothing
        attributes = Dict("description" => description)
    else
        attributes = ()
    end

    return defGroup(parent, name, attrib=attributes)
end

function get_group(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        return file_or_group.group[name]
    catch
        println("An error occured while opening the $name group")
        rethrow()
    end
end

function is_group(file_or_group::NCDataset, name::String)
    println("check groups ", NCDatasets.groupnames(file_or_group))
    return name ∈ NCDatasets.groupnames(file_or_group)
end

function get_subgroup_keys(file_or_group::NCDataset)
    return NCDatasets.groupnames(file_or_group)
end

function get_variable_keys(file_or_group::NCDataset)
    return keys(file_or_group)
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
function maybe_create_netcdf_dim(file_or_group::NCDataset, coord)
    return maybe_create_netcdf_dim(file_or_group, coord.name, coord.n)
end

function write_single_value!(file_or_group::NCDataset, name,
                             value::Union{Number, AbstractString, AbstractArray{T,N}},
                             coords...; parallel_io, n_ion_species=nothing,
                             n_neutral_species=nothing, description=nothing,
                             units=nothing) where {T,N}
    if description !== nothing || units !== nothing
        attributes = Dict{String, Any}()
        if description !== nothing
            attributes["description"] = description
        end
        if units !== nothing
            attributes["units"] = units
        end
    else
        attributes = ()
    end

    if n_ion_species !== nothing && n_neutral_species != nothing
        error("Cannot have both ion-species and neutral species dimensions." *
              "Got n_ion_species=$n_ion_species, n_neutral_species=$n_neutral_species")
    end

    if isa(value, Number) || isa(value, AbstractString)
        coords !== () && error("cannot pass coordinates with a scalar")

        if isa(value, AbstractString)
            # Trying to write a SubString{String} causes an error, so force anything
            # string-like to be a String
            value = String(value)
        end

        type = typeof(value)
        dims = ()
    else
        type = T
        for c ∈ coords
            maybe_create_netcdf_dim(file_or_group, c)
        end
        dims = Tuple(c.name for c in coords)

        if n_ion_species !== nothing
            if n_ion_species < 0
                error("n_ion_species must be non-negative, got $n_ion_species")
            elseif n_ion_species == 0
                # No data to write
                return nothing
            end
            maybe_create_netcdf_dim(file_or_group, "ion_species", n_ion_species)
            dims = tuple(dims..., "ion_species")
        elseif n_neutral_species !== nothing
            if n_neutral_species < 0
                error("n_neutral_species must be non-negative, got $n_neutral_species")
            elseif n_neutral_species == 0
                # No data to write
                return nothing
            end
            maybe_create_netcdf_dim(file_or_group, "neutral_species", n_neutral_species)
            dims = tuple(dims..., "neutral_species")
        end
    end
    if isa(value, Bool)
        # As a hack, write bools to NetCDF as Char, as NetCDF does not support bools (?),
        # and we do not use Char for anything else
        var = defVar(file_or_group, name, Char, dims, attrib=attributes)
        var[:] = Char(value)
    else
        var = defVar(file_or_group, name, type, dims, attrib=attributes)
        var[:] = value
    end

    return nothing
end

function create_dynamic_variable!(file_or_group::NCDataset, name, type,
                                  coords::coordinate...; parallel_io,
                                  n_ion_species=nothing, n_neutral_species=nothing,
                                  description=nothing, units=nothing)

    if n_ion_species !== nothing && n_neutral_species !== nothing
        error("Variable should not contain both ion and neutral species dimensions. "
              * "Got n_ion_species=$n_ion_species and "
              * "n_neutral_species=$n_neutral_species")
    end

    # Create time dimension if necessary
    maybe_create_netcdf_dim(file_or_group, "time", Inf)

    # Create species dimension if necessary
    if n_ion_species !== nothing
        if n_ion_species < 0
            error("n_ion_species must be non-negative, got $n_ion_species")
        elseif n_ion_species == 0
            # No data to write
            return nothing
        end
        maybe_create_netcdf_dim(file_or_group, "ion_species", n_ion_species)
    end
    if n_neutral_species !== nothing
        if n_neutral_species < 0
            error("n_neutral_species must be non-negative, got $n_neutral_species")
        elseif n_neutral_species == 0
            # No data to write
            return nothing
        end
        maybe_create_netcdf_dim(file_or_group, "neutral_species", n_neutral_species)
    end

    # Create other dimensions if necessary
    for c ∈ coords
        maybe_create_netcdf_dim(file_or_group, c)
    end

    # create the variable so it can be expanded indefinitely (up to the largest unsigned
    # integer in size) in the time dimension
    coord_dims = Tuple(c.name for c ∈ coords)
    if n_ion_species !== nothing
        fixed_dims = tuple(coord_dims..., "ion_species")
    elseif n_neutral_species !== nothing
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
                               data::Union{Number,AbstractArray{T,N}}, t_idx,
                               coords...) where {T,N}

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
