# This extension provides an interface to the optional file_io_netcdf package, which
# provides NetCDF I/O
#
# Note that if there are errors when precompiling an extension, they may not be shown by
# default. To see the error, precompile by running
# `using Pkg; Pkg.precompile(strict=true)`.
module file_io_netcdf

import moment_kinetics.file_io: io_has_implementation, io_has_parallel,
                                open_output_file_implementation, create_io_group,
                                get_io_variable, get_group, is_group, get_subgroup_keys,
                                get_variable_keys, add_attribute!, write_single_value!,
                                create_dynamic_variable!, append_to_dynamic_var
import moment_kinetics.load_data: open_file_to_read, get_attribute, has_attribute,
                                  load_variable, load_slice
using moment_kinetics.coordinates: coordinate
using moment_kinetics.input_structs: netcdf

using NCDatasets

function io_has_implementation(::Val{netcdf})
    return true
end

function io_has_parallel(::Val{netcdf})
    # NCDatasets.jl does not support parallel I/O yet
    return false
end

function open_output_file_implementation(::Val{netcdf}, prefix, io_input, io_comm,
                                         mode="c")
    io_input.parallel_io && error("NetCDF interface does not support parallel I/O")

    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix, ".cdf")
    # if a netcdf file with the requested name already exists, remove it
    mode == "c" && isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename, mode)

    return fid, (filename, io_input, io_comm)
end

function create_io_group(parent::NCDataset, name; description=nothing)
    if description !== nothing
        attributes = Dict("description" => description)
    else
        attributes = ()
    end

    return defGroup(parent, name, attrib=attributes)
end

function get_io_variable(file_or_group::NCDataset, name::String)
    return file_or_group[name]
end

function get_group(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        if occursin("/", name)
            split_names = split(name, "/")
            this_group = file_or_group
            for n ∈ split_names
                this_group = this_group.group[n]
            end
            return this_group
        else
            return file_or_group.group[name]
        end
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

function has_attribute(file_or_group_or_var::Union{NCDataset,NCDatasets.CFVariable}, name)
    return name ∈ keys(file_or_group_or_var.attrib)
end

function get_attribute(file_or_group_or_var::Union{NCDataset,NCDatasets.CFVariable}, name)
    return var.attrib[name]
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
                             coords::Union{coordinate,NamedTuple}...; parallel_io,
                             description=nothing, units=nothing,
                             overwrite=false) where {T,N}

    if any(c.n < 0 for c ∈ coords)
        error("Got a negative `n` in $coords")
    end
    if any(c.n == 0 for c ∈ coords)
        # No data to write
        return nothing
    end

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
    end
    if isa(value, Bool)
        # As a hack, write bools to NetCDF as Char, as NetCDF does not support bools (?),
        # and we do not use Char for anything else
        if overwrite && name ∈ keys(file_or_group)
            var = file_or_group[name]
        else
            var = defVar(file_or_group, name, Char, dims, attrib=attributes)
        end
        var[:] = Char(value)
    else
        if overwrite && name ∈ keys(file_or_group)
            var = file_or_group[name]
        else
            var = defVar(file_or_group, name, type, dims, attrib=attributes)
        end
        var[:] = value
    end

    return nothing
end

function create_dynamic_variable!(file_or_group::NCDataset, name, type,
                                  coords::Union{coordinate,NamedTuple}...; parallel_io,
                                  description=nothing, units=nothing)

    if any(c.n < 0 for c ∈ coords)
        error("Got a negative `n` in $coords")
    end
    if any(c.n == 0 for c ∈ coords)
        # No data to write
        return nothing
    end

    # Create time dimension if necessary
    maybe_create_netcdf_dim(file_or_group, "time", Inf)

    # Create other dimensions if necessary
    for c ∈ coords
        maybe_create_netcdf_dim(file_or_group, c)
    end

    # create the variable so it can be expanded indefinitely (up to the largest unsigned
    # integer in size) in the time dimension
    coord_dims = Tuple(c.name for c ∈ coords)
    dims = tuple(coord_dims..., "time")

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
                               data::Union{Nothing,Number,AbstractArray{T,N}}, t_idx,
                               parallel_io::Bool,
                               coords...; only_root=false,
                               write_from_this_rank=nothing) where {T,N}
    if only_root && parallel_io && global_rank[] != 0
        # Variable should only be written from root, and this process is not root for the
        # output file
        return nothing
    end

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

function open_file_to_read(::Val{netcdf}, filename)
    return NCDataset(filename, "r")
end

function load_variable(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        if size(file_or_group[name].var) == ()
            var = file_or_group[name].var[]
        else
            var = copy(file_or_group[name].var)
        end
        if isa(var, Char)
            var = (var == Char(true))
        end
        return var
    catch
        println("An error occured while loading $name")
        rethrow()
    end
end

function load_slice(file_or_group::NCDataset, name::String, slices_or_indices...)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        var = file_or_group[name].var[slices_or_indices...]
        return var
    catch
        println("An error occured while loading $name")
        rethrow()
    end
end

end # file_io_netcdf
