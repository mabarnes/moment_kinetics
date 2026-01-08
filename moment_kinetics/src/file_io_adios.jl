# This extension provides an interface to the optional file_io_adios module, which
# provides ADIOS2 I/O
#
# Note that if there are errors when precompiling an extension, they may not be shown by
# default. To see the error, precompile by running
# `using Pkg; Pkg.precompile(strict=true)`.
module file_io_adios

import moment_kinetics.file_io: io_has_implementation, io_has_parallel,
                                open_output_file_implementation, create_io_group,
                                get_variable, get_group, is_group, get_subgroup_keys,
                                get_variable_keys, add_attribute!, write_single_value!,
                                create_dynamic_variable!, append_to_dynamic_var
import moment_kinetics.load_data: open_file_to_read, get_attribute, has_attribute,
                                  load_variable, load_slice
using moment_kinetics.communication
using moment_kinetics.coordinates: coordinate
using moment_kinetics.input_structs: adios
using moment_kinetics.type_definitions

using ADIOS2
using MPI

function io_has_implementation(::Val{adios})
    return true
end

function io_has_parallel(::Val{adios})
    return true
end

function open_output_file_implementation(::Val{adios}, prefix, io_input, aio_or_comm)
    # The ADIOS file will be given by output_dir/run_name with .bp appended.
    filename = string(prefix, ".bp")

    if !io_input.parallel_io
        error("ADIOS I/O requires parallel_io=true.")
    end

    # Create the new ADIOS file.
    # ADIOS always uses parallel I/O.
    if !isa(aio_or_comm, Tuple{AIO,MPI.Comm}) && MPI.Comm_rank(aio_or_comm) == 0 && isfile(filename)
        # If a file with the requested name already exists, remove it.
        rm(filename)
    end

    if isa(aio_or_comm, Tuple{AIO,MPI.Comm})
        # aio_or_comm is not an MPI communicator, but rather a Tuple with an existing ADIOS
        # I/O handler and the communicator. It is passed this way for compatibility with
        # the structure set up for other I/O backends.
        adios_io, io_comm = aio_or_comm
        MPI.Barrier(io_comm)

        # Apen file for writing - here appending to an existing file.
        adios_writer = open(adios_io, filename, mode_append)
    else
        io_comm = aio_or_comm
        adios = adios_init_mpi(io_comm)
        adios_io = declare_io(adios, "WriteIO")
        MPI.Barrier(io_comm)

        # Open file for writing - here appending to an existing file.
        adios_writer = open(adios_io, filename, mode_write)
    end

    # ADIOS requires that we begin a 'step' as well as opening the file.
    begin_step(adios_writer)

    fid = (adios_io, adios_writer)

    return fid, (filename, io_input, (adios_io, io_comm))
end

function create_io_group(parent::Union{Tuple{AIO,Engine},Tuple{Tuple{AIO,Engine},String}}, group_name; description=nothing)
    # The C API that ADIOS2.jl wraps apparently does not support independent handles for
    # groups. As a workaround, we represent a group by a Tuple of the AIO/Engine and the
    # group group_name. The group should be implicitly created if it does not already exsist
    # when a variable or attribute is added to it.

    if isa(parent, Tuple{AIO,Engine})
        group = (parent, group_name)
    else
        file, parent_name = parent
        group = (file, parent_name * "/" * group_name)
    end

    if description !== nothing
        add_attribute!(group, "description", description)
    end

    return group
end

function add_attribute!(file::Tuple{AIO,Engine}, attribute_name, value)
    adios_io, writer = file
    return define_attribute(adios_io, attribute_name, value)
end
function add_attribute!(group::Tuple{Tuple{AIO,Engine},String}, attribute_name, value)
    # Add attribute to a Group.
    file, group_name = group
    adios_io, writer = file
    return define_attribute(adios_io, group_name * "/" * attribute_name, value)
end
function add_attribute!(io_var::Variable, attribute_name, value)
    error("Don't know how to define attribute from just an ADIOS2.Variable instance. "
          * "Would also need the AIO instance")
end

function add_variable_attribute!(adios_io::AIO, variable_name, attribute_name, value)
    return define_variable_attribute(adios_io, attribute_name, value, variable_name)
end

function has_attribute(file::Tuple{AIO,Engine}, attribute_name)
    adios_io, writer = file
    attrs = inquire_group_attributes(adios_io, "")
    return attribute_name ∈ [basename(name(a)) for a ∈ attrs]
end
function has_attribute(group::Tuple{Tuple{AIO,Engine},String}, attribute_name)
    file, group_name = group
    adios_io, writer = file
    attrs = inquire_group_attributes(adios_io, group_name)
    return attribute_name ∈ [basename(name(a)) for a ∈ attrs]
end
function has_attribute(io_var::Variable, attribute_name)
    error("Don't know how to check attribute from just an ADIOS2.Variable instance. "
          * "Would also need the AIO instance")
end

function get_variable(file::Tuple{AIO,Engine}, variable_name::String)
    adios_io, writer = file
    return (writer, inquire_variable(adios_io, variable_name))
end
function get_variable(group::Tuple{Tuple{AIO,Engine},String}, variable_name::String)
    file, group_name = group
    adios_io, writer = file
    return (writer, inquire_variable(adios_io, group_name * "/" * variable_name))
end

function get_group(file::Tuple{AIO,Engine}, group_name::String)
    return (file, group_name)
end
function get_group(parent::Tuple{Tuple{AIO,Engine},String}, group_name::String)
    file, parent_name = parent
    return (file, parent_name * "/" * group_name)
end

function is_group(file::Tuple{AIO,Engine}, group_name::String)
    adios_io, writer = file
    return group_name ∈ inquire_subgroups(adios_io, "")
end
function is_group(parent::Tuple{Tuple{AIO,Engine},String}, group_name::String)
    file, parent_name = parent
    adios_io, writer = file
    return group_name ∈ inquire_subgroups(adios_io, parent_name)
end

function get_subgroup_keys(file::Tuple{AIO,Engine})
    adios_io, writer = file
    return inquire_subgroups(adios_io, "")
end
function get_subgroup_keys(parent::Tuple{Tuple{AIO,Engine},String})
    file, parent_name = parent
    adios_io, writer = file
    return inquire_subgroups(adios_io, parent_name)
end

function get_variable_keys(file::Tuple{AIO,Engine})
    adios_io, writer = file
    variables = inquire_group_variables(adios_io, "")
    return [name(v) for v ∈ variables]
end
function get_variable_keys(parent::Tuple{Tuple{AIO,Engine},String})
    file, parent_name = parent
    adios_io, writer = file
    variables = inquire_group_variables(adios_io, parent_name)
    return [basename(name(v)) for v ∈ variables]
end

function write_single_value!(group::Tuple{Tuple{AIO,Engine},String}, variable_name, args...;
                             kwargs...)
    file, group_name = group
    return write_single_value!(file, group_name * "/" * variable_name, args...; kwargs...)
end
function write_single_value!(file::Tuple{AIO,Engine}, variable_name,
                             data::Union{Number, AbstractString, AbstractArray{T,N}},
                             coords::Union{coordinate,mk_int,NamedTuple}...; parallel_io,
                             description=nothing, units=nothing,
                             overwrite=false) where {T,N}

    adios_io, adios_writer = file

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
        if isa(data, Bool)
            # Convert to UInt8 because ADIOS cannot read/write Bool.
            data = UInt8(data)
        end
        if !(overwrite && variable_name ∈ [name(v) for v ∈ inquire_all_variables(adios_io)])
            io_var = define_variable(adios_io, variable_name, typeof(data))
            if description !== nothing
                add_variable_attribute!(adios_io, variable_name, "description", description)
            end
            if units !== nothing
                add_variable_attribute!(adios_io, variable_name, "units", units)
            end
        else
            io_var = inquire_variable(adios_io, variable_name)
        end
        if global_rank[] == 0
            put!(adios_writer, io_var, data)
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

    dim_sizes = get_fixed_dim_sizes(coords)
    local_ranges = Tuple(isa(c, mk_int) ? (1:c) : isa(c, coordinate) ? c.local_io_range : c.n for c ∈ coords)
    global_ranges = Tuple(isa(c, mk_int) ? (1:c) : isa(c, coordinate) ? c.global_io_range : c.n for c ∈ coords)
    if overwrite && variable_name ∈ keys(file_or_group)
        io_var = inquire_variable(adios_io, variable_name)
    else
        # Final `true` argument ('constant_dims') indicates that the dimensions passed
        # here never change.
        io_var = define_variable(adios_io, variable_name, T, dim_sizes,
                                 Tuple(first(r) - 1 for r ∈ global_ranges), # Note - need to convert to 0-based indexing for the offset
                                 Tuple(length(r) for r ∈ local_ranges);
                                 constant_dims=true)
    end

    if description !== nothing
        add_attribute!(file, variable_name * "/description", description)
    end

    put!(adios_writer, io_var, @view(data[local_ranges...]))

    return nothing
end

# Convert Enum values to String to be written to file
function write_single_value!(group::Tuple{Tuple{AIO,Engine},String}, variable_name, data::Enum; kwargs...)
    return write_single_value!(group, variable_name, string(data); kwargs...)
end
function write_single_value!(file::Tuple{AIO,Engine}, variable_name, data::Enum; kwargs...)
    return write_single_value!(file, variable_name, string(data); kwargs...)
end

"""
Get sizes of fixed dimensions (i.e. everything but time) for I/O

`coords` should be a Tuple whose elements are coordinate structs or integers (e.g. number
of species).
"""
function get_fixed_dim_sizes(coords)
    return Tuple(isa(c, mk_int) ? c : (isa(c, coordinate) ? c.n_global : c.n) for c in coords)
end

function create_dynamic_variable!(group::Tuple{Tuple{AIO,Engine},String}, variable_name, args...;
                                  kwargs...)
    file, group_name = group
    return create_dynamic_variable!(file, group_name * "/" * variable_name, args...; kwargs...)
end
function create_dynamic_variable!(file::Tuple{AIO,Engine}, variable_name, type,
                                  coords::Union{coordinate,NamedTuple}...; parallel_io,
                                  description=nothing, units=nothing)

    adios_io, adios_writer = file

    if any(isa(c, mk_int) ? c < 0 : c.n < 0 for c ∈ coords)
        error("Got a negative `n` in $coords")
    end
    if any(isa(c, mk_int) ? c == 0 : c.n == 0 for c ∈ coords)
        # No data to write
        return nothing
    end

    dim_sizes = get_fixed_dim_sizes(coords)
    local_ranges = Tuple(isa(c, mk_int) ? (1:c) : isa(c, coordinate) ? c.local_io_range : c.n for c ∈ coords)
    global_ranges = Tuple(isa(c, mk_int) ? (1:c) : isa(c, coordinate) ? c.global_io_range : c.n for c ∈ coords)
    # Final `true` argument ('constant_dims') indicates that the dimensions passed
    # here never change.
    io_var = define_variable(adios_io, variable_name, type, dim_sizes,
                             Tuple(first(r) - 1 for r ∈ global_ranges), # Note - need to convert to 0-based indexing for the offset
                             Tuple(length(r) for r ∈ local_ranges); constant_dims=true)

    # Add attribute listing the dimensions belonging to this variable
    dim_names = Tuple(c.name for c ∈ coords)
    add_variable_attribute!(adios_io, variable_name, "dims", join(dim_names, ","))

    if description !== nothing
        add_variable_attribute!(adios_io, variable_name, "description", description)
    end
    if units !== nothing
        add_variable_attribute!(adios_io, variable_name, "units", units)
    end

    return io_var
end

function append_to_dynamic_var(writer_var::Tuple{Engine,Variable},
                               data::Union{Nothing,Number,String,AbstractArray{T,N}}, t_idx,
                               parallel_io::Bool,
                               coords::Union{coordinate,NamedTuple,Integer}...;
                               only_root=false, write_from_this_rank=nothing) where {T,N}

    adios_writer, io_var = writer_var

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

    put!(adios_writer, io_var, data)

    return nothing
end

function Base.close(file::Tuple{AIO,Engine})

    adios_io, adios_writer = file

    # ADIOS requires that we end a 'step' as well as closing the file.
    end_step(adios_writer)

    close(adios_writer)

    return nothing
end

# Overload to get dimensions of a variable from our Tuple{Engine,Variable}
function Base.ndims(engine_var::Tuple{Engine,Variable})
    writer, io_var = engine_var
    return ndims(io_var)
end

function open_file_to_read(::Val{adios}, filename)
    adios = adios_init_serial()
    adios_io = declare_io(adios, "ReadIO")

    # Open file for writing - here appending to an existing file.
    adios_reader = open(adios_io, filename, mode_readRandomAccess)

    return (adios_io, adios_reader)
end

function load_variable(file_or_group::Tuple{AIO,Engine}, variable_name::String)
    error("load_variable not implemented yet for ADIOS")
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        if size(file_or_group[variable_name].var) == ()
            var = file_or_group[variable_name].var[]
        else
            var = copy(file_or_group[variable_name].var)
        end
        if isa(var, UInt8)
            var = (var == UInt8(true))
        end
        return var
    catch
        println("An error occured while loading $variable_name")
        rethrow()
    end
end

function load_slice(file_or_group::Tuple{AIO,Engine}, variable_name::String, slices_or_indices...)
    error("load_slice not implemented yet for ADIOS")
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        var = file_or_group[variable_name].var[slices_or_indices...]
        return var
    catch
        println("An error occured while loading $variable_name")
        rethrow()
    end
end

end # file_io_adios
