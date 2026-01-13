# This extension provides an interface to the optional file_io_adios module, which
# provides ADIOS2 I/O
#
# Note that if there are errors when precompiling an extension, they may not be shown by
# default. To see the error, precompile by running
# `using Pkg; Pkg.precompile(strict=true)`.
module file_io_adios

import moment_kinetics.file_io: io_has_implementation, io_has_parallel, io_close,
                                io_finalize!, open_output_file_implementation,
                                create_io_group, get_io_variable, get_group, is_group,
                                get_subgroup_keys, get_variable_keys, add_attribute!,
                                write_single_value!, create_dynamic_variable!,
                                append_to_dynamic_var
import moment_kinetics.load_data: open_file_to_read, get_attribute, has_attribute,
                                  load_variable, load_slice
using moment_kinetics.file_io: io_input_struct
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
    if !isa(aio_or_comm, Tuple{Adios,AIO,MPI.Comm}) && MPI.Comm_rank(aio_or_comm) == 0 && isfile(filename)
        # If a file with the requested name already exists, remove it.
        rm(filename)
    end

    if isa(aio_or_comm, Tuple{Adios,AIO,MPI.Comm})
        # aio_or_comm is not an MPI communicator, but rather a Tuple with an existing
        # Adios struct, ADIOS I/O handler and the communicator. It is passed this way for
        # compatibility with the structure set up for other I/O backends.
        adios, adios_io, io_comm = aio_or_comm
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
    # Note that if we did not always close the file after each output, for ADIOS we would
    # need to add functions to the file_io interface that could call
    # begin_step()/end_step() for each output.
    begin_step(adios_writer)

    fid = AdiosFile(adios, adios_io, adios_writer)

    return fid, (filename, io_input, (adios, adios_io, io_comm))
end

function create_io_group(parent::Union{AdiosFile,Tuple{AdiosFile,String}}, group_name; description=nothing)
    # The C API that ADIOS2.jl wraps apparently does not support independent handles for
    # groups. As a workaround, we represent a group by a Tuple of the AIO/Engine and the
    # group group_name. The group should be implicitly created if it does not already exsist
    # when a variable or attribute is added to it.

    if isa(parent, AdiosFile)
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

function add_attribute!(file::AdiosFile, attribute_name, value)
    return define_attribute(file.io, attribute_name, value)
end
function add_attribute!(group::Tuple{AdiosFile,String}, attribute_name, value)
    # Add attribute to a Group.
    file, group_name = group
    return define_attribute(file.io, group_name * "/" * attribute_name, value)
end
function add_attribute!(io_var::Tuple{Variable,AdiosFile}, attribute_name, value)
    var, file = io_var
    return add_attribute!(file, name(var) * "/" * attribute_name, value)
end

function add_variable_attribute!(file::AdiosFile, variable_name, attribute_name, value)
    return define_variable_attribute(file.io, attribute_name, value, variable_name)
end

function has_attribute(file::AdiosFile, attribute_name)
    attrs = inquire_group_attributes(file.io, "")
    return attribute_name ∈ [basename(name(a)) for a ∈ attrs]
end
function has_attribute(group::Tuple{AdiosFile,String}, attribute_name)
    file, group_name = group
    # Extra separator for safety. Multiple "/" separators are ignored, so this should be
    # safe to do.
    attrs = inquire_group_attributes(file.io, group_name * "/")
    return attribute_name ∈ [basename(name(a)) for a ∈ attrs]
end
function has_attribute(io_var::Tuple{Variable,AdiosFile}, attribute_name)
    var, file = io_var
    return has_attribute(file, name(var) * "/" * attribute_name)
end

function get_attribute(file::AdiosFile, attribute_name)
    attr = inquire_attribute(file.io, attribute_name)
    return data(attr)
end
function get_attribute(group::Tuple{AdiosFile,String}, attribute_name)
    file, group_name = group
    return get_attribute(file, group_name * "/" * attribute_name)
end
function get_attribute(io_var::Tuple{Variable,AdiosFile}, attribute_name)
    var, file = io_var
    return get_attribute(file, name(var) * "/" * attribute_name)
end

function get_io_variable(file::AdiosFile, variable_name::AbstractString)
    return (inquire_variable(file.io, variable_name), file)
end
function get_io_variable(group::Tuple{AdiosFile,String}, variable_name::AbstractString)
    file, group_name = group
    return (inquire_variable(file.io, group_name * "/" * variable_name), file)
end

function get_group(file::AdiosFile, group_name::AbstractString)
    return (file, String(group_name))
end
function get_group(parent::Tuple{AdiosFile,String}, group_name::AbstractString)
    file, parent_name = parent
    return (file, String(parent_name * "/" * group_name))
end

function is_group(file::AdiosFile, group_name::AbstractString)
    return group_name ∈ inquire_subgroups(file.io, "")
end
function is_group(parent::Tuple{AdiosFile,String}, group_name::AbstractString)
    file, parent_name = parent
    # Extra separator needed to ensure that for example "_bar" is not returned as a
    # subgroup of "foo" when there is another subgroup called "foo_bar" (not sure if this
    # is a bug in ADIOS2/ADIOS2.jl?). Multiple "/" separators are ignored, so this should
    # be safe to do.
    return group_name ∈ inquire_subgroups(file.io, parent_name * "/")
end

function get_subgroup_keys(file::AdiosFile)
    return [lstrip(s, '/') for s ∈ inquire_subgroups(file.io, "")]
end
function get_subgroup_keys(parent::Tuple{AdiosFile,String})
    file, parent_name = parent
    # Extra separator needed to ensure that for example "_bar" is not returned as a
    # subgroup of "foo" when there is another subgroup called "foo_bar" (not sure if this
    # is a bug in ADIOS2/ADIOS2.jl?). Multiple "/" separators are ignored, so this should
    # be safe to do.
    return [lstrip(s, '/') for s ∈ inquire_subgroups(file.io, parent_name * "/")]
end

function get_variable_keys(file::AdiosFile)
    variables = inquire_group_variables(file.io, "")
    return [name(v) for v ∈ variables]
end
function get_variable_keys(parent::Tuple{AdiosFile,String})
    file, parent_name = parent
    # Extra separator for safety. Multiple "/" separators are ignored, so this should be
    # safe to do.
    variables = inquire_group_variables(file.io, parent_name * "/")
    return [basename(name(v)) for v ∈ variables]
end

function write_single_value!(group::Tuple{AdiosFile,String}, variable_name, args...;
                             kwargs...)
    file, group_name = group
    return write_single_value!(file, group_name * "/" * variable_name, args...; kwargs...)
end
function write_single_value!(file::AdiosFile, variable_name,
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
        if isa(data, Bool)
            # Convert to UInt8 because ADIOS cannot read/write Bool.
            data = UInt8(data)
        end
        if !(overwrite && variable_name ∈ [name(v) for v ∈ inquire_all_variables(file.io)])
            io_var = define_variable(file.io, variable_name, typeof(data))
            if description !== nothing
                add_variable_attribute!(file, variable_name, "description", description)
            end
            if units !== nothing
                add_variable_attribute!(file, variable_name, "units", units)
            end
        else
            io_var = inquire_variable(file.io, variable_name)
        end
        if global_rank[] == 0
            put!(file.engine, io_var, data)
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
        io_var = inquire_variable(file.io, variable_name)
    else
        # Final `true` argument ('constant_dims') indicates that the dimensions passed
        # here never change.
        io_var = define_variable(file.io, variable_name, T, dim_sizes,
                                 Tuple(first(r) - 1 for r ∈ global_ranges), # Note - need to convert to 0-based indexing for the offset
                                 Tuple(length(r) for r ∈ local_ranges);
                                 constant_dims=true)
    end

    if description !== nothing
        add_attribute!(file, variable_name * "/description", description)
    end

    put!(file.engine, io_var, @view(data[local_ranges...]))

    return nothing
end

# Convert Enum values to String to be written to file
function write_single_value!(group::Tuple{AdiosFile,String}, variable_name, data::Enum; kwargs...)
    return write_single_value!(group, variable_name, string(data); kwargs...)
end
function write_single_value!(file::AdiosFile, variable_name, data::Enum; kwargs...)
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

function create_dynamic_variable!(group::Tuple{AdiosFile,String}, variable_name, args...;
                                  kwargs...)
    file, group_name = group
    return create_dynamic_variable!(file, group_name * "/" * variable_name, args...; kwargs...)
end
function create_dynamic_variable!(file::AdiosFile, variable_name, type,
                                  coords::Union{coordinate,NamedTuple}...; parallel_io,
                                  description=nothing, units=nothing)

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
    io_var = define_variable(file.io, variable_name, type, dim_sizes,
                             Tuple(first(r) - 1 for r ∈ global_ranges), # Note - need to convert to 0-based indexing for the offset
                             Tuple(length(r) for r ∈ local_ranges); constant_dims=true)

    # Add attribute listing the dimensions belonging to this variable
    dim_names = Tuple(c.name for c ∈ coords)
    add_variable_attribute!(file, variable_name, "dims", join(dim_names, ","))

    if description !== nothing
        add_variable_attribute!(file, variable_name, "description", description)
    end
    if units !== nothing
        add_variable_attribute!(file, variable_name, "units", units)
    end

    return io_var
end

function append_to_dynamic_var(io_var::Tuple{Variable,AdiosFile},
                               data::Union{Nothing,Number,String,AbstractArray{T,N}}, t_idx,
                               parallel_io::Bool,
                               coords::Union{coordinate,NamedTuple,Integer}...;
                               only_root=false, write_from_this_rank=nothing) where {T,N}

    var, file = io_var

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

    put!(file.engine, var, data)

    return nothing
end

function io_close(file::AdiosFile)

    # ADIOS requires that we end a 'step' as well as closing the file.
    if openmode(file.engine) != mode_readRandomAccess
        end_step(file.engine)
    end

    close(file.engine)

    return nothing
end

function io_finalize!(file_info::Tuple{String,io_input_struct,Tuple{Adios,AIO,MPI.Comm}})
    # Finalize ADIOS2 structs
    filename, io_input, (adios, adios_io, io_comm) = file_info
    adios_finalize(adios)
    return nothing
end

# Overload to get dimensions of a variable from our Tuple{Variable,AdiosFile}
function Base.ndims(io_var::Tuple{Variable,AdiosFile})
    var, file = io_var
    nsteps = steps(var)
    nd = ndims(var)
    if nsteps > 1
        # Time-dependent variable. ADIOS2 does not include 'steps' (i.e. the time
        # dimension) in result of `ndims()`, so need to add.
        nd += 1
    end
    return nd
end

# Overload to get element type of a variable from our Tuple{Variable,AdiosFile}
function Base.eltype(io_var::Tuple{Variable,AdiosFile})
    var, file = io_var
    return type(var)
end

# Overload to get size of a variable from our Tuple{Variable,AdiosFile}
function Base.size(io_var::Tuple{Variable,AdiosFile})
    var, file = io_var
    var_count = mk_int.(count(var))
    nsteps = steps(var)
    if nsteps > 1
        # Time-dependent variable. ADIOS2 does not include 'steps' (i.e. the time
        # dimension) in result of `ndims()`, so need to add.
        var_size = tuple(var_count..., nsteps)
    else
        var_size = var_count
    end
    return var_size
end
function Base.size(io_var::Tuple{Variable,AdiosFile}, d::Integer)
    return size(io_var)[d]
end

function open_file_to_read(::Val{adios}, filename)
    return adios_open_serial(filename, mode_readRandomAccess)
end

function load_variable(group::Tuple{AdiosFile,String}, variable_name::AbstractString)
    file, group_name = group
    return load_variable(file, group_name * "/" * variable_name)
end
function load_variable(file::AdiosFile, variable_name::AbstractString)
    var = adios_load(file, variable_name)
    if isa(var, AbstractArray{UInt8,0}) && size(var) == ()
        return var[] == UInt8(true)
    elseif isa(var, UInt8)
        return var == UInt8(true)
    elseif isa(var, AbstractArray) && size(var) == ()
        return var[]
    else
        return var
    end
end

function load_slice(group::Tuple{AdiosFile,String}, variable_name::AbstractString, slices_or_indices...)
    file, group_name = group
    return load_slice(file, group_name * "/" * variable_name, slices_or_indices...)
end
function load_slice(file::AdiosFile, variable_name::AbstractString, slices_or_indices...)
    return load_slice((variable_name, file), slices_or_indices...)
end
function load_slice(io_variable::Union{Tuple{<:AbstractString,AdiosFile},Tuple{Variable,AdiosFile}},
                    slices_or_indices...)

    variable, file = io_variable
    var_size = size(io_variable)

    nd = length(slices_or_indices)

    # Any dimensions that were indexed with an Integer (rather than a UnitRange or
    # AbstractArray) should be dropped from the result after loading.
    drop_dims = ntuple(i->isa(slices_or_indices[i], Integer), nd)

    # Assume all arrays being loaded are time-dependent, so the last element of
    # slices_or_indices is the time index.
    it = slices_or_indices[end]

    # All other indices are required to be contiguous (i.e. Integer or UnitRange) and need
    # to be converted to `start` and `count` values for ADIOS2.
    function get_start_value(r)
        if isa(r, Integer)
            return r
        elseif isa(r, UnitRange)
            return first(r)
        elseif isa(r, AbstractVector)
            if !(r[2:end] .- 1 == r[1:end-1])
                error("file_io_adios requires that slices to be loaded are contiguous. "
                      * "Got $r.")
            end
            return first(r)
        elseif isa(r, Colon)
            return 1
        else
            error("Range type $(typeof(r)) ($r) is unsupported by file_io_adios.")
        end
    end
    function get_count_value(r, dim_size)
        if isa(r, Integer)
            return 1
        elseif isa(r, UnitRange)
            return length(r)
        elseif isa(r, AbstractVector)
            if !(r[2:end] .- 1 == r[1:end-1])
                error("file_io_adios requires that slices to be loaded are contiguous. "
                      * "Got $r.")
            end
            return length(r)
        elseif isa(r, Colon)
            return dim_size
        else
            error("Range type $(typeof(r)) ($r) is unsupported by file_io_adios.")
        end
    end
    start = ntuple(i->get_start_value(slices_or_indices[i]), nd-1)
    count = ntuple(i->get_count_value(slices_or_indices[i], var_size[i]), nd-1)

    if isa(it, StepRange)
        # ADIOS2.jl only supports UnitRange slices
        if it.step != 1
            error("Non-unit step not supported in file_io_adios. Got it=$it.")
        end
        it = it.start:it.stop
    end

    # ADIOS2 uses 0-based indexing, so convert `it`.
    it = it .- 1

    try
        # If/when https://github.com/eschnett/ADIOS2.jl/pull/21 is merged and released,
        # this should simplify to:
        #var = adios_load(file, variable_name, it; start=start, count=count)
        var = adios_load(file, variable, it; start=start, count=count)

        # Remove unwanted dimensions of `var`
        vardims = Tuple(d for (i,d) ∈ enumerate(size(var)) if !drop_dims[i])
        var = reshape(var, vardims)

        return var
    catch
        if isa(variable, Variable)
            println("An error occured while loading ", name(variable))
        else
            println("An error occured while loading $variable")
        end
        rethrow()
    end
end

end # file_io_adios
