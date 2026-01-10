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
using file_io: io_input_struct
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
    attrs = inquire_group_attributes(file.io, group_name)
    return attribute_name ∈ [basename(name(a)) for a ∈ attrs]
end
function has_attribute(io_var::Tuple{Variable,AdiosFile}, attribute_name)
    var, file = io_var
    return has_attribute(file, name(var) * "/" * attribute_name)
end

function get_variable(file::AdiosFile, variable_name::String)
    return (file, inquire_variable(file.io, variable_name))
end
function get_variable(group::Tuple{AdiosFile,String}, variable_name::String)
    file, group_name = group
    return (writer, inquire_variable(file.io, group_name * "/" * variable_name))
end

function get_group(file::AdiosFile, group_name::String)
    return (file, group_name)
end
function get_group(parent::Tuple{AdiosFile,String}, group_name::String)
    file, parent_name = parent
    return (file, parent_name * "/" * group_name)
end

function is_group(file::AdiosFile, group_name::String)
    return group_name ∈ inquire_subgroups(file.io, "")
end
function is_group(parent::Tuple{AdiosFile,String}, group_name::String)
    file, parent_name = parent
    return group_name ∈ inquire_subgroups(file.io, parent_name)
end

function get_subgroup_keys(file::AdiosFile)
    return inquire_subgroups(file.io, "")
end
function get_subgroup_keys(parent::Tuple{AdiosFile,String})
    file, parent_name = parent
    return inquire_subgroups(file.io, parent_name)
end

function get_variable_keys(file::AdiosFile)
    variables = inquire_group_variables(file.io, "")
    return [name(v) for v ∈ variables]
end
function get_variable_keys(parent::Tuple{AdiosFile,String})
    file, parent_name = parent
    variables = inquire_group_variables(file.io, parent_name)
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
            put!(file.writer, io_var, data)
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

    file, var = io_var

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

function mk_close(file::AdiosFile)

    # ADIOS requires that we end a 'step' as well as closing the file.
    end_step(file.engine)

    close(file.engine)

    return nothing
end

function io_finalize!(file_info::Tuple{String,io_input_struct,Tuple{Adios,AIO,MPI.Comm}})
    # Finalize ADIOS2 structs
    filename, io_input, (adios, adios_io, io_comm) = file_info
    adios_finalize(adios)
    return nothing
end

# Overload to get dimensions of a variable from our Tuple{AdiosFile,Variable}
function Base.ndims(io_var::Tuple{AdiosFile,Variable})
    file, var = io_var
    return ndims(var)
end

function open_file_to_read(::Val{adios}, filename)
    return adios_open_serial(filename, mode_readRandomAcces)
end

function load_variable(group::Tuple{AdiosFile,String}, variable_name::String)
    file, group_name = group
    return load_variable(file, group_name * "/" * variable_name)
end
function load_variable(file::AdiosFile, variable_name::String)
    return adios_load(file, variable_name)
end

function load_slice(group::Tuple{AdiosFile,String}, variable_name::String, slices_or_indices...)
    file, group_name = group
    return load_variable(file, group_name * "/" * variable_name, slices_or_indices...)
end
function load_slice(file::AdiosFile, variable_name::String, slices_or_indices...)

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
        else
            error("Range type $(typeof(r)) ($r) is unsupported by file_io_adios.")
        end
    end
    function get_count_value(r)
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
        else
            error("Range type $(typeof(r)) ($r) is unsupported by file_io_adios.")
        end
    end
    start = ntuple(i->get_start_value(slices_or_indices[i]), nd-1)
    count = ntuple(i->get_count_value(slices_or_indices[i]), nd-1)

    try
        # If/when https://github.com/eschnett/ADIOS2.jl/pull/21 is merged and released,
        # this should simplify to:
        #var = adios_load(file, variable_name, it; start=start, count=count)

        # Cut and paste code from https://github.com/eschnett/ADIOS2.jl/pull/21 as a
        # temporary workaround:
        step_list = it
        @assert openmode(file.engine) === mode_readRandomAccess "File must be opened with `mode_readRandomAccess`"
        ADIOS2._check_validity_of_steps(file, step_list)

        # body of _schedule_tasks_randomAccess
        if isa(it, Integer)
            it = [it]
        end
        if isa(it, AbstractUnitRange)
            @assert minimum(step_list) >= 0 "Steps must be non-negative integers"
            @assert maximum(step_list) < steps(file.engine) "Steps must be less than total steps"

            adios_var = inquire_variable(file.io, varName)
            if adios_var === nothing
                error("Variable '$varName' not found in the file")
            end
            T, D, sh = _get_var_type_ndims_shape(adios_var)

            if start !== nothing || count !== nothing
                if start === nothing
                    start = ntuple(i->0, D)
                else
                    # Convert `start` to 0-based indexing
                    start = start .- 1
                end
                if count === nothing
                    count = ntuple(i->(Int(sh[i]) - start[i]), D)
                end
                set_selection(adios_var, start, count)
                orig_sh = sh
                sh = count
            end

            # For contiguous UnitRange, create a single IORef
            set_step_selection(adios_var, step_list[1], length(step_list))

            ioref = IORef{T,D + 1}(file.engine,
                                   Array{T,D + 1}(undef, (sh..., length(step_list))))
            get(file.engine, adios_var, ioref.array)
            push!(file.engine.get_tasks, () -> (ioref.engine = nothing))

            if start !== nothing || count !== nothing
                # Unset selection on variable to avoid messing up future operations.
                zero_start = ntuple(i->0, D)
                set_selection(adios_var, zero_start, Int.(orig_sh))
            end
        else
            adios_var = inquire_variable(file.io, varName)
            if adios_var === nothing
                error("Variable '$varName' not found in the file")
            end
            T, D, sh = _get_var_type_ndims_shape(adios_var)

            if start !== nothing || count !== nothing
                if start === nothing
                    start = ntuple(i->0, D)
                else
                    # Convert `start` to 0-based indexing
                    start = start .- 1
                end
                if count === nothing
                    count = ntuple(i->(Int(sh[i]) - start[i]), D)
                end
                set_selection(adios_var, start, count)
                orig_sh = sh
                sh = count
            end

            # Schedule reading for all requested steps
            ioref = IORef[]
            sizehint!(ioref, length(steps))
            for step in steps
                set_step_selection(adios_var, step, 1)

                # Schedule reading data for the current step
                ioref = IORef{T,D}(file.engine, Array{T,D}(undef, Tuple(sh)))
                get(file.engine, adios_var, ioref.array)
                push!(file.engine.get_tasks, () -> (ioref.engine = nothing))
                push!(ioref, ioref)
            end

            if start !== nothing || count !== nothing
                # Unset selection on variable to avoid messing up future operations.
                zero_start = ntuple(i->0, D)
                set_selection(adios_var, zero_start, Int.(orig_sh))
            end
        end
        # end _schedule_tasks_randomAccess

        # Perform all reads at once
        perform_gets(file.engine)

        # If `start` or `count` was passed, then the data is not a scalar and so never needs
        # to be normalized.
        # body of _normalize_data_shape
        maybe_normalize = (start === nothing && count === nothing)
        if isa(ioref, AbstractArray)
            @assert all(isready.(iorefs)) "All IORefs must be ready before assembling data"

            N_steps = length(iorefs)

            # Create result array with time as the last dimension (column-major optimized)
            # Get dimensions from the first IORef's data
            data_arr = [fetch(ioref) for ioref in iorefs]
            dims = size(data_arr[1])
            T = eltype(data_arr[1])

            # Create the final array with time as the last dimension
            result_shape = (dims..., N_steps)
            last_index = findlast(x -> x != 1, result_shape)

            if last_index === nothing
                # scalar
                result = data_arr[1][]
            else
                # Array
                if maybe_normalize
                    result_shape = result_shape[1:last_index]
                    if result_shape[1] == 1
                        result_shape = result_shape[2:end]
                    end
                elseif N_steps == 1
                    # Only one time point, so remove time dimension
                    result_shape = result_shape[1:end-1]
                end

                D = length(result_shape)

                result = Array{T,D}(undef, result_shape)

                if N_steps == 1
                    @views result .= data_arr[1]
                else
                    # Copy data from each step into the result array
                    for i in 1:N_steps
                        selectdim(result, D, i) .= data_arr[i]
                    end
                end
            end
        else
            @assert isready(ioref) "IORefs must be ready before assembling data"

            if maybe_normalize && ndims(ioref.array) == 2 && size(ioref.array, 1) == 1
                # This is a collection of scalar
                # return it as a Vector, not as a Matrix
                result = ioref.array[:]
            else
                # Otherwise, return it as is
                result = ioref.array
            end
        end
        # end _normalize_data_shape
        var = result

        return var
    catch
        println("An error occured while loading $variable_name")
        rethrow()
    end
end

end # file_io_adios
