module parameter_scans

export get_scan_inputs, generate_scan_input_files

using ..command_line_options: get_options
using ..input_structs: options_to_TOML
using ..moment_kinetics_input: read_input_file
using ..type_definitions: OptionsDict

using Glob

"""
    get_scan_inputs(scan_inputs::AbstractDict)

Make a set of inputs for a parameter scan.

`scan_inputs` is like a Dict of inputs for `run_moment_kinetics`, except that any value
may be an array instead of a scalar. The values passed as arrays will be combined as
follows.

A special, extra, setting `combine_outer` can be passed, with the names of
options to combine using an 'outer product'

By default, inputs are combined with an 'inner product', i.e. inputs a,b,c
are combined as (a[1],b[1],c[1]), (a[2],b[2],c[2]), etc.
Any inputs named in 'combine_outer' are instead combined with an 'outer
product', i.e. an entry is created for every value of those inputs combined
with every combination of the other inputs.

Returns a `Vector{OptionsDict}` whose entries are the input for a single run in the
parameter scan.
"""
function get_scan_inputs(scan_inputs::AbstractDict)
    scan_inputs = OptionsDict(scan_inputs)

    if "base_directory" ∉ keys(scan_inputs["output"])
        # Set up base_directory so that the runs in the scan are created in subdirectories
        # under a directory for the whole scan.
        scan_inputs["output"]["base_directory"] = joinpath("runs", scan_inputs["output"]["run_name"])
    end

    combine_outer = pop!(scan_inputs, "combine_outer", String[])
    if combine_outer isa String
        combine_outer = [combine_outer]
    end

    # Collect inputs to be combined as an 'outer product' to be treated specially
    outer_inputs = OptionsDict()
    for x ∈ combine_outer
        x_parts = split(x, ".")
        scan_inputs_section = scan_inputs
        outer_inputs_section = outer_inputs
        for section_name ∈ x_parts[1:end-1]
            scan_inputs_section = scan_inputs_section[section_name]
            new_section = get(outer_inputs_section, section_name, OptionsDict())
            outer_inputs_section[section_name] = new_section
            outer_inputs_section = new_section
        end
        outer_inputs_section[x_parts[end]] = pop!(scan_inputs_section, x_parts[end])
    end

    # First combine the 'inner product' inputs
    ##########################################

    # check all inputs have same size
    inner_input_lengths = OptionsDict()
    function add_to_inner_input_lengths(d, section_name=nothing)
        for (k,v) ∈ d
            long_name = section_name === nothing ? k : "$section_name.$k"
            if v isa AbstractDict
                add_to_inner_input_lengths(v, long_name)
            elseif v isa Vector
                inner_input_lengths[long_name] = length(v)
            end
        end
    end
    add_to_inner_input_lengths(scan_inputs)
    if length(inner_input_lengths) > 0
        length_inner_product = first(values(inner_input_lengths))
        if length(inner_input_lengths) > 0 &&
            !all(l == length_inner_product for l ∈ values(inner_input_lengths))
            error("Lengths of all scan inputs to be combined as an 'inner product' should be "
                  * "the same. Got lengths $inner_input_lengths")
        end
    end

    result = Vector{OptionsDict}(undef, length_inner_product)
    for i ∈ 1:length_inner_product
        run_name = scan_inputs["output"]["run_name"]
        result[i] = OptionsDict()
        function add_section_to_result(result_section, scan_inputs_section)
            for (k,v) ∈ scan_inputs_section
                if v isa AbstractDict
                    result_section[k] = OptionsDict()
                    add_section_to_result(result_section[k], v)
                elseif v isa Vector
                    result_section[k] = v[i]
                    # Truncate `key` - seems that if file names are too long, HDF5 has a
                    # buffer overflow
                    run_name *= "_$(k[1:min(3, length(k))])_$(v[i])"
                else
                    result_section[k] = v
                end
            end
        end
        add_section_to_result(result[i], scan_inputs)
        result[i]["output"]["run_name"] = run_name
    end

    # Combine 'result' with 'combine_outer' fields
    ##############################################

    function create_level(i, scan_list)
        # This function recursively builds an outer-product of all the
        # option values in the scan
        if i > length(outer_inputs.keys)
            # Done building the scan_list, so just return it, ending the
            # recursion
            return scan_list
        end

        key = outer_inputs.keys[i]
        vals = outer_inputs.vals[i]

        l = length(scan_list) * length(vals)
        new_scan_inputs = Vector{OptionsDict}(undef, l)
        count = 0
        for partial_dict ∈ scan_list
            for j ∈ 1:length(vals)
                count = count + 1
                new_dict = deepcopy(partial_dict)
                key_parts = split(key, ".")
                this_section = new_dict
                for section_name ∈ key_parts[1:end-1]
                    this_section = this_section[section_name]
                end
                this_section[key_parts[end]] = vals[j]
                # Truncate `key` - seems that if file names are too long, HDF5 has a
                # buffer overflow
                new_dict["output"]["run_name"] = new_dict["output"]["run_name"] *
                                       "_$(key[1:min(3, length(key))])_$(vals[j])"
                new_scan_inputs[count] = new_dict
            end
        end

        return create_level(i + 1, new_scan_inputs)
    end

    result = create_level(1, result)

    for x ∈ result
        # Sort the inputs to make them more readable
        sort!(x)
    end

    println("Running scan:")
    for x in result
        println(x["output"]["run_name"])
    end

    return result
end

"""
    get_scan_inputs(file_or_dir::AbstractString)

If `file_or_dir` is a file, read input from it using TOML , and call
`get_scan_inputs(scan_inputs::AbstractDict)`.

If `file_or_dir` is a directory, read input from all the `.toml` files in the directory,
returning the inputs as a `Vector{OptionsDict}`.
"""
function get_scan_inputs(file_or_dir::AbstractString)
    if isfile(file_or_dir)
        scan_inputs = sort(OptionsDict(read_input_file(file_or_dir)))
        return get_scan_inputs(scan_inputs)
    elseif isdir(file_or_dir)
        input_filenames = glob(joinpath(file_or_dir, "*.toml"))
        scan_inputs = collect(sort(OptionsDict(read_input_file(f)))
                              for f ∈ input_filenames)
        return scan_inputs
    else
        error("$file_or_dir does not exist")
    end
end

"""
    get_scan_inputs()

Get input file name from command line options, and call
`get_scan_inputs(filename::AbstractString)`
"""
function get_scan_inputs()
    inputfile = get_options()["inputfile"]
    return get_scan_inputs(inputfile)
end

"""
    generate_scan_input_files(scan_input::AbstractDict, dirname::AbstractString)

Generate individual input files for each run in the scan specified by `scan_input`, saving
the generated files in `dirname`

Inputs are generated by calling [`get_scan_inputs(scan_inputs::AbstractDict)`](@ref).
"""
function generate_scan_input_files(scan_input::AbstractDict, dirname::AbstractString)
    input_dicts = get_scan_inputs(scan_input)

    # Create the directory if it does not exist
    mkpath(dirname)

    for input ∈ input_dicts
        # Write the file, but do not overwrite
        filename = joinpath(dirname, input["output"]["run_name"] * ".toml")
        ispath(filename) && error("The file $filename already exists.")
        open(filename, "w") do io
            # The run name will be created from the name of the input file, so do not need
            # to save "run_name" in the file.
            pop!(input["output"], "run_name")
            options_to_TOML(io, input)
        end
    end

    return nothing
end

"""
    generate_scan_input_files(filename::AbstractString, dirname=nothing)

Read inputs for a scan from a TOML file and call
[`generate_scan_input_files(scan_input::AbstractDict, dirname::AbstractString)`](@ref).

By default, `dirname` will be set to `filename` with the `.toml` extension removed.
"""
function generate_scan_input_files(filename::AbstractString, dirname=nothing)

    scan_input = sort(OptionsDict(read_input_file(filename)))

    if dirname === nothing
        dirname = splitext(filename)[1]
    end

    return generate_scan_input_files(scan_input, dirname)
end

end
