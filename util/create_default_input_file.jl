using moment_kinetics.input_structs: options_to_TOML
using moment_kinetics.moment_kinetics_input: mk_input
using moment_kinetics.type_definitions: OptionsDict

"""
Create an example input file for moment_kinetics, with all options included with their
current default values.
"""
function create_default_input_file(filename="example.toml")
    # run_name is required when passing a Dict to mk_input...
    input_dict = OptionsDict("output" => OptionsDict("run_name" => "example"))

    mk_input(input_dict)

    # ...but we don't really want to write run_name as it is usually created from the
    # input file name, and does not have to be set in the input file.
    pop!(input_dict["output"], "run_name")

    # This is an internal implementation detail, that we don't want to write to the
    # example file.
    pop!(input_dict, "_section_check_store")

    if isfile(filename)
        error("$filename already exists")
    end
    open(filename; write=true) do io
        options_to_TOML(io, input_dict)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        create_default_input_file(ARGS[1])
    else
        create_default_input_file()
    end
end
