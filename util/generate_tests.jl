"""
Script designed to write new test files for moment kinetics. This should make it much 
easier for people (and perhaps more likely!) to make new tests. This is important for 
when new functionality is added. 

The usage is as follows:

julia --project -O3 util/generate_tests.jl input_1.toml input_2.toml ... input_N.toml test_file.jl

The script will run these .toml files, extract the final phi values and write a test file
that runs the exact same input parameters and checks that the final values match the original 
values of phi. Note that this is designed for a set of inputs that are mostly similar, with
small changes between them. In theory it should work for very different input files as well,
but the generated test will have many extra lines to write out all the input changes.

This script was mostly written by a combination of ChatGPT and Claude Sonnet 4.5 on 8/10/2025.
"""

# Capture command-line arguments BEFORE including setup.jl
# setup.jl contains command-line parsing that would interfere with our arguments
const SCRIPT_ARGS = copy(ARGS)

# Clear ARGS so setup.jl doesn't try to parse our script arguments
empty!(ARGS)

include("../moment_kinetics/test/setup.jl")
using TOML, Printf
using Base.Filesystem: tempname, mkpath, joinpath, realpath, rm
using MPI
using moment_kinetics
using moment_kinetics.utils: OptionsDict, merge_dict_with_kwargs!
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info, postproc_load_variable
using moment_kinetics.interpolation: interpolate_to_grid_z

# recursively render a Dict{String,Any} as an OptionsDict literal:
function dict_to_optionsdict_str(d::Dict{String,Any}, indent::Int=0)
    ind = " "^indent
    lines = String[]
    for (k,v) in d
        val = if isa(v,Dict)
            dict_to_optionsdict_str(v, indent+4)
        elseif isa(v,String)
            "\"$(v)\""
        else
            string(v)
        end
        push!(lines, ind * " " * "\"$k\" => $val")
    end
    return "OptionsDict(\n" * join(lines, ",\n") * "\n" * ind * ")"
end

# Find differences between two nested dicts
function find_dict_differences(base::Dict, new::Dict, prefix::String="")
    diffs = Dict{String,Any}()
    
    for (k, v) in new
        full_key = prefix == "" ? k : "$prefix.$k"
        
        if !haskey(base, k)
            # Key exists in new but not in base
            diffs[k] = v
        elseif isa(v, Dict) && isa(base[k], Dict)
            # Both are dicts, recurse
            nested_diffs = find_dict_differences(base[k], v, full_key)
            if !isempty(nested_diffs)
                diffs[k] = nested_diffs
            end
        elseif v != base[k]
            # Values differ
            diffs[k] = v
        end
    end
    
    return diffs
end

# Convert differences dict to recursive_merge OptionsDict syntax
function diffs_to_recursive_merge_str(diffs::Dict{String,Any}, base_name::String, indent::Int=0)
    ind = " "^indent
    result = "recursive_merge($base_name,\n"
    result *= ind * "               " * dict_to_optionsdict_str(diffs, indent+15)
    result *= ")"
    return result
end

println("Number of arguments: ", length(SCRIPT_ARGS))
println("Arguments received: ", SCRIPT_ARGS)

if length(SCRIPT_ARGS) < 2
    println("Usage: julia --project -O3 util/generate_tests.jl <input1.toml> [input2.toml ...] <output_test.jl>")
    println("Received $(length(SCRIPT_ARGS)) arguments")
    error("incorrect argument usage")
end

# Last argument is output file, rest are input TOML files
toml_files = SCRIPT_ARGS[1:end-1]
out_file = "moment_kinetics/test/" * SCRIPT_ARGS[end]
out_file_without_extension = splitext(basename(out_file))[1]

println("TOML input files: ", toml_files)
println("Output file: ", out_file)

# Parse all TOML files
all_toml_data = [TOML.parsefile(f) for f in toml_files]

# Generate run names from TOML filenames (without extension)
run_names = [splitext(basename(f))[1] for f in toml_files]

# Use first TOML as base
base_toml = all_toml_data[1]
base_optstr = dict_to_optionsdict_str(base_toml)

# Process each TOML file to get expected phi values
test_configs = []
for (idx, toml_data) in enumerate(all_toml_data)
    # eval into a real OptionsDict so we can run the sim
    opts = eval(Meta.parse(dict_to_optionsdict_str(toml_data)))
    
    # inject a temp run directory + name
    run_name = "generated_test_$idx"
    base_dir = tempname()
    mkpath(base_dir)
    opts["output"]["base_directory"] = base_dir
    opts["output"]["run_name"] = run_name
    
    # run the sim quietly
    quietoutput() do
        run_moment_kinetics(opts)
    end
    
    # post‐process phi
    path = joinpath(realpath(base_dir), run_name)
    run_info = get_run_info_no_setup(path)
    phi_zrt = postproc_load_variable(run_info, "phi")
    close_run_info(run_info)
    phi = phi_zrt[:,1,:]
    exp_phi = phi[begin:3:end, end]
    
    # clean up
    rm(realpath(base_dir); recursive=true)
    
    # format as Julia array
    phi_literal = "[" * join([ @sprintf("%.17g", v) for v in exp_phi ], ", ") * "]"
    
    # Find differences from base (for tests after the first one)
    if idx == 1
        config_str = "test_input"
        test_name = run_names[idx]
    else
        diffs = find_dict_differences(base_toml, toml_data)
        config_str = "test_input_$idx"
        test_name = run_names[idx]
    end
    
    push!(test_configs, (
        idx = idx,
        config_str = config_str,
        test_name = test_name,
        phi_literal = phi_literal,
        toml_file = toml_files[idx],
        diffs = idx == 1 ? nothing : find_dict_differences(base_toml, toml_data),
        run_name = run_names[idx]
    ))
end

# Generate the test file
open(out_file, "w") do io
    print(io, """
module $out_file_without_extension

# Test generated from TOML input files

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.utils: merge_dict_with_kwargs!

# default inputs for tests
test_input = $base_optstr

""")

    # Generate test_input_N for subsequent tests
    for config in test_configs[2:end]
        diffs_str = dict_to_optionsdict_str(config.diffs)
        print(io, "test_input_$(config.idx) = recursive_merge(test_input,\n")
        print(io, "                               $diffs_str)\n")
    end
    
    # Add run names to all test inputs
    print(io, "# Here choose the names for each test\n")
    for config in test_configs
        print(io, "$(config.config_str) = recursive_merge($(config.config_str),\n")
        print(io, "                               OptionsDict(\"output\" => OptionsDict(\"run_name\" => \"$(config.run_name)\")))\n")
    end

    print(io, """

\"\"\"
Run a test for a single set of parameters
\"\"\"
function run_test(test_input, expected_phi; rtol=4.e-14, atol=1.e-15, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = input["output"]["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = name

    # Suppress console output while running
    phi = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    if global_rank[] == 0
        quietoutput() do
            # Load and analyse output
            #########################

            path = joinpath(realpath(input["output"]["base_directory"]), name)

            # open the output file(s)
            run_info = get_run_info_no_setup(path)

            # load fields data
            phi_zrt = postproc_load_variable(run_info, "phi")

            close_run_info(run_info)
            
            phi = phi_zrt[:,1,:]
        end

        # Regression test
        actual_phi = phi[begin:3:end, end]
        if expected_phi == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_phi)
            @test false
        else
            @test isapprox(actual_phi, expected_phi, rtol=rtol, atol=atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "$out_file_without_extension tests" verbose=use_verbose begin
        println("$out_file_without_extension tests")
""")

    # Generate test cases
    for config in test_configs
        print(io, """
        @testset "$(config.test_name)" begin
            $(config.config_str)["output"]["base_directory"] = test_output_directory
            run_test($(config.config_str),
                     $(config.phi_literal))
        end
""")
    end

    print(io, """
    end
    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end

using .$out_file_without_extension

$out_file_without_extension.runtests()
""")
end

println("✅ Wrote new test to $out_file")
println("Test configurations:")
for config in test_configs
    println("  - $(config.test_name) from $(config.toml_file)")
    println("    Expected φ (first few) = ", config.phi_literal[1:min(100, length(config.phi_literal))], "...")
end