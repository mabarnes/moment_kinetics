using Coverage

n = 5
if length(ARGS) > 0
    n = parse(Int, ARGS[1])
end

const context = 2

# Get the path to a module, as suggested here:
# https://stackoverflow.com/a/63883681/13577592
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

# import REPL just so we can use it to get a path to the Julia install location
using REPL
repl_file = module_file(REPL)
# extract the install directory by stripping of the last few components of the path
julia_dir = joinpath(splitpath(repl_file)[begin:end-5]...)

results = analyze_malloc([joinpath("..", "src"), joinpath(homedir(), ".julia"), julia_dir])
n_results = length(results)

"""
Format a single result from the Vector returned by Coverage.analyze_malloc()

Returns
-------
result : String
    Memory usage, source file and line number, followed by the source code with $context
    lines of context either side.
mem_file : String
    name of the .mem file that the result came from
"""
function format_result(r)
    filename_parts = split(r.filename, ".")
    src_file = join(filename_parts[begin:end-2], ".")

    # Print allocation size and source location
    result = string(r.bytes / 1024^2, " MB from $src_file:", r.linenumber, "\n")

    start = max(r.linenumber - context, 1)
    finish = r.linenumber + context
    w = ndigits(finish)

    open(src_file, "r") do io
        # Skip to line number 'start'
        for i ∈ 1:start-1
            readline(io)
        end

        # Read and print the source lines
        for i ∈ start:finish
            # format and print the line number
            result *= lpad(i, w)
            if i == r.linenumber
                result *= "* "
            else
                result *= "  "
            end

            # print the source line
            result *= readline(io) * "\n"

            if eof(io)
                # Stop if we reach the end of the file
                break
            end
        end
    end
    result *= "\n"

    return result, r.filename
end

if n <= 0
    n = n_results
end

mem_file_list = Vector{String}(undef, 0)
open("memory_profile.txt", "w") do io
    # `results` are sorted from smallest to largest allocations.
    # Here print largest allocations first
    for (i, r) ∈ enumerate(results[end:-1:begin])
        (result_string, mem_file) = format_result(r)

        # Save the .mem file name, if it is not a duplicate
        if ! (mem_file in mem_file_list)
            push!(mem_file_list, mem_file)
        end

        # write all results to log file
        write(io, result_string)

        # print selected number of results
        if i <= n
            print(result_string)
        end
    end
end

# Write the collected list of *.mem files, to make it easy to delete them all with
# clean_up_mem_files.sh
open("mem_files_list.txt", "w") do io
    for mem_file in mem_file_list
        write(io, mem_file * "\n")
    end
end
