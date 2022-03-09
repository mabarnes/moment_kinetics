# Note, arrays are flattened at each time point so that, for example, a 1d1v problem can
# be compared before and after adding a size-1 vperp dimension.

# Do this to avoid needing to pass '--project' argument to julia
using Pkg
Pkg.activate(".")

using ArgParse
using NCDatasets

# For now, just hard-code relative and absolute tolerances
rtol = 1.e-14
atol = 1.e-14

s = ArgParseSettings(description="Compare two sets of debug output produced by "
                     * "`debug_dump()` and identify the first place with a difference.")
@add_arg_table! s begin
    "file1"
        help = "First debug output file."
        arg_type = String
    "file2"
        help = "Second debug output file."
        arg_type = String
end

options = parse_args(s)

file1 = NCDataset(options["file1"], "r")
file2 = NCDataset(options["file2"], "r")

nt1 = length(file1["time"])
nt2 = length(file2["time"])
nt = min(nt1, nt2)

function testfield(name)
    var1 = reshape(file1[name], :, nt1)
    var2 = reshape(file2[name], :, nt2)

    first_difference = -1
    for i ∈ 1:nt
        if any(@. abs(var1[:,i] - var2[:,i]) >=
                  max(atol, rtol*max(var1[:,i], var2[:,i])))
            first_difference = i
            break
        end
    end

    if first_difference == -1
        return first_difference, nothing
    else
        max_difference = maximum(@. abs(var1[:,first_difference] - var2[:,first_difference]))
        return first_difference, max_difference
    end
end

differing_fields = Vector{Any}(undef, 0)

global_first_difference = -1

for name in keys(file1)
    global differing_fields, global_first_difference

    if name ∈ ("time", "istage", "label")
        # Skip these 'metadata' variables - they are just used to identify the point at
        # which the first difference occured.
        continue
    end

    first_difference, max_difference = testfield(name)
    if first_difference > 0 && first_difference == global_first_difference
        # Found another variable that starts to differ at the same point, so add to
        # differing_fields.
        push!(differing_fields, (name, max_difference))
    elseif (first_difference > 0
            && (first_difference < global_first_difference
                || global_first_difference == -1))
        # Found a variable that starts to differ before any of the others tested so far.
        differing_fields = Vector{Any}(undef, 0)
        push!(differing_fields, (name, max_difference))
        global_first_difference = first_difference
    end
end

if global_first_difference < 0
    #close(file1)
    #close(file2)

    println("All variables are the same")
else
    t1 = file1["time"][global_first_difference]
    t2 = file2["time"][global_first_difference]
    istage1 = file1["istage"][global_first_difference]
    istage2 = file2["istage"][global_first_difference]
    label1 = file1["label"][global_first_difference]
    label2 = file2["label"][global_first_difference]

    #close(file1)
    #close(file2)

    if any((t1 != t2, istage1 != istage2, label1 != label2))
        error("""
              Debugging output not written at compatible points. At index \
              $global_first_difference, found:
              file1["time"] == $t1
              file2["time"] == $t2
              file1["istage"] == $istage1
              file2["istage"] == $istage2
              file1["label"] == $label1
              file2["label"] == $label2
              """)
    end

    println("First difference found at time=$t1, istage=$istage1, label=$label1.\n")
    println("Var\tmax_difference")
    for difference in differing_fields
        name, max_difference = difference
        println("$name\t$max_difference")
    end
end
