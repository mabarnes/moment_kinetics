using moment_kinetics
using moment_kinetics.Glob

function runtests()
    println("this file: ", @__FILE__)
    for filename ∈ (basename(f) for f ∈ glob("test_scripts/*.jl"))
        if filename == basename(@__FILE__)
            # Skip this file to avoid recursion
            continue
        end
        println("\n", filename)
        include(filename)
    end

    return nothing
end

runtests()
