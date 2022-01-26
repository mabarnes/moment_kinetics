module PrintMacros
"""
This script prints expanded version of auto-generated macros.
Intended to help developers see what the macros are doing.
"""

export print_macros

using moment_kinetics.looping
using moment_kinetics.looping: dimension_combinations, dims_string

function print_macros()
    println("Here is a set of examples of expanding the macros defined in the `looping` module")
    println()

    # Print loop macros
    for dims ∈ dimension_combinations
        println()

        iteration_vars = Tuple(string("i", d) for d ∈ dims)
        macro_name = string("@loop_", dims_string(dims))
        macro_example = """
            $macro_name $(join(iteration_vars, " ")) begin
                foo[$(join(reverse(iteration_vars), ","))] = something
            end
        """
        println("```")
        println(macro_example)
        println("```")
        println("expands to:")
        println("```")
        println(eval(Meta.parse("@macroexpand $macro_example")))
        println("```")
    end
end

end # PrintMacros

using .PrintMacros
print_macros()
