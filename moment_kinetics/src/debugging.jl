"""
Define debugging levels that can be used to include extra debugging steps

Provides a bunch of macros (see the `macronames` Vector) that can be used to surround code in
other modules so that it only runs if the 'debug level' passed to the `--debug` or `-d`
command line argument is high enough.

Also provides macro `*_ifelse` whose names are taken from `macronames`, which can be
used to switch definitions, etc. For example, if `debug_shared_array` is in
`macronames`, then
```
const MPISharedArray = @debug_shared_array_ifelse(DebugMPISharedArray, Array)
```
can be used to make the type represented by `MPISharedArray` depend on the debug level.
"""
module debugging

import ..moment_kinetics

"""
"""
macronames = [
    ("debug_initialize_NaN", 1, "Initialize arrays with NaN."),

    ("debug_consistency_checks", 1, "Some extra checks that we do not want in production runs."),

    ("debug_shared_array", 2,
     "Check for incorrect reads/writes to shared-memory arrays"),

    ("debug_track_initialized", 3,
     "Check that every array is initialized before being used."),

    ("debug_track_array_allocate_location", 3,
     "Record where every array was allocated."),

    ("debug_shared_array_allocate", 4,
     "Check that allocate_shared() was called from the same place on every process."),

    ("debug_block_synchronize_quick", 1,
     "Check _block_synchronize() was called from the same place on every process, with quick check of call site."),

    ("debug_block_synchronize_backtrace", 4,
     "Check _block_synchronize() was called from the same place on every process, checking the full backtrace of each call."),

    ("debug_detect_redundant_block_synchronize", 5,
     "Check if any _block_synchronize() call could have been skipped without resulting "
     * "in an error.")
]

using ..command_line_options: get_options

const active_debug_macros = Vector{String}(undef, 0)

"""
"""
_debug_level = get_options()["debug"]

for (macroname, minlevel, macro_docstring) ∈ macronames
    m = Symbol(macroname)
    export_string = Symbol(string("@", macroname))
    ifelse_string = macroname * "_ifelse"
    ifelse_symbol = Symbol(ifelse_string)
    export_ifelse_string = Symbol(string("@", ifelse_string))
    ifelse_docstring = "Evaluate first expression if $macroname is active, second " *
                       "expression if not"
    macro_docstring *= "\n Activated at `_debug_level >= $minlevel`"

    if _debug_level >= minlevel
        println("$export_string activated")
        push!(active_debug_macros, string(export_string))
        macro_docstring *= "\n Currently active (`_debug_level = $_debug_level`)."
        ifelse_docstring *= "\n $macroname is active (`_debug_level = $_debug_level " *
                            ">= $minlevel`)."
        macro_block = quote
            """
            $($macro_docstring)
            """
            macro $m(blk)
                return quote
                    # Uncomment the following line to print the macro name each time
                    # the debug block is called. Can be useful to see progress, since
                    # debugging blocks can make the code run very slowly.
                    #println($$macroname)

                    $(esc(blk))
                end
            end

            """
            $($ifelse_docstring)
            """
            macro $ifelse_symbol(debug, standard)
                return :( $(esc(debug)) )
            end

            export $export_string, $export_ifelse_string
        end
    else
        macro_docstring *= "\n Currently inactive (`_debug_level = $_debug_level`)."
        ifelse_docstring *= "\n $macroname is inactive (`_debug_level = $_debug_level " *
                            "< $minlevel`)."
        macro_block = quote
            """
            $($macro_docstring)
            """
            macro $m(blk)
                return
            end

            """
            $($ifelse_docstring)
            """
            macro $ifelse_symbol(debug, standard)
                return :( $(esc(standard)) )
            end

            export $export_string, $export_ifelse_string
        end
    end

    eval(macro_block)
end

function __init__()
    for name in active_debug_macros
        println("$name is active")
    end
end

end # debugging
