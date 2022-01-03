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

macronames = [
              ("debug_initialize_NaN", 1),
              ("debug_error_stop_all", 1),
              ("debug_loop_type_region", 1),
              ("debug_block_synchronize", 2),
              ("debug_shared_array", 2),
              ("debug_shared_array_allocate", 3),
              ("debug_detect_redundant_block_synchronize", 4)
             ]

using ..command_line_options: get_options
_debug_level = get_options()["debug"]

for (macroname, minlevel) ∈ macronames
    m = Symbol(macroname)
    export_string = Symbol(string("@", macroname))
    ifelse_string = macroname * "_ifelse"
    ifelse_symbol = Symbol(ifelse_string)
    export_ifelse_string = Symbol(string("@", ifelse_string))

    if _debug_level >= minlevel
        println("$export_string activated")
        macro_block = quote
            macro $m(blk)
                return quote
                    # Uncomment the following line to print the macro name each time
                    # the debug block is called. Can be useful to see progress, since
                    # debugging blocks can make the code run very slowly.
                    #println($$macroname)

                    $(esc(blk))
                end
            end

            macro $ifelse_symbol(debug, standard)
                return :( $(esc(debug)) )
            end

            export $export_string, $export_ifelse_string
        end
    else
        macro_block = quote
            macro $m(blk)
                return
            end

            macro $ifelse_symbol(debug, standard)
                return :( $(esc(standard)) )
            end

            export $export_string, $export_ifelse_string
        end
    end

    eval(macro_block)
end

@debug_loop_type_region begin
    # Add checks that loop macros are used within the correct 'parallel region'. Helps
    # ensure the `begin_*_region()` functions were called in the right places.
    #
    # Define this variable in `debugging` so that it can be imported in both
    # communication.jl and looping.jl
    const current_loop_region_type = Ref("serial")
    export current_loop_region_type
end

end # debugging
