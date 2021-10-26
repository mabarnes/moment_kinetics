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

using ArgParse

macronames = [
              ("debug_initialize_NaN", 1),
              ("debug_error_stop_all", 1),
              ("debug_block_synchronize", 2),
              ("debug_shared_array", 2),
              ("debug_shared_array_allocate", 3),
             ]

#s = ArgParseSettings()
#@add_arg_table! s begin
#    "--debug", "-d"
#        help = "Set debugging level, default is 0 (no extra debugging). Higher " *
#               "integer values activate more checks (and increase run time)"
#        arg_type = Int
#        default = 0
#end
#options = parse_args(s)
#_debug_level = options["debug"]
## Problems with trying to call ArgParse with a partial options list - for now just
## hard-code _debug_level
_debug_level = 0

for (macroname, minlevel) ∈ macronames
    m = Symbol(macroname)
    export_string = Symbol(string("@", macroname))
    ifelse_string = macroname * "_ifelse"
    ifelse_symbol = Symbol(ifelse_string)
    export_ifelse_string = Symbol(string("@", ifelse_string))

    if _debug_level >= minlevel
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
                return :()
            end

            macro $ifelse_symbol(debug, standard)
                return :( $(esc(standard)) )
            end

            export $export_string, $export_ifelse_string
        end
    end

    eval(macro_block)
end

end # debugging
