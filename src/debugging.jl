"""
Define debugging levels that can be used to include extra debugging steps

Provides a bunch of macros (see the `macronames` Vector) that can be used to surround code in
other modules so that it only runs if the 'debug level' passed to the `--debug` or `-d`
command line argument is high enough.
"""
module debugging

using ArgParse

macronames = [
              ("debug_block_synchronize", 1),
              ("debug_shared_array", 1),
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

    macro_block = quote
        macro $m(blk)
            if _debug_level >= $minlevel
                return :( $(esc(blk)) )
            else
                return :()
            end
        end

        export $export_string
    end

    eval(macro_block)
end

end # debugging
