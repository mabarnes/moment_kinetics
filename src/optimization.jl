module optimization

# 'Macro hygiene' in Julia means expressions are evaluated by default with variables,
# etc.  local to the macro or defined in the module where the macro is defined. The
# `esc()` function can be used to 'escape' an expression so variables in the expression
# are taken from the calling context of the macro.  Approach here is to wrap everything
# in `esc()` (i.e. disable all 'macro hygiene') because we define macros that wrap other
# macros (see comments below). This works OK for us because there is not much code in
# the @innerloop and @outerloop macros - the lack of 'hygiene' means we need to export
# things that are used in the loops (e.g. `Polyester`) so that they are available when
# `using optimization` is used.

export @innerloop, @outerloop

using Base.Threads: threadid
export threadid

using Polyester
export Polyester

using FLoops
export FLoops

"""
Macro to use for optimizing inner loops

Useful to have one place to change things for many loops.
"""
macro innerloop(ex)
    # Note calling a macro within a macro is awful, and an open issue
    # https://github.com/JuliaLang/julia/issues/37691
    # Seems like the following should work, but doesn't because @simd doesn't
    # understand the expression with an :escape in it.
    #return :( @simd(ivdep,$(esc(ex))) )
    #
    # Workarounds discussed here
    # https://discourse.julialang.org/t/calling-a-macro-from-within-a-macro-revisited/19680/11
    #
    # Put the esc() outside everything so we can wrap other macros - idea
    # borrowed from Base.SimdLoop.@simd
    # (https://github.com/JuliaLang/julia/blob/master/base/simdloop.jl).
    #
    # These macros don't seem to help, so skip
    #return esc( :( @fastmath @simd ivdep $ex ) )
    #
    return esc( :( $ex ) )
end

macro outerloop(ex)
    # Do-nothing
    #return esc( :( $ex ) )

    # Threading from Julia Base
    #return esc( :( Base.Threads.@threads $ex ) )

    # Lightweight threads using Polyester.jl
    #return esc( :( Polyester.@batch $ex ) )

    # Parallelise nested loops using FLoops.jl
    block = quote
        FLoops.@floop FLoops.ThreadedEx() $ex
    end
    return esc(block)
end

end # optimization
