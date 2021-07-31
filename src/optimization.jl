module optimization

export @innerloop

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

end # optimization
