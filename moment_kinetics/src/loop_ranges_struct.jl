module loop_ranges_struct

using ..type_definitions: mk_int

# The ion dimensions and neutral dimensions are separated in order to restrict the
# supported parallel loop types to correct combinations. This also reduces the number
# of combinations - for some of the debugging features this helps.
const ion_dimensions = (:s, :r, :z, :vperp, :vpa)
const neutral_dimensions = (:sn, :r, :z, :vzeta, :vr, :vz)
const all_dimensions = unique((ion_dimensions..., neutral_dimensions...))

# Create struct to store ranges for loops over all combinations of dimensions
LoopRanges_body = quote
    parallel_dims::Tuple{Vararg{Symbol}}
    rank0::Bool
    is_anysv::Bool
    anysv_rank0::Bool
    is_anyzv::Bool
    anyzv_rank0::Bool
end
for dim ∈ all_dimensions
    global LoopRanges_body
    LoopRanges_body = quote
        $LoopRanges_body;
        $dim::UnitRange{mk_int}
    end
end
eval(quote
         """
         LoopRanges structs contain information on which points should be included on
         this process in loops over shared-memory arrays.

         Members
         -------
         parallel_dims::Tuple{Vararg{Symbol}}
                Indicates which dimensions are (or might be) parallelized when using
                this LoopRanges. Provided for information for developers, to make it
                easier to tell (when using a Debugger, or printing debug informatino)
                which LoopRanges instance is active in looping.loop_ranges at any point
                in the code.
         rank0::Bool
                Is this process the one with rank 0 in the 'block' which work in
                parallel on shared memory arrays.
         <d>::UnitRange{mk_int}
                Loop ranges for each dimension <d> in looping.all_dimensions.
         """
         Base.@kwdef struct LoopRanges
             $LoopRanges_body
         end
     end)

"""
module variable that we can access by giving fully-qualified name in loop
macros
"""
const loop_ranges = Ref{LoopRanges}()

"""
module variable used to store LoopRanges that are swapped into the loop_ranges
variable in @begin_*_region() functions
"""
const loop_ranges_store = Dict{Tuple{Vararg{Symbol}}, LoopRanges}()

end # loop_ranges_struct
