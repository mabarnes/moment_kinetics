"""
"""
module type_definitions

export mk_float
export mk_int
export OptionsDict
export MPISharedArray

using ..debugging

using OrderedCollections: OrderedDict

"""
"""
const mk_float = Float64

"""
"""
const mk_int = Int64

"""
"""
const OptionsDict = OrderedDict{String,Any}

@debug_shared_array begin
    """
    Special type for debugging race conditions in accesses to shared-memory arrays.
    Only used if debugging._debug_level is high enough.
    """
    struct DebugMPISharedArray{T, N, TArray <: AbstractArray{T,N}, TIntArray <: AbstractArray{mk_int,N}, TBoolArray <: AbstractArray{Bool,N}, TRanges} <: AbstractArray{T, N}
        data::TArray
        dim_names::NTuple{N, Symbol}
        dim_ranges::TRanges
        accessed::Base.RefValue{Bool}
        is_initialized::TIntArray
        is_read::TBoolArray
        is_written::TBoolArray
        creation_stack_trace::String
        @debug_detect_redundant_block_synchronize begin
            previous_is_read::TBoolArray
            previous_is_written::TBoolArray
        end
    end

    export DebugMPISharedArray
end

"""
Type used to declare a shared-memory array. When debugging is not active `MPISharedArray`
is just an alias for `Array`, but when `@debug_shared_array` is activated, it is instead
defined as an alias for `DebugMPISharedArray`.
"""
const MPISharedArray = @debug_shared_array_ifelse(DebugMPISharedArray, Array)

end
