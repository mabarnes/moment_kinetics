"""
"""
module type_definitions

export mk_float
export mk_int
export mk_zeros
export MKArray
export MKVector
export MKMatrix
export AbstractMKArray
export AbstractMKVector
export AbstractMKMatrix
export OptionsDict

using InboundsArrays
using OrderedCollections: OrderedDict

"""
"""
const mk_float = Float64

"""
"""
const mk_int = Int64

"""
"""
const MKArray{T,N} = InboundsArray{T,N,Array{T,N}} where {T,N}

"""
"""
const MKVector{T} = MKArray{T,1}

"""
"""
const MKMatrix{T} = MKArray{T,2}

"""
"""
const AbstractMKArray{T,N} = AbstractInboundsArray{T,N}

"""
"""
const AbstractMKVector{T} = AbstractInboundsArray{T,1}

"""
"""
const AbstractMKMatrix{T} = AbstractInboundsArray{T,2}

@inline function MKArray(x)
    return InboundsArray(x)
end

@inline function MKArray{T}(args...) where T
    return InboundsArray{T}(args...)
end

@inline function MKVector(x)
    return InboundsVector(x)
end

@inline function MKMatrix(x)
    return InboundsMatrix(x)
end

function mk_zeros(args...)
    if args[1] isa Type
        a = zeros(args...)
    else
        a = zeros(mk_float, args...)
    end
    return MKArray(a)
end

"""
"""
const OptionsDict = OrderedDict{String,Any}

end
