module type_definitions

export mk_float, mk_int, pdf_dimensions, moment_dimensions, spatial_dims,
       pdf_ndims, spatial_ndims, moment_ndims

using NamedDims

const mk_float = Float64
const mk_int = Int64

function expand_Val_dimnames(::Val{dimnames}, ::Val{dim}) where {dimnames, dim}
    return Val(NamedDims.expand_dimnames(dimnames, dim))
end

const phase_space_dims_tuple = (:vpa, :z)
const phase_space_ndims = 2
const spatial_dims_tuple = (:z,)
const spatial_ndims = 1

const phase_space_dims = Val(phase_space_dims_tuple)
const pdf_dims_tuple = NamedDims.expand_dimnames(phase_space_dims_tuple, :s)
const pdf_dims = Val(pdf_dims_tuple)
const pdf_ndims = phase_space_ndims + 1
const spatial_dims = Val(spatial_dims_tuple)
const moment_dims_tuple = NamedDims.expand_dimnames(spatial_dims_tuple, :s)
const moment_dims = Val(moment_dims_tuple)
const moment_ndims = spatial_ndims + 1

_check_length(::Val{dims}, ndims) where dims = (length(dims) == ndims)
@assert _check_length(pdf_dims, pdf_ndims)
@assert _check_length(spatial_dims, spatial_ndims)
@assert _check_length(moment_dims, moment_ndims)

end
