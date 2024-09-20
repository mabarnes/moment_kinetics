"""
module for storing sparse matrix worker functions
"""
module sparse_matrix_functions

export icsc_func, ic_func
export allocate_sparse_matrix_constructor
export assemble_constructor_data!
export assign_constructor_data!
export sparse_matrix_constructor
export create_sparse_matrix
export get_global_compound_index


using ..type_definitions: mk_float, mk_int
using SparseArrays: sparse
using ..coordinates: coordinate

"""
function that returns the sparse matrix index
used to directly construct the nonzero entries
of a 2D assembled sparse matrix

Assume that this indexing is for an array A that 
in a 2D view is indexed by A[ix,iy]
"""
function icsc_func(ix_local::mk_int,ixp_local::mk_int,
                   ielement_x::mk_int,
                   ngrid_x::mk_int,nelement_x::mk_int,
                   iy_local::mk_int,iyp_local::mk_int,
                   ielement_y::mk_int,
                   ngrid_y::mk_int,nelement_y::mk_int)
    ntot_x = (nelement_x - 1)*(ngrid_x^2 - 1) + ngrid_x^2
    #ntot_y = (nelement_y - 1)*(ngrid_y^2 - 1) + ngrid_y^2
    
    icsc_x = ((ixp_local - 1) + (ix_local - 1)*ngrid_x +
                (ielement_x - 1)*(ngrid_x^2 - 1))
    icsc_y = ((iyp_local - 1) + (iy_local - 1)*ngrid_y + 
                    (ielement_y - 1)*(ngrid_y^2 - 1))
    icsc = 1 + icsc_x + ntot_x*icsc_y
    return icsc
end

"""
    ic_func(ix::mk_int,iy::mk_int,nx::mk_int)

Get the 'linear index' corresponding to `ix` and `iy`. Defined so that the linear
index corresponds to the underlying layout in memory of a 2d array indexed by
`[ix,iy]`, i.e. for a 2d array `f2d`:
* `size(f2d) == (x.n, y.n)`
* For a reference to `f2d` that is reshaped to a vector (a 1d array) `f1d = vec(f2d)` than
  for any `ix` and `iy` it is true that `f1d[ic_func(ix,iy)] ==
  f2d[ix,iy]`.
"""
function ic_func(ix::mk_int,iy::mk_int,nx::mk_int)
    return ix + nx*(iy-1)
end

"""
    get_global_compound_index(x,y,ielement_x,ielement_y,ix_local,iy_local)

For local (within the single element specified by `ielement_x` and `ielement_y`)
indices `ix_local` and `iy_local`, get the global index in the 'linear-indexed' 2d
space of size `(x.n, y.n)` (as returned by [`ic_func`](@ref)).
"""
function get_global_compound_index(x::coordinate,y::coordinate,
                                   ielement_x::mk_int,ielement_y::mk_int,
                                   ix_local::mk_int,iy_local::mk_int)
    # global indices on the grids
    ix_global = x.igrid_full[ix_local,ielement_x]
    iy_global = y.igrid_full[iy_local,ielement_y]
    # global compound index
    ic_global = ic_func(ix_global,iy_global,x.n)
    return ic_global
end

"""
"""
struct sparse_matrix_constructor
    # the Ith row
    II::Array{mk_float,1}
    # the Jth column
    JJ::Array{mk_float,1}
    # the data S[I,J]
    SS::Array{mk_float,1}
end

"""
"""
function allocate_sparse_matrix_constructor(nsparse::mk_int)
    II = Array{mk_int,1}(undef,nsparse)
    @. II = 0
    JJ = Array{mk_int,1}(undef,nsparse)
    @. JJ = 0
    SS = Array{mk_float,1}(undef,nsparse)
    @. SS = 0.0
    return sparse_matrix_constructor(II,JJ,SS)
end

"""
"""
function assign_constructor_data!(data::sparse_matrix_constructor,icsc::mk_int,ii::mk_int,jj::mk_int,ss::mk_float)
    data.II[icsc] = ii
    data.JJ[icsc] = jj
    data.SS[icsc] = ss
    return nothing
end

"""
"""
function assemble_constructor_data!(data::sparse_matrix_constructor,icsc::mk_int,ii::mk_int,jj::mk_int,ss::mk_float)
    data.II[icsc] = ii
    data.JJ[icsc] = jj
    data.SS[icsc] += ss
    return nothing
end

"""
"""
function create_sparse_matrix(data::sparse_matrix_constructor)
    return sparse(data.II,data.JJ,data.SS)
end

end
