"""
LU solver with the solve phase parallelised with shared-memory MPI.

Uses LinearAlgebra.jl for the factorization.

Solve borrows heavily from ILUZero.jl
"""
module sharedmem_lu_solver

# The `ldiv!()` function in this module was based on the one in ILUZero.jl. ILUZero.jl is
# open source. The following is its license:
#
# MIT License
#
# Copyright (c) 2017 Matt Covalt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

export sharedmem_lu, sharedmem_lu!

using LinearAlgebra
import LinearAlgebra: ldiv!
using MPI
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC

using ..array_allocation: allocate_shared_float, allocate_shared_int
using ..communication
using ..looping
using ..type_definitions: mk_int, mk_float

struct SparseMatrixCSCShared <: AbstractSparseMatrixCSC{mk_float,mk_int}
    m::mk_int
    n::mk_int
    colptr::MPISharedArray{mk_int,1}
    rowval::MPISharedArray{mk_int,1}
    nzval::MPISharedArray{mk_float,1}
end

struct ParallelSparseLU
    m::mk_int
    n::mk_int
    L::SparseMatrixCSCShared
    U::SparseMatrixCSCShared
    wrk::MPISharedArray{mk_float,1}
    # Keep the object created by `lu()` so that we can do in-place updates.
    lu_object::Union{SparseArrays.UMFPACK.UmfpackLU{mk_float,mk_int},Nothing}
    zero_wrk_range::UnitRange{Int64}
    forward_substitution_ranges::Vector{UnitRange{Int64}}
    backward_substitution_ranges::Vector{UnitRange{Int64}}
end

function sharedmem_lu(A::Union{SparseMatrixCSC{mk_float,mk_int},Nothing})
    # Use `lu()` from LinearAlgebra to calculate the L and U factors
    lu_object = nothing
    L_serial = nothing
    U_serial = nothing
    @serial_region begin
        # Set control parameters to disable pivoting, scaling and ordering, so that we can
        # implement a simple `ldiv!()` function, which assumes that L_serial*U_serial = A.
        # Note that this will make the solution less accurate!
        umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(mk_float, mk_int)
        umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_PIVOT_TOLERANCE] = 0.0
        umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_SYM_PIVOT_TOLERANCE] = 0.0
        umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_SCALE] = 0.0
        umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_ORDERING] = 5.0

        lu_object = lu(A; control=umfpack_control)
        L_serial = lu_object.L
        U_serial = lu_object.U
    end

    this_length = Ref(0)

    if block_rank[] == 0
        m, n = size(A)
        this_length[] = m
        MPI.Bcast!(this_length, comm_block[])
        this_length[] = n
        MPI.Bcast!(this_length, comm_block[])
    else
        MPI.Bcast!(this_length, comm_block[])
        m = this_length[]
        MPI.Bcast!(this_length, comm_block[])
        n = this_length[]
    end

    # Allocate and fill the L SparseMatrixCSCShared object
    ######################################################

    if block_rank[] == 0
        this_length[] = length(L_serial.colptr)
        MPI.Bcast!(this_length, comm_block[])
    else
        MPI.Bcast!(this_length, comm_block[])
    end
    L_colptr = allocate_shared_int(this_length[])

    if block_rank[] == 0
        this_length[] = length(L_serial.rowval)
        MPI.Bcast!(this_length, comm_block[])
    else
        MPI.Bcast!(this_length, comm_block[])
    end
    L_rowval = allocate_shared_int(this_length[])
    L_nzval = allocate_shared_float(this_length[])

    L = SparseMatrixCSCShared(m, n, L_colptr, L_rowval, L_nzval)

    @serial_region begin
        L.colptr .= L_serial.colptr
        L.rowval .= L_serial.rowval
        L.nzval .= L_serial.nzval
    end

    # Allocate and fill the U SparseMatrixCSCShared object
    ######################################################

    this_length = Ref(0)
    if block_rank[] == 0
        this_length[] = length(U_serial.colptr)
        MPI.Bcast!(this_length, comm_block[])
    else
        MPI.Bcast!(this_length, comm_block[])
    end
    U_colptr = allocate_shared_int(this_length[])

    if block_rank[] == 0
        this_length[] = length(U_serial.rowval)
        MPI.Bcast!(this_length, comm_block[])
    else
        MPI.Bcast!(this_length, comm_block[])
    end
    U_rowval = allocate_shared_int(this_length[])
    U_nzval = allocate_shared_float(this_length[])

    U = SparseMatrixCSCShared(m, n, U_colptr, U_rowval, U_nzval)

    @serial_region begin
        U.colptr .= U_serial.colptr
        U.rowval .= U_serial.rowval
        U.nzval .= U_serial.nzval
    end

    wrk = allocate_shared_float(m)

    # Get a sub-range for this process to calculate from a global index range
    function get_range(global_range)
        irank = block_rank[]
        nrank = block_size[]
        range_length = length(global_range)

        # Define chunk_size so that nrank*chunk_size ≥ range_length, with equality if
        # nrank divides range_length exactly.
        chunk_size = (range_length + nrank - 1) ÷ nrank

        first_ind = first(global_range)
        last_ind = last(global_range)

        imin = irank * chunk_size + first_ind
        imax = min((irank + 1) * chunk_size + first_ind - 1, last_ind)
        if imin > last_ind
            return 1:0
        end
        return imin:imax
    end
    _block_synchronize()

    zero_wrk_range = get_range(1:m)

    forward_substitution_ranges = Vector{UnitRange{Int64}}(undef, n)
    l_colptr = L.colptr
    for i ∈ 1:n
        # This indexing is different from ILUZero.jl, which has
        # `l_colptr[i]+1:l_colptr[i + 1] - 1`. This is because ILUZero does not store the
        # diagonal entries of L (the diagonal entries L are all 1 by definition), whereas
        # LinearAlgebra.jl/UMFPACK does.
        forward_substitution_ranges[i] = get_range(l_colptr[i]+1:l_colptr[i + 1] - 1)
    end

    backward_substitution_ranges = Vector{UnitRange{Int64}}(undef, n)
    u_colptr = U.colptr
    for i ∈ n:-1:1
        backward_substitution_ranges[i] = get_range(u_colptr[i]:u_colptr[i + 1] - 2)
    end

    return ParallelSparseLU(m, n, L, U, wrk, lu_object, zero_wrk_range,
                            forward_substitution_ranges, backward_substitution_ranges)
end

function sharedmem_lu!(F::ParallelSparseLU,
                       A::Union{SparseMatrixCSC{mk_float,mk_int},Nothing})
    @serial_region begin
        lu!(F.lu_object, A)
        new_L = F.lu_object.L
        F.L.nzval .= new_L.nzval

        new_U = F.lu_object.U
        F.U.nzval .= new_U.nzval
    end
    _block_synchronize()
    return nothing
end

"""
    ldiv!(x::MPISharedArray{mk_float,1}, F::ParallelSparseLU,
          b::MPISharedArray{mk_float,1})

Solves `F\\b` overwriting `x`.
"""
function ldiv!(x::MPISharedArray{mk_float,1}, F::ParallelSparseLU,
               b::MPISharedArray{mk_float,1})
    @boundscheck (length(b) == F.n) || throw(DimensionMismatch("length(b)=$(length(b)), F.n=$(F.n)"))
    n = F.n
    l_colptr = F.L.colptr
    l_rowval = F.L.rowval
    l_nzval = F.L.nzval
    u_colptr = F.U.colptr
    u_rowval = F.U.rowval
    u_nzval = F.U.nzval
    wrk = F.wrk
    zero_wrk_range = F.zero_wrk_range
    forward_substitution_ranges = F.forward_substitution_ranges
    backward_substitution_ranges = F.backward_substitution_ranges

    for i ∈ zero_wrk_range
        wrk[i] = 0.0
    end

    @inbounds for i ∈ 1:n
        @serial_region begin
            wrk[i] += b[i]
        end
        _block_synchronize()
        for j ∈ forward_substitution_ranges[i]
            wrk[l_rowval[j]] -= l_nzval[j] * wrk[i]
        end
        # The following synchronization should not be needed because although the loop
        # just before might modify wrk[i+1], the root process will always get the first
        # element of the range, and is also the process that will do the `wrk[i] += b[i]`
        # on the next step of the loop.
        #_block_synchronize()
    end
    _block_synchronize()
    @inbounds for i ∈ n:-1:1
        @serial_region begin
            x[i] = u_nzval[u_colptr[i + 1] - 1] \ wrk[i]
        end
        _block_synchronize()
        for j ∈ backward_substitution_ranges[i]
            wrk[u_rowval[j]] -= u_nzval[j] * x[i]
        end
        _block_synchronize()
    end
    return x
end

end # sharedmem_lu_solver
