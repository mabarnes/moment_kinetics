"""
Provide interface to the MUMPS package, if it is installed
"""
module mumps_lu_ext

using MUMPS

"""
"""
const global_mumps_store = Mumps{mk_float}[]

function get_mumps_instance(comm, matrix_type=mumps_unsymmetric, icntl=get_icntl(),
                            cntl64=default_cntl64)

    # MUMPS is written in Fortran and wants a 'Fortran communicator', so we use the
    # low-level function MPI_Comm_f2c() to get one. This function was only recently added
    # to MPI.jl (https://github.com/JuliaParallel/MPI.jl/pull/798), and the addition of
    # 'Fortran conversions' may not be complete yet
    # (https://github.com/JuliaParallel/MPI.jl/issues/784), so watch out for API changes
    # in MPI.jl that might affect this!
    fortran_comm = MPI.API.MPI_Comm_f2c(comm.val)

    m = Mumps{mk_float}(matrix_type, icntl, cntl64; comm=fortran_comm)
    push!(global_mumps_store, m)
    return m
end

function cleanup_mumps_instances()
    while length(global_mumps_store) > 0
        m = pop!(global_mumps_store)
        finalize(m)
    end
    return nothing
end

end # mumps_lu_ext
