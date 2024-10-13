module SparseLUTests

include("setup.jl")

using moment_kinetics.array_allocation: allocate_shared_float
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.sparse_lu_solver

using LinearAlgebra
using MPI
using SparseArrays

# Get a sparse test matrix with a structure similar to a finite element derivative
function test_matrix(nel=6,ngr=5)
    n = nel*(ngr-1)+1
    mat = zeros(n,n)
    for iel in 1:nel
        imin = (iel-1) * (ngr-1) + 1
        imax = iel * (ngr-1) + 1
        mat[imin:imax,imin:imax] .= rand(ngr,ngr)
    end
    return sparse(mat)
end

function runtests()

    atol = 1.0e-10

    # Set up to use only shared-memory MPI
    setup_distributed_memory_MPI(1, 1, 1, 1)
    setup_loop_ranges!(block_rank[], block_size[]; s=1, sn=0, r=1, z=1, vpa=1, vperp=1, vzeta=1, vr=1, vz=1)

    @testset "sparse_lu" verbose=use_verbose begin
        println("sparse_lu tests")

        @testset "dense matrix" begin
            n = 42

            if block_rank[] == 0
                A = rand(n,n)
                A_sparse = sparse(A)
            else
                A = nothing
                A_sparse = nothing
            end
            A_lu = sparse_lu(A_sparse)

            # Create rhs
            b = allocate_shared_float(n)
            @serial_region begin
                b .= rand(n)
            end
            x = allocate_shared_float(n)
            _block_synchronize()

            ldiv!(x, A_lu, b)

            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end

            # Check we can update the rhs
            @serial_region begin
                b .= rand(n)
            end
            _block_synchronize()

            ldiv!(x, A_lu, b)
            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end

            # Check we can update the matrix
            if block_rank[] == 0
                A = rand(n,n)
                A_sparse = sparse(A)
            else
                A = nothing
                A_sparse = nothing
            end
            sparse_lu!(A_lu, A_sparse)

            # Create rhs
            @serial_region begin
                b .= rand(n)
            end
            _block_synchronize()

            ldiv!(x, A_lu, b)

            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end

            # Check we can update the rhs again
            @serial_region begin
                b .= rand(n)
            end
            _block_synchronize()

            ldiv!(x, A_lu, b)
            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end
        end

        @testset "sparse matrix" begin
            if block_rank[] == 0
                A = test_matrix()
            else
                A = nothing
            end

            this_length = Ref(0)
            if block_rank[] == 0
                this_length[] = size(A, 1)
                MPI.Bcast!(this_length, comm_block[])
            else
                MPI.Bcast!(this_length, comm_block[])
            end
            n = this_length[]

            A_lu = sparse_lu(A)

            # Create rhs
            b = allocate_shared_float(n)
            @serial_region begin
                b .= rand(n)
            end
            x = allocate_shared_float(n)
            _block_synchronize()

            ldiv!(x, A_lu, b)

            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end

            # Check we can update the rhs
            @serial_region begin
                b .= rand(n)
            end
            _block_synchronize()

            ldiv!(x, A_lu, b)
            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end

            # Check we can update the matrix
            if block_rank[] == 0
                A = test_matrix()
            else
                A = nothing
            end
            sparse_lu!(A_lu, A)

            # Create rhs
            @serial_region begin
                b .= rand(n)
            end
            _block_synchronize()

            ldiv!(x, A_lu, b)

            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end

            # Check we can update the rhs again
            @serial_region begin
                b .= rand(n)
            end
            _block_synchronize()

            ldiv!(x, A_lu, b)
            @serial_region begin
                @test isapprox(x, A \ b, atol=atol)
            end
        end
    end
end

end # SparseLUTests


using .SparseLUTests

SparseLUTests.runtests()
