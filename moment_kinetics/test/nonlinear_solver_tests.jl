module NonlinearSolverTests

include("setup.jl")

using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.communication
using moment_kinetics.coordinates: coordinate
using moment_kinetics.looping
using moment_kinetics.looping: setup_loop_ranges!
using moment_kinetics.nonlinear_solvers
using moment_kinetics.type_definitions: mk_float, mk_int

using MPI
using LinearAlgebra
using SparseArrays

function linear_test()
    println("    - linear test")
    @testset "linear test $coord_names" for (coord_names, serial_solve) ∈ (((:z,), false), ((:vpa,), true))
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        #
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

        n = 16
        restart = 8
        max_restarts = 1
        atol = 1.0e-10

        irank_z, nrank_z, comm_sub_z, irank_r, nrank_r, comm_sub_r =
            setup_distributed_memory_MPI(1, 1, 1, 1)

        setup_loop_ranges!(block_rank[], block_size[]; s=1, sn=0, r=1, z=n, vperp=1, vpa=1,
                           vzeta=1, vr=1, vz=1)

        A = zeros(n,n)
        i = 1
        A[i,i] = -2.0
        A[i,i+1] = 1.0
        for i ∈ 2:n-1
            A[i,i-1] = 1.0
            A[i,i] = -2.0
            A[i,i+1] = 1.0
        end
        i = n
        A[i,i-1] = 1.0
        A[i,i] = -2.0

        z = collect(0:n-1) ./ (n-1)
        b = @. - z * (1.0 - z)

        if serial_solve
            coord_comm = MPI.COMM_NULL
        else
            coord_comm = comm_sub_z
        end
        the_coord = coordinate("foo", n, n, n, 1, 1, 1, 0, Cint(0), Cint(0), 1.0,
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0, 0), "", "", "", "", false, nothing,
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_int, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_float, 0, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), coord_comm, 1:n, 1:n,
                               zeros(mk_float, 0), zeros(mk_float, 0), "",
                               zeros(mk_float, 0), false, zeros(mk_float, 0, 0, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0))
        coords = NamedTuple(c => the_coord for c ∈ coord_names)

        function rhs_func!(residual, x; krylov=false)
            if serial_solve
                residual .= A * x - b
            else
                @begin_anyzv_region()
                @anyzv_serial_region begin
                    residual .= A * x - b
                end
            end
            return nothing
        end

        if serial_solve
            x = allocate_float(:vpa=>n)
            residual = allocate_float(:vpa=>n)
            delta_x = allocate_float(:vpa=>n)
            rhs_delta = allocate_float(:vpa=>n)
            v = allocate_float(:vpa=>n)
            w = allocate_float(:vpa=>n)

            x .= 0.0
            residual .= 0.0
            delta_x .= 0.0
            rhs_delta .= 0.0
            v .= 0.0
            w .= 0.0
        else
            x = allocate_shared_float(:z=>n)
            residual = allocate_shared_float(:z=>n)
            delta_x = allocate_shared_float(:z=>n)
            rhs_delta = allocate_shared_float(:z=>n)
            v = allocate_shared_float(:z=>n)
            w = allocate_shared_float(:z=>n)

            @begin_serial_region()
            @serial_region begin
                x .= 0.0
                residual .= 0.0
                delta_x .= 0.0
                rhs_delta .= 0.0
                v .= 0.0
                w .= 0.0
            end
        end

        nl_solver_params = setup_nonlinear_solve(
            (rtol=0.0, atol=atol, linear_restart=restart,
             linear_max_restarts=max_restarts, nonlinear_max_iterations=20,
             linear_rtol=1.0e-3, linear_atol=1.0, preconditioner_update_interval=300,
             total_its_soft_limit=50, adi_precon_iterations=1),
            coords; serial_solve=serial_solve, anyzv_region=!serial_solve)

        if !serial_solve
            @begin_r_anyzv_region()
        end
        newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, v, w, nl_solver_params;
                      coords)

        if serial_solve
            x_direct = A \ b

            @test isapprox(x, x_direct; atol=100.0*atol)
        else
            @begin_serial_region()
            @serial_region begin
                x_direct = A \ b

                @test isapprox(x, x_direct; atol=100.0*atol)
            end
        end
    end
end

function nonlinear_test()
    println("    - non-linear test")
    @testset "non-linear test, serial_solve=$serial_solve" for (coord_names, serial_solve) ∈ (((:z,), false), ((:vpa,), true))
        # Test represents diffusion with a coefficient proportional to the variable to the
        # power 5/2 (similar to collisional, parallel, thermal diffusion in Braginskii),
        # in 1D steady state, with a central finite-difference discretisation of the
        # second derivative.

        n = 16
        restart = 10
        max_restarts = 0
        atol = 1.0e-10

        irank_z, nrank_z, comm_sub_z, irank_r, nrank_r, comm_sub_r =
            setup_distributed_memory_MPI(1, 1, 1, 1)

        setup_loop_ranges!(block_rank[], block_size[]; s=1, sn=0, r=1, z=n, vperp=1, vpa=1,
                           vzeta=1, vr=1, vz=1)

        z = collect(0:n-1) ./ (n-1)
        b = @. - z * (1.0 - z)

        if serial_solve
            coord_comm = MPI.COMM_NULL
        else
            coord_comm = comm_sub_z
        end
        the_coord = coordinate("foo", n, n, n, 1, 1, 1, 0, Cint(0), Cint(0), 1.0,
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0, 0), "", "", "", "", false, nothing,
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_int, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_float, 0, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), coord_comm, 1:n, 1:n,
                               zeros(mk_float, 0), zeros(mk_float, 0), "",
                               zeros(mk_float, 0), false, zeros(mk_float, 0, 0, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0))
        coords = NamedTuple(c => the_coord for c ∈ coord_names)

        function rhs_func!(residual, x; krylov=false)
            if serial_solve
                i = 1
                D = abs(x[i])^2.5
                residual[i] = D * (- 2.0 * x[i] + x[i+1]) - b[i]
                for i ∈ 2:n-1
                    D = abs(x[i])^2.5
                    residual[i] = D * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
                end
                i = n
                D = abs(x[i])^2.5
                residual[i] = D * (x[i-1] - 2.0 * x[i]) - b[i]
            else
                @begin_anyzv_region()
                @anyzv_serial_region begin
                    i = 1
                    D = abs(x[i])^2.5
                    residual[i] = D * (- 2.0 * x[i] + x[i+1]) - b[i]
                    for i ∈ 2:n-1
                        D = abs(x[i])^2.5
                        residual[i] = D * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
                    end
                    i = n
                    D = abs(x[i])^2.5
                    residual[i] = D * (x[i-1] - 2.0 * x[i]) - b[i]
                end
            end
            return nothing
        end

        if serial_solve
            x = allocate_float(:vpa=>n)
            residual = allocate_float(:vpa=>n)
            delta_x = allocate_float(:vpa=>n)
            rhs_delta = allocate_float(:vpa=>n)
            v = allocate_float(:vpa=>n)
            w = allocate_float(:vpa=>n)
        else
            x = allocate_shared_float(:z=>n)
            residual = allocate_shared_float(:z=>n)
            delta_x = allocate_shared_float(:z=>n)
            rhs_delta = allocate_shared_float(:z=>n)
            v = allocate_shared_float(:z=>n)
            w = allocate_shared_float(:z=>n)
        end

        if serial_solve
            x .= 1.0
            residual .= 0.0
            delta_x .= 0.0
            rhs_delta .= 0.0
            v .= 0.0
            w .= 0.0
        else
            @begin_serial_region()
            @serial_region begin
                x .= 1.0
                residual .= 0.0
                delta_x .= 0.0
                rhs_delta .= 0.0
                v .= 0.0
                w .= 0.0
            end
        end

        nl_solver_params = setup_nonlinear_solve(
            (rtol=0.0, atol=atol, linear_restart=restart,
             linear_max_restarts=max_restarts, nonlinear_max_iterations=100,
             linear_rtol=1.0e-3, linear_atol=1.0, preconditioner_update_interval=300,
             total_its_soft_limit=50, adi_precon_iterations=1),
            coords; serial_solve=serial_solve, anyzv_region=!serial_solve)

        if !serial_solve
            @begin_r_anyzv_region()
        end
        newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, v, w, nl_solver_params;
                      coords)

        rhs_func!(residual, x)

        if serial_solve
            @test isapprox(residual, zeros(n); atol=4.0*atol)
        else
            @begin_serial_region()
            @serial_region begin
                @test isapprox(residual, zeros(n); atol=4.0*atol)
            end
        end
    end
end

function nonlinear_preconditioning_test()
    println("    - non-linear test")
    @testset "non-linear test, precon=$precon" for precon ∈ (:none, :left, :right)
        # Test represents diffusion with a coefficient proportional to the variable to the
        # power 5/2 (similar to collisional, parallel, thermal diffusion in Braginskii),
        # in 1D steady state, with a central finite-difference discretisation of the
        # second derivative.
        #
        # Just do this test with 'serial solve' because it makes implementing a dummy
        # preconditioner easier. This test checks the basic structure of the
        # implementation (e.g. the places where the preconditioner calls are made in the
        # nonlinear solver), not the parallelisation of any particular preconditioner.

        coord_names = (:vpa,)
        serial_solve = true
        n = 16
        restart = 10
        max_restarts = 0
        atol = 1.0e-10

        irank_z, nrank_z, comm_sub_z, irank_r, nrank_r, comm_sub_r =
            setup_distributed_memory_MPI(1, 1, 1, 1)

        setup_loop_ranges!(block_rank[], block_size[]; s=1, sn=0, r=1, z=n, vperp=1, vpa=1,
                           vzeta=1, vr=1, vz=1)

        z = collect(0:n-1) ./ (n-1)
        b = @. - z * (1.0 - z)

        if serial_solve
            coord_comm = MPI.COMM_NULL
        else
            coord_comm = comm_sub_z
        end
        the_coord = coordinate("foo", n, n, n, 1, 1, 1, 0, Cint(0), Cint(0), 1.0,
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0, 0), "", "", "", "", false, nothing,
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_int, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_float, 0, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), coord_comm, 1:n, 1:n,
                               zeros(mk_float, 0), zeros(mk_float, 0), "",
                               zeros(mk_float, 0), false, zeros(mk_float, 0, 0, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0))
        coords = NamedTuple(c => the_coord for c ∈ coord_names)

        function rhs_func!(residual, x; krylov=false)
            if serial_solve
                i = 1
                D = abs(x[i])^2.5
                residual[i] = D * (- 2.0 * x[i] + x[i+1]) - b[i]
                for i ∈ 2:n-1
                    D = abs(x[i])^2.5
                    residual[i] = D * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
                end
                i = n
                D = abs(x[i])^2.5
                residual[i] = D * (x[i-1] - 2.0 * x[i]) - b[i]
            else
                @begin_anyzv_region()
                @anyzv_serial_region begin
                    i = 1
                    D = abs(x[i])^2.5
                    residual[i] = D * (- 2.0 * x[i] + x[i+1]) - b[i]
                    for i ∈ 2:n-1
                        D = abs(x[i])^2.5
                        residual[i] = D * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
                    end
                    i = n
                    D = abs(x[i])^2.5
                    residual[i] = D * (x[i-1] - 2.0 * x[i]) - b[i]
                end
            end
            return nothing
        end

        if serial_solve
            x = allocate_float(:vpa=>n)
            residual = allocate_float(:vpa=>n)
            delta_x = allocate_float(:vpa=>n)
            rhs_delta = allocate_float(:vpa=>n)
            v = allocate_float(:vpa=>n)
            w = allocate_float(:vpa=>n)
        else
            x = allocate_shared_float(:z=>n)
            residual = allocate_shared_float(:z=>n)
            delta_x = allocate_shared_float(:z=>n)
            rhs_delta = allocate_shared_float(:z=>n)
            v = allocate_shared_float(:z=>n)
            w = allocate_shared_float(:z=>n)
        end

        if serial_solve
            x .= 1.0
            residual .= 0.0
            delta_x .= 0.0
            rhs_delta .= 0.0
            v .= 0.0
            w .= 0.0
        else
            @begin_serial_region()
            @serial_region begin
                x .= 1.0
                residual .= 0.0
                delta_x .= 0.0
                rhs_delta .= 0.0
                v .= 0.0
                w .= 0.0
            end
        end

        function get_diffusion_matrix(x)
            # This is not a particularly efficient way to construct the preconditioner
            # matrix, but this is only a test so simplicity is more important.

            M = zeros(n,n)

            # Dirichlet boundary conditions, so x at end points does not change.
            M[1,1] = 1.0
            M[end,end] = 1.0

            for i ∈ 2:n-1
                D = abs(x[i])^2.5
                M[i,i-1] = D
                M[i,i] = -2.0 * D
                M[i,i+1] = D
            end

            return sparse(M)
        end

        diffusion_matrix = get_diffusion_matrix(x)
        diffusion_lu = lu(diffusion_matrix)

        function recalculate_preconditioner()
            diffusion_matrix = get_diffusion_matrix(x)
            lu!(diffusion_lu, diffusion_matrix)
        end

        function preconditioner(x)
            ldiv!(diffusion_lu, x)
            return nothing
        end

        if precon === :none
            left_preconditioner = identity
            right_preconditioner = identity
            recalculate_preconditioner = nothing
        elseif precon === :left
            left_preconditioner = preconditioner
            right_preconditioner = identity
            recalculate_preconditioner = recalculate_preconditioner
        elseif precon === :right
            left_preconditioner = identity
            right_preconditioner = preconditioner
            recalculate_preconditioner = recalculate_preconditioner
        else
            error("Unrecognised value for precon=$precon.")
        end

        nl_solver_params = setup_nonlinear_solve(
            (rtol=0.0, atol=atol, linear_restart=restart,
             linear_max_restarts=max_restarts, nonlinear_max_iterations=100,
             linear_rtol=1.0e-3, linear_atol=1.0, preconditioner_update_interval=5,
             total_its_soft_limit=50, adi_precon_iterations=1),
            coords; serial_solve=serial_solve, anyzv_region=!serial_solve)

        if !serial_solve
            @begin_r_anyzv_region()
        end
        newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, v, w, nl_solver_params;
                      coords, left_preconditioner=left_preconditioner,
                      right_preconditioner=right_preconditioner,
                      recalculate_preconditioner=recalculate_preconditioner)

        rhs_func!(residual, x)

        if serial_solve
            @test isapprox(residual, zeros(n); atol=4.0*atol)
        else
            @begin_serial_region()
            @serial_region begin
                @test isapprox(residual, zeros(n); atol=4.0*atol)
            end
        end
    end
end

function runtests()
    @testset "non-linear solvers" begin
        println("non-linear solver tests")
        linear_test()
        nonlinear_test()
        nonlinear_preconditioning_test()
    end
end

end # NonlinearSolverTests

using .NonlinearSolverTests
NonlinearSolverTests.runtests()
