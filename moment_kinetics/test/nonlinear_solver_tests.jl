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
            true,
            OptionsDict("nonlinear_solver" =>
                        OptionsDict("rtol" => 0.0,
                                    "atol" => atol,
                                    "linear_restart" => restart,
                                    "linear_max_restarts" => max_restarts)),
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
    @testset "non-linear test" for (coord_names, serial_solve) ∈ (((:z,), false), ((:vpa,), true))
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        #
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

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
            true,
            OptionsDict("nonlinear_solver" =>
                        OptionsDict("rtol" => 0.0,
                                    "atol" => atol,
                                    "linear_restart" => restart,
                                    "linear_max_restarts" => max_restarts,
                                    "nonlinear_max_iterations" => 100)),
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

function runtests()
    @testset "non-linear solvers" begin
        println("non-linear solver tests")
        linear_test()
        nonlinear_test()
    end
end

end # NonlinearSolverTests

using .NonlinearSolverTests
NonlinearSolverTests.runtests()
