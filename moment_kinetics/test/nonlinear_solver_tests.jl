module NonlinearSolverTests

include("setup.jl")

using moment_kinetics.array_allocation: allocate_shared_float
using moment_kinetics.communication
using moment_kinetics.coordinates: coordinate
using moment_kinetics.input_structs: advection_input
using moment_kinetics.looping
using moment_kinetics.looping: setup_loop_ranges!
using moment_kinetics.nonlinear_solvers
using moment_kinetics.type_definitions: mk_float, mk_int

using MPI

function linear_test()
    println("    - linear test")
    @testset "linear test $coord_names" for coord_names ∈ ((:z,), (:vpa,))
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

        the_coord = coordinate("foo", n, n, n, 1, 1, 1, 0, 1.0, zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_int, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_int, 0, 0),
                               "", "", "", "", zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0, 0),
                               advection_input("", 0.0, 0.0, 0.0), zeros(mk_float, 0),
                               zeros(mk_float, 0), MPI.COMM_NULL, 1:n, 1:n,
                               zeros(mk_float, 0), zeros(mk_float, 0), "",
                               zeros(mk_float, 0), false, zeros(mk_float, 0, 0, 0),
                               zeros(mk_float, 0, 0))
        coords = NamedTuple(c => the_coord for c ∈ coord_names)

        function rhs_func!(residual, x)
            begin_serial_region()
            @serial_region begin
                residual .= A * x - b
            end
            return nothing
        end

        x = allocate_shared_float(n)
        residual = allocate_shared_float(n)
        delta_x = allocate_shared_float(n)
        rhs_delta = allocate_shared_float(n)
        v = allocate_shared_float(n)
        w = allocate_shared_float(n)

        begin_serial_region()
        @serial_region begin
            x .= 0.0
            residual .= 0.0
            delta_x .= 0.0
            rhs_delta .= 0.0
            v .= 0.0
            w .= 0.0
        end

        nl_solver_params = setup_nonlinear_solve(
            Dict{String,Any}("nonlinear_solver" =>
                             Dict{String,Any}("rtol" => 0.0,
                                              "atol" => atol,
                                              "linear_restart" => restart,
                                              "linear_max_restarts" => max_restarts)),
            coords)

        newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, v, w, nl_solver_params;
                      coords)

        begin_serial_region()
        @serial_region begin
            x_direct = A \ b

            @test isapprox(x, x_direct; atol=100.0*atol)
        end
    end
end

function nonlinear_test()
    println("    - non-linear test")
    @testset "non-linear test" for coord_names ∈ ((:z,), (:vpa,))
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

        the_coord = coordinate("foo", n, n, n, 1, 1, 1, 0, 1.0, zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_int, 0), zeros(mk_int, 0),
                               zeros(mk_int, 0), zeros(mk_int, 0), zeros(mk_int, 0, 0),
                               "", "", "", "", zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0), zeros(mk_float, 0),
                               zeros(mk_float, 0, 0), zeros(mk_float, 0, 0),
                               advection_input("", 0.0, 0.0, 0.0), zeros(mk_float, 0),
                               zeros(mk_float, 0), MPI.COMM_NULL, 1:n, 1:n,
                               zeros(mk_float, 0), zeros(mk_float, 0), "",
                               zeros(mk_float, 0), false, zeros(mk_float, 0, 0, 0),
                               zeros(mk_float, 0, 0))
        coords = NamedTuple(c => the_coord for c ∈ coord_names)

        function rhs_func!(residual, x)
            begin_serial_region()
            @serial_region begin
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
            return nothing
        end

        x = allocate_shared_float(n)
        residual = allocate_shared_float(n)
        delta_x = allocate_shared_float(n)
        rhs_delta = allocate_shared_float(n)
        v = allocate_shared_float(n)
        w = allocate_shared_float(n)

        begin_serial_region()
        @serial_region begin
            x .= 1.0
            residual .= 0.0
            delta_x .= 0.0
            rhs_delta .= 0.0
            v .= 0.0
            w .= 0.0
        end

        nl_solver_params = setup_nonlinear_solve(
            Dict{String,Any}("nonlinear_solver" =>
                             Dict{String,Any}("rtol" => 0.0,
                                              "atol" => atol,
                                              "linear_restart" => restart,
                                              "linear_max_restarts" => max_restarts,
                                              "nonlinear_max_iterations" => 100)),
            coords)

        newton_solve!(x, rhs_func!, residual, delta_x, rhs_delta, v, w, nl_solver_params;
                      coords)

        rhs_func!(residual, x)

        begin_serial_region()
        @serial_region begin
            @test isapprox(residual, zeros(n); atol=4.0*atol)
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
