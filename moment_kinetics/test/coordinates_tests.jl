module CoordinatesTests

include("setup.jl")

using moment_kinetics.coordinates
using moment_kinetics.coordinates: define_test_coordinate

function runtests()
    @testset "coordinates" verbose=use_verbose begin
        println("coordinates tests")

        @testset "get_global_indices bc=$bc" for bc ∈ ("default", "periodic")
            # x.n = 5   x.n_global = 5
            # y.n = 10   y.n_global = 10
            # z.n = 17  z.n_global = 17
            if bc != "periodic"
                local_inds_1d = collect(1:17)
                local_inds_2d = vcat(local_inds_1d, local_inds_1d .+ 17,
                                     local_inds_1d .+ 2*17, local_inds_1d .+ 3*17,
                                     local_inds_1d .+ 4*17, local_inds_1d .+ 5*17,
                                     local_inds_1d .+ 6*17, local_inds_1d .+ 7*17,
                                     local_inds_1d .+ 8*17, local_inds_1d .+ 9*17)
                local_inds_3d = vcat(local_inds_2d, local_inds_2d .+ 10*17,
                                     local_inds_2d .+ 2*10*17, local_inds_2d .+ 3*10*17,
                                     local_inds_2d .+ 4*10*17)
            else
                local_inds_1d = collect(1:17)
                local_inds_1d[end] = 1
                local_inds_2d = vcat(local_inds_1d, local_inds_1d .+ 17,
                                     local_inds_1d .+ 2*17, local_inds_1d .+ 3*17,
                                     local_inds_1d .+ 4*17, local_inds_1d .+ 5*17,
                                     local_inds_1d .+ 6*17, local_inds_1d .+ 7*17,
                                     local_inds_1d .+ 8*17, local_inds_1d)
                local_inds_3d = vcat(local_inds_2d, local_inds_2d .+ 10*17,
                                     local_inds_2d .+ 2*10*17, local_inds_2d .+ 3*10*17,
                                     local_inds_2d)
            end
            @testset "local" begin
                x, _ = define_test_coordinate("x"; ngrid=3, nelement=2, L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false)
                y, _ = define_test_coordinate("y"; ngrid=4, nelement=3, L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false)
                z, _ = define_test_coordinate("z"; ngrid=5, nelement=4, L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false)

                # One variable
                @test get_global_indices((z,)) == local_inds_1d
                @test get_global_indices((z, y)) == local_inds_2d
                @test get_global_indices((z, y, x)) == local_inds_3d

                # Two identical variables
                @test get_global_indices(2, (z,)) == vcat(local_inds_1d, local_inds_1d .+ 17)
                @test get_global_indices(2, (z, y)) == vcat(local_inds_2d, local_inds_2d .+ 10*17)
                @test get_global_indices(2, (z, y, x)) == vcat(local_inds_3d, local_inds_3d .+ 5*10*17)

                # Three identical variables
                @test get_global_indices(3, (z,)) == vcat(local_inds_1d, local_inds_1d .+ 17, local_inds_1d .+ 2*17)
                @test get_global_indices(3, (z, y)) == vcat(local_inds_2d, local_inds_2d .+ 10*17, local_inds_2d .+ 2*10*17)
                @test get_global_indices(3, (z, y, x)) == vcat(local_inds_3d, local_inds_3d .+ 5*10*17, local_inds_3d .+ 2*5*10*17)

                # Two different variables
                @test get_global_indices((z,), (z, y)) == vcat(local_inds_1d, local_inds_2d .+ 17)
                @test get_global_indices((z, y), (z,)) == vcat(local_inds_2d, local_inds_1d .+ 10*17)
                @test get_global_indices((z,), (z, y, x)) == vcat(local_inds_1d, local_inds_3d .+ 17)
                @test get_global_indices((z, y, x), (z,)) == vcat(local_inds_3d, local_inds_1d .+ 5*10*17)
                @test get_global_indices((z, y), (z, y, x)) == vcat(local_inds_2d, local_inds_3d .+ 10*17)
                @test get_global_indices((z, y, x), (z, y)) == vcat(local_inds_3d, local_inds_2d .+ 5*10*17)
            end

            # x.n = 3   x.n_global = 5
            # y.n = 7   y.n_global = 13
            # z.n = 13  z.n_global = 25
            local_inds_1d = collect(1:13)
            local_inds_2d = vcat(local_inds_1d, local_inds_1d .+ 25,
                                 local_inds_1d .+ 2*25, local_inds_1d .+ 3*25,
                                 local_inds_1d .+ 4*25, local_inds_1d .+ 5*25,
                                 local_inds_1d .+ 6*25)
            local_inds_3d = vcat(local_inds_2d, local_inds_2d .+ 13*25,
                                 local_inds_2d .+ 2*13*25)
            if bc != "periodic"
                local_inds_1d_end = local_inds_1d
                local_inds_2d_yend = local_inds_2d
                local_inds_2d_zend = local_inds_2d
                local_inds_2d_zend_yend = local_inds_2d
                local_inds_3d_xend = local_inds_3d
                local_inds_3d_yend = local_inds_3d
                local_inds_3d_zend = local_inds_3d
                local_inds_3d_zend_yend = local_inds_3d
                local_inds_3d_yend_xend = local_inds_3d
                local_inds_3d_zend_xend = local_inds_3d
                local_inds_3d_zend_yend_xend = local_inds_3d
            else
                local_inds_1d_end = collect(1:13)
                local_inds_1d_end[end] = -11
                local_inds_2d_yend = vcat(local_inds_1d, local_inds_1d .+ 25,
                                          local_inds_1d .+ 2*25, local_inds_1d .+ 3*25,
                                          local_inds_1d .+ 4*25, local_inds_1d .+ 5*25,
                                          local_inds_1d .- 25*6)
                local_inds_2d_zend = vcat(local_inds_1d_end, local_inds_1d_end .+ 25,
                                          local_inds_1d_end .+ 2*25, local_inds_1d_end .+ 3*25,
                                          local_inds_1d_end .+ 4*25, local_inds_1d_end .+ 5*25,
                                          local_inds_1d_end .+ 6*25)
                local_inds_2d_zend_yend = vcat(local_inds_1d_end, local_inds_1d_end .+ 25,
                                               local_inds_1d_end .+ 2*25, local_inds_1d_end .+ 3*25,
                                               local_inds_1d_end .+ 4*25, local_inds_1d_end .+ 5*25,
                                               local_inds_1d_end .- 25*6)
                local_inds_3d_xend = vcat(local_inds_2d, local_inds_2d .+ 13*25,
                                          local_inds_2d .- 25*13*2)
                local_inds_3d_yend = vcat(local_inds_2d_yend, local_inds_2d_yend .+ 13*25,
                                          local_inds_2d_yend .+ 2*13*25)
                local_inds_3d_zend = vcat(local_inds_2d_zend, local_inds_2d_zend .+ 13*25,
                                          local_inds_2d_zend .+ 2*13*25)
                local_inds_3d_zend_yend = vcat(local_inds_2d_zend_yend, local_inds_2d_zend_yend .+ 13*25,
                                               local_inds_2d_zend_yend .+ 2*13*25)
                local_inds_3d_yend_xend = vcat(local_inds_2d_yend, local_inds_2d_yend .+ 13*25,
                                               local_inds_2d_yend .- 25*13*2)
                local_inds_3d_zend_xend = vcat(local_inds_2d_zend, local_inds_2d_zend .+ 13*25,
                                               local_inds_2d_zend .- 25*13*2)
                local_inds_3d_zend_yend_xend = vcat(local_inds_2d_zend_yend, local_inds_2d_zend_yend .+ 13*25,
                                                    local_inds_2d_zend_yend .- 25*13*2)
            end
            @testset "2 subdomains per coord ($irankx,$iranky,$irankz)" for (irankx, iranky, irankz, inds_1d, inds_2d, inds_3d) ∈ (
                         (0, 0, 0, local_inds_1d, local_inds_2d, local_inds_3d),
                         (1, 0, 0, local_inds_1d, local_inds_2d, local_inds_3d_xend .+ 25*13*2),
                         (0, 1, 0, local_inds_1d, local_inds_2d_yend .+ 25*6, local_inds_3d_yend .+ 25*6),
                         (0, 0, 1, local_inds_1d_end .+ 12, local_inds_2d_zend .+ 12, local_inds_3d_zend .+ 12),
                         (1, 1, 0, local_inds_1d, local_inds_2d_yend .+ 25*6, local_inds_3d_yend_xend .+ 25*6 .+ 25*13*2),
                         (1, 0, 1, local_inds_1d_end .+ 12, local_inds_2d_zend .+ 12, local_inds_3d_zend_xend .+ 12 .+ 25*13*2),
                         (0, 1, 1, local_inds_1d_end .+ 12, local_inds_2d_zend_yend .+ 12 .+ 25*6, local_inds_3d_zend_yend .+ 12 .+ 25*6),
                         (1, 1, 1, local_inds_1d_end .+ 12, local_inds_2d_zend_yend .+ 12 .+ 25*6, local_inds_3d_zend_yend_xend .+ 12 .+ 25*6 .+ 25*13*2),
                        )
                # Name all coordinates "z" so that `define_coordinate()` thinks that they
                # should support distributed MPI.
                x, _ = define_test_coordinate("z"; ngrid=3, nelement=2, nelement_local=1,
                                              L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false,
                                              irank=irankx, nrank=2)
                y, _ = define_test_coordinate("z"; ngrid=4, nelement=4, nelement_local=2,
                                              L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false,
                                              irank=iranky, nrank=2)
                z, _ = define_test_coordinate("z"; ngrid=5, nelement=6, nelement_local=3,
                                              L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false,
                                              irank=irankz, nrank=2)

                # One variable
                @test get_global_indices((z,)) == inds_1d
                @test get_global_indices((z, y)) == inds_2d
                @test get_global_indices((z, y, x)) == inds_3d

                # Two identical variables
                @test get_global_indices(2, (z,)) == vcat(inds_1d, inds_1d .+ 25)
                @test get_global_indices(2, (z, y)) == vcat(inds_2d, inds_2d .+ 25*13)
                @test get_global_indices(2, (z, y, x)) == vcat(inds_3d, inds_3d .+ 25*13*5)

                # Three identical variables
                @test get_global_indices(3, (z,)) == vcat(inds_1d, inds_1d .+ 25, inds_1d .+ 2*25)
                @test get_global_indices(3, (z, y)) == vcat(inds_2d, inds_2d .+ 25*13, inds_2d .+ 2*25*13)
                @test get_global_indices(3, (z, y, x)) == vcat(inds_3d, inds_3d .+ 25*13*5, inds_3d .+ 2*25*13*5)

                # Two different variables
                @test get_global_indices((z,), (z, y)) == vcat(inds_1d, inds_2d .+ 25)
                @test get_global_indices((z, y), (z,)) == vcat(inds_2d, inds_1d .+ 25*13)
                @test get_global_indices((z,), (z, y, x)) == vcat(inds_1d, inds_3d .+ 25)
                @test get_global_indices((z, y, x), (z,)) == vcat(inds_3d, inds_1d .+ 25*13*5)
                @test get_global_indices((z, y), (z, y, x)) == vcat(inds_2d, inds_3d .+ 25*13)
                @test get_global_indices((z, y, x), (z, y)) == vcat(inds_3d, inds_2d .+ 25*13*5)
            end

            # x.n = 3   x.n_global = 7
            # y.n = 7   y.n_global = 19
            # z.n = 13  z.n_global = 37
            local_inds_1d = collect(1:13)
            local_inds_2d = vcat(local_inds_1d, local_inds_1d .+ 37,
                                 local_inds_1d .+ 2*37, local_inds_1d .+ 3*37,
                                 local_inds_1d .+ 4*37, local_inds_1d .+ 5*37,
                                 local_inds_1d .+ 6*37)
            local_inds_3d = vcat(local_inds_2d, local_inds_2d .+ 19*37,
                                 local_inds_2d .+ 2*19*37)
            if bc != "periodic"
                local_inds_1d_end = local_inds_1d
                local_inds_2d_yend = local_inds_2d
                local_inds_2d_zend = local_inds_2d
                local_inds_2d_zend_yend = local_inds_2d
                local_inds_3d_xend = local_inds_3d
                local_inds_3d_yend = local_inds_3d
                local_inds_3d_zend = local_inds_3d
                local_inds_3d_zend_yend = local_inds_3d
                local_inds_3d_yend_xend = local_inds_3d
                local_inds_3d_zend_xend = local_inds_3d
                local_inds_3d_zend_yend_xend = local_inds_3d
            else
                local_inds_1d_end = collect(1:13)
                local_inds_1d_end[end] = -23
                local_inds_2d_yend = vcat(local_inds_1d, local_inds_1d .+ 37,
                                          local_inds_1d .+ 2*37, local_inds_1d .+ 3*37,
                                          local_inds_1d .+ 4*37, local_inds_1d .+ 5*37,
                                          local_inds_1d .- 37*12)
                local_inds_2d_zend = vcat(local_inds_1d_end, local_inds_1d_end .+ 37,
                                          local_inds_1d_end .+ 2*37, local_inds_1d_end .+ 3*37,
                                          local_inds_1d_end .+ 4*37, local_inds_1d_end .+ 5*37,
                                          local_inds_1d_end .+ 6*37)
                local_inds_2d_zend_yend = vcat(local_inds_1d_end, local_inds_1d_end .+ 37,
                                               local_inds_1d_end .+ 2*37, local_inds_1d_end .+ 3*37,
                                               local_inds_1d_end .+ 4*37, local_inds_1d_end .+ 5*37,
                                               local_inds_1d_end .- 37*12)
                local_inds_3d_zend = vcat(local_inds_2d_zend, local_inds_2d_zend .+ 19*37,
                                          local_inds_2d_zend .+ 2*19*37)
                local_inds_3d_yend = vcat(local_inds_2d_yend, local_inds_2d_yend .+ 19*37,
                                          local_inds_2d_yend .+ 2*19*37)
                local_inds_3d_xend = vcat(local_inds_2d, local_inds_2d .+ 19*37,
                                          local_inds_2d .- 37*19*4)
                local_inds_3d_zend_yend = vcat(local_inds_2d_zend_yend, local_inds_2d_zend_yend .+ 19*37,
                                               local_inds_2d_zend_yend .+ 2*19*37)
                local_inds_3d_yend_xend = vcat(local_inds_2d_yend, local_inds_2d_yend .+ 19*37,
                                               local_inds_2d_yend .- 37*19*4)
                local_inds_3d_zend_xend = vcat(local_inds_2d_zend, local_inds_2d_zend .+ 19*37,
                                               local_inds_2d_zend .- 37*19*4)
                local_inds_3d_zend_yend_xend = vcat(local_inds_2d_zend_yend, local_inds_2d_zend_yend .+ 19*37,
                                                    local_inds_2d_zend_yend .- 37*19*4)
            end
            @testset "3 subdomains per coord ($irankx,$iranky,$irankz)" for (irankx, iranky, irankz, inds_1d, inds_2d, inds_3d) ∈ (
                         (0, 0, 0, local_inds_1d, local_inds_2d, local_inds_3d),
                         (1, 0, 0, local_inds_1d, local_inds_2d, local_inds_3d .+ 37*19*2),
                         (0, 1, 0, local_inds_1d, local_inds_2d .+ 37*6, local_inds_3d .+ 37*6),
                         (0, 0, 1, local_inds_1d .+ 12, local_inds_2d .+ 12, local_inds_3d .+ 12),
                         (1, 1, 0, local_inds_1d, local_inds_2d .+ 37*6, local_inds_3d .+ 37*6 .+ 37*19*2),
                         (1, 0, 1, local_inds_1d .+ 12, local_inds_2d .+ 12, local_inds_3d .+ 12 .+ 37*19*2),
                         (0, 1, 1, local_inds_1d .+ 12, local_inds_2d .+ 12 .+ 37*6, local_inds_3d .+ 12 .+ 37*6),
                         (1, 1, 1, local_inds_1d .+ 12, local_inds_2d .+ 12 .+ 37*6, local_inds_3d .+ 12 .+ 37*6 .+ 37*19*2),
                         (2, 0, 0, local_inds_1d, local_inds_2d, local_inds_3d_xend .+ 37*19*4),
                         (0, 2, 0, local_inds_1d, local_inds_2d_yend .+ 37*12, local_inds_3d_yend .+ 37*12),
                         (0, 0, 2, local_inds_1d_end .+ 24, local_inds_2d_zend .+ 24, local_inds_3d_zend .+ 24),
                         (2, 2, 0, local_inds_1d, local_inds_2d_yend .+ 37*12, local_inds_3d_yend_xend .+ 37*12 .+ 37*19*4),
                         (2, 0, 2, local_inds_1d_end .+ 24, local_inds_2d_zend .+ 24, local_inds_3d_zend_xend .+ 24 .+ 37*19*4),
                         (0, 2, 2, local_inds_1d_end .+ 24, local_inds_2d_zend_yend .+ 24 .+ 37*12, local_inds_3d_zend_yend .+ 24 .+ 37*12),
                         (2, 2, 2, local_inds_1d_end .+ 24, local_inds_2d_zend_yend .+ 24 .+ 37*12, local_inds_3d_zend_yend_xend .+ 24 .+ 37*12 .+ 37*19*4),
                         (2, 1, 0, local_inds_1d, local_inds_2d .+ 37*6, local_inds_3d_xend .+ 37*6 .+ 37*19*4),
                         (2, 0, 1, local_inds_1d .+ 12, local_inds_2d .+ 12, local_inds_3d_xend .+ 12 .+ 37*19*4),
                         (1, 2, 0, local_inds_1d, local_inds_2d_yend .+ 37*12, local_inds_3d_yend .+ 37*12 .+ 37*19*2),
                         (0, 2, 1, local_inds_1d .+ 12, local_inds_2d_yend .+ 12 .+ 37*12, local_inds_3d_yend .+ 12 .+ 37*12),
                         (1, 0, 2, local_inds_1d_end .+ 24, local_inds_2d_zend .+ 24, local_inds_3d_zend .+ 24 .+ 37*19*2),
                         (0, 1, 2, local_inds_1d_end .+ 24, local_inds_2d_zend .+ 24 .+ 37*6, local_inds_3d_zend .+ 24 .+ 37*6),
                         (2, 2, 1, local_inds_1d .+ 12, local_inds_2d_yend .+ 12 .+ 37*12, local_inds_3d_yend_xend .+ 12 .+ 37*12 .+ 37*19*4),
                         (2, 1, 2, local_inds_1d_end .+ 24, local_inds_2d_zend .+ 24 .+ 37*6, local_inds_3d_zend_xend .+ 24 .+ 37*6 .+ 37*19*4),
                         (1, 2, 2, local_inds_1d_end .+ 24, local_inds_2d_zend_yend .+ 24 .+ 37*12, local_inds_3d_zend_yend .+ 24 .+ 37*12 .+ 37*19*2),
                         (2, 1, 1, local_inds_1d .+ 12, local_inds_2d .+ 12 .+ 37*6, local_inds_3d_xend .+ 12 .+ 37*6 .+ 37*19*4),
                         (1, 2, 1, local_inds_1d .+ 12, local_inds_2d_yend .+ 12 .+ 37*12, local_inds_3d_yend .+ 12 .+ 37*12 .+ 37*19*2),
                         (1, 1, 2, local_inds_1d_end .+ 24, local_inds_2d_zend .+ 24 .+ 37*6, local_inds_3d_zend .+ 24 .+ 37*6 .+ 37*19*2),
                        )
                # Name all coordinates "z" so that `define_coordinate()` thinks that they
                # should support distributed MPI.
                x, _ = define_test_coordinate("z"; ngrid=3, nelement=3, nelement_local=1,
                                              L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false,
                                              irank=irankx, nrank=3)
                y, _ = define_test_coordinate("z"; ngrid=4, nelement=6, nelement_local=2,
                                              L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false,
                                              irank=iranky, nrank=3)
                z, _ = define_test_coordinate("z"; ngrid=5, nelement=9, nelement_local=3,
                                              L=1.0, bc=bc,
                                              discretization="chebyshev_pseudospectral",
                                              collision_operator_dim=false,
                                              irank=irankz, nrank=3)

                # One variable
                @test get_global_indices((z,)) == inds_1d
                @test get_global_indices((z, y)) == inds_2d
                @test get_global_indices((z, y, x)) == inds_3d

                # Two identical variables
                @test get_global_indices(2, (z,)) == vcat(inds_1d, inds_1d .+ 37)
                @test get_global_indices(2, (z, y)) == vcat(inds_2d, inds_2d .+ 37*19)
                @test get_global_indices(2, (z, y, x)) == vcat(inds_3d, inds_3d .+ 37*19*7)

                # Three identical variables
                @test get_global_indices(3, (z,)) == vcat(inds_1d, inds_1d .+ 37, inds_1d .+ 2*37)
                @test get_global_indices(3, (z, y)) == vcat(inds_2d, inds_2d .+ 37*19, inds_2d .+ 2*37*19)
                @test get_global_indices(3, (z, y, x)) == vcat(inds_3d, inds_3d .+ 37*19*7, inds_3d .+ 2*37*19*7)

                # Two different variables
                @test get_global_indices((z,), (z, y)) == vcat(inds_1d, inds_2d .+ 37)
                @test get_global_indices((z, y), (z,)) == vcat(inds_2d, inds_1d .+ 37*19)
                @test get_global_indices((z,), (z, y, x)) == vcat(inds_1d, inds_3d .+ 37)
                @test get_global_indices((z, y, x), (z,)) == vcat(inds_3d, inds_1d .+ 37*19*7)
                @test get_global_indices((z, y), (z, y, x)) == vcat(inds_2d, inds_3d .+ 37*19)
                @test get_global_indices((z, y, x), (z, y)) == vcat(inds_3d, inds_2d .+ 37*19*7)
            end
        end
    end
end

end # CoordinatesTests


using .CoordinatesTests

CoordinatesTests.runtests()
