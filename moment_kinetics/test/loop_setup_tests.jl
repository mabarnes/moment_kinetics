module LoopSetupTests

include("setup.jl")

using moment_kinetics.looping: dims_string, get_splits, get_load_balance,
                               get_best_ranges, get_best_split_from_sizes,
                               get_ranges_from_split, get_best_anyv_split,
                               debug_setup_loop_ranges_split_one_combination!,
                               loop_ranges_store
using moment_kinetics.communication

function runtests()
    @testset "loop setup" verbose=use_verbose begin
        println("loop setup tests")
        @testset "dims_string" begin
            @test dims_string((:z,:y,:x)) == "z_y_x"
            @test dims_string((:a,:b)) == "a_b"
            @test dims_string((:i,:j,:k,:l,:m,:n)) == "i_j_k_l_m_n"
        end
        @testset "get_splits" begin
            @test get_splits(1, 4) == [[1, 1, 1, 1]]
            @test get_splits(6, 2) == [[1, 6],
                                       [2, 3],
                                       [3, 2],
                                       [6, 1]]
            @test get_splits(8, 3) == [[1, 1, 8],
                                       [1, 2, 4],
                                       [1, 4, 2],
                                       [1, 8, 1],
                                       [2, 1, 4],
                                       [2, 2, 2],
                                       [2, 4, 1],
                                       [4, 1, 2],
                                       [4, 2, 1],
                                       [8, 1, 1]]
        end
        @testset "get_load_balance" begin
            @test isapprox(get_load_balance([1], [4]), 1.0; atol=1.e-8)
            @test isapprox(get_load_balance([2,2], [4,6]), 1.0; atol=1.e-8)
            @test isapprox(get_load_balance([3,2], [4,5]), (2.0*3.0)/(1.0*2.0); atol=1.e-8)
        end
        @testset "get_best_ranges_from_sizes" begin
            function get_best_ranges_from_sizes(block_rank, block_size, dim_sizes_list)
                # Define this function for testing, since it was split in two in
                # looping.jl
                best_split = get_best_split_from_sizes(block_size, dim_sizes_list)
                return get_ranges_from_split(block_rank, block_size, best_split,
                                             dim_sizes_list)
            end
            @test get_best_ranges_from_sizes(0, 1, [3,4,5]) == [1:3, 1:4, 1:5]

            @test get_best_ranges_from_sizes(0, 2, [3,4,5]) == [1:3, 1:2, 1:5]
            @test get_best_ranges_from_sizes(1, 2, [3,4,5]) == [1:3, 3:4, 1:5]

            @test get_best_ranges_from_sizes(0, 3, [3,4,5]) == [1:1, 1:4, 1:5]
            @test get_best_ranges_from_sizes(1, 3, [3,4,5]) == [2:2, 1:4, 1:5]
            @test get_best_ranges_from_sizes(2, 3, [3,4,5]) == [3:3, 1:4, 1:5]

            @test get_best_ranges_from_sizes(0, 4, [3,4,5]) == [1:3, 1:1, 1:5]
            @test get_best_ranges_from_sizes(1, 4, [3,4,5]) == [1:3, 2:2, 1:5]
            @test get_best_ranges_from_sizes(2, 4, [3,4,5]) == [1:3, 3:3, 1:5]
            @test get_best_ranges_from_sizes(3, 4, [3,4,5]) == [1:3, 4:4, 1:5]

            @test get_best_ranges_from_sizes(0, 5, [3,4,5]) == [1:3, 1:4, 1:1]
            @test get_best_ranges_from_sizes(1, 5, [3,4,5]) == [1:3, 1:4, 2:2]
            @test get_best_ranges_from_sizes(2, 5, [3,4,5]) == [1:3, 1:4, 3:3]
            @test get_best_ranges_from_sizes(3, 5, [3,4,5]) == [1:3, 1:4, 4:4]
            @test get_best_ranges_from_sizes(4, 5, [3,4,5]) == [1:3, 1:4, 5:5]

            @test get_best_ranges_from_sizes(0, 6, [3,4,5]) == [1:1, 1:2, 1:5]
            @test get_best_ranges_from_sizes(1, 6, [3,4,5]) == [1:1, 3:4, 1:5]
            @test get_best_ranges_from_sizes(2, 6, [3,4,5]) == [2:2, 1:2, 1:5]
            @test get_best_ranges_from_sizes(3, 6, [3,4,5]) == [2:2, 3:4, 1:5]
            @test get_best_ranges_from_sizes(4, 6, [3,4,5]) == [3:3, 1:2, 1:5]
            @test get_best_ranges_from_sizes(5, 6, [3,4,5]) == [3:3, 3:4, 1:5]

            @test get_best_ranges_from_sizes(0, 7, [3,4,5]) == [1:3, 1:4, 1:0]
            @test get_best_ranges_from_sizes(1, 7, [3,4,5]) == [1:3, 1:4, 1:0]
            @test get_best_ranges_from_sizes(2, 7, [3,4,5]) == [1:3, 1:4, 1:1]
            @test get_best_ranges_from_sizes(3, 7, [3,4,5]) == [1:3, 1:4, 2:2]
            @test get_best_ranges_from_sizes(4, 7, [3,4,5]) == [1:3, 1:4, 3:3]
            @test get_best_ranges_from_sizes(5, 7, [3,4,5]) == [1:3, 1:4, 4:4]
            @test get_best_ranges_from_sizes(6, 7, [3,4,5]) == [1:3, 1:4, 5:5]

            @test get_best_ranges_from_sizes(0, 8, [3,4,5]) == [1:3, 1:1, 1:2]
            @test get_best_ranges_from_sizes(1, 8, [3,4,5]) == [1:3, 1:1, 3:5]
            @test get_best_ranges_from_sizes(2, 8, [3,4,5]) == [1:3, 2:2, 1:2]
            @test get_best_ranges_from_sizes(3, 8, [3,4,5]) == [1:3, 2:2, 3:5]
            @test get_best_ranges_from_sizes(4, 8, [3,4,5]) == [1:3, 3:3, 1:2]
            @test get_best_ranges_from_sizes(5, 8, [3,4,5]) == [1:3, 3:3, 3:5]
            @test get_best_ranges_from_sizes(6, 8, [3,4,5]) == [1:3, 4:4, 1:2]
            @test get_best_ranges_from_sizes(7, 8, [3,4,5]) == [1:3, 4:4, 3:5]

            @test get_best_ranges_from_sizes(0, 9, [3,4,5]) == [1:1, 1:4, 1:1]
            @test get_best_ranges_from_sizes(1, 9, [3,4,5]) == [1:1, 1:4, 2:3]
            @test get_best_ranges_from_sizes(2, 9, [3,4,5]) == [1:1, 1:4, 4:5]
            @test get_best_ranges_from_sizes(3, 9, [3,4,5]) == [2:2, 1:4, 1:1]
            @test get_best_ranges_from_sizes(4, 9, [3,4,5]) == [2:2, 1:4, 2:3]
            @test get_best_ranges_from_sizes(5, 9, [3,4,5]) == [2:2, 1:4, 4:5]
            @test get_best_ranges_from_sizes(6, 9, [3,4,5]) == [3:3, 1:4, 1:1]
            @test get_best_ranges_from_sizes(7, 9, [3,4,5]) == [3:3, 1:4, 2:3]
            @test get_best_ranges_from_sizes(8, 9, [3,4,5]) == [3:3, 1:4, 4:5]

            @test get_best_ranges_from_sizes(0, 10, [3,4,5]) == [1:3, 1:2, 1:1]
            @test get_best_ranges_from_sizes(1, 10, [3,4,5]) == [1:3, 1:2, 2:2]
            @test get_best_ranges_from_sizes(2, 10, [3,4,5]) == [1:3, 1:2, 3:3]
            @test get_best_ranges_from_sizes(3, 10, [3,4,5]) == [1:3, 1:2, 4:4]
            @test get_best_ranges_from_sizes(4, 10, [3,4,5]) == [1:3, 1:2, 5:5]
            @test get_best_ranges_from_sizes(5, 10, [3,4,5]) == [1:3, 3:4, 1:1]
            @test get_best_ranges_from_sizes(6, 10, [3,4,5]) == [1:3, 3:4, 2:2]
            @test get_best_ranges_from_sizes(7, 10, [3,4,5]) == [1:3, 3:4, 3:3]
            @test get_best_ranges_from_sizes(8, 10, [3,4,5]) == [1:3, 3:4, 4:4]
            @test get_best_ranges_from_sizes(9, 10, [3,4,5]) == [1:3, 3:4, 5:5]

            # Check that outer-most loop gets parallelised if load balance is the same
            # either way
            @test get_best_ranges_from_sizes(0, 2, [2, 2]) == [1:1, 1:2]
            @test get_best_ranges_from_sizes(1, 2, [2, 2]) == [2:2, 1:2]
        end
        @testset "get_best_ranges" begin
            @test get_best_ranges(0, 1, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>1:4, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

            @test get_best_ranges(0, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>1:1, :z=>1:2, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(1, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>1:1, :z=>3:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(2, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>2:2, :z=>1:2, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(3, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>2:2, :z=>3:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(4, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>3:3, :z=>1:2, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(5, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>3:3, :z=>3:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(6, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>4:4, :z=>1:2, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
            @test get_best_ranges(7, 8, (:s,:r,:z),
                                  Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                       :vzeta=>11, :vr=>13, :vz=>17)) ==
                Dict(:s=>1:3, :r=>4:4, :z=>3:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                     :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        end
        @testset "get_best_anyv_split" begin
            @test get_best_anyv_split(1, Dict(:s=>1, :r=>1, :z=>2, :vpa=>2, :vperp=>2)) == [1,1,1,1]
            @test get_best_anyv_split(2, Dict(:s=>1, :r=>1, :z=>2, :vpa=>2, :vperp=>2)) == [1,1,2,1]
            @test get_best_anyv_split(3, Dict(:s=>1, :r=>1, :z=>2, :vpa=>2, :vperp=>2)) == [1,1,3,1]
            @test get_best_anyv_split(4, Dict(:s=>1, :r=>1, :z=>2, :vpa=>2, :vperp=>2)) == [1,1,2,2]
            @test get_best_anyv_split(2, Dict(:s=>1, :r=>1, :z=>3, :vpa=>2, :vperp=>2)) == [1,1,1,2]
            # Splitting the v-space dimension would be slightly more efficient for the
            # following case if parallelisation was perfect, but because 'anyv' is used
            # for the collision operator, some parts of which do not parallelise over
            # velocity space, we penalise splitting the velocity space in favour of
            # species or spatial dimensions if there is an 'efficient enough' arrangement
            # of processes that minimises the number of processes used for velocity space.
            @test get_best_anyv_split(2, Dict(:s=>1, :r=>1, :z=>7, :vpa=>2, :vperp=>2)) == [1,1,2,1]
        end
        @testset "debug_setup_loop_ranges_split_one_combination" begin
            # Need to set comm_block[] to avoid MPI errors when creating 'anyv'
            # communicator in debug_setup_loop_ranges_split_one_combination!()
            comm_block[] = comm_world

            debug_setup_loop_ranges_split_one_combination!(
                0, 2, (:s, :z), :s; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:2
            @test loop_ranges_store[()].r == 1:3
            @test loop_ranges_store[()].z == 1:4

            @test loop_ranges_store[(:s,)].s == 1:2
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:3
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:4

            @test loop_ranges_store[(:s,:r)].s == 1:2
            @test loop_ranges_store[(:s,:r)].r == 1:3
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 1:1
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:4

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 1:3
            @test loop_ranges_store[(:r,:z)].z == 1:4

            @test loop_ranges_store[(:s,:r,:z)].s == 1:2
            @test loop_ranges_store[(:s,:r,:z)].r == 1:3
            @test loop_ranges_store[(:s,:r,:z)].z == 1:4

            debug_setup_loop_ranges_split_one_combination!(
                1, 2, (:s, :z), :s; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:0
            @test loop_ranges_store[()].r == 1:0
            @test loop_ranges_store[()].z == 1:0

            @test loop_ranges_store[(:s,)].s == 1:0
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:0
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:0

            @test loop_ranges_store[(:s,:r)].s == 1:0
            @test loop_ranges_store[(:s,:r)].r == 1:0
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 2:2
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:4

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 1:0
            @test loop_ranges_store[(:r,:z)].z == 1:0

            @test loop_ranges_store[(:s,:r,:z)].s == 1:0
            @test loop_ranges_store[(:s,:r,:z)].r == 1:0
            @test loop_ranges_store[(:s,:r,:z)].z == 1:0

            debug_setup_loop_ranges_split_one_combination!(
                0, 4, (:r, :z), :r, :z; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:2
            @test loop_ranges_store[()].r == 1:3
            @test loop_ranges_store[()].z == 1:4

            @test loop_ranges_store[(:s,)].s == 1:2
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:3
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:4

            @test loop_ranges_store[(:s,:r)].s == 1:2
            @test loop_ranges_store[(:s,:r)].r == 1:3
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 1:2
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:4

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 1:1
            @test loop_ranges_store[(:r,:z)].z == 1:2

            @test loop_ranges_store[(:s,:r,:z)].s == 1:2
            @test loop_ranges_store[(:s,:r,:z)].r == 1:3
            @test loop_ranges_store[(:s,:r,:z)].z == 1:4

            debug_setup_loop_ranges_split_one_combination!(
                1, 4, (:r, :z), :r, :z; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:0
            @test loop_ranges_store[()].r == 1:0
            @test loop_ranges_store[()].z == 1:0

            @test loop_ranges_store[(:s,)].s == 1:0
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:0
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:0

            @test loop_ranges_store[(:s,:r)].s == 1:0
            @test loop_ranges_store[(:s,:r)].r == 1:0
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 1:0
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:0

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 1:1
            @test loop_ranges_store[(:r,:z)].z == 3:4

            @test loop_ranges_store[(:s,:r,:z)].s == 1:0
            @test loop_ranges_store[(:s,:r,:z)].r == 1:0
            @test loop_ranges_store[(:s,:r,:z)].z == 1:0

            debug_setup_loop_ranges_split_one_combination!(
                2, 4, (:r, :z), :r, :z; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:0
            @test loop_ranges_store[()].r == 1:0
            @test loop_ranges_store[()].z == 1:0

            @test loop_ranges_store[(:s,)].s == 1:0
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:0
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:0

            @test loop_ranges_store[(:s,:r)].s == 1:0
            @test loop_ranges_store[(:s,:r)].r == 1:0
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 1:0
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:0

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 2:3
            @test loop_ranges_store[(:r,:z)].z == 1:2

            @test loop_ranges_store[(:s,:r,:z)].s == 1:0
            @test loop_ranges_store[(:s,:r,:z)].r == 1:0
            @test loop_ranges_store[(:s,:r,:z)].z == 1:0

            debug_setup_loop_ranges_split_one_combination!(
                3, 4, (:r, :z), :r, :z; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:0
            @test loop_ranges_store[()].r == 1:0
            @test loop_ranges_store[()].z == 1:0

            @test loop_ranges_store[(:s,)].s == 1:0
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:0
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:0

            @test loop_ranges_store[(:s,:r)].s == 1:0
            @test loop_ranges_store[(:s,:r)].r == 1:0
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 1:0
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:0

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 2:3
            @test loop_ranges_store[(:r,:z)].z == 3:4

            @test loop_ranges_store[(:s,:r,:z)].s == 1:0
            @test loop_ranges_store[(:s,:r,:z)].r == 1:0
            @test loop_ranges_store[(:s,:r,:z)].z == 1:0

            debug_setup_loop_ranges_split_one_combination!(
                3, 8, (:s, :r, :z), :s, :r, :z; s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[()].s == 1:0
            @test loop_ranges_store[()].r == 1:0
            @test loop_ranges_store[()].z == 1:0

            @test loop_ranges_store[(:s,)].s == 1:0
            @test loop_ranges_store[(:s,)].r == 1:3
            @test loop_ranges_store[(:s,)].z == 1:4

            @test loop_ranges_store[(:r,)].s == 1:2
            @test loop_ranges_store[(:r,)].r == 1:0
            @test loop_ranges_store[(:r,)].z == 1:4

            @test loop_ranges_store[(:z,)].s == 1:2
            @test loop_ranges_store[(:z,)].r == 1:3
            @test loop_ranges_store[(:z,)].z == 1:0

            @test loop_ranges_store[(:s,:r)].s == 1:0
            @test loop_ranges_store[(:s,:r)].r == 1:0
            @test loop_ranges_store[(:s,:r)].z == 1:4

            @test loop_ranges_store[(:s,:z)].s == 1:0
            @test loop_ranges_store[(:s,:z)].r == 1:3
            @test loop_ranges_store[(:s,:z)].z == 1:0

            @test loop_ranges_store[(:r,:z)].s == 1:2
            @test loop_ranges_store[(:r,:z)].r == 1:0
            @test loop_ranges_store[(:r,:z)].z == 1:0

            @test loop_ranges_store[(:s,:r,:z)].s == 1:1
            @test loop_ranges_store[(:s,:r,:z)].r == 2:3
            @test loop_ranges_store[(:s,:r,:z)].z == 3:4

            debug_setup_loop_ranges_split_one_combination!(
                0, 2, (:anyv, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:7
            @test loop_ranges_store[(:anyv,)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:5

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:11

            debug_setup_loop_ranges_split_one_combination!(
                1, 2, (:anyv, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
                vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 6:11

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:0

            debug_setup_loop_ranges_split_one_combination!(
                0, 2, (:anyv, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
                vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:7
            @test loop_ranges_store[(:anyv,)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:11

            debug_setup_loop_ranges_split_one_combination!(
                1, 2, (:anyv, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
                vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 4:7
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:0

            debug_setup_loop_ranges_split_one_combination!(
                0, 2, (:anyv, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
                vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:7
            @test loop_ranges_store[(:anyv,)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:5

            debug_setup_loop_ranges_split_one_combination!(
                1, 2, (:anyv, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
                vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 6:11

            debug_setup_loop_ranges_split_one_combination!(
                0, 2, (:anyv, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
                vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:7
            @test loop_ranges_store[(:anyv,)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:11

            debug_setup_loop_ranges_split_one_combination!(
                1, 2, (:anyv, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
                vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 4:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:11

            debug_setup_loop_ranges_split_one_combination!(
                0, 4, (:anyv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
                vpa=11, vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:7
            @test loop_ranges_store[(:anyv,)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:7
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:11

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:5

            debug_setup_loop_ranges_split_one_combination!(
                1, 4, (:anyv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
                vpa=11, vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 6:11

            debug_setup_loop_ranges_split_one_combination!(
                2, 4, (:anyv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
                vpa=11, vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 4:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 1:5

            debug_setup_loop_ranges_split_one_combination!(
                3, 4, (:anyv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
                vpa=11, vzeta=13, vr=17, vz=19)

            @test loop_ranges_store[(:anyv,)].s == 1:2
            @test loop_ranges_store[(:anyv,)].r == 1:3
            @test loop_ranges_store[(:anyv,)].z == 1:4
            @test loop_ranges_store[(:anyv,)].vperp == 1:0
            @test loop_ranges_store[(:anyv,)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vperp)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vpa)].vperp == 1:0
            @test loop_ranges_store[(:anyv,:vpa)].vpa == 1:0

            @test loop_ranges_store[(:anyv,:vperp,:vpa,)].s == 1:2
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].r == 1:3
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].z == 1:4
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vperp == 4:7
            @test loop_ranges_store[(:anyv,:vperp,:vpa)].vpa == 6:11
        end
    end
end

end # LoopSetupTests


using .LoopSetupTests

LoopSetupTests.runtests()
