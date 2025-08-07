module LoopSetupTests

include("setup.jl")

using moment_kinetics.looping: dims_string, get_splits, get_max_work,
                               get_best_ranges, get_best_split_from_sizes,
                               get_ranges_from_split, get_best_anysv_split,
                               get_anysv_ranges, get_best_anyzv_split, get_anyzv_ranges,
                               debug_setup_loop_ranges_split_one_combination!,
                               loop_ranges_store
using moment_kinetics.communication

function test_low_level_utils()
    @testset "dims_string" begin
        @test dims_string((:z,:y,:x)) == "z_y_x"
        @test dims_string((:a,:b)) == "a_b"
        @test dims_string((:i,:j,:k,:l,:m,:n)) == "i_j_k_l_m_n"
    end
    @testset "get_splits" begin
        @test get_splits(1, 4) == [[1, 1, 1, 1]]
        @test get_splits(6, 2) == [[1, 6], [2, 3], [3, 2], [6, 1],
                                   [1, 5], [5, 1],
                                   [1, 4], [2, 2], [4, 1],
                                   [1, 3], [3, 1],
                                   [1, 2], [2, 1],
                                   [1, 1]]
        @test get_splits(8, 3) == [[1, 1, 8], [1, 2, 4], [1, 4, 2], [1, 8, 1],
                                   [2, 1, 4], [2, 2, 2], [2, 4, 1], [4, 1, 2],
                                   [4, 2, 1], [8, 1, 1],
                                   [1, 1, 7], [1, 7, 1], [7, 1, 1],
                                   [1, 1, 6], [1, 2, 3], [1, 3, 2], [1, 6, 1],
                                   [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1],
                                   [6, 1, 1],
                                   [1, 1, 5], [1, 5, 1], [5, 1, 1],
                                   [1, 1, 4], [1, 2, 2], [1, 4, 1], [2, 1, 2],
                                   [2, 2, 1], [4, 1, 1],
                                   [1, 1, 3], [1, 3, 1], [3, 1, 1],
                                   [1, 1, 2], [1, 2, 1], [2, 1, 1],
                                   [1, 1, 1]]
    end
    @testset "get_max_work" begin
        @test isapprox(get_max_work([1], [4]), 4)
        @test isapprox(get_max_work([2,2], [4,6]), 6)
        @test isapprox(get_max_work([3,2], [4,5]), 6)
    end

    return nothing
end

function test_get_best_ranges()
    @testset "get_best_ranges_from_sizes" begin
        function get_best_ranges_from_sizes(block_rank, block_size, dim_sizes_list)
            # Define this function for testing, since it was split in two in
            # looping.jl
            best_split = get_best_split_from_sizes(block_size, dim_sizes_list)
            effective_block_size = prod(best_split)
            return get_ranges_from_split(block_rank, effective_block_size, best_split,
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

        @test get_best_ranges_from_sizes(0, 7, [3,4,5]) == [1:1, 1:2, 1:5]
        @test get_best_ranges_from_sizes(1, 7, [3,4,5]) == [1:1, 3:4, 1:5]
        @test get_best_ranges_from_sizes(2, 7, [3,4,5]) == [2:2, 1:2, 1:5]
        @test get_best_ranges_from_sizes(3, 7, [3,4,5]) == [2:2, 3:4, 1:5]
        @test get_best_ranges_from_sizes(4, 7, [3,4,5]) == [3:3, 1:2, 1:5]
        @test get_best_ranges_from_sizes(5, 7, [3,4,5]) == [3:3, 3:4, 1:5]
        @test get_best_ranges_from_sizes(6, 7, [3,4,5]) == [1:0, 1:0, 1:0]

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

        @test get_best_ranges(0, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>1:1, :r=>1:2, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                 :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_best_ranges(1, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>1:1, :r=>3:4, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                 :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_best_ranges(2, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>2:2, :r=>1:2, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                 :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_best_ranges(3, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>2:2, :r=>3:4, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                 :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_best_ranges(4, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>3:3, :r=>1:2, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                 :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_best_ranges(5, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>3:3, :r=>3:4, :z=>1:5, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
                 :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_best_ranges(6, 7, (:s,:r,:z),
                              Dict(:s=>3, :r=>4, :z=>5, :sn=>1, :vperp=>2, :vpa=>7,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
            Dict(:s=>1:0, :r=>1:0, :z=>1:0, :sn=>1:1, :vperp=>1:2, :vpa=>1:7,
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

    return nothing
end

function test_anysv()
    @testset "get_best_anysv_split" begin
        @test get_best_anysv_split(1, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,1,1]
        @test get_best_anysv_split(2, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,2,1]
        @test get_best_anysv_split(3, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,3,1]
        @test get_best_anysv_split(4, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,2,2]
        @test get_best_anysv_split(5, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,2,2]
        @test get_best_anysv_split(6, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,3,2]
        @test get_best_anysv_split(7, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,3,2]
        @test get_best_anysv_split(8, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,2,4]
        @test get_best_anysv_split(9, Dict(:s=>1, :r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,2,4]
        @test get_best_anysv_split(2, Dict(:s=>1, :r=>1, :z=>3, :vperp=>2, :vpa=>2)) == [1,1,2]
        # Splitting the v-space dimension would be slightly more efficient for the
        # following case if parallelisation was perfect, but because 'anysv' is used
        # for the collision operator, some parts of which do not parallelise over
        # velocity space, we penalise splitting the velocity space in favour of
        # species or spatial dimensions if there is an 'efficient enough' arrangement
        # of processes that minimises the number of processes used for velocity space.
        @test get_best_anysv_split(2, Dict(:s=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2)) == [1,2,1]
    end
    @testset "get_anysv_ranges" begin
        @test get_anysv_ranges(0, [1, 1, 1], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 1, 1], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 1, 1], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 1, 1], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

        @test get_anysv_ranges(0, [1, 2, 1], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 2, 1], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 2, 1], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 2, 1], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 1], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 1], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 1], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 1], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

        @test get_anysv_ranges(0, [1, 2, 2], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 2, 2], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 2, 2], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [1, 2, 2], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 2], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 2], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 2], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [1, 2, 2], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [1, 2, 2], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [1, 2, 2], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [1, 2, 2], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [1, 2, 2], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [1, 2, 2], (:anysv,),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [1, 2, 2], (:anysv,:vperp),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [1, 2, 2], (:anysv,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [1, 2, 2], (:anysv,:vperp,:vpa),
                              Dict(:s=>1, :sn=>1, :r=>1, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(0, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(1, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(2, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(3, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(4, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(5, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(6, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(7, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(8, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(9, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(10, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(11, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(12, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(13, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(14, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(15, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>1:2, :z=>4:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(16, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(17, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(18, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(19, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(20, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(21, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(22, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(23, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(24, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(25, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(26, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(27, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(28, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:0, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(29, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(30, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:s),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:s,:vperp),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:s,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anysv_ranges(31, [2, 2, 8], (:anysv,:s,:vperp,:vpa),
                              Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                   :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>2:2, :sn=>1:1, :r=>3:5, :z=>4:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
    end

    return nothing
end

function test_anyzv()
    @testset "get_best_anyzv_split" begin
        @test get_best_anyzv_split(1, Dict(:r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,1]
        @test get_best_anyzv_split(2, Dict(:r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,2]
        @test get_best_anyzv_split(3, Dict(:r=>1, :z=>2, :vperp=>2, :vpa=>2)) == [1,3]
        @test get_best_anyzv_split(1, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [1,1]
        @test get_best_anyzv_split(2, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [2,1]
        @test get_best_anyzv_split(3, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [3,1]
        @test get_best_anyzv_split(4, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [2,2]
        @test get_best_anyzv_split(5, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [5,1]
        @test get_best_anyzv_split(6, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [2,3]
        @test get_best_anyzv_split(7, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [7,1]
        @test get_best_anyzv_split(8, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [2,4]
        @test get_best_anyzv_split(9, Dict(:r=>3, :z=>2, :vperp=>2, :vpa=>2)) == [3,3]
        @test get_best_anyzv_split(6, Dict(:r=>4, :z=>3, :vperp=>2, :vpa=>2)) == [3,2]
    end
    @testset "get_anyzv_ranges" begin
        @test get_anyzv_ranges(0, [1, 1], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [1, 1], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [1, 1], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [1, 1], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

        @test get_anyzv_ranges(0, [2, 1], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [2, 1], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [2, 1], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [2, 1], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 1], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 1], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 1], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 1], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

        @test get_anyzv_ranges(0, [2, 2], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [2, 2], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [2, 2], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [2, 2], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 2], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 2], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 2], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [2, 2], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [2, 2], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [2, 2], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [2, 2], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [2, 2], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [2, 2], (:anyzv,),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [2, 2], (:anyzv,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [2, 2], (:anyzv,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [2, 2], (:anyzv,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>3, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)

        @test get_anyzv_ranges(0, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(0, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:1, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(1, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>2:2, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(2, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>3:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(3, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:4, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>7:9, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>7:9, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(4, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>5:5, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>7:9, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>7:9, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(5, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>6:6, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>10:12, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>10:12, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(6, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>1:1, :z=>1:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>7:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>10:12, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>10:12, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(7, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>1:1, :z=>4:6, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(8, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:1, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(9, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>2:2, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(10, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>3:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(11, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:4, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>7:9, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>7:9, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(12, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>5:5, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>7:9, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>7:9, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(13, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>6:6, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>10:12, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>10:12, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(14, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>2:2, :z=>1:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>7:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>10:12, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>10:12, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(15, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>2:2, :z=>4:6, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(16, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:1, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(17, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>2:2, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(18, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>3:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(19, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:4, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>7:9, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>7:9, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(20, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>5:5, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>7:9, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>7:9, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(21, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>6:6, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>10:12, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>10:12, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(22, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>3:3, :z=>1:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>7:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>10:12, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>10:12, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(23, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>3:3, :z=>4:6, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(24, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:1, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(25, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>2:2, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(26, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>3:3, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(27, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>1:3, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:4, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>7:9, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>7:9, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(28, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>1:1, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:0, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>5:5, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>7:9, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>7:9, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(29, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>1:1, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>6:6, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>10:12, :vperp=>1:1, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>10:12, :vperp=>1:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(30, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>2:2, :vpa=>1:1,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:0, :vperp=>1:0, :vpa=>1:0,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:vperp),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:vperp,:vpa),
                               Dict(:s=>2, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:2, :sn=>1:1, :r=>4:4, :z=>1:7, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:z),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>7, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>7:7, :vperp=>1:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:z,:vperp),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>10:12, :vperp=>2:2, :vpa=>1:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:z,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>12, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>10:12, :vperp=>1:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
        @test get_anyzv_ranges(31, [4, 8], (:anyzv,:z,:vperp,:vpa),
                               Dict(:s=>1, :sn=>1, :r=>5, :z=>6, :vperp=>2, :vpa=>2,
                                    :vzeta=>11, :vr=>13, :vz=>17)) ==
              Dict(:s=>1:1, :sn=>1:1, :r=>4:4, :z=>4:6, :vperp=>2:2, :vpa=>2:2,
                   :vzeta=>1:11, :vr=>1:13, :vz=>1:17)
    end

    return nothing
end

function test_debug_setup_loop_ranges_split_one_combination!()
    @testset "debug_setup_loop_ranges_split_one_combination" begin
        # Need to set comm_block[] to avoid MPI errors when creating 'anysv' or
        # 'anyzv' communicators in debug_setup_loop_ranges_split_one_combination!()
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
    end

    return nothing
end

function test_debug_setup_loop_ranges_split_one_combination_anysv()
    @testset "debug_setup_loop_ranges_split_one_combination_anysv" begin
        # Need to set comm_block[] to avoid MPI errors when creating 'anysv' or
        # 'anyzv' communicators in debug_setup_loop_ranges_split_one_combination!()
        comm_block[] = comm_world

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anysv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anysv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anysv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anysv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vpa), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vpa), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anysv, :s, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anysv, :s, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anysv, :s, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anysv, :s, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vperp), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vperp), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anysv, :s, :vperp), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anysv, :s, :vperp), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anysv, :s, :vperp), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anysv, :s, :vperp), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vperp, :vpa), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vperp, :vpa), :s, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anysv, :s, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anysv, :s, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anysv, :s, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anysv, :s, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anysv, :s, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anysv, :s, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anysv, :s, :vperp, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anysv, :s, :vperp, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anysv, :s, :vperp, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anysv, :s, :vperp, :vpa), :s, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anysv, :s, :vperp, :vpa), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anysv, :s, :vperp, :vpa), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anysv, :s, :vperp, :vpa), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anysv, :s, :vperp, :vpa), :s, :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:2
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:7
        @test loop_ranges_store[(:anysv,)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            3, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 1:1
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            4, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            5, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            6, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            7, 8, (:anysv, :s, :vperp, :vpa), :s, :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anysv,)].s == 1:0
        @test loop_ranges_store[(:anysv,)].r == 1:3
        @test loop_ranges_store[(:anysv,)].z == 1:4
        @test loop_ranges_store[(:anysv,)].vperp == 1:0
        @test loop_ranges_store[(:anysv,)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s)].r == 1:3
        @test loop_ranges_store[(:anysv,:s)].z == 1:4
        @test loop_ranges_store[(:anysv,:s)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:vperp,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vpa,)].s == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anysv,:s,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa,)].s == 2:2
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anysv,:s,:vperp,:vpa)].vpa == 6:11
    end

    return nothing
end

function test_debug_setup_loop_ranges_split_one_combination_anyzv()
    @testset "debug_setup_loop_ranges_split_one_combination_anyzv" begin
        # Need to set comm_block[] to avoid MPI errors when creating 'anysv' or
        # 'anyzv' communicators in debug_setup_loop_ranges_split_one_combination!()
        comm_block[] = comm_world

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z), :z, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z), :z, s=2, r=3, z=4, sn=5, vperp=7, vpa=11, vzeta=13,
            vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anyzv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anyzv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anyzv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anyzv, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vperp), :vperp, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vperp), :z, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vperp), :z, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anyzv, :z, :vperp), :z, :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anyzv, :z, :vperp), :z, :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anyzv, :z, :vperp), :z, :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anyzv, :z, :vperp), :z, :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vpa), :z, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vpa), :z, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anyzv, :z, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anyzv, :z, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anyzv, :z, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:5

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0


        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anyzv, :z, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7, vpa=11,
            vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 6:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:0

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vperp, :vpa), :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vperp, :vpa), :vperp, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 2, (:anyzv, :z, :vperp, :vpa), :z, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 2, (:anyzv, :z, :vperp, :vpa), :z, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anyzv, :z, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anyzv, :z, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anyzv, :z, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anyzv, :z, :vperp, :vpa), :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anyzv, :z, :vperp, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anyzv, :z, :vperp, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anyzv, :z, :vperp, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anyzv, :z, :vperp, :vpa), :z, :vpa, s=2, r=3, z=4, sn=5, vperp=7,
            vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 4, (:anyzv, :z, :vperp, :vpa), :z, :vperp, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            1, 4, (:anyzv, :z, :vperp, :vpa), :z, :vperp, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 4, (:anyzv, :z, :vperp, :vpa), :z, :vperp, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            3, 4, (:anyzv, :z, :vperp, :vpa), :z, :vperp, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:11

        debug_setup_loop_ranges_split_one_combination!(
            0, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:4
        @test loop_ranges_store[(:anyzv,)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:4
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:7
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:11

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            1, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            2, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            3, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            4, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            5, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11

        debug_setup_loop_ranges_split_one_combination!(
            6, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 1:5

        debug_setup_loop_ranges_split_one_combination!(
            7, 8, (:anyzv, :z, :vperp, :vpa), :z, :vperp, :vpa, s=2, r=3, z=4, sn=5,
            vperp=7, vpa=11, vzeta=13, vr=17, vz=19)

        @test loop_ranges_store[(:anyzv,)].s == 1:2
        @test loop_ranges_store[(:anyzv,)].r == 1:3
        @test loop_ranges_store[(:anyzv,)].z == 1:0
        @test loop_ranges_store[(:anyzv,)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:vperp,:vpa,)].s == 1:2
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:vperp,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vperp)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vpa)].z == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vperp == 1:0
        @test loop_ranges_store[(:anyzv,:z,:vpa)].vpa == 1:0

        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].s == 1:2
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].r == 1:3
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].z == 3:4
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vperp == 4:7
        @test loop_ranges_store[(:anyzv,:z,:vperp,:vpa)].vpa == 6:11
    end

    return nothing
end

function runtests()
    @testset "loop setup" verbose=use_verbose begin
        println("loop setup tests")
        test_low_level_utils()
        test_get_best_ranges()
        test_anysv()
        test_anyzv()
        test_debug_setup_loop_ranges_split_one_combination!()
        test_debug_setup_loop_ranges_split_one_combination_anysv()
        test_debug_setup_loop_ranges_split_one_combination_anyzv()
    end

    return nothing
end

end # LoopSetupTests


using .LoopSetupTests

LoopSetupTests.runtests()
