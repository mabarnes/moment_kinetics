module LoopSetupTests

include("setup.jl")

using moment_kinetics.looping: dims_string, dims_string_underscores,
      loop_range_names, get_splits, get_load_balance, get_best_ranges,
      get_best_ranges_from_sizes

function runtests()
    @testset "loop setup" verbose=use_verbose begin
        println("loop setup tests")
        @testset "dims_string" begin
            @test dims_string((:z,:y,:x)) == "zyx"
            @test dims_string((:a,:b)) == "ab"
            @test dims_string((:i,:j,:k,:l,:m,:n)) == "ijklmn"
        end
        @testset "dims_string_underscores" begin
            @test dims_string_underscores((:z,:y,:x)) == "z_y_x"
            @test dims_string_underscores((:a,:b)) == "a_b"
            @test dims_string_underscores((:i,:j,:k,:l,:m,:n)) == "i_j_k_l_m_n"
        end
        @testset "loop_range_names" begin
            @test loop_range_names((:z,:y,:x)) == Dict(:z=>:z_range_zyx,
                                                       :y=>:y_range_zyx,
                                                       :x=>:x_range_zyx)
            @test loop_range_names((:a,:b)) == Dict(:a=>:a_range_ab,
                                                    :b=>:b_range_ab)
            @test loop_range_names((:i,:j,:k,:l,:m,:n)) ==
                Dict(:i=>:i_range_ijklmn, :j=>:j_range_ijklmn,
                     :k=>:k_range_ijklmn, :l=>:l_range_ijklmn,
                     :m=>:m_range_ijklmn, :n=>:n_range_ijklmn)
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
            @test get_load_balance([1], [4]) == 1.0
            @test get_load_balance([2,2], [4,6]) == 1.0
            @test get_load_balance([3,2], [4,5]) == (2.0*3.0)/(1.0*2.0)
        end
        @testset "get_best_ranges_from_sizes" begin
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
            @test get_best_ranges_from_sizes(1, 6, [3,4,5]) == [2:2, 1:2, 1:5]
            @test get_best_ranges_from_sizes(2, 6, [3,4,5]) == [3:3, 1:2, 1:5]
            @test get_best_ranges_from_sizes(3, 6, [3,4,5]) == [1:1, 3:4, 1:5]
            @test get_best_ranges_from_sizes(4, 6, [3,4,5]) == [2:2, 3:4, 1:5]
            @test get_best_ranges_from_sizes(5, 6, [3,4,5]) == [3:3, 3:4, 1:5]

            @test get_best_ranges_from_sizes(0, 7, [3,4,5]) == [1:3, 1:4, 1:0]
            @test get_best_ranges_from_sizes(1, 7, [3,4,5]) == [1:3, 1:4, 1:0]
            @test get_best_ranges_from_sizes(2, 7, [3,4,5]) == [1:3, 1:4, 1:1]
            @test get_best_ranges_from_sizes(3, 7, [3,4,5]) == [1:3, 1:4, 2:2]
            @test get_best_ranges_from_sizes(4, 7, [3,4,5]) == [1:3, 1:4, 3:3]
            @test get_best_ranges_from_sizes(5, 7, [3,4,5]) == [1:3, 1:4, 4:4]
            @test get_best_ranges_from_sizes(6, 7, [3,4,5]) == [1:3, 1:4, 5:5]

            @test get_best_ranges_from_sizes(0, 8, [3,4,5]) == [1:3, 1:1, 1:2]
            @test get_best_ranges_from_sizes(1, 8, [3,4,5]) == [1:3, 2:2, 1:2]
            @test get_best_ranges_from_sizes(2, 8, [3,4,5]) == [1:3, 3:3, 1:2]
            @test get_best_ranges_from_sizes(3, 8, [3,4,5]) == [1:3, 4:4, 1:2]
            @test get_best_ranges_from_sizes(4, 8, [3,4,5]) == [1:3, 1:1, 3:5]
            @test get_best_ranges_from_sizes(5, 8, [3,4,5]) == [1:3, 2:2, 3:5]
            @test get_best_ranges_from_sizes(6, 8, [3,4,5]) == [1:3, 3:3, 3:5]
            @test get_best_ranges_from_sizes(7, 8, [3,4,5]) == [1:3, 4:4, 3:5]

            @test get_best_ranges_from_sizes(0, 9, [3,4,5]) == [1:1, 1:4, 1:1]
            @test get_best_ranges_from_sizes(1, 9, [3,4,5]) == [2:2, 1:4, 1:1]
            @test get_best_ranges_from_sizes(2, 9, [3,4,5]) == [3:3, 1:4, 1:1]
            @test get_best_ranges_from_sizes(3, 9, [3,4,5]) == [1:1, 1:4, 2:3]
            @test get_best_ranges_from_sizes(4, 9, [3,4,5]) == [2:2, 1:4, 2:3]
            @test get_best_ranges_from_sizes(5, 9, [3,4,5]) == [3:3, 1:4, 2:3]
            @test get_best_ranges_from_sizes(6, 9, [3,4,5]) == [1:1, 1:4, 4:5]
            @test get_best_ranges_from_sizes(7, 9, [3,4,5]) == [2:2, 1:4, 4:5]
            @test get_best_ranges_from_sizes(8, 9, [3,4,5]) == [3:3, 1:4, 4:5]

            @test get_best_ranges_from_sizes(0, 10, [3,4,5]) == [1:3, 1:2, 1:1]
            @test get_best_ranges_from_sizes(1, 10, [3,4,5]) == [1:3, 3:4, 1:1]
            @test get_best_ranges_from_sizes(2, 10, [3,4,5]) == [1:3, 1:2, 2:2]
            @test get_best_ranges_from_sizes(3, 10, [3,4,5]) == [1:3, 3:4, 2:2]
            @test get_best_ranges_from_sizes(4, 10, [3,4,5]) == [1:3, 1:2, 3:3]
            @test get_best_ranges_from_sizes(5, 10, [3,4,5]) == [1:3, 3:4, 3:3]
            @test get_best_ranges_from_sizes(6, 10, [3,4,5]) == [1:3, 1:2, 4:4]
            @test get_best_ranges_from_sizes(7, 10, [3,4,5]) == [1:3, 3:4, 4:4]
            @test get_best_ranges_from_sizes(8, 10, [3,4,5]) == [1:3, 1:2, 5:5]
            @test get_best_ranges_from_sizes(9, 10, [3,4,5]) == [1:3, 3:4, 5:5]
        end
        @testset "get_best_ranges" begin
            @test get_best_ranges(0, 1, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>1:4, :z=>1:5)

            @test get_best_ranges(0, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>1:1, :z=>1:2)
            @test get_best_ranges(1, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>2:2, :z=>1:2)
            @test get_best_ranges(2, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>3:3, :z=>1:2)
            @test get_best_ranges(3, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>4:4, :z=>1:2)
            @test get_best_ranges(4, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>1:1, :z=>3:5)
            @test get_best_ranges(5, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>2:2, :z=>3:5)
            @test get_best_ranges(6, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>3:3, :z=>3:5)
            @test get_best_ranges(7, 8, (:x,:y,:z), Dict(:x=>3, :y=>4, :z=>5)) ==
                Dict(:x=>1:3, :y=>4:4, :z=>3:5)
        end
    end
end

end # LoopSetupTests


using .LoopSetupTests

LoopSetupTests.runtests()
