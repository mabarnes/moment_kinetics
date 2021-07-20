using Test: @testset

@testset "moment_kinetics tests" begin
    include("calculus_tests.jl")
    include("sound_wave_tests.jl")
end
