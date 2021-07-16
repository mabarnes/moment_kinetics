using Test: @testset

@testset "moment_kinetics tests" begin
    include("calculus_tests.jl")
    include("interpolation_tests.jl")
    include("sound_wave_tests.jl")
    include("nonlinear_sound_wave_tests.jl")
end
