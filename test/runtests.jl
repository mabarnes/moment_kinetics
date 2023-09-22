module MomentKineticsTests

include("setup.jl")

function runtests()
    @testset "moment_kinetics tests" verbose=use_verbose begin
        include(joinpath(@__DIR__, "calculus_tests.jl"))
        include(joinpath(@__DIR__, "interpolation_tests.jl"))
        include(joinpath(@__DIR__, "loop_setup_tests.jl"))
        include(joinpath(@__DIR__, "sound_wave_tests.jl"))
        include(joinpath(@__DIR__, "nonlinear_sound_wave_tests.jl"))
        include(joinpath(@__DIR__, "Krook_collisions_tests.jl"))
        include(joinpath(@__DIR__, "restart_interpolation_tests.jl"))
        include(joinpath(@__DIR__, "harrisonthompson.jl"))
        include(joinpath(@__DIR__, "wall_bc_tests.jl"))
    end
end

end # MomentKineticsTests

using .MomentKineticsTests

MomentKineticsTests.runtests()
