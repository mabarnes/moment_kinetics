module MomentKineticsDebugTests

include("setup.jl")

function runtests()
    @testset "moment_kinetics tests" begin
        include(joinpath(@__DIR__, "sound_wave_tests.jl"))
        include(joinpath(@__DIR__, "wall_bc_tests.jl"))
        include(joinpath(@__DIR__, "harrisonthompson.jl"))
    end
end

end # MomentKineticsDebugTests

using .MomentKineticsDebugTests

MomentKineticsDebugTests.runtests()
