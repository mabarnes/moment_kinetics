module MomentKineticsTests

include("setup.jl")

function runtests()
    @testset "moment_kinetics tests" verbose=use_verbose begin
        include("calculus_tests.jl")
        include("interpolation_tests.jl")
        include("sound_wave_tests.jl")
        include("nonlinear_sound_wave_tests.jl")
        include("harrisonthompson.jl")
    end
end

end # MomentKineticsTests

using .MomentKineticsTests

MomentKineticsTests.runtests()
