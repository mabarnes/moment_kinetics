module MomentKineticsDebugTests

include("setup.jl")

function runtests()
    @testset "moment_kinetics tests" begin
        include(joinpath(@__DIR__, "kinetic_electron_tests.jl"))
        include(joinpath(@__DIR__, "sound_wave_tests.jl"))
        include(joinpath(@__DIR__, "fokker_planck_collisions_tests.jl"))
        include(joinpath(@__DIR__, "wall_bc_tests.jl"))
        include(joinpath(@__DIR__, "restart_interpolation_tests.jl"))
        include(joinpath(@__DIR__, "recycling_fraction_tests.jl"))
        include(joinpath(@__DIR__, "gyroaverage_tests.jl"))

        manufactured_solutions_ext = Base.get_extension(moment_kinetics, :manufactured_solns_ext)
        if manufactured_solutions_ext !== nothing
            include(joinpath(@__DIR__, "mms_tests.jl"))
        end
    end
end

end # MomentKineticsDebugTests

using .MomentKineticsDebugTests

MomentKineticsDebugTests.runtests()
