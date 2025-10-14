module MomentKineticsTests

include("setup.jl")

function runtests()
    @testset "moment_kinetics tests" verbose=use_verbose begin
        include(joinpath(@__DIR__, "calculus_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "interpolation_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "loop_setup_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "nonlinear_solver_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "velocity_integral_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "sound_wave_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "nonlinear_sound_wave_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "Krook_collisions_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "moment_kinetic_2V_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "multi_source_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "numerical_dissipation.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "regridding_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "restart_interpolation_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "harrisonthompson.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "wall_bc_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "recycling_fraction_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "braginskii_electrons_imex_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "fokker_planck_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "fokker_planck_time_evolution_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "gyroaverage_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "jacobian_matrix_tests.jl"))
        GC.gc()
        include(joinpath(@__DIR__, "kinetic_electron_tests.jl"))
    end
end

end # MomentKineticsTests

using .MomentKineticsTests

MomentKineticsTests.runtests()
