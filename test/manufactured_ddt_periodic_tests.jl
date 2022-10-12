"""
Test cases using the method of manufactured solutions (MMS)
"""
module ManufacturedDDTPeriodicTests

include("setup.jl")
include("mms_utils.jl")

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

ngrid = 7
const input_sound_wave_periodic = Dict(
    "use_manufactured_solns" => true,
    "n_ion_species" => 1,
    "n_neutral_species" => 0,
    "boltzmann_electron_response" => true,
    "run_name" => "MMS-rperiodic",
    "base_directory" => test_output_directory,
    "evolve_moments_density" => false,
    "evolve_moments_parallel_flow" => false,
    "evolve_moments_parallel_pressure" => false,
    "evolve_moments_conservation" => false,
    "T_e" => 1.0,
    "rhostar" => 1.0,
    "initial_density1" => 0.5,
    "initial_temperature1" => 1.0,
    "initial_density2" => 0.5,
    "initial_temperature2" => 1.0,
    "z_IC_option1" => "sinusoid",
    "z_IC_density_amplitude1" => 0.001,
    "z_IC_density_phase1" => 0.0,
    "z_IC_upar_amplitude1" => 0.0,
    "z_IC_upar_phase1" => 0.0,
    "z_IC_temperature_amplitude1" => 0.0,
    "z_IC_temperature_phase1" => 0.0,
    "z_IC_option2" => "sinusoid",
    "z_IC_density_amplitude2" => 0.001,
    "z_IC_density_phase2" => 0.0,
    "z_IC_upar_amplitude2" => 0.0,
    "z_IC_upar_phase2" => 0.0,
    "z_IC_temperature_amplitude2" => 0.0,
    "z_IC_temperature_phase2" => 0.0,
    "charge_exchange_frequency" => 0.62831853071,
    "ionization_frequency" => 0.0,
    #"nstep" => 10, #1700,
    #"dt" => 0.002,
    #"nwrite" => 10, #1700,
    "nstep" => 1700, #1700,
    "dt" => 0.0002, #0.002,
    "nwrite" => 1700, #1700,
    "use_semi_lagrange" => false,
    "n_rk_stages" => 1,
    "split_operators" => false,
    "z_ngrid" => ngrid,
    "z_nelement" => 2,
    "z_bc" => "periodic",
    "z_discretization" => "chebyshev_pseudospectral",
    "r_ngrid" => ngrid,
    "r_nelement" => 2,
    "r_bc" => "periodic",
    "r_discretization" => "chebyshev_pseudospectral",
    "vpa_ngrid" => ngrid,
    "vpa_nelement" => 4,
    "vpa_L" => 8.0,
    "vpa_bc" => "zero",
    "vpa_discretization" => "chebyshev_pseudospectral",
    "vperp_ngrid" => ngrid,
    "vperp_nelement" => 4,
    "vperp_L" => 8.0,
    "vperp_bc" => "zero",
    "vperp_discretization" => "chebyshev_pseudospectral",
    "gyrophase_ngrid" => ngrid,
    "gyrophase_nelement" => 4,
    "vperp_discretization" => "chebyshev_pseudospectral",
    "vz_ngrid" => ngrid,
    "vz_nelement" => 4,
    "vz_L" => 8.0,
    "vz_bc" => "zero",
    "vz_discretization" => "chebyshev_pseudospectral",
    "vr_ngrid" => ngrid,
    "vr_nelement" => 4,
    "vr_L" => 8.0,
    "vr_bc" => "zero",
    "vr_discretization" => "chebyshev_pseudospectral",
    "vzeta_ngrid" => ngrid,
    "vzeta_nelement" => 4,
    "vzeta_L" => 8.0,
    "vzeta_bc" => "zero",
    "vzeta_discretization" => "chebyshev_pseudospectral",
)

function runtests(ngrid=nothing)
    @testset "MMS" verbose=use_verbose begin
        global_rank[] == 0 && println("MMS tests")

        @testset "r-periodic, z-periodic" begin
            @testset "vpa_advection" begin
                global_rank[] == 0 && println("\nvpa_advection")
                testconvergence(input_sound_wave_periodic, :vpa_advection, ngrid=ngrid)
            end
            @testset "z_advection" begin
                global_rank[] == 0 && println("\nz_advection")
                testconvergence(input_sound_wave_periodic, :z_advection, ngrid=ngrid)
            end
            @testset "r_advection" begin
                global_rank[] == 0 && println("\nr_advection")
                testconvergence(input_sound_wave_periodic, :r_advection, ngrid=ngrid)
            end
            #@testset "neutral_z_advection" begin
            #    global_rank[] == 0 && println("\nneutral_z_advection")
            #    testconvergence(input_sound_wave_periodic, :neutral_z_advection,
            #                    ngrid=ngrid)
            #end
            #@testset "neutral_r_advection" begin
            #    global_rank[] == 0 && println("\nneutral_r_advection")
            #    testconvergence(input_sound_wave_periodic, :neutral_r_advection,
            #                    ngrid=ngrid)
            #end
            #@testset "cx_collisions" begin
            #    global_rank[] == 0 && println("\ncx_collisions")
            #    testconvergence(input_sound_wave_periodic, :cx_collisions, ngrid=ngrid)
            #end
            #@testset "ionization_collisions" begin
            #    global_rank[] == 0 && println("\nionization_collisions")
            #    testconvergence(input_sound_wave_periodic, :ionization_collisions,
            #                    ngrid=ngrid)
            #end
            #@testset "continuity" begin
            #    global_rank[] == 0 && println("\ncontinuity")
            #    testconvergence(input_sound_wave_periodic, :continuity, ngrid=ngrid)
            #end
            #@testset "force_balance" begin
            #    global_rank[] == 0 && println("\nforce_balance")
            #    testconvergence(input_sound_wave_periodic, :force_balance, ngrid=ngrid)
            #end
            #@testset "energy" begin
            #    global_rank[] == 0 && println("\nenergy")
            #    testconvergence(input_sound_wave_periodic, :energy, ngrid=ngrid)
            #end
            @testset "all" begin
                global_rank[] == 0 && println("\nall terms")
                testconvergence(input_sound_wave_periodic, :all, ngrid=ngrid)
            end
        end
    end

    return nothing
end

function runtests_return(which_term::Symbol=:all; ngrid=nothing)
    global_rank[] == 0 && println("MMS tests with return")

    results = testconvergence(input_sound_wave_periodic, which_term; ngrid=ngrid,
                              returnstuff=true)

    return results
end

end # ManufacturedDDTPeriodicTests


using .ManufacturedDDTPeriodicTests

ManufacturedDDTPeriodicTests.runtests()
