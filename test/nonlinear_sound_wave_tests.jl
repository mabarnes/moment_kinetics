module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
using MPI
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_charged_particle_moments_data, load_pdf_data,
                                 load_neutral_particle_moments_data,
                                 load_neutral_pdf_data, load_time_data, load_species_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.type_definitions: mk_float

const analytical_rtol = 3.e-2
const regression_rtol = 2.e-8

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = 8.0

# The expected output
struct expected_data
    z::Array{mk_float, 1}
    vpa::Array{mk_float, 1}
    phi::Array{mk_float, 2}
    n_charged::Array{mk_float, 2}
    n_neutral::Array{mk_float, 2}
    upar_charged::Array{mk_float, 2}
    upar_neutral::Array{mk_float, 2}
    ppar_charged::Array{mk_float, 2}
    ppar_neutral::Array{mk_float, 2}
    f_charged::Array{mk_float, 3}
    f_neutral::Array{mk_float, 3}
end

# Use very small number of points in vpa_expected to reduce the amount of entries we
# need to store. First and last entries are within the grid (rather than at the ends) in
# order to get non-zero values.
# Note: in the arrays of numbers for expected data, space-separated entries have to stay
# on the same line.
const expected =
  expected_data(
   [z for z in range(-0.5 * z_L, 0.5 * z_L, length=11)],
   [vpa for vpa in range(-0.2 * vpa_L, 0.2 * vpa_L, length=3)],
   # Expected phi:
   [-1.386282080324426 -1.2382641646968997; -1.2115129555832849 -1.130635565831393;
    -0.8609860698164498 -0.872637046489647; -0.5494724768120176 -0.5903597644911374;
    -0.35345976364887166 -0.37552658974835484; -0.28768207245186167 -0.2921445534164449;
    -0.353459763648872 -0.3755265897483555; -0.5494724768120175 -0.5903597644911376;
    -0.8609860698164502 -0.8726370464896476; -1.2115129555832849 -1.1306355658313922;
    -1.3862820803244258 -1.2382641646968997],
   # Expected n_charged:
   [0.2500030702177186 0.2898869775083742; 0.2977473631375158 0.3228278662412625;
    0.42274585818529853 0.417848119539277; 0.5772542465450629 0.5541281150892785;
    0.7022542481909738 0.6869277664245242; 0.7499999999999394 0.7466605958319346;
    0.7022542481909738 0.6869277664245237; 0.577254246545063 0.5541281150892783;
    0.42274585818529864 0.41784811953927686; 0.2977473631375159 0.32282786624126253;
    0.2500030702177185 0.2898869775083743],
   # Expected n_neutral:
   [0.7499999999999382 0.7736769553678673; 0.7022542481909748 0.7056866352169496;
    0.5772542465450632 0.5582977481633454; 0.4227458581852985 0.40969188756651037;
    0.29774736313751604 0.30539644783353687; 0.2500030702177186 0.268198658560817;
    0.29774736313751604 0.305396447833537; 0.42274585818529836 0.4096918875665103;
    0.5772542465450631 0.5582977481633457; 0.7022542481909745 0.7056866352169494;
    0.7499999999999383 0.7736769553678673],
   # Expected upar_charged:
   [-2.7135787559953277e-17 -6.299214622140781e-17;
    -9.321028970172899e-18 -0.1823721921091055;
    -2.8374879811351724e-18 -0.19657035490893093;
    1.2124327390522635e-17 -0.11139486685283827;
    3.6525788403693063e-17 -0.033691837771623996;
    -2.0930856430671915e-17 4.84147091991613e-17;
    8.753545920086251e-18 0.033691837771624024;
    1.1293771270243255e-17 0.11139486685283813;
    1.3739171132886587e-17 0.19657035490893102;
    -6.840453743089351e-18 0.18237219210910513;
    -2.7135787559953277e-17 -4.656897959900552e-17],
   # Expected upar_neutral:
   [6.5569385065066925e-18 7.469475342067322e-17;
    1.1054500872839027e-17 -0.036209130454625794;
    -3.241833393685864e-17 -0.00915544640981337;
    -3.617637280460899e-17 0.05452268209340691; 4.417578961284041e-17 0.07606644718003618;
    4.9354467746194965e-17 4.452343983947504e-17;
    6.573091229872379e-18 -0.07606644718003616;
    2.989662686945165e-17 -0.05452268209340687;
    -3.1951996361666834e-17 0.009155446409813412;
    -4.395464518158184e-18 0.03620913045462582;
    6.5569385065066925e-18 7.150569974151354e-17],
   # Expected ppar_charged:
   [0.18749999999999992 0.2328164829490338; 0.20909325514551116 0.21912575009260987;
    0.24403180771238264 0.20822611102296495; 0.24403180771238278 0.21506741942934832;
    0.2090932551455113 0.22097085045011763; 0.1875 0.22119050467096843;
    0.20909325514551128 0.2209708504501176; 0.2440318077123828 0.2150674194293483;
    0.24403180771238256 0.20822611102296476; 0.20909325514551116 0.21912575009260982;
    0.18749999999999992 0.2328164829490338],
   # Expected ppar_neutral:
   [0.18750000000000003 0.2480244331470989; 0.20909325514551122 0.2440075646485762;
    0.24403180771238286 0.22861256884534023; 0.24403180771238278 0.20588932618946498;
    0.20909325514551144 0.19263633346696638; 0.18749999999999992 0.19091848744561835;
    0.20909325514551141 0.19263633346696654; 0.2440318077123828 0.20588932618946482;
    0.24403180771238286 0.22861256884534029; 0.20909325514551114 0.24400756464857642;
    0.18750000000000006 0.24802443314709893],
   # Expected f_charged:
   [0.0370462360994826 0.04059927063892091 0.0428431419871786 0.030398267195914062 0.01236045902698859 0.006338529470383425 0.012360459026988587 0.030398267195914028 0.04284314198717859 0.0405992706389209 0.0370462360994826;
    0.20411991941198782 0.25123395823993105 0.3934413727192304 0.6277900619432855 0.9100364506661008 1.0606601717796504 0.910036450666101 0.6277900619432859 0.39344137271923046 0.25123395823993094 0.20411991941198776;
    0.0370462360994826 0.04059927063892091 0.0428431419871786 0.030398267195914062 0.01236045902698859 0.006338529470383425 0.012360459026988587 0.030398267195914028 0.04284314198717859 0.0405992706389209 0.0370462360994826;;;
    0.05392403019146985 0.06057819609646438 0.03676744157455075 0.013740507879552622 0.010777319583092297 0.019330359159894384 0.027982173790396116 0.027603104735767332 0.02667986700464528 0.035654512254837005 0.05392403019146984;
    0.21177720235387912 0.24902901234066305 0.3729377138332225 0.596281539172339 0.8870867512643452 1.0533860567375264 0.887086751264345 0.5962815391723388 0.3729377138332225 0.24902901234066285 0.21177720235387912;
    0.053924030191469796 0.035654512254837074 0.02667986700464531 0.02760310473576733 0.02798217379039615 0.019330359159894287 0.010777319583092311 0.013740507879552624 0.03676744157455069 0.060578196096464365 0.05392403019146979],
   # Expected f_neutral:
   [0.0063385294703834595 0.012360459026988546 0.030398267195914108 0.04284314198717859 0.040599270638920985 0.03704623609948259 0.040599270638920965 0.0428431419871786 0.030398267195914094 0.012360459026988546 0.006338529470383456;
    1.0606601717796493 0.9100364506661016 0.6277900619432857 0.3934413727192303 0.2512339582399308 0.20411991941198754 0.2512339582399307 0.3934413727192301 0.6277900619432853 0.9100364506661016 1.0606601717796487;
    0.0063385294703834595 0.012360459026988546 0.030398267195914108 0.04284314198717859 0.040599270638920985 0.03704623609948259 0.040599270638920965 0.0428431419871786 0.030398267195914094 0.012360459026988546 0.006338529470383456;;;
    0.024285034070612683 0.04071236753946936 0.04190483876050118 0.036374533667106086 0.0369234055803037 0.04165072188572372 0.03672486160719089 0.019283695804388743 0.008424202166370513 0.010011778724856858 0.02428503407061268;
    1.05300288883775 0.9036794996386066 0.6251037063201776 0.39552476644559265 0.25711045639921726 0.2113940344541052 0.25711045639921726 0.39552476644559253 0.6251037063201775 0.9036794996386066 1.0530028888377503;
    0.024285034070612672 0.01001177872485688 0.00842420216637052 0.019283695804388705 0.03672486160719087 0.04165072188572364 0.0369234055803037 0.036374533667106045 0.041904838760501203 0.040712367539469434 0.02428503407061266])

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 1,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "finite_difference",
                                    "base_directory" => test_output_directory,
                                    "evolve_moments_density" => false,
                                    "evolve_moments_parallel_flow" => false,
                                    "evolve_moments_parallel_pressure" => false,
                                    "evolve_moments_conservation" => true,
                                    "T_e" => 1.0,
                                    "initial_density1" => 0.5,
                                    "initial_temperature1" => 1.0,
                                    "initial_density2" => 0.5,
                                    "initial_temperature2" => 1.0,
                                    "z_IC_option1" => "sinusoid",
                                    "z_IC_density_amplitude1" => 0.5,
                                    "z_IC_density_phase1" => 0.0,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_upar_phase1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.5,
                                    "z_IC_temperature_phase1" => mk_float(π),
                                    "z_IC_option2" => "sinusoid",
                                    "z_IC_density_amplitude2" => 0.5,
                                    "z_IC_density_phase2" => mk_float(π),
                                    "z_IC_upar_amplitude2" => 0.0,
                                    "z_IC_upar_phase2" => 0.0,
                                    "z_IC_temperature_amplitude2" => 0.5,
                                    "z_IC_temperature_phase2" => 0.0,
                                    "charge_exchange_frequency" => 2*π*0.1,
                                    "ionization_frequency" => 0.0,
                                    "nstep" => 100,
                                    "dt" => 0.001,
                                    "nwrite" => 100,
                                    "nwrite_dfns" => 100,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "r_ngrid" => 1,
                                    "r_nelement" => 1,
                                    "r_bc" => "periodic",
                                    "r_discretization" => "finite_difference",
                                    "z_ngrid" => 100,
                                    "z_nelement" => 1,
                                    "z_bc" => "periodic",
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 400,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => vpa_L,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference",
                                    "vz_ngrid" => 400,
                                    "vz_nelement" => 1,
                                    "vz_L" => vpa_L,
                                    "vz_bc" => "periodic",
                                    "vz_discretization" => "finite_difference")

test_input_finite_difference_split_1_moment =
    merge(test_input_finite_difference,
          Dict("run_name" => "finite_difference_split_1_moment",
               "evolve_moments_density" => true))

test_input_finite_difference_split_2_moments =
    merge(test_input_finite_difference_split_1_moment,
          Dict("run_name" => "finite_difference_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_finite_difference_split_3_moments =
    merge(test_input_finite_difference_split_2_moments,
          Dict("run_name" => "finite_difference_split_3_moments",
               "evolve_moments_parallel_pressure" => true,
               "vpa_L" => 12.0, "vz_L" => 12.0))

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 4,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 8,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 17,
                                  "vz_nelement" => 8))

test_input_chebyshev_split_1_moment =
    merge(test_input_chebyshev,
          Dict("run_name" => "chebyshev_pseudospectral_split_1_moment",
               "evolve_moments_density" => true))

test_input_chebyshev_split_2_moments =
    merge(test_input_chebyshev_split_1_moment,
          Dict("run_name" => "chebyshev_pseudospectral_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_chebyshev_split_3_moments =
    merge(test_input_chebyshev_split_2_moments,
          Dict("run_name" => "chebyshev_pseudospectral_split_3_moments",
               "evolve_moments_parallel_pressure" => true,
               "vpa_L" => 12.0, "vz_L" => 12.0))


# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, rtol, atol, upar_rtol=nothing; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    if upar_rtol === nothing
        upar_rtol = rtol
    end

    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Convert dict from symbol keys to String keys
    modified_inputs = Dict(String(k) => v for (k, v) in args)

    # Update default inputs with values to be changed
    input = merge(test_input, modified_inputs)

    input["run_name"] = name

    # Suppress console output while running
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
    end

    phi = nothing
    n_charged = nothing
    upar_charged = nothing
    ppar_charged = nothing
    f_charged = nothing
    n_neutral = nothing
    upar_neutral = nothing
    ppar_neutral = nothing
    f_neutral = nothing
    z, z_spectral = nothing, nothing
    vpa, vpa_spectral = nothing, nothing

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file containing moments data and give it the handle 'fid'
            fid = open_readonly_output_file(path, "moments")

            # load species, time coordinate data
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            # load velocity moments data
            n_charged_zrst, upar_charged_zrst, ppar_charged_zrst, qpar_charged_zrst, v_t_charged_zrst = load_charged_particle_moments_data(fid)
            n_neutral_zrst, upar_neutral_zrst, ppar_neutral_zrst, qpar_neutral_zrst, v_t_neutral_zrst = load_neutral_particle_moments_data(fid)
            z, z_spectral = load_coordinate_data(fid, "z")

            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            
            # load particle distribution function (pdf) data
            f_charged_vpavperpzrst = load_pdf_data(fid)
            f_neutral_vzvrvzetazrst = load_neutral_pdf_data(fid)
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa")

            close(fid)
            
            phi = phi_zrt[:,1,:]
            n_charged = n_charged_zrst[:,1,:,:]
            upar_charged = upar_charged_zrst[:,1,:,:]
            ppar_charged = ppar_charged_zrst[:,1,:,:]
            qpar_charged = qpar_charged_zrst[:,1,:,:]
            v_t_charged = v_t_charged_zrst[:,1,:,:]
            f_charged = f_charged_vpavperpzrst[:,1,:,1,:,:]
            n_neutral = n_neutral_zrst[:,1,:,:]
            upar_neutral = upar_neutral_zrst[:,1,:,:]
            ppar_neutral = ppar_neutral_zrst[:,1,:,:]
            qpar_neutral = qpar_neutral_zrst[:,1,:,:]
            v_t_neutral = v_t_neutral_zrst[:,1,:,:]
            f_neutral = f_neutral_vzvrvzetazrst[:,1,1,:,1,:,:]

            # Unnormalize f
            if input["evolve_moments_density"]
                for it ∈ 1:length(time), is ∈ 1:n_ion_species, iz ∈ 1:z.n
                    f_charged[:,iz,is,it] .*= n_charged[iz,is,it]
                end
                for it ∈ 1:length(time), isn ∈ 1:n_neutral_species, iz ∈ 1:z.n
                    f_neutral[:,iz,isn,it] .*= n_neutral[iz,isn,it]
                end
            end
            if input["evolve_moments_parallel_pressure"]
                for it ∈ 1:length(time), is ∈ 1:n_ion_species, iz ∈ 1:z.n
                    f_charged[:,iz,is,it] ./= v_t_charged[iz,is,it]
                end
                for it ∈ 1:length(time), isn ∈ 1:n_neutral_species, iz ∈ 1:z.n
                    f_neutral[:,iz,isn,it] ./= v_t_neutral[iz,isn,it]
                end
            end
        end

        # Test against values interpolated onto 'expected' grid which is fairly coarse no we
        # do not have to save too much data in this file

        # Use commented-out lines to get the test data to put in `expected`
        #newgrid_phi = cat(interpolate_to_grid_z(expected.z, phi[:, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, phi[:, 2], z, z_spectral);
        #                   dims=2)
        #println("phi ", size(newgrid_phi))
        #println(newgrid_phi)
        #println()
        #newgrid_n_charged = cat(interpolate_to_grid_z(expected.z, n_charged[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, n_charged[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("n_charged ", size(newgrid_n_charged))
        #println(newgrid_n_charged)
        #println()
        #newgrid_n_neutral = cat(interpolate_to_grid_z(expected.z, n_neutral[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, n_neutral[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("n_neutral ", size(newgrid_n_neutral))
        #println(newgrid_n_neutral)
        #println()
        #newgrid_upar_charged = cat(interpolate_to_grid_z(expected.z, upar_charged[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, upar_charged[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("upar_charged ", size(newgrid_upar_charged))
        #println(newgrid_upar_charged)
        #println()
        #newgrid_upar_neutral = cat(interpolate_to_grid_z(expected.z, upar_neutral[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, upar_neutral[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("upar_neutral ", size(newgrid_upar_neutral))
        #println(newgrid_upar_neutral)
        #println()
        #newgrid_ppar_charged = cat(interpolate_to_grid_z(expected.z, ppar_charged[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, ppar_charged[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("ppar_charged ", size(newgrid_ppar_charged))
        #println(newgrid_ppar_charged)
        #println()
        #newgrid_ppar_neutral = cat(interpolate_to_grid_z(expected.z, ppar_neutral[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, ppar_neutral[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("ppar_neutral ", size(newgrid_ppar_neutral))
        #println(newgrid_ppar_neutral)
        #println()
        #newgrid_f_charged = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_charged[:, :, :, 1], z, z_spectral), vpa, vpa_spectral)[:,:,1],
        #                        interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_charged[:, :, :, 2], z, z_spectral), vpa, vpa_spectral)[:,:,1];
        #                        dims=4)
        #println("f_charged ", size(newgrid_f_charged))
        #println(newgrid_f_charged)
        #println()
        #newgrid_f_neutral = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_neutral[:, :, :, 1], z, z_spectral), vpa, vpa_spectral)[:,:,1],
        #                        interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_neutral[:, :, :, 2], z, z_spectral), vpa, vpa_spectral)[:,:,1];
        #                        dims=4)
        #println("f_neutral ", size(newgrid_f_neutral))
        #println(newgrid_f_neutral)
        #println()
        function test_values(tind)
            @testset "tind=$tind" begin
                newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, tind], z, z_spectral)
                @test isapprox(expected.phi[:, tind], newgrid_phi, rtol=rtol)

                # Check charged particle moments and f
                ######################################

                newgrid_n_charged = interpolate_to_grid_z(expected.z, n_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_charged[:, tind], newgrid_n_charged[:,1], rtol=rtol)

                newgrid_upar_charged = interpolate_to_grid_z(expected.z, upar_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_charged[:, tind], newgrid_upar_charged[:,1], rtol=upar_rtol, atol=atol)

                newgrid_ppar_charged = 0.5*interpolate_to_grid_z(expected.z, ppar_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_charged[:, tind], newgrid_ppar_charged[:,1], rtol=rtol)

                newgrid_vth_charged = @. sqrt(2.0*newgrid_ppar_charged/newgrid_n_charged)
                newgrid_f_charged = interpolate_to_grid_z(expected.z, f_charged[:, :, :, tind], z, z_spectral)
                temp = newgrid_f_charged
                newgrid_f_charged = fill(NaN, length(expected.vpa),
                                         size(newgrid_f_charged, 2),
                                         size(newgrid_f_charged, 3),
                                         size(newgrid_f_charged, 4))
                for iz ∈ 1:length(expected.z)
                    wpa = copy(expected.vpa)
                    if input["evolve_moments_parallel_flow"]
                        wpa .-= newgrid_upar_charged[iz,1]
                    end
                    if input["evolve_moments_parallel_pressure"]
                        wpa ./= newgrid_vth_charged[iz,1]
                    end
                    newgrid_f_charged[:,iz,1] = interpolate_to_grid_vpa(wpa, temp[:,iz,1], vpa, vpa_spectral)
                end
                @test isapprox(expected.f_charged[:, :, tind], newgrid_f_charged[:,:,1], rtol=rtol)

                # Check neutral particle moments and f
                ######################################

                newgrid_n_neutral = interpolate_to_grid_z(expected.z, n_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_neutral[:, tind], newgrid_n_neutral[:,:,1], rtol=rtol)

                newgrid_upar_neutral = interpolate_to_grid_z(expected.z, upar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_neutral[:, tind], newgrid_upar_neutral[:,:,1], rtol=upar_rtol, atol=atol)

                newgrid_ppar_neutral = interpolate_to_grid_z(expected.z, ppar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_neutral[:, tind], newgrid_ppar_neutral[:,:,1], rtol=rtol)

                newgrid_vth_neutral = @. sqrt(2.0*newgrid_ppar_neutral/newgrid_n_neutral)
                newgrid_f_neutral = interpolate_to_grid_z(expected.z, f_neutral[:, :, :, tind], z, z_spectral)
                temp = newgrid_f_neutral
                newgrid_f_neutral = fill(NaN, length(expected.vpa),
                                         size(newgrid_f_neutral, 2),
                                         size(newgrid_f_neutral, 3),
                                         size(newgrid_f_neutral, 4))
                for iz ∈ 1:length(expected.z)
                    wpa = copy(expected.vpa)
                    if input["evolve_moments_parallel_flow"]
                        wpa .-= newgrid_upar_neutral[iz,1]
                    end
                    if input["evolve_moments_parallel_pressure"]
                        wpa ./= newgrid_vth_neutral[iz,1]
                    end
                    newgrid_f_neutral[:,iz,1] = interpolate_to_grid_vpa(wpa, temp[:,iz,1], vpa, vpa_spectral)
                end
                @test isapprox(expected.f_neutral[:, :, tind], newgrid_f_neutral[:,:,1], rtol=rtol)
            end
        end

        # Test initial values
        test_values(1)

        # Test final values
        test_values(2)
    end
end


function runtests()
    @testset "nonlinear sound wave" verbose=use_verbose begin
        println("nonlinear sound wave tests")

        # finite difference
        @testset "FD base" begin
            run_test(test_input_finite_difference, 1.e-3, 1.e-11, 2.e-3)
        end
        @testset "FD split 1" begin
            run_test(test_input_finite_difference_split_1_moment, 1.e-3, 1.e-11)
        end
        @testset "FD split 2" begin
            run_test(test_input_finite_difference_split_2_moments, 1.e-3, 1.e-11)
        end
        @testset "FD split 3" begin
            run_test(test_input_finite_difference_split_3_moments, 1.e-3, 1.e-11)
        end

        # Chebyshev pseudospectral
        # Benchmark data is taken from this run (Chebyshev with no splitting)
        @testset "Chebyshev base" begin
            run_test(test_input_chebyshev, 1.e-10, 3.e-16)
        end
        @testset "Chebyshev split 1" begin
            run_test(test_input_chebyshev_split_1_moment, 1.e-3, 1.e-15)
        end
        @testset "Chebyshev split 2" begin
            run_test(test_input_chebyshev_split_2_moments, 1.e-3, 1.e-15)
        end
        @testset "Chebyshev split 3" begin
            run_test(test_input_chebyshev_split_3_moments, 1.e-3, 1.e-15)
        end
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
