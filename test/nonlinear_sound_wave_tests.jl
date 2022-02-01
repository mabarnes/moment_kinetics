module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs

using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_netcdf_file, load_coordinate_data,
                                 load_fields_data, load_moments_data, load_pdf_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa,
                                     interpolate_to_grid_vpa!
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
    n::Array{mk_float, 3}
    upar::Array{mk_float, 3}
    ppar::Array{mk_float, 3}
    f::Array{mk_float, 4}
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
   [-1.3862943611198908 -1.3843293874034386; -1.2115153398469691 -1.21058953020971;
    -0.8609843411405558 -0.8612724801562335; -0.5494724707161711 -0.5499504909887154;
    -0.35345976304280385 -0.3536363535947117; -0.2876820724517809 -0.2876820196023221;
    -0.35345976304280385 -0.3536363535947118; -0.5494724707161711 -0.5499504909887154;
    -0.8609843411405558 -0.8612724801562346; -1.2115153398469691 -1.2105895302097092;
    -1.3862943611198906 -1.3843293874034388],
   # Expected n:
   cat(
     [0.25 0.7500000000000002; 0.2977457514062631 0.7022542485937371;
      0.4227457514062633 0.577254248593737; 0.577254248593737 0.4227457514062633;
      0.702254248593737 0.2977457514062631; 0.7500000000000002 0.2500000000000001;
      0.702254248593737 0.2977457514062632; 0.577254248593737 0.4227457514062632;
      0.4227457514062633 0.5772542485937368; 0.2977457514062631 0.7022542485937369;
      0.25 0.75],
     [0.25049172638560674 0.7502472805801291; 0.29802153493417305 0.7023298876211871;
      0.42262395940907915 0.5770539798260521; 0.5769783753022829 0.42254736822989586;
      0.7021302480773779 0.297822380110277; 0.7500000396370949 0.2502454878502576;
      0.7021302480773778 0.29782238011027684; 0.5769783753022828 0.4225473682298958;
      0.42262395940907876 0.5770539798260513; 0.2980215349341734 0.7023298876211883;
      0.25049172638560674 0.7502472805801292], dims=3),
   # Expected upar:
   cat(
     [0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0;
      0.0 0.0; 0.0 0.0],
     [-5.2049539334066315e-17 -5.269713612596165e-17;
      -0.027875375865651277 -0.005319613768201196;
      -0.02311645486407955 -0.00394844875377928;
      -0.008993359438991082 0.005498163698709551;
      -0.001269841413965618 0.01245660613001191;
      8.292245445042558e-17 4.293615593101183e-17;
      0.0012698414139656938 -0.012456606130011863;
      0.008993359438991051 -0.005498163698709525;
      0.02311645486407926 0.003948448753779315;
      0.027875375865651114 0.005319613768201185;
      -4.870928214270031e-17 -5.239811854297711e-17], dims=3),
   # Expected ppar:
   cat(
     [0.18749999999999994 0.1875; 0.2090932189257829 0.2090932189257829;
      0.2440317810742172 0.24403178107421716; 0.24403178107421716 0.2440317810742172;
      0.2090932189257829 0.2090932189257829; 0.1875 0.1875;
      0.20909321892578284 0.20909321892578295; 0.24403178107421705 0.24403178107421716;
      0.2440317810742171 0.2440317810742171; 0.20909321892578295 0.2090932189257829;
      0.18749999999999997 0.18750000000000006],
     [0.18773809369893696 0.18855252147797147; 0.20885756070469555 0.21004308083161782;
      0.2433789886790452 0.2442245318469268; 0.24402969447364153 0.24323931498640963;
      0.20980541455443397 0.20837463440422174; 0.18836509350039932 0.18718624075121015;
      0.2098054145544336 0.20837463440422158; 0.24402969447364134 0.2432393149864097;
      0.24337898867904517 0.2442245318469269; 0.20885756070469536 0.21004308083161768;
      0.18773809369893701 0.18855252147797152], dims=3),
   # Expected f:
   cat(
     cat(
       [0.03704234423812432 0.04059701527559221 0.04284289279243935 0.03039799812235904 0.012360547822138707 0.006338529470377173 0.012360547822138674 0.03039799812235902 0.04284289279243931 0.04059701527559221 0.0370423442381243;
        0.20412414523193148 0.2512368778077274 0.39344196895256145 0.627787283492833 0.9100318469851268 1.0606601717798214 0.910031846985127 0.627787283492833 0.3934419689525617 0.25123687780772724 0.20412414523193148; 0.037042344238124324 0.040597015275592216 0.04284289279243936 0.03039799812235902 0.0123605478221387 0.006338529470377172 0.012360547822138683 0.030397998122359025 0.042842892792439316 0.040597015275592216 0.03704234423812432],
       [0.00633852947037717 0.012360547822138699 0.03039799812235904 0.04284289279243935 0.04059701527559221 0.0370423442381243 0.04059701527559219 0.042842892792439344 0.030397998122359 0.012360547822138693 0.006338529470377167; 1.0606601717798205 0.9100318469851268 0.627787283492833 0.39344196895256145 0.25123687780772747 0.20412414523193156 0.25123687780772735 0.3934419689525615 0.6277872834928327 0.9100318469851266 1.0606601717798205;
        0.006338529470377172 0.012360547822138699 0.03039799812235902 0.04284289279243936 0.0405970152755922 0.03704234423812432 0.04059701527559219 0.04284289279243936 0.030397998122359014 0.012360547822138695 0.006338529470377171],
       dims=3),
     cat(
       [0.03711371137589077 0.042877028078271845 0.044096637810611905 0.02873375787367191 0.011006726224290914 0.006618988204190726 0.014253100048141364 0.031875531344215814 0.04115449578340678 0.03826902165273537 0.037113711375890764; 0.20482709207602842 0.25171398146865814 0.39340886053339585 0.6272471056095726 0.9092879021115668 1.0599576654317882 0.909287902111568 0.6272471056095728 0.39340886053339574 0.251713981468658 0.20482709207602834;
        0.037113711375890764 0.03826902165273538 0.041154495783406804 0.031875531344215766 0.014253100048141362 0.006618988204190763 0.011006726224290919 0.02873375787367197 0.04409663781061185 0.04287702807827178 0.03711371137589074],
       [0.006641083217338873 0.014773417436585211 0.033437736030449534 0.043167959734650305 0.03955927953494643 0.036989390675581914 0.04130953259769147 0.041934568741473735 0.02732874560409102 0.010636936113964652 0.006641083217338871; 1.0599572249357163 0.909439130045617 0.6275471932578486 0.393681425273695 0.2518290561147449 0.20482665157995605 0.25182905611474476 0.3936814252736953 0.6275471932578488 0.9094391300456172 1.0599572249357156;
        0.00664108321733888 0.010636936113964622 0.02732874560409102 0.04193456874147367 0.04130953259769151 0.03698939067558195 0.03955927953494644 0.043167959734650235 0.033437736030449534 0.0147734174365852 0.006641083217338897],
       dims=3), dims=4))

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
                                    "z_IC_temperature_phase1" => π,
                                    "z_IC_option2" => "sinusoid",
                                    "z_IC_density_amplitude2" => 0.5,
                                    "z_IC_density_phase2" => π,
                                    "z_IC_upar_amplitude2" => 0.0,
                                    "z_IC_upar_phase2" => 0.0,
                                    "z_IC_temperature_amplitude2" => 0.5,
                                    "z_IC_temperature_phase2" => 0.0,
                                    "charge_exchange_frequency" => 2*π*0.1,
                                    "ionization_frequency" => 0.0,
                                    "nstep" => 100,
                                    "dt" => 0.0001,
                                    "nwrite" => 100,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "z_ngrid" => 100,
                                    "z_nelement" => 1,
                                    "z_bc" => "periodic",
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 400,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => vpa_L,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference")

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
               "evolve_moments_parallel_pressure" => true))

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 17,
                                  "z_nelement" => 4,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_L" => 8.0 * vpa_L,
                                  "vpa_ngrid" => 25,
                                  "vpa_nelement" => 64))

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
               "evolve_moments_parallel_pressure" => true))


# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, rtol; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

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
    phi = undef
    n = undef
    upar = undef
    ppar = undef
    f = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
    end

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_netcdf_file(path)

            # load space-time coordinate data
            nz, _, z_wgts, Lz, nvpa, _, vpa_wgts, ntime, time = load_coordinate_data(fid)

            # load fields data
            phi = load_fields_data(fid)

            # load velocity moments data
            n, upar, ppar, qpar, v_t, n_species, evolve_ppar = load_moments_data(fid)

            # load particle distribution function (pdf) data
            f = load_pdf_data(fid)

            close(fid)
        end

        # Create coordinates
        #
        # create the 'input' struct containing input info needed to create a coordinate
        # adv_input not actually used in this test so given values unimportant
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        z_input = grid_input("coord", input["z_ngrid"], input["z_nelement"],
                             z_L, input["z_discretization"], "",
                             "periodic", #input["z_bc"],
                             adv_input)
        z = define_coordinate(z_input)
        if input["z_discretization"] == "chebyshev_pseudospectral"
            z_spectral = setup_chebyshev_pseudospectral(z)
        else
            z_spectral = false
        end
        vpa_input = grid_input("coord", input["vpa_ngrid"], input["vpa_nelement"],
                               input["vpa_L"], input["vpa_discretization"], "",
                               input["vpa_bc"], adv_input)
        vpa = define_coordinate(vpa_input)
        if input["vpa_discretization"] == "chebyshev_pseudospectral"
            vpa_spectral = setup_chebyshev_pseudospectral(vpa)
        else
            vpa_spectral = false
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
        #newgrid_n = cat(interpolate_to_grid_z(expected.z, n[:, :, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, n[:, :, 2], z, z_spectral);
        #                   dims=3)
        #println("n ", size(newgrid_n))
        #println(newgrid_n)
        #println()
        #newgrid_upar = cat(interpolate_to_grid_z(expected.z, upar[:, :, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, upar[:, :, 2], z, z_spectral);
        #                   dims=3)
        #println("upar ", size(newgrid_upar))
        #println(newgrid_upar)
        #println()
        #newgrid_ppar = cat(interpolate_to_grid_z(expected.z, ppar[:, :, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, ppar[:, :, 2], z, z_spectral);
        #                   dims=3)
        #println("ppar ", size(newgrid_ppar))
        #println(newgrid_ppar)
        #println()
        #newgrid_f = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f[:, :, :, 1], z, z_spectral), vpa, vpa_spectral),
        #                interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f[:, :, :, 2], z, z_spectral), vpa, vpa_spectral);
        #                dims=4)
        #println("f ", size(newgrid_f))
        #println(newgrid_f)
        #println()
        function test_values(tind)
            @testset "tind=$tind" begin
                newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, tind], z, z_spectral)
                @test isapprox(expected.phi[:, tind], newgrid_phi, rtol=rtol)

                newgrid_n = interpolate_to_grid_z(expected.z, n[:, :, tind], z, z_spectral)
                @test isapprox(expected.n[:, :, tind], newgrid_n, rtol=rtol)

                newgrid_upar = interpolate_to_grid_z(expected.z, upar[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar[:, :, tind], newgrid_upar, rtol=rtol)

                newgrid_ppar = interpolate_to_grid_z(expected.z, ppar[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar[:, :, tind], newgrid_ppar, rtol=rtol)

                n_species = size(expected.f, 3)
                n_z = size(f, 2)
                n_vpa = length(expected.vpa)
                newvpa_f = Array{mk_float}(undef, n_vpa, n_z, n_species)
                for is ∈ 1:n_species, iz ∈ 1:n_z
                    if input["evolve_moments_parallel_flow"]
                        # Need to shift grid to interpolate to so that its values are in
                        # w_par = vpa - upar
                        new_wpa_grid = expected.vpa .- upar[iz,is,tind]
                    else
                        new_wpa_grid = expected.vpa
                    end
                    @views interpolate_to_grid_vpa!(newvpa_f[:,iz,is], new_wpa_grid,
                                                    f[:,iz,is,tind], vpa, vpa_spectral)
                end
                newgrid_f = interpolate_to_grid_z(expected.z, newvpa_f[:, :, :], z, z_spectral)

                @test isapprox(expected.f[:, :, :, tind], newgrid_f, rtol=rtol)
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
            run_test(test_input_finite_difference, 1.e-4)
        end
        @testset "FD split 1" begin
            run_test(test_input_finite_difference_split_1_moment, 1.e-4)
        end
        @testset "FD split 2" begin
            run_test(test_input_finite_difference_split_2_moments, 1.e-3)
        end
        @testset_skip "grids need shift/scale for collisions" "FD split 3" begin
            run_test(test_input_finite_difference_split_3_moments, 1.e-3)
        end

        # Chebyshev pseudospectral
        # Benchmark data is taken from this run (Chebyshev with no splitting)
        @testset "Chebyshev base" begin
            run_test(test_input_chebyshev, 1.e-10)
        end
        @testset "Chebyshev split 1" begin
            run_test(test_input_chebyshev_split_1_moment, 1.e-4)
        end
        @testset "Chebyshev split 2" begin
            run_test(test_input_chebyshev_split_2_moments, 1.e-3)
        end
        @testset_skip "grids need shift/scale for collisions" "Chebyshev split 3" begin
            run_test(test_input_chebyshev_split_3_moments, 1.e-3)
        end
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
