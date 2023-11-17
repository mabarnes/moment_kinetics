module FokkerPlanckTimeEvolutionTests
include("setup.jl")

using Base.Filesystem: tempname
using MPI
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_charged_particle_moments_data, load_pdf_data,
                                 load_time_data, load_species_data
using moment_kinetics.type_definitions: mk_float

const analytical_rtol = 3.e-2
const regression_rtol = 2.e-8

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# The expected output
struct expected_data
    vpa::Array{mk_float, 1}
    vperp::Array{mk_float, 1}
    phi::Array{mk_float, 1} #time
    n_charged::Array{mk_float, 1} #time
    upar_charged::Array{mk_float, 1} # time
    ppar_charged::Array{mk_float, 1} # time
    pperp_charged::Array{mk_float, 1} # time
    qpar_charged::Array{mk_float, 1} # time
    v_t_charged::Array{mk_float, 1} # time
    dSdt::Array{mk_float, 1} # time
    f_charged::Array{mk_float, 3} # vpa, vperp, time
end

const expected =
  expected_data(
   vec([-3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 2.5 3]),
   vec([0.155051 0.644949 1 1.5 2 2.5 3]),
   # Expected phi:
   vec([-1.267505494648937  -1.275240686285454]),
   # Expected n_charged:
   vec([0.281533032234007 0.279363721112678]),
   # Expected upar_charged:
   vec([0.0 -0.000000000000001]),
   # Expected ppar_charged:
   vec([0.179822802480489 0.147624463313134]),
   # Expected pperp_charged
   vec([0.143401466675068 0.158821758231315]),
   # Expected qpar_charged
   vec([0.0 0.0]),
   # Expected v_t_charged
   vec([1.051172608301042 1.053709630931266]),
   # Expected dSdt
   vec([0.0 0.000008785979565]),
   # Expected f_charged:
   [0.000000000000000; 0.000619960016181; 0.005882016862627; 0.033848669972256; 0.118138103172003; 0.229579465209362; 0.000000000000000; 0.229579465209362; 0.118138103172003; 0.033848669972256; 0.005882016862627; 0.000619960016181; 0.000000000000000;;
    0.000000000000000; 0.000478053009971; 0.004535640674379; 0.026100809957764; 0.091096642266611; 0.177029407552679; 0.000000000000000; 0.177029407552679; 0.091096642266611; 0.026100809957764; 0.004535640674379; 0.000478053009971; 0.000000000000000;;
    0.000000000000000; 0.000266581711212; 0.002529256854782; 0.014554868262370; 0.050799175561231; 0.098718764270694; 0.000000000000000; 0.098718764270694; 0.050799175561231; 0.014554868262370; 0.002529256854782; 0.000266581711212; 0.000000000000000;;
    0.000000000000000; 0.000076376939017; 0.000724644221386; 0.004170039574837; 0.014554207474836; 0.028283399503664; 0.000000000000000; 0.028283399503664; 0.014554207474836; 0.004170039574837; 0.000724644221386; 0.000076376939017; 0.000000000000000;;
    0.000000000000000; 0.000013272321882; 0.000125924283949; 0.000724644221264; 0.002529142026698; 0.004914917865936; 0.000000000000000; 0.004914917865936; 0.002529142026698; 0.000724644221264; 0.000125924283949; 0.000013272321882; 0.000000000000000;;
    0.000000000000000; 0.000001398892434; 0.000013272321882; 0.000076376939004; 0.000266569608421; 0.000518028531855; 0.000000000000000; 0.000518028531855; 0.000266569608421; 0.000076376939004; 0.000013272321882; 0.000001398892434; 0.000000000000000;;
    0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000;;;
    0.000000000000000; 0.000138780288619; 0.005791061761083; 0.027815242206659; 0.081445074789095; 0.152354938162618; 0.186355844713317; 0.152354938162617; 0.081445074789094; 0.027815242206658; 0.005791061761083; 0.000138780288619; 0.000000000000000;;
    0.000000000000000; 0.000041802335902; 0.004577414397901; 0.022242056574201; 0.065733717960889; 0.123790379312686; 0.151688746523960; 0.123790379312685; 0.065733717960889; 0.022242056574200; 0.004577414397901; 0.000041802335902; 0.000000000000000;;
    0.000000000000000; -0.000102715219360; 0.002768824455923; 0.013936837294347; 0.042320490365826; 0.081223176185748; 0.100027475709292; 0.081223176185748; 0.042320490365826; 0.013936837294347; 0.002768824455923; -0.000102715219360; 0.000000000000000;;
    0.000000000000000; -0.000164682482097; 0.000767345547592; 0.004535511737563; 0.014877992136837; 0.029791500388954; 0.037097641507925; 0.029791500388954; 0.014877992136837; 0.004535511737563; 0.000767345547592; -0.000164682482097; 0.000000000000000;;
    0.000000000000000; -0.000091767217551; 0.000022353567834; 0.000693446610930; 0.002889786006257; 0.006284514039983; 0.007979739613551; 0.006284514039983; 0.002889786006257; 0.000693446610930; 0.000022353567834; -0.000091767217551; 0.000000000000000;;
    0.000000000000000; -0.000032024960899; -0.000089243114436; -0.000201682249055; -0.000175449913982; 0.000111463338879; 0.000290773816217; 0.000111463338879; -0.000175449913982; -0.000201682249055; -0.000089243114436; -0.000032024960899; 0.000000000000000;;
    0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000; 0.000000000000000])

# default inputs for tests
test_input_gauss_legendre = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 0,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "gausslegendre_pseudospectral",
                                    "base_directory" => test_output_directory,
                                    "evolve_moments_density" => false,
                                    "evolve_moments_parallel_flow" => false,
                                    "evolve_moments_parallel_pressure" => false,
                                    "evolve_moments_conservation" => false,
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
                                    "T_e" => 1.0,
                                    "charge_exchange_frequency" => 0.0,
                                    "ionization_frequency" => 0.0,
                                    "nuii" => 1.0,
                                    "nstep" => 50000,
                                    "dt" => 1.0e-3,
                                    "nwrite" => 50000,
                                    "nwrite_dfns" => 50000,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "r_ngrid" => 1,
                                    "r_nelement" => 1,
                                    "r_nelement_local" => 1,
                                    "r_bc" => "periodic",
                                    "r_discretization" => "chebyshev_pseudospectral",
                                    "z_ngrid" => 1,
                                    "z_nelement" => 1,
                                    "z_nelement_local" => 1,
                                    "z_bc" => "wall",
                                    "z_discretization" => "chebyshev_pseudospectral",
                                    "vpa_ngrid" => 3,
                                    "vpa_nelement" => 6,
                                    "vpa_L" => 6.0,
                                    "vpa_bc" => "zero",
                                    "vpa_discretization" => "gausslegendre_pseudospectral",
                                    "vperp_ngrid" => 3,
                                    "vperp_nelement" => 3,
                                    "vperp_L" => 3.0,
                                    "vperp_bc" => "zero",
                                    "vperp_discretization" => "gausslegendre_pseudospectral")


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
    #input = merge(test_input, modified_inputs)
    input = test_input

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
    pperp_charged = nothing
    qpar_charged = nothing
    v_t_charged = nothing
    dSdt = nothing
    f_charged = nothing
    vpa, vpa_spectral = nothing, nothing
    vperp, vperp_spectral = nothing, nothing

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
            n_charged_zrst, upar_charged_zrst, ppar_charged_zrst, 
            pperp_charged_zrst, qpar_charged_zrst, v_t_charged_zrst, dSdt_zrst = load_charged_particle_moments_data(fid,extended_moments=true)
            
            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            # load coordinates
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa")
            vperp, vperp_spectral = load_coordinate_data(fid, "vperp")

            # load particle distribution function (pdf) data
            f_charged_vpavperpzrst = load_pdf_data(fid)
            
            close(fid)
            # select the single z, r, s point
            # keep the two time points in the arrays
            phi = phi_zrt[1,1,:]
            n_charged = n_charged_zrst[1,1,1,:]
            upar_charged = upar_charged_zrst[1,1,1,:]
            ppar_charged = ppar_charged_zrst[1,1,1,:]
            pperp_charged = pperp_charged_zrst[1,1,1,:]
            qpar_charged = qpar_charged_zrst[1,1,1,:]
            v_t_charged = v_t_charged_zrst[1,1,1,:]
            dSdt = dSdt_zrst[1,1,1,:]
            f_charged = f_charged_vpavperpzrst[:,:,1,1,1,:]
            
            # Unnormalize f
            # NEED TO UPGRADE TO 2V MOMENT KINETICS HERE
            
        end
        
        function test_values(tind)
            @testset "tind=$tind" begin
                # Check grids
                #############
                
                @test isapprox(expected.vpa[:], vpa.grid[:], atol=atol)
                @test isapprox(expected.vperp[:], vperp.grid[:], atol=atol)
            
                # Check electrostatic potential
                ###############################
                
                @test isapprox(expected.phi[tind], phi[tind], rtol=rtol)

                # Check charged particle moments and f
                ######################################

                @test isapprox(expected.n_charged[tind], n_charged[tind], atol=atol)
                @test isapprox(expected.upar_charged[tind], upar_charged[tind], atol=atol)
                @test isapprox(expected.ppar_charged[tind], ppar_charged[tind], atol=atol)
                @test isapprox(expected.pperp_charged[tind], pperp_charged[tind], atol=atol)
                @test isapprox(expected.qpar_charged[tind], qpar_charged[tind], atol=atol)
                @test isapprox(expected.v_t_charged[tind], v_t_charged[tind], atol=atol)
                @test isapprox(expected.dSdt[tind], dSdt[tind], atol=atol)
                @test isapprox(expected.f_charged[:,:,tind], f_charged[:,:,tind], atol=atol)
            end
        end

        # Test initial values
        test_values(1)

        # Test final values
        test_values(2)
    end
end


function runtests()
    @testset "Fokker Planck dFdt = C[F,F] relaxation test" verbose=use_verbose begin
        println("Fokker Planck dFdt = C[F,F] relaxation test")

        # GaussLegendre pseudospectral
        # Benchmark data is taken from this run (GaussLegendre)
        @testset "Gauss Legendre base" begin
            run_test(test_input_gauss_legendre, 1.e-2, 1.0e-2)
        end
    end
end

end # FokkerPlanckTimeEvolutionTests


using .FokkerPlanckTimeEvolutionTests

FokkerPlanckTimeEvolutionTests.runtests()
