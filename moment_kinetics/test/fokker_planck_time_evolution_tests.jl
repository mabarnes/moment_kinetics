module FokkerPlanckTimeEvolutionTests
include("setup.jl")

using Base.Filesystem: tempname
using MPI

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
   [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
   [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
   # Expected phi:
   [-1.267505494648937, -1.275683298550937],
   # Expected n_charged:
   [0.2815330322340072, 0.2792400986636072],
   # Expected upar_charged:
   [0.0, 0.0],
   # Expected ppar_charged:
   [0.17982280248048935, 0.14891126175332367],
   # Expected pperp_charged
   [0.14340146667506784, 0.1581377822859991],
   # Expected qpar_charged
   [0.0, 0.0],
   # Expected v_t_charged
   [1.0511726083010418, 1.0538509291794658],
   # Expected dSdt
   [0.0, 1.1853081348031516e-5],
   # Expected f_charged:
   [0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0006199600161806666 0.00047805300997075977 0.0002665817112117718 7.637693901737056e-5 1.3272321881722645e-5 1.3988924344690309e-6 0.0;
    0.005882016862626724 0.0045356406743786385 0.002529256854781707 0.0007246442213864763 0.00012592428394890537 1.3272321881722645e-5 0.0;
    0.03384866997225574 0.026100809957763767 0.01455486826237011 0.004170039574837177 0.000724644221263874 7.637693900444835e-5 0.0;
    0.11813810317200342 0.09109664226661075 0.05079917556123135 0.01455420747483572 0.0025291420266981487 0.0002665696084208068 0.0;
    0.22957946520936198 0.17702940755267918 0.0987187642706944 0.0282833995036643 0.004914917865936108 0.0005180285318549189 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.22957946520936204 0.1770294075526792 0.0987187642706944 0.0282833995036643 0.004914917865936108 0.0005180285318549189 0.0;
    0.11813810317200349 0.0910966422666108 0.050799175561231376 0.01455420747483573 0.0025291420266981487 0.0002665696084208068 0.0;
    0.03384866997225574 0.026100809957763767 0.01455486826237011 0.004170039574837177 0.000724644221263874 7.637693900444835e-5 0.0;
    0.005882016862626724 0.0045356406743786385 0.002529256854781707 0.0007246442213864763 0.00012592428394890537 1.3272321881722645e-5 0.0;
    0.0006199600161806666 0.00047805300997075977 0.0002665817112117718 7.637693901737056e-5 1.3272321881722645e-5 1.3988924344690309e-6 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;;;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0001712743622973216 7.105465094508053e-5 -7.829380680167827e-5 -0.00015364081956318698 -9.097098213067502e-5 -3.311284120491419e-5 0.0;
    0.005883280697248667 0.004667594200766182 0.002855965521103658 0.0008138347136178689 2.44260649525292e-5 -9.753249634264602e-5 0.0;
    0.02792209301450194 0.022385716644538384 0.01413535091105969 0.004677801530322722 0.0007105315221401102 -0.00022400635166536323 0.0;
    0.08117458037332098 0.06563459159004267 0.04247673844050208 0.015087784332275832 0.0029056314178876035 -0.00023019804543218203 0.0;
    0.15133793170654106 0.12313903060106579 0.08111673445361306 0.029975277983613262 0.00626735398468981 7.553501812465833e-6 0.0;
    0.18493902160817713 0.15073513412904313 0.09976414473955808 0.037251581926306565 0.007941836186495122 0.00016196175024033304 0.0;
    0.15133793170654092 0.12313903060106571 0.08111673445361306 0.02997527798361324 0.006267353984689816 7.553501812469816e-6 0.0;
    0.081174580373321 0.06563459159004267 0.042476738440502065 0.015087784332275821 0.002905631417887614 -0.0002301980454321778 0.0;
    0.027922093014501933 0.022385716644538384 0.014135350911059698 0.004677801530322729 0.0007105315221401184 -0.00022400635166536134 0.0;
    0.005883280697248667 0.004667594200766184 0.002855965521103663 0.0008138347136178759 2.4426064952530956e-5 -9.753249634264635e-5 0.0;
    0.0001712743622973275 7.105465094508572e-5 -7.829380680167411e-5 -0.00015364081956318568 -9.097098213067551e-5 -3.311284120491447e-5 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0])
###########################################################################################
# to modify the test, with a new expected f, print the new f using the following commands
# in an interative Julia REPL. The path is the path to the .dfns file. 
########################################################################################## 
"""
fid = open_readonly_output_file(path, "dfns")
f_charged_vpavperpzrst = load_pdf_data(fid)
f_charged = f_charged_vpavperpzrst[:,:,1,1,1,:]
ntind = 2
nvpa = 13  #subject to grid choices
nvperp = 7 #subject to grid choices
for k in 1:ntind
  for j in 1:nvperp-1
      for i in 1:nvpa-1
         @printf("%.15f ", f_charged[i,j,k])
         print("; ")
      end
      @printf("%.15f ", f_charged[nvpa,j,k])
      print(";;\n")
  end
  for i in 1:nvpa-1
    @printf("%.15f ", f_charged[i,nvperp,k])
    print("; ")
  end
  @printf("%.15f ", f_charged[nvpa,nvperp,k])
  if k < ntind
      print(";;;\n")
  end  
end
"""
# default inputs for tests
test_input_gauss_legendre = Dict("run_name" => "gausslegendre_pseudospectral",
                              "base_directory" => test_output_directory,
                              "n_ion_species" => 1,
                              "n_neutral_species" => 0,
                              "T_wall" => 1.0,
                              "T_e" => 1.0,
                              "initial_temperature2" => 1.0,
                              "vpa_ngrid" => 3,
                              "vpa_L" => 6.0,
                              "vpa_nelement" => 6,
                              "vpa_bc" => "zero",
                              "vpa_discretization" => "gausslegendre_pseudospectral",
                              "vperp_ngrid" => 3,
                              "vperp_nelement" => 3,
                              "vperp_L" => 3.0,
                              "vperp_discretization" => "gausslegendre_pseudospectral",
                              "n_rk_stages" => 4,
                              "split_operators" => false,
                              "ionization_frequency" => 0.0,
                              "charge_exchange_frequency" => 0.0,
                              "constant_ionization_rate" => false,
                              "electron_physics" => "boltzmann_electron_response",
                              "nuii" => 1.0,
                              "Bzed" => 1.0,
                              "Bmag" => 1.0,
                              "rhostar" => 1.0,
                              "z_IC_upar_amplitude1" => 0.0,
                              "z_IC_density_amplitude1" => 0.001,
                              "z_IC_upar_amplitude2" => 0.0,
                              "z_IC_temperature_phase1" => 0.0,
                              "z_IC_temperature_amplitude1" => 0.0,
                              "evolve_moments_parallel_pressure" => false,
                              "evolve_moments_conservation" => false,
                              "z_IC_option1" => "sinusoid",
                              "evolve_moments_parallel_flow" => false,
                              "z_IC_density_phase2" => 0.0,
                              "z_discretization" => "chebyshev_pseudospectral",                              
                              "z_IC_upar_phase2" => 0.0,
                              "evolve_moments_density" => false,
                              "z_IC_temperature_amplitude2" => 0.0,
                              "initial_density1" => 0.5,
                              "z_IC_upar_phase1" => 0.0,
                              "initial_density2" => 0.5,
                              "z_IC_density_phase1" => 0.0,
                              "z_IC_option2" => "sinusoid",
                              "z_IC_density_amplitude2" => 0.001,
                              "initial_temperature1" => 1.0,
                              "z_IC_temperature_phase2" => 0.0,
                              "z_ngrid" => 1,
                              "z_nelement_local" => 1,  
                              "z_nelement" => 1,
                              "z_bc" => "wall",
                              "r_discretization" => "chebyshev_pseudospectral",
                              "r_ngrid" => 1, 
                              "r_nelement" => 1,
                              "r_nelement_local" => 1,
                              "r_bc" => "periodic",   
                              "timestepping" => Dict{String,Any}("dt" => 0.01,
                                                                 "nstep" => 5000,
                                                                 "nwrite" => 5000,
                                                                 "nwrite_dfns" => 5000 ))


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
    #input = test_input

    input["run_name"] = name

    # Suppress console output while running
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
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
    f_err = nothing
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
            f_err = copy(f_charged)
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
                @. f_err = abs(expected.f_charged - f_charged)
                max_f_err = maximum(f_err)
                @test isapprox(max_f_err, 0.0, atol=atol)
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
            run_test(test_input_gauss_legendre, 1.e-14, 1.0e-14 )
        end
    end
end

end # FokkerPlanckTimeEvolutionTests


using .FokkerPlanckTimeEvolutionTests

FokkerPlanckTimeEvolutionTests.runtests()
