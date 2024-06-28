module FokkerPlanckTimeEvolutionTests
include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_ion_moments_data, load_pdf_data,
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
    n_ion::Array{mk_float, 1} #time
    upar_ion::Array{mk_float, 1} # time
    ppar_ion::Array{mk_float, 1} # time
    pperp_ion::Array{mk_float, 1} # time
    qpar_ion::Array{mk_float, 1} # time
    v_t_ion::Array{mk_float, 1} # time
    dSdt::Array{mk_float, 1} # time
    f_ion::Array{mk_float, 3} # vpa, vperp, time
end

const expected_base =
  expected_data(
   [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
   [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
   # Expected phi:
   [-1.268504994982021, -1.276675261324553],
   # Expected n_ion:
   [0.28125178045355353, 0.278963240220169],
   # Expected upar_ion:
   [0.0, 0.0],
   # Expected ppar_ion:
   [0.17964315932116812, 0.148762861611732],
   # Expected pperp_ion
   [0.14325820846660123, 0.15798027481288696],
   # Expected qpar_ion
   [0.0, 0.0],
   # Expected v_t_ion
   [1.0511726083010418, 1.0538484394097123],
   # Expected dSdt
   [0.0, 1.1831920390587679e-5],
   # Expected f_ion:
   [0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0006193406755051616 0.0004775754345362236 0.0002663153958159559 7.630063837899157e-5 1.3259062818903743e-5 1.3974949395295016e-6 0.0;
    0.005876140721904817 0.0045311095648138235 0.00252673012465705 0.000723920301085391 0.00012579848546344195 1.3259062818903743e-5 0.0;
    0.0338148551171386 0.026074735222541223 0.01454032793443568 0.004165873701136042 0.000723920300962911 7.630063836608227e-5 0.0;
    0.11802008308891455 0.09100563662998079 0.05074842713409726 0.014539667807028695 0.0025266154112868616 0.0002663033051156912 0.0;
    0.22935011509426767 0.1768525549976815 0.09862014412656786 0.028255144359305 0.00491000785807803 0.0005175110208340848 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.22935011509426778 0.17685255499768154 0.09862014412656786 0.028255144359305 0.00491000785807803 0.0005175110208340848 0.0;
    0.1180200830889146 0.09100563662998083 0.05074842713409728 0.014539667807028703 0.0025266154112868616 0.0002663033051156912 0.0;
    0.0338148551171386 0.026074735222541223 0.01454032793443568 0.004165873701136042 0.000723920300962911 7.630063836608227e-5 0.0;
    0.005876140721904817 0.0045311095648138235 0.00252673012465705 0.000723920301085391 0.00012579848546344195 1.3259062818903743e-5 0.0;
    0.0006193406755051616 0.0004775754345362236 0.0002663153958159559 7.630063837899157e-5 1.3259062818903743e-5 1.3974949395295016e-6 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;;;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.00017108987342944037 7.097261590252227e-5 -7.822316408658004e-5 -0.000153489754637506 -9.087984332447761e-5 -3.307937957312587e-5 0.0;
    0.005877384425022921 0.004662912823128315 0.002853094592035005 0.0008130106752860298 2.4399432485772724e-5 -9.74348090421241e-5 0.0;
    0.02789429969714529 0.022363413859309983 0.014121230173998248 0.004673094872084004 0.0007098078939540657 -0.0002237857510708618 0.0;
    0.08109427896718696 0.06556961121847364 0.04243458963422585 0.01507272286376149 0.0029027058292680945 -0.0002299783231637593 0.0;
    0.1511887162169901 0.12301753182687214 0.08103653941734254 0.029945484317773212 0.0062610744742386155 7.528837370841561e-6 0.0;
    0.184756848099325 0.1505865499710698 0.09966561578213134 0.037214599991686845 0.007933889035324836 0.00016178010211991204 0.0;
    0.1511887162169905 0.12301753182687247 0.08103653941734275 0.029945484317773295 0.006261074474238628 7.528837370846196e-6 0.0;
    0.08109427896718732 0.06556961121847393 0.042434589634226055 0.015072722863761552 0.0029027058292681127 -0.0002299783231637549 0.0;
    0.027894299697145436 0.022363413859310108 0.014121230173998337 0.00467309487208404 0.0007098078939540783 -0.00022378575107086105 0.0;
    0.005877384425022965 0.00466291282312835 0.0028530945920350282 0.0008130106752860425 2.4399432485774198e-5 -9.743480904212476e-5 0.0;
    0.00017108987342944414 7.097261590252536e-5 -7.822316408657793e-5 -0.0001534897546375058 -9.087984332447822e-5 -3.3079379573126077e-5 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0])
const expected_no_regularity = 
expected_data(
   [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
   [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
   # Expected phi:
   [-1.249777922692213, -1.249777922692476],
   # Expected n_ion:
   [0.286568430139637, 0.286568430139562],
   # Expected upar_ion:
   [0.0, 0.0],
   # Expected ppar_ion:
   [0.182900778868619, 0.157103495982094],
   # Expected pperp_ion
   [0.143950337301367, 0.156848978744563],
   # Expected qpar_ion
   [0.0, 0.0],
   # Expected v_t_ion
   [1.046547864034549, 1.046547864034539],
   # Expected dSdt
   [0.000000000000000, -0.000000000409521],
   # Expected f_ion:
   [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000709315517525 0.000479326544463 0.000267291885601 0.000076580407386 0.000013307679383 0.000001402619088 0.000000000000000 ;
    0.006729798254906 0.004547723633218 0.002535994801781 0.000726574675523 0.000126259746577 0.000013307679383 0.000000000000000 ;
    0.038727315040126 0.026170342585036 0.014593642470202 0.004181148571375 0.000726574675400 0.000076580407373 0.000000000000000 ;
    0.135165490857824 0.091339334982523 0.050934510844468 0.014592981682667 0.002535879973697 0.000267279782810 0.000000000000000 ;
    0.262731408806302 0.177543188036297 0.099005269067555 0.028365484502288 0.004929182099845 0.000519531971048 0.000000000000000 ;
    0.000976245858025 0.000659707199564 0.000367879441171 0.000105399224562 0.000018315638889 0.000001930454136 0.000000000000000 ;
    0.262731408806302 0.177543188036297 0.099005269067555 0.028365484502288 0.004929182099845 0.000519531971048 0.000000000000000 ;
    0.135165490857824 0.091339334982523 0.050934510844468 0.014592981682667 0.002535879973697 0.000267279782810 0.000000000000000 ;
    0.038727315040126 0.026170342585036 0.014593642470202 0.004181148571375 0.000726574675400 0.000076580407373 0.000000000000000 ;
    0.006729798254906 0.004547723633218 0.002535994801781 0.000726574675523 0.000126259746577 0.000013307679383 0.000000000000000 ;
    0.000709315517525 0.000479326544463 0.000267291885601 0.000076580407386 0.000013307679383 0.000001402619088 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;;;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000474751734681 0.000190566474175 -0.000068453615894 -0.000220474853312 -0.000166467621731 -0.000070269966866 0.000000000000000 ;
    0.008012848857986 0.005724808854515 0.003340228975343 0.000928890127606 -0.000038607624651 -0.000175709447738 0.000000000000000 ;
    0.034159249923911 0.024646971077049 0.014803095427209 0.004994733815267 0.000871813908241 -0.000279350037847 0.000000000000000 ;
    0.095764438891876 0.069477778980504 0.042139603659514 0.014773029429542 0.003247498909427 -0.000201036843458 0.000000000000000 ;
    0.177293203999877 0.129346984538368 0.079203821037167 0.028272781052345 0.006554590511338 0.000118708200311 0.000000000000000 ;
    0.216600640826244 0.158225686082234 0.097107273632873 0.034812857798807 0.008154333928791 0.000303091030245 0.000000000000000 ;
    0.177293203999877 0.129346984538368 0.079203821037167 0.028272781052345 0.006554590511338 0.000118708200311 0.000000000000000 ;
    0.095764438891876 0.069477778980504 0.042139603659514 0.014773029429542 0.003247498909427 -0.000201036843458 0.000000000000000 ;
    0.034159249923911 0.024646971077049 0.014803095427209 0.004994733815267 0.000871813908241 -0.000279350037847 0.000000000000000 ;
    0.008012848857986 0.005724808854515 0.003340228975343 0.000928890127606 -0.000038607624651 -0.000175709447738 0.000000000000000 ;
    0.000474751734681 0.000190566474175 -0.000068453615894 -0.000220474853312 -0.000166467621731 -0.000070269966866 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000])
###########################################################################################
# to modify the test, with a new expected f, print the new f using the following commands
# in an interative Julia REPL. The path is the path to the .dfns file. 
########################################################################################## 
"""
using moment_kinetics.load_data: open_readonly_output_file, load_pdf_data, load_ion_moments_data, load_fields_data
fid = open_readonly_output_file(path, "dfns")
f_ion_vpavperpzrst = load_pdf_data(fid)
f_ion = f_ion_vpavperpzrst[:,:,1,1,1,:]
ntind = 2
nvpa = 13  #subject to grid choices
nvperp = 7 #subject to grid choices
# pdf
for k in 1:ntind
     for i in 1:nvpa-1
         for j in 1:nvperp-1
            @printf("%.15f ", f_ion[i,j,k])
         end
         @printf("%.15f ", f_ion[i,nvperp,k])
         print(";\n")
     end
     for j in 1:nvperp-1
       @printf("%.15f ", f_ion[nvpa,j,k])
     end
     @printf("%.15f ", f_ion[nvpa,nvperp,k])
     if k < ntind
         print(";;;\n")
     end
end
# a moment
n_ion_zrst, upar_ion_zrst, ppar_ion_zrst, pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
for k in 1:ntind
   @printf("%.15f", n_ion_zrst[1,1,1,k])
   if k < ntind
       print(", ")
   end
end
# a field
phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)
for k in 1:ntind
   @printf("%.15f", phi_zrt[1,1,k])
   if k < ntind
       print(", ")
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
                              "split_operators" => false,
                              "ionization_frequency" => 0.0,
                              "charge_exchange_frequency" => 0.0,
                              "constant_ionization_rate" => false,
                              "electron_physics" => "boltzmann_electron_response",
                              "fokker_planck_collisions" => Dict{String,Any}("use_fokker_planck" => true, "nuii" => 1.0, "frequency_option" => "manual"),
                              "z_IC_upar_amplitude1" => 0.0,
                              "z_IC_density_amplitude1" => 0.0,
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
function run_test(test_input, run_name, vperp_bc, expected, rtol, atol, upar_rtol=nothing; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    test_input = deepcopy(test_input)
    test_input["vperp_bc"] = vperp_bc
    
    if upar_rtol === nothing
        upar_rtol = rtol
    end

    # Convert keyword arguments to a unique name
    test_input["run_name"] = run_name
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
    print(input)
    # Suppress console output while running
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    phi = nothing
    n_ion = nothing
    upar_ion = nothing
    ppar_ion = nothing
    pperp_ion = nothing
    qpar_ion = nothing
    v_t_ion = nothing
    dSdt = nothing
    f_ion = nothing
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
            n_ion_zrst, upar_ion_zrst, ppar_ion_zrst, 
            pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
            
            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            # load coordinates
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa")
            vperp, vperp_spectral = load_coordinate_data(fid, "vperp")

            # load particle distribution function (pdf) data
            f_ion_vpavperpzrst = load_pdf_data(fid)
            
            close(fid)
            # select the single z, r, s point
            # keep the two time points in the arrays
            phi = phi_zrt[1,1,:]
            n_ion = n_ion_zrst[1,1,1,:]
            upar_ion = upar_ion_zrst[1,1,1,:]
            ppar_ion = ppar_ion_zrst[1,1,1,:]
            pperp_ion = pperp_ion_zrst[1,1,1,:]
            qpar_ion = qpar_ion_zrst[1,1,1,:]
            v_t_ion = v_t_ion_zrst[1,1,1,:]
            dSdt = dSdt_zrst[1,1,1,:]
            f_ion = f_ion_vpavperpzrst[:,:,1,1,1,:]
            f_err = copy(f_ion)
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

                # Check ion particle moments and f
                ######################################

                @test isapprox(expected.n_ion[tind], n_ion[tind], atol=atol)
                println(expected.n_ion[tind])
                println(n_ion[tind])
                @test isapprox(expected.upar_ion[tind], upar_ion[tind], atol=atol)
                @test isapprox(expected.ppar_ion[tind], ppar_ion[tind], atol=atol)
                @test isapprox(expected.pperp_ion[tind], pperp_ion[tind], atol=atol)
                @test isapprox(expected.qpar_ion[tind], qpar_ion[tind], atol=atol)
                @test isapprox(expected.v_t_ion[tind], v_t_ion[tind], atol=atol)
                @test isapprox(expected.dSdt[tind], dSdt[tind], atol=atol)
                @. f_err = abs(expected.f_ion - f_ion)
                max_f_err = maximum(f_err)
                @test isapprox(max_f_err, 0.0, atol=atol)
                @test isapprox(expected.f_ion[:,:,tind], f_ion[:,:,tind], atol=atol)
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
            run_name = "gausslegendre_pseudospectral"
            vperp_bc = "zero"
            run_test(test_input_gauss_legendre, run_name, vperp_bc, expected_base, 1.0e-14, 1.0e-14 )
        end
        @testset "Gauss Legendre no enforced regularity condition at vperp = 0" begin
            run_name = "gausslegendre_pseudospectral_no_regularity"
            vperp_bc = "zero-no-regularity"
            run_test(test_input_gauss_legendre, run_name, vperp_bc, expected_no_regularity,
             1.0e-13, 1.0e-13)
        end
    end
end

end # FokkerPlanckTimeEvolutionTests


using .FokkerPlanckTimeEvolutionTests

FokkerPlanckTimeEvolutionTests.runtests()
