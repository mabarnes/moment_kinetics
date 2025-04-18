module FokkerPlanckTimeEvolutionTests
include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_ion_moments_data, load_pdf_data,
                                 load_time_data, load_species_data
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.utils: merge_dict_with_kwargs!

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

const expected_zero_impose_regularity =
  expected_data(
   sqrt(2) .* [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
   sqrt(2) .* [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
   # Expected phi:
   [-1.268504994982021, -1.276675261324553],
   # Expected n_ion:
   [0.28125178045355353, 0.278963240220169],
   # Expected upar_ion:
   [0.0, 0.0],
   # Expected ppar_ion:
   2.0 .* [0.17964315932116812, 0.148762861611732],
   # Expected pperp_ion
   2.0 .* [0.14325820846660123, 0.15798027481288696],
   # Expected qpar_ion
   [0.0, 0.0],
   # Expected v_t_ion
   sqrt(2) .* [1.0511726083010418, 1.0538484394097123],
   # Expected dSdt
   sqrt(2) .* [0.0, 1.1831920390587679e-5],
   # Expected f_ion:
   (1.0 / (2.0 * π)^1.5) .* [0.0 0.0 0.0 0.0 0.0 0.0 0.0;
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
const expected_zero = 
expected_data(
   sqrt(2) .* [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
   sqrt(2) .* [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
   # Expected phi:
   [-1.254259088243025, -1.254259088243286],
   # Expected n_ion:
   [0.285287142537587, 0.285287142537513],
   # Expected upar_ion:
   [0.0, 0.0],
   # Expected ppar_ion:
   2.0 .* [0.182220654804438, 0.156448883483764],
   # Expected pperp_ion
   2.0 .* [0.143306715174515, 0.156192600834786],
   # Expected qpar_ion
   [0.0, 0.0],
   # Expected v_t_ion
   sqrt(2) .* [1.046701532502699, 1.046701532502689],
   # Expected dSdt
   sqrt(2) .* [0.000000000000000, -0.000000000865997],
   # Expected f_ion:
   (1.0 / (2.0 * π)^1.5) .* [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000706724195475 0.000477575434536 0.000266315395816 0.000076300638379 0.000013259062819 0.000001397494940 0.000000000000000 ;
    0.006705212475828 0.004531109564814 0.002526730124657 0.000723920301085 0.000125798485463 0.000013259062819 0.000000000000000 ;
    0.038585833650058 0.026074735222541 0.014540327934436 0.004165873701136 0.000723920300963 0.000076300638366 0.000000000000000 ;
    0.134671678398729 0.091005636629981 0.050748427134097 0.014539667807029 0.002526615411287 0.000266303305116 0.000000000000000 ;
    0.261709398369233 0.176852554997681 0.098620144126568 0.028255144359305 0.004910007858078 0.000517511020834 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.261709398369233 0.176852554997682 0.098620144126568 0.028255144359305 0.004910007858078 0.000517511020834 0.000000000000000 ;
    0.134671678398729 0.091005636629981 0.050748427134097 0.014539667807029 0.002526615411287 0.000266303305116 0.000000000000000 ;
    0.038585833650058 0.026074735222541 0.014540327934436 0.004165873701136 0.000723920300963 0.000076300638366 0.000000000000000 ;
    0.006705212475828 0.004531109564814 0.002526730124657 0.000723920301085 0.000125798485463 0.000013259062819 0.000000000000000 ;
    0.000706724195475 0.000477575434536 0.000266315395816 0.000076300638379 0.000013259062819 0.000001397494940 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;;;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000473874379555 0.000190726186343 -0.000067413540342 -0.000219129923463 -0.000165609965457 -0.000069940808382 0.000000000000000 ;
    0.007980077110978 0.005701898246888 0.003327438535529 0.000925955494251 -0.000037963012435 -0.000174809935222 0.000000000000000 ;
    0.034010103035913 0.024541441396039 0.014741892515397 0.004975785949797 0.000869171347288 -0.000277723510529 0.000000000000000 ;
    0.095320265225675 0.069160937049138 0.041953067849510 0.014711859721633 0.003235165444330 -0.000199356792174 0.000000000000000 ;
    0.176443375955397 0.128736406466194 0.078839749105573 0.028150317244036 0.006528121645593 0.000119350436884 0.000000000000000 ;
    0.215552595150568 0.157471466895046 0.096656199821596 0.034660213647059 0.008120924795998 0.000303073103709 0.000000000000000 ;
    0.176443375955397 0.128736406466194 0.078839749105573 0.028150317244036 0.006528121645593 0.000119350436884 0.000000000000000 ;
    0.095320265225675 0.069160937049138 0.041953067849510 0.014711859721633 0.003235165444330 -0.000199356792174 0.000000000000000 ;
    0.034010103035913 0.024541441396039 0.014741892515397 0.004975785949797 0.000869171347288 -0.000277723510529 0.000000000000000 ;
    0.007980077110978 0.005701898246888 0.003327438535529 0.000925955494251 -0.000037963012435 -0.000174809935222 0.000000000000000 ;
    0.000473874379555 0.000190726186343 -0.000067413540342 -0.000219129923463 -0.000165609965457 -0.000069940808382 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000])

const expected_none_bc = 
expected_data(
   sqrt(2) .* [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
   sqrt(2) .* [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
   # Expected phi:
   [-1.254104813718096, -1.254104813718360],
   # Expected n_ion:
   [0.285331158471152, 0.285331158471077],
   # Expected upar_ion:
   [0.0, 0.0],
   # Expected ppar_ion:
   2.0 .* [0.182321263834348, 0.154595996906667],
   # Expected pperp_ion
   2.0 .* [0.143470130069393, 0.157332763533171],
   # Expected qpar_ion
   [0.0, 0.0],
   # Expected v_t_ion
   sqrt(2) .* [1.047097792428007, 1.047097792428005],
   # Expected dSdt
   sqrt(2) .* [0.000000000000000, 0.000000019115425],
   # Expected f_ion:
   (1.0 / (2.0 * π)^1.5) .* [0.000045179366280 0.000030530376095 0.000017024973661 0.000004877736620 0.000000847623528 0.000000089338863 0.000000005711242 ;
    0.000706724195475 0.000477575434536 0.000266315395816 0.000076300638379 0.000013259062819 0.000001397494940 0.000000089338863 ;
    0.006705212475828 0.004531109564814 0.002526730124657 0.000723920301085 0.000125798485463 0.000013259062819 0.000000847623528 ;
    0.038585833650058 0.026074735222541 0.014540327934436 0.004165873701136 0.000723920300963 0.000076300638366 0.000004877736619 ;
    0.134671678398729 0.091005636629981 0.050748427134097 0.014539667807029 0.002526615411287 0.000266303305116 0.000017024200728 ;
    0.261709398369233 0.176852554997681 0.098620144126568 0.028255144359305 0.004910007858078 0.000517511020834 0.000033083372713 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.261709398369233 0.176852554997682 0.098620144126568 0.028255144359305 0.004910007858078 0.000517511020834 0.000033083372713 ;
    0.134671678398729 0.091005636629981 0.050748427134097 0.014539667807029 0.002526615411287 0.000266303305116 0.000017024200728 ;
    0.038585833650058 0.026074735222541 0.014540327934436 0.004165873701136 0.000723920300963 0.000076300638366 0.000004877736619 ;
    0.006705212475828 0.004531109564814 0.002526730124657 0.000723920301085 0.000125798485463 0.000013259062819 0.000000847623528 ;
    0.000706724195475 0.000477575434536 0.000266315395816 0.000076300638379 0.000013259062819 0.000001397494940 0.000000089338863 ;
    0.000045179366280 0.000030530376095 0.000017024973661 0.000004877736620 0.000000847623528 0.000000089338863 0.000000005711242 ;;;
    0.000447615535468 0.000364801746561 0.000270093752229 0.000138454732861 0.000052917434226 0.000014331973588 -0.000000022222325 ;
    0.000530676740860 0.000337644325418 0.000161654162384 0.000044642657051 0.000030848860156 0.000021605350900 0.000014496785601 ;
    0.006652401806692 0.004725322877613 0.002773765135978 0.000908975538425 0.000210924069594 0.000037798409136 0.000057517483301 ;
    0.030674916797523 0.021423584760802 0.012211671337997 0.003838328661082 0.000895825759878 0.000045643481270 0.000159762122747 ;
    0.097751022951371 0.068620792366922 0.039365625553724 0.012397445282748 0.002818297719268 0.000149127033057 0.000311439525732 ;
    0.194564361178625 0.137570544970062 0.079900546066037 0.025549350260855 0.005673491663727 0.000408759661293 0.000445141805523 ;
    0.242684098382075 0.171867841736931 0.100113917461427 0.032148650920148 0.007091277425658 0.000554833974109 0.000500018986081 ;
    0.194564361178625 0.137570544970062 0.079900546066037 0.025549350260855 0.005673491663727 0.000408759661293 0.000445141805523 ;
    0.097751022951371 0.068620792366922 0.039365625553724 0.012397445282748 0.002818297719268 0.000149127033057 0.000311439525732 ;
    0.030674916797523 0.021423584760802 0.012211671337997 0.003838328661082 0.000895825759878 0.000045643481270 0.000159762122747 ;
    0.006652401806692 0.004725322877613 0.002773765135978 0.000908975538425 0.000210924069594 0.000037798409136 0.000057517483301 ;
    0.000530676740860 0.000337644325418 0.000161654162384 0.000044642657051 0.000030848860156 0.000021605350900 0.000014496785601 ;
    0.000447615535468 0.000364801746561 0.000270093752229 0.000138454732861 0.000052917434226 0.000014331973588 -0.000000022222325])
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
n_ion_zrst, upar_ion_zrst, p_ion_zrst, ppar_ion_zrst, pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
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
test_input_gauss_legendre = OptionsDict("output" => OptionsDict("run_name" => "gausslegendre_pseudospectral",
                                                                "base_directory" => test_output_directory),
                                        "composition" => OptionsDict("n_ion_species" => 1,
                                                                     "n_neutral_species" => 0,
                                                                     "electron_physics" => "boltzmann_electron_response",
                                                                     "T_e" => 1.0),
                                        "ion_species_1" => OptionsDict("initial_density" => 0.5,
                                                                       "initial_temperature" => 1.0),
                                        "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                            "density_amplitude" => 0.0,
                                                                            "density_phase" => 0.0,
                                                                            "upar_amplitude" => 0.0,
                                                                            "upar_phase" => 0.0,
                                                                            "temperature_amplitude" => 0.0,
                                                                            "temperature_phase" => 0.0),
                                        "vpa" => OptionsDict("ngrid" => 3,
                                                             "L" => 8.485281374238571,
                                                             "nelement" => 6,
                                                             "bc" => "zero",
                                                             "discretization" => "gausslegendre_pseudospectral"),
                                        "vperp" => OptionsDict("ngrid" => 3,
                                                               "nelement" => 3,
                                                               "L" => 4.242640687119286,
                                                               "discretization" => "gausslegendre_pseudospectral"),
                                        "reactions" => OptionsDict("ionization_frequency" => 0.0,
                                                                   "charge_exchange_frequency" => 0.0),
                                        "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true,
                                                                                  "nuii" => 4.0,
                                                                                  "frequency_option" => "manual"),
                                        "evolve_moments" => OptionsDict("pressure" => false,
                                                                        "moments_conservation" => false,
                                                                        "parallel_flow" => false,
                                                                        "density" => false),
                                        "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                           "ngrid" => 1,
                                                           "nelement_local" => 1,
                                                           "nelement" => 1,
                                                           "bc" => "wall"),
                                        "r" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                           "ngrid" => 1,
                                                           "nelement" => 1,
                                                           "nelement_local" => 1,
                                                           "bc" => "periodic"),
                                        "timestepping" => OptionsDict("dt" => 0.0070710678118654745,
                                                                      "nstep" => 5000,
                                                                      "nwrite" => 5000,
                                                                      "nwrite_dfns" => 5000))


"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, expected, rtol, atol, upar_rtol=nothing; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)
    
    if upar_rtol === nothing
        upar_rtol = rtol
    end

    # Convert keyword arguments to a unique name
    function stringify_arg(key, value)
        if isa(value, AbstractDict)
            return string(string(key)[1], (stringify_arg(k, v) for (k, v) in value)...)
        else
            return string(string(key)[1], value)
        end
    end
    name = input["output"]["run_name"]
    if length(args) > 0
        name = string(name, "_", (stringify_arg(k, v) for (k, v) in args)...)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = name
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

            path = joinpath(realpath(input["output"]["base_directory"]), name, name)

            # open the netcdf file containing moments data and give it the handle 'fid'
            fid = open_readonly_output_file(path, "moments")

            # load species, time coordinate data
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            # load velocity moments data
            n_ion_zrst, upar_ion_zrst, p_ion_zrst,  ppar_ion_zrst,
            pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
            
            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            # load coordinates
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa"; ignore_MPI=true)
            vperp, vperp_spectral = load_coordinate_data(fid, "vperp"; ignore_MPI=true)

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
                @test isapprox(expected.upar_ion[tind], upar_ion[tind], atol=atol)
                @test isapprox(expected.ppar_ion[tind], ppar_ion[tind], atol=atol)
                @test isapprox(expected.pperp_ion[tind], pperp_ion[tind], atol=atol)
                @test isapprox(expected.qpar_ion[tind], qpar_ion[tind], atol=atol)
                @test isapprox(expected.v_t_ion[tind], v_t_ion[tind], atol=atol)
                @test isapprox(expected.dSdt[tind], dSdt[tind], atol=atol)
                @. f_err = abs(expected.f_ion[:,:,tind] - f_ion[:,:,tind])
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
            vperp_bc = "zero-impose-regularity"
            run_test(test_input_gauss_legendre,
             expected_zero_impose_regularity, 2.0e-14, 1.0e-14;
             vperp=OptionsDict("bc" => vperp_bc))
        end
        @testset "Gauss Legendre no enforced regularity condition at vperp = 0" begin
            run_name = "gausslegendre_pseudospectral_no_regularity"
            vperp_bc = "zero"
            run_test(test_input_gauss_legendre,
            expected_zero,
             1.0e-14, 1.0e-14; vperp=OptionsDict("bc" => vperp_bc))
        end
        @testset "Gauss Legendre no (explicitly) enforced boundary conditions" begin
            run_name = "gausslegendre_pseudospectral_none_bc"
            vperp_bc = "none"
            vpa_bc = "none"
            run_test(test_input_gauss_legendre, expected_none_bc, 1.0e-14, 1.0e-14;
                     vperp=OptionsDict("bc" => vperp_bc), vpa=OptionsDict("bc" => vpa_bc))
        end
    end
end

end # FokkerPlanckTimeEvolutionTests


using .FokkerPlanckTimeEvolutionTests

FokkerPlanckTimeEvolutionTests.runtests()
