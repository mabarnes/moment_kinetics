module FokkerPlanckTimeEvolutionTests
include("setup.jl")

export print_output_data_for_test_update

using Base.Filesystem: tempname
using MPI
using Printf
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_ion_moments_data, load_pdf_data,
                                 load_time_data, load_species_data,
                                 load_input
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.utils: merge_dict_with_kwargs!
using moment_kinetics.input_structs: options_to_TOML
using moment_kinetics.fokker_planck_test: F_Maxwellian, print_test_data
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure

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
    maxnorm_ion::Array{mk_float, 1} # time
    L2norm_ion::Array{mk_float, 1} # time
    f_ion::Array{mk_float, 3} # vpa, vperp, time
end

const expected_zero_impose_regularity =
  expected_data(
    # Expected vpa
    [-3.000000000000000, -2.500000000000000, -2.000000000000000, -1.500000000000000, -1.000000000000000, -0.500000000000000, 0.000000000000000, 0.500000000000000, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
    # Expected vperp
    [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
    # Expected phi
    [0.018846102257925, -0.019137897530430],
    # Expected n_ion
    [1.019024810931714, 0.981044069360335],
    # Expected upar_ion
    [0.127188034386486, 0.127125252973134],
    # Expected ppar_ion
    [0.156668641253299, 0.501846847278676],
    # Expected pperp_ion
    [0.719549591157190, 0.535875941382659],
    # Expected qpar_ion
    [-0.019194164922596, -0.007216264716710],
    # Expected v_t_ion
    [1.021755168637407, 1.034087075929172],
    # Expected dSdt_ion
    [0.000000000000000, 0.000164440610574],
    # Expected maxnorm_ion
    [0.808346186957451, 0.179973032115546],
    # Expected L2norm_ion
    [0.087661894753377, 0.011357320066030],
    # Expected f_ion
    [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000000000000743 0.000000000001327 0.000000000002198 0.000000000000808 0.000000000000040 0.000000000000000 0.000000000000000 ;
    0.000000008984786 0.000000016045073 0.000000026566386 0.000000009773227 0.000000000486580 0.000000000003279 0.000000000000000 ;
    0.000014698969447 0.000026249488962 0.000043462193455 0.000015988847440 0.000000796037841 0.000000005363661 0.000000000000000 ;
    0.003254446147102 0.005811805278321 0.009622808493146 0.003540033410959 0.000176247885456 0.000001187548911 0.000000000000000 ;
    0.097516549950631 0.174145514815615 0.288338796425582 0.106073915297089 0.005281109272943 0.000035583834377 0.000000000000000 ;
    0.395449110126606 0.706194885906433 1.169271477705052 0.430150937795841 0.021415954148542 0.000144299563988 0.000000000000000 ;
    0.217027073120509 0.387567970735544 0.641709792717390 0.236071839939115 0.011753324834776 0.000079193279800 0.000000000000000 ;
    0.016119377290044 0.028786059987797 0.047662082480299 0.017533900267920 0.000872961491394 0.000005881968261 0.000000000000000 ;
    0.000162029332825 0.000289352746733 0.000479091424339 0.000176247885456 0.000008774865523 0.000000059124579 0.000000000000000 ;
    0.000000220419526 0.000000393626228 0.000000651740661 0.000000239761990 0.000000011937047 0.000000000080431 0.000000000000000 ;
    0.000000000040581 0.000000000072469 0.000000000119989 0.000000000044142 0.000000000002198 0.000000000000015 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;;;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    -0.000059253318543 -0.000202706880802 -0.000416482873495 -0.000429126481104 -0.000213752671053 -0.000068671546683 0.000000000000000 ;
    0.012544254911844 0.009954955824274 0.006096355353809 0.001722895833706 0.000051829703215 -0.000198421773570 0.000000000000000 ;
    0.063679424634785 0.050427746945806 0.030679958824065 0.009143813174947 0.001061606428502 -0.000681573096792 0.000000000000000 ;
    0.226479549633422 0.181509298874425 0.114494162782726 0.038010681381956 0.006552052453867 -0.001032574384107 0.000000000000000 ;
    0.492991700438350 0.398529061979853 0.257759854225088 0.090684885050888 0.017586927378913 -0.000565146748644 0.000000000000000 ;
    0.674483355364362 0.546650384334622 0.356152359396428 0.127399336414645 0.025465810739355 -0.000009953608323 0.000000000000000 ;
    0.608383375502228 0.492533633406629 0.319893141043541 0.113643595535264 0.022475183283330 -0.000268538435149 0.000000000000000 ;
    0.359369621711741 0.289340454268123 0.184982259519043 0.063469637765334 0.011781025340642 -0.000942129729804 0.000000000000000 ;
    0.135055462433685 0.107885414590990 0.067396326292281 0.021672903647779 0.003313556630286 -0.000927721860814 0.000000000000000 ;
    0.028957379332219 0.022792588262078 0.013605752396821 0.003636932966570 0.000090173339172 -0.000455850085963 0.000000000000000 ;
    0.001030709244163 0.000503486200893 -0.000282187069750 -0.000679645543723 -0.000385444759803 -0.000145557385042 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ])
const expected_zero = 
expected_data(
    # Expected vpa
    [-3.000000000000000, -2.500000000000000, -2.000000000000000, -1.500000000000000, -1.000000000000000, -0.500000000000000, 0.000000000000000, 0.500000000000000, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
    # Expected vperp
    [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
    # Expected phi
    [-0.000000065347221, -0.000000065347489],
    # Expected n_ion
    [0.999999934652782, 0.999999934652514],
    # Expected upar_ion
    [0.127188034386486, 0.127188034386492],
    # Expected ppar_ion
    [0.153743686448806, 0.530406198086908],
    # Expected pperp_ion
    [0.719320904338625, 0.530989648519366],
    # Expected qpar_ion
    [-0.018835815833337, -0.003341501709899],
    # Expected v_t_ion
    [1.030335090859288, 1.030335090859291],
    # Expected dSdt_ion
    [0.000000000000000, -0.000000000001616],
    # Expected maxnorm_ion
    [0.818237925050035, 0.106008532902116],
    # Expected L2norm_ion
    [0.091263900048123, 0.007627575466069],
    # Expected f_ion
    [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000000000000126 0.000000000001327 0.000000000002198 0.000000000000808 0.000000000000040 0.000000000000000 0.000000000000000 ;
    0.000000001527896 0.000000016045073 0.000000026566386 0.000000009773227 0.000000000486580 0.000000000003279 0.000000000000000 ;
    0.000002499614119 0.000026249488962 0.000043462193455 0.000015988847440 0.000000796037841 0.000000005363661 0.000000000000000 ;
    0.000553430604042 0.005811805278321 0.009622808493146 0.003540033410959 0.000176247885456 0.000001187548911 0.000000000000000 ;
    0.016583049988818 0.174145514815615 0.288338796425582 0.106073915297089 0.005281109272943 0.000035583834377 0.000000000000000 ;
    0.067247583764838 0.706194885906433 1.169271477705052 0.430150937795841 0.021415954148542 0.000144299563988 0.000000000000000 ;
    0.036906256469351 0.387567970735544 0.641709792717390 0.236071839939115 0.011753324834776 0.000079193279800 0.000000000000000 ;
    0.002741159726475 0.028786059987797 0.047662082480299 0.017533900267920 0.000872961491394 0.000005881968261 0.000000000000000 ;
    0.000027553687320 0.000289352746733 0.000479091424339 0.000176247885456 0.000008774865523 0.000000059124579 0.000000000000000 ;
    0.000000037483156 0.000000393626228 0.000000651740661 0.000000239761990 0.000000011937047 0.000000000080431 0.000000000000000 ;
    0.000000000006901 0.000000000072469 0.000000000119989 0.000000000044142 0.000000000002198 0.000000000000015 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;;;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000354236210033 -0.000198861521027 -0.000676024980821 -0.000835496073485 -0.000532610943108 -0.000196819687444 0.000000000000000 ;
    0.018422730045861 0.013105347750284 0.007539779911345 0.001850703806574 -0.000367607892099 -0.000515097109289 0.000000000000000 ;
    0.081652388562191 0.058165619982996 0.034152280010850 0.010827556056590 0.001422296034713 -0.001021692695536 0.000000000000000 ;
    0.267720565212289 0.192364032081158 0.114716448550907 0.038735768661154 0.007946079277555 -0.001096986454193 0.000000000000000 ;
    0.568687889959713 0.411691118275018 0.248757404704419 0.086275797335986 0.019281896440063 -0.000234274234501 0.000000000000000 ;
    0.774252653160083 0.561365462403124 0.340174143951638 0.118690543653993 0.026994415372131 0.000529710009427 0.000000000000000 ;
    0.699628335529835 0.506702309189828 0.306458833028253 0.106545726007728 0.024087348937672 0.000186028160530 0.000000000000000 ;
    0.418586219283289 0.301716205678227 0.180974972382888 0.061948160151029 0.013426517427252 -0.000824159333905 0.000000000000000 ;
    0.164122153043431 0.117722270272710 0.069973087007820 0.023145364157431 0.004129423363668 -0.001178218517235 0.000000000000000 ;
    0.039492164044392 0.027881819205308 0.015950508097287 0.004313401974877 -0.000126748613998 -0.000820123010430 0.000000000000000 ;
    0.002852837253217 0.001333016953284 -0.000046956343774 -0.000893627137459 -0.000715236681916 -0.000324634428591 0.000000000000000 ;
    0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ])

const expected_none_bc = 
expected_data(
    # Expected vpa
    [-3.000000000000000, -2.500000000000000, -2.000000000000000, -1.500000000000000, -1.000000000000000, -0.500000000000000, 0.000000000000000, 0.500000000000000, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
    # Expected vperp
    [0.155051025721682, 0.644948974278318, 1.000000000000000, 1.500000000000000, 2.000000000000000, 2.500000000000000, 3.000000000000000],
    # Expected phi
    [-0.000000000000000, -0.000000000000255],
    # Expected n_ion
    [1.000000000000000, 0.999999999999745],
    # Expected upar_ion
    [0.127188034386488, 0.127188034386486],
    # Expected ppar_ion
    [0.153743696495533, 0.523721622597214],
    # Expected pperp_ion
    [0.719321198401106, 0.534332235350059],
    # Expected qpar_ion
    [-0.018835817064195, -0.003073200116913],
    # Expected v_t_ion
    [1.030335250714622, 1.030335250714620],
    # Expected dSdt_ion
    [0.000000000000000, 0.000000000000006],
    # Expected maxnorm_ion
    [0.818237961233122, 0.011544489536890],
    # Expected L2norm_ion
    [0.091263890321732, 0.001724403977964],
    # Expected f_ion
    [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 ;
    0.000000000000126 0.000000000001327 0.000000000002198 0.000000000000808 0.000000000000040 0.000000000000000 0.000000000000000 ;
    0.000000001527896 0.000000016045073 0.000000026566386 0.000000009773227 0.000000000486580 0.000000000003279 0.000000000000003 ;
    0.000002499614119 0.000026249488962 0.000043462193455 0.000015988847440 0.000000796037841 0.000000005363661 0.000000000004891 ;
    0.000553430604042 0.005811805278321 0.009622808493146 0.003540033410959 0.000176247885456 0.000001187548911 0.000000001082904 ;
    0.016583049988818 0.174145514815615 0.288338796425582 0.106073915297089 0.005281109272943 0.000035583834377 0.000000032448257 ;
    0.067247583764838 0.706194885906433 1.169271477705052 0.430150937795841 0.021415954148542 0.000144299563988 0.000000131584170 ;
    0.036906256469351 0.387567970735544 0.641709792717390 0.236071839939115 0.011753324834776 0.000079193279800 0.000000072214924 ;
    0.002741159726475 0.028786059987797 0.047662082480299 0.017533900267920 0.000872961491394 0.000005881968261 0.000000005363661 ;
    0.000027553687320 0.000289352746733 0.000479091424339 0.000176247885456 0.000008774865523 0.000000059124579 0.000000000053915 ;
    0.000000037483156 0.000000393626228 0.000000651740661 0.000000239761990 0.000000011937047 0.000000000080431 0.000000000000073 ;
    0.000000000006901 0.000000000072469 0.000000000119989 0.000000000044142 0.000000000002198 0.000000000000015 0.000000000000000 ;
    0.000000000000000 0.000000000000002 0.000000000000003 0.000000000000001 0.000000000000000 0.000000000000000 0.000000000000000 ;;;
    0.001112929959917 0.000928574020373 0.000716257049725 0.000386347349794 0.000146064596586 0.000038162643410 0.000002579312656 ;
    0.000864183488743 0.000507744571925 0.000195284355825 0.000041161503156 0.000071835761844 0.000057319780464 0.000030186461744 ;
    0.015564221006663 0.011155935143900 0.006654603228218 0.002232909016491 0.000531799966324 0.000128180348125 0.000137775831815 ;
    0.068941020974092 0.047208617125357 0.026035526684578 0.007738045360968 0.001866911812298 0.000086133996390 0.000446198328738 ;
    0.267910253062402 0.185260775232493 0.103457563876369 0.030840272803816 0.006977692351260 0.000220035700589 0.000949364226388 ;
    0.626991230093102 0.438096127671547 0.249185866389679 0.076147559940302 0.016560482918077 0.000928436079582 0.001451637477326 ;
    0.880946582830658 0.616870248152733 0.352362877334522 0.108413428238932 0.023317178008344 0.001535094138399 0.001734400194449 ;
    0.784248154106488 0.548489037556264 0.312613444675541 0.095901707030626 0.020780602552716 0.001295077448617 0.001640605336678 ;
    0.438604515301999 0.304914270950157 0.171968578479234 0.052105605360981 0.011631879022997 0.000524436887961 0.001226955651128 ;
    0.151869442788786 0.105148527819512 0.058968720625806 0.017833659916292 0.004104783353034 0.000145877095888 0.000675515135110 ;
    0.032535975691312 0.022654155133716 0.012841213948593 0.003941484615372 0.000897238495334 0.000114140695862 0.000265060225606 ;
    0.002792681217546 0.001763495047780 0.000838830276554 0.000225680044807 0.000140352997912 0.000089217284027 0.000071212403173 ;
    0.002184617240084 0.001744920353304 0.001246223675011 0.000599319182019 0.000219873042107 0.000059482467365 -0.000001774176858 ])
###########################################################################################
# to modify the test, with a new expected f, print the new f using the following commands
# in an interative Julia REPL. The path is the path to the .dfns file. 
########################################################################################## 

"""
Function to print data from a moment_kinetics run suitable
for copying into the expected data structure.
"""
function print_output_data_for_test_update(path; write_grid=true, write_pdf=true)
    fid = open_readonly_output_file(path, "dfns")
    input = load_input(fid)
    f_ion_vpavperpzrst = load_pdf_data(fid)
    f_ion = f_ion_vpavperpzrst[:,:,1,1,1,:]
    ntind = size(f_ion,3)
    nvperp = size(f_ion,2)
    nvpa = size(f_ion,1)
    vpa, vpa_spectral = load_coordinate_data(fid, "vpa"; ignore_MPI=true)
    vperp, vperp_spectral = load_coordinate_data(fid, "vperp"; ignore_MPI=true)
    # grid
    function print_grid(coord)
        println("# Expected "*coord.name)
        print("[")
        for k in 1:coord.n
            @printf("%.15f", coord.grid[k])
            if k < coord.n
                print(", ")
            end
        end
        print("],\n")
        return nothing
    end
    # pdf
    function print_pdf(pdf)
        println("# Expected f_ion")
        print("[")
        for k in 1:ntind
            for i in 1:nvpa-1
                for j in 1:nvperp-1
                    @printf("%.15f ", pdf[i,j,k])
                end
                @printf("%.15f ", pdf[i,nvperp,k])
                print(";\n")
            end
            for j in 1:nvperp-1
                @printf("%.15f ", pdf[nvpa,j,k])
            end
            @printf("%.15f ", pdf[nvpa,nvperp,k])
            if k < ntind
                print(";;;\n")
            end
        end
        print("]\n")
        return nothing
    end
    # a moment
    function print_moment(moment,moment_name)
        println("# Expected "*moment_name)
        print("[")
        for k in 1:ntind
            @printf("%.15f", moment[1,1,1,k])
            if k < ntind
                print(", ")
            end
        end
        print("],\n")
        return nothing
    end    
    # a field
    function print_field(field,field_name)
        println("# Expected "*field_name)
        print("[")
        for k in 1:ntind
            @printf("%.15f", field[1,1,k])
            if k < ntind
                print(", ")
            end
        end
        print("],\n")
        return nothing
    end
    # the norms
    function print_norms(pdf)
        L2norm_ion = copy(pdf[1,1,:])
        maxnorm_ion = copy(pdf[1,1,:])
        f_dummy_1 = copy(pdf[:,:,1])
        f_dummy_2 = copy(pdf[:,:,1])
        f_dummy_3 = copy(pdf[:,:,1])
        mass = input["ion_species_1"]["mass"]
        for it in 1:ntind
            @views output = diagnose_F_Maxwellian_serial(pdf[:,:,it],
                                                        f_dummy_1,f_dummy_2,f_dummy_3,
                                                        vpa,vperp,mass)
            maxnorm_ion[it] = output[1]
            L2norm_ion[it] = output[2]
        end
        println("# Expected maxnorm_ion")
        print("[")
        for k in 1:ntind
            @printf("%.15f", maxnorm_ion[k])
            if k < ntind
                print(", ")
            end
        end
        print("],\n")
        println("# Expected L2norm_ion")
        print("[")
        for k in 1:ntind
            @printf("%.15f", L2norm_ion[k])
            if k < ntind
                print(", ")
            end
        end
        print("],\n")
        return nothing
    end
    n_ion_zrst, upar_ion_zrst, ppar_ion_zrst, pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
    phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)
    if write_grid
        print_grid(vpa)
        print_grid(vperp)
    end
    print_field(phi_zrt,"phi")
    print_moment(n_ion_zrst,"n_ion")
    print_moment(upar_ion_zrst,"upar_ion")
    print_moment(ppar_ion_zrst,"ppar_ion")
    print_moment(pperp_ion_zrst,"pperp_ion")
    print_moment(qpar_ion_zrst,"qpar_ion")
    print_moment(v_t_ion_zrst,"v_t_ion")
    print_moment(dSdt_zrst,"dSdt_ion")
    print_norms(f_ion)
    if write_pdf
        print_pdf(f_ion)
    end
    return nothing
end

function diagnose_F_Maxwellian_serial(pdf,pdf_exact,pdf_dummy_1,pdf_dummy_2,vpa,vperp,mass)
    # call this function from a single process
    # construct the local-in-time Maxwellian for this pdf
    dens = get_density(pdf,vpa,vperp)
    upar = get_upar(pdf,vpa,vperp,dens)
    ppar = get_ppar(pdf,vpa,vperp,upar)
    pperp = get_pperp(pdf,vpa,vperp)
    pres = get_pressure(ppar,pperp) 
    vth = sqrt(2.0*pres/(dens*mass))
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            pdf_exact[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
    end
    # check how close the pdf is to the Maxwellian with
    # maximum of difference and L2 of difference
    max_err, L2norm = print_test_data(pdf_exact,pdf,pdf_dummy_1,"F",vpa,vperp,pdf_dummy_2;print_to_screen=false)
    return max_err, L2norm
end

# default inputs for tests
test_input_gauss_legendre = OptionsDict("output" => OptionsDict("run_name" => "gausslegendre_pseudospectral",
                                                                "base_directory" => test_output_directory),
                                        "composition" => OptionsDict("n_ion_species" => 1,
                                                                     "n_neutral_species" => 0,
                                                                     "electron_physics" => "boltzmann_electron_response",
                                                                     "T_e" => 1.0),
                                        "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                                       "initial_temperature" => 1.0,
                                                                       "mass" => 1.0),
                                        "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                            "density_amplitude" => 0.0,
                                                                            "density_phase" => 0.0,
                                                                            "upar_amplitude" => 0.0,
                                                                            "upar_phase" => 0.0,
                                                                            "temperature_amplitude" => 0.0,
                                                                            "temperature_phase" => 0.0),
                                        "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "directed-beam",
                                                                              "vpa0" => 0.1,
                                                                              "vperp0" => 1.0,
                                                                              "vth0" => 0.5),
                                        "vpa" => OptionsDict("ngrid" => 3,
                                                             "L" => 6.0,
                                                             "nelement" => 6,
                                                             "bc" => "zero",
                                                             "discretization" => "gausslegendre_pseudospectral"),
                                        "vperp" => OptionsDict("ngrid" => 3,
                                                               "nelement" => 3,
                                                               "L" => 3.0,
                                                               "discretization" => "gausslegendre_pseudospectral"),
                                        "reactions" => OptionsDict("ionization_frequency" => 0.0,
                                                                   "charge_exchange_frequency" => 0.0),
                                        "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true, "nuii" => 1.0, "frequency_option" => "manual"),
                                        "evolve_moments" => OptionsDict("parallel_pressure" => false,
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
                                        "timestepping" => OptionsDict("dt" => 0.01,
                                                                      "nstep" => 5000,
                                                                      "nwrite" => 5000,
                                                                      "nwrite_dfns" => 5000),
                                       )

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
        name = string(name[1], "_", (stringify_arg(k, v) for (k, v) in args)...)
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
    maxnorm_ion = nothing
    L2norm_ion = nothing
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
            n_ion_zrst, upar_ion_zrst, ppar_ion_zrst, 
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
            f_dummy_1 = copy(f_ion[:,:,1])
            f_dummy_2 = copy(f_ion[:,:,1])
            f_dummy_3 = copy(f_ion[:,:,1])
            L2norm_ion = copy(phi)
            maxnorm_ion = copy(phi)
            mass = input["ion_species_1"]["mass"]
            for it in 1:size(phi,1)
                @views output = diagnose_F_Maxwellian_serial(f_ion[:,:,it],
                                                            f_dummy_1,f_dummy_2,f_dummy_3,
                                                            vpa,vperp,mass)
                maxnorm_ion[it] = output[1]
                L2norm_ion[it] = output[2]
            end
        end
        
        function test_values(tind)
            @testset "tind=$tind" begin
                # Check grids
                #############
                
                @test isapprox(expected.vpa[:], vpa.grid[:], atol=atol)
                @test isapprox(expected.vperp[:], vperp.grid[:], atol=atol)
            
                # Check electrostatic potential
                ###############################
                
                @test isapprox(expected.phi[tind], phi[tind], atol=atol)

                # Check ion particle moments and f
                ######################################

                @test isapprox(expected.n_ion[tind], n_ion[tind], atol=atol)
                @test isapprox(expected.upar_ion[tind], upar_ion[tind], atol=atol)
                @test isapprox(expected.ppar_ion[tind], ppar_ion[tind], atol=atol)
                @test isapprox(expected.pperp_ion[tind], pperp_ion[tind], atol=atol)
                @test isapprox(expected.qpar_ion[tind], qpar_ion[tind], atol=atol)
                @test isapprox(expected.v_t_ion[tind], v_t_ion[tind], atol=atol)
                @test isapprox(expected.dSdt[tind], dSdt[tind], atol=atol)
                @test isapprox(expected.maxnorm_ion[tind], maxnorm_ion[tind], atol=atol)
                @test isapprox(expected.L2norm_ion[tind], L2norm_ion[tind], atol=atol)                                
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
             expected_zero_impose_regularity, 1.0e-14, 2.0e-14;
             vperp=OptionsDict("bc" => vperp_bc))
        end
        @testset "Gauss Legendre no enforced regularity condition at vperp = 0" begin
            run_name = "gausslegendre_pseudospectral_no_regularity"
            vperp_bc = "zero"
            run_test(test_input_gauss_legendre,
            expected_zero,
             1.0e-14, 1.0e-14; vperp=OptionsDict("bc" => vperp_bc))
        end
        @testset "Gauss Legendre no (explicitly) enforced boundary conditions: explicit timestepping" begin
            run_name = "gausslegendre_pseudospectral_none_bc"
            vperp_bc = "none"
            vpa_bc = "none"
            run_test(test_input_gauss_legendre, expected_none_bc, 1.0e-14, 1.0e-14;
                     vperp=OptionsDict("bc" => vperp_bc), vpa=OptionsDict("bc" => vpa_bc))
        end
        @testset "Gauss Legendre no (explicitly) enforced boundary conditions: IMEX timestepping" begin
            run_name = "gausslegendre_pseudospectral_none_bc"
            vperp_bc = "none"
            vpa_bc = "none"
            run_test(test_input_gauss_legendre, expected_none_bc, 5.0e-12, 5.0e-12;
                     vperp=OptionsDict("bc" => vperp_bc), vpa=OptionsDict("bc" => vpa_bc),
                     fokker_planck_collisions_nonlinear_solver=OptionsDict("rtol" => 0.0,
                                                                           "atol" => 1.0e-14,
                                                                           "nonlinear_max_iterations" => 20,),
                     timestepping=OptionsDict("kinetic_ion_solver" => "implicit_ion_fp_collisions",
                                              "type" => "PareschiRusso3(4,3,3)",))
        end
    end
end

end # FokkerPlanckTimeEvolutionTests


using .FokkerPlanckTimeEvolutionTests

FokkerPlanckTimeEvolutionTests.runtests()
