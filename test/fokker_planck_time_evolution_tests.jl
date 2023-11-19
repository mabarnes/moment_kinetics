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
   vec([-3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 ]),
   vec([0.155051025721682 0.644948974278318 1.000000000000000 1.500000000000000 2.000000000000000 2.500000000000000 3.000000000000000 ]),
   # Expected phi:
   vec([-1.267505494648937 -1.275240686416660 ]),
   # Expected n_charged:
   vec([0.281533032234007 0.279363721076024 ]),
   # Expected upar_charged:
   vec([0.0 0.0 ]),
   # Expected ppar_charged:
   vec([0.179822802480489 0.147624463307870 ]),
   # Expected pperp_charged
   vec([0.143401466675068 0.158821758223731 ]),
   # Expected qpar_charged
   vec([0.0 0.0 ]),
   # Expected v_t_charged
   vec([1.051172608301042 1.053709630977256 ]),
   # Expected dSdt
   vec([0.0 0.000008786074305 ]),
   # Expected f_charged:
  [ 0.000000000000000 ; 0.000619960016181 ; 0.005882016862627 ; 0.033848669972256 ; 0.118138103172003 ; 0.229579465209362 ; 0.000000000000000 ; 0.229579465209362 ; 0.118138103172003 ; 0.033848669972256 ; 0.005882016862627 ; 0.000619960016181 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000478053009971 ; 0.004535640674379 ; 0.026100809957764 ; 0.091096642266611 ; 0.177029407552679 ; 0.000000000000000 ; 0.177029407552679 ; 0.091096642266611 ; 0.026100809957764 ; 0.004535640674379 ; 0.000478053009971 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000266581711212 ; 0.002529256854782 ; 0.014554868262370 ; 0.050799175561231 ; 0.098718764270694 ; 0.000000000000000 ; 0.098718764270694 ; 0.050799175561231 ; 0.014554868262370 ; 0.002529256854782 ; 0.000266581711212 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000076376939017 ; 0.000724644221386 ; 0.004170039574837 ; 0.014554207474836 ; 0.028283399503664 ; 0.000000000000000 ; 0.028283399503664 ; 0.014554207474836 ; 0.004170039574837 ; 0.000724644221386 ; 0.000076376939017 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000013272321882 ; 0.000125924283949 ; 0.000724644221264 ; 0.002529142026698 ; 0.004914917865936 ; 0.000000000000000 ; 0.004914917865936 ; 0.002529142026698 ; 0.000724644221264 ; 0.000125924283949 ; 0.000013272321882 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000001398892434 ; 0.000013272321882 ; 0.000076376939004 ; 0.000266569608421 ; 0.000518028531855 ; 0.000000000000000 ; 0.000518028531855 ; 0.000266569608421 ; 0.000076376939004 ; 0.000013272321882 ; 0.000001398892434 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ;;;
    0.000000000000000 ; 0.000138780288865 ; 0.005791061761509 ; 0.027815242205125 ; 0.081445074775077 ; 0.152354938126957 ; 0.186355844666422 ; 0.152354938126957 ; 0.081445074775077 ; 0.027815242205125 ; 0.005791061761509 ; 0.000138780288865 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000041802336103 ; 0.004577414398302 ; 0.022242056573360 ; 0.065733717950559 ; 0.123790379285355 ; 0.151688746487743 ; 0.123790379285355 ; 0.065733717950559 ; 0.022242056573360 ; 0.004577414398302 ; 0.000041802336103 ; 0.000000000000000 ;;
    0.000000000000000 ; -0.000102715219225 ; 0.002768824456286 ; 0.013936837294540 ; 0.042320490360991 ; 0.081223176170832 ; 0.100027475688989 ; 0.081223176170832 ; 0.042320490360991 ; 0.013936837294540 ; 0.002768824456286 ; -0.000102715219225 ; 0.000000000000000 ;;
    0.000000000000000 ; -0.000164682482058 ; 0.000767345547797 ; 0.004535511738255 ; 0.014877992136829 ; 0.029791500386470 ; 0.037097641503996 ; 0.029791500386470 ; 0.014877992136829 ; 0.004535511738255 ; 0.000767345547797 ; -0.000164682482058 ; 0.000000000000000 ;;
    0.000000000000000 ; -0.000091767217558 ; 0.000022353567868 ; 0.000693446611185 ; 0.002889786006700 ; 0.006284514040367 ; 0.007979739613849 ; 0.006284514040367 ; 0.002889786006700 ; 0.000693446611185 ; 0.000022353567868 ; -0.000091767217558 ; 0.000000000000000 ;;
    0.000000000000000 ; -0.000032024960906 ; -0.000089243114443 ; -0.000201682249005 ; -0.000175449913808 ; 0.000111463339173 ; 0.000290773816557 ; 0.000111463339173 ; -0.000175449913808 ; -0.000201682249005 ; -0.000089243114443 ; -0.000032024960906 ; 0.000000000000000 ;;
    0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ; 0.000000000000000 ])
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
                              "vperp_bc" => "periodic",
                              "vperp_discretization" => "gausslegendre_pseudospectral",
                              "n_rk_stages" => 4,
                              "split_operators" => false,
                              "ionization_frequency" => 0.0,
                              "charge_exchange_frequency" => 0.0,
                              "constant_ionization_rate" => false,
                              "electron_physics" => "boltzmann_electron_response",
                              "nuii" => 1.0,
                              "use_semi_lagrange" => false,
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
                              "dt" => 0.01,
                              "nstep" => 5000,
                              "nwrite" => 5000,
                              "nwrite_dfns" => 5000 )


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
    #input = test_input

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
