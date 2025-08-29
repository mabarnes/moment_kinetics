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
                                 load_input, load_mk_options, regrid_ion_pdf
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.utils: merge_dict_with_kwargs!
using moment_kinetics.input_structs: options_to_TOML
using moment_kinetics.fokker_planck_test: F_Maxwellian, print_test_data
using moment_kinetics.velocity_moments: get_density, get_upar, get_p

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# The expected output
struct expected_data
    vpa::Array{mk_float, 1}
    vpa_wgts::Array{mk_float, 1}
    vperp::Array{mk_float, 1}
    vperp_wgts::Array{mk_float, 1}
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

include("fokker_planck_time_evolution_expected.jl")
include("fokker_planck_time_evolution_highres_expected.jl")

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
            @printf("%.15e", coord.grid[k])
            if k < coord.n
                print(", ")
            end
        end
        print("],\n")
        println("# Expected "*coord.name*" wgts")
        print("[")
        for k in 1:coord.n
            @printf("%.15e", coord.wgts[k])
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
                    @printf("%.15e ", pdf[i,j,k])
                end
                @printf("%.15e ", pdf[i,nvperp,k])
                print(";\n")
            end
            for j in 1:nvperp-1
                @printf("%.15e ", pdf[nvpa,j,k])
            end
            @printf("%.15e ", pdf[nvpa,nvperp,k])
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
            @printf("%.15e", moment[1,1,1,k])
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
            @printf("%.15e", field[1,1,k])
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
            @printf("%.15e", maxnorm_ion[k])
            if k < ntind
                print(", ")
            end
        end
        print("],\n")
        println("# Expected L2norm_ion")
        print("[")
        for k in 1:ntind
            @printf("%.15e", L2norm_ion[k])
            if k < ntind
                print(", ")
            end
        end
        print("],\n")
        return nothing
    end
    n_ion_zrst, upar_ion_zrst, p_ion_zrst, ppar_ion_zrst, pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
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
    upar = get_upar(pdf, dens, vpa, vperp, false)
    pressure = get_p(pdf, dens, upar, vpa, vperp, false, false)
    vth = sqrt(2.0*pressure/(dens*mass))
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
                                        "ion_species_1" => OptionsDict("initial_density" => 1.5,
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
                                                                              "vpa0" => sqrt(2) * 0.1,
                                                                              "vperp0" => sqrt(2) * 1.0,
                                                                              "vth0" => sqrt(2) * 0.5),
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
                                                           "nelement_local" => 1),
                                        "timestepping" => OptionsDict("dt" => 0.0070710678118654745,
                                                                      "nstep" => 5000,
                                                                      "nwrite" => 500,
                                                                      "nwrite_dfns" => 500))

test_input_gauss_legendre_highres = deepcopy(test_input_gauss_legendre)
test_input_gauss_legendre_highres["vpa"]["ngrid"] = 5
test_input_gauss_legendre_highres["vpa"]["nelement"] = 16
test_input_gauss_legendre_highres["vperp"]["ngrid"] = 5
test_input_gauss_legendre_highres["vperp"]["nelement"] = 8
test_input_gauss_legendre_highres["timestepping"]["dt"] = 0.000070710678118654745

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, expected, atol, final_atol; interp_to_expected=true, args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)
    
    # Convert keyword arguments to a unique name
    function stringify_arg(key, value)
        if isa(value, AbstractDict)
            return string(string(key)[1], (stringify_arg(k, v) for (k, v) in value)...)
        elseif isa(value, AbstractFloat)
            return string(string(key)[1], round(value; sigdigits=2))
        elseif isa(value, AbstractString)
            return string(string(key)[1], value[1:min(6,length(value))])
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
    evolve_density, evolve_upar, evolve_p = nothing, nothing, nothing

    if global_rank[] == 0
        ntime = 0
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
            evolve_density, evolve_upar, evolve_p = load_mk_options(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            # load velocity moments data
            n_ion_zrst, upar_ion_zrst, p_ion_zrst,  ppar_ion_zrst,
            pperp_ion_zrst, qpar_ion_zrst, v_t_ion_zrst, dSdt_zrst = load_ion_moments_data(fid,extended_moments=true)
            
            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            # load coordinates
            r, r_spectral = load_coordinate_data(fid, "r"; ignore_MPI=true)
            z, z_spectral = load_coordinate_data(fid, "z"; ignore_MPI=true)
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa"; ignore_MPI=true)
            vperp, vperp_spectral = load_coordinate_data(fid, "vperp"; ignore_MPI=true)

            # load particle distribution function (pdf) data
            f_ion_vpavperpzrst = load_pdf_data(fid)
            
            if interp_to_expected
                # In case expected data is from a 'full-f' run, but simulation is
                # moment-kinetic, so that f_ion is a shape function rather than a
                # distribution function, convert f_ion to an 'unnormalised' distribution
                # function.
                for it ∈ 1:ntime
                    @views f_ion_vpavperpzrst[:,:,:,:,:,it] .=
                        regrid_ion_pdf(f_ion_vpavperpzrst[:,:,:,:,:,it],
                                       (r=r, z=z, vperp=(grid=expected.vperp, n=length(expected.vperp)),
                                        vpa=(grid=expected.vpa, n=length(expected.vpa))),
                                       (r=r, r_spectral=r_spectral, z=z,
                                        z_spectral=z_spectral, vperp=vperp,
                                        vperp_spectral=vperp_spectral, vpa=vpa,
                                        vpa_spectral=vpa_spectral),
                                       Dict("r"=>false, "z"=>false, "vperp"=>evolve_p,
                                            "vpa"=>(evolve_upar || evolve_p)),
                                       (evolve_density=false, evolve_upar=false,
                                        evolve_p=false,
                                        ion=(dens=n_ion_zrst[:,:,:,it],
                                             upar=upar_ion_zrst[:,:,:,it],
                                             vth=v_t_ion_zrst[:,:,:,it])),
                                       evolve_density, evolve_upar, evolve_p)
                end
            end

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
            f_dummy_1 = copy(f_ion[:,:,1])
            f_dummy_2 = copy(f_ion[:,:,1])
            f_dummy_3 = copy(f_ion[:,:,1])
            L2norm_ion = copy(phi)
            maxnorm_ion = copy(phi)
            mass = input["ion_species_1"]["mass"]

            for it in 1:ntime
                @views output = diagnose_F_Maxwellian_serial(f_ion[:,:,it],
                                                             f_dummy_1, f_dummy_2, f_dummy_3,
                                                             (grid=expected.vpa, n=length(expected.vpa), wgts=expected.vpa_wgts),
                                                             (grid=expected.vperp, n=length(expected.vperp), wgts=expected.vperp_wgts), mass)
                maxnorm_ion[it] = output[1]
                L2norm_ion[it] = output[2]
            end
        end
        
        function test_values(tind)
            @testset "tind=$tind" begin
                # Check grids
                #############

                if tind == ntime
                    this_atol = final_atol
                else
                    this_atol = atol
                end
                
                if !evolve_density
                    # For moment kinetic runs these are grids in the normalised velocity
                    # coordinates. It does not make sense to compare these to the
                    # unnormalised velocity from the 'full-f' run that generated the
                    # expected data.
                    @test isapprox(expected.vpa[:], vpa.grid[:], atol=this_atol)
                    @test isapprox(expected.vperp[:], vperp.grid[:], atol=this_atol)
                end
            
                # Check electrostatic potential
                ###############################
                
                @test isapprox(expected.phi[tind], phi[tind], atol=this_atol)

                # Check ion particle moments and f
                ######################################

                @test isapprox(expected.n_ion[tind], n_ion[tind], atol=this_atol)
                @test isapprox(expected.upar_ion[tind], upar_ion[tind], atol=this_atol)
                @test isapprox(expected.ppar_ion[tind], ppar_ion[tind], atol=this_atol)
                @test isapprox(expected.pperp_ion[tind], pperp_ion[tind], atol=this_atol)
                @test isapprox(expected.qpar_ion[tind], qpar_ion[tind], atol=this_atol)
                @test isapprox(expected.v_t_ion[tind], v_t_ion[tind], atol=this_atol)
                @test isapprox(expected.dSdt[tind], dSdt[tind], atol=this_atol)
                @test isapprox(expected.maxnorm_ion[tind], maxnorm_ion[tind], atol=this_atol)
                @test isapprox(expected.L2norm_ion[tind], L2norm_ion[tind], atol=this_atol)
                f_err = @. abs(expected.f_ion[:,:,tind] - f_ion[:,:,tind])
                max_f_err = maximum(f_err)
                @test isapprox(max_f_err, 0.0, atol=this_atol)
                @test elementwise_isapprox(expected.f_ion[:,:,tind], f_ion[:,:,tind], rtol=this_atol, atol=this_atol*1.0e-3)
            end
        end

        for it ∈ 1:ntime
            test_values(it)
        end
    end
end


"""
The `highres=true` version of these tests is included as it has enough resolution to give
reasonable agreement between 'full-f' and moment-kinetic versions of each test. However,
it is too slow to include even in the `@long` tests, so is only available to be run
manually.
"""
function runtests(; highres=false)
    @testset "Fokker Planck dFdt = C[F,F] relaxation test" verbose=use_verbose begin
        println("Fokker Planck dFdt = C[F,F] relaxation test")

        @testset "evolve_density=$evolve_density, evolve_upar=$evolve_upar, evolve_p=$evolve_p" for
                (evolve_density, evolve_upar, evolve_p, tol1, tol2, tol3, tol4) ∈
                    ((false, false, false, 2.0e-14, 2.0e-14, highres ? 4.0e-3 : 1.0e-5, highres ? 1.0e-4 : 5.0e-12),
                     (true, false, false, highres ? 2.0e-8 : 2.0e-14, highres ? 2.0e-8 : 1.0e-14, highres ? 4.0e-3 : 1.0e-5, highres ? 1.0e-4 : 5.0e-12),
                     (true, true, false, highres ? 2.0e-3 : 2.0e-14, highres ? 3.0e-4 : 2.0e-14, highres ? 4.0e-3 : 1.0e-5, highres ? 3.0e-4 : 5.0e-12),
                     (true, true, true, highres ? 2.0e-3 : 2.0e-14, highres ? 2.0e-3 : 2.0e-14, highres ? 4.0e-3 : 1.0e-5, highres ? 2.0e-3 : 5.0e-12),
                    )
            println("  evolve_density=$evolve_density, evolve_upar=$evolve_upar, evolve_p=$evolve_p:")

            if evolve_p
                Lvperp = test_input_gauss_legendre["vperp"]["L"] / sqrt(2.0)
                Lvpa = test_input_gauss_legendre["vpa"]["L"] / sqrt(2.0)
            else
                Lvperp = test_input_gauss_legendre["vperp"]["L"]
                Lvpa = test_input_gauss_legendre["vpa"]["L"]
            end

            # GaussLegendre pseudospectral
            if !highres || !evolve_density
                # This case does not conserve moments well enough to compare full-f to
                # moment-kinetic cases, so skip unless separate expected output is saved
                # for full-f and moment-kinetic.
                @testset "Gauss Legendre base" begin
                    run_name = "gausslegendre_pseudospectral"
                    vperp_bc = "zero-impose-regularity"
                    if highres
                        this_expected = expected_zero_impose_regularity_highres
                        this_input = test_input_gauss_legendre_highres
                    else
                        this_expected = expected_zero_impose_regularity[(evolve_density, evolve_upar, evolve_p)]
                        this_input = test_input_gauss_legendre
                    end
                    run_test(this_input, this_expected, tol1, tol1;
                             interp_to_expected=highres,
                             vperp=OptionsDict("bc" => vperp_bc, "L" => Lvperp),
                             vpa=OptionsDict("L" => Lvpa),
                             evolve_moments=OptionsDict("density" => evolve_density, "parallel_flow" => evolve_upar, "pressure" => evolve_p))
                end
            end
            @testset "Gauss Legendre no enforced regularity condition at vperp = 0" begin
                run_name = "gausslegendre_pseudospectral_no_regularity"
                vperp_bc = "zero"
                if highres
                    this_expected = expected_zero_highres
                    this_input = test_input_gauss_legendre_highres
                else
                    this_expected = expected_zero[(evolve_density, evolve_upar, evolve_p)]
                    this_input = test_input_gauss_legendre
                end
                run_test(this_input, this_expected, tol1, tol2;
                         interp_to_expected=highres,
                         vperp=OptionsDict("bc" => vperp_bc, "L" => Lvperp),
                         vpa=OptionsDict("L" => Lvpa),
                         evolve_moments=OptionsDict("density" => evolve_density, "parallel_flow" => evolve_upar, "pressure" => evolve_p))
            end
            @testset "Gauss Legendre no (explicitly) enforced boundary conditions: explicit timestepping" begin
                run_name = "gausslegendre_pseudospectral_none_bc"
                vperp_bc = "none"
                vpa_bc = "none"
                if highres
                    this_expected = expected_none_bc_highres
                    this_input = test_input_gauss_legendre_highres
                else
                    this_expected = expected_none_bc[(evolve_density, evolve_upar, evolve_p)]
                    this_input = test_input_gauss_legendre
                end
                run_test(this_input, this_expected, tol1, tol2;
                         interp_to_expected=highres,
                         vperp=OptionsDict("bc" => vperp_bc, "L" => Lvperp),
                         vpa=OptionsDict("bc" => vpa_bc, "L" => Lvpa),
                         evolve_moments=OptionsDict("density" => evolve_density, "parallel_flow" => evolve_upar, "pressure" => evolve_p))
            end
            @testset "Gauss Legendre no (explicitly) enforced boundary conditions: IMEX timestepping PareschiRusso3(4,3,3)" begin
                run_name = "gausslegendre_pseudospectral_none_bc"
                vperp_bc = "none"
                vpa_bc = "none"
                if highres
                    this_expected = expected_none_bc_highres
                    this_input = test_input_gauss_legendre_highres
                else
                    this_expected = expected_none_bc[(evolve_density, evolve_upar, evolve_p)]
                    this_input = test_input_gauss_legendre
                end
                run_test(this_input, this_expected, tol3, tol4;
                         interp_to_expected=highres,
                         vperp=OptionsDict("bc" => vperp_bc, "L" => Lvperp),
                         vpa=OptionsDict("bc" => vpa_bc, "L" => Lvpa),
                         evolve_moments=OptionsDict("density" => evolve_density, "parallel_flow" => evolve_upar, "pressure" => evolve_p),
                         fokker_planck_collisions_nonlinear_solver=OptionsDict("rtol" => 0.0,
                                                                               "atol" => 1.0e-14,
                                                                               "nonlinear_max_iterations" => 20,),
                         timestepping=OptionsDict("kinetic_ion_solver" => "implicit_ion_fp_collisions",
                                                  "type" => "PareschiRusso3(4,3,3)",))
            end
            @testset "Gauss Legendre no (explicitly) enforced boundary conditions: IMEX timestepping EulerIMEX" begin
                run_name = "gausslegendre_pseudospectral_none_bc"
                vperp_bc = "none"
                vpa_bc = "none"
                if highres
                    this_expected = expected_none_bc_highres
                    this_input = test_input_gauss_legendre_highres
                else
                    this_expected = expected_none_bc[(evolve_density, evolve_upar, evolve_p)]
                    this_input = test_input_gauss_legendre
                end
                run_test(this_input, this_expected, 10.0 * tol3, tol4;
                         interp_to_expected=highres,
                         vperp=OptionsDict("bc" => vperp_bc, "L" => Lvperp),
                         vpa=OptionsDict("bc" => vpa_bc, "L" => Lvpa),
                         evolve_moments=OptionsDict("density" => evolve_density, "parallel_flow" => evolve_upar, "pressure" => evolve_p),
                         fokker_planck_collisions_nonlinear_solver=OptionsDict("rtol" => 0.0,
                                                                               "atol" => 1.0e-14,
                                                                               "nonlinear_max_iterations" => 20,),
                         timestepping=OptionsDict("kinetic_ion_solver" => "implicit_ion_fp_collisions",
                                                  "type" => "EulerIMEX",))
            end
        end
    end
end

end # FokkerPlanckTimeEvolutionTests


using .FokkerPlanckTimeEvolutionTests

FokkerPlanckTimeEvolutionTests.runtests()
