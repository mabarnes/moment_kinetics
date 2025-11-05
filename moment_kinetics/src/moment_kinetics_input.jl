"""
"""
module moment_kinetics_input

export mk_input
export performance_test
#export advective_form
export read_input_file

using ..type_definitions: mk_float, mk_int, OptionsDict
using ..communication
using ..coordinates: define_coordinate, get_coordinate_input
using ..external_sources
using ..file_io: io_has_parallel, input_option_error, open_ascii_output_file,
                 setup_io_input
using ..krook_collisions: setup_krook_collisions_input
using ..maxwell_diffusion: setup_mxwl_diff_collisions_input
using ..fokker_planck: setup_fkpl_collisions_input
using ..finite_differences: fd_check_option
using ..input_structs
using ..manufactured_solns: setup_manufactured_solutions
using ..numerical_dissipation: setup_numerical_dissipation
using ..reference_parameters
using ..geo: init_magnetic_geometry, setup_geometry_input
using ..species_input: get_species_input
using ..looping
using MPI
using TOML

"""
Read input from a TOML file
"""
function read_input_file(input_filename::String)
    input = convert_to_sorted_nested_OptionsDict(TOML.parsefile(input_filename))

    # Use input_filename (without the extension) as default for "run_name"
    if !("output" ∈ keys(input) && "run_name" in keys(input["output"]))
        if !("output" ∈ keys(input))
            input["output"] = OptionsDict()
        end
        input["output"]["run_name"] = splitdir(splitext(input_filename)[1])[end]
    end

    return input
end

"""
Process user-supplied inputs

`ignore_MPI` should be false when actually running a simulation, but defaults to true for
other situations (e.g. when post-processing).
"""
function mk_input(input_dict=OptionsDict("output" => OptionsDict("run_name" => "default"));
                  save_inputs_to_txt=false, ignore_MPI=true,
                  debug_loop_type::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
                  debug_loop_parallel_dims::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
                  warn_unexpected=false, write_output=true)

    # Check for input options that used to exist, but do not any more. If these are
    # present, the user probably needs to update their input file.
    removed_options_list = ("Bzed", "Bmag", "rhostar", "geometry_option", "pitch",
                            "DeltaB", "n_ion_species", "n_neutral_species",
                            "recycling_fraction", "gyrokinetic_ions", "T_e", "T_wall",
                            "z_IC_option1", "z_IC_option2", "vpa_IC_option1",
                            "vpa_IC_option2", "boltzmann_electron_response",
                            "boltzmann_electron_response_with_simple_sheath",
                            "electron_physics", "nstep", "dt",
                            ("$(c)_$(opt)"
                             for c ∈ ("r", "z", "vperp", "vpa", "vzeta", "vr", "vz"),
                                 opt ∈ ("ngrid", "nelement", "nelement_local", "L",
                                        "discretization", "cheb_option",
                                        "finite_difference_option",
                                        "element_spacing_option", "bc")
                            )...,
                            "force_Er_zero_at_wall", "evolve_moments_density",
                            "evolve_moments_parallel_flow",
                            "evolve_moments_parallel_pressure",
                            "evolve_moments_conservation", "charge_exchange_frequency",
                            "electron_charge_exchange_frequency", "ionization_frequency",
                            "electron_ionization_frequency", "ionization_energy", "nu_ei",
                            "run_name", "base_directory",
                           )
    input_keys = keys(input_dict)
    for opt in removed_options_list
        if opt ∈ input_keys
            message = "Option '$opt' is no longer used. Please update your input file. " *
                      "The option may have been moved into an input file section - " *
                      "there are no longer any top-level options (i.e. ones not in a " *
                      "section). You may need to set some new options to replicate the " *
                      "effect of the removed ones."
            if warn_unexpected
                println(message)
            else
                error(message)
            end
        end
    end
    # Check for input options that used to exist in some section, but do not any more. If
    # these are present, the user probably needs to update their input file.
    removed_section_options_list = Dict("evolve_moments" => ("parallel_pressure",),
                                        "timestepping" => ("implicit_electron_advance",
                                                           "implicit_electron_time_evolving",
                                                           "implicit_electron_ppar",),
                                       )
    for (section_name, removed_options) ∈ pairs(removed_section_options_list)
        section = get(input_dict, section_name, OptionsDict())
        section_keys = keys(section)
        for opt ∈ removed_options
            if opt ∈ section_keys
                message = "Option '$opt' in section [$section_name] is no longer used. " *
                          "Please update your input file. You may need to set some new " *
                          "option(s) to replicate the effect of the removed ones."
                if warn_unexpected
                    println(message)
                else
                    error(message)
                end
            end
        end
    end

    # if evolve_moments.density = true, evolve density via continuity eqn
    # and g = f/n via modified drift kinetic equation
    evolve_moments_settings = set_defaults_and_check_section!(
        input_dict, "evolve_moments", warn_unexpected;
        density=false,
        parallel_flow=false,
        pressure=false,
        moments_conservation=false,
       )
    evolve_moments = Dict_to_NamedTuple(evolve_moments_settings)

    # Reference parameters that define the conversion between physical quantities and
    # normalised values used in the code.
    reference_params = setup_reference_parameters(input_dict, warn_unexpected)
    
    ## set geometry_input
    geometry_in = setup_geometry_input(input_dict, warn_unexpected)
    
    em_input = set_defaults_and_check_section!(
        input_dict, em_fields_input, warn_unexpected, "em_fields"
       )

    manufactured_solns_input = setup_manufactured_solutions(input_dict, warn_unexpected)

    reactions_input = set_defaults_and_check_section!(
        input_dict, reactions, warn_unexpected
       )
    electron_fluid_collisions_input = set_defaults_and_check_section!(
        input_dict, electron_fluid_collisions, warn_unexpected
       )
    # set up krook collision inputs
    krook_input = setup_krook_collisions_input(input_dict, warn_unexpected)
    # set up Fokker-Planck collision inputs
    fkpl_input = setup_fkpl_collisions_input(input_dict, warn_unexpected)
    # set up maxwell diffusion collision inputs
    mxwl_diff_input = setup_mxwl_diff_collisions_input(input_dict, warn_unexpected)
    # write total collision struct using the structs above, as each setup function 
    # for the collisions outputs itself a struct of the type of collision, which
    # is a substruct of the overall collisions_input struct.
    collisions = collisions_input(reactions_input, electron_fluid_collisions_input,
                                  krook_input, fkpl_input, mxwl_diff_input)

    num_diss_params = setup_numerical_dissipation(input_dict, warn_unexpected)

    # set up distributed-memory MPI information for z and r coords
    # need grid and MPI information to determine these values 
    # MRH just put dummy values now 
    r_coord_input = get_coordinate_input(input_dict, "r"; ignore_MPI=ignore_MPI,
                                         warn_unexpected=warn_unexpected)
    z_coord_input = get_coordinate_input(input_dict, "z"; ignore_MPI=ignore_MPI,
                                         warn_unexpected=warn_unexpected)
    if ignore_MPI
        irank_z = irank_r = 0
        nrank_z = nrank_r = 1
        comm_sub_z, comm_sub_r = setup_serial_MPI()
    else
        irank_z, nrank_z, comm_sub_z, irank_r, nrank_r, comm_sub_r =
            setup_distributed_memory_MPI(z_coord_input.nelement,
                                         z_coord_input.nelement_local,
                                         r_coord_input.nelement,
                                         r_coord_input.nelement_local)
    end

    # read composition and species data
    composition = get_species_input(input_dict, warn_unexpected; ignore_MPI=ignore_MPI)
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species

    # parameters related to the time stepping
    timestepping_section = set_defaults_and_check_section!(
        input_dict, "timestepping", warn_unexpected;
        nstep=5,
        dt=0.00025/sqrt(2.0*composition.ion[1].initial_temperature),
        CFL_prefactor=-1.0,
        nwrite=1,
        nwrite_dfns=-1,
        exact_output_times=false,
        type="SSPRK4",
        split_operators=false,
        add_random_noise=-1.0,
        random_noise_seed=-1,
        print_nT_live=false,
        steady_state_residual=false,
        converged_residual_value=-1.0,
        rtol=1.0e-5,
        atol=1.0e-12,
        atol_upar=-1.0,
        step_update_prefactor=0.9,
        max_increase_factor=1.05,
        max_increase_factor_near_last_fail=Inf,
        last_fail_proximity_factor=1.05,
        minimum_dt=0.0,
        maximum_dt=Inf,
        implicit_braginskii_conduction=true,
        kinetic_electron_solver=(composition.electron_physics ∈ (kinetic_electrons,
                                                                 kinetic_electrons_with_temperature_equation)
                                 ? implicit_p_implicit_pseudotimestep : null_kinetic_electrons),
        kinetic_electron_preconditioner="default",
        kinetic_ion_solver=full_explicit_ion_advance,
        constraint_forcing_rate=0.0,
        write_after_fixed_step_count=false,
        write_error_diagnostics=false,
        write_steady_state_diagnostics=false,
        high_precision_error_sum=false,
        debug_io=false,
       )
    if timestepping_section["nwrite"] > timestepping_section["nstep"]
        timestepping_section["nwrite"] = timestepping_section["nstep"]
    end
    if timestepping_section["nwrite_dfns"] < 0
        timestepping_section["nwrite_dfns"] = timestepping_section["nstep"]
    elseif timestepping_section["nwrite_dfns"] > timestepping_section["nstep"]
        timestepping_section["nwrite_dfns"] = timestepping_section["nstep"]
    end
    if timestepping_section["atol_upar"] < 0.0
        timestepping_section["atol_upar"] = 1.0e-2 * timestepping_section["rtol"]
    end

    # parameters related to electron time stepping
    electron_timestepping_section = set_defaults_and_check_section!(
        input_dict, "electron_timestepping", warn_unexpected;
        nstep=50000,
        dt=timestepping_section["dt"] * sqrt(composition.me_over_mi),
        CFL_prefactor=timestepping_section["CFL_prefactor"],
        nwrite=-1,
        nwrite_dfns=-1,
        type=timestepping_section["type"],
        split_operators=false,
        converged_residual_value=1.0e-3,
        rtol=timestepping_section["rtol"],
        atol=timestepping_section["atol"],
        step_update_prefactor=timestepping_section["step_update_prefactor"],
        max_increase_factor=timestepping_section["max_increase_factor"],
        max_increase_factor_near_last_fail=timestepping_section["max_increase_factor_near_last_fail"],
        last_fail_proximity_factor=timestepping_section["last_fail_proximity_factor"],
        minimum_dt=timestepping_section["minimum_dt"] * sqrt(composition.me_over_mi),
        maximum_dt=timestepping_section["maximum_dt"] * sqrt(composition.me_over_mi),
        constraint_forcing_rate=1.0e5,
        write_after_fixed_step_count=false,
        write_error_diagnostics=false,
        write_steady_state_diagnostics=false,
        high_precision_error_sum=timestepping_section["high_precision_error_sum"],
        initialization_residual_value=1.0,
        decrease_dt_iteration_threshold=100,
        increase_dt_iteration_threshold=20,
        cap_factor_ion_dt=10.0,
        max_pseudotimesteps=1000,
        max_pseudotime=1.0e-2,
        include_wall_bc_in_preconditioner=false,
        no_restart=false,
        skip_electron_initial_solve=false,
        debug_io=false,
       )
    if electron_timestepping_section["nwrite"] < 0
        electron_timestepping_section["nwrite"] = electron_timestepping_section["nstep"]
    elseif electron_timestepping_section["nwrite"] > electron_timestepping_section["nstep"]
        electron_timestepping_section["nwrite"] = electron_timestepping_section["nstep"]
    end
    if electron_timestepping_section["nwrite_dfns"] < 0
        electron_timestepping_section["nwrite_dfns"] = electron_timestepping_section["nstep"]
    elseif electron_timestepping_section["nwrite_dfns"] > electron_timestepping_section["nstep"]
        electron_timestepping_section["nwrite_dfns"] = electron_timestepping_section["nstep"]
    end
    # Make a copy because "stopfile_name" is not a separate input for the electrons, so we
    # do not want to add a value to the `input_dict`. We also add a few dummy inputs that
    # are not actually used for electrons.
    electron_timestepping_section = copy(electron_timestepping_section)
    electron_timestepping_section["atol_upar"] = NaN
    electron_timestepping_section["steady_state_residual"] = true
    electron_timestepping_section["print_nT_live"] = false
    if !(0.0 < electron_timestepping_section["step_update_prefactor"] < 1.0)
        error("[electron_timestepping] step_update_prefactor="
              * "$(electron_timestepping_section["step_update_prefactor"]) must be between "
              * "0.0 and 1.0.")
    end
    if electron_timestepping_section["max_increase_factor"] ≤ 1.0
        error("[electron_timestepping] max_increase_factor="
              * "$(electron_timestepping_section["max_increase_factor"]) must be greater than "
              * "1.0.")
    end
    if electron_timestepping_section["max_increase_factor_near_last_fail"] ≤ 1.0
        error("[electron_timestepping] max_increase_factor_near_last_fail="
              * "$(electron_timestepping_section["max_increase_factor_near_last_fail"]) must "
              * "be greater than 1.0.")
    end
    if !isinf(electron_timestepping_section["max_increase_factor_near_last_fail"]) &&
        electron_timestepping_section["max_increase_factor_near_last_fail"] > electron_timestepping_section["max_increase_factor"]
        error("[electron_timestepping] max_increase_factor_near_last_fail="
              * "$(electron_timestepping_section["max_increase_factor_near_last_fail"]) should be "
              * "less than max_increase_factor="
              * "$(electron_timestepping_section["max_increase_factor"]).")
    end
    if electron_timestepping_section["last_fail_proximity_factor"] ≤ 1.0
        error("[electron_timestepping] last_fail_proximity_factor="
              * "$(electron_timestepping_section["last_fail_proximity_factor"]) must be "
              * "greater than 1.0.")
    end
    if electron_timestepping_section["minimum_dt"] > electron_timestepping_section["maximum_dt"]
        error("[electron_timestepping] minimum_dt="
              * "$(electron_timestepping_section["minimum_dt"]) must be less than "
              * "maximum_dt=$(electron_timestepping_section["maximum_dt"])")
    end
    if electron_timestepping_section["maximum_dt"] ≤ 0.0
        error("[electron_timestepping] maximum_dt="
              * "$(electron_timestepping_section["maximum_dt"]) must be positive")
    end

    # Make a copy of `timestepping_section` here as we do not want to add
    # `electron_timestepping_section` to the `input_dict` because there is already an
    # "electron_timestepping" section containing the input info - we only want to put
    # `electron_timestepping_section` into the Dict that is used to make
    # `timestepping_input`, so that it becomes part of `timestepping_input`.
    timestepping_section = copy(timestepping_section)
    timestepping_section["electron_t_input"] = electron_timestepping_section
    if !(0.0 < timestepping_section["step_update_prefactor"] < 1.0)
        error("step_update_prefactor=$(timestepping_section["step_update_prefactor"]) must "
              * "be between 0.0 and 1.0.")
    end
    if timestepping_section["max_increase_factor"] ≤ 1.0
        error("max_increase_factor=$(timestepping_section["max_increase_factor"]) must "
              * "be greater than 1.0.")
    end
    if timestepping_section["max_increase_factor_near_last_fail"] ≤ 1.0
        error("max_increase_factor_near_last_fail="
              * "$(timestepping_section["max_increase_factor_near_last_fail"]) must be "
              * "greater than 1.0.")
    end
    if !isinf(timestepping_section["max_increase_factor_near_last_fail"]) &&
        timestepping_section["max_increase_factor_near_last_fail"] > timestepping_section["max_increase_factor"]
        error("max_increase_factor_near_last_fail="
              * "$(timestepping_section["max_increase_factor_near_last_fail"]) should be "
              * "less than max_increase_factor="
              * "$(timestepping_section["max_increase_factor"]).")
    end
    if timestepping_section["last_fail_proximity_factor"] ≤ 1.0
        error("last_fail_proximity_factor="
              * "$(timestepping_section["last_fail_proximity_factor"]) must be "
              * "greater than 1.0.")
    end
    if timestepping_section["minimum_dt"] > timestepping_section["maximum_dt"]
        error("minimum_dt=$(timestepping_section["minimum_dt"]) must be less than "
              * "maximum_dt=$(timestepping_section["maximum_dt"])")
    end
    if timestepping_section["maximum_dt"] ≤ 0.0
        error("maximum_dt=$(timestepping_section["maximum_dt"]) must be positive")
    end

    io_immutable = setup_io_input(input_dict, timestepping_section, warn_unexpected;
                                  ignore_MPI=ignore_MPI, write_output=write_output)

    # this is the directory where the simulation data will be stored
    timestepping_section["stopfile_name"] = joinpath(io_immutable.output_dir, "stop")
    electron_timestepping_section["stopfile_name"] = joinpath(io_immutable.output_dir, "stop")

    # initialize z grid and write grid point locations to file
    if ignore_MPI || !write_output
        run_directory = nothing
    else
        run_directory = io_immutable.output_dir
    end
    z, z_spectral = define_coordinate(z_coord_input; parallel_io=io_immutable.parallel_io,
                                      run_directory=run_directory, ignore_MPI=ignore_MPI,
                                      irank=irank_z, nrank=nrank_z, comm=comm_sub_z)
    # initialize r grid and write grid point locations to file
    r, r_spectral = define_coordinate(r_coord_input; parallel_io=io_immutable.parallel_io,
                                      run_directory=run_directory, ignore_MPI=ignore_MPI,
                                      irank=irank_r, nrank=nrank_r, comm=comm_sub_r)
    # initialize vpa grid and write grid point locations to file
    vpa, vpa_spectral = define_coordinate(input_dict, "vpa", warn_unexpected;
                                          parallel_io=io_immutable.parallel_io,
                                          run_directory=run_directory,
                                          ignore_MPI=ignore_MPI)
    # initialize vperp grid and write grid point locations to file
    vperp, vperp_spectral = define_coordinate(input_dict, "vperp", warn_unexpected;
                                              parallel_io=io_immutable.parallel_io,
                                              run_directory=run_directory,
                                              ignore_MPI=ignore_MPI)
    # initialize gyrophase grid and write grid point locations to file
    gyrophase, gyrophase_spectral = define_coordinate(input_dict, "gyrophase",
                                                      warn_unexpected;
                                                      parallel_io=io_immutable.parallel_io,
                                                      run_directory=run_directory,
                                                      ignore_MPI=ignore_MPI)
    # initialize vz grid and write grid point locations to file
    vz, vz_spectral = define_coordinate(input_dict, "vz", warn_unexpected;
                                        parallel_io=io_immutable.parallel_io,
                                        run_directory=run_directory,
                                        ignore_MPI=ignore_MPI)
    # initialize vr grid and write grid point locations to file
    vr, vr_spectral = define_coordinate(input_dict, "vr", warn_unexpected;
                                        parallel_io=io_immutable.parallel_io,
                                        run_directory=run_directory,
                                        ignore_MPI=ignore_MPI)
    # initialize vr grid and write grid point locations to file
    vzeta, vzeta_spectral = define_coordinate(input_dict, "vzeta", warn_unexpected;
                                              parallel_io=io_immutable.parallel_io,
                                              run_directory=run_directory,
                                              ignore_MPI=ignore_MPI)

        # Create loop range variables for shared-memory-parallel loops
    if debug_loop_type === nothing
        # Non-debug case used for all simulations
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                   s=composition.n_ion_species,
                                   sn=composition.n_neutral_species,
                                   r=r.n, z=z.n, vperp=vperp.n, vpa=vpa.n,
                                   vzeta=vzeta.n, vr=vr.n, vz=vz.n)
    else
        if debug_loop_parallel_dims === nothing
            error("debug_loop_parallel_dims must not be `nothing` when debug_loop_type "
                  * "is not `nothing`.")
        end
        # Debug initialisation only used by tests in `debug_test/`
        looping.debug_setup_loop_ranges_split_one_combination!(
            debug_loop_type, debug_loop_parallel_dims...; s=composition.n_ion_species,
            sn=composition.n_neutral_species, r=r.n, z=z.n, vperp=vperp.n, vpa=vpa.n,
            vzeta=vzeta.n, vr=vr.n, vz=vz.n)
    end

    geometry = init_magnetic_geometry(geometry_in,z,r,z_spectral)
    if any(geometry.dBdz .!= 0.0) &&
            (evolve_moments.density || evolve_moments.parallel_flow ||
             evolve_moments.pressure)
        error("Mirror terms not yet implemented for moment-kinetic modes")
    end

    species_immutable = (ion = composition.ion, neutral = composition.neutral)

    # Ideally `check_sections!(input_dict) would be called here to check that no
    # unexpected sections or top-level options were passed (helps to catch typos in input
    # files). However, it needs to be called after calls to `setup_nonlinear_solve()`
    # because the inputs for nonlinear solvers are only read there, but before electron
    # setup, because `input_dict` needs to be written to the output files, and it cannot
    # be with the `_section_check_store` variable still contained in it (which is used and
    # removed by `check_sections!()`) - it therefore has to be called in the middle of
    # `setup_time_advance!()`.

    if !write_output
        save_inputs_to_txt = false
    else
        save_inputs_to_txt = io_immutable.save_inputs_to_txt
    end

    if global_rank[] == 0 && save_inputs_to_txt
        # Make file to log some information about inputs into.
        io = open_ascii_output_file(joinpath(io_immutable.output_dir, io_immutable.run_name), "input")
    else
        io = devnull
    end
    
    # check input (and initialized coordinate structs) to catch errors/unsupported options
    check_input(io, io_immutable.output_dir, timestepping_section, r, z, vpa, vperp, composition,
                species_immutable, evolve_moments, num_diss_params, save_inputs_to_txt,
                collisions)

    # return immutable structs for z, vpa, species and composition
    all_inputs = (io_immutable, evolve_moments, timestepping_section, z, z_spectral, r,
                  r_spectral, vpa, vpa_spectral, vperp, vperp_spectral, gyrophase,
                  gyrophase_spectral, vz, vz_spectral, vr, vr_spectral, vzeta,
                  vzeta_spectral, composition, species_immutable, collisions, geometry,
                  em_input, num_diss_params, manufactured_solns_input)
    println(io, "\nAll inputs returned from mk_input():")
    println(io, all_inputs)
    close(io)

    return all_inputs
end

"""
check various input options to ensure they are all valid/consistent
"""
function check_input(io, output_dir, timestepping_section, r, z, vpa, vperp, composition, species,
                     evolve_moments, num_diss_params, save_inputs_to_txt, collisions)
    nstep = timestepping_section["nstep"]
    dt = timestepping_section["dt"]
    # copy the input file to the output directory to be saved
    if save_inputs_to_txt && global_rank[] == 0
        cp(joinpath(@__DIR__, "moment_kinetics_input.jl"), joinpath(output_dir, "moment_kinetics_input.jl"), force=true)
    end
    # open ascii file in which informtaion about input choices will be written
    check_input_time_advance(nstep, dt, io)
    check_coordinate_input(r, "r", io)
    check_coordinate_input(z, "z", io)
    check_coordinate_input(vpa, "vpa", io)
    check_coordinate_input(vperp, "vperp", io)
    # if the parallel flow is evolved separately, then the density must also be evolved separately
    if evolve_moments.parallel_flow && !evolve_moments.density
        error("evolve_moments.parallel_flow = true, but evolve_moments.density = false."
              * "this is not a supported option.")
    end
    if collisions.fkpl.nuii > 0.0
    # check that the grids support the collision operator
        print(io, "The self-collision operator is switched on \n nuii = $collisions.fkpl.nuii \n")
        if !(vpa.discretization == "gausslegendre_pseudospectral") || !(vperp.discretization == "gausslegendre_pseudospectral")
            error("ERROR: you are using \n      vpa.discretization='"*vpa.discretization*
              "' \n      vperp.discretization='"*vperp.discretization*"' \n      with the ion self-collision operator \n"*
              "ERROR: you should use \n       vpa.discretization='gausslegendre_pseudospectral' \n       vperp.discretization='gausslegendre_pseudospectral'")
        end
        if (timestepping_section["kinetic_ion_solver"] == implicit_ion_fp_collisions) && (vperp.bc == "zero-impose-regularity")
            errorstring = (
                """ERROR: you are using

                    vperp.bc = "$(vperp.bc)"
                    kinetic_ion_solver="$(implicit_ion_fp_collisions)"

                Only vperp.bc = "none" or vperp.bc = "zero" are currently
                supported for kinetic_ion_solver="$(implicit_ion_fp_collisions)".
                """)
            error(errorstring)
        end
    end
end

"""
"""
function check_input_time_advance(nstep, dt, io)
    println(io,"##### time advance #####")
    println(io)
    println(io,">running for ", nstep, " time steps, with step size ", dt, ".")
end

"""
Check input for a coordinate
"""
function check_coordinate_input(coord, coord_name, io)
    if coord.ngrid * coord.nelement_global == 1
        # Coordinate is not being used for this run
        return
    end

    println(io)
    println(io,"######## $coord_name-grid ########")
    println(io)
    # discretization_option determines discretization in coord
    # supported options are chebyshev_pseudospectral and finite_difference
    if coord.discretization == "chebyshev_pseudospectral"
        print(io,">$coord_name.discretization = 'chebyshev_pseudospectral'.  ")
        println(io,"using a Chebyshev pseudospectral method in $coord_name.")
    elseif coord.discretization == "gausslegendre_pseudospectral"
        print(io,">$coord_name.discretization = 'gausslegendre_pseudospectral'.  ")
        println(io,"using a Gauss-Legendre-Lobatto pseudospectral method in $coord_name.")
    elseif coord.discretization == "fourier_pseudospectral"
        print(io,">$coord_name.discretization = 'fourier_pseudospectral'.  ")
        println(io,"using a Fourier pseudospectral method in $coord_name.")
    elseif coord.discretization == "finite_difference"
        println(io,">$coord_name.discretization = 'finite_difference', ",
            "and $coord_name.finite_difference_option = ", coord.finite_difference_option,
            "  using finite differences on an equally spaced grid in $coord_name.")
        fd_check_option(coord.finite_difference_option, coord.ngrid)
    else
        input_option_error("$coord_name.discretization", coord.discretization)
    end
    # boundary_option determines coord boundary condition
    if coord_name == "r"
        # Radial boundary conditions are not set in the coordinate input
        if coord.discretization == "finite_difference" && coord.n > 1
            error("finite_difference discretization is not supported for the radial "
                  * "dimension because it does not (yet?) support boundary conditions "
                  * "implemented using `r_boundary_section`s.")
        elseif coord.bc ∉ ("default", "periodic")
            error("Radial boundary conditions other than \"periodic\" should be set in "
                  * "[inner_r_bc_*] and [outer_r_bc_*] sections, not in [r], but got "
                  * "bc=$(coord_input_dict["bc"]) in [r] section.")
        end
    elseif coord.bc == "constant"
        println(io,">$coord_name.bc = 'constant'.  enforcing constant incoming BC in $coord_name.")
    elseif coord.bc == "zero"
        println(io,">$coord_name.bc = 'zero'.  enforcing zero incoming BC in $coord_name. Enforcing zero at both boundaries if diffusion operator is present.")
    elseif coord.bc == "zero-impose-regularity"
        println(io,">$coord_name.bc = 'zero'.  enforcing zero incoming BC in $coord_name. Enforcing zero at both boundaries if diffusion operator is present. Enforce dF/dcoord = 0 at origin if coord = vperp.")
    elseif coord.bc == "zero_gradient"
        println(io,">$coord_name.bc = 'zero_gradient'.  enforcing zero gradients at both limits of $coord_name domain.")
    elseif coord.bc == "both_zero"
        println(io,">$coord_name.bc = 'both_zero'.  enforcing zero BC in $coord_name.")
    elseif coord.periodic
        println(io,">$coord_name.bc = 'periodic'.  enforcing periodicity in $coord_name.")
    elseif coord_name == "z" && coord.bc == "wall"
        println(io,">$coord_name.bc = 'wall'.  enforcing wall BC in $coord_name.")
    elseif coord.bc == "none"
        println(io,">$coord_name.bc = 'none'.  not enforcing any BC in $coord_name.")
    else
        input_option_error("$coord_name.bc", coord.bc)
    end
    if coord.name == "vperp"
        println(io,">using ", coord.ngrid, " grid points per $coord_name element on ",
                coord.nelement_global, " elements across the $coord_name domain [",
                0.0, ",", coord.L, "].")

        if coord.bc == "zero-impose-regularity" && coord.n_global > 1 && global_rank[] == 0
            println("WARNING: regularity condition (df/dvperp=0 at vperp=0) being "
                    * "imposed explicitly.")
        end
    else
        println(io,">using ", coord.ngrid, " grid points per $coord_name element on ",
                coord.nelement_global, " elements across the $coord_name domain [",
                -0.5*coord.L, ",", 0.5*coord.L, "].")
    end
end

"""
"""
function check_input_initialization(composition, species, io)
    println(io)
    println(io,"####### initialization #######")
    println(io)
    # xx_initialization_option determines the initial condition for coordinate xx
    # currently supported options are "gaussian" and "monomial"
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    for is ∈ 1:composition.n_species
        if is <= n_ion_species
            print(io,">initial distribution function for ion species ", is)
        elseif is <= n_ion_species + n_neutral_species
            print(io,">initial distribution function for neutral species ", is-n_ion_species)
        else
            print(io,">initial distribution function for the electrons")
        end
        println(io," is of the form f(z,r,vpa,t=0)=Fz(z)*Fr(r)*G(vpa).")
        if species[is].z_IC.initialization_option == "gaussian"
            print(io,">z intialization_option = 'gaussian'.")
            println(io,"  setting Fz(z) = initial_density + exp(-(z/z_width)^2).")
        elseif species[is].z_IC.initialization_option == "monomial"
            print(io,">z_intialization_option = 'monomial'.")
            println(io,"  setting Fz(z) = (z + L_z/2)^", species[is].z_IC.monomial_degree, ".")
        elseif species[is].z_IC.initialization_option == "sinusoid"
            print(io,">z_initialization_option = 'sinusoid'.")
            println(io,"  setting Fz(z) = initial_density + z_amplitude*sinpi(z_wavenumber*z/L_z).")
        elseif species[is].z_IC.initialization_option == "smoothedsquare"
            print(io,">z_initialization_option = 'smoothedsquare'.")
            println(io,"  setting Fz(z) = initial_density + z_amplitude*(cospi(z_wavenumber*z/L_z - sinpi(2*z_wavenumber*z/Lz))).")
        elseif species[is].z_IC.initialization_option == "2D-instability-test"
            print(io,">z_initialization_option = '2D-instability-test'.")
            println(io,"  setting Fz(z) for 2D instability test.")
        elseif species[is].z_IC.initialization_option == "bgk"
            print(io,">z_initialization_option = 'bgk'.")
            println(io,"  setting Fz(z,vpa) = F(vpa^2 + phi), with phi_max = 0.")
        else
            input_option_error("z_initialization_option", species[is].z_IC.initialization_option)
        end
        if species[is].r_IC.initialization_option == "gaussian"
            print(io,">r intialization_option = 'gaussian'.")
            println(io,"  setting Fr(r) = initial_density + exp(-(r/r_width)^2).")
        elseif species[is].r_IC.initialization_option == "monomial"
            print(io,">r_intialization_option = 'monomial'.")
            println(io,"  setting Fr(r) = (r + L_r/2)^", species[is].r_IC.monomial_degree, ".")
        elseif species[is].r_IC.initialization_option == "sinusoid"
            print(io,">r_initialization_option = 'sinusoid'.")
            println(io,"  setting Fr(r) = initial_density + r_amplitude*sinpi(r_wavenumber*r/L_r).")
        elseif species[is].r_IC.initialization_option == "smoothedsquare"
            print(io,">r_initialization_option = 'smoothedsquare'.")
            println(io,"  setting Fr(r) = initial_density + r_amplitude*(cospi(r_wavenumber*r/L_r - sinpi(2*r_wavenumber*r/Lr))).")
        else
            input_option_error("r_initialization_option", species[is].r_IC.initialization_option)
        end
        if species[is].vpa_IC.initialization_option == "gaussian"
            print(io,">vpa_intialization_option = 'gaussian'.")
            println(io,"  setting G(vpa) = Maxwellian_prefactor*exp(-(vpa/vpa_width)^2).")
        elseif species[is].vpa_IC.initialization_option == "monomial"
            print(io,">vpa_intialization_option = 'monomial'.")
            println(io,"  setting G(vpa) = (vpa + L_vpa/2)^", species[is].vpa_IC._monomial_degree, ".")
        elseif species[is].vpa_IC.initialization_option == "sinusoid"
            print(io,">vpa_initialization_option = 'sinusoid'.")
            println(io,"  setting G(vpa) = vpa_amplitude*sinpi(vpa_wavenumber*vpa/L_vpa).")
        elseif species[is].vpa_IC.initialization_option == "bgk"
            print(io,">vpa_initialization_option = 'bgk'.")
            println(io,"  setting F(z,vpa) = F(vpa^2 + phi), with phi_max = 0.")
        elseif species[is].vpa_IC.initialization_option == "vpagaussian"
            print(io,">vpa_initialization_option = 'vpagaussian'.")
            println(io,"  setting G(vpa) = vpa^2*Maxwellian_prefactor*exp(-(vpa/vpa_width)^2).")
        else
            input_option_error("vpa_initialization_option", species[is].vpa_IC.initialization_option)
        end
        println(io)
    end
end

end
