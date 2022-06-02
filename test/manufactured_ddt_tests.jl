"""
Test cases using the method of manufactured solutions (MMS)
"""
module ManufacturedDDTTests

include("setup.jl")
include("mms_utils.jl")

using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!
using moment_kinetics.looping
using moment_kinetics.manufactured_solns
using moment_kinetics.post_processing: L2_error_norm, L_infinity_error_norm
using moment_kinetics.time_advance: evaluate_ddt!, advance_info
using moment_kinetics.type_definitions

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

ngrid = 7
const input_sound_wave_periodic = Dict(
    "use_manufactured_solns" => true,
    "n_ion_species" => 1,
    "n_neutral_species" => 0,
    "boltzmann_electron_response" => true,
    "run_name" => "MMS-rperiodic",
    "base_directory" => test_output_directory,
    "evolve_moments_density" => false,
    "evolve_moments_parallel_flow" => false,
    "evolve_moments_parallel_pressure" => false,
    "evolve_moments_conservation" => false,
    "T_e" => 1.0,
    "rhostar" => 1.0,
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
    "charge_exchange_frequency" => 0.62831853071,
    "ionization_frequency" => 0.0,
    #"nstep" => 10, #1700,
    #"dt" => 0.002,
    #"nwrite" => 10, #1700,
    "nstep" => 1700, #1700,
    "dt" => 0.0002, #0.002,
    "nwrite" => 1700, #1700,
    "use_semi_lagrange" => false,
    "n_rk_stages" => 1,
    "split_operators" => false,
    "z_ngrid" => ngrid,
    "z_nelement" => 2,
    "z_bc" => "periodic",
    "z_discretization" => "chebyshev_pseudospectral",
    "r_ngrid" => ngrid,
    "r_nelement" => 2,
    "r_bc" => "periodic",
    "r_discretization" => "chebyshev_pseudospectral",
    "vpa_ngrid" => ngrid,
    "vpa_nelement" => 4,
    "vpa_L" => 8.0,
    "vpa_bc" => "periodic",
    "vpa_discretization" => "chebyshev_pseudospectral",
    "vperp_ngrid" => ngrid,
    "vperp_nelement" => 4,
    "vperp_L" => 8.0,
    "vperp_bc" => "periodic",
    "vperp_discretization" => "chebyshev_pseudospectral",
)

"""
    evaluate_initial_ddt(input_dict::Dict, advance_input::advance_info)

Evaluate df/dt for the initial state of f.

Very similar to combination of run_moment_kinetics function and various parts of
time_advance module.
"""
function evaluate_initial_ddt(input_dict::Dict, advance_input::advance_info)
    # set up all the structs, etc. needed for a run
    mk_state = setup_moment_kinetics(input_dict)

    # Split mk_state tuple into separate variables
    pdf, scratch, t, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, moments,
    fields, spectral_objects, advect_objects, composition, collisions, geometry,
    boundary_distributions, advance, scratch_dummy, manufactured_source_list, io,
    cdf = mk_state

    # Put initial state into scratch[1]
    begin_s_r_z_region()
    first_scratch = scratch[1]
    output = scratch[2]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        first_scratch.pdf[ivpa,ivperp,iz,ir,is] = pdf.charged.norm[ivpa,ivperp,iz,ir,is]
        output.pdf[ivpa,ivperp,iz,ir,is] = 0.0
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.charged.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.charged.upar[iz,ir,is]
        first_scratch.ppar[iz,ir,is] = moments.charged.ppar[iz,ir,is]
        output.density[iz,ir,is] = 0.0
        output.upar[iz,ir,is] = 0.0
        output.ppar[iz,ir,is] = 0.0
    end

    # modify advance_info argument so that no manufactured sources are added when
    # evaluating the time advance, as in this test we only want manufactured initial
    # conditions. Also take the terms to evaluate from advance_input so that the calling
    # function can control which term to use.
    modified_advance = advance_info(advance_input.vpa_advection,
        advance_input.z_advection, advance_input.r_advection,
        advance_input.neutral_z_advection, advance_input.neutral_r_advection,
        advance_input.cx_collisions, advance_input.cx_collisions_1V,
        advance_input.ionization_collisions, advance_input.ionization_collisions_1V,
        advance_input.source_terms, advance_input.continuity,
        advance_input.force_balance, advance_input.energy, advance.rk_coefs, false)

    evaluate_ddt!(output, first_scratch, pdf, fields, moments, advect_objects, vz, vr,
        vzeta, vpa, vperp, gyrophase, z, r, t, t_input, spectral_objects, composition,
        collisions, geometry, boundary_distributions, scratch_dummy,
        manufactured_source_list, modified_advance)

    # Make a copy of the array with df/dt so it doesn't get messed up by MPI cleanup in
    # cleanup_moment_kinetics!()
    dfdt = nothing
    if global_rank[] == 0
        dfdt = copy(output.pdf)
    end

    # clean up i/o and communications
    # last 2 elements of mk_state are `io` and `cdf`
    cleanup_moment_kinetics!(mk_state[end-1:end]...)

    return dfdt, r, z, vperp, vpa, composition, geometry, collisions, modified_advance
end

"""
    runcase(input::Dict, advance::advance_info, returnstuff=false)

Run a simulation with parameters set by `input` using manufactured sources and return
the errors in each variable compared to the manufactured solution.
"""
function runcase(input::Dict, advance::advance_info, returnstuff=false)
    dfdt = nothing
    manufactured_inputs = nothing
    quietoutput() do
        dfdt, manufactured_inputs... = evaluate_initial_ddt(input, advance)
    end

    error_2 = nothing
    error_inf = nothing
    rhs_manf = nothing
    if global_rank[] == 0
        rhs_manf = manufactured_rhs_as_array(mk_float(0.0), manufactured_inputs...)

        # Only one species, so get rid of species index
        dfdt = dfdt[:,:,:,:,1]

        error_2 = L2_error_norm(dfdt, rhs_manf)
        error_inf = L_infinity_error_norm(dfdt, rhs_manf)

        println("error ", error_2, " ", error_inf)
    end

    if returnstuff
        return error_2, error_inf, dfdt, rhs_manf
    else
        return error_2, error_inf
    end
end

"""
    calculate_convergence_orders(nelements, errors)

Calculate estimated convergence order between consecutive steps of the nelement
scan.

Don't assume the final error is the 'best' value in case it is affected by
rounding errors, etc.
"""
function calculate_convergence_orders(nelements, errors)
    @assert size(nelements) == size(errors)

    error_ratios = errors[1:end-1] ./ errors[2:end]
    nelements_ratios = nelements[2:end] ./ nelements[1:end-1]
    orders = @. log(error_ratios) / log(nelements_ratios)
    return orders
end

"""
    testconvergence(input::Dict, advance::advance_info; returnstuff::Bool)

Test convergence with spatial resolution

The parameters for the run are given in `input::Dict`. 
`which_term` controls which term to include in the evolution equation, or include all terms.
`returnstuff` can be set to true to return the calculated and manfactured RHS of df/dt=(...).
"""
function testconvergence(input::Dict, advance::advance_info; ngrid=nothing, returnstuff=false)
    errors_2 = Vector{mk_float}(undef, 0)
    errors_inf = Vector{mk_float}(undef, 0)

    if ngrid === nothing
        ngrid = get_and_check_ngrid(input)
    else
        set_ngrid(input, ngrid)
    end
    global_rank[] == 0 && println("ngrid=$ngrid")

    #nelement_values = [2, 4, 6, 8, 10, 12, 14, 16]
    nelement_values = ngrid > 6 ? [2, 4, 6] : [2, 4, 6, 8]
    if returnstuff
        nelement_values = [nelement_values[end]]
    end
    lastrhs, lastrhs_manf = nothing, nothing
    for nelement ∈ nelement_values
        global_rank[] == 0 && println("testing nelement=$nelement")
        case_input = increase_resolution(input, nelement)

        if returnstuff
            error_2, error_inf, lastrhs, lastrhs_manf = runcase(case_input, advance, returnstuff)
        else
            error_2, error_inf = runcase(case_input, advance)
        end

        if global_rank[] == 0
            push!(errors_2, error_2)
            push!(errors_inf, error_inf)
        end
    end

    if global_rank[] == 0
        convergence_2 = errors_2[1:end] ./ errors_2[end]
        convergence_inf = errors_inf[1:end] ./ errors_inf[end]
        expected_convergence = @. (nelement_values[end] / nelement_values[1:end])^(ngrid - 1)
        println("errors")
        println(errors_2)
        println(errors_inf)
        println("convergence")
        println(convergence_2)
        println(convergence_inf)
        println("expected convergence")
        println(expected_convergence)
        println("convergence orders")
        println(calculate_convergence_orders(nelement_values, errors_2))
        println(calculate_convergence_orders(nelement_values, errors_inf))
    end

    if returnstuff
        return lastrhs, lastrhs_manf
    else
        return nothing
    end
end

function setup_advance(which_term)
    advance = advance_info(false, false, false, false, false, false, false, false,
        false, false, false, false, false, zeros(1,1), false)

    if which_term == :all
        advance.vpa_advection = true
        advance.z_advection = true
        advance.r_advection = true
        #advance.neutral_z_advection = true
        #advance.neutral_r_advection = true
        #advance.cx_collisions = true
        #advance.ionization_collisions = true
        #advance.source_terms = true
        #advance.continuity = true
        #advance.force_balance = true
        #advance.energy = true
    elseif which_term == :vpa_advection
        advance.vpa_advection = true
    elseif which_term == :z_advection
        advance.z_advection = true
    elseif which_term == :r_advection
        advance.r_advection = true
    elseif which_term == :neutral_z_advection
        advance.neutral_z_advection = true
    elseif which_term == :neutral_r_advection
        advance.neutral_r_advection = true
    elseif which_term == :cx_collisions
        advance.cx_collisions = true
    elseif which_term == :ionization_collisions
        advance.ionization_collisions = true
    elseif which_term == :source_terms
        advance.source_terms = true
    elseif which_term == :continuity
        advance.continuity = true
    elseif which_term == :force_balance
        advance.force_balance = true
    elseif which_term == :energy
        advance.energy = true
    end

    return advance
end

function runtests(ngrid=nothing)
    @testset "MMS" verbose=use_verbose begin
        global_rank[] == 0 && println("MMS tests")

        @testset "r-periodic, z-periodic" begin
            @testset "vpa_advection" begin
                global_rank[] == 0 && println("\nvpa_advection")
                testconvergence(input_sound_wave_periodic,
                                setup_advance(:vpa_advection), ngrid=ngrid)
            end
            @testset "z_advection" begin
                global_rank[] == 0 && println("\nz_advection")
                testconvergence(input_sound_wave_periodic, setup_advance(:z_advection),
                                ngrid=ngrid)
            end
            @testset "r_advection" begin
                global_rank[] == 0 && println("\nr_advection")
                testconvergence(input_sound_wave_periodic, setup_advance(:r_advection),
                                ngrid=ngrid)
            end
            #@testset "cx_collisions" begin
            #    global_rank[] == 0 && println("\ncx_collisions")
            #    testconvergence(input_sound_wave_periodic,
            #                    setup_advance(:cx_collisions), ngrid=ngrid)
            #end
            #@testset "ionization_collisions" begin
            #    global_rank[] == 0 && println("\nionization_collisions")
            #    testconvergence(input_sound_wave_periodic,
            #                    setup_advance(:ionization_collisions), ngrid=ngrid)
            #end
            #@testset "continuity" begin
            #    global_rank[] == 0 && println("\ncontinuity")
            #    testconvergence(input_sound_wave_periodic, setup_advance(:continuity),
            #                    ngrid=ngrid)
            #end
            #@testset "force_balance" begin
            #    global_rank[] == 0 && println("\nforce_balance")
            #    testconvergence(input_sound_wave_periodic,
            #                    setup_advance(:force_balance), ngrid=ngrid)
            #end
            #@testset "energy" begin
            #    global_rank[] == 0 && println("\nenergy")
            #    testconvergence(input_sound_wave_periodic, setup_advance(:energy),
            #                    ngrid=ngrid)
            #end
            @testset "all" begin
                global_rank[] == 0 && println("\nall terms")
                testconvergence(input_sound_wave_periodic, setup_advance(:all),
                                ngrid=ngrid)
            end
        end
    end

    return nothing
end

function runtests_return(which_term::Symbol=:all; ngrid=nothing)
    global_rank[] == 0 && println("MMS tests with return")

    rhs, rhs_manf = testconvergence(input_sound_wave_periodic,
                                    setup_advance(which_term); ngrid=ngrid,
                                    returnstuff=true)

    return rhs, rhs_manf
end

end # ManufacturedSolutionsTests


using .ManufacturedDDTTests

ManufacturedDDTTests.runtests()
