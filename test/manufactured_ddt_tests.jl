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
using Statistics: mean

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

ngrid = 7
const input_sound_wave_periodic = Dict(
    "use_manufactured_solns" => true,
    "n_ion_species" => 1,
    "n_neutral_species" => 1,
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
    "vpa_bc" => "zero",
    "vpa_discretization" => "chebyshev_pseudospectral",
    "vperp_ngrid" => ngrid,
    "vperp_nelement" => 4,
    "vperp_L" => 8.0,
    "vperp_bc" => "zero",
    "vperp_discretization" => "chebyshev_pseudospectral",
    "gyrophase_ngrid" => ngrid,
    "gyrophase_nelement" => 4,
    "vperp_discretization" => "chebyshev_pseudospectral",
    "vz_ngrid" => ngrid,
    "vz_nelement" => 4,
    "vz_L" => 8.0,
    "vz_bc" => "zero",
    "vz_discretization" => "chebyshev_pseudospectral",
    "vr_ngrid" => ngrid,
    "vr_nelement" => 4,
    "vr_L" => 8.0,
    "vr_bc" => "zero",
    "vr_discretization" => "chebyshev_pseudospectral",
    "vzeta_ngrid" => ngrid,
    "vzeta_nelement" => 4,
    "vzeta_L" => 8.0,
    "vzeta_bc" => "zero",
    "vzeta_discretization" => "chebyshev_pseudospectral",
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
    begin_sn_r_z_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        first_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
        output.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = 0.0
    end
    @loop_sn_r_z isn ir iz begin
        first_scratch.density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
        output.density_neutral[iz,ir,isn] = 0.0
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
    dfdt_ion = nothing
    dfdt_neutral = nothing
    if global_rank[] == 0
        dfdt_ion = copy(output.pdf)
        dfdt_neutral = copy(output.pdf_neutral)
    end

    # clean up i/o and communications
    # last 2 elements of mk_state are `io` and `cdf`
    cleanup_moment_kinetics!(mk_state[end-1:end]...)

    return dfdt_ion, dfdt_neutral, r, z, vperp, vpa, vzeta, vr, vz, composition,
           geometry, collisions, modified_advance
end

"""
    remove_boundary_points(f::AbstractArray{mk_float,4})

Remove points which are modified by boundary conditions* and therefore not
expected to match the manufactured rhs.

* Boundary conditions are applied at the end of the euler_time_advance!()
function. In normal operation, these are applied to the distribution function
(and possibly the moments). Here we hack around to get df/dt stored in the
fvec_out argument, so the bcs are applied to our df/dt. Physically this makes
sense for zero-value bcs (as df/dt=0 on the boundary for them), but means that
the calculated df/dt does not match the one evaluated from the manufactured
solution, for which evaluating the rhs doesn't necessarily give zero at the
boundary.
"""
function remove_ion_boundary_points(f::AbstractArray{mk_float,4})
    return f[2:end-1,:,:,:]
end

"""
    runcase(input::Dict, advance::advance_info, returnstuff=false)

Run a simulation with parameters set by `input` using manufactured sources and return
the errors in each variable compared to the manufactured solution.
"""
function runcase(input::Dict, advance::advance_info, returnstuff=false)
    dfdt_ion = nothing
    dfdt_neutral = nothing
    manufactured_inputs = nothing
    quietoutput() do
        dfdt_ion, dfdt_neutral, manufactured_inputs... = evaluate_initial_ddt(input, advance)
    end

    error_2 = nothing
    error_inf = nothing
    rhs_ion_manf = nothing
    rhs_neutral_manf = nothing
    if global_rank[] == 0
        rhs_ion_manf, rhs_neutral_manf = manufactured_rhs_as_array(mk_float(0.0),
                                                                   manufactured_inputs...)

        if input["n_ion_species"] > 0
            # Only one species, so get rid of species index
            dfdt_ion = dfdt_ion[:,:,:,:,1]

            # Get rid of vpa boundary points which are (possibly) overwritten by
            # boundary conditions in the numerically evaluated result.
            # Not necessary for neutrals as there are no v-space boundary conditions for
            # neutrals.
            dfdt_ion = remove_ion_boundary_points(dfdt_ion)
            rhs_ion_manf = remove_ion_boundary_points(rhs_ion_manf)

            error_2_ion = L2_error_norm(dfdt_ion, rhs_ion_manf)
            error_inf_ion = L_infinity_error_norm(dfdt_ion, rhs_ion_manf)
        else
            error_2_ion = mk_float(0.0)
            error_inf_ion = mk_float(0.0)
        end

        if input["n_neutral_species"] > 0
            # Only one species, so get rid of species index
            dfdt_neutral = dfdt_neutral[:,:,:,:,:,1]
            error_2_neutral = L2_error_norm(dfdt_neutral, rhs_neutral_manf)
            error_inf_neutral = L_infinity_error_norm(dfdt_neutral, rhs_neutral_manf)
        else
            error_2_neutral = 0.0
            error_inf_neutral = 0.0
        end

        # Create combined errors
        error_2 = mean([error_2_ion, error_2_neutral])
        error_inf = max(error_inf_ion, error_inf_neutral)

        println("ion error ", error_2_ion, " ", error_inf_ion)
        println("neutral error ", error_2_neutral, " ", error_inf_neutral)
        println("combined error ", error_2, " ", error_inf)
    end

    if returnstuff
        return error_2, error_inf, dfdt_ion, rhs_ion_manf, dfdt_neutral, rhs_neutral_manf
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
function testconvergence(input::Dict, which_term::Symbol; ngrid=nothing, returnstuff=false)
    advance = setup_advance(which_term, input)

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
    lastrhs_ion, lastrhs_manf_ion, lastrhs_neutral, lastrhs_manf_neutral = nothing, nothing, nothing, nothing
    for nelement ∈ nelement_values
        global_rank[] == 0 && println("testing nelement=$nelement")
        case_input = increase_resolution(input, nelement)

        if returnstuff
            error_2, error_inf, lastrhs_ion, lastrhs_manf_ion, lastrhs_neutral,
            lastrhs_manf_neutral = runcase(case_input, advance, returnstuff)
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
        return lastrhs_ion, lastrhs_manf_ion, lastrhs_neutral, lastrhs_manf_neutral
    else
        return nothing
    end
end

function setup_advance(which_term::Symbol, input::Dict)
    advance = advance_info(false, false, false, false, false, false, false, false,
        false, false, false, false, false, zeros(1,1), false)

    has_ions = input["n_ion_species"] > 0
    has_neutrals = input["n_neutral_species"] > 0
    if which_term == :all
        advance.vpa_advection = has_ions
        advance.z_advection = has_ions
        advance.r_advection = has_ions
        advance.neutral_z_advection = has_neutrals
        advance.neutral_r_advection = has_neutrals
        advance.cx_collisions = has_ions && has_neutrals
        #advance.ionization_collisions = true
        advance.source_terms = true
        advance.continuity = true
        advance.force_balance = true
        advance.energy = true
    elseif which_term == :vpa_advection
        !has_ions && error("vpa_advection requires at least 1 ion species")
        advance.vpa_advection = true
    elseif which_term == :z_advection
        !has_ions && error("z_advection requires at least 1 ion species")
        advance.z_advection = true
    elseif which_term == :r_advection
        !has_ions && error("r_advection requires at least 1 ion species")
        advance.r_advection = true
    elseif which_term == :neutral_z_advection
        !has_neutrals && error("neutral_z_advection requires at least 1 neutral species")
        advance.neutral_z_advection = true
    elseif which_term == :neutral_r_advection
        !has_neutrals && error("neutral_r_advection requires at least 1 neutral species")
        advance.neutral_r_advection = true
    elseif which_term == :cx_collisions
        if !(has_ions && has_neutrals)
            error("cx_collisions requires both ion and neutral species")
        end
        advance.cx_collisions = true
    elseif which_term == :ionization_collisions
        if !(has_ions && has_neutrals)
            error("ionization_collisions requires both ion and neutral species")
        end
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
                testconvergence(input_sound_wave_periodic, :vpa_advection, ngrid=ngrid)
            end
            @testset "z_advection" begin
                global_rank[] == 0 && println("\nz_advection")
                testconvergence(input_sound_wave_periodic, :z_advection, ngrid=ngrid)
            end
            @testset "r_advection" begin
                global_rank[] == 0 && println("\nr_advection")
                testconvergence(input_sound_wave_periodic, :r_advection, ngrid=ngrid)
            end
            @testset "neutral_z_advection" begin
                global_rank[] == 0 && println("\nneutral_z_advection")
                testconvergence(input_sound_wave_periodic, :neutral_z_advection,
                                ngrid=ngrid)
            end
            @testset "neutral_r_advection" begin
                global_rank[] == 0 && println("\nneutral_r_advection")
                testconvergence(input_sound_wave_periodic, :neutral_r_advection,
                                ngrid=ngrid)
            end
            @testset "cx_collisions" begin
                global_rank[] == 0 && println("\ncx_collisions")
                testconvergence(input_sound_wave_periodic, :cx_collisions, ngrid=ngrid)
            end
            #@testset "ionization_collisions" begin
            #    global_rank[] == 0 && println("\nionization_collisions")
            #    testconvergence(input_sound_wave_periodic, :ionization_collisions,
            #                    ngrid=ngrid)
            #end
            #@testset "continuity" begin
            #    global_rank[] == 0 && println("\ncontinuity")
            #    testconvergence(input_sound_wave_periodic, :continuity, ngrid=ngrid)
            #end
            #@testset "force_balance" begin
            #    global_rank[] == 0 && println("\nforce_balance")
            #    testconvergence(input_sound_wave_periodic, :force_balance, ngrid=ngrid)
            #end
            #@testset "energy" begin
            #    global_rank[] == 0 && println("\nenergy")
            #    testconvergence(input_sound_wave_periodic, :energy, ngrid=ngrid)
            #end
            @testset "all" begin
                global_rank[] == 0 && println("\nall terms")
                testconvergence(input_sound_wave_periodic, :all, ngrid=ngrid)
            end
        end
    end

    return nothing
end

function runtests_return(which_term::Symbol=:all; ngrid=nothing)
    global_rank[] == 0 && println("MMS tests with return")

    results = testconvergence(input_sound_wave_periodic, which_term; ngrid=ngrid,
                              returnstuff=true)

    return results
end

end # ManufacturedSolutionsTests


using .ManufacturedDDTTests

ManufacturedDDTTests.runtests()
