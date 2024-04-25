module shared_utils

export calculate_and_write_frequencies, get_geometry, get_composition

using moment_kinetics.analysis: fit_delta_phi_mode
using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.geo: init_magnetic_geometry
using moment_kinetics.input_structs: boltzmann_electron_response,
                                     boltzmann_electron_response_with_simple_sheath,
                                     grid_input, geometry_input, species_composition
using moment_kinetics.moment_kinetics_input: get_default_rhostar, setup_reference_parameters
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.reference_parameters: setup_reference_parameters
using moment_kinetics.moment_kinetics_input: get_default_rhostar
using moment_kinetics.geo: init_magnetic_geometry
using MPI

"""
"""
function calculate_and_write_frequencies(run_name, ntime, time, z, itime_min, itime_max,
                                         iz0, delta_phi, pp)
    if pp.calculate_frequencies
        println("Calculating the frequency and damping/growth rate...")
        # shifted_time = t - t0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        # assume phi(z0,t) = A*exp(growth_rate*t)*cos(ω*t + φ)
        # and fit phi(z0,t)/phi(z0,t0), which eliminates the constant A pre-factor
        @views phi_fit = fit_delta_phi_mode(shifted_time[itime_min:itime_max], z,
                                            delta_phi[:, itime_min:itime_max])
        frequency = phi_fit.frequency
        growth_rate = phi_fit.growth_rate

        # write info related to fit to file
        io = open_ascii_output_file(run_name, "frequency_fit.txt")
        println(io, "#growth_rate: ", phi_fit.growth_rate,
                "  frequency: ", phi_fit.frequency,
                " fit_errors: ", phi_fit.amplitude_fit_error, " ",
                phi_fit.offset_fit_error, " ", phi_fit.cosine_fit_error)
        println(io)

        # Calculate the fitted phi as a function of time at index iz0
        L = z[end] - z[begin]
        fitted_delta_phi =
            @. (phi_fit.amplitude0 * cos(2.0 * π * (z[iz0] + phi_fit.offset0) / L)
                * exp(phi_fit.growth_rate * shifted_time)
                * cos(phi_fit.frequency * shifted_time + phi_fit.phase))
        for i ∈ 1:ntime
            println(io, "time: ", time[i], "  delta_phi: ", delta_phi[iz0,i],
                    "  fitted_delta_phi: ", fitted_delta_phi[i])
        end
        close(io)
    else
        frequency = 0.0
        growth_rate = 0.0
        phase = 0.0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        fitted_delta_phi = zeros(ntime)

    end
    return frequency, growth_rate, shifted_time, fitted_delta_phi
end

"""
"""
function get_composition(scan_input)
    reference_params = setup_reference_parameters(scan_input)
    # set composition input
    # MRH need to get this in way that does not duplicate code
    # MRH from moment_kinetics_input.jl
    electron_physics = get(scan_input, "electron_physics", boltzmann_electron_response)

    n_ion_species = get(scan_input, "n_ion_species", 1)
    n_neutral_species = get(scan_input, "n_neutral_species", 1)
    if electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
        n_species = n_ion_species + n_neutral_species
    else
        n_species = n_ion_species + n_neutral_species + 1
    end
    T_e = get(scan_input, "T_e", 1.0)
    # set wall temperature T_wall = Tw/Te
    T_wall = get(scan_input, "T_wall", 1.0)
    # set initial neutral temperature Tn/Tₑ = 1
    # set initial nᵢ/Nₑ = 1.0
    # set phi_wall at z = 0
    phi_wall = get(scan_input, "phi_wall", 0.0)
    # if false use true Knudsen cosine for neutral wall bc
    use_test_neutral_wall_pdf = get(scan_input, "use_test_neutral_wall_pdf", false)
    gyrokinetic_ions = get(scan_input, "gyrokinetic_ions", false)
    # constant to be used to test nonzero Er in wall boundary condition
    Er_constant = get(scan_input, "Er_constant", 0.0)
    recycling_fraction = get(scan_input, "recycling_fraction", 1.0)
    # constant to be used to control Ez divergences
    epsilon_offset = get(scan_input, "epsilon_offset", 0.001)
    # bool to control if dfni is a function of vpa or vpabar in MMS test
    use_vpabar_in_mms_dfni = get(scan_input, "use_vpabar_in_mms_dfni", true)
    if use_vpabar_in_mms_dfni
        alpha_switch = 1.0
    else
        alpha_switch = 0.0
    end
    # ratio of the neutral particle mass to the ion particle mass
    mn_over_mi = 1.0
    # ratio of the electron particle mass to the ion particle mass
    me_over_mi = 1.0/1836.0
    composition = species_composition(n_species, n_ion_species, n_neutral_species,
        electron_physics, use_test_neutral_wall_pdf, T_e, T_wall, phi_wall, Er_constant,
        mn_over_mi, me_over_mi, recycling_fraction, gyrokinetic_ions, allocate_float(n_species))
    return composition

end

function get_geometry(scan_input,z,r)
    reference_params = setup_reference_parameters(scan_input)
    # set geometry_input
    # MRH need to get this in way that does not duplicate code
    # MRH from moment_kinetics_input.jl
    option = get(scan_input, "geometry_option", "constant-helical") #"1D-mirror"
    pitch = get(scan_input, "pitch", 1.0)
    rhostar = get(scan_input, "rhostar", get_default_rhostar(reference_params))
    DeltaB = get(scan_input, "DeltaB", 1.0)
    geo_in = geometry_input(rhostar,option,pitch,DeltaB)
    geometry = init_magnetic_geometry(geo_in,z,r)
    
    return geometry

end

end # shared_utils.jl
