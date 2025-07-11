module shared_utils

export calculate_and_write_frequencies, get_geometry

using moment_kinetics.analysis: fit_delta_phi_mode
using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.file_io: open_ascii_output_file
using moment_kinetics.input_structs: boltzmann_electron_response,
                                     boltzmann_electron_response_with_simple_sheath,
                                     geometry_input
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.reference_parameters: setup_reference_parameters
using moment_kinetics.geo: init_magnetic_geometry, setup_geometry_input
using moment_kinetics.species_input: get_species_input
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

function get_geometry(scan_input,z,r)
    geo_in = setup_geometry_input(scan_input, true)
    geometry = init_magnetic_geometry(geo_in,z,r)
    return geometry
end

end # shared_utils.jl
