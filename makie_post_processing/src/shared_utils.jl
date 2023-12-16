module shared_utils

export calculate_and_write_frequencies, construct_global_zr_coords,
       get_geometry_and_composition, read_distributed_zr_data!

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input
using moment_kinetics.type_definitions: mk_float, mk_int

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
function construct_global_zr_coords(r_local, z_local)

    function make_global_input(coord_local)
        return grid_input(coord_local.name, coord_local.ngrid,
            coord_local.nelement_global, coord_local.nelement_global, 1, 0, coord_local.L,
            coord_local.discretization, coord_local.fd_option, coord_local.cheb_option, coord_local.bc,
            coord_local.advection, MPI.COMM_NULL, coord_local.element_spacing_option)
    end

    r_global, r_global_spectral = define_coordinate(make_global_input(r_local))
    z_global, z_global_spectral = define_coordinate(make_global_input(z_local))

    return r_global, r_global_spectral, z_global, z_global_spectral
end

"""
"""
function get_geometry_and_composition(scan_input,n_ion_species,n_neutral_species)
    # set geometry_input
    # MRH need to get this in way that does not duplicate code
    # MRH from moment_kinetics_input.jl
    Bzed = get(scan_input, "Bzed", 1.0)
    Bmag = get(scan_input, "Bmag", 1.0)
    bzed = Bzed/Bmag
    bzeta = sqrt(1.0 - bzed^2.0)
    Bzeta = Bmag*bzeta
    rhostar = get(scan_input, "rhostar", 0.0)
    geometry = geometry_input(Bzed,Bmag,bzed,bzeta,Bzeta,rhostar)

    # set composition input
    # MRH need to get this in way that does not duplicate code
    # MRH from moment_kinetics_input.jl
    electron_physics = get(scan_input, "electron_physics", boltzmann_electron_response)

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
        mn_over_mi, me_over_mi, recycling_fraction, allocate_float(n_species))
    return geometry, composition

end

"""
Read data which is a function of (z,r,t) or (z,r,species,t)

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.
"""
function read_distributed_zr_data!(var::Array{mk_float,N}, var_name::String,
   run_names::Tuple, file_key::String, nblocks::Tuple,
   nz_local::mk_int,nr_local::mk_int,iskip::mk_int) where N
    # dimension of var is [z,r,species,t]

    local_tind_start = 1
    local_tind_end = -1
    global_tind_start = 1
    global_tind_end = -1
    for (run_name, nb) in zip(run_names, nblocks)
        for iblock in 0:nb-1
            fid = open_readonly_output_file(run_name,file_key,iblock=iblock,printout=false)
            group = get_group(fid, "dynamic_data")
            var_local = load_variable(group, var_name)

            ntime_local = size(var_local, N)

            # offset is the amount we have to skip at the beginning of this restart to
            # line up properly with having outputs every iskip since the beginning of the
            # first restart.
            # Note: use rem(x,y,RoundDown) here because this gives a result that's
            # definitely between 0 and y, whereas rem(x,y) or mod(x,y) give negative
            # results for negative x.
            offset = rem(1 - (local_tind_start-1), iskip, RoundDown)
            if offset == 0
                # Actually want offset in the range [1,iskip], so correct if rem()
                # returned 0
                offset = iskip
            end
            if local_tind_start > 1
                # The run being loaded is a restart (as local_tind_start=1 for the first
                # run), so skip the first point, as this is a duplicate of the last point
                # of the previous restart
                offset += 1
            end

            local_tind_end = local_tind_start + ntime_local - 1
            global_tind_end = global_tind_start + length(offset:iskip:ntime_local) - 1

            z_irank, z_nrank, r_irank, r_nrank = load_rank_data(fid)

            # min index set to avoid double assignment of repeated points
            # 1 if irank = 0, 2 otherwise
            imin_r = min(1,r_irank) + 1
            imin_z = min(1,z_irank) + 1
            for ir_local in imin_r:nr_local
                for iz_local in imin_z:nz_local
                    ir_global = iglobal_func(ir_local,r_irank,nr_local)
                    iz_global = iglobal_func(iz_local,z_irank,nz_local)
                    if N == 4
                        var[iz_global,ir_global,:,global_tind_start:global_tind_end] .= var_local[iz_local,ir_local,:,offset:iskip:end]
                    elseif N == 3
                        var[iz_global,ir_global,global_tind_start:global_tind_end] .= var_local[iz_local,ir_local,offset:iskip:end]
                    else
                        error("Unsupported number of dimensions: $N")
                    end
                end
            end
            close(fid)
        end
        local_tind_start = local_tind_end + 1
        global_tind_start = global_tind_end + 1
    end
end

end # shared_utils.jl
