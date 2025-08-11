"""
"""
module velocity_moments

export integrate_over_positive_vpa, integrate_over_negative_vpa
export integrate_over_positive_vz, integrate_over_negative_vz
export create_moments_ion, create_moments_electron, create_moments_neutral
export update_moments!
export update_density!
export update_upar!
export update_ppar!
export update_pperp!
export update_ion_qpar!
export update_derived_ion_moment_time_derivatives!
export update_vth!
export reset_moments_status!
export update_derived_electron_moment_time_derivatives!
export update_neutral_density!
export update_neutral_uz!
export update_neutral_ur!
export update_neutral_uzeta!
export update_neutral_pz!
export update_neutral_pr!
export update_neutral_pzeta!
export update_neutral_qz!
export update_derived_neutral_moment_time_derivatives!
export update_chodura!

# for testing 
export get_density
export get_upar
export get_ppar
export get_pperp
export get_qpar
export get_rmom

using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float, allocate_bool, allocate_float
using ..calculus: integral
using ..communication
using ..derivatives: derivative_z!, derivative_z_anyzv!, second_derivative_z!
using ..derivatives: derivative_r!, second_derivative_r!
using ..looping
using ..gyroaverages: gyro_operators, gyroaverage_pdf!
using ..collision_frequencies: get_collision_frequency_ii
using ..input_structs
using ..moment_kinetics_structs: moments_ion_substruct, moments_electron_substruct,
                                 moments_neutral_substruct


#global tmpsum1 = 0.0
#global tmpsum2 = 0.0
#global dens_hist = zeros(17,1)
#global n_hist = 0

"""
"""
function create_moments_ion(nz, nr, n_species, evolve_density, evolve_upar,
                            evolve_p, ion_source_settings, num_diss_params)
    # allocate array used for the particle density
    density = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the density is updated for each species
    density_updated = allocate_bool(n_species)
    density_updated .= false
    # allocate array used for the parallel flow
    parallel_flow = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    parallel_flow_updated = allocate_bool(n_species)
    parallel_flow_updated .= false
    # allocate array used for the pressure
    pressure = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the pressure is updated for each species
    pressure_updated = allocate_bool(n_species)
    pressure_updated .= false
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_shared_float(nz, nr, n_species)
    # allocate array used for the perpendicular pressure
    perpendicular_pressure = allocate_shared_float(nz, nr, n_species)
    # allocate array used for the temperature
    temperature = allocate_shared_float(nz, nr, n_species)
    # allocate array used for the parallel flow
    parallel_heat_flux = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    parallel_heat_flux_updated = allocate_bool(n_species)
    parallel_heat_flux_updated .= false
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    # allocate array used for the thermal speed
    thermal_speed = allocate_shared_float(nz, nr, n_species)
    chodura_integral_lower = allocate_shared_float(nr, n_species)
    chodura_integral_upper = allocate_shared_float(nr, n_species)
    if evolve_p
        v_norm_fac = thermal_speed
    else
        v_norm_fac = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            v_norm_fac .= 1.0
        end
    end

    if evolve_density
        ddens_dz_upwind = allocate_shared_float(nz, nr, n_species)
        ddens_dz = allocate_shared_float(nz, nr, n_species)
        ddens_dr_upwind = allocate_shared_float(nz, nr, n_species)
        ddens_dr = allocate_shared_float(nz, nr, n_species)
        ddens_dt = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            ddens_dt .= 0.0
            if nr == 1
                ddens_dr_upwind .= 0.0
                ddens_dr .= 0.0
            end
        end
    else
        ddens_dr_upwind = nothing
        ddens_dr = nothing
        ddens_dz = nothing
        ddens_dz_upwind = nothing
        ddens_dt = nothing
    end
    if evolve_density && num_diss_params.ion.moment_dissipation_coefficient > 0.0

        d2dens_dz2 = allocate_shared_float(nz, nr, n_species)
    else
        d2dens_dz2 = nothing
    end
    if evolve_density || evolve_upar || evolve_p
        dupar_dz = allocate_shared_float(nz, nr, n_species)
    else
        dupar_dz = nothing
    end
    if evolve_upar
        dupar_dr = allocate_shared_float(nz, nr, n_species)
        dupar_dr_upwind = allocate_shared_float(nz, nr, n_species)
        dupar_dz_upwind = allocate_shared_float(nz, nr, n_species)
        dupar_dt = allocate_shared_float(nz, nr, n_species)
        dnupar_dt = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            dupar_dt .= 0.0
            dnupar_dt .= 0.0
            if nr == 1
                dupar_dr .= 0.0
                dupar_dr_upwind .= 0.0
            end
        end
    else
        dupar_dr = nothing
        dupar_dr_upwind = nothing
        dupar_dz_upwind = nothing
        dupar_dt = nothing
        dnupar_dt = nothing
    end
    if evolve_upar && num_diss_params.ion.moment_dissipation_coefficient > 0.0

        d2upar_dz2 = allocate_shared_float(nz, nr, n_species)
    else
        d2upar_dz2 = nothing
    end
    if evolve_upar
        dppar_dz = allocate_shared_float(nz, nr, n_species)
    else
        dppar_dz = nothing
    end
    if evolve_p
        dp_dr_upwind = allocate_shared_float(nz, nr, n_species)
        dp_dz = allocate_shared_float(nz, nr, n_species)
        dp_dz_upwind = allocate_shared_float(nz, nr, n_species)
        d2p_dz2 = allocate_shared_float(nz, nr, n_species)
        dqpar_dz = allocate_shared_float(nz, nr, n_species)
        dvth_dr = allocate_shared_float(nz, nr, n_species)
        dvth_dz = allocate_shared_float(nz, nr, n_species)
        dT_dz = allocate_shared_float(nz, nr, n_species)
        dp_dt = allocate_shared_float(nz, nr, n_species)
        dvth_dt = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            dp_dt .= 0.0
            dvth_dt .= 0.0
            if nr == 1
                dvth_dr .= 0.0
                dp_dr_upwind .= 0.0
            end
        end
    else
        dp_dr_upwind = nothing
        dp_dz = nothing
        dp_dz_upwind = nothing
        d2p_dz2 = nothing
        dqpar_dz = nothing
        dvth_dr = nothing
        dvth_dz = nothing
        dT_dz = nothing
        dp_dt = nothing
        dvth_dt = nothing
    end

    entropy_production = allocate_shared_float(nz, nr, n_species)

    n_sources = length(ion_source_settings)
    if any(x -> x.active, ion_source_settings)
        external_source_amplitude = allocate_shared_float(nz, nr, n_sources)
        external_source_T_array = allocate_shared_float(nz, nr, n_sources)
        if evolve_density
            external_source_density_amplitude = allocate_shared_float(nz, nr, n_sources)
        else
            external_source_density_amplitude = allocate_shared_float(1, 1, n_sources)
        end
        if evolve_upar
            external_source_momentum_amplitude = allocate_shared_float(nz, nr, n_sources)
        else
            external_source_momentum_amplitude = allocate_shared_float(1, 1, n_sources)
        end
        if evolve_p
            external_source_pressure_amplitude = allocate_shared_float(nz, nr, n_sources)
        else
            external_source_pressure_amplitude = allocate_shared_float(1, 1, n_sources)
        end
        if any(x -> x.PI_density_controller_I != 0.0 && x.source_type ∈ 
                    ("density_profile_control", "density_midpoint_control"), ion_source_settings)
            if any(x -> x.source_type == "density_profile_control", ion_source_settings)
                external_source_controller_integral = allocate_shared_float(nz, nr, n_sources)
            else
                external_source_controller_integral = allocate_shared_float(1, 1, n_sources)
            end
        else
            external_source_controller_integral = allocate_shared_float(1, 1, n_sources)
        end
    else
        external_source_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_T_array = allocate_shared_float(1, 1, n_sources)
        external_source_density_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_momentum_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_pressure_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_controller_integral = allocate_shared_float(1, 1, n_sources)
    end
    # If not using PI controllers, the integrals might not otherwise be initialised, so
    # zero-initialise here.
    @begin_serial_region()
    @serial_region begin
        external_source_controller_integral .= 0.0
    end

    if evolve_density || evolve_upar || evolve_p
        constraints_A_coefficient = allocate_shared_float(nz, nr, n_species)
        constraints_B_coefficient = allocate_shared_float(nz, nr, n_species)
        constraints_C_coefficient = allocate_shared_float(nz, nr, n_species)
    else
        constraints_A_coefficient = nothing
        constraints_B_coefficient = nothing
        constraints_C_coefficient = nothing
    end

    # return struct containing arrays needed to update moments
    return moments_ion_substruct(density, density_updated, parallel_flow,
        parallel_flow_updated, pressure, pressure_updated, parallel_pressure,
        perpendicular_pressure, parallel_heat_flux, parallel_heat_flux_updated,
        thermal_speed, temperature, chodura_integral_lower, chodura_integral_upper,
        v_norm_fac, ddens_dr, ddens_dr_upwind, ddens_dz, ddens_dz_upwind, d2dens_dz2,
        dupar_dr, dupar_dr_upwind, dupar_dz, dupar_dz_upwind, d2upar_dz2, dp_dr_upwind,
        dp_dz, dp_dz_upwind, d2p_dz2, dppar_dz, dqpar_dz, dvth_dr, dvth_dz, dT_dz,
        ddens_dt, dupar_dt, dnupar_dt, dp_dt, dvth_dt, entropy_production,
        external_source_amplitude, external_source_T_array,
        external_source_density_amplitude, external_source_momentum_amplitude,
        external_source_pressure_amplitude, external_source_controller_integral,
        constraints_A_coefficient, constraints_B_coefficient, constraints_C_coefficient)
end

"""
create a moment struct containing information about the electron moments
"""
function create_moments_electron(nz, nr, electron_model, num_diss_params, n_sources)
    # allocate array used for the particle density
    density = allocate_shared_float(nz, nr)
    # initialise Bool variable that indicates if the density is updated for each species
    density_updated = Ref(false)
    # allocate array used for the parallel flow
    parallel_flow = allocate_shared_float(nz, nr)
    # allocate Bool variable that indicates if the parallel flow is updated for each species
    parallel_flow_updated = Ref(false)
    # allocate array used for the pressure
    pressure = allocate_shared_float(nz, nr)
    # allocate array of Bools that indicate if the pressure is updated for each species
    pressure_updated = Ref(false)
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_shared_float(nz, nr)
    # allocate array used for the perpendicular pressure
    perpendicular_pressure = allocate_shared_float(nz, nr)
    # allocate array used for the temperature
    temperature = allocate_shared_float(nz, nr)
    # allocate Bool variable that indicates if the temperature is updated for each species
    temperature_updated = Ref(false)
    # allocate array used for the parallel flow
    parallel_heat_flux = allocate_shared_float(nz, nr)
    # allocate Bool variables that indicates if the parallel flow is updated for each species
    parallel_heat_flux_updated = Ref(false)
    # allocate array used for the election-ion parallel friction force
    parallel_friction_force = allocate_shared_float(nz, nr)
    # allocate arrays used for external sources (third index is for the different sources)
    external_source_amplitude = allocate_shared_float(nz, nr, n_sources)
    external_source_T_array = allocate_shared_float(nz, nr, n_sources)
    external_source_density_amplitude = allocate_shared_float(nz, nr, n_sources)
    external_source_momentum_amplitude = allocate_shared_float(nz, nr, n_sources)
    external_source_pressure_amplitude = allocate_shared_float(nz, nr, n_sources)
    # allocate array used for the thermal speed
    thermal_speed = allocate_shared_float(nz, nr)
    # if evolving the electron pdf, it will be a function of the vth-normalised peculiar velocity
    v_norm_fac = thermal_speed
    # dn/dz is needed to obtain dT/dz (appearing in, e.g., Braginskii qpar) from dppar/dz
    ddens_dz = allocate_shared_float(nz, nr)
    # need dupar/dz to obtain, e.g., the updated electron temperature
    dupar_dz = allocate_shared_float(nz, nr)
    dppar_dz = allocate_shared_float(nz, nr)
    dp_dz = allocate_shared_float(nz, nr)
    if electron_model ∈ (braginskii_fluid, kinetic_electrons,
                         kinetic_electrons_with_temperature_equation)
        dT_dz_upwind = allocate_shared_float(nz, nr)
        if electron_model == kinetic_electrons_with_temperature_equation
            dp_dt = nothing
            dT_dt = allocate_shared_float(nz, nr)
            @serial_region begin
                # Initialise time derivatives so that we can use them without errors when
                # initialising advection speeds. Note the initial values of the speeds are
                # never actually used, as they are updated again in the first timestep.
                dT_dt .= 0.0
            end
        else
            dp_dt = allocate_shared_float(nz, nr)
            dT_dt = nothing
            @serial_region begin
                # Initialise time derivatives so that we can use them without errors when
                # initialising advection speeds. Note the initial values of the speeds are
                # never actually used, as they are updated again in the first timestep.
                dp_dt .= 0.0
            end
        end
        dvth_dt = allocate_shared_float(nz, nr)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            dvth_dt .= 0.0
        end
    else
        dT_dz_upwind = nothing
        dp_dt = nothing
        dT_dt = nothing
        dvth_dt = nothing
    end
    if num_diss_params.electron.moment_dissipation_coefficient > 0.0
        d2p_dz2 = allocate_shared_float(nz, nr)
    else
        d2p_dz2 = nothing
    end
    dqpar_dz = allocate_shared_float(nz, nr)
    dT_dz = allocate_shared_float(nz, nr)
    dvth_dz = allocate_shared_float(nz, nr)
    
    constraints_A_coefficient = allocate_shared_float(nz, nr)
    constraints_B_coefficient = allocate_shared_float(nz, nr)
    constraints_C_coefficient = allocate_shared_float(nz, nr)
    @serial_region begin
        constraints_A_coefficient .= 1.0
        constraints_B_coefficient .= 0.0
        constraints_C_coefficient .= 0.0
    end

    # return struct containing arrays needed to update moments
    return moments_electron_substruct(density, density_updated, parallel_flow,
        parallel_flow_updated, pressure, pressure_updated, parallel_pressure,
        perpendicular_pressure, temperature, temperature_updated, parallel_heat_flux,
        parallel_heat_flux_updated, thermal_speed, parallel_friction_force,
        external_source_amplitude, external_source_T_array,
        external_source_density_amplitude, external_source_momentum_amplitude,
        external_source_pressure_amplitude, v_norm_fac, ddens_dz, dupar_dz, dp_dz,
        d2p_dz2, dppar_dz, dqpar_dz, dT_dz, dT_dz_upwind, dvth_dz, dp_dt, dT_dt, dvth_dt,
        constraints_A_coefficient, constraints_B_coefficient, constraints_C_coefficient)
end

# neutral particles have natural mean velocities 
# uz, ur, uzeta =/= upar 
# and similarly for heat fluxes
# therefore separate moments object for neutrals 
    
function create_moments_neutral(nz, nr, n_species, evolve_density, evolve_upar,
                                evolve_p, neutral_source_settings, num_diss_params)
    density = allocate_shared_float(nz, nr, n_species)
    density_updated = allocate_bool(n_species)
    density_updated .= false
    uz = allocate_shared_float(nz, nr, n_species)
    uz_updated = allocate_bool(n_species)
    uz_updated .= false
    ur = allocate_shared_float(nz, nr, n_species)
    ur_updated = allocate_bool(n_species)
    ur_updated .= false
    uzeta = allocate_shared_float(nz, nr, n_species)
    uzeta_updated = allocate_bool(n_species)
    uzeta_updated .= false
    p = allocate_shared_float(nz, nr, n_species)
    p_updated = allocate_bool(n_species)
    p_updated .= false
    pz = allocate_shared_float(nz, nr, n_species)
    pz_updated = allocate_bool(n_species)
    pz_updated .= false
    pr = allocate_shared_float(nz, nr, n_species)
    pr_updated = allocate_bool(n_species)
    pr_updated .= false
    pzeta = allocate_shared_float(nz, nr, n_species)
    pzeta_updated = allocate_bool(n_species)
    pzeta_updated .= false
    vth = allocate_shared_float(nz, nr, n_species)
    if evolve_p
        v_norm_fac = vth
    else
        v_norm_fac = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            v_norm_fac .= 1.0
        end
    end
    qz = allocate_shared_float(nz, nr, n_species)
    qz_updated = allocate_bool(n_species)
    qz_updated .= false

    if evolve_density
        ddens_dz = allocate_shared_float(nz, nr, n_species)
        ddens_dz_upwind = allocate_shared_float(nz, nr, n_species)
        ddens_dt = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            ddens_dt .= 0.0
        end
    else
        ddens_dz = nothing
        ddens_dz_upwind = nothing
        ddens_dt = nothing
    end
    if evolve_density && num_diss_params.neutral.moment_dissipation_coefficient > 0.0

        d2dens_dz2 = allocate_shared_float(nz, nr, n_species)
    else
        d2dens_dz2 = nothing
    end
    if evolve_density || evolve_upar || evolve_p
        duz_dz = allocate_shared_float(nz, nr, n_species)
    else
        duz_dz = nothing
    end
    if evolve_upar
        duz_dz_upwind = allocate_shared_float(nz, nr, n_species)
        duz_dt = allocate_shared_float(nz, nr, n_species)
        dnuz_dt = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            duz_dt .= 0.0
            dnuz_dt .= 0.0
        end
    else
        duz_dz_upwind = nothing
        duz_dt = nothing
        dnuz_dt = nothing
    end
    if evolve_upar && num_diss_params.neutral.moment_dissipation_coefficient > 0.0

        d2uz_dz2 = allocate_shared_float(nz, nr, n_species)
    else
        d2uz_dz2 = nothing
    end
    if evolve_upar
        dpz_dz = allocate_shared_float(nz, nr, n_species)
    else
        dpz_dz = nothing
    end
    if evolve_p
        dp_dz = allocate_shared_float(nz, nr, n_species)
        dp_dz_upwind = allocate_shared_float(nz, nr, n_species)
        d2p_dz2 = allocate_shared_float(nz, nr, n_species)
        dqz_dz = allocate_shared_float(nz, nr, n_species)
        dvth_dz = allocate_shared_float(nz, nr, n_species)
        dp_dt = allocate_shared_float(nz, nr, n_species)
        dvth_dt = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            # Initialise time derivatives so that we can use them without errors when
            # initialising advection speeds. Note the initial values of the speeds are
            # never actually used, as they are updated again in the first timestep.
            dp_dt .= 0.0
            dvth_dt .= 0.0
        end
    else
        dp_dz = nothing
        dp_dz_upwind = nothing
        d2p_dz2 = nothing
        dqz_dz = nothing
        dvth_dz = nothing
        dp_dt = nothing
        dvth_dt = nothing
    end

    n_sources = length(neutral_source_settings)
    if any(x -> x.active, neutral_source_settings)
        external_source_amplitude = allocate_shared_float(nz, nr, n_sources)
        external_source_T_array = allocate_shared_float(nz, nr, n_sources)
        if evolve_density
            external_source_density_amplitude = allocate_shared_float(nz, nr, n_sources)
        else
            external_source_density_amplitude = allocate_shared_float(1, 1, n_sources)
        end
        if evolve_upar
            external_source_momentum_amplitude = allocate_shared_float(nz, nr, n_sources)
        else
            external_source_momentum_amplitude = allocate_shared_float(1, 1, n_sources)
        end
        if evolve_p
            external_source_pressure_amplitude = allocate_shared_float(nz, nr, n_sources)
        else
            external_source_pressure_amplitude = allocate_shared_float(1, 1, n_sources)
        end
        if any(x -> x.PI_density_controller_I != 0.0 && x.source_type ∈ 
                    ("density_profile_control", "density_midpoint_control"), neutral_source_settings)
            if any(x -> x.source_type == "density_profile_control", neutral_source_settings)
                external_source_controller_integral = allocate_shared_float(nz, nr, n_sources)
            else
                external_source_controller_integral = allocate_shared_float(1, 1, n_sources)
            end
        else
            external_source_controller_integral = allocate_shared_float(1, 1, n_sources)
        end
    else
        external_source_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_T_array = allocate_shared_float(1, 1, n_sources)
        external_source_density_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_momentum_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_pressure_amplitude = allocate_shared_float(1, 1, n_sources)
        external_source_controller_integral = allocate_shared_float(1, 1, n_sources)
    end
    # If not using PI controllers, the integrals might not otherwise be initialised, so
    # zero-initialise here.
    @begin_serial_region()
    @serial_region begin
        external_source_controller_integral .= 0.0
    end

    if evolve_density || evolve_upar || evolve_p
        constraints_A_coefficient = allocate_shared_float(nz, nr, n_species)
        constraints_B_coefficient = allocate_shared_float(nz, nr, n_species)
        constraints_C_coefficient = allocate_shared_float(nz, nr, n_species)
    else
        constraints_A_coefficient = nothing
        constraints_B_coefficient = nothing
        constraints_C_coefficient = nothing
    end

    # return struct containing arrays needed to update moments
    return moments_neutral_substruct(density, density_updated, uz, uz_updated, ur,
        ur_updated, uzeta, uzeta_updated, p, p_updated, pz, pz_updated, pr, pr_updated,
        pzeta, pzeta_updated, qz, qz_updated, vth, v_norm_fac, ddens_dz, ddens_dz_upwind,
        d2dens_dz2, duz_dz, duz_dz_upwind, d2uz_dz2, dp_dz, dp_dz_upwind, d2p_dz2, dpz_dz,
        dqz_dz, dvth_dz, ddens_dt, duz_dt, dnuz_dt, dp_dt, dvth_dt,
        external_source_amplitude, external_source_T_array,
        external_source_density_amplitude, external_source_momentum_amplitude,
        external_source_pressure_amplitude, external_source_controller_integral,
        constraints_A_coefficient, constraints_B_coefficient, constraints_C_coefficient)
end

"""
calculate the updated density (dens) and parallel pressure (ppar) for all species
this function is only used once after initialisation
the function used to update moments at run time is update_derived_moments! in time_advance.jl
"""
function update_moments!(moments, ff_in, gyroavs::gyro_operators, vpa, vperp, z, r, composition,
        r_spectral, geometry, scratch_dummy, z_advect, collisions)
    if composition.ion_physics == gyrokinetic_ions
        ff = scratch_dummy.buffer_vpavperpzrs_1 # the buffer array for the ion pdf -> make sure not to reuse this array below
        # fill buffer with ring-averaged F (gyroaverage at fixed position)
        gyroaverage_pdf!(ff,ff_in,gyroavs,vpa,vperp,z,r,composition)
    else
        ff = ff_in
    end
    @begin_r_z_region()
    n_species = size(ff,5)
    @boundscheck n_species == size(moments.ion.dens,3) || throw(BoundsError(moments))
    @loop_s is begin
        if moments.ion.dens_updated[is] == false
            @views update_density_species!(moments.ion.dens[:,:,is], ff[:,:,:,:,is],
                                           vpa, vperp, z, r)
            moments.ion.dens_updated[is] = true
        end
        if moments.ion.upar_updated[is] == false
            # Can pass moments.ppar here even though it has not been updated yet,
            # because moments.ppar is only needed if evolve_p=true, in which case it
            # will not be updated because it is not calculated from the distribution
            # function
            @views update_upar_species!(moments.ion.upar[:,:,is],
                                        moments.ion.dens[:,:,is],
                                        moments.ion.ppar[:,:,is], ff[:,:,:,:,is], vpa,
                                        vperp, z, r, moments.evolve_density,
                                        moments.evolve_p)
            moments.ion.upar_updated[is] = true
        end
        if moments.ion.p_updated[is] == false
            @views update_p_species!(moments.ion.p[:,:,is], moments.ion.dens[:,:,is],
                                     moments.ion.upar[:,:,is], ff[:,:,:,:,is], vpa, vperp,
                                     z, r, moments.evolve_density, moments.evolve_upar)
            moments.ion.p_updated[is] = true
        end
        @views update_ppar_species!(moments.ion.ppar[:,:,is], moments.ion.dens[:,:,is],
                                    moments.ion.upar[:,:,is], moments.ion.vth[:,:,is],
                                    moments.ion.p[:,:,is], ff[:,:,:,:,is], vpa, vperp, z, r,
                                    moments.evolve_density, moments.evolve_upar,
                                    moments.evolve_p)
        @views update_pperp_species!(moments.ion.pperp[:,:,is], moments.ion.p[:,:,is],
                                     moments.ion.ppar[:,:,is], z, r)
        if moments.ion.qpar_updated[is] == false
            @views update_ion_qpar_species!(moments.ion.qpar[:,:,is],
                                            moments.ion.dens[:,:,is],
                                            moments.ion.upar[:,:,is],
                                            moments.ion.vth[:,:,is],  moments.ion.dT_dz,
                                            ff[:,:,:,:,is], vpa, vperp, z, r,
                                            moments.evolve_density, moments.evolve_upar,
                                            moments.evolve_p, composition.ion_physics,
                                            collisions, composition.T_e)
            moments.ion.qpar_updated[is] = true
        end
    end

    update_vth!(moments.ion.vth, moments.ion.p, moments.ion.dens, z, r, composition)
    # update the Chodura diagnostic -- note that the pdf should be the unnormalised one
    # so this will break for the split moments cases
    if composition.ion_physics != coll_krook_ions
        update_chodura!(moments,ff,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
    end
    return nothing
end

"""
NB: if this function is called and if dens_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_density!(dens, dens_updated, pdf, vpa, vperp, z, r, composition)

    @begin_r_z_region()

    n_species = size(pdf,5)
    @boundscheck n_species == size(dens,3) || throw(BoundsError(dens))
    @loop_s is begin
        if dens_updated[is] == false
            @views update_density_species!(dens[:,:,is], pdf[:,:,:,:,is], vpa, vperp, z, r)
            dens_updated[is] = true
        end
    end
end

"""
calculate the updated density (dens) for a given species;
should only be called when evolve_density = false,
in which case the vpa coordinate is vpa/c_s
"""
function update_density_species!(dens, ff, vpa, vperp, z, r)
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck z.n == size(dens, 1) || throw(BoundsError(dens))
    @boundscheck r.n == size(dens, 2) || throw(BoundsError(dens))
    @loop_r_z ir iz begin
        # When evolve_density = false, the evolved pdf is the 'true' pdf, and the vpa
        # coordinate is (dz/dt) / c_s.
        dens[iz,ir] = get_density(@view(ff[:,:,iz,ir]), vpa, vperp)
    end
    return nothing
end

function get_density(ff, vpa, vperp)
    # Integrating calculates n_s / nref = ∫d(vpa/cref) (f_s c_ref / N_e) in 1V
    # or n_s / nref = ∫d^3(v/cref) (f_s c_ref^3 / N_e) in 2V
    return integral(ff, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
end

"""
NB: if this function is called and if upar_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_upar!(upar, upar_updated, density, vth, pdf, vpa, vperp, z, r,
                      composition, evolve_density, evolve_p)
    @begin_r_z_region()

    n_species = size(pdf,5)
    @boundscheck n_species == size(upar,3) || throw(BoundsError(upar))
    @loop_s is begin
        if upar_updated[is] == false
            @views update_upar_species!(upar[:,:,is], density[:,:,is], vth[:,:,is],
                                        pdf[:,:,:,:,is], vpa, vperp, z, r, evolve_density,
                                        evolve_p)
            upar_updated[is] = true
        end
    end
end

"""
calculate the updated parallel flow (upar) for a given species
"""
function update_upar_species!(upar, density, vth, ff, vpa, vperp, z, r, evolve_density,
                              evolve_p)
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck z.n == size(upar, 1) || throw(BoundsError(upar))
    @boundscheck r.n == size(upar, 2) || throw(BoundsError(upar))
    if evolve_density && evolve_p
        # this is the case where the density and parallel pressure are evolved
        # separately from the normalized pdf, g_s = (√π f_s vth_s / n_s); the vpa
        # coordinate is (dz/dt) / vth_s.
        # Integrating calculates
        # (upar_s / vth_s) = (1/√π)∫d(vpa/vth_s) * (vpa/vth_s) * (√π f_s vth_s / n_s)
        # so convert from upar_s / vth_s to upar_s / c_s
        # we set the input density to get_upar = 1.0 as the normalised distribution has density of 1.0
        @loop_r_z ir iz begin
            upar[iz,ir] = vth[iz,ir]*get_upar(@view(ff[:,:,iz,ir]), 1.0, vpa, vperp,
                                              evolve_density)
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf, given by g_s = (√π f_s c_s / n_s); the vpa coordinate is
        # (dz/dt) / c_s.
        # Integrating calculates
        # (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / n_s)
        # we set the input density to get_upar = 1.0 as the normalised distribution has density of 1.0
        @loop_r_z ir iz begin
            upar[iz,ir] = get_upar(@view(ff[:,:,iz,ir]), 1.0, vpa, vperp, evolve_density)
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vpa coordinate is (dz/dt) / c_s.
        # Integrating calculates
        # (n_s / N_e) * (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / N_e)
        @loop_r_z ir iz begin
            upar[iz,ir] = get_upar(@view(ff[:,:,iz,ir]), density[iz,ir], vpa, vperp,
                                   evolve_density)
        end
    end
    return nothing
end

function get_upar(ff, density, vpa, vperp, evolve_density)
    # Integrating calculates
    # (n_s / N_e) * (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / N_e)
    # so we divide by the density of f_s
    upar = integral(ff, vpa.grid, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)
    if !evolve_density
        upar /= density
    end
    return upar
end

"""
NB: if this function is called and if p_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_p!(p, p_updated, density, upar, pdf, vpa, vperp, z, r, composition,
                   evolve_density, evolve_upar)
    @boundscheck composition.n_ion_species == size(p,3) || throw(BoundsError(p))
    @boundscheck r.n == size(p,2) || throw(BoundsError(p))
    @boundscheck z.n == size(p,1) || throw(BoundsError(p))

    @begin_r_z_region()

    @loop_s is begin
        if p_updated[is] == false
            @views update_p_species!(p[:,:,is], density[:,:,is], upar[:,:,is],
                                     pdf[:,:,:,:,is], vpa, vperp, z, r, evolve_density,
                                     evolve_upar)
            p_updated[is] = true
        end
    end
end

"""
calculate the updated energy density (or pressure, p) for a given species;
which of these is calculated depends on the definition of the vpa coordinate
"""
function update_p_species!(p, density, upar, ff, vpa, vperp, z, r, evolve_density,
                           evolve_upar)
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck z.n == size(p, 1) || throw(BoundsError(p))
    @boundscheck r.n == size(p, 2) || throw(BoundsError(p))
    if evolve_upar
        # this is the case where the parallel flow and density are evolved separately
        # from the normalised distribution function; the vpa coordinate is
        # <(v_∥ - upar_s) / cref> and so we set upar = 0 in the call to get_p
        # because the mean flow of the shape function is zero
        @loop_r_z ir iz begin
            p[iz,ir] = 1.0 / 3.0 * density[iz,ir] * get_moment_for_pressure(@view(ff[:,:,iz,ir]), vpa, vperp, 0.0)
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised distribution function; the vpa coordinate is v_∥ / cref.
        @loop_r_z ir iz begin
            p[iz,ir] = 1.0 / 3.0 * density[iz,ir] * get_moment_for_pressure(@view(ff[:,:,iz,ir]), vpa, vperp, upar[iz,ir])
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vpa coordinate is v_∥ / cref.
        @loop_r_z ir iz begin
            p[iz,ir] = 1.0 / 3.0 * get_moment_for_pressure(@view(ff[:,:,iz,ir]), vpa, vperp, upar[iz,ir])
        end
    end
    return nothing
end

function get_moment_for_pressure(ff, vpa, vperp, upar)
    # Integrating calculates
    # ∫d^3v (((vpa-upar))^2 + vperp^2) * ff
    return integral((vperp,vpa)->((vpa - upar)^2 + vperp^2), ff, vperp, vpa)
end

function get_p(ff, density, upar, vpa, vperp, evolve_density, evolve_upar)
    # Integrating calculates
    # (p/mref nref Tref) = ∫d^3(v/cref) (1/3 * (vpa-upar)/cref)^2 * (f_s cref^3 / nref)
    # the internal energy density (aka pressure of f_s)

    if evolve_upar
        return 1.0 / 3.0 * density * get_moment_for_pressure(ff, vpa, vperp, 0.0)
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised distribution function; the vpa coordinate is v_∥ / cref.
        return 1.0 / 3.0 * density * get_moment_for_pressure(ff, vpa, vperp, upar)
    else
        return 1.0 / 3.0 * get_moment_for_pressure(ff, vpa, vperp, upar)
    end
end

"""
Calculate the parallel pressure p_∥=∫d^3v (v_∥ - u_∥)^2 f
"""
function update_ppar!(ppar, density, upar, vth, p, pdf, vpa, vperp, z, r, composition,
                      evolve_density, evolve_upar, evolve_p)
    @boundscheck composition.n_ion_species == size(ppar,3) || throw(BoundsError(ppar))
    @boundscheck r.n == size(ppar,2) || throw(BoundsError(ppar))
    @boundscheck z.n == size(ppar,1) || throw(BoundsError(ppar))

    @begin_s_r_z_region()

    @loop_s is begin
        @views update_ppar_species!(ppar[:,:,is], density[:,:,is], upar[:,:,is],
                                    vth[:,:,is], p[:,:,is], pdf[:,:,:,:,is], vpa, vperp, 
                                    z, r, evolve_density, evolve_upar, evolve_p)
    end
end

"""
calculate the updated energy density (or parallel pressure, ppar) for a given species;
which of these is calculated depends on the definition of the vpa coordinate
"""
function update_ppar_species!(ppar, density, upar, vth, p, ff, vpa, vperp, z, r,
                              evolve_density, evolve_upar, evolve_p)
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck z.n == size(ppar, 1) || throw(BoundsError(ppar))
    @boundscheck r.n == size(ppar, 2) || throw(BoundsError(ppar))
    if evolve_p
        # this is the case where the pressure, parallel flow and density are evolved
        # separately from the shape function; the vpa coordinate
        # is <(v_∥ - upar_s) / vth_s> and so we set upar = 0 in the call to
        # get_moment_for_ppar because the mean flow of the shape function is zero
        if vperp.n == 1
            @loop_r_z ir iz begin
                ppar[iz,ir] = 3.0 * p[iz,ir]
            end
        else
            @loop_r_z ir iz begin
                ppar[iz,ir] = density[iz,ir] * vth[iz,ir]^2 *
                              get_moment_for_ppar(@view(ff[:,:,iz,ir]), vpa, vperp, 0.0)
            end
        end
    elseif evolve_upar
        # this is the case where the parallel flow and density are evolved separately
        # from the normalized pdf; the vpa coordinate is <v_∥ - upar_s) / c_ref> and so we
        # set upar = 0 in the call to get_moment_for_ppar because the mean flow of the
        # normalised ff is zero
        @loop_r_z ir iz begin
            ppar[iz,ir] = density[iz,ir]*get_moment_for_ppar(@view(ff[:,:,iz,ir]), vpa, vperp, 0.0)
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf; the vpa coordinate is v_\parallel / cref.
        @loop_r_z ir iz begin
            ppar[iz,ir] = density[iz,ir]*get_moment_for_ppar(@view(ff[:,:,iz,ir]), vpa, vperp, upar[iz,ir])
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vpa coordinate is v_∥ / cref.
        @loop_r_z ir iz begin
            ppar[iz,ir] = get_moment_for_ppar(@view(ff[:,:,iz,ir]), vpa, vperp, upar[iz,ir])
        end
    end
    return nothing
end

function get_moment_for_ppar(ff, vpa, vperp, upar)
    # Calculate ∫d^3v (vpa-upar)^2 ff

    # modify input vpa.grid to account for the mean flow
    @. vpa.scratch = (vpa.grid - upar)^2

    return integral(ff, vpa.scratch, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)
end

function get_ppar(density, upar, p, vth, ff, vpa, vperp, evolve_density, evolve_upar,
                  evolve_p)
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    if evolve_p
        if vperp.n == 1
            return 3.0 * p
        else
            return density * vth^2 * get_moment_for_ppar(ff, vpa, vperp, 0.0)
        end
    elseif evolve_upar
        return density * get_moment_for_ppar(ff, vpa, vperp, 0.0)
    elseif evolve_density
        return density * get_moment_for_ppar(ff, vpa, vperp, upar)
    else
        return get_moment_for_ppar(ff, vpa, vperp, upar)
    end
    return nothing
end

function update_pperp!(pperp, p, ppar, vperp, z, r, composition)
    @boundscheck composition.n_ion_species == size(pperp,3) || throw(BoundsError(pperp))
    @boundscheck r.n == size(pperp,2) || throw(BoundsError(pperp))
    @boundscheck z.n == size(pperp,1) || throw(BoundsError(pperp))
    @boundscheck r.n == size(p,2) || throw(BoundsError(p))
    @boundscheck z.n == size(p,1) || throw(BoundsError(p))
    @boundscheck r.n == size(ppar,2) || throw(BoundsError(ppar))
    @boundscheck z.n == size(ppar,1) || throw(BoundsError(ppar))
    
    @begin_s_r_z_region()

    if vperp.n == 1
        @loop_s_r_z is ir iz begin
            pperp[iz,ir,is] = 0.0
        end
    else
        @loop_s is begin
            @views update_pperp_species!(pperp[:,:,is], p[:,:,is], ppar[:,:,is], z, r)
        end
    end
end

"""
calculate the updated perpendicular pressure (pperp) for a given species
"""
function update_pperp_species!(pperp, p, ppar, z, r)
    @boundscheck z.n == size(pperp, 1) || throw(BoundsError(pperp))
    @boundscheck r.n == size(pperp, 2) || throw(BoundsError(pperp))
    @boundscheck r.n == size(p,2) || throw(BoundsError(p))
    @boundscheck z.n == size(p,1) || throw(BoundsError(p))
    @boundscheck r.n == size(ppar,2) || throw(BoundsError(ppar))
    @boundscheck z.n == size(ppar,1) || throw(BoundsError(ppar))
    @loop_r_z ir iz begin
        pperp[iz,ir] = get_pperp(p[iz,ir], ppar[iz,ir])
    end
    return nothing
end

function get_pperp(p, ppar)
    return 1.5 * p - 0.5 * ppar
end

function update_vth!(vth, p, dens, z, r, composition)
    @boundscheck composition.n_ion_species == size(vth,3) || throw(BoundsError(vth))
    @boundscheck r.n == size(vth,2) || throw(BoundsError(vth))
    @boundscheck z.n == size(vth,1) || throw(BoundsError(vth))

    @begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        vth[iz,ir,is] = sqrt(2.0 * p[iz,ir,is] / dens[iz,ir,is])
    end
end

"""
NB: the incoming pdf is the normalized pdf
"""
function update_ion_qpar!(qpar, qpar_updated, density, upar, vth, dT_dz, pdf, vpa, vperp, z, r,
                          composition, ion_physics, collisions, evolve_density, evolve_upar, evolve_p)
    @boundscheck composition.n_ion_species == size(qpar,3) || throw(BoundsError(qpar))

    @begin_r_z_region()

    @loop_s is begin
        if qpar_updated[is] == false
            @views update_ion_qpar_species!(qpar[:,:,is], density[:,:,is], upar[:,:,is],
                                        vth[:,:,is], dT_dz, pdf[:,:,:,:,is], vpa, vperp, z, r,
                                        evolve_density, evolve_upar, evolve_p,
                                        ion_physics, collisions, composition.T_e)
            qpar_updated[is] = true
        end
    end
end

"""
calculate the updated parallel heat flux (qpar) for a given species
"""
function update_ion_qpar_species!(qpar, density, upar, vth, dT_dz, ff, vpa, vperp, z, r, evolve_density,
                                  evolve_upar, evolve_p, ion_physics, collisions, T_e)
    if ion_physics ∈ (drift_kinetic_ions, gyrokinetic_ions)
        calculate_ion_qpar_from_pdf!(qpar, density, upar, vth, ff, vpa, vperp, z, r, evolve_density,
                                     evolve_upar, evolve_p)
    elseif ion_physics == coll_krook_ions
        calculate_ion_qpar_from_coll_krook!(qpar, density, upar, vth, dT_dz, z, r, vperp, collisions, evolve_density, 
                                            evolve_upar, evolve_p, T_e)
    else
        throw(ArgumentError("ion model $ion_physics not implemented for qpar calculation"))
    end
    return nothing
end

"""
calculate parallel heat flux if ion composition flag is kinetic ions
"""
function calculate_ion_qpar_from_pdf!(qpar, density, upar, vth, ff, vpa, vperp, z, r, evolve_density, 
                                      evolve_upar, evolve_p)
    @boundscheck r.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck r.n == size(qpar, 2) || throw(BoundsError(qpar))
    @boundscheck z.n == size(qpar, 1) || throw(BoundsError(qpar))
    if evolve_upar && evolve_p
        @loop_r_z ir iz begin
            qpar[iz,ir] = 0.5 * density[iz,ir] * vth[iz,ir]^3 *
                          integral((vperp,vpa)->(vpa*(vpa^2+vperp^2)), @view(ff[:,:,iz,ir]), vperp, vpa)
        end
    elseif evolve_upar
        @loop_r_z ir iz begin
            qpar[iz,ir] = 0.5 * density[iz,ir] *
                          integral((vperp,vpa)->(vpa*(vpa^2+vperp^2)), @view(ff[:,:,iz,ir]), vperp, vpa)
        end
    elseif evolve_p
        @loop_r_z ir iz begin
            qpar[iz,ir] = 0.5 * density[iz,ir] * vth[iz,ir]^3 *
                          integral((vperp,vpa)->((vpa-upar[iz,ir]/vth[iz,ir])*((vpa-upar[iz,ir]/vth[iz,ir])^2+vperp^2)),
                                   @view(ff[:,:,iz,ir]), vperp, vpa)
        end
    elseif evolve_density
        @loop_r_z ir iz begin
            qpar[iz,ir] = 0.5 * density[iz,ir] *
                          integral((vperp,vpa)->((vpa-upar[iz,ir])*((vpa-upar[iz,ir])^2+vperp^2)),
                                   @view(ff[:,:,iz,ir]), vperp, vpa)
        end
    else
        @loop_r_z ir iz begin
            qpar[iz,ir] = 0.5 *
                          integral((vperp,vpa)->((vpa-upar[iz,ir])*((vpa-upar[iz,ir])^2+vperp^2)),
                                   @view(ff[:,:,iz,ir]), vperp, vpa)
        end
    end
    return nothing
end
"""
calculate parallel heat flux if ion composition flag is coll_krook fluid ions
"""
function calculate_ion_qpar_from_coll_krook!(qpar, density, upar, vth, dT_dz, z, r, vperp, collisions, evolve_density, evolve_upar, evolve_p, T_e)
    # Note that this is a braginskii heat flux for ions using the krook operator. The full Fokker-Planck operator
    # Braginskii heat flux is different! This also assumes one ion species, and so no friction between ions.
    @boundscheck r.n == size(qpar, 2) || throw(BoundsError(qpar))
    @boundscheck z.n == size(qpar, 1) || throw(BoundsError(qpar))

    if vperp.n == 1
        # For 1V need to use parallel temperature for Maxwellian in Krook
        # operator, and for consistency with old 1D1V results also calculate
        # collision frequency using parallel temperature.
        Krook_vth = sqrt(3.0) * vth
        adjust_1V = 1.0 / sqrt(3.0)
    else
        Krook_vth = vth
        adjust_1V = 1.0
    end
    
    # calculate coll_krook heat flux. Currently only works for one ion species! (hence the 1 in dT_dz[iz,ir,1])
    if evolve_density && evolve_upar && evolve_p
        @begin_r_z_region()
        @loop_r_z ir iz begin
            nu_ii = get_collision_frequency_ii(collisions, density[iz,ir], Krook_vth[iz,ir])
            qpar[iz,ir] = -(3/2) * 1/2 * density[iz,ir] * Krook_vth[iz,ir]^2 /nu_ii * 3 * dT_dz[iz,ir,1]
        end
    else
        throw(ArgumentError("coll_krook heat flux simulation requires evolve_density, 
              evolve_upar and evolve_p to be true, since it is a purely fluid simulation"))
    end

    # add boundary condition to the heat flux, since now there is no distribution function 
    # (in this case shape function) whose cutoff boundary condition can hold the parallel heat
    # flux in check. See Stangeby textbook, equations (2.92) and (2.93), and the paragraph between.

    if z.bc != "wall"
        # There's no wall boundary condition here, do nothing (qpar can be what it wants)
        return nothing
    end

    @begin_r_region()

    if z.irank == 0 && (z.irank == z.nrank - 1)
        z_indices = (1, z.n)
    elseif z.irank == 0
        z_indices = (1,)
    elseif z.irank == z.nrank - 1
        z_indices = (z.n,)
    else
        return nothing
    end
    # Stangeby (25.2) suggests that, when including kinetic effects, a value
    # for gamma_i of around 2.5 is sensible.
    # However, maybe for the purposes of this coll_krook scan, at very high
    # collisionality we expect the distribution function of the ions at the
    # sheath entrance to be close to a drifting maxwellian, in which case
    # the original Stangeby (2.92) would be more appropriate. However, this
    # also depends on whether we're 1V or 2V - as in 1V gamma_i = 2.5, 
    # in 2V gamma_i = 3.5.

    @loop_r ir begin
        for iz ∈ z_indices
            this_p = Krook_vth[iz,ir]^2 * density[iz,ir] * 1/2
            this_upar = upar[iz,ir]
            this_dens = density[iz,ir]
            particle_flux = this_dens * this_upar
            T_i = 1/2 * Krook_vth[iz,ir]^2

            if vperp.n == 1
                gamma_i = 2 + T_e/(2 * T_i)
                convective_coefficient = 1.5
            else
                gamma_i = 3.5
                convective_coefficient = 2.5
            end

            # Stangeby (2.92)
            total_heat_flux = gamma_i * T_i * particle_flux

            # E.g. Helander&Sigmar (2.14), but in 1V we have no viscosity and only 3/2
            # rather than 5/2.
            conductive_heat_flux = total_heat_flux -
                                   convective_coefficient * this_p * this_upar -
                                   0.5 * this_dens * this_upar^3

            qpar[iz,ir] = conductive_heat_flux
        end
    end
    return nothing
end

function get_qpar(ff, density, upar, p, vth, vpa, vperp, evolve_density, evolve_upar,
                  evolve_p)
    if evolve_p
        return 0.5 * density * vth^3
               integral((vperp,vpa) -> vpa*(vpa^2 + vperp^2), ff, vperp, vpa)
    elseif evolve_upar
        return 0.5 * density *
               integral((vperp,vpa) -> vpa*(vpa^2 + vperp^2), ff, vperp, vpa)
    elseif evolve_density
        return 0.5 * density *
               integral((vperp,vpa) -> (vpa-upar)*((vpa-upar)^2 + vperp^2), ff, vperp, vpa)
    else
        return 0.5 *
               integral((vperp,vpa) -> (vpa-upar)*((vpa-upar)^2 + vperp^2), ff, vperp, vpa)
    end
end

# generalised moment useful for computing numerical conserving terms in the collision operator
function get_rmom(ff, upar, vpa, vperp)
    return integral((vperp,vpa)->((vpa-upar)^2 + vperp^2)^2, ff, vperp, vpa)
end

"""
'Primary' time derivatives are calculated when doing the time advance for moment
quantities. Moment kinetic equations require in addition some 'derived' time derivatives,
which we can calculate by applying the chain rule.
"""
function update_derived_ion_moment_time_derivatives!(fvec_in, moments)
    @begin_s_r_z_region()

    n = fvec_in.density
    upar = fvec_in.upar
    p = fvec_in.p
    vth = moments.ion.vth
    dn_dt = moments.ion.ddens_dt
    dnupar_dt = moments.ion.dnupar_dt
    dp_dt = moments.ion.dp_dt

    dupar_dt = moments.ion.dupar_dt
    dvth_dt = moments.ion.dvth_dt

    if dupar_dt !== nothing
        @loop_s_r_z is ir iz begin
            dupar_dt[iz,ir,is] = (dnupar_dt[iz,ir,is] - upar[iz,ir,is] * dn_dt[iz,ir,is]) / n[iz,ir,is]
        end
    end

    if dvth_dt !== nothing
        @loop_s_r_z is ir iz begin
            # vth = sqrt(2*ppar/n)
            # dvth/dt = 1 / sqrt(2*ppar*n) * dppar/dt - sqrt(ppar/2/n^3) * dn/dt
            dvth_dt[iz,ir,is] = 0.5 * vth[iz,ir,is] *
                                (dp_dt[iz,ir,is] / p[iz,ir,is] - dn_dt[iz,ir,is] / n[iz,ir,is])
        end
    end

    return nothing
end

"""
'Primary' time derivatives are calculated when doing the time advance for moment
quantities. Moment kinetic equations require in addition some 'derived' time derivatives,
which we can calculate by applying the chain rule.

For electrons the kinetic equation that is solved for the shape function is simplified by
using \$\\sqrt{m_e/m_i}\$ as a small parameter _after_ substituting in the moment
equations. Therefore it is not possible (or at least not convenient) to use `dp_dt`
as calculated from the moment equation. The electron moment time derivatives may still be
useful as diagnostics, so we calculate them anyway, including the derived versions
(calculated in this function), similar to the ion moment time derivatives.

Note that due to the implicit/explicit splitting of terms in the timestep, which means
that the density (which is equal to the ion density) does not update during the implicit
electron timestep, the time derivative of density dn/dt does not contribute to the chain
rule calculations in this function, even though analytically it would contribute (although
the contribution would be small in sqrt(me/mi)). Another way of saying this is that due to
the operator splitting, the time derivatives during the implicit step are done with the
explicit variables (density in particular) held fixed, and so
`dvth_dt|_n = 0.5 * vth * dp_dt / p`.
"""
function update_derived_electron_moment_time_derivatives!(p, moments, electron_physics,
                                                          ir)
    @begin_anyzv_z_region()

    if moments.electron.dvth_dt !== nothing
        dvth_dt = @view moments.electron.dvth_dt[:,ir]
        vth = @view moments.electron.vth[:,ir]
        if electron_physics == kinetic_electrons_with_temperature_equation
            dT_dt = @view moments.electron.dT_dt[:,ir]
            T = @view moments.electron.temp[:,ir]
            @loop_z iz begin
                # vth = sqrt(2*T)
                # dvth/dt = 1 / (2*T) * dT/dt
                dvth_dt[iz] = 0.5 * vth[iz] * dT_dt[iz] / T[iz]
            end
        else
            dp_dt = @view moments.electron.dp_dt[:,ir]
            p = @view moments.electron.p[:,ir]
            @loop_z iz begin
                # vth = sqrt(2*p/n)
                # dvth/dt = 1 / sqrt(2*p*n) * dp/dt - sqrt(p/2/n^3) * dn/dt
                # but no dn/dt contribution because due to the implicit/explicit splitting
                # of terms, the density does not update during the update of the electron
                # pressure and shape function. Therefore T_out = p_out / density_in,
                # T_in = p_in / density_in so that effectively, for the electron update
                # dn_dt = (density_in - density_in) / dt_implicit = 0
                dvth_dt[iz] = 0.5 * vth[iz] * dp_dt[iz] / p[iz]
            end
        end
    end

    return nothing
end

"""
'Primary' time derivatives are calculated when doing the time advance for moment
quantities. Moment kinetic equations require in addition some 'derived' time derivatives,
which we can calculate by applying the chain rule.
"""
function update_derived_neutral_moment_time_derivatives!(fvec_in, moments)
    @begin_sn_r_z_region()

    n = fvec_in.density_neutral
    uz = fvec_in.uz_neutral
    p = fvec_in.p_neutral
    vth = moments.neutral.vth
    dn_dt = moments.neutral.ddens_dt
    dnuz_dt = moments.neutral.dnuz_dt
    dp_dt = moments.neutral.dp_dt

    duz_dt = moments.neutral.duz_dt
    dvth_dt = moments.neutral.dvth_dt

    if duz_dt !== nothing
        @loop_sn_r_z isn ir iz begin
            duz_dt[iz,ir,isn] = (dnuz_dt[iz,ir,isn] - uz[iz,ir,isn] * dn_dt[iz,ir,isn]) / n[iz,ir,isn]
        end
    end

    if dvth_dt !== nothing
        @loop_sn_r_z isn ir iz begin
            # vth = sqrt(2*p/n)
            # dvth/dt = 1 / sqrt(2*p*n) * dp/dt - sqrt(p/2/n^3) * dn/dt
            dvth_dt[iz,ir,isn] = 0.5 * vth[iz,ir,isn] *
                                 (dp_dt[iz,ir,isn] / p[iz,ir,isn] - dn_dt[iz,ir,isn] / n[iz,ir,isn])
        end
    end

    return nothing
end

"""
runtime diagnostic routine for computing the Chodura ratio
in a single species plasma with Z = 1
"""
function update_chodura!(moments,ff,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
    @boundscheck composition.n_ion_species == size(ff, 5) || throw(BoundsError(ff))

    if vpa.n == 1
        # Cannot calculate integral sensibly
        @begin_s_r_region()
        @loop_s_r is ir begin
            moments.ion.chodura_integral_lower[ir,is] = 0.0
            moments.ion.chodura_integral_upper[ir,is] = 0.0
        end
        return nothing
    end

    @begin_s_z_vperp_vpa_region()
    # use buffer_vpavperpzrs_2 here as buffer_vpavperpzrs_1 is in use storing ff
    dffdr = scratch_dummy.buffer_vpavperpzrs_2 
    ff_dummy = scratch_dummy.dummy_vpavperp
    if r.n > 1
    # first compute d f / d r using centred reconciliation and place in dummy array #1
    derivative_r!(dffdr, ff[:,:,:,:,:],
                  scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
                  scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
                  r_spectral,r)
    else
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            dffdr[ivpa,ivperp,iz,ir,is] = 0.0
        end
    end 
    
    del_vpa = minimum(vpa.grid[2:vpa.ngrid].-vpa.grid[1:vpa.ngrid-1])
    @begin_s_r_region()
    if z.irank == 0
        @loop_s_r is ir begin
            @views moments.ion.chodura_integral_lower[ir,is] = update_chodura_integral_species!(ff[:,:,1,ir,is],dffdr[:,:,1,ir,is],
            ff_dummy[:,:],vpa,vperp,z,r,composition,geometry,z_advect[is].speed[1,:,:,ir],moments.ion.dens[1,ir,is],del_vpa,1,ir)
        end
    else # we do not save this Chodura integral to the output file
        @loop_s_r is ir begin
            moments.ion.chodura_integral_lower[ir,is] = 0.0
        end
    end
    if z.irank == z.nrank - 1
        @loop_s_r is ir begin
            @views moments.ion.chodura_integral_upper[ir,is] = update_chodura_integral_species!(ff[:,:,end,ir,is],dffdr[:,:,end,ir,is],
            ff_dummy[:,:],vpa,vperp,z,r,composition,geometry,z_advect[is].speed[end,:,:,ir],moments.ion.dens[end,ir,is],del_vpa,z.n,ir)
        end
    else # we do not save this Chodura integral to the output file
        @loop_s_r is ir begin
            moments.ion.chodura_integral_upper[ir,is] =  0.0
        end
    end
end

"""
compute the integral needed for the generalised Chodura condition

 IChodura = (Z^2 vBohm^2 / cref^2) * int ( f bz^2 / vz^2 + dfdr*rhostar/vz )
 vBohm = sqrt(Z Te/mi)
 with Z = 1 and mref = mi
 cref = sqrt(2Ti/mi)
and normalise to the local ion density, appropriate to assessing the 
Chodura condition 

    IChodura <= (Te/e)d ne / dphi |(sheath entrance) = ni
 to a single species plasma with Z = 1

"""
function update_chodura_integral_species!(ff,dffdr,ff_dummy,vpa,vperp,z,r,composition,geometry,vz,dens,del_vpa,iz,ir)
    @boundscheck vpa.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vperp.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vpa.n == size(dffdr, 1) || throw(BoundsError(dffdr))
    @boundscheck vperp.n == size(dffdr, 2) || throw(BoundsError(dffdr))
    @boundscheck vpa.n == size(ff_dummy, 1) || throw(BoundsError(ff_dummy))
    @boundscheck vperp.n == size(ff_dummy, 2) || throw(BoundsError(ff_dummy))
    bzed = geometry.bzed
    @loop_vperp_vpa ivperp ivpa begin
        # avoid divide by zero by making sure 
        # we are more than a vpa mimimum grid spacing away from 
        # the vz(vpa,r) = 0 velocity boundary
        if abs(vz[ivpa,ivperp]) > 0.5*del_vpa
            ff_dummy[ivpa,ivperp] = (ff[ivpa,ivperp]*bzed[iz,ir]^2/(vz[ivpa,ivperp]^2) + 
                                geometry.rhostar*dffdr[ivpa,ivperp]/vz[ivpa,ivperp])
        else
            ff_dummy[ivpa,ivperp] = 0.0
        end
    end
    chodura_integral = integral(@view(ff_dummy[:,:]), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
    # multiply by Te factor from vBohm and divide by the local ion density
    chodura_integral *= composition.T_e/dens
    #println("chodura_integral: ",chodura_integral)
    return chodura_integral
end

"""
Pre-calculate spatial derivatives of the moments that will be needed for the time advance
"""
function calculate_ion_moment_derivatives!(moments, fields, geometry, scratch,
                                           scratch_dummy, r, z, r_spectral, z_spectral,
                                           ion_mom_diss_coeff)
    @begin_s_r_region()

    density = scratch.density
    upar = scratch.upar
    p = scratch.p
    ppar = moments.ion.ppar
    qpar = moments.ion.qpar
    vth = moments.ion.vth
    bz = geometry.bzed
    dummy_zrs = scratch_dummy.dummy_zrs
    buffer_r_1 = scratch_dummy.buffer_rs_1
    buffer_r_2 = scratch_dummy.buffer_rs_2
    buffer_r_3 = scratch_dummy.buffer_rs_3
    buffer_r_4 = scratch_dummy.buffer_rs_4
    buffer_r_5 = scratch_dummy.buffer_rs_5
    buffer_r_6 = scratch_dummy.buffer_rs_6
    buffer_z_1 = scratch_dummy.buffer_zs_1
    buffer_z_2 = scratch_dummy.buffer_zs_2
    buffer_z_3 = scratch_dummy.buffer_zs_3
    buffer_z_4 = scratch_dummy.buffer_zs_4
    buffer_z_5 = scratch_dummy.buffer_zs_5
    buffer_z_6 = scratch_dummy.buffer_zs_6

    # Non-upwinded derivatives
    if moments.evolve_density
        @views derivative_z!(moments.ion.ddens_dz, density, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
    end
    if moments.evolve_density && ion_mom_diss_coeff > 0.0

        # centred second derivative for dissipation
        @views second_derivative_z!(moments.ion.d2dens_dz2, density, buffer_r_1,
                                    buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
    end
    if moments.evolve_density || moments.evolve_upar || moments.evolve_p
        @views derivative_z!(moments.ion.dupar_dz, upar, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
    end
    if moments.evolve_upar && ion_mom_diss_coeff > 0.0
        # centred second derivative for dissipation
        @views second_derivative_z!(moments.ion.d2upar_dz2, upar, buffer_r_1, buffer_r_2,
                                    buffer_r_3, buffer_r_4, z_spectral, z)
    end
    if moments.evolve_upar
        @views derivative_z!(moments.ion.dppar_dz, ppar, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
    end
    if moments.evolve_p
        @views derivative_z!(moments.ion.dp_dz, p, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)

        if ion_mom_diss_coeff > 0.0
            # centred second derivative for dissipation
            @views second_derivative_z!(moments.ion.d2p_dz2, p, buffer_r_1,
                                        buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
        end

        @views derivative_z!(moments.ion.dqpar_dz, qpar, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
        @views derivative_z!(moments.ion.dvth_dz, vth, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)

        # calculate the z derivative of the ion temperature
        @loop_s_r_z is ir iz begin
            # store the temperature in dummy_zrs
            dummy_zrs[iz,ir,is] = p[iz,ir,is]/density[iz,ir,is]
        end
        @views derivative_z!(moments.ion.dT_dz, dummy_zrs, buffer_r_1,
                            buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
    end

    # z-upwinded derivatives
    vEz = fields.vEz
    if moments.evolve_density
        # Upwinded with z-advection velocity, to be used in continuity equation
        @loop_s_r_z is ir iz begin
            dummy_zrs[iz,ir,is] = -(vEz[iz,ir] + bz[iz,ir] * upar[iz,ir,is])
        end
        @views derivative_z!(moments.ion.ddens_dz_upwind, density, dummy_zrs, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, buffer_r_5, buffer_r_6,
                             z_spectral, z)
    end
    if moments.evolve_upar
        # Upwinded using upar as advection velocity, to be used in force-balance
        # equation
        if !moments.evolve_density
            # If evolving density this was already calculated
            @loop_s_r_z is ir iz begin
                dummy_zrs[iz,ir,is] = -(vEz[iz,ir] + bz[iz,ir] * upar[iz,ir,is])
            end
        end
        @views derivative_z!(moments.ion.dupar_dz_upwind, upar, dummy_zrs, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, buffer_r_5, buffer_r_6,
                             z_spectral, z)
    end
    if moments.evolve_p
        # Upwinded using upar as advection velocity, to be used in energy equation
        if !(moments.evolve_density || moments.evolve_upar)
            # If evolving density or upar this was already calculated
            @loop_s_r_z is ir iz begin
                dummy_zrs[iz,ir,is] = -(vEz[iz,ir] + bz[iz,ir] * upar[iz,ir,is])
            end
        end
        @views derivative_z!(moments.ion.dp_dz_upwind, p, dummy_zrs, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, buffer_r_5, buffer_r_6,
                             z_spectral, z)
    end

    # r derivatives
    if r.n > 1
        @begin_s_z_region()

        vEr = fields.vEr
        if moments.evolve_density
            @views derivative_r!(moments.ion.ddens_dr, density, buffer_z_1, buffer_z_2,
                                 buffer_z_3, buffer_z_4, r_spectral, r)
            # Upwinded with r-advection velocity, to be used in continuity equation
            @loop_s_r_z is ir iz begin
                dummy_zrs[iz,ir,is] = -vEr[iz,ir]
            end
            @views derivative_r!(moments.ion.ddens_dr_upwind, density,
                                 dummy_zrs, buffer_z_1, buffer_z_2, buffer_z_3, buffer_z_4,
                                 buffer_z_5, buffer_z_6, r_spectral, r)
        end
        if moments.evolve_upar
            @views derivative_r!(moments.ion.dupar_dr, upar, buffer_z_1, buffer_z_2,
                                 buffer_z_3, buffer_z_4, r_spectral, r)
            # Upwinded using upar as advection velocity, to be used in force-balance
            # equation
            if !moments.evolve_density
                # If evolving density this was already calculated
                @loop_s_r_z is ir iz begin
                    dummy_zrs[iz,ir,is] = -vEr[iz,ir]
                end
            end
            @views derivative_r!(moments.ion.dupar_dr_upwind, upar, dummy_zrs, buffer_z_1,
                                 buffer_z_2, buffer_z_3, buffer_z_4, buffer_z_5,
                                 buffer_z_6, r_spectral, r)
        end
        if moments.evolve_p
            @views derivative_r!(moments.ion.dvth_dr, vth, buffer_z_1,
                                 buffer_z_2, buffer_z_3, buffer_z_4, r_spectral, r)
            # Upwinded using upar as advection velocity, to be used in energy equation
            if !(moments.evolve_density || moments.evolve_upar)
                # If evolving density or upar this was already calculated
                @loop_s_r_z is ir iz begin
                    dummy_zrs[iz,ir,is] = -vEr[iz,ir]
                end
            end
            @views derivative_r!(moments.ion.dp_dr_upwind, p, dummy_zrs, buffer_z_1,
                                 buffer_z_2, buffer_z_3, buffer_z_4, buffer_z_5,
                                 buffer_z_6, r_spectral, r)
        end
    end

    return nothing
end

"""
Pre-calculate spatial derivatives of the electron moments that will be needed for the time advance
"""
function calculate_electron_moment_derivatives!(moments, scratch, scratch_dummy, z, z_spectral,
                                                electron_mom_diss_coeff, electron_model)
    @begin_r_region()

    dens = scratch.electron_density
    upar = scratch.electron_upar
    p = scratch.electron_p
    ppar = moments.electron.ppar
    qpar = moments.electron.qpar
    vth = moments.electron.vth
    dummy_zr = @view scratch_dummy.dummy_zrs[:,:,1]
    buffer_r_1 = @view scratch_dummy.buffer_rs_1[:,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[:,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[:,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[:,1]
       
    @views derivative_z!(moments.electron.dupar_dz, upar, buffer_r_1,
                         buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)

    @views derivative_z!(moments.electron.ddens_dz, dens, buffer_r_1, buffer_r_2,
                         buffer_r_3, buffer_r_4, z_spectral, z)
    @views derivative_z!(moments.electron.dp_dz, p, buffer_r_1, buffer_r_2, buffer_r_3,
                         buffer_r_4, z_spectral, z)
    @views derivative_z!(moments.electron.dppar_dz, ppar, buffer_r_1, buffer_r_2,
                         buffer_r_3, buffer_r_4, z_spectral, z)
    @views derivative_z!(moments.electron.dqpar_dz, qpar, buffer_r_1, buffer_r_2,
                         buffer_r_3, buffer_r_4, z_spectral, z)
    @views derivative_z!(moments.electron.dvth_dz, vth, buffer_r_1, buffer_r_2,
                         buffer_r_3, buffer_r_4, z_spectral, z)
    # calculate the zed derivative of the electron temperature
    @views derivative_z!(moments.electron.dT_dz, moments.electron.temp, buffer_r_1, buffer_r_2,
                         buffer_r_3, buffer_r_4, z_spectral, z)
    @views derivative_z!(moments.electron.dvth_dz, moments.electron.vth, buffer_r_1,
                         buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)

    # centred second derivative for dissipation
    if electron_mom_diss_coeff > 0.0
        @views derivative_z!(moments.electron.d2p_dz2, moments.electron.dp_dz, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
    end
end

"""
Calculate spatial derivatives of the electron moments.

This version, for use in implicit solvers for electrons, works with a single point in `r`,
given by `ir`.
"""
function calculate_electron_moment_derivatives_no_r!(moments, dens, upar, p,
                                                     scratch_dummy, z, z_spectral,
                                                     electron_mom_diss_coeff, ir)
    @begin_anyzv_region()

    ppar = @view moments.electron.ppar[:,ir]
    qpar = @view moments.electron.qpar[:,ir]
    vth = @view moments.electron.vth[:,ir]
    temp = @view moments.electron.temp[:,ir]
    dummy_z = @view scratch_dummy.dummy_zrs[:,ir,1]
    buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

    @views derivative_z_anyzv!(moments.electron.dupar_dz[:,ir], upar, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)

    @views derivative_z_anyzv!(moments.electron.ddens_dz[:,ir], dens, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)
    @views derivative_z_anyzv!(moments.electron.dp_dz[:,ir], p, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)
    @views derivative_z_anyzv!(moments.electron.dppar_dz[:,ir], ppar, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)
    @views derivative_z_anyzv!(moments.electron.dqpar_dz[:,ir], qpar, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)
    @views derivative_z_anyzv!(moments.electron.dvth_dz[:,ir], vth, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)
    # calculate the zed derivative of the electron temperature
    @views derivative_z_anyzv!(moments.electron.dT_dz[:,ir], temp, buffer_1, buffer_2,
                               buffer_3, buffer_4, z_spectral, z)
    @views derivative_z_anyzv!(moments.electron.dvth_dz[:,ir], moments.electron.vth[:,ir],
                               buffer_1, buffer_2, buffer_3, buffer_4, z_spectral, z)

    # centred second derivative for dissipation
    if electron_mom_diss_coeff > 0.0
        @views derivative_z_anyzv!(moments.electron.d2p_dz2[:,ir],
                                   moments.electron.dp_dz[:,ir], buffer_1, buffer_2,
                                   buffer_3, buffer_4, z_spectral, z)
    end
end

"""
update velocity moments of the evolved neutral pdf
"""
function update_moments_neutral!(moments, pdf, vz, vr, vzeta, z, r, composition)
    @begin_r_z_region()
    n_species = size(pdf,6)
    @boundscheck n_species == size(moments.neutral.dens,3) || throw(BoundsError(moments))
    @loop_sn isn begin
        if moments.neutral.dens_updated[isn] == false
            @views update_neutral_density_species!(moments.neutral.dens[:,:,isn],
                                                   pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r)
            moments.neutral.dens_updated[isn] = true
        end
        if moments.neutral.uz_updated[isn] == false
            # Can pass moments.neutral.pz here even though it has not been updated yet,
            # because moments.neutral.pz isn only needed if evolve_p=true, in which
            # case it will not be updated because it isn not calculated from the
            # distribution function
            @views update_neutral_uz_species!(moments.neutral.uz[:,:,isn],
                                              moments.neutral.dens[:,:,isn],
                                              moments.neutral.pz[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              moments.evolve_density, moments.evolve_p)
            moments.neutral.uz_updated[isn] = true
        end
        if moments.neutral.ur_updated[isn] == false
            @views update_neutral_ur_species!(moments.neutral.ur[:,:,isn],
                                              moments.neutral.dens[:,:,isn],
                                              moments.neutral.vth[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              moments.evolve_density, moments.evolve_p)
            moments.neutral.ur_updated[isn] = true
        end
        if moments.neutral.uzeta_updated[isn] == false
            @views update_neutral_uzeta_species!(moments.neutral.uzeta[:,:,isn],
                                                 moments.neutral.dens[:,:,isn],
                                                 moments.neutral.vth[:,:,isn],
                                                 pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                                 moments.evolve_density, moments.evolve_p)
            moments.neutral.uzeta_updated[isn] = true
        end
        if moments.neutral.pz_updated[isn] == false
            @views update_neutral_pz_species!(moments.neutral.pz[:,:,isn],
                                              moments.neutral.dens[:,:,isn],
                                              moments.neutral.uz[:,:,isn],
                                              moments.neutral.vth[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              moments.evolve_density, moments.evolve_upar,
                                              moments.evolve_p)
            moments.neutral.pz_updated[isn] = true
        end
        if moments.neutral.pr_updated[isn] == false
            @views update_neutral_pr_species!(moments.neutral.pr[:,:,isn],
                                              moments.neutral.dens[:,:,isn],
                                              moments.neutral.ur[:,:,isn],
                                              moments.neutral.vth[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              moments.evolve_density, moments.evolve_upar,
                                              moments.evolve_p)
            moments.neutral.pr_updated[isn] = true
        end
        if moments.neutral.pzeta_updated[isn] == false
            @views update_neutral_pzeta_species!(moments.neutral.pzeta[:,:,isn],
                                                 moments.neutral.dens[:,:,isn],
                                                 moments.neutral.uzeta[:,:,isn],
                                                 moments.neutral.vth[:,:,isn],
                                                 pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                                 moments.evolve_density,
                                                 moments.evolve_upar, moments.evolve_p)
            moments.neutral.pr_updated[isn] = true
        end
        if !moments.evolve_p
            @loop_r_z ir iz begin
                moments.neutral.p[iz,ir,isn] = 1.0/3.0 * (moments.neutral.pzeta[iz,ir,isn] +
                                                          moments.neutral.pr[iz,ir,isn] +
                                                          moments.neutral.pz[iz,ir,isn])
            end
        end
        @loop_r_z ir iz begin
            moments.neutral.vth[iz,ir,isn] =
                sqrt(2*moments.neutral.p[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
        end
        if moments.neutral.qz_updated[isn] == false
            @views update_neutral_qz_species!(moments.neutral.qz[:,:,isn],
                                              moments.neutral.dens[:,:,isn],
                                              moments.neutral.uz[:,:,isn],
                                              moments.neutral.vth[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              moments.evolve_density, moments.evolve_upar,
                                              moments.evolve_p)
            moments.neutral.qz_updated[isn] = true
        end
    end
    return nothing
end

"""
calculate the neutral density from the neutral pdf
"""
function update_neutral_density!(dens, dens_updated, pdf, vz, vr, vzeta, z, r,
                                 composition)
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(dens, 3) || throw(BoundsError(dens))
    @loop_sn isn begin
        if dens_updated[isn] == false
            @views update_neutral_density_species!(dens[:,:,isn], pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r)
            dens_updated[isn] = true
        end
    end
end

"""
calculate the updated density (dens) for a given species
"""
function update_neutral_density_species!(dens, ff, vz, vr, vzeta, z, r)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(dens, 1) || throw(BoundsError(dens))
    @boundscheck r.n == size(dens, 2) || throw(BoundsError(dens))
    @loop_r_z ir iz begin
        dens[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 0,
                               vr.wgts, vzeta.grid, 0, vzeta.wgts)
    end
    return nothing
end

function get_neutral_density(ff, vz, vr, vzeta)
    return integral(ff, vz.grid, 0, vz.wgts, vr.grid, 0, vr.wgts, vzeta.grid, 0,
                    vzeta.wgts)
end


function update_neutral_uz!(uz, uz_updated, density, vth, pdf, vz, vr, vzeta, z, r,
                            composition, evolve_density, evolve_p)
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(uz, 3) || throw(BoundsError(uz))
    @loop_sn isn begin
        if uz_updated[isn] == false
            @views update_neutral_uz_species!(uz[:,:,isn], density[:,:,isn], vth[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              evolve_density, evolve_p)
            uz_updated[isn] = true
        end
    end
end

"""
calculate the updated uz (mean velocity in z) for a given species
"""
function update_neutral_uz_species!(uz, density, vth, ff, vz, vr, vzeta, z, r,
                                    evolve_density, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(uz, 1) || throw(BoundsError(uz))
    @boundscheck r.n == size(uz, 2) || throw(BoundsError(uz))
    if evolve_density && evolve_p
        # this is the case where the density and parallel pressure are evolved
        # separately from the normalized pdf, g_s = (√π f_s vth_s / n_s); the vz
        # coordinate is (dz/dt) / vth_s.
        # Integrating calculates
        # (upar_s / vth_s) = (1/√π)∫d(vz/vth_s) * (vz/vth_s) * (√π f_s vth_s / n_s)
        # so convert from upar_s / vth_s to upar_s / c_s
        @loop_r_z ir iz begin
            uz[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 1, vz.wgts, vr.grid, 0,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) * vth[iz,ir]
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf, given by g_s = (√π f_s c_s / n_s); the vz coordinate is
        # (dz/dt) / c_s.
        # Integrating calculates
        # (upar_s / c_s) = (1/√π)∫d(vz/c_s) * (vz/c_s) * (√π f_s c_s / n_s)
        @loop_r_z ir iz begin
            uz[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 1, vz.wgts, vr.grid, 0,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts)
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vz coordinate is (dz/dt) / c_s.
        # Integrating calculates
        # (n_s / N_e) * (uz / c_s) = (1/√π)∫d(vz/c_s) * (vz/c_s) * (√π f_s c_s / N_e)
        @loop_r_z ir iz begin
            uz[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 1, vz.wgts, vr.grid, 0,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) / density[iz,ir]
        end
    end
    return nothing
end

function get_neutral_uz(ff, density, vz, vr, vzeta, evolve_density)
    upar = integral(ff, vz.grid, 1, vz.wgts, vr.grid, 0, vr.wgts, vzeta.grid, 0,
                    vzeta.wgts)
    if !evolve_density
        upar /= density
    end
    return upar
end

function update_neutral_ur!(ur, ur_updated, density, vth, pdf, vz, vr, vzeta, z, r,
                            composition, evolve_density, evolve_p)
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(ur, 3) || throw(BoundsError(ur))
    @loop_sn isn begin
        if ur_updated[isn] == false
            @views update_neutral_ur_species!(ur[:,:,isn], density[:,:,isn], vth[:,:,isn],
                                              pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                              evolve_density, evolve_p)
            ur_updated[isn] = true
        end
    end
end

"""
calculate the updated ur (mean velocity in r) for a given species
"""
function update_neutral_ur_species!(ur, density, vth, ff, vz, vr, vzeta, z, r,
                                    evolve_density, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(ur, 1) || throw(BoundsError(ur))
    @boundscheck r.n == size(ur, 2) || throw(BoundsError(ur))
    if evolve_p
        @loop_r_z ir iz begin
            ur[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 1,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) * vth[iz,ir]
        end
    elseif evolve_density
        @loop_r_z ir iz begin
            ur[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 1,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts)
        end
    else
        @loop_r_z ir iz begin
            ur[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 1,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) / density[iz,ir]
        end
    end
    return nothing
end

function update_neutral_uzeta!(uzeta, uzeta_updated, density, vth, pdf, vz, vr, vzeta, z,
                               r, composition, evolve_density, evolve_p)

    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(uzeta, 3) || throw(BoundsError(uzeta))
    @loop_sn isn begin
        if uzeta_updated[isn] == false
            @views update_neutral_uzeta_species!(uzeta[:,:,isn], density[:,:,isn],
                                                 vth[:,:,isn], pdf[:,:,:,:,:,isn], vz, vr,
                                                 vzeta, z, r, evolve_density, evolve_p)
            uzeta_updated[isn] = true
        end
    end
end

"""
calculate the updated uzeta (mean velocity in zeta) for a given species
"""
function update_neutral_uzeta_species!(uzeta, density, vth, ff, vz, vr, vzeta, z, r,
                                       evolve_density, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(uzeta, 1) || throw(BoundsError(uzeta))
    @boundscheck r.n == size(uzeta, 2) || throw(BoundsError(uzeta))
    if evolve_p
        @loop_r_z ir iz begin
            uzeta[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid,
                                    0, vr.wgts, vzeta.grid, 1, vzeta.wgts) * vth[iz,ir]
        end
    elseif evolve_density
        @loop_r_z ir iz begin
            uzeta[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid,
                                    0, vr.wgts, vzeta.grid, 1, vzeta.wgts)
        end
    else
        @loop_r_z ir iz begin
            uzeta[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid,
                                    0, vr.wgts, vzeta.grid, 1, vzeta.wgts) / density[iz,ir]
        end
    end
    return nothing
end

function update_neutral_p!(p, p_updated, density, uz, ur, uzeta, vth, pdf, vz, vr, vzeta,
                           z, r, composition, evolve_density, evolve_upar, evolve_p)
    @boundscheck r.n == size(p,2) || throw(BoundsError(p))
    @boundscheck z.n == size(p,1) || throw(BoundsError(p))

    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(p, 3) || throw(BoundsError(p))

    @loop_sn isn begin
        if p_updated[isn] == false
            @views update_neutral_p_species!(p[:,:,isn], density[:,:,isn], uz[:,:,isn],
                                             ur[:,:,isn], uzeta[:,:,isn], vth[:,:,isn],
                                             pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                             evolve_density, evolve_upar, evolve_p)
            p_updated[isn] = true
        end
    end
end

"""
calculate the updated pressure (p) for a given species
"""
function update_neutral_p_species!(p, density, uz, ur, uzeta, vth, ff, vz, vr, vzeta, z,
                                   r, evolve_density, evolve_upar, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(p, 1) || throw(BoundsError(p))
    @boundscheck r.n == size(p, 2) || throw(BoundsError(p))
    if evolve_p
        error("update_neutral_p_species!() should not be called when evolve_p=true")
    elseif evolve_upar
        # this is the case where the parallel flow and density are evolved separately
        # from the normalized pdf; the vz coordinate is
        # <(vz - uz_s) / cref>.
        # Integrating calculates p_s/n_s + m_s/3*uzeta_s^2 + m_/3s*ur_s^2 = ∫d^3w w^2/3 * g_s
        @loop_r_z ir iz begin
            p[iz,ir] = 1.0/3.0 * (integral((vzeta,vr,vz)->(vz^2 + vzeta^2 + vr^2),
                                           @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
                                  - uzeta[iz,ir]^2 - ur[iz,ir]^2) *
                       density[iz,ir]
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf; the vz coordinate is <vz / c_s>.
        # Integrating calculates
        # p_s/n_s + m_s/3*uz_s^2 + m_s/3 u*eta_s^2 + m_s/3*ur_s^2 = ∫d^3v v^2/3 * f_s
        # so subtract off u^2 and multiply by density to get the internal energy density
        # (aka pressure)
        @loop_r_z ir iz begin
            p[iz,ir] = 1.0/3.0 * (integral((vzeta,vr,vz)->(vz^2 + vzeta^2 + vr^2),
                                           @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
                                  - uz[iz,ir]^2 - uzeta[iz,ir]^2 - ur[iz,ir]^2) *
                       density[iz,ir]
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vz coordinate is <vz / cref>.
        # Integrating calculates
        # p_s + m_s/3*n_s*uz_s^2 + m_s/3*n_s*uzeta_s^2 + m_s/3*n_s*ur_s^2 = ∫d^3v v^2/3 * f_s
        # so subtract off twice the mean kinetic energy density to get the
        # internal energy density (aka pressure)
        @loop_r_z ir iz begin
            p[iz,ir] = 1.0/3.0 * (integral((vzeta,vr,vz)->(vz^2 + vzeta^2 + vr^2),
                                           @view(ff[:,:,:,iz,ir]), vzeta, vr, vz) -
                                  density[iz,ir]*(uz[iz,ir]^2 + ur[iz,ir]^2 + uzeta[iz,ir]^2))
        end
    end
    return nothing
end

function get_neutral_p(ff, density, upar, vzeta, vr, vz, evolve_density, evolve_upar)
    if evolve_upar
        return density * integral((vzeta,vr,vz)->(vz^2 + vzeta^2 + vr^2), ff, vz, vr, vzeta)
    elseif evolve_density
        return density * integral((vzeta,vr,vz)->((vz-upar)^2 + vzeta^2 + vr^2), ff, vz, vr, vzeta)
    else
        return integral((vzeta,vr,vz)->((vz-upar)^2 + vzeta^2 + vr^2), ff, vz, vr, vzeta)
    end
end

function update_neutral_pz!(pz, pz_updated, density, uz, p, vth, pdf, vz, vr, vzeta, z, r,
                            composition, evolve_density, evolve_upar, evolve_p)
    @boundscheck r.n == size(pz,2) || throw(BoundsError(pz))
    @boundscheck z.n == size(pz,1) || throw(BoundsError(pz))
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(pz, 3) || throw(BoundsError(pz))
    
    if vzeta.n == 1 && vr.n == 1
        @loop_sn isn begin
            if pz_updated[isn] == false
                @loop_r_z ir iz begin
                    pz[iz,ir,isn] = 3.0 * p[iz,ir,isn]
                end
                pz_updated[isn] = true
            end
        end
    else
        @loop_sn isn begin
            if pz_updated[isn] == false
                @views update_neutral_pz_species!(pz[:,:,isn], density[:,:,isn], uz[:,:,isn],
                                                  vth[:,:,isn], pdf[:,:,:,:,:,isn], vz, vr,
                                                  vzeta, z, r, evolve_density, evolve_upar,
                                                  evolve_p)
                pz_updated[isn] = true
            end
        end
    end
end

"""
calculate the updated pressure in zz direction (pz) for a given species
"""
function update_neutral_pz_species!(pz, density, uz, vth, ff, vz, vr, vzeta, z, r,
                                    evolve_density, evolve_upar, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(pz, 1) || throw(BoundsError(pz))
    @boundscheck r.n == size(pz, 2) || throw(BoundsError(pz))
    if evolve_p
        # this is the case where the pressure, parallel flow, and density are evolved
        # separately from the shape function; the vz coordinate is
        # <(vz - uz_s) / vth_s>.
        # Integrating calculates pz_s/n_s/vth_s^2 = ∫d^3w wz^2 * g_s
        # so convert from pz_s/n_s/vth_s^2 to pz_s.
        @loop_r_z ir iz begin
            pz[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 2, vz.wgts, vr.grid, 0,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) *
                        density[iz,ir] * vth[iz,ir]^2
        end
    elseif evolve_upar
        # this is the case where the parallel flow and density are evolved separately
        # from the normalized pdf; the vz coordinate is
        # <(vz - uz_s) / cref>.
        # Integrating calculates pz_s/n_s = ∫d^3w wz^2 * g_s
        # so convert from pz_s/n_s to pz_s.

        @loop_r_z ir iz begin
            pz[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 2, vz.wgts, vr.grid, 0,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) *
                        density[iz,ir]
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf; the vz coordinate is <vz / c_s>.
        # Integrating calculates
        # pz_s/n_s + uz_s^2 = ∫d^3v vz^2 * f_s
        # so subtract off uz^2 and multiply by density to get the internal energy density
        # (aka pressure)
        @loop_r_z ir iz begin
            pz[iz,ir] = (integral(@view(ff[:,:,:,iz,ir]), vz.grid, 2, vz.wgts, vr.grid, 0,
                                  vr.wgts, vzeta.grid, 0, vzeta.wgts) -
                         uz[iz,ir]^2) *
                        density[iz,ir]
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vz coordinate is <vz / cref>.
        # Integrating calculates
        # pz_s + n_s*uz_s^2 = ∫d^3v vz^2 * f_s
        # so subtract off twice the mean z-direction kinetic energy density to get the
        # internal energy density (aka pressure)
        @loop_r_z ir iz begin
            pz[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 2, vz.wgts, vr.grid, 0,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) -
                        density[iz,ir]*uz[iz,ir]^2
        end
    end
    return nothing
end

function update_neutral_pr!(pr, pr_updated, density, ur, vth, pdf, vz, vr, vzeta, z, r,
                            composition, evolve_density, evolve_upar, evolve_p)
    @boundscheck r.n == size(pr,2) || throw(BoundsError(pr))
    @boundscheck z.n == size(pr,1) || throw(BoundsError(pr))
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(pr, 3) || throw(BoundsError(pr))
    
    @loop_sn isn begin
        if pr_updated[isn] == false
            @views update_neutral_pr_species!(pr[:,:,isn], density[:,:,isn], ur[:,:,isn],
                                              vth[:,:,isn], pdf[:,:,:,:,:,isn], vz, vr,
                                              vzeta, z, r, evolve_density, evolve_upar,
                                              evolve_p)
            pr_updated[isn] = true
        end
    end
end

"""
calculate the updated pressure in rr direction (pr) for a given species
"""
function update_neutral_pr_species!(pr, density, ur, vth, ff, vz, vr, vzeta, z, r,
                                    evolve_density, evolve_upar, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(pr, 1) || throw(BoundsError(pr))
    @boundscheck r.n == size(pr, 2) || throw(BoundsError(pr))
    if evolve_p
        # this is the case where the pressure, parallel flow, and density are evolved
        # separately from the shape function; the vr coordinate is <vr / vth_s>.
        # Integrating calculates pr_s/n_s/vth_s^2 + ur_s^2/vth_s^2 = ∫d^3w wr^2 * g_s
        # so convert from pr_s/n_s/vth_s^2+ur_s^2/vth_s^2 to pr_s.
        @loop_r_z ir iz begin
            pr[iz,ir] = (integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 2,
                                  vr.wgts, vzeta.grid, 0, vzeta.wgts) * vth[iz,ir]^2 -
                         ur[iz,ir]^2) *
                        density[iz,ir]
        end
    elseif evolve_density
        # corresponds to case where only the density, or parallel flow and density, are
        # evolved separately from the normalised pdf; the vr coordinate is <vr / c_s>.
        # Integrating calculates
        # pr_s/n_s + ur_s^2 = ∫d^3v vr^2 * f_s
        # so subtract off ur^2 and multiply by density to get the internal energy density
        # (aka pressure)
        @loop_r_z ir iz begin
            pr[iz,ir] = (integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 2,
                                  vr.wgts, vzeta.grid, 0, vzeta.wgts) -
                         ur[iz,ir]^2) *
                        density[iz,ir]
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vr coordinate is <vr / cref>.
        # Integrating calculates
        # pr_s + n_s*ur_s^2 = ∫d^3v vr^2 * f_s
        # so subtract off twice the mean r-direction kinetic energy density to get the
        # internal energy density (aka pressure)
        @loop_r_z ir iz begin
            pr[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid, 2,
                                 vr.wgts, vzeta.grid, 0, vzeta.wgts) -
                        density[iz,ir]*ur[iz,ir]^2
        end
    end
    return nothing
end

function update_neutral_pzeta!(pzeta, pzeta_updated, density, uzeta, vth, pdf, vz, vr,
                               vzeta, z, r, composition, evolve_density, evolve_upar,
                               evolve_p)
    @boundscheck r.n == size(pzeta,2) || throw(BoundsError(pzeta))
    @boundscheck z.n == size(pzeta,1) || throw(BoundsError(pzeta))
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(pzeta, 3) || throw(BoundsError(pzeta))
    
    @loop_sn isn begin
        if pzeta_updated[isn] == false
            @views update_neutral_pzeta_species!(pzeta[:,:,isn], density[:,:,isn],
                                                 uzeta[:,:,isn], vth[:,:,isn],
                                                 pdf[:,:,:,:,:,isn], vz, vr, vzeta, z, r,
                                                 evolve_density, evolve_upar, evolve_p)
            pzeta_updated[isn] = true
        end
    end
end

"""
calculate the updated pressure in zeta-zeta direction (pzeta) for a given species
"""
function update_neutral_pzeta_species!(pzeta, density, uzeta, vth, ff, vz, vr, vzeta, z,
                                       r, evolve_density, evolve_upar, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(pzeta, 1) || throw(BoundsError(pzeta))
    @boundscheck r.n == size(pzeta, 2) || throw(BoundsError(pzeta))
    if evolve_p
        # this is the case where the pressure, parallel flow, and density are evolved
        # separately from the shape function; the vzeta coordinate is <vzeta / vth_s>.
        # Integrating calculates pzeta_s/n_s/vth_s^2 + uzeta_s^2/vth_s^2 = ∫d^3w wzeta^2 * g_s
        # so convert from pzeta_s/n_s/vth_s^2+uzeta_s^2/vth_s^2 to pzeta_s.
        @loop_r_z ir iz begin
            pzeta[iz,ir] = (integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts,
                                     vr.grid, 0, vr.wgts, vzeta.grid, 2, vzeta.wgts) *
                            vth[iz,ir]^2
                            - uzeta[iz,ir]^2) *
                           density[iz,ir]
        end
    elseif evolve_density
        # corresponds to case where only the density, or parallel flow and density, are
        # evolved separately from the normalised pdf; the vzeta coordinate is <vzeta / c_s>.
        # Integrating calculates
        # pzeta_s/n_s + uzeta_s^2 = ∫d^3v vzeta^2 * f_s
        # so subtract off uzeta^2 and multiply by density to get the internal energy density
        # (aka pressure)
        @loop_r_z ir iz begin
            pzeta[iz,ir] = (integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid,
                                     0, vr.wgts, vzeta.grid, 2, vzeta.wgts) -
                            uzeta[iz,ir]^2) *
                           density[iz,ir]
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vzeta coordinate is <vzeta / cref>.
        # Integrating calculates
        # pzeta_s + n_s*uzeta_s^2 = ∫d^3v vzeta^2 * f_s
        # so subtract off twice the mean zeta-direction kinetic energy density to get the
        # internal energy density (aka pressure)
        @loop_r_z ir iz begin
            pzeta[iz,ir] = integral(@view(ff[:,:,:,iz,ir]), vz.grid, 0, vz.wgts, vr.grid,
                                    0, vr.wgts, vzeta.grid, 2, vzeta.wgts) -
                           density[iz,ir]*uzeta[iz,ir]^2
        end
    end
    return nothing
end

function update_neutral_vth!(vth, p, dens, z, r, composition)
    @boundscheck composition.n_neutral_species == size(vth,3) || throw(BoundsError(vth))
    @boundscheck r.n == size(vth,2) || throw(BoundsError(vth))
    @boundscheck z.n == size(vth,1) || throw(BoundsError(vth))

    @begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        vth[iz,ir,isn] = sqrt(2.0 * p[iz,ir,isn] / dens[iz,ir,isn])
    end
end

function update_neutral_qz!(qz, qz_updated, density, uz, vth, pdf, vz, vr, vzeta, z, r,
                            composition, evolve_density, evolve_upar, evolve_p)
    @boundscheck r.n == size(qz,2) || throw(BoundsError(qz))
    @boundscheck z.n == size(qz,1) || throw(BoundsError(qz))
    
    @begin_r_z_region()
    @boundscheck composition.n_neutral_species == size(pdf, 6) || throw(BoundsError(pdf))
    @boundscheck composition.n_neutral_species == size(qz, 3) || throw(BoundsError(qz))
    
    @loop_sn isn begin
        if qz_updated[isn] == false
            @views update_neutral_qz_species!(qz[:,:,isn], density[:,:,isn], uz[:,:,isn],
                                              vth[:,:,isn], pdf[:,:,:,:,:,isn], vz, vr,
                                              vzeta, z, r, evolve_density, evolve_upar,
                                              evolve_p)
            qz_updated[isn] = true
        end
    end
end

"""
calculate the updated heat flux zzz direction (qz) for a given species
"""
function update_neutral_qz_species!(qz, density, uz, vth, ff, vz, vr, vzeta, z, r,
                                    evolve_density, evolve_upar, evolve_p)
    @boundscheck vz.n == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck vr.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck vzeta.n == size(ff, 3) || throw(BoundsError(ff))
    @boundscheck z.n == size(ff, 4) || throw(BoundsError(ff))
    @boundscheck r.n == size(ff, 5) || throw(BoundsError(ff))
    @boundscheck z.n == size(qz, 1) || throw(BoundsError(qz))
    @boundscheck r.n == size(qz, 2) || throw(BoundsError(qz))
    if evolve_upar && evolve_p
        @loop_r_z ir iz begin
            qz[iz,ir] = 0.5 * density[iz,ir] * vth[iz,ir]^3 *
                        integral((vzeta,vr,vz)->(vz*(vz^2+vzeta^2+vr^2)),
                                 @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
        end
    elseif evolve_upar
        @loop_r_z ir iz begin
            qz[iz,ir] = 0.5 * density[iz,ir] *
                        integral((vzeta,vr,vz)->(vz*(vz^2+vzeta^2+vr^2)),
                                 @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
        end
    elseif evolve_p
        @loop_r_z ir iz begin
            qz[iz,ir] = 0.5 * density[iz,ir] * vth[iz,ir]^3 *
                        integral((vzeta,vr,vz)->((vz-uz[iz,ir])*((vz-uz[iz,ir])^2+vzeta^2+vr^2)),
                                 @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
        end
    elseif evolve_density
        @loop_r_z ir iz begin
            qz[iz,ir] = 0.5 * density[iz,ir] *
                        integral((vzeta,vr,vz)->((vz-uz[iz,ir])*((vz-uz[iz,ir])^2+vzeta^2+vr^2)),
                                 @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
        end
    else
        @loop_r_z ir iz begin
            qz[iz,ir] = 0.5 *
                        integral((vzeta,vr,vz)->((vz-uz[iz,ir])*((vz-uz[iz,ir])^2+vzeta^2+vr^2)),
                                 @view(ff[:,:,:,iz,ir]), vzeta, vr, vz)
        end
    end
    return nothing
end

"""
Pre-calculate spatial derivatives of the neutral moments that will be needed for the time
advance
"""
function calculate_neutral_moment_derivatives!(moments, scratch, scratch_dummy, z,
                                               z_spectral, neutral_mom_diss_coeff)
    @begin_sn_r_region()

    density = scratch.density_neutral
    uz = scratch.uz_neutral
    p = scratch.p_neutral
    pz = moments.neutral.pz
    qz = moments.neutral.qz
    vth = moments.neutral.vth
    dummy_zrsn = scratch_dummy.dummy_zrsn
    buffer_r_1 = scratch_dummy.buffer_rsn_1
    buffer_r_2 = scratch_dummy.buffer_rsn_2
    buffer_r_3 = scratch_dummy.buffer_rsn_3
    buffer_r_4 = scratch_dummy.buffer_rsn_4
    buffer_r_5 = scratch_dummy.buffer_rsn_5
    buffer_r_6 = scratch_dummy.buffer_rsn_6
    if moments.evolve_density
        @views derivative_z!(moments.neutral.ddens_dz, density, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z;
                             neutrals=true)
        # Upwinded using upar as advection velocity, to be used in continuity equation
        @loop_sn_r_z isn ir iz begin
            dummy_zrsn[iz,ir,isn] = -uz[iz,ir,isn]
        end
        @views derivative_z!(moments.neutral.ddens_dz_upwind, density,
                             dummy_zrsn, buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                             buffer_r_5, buffer_r_6, z_spectral, z; neutrals=true)
    end
    if moments.evolve_density && neutral_mom_diss_coeff > 0.0
        # centred second derivative for dissipation
        @views second_derivative_z!(moments.neutral.d2dens_dz2, density, buffer_r_1,
                                    buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z;
                                    neutrals=true)
    end
    if moments.evolve_density || moments.evolve_upar || moments.evolve_p
        @views derivative_z!(moments.neutral.duz_dz, uz, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z;
                             neutrals=true)
    end
    if moments.evolve_upar
        # Upwinded using upar as advection velocity, to be used in force-balance
        # equation
        @loop_sn_r_z isn ir iz begin
            dummy_zrsn[iz,ir,isn] = -uz[iz,ir,isn]
        end
        @views derivative_z!(moments.neutral.duz_dz_upwind, uz, dummy_zrsn,
                             buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                             buffer_r_5, buffer_r_6, z_spectral, z; neutrals=true)
    end
    if moments.evolve_upar && neutral_mom_diss_coeff > 0.0
        # centred second derivative for dissipation
        @views second_derivative_z!(moments.neutral.d2uz_dz2, uz, buffer_r_1, buffer_r_2,
                                    buffer_r_3, buffer_r_4, z_spectral, z; neutrals=true)
    end
    if moments.evolve_upar
        @views derivative_z!(moments.neutral.dpz_dz, pz, buffer_r_1, buffer_r_2,
                             buffer_r_3, buffer_r_4, z_spectral, z; neutrals=true)
    end
    if moments.evolve_p
        @views derivative_z!(moments.neutral.dp_dz, p, buffer_r_1, buffer_r_2, buffer_r_3,
                             buffer_r_4, z_spectral, z; neutrals=true)
        # Upwinded using upar as advection velocity, to be used in energy equation
        @loop_sn_r_z isn ir iz begin
            dummy_zrsn[iz,ir,isn] = -uz[iz,ir,isn]
        end
        @views derivative_z!(moments.neutral.dp_dz_upwind, p, dummy_zrsn,
                             buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                             buffer_r_5, buffer_r_6, z_spectral, z; neutrals=true)

        if neutral_mom_diss_coeff > 0.0
            # centred second derivative for dissipation
            @views second_derivative_z!(moments.neutral.d2p_dz2, p, buffer_r_1,
                                        buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z;
                                        neutrals=true)
        end

        @views derivative_z!(moments.neutral.dqz_dz, qz, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z;
                             neutrals=true)
        @views derivative_z!(moments.neutral.dvth_dz, vth, buffer_r_1,
                             buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z;
                             neutrals=true)
    end
end

"""
update velocity moments that are calculable from the evolved ion pdf
"""
function update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition,
                                 r_spectral, geometry, gyroavs, scratch_dummy, z_advect,
                                 collisions, diagnostic_moments)

    if composition.ion_physics == gyrokinetic_ions
        ff = scratch_dummy.buffer_vpavperpzrs_1
        # fill buffer with ring-averaged F (gyroaverage at fixed position)
        gyroaverage_pdf!(ff,new_scratch.pdf,gyroavs,vpa,vperp,z,r,composition)
    else
        ff = new_scratch.pdf
    end

    if !moments.evolve_density
        update_density!(new_scratch.density, moments.ion.dens_updated,
                        ff, vpa, vperp, z, r, composition)
    end
    if !moments.evolve_upar
        update_upar!(new_scratch.upar, moments.ion.upar_updated, new_scratch.density,
                     moments.ion.vth, ff, vpa, vperp, z, r, composition,
                     moments.evolve_density, moments.evolve_p)
    end
    if !moments.evolve_p
        # update_ppar! calculates (p_parallel/m_s N_e c_s^2) + (n_s/N_e)*(upar_s/c_s)^2 = (1/√π)∫d(vpa/c_s) (vpa/c_s)^2 * (√π f_s c_s / N_e)
        update_p!(new_scratch.p, moments.ion.p_updated, new_scratch.density,
                  new_scratch.upar, ff, vpa, vperp, z, r, composition,
                  moments.evolve_density, moments.evolve_upar)
    end
    update_ppar!(moments.ion.ppar, new_scratch.density, new_scratch.upar, moments.ion.vth,
                 new_scratch.p, ff, vpa, vperp, z, r, composition, moments.evolve_density,
                 moments.evolve_upar, moments.evolve_p)
    update_pperp!(moments.ion.pperp, new_scratch.p, moments.ion.ppar, vperp, z, r,
                  composition)

    # if diagnostic time step/RK stage
    # update the diagnostic chodura condition
    if diagnostic_moments
        if composition.ion_physics != coll_krook_ions
            update_chodura!(moments,ff,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
        end
    end
    # update the thermal speed
    @begin_s_r_z_region()
    update_vth!(moments.ion.vth, new_scratch.p, new_scratch.density, z, r,
                composition)
    # update the parallel heat flux
    update_ion_qpar!(moments.ion.qpar, moments.ion.qpar_updated, new_scratch.density,
                     new_scratch.upar, moments.ion.vth, moments.ion.dT_dz, ff, vpa, vperp,
                     z, r, composition, composition.ion_physics, collisions,
                     moments.evolve_density, moments.evolve_upar, moments.evolve_p)
    # add further moments to be computed here

    return nothing
end

"""
update velocity moments that are calculable from the evolved neutral pdf
"""
function update_derived_moments_neutral!(new_scratch, moments, vz, vr, vzeta, z, r,
                                         composition)

    if !moments.evolve_density
        update_neutral_density!(new_scratch.density_neutral, moments.neutral.dens_updated,
                                new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition)
    end
    if !moments.evolve_upar
        update_neutral_uz!(new_scratch.uz_neutral, moments.neutral.uz_updated,
                           new_scratch.density_neutral, moments.neutral.vth,
                           new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_p)
    end
    if !moments.evolve_p
        update_neutral_uzeta!(moments.neutral.uzeta, moments.neutral.uzeta_updated,
                              new_scratch.density_neutral, moments.neutral.vth,
                              new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                              moments.evolve_density, moments.evolve_p)
        update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                           new_scratch.density_neutral, moments.neutral.vth,
                           new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_p)
        update_neutral_p!(new_scratch.p_neutral, moments.neutral.p_updated,
                          new_scratch.density_neutral, new_scratch.uz_neutral,
                          moments.neutral.ur, moments.neutral.uzeta,
                          moments.neutral.vth, new_scratch.pdf_neutral, vz, vr, vzeta, z,
                          r, composition, moments.evolve_density, moments.evolve_upar,
                          moments.evolve_p)
    end
    # pz is needed for the neutral parallel momentum equation if evolving uz and the
    # neutral pressure equation if evolving p.
    update_neutral_pz!(moments.neutral.pz, moments.neutral.pz_updated,
                       new_scratch.density_neutral, new_scratch.uz_neutral,
                       new_scratch.p_neutral, moments.neutral.vth,
                       new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                       moments.evolve_density, moments.evolve_upar, moments.evolve_p)
    update_neutral_vth!(moments.neutral.vth, new_scratch.p_neutral,
                        new_scratch.density_neutral, z, r, composition)
end

"""
computes the integral over vpa >= 0 of the integrand, using the input vpa_wgts
this could be made more efficient for the case that dz/dt = vpa is time-independent,
but it has been left general for the cases where, e.g., dz/dt = wpa*vth + upar
varies in time
"""
function integrate_over_positive_vpa(integrand, dzdt, vpa_wgts, wgts_mod, vperp_grid,
                                     vperp_wgts)
    # define the nvpa variable for convenience
    nvpa = length(vpa_wgts)
    nvperp = length(vperp_wgts)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = -1.0e-15
    # if dzdt at the maximum vpa index is negative, then dzdt < 0 everywhere
    # the integral over positive dzdt is thus zero, as we assume the distribution
    # function is zero beyond the simulated vpa domain
    if dzdt[nvpa] < zero
        velocity_integral = 0.0
    else
        # do bounds checks on arrays that will be used in the below loop
        @boundscheck nvpa == size(integrand,1) || throw(BoundsError(integrand))
        @boundscheck nvperp == size(integrand,2) || throw(BoundsError(integrand))
        @boundscheck nvpa == length(dzdt) || throw(BoundsError(dzdt))
        @boundscheck nvpa == length(wgts_mod) || throw(BoundsError(wgts_mod))
        # initialise the integration weights, wgts_mod, to be the input vpa_wgts
        # this will only change at the dzdt = 0 point, if it exists on the grid
        @. wgts_mod = vpa_wgts
        # ivpa_zero will be the minimum index for which dzdt[ivpa_zero] >= 0
        ivpa_zero = nvpa
        @inbounds for ivpa ∈ 1:nvpa
            if dzdt[ivpa] >= zero
                ivpa_zero = ivpa
                # if dzdt = 0, need to divide its associated integration
                # weight by a factor of 2 to avoid double-counting
                if abs(dzdt[ivpa]) < abs(zero)
                    wgts_mod[ivpa] /= 2.0
                end
                break
            end
        end
        @views velocity_integral = integral(integrand[ivpa_zero:end,:],
                                            dzdt[ivpa_zero:end], 0,
                                            wgts_mod[ivpa_zero:end], vperp_grid, 0,
                                            vperp_wgts)
        # n.b. we pass more arguments than might appear to be required here
        # to avoid needing a special integral function definition
        # the 0 integers are the powers by which dzdt and vperp_grid are raised to in the integral
    end
    return velocity_integral
end

function integrate_over_positive_vz(integrand, dzdt, vz_wgts, wgts_mod, vr_grid, vr_wgts,
                                    vzeta_grid, vzeta_wgts)
    # define the nvz nvr nvzeta variable for convenience
    nvz = length(vz_wgts)
    nvr = length(vr_wgts)
    nvzeta = length(vzeta_wgts)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = -1.0e-15
    # if dzdt at the maximum vz index is negative, then dzdt < 0 everywhere
    # the integral over positive dzdt is thus zero, as we assume the distribution
    # function is zero beyond the simulated vpa domain
    if dzdt[nvz] < zero
        velocity_integral = 0.0
    else
        # do bounds checks on arrays that will be used in the below loop
        @boundscheck nvz == size(integrand,1) || throw(BoundsError(integrand))
        @boundscheck nvr == size(integrand,2) || throw(BoundsError(integrand))
        @boundscheck nvzeta == size(integrand,3) || throw(BoundsError(integrand))
        @boundscheck nvz == length(dzdt) || throw(BoundsError(dzdt))
        @boundscheck nvz == length(wgts_mod) || throw(BoundsError(wgts_mod))
        # initialise the integration weights, wgts_mod, to be the input vz_wgts
        # this will only change at the dzdt = 0 point, if it exists on the grid
        @. wgts_mod = vz_wgts
        # ivz_zero will be the minimum index for which dzdt[ivz_zero] >= 0
        ivz_zero = nvz
        @inbounds for ivz ∈ 1:nvz
            if dzdt[ivz] >= zero
                ivz_zero = ivz
                # if dzdt = 0, need to divide its associated integration
                # weight by a factor of 2 to avoid double-counting
                if abs(dzdt[ivz]) < abs(zero)
                    wgts_mod[ivz] /= 2.0
                end
                break
            end
        end
        @views velocity_integral = integral(integrand[ivz_zero:end,:,:],
                                            dzdt[ivz_zero:end], 0, wgts_mod[ivz_zero:end],
                                            vr_grid, 0, vr_wgts, vzeta_grid, 0,
                                            vzeta_wgts)
        # n.b. we pass more arguments than might appear to be required here
        # to avoid needing a special integral function definition
        # the 0 integers are the powers by which dzdt vr_grid and vzeta_grid are raised to in the integral
    end
    return velocity_integral
end

"""
computes the integral over vpa <= 0 of the integrand, using the input vpa_wgts
this could be made more efficient for the case that dz/dt = vpa is time-independent,
but it has been left general for the cases where, e.g., dz/dt = wpa*vth + upar
varies in time
"""
function integrate_over_negative_vpa(integrand, dzdt, vpa_wgts, wgts_mod, vperp_grid,
                                     vperp_wgts)
    # define the nvpa nvperp variables for convenience
    nvpa = length(vpa_wgts)
    nvperp = length(vperp_wgts)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = 1.0e-15
    # if dzdt at the mimimum vpa index is positive, then dzdt > 0 everywhere
    # the integral over negative dzdt is thus zero, as we assume the distribution
    # function is zero beyond the simulated vpa domain
    if dzdt[1] > zero
        velocity_integral = 0.0
    else
        # do bounds checks on arrays that will be used in the below loop
        @boundscheck nvpa == size(integrand,1) || throw(BoundsError(integrand))
        @boundscheck nvperp == size(integrand,2) || throw(BoundsError(integrand))
        @boundscheck nvpa == length(dzdt) || throw(BoundsError(dzdt))
        @boundscheck nvpa == length(wgts_mod) || throw(BoundsError(wgts_mod))
        # initialise the integration weights, wgts_mod, to be the input vpa_wgts
        # this will only change at the dzdt = 0 point, if it exists on the grid
        @. wgts_mod = vpa_wgts
        # ivpa_zero will be the maximum index for which dzdt[ivpa_zero] <= 0
        ivpa_zero = 1
        @inbounds for ivpa ∈ nvpa:-1:1
            if dzdt[ivpa] <= zero
                ivpa_zero = ivpa
                # if dzdt = 0, need to divide its associated integration
                # weight by a factor of 2 to avoid double-counting
                if abs(dzdt[ivpa]) < zero
                    wgts_mod[ivpa] /= 2.0
                end
                break
            end
        end
        @views velocity_integral = integral(integrand[1:ivpa_zero,:], dzdt[1:ivpa_zero],
                                            0, wgts_mod[1:ivpa_zero], vperp_grid, 0,
                                            vperp_wgts)
        # n.b. we pass more arguments than might appear to be required here
        # to avoid needing a special integral function definition
        # the 0 integers are the powers by which dzdt and vperp_grid are raised to in the integral
    end
    return velocity_integral
end
function integrate_over_negative_vz(integrand, dzdt, vz_wgts, wgts_mod, vr_grid, vr_wgts,
                                    vzeta_grid, vzeta_wgts)
    # define the nvz nvr nvzeta variables for convenience
    nvz = length(vz_wgts)
    nvr = length(vr_wgts)
    nvzeta = length(vzeta_wgts)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = 1.0e-15
    # if dzdt at the mimimum vz index is positive, then dzdt > 0 everywhere
    # the integral over negative dzdt is thus zero, as we assume the distribution
    # function is zero beyond the simulated vpa domain
    if dzdt[1] > zero
        velocity_integral = 0.0
    else
        # do bounds checks on arrays that will be used in the below loop
        @boundscheck nvz == size(integrand,1) || throw(BoundsError(integrand))
        @boundscheck nvr == size(integrand,2) || throw(BoundsError(integrand))
        @boundscheck nvzeta == size(integrand,3) || throw(BoundsError(integrand))
        @boundscheck nvz == length(dzdt) || throw(BoundsError(dzdt))
        @boundscheck nvz == length(wgts_mod) || throw(BoundsError(wgts_mod))
        # initialise the integration weights, wgts_mod, to be the input vz_wgts
        # this will only change at the dzdt = 0 point, if it exists on the grid
        @. wgts_mod = vz_wgts
        # ivz_zero will be the maximum index for which dzdt[ivz_zero] <= 0
        ivz_zero = 1
        @inbounds for ivz ∈ nvz:-1:1
            if dzdt[ivz] <= zero
                ivz_zero = ivz
                # if dzdt = 0, need to divide its associated integration
                # weight by a factor of 2 to avoid double-counting
                if abs(dzdt[ivz]) < zero
                    wgts_mod[ivz] /= 2.0
                end
                break
            end
        end
        @views velocity_integral = integral(integrand[1:ivz_zero,:,:], dzdt[1:ivz_zero],
                                            0, wgts_mod[1:ivz_zero], vr_grid, 0, vr_wgts,
                                            vzeta_grid, 0, vzeta_wgts)
        # n.b. we pass more arguments than might appear to be required here
        # to avoid needing a special integral function definition
        # the 0 integers are the powers by which dzdt and vperp_grid are raised to in the integral
    end
    return velocity_integral
end

"""
"""
function reset_moments_status!(moments)
    if moments.evolve_density == false
        moments.ion.dens_updated .= false
        moments.neutral.dens_updated .= false
    end
    if moments.evolve_upar == false
        moments.ion.upar_updated .= false
        moments.neutral.uz_updated .= false
    end
    if moments.evolve_p == false
        moments.ion.p_updated .= false
        moments.neutral.p_updated .= false
    end
    moments.ion.qpar_updated .= false
    moments.neutral.uzeta_updated .= false
    moments.neutral.ur_updated .= false
    moments.neutral.pzeta_updated .= false
    moments.neutral.pr_updated .= false
    moments.neutral.pz_updated .= false
    moments.neutral.qz_updated .= false
    moments.electron.dens_updated[] = false
    moments.electron.upar_updated[] = false
    if false
        moments.electron.p_updated[] = false
    end
    moments.electron.qpar_updated[] = false
end

end
