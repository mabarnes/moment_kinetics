"""
"""
module input_structs

export advance_info
export time_info
export advection_input
export initial_condition_input
export spatial_initial_condition_input, velocity_initial_condition_input
export ion_species_parameters, neutral_species_parameters
export species_composition
export em_fields_input
export ion_source_data, electron_source_data, neutral_source_data
export collisions_input, reactions, electron_fluid_collisions, krook_collisions_input,
       fkpl_collisions_input, mxwl_diff_collisions_input
export io_input
export pp_input
export geometry_input
export set_defaults_and_check_top_level!, set_defaults_and_check_section!,
       check_sections!, options_to_TOML, Dict_to_NamedTuple

using ..communication
using ..type_definitions: mk_float, mk_int

using MPI
using TOML

"""
`t_error_sum` is included so that a type which might be mk_float or Float128 can be set by
an option but known at compile time when a `time_info` struct is passed as a function
argument.
"""
struct time_info{Terrorsum <: Real, T_debug_output, T_electron, Trkimp, Timpzero}
    n_variables::mk_int
    nstep::mk_int
    end_time::mk_float
    t::MPISharedArray{mk_float,1}
    dt::MPISharedArray{mk_float,1}
    previous_dt::MPISharedArray{mk_float,1}
    next_output_time::MPISharedArray{mk_float,1}
    dt_before_output::MPISharedArray{mk_float,1}
    dt_before_last_fail::MPISharedArray{mk_float,1}
    CFL_prefactor::mk_float
    step_to_moments_output::MPISharedArray{Bool,1}
    step_to_dfns_output::MPISharedArray{Bool,1}
    write_moments_output::MPISharedArray{Bool,1}
    write_dfns_output::MPISharedArray{Bool,1}
    step_counter::Ref{mk_int}
    max_step_count_this_ion_step::Ref{mk_int}
    max_t_increment_this_ion_step::Ref{mk_float}
    moments_output_counter::Ref{mk_int}
    dfns_output_counter::Ref{mk_int}
    failure_counter::Ref{mk_int}
    failure_caused_by::Vector{mk_int}
    limit_caused_by::Vector{mk_int}
    nwrite_moments::mk_int
    nwrite_dfns::mk_int
    moments_output_times::Vector{mk_float}
    dfns_output_times::Vector{mk_float}
    type::String
    rk_coefs::Array{mk_float,2}
    rk_coefs_implicit::Trkimp
    implicit_coefficient_is_zero::Timpzero
    n_rk_stages::mk_int
    rk_order::mk_int
    adaptive::Bool
    low_storage::Bool
    rtol::mk_float
    atol::mk_float
    atol_upar::mk_float
    step_update_prefactor::mk_float
    max_increase_factor::mk_float
    max_increase_factor_near_last_fail::mk_float
    last_fail_proximity_factor::mk_float
    minimum_dt::mk_float
    maximum_dt::mk_float
    implicit_braginskii_conduction::Bool
    implicit_electron_advance::Bool
    implicit_ion_advance::Bool
    implicit_vpa_advection::Bool
    implicit_electron_ppar::Bool
    constraint_forcing_rate::mk_float
    decrease_dt_iteration_threshold::mk_int
    increase_dt_iteration_threshold::mk_int
    cap_factor_ion_dt::mk_float
    write_after_fixed_step_count::Bool
    error_sum_zero::Terrorsum
    split_operators::Bool
    steady_state_residual::Bool
    converged_residual_value::mk_float
    use_manufactured_solns_for_advance::Bool
    stopfile::String
    debug_io::T_debug_output # Currently only used by electrons
    electron::T_electron
end

"""
"""
struct advance_info
    vpa_advection::Bool
    vperp_advection::Bool
    z_advection::Bool
    r_advection::Bool
    neutral_z_advection::Bool
    neutral_r_advection::Bool
    neutral_vz_advection::Bool
    ion_cx_collisions::Bool
    neutral_cx_collisions::Bool
    ion_cx_collisions_1V::Bool
    neutral_cx_collisions_1V::Bool
    ion_ionization_collisions::Bool
    neutral_ionization_collisions::Bool
    ion_ionization_collisions_1V::Bool
    neutral_ionization_collisions_1V::Bool
    krook_collisions_ii::Bool
    mxwl_diff_collisions_ii::Bool
    mxwl_diff_collisions_nn::Bool
    explicit_weakform_fp_collisions::Bool
    external_source::Bool
    ion_numerical_dissipation::Bool
    neutral_numerical_dissipation::Bool
    source_terms::Bool
    continuity::Bool
    force_balance::Bool
    energy::Bool
    electron_energy::Bool
    electron_conduction::Bool
    neutral_external_source::Bool
    neutral_source_terms::Bool
    neutral_continuity::Bool
    neutral_force_balance::Bool
    neutral_energy::Bool
    manufactured_solns_test::Bool
    r_diffusion::Bool #flag to control how r bc is imposed when r diffusion terms are present
    vpa_diffusion::Bool #flag to control how vpa bc is imposed when vpa diffusion terms are present
    vperp_diffusion::Bool #flag to control how vperp bc is imposed when vperp diffusion terms are present
    vz_diffusion::Bool #flag to control how vz bc is imposed when vz diffusion terms are present
end

"""
"""
struct advection_input
    # advection speed option
    option::String
    # constant advection speed to use with the "constant" advection option
    constant_speed::mk_float
    # for option = "oscillating", advection speed is of form
    # speed = constant_speed*(1 + oscillation_amplitude*sinpi(frequency*t))
    frequency::mk_float
    oscillation_amplitude::mk_float
end

"""
"""
@enum electron_physics_type begin
    boltzmann_electron_response 
    boltzmann_electron_response_with_simple_sheath 
    braginskii_fluid
    kinetic_electrons
    kinetic_electrons_with_temperature_equation
end
export electron_physics_type
export boltzmann_electron_response
export boltzmann_electron_response_with_simple_sheath
export braginskii_fluid
export kinetic_electrons
export kinetic_electrons_with_temperature_equation

"""
"""
Base.@kwdef struct spatial_initial_condition_input
    # initialization inputs for one coordinate of a separable distribution function
    initialization_option::String = "gaussian"
    # inputs for "gaussian" initial condition
    width::mk_float = 0.125
    # inputs for "sinusoid" initial condition
    wavenumber::mk_int = 1
    density_amplitude::mk_float = 0.001
    density_phase::mk_float = 0.0
    upar_amplitude::mk_float = 0.0
    upar_phase::mk_float = 0.0
    temperature_amplitude::mk_float = 0.0
    temperature_phase::mk_float = 0.0
    # inputs for "monomial" initial condition
    monomial_degree::mk_int = 2
end

"""
"""
Base.@kwdef struct velocity_initial_condition_input
    # initialization inputs for one coordinate of a separable distribution function
    initialization_option::String = "gaussian"
    # inputs for "gaussian" initial condition
    width::mk_float = 1.0
    # inputs for "sinusoid" initial condition
    wavenumber::mk_int = 1
    density_amplitude::mk_float = 1.0
    density_phase::mk_float = 0.0
    upar_amplitude::mk_float = 0.0
    upar_phase::mk_float = 0.0
    temperature_amplitude::mk_float = 0.0
    temperature_phase::mk_float = 0.0
    # inputs for "monomial" initial condition
    monomial_degree::mk_int = 2
    # inputs for "isotropic-beam", "directed-beam" initial conditions
    v0::mk_float = 1.0
    vth0::mk_float = 1.0
    vpa0::mk_float = 1.0
    vperp0::mk_float = 1.0
end

"""
"""
Base.@kwdef struct ion_species_parameters
    # type is the type of species
    type::String
    # mass/reference mass
    mass::mk_float
    # charge number, absolute w.r.t. proton charge
    zeds::mk_float
    # array containing the initial line-averaged temperature for this species
    initial_temperature::mk_float
    # array containing the initial line-averaged density for this species
    initial_density::mk_float
    # struct containing the initial condition info in z for this species
    z_IC::spatial_initial_condition_input
    # struct containing the initial condition info in r for this species
    r_IC::spatial_initial_condition_input
    # struct containing the initial condition info in vpa for this species
    vpa_IC::velocity_initial_condition_input
end

"""
"""
Base.@kwdef struct neutral_species_parameters
    # type is the type of species
    type::String
    # mass/reference mass
    mass::mk_float
    # array containing the initial line-averaged temperature for this species
    initial_temperature::mk_float
    # array containing the initial line-averaged density for this species
    initial_density::mk_float
    # struct containing the initial condition info in z for this species
    z_IC::spatial_initial_condition_input
    # struct containing the initial condition info in r for this species
    r_IC::spatial_initial_condition_input
    # struct containing the initial condition info in vz for this species
    vz_IC::velocity_initial_condition_input
end

"""
"""
Base.@kwdef struct species_composition
    # n_species = total number of evolved species (including ions, neutrals and electrons)
    # a diagnostic, not an input parameter
    n_species::mk_int
    # n_ion_species is the number of evolved ion species
    n_ion_species::mk_int
    # n_neutral_species is the number of evolved neutral species
    n_neutral_species::mk_int
    # * if electron_physics=boltzmann_electron_response, the electron density is fixed
    #   to be Nₑ*(eϕ/T_e)
    # * if electron_physics=boltzmann_electron_response_with_simple_sheath, the electron
    #   density is fixed to be Nₑ*(eϕ/T_e) and N_e is calculated using a current
    #   condition at the wall
    electron_physics::electron_physics_type
    # if false -- wall bc uses true Knudsen cosine to specify neutral pdf leaving the wall
    # if true -- use a simpler pdf that is easier to integrate
    use_test_neutral_wall_pdf::Bool
    # electron temperature used for Boltzmann response
    T_e::mk_float
    # wall temperature used if 'wall' BC selected for z coordinate; normalised by electron temperature
    T_wall::mk_float
    # wall potential used if electron_physics=boltzmann_electron_response_with_simple_sheath
    phi_wall::mk_float
    # ratio of the neutral particle mass to the ion mass
    mn_over_mi::mk_float
    # ratio of the electron particle mass to the ion mass
    me_over_mi::mk_float
    # The ion flux reaching the wall that is recycled as neutrals is reduced by
    # `recycling_fraction` to account for ions absorbed by the wall.
    recycling_fraction::mk_float
    # gyrokinetic_ions is a flag determining if the ion species is gyrokinetic
    # gyrokinetic_ions = true -> use gyroaveraged fields at fixed guiding centre and moments of the pdf computed at fixed r
    # gyrokinetic_ions = false -> use drift kinetic approximation
    gyrokinetic_ions::Bool
    # array of structs of parameters for each ion species
    ion::Vector{ion_species_parameters}
    # array of structs of parameters for each neutral species
    neutral::Vector{neutral_species_parameters}
end

"""
Settings for electronmagenetic fields
"""
Base.@kwdef struct em_fields_input
    force_Er_zero_at_wall::Bool = false
end

"""
Source profile structs for ions and electrons which allows them to have any number 
of different sources (from wall perhaps, superposition of core sources, etc.). These
sources are then contained in a vector of structs.

Since the ion source must be the same as the electron source in all respects (apart
from possibly a different electron temperature or source strength), the electron
vector of source profile structs will be a kind of mirror of the ion vector of structs. 
"""
Base.@kwdef struct ion_source_data
    # struct containing source data for ions
    # is the source active or not
    active::Bool
    # An overall multiplier for the strength (i.e. 0.0 would turn off the source)
    source_strength::mk_float
    # For use with "energy" option, Krook source (...)
    source_n::mk_float
    # Temperature of source (variation along z can be introduced later)
    source_T::mk_float
    # birth speed for "alphas" option
    source_v0::mk_float
    # birth vpa for "beam" option
    source_vpa0::mk_float
    # birth vperp for "beam" option
    source_vperp0::mk_float
    # strength of sink in "alphas-with-losses" & "beam-with-losses" option
    sink_strength::mk_float
    # thermal speed for sink in "alphas-with-losses" & "beam-with-losses" option
    sink_vth::mk_float
    # profile for source in r ('constant' or 'gaussian' or 'parabolic' etc.) 
    r_profile::String
    # width of source in r (doesn't apply to constant profile)
    r_width::mk_float
    # relative minimum of source in r, acts as the baseline
    r_relative_minimum::mk_float
    # profile for source in z ('constant' or 'gaussian' or 'parabolic' etc.)
    z_profile::String
    # width of source in z (doesn't apply to constant profile)
    z_width::mk_float
    # relative minimum of source in z, acts as the baseline (so you can elevate
    # your gaussian source above 0, for example)
    z_relative_minimum::mk_float
    # velocity profile for the source, Maxwellian would be default, can have beams too 
    source_type::String #"Maxwellian" "energy", "alphas", "alphas-with-losses", "beam", "beam-with-losses"
    # proportional integral controllers - for when you want to find a source profile that matches
    # a target density 
    PI_density_controller_P::mk_float
    PI_density_controller_I::mk_float
    PI_density_target_amplitude::mk_float
    PI_density_target_r_profile::String
    PI_density_target_r_width::mk_float
    PI_density_target_r_relative_minimum::mk_float
    PI_density_target_z_profile::String
    PI_density_target_z_width::mk_float
    PI_density_target_z_relative_minimum::mk_float
    recycling_controller_fraction::mk_float
    # r_amplitude through the r coordinate (in 1D this can just be set to 1.0)
    r_amplitude::Vector{mk_float}
    # z_amplitude through the z coordinate, which will have your gaussian profile,
    # constant profile, parabolic, etc..
    z_amplitude::Vector{mk_float}
    PI_density_target::Union{mk_float, Nothing, MPISharedArray{mk_float,2}}
    PI_controller_amplitude::Union{Nothing, MPISharedArray{mk_float,1}}
    controller_source_profile::Union{Nothing, MPISharedArray{mk_float,2}, Array{mk_float, 2}}
    PI_density_target_ir::Union{mk_int, Nothing}
    PI_density_target_iz::Union{mk_int, Nothing}
    PI_density_target_rank::Union{mk_int, Nothing} #possibly this should have Int64 as well, 
    # in the event that the code is running with mk_int = Int32 but the rank is set to 0::Int64
end

Base.@kwdef struct electron_source_data
    # most of the electron parameters must be the same as for ions, so only 
    # source strength (in the case of an ion energy source) and source Temperature
    # can be different. The other four are set by the ion source data.
    source_strength::mk_float
    source_T::mk_float
    active::Bool
    r_amplitude::Vector{mk_float}
    z_amplitude::Vector{mk_float}
    source_type::String
end

Base.@kwdef struct neutral_source_data
    # struct containing source data for neutrals
    # is the source active or not
    active::Bool
    # An overall multiplier for the strength (i.e. 0.0 would turn off the source)
    source_strength::mk_float
    # For use with "energy" option, Krook source (...)
    source_n::mk_float
    # Temperature of source (variation along z can be introduced later)
    source_T::mk_float
    # birth speed for "alphas" option
    source_v0::mk_float
    # birth vpa for "beam" option
    source_vpa0::mk_float
    # birth vperp for "beam" option
    source_vperp0::mk_float
    # strength of sink in "alphas-with-losses" & "beam-with-losses" option
    sink_strength::mk_float
    # thermal speed for sink in "alphas-with-losses" & "beam-with-losses" option
    sink_vth::mk_float
    # profile for source in r ('constant' or 'gaussian' or 'parabolic' etc.) 
    r_profile::String
    # width of source in r (doesn't apply to constant profile)
    r_width::mk_float
    # relative minimum of source in r, acts as the baseline
    r_relative_minimum::mk_float
    # profile for source in z ('constant' or 'gaussian' or 'parabolic' etc.)
    z_profile::String
    # width of source in z (doesn't apply to constant profile)
    z_width::mk_float
    # relative minimum of source in z, acts as the baseline (so you can elevate
    # your gaussian source above 0, for example)
    z_relative_minimum::mk_float
    # velocity profile for the source, Maxwellian would be default, can have beams too 
    source_type::String #"Maxwellian" "energy", "alphas", "alphas-with-losses", "beam", "beam-with-losses"
    # proportional integral controllers - for when you want to find a source profile that matches
    # a target density 
    PI_density_controller_P::mk_float
    PI_density_controller_I::mk_float
    PI_density_target_amplitude::mk_float
    PI_density_target_r_profile::String
    PI_density_target_r_width::mk_float
    PI_density_target_r_relative_minimum::mk_float
    PI_density_target_z_profile::String
    PI_density_target_z_width::mk_float
    PI_density_target_z_relative_minimum::mk_float
    recycling_controller_fraction::mk_float
    # r_amplitude through the r coordinate (in 1D this can just be set to 1.0)
    r_amplitude::Vector{mk_float}
    # z_amplitude through the z coordinate, which will have your gaussian profile,
    # constant profile, parabolic, etc..
    z_amplitude::Vector{mk_float}
    PI_density_target::Union{mk_float, Nothing, MPISharedArray{mk_float,2}}
    PI_controller_amplitude::Union{Nothing, MPISharedArray{mk_float,1}}
    controller_source_profile::Union{Nothing, MPISharedArray{mk_float,2}, Array{mk_float, 2}}
    PI_density_target_ir::Union{mk_int, Nothing}
    PI_density_target_iz::Union{mk_int, Nothing}
    PI_density_target_rank::Union{mk_int, Nothing} #possibly this should have Int64 as well, 
    # in the event that the code is running with mk_int = Int32 but the rank is set to 0::Int64
end

#Structs set up for the collision operators so far in use. These will each
#be contained in the main collisions_input struct below, as substructs.

Base.@kwdef struct reactions
    charge_exchange_frequency::mk_float = 0.0
    electron_charge_exchange_frequency::mk_float = 0.0
    ionization_frequency::mk_float = 0.0
    electron_ionization_frequency::mk_float = 0.0
    ionization_energy::mk_float = 0.0
end

Base.@kwdef struct electron_fluid_collisions
    nu_ei::mk_float = 0.0
end

Base.@kwdef struct mxwl_diff_collisions_input
    use_maxwell_diffusion::Bool
    # different diffusion coefficients for each species, has units of 
    # frequency * velocity^2. Diffusion coefficients usually denoted D
    D_ii::mk_float
    D_nn::mk_float
    # Setting to switch between different options for Krook collision operator
    diffusion_coefficient_option::String # "reference_parameters" # "manual", 
end

Base.@kwdef struct krook_collisions_input
    use_krook::Bool
    # Ion-ion Coulomb collision rate at the reference density and temperature
    nuii0::mk_float
    # Electron-electron Coulomb collision rate at the reference density and temperature
    nuee0::mk_float
    # Electron-ion Coulomb collision rate at the reference density and temperature
    nuei0::mk_float
    # Setting to switch between different options for Krook collision operator
    frequency_option::String # "reference_parameters" # "manual", 
end

Base.@kwdef struct fkpl_collisions_input
    # option to check if fokker planck frequency should be > 0
    use_fokker_planck::Bool
    # ion-ion self collision frequency (for a species with Z = 1)
    # nu_{ii} = (L/c_{ref}) * gamma_{ref} n_{ref} /(m_s)^2 (c_{ref})^3
    # with gamma_ref = 2 pi e^4 ln \Lambda_{ii} / (4 pi \epsilon_0)^2
    # and ln \Lambda_{ii} the Coulomb logarithm for ion-ion collisions
    nuii::mk_float
    # option to determine if self collisions are used (for physics test)
    self_collisions::Bool
    # option to determine if ad-hoc moment_kinetics-style conserving corrections are used
    use_conserving_corrections::Bool
    # option to determine if cross-collisions against fixed Maxwellians are used
    slowing_down_test::Bool
    # Setting to switch between different options for Fokker-Planck collision frequency input
    frequency_option::String # "manual" # "reference_parameters"
    # options for fixed Maxwellian species in slowing down test operator
    # ion density - electron density determined from quasineutrality
    sd_density::mk_float
    # ion temperature - electron temperature assumed identical
    sd_temp::mk_float
    # ion charge number of fixed Maxwellian species
    sd_q::mk_float
    # ion mass with respect to reference
    sd_mi::mk_float
    # electron mass with respect to reference
    sd_me::mk_float
    # charge number of evolved ion species
    # kept here because charge number different from 1
    # is not supported for other physics features
    Zi::mk_float
end

"""
Collisions input struct to contain all the different collisions substructs and overall 
collision input parameters.
"""
struct collisions_input
    # atomic reaction parameters
    reactions::reactions
    # electron fluid collision parameters
    electron_fluid::electron_fluid_collisions
    # struct of parameters for the Krook operator
    krook::krook_collisions_input
    # struct of parameters for the Fokker-Planck operator
    fkpl::fkpl_collisions_input
    # struct of parameters for the Maxwellian Diffusion operator
    mxwl_diff::mxwl_diff_collisions_input
end

"""
"""
Base.@kwdef struct geometry_input
    # rhostar ion (ref)
    rhostar::mk_float = 0.0 #used to premultiply ExB drift terms
    # magnetic geometry option
    option::String = "constant-helical" # "1D-mirror"
    # pitch ( = Bzed/Bmag if geometry_option == "constant-helical")
    pitch::mk_float = 1.0
    # DeltaB ( = (Bzed(z=L/2) - Bzed(0))/Bref if geometry_option == "1D-mirror")
    DeltaB::mk_float = 0.0
    # constant for testing nonzero Er when nr = 1
    Er_constant::mk_float
    # constant for testing nonzero Ez when nz = 1
    Ez_constant::mk_float
    # constant for testing nonzero dBdz when nz = 1
    dBdz_constant::mk_float
    # constant for testing nonzero dBdr when nr = 1
    dBdr_constant::mk_float
end

@enum binary_format_type hdf5 netcdf
export binary_format_type, hdf5, netcdf

"""
"""
struct pp_input
    # if calculate_frequencies = true, calculate and print the frequency and growth/decay
    # rate of phi, using values at iz = iz0
    calculate_frequencies::Bool
    # if plot_phi0_vs_t = true, create plot of phi(z0) vs time
    plot_phi0_vs_t::Bool
    # if plot_phi_vs_z_t = true, create plot of phi vs z and time
    plot_phi_vs_z_t::Bool
    # if animate_phi_vs_z = true, create animation of phi vs z at different time slices
    animate_phi_vs_z::Bool
    # if plot_dens0_vs_t = true, create plots of species density(z0) vs time
    plot_dens0_vs_t::Bool
    # if plot_upar0_vs_t = true, create plots of species upar(z0) vs time
    plot_upar0_vs_t::Bool
    # if plot_ppar0_vs_t = true, create plots of species ppar(z0) vs time
    plot_ppar0_vs_t::Bool
    # if plot_pperp0_vs_t = true, create plots of species pperp(z0) vs time
    plot_pperp0_vs_t::Bool
    # if plot_vth0_vs_t = true, create plots of species vth(z0) vs time
    plot_vth0_vs_t::Bool
    # if plot_dSdt0_vs_t = true, create plots of species vth(z0) vs time
    plot_dSdt0_vs_t::Bool
    # if plot_qpar0_vs_t = true, create plots of species qpar(z0) vs time
    plot_qpar0_vs_t::Bool
    # if plot_dens_vs_z_t = true, create plot of species density vs z and time
    plot_dens_vs_z_t::Bool
    # if plot_upar_vs_z_t = true, create plot of species parallel flow vs z and time
    plot_upar_vs_z_t::Bool
    # if plot_ppar_vs_z_t = true, create plot of species parallel pressure vs z and time
    plot_ppar_vs_z_t::Bool
    # if plot_Tpar_vs_z_t = true, create plot of species parallel temperature vs z and time
    plot_Tpar_vs_z_t::Bool
    # if plot_qpar_vs_z_t = true, create plot of species parallel heat flux vs z and time
    plot_qpar_vs_z_t::Bool
    # if animate_dens_vs_z = true, create animation of species density vs z at different time slices
    animate_dens_vs_z::Bool
    # if animate_upar_vs_z = true, create animation of species parallel flow vs z at different time slices
    animate_upar_vs_z::Bool
    # if animate_ppar_vs_z = true, create animation of species parallel pressure vs z at different time slices
    animate_ppar_vs_z::Bool
    # if animate_Tpar_vs_z = true, create animation of species parallel temperature vs z at different time slices
    animate_Tpar_vs_z::Bool
    # if animate_vth_vs_z = true, create animation of species thermal speed vs z at different time slices
    animate_vth_vs_z::Bool
    # if animate_qpar_vs_z = true, create animation of species parallel heat flux vs z at different time slices
    animate_qpar_vs_z::Bool
    # if plot_f_unnormalized_vs_vpa_z = true, create plot of f_unnorm(v_parallel_unnorm,z)
    # at it=itime_max
    plot_f_unnormalized_vs_vpa_z::Bool
    # if animate_f_vs_vpa_z = true, create animation of f(z,vpa) at different time slices
    animate_f_vs_vpa_z::Bool
    # if animate_f_unnormalized = true, create animation of
    # f_unnorm(v_parallel_unnorm,z) at different time slices
    animate_f_unnormalized::Bool
    # if animate_f_vs_z_vpa0 = true, create animation of f(z,vpa0) at different time slices
    animate_f_vs_vpa0_z::Bool
    # if animate_f_vs_z0_vpa = true, create animation of f(z0,vpa) at different time slices
    animate_f_vs_vpa_z0::Bool
    # if animate_deltaf_vs_z_vpa = true, create animation of δf(z,vpa) at different time slices
    animate_deltaf_vs_vpa_z::Bool
    # if animate_deltaf_vs_z_vpa0 = true, create animation of δf(z,vpa0) at different time slices
    animate_deltaf_vs_vpa0_z::Bool
    # if animate_deltaf_vs_z0_vpa = true, create animation of δf(z0,vpa) at different time slices
    animate_deltaf_vs_vpa_z0::Bool
    # if animate_f_vs_vpa_r = true, create animation of f(vpa,r) at different time slices
    animate_f_vs_vpa_r::Bool
    # if animate_f_vs_vperp_z = true, create animation of f(vperp,z) at different time slices
    animate_f_vs_vperp_z::Bool
    # if animate_f_vs_vperp_r = true, create animation of f(vperp,r) at different time slices
    animate_f_vs_vperp_r::Bool
    # if animate_f_vs_vperp_vpa = true, create animation of f(vperp,vpa) at different time slices
    animate_f_vs_vperp_vpa::Bool
    # if animate_f_vs_r_z = true, create animation of f(r,z) at different time slices
    animate_f_vs_r_z::Bool
    # if animate_f_vs_vz_z = true, create animation of f(vz,z) at different time slices
    animate_f_vs_vz_z::Bool
    # if animate_f_vs_vr_r = true, create animation of f(vr,r) at different time slices
    animate_f_vs_vr_r::Bool
    # if animate_Er_vs_r_z = true, create animation of Er(r,z) at different time slices
    animate_Er_vs_r_z::Bool
    # if animate_Ez_vs_r_z = true, create animation of Ez(r,z) at different time slices
    animate_Ez_vs_r_z::Bool
    # if animate_phi_vs_r_z = true, create animation of phi(r,z) at different time slices
    animate_phi_vs_r_z::Bool
    # if plot_phi_vs_r0_z = true, plot last timestep phi[z,ir0]
    plot_phi_vs_r0_z::Bool
    # if plot_Ez_vs_r0_z = true, plot last timestep Ez[z,ir0]
    plot_Ez_vs_r0_z::Bool
    # if plot_wall_Ez_vs_r = true, plot last timestep Ez[z_wall,r]
    plot_wall_Ez_vs_r::Bool
    # if plot_Er_vs_r0_z  = true, plot last timestep Er[z,ir0]
    plot_Er_vs_r0_z::Bool
    # if plot_wall_Er_vs_r = true, plot last timestep Er[z_wall,r]
    plot_wall_Er_vs_r::Bool
	# if plot_density_vs_r0_z = true  plot last timestep density[z,ir0]
	plot_density_vs_r0_z::Bool
	# if plot_wall_density_vs_r = true  plot last timestep density[z_wall,r]
	plot_wall_density_vs_r::Bool
    # if plot_density_vs_r_z = true plot density vs r z at last timestep
	plot_density_vs_r_z::Bool
	# if animate_density_vs_r_z = true animate density vs r z
	animate_density_vs_r_z::Bool
	# if plot_parallel_flow_vs_r0_z = true  plot last timestep parallel_flow[z,ir0]
	plot_parallel_flow_vs_r0_z::Bool
	# if plot_wall_parallel_flow_vs_r = true  plot last timestep parallel_flow[z_wall,r]
	plot_wall_parallel_flow_vs_r::Bool
    # if plot_parallel_flow_vs_r_z = true plot parallel_flow vs r z at last timestep
	plot_parallel_flow_vs_r_z::Bool
	# if animate_parallel_flow_vs_r_z = true animate parallel_flow vs r z
	animate_parallel_flow_vs_r_z::Bool
	# if plot_parallel_pressure_vs_r0_z = true  plot last timestep parallel_pressure[z,ir0]
	plot_parallel_pressure_vs_r0_z::Bool
	# if plot_perpendicular_pressure_vs_r0_z = true  plot last timestep perpendicular_pressure[z,ir0]
	plot_perpendicular_pressure_vs_r0_z::Bool
	# if plot_wall_parallel_pressure_vs_r = true  plot last timestep parallel_pressure[z_wall,r]
	plot_wall_parallel_pressure_vs_r::Bool
    # if plot_parallel_pressure_vs_r_z = true plot parallel_pressure vs r z at last timestep 
	plot_parallel_pressure_vs_r_z::Bool
    # if animate_parallel_pressure_vs_r_z = true animate parallel_pressure vs r z
	animate_parallel_pressure_vs_r_z::Bool
    # if plot_parallel_temperature_vs_r0_z = true  plot last timestep parallel_temperature[z,ir0]
    plot_parallel_temperature_vs_r0_z::Bool
    # if plot_wall_parallel_temperature_vs_r = true  plot last timestep parallel_temperature[z_wall,r]
    plot_wall_parallel_temperature_vs_r::Bool
    # if plot_parallel_temperature_vs_r_z = true plot parallel_temperature vs r z at last timestep
    plot_parallel_temperature_vs_r_z::Bool
    # if animate_parallel_temperature_vs_r_z = true animate parallel_temperature vs r z
    animate_parallel_temperature_vs_r_z::Bool
    # if plot_chodura_integral = true then plots of the in-simulation Chodura integrals are generated
    plot_chodura_integral::Bool
    # if plot_wall_pdf = true then plot the ion distribution (vpa,vperp,z,r) in the element nearest the wall at the last timestep 
    plot_wall_pdf::Bool
    # run analysis for a 2D (in R-Z) linear mode?
    instability2D::Bool
    # animations of moments will use one in every nwrite_movie data slices
    nwrite_movie::mk_int
    # itime_min is the minimum time index at which to start animations of the moments
    itime_min::mk_int
    # itime_max is the final time index at which to end animations of the moments
    # if itime_max < 0, the value used will be the total number of time slices
    itime_max::mk_int
    # Only load every itime_skip'th time-point when loading data, to save memory
    itime_skip::mk_int
    # animations of pdfs will use one in every nwrite_movie data slices
    nwrite_movie_pdfs::mk_int
    # itime_min_pdfs is the minimum time index at which to start animations of the pdfs
    itime_min_pdfs::mk_int
    # itime_max_pdfs is the final time index at which to end animations of the pdfs
    # if itime_max < 0, the value used will be the total number of time slices
    itime_max_pdfs::mk_int
    # Only load every itime_skip_pdfs'th time-point when loading pdf data, to save memory
    itime_skip_pdfs::mk_int
    # ivpa0 is the ivpa index used when plotting data at a single vpa location
    ivpa0::mk_int
    # ivperp0 is the ivperp index used when plotting data at a single vperp location
    ivperp0::mk_int
    # iz0 is the iz index used when plotting data at a single z location
    iz0::mk_int
    # ir0 is the ir index used when plotting data at a single r location
    ir0::mk_int
    # ivz0 is the ivz index used when plotting data at a single vz location
    ivz0::mk_int
    # ivr0 is the ivr index used when plotting data at a single vr location
    ivr0::mk_int
    # ivzeta0 is the ivzeta index used when plotting data at a single vzeta location
    ivzeta0::mk_int
    # Calculate and plot the 'Chodura criterion' at the wall boundaries vs t at fixed r
    diagnostics_chodura_t::Bool
    # Calculate and plot the 'Chodura criterion' at the wall boundaries vs r at fixed t
    diagnostics_chodura_r::Bool
end

import Base: get
"""
Utility method for converting a string to an Enum when getting from a Dict, based on the
type of the default value
"""
function get(d::Dict, key, default::Enum)
    val_maybe_string = get(d, key, nothing)
    if val_maybe_string == nothing
        return default
    elseif isa(val_maybe_string, Enum)
        return val_maybe_string
    # instances(typeof(default)) gets the possible values of the Enum. Then convert to
    # Symbol, then to String.
    elseif val_maybe_string ∈ Tuple(split(s, ".")[end] for s ∈ string.(instances(typeof(default))))
        return eval(Symbol(val_maybe_string))
    else
        error("Expected a $(typeof(default)), but '$val_maybe_string' is not in "
              * "$(instances(typeof(default)))")
    end
end

"""
Set the defaults for options in the top level of the input, and check that there are not
any unexpected options (i.e. options that have no default).

Modifies the options[section_name]::Dict by adding defaults for any values that are not
already present.

Ignores any sections, as these will be checked separately.
"""
function set_defaults_and_check_top_level!(options::AbstractDict; kwargs...)
    DictType = typeof(options)

    # Check for any unexpected values in the options - all options that are set should be
    # present in the kwargs of this function call
    options_keys_symbols = keys(kwargs)
    options_keys = (String(k) for k ∈ options_keys_symbols)
    for (key, value) in options
        # Ignore any ssections when checking
        if !(isa(value, AbstractDict) || key ∈ options_keys)
            error("Unexpected option '$key=$value' in top-level options")
        end
    end

    # Set default values if a key was not set explicitly
    explicit_keys = keys(options)
    for (key_sym, value) ∈ kwargs
        key = String(key_sym)
        if !(key ∈ explicit_keys)
            options[key] = value
        end
    end

    return options
end

function _get_section_and_check_option_names(options, section_name, section_keys)

    DictType = typeof(options)

    if !(section_name ∈ keys(options))
        # If section is not present, create it
        options[section_name] = DictType()
    end

    if !isa(options[section_name], AbstractDict)
        error("Expected '$section_name' to be a section in the input file, but it has a "
              * "value '$(options[section_name])'")
    end

    section = options[section_name]

    # Check for any unexpected values in the section - all options that are set should be
    # present in the kwargs of this function call
    unexpected_options = String[]
    for (key, value) in section
        if !(key ∈ section_keys)
            push!(unexpected_options, key)
        end
    end
    if length(unexpected_options) > 0
        error("Unexpected options $(join(("$k=$(section[k])" for k ∈ unexpected_options), ", ", ", and ")) in section '$section_name'")
    end

    return section
end

function _check_for_nothing(section, section_name)
    nothing_defaults = Tuple((k,v) for (k,v) ∈ section if v === nothing)
    if length(nothing_defaults) > 0
        error("`nothing` cannot be used as a default value, because it cannot be written "
              * "to TOML or HDF5. In section [$section_name], found "
              * "$(join(("$k=$v" for (k,v) ∈ nothing_defaults), ", ", ", and ")).")
    end
end

function _store_section_name_for_check!(options, section_name)
    # Record the defined section_name in a temporary, private subsection of `options`, so
    # we can use it to check the existing sections later.
    if !(_section_check_store_name ∈ keys(options))
        # If section is not present, create it
        options[_section_check_store_name] = String[]
    end
    push!(options[_section_check_store_name], section_name)

    return nothing
end

const _section_check_store_name = "_section_check_store"

"""
Set the defaults for options in a section, and check that there are not any unexpected
options (i.e. options that have no default).

Modifies the options[section_name]::Dict by adding defaults for any values that are not
already present.
"""
function set_defaults_and_check_section!(options::AbstractDict, section_name;
                                         kwargs...)

    section_keys_symbols = keys(kwargs)
    section_keys = (String(k) for k ∈ section_keys_symbols)
    section = _get_section_and_check_option_names(options, section_name, section_keys)

    # Set default values if a key was not set explicitly
    for (key_sym, default_value) ∈ kwargs
        key = String(key_sym)
        # Use `Base.get()` here to take advantage of our `Enum`-handling method of
        # `Base.get()` defined above.
        section[key] = get(section, key, default_value)
    end

    _check_for_nothing(section, section_name)

    _store_section_name_for_check!(options, section_name)

    return section
end

"""
    set_defaults_and_check_section!(options::AbstractDict, struct_type::Type,
                                    name::Union{String,Nothing}=nothing)

Alternative form to be used when the options should be stored in a struct of type
`struct_type` rather than a `NamedTuple`. `struct_type` must be defined using `@kwdef`.

The returned instance of `struct_type` is immutable, so if you need to modify the settings
- e.g. to apply some logic to set defaults depending on other settings/parameters - then
you should use the 'standard' version of [`set_defaults_and_check_section!`](@ref) that
returns a `Dict` that can be modified, and then use that `Dict` to initialise the
`struct_type`.

The name of the section in the options that will be read defaults to the name of
`struct_type`, but can be set using the `section_name` argument.

Returns an instance of `struct_type`.
"""
function set_defaults_and_check_section!(options::AbstractDict, struct_type::Type,
                                         section_name::Union{String,Nothing}=nothing)

    if section_name === nothing
        section_name = String(nameof(struct_type))
    end
    section_keys = (String(key) for key ∈ fieldnames(struct_type))
    section = _get_section_and_check_option_names(options, section_name, section_keys)

    # Pass the settings in `section` as kwargs to the constructor for `struct_type`.
    # `struct_type` is an `@kwdef` struct, so this constructor takes care of the default
    # values.
    settings_instance = struct_type(; (Symbol(k) => v for (k,v) ∈ pairs(section))...)

    # Save the settings with defaults applied back into `options` so they can be stored in
    # the output file.
    for k ∈ fieldnames(struct_type)
        section[String(k)] = getfield(settings_instance, k)
    end

    _check_for_nothing(section, section_name)

    _store_section_name_for_check!(options, section_name)

    return settings_instance
end

"""
    check_sections!(options::AbstractDict)

Check that there are no unexpected sections in `options`. The 'expected sections' are the
ones that were defined with [`set_defaults_and_check_section!`](@ref).
"""
function check_sections!(options::AbstractDict; check_no_top_level_options=true)

    expected_section_names = pop!(options, _section_check_store_name)

    unexpected_section_names = String[]
    unexpected_top_level_options = String[]
    for (k,v) ∈ pairs(options)
        if isa(v, AbstractDict)
            if k ∉ expected_section_names
                push!(unexpected_section_names, k)
            end
        elseif check_no_top_level_options
            push!(unexpected_top_level_options, k)
        end
    end

    if length(unexpected_section_names) > 0 && length(unexpected_top_level_options) > 0
        error("Input had unexpected sections $unexpected_section_names, and unexpected "
              * "options in the top level $unexpected_top_level_options")
    elseif length(unexpected_section_names) > 0
        error("Input had unexpected sections $unexpected_section_names.")
    elseif length(unexpected_top_level_options) > 0
        error("Input had unexpected options in the top level "
              * "$unexpected_top_level_options")
    end

    return nothing
end

"""
Convert a Dict whose keys are String or Symbol to a NamedTuple

Useful as NamedTuple is immutable, so option values cannot be accidentally changed.
"""
function Dict_to_NamedTuple(d)
    return NamedTuple(Symbol(k)=>v for (k,v) ∈ d)
end

"""
    options_to_toml(io::IO [=stdout], data::AbstractDict; sorted=false, by=identity)

Convert `moment_kinetics` 'options' (in the form of a `Dict`) to TOML format.

This function is defined so that we can handle some extra types, for example `Enum`.

For descriptions of the arguments, see `TOML.print`.
"""
function options_to_TOML(args...; kwargs...)
    function handle_extra_types(x)
        if isa(x, Enum)
            return string(x)
        else
            error("Unhandled type $(typeof(x)) for x=$x")
        end
    end

    return TOML.print(handle_extra_types, args...; kwargs...)
end

end
