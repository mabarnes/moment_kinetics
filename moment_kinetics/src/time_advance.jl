"""
"""
module time_advance

export setup_time_advance!
export time_advance!
export setup_dummy_and_buffer_arrays
# functional testing
export setup_runge_kutta_coefficients

using MPI
using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication
using ..communication: _block_synchronize
using ..debugging
using ..file_io: write_data_to_ascii, write_moments_data_to_binary, write_dfns_data_to_binary, debug_dump
using ..looping
using ..moment_kinetics_structs: scratch_pdf
using ..velocity_moments: update_moments!, update_moments_neutral!, reset_moments_status!
using ..velocity_moments: update_density!, update_upar!, update_ppar!, update_pperp!, update_qpar!, update_vth!
using ..velocity_moments: update_neutral_density!, update_neutral_qz!
using ..velocity_moments: update_neutral_uzeta!, update_neutral_uz!, update_neutral_ur!
using ..velocity_moments: update_neutral_pzeta!, update_neutral_pz!, update_neutral_pr!
using ..velocity_moments: calculate_moment_derivatives!, calculate_moment_derivatives_neutral!
using ..velocity_moments: update_chodura!
using ..velocity_grid_transforms: vzvrvzeta_to_vpavperp!, vpavperp_to_vzvrvzeta!
using ..initial_conditions: enforce_boundary_conditions!
using ..initial_conditions: enforce_neutral_boundary_conditions!
using ..input_structs: advance_info, time_info
using ..moment_constraints: hard_force_moment_constraints!,
                            hard_force_moment_constraints_neutral!
using ..advection: setup_advection
using ..z_advection: update_speed_z!, z_advection!
using ..r_advection: update_speed_r!, r_advection!
using ..neutral_r_advection: update_speed_neutral_r!, neutral_advection_r!
using ..neutral_z_advection: update_speed_neutral_z!, neutral_advection_z!
using ..neutral_vz_advection: update_speed_neutral_vz!, neutral_advection_vz!
using ..vperp_advection: update_speed_vperp!, vperp_advection!
using ..vpa_advection: update_speed_vpa!, vpa_advection!
using ..charge_exchange: charge_exchange_collisions_1V!, charge_exchange_collisions_3V!
using ..ionization: ionization_collisions_1V!, ionization_collisions_3V!, constant_ionization_source!
using ..krook_collisions: krook_collisions!
using ..external_sources
using ..numerical_dissipation: vpa_boundary_buffer_decay!,
                               vpa_boundary_buffer_diffusion!, vpa_dissipation!,
                               z_dissipation!, r_dissipation!, vperp_dissipation!,
                               vz_dissipation_neutral!, z_dissipation_neutral!,
                               r_dissipation_neutral!,
                               vpa_boundary_force_decreasing!, force_minimum_pdf_value!,
                               force_minimum_pdf_value_neutral!
using ..source_terms: source_terms!, source_terms_neutral!, source_terms_manufactured!
using ..continuity: continuity_equation!, neutral_continuity_equation!
using ..force_balance: force_balance!, neutral_force_balance!
using ..energy_equation: energy_equation!, neutral_energy_equation!
using ..em_fields: setup_em_fields, update_phi!
using ..fokker_planck: init_fokker_planck_collisions_weak_form, explicit_fokker_planck_collisions_weak_form!
using ..manufactured_solns: manufactured_sources
using ..advection: advection_info
using ..utils: to_minutes
@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

using Dates
using ..analysis: steady_state_residuals
#using ..post_processing: draw_v_parallel_zero!

struct scratch_dummy_arrays
    dummy_s::Array{mk_float,1}
    dummy_sr::Array{mk_float,2}
    dummy_vpavperp::Array{mk_float,2}
    dummy_zrs::MPISharedArray{mk_float,3}
    dummy_zrsn::MPISharedArray{mk_float,3}

    #buffer arrays for MPI 
    buffer_z_1::MPISharedArray{mk_float,1}
    buffer_z_2::MPISharedArray{mk_float,1}
    buffer_z_3::MPISharedArray{mk_float,1}
    buffer_z_4::MPISharedArray{mk_float,1}
    buffer_r_1::MPISharedArray{mk_float,1}
    buffer_r_2::MPISharedArray{mk_float,1}
    buffer_r_3::MPISharedArray{mk_float,1}
    buffer_r_4::MPISharedArray{mk_float,1}
    
    buffer_zs_1::MPISharedArray{mk_float,2}
    buffer_zs_2::MPISharedArray{mk_float,2}
    buffer_zs_3::MPISharedArray{mk_float,2}
    buffer_zs_4::MPISharedArray{mk_float,2}
    buffer_zsn_1::MPISharedArray{mk_float,2}
    buffer_zsn_2::MPISharedArray{mk_float,2}
    buffer_zsn_3::MPISharedArray{mk_float,2}
    buffer_zsn_4::MPISharedArray{mk_float,2}

    buffer_rs_1::MPISharedArray{mk_float,2}
    buffer_rs_2::MPISharedArray{mk_float,2}
    buffer_rs_3::MPISharedArray{mk_float,2}
    buffer_rs_4::MPISharedArray{mk_float,2}
    buffer_rs_5::MPISharedArray{mk_float,2}
    buffer_rs_6::MPISharedArray{mk_float,2}
    buffer_rsn_1::MPISharedArray{mk_float,2}
    buffer_rsn_2::MPISharedArray{mk_float,2}
    buffer_rsn_3::MPISharedArray{mk_float,2}
    buffer_rsn_4::MPISharedArray{mk_float,2}
    buffer_rsn_5::MPISharedArray{mk_float,2}
    buffer_rsn_6::MPISharedArray{mk_float,2}

    buffer_zrs_1::MPISharedArray{mk_float,3}
    buffer_zrs_2::MPISharedArray{mk_float,3}
    buffer_zrs_3::MPISharedArray{mk_float,3}
    
    buffer_vpavperpzs_1::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_2::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_3::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_4::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_5::MPISharedArray{mk_float,4}
    buffer_vpavperpzs_6::MPISharedArray{mk_float,4}

    buffer_vpavperprs_1::MPISharedArray{mk_float,4}
    buffer_vpavperprs_2::MPISharedArray{mk_float,4}
    buffer_vpavperprs_3::MPISharedArray{mk_float,4}
    buffer_vpavperprs_4::MPISharedArray{mk_float,4}
    buffer_vpavperprs_5::MPISharedArray{mk_float,4}
    buffer_vpavperprs_6::MPISharedArray{mk_float,4}

    # buffer to hold derivative after MPI communicates
    # needs to be shared memory
    buffer_vpavperpzrs_1::MPISharedArray{mk_float,5}
    buffer_vpavperpzrs_2::MPISharedArray{mk_float,5}
    
    buffer_vzvrvzetazsn_1::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_2::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_3::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_4::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_5::MPISharedArray{mk_float,5}
    buffer_vzvrvzetazsn_6::MPISharedArray{mk_float,5}

    buffer_vzvrvzetarsn_1::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_2::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_3::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_4::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_5::MPISharedArray{mk_float,5}
    buffer_vzvrvzetarsn_6::MPISharedArray{mk_float,5}

    # buffer to hold derivative after MPI communicates
    # needs to be shared memory
    buffer_vzvrvzetazrsn_1::MPISharedArray{mk_float,6}
    buffer_vzvrvzetazrsn_2::MPISharedArray{mk_float,6}
    
    buffer_vpavperp_1::MPISharedArray{mk_float,2}
    buffer_vpavperp_2::MPISharedArray{mk_float,2}
    buffer_vpavperp_3::MPISharedArray{mk_float,2}

end 

struct advect_object_struct
    vpa_advect::Vector{advection_info{4,5}}
    vperp_advect::Vector{advection_info{4,5}}
    z_advect::Vector{advection_info{4,5}}
    r_advect::Vector{advection_info{4,5}}
    neutral_z_advect::Vector{advection_info{5,6}}
    neutral_r_advect::Vector{advection_info{5,6}}
    neutral_vz_advect::Vector{advection_info{5,6}}
end

# consider changing code structure so that
# we can avoid arbitrary types below?
struct spectral_object_struct{Tvz,Tvr,Tvzeta,Tvpa,Tvperp,Tz,Tr}
    vz_spectral::Tvz
    vr_spectral::Tvr
    vzeta_spectral::Tvzeta
    vpa_spectral::Tvpa
    vperp_spectral::Tvperp
    z_spectral::Tz
    r_spectral::Tr
end

"""
create arrays and do other work needed to setup
the main time advance loop.
this includes creating and populating structs
for Chebyshev transforms, velocity space moments,
EM fields, and advection terms
"""
function setup_time_advance!(pdf, vz, vr, vzeta, vpa, vperp, z, r, vz_spectral,
                             vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral,
                             z_spectral, r_spectral, composition, drive_input, moments,
                             t_params, collisions, species, geometry,
                             boundary_distributions, external_source_settings,
                             num_diss_params, manufactured_solns_input, restarting)
    # define some local variables for convenience/tidiness
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    # create array containing coefficients needed for the Runge Kutta time advance
    rk_coefs = setup_runge_kutta_coefficients(t_params.n_rk_stages)
    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    advance = setup_advance_flags(moments, composition, t_params, collisions,
                                  external_source_settings, num_diss_params,
                                  manufactured_solns_input, rk_coefs, r, z, vperp, vpa,
                                  vzeta, vr, vz)

    begin_serial_region()

    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf.charged.norm, pdf.neutral.norm, t_params.n_rk_stages)
    # setup dummy arrays & buffer arrays for z r MPI
    n_neutral_species_alloc = max(1,composition.n_neutral_species)
    scratch_dummy = setup_dummy_and_buffer_arrays(r.n,z.n,vpa.n,vperp.n,vz.n,vr.n,vzeta.n,
                                   composition.n_ion_species,n_neutral_species_alloc)
    # create arrays for Fokker-Planck collisions 
    if advance.explicit_weakform_fp_collisions
        fp_arrays = init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; precompute_weights=true)
    else
        fp_arrays = nothing
    end
    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, r.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency, drive_input.force_Er_zero_at_wall)
    # initialize the electrostatic potential
    begin_serial_region()
    update_phi!(fields, scratch[1], z, r, composition, z_spectral, r_spectral, scratch_dummy)
    @serial_region begin
        # save the initial phi(z) for possible use later (e.g., if forcing phi)
        fields.phi0 .= fields.phi
    end

    # Preliminary calculation of moment derivatives, to be used for initial version of
    # 'speed' in advect objects, which are needed for boundary conditions on the
    # distribution function which is then used to (possibly) re-calculate the moments
    # after which the initial values of moment derivatives are re-calculated.
    calculate_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, num_diss_params)
    calculate_moment_derivatives_neutral!(moments, scratch[1], scratch_dummy, z,
                                          z_spectral, num_diss_params)

    ##
    # Charged particle advection only
    ##

    # create structure r_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in r
    begin_serial_region()
    r_advect = setup_advection(n_ion_species, r, vpa, vperp, z)
    if r.n > 1
        # initialise the r advection speed
        begin_s_z_vperp_vpa_region()
        @loop_s is begin
            @views update_speed_r!(r_advect[is], moments.charged.upar[:,:,is],
                                   moments.charged.vth[:,:,is], fields, moments.evolve_upar,
                                   moments.evolve_ppar, vpa, vperp, z, r, geometry)
        end
        # enforce prescribed boundary condition in r on the distribution function f
    end

    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in z
    begin_serial_region()
    z_advect = setup_advection(n_ion_species, z, vpa, vperp, r)
    if z.n > 1
        # initialise the z advection speed
        begin_s_r_vperp_vpa_region()
        @loop_s is begin
            @views update_speed_z!(z_advect[is], moments.charged.upar[:,:,is],
                                   moments.charged.vth[:,:,is], moments.evolve_upar,
                                   moments.evolve_ppar, fields, vpa, vperp, z, r, 0.0,
                                   geometry)
        end
    end
    begin_serial_region()

    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_advect = setup_advection(n_ion_species, vpa, vperp, z, r)
    # initialise the vpa advection speed
    begin_s_r_z_vperp_region()
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, vperp, z, r,
                      composition, collisions, external_source_settings.ion, 0.0,
                      geometry)

    # create structure vperp_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vperp
    begin_serial_region()
    vperp_advect = setup_advection(n_ion_species, vperp, vpa, z, r)
    # initialise the vperp advection speed
    if vperp.n > 1
        begin_serial_region()
        @serial_region begin
            for is ∈ 1:n_ion_species
                @views update_speed_vperp!(vperp_advect[is], vpa, vperp, z, r)
            end
        end
    end
    
    ##
    # Neutral particle advection
    ##

    # create structure neutral_r_advect for neutral particle advection
    begin_serial_region()
    neutral_r_advect = setup_advection(n_neutral_species_alloc, r, vz, vr, vzeta, z)
    if n_neutral_species > 0 && r.n > 1
        # initialise the r advection speed
        begin_sn_z_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_r!(neutral_r_advect[isn], r, z, vzeta, vr, vz)
        end
    end

    # create structure neutral_z_advect for neutral particle advection
    begin_serial_region()
    neutral_z_advect = setup_advection(n_neutral_species_alloc, z, vz, vr, vzeta, r)
    if n_neutral_species > 0 && z.n > 1
        # initialise the z advection speed
        begin_sn_r_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_z!(neutral_z_advect[isn],
                                           moments.neutral.uz[:,:,isn],
                                           moments.neutral.vth[:,:,isn],
                                           moments.evolve_upar, moments.evolve_ppar, vz,
                                           vr, vzeta, z, r, 0.0)
        end
    end

    # create structure neutral_vz_advect for neutral particle advection
    begin_serial_region()
    neutral_vz_advect = setup_advection(n_neutral_species_alloc, vz, vr, vzeta, z, r)
    if n_neutral_species > 0
        # initialise the z advection speed
        @serial_region begin
            # Initialise the vz 'advection speed' in case it does not need updating. It
            # may still be used to decide which boundary is 'incoming' in the vz boundary
            # condition.
            @loop_sn isn begin
                neutral_vz_advect[isn].speed .= 0.0
            end
        end
        begin_sn_r_z_vzeta_vr_region()
        @views update_speed_neutral_vz!(neutral_vz_advect, fields, scratch[1], moments,
                                        vz, vr, vzeta, z, r, composition, collisions,
                                        external_source_settings.neutral)
    end

    ##
    # construct advect & spectral objects to compactify arguments
    ##

    advect_objects = advect_object_struct(vpa_advect, vperp_advect, z_advect, r_advect, neutral_z_advect, neutral_r_advect, neutral_vz_advect)
    spectral_objects = spectral_object_struct(vz_spectral, vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral, z_spectral, r_spectral)
    if(advance.manufactured_solns_test)
        manufactured_source_list = manufactured_sources(manufactured_solns_input, r, z,
                                                        vperp, vpa, vzeta, vr, vz,
                                                        composition, geometry, collisions,
                                                        num_diss_params, species)
    else
        manufactured_source_list = false # dummy Bool to be passed as argument instead of list
    end

    if !restarting
        begin_serial_region()
        # ensure initial pdf has no negative values
        force_minimum_pdf_value!(pdf.charged.norm, num_diss_params)
        force_minimum_pdf_value_neutral!(pdf.neutral.norm, num_diss_params)
        # enforce boundary conditions and moment constraints to ensure a consistent initial
        # condition
        enforce_boundary_conditions!(
            pdf.charged.norm, boundary_distributions.pdf_rboundary_charged,
            moments.charged.dens, moments.charged.upar, moments.charged.ppar, moments,
            vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_spectral, vperp_spectral,
            vpa_advect, z_advect, r_advect,
            composition, scratch_dummy, advance.r_diffusion, advance.vpa_diffusion)
        # Ensure normalised pdf exactly obeys integral constraints if evolving moments
        begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
            @views hard_force_moment_constraints!(pdf.charged.norm[:,:,iz,ir,is], moments,
                                                  vpa)
        end
        # update moments in case they were affected by applying boundary conditions or
        # constraints to the pdf
        reset_moments_status!(moments)
        update_moments!(moments, pdf.charged.norm, vpa, vperp, z, r, composition)
        # update the Chodura diagnostic -- note that the pdf should be the unnormalised one
        # so this will break for the split moments cases
        update_chodura!(moments,pdf.charged.norm,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
        # enforce boundary conditions in r and z on the neutral particle distribution function
        if n_neutral_species > 0
            # Note, so far vr and vzeta do not need advect objects, so pass `nothing` for
            # those as a placeholder
            enforce_neutral_boundary_conditions!(
                pdf.neutral.norm, pdf.charged.norm, boundary_distributions,
                moments.neutral.dens, moments.neutral.uz, moments.neutral.pz, moments,
                moments.charged.dens, moments.charged.upar, fields.Er, vzeta_spectral,
                vr_spectral, vz_spectral, neutral_r_advect, neutral_z_advect, nothing,
                nothing, neutral_vz_advect, r, z, vzeta, vr, vz, composition, geometry,
                scratch_dummy, advance.r_diffusion, advance.vz_diffusion)
            begin_sn_r_z_region()
            @loop_sn_r_z isn ir iz begin
                @views hard_force_moment_constraints_neutral!(
                    pdf.neutral.norm[:,:,:,iz,ir,isn], moments, vz)
            end
            update_moments_neutral!(moments, pdf.neutral.norm, vz, vr, vzeta, z, r,
                                    composition)
        end

        # update scratch arrays in case they were affected by applying boundary conditions
        # or constraints to the pdf
        begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
            scratch[1].pdf[:,:,iz,ir,is] .= pdf.charged.norm[:,:,iz,ir,is]
            scratch[1].density[iz,ir,is] = moments.charged.dens[iz,ir,is]
            scratch[1].upar[iz,ir,is] = moments.charged.upar[iz,ir,is]
            scratch[1].ppar[iz,ir,is] = moments.charged.ppar[iz,ir,is]
        end

        begin_sn_r_z_region(no_synchronize=true)
        @loop_sn_r_z isn ir iz begin
            scratch[1].pdf_neutral[:,:,:,iz,ir,isn] .= pdf.neutral.norm[:,:,:,iz,ir,isn]
            scratch[1].density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            scratch[1].uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            scratch[1].pz_neutral[iz,ir,isn] = moments.neutral.pz[iz,ir,isn]
        end
    end

    update_phi!(fields, scratch[1], z, r, composition, z_spectral, r_spectral,
                scratch_dummy)
    calculate_moment_derivatives!(moments, scratch[1], scratch_dummy, z, z_spectral, num_diss_params)
    calculate_moment_derivatives_neutral!(moments, scratch[1], scratch_dummy, z,
                                          z_spectral, num_diss_params)

    # Ensure all processes are synchronized at the end of the setup
    _block_synchronize()

    return moments, fields, spectral_objects, advect_objects,
    scratch, advance, fp_arrays, scratch_dummy, manufactured_source_list
end

"""
create the 'advance_info' struct to be used in later Euler advance to
indicate which parts of the equations are to be advanced concurrently.
if no splitting of operators, all terms advanced concurrently;
else, will advance one term at a time.
"""
function setup_advance_flags(moments, composition, t_params, collisions,
                             external_source_settings, num_diss_params,
                             manufactured_solns_input, rk_coefs, r, z, vperp, vpa, vzeta,
                             vr, vz)
    # default is not to concurrently advance different operators
    advance_vpa_advection = false
    advance_z_advection = false
    advance_r_advection = false
    advance_cx_1V = false
    advance_cx = false
    advance_ionization = false
    advance_ionization_1V = false
    advance_ionization_source = false
    advance_krook_collisions = false
    advance_external_source = false
    advance_numerical_dissipation = false
    advance_sources = false
    advance_continuity = false
    advance_force_balance = false
    advance_energy = false
    advance_neutral_z_advection = false
    advance_neutral_r_advection = false
    advance_neutral_vz_advection = false
    advance_neutral_external_source = false
    advance_neutral_sources = false
    advance_neutral_continuity = false
    advance_neutral_force_balance = false
    advance_neutral_energy = false
    r_diffusion = false
    vpa_diffusion = false
    vz_diffusion = false
    explicit_weakform_fp_collisions = false
    # all advance flags remain false if using operator-splitting
    # otherwise, check to see if the flags need to be set to true
    if !t_params.split_operators
        # default for non-split operators is to include both vpa and z advection together
        advance_vpa_advection = true && vpa.n > 1 && z.n > 1
        advance_z_advection = true && z.n > 1
        advance_r_advection = true && r.n > 1
        if collisions.nuii > 0.0 && vperp.n > 1
            explicit_weakform_fp_collisions = true
        else
            explicit_weakform_fp_collisions = false    
        end
        # if neutrals present, check to see if different ion-neutral
        # collisions are enabled
        if composition.n_neutral_species > 0
            advance_neutral_z_advection = true
            if r.n > 1
                advance_neutral_r_advection = true
            end
            if moments.evolve_upar || moments.evolve_ppar
                advance_neutral_vz_advection = true
            end
            # if charge exchange collision frequency non-zero,
            # account for charge exchange collisions
            if abs(collisions.charge_exchange) > 0.0
                if vz.n == vpa.n && vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                    advance_cx_1V = true
                elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                    advance_cx = true
                else
                    error("If any perpendicular velocity has length>1 they all must. "
                          * "If all perpendicular velocities have length=1, then vpa and "
                          * "vz should be the same.\n"
                          * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                          * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
                end
            end
            # if ionization collision frequency non-zero,
            # account for ionization collisions
            if abs(collisions.ionization) > 0.0
                if vz.n == vpa.n && vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                    advance_ionization_1V = true
                elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                    advance_ionization = true
                else
                    error("If any perpendicular velocity has length>1 they all must. "
                          * "If all perpendicular velocities have length=1, then vpa and "
                          * "vz should be the same.\n"
                          * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n), "
                          * "vpa.n=$(vpa.n), vz.n=$(vz.n)")
                end
            end
        end
        # exception for the case where ions are evolved alone but sourced by ionization
        if collisions.ionization > 0.0 && collisions.constant_ionization_rate
            advance_ionization_source = true
        end
        if collisions.krook_collision_frequency_prefactor > 0.0
            advance_krook_collisions = true
        end
        advance_external_source = external_source_settings.ion.active
        advance_neutral_external_source = external_source_settings.neutral.active
        advance_numerical_dissipation = true
        # if evolving the density, must advance the continuity equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_density
            advance_sources = true
            advance_continuity = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_continuity = true
            end
        end
        # if evolving the parallel flow, must advance the force balance equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_upar
            advance_sources = true
            advance_force_balance = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_force_balance = true
            end
        end
        # if evolving the parallel pressure, must advance the energy equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_ppar
            advance_sources = true
            advance_energy = true
            if composition.n_neutral_species > 0
                advance_neutral_sources = true
                advance_neutral_energy = true
            end
        end

        # flag to determine if a d^2/dr^2 operator is present
        r_diffusion = (advance_numerical_dissipation && num_diss_params.r_dissipation_coefficient > 0.0)
        # flag to determine if a d^2/dvpa^2 operator is present
        vpa_diffusion = ((advance_numerical_dissipation && num_diss_params.vpa_dissipation_coefficient > 0.0) || explicit_weakform_fp_collisions)
        vz_diffusion = (advance_numerical_dissipation && num_diss_params.vz_dissipation_coefficient > 0.0)
    end

    manufactured_solns_test = manufactured_solns_input.use_for_advance

    return advance_info(advance_vpa_advection, advance_z_advection, advance_r_advection,
                        advance_neutral_z_advection, advance_neutral_r_advection,
                        advance_neutral_vz_advection, advance_cx, advance_cx_1V,
                        advance_ionization, advance_ionization_1V,
                        advance_ionization_source, advance_krook_collisions,
                        explicit_weakform_fp_collisions,
                        advance_external_source, advance_numerical_dissipation,
                        advance_sources, advance_continuity, advance_force_balance,
                        advance_energy, advance_neutral_external_source,
                        advance_neutral_sources, advance_neutral_continuity,
                        advance_neutral_force_balance, advance_neutral_energy, rk_coefs,
                        manufactured_solns_test, r_diffusion, vpa_diffusion, vz_diffusion)
end

function setup_dummy_and_buffer_arrays(nr,nz,nvpa,nvperp,nvz,nvr,nvzeta,nspecies_ion,nspecies_neutral)

    dummy_s = allocate_float(nspecies_ion)
    dummy_sr = allocate_float(nr, nspecies_ion)
    dummy_zrs = allocate_shared_float(nz, nr, nspecies_ion)
    dummy_zrsn = allocate_shared_float(nz, nr, nspecies_neutral)
    dummy_vpavperp = allocate_float(nvpa, nvperp)
    
    buffer_z_1 = allocate_shared_float(nz)
    buffer_z_2 = allocate_shared_float(nz)
    buffer_z_3 = allocate_shared_float(nz)
    buffer_z_4 = allocate_shared_float(nz)
    
    buffer_r_1 = allocate_shared_float(nr)
    buffer_r_2 = allocate_shared_float(nr)
    buffer_r_3 = allocate_shared_float(nr)
    buffer_r_4 = allocate_shared_float(nr)

    buffer_zs_1 = allocate_shared_float(nz,nspecies_ion)
    buffer_zs_2 = allocate_shared_float(nz,nspecies_ion)
    buffer_zs_3 = allocate_shared_float(nz,nspecies_ion)
    buffer_zs_4 = allocate_shared_float(nz,nspecies_ion)
    buffer_zsn_1 = allocate_shared_float(nz,nspecies_neutral)
    buffer_zsn_2 = allocate_shared_float(nz,nspecies_neutral)
    buffer_zsn_3 = allocate_shared_float(nz,nspecies_neutral)
    buffer_zsn_4 = allocate_shared_float(nz,nspecies_neutral)

    buffer_rs_1 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_2 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_3 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_4 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_5 = allocate_shared_float(nr,nspecies_ion)
    buffer_rs_6 = allocate_shared_float(nr,nspecies_ion)
    buffer_rsn_1 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_2 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_3 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_4 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_5 = allocate_shared_float(nr,nspecies_neutral)
    buffer_rsn_6 = allocate_shared_float(nr,nspecies_neutral)

    buffer_zrs_1 = allocate_shared_float(nz,nr,nspecies_ion)
    buffer_zrs_2 = allocate_shared_float(nz,nr,nspecies_ion)
    buffer_zrs_3 = allocate_shared_float(nz,nr,nspecies_ion)
    
    buffer_vpavperpzs_1 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_2 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_3 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_4 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_5 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)
    buffer_vpavperpzs_6 = allocate_shared_float(nvpa,nvperp,nz,nspecies_ion)

    buffer_vpavperprs_1 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_2 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_3 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_4 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_5 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)
    buffer_vpavperprs_6 = allocate_shared_float(nvpa,nvperp,nr,nspecies_ion)

    buffer_vpavperpzrs_1 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
    buffer_vpavperpzrs_2 = allocate_shared_float(nvpa,nvperp,nz,nr,nspecies_ion)
    
    buffer_vzvrvzetazsn_1 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_2 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_3 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_4 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_5 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)
    buffer_vzvrvzetazsn_6 = allocate_shared_float(nvz,nvr,nvzeta,nz,nspecies_neutral)

    buffer_vzvrvzetarsn_1 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_2 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_3 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_4 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_5 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)
    buffer_vzvrvzetarsn_6 = allocate_shared_float(nvz,nvr,nvzeta,nr,nspecies_neutral)

    buffer_vzvrvzetazrsn_1 = allocate_shared_float(nvz,nvr,nvzeta,nz,nr,nspecies_neutral)
    buffer_vzvrvzetazrsn_2 = allocate_shared_float(nvz,nvr,nvzeta,nz,nr,nspecies_neutral)
    
    buffer_vpavperp_1 = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_3 = allocate_shared_float(nvpa,nvperp)
    
    return scratch_dummy_arrays(dummy_s,dummy_sr,dummy_vpavperp,dummy_zrs,dummy_zrsn,
        buffer_z_1,buffer_z_2,buffer_z_3,buffer_z_4,
        buffer_r_1,buffer_r_2,buffer_r_3,buffer_r_4,
        buffer_zs_1,buffer_zs_2,buffer_zs_3,buffer_zs_4,
        buffer_zsn_1,buffer_zsn_2,buffer_zsn_3,buffer_zsn_4,
        buffer_rs_1,buffer_rs_2,buffer_rs_3,buffer_rs_4,buffer_rs_5,buffer_rs_6,
        buffer_rsn_1,buffer_rsn_2,buffer_rsn_3,buffer_rsn_4,buffer_rsn_5,buffer_rsn_6,
        buffer_zrs_1,buffer_zrs_2,buffer_zrs_3,
        buffer_vpavperpzs_1,buffer_vpavperpzs_2,buffer_vpavperpzs_3,buffer_vpavperpzs_4,buffer_vpavperpzs_5,buffer_vpavperpzs_6,
        buffer_vpavperprs_1,buffer_vpavperprs_2,buffer_vpavperprs_3,buffer_vpavperprs_4,buffer_vpavperprs_5,buffer_vpavperprs_6,
        buffer_vpavperpzrs_1,buffer_vpavperpzrs_2,
        buffer_vzvrvzetazsn_1,buffer_vzvrvzetazsn_2,buffer_vzvrvzetazsn_3,buffer_vzvrvzetazsn_4,buffer_vzvrvzetazsn_5,buffer_vzvrvzetazsn_6,
        buffer_vzvrvzetarsn_1,buffer_vzvrvzetarsn_2,buffer_vzvrvzetarsn_3,buffer_vzvrvzetarsn_4,buffer_vzvrvzetarsn_5,buffer_vzvrvzetarsn_6,
        buffer_vzvrvzetazrsn_1, buffer_vzvrvzetazrsn_2,
        buffer_vpavperp_1,buffer_vpavperp_2,buffer_vpavperp_3)

end

"""
if evolving the density via continuity equation, redefine the normalised f → f/n
if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
'scratch' should be a (nz,nspecies) array
"""
function normalize_pdf!(pdf, moments, scratch)
    error("Function normalise_pdf() has not been updated to be parallelized. Does not "
          * "seem to be used at the moment.")
    if moments.evolve_ppar
        @. scratch = moments.vth/moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    elseif moments.evolve_density
        @. scratch = 1.0 / moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    end
    return nothing
end

"""
create an array of structs containing scratch arrays for the normalised pdf and low-order moments
that may be evolved separately via fluid equations
"""
function setup_scratch_arrays(moments, pdf_charged_in, pdf_neutral_in, n_rk_stages)
    # create n_rk_stages+1 structs, each of which will contain one pdf,
    # one density, and one parallel flow array
    scratch = Vector{scratch_pdf{5,3,6,3}}(undef, n_rk_stages+1)
    pdf_dims = size(pdf_charged_in)
    moment_dims = size(moments.charged.dens)
    pdf_neutral_dims = size(pdf_neutral_in)
    moment_neutral_dims = size(moments.neutral.dens)
    # populate each of the structs
    for istage ∈ 1:n_rk_stages+1
        # Allocate arrays in temporary variables so that we can identify them
        # by source line when using @debug_shared_array
        pdf_array = allocate_shared_float(pdf_dims...)
        density_array = allocate_shared_float(moment_dims...)
        upar_array = allocate_shared_float(moment_dims...)
        ppar_array = allocate_shared_float(moment_dims...)
        pperp_array = allocate_shared_float(moment_dims...)
        temp_z_s_array = allocate_shared_float(moment_dims...)

        pdf_neutral_array = allocate_shared_float(pdf_neutral_dims...)
        density_neutral_array = allocate_shared_float(moment_neutral_dims...)
        uz_neutral_array = allocate_shared_float(moment_neutral_dims...)
        pz_neutral_array = allocate_shared_float(moment_neutral_dims...)


        scratch[istage] = scratch_pdf(pdf_array, density_array, upar_array,
                                      ppar_array, pperp_array, temp_z_s_array,
                                      pdf_neutral_array, density_neutral_array,
                                      uz_neutral_array, pz_neutral_array)
        @serial_region begin
            scratch[istage].pdf .= pdf_charged_in
            scratch[istage].density .= moments.charged.dens
            scratch[istage].upar .= moments.charged.upar
            scratch[istage].ppar .= moments.charged.ppar
            scratch[istage].pperp .= moments.charged.pperp

            scratch[istage].pdf_neutral .= pdf_neutral_in
            scratch[istage].density_neutral .= moments.neutral.dens
            scratch[istage].uz_neutral .= moments.neutral.uz
            scratch[istage].pz_neutral .= moments.neutral.pz
        end
    end
    return scratch
end

"""
given the number of Runge Kutta stages that are requested,
returns the needed Runge Kutta coefficients;
e.g., if f is the function to be updated, then
f^{n+1}[stage+1] = rk_coef[1,stage]*f^{n} + rk_coef[2,stage]*f^{n+1}[stage] + rk_coef[3,stage]*(f^{n}+dt*G[f^{n+1}[stage]]
"""
function setup_runge_kutta_coefficients(n_rk_stages)
    rk_coefs = allocate_float(3,n_rk_stages)
    rk_coefs .= 0.0
    if n_rk_stages == 4
        rk_coefs[1,1] = 0.5
        rk_coefs[3,1] = 0.5
        rk_coefs[2,2] = 0.5
        rk_coefs[3,2] = 0.5
        rk_coefs[1,3] = 2.0/3.0
        rk_coefs[2,3] = 1.0/6.0
        rk_coefs[3,3] = 1.0/6.0
        rk_coefs[2,4] = 0.5
        rk_coefs[3,4] = 0.5
    elseif n_rk_stages == 3
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.75
        rk_coefs[3,2] = 0.25
        rk_coefs[1,3] = 1.0/3.0
        rk_coefs[3,3] = 2.0/3.0
    elseif n_rk_stages == 2
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.5
        rk_coefs[3,2] = 0.5
    else
        rk_coefs[3,1] = 1.0
    end
    return rk_coefs
end

"""
solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
define approximate characteristic velocity
v₀(z)=vⁿ(z) and take time derivative along this characteristic
df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
for prudent choice of v₀, expect δv≪v so that explicit
time integrator can be used without severe CFL condition
"""
function time_advance!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions, 
           external_source_settings, num_diss_params, advance, fp_arrays, scratch_dummy,
           manufactured_source_list, ascii_io, io_moments, io_dfns)

    @debug_detect_redundant_block_synchronize begin
        # Only want to check for redundant _block_synchronize() calls during the
        # time advance loop, so activate these checks here
        debug_detect_redundant_is_active[] = true
    end

    if isfile(t_params.stopfile)
        if filesize(t_params.stopfile) > 0
            error("Found a 'stop file' at $(t_params.stopfile), but it contains some data "
                  * "(file size is greater than zero), so will not delete.")
        end
        if global_rank[] == 0
            rm(t_params.stopfile)
        end
    end

    @serial_region begin
        if global_rank[] == 0
             println("beginning time advance   ", Dates.format(now(), dateformat"H:MM:SS"))
             flush(stdout)
        end
    end

    start_time = now()

    # main time advance loop
    iwrite_moments = 2
    iwrite_dfns = 2
    finish_now = false
    for i ∈ 1:t_params.nstep
        if t_params.split_operators
            # MRH NOT SUPPORTED
            time_advance_split_operators!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, i)
        else
            time_advance_no_splitting!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
                moments, fields, spectral_objects, advect_objects,
                composition, collisions, geometry, boundary_distributions,
                external_source_settings, num_diss_params, advance, fp_arrays,  scratch_dummy,
                manufactured_source_list, i)
        end
        # update the time
        t += t_params.dt

        if i == t_params.nstep
            # Ensure all output is written at the final step
            finish_now = true
        end

        if mod(i,t_params.nwrite_moments) == 0 || mod(i,t_params.nwrite_dfns) == 0 || finish_now
            # update the diagnostic chodura condition
            update_chodura!(moments,scratch[end].pdf,vpa,vperp,z,r,spectral_objects.r_spectral,composition,geometry,scratch_dummy,advect_objects.z_advect)

            # Always synchronise here, regardless of if we changed region or not
            begin_serial_region(no_synchronize=true)
            _block_synchronize()

            if isfile(t_params.stopfile)
                # Stop cleanly if a file called 'stop' was created
                println("Found 'stop' file $(t_params.stopfile), aborting run")
                flush(stdout)
                finish_now = true
            end

            # If NaNs are present, they should propagate into every field, so only need to
            # check one. Choose phi because it is small (it has no species or velocity
            # dimensions). If a NaN is found, stop the simulation.
            if block_rank[] == 0
                if any(isnan.(fields.phi))
                    println("Found NaN, stopping simulation")
                    found_nan = 1
                else
                    found_nan = 0
                end
                found_nan = MPI.Allreduce(found_nan, +, comm_inter_block[])
            else
                found_nan = 0
            end
            found_nan = MPI.Bcast(found_nan, 0, comm_block[])
            if found_nan != 0
                finish_now = true
            end

            time_for_run = to_minutes(now() - start_time)
        end
        # write moments data to file every nwrite_moments time steps
        if mod(i,t_params.nwrite_moments) == 0 || finish_now
            @debug_detect_redundant_block_synchronize begin
                # Skip check for redundant _block_synchronize() during file I/O because
                # it only runs infrequently
                debug_detect_redundant_is_active[] = false
            end
            begin_serial_region()
            @serial_region begin
                if global_rank[] == 0
                    print("finished time step ", rpad(string(i), 7),"  ",
                          Dates.format(now(), dateformat"H:MM:SS"))
                end
            end
            write_data_to_ascii(moments, fields, vpa, vperp, z, r, t,
                                composition.n_ion_species, composition.n_neutral_species,
                                ascii_io)
            write_moments_data_to_binary(moments, fields, t, composition.n_ion_species,
                                         composition.n_neutral_species, io_moments,
                                         iwrite_moments, time_for_run, r, z)

            if t_params.steady_state_residual
                # Calculate some residuals to see how close simulation is to steady state
                begin_r_z_region()
                result_string = ""
                all_residuals = Vector{mk_float}()
                @loop_s is begin
                    @views residual_ni =
                        steady_state_residuals(scratch[end].density[:,:,is],
                                               scratch[1].density[:,:,is], t_params.dt;
                                               use_mpi=true, only_max_abs=true)
                    if global_rank[] == 0
                        residual_ni = first(values(residual_ni))[1]
                        push!(all_residuals, residual_ni)
                        result_string *= "  density "
                        result_string *= rpad(string(round(residual_ni; sigdigits=4)), 11)
                    end
                end
                if composition.n_neutral_species > 0
                    @loop_sn isn begin
                        residual_nn =
                            steady_state_residuals(scratch[end].density_neutral[:,:,isn],
                                                   scratch[1].density_neutral[:,:,isn],
                                                   t_params.dt; use_mpi=true,
                                                   only_max_abs=true)
                        if global_rank[] == 0
                            residual_nn = first(values(residual_nn))[1]
                            push!(all_residuals, residual_nn)
                            result_string *= " density_neutral "
                            result_string *= rpad(string(round(residual_nn; sigdigits=4)), 11)
                        end
                    end
                end
                if global_rank[] == 0
                    println("    residuals:", result_string)
                    flush(stdout)
                end
                if t_params.converged_residual_value > 0.0
                    if global_rank[] == 0
                        if all(r < t_params.converged_residual_value for r ∈ all_residuals)
                            println("Run converged! All tested residuals less than ",
                                    t_params.converged_residual_value)
                            flush(stdout)
                            finish_now = true
                        end
                    end
                    finish_now = MPI.Bcast(finish_now, 0, comm_world)
                end
            else
                if global_rank[] == 0
                    println()
                    flush(stdout)
                end
            end

            iwrite_moments += 1
            begin_s_r_z_vperp_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end
        if mod(i,t_params.nwrite_dfns) == 0 || finish_now
            @debug_detect_redundant_block_synchronize begin
                # Skip check for redundant _block_synchronize() during file I/O because
                # it only runs infrequently
                debug_detect_redundant_is_active[] = false
            end
            begin_serial_region()
            @serial_region begin
                if global_rank[] == 0
                    println("writing distribution functions at step ", i,"  ",
                                   Dates.format(now(), dateformat"H:MM:SS"))
                    flush(stdout)
                end
            end
            write_dfns_data_to_binary(pdf.charged.norm, pdf.neutral.norm, moments, fields,
                                      t, composition.n_ion_species,
                                      composition.n_neutral_species, io_dfns, iwrite_dfns,
                                      time_for_run, r, z, vperp, vpa, vzeta, vr, vz)
            iwrite_dfns += 1
            begin_s_r_z_vperp_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end

        if finish_now
            break
        end
    end
    return nothing
end

"""
"""
function time_advance_split_operators!(pdf, scratch, t, t_params, vpa, z,
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
    composition, collisions, external_source_settings, num_diss_params, advance, istep)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    dt = t_params.dt
    n_rk_stages = t_params.n_rk_stages
    # to ensure 2nd order accuracy in time for operator-split advance,
    # have to reverse order of operations every other time step
    flipflop = (mod(istep,2)==0)
    if flipflop
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.cx_collisions = false
            end
            if collisions.ionization > 0.0
                advance.ionization_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.ionization_collisions = false
            end
        end
        if collisions.krook_collision_frequency_prefactor  > 0.0
            advance.krook_collisions = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, z, vpa,
                z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                z_SL, vpa_SL, composition, collisions, sources, num_diss_params,
                advance, istep)
            advance.krook_collisions = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.source_terms = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.continuity = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.force_balance = false
        end
        # use the energy equation to update the parallel pressure
        if moments.evolve_ppar
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.energy = false
        end
    else
        # use the energy equation to update the parallel pressure
        if moments.evolve_ppar
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.energy = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.force_balance = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.continuity = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                composition, collisions, external_source_settings, num_diss_params,
                advance, istep)
            advance.source_terms = false
        end
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.ionization > 0.0
                advance.ionization = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.ionization = false
            end
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    composition, collisions, external_source_settings, num_diss_params,
                    advance, istep)
                advance.cx_collisions = false
            end
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_params, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            composition, collisions, external_source_settings, num_diss_params, advance,
            istep)
        advance.vpa_advection = false
    end
    return nothing
end

"""
"""
function time_advance_no_splitting!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions,
           external_source_settings, num_diss_params, advance, fp_arrays, scratch_dummy,
           manufactured_source_list, istep)

    ssp_rk!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
        moments, fields, spectral_objects, advect_objects, composition, collisions,
        geometry, boundary_distributions, external_source_settings, num_diss_params,
        advance, fp_arrays, scratch_dummy, manufactured_source_list, istep)

    return nothing
end

"""
use information obtained from the Runge-Kutta stages to compute the updated pdf;
for the quantities (density, upar, ppar, vth, qpar and phi) that are derived
from the 'true', un-modified pdf, either: update them using info from Runge Kutta
stages, if the quantities are evolved separately from the modified pdf;
or update them by taking the appropriate velocity moment of the evolved pdf
"""
function rk_update!(scratch, pdf, moments, fields, boundary_distributions, vz, vr, vzeta,
                    vpa, vperp, z, r, spectral_objects, advect_objects, rk_coefs, istage, composition,
                    geometry, num_diss_params, advance, scratch_dummy)
    begin_s_r_z_region()

    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]

    z_spectral, r_spectral, vpa_spectral, vperp_spectral = spectral_objects.z_spectral, spectral_objects.r_spectral, spectral_objects.vpa_spectral, spectral_objects.vperp_spectral
    vzeta_spectral, vr_spectral, vz_spectral = spectral_objects.vzeta_spectral, spectral_objects.vr_spectral, spectral_objects.vz_spectral
    vpa_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.r_advect, advect_objects.z_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    ##
    # update the charged particle distribution and moments
    ##
    # here we seem to have duplicate arrays for storing n, u||, p||, etc, but not for vth
    # 'scratch' is for the multiple stages of time advanced quantities, but 'moments' can be updated directly at each stage
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_scratch.pdf[ivpa,ivperp,iz,ir,is] = rk_coefs[1]*pdf.charged.norm[ivpa,ivperp,iz,ir,is] + rk_coefs[2]*old_scratch.pdf[ivpa,ivperp,iz,ir,is] + rk_coefs[3]*new_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    # use Runge Kutta to update any velocity moments evolved separately from the pdf
    rk_update_evolved_moments!(new_scratch, old_scratch, moments, rk_coefs)

    # Ensure there are no negative values in the pdf before applying boundary
    # conditions, so that negative deviations do not mess up the integral-constraint
    # corrections in the sheath boundary conditions.
    force_minimum_pdf_value!(new_scratch.pdf, num_diss_params)

    # Enforce boundary conditions in z and vpa on the distribution function.
    # Must be done after Runge Kutta update so that the boundary condition applied to
    # the updated pdf is consistent with the updated moments - otherwise different upar
    # between 'pdf', 'old_scratch' and 'new_scratch' might mean a point that should be
    # set to zero at the sheath boundary according to the final upar has a non-zero
    # contribution from one or more of the terms.
    # NB: probably need to do the same for the evolved moments
    enforce_boundary_conditions!(new_scratch, moments,
        boundary_distributions.pdf_rboundary_charged, vpa.bc, z.bc, r.bc, vpa, vperp, z,
        r, vpa_spectral, vperp_spectral, 
        vpa_advect, z_advect, r_advect, composition, scratch_dummy,
        advance.r_diffusion, advance.vpa_diffusion)

    if moments.evolve_density && moments.enforce_conservation
        begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
            @views hard_force_moment_constraints!(new_scratch.pdf[:,:,iz,ir,is], moments,
                                                  vpa)
        end
    end
    # update remaining velocity moments that are calculable from the evolved pdf
    update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition)
    # update the diagnostic chodura condition
    # update_chodura!(moments,new_scratch.pdf,vpa,vperp,z,r,r_spectral,composition,geometry,scratch_dummy,z_advect)
    # update the thermal speed
    begin_s_r_z_region()
    try #below block causes DomainError if ppar < 0 or density, so exit cleanly if possible
        update_vth!(moments.charged.vth, new_scratch.ppar, new_scratch.pperp, new_scratch.density, vperp, z, r, composition)
    catch e
        if global_size[] > 1
            println("ERROR: error calculating vth in time_advance.jl")
            println(e)
            display(stacktrace(catch_backtrace()))
            flush(stdout)
            flush(stderr)
            MPI.Abort(comm_world, 1)
        end
        rethrow(e)
    end
    # update the parallel heat flux
    update_qpar!(moments.charged.qpar, moments.charged.qpar_updated, new_scratch.density,
                 new_scratch.upar, moments.charged.vth, new_scratch.pdf, vpa, vperp, z, r,
                 composition, moments.evolve_density, moments.evolve_upar,
                 moments.evolve_ppar)

    calculate_moment_derivatives!(moments, new_scratch, scratch_dummy, z, z_spectral,
                                  num_diss_params)

    ##
    # update the neutral particle distribution and moments
    ##

    if composition.n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            new_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = ( rk_coefs[1]*pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
             + rk_coefs[2]*old_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] + rk_coefs[3]*new_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn])
        end
        # use Runge Kutta to update any velocity moments evolved separately from the pdf
        rk_update_evolved_moments_neutral!(new_scratch, old_scratch, moments, rk_coefs)

        # Ensure there are no negative values in the pdf before applying boundary
        # conditions, so that negative deviations do not mess up the integral-constraint
        # corrections in the sheath boundary conditions.
        force_minimum_pdf_value_neutral!(new_scratch.pdf_neutral, num_diss_params)

        # Enforce boundary conditions in z and vpa on the distribution function.
        # Must be done after Runge Kutta update so that the boundary condition applied to
        # the updated pdf is consistent with the updated moments - otherwise different upar
        # between 'pdf', 'old_scratch' and 'new_scratch' might mean a point that should be
        # set to zero at the sheath boundary according to the final upar has a non-zero
        # contribution from one or more of the terms.
        # NB: probably need to do the same for the evolved moments
        # Note, so far vr and vzeta do not need advect objects, so pass `nothing` for
        # those as a placeholder
        enforce_neutral_boundary_conditions!(new_scratch.pdf_neutral, new_scratch.pdf,
            boundary_distributions, new_scratch.density_neutral, new_scratch.uz_neutral,
            new_scratch.pz_neutral, moments, new_scratch.density, new_scratch.upar,
            fields.Er, vzeta_spectral, vr_spectral, vz_spectral, neutral_r_advect,
            neutral_z_advect, nothing, nothing, neutral_vz_advect, r, z, vzeta, vr, vz,
            composition, geometry, scratch_dummy, advance.r_diffusion,
            advance.vz_diffusion)

        if moments.evolve_density && moments.enforce_conservation
            begin_sn_r_z_region()
            @loop_sn_r_z isn ir iz begin
                @views hard_force_moment_constraints_neutral!(
                    new_scratch.pdf_neutral[:,:,:,iz,ir,isn], moments, vz)
            end
        end

        # update remaining velocity moments that are calculable from the evolved pdf
        update_derived_moments_neutral!(new_scratch, moments, vz, vr, vzeta, z, r,
                                        composition)
        # update the thermal speed
        begin_sn_r_z_region()
        @loop_sn_r_z isn ir iz begin
            moments.neutral.vth[iz,ir,isn] = sqrt(2.0*new_scratch.pz_neutral[iz,ir,isn]/new_scratch.density_neutral[iz,ir,isn])
        end

        # update the parallel heat flux
        update_neutral_qz!(moments.neutral.qz, moments.neutral.qz_updated,
                           new_scratch.density_neutral, new_scratch.uz_neutral,
                           moments.neutral.vth, new_scratch.pdf_neutral, vz, vr, vzeta, z,
                           r, composition, moments.evolve_density, moments.evolve_upar,
                           moments.evolve_ppar)

        calculate_moment_derivatives_neutral!(moments, new_scratch, scratch_dummy, z,
                                              z_spectral, num_diss_params)
    end

    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], z, r, composition, z_spectral, r_spectral, scratch_dummy)
    if !(( moments.evolve_upar || moments.evolve_ppar) &&
              istage == length(scratch)-1)
        # _block_synchronize() here because phi needs to be read on different ranks than
        # it was written on, even though the loop-type does not change here. However,
        # after the final RK stage can skip if:
        #  * evolving upar or ppar as synchronization will be triggered after moments
        #    updates at the beginning of the next RK step
        _block_synchronize()
    end
end

"""
use Runge Kutta to update any charged-particle velocity moments evolved separately from
the pdf
"""
function rk_update_evolved_moments!(new_scratch, old_scratch, moments, rk_coefs)
    # if separately evolving the particle density, update using RK
    if moments.evolve_density
        @loop_s_r_z is ir iz begin
            new_scratch.density[iz,ir,is] = rk_coefs[1]*moments.charged.dens[iz,ir,is] + rk_coefs[2]*old_scratch.density[iz,ir,is] + rk_coefs[3]*new_scratch.density[iz,ir,is]
        end
    end
    # if separately evolving the parallel flow, update using RK
    if moments.evolve_upar
        @loop_s_r_z is ir iz begin
            new_scratch.upar[iz,ir,is] = rk_coefs[1]*moments.charged.upar[iz,ir,is] + rk_coefs[2]*old_scratch.upar[iz,ir,is] + rk_coefs[3]*new_scratch.upar[iz,ir,is]
        end
    end
    # if separately evolving the parallel pressure, update using RK;
    if moments.evolve_ppar
        @loop_s_r_z is ir iz begin
            new_scratch.ppar[iz,ir,is] = rk_coefs[1]*moments.charged.ppar[iz,ir,is] + rk_coefs[2]*old_scratch.ppar[iz,ir,is] + rk_coefs[3]*new_scratch.ppar[iz,ir,is]
        end
    end
end

"""
use Runge Kutta to update any neutral-particle velocity moments evolved separately from
the pdf
"""
function rk_update_evolved_moments_neutral!(new_scratch, old_scratch, moments, rk_coefs)
    # if separately evolving the particle density, update using RK
    if moments.evolve_density
        @loop_sn_r_z isn ir iz begin
            new_scratch.density_neutral[iz,ir,isn] = rk_coefs[1]*moments.neutral.dens[iz,ir,isn] + rk_coefs[2]*old_scratch.density_neutral[iz,ir,isn] + rk_coefs[3]*new_scratch.density_neutral[iz,ir,isn]
        end
    end
    # if separately evolving the parallel flow, update using RK
    if moments.evolve_upar
        @loop_sn_r_z isn ir iz begin
            new_scratch.uz_neutral[iz,ir,isn] = rk_coefs[1]*moments.neutral.uz[iz,ir,isn] + rk_coefs[2]*old_scratch.uz_neutral[iz,ir,isn] + rk_coefs[3]*new_scratch.uz_neutral[iz,ir,isn]
        end
    end
    # if separately evolving the parallel pressure, update using RK;
    if moments.evolve_ppar
        @loop_sn_r_z isn ir iz begin
            new_scratch.pz_neutral[iz,ir,isn] = rk_coefs[1]*moments.neutral.pz[iz,ir,isn] + rk_coefs[2]*old_scratch.pz_neutral[iz,ir,isn] + rk_coefs[3]*new_scratch.pz_neutral[iz,ir,isn]
        end
    end
end

"""
update velocity moments that are calculable from the evolved charged pdf
"""
function update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition)
    if !moments.evolve_density
        update_density!(new_scratch.density, moments.charged.dens_updated,
                        new_scratch.pdf, vpa, vperp, z, r, composition)
    end
    if !moments.evolve_upar
        update_upar!(new_scratch.upar, moments.charged.upar_updated, new_scratch.density,
                     new_scratch.ppar, new_scratch.pdf, vpa, vperp, z, r, composition,
                     moments.evolve_density, moments.evolve_ppar)
    end
    if !moments.evolve_ppar
        # update_ppar! calculates (p_parallel/m_s N_e c_s^2) + (n_s/N_e)*(upar_s/c_s)^2 = (1/√π)∫d(vpa/c_s) (vpa/c_s)^2 * (√π f_s c_s / N_e)
        update_ppar!(new_scratch.ppar, moments.charged.ppar_updated, new_scratch.density,
                     new_scratch.upar, new_scratch.pdf, vpa, vperp, z, r, composition,
                     moments.evolve_density, moments.evolve_upar)
    end 
    update_pperp!(new_scratch.pperp, new_scratch.pdf, vpa, vperp, z, r, composition)
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
                           new_scratch.density_neutral, new_scratch.pz_neutral,
                           new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_ppar)
    end
    if !moments.evolve_ppar
        update_neutral_pz!(new_scratch.pz_neutral, moments.neutral.pz_updated,
                           new_scratch.density_neutral, new_scratch.uz_neutral,
                           new_scratch.pdf_neutral, vz, vr, vzeta, z, r, composition,
                           moments.evolve_density, moments.evolve_upar)
    end
end

"""
"""
function ssp_rk!(pdf, scratch, t, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects, composition, collisions,
           geometry, boundary_distributions, external_source_settings, num_diss_params,
           advance, fp_arrays, scratch_dummy, manufactured_source_list,  istep)

    begin_s_r_z_region()

    n_rk_stages = t_params.n_rk_stages

    first_scratch = scratch[1]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        first_scratch.pdf[ivpa,ivperp,iz,ir,is] = pdf.charged.norm[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.charged.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.charged.upar[iz,ir,is]
        first_scratch.ppar[iz,ir,is] = moments.charged.ppar[iz,ir,is]
        first_scratch.pperp[iz,ir,is] = moments.charged.pperp[iz,ir,is]
    end

    if composition.n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            first_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            first_scratch.density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            first_scratch.uz_neutral[iz,ir,isn] = moments.neutral.uz[iz,ir,isn]
            first_scratch.pz_neutral[iz,ir,isn] = moments.neutral.pz[iz,ir,isn]
            # other neutral moments here if required
        end
    end
    if moments.evolve_upar
        # moments may be read on all ranks, even though loop type is z_s, so need to
        # synchronize here
        _block_synchronize()
    end

    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage, composition, vpa, vperp, z, r)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments,
            advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t,
            t_params, spectral_objects, composition,
            collisions, geometry, scratch_dummy, manufactured_source_list,
            external_source_settings, num_diss_params, advance, fp_arrays, istage)
        @views rk_update!(scratch, pdf, moments, fields, boundary_distributions, vz, vr,
                          vzeta, vpa, vperp, z, r, spectral_objects, advect_objects,
                          advance.rk_coefs[:,istage], istage, composition, geometry,
                          num_diss_params, advance, scratch_dummy)
    end

    istage = n_rk_stages+1

    # update the pdf.norm and moments arrays as needed
    begin_s_r_z_region()
    final_scratch = scratch[istage]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.charged.norm[ivpa,ivperp,iz,ir,is] = final_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        moments.charged.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
        moments.charged.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
        moments.charged.ppar[iz,ir,is] = final_scratch.ppar[iz,ir,is]
        moments.charged.pperp[iz,ir,is] = final_scratch.pperp[iz,ir,is]
    end
    if composition.n_neutral_species > 0
        # No need to synchronize here as we only change neutral quantities and previous
        # region only changed plasma quantities.
        begin_sn_r_z_region(no_synchronize=true)
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = final_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
            moments.neutral.uz[iz,ir,isn] = final_scratch.uz_neutral[iz,ir,isn]
            moments.neutral.pz[iz,ir,isn] = final_scratch.pz_neutral[iz,ir,isn]
        end
        # for now update moments.neutral object directly for diagnostic moments
        # that are not used in Runga-Kutta steps
        update_neutral_pr!(moments.neutral.pr, moments.neutral.pr_updated, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        update_neutral_pzeta!(moments.neutral.pzeta, moments.neutral.pzeta_updated, pdf.neutral.norm, vz, vr, vzeta, z, r, composition)
        # Update ptot (isotropic pressure)
        if r.n > 1 #if 2D geometry
            @loop_sn_r_z isn ir iz begin
                moments.neutral.ptot[iz,ir,isn] = (moments.neutral.pz[iz,ir,isn] + moments.neutral.pr[iz,ir,isn] + moments.neutral.pzeta[iz,ir,isn])/3.0
            end
        else # 1D model
            @loop_sn_r_z isn ir iz begin
                moments.neutral.ptot[iz,ir,isn] = moments.neutral.pz[iz,ir,isn]
            end
        end
        # get particle fluxes (n.b. bad naming convention uz -> means -> n uz here)
        update_neutral_ur!(moments.neutral.ur, moments.neutral.ur_updated,
                           moments.neutral.dens, pdf.neutral.norm, vz, vr, vzeta, z, r,
                           composition)
        update_neutral_uzeta!(moments.neutral.uzeta, moments.neutral.uzeta_updated,
                              moments.neutral.dens, pdf.neutral.norm, vz, vr, vzeta, z,
                              r, composition)
        try #below loop can cause DomainError if ptot < 0 or density < 0, so exit cleanly if possible
            @loop_sn_r_z isn ir iz begin
                # update density using last density from Runga-Kutta stages
                moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
                # get vth for neutrals
                moments.neutral.vth[iz,ir,isn] = sqrt(2.0*moments.neutral.ptot[iz,ir,isn]/moments.neutral.dens[iz,ir,isn])
            end
        catch e
            if global_size[] > 1
                println("ERROR: error at line 724 of time_advance.jl")
                println(e)
                display(stacktrace(catch_backtrace()))
                flush(stdout)
                flush(stderr)
                MPI.Abort(comm_world, 1)
            end
            rethrow(e)
        end
    end

    return nothing
end

"""
euler_time_advance! advances the vector equation dfvec/dt = G[f]
that includes the kinetic equation + any evolved moment equations
using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
with fvec_in an input and fvec_out the output
"""
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments,
    advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t, t_params,
    spectral_objects, composition, collisions, geometry, scratch_dummy,
    manufactured_source_list, external_source_settings, num_diss_params, advance, fp_arrays, istage)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    dt = t_params.dt
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa

    vpa_spectral, vperp_spectral, r_spectral, z_spectral = spectral_objects.vpa_spectral, spectral_objects.vperp_spectral, spectral_objects.r_spectral, spectral_objects.z_spectral
    vz_spectral, vr_spectral, vzeta_spectral = spectral_objects.vz_spectral, spectral_objects.vr_spectral, spectral_objects.vzeta_spectral
    vpa_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.r_advect, advect_objects.z_advect
    neutral_z_advect, neutral_r_advect, neutral_vz_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect, advect_objects.neutral_vz_advect

    if advance.external_source
        external_ion_source_controller!(fvec_in, moments, external_source_settings.ion,
                                        dt)
    end
    if advance.neutral_external_source
        external_neutral_source_controller!(fvec_in, moments,
                                            external_source_settings.neutral, r, z, dt)
    end

    if advance.vpa_advection
        vpa_advection!(fvec_out.pdf, fvec_in, fields, moments, vpa_advect, vpa, vperp, z, r, dt, t,
            vpa_spectral, composition, collisions, external_source_settings.ion, geometry)
    end

    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to charged species

    if advance.z_advection
        z_advection!(fvec_out.pdf, fvec_in, moments, fields, z_advect, z, vpa, vperp, r,
                     dt, t, z_spectral, composition, geometry, scratch_dummy)
    end

    # r advection relies on derivatives in z to get ExB
    if advance.r_advection && r.n > 1
        r_advection!(fvec_out.pdf, fvec_in, moments, fields, r_advect, r, z, vperp, vpa,
                     dt, r_spectral, composition, geometry, scratch_dummy)
    end

    #if advance.vperp_advection
    # PLACEHOLDER
    #end

    if advance.source_terms
        source_terms!(fvec_out.pdf, fvec_in, moments, vpa, z, r, dt, z_spectral,
                      composition, collisions, external_source_settings.ion)
    end

    if advance.neutral_z_advection
        neutral_advection_z!(fvec_out.pdf_neutral, fvec_in, moments, neutral_z_advect,
            r, z, vzeta, vr, vz, dt, t, z_spectral, composition, scratch_dummy)
    end

    if advance.neutral_r_advection && r.n > 1
        neutral_advection_r!(fvec_out.pdf_neutral, fvec_in, neutral_r_advect,
            r, z, vzeta, vr, vz, dt, r_spectral, composition, geometry, scratch_dummy)
    end

    if advance.neutral_vz_advection
        neutral_advection_vz!(fvec_out.pdf_neutral, fvec_in, fields, moments,
                              neutral_vz_advect, vz, vr, vzeta, z, r, dt, vz_spectral,
                              composition, collisions, external_source_settings.neutral)
    end

    if advance.neutral_source_terms
        source_terms_neutral!(fvec_out.pdf_neutral, fvec_in, moments, vpa, z, r, dt, z_spectral,
                      composition, collisions, external_source_settings.neutral)
    end

    if advance.manufactured_solns_test
        source_terms_manufactured!(fvec_out.pdf, fvec_out.pdf_neutral, vz, vr, vzeta, vpa, vperp, z, r, t, dt, composition, manufactured_source_list)
    end

    if advance.cx_collisions || advance.ionization_collisions
        # gyroaverage neutral dfn and place it in the charged.buffer array for use in the collisions step
        vzvrvzeta_to_vpavperp!(pdf.charged.buffer, fvec_in.pdf_neutral, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, geometry, composition)
        # interpolate charged particle dfn and place it in the neutral.buffer array for use in the collisions step
        vpavperp_to_vzvrvzeta!(pdf.neutral.buffer, fvec_in.pdf, vz, vr, vzeta, vpa, vperp, z, r, geometry, composition)
    end

    # account for charge exchange collisions between ions and neutrals
    if advance.cx_collisions_1V
        charge_exchange_collisions_1V!(fvec_out.pdf, fvec_out.pdf_neutral, fvec_in,
                                       moments, composition, vpa, vz,
                                       collisions.charge_exchange, vpa_spectral,
                                       vz_spectral, dt)
    elseif advance.cx_collisions
        charge_exchange_collisions_3V!(fvec_out.pdf, fvec_out.pdf_neutral, pdf.charged.buffer, pdf.neutral.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions.charge_exchange, dt)
    end
    # account for ionization collisions between ions and neutrals
    if advance.ionization_collisions_1V
        ionization_collisions_1V!(fvec_out.pdf, fvec_out.pdf_neutral, fvec_in, vz, vpa,
                                  vperp, z, r, vz_spectral, moments, composition,
                                  collisions, dt)
    elseif advance.ionization_collisions
        ionization_collisions_3V!(fvec_out.pdf, fvec_out.pdf_neutral, pdf.charged.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions, dt)
    end
    if advance.ionization_source
        constant_ionization_source!(fvec_out.pdf, fvec_in, vpa, vperp, z, r, moments,
                                    composition, collisions, dt)
    end

    # Add Krook collision operator for ions
    if advance.krook_collisions
        krook_collisions!(fvec_out.pdf, fvec_in, moments, composition, collisions,
                          vperp, vpa, dt)
    end

    if advance.external_source
        external_ion_source!(fvec_out.pdf, fvec_in, moments, external_source_settings.ion,
                            vperp, vpa, dt)
    end
    if advance.neutral_external_source
        external_neutral_source!(fvec_out.pdf_neutral, fvec_in, moments,
                                external_source_settings.neutral, vzeta, vr, vz, dt)
    end

    # add numerical dissipation
    if advance.numerical_dissipation
        vpa_dissipation!(fvec_out.pdf, fvec_in.pdf, vpa, vpa_spectral, dt,
                         num_diss_params)
        vperp_dissipation!(fvec_out.pdf, fvec_in.pdf, vperp, vperp_spectral, dt,
                         num_diss_params)
        z_dissipation!(fvec_out.pdf, fvec_in.pdf, z, z_spectral, dt,
                       num_diss_params, scratch_dummy)
        r_dissipation!(fvec_out.pdf, fvec_in.pdf, r, r_spectral, dt,
                       num_diss_params, scratch_dummy)
        vz_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, vz,
                                vz_spectral, dt, num_diss_params)
        z_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, z, z_spectral,
                               dt, num_diss_params, scratch_dummy)
        r_dissipation_neutral!(fvec_out.pdf_neutral, fvec_in.pdf_neutral, r, r_spectral,
                               dt, num_diss_params, scratch_dummy)
    end
    # advance with the Fokker-Planck self-collision operator
    if advance.explicit_weakform_fp_collisions
        update_entropy_diagnostic = (istage == 1)
        explicit_fokker_planck_collisions_weak_form!(fvec_out.pdf,fvec_in.pdf,moments.charged.dSdt,composition,collisions,dt,
                                             fp_arrays,r,z,vperp,vpa,vperp_spectral,vpa_spectral,scratch_dummy,
                                             diagnose_entropy_production = update_entropy_diagnostic)
    end
    
    # End of advance for distribution function

    # Start advancing moments
    if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
        # Only need to change region type if moment evolution equations will be used.
        # Exept when using wall boundary conditions, do not actually need to synchronize
        # here because above we only modify the distribution function and below we only
        # modify the moments, so there is no possibility of race conditions.
        begin_s_r_z_region(no_synchronize=true)
    end
    if advance.continuity
        continuity_equation!(fvec_out.density, fvec_in, moments, composition, dt,
                             z_spectral, collisions.ionization,
                             external_source_settings.ion, num_diss_params)
    end
    if advance.force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        force_balance!(fvec_out.upar, fvec_out.density, fvec_in, moments, fields,
                       collisions, dt, z_spectral, composition, geometry,
                       external_source_settings.ion, num_diss_params)
    end
    if advance.energy
        energy_equation!(fvec_out.ppar, fvec_in, moments, collisions, dt, z_spectral,
                         composition, external_source_settings.ion, num_diss_params)
    end
    if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
        # Only need to change region type if moment evolution equations will be used.
        # Exept when using wall boundary conditions, do not actually need to synchronize
        # here because above we only modify the distribution function and below we only
        # modify the moments, so there is no possibility of race conditions.
        begin_sn_r_z_region(no_synchronize=true)
    end
    if advance.neutral_continuity
        neutral_continuity_equation!(fvec_out.density_neutral, fvec_in, moments,
                                     composition, dt, z_spectral, collisions.ionization,
                                     external_source_settings.neutral, num_diss_params)
    end
    if advance.neutral_force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        neutral_force_balance!(fvec_out.uz_neutral, fvec_out.density_neutral, fvec_in,
                               moments, fields, collisions, dt, z_spectral, composition,
                               geometry, external_source_settings.neutral,
                               num_diss_params)
    end
    if advance.neutral_energy
        neutral_energy_equation!(fvec_out.pz_neutral, fvec_in, moments, collisions, dt,
                                 z_spectral, composition,
                                 external_source_settings.neutral, num_diss_params)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments)
    return nothing
end

"""
update the vector containing the pdf and any evolved moments of the pdf
for use in the Runge-Kutta time advance
"""
function update_solution_vector!(evolved, moments, istage, composition, vpa, vperp, z, r)
    new_evolved = evolved[istage+1]
    old_evolved = evolved[istage]
    begin_s_r_z_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_evolved.pdf[ivpa,ivperp,iz,ir,is] = old_evolved.pdf[ivpa,ivperp,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        new_evolved.density[iz,ir,is] = old_evolved.density[iz,ir,is]
        new_evolved.upar[iz,ir,is] = old_evolved.upar[iz,ir,is]
        new_evolved.ppar[iz,ir,is] = old_evolved.ppar[iz,ir,is]
    end
    if composition.n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            new_evolved.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = old_evolved.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
        end
        @loop_sn_r_z isn ir iz begin
            new_evolved.density_neutral[iz,ir,isn] = old_evolved.density_neutral[iz,ir,isn]
            new_evolved.uz_neutral[iz,ir,isn] = old_evolved.uz_neutral[iz,ir,isn]
            new_evolved.pz_neutral[iz,ir,isn] = old_evolved.pz_neutral[iz,ir,isn]
        end
    end
    return nothing
end

end
