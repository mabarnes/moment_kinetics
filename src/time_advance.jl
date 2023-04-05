"""
"""
module time_advance

export setup_time_advance!
export time_advance!

using MPI
using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: _block_synchronize, global_size, comm_world, MPISharedArray, global_rank
using ..debugging
using ..file_io: write_data_to_ascii, write_moments_data_to_binary, write_dfns_data_to_binary, debug_dump
using ..looping
using ..moment_kinetics_structs: scratch_pdf
using ..chebyshev: setup_chebyshev_pseudospectral
using ..chebyshev: chebyshev_derivative!
using ..velocity_moments: update_moments!, reset_moments_status!
using ..velocity_moments: enforce_moment_constraints!
using ..velocity_moments: update_density!, update_upar!, update_ppar!, update_qpar!
using ..velocity_moments: update_neutral_density!, update_neutral_qz!
using ..velocity_moments: update_neutral_uzeta!, update_neutral_uz!, update_neutral_ur!
using ..velocity_moments: update_neutral_pzeta!, update_neutral_pz!, update_neutral_pr!
using ..velocity_grid_transforms: vzvrvzeta_to_vpavperp!, vpavperp_to_vzvrvzeta!
using ..initial_conditions: enforce_z_boundary_condition!, enforce_boundary_conditions!
using ..initial_conditions: enforce_vpa_boundary_condition!, enforce_r_boundary_condition!
using ..initial_conditions: enforce_neutral_boundary_conditions!
using ..initial_conditions: enforce_neutral_z_boundary_condition!, enforce_neutral_r_boundary_condition!
using ..input_structs: advance_info, time_input
using ..advection: setup_advection #, update_boundary_indices!
using ..z_advection: update_speed_z!, z_advection!
using ..r_advection: update_speed_r!, r_advection!
using ..neutral_advection: update_speed_neutral_r!, neutral_advection_r!, update_speed_neutral_z!, neutral_advection_z!
using ..vperp_advection: update_speed_vperp!, vperp_advection!
using ..vpa_advection: update_speed_vpa!, vpa_advection!
using ..charge_exchange: charge_exchange_collisions_1V!, charge_exchange_collisions_3V!
using ..ionization: ionization_collisions_1V!, ionization_collisions_3V!, constant_ionization_source!
using ..numerical_dissipation: vpa_dissipation!, z_dissipation!, r_dissipation!
using ..source_terms: source_terms!, source_terms_manufactured!
using ..continuity: continuity_equation!
using ..force_balance: force_balance!
using ..energy_equation: energy_equation!
using ..em_fields: setup_em_fields, update_phi!
using ..fokker_planck: init_fokker_planck_collisions, explicit_fokker_planck_collisions!
#using ..semi_lagrange: setup_semi_lagrange
using Dates

using ..manufactured_solns: manufactured_sources
using ..advection: advection_info
@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

mutable struct scratch_dummy_arrays
    dummy_sr::Array{mk_float,2}
    dummy_vpavperp::Array{mk_float,2}
    dummy_zr::MPISharedArray{mk_float,2}
	
	#buffer arrays for MPI 
	buffer_z_1::MPISharedArray{mk_float,1}
	buffer_z_2::MPISharedArray{mk_float,1}
	buffer_z_3::MPISharedArray{mk_float,1}
	buffer_z_4::MPISharedArray{mk_float,1}
	
	buffer_r_1::MPISharedArray{mk_float,1}
	buffer_r_2::MPISharedArray{mk_float,1}
	buffer_r_3::MPISharedArray{mk_float,1}
	buffer_r_4::MPISharedArray{mk_float,1}
	
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
	buffer_vzvrvzetazrsn::MPISharedArray{mk_float,6}
	
end 

struct advect_object_struct
    vpa_advect::Vector{advection_info{4,5}}
    vperp_advect::Vector{advection_info{4,5}}
    z_advect::Vector{advection_info{4,5}}
    r_advect::Vector{advection_info{4,5}}
    neutral_z_advect::Vector{advection_info{5,6}}
    neutral_r_advect::Vector{advection_info{5,6}}
end

# consider changing code structure so that
# we can avoid arbitrary types below?
struct spectral_object_struct
    vz_spectral::T where T
    vr_spectral::T where T
    vzeta_spectral::T where T
    vpa_spectral::T where T
    vperp_spectral::T where T
    z_spectral::T where T
    r_spectral::T where T
end

"""
create arrays and do other work needed to setup
the main time advance loop.
this includes creating and populating structs
for Chebyshev transforms, velocity space moments,
EM fields, semi-Lagrange treatment, and advection terms
"""
function setup_time_advance!(pdf, vz, vr, vzeta, vpa, vperp, z, r, composition, drive_input, moments,
                             t_input, collisions, species, geometry, boundary_distributions, num_diss_params)
    # define some local variables for convenience/tidiness
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    # create array containing coefficients needed for the Runge Kutta time advance
    rk_coefs = setup_runge_kutta_coefficients(t_input.n_rk_stages)
    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    manufactured_solns_test = t_input.use_manufactured_solns_for_advance

    if composition.n_neutral_species > 0
        advance_neutral_z_advection = true
        advance_neutral_r_advection = true
        if collisions.charge_exchange > 0.0
            if vz.n == vpa.n && vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                advance_cx_1V = true
                advance_cx = false
            elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                advance_cx = true
                advance_cx_1V = false
            else
                error("If any perpendicular velocity has length>1 they all must. "
                      * "vperp.n=$(vperp.n), vr.n=$(vr.n), vzeta.n=$(vzeta.n)")
            end
        else
            advance_cx = false
            advance_cx_1V = false
        end
        if collisions.ionization > 0.0
            if vz.n == vpa.n && vperp.n == 1 && vr.n == 1 && vzeta.n == 1
                advance_ionization_1V = true
                advance_ionization = false
            elseif vperp.n > 1 && vr.n > 1 && vzeta.n > 1
                advance_ionization_1V = false
                advance_ionization = true
            end
        else
            advance_ionization = false
            advance_ionization_1V = false
        end
    else
        advance_neutral_z_advection = false
        advance_neutral_r_advection = false
        advance_cx = false
        advance_cx_1V = false
        advance_ionization = false
        advance_ionization_1V = false
    end
    # exception for the case where ions are evolved alone but sourced by ionization
    if collisions.ionization > 0.0 && collisions.constant_ionization_rate
        advance_ionization_source = true
        #println("collisions.constant_ionization_rate == true")
        #println("set advance_ionization = false")
        #println("set advance_ionization_1V = false")
        advance_ionization = false
        advance_ionization_1V = false
    else 
        advance_ionization_source = false 
    end
    
    advance_vpa_advection = true
    advance_z_advection = true
    advance_r_advection = true
    advance_numerical_dissipation = true
    advance_sources = false
    advance_continuity = false
    advance_force_balance = false
    advance_energy = false
    if collisions.nuii > 0.0
        explicit_fp_collisions = true
    else 
        explicit_fp_collisions = false    
    end
    # flag to determine if a d^2/dr^2 operator is present 
    r_diffusion = (advance_numerical_dissipation && num_diss_params.r_dissipation_coefficient > 0.0)
    # flag to determine if a d^2/dvpa^2 operator is present
    vpa_diffusion = (advance_numerical_dissipation && num_diss_params.vpa_dissipation_coefficient > 0.0 || explicit_fp_collisions)
    advance = advance_info(advance_vpa_advection, advance_z_advection, advance_r_advection, advance_neutral_z_advection, advance_neutral_r_advection,
                           advance_cx, advance_cx_1V, advance_ionization, advance_ionization_1V, advance_ionization_source, advance_numerical_dissipation, 
                           advance_sources, advance_continuity, advance_force_balance, advance_energy, rk_coefs,
                           manufactured_solns_test, r_diffusion, vpa_diffusion, explicit_fp_collisions)


    if z.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        z_spectral = setup_chebyshev_pseudospectral(z)
        # obtain the local derivatives of the uniform z-grid with respect to the used z-grid
        chebyshev_derivative!(z.duniform_dgrid, z.uniform_grid, z_spectral, z)
    else
        # create dummy Bool variable to return in place of the above struct
        z_spectral = false
        z.duniform_dgrid .= 1.0
    end

    if r.discretization == "chebyshev_pseudospectral" && r.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        r_spectral = setup_chebyshev_pseudospectral(r)
        # obtain the local derivatives of the uniform r-grid with respect to the used r-grid
        chebyshev_derivative!(r.duniform_dgrid, r.uniform_grid, r_spectral, r)
    else
        # create dummy Bool variable to return in place of the above struct
        r_spectral = false
        r.duniform_dgrid .= 1.0
    end

    if vpa.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        vpa_spectral = setup_chebyshev_pseudospectral(vpa)
        # obtain the local derivatives of the uniform vpa-grid with respect to the used vpa-grid
        chebyshev_derivative!(vpa.duniform_dgrid, vpa.uniform_grid, vpa_spectral, vpa)
    else
        # create dummy Bool variable to return in place of the above struct
        vpa_spectral = false
        vpa.duniform_dgrid .= 1.0
    end

    if vperp.discretization == "chebyshev_pseudospectral" && vperp.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vperp
        # and create the plans for the forward and backward fast Chebyshev transforms
        vperp_spectral = setup_chebyshev_pseudospectral(vperp)
        # obtain the local derivatives of the uniform vperp-grid with respect to the used vperp-grid
        chebyshev_derivative!(vperp.duniform_dgrid, vperp.uniform_grid, vperp_spectral, vperp)
    else
        # create dummy Bool variable to return in place of the above struct
        vperp_spectral = false
        vperp.duniform_dgrid .= 1.0
    end

    if vz.discretization == "chebyshev_pseudospectral" && vz.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vz
        # and create the plans for the forward and backward fast Chebyshev transforms
        vz_spectral = setup_chebyshev_pseudospectral(vz)
        # obtain the local derivatives of the uniform vz-grid with respect to the used vz-grid
        chebyshev_derivative!(vz.duniform_dgrid, vz.uniform_grid, vz_spectral, vz)
    else
        # create dummy Bool variable to return in place of the above struct
        vz_spectral = false
        vz.duniform_dgrid .= 1.0
    end

    if vr.discretization == "chebyshev_pseudospectral" && vr.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vr
        # and create the plans for the forward and backward fast Chebyshev transforms
        vr_spectral = setup_chebyshev_pseudospectral(vr)
        # obtain the local derivatives of the uniform vr-grid with respect to the used vr-grid
        chebyshev_derivative!(vr.duniform_dgrid, vr.uniform_grid, vr_spectral, vr)
    else
        # create dummy Bool variable to return in place of the above struct
        vr_spectral = false
        vr.duniform_dgrid .= 1.0
    end

    if vzeta.discretization == "chebyshev_pseudospectral" && vzeta.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vzeta
        # and create the plans for the forward and backward fast Chebyshev transforms
        vzeta_spectral = setup_chebyshev_pseudospectral(vzeta)
        # obtain the local derivatives of the uniform vzeta-grid with respect to the used vzeta-grid
        chebyshev_derivative!(vzeta.duniform_dgrid, vzeta.uniform_grid, vzeta_spectral, vzeta)
    else
        # create dummy Bool variable to return in place of the above struct
        vzeta_spectral = false
        vzeta.duniform_dgrid .= 1.0
    end

    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf.charged.norm, pdf.neutral.norm, t_input.n_rk_stages)
    # setup dummy arrays & buffer arrays for z r MPI
    n_neutral_species_alloc = max(1,composition.n_neutral_species)
    scratch_dummy = setup_dummy_and_buffer_arrays(r.n,z.n,vpa.n,vperp.n,vz.n,vr.n,vzeta.n,
                                   composition.n_ion_species,n_neutral_species_alloc)
    # create arrays for Fokker-Planck collisions 
    fp_arrays = init_fokker_planck_collisions(vperp,vpa)
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
            @views update_speed_r!(r_advect[is], fields, vpa, vperp, z, r, geometry)
        end
        # enforce prescribed boundary condition in r on the distribution function f
        @views enforce_r_boundary_condition!(pdf.charged.unnorm, boundary_distributions.pdf_rboundary_charged,
                                            r.bc, r_advect, vpa, vperp, z, r, composition,
                                            scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
                                            scratch_dummy.buffer_vpavperpzs_3, scratch_dummy.buffer_vpavperpzs_4,
                                            scratch_dummy.buffer_vpavperpzrs_1, advance.r_diffusion)
    end

    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in z
    begin_serial_region()
    z_advect = setup_advection(n_ion_species, z, vpa, vperp, r)
    # initialise the z advection speed
    begin_s_r_vperp_vpa_region()
    @loop_s is begin
        @views update_speed_z!(z_advect[is], fields, vpa, vperp, z, r, 0.0, geometry)
    end
    # enforce prescribed boundary condition in z on the distribution function f
    @views enforce_z_boundary_condition!(pdf.charged.unnorm, z.bc, z_advect, vpa, vperp, z, r, composition,
            scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
            scratch_dummy.buffer_vpavperprs_3, scratch_dummy.buffer_vpavperprs_4,
            scratch_dummy.buffer_vpavperpzrs_1)

    begin_serial_region()


    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_advect = setup_advection(n_ion_species, vpa, vperp, z, r)
    # initialise the vpa advection speed
    begin_s_r_z_vperp_region()
    update_speed_vpa!(vpa_advect, fields, vpa, vperp, z, r, composition, geometry)

    begin_serial_region()
    @serial_region begin
        for is ∈ 1:n_ion_species
            # enforce prescribed boundary condition in vpa on the distribution function f
            @views enforce_vpa_boundary_condition!(pdf.charged.norm[:,:,:,:,is], vpa.bc, vpa_advect[is], advance.vpa_diffusion)
        end
    end

    # create structure vperp_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vperp
    begin_serial_region()
    vperp_advect = setup_advection(n_ion_species, vperp, vpa, z, r)
    # initialise the vperp advection speed
    begin_serial_region()
    @serial_region begin
        for is ∈ 1:n_ion_species
            @views update_speed_vperp!(vperp_advect[is], vpa, vperp, z, r)
            # enforce prescribed boundary condition in vpa on the distribution function f
            #PLACEHOLDER
            #@views enforce_vperp_boundary_condition!(pdf.norm[:,:,:,:,is], vpa.bc, vpa_advect[is])
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
        begin_sn_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_r!(neutral_r_advect[isn], r, z, vzeta, vr, vz)
        end
        # enforce prescribed boundary condition in r on the neutral distribution function f
        @views enforce_neutral_r_boundary_condition!(pdf.neutral.unnorm,
            boundary_distributions.pdf_rboundary_neutral, neutral_r_advect, vz, vr, vzeta, z, r, composition,
            scratch_dummy.buffer_vzvrvzetazsn_1, scratch_dummy.buffer_vzvrvzetazsn_2,
            scratch_dummy.buffer_vzvrvzetazsn_3, scratch_dummy.buffer_vzvrvzetazsn_4,
            scratch_dummy.buffer_vzvrvzetazrsn)
    end

    # create structure neutral_z_advect for neutral particle advection
    begin_serial_region()
    neutral_z_advect = setup_advection(n_neutral_species_alloc, z, vz, vr, vzeta, r)
    if n_neutral_species > 0
        # initialise the z advection speed
        begin_sn_vzeta_vr_vz_region()
        @loop_sn isn begin
            @views update_speed_neutral_z!(neutral_z_advect[isn], r, z, vzeta, vr, vz)
        end
        # enforce prescribed boundary condition in z on the neutral distribution function f
        @views enforce_neutral_z_boundary_condition!(pdf.neutral.unnorm, pdf.charged.unnorm, boundary_distributions,
            neutral_z_advect, z_advect, vz, vr, vzeta, vpa, vperp, z, r, composition,
            scratch_dummy.buffer_vzvrvzetarsn_1, scratch_dummy.buffer_vzvrvzetarsn_2,
            scratch_dummy.buffer_vzvrvzetarsn_3, scratch_dummy.buffer_vzvrvzetarsn_4,
            scratch_dummy.buffer_vzvrvzetazrsn)
    end

    ##
    # construct named list of advect & spectral objects to compactify arguments
    ##

    #advect_objects = (vpa_advect = vpa_advect, vperp_advect = vperp_advect, z_advect = z_advect,
    # r_advect = r_advect, neutral_z_advect = neutral_z_advect, neutral_r_advect = neutral_r_advect)
    advect_objects = advect_object_struct(vpa_advect, vperp_advect, z_advect, r_advect, neutral_z_advect, neutral_r_advect)
    #spectral_objects = (vz_spectral = vz_spectral, vr_spectral = vr_spectral, vzeta_spectral = vzeta_spectral,
    # vpa_spectral = vpa_spectral, vperp_spectral = vperp_spectral, z_spectral = z_spectral, r_spectral = r_spectral)
    spectral_objects = spectral_object_struct(vz_spectral, vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral, z_spectral, r_spectral)
    if(advance.manufactured_solns_test)
        manufactured_source_list = manufactured_sources(r,z,vperp,vpa,vzeta,vr,vz,composition,geometry,collisions,num_diss_params)
    else
        manufactured_source_list = false # dummy Bool to be passed as argument instead of list
    end

    # Ensure all processes are synchronized at the end of the setup
    _block_synchronize()

    return moments, fields, spectral_objects, advect_objects,
    scratch, advance, fp_arrays, scratch_dummy, manufactured_source_list
end

function setup_dummy_and_buffer_arrays(nr,nz,nvpa,nvperp,nvz,nvr,nvzeta,nspecies_ion,nspecies_neutral)

	dummy_sr = allocate_float(nr, nspecies_ion)
    dummy_zr = allocate_shared_float(nz, nr)
    dummy_vpavperp = allocate_float(nvpa, nvperp)
	
	# should the arrays below be shared memory arrays? MRH
	buffer_z_1 = allocate_shared_float(nz)
	buffer_z_2 = allocate_shared_float(nz)
	buffer_z_3 = allocate_shared_float(nz)
	buffer_z_4 = allocate_shared_float(nz)
	
	buffer_r_1 = allocate_shared_float(nr)
	buffer_r_2 = allocate_shared_float(nr)
	buffer_r_3 = allocate_shared_float(nr)
	buffer_r_4 = allocate_shared_float(nr)
	
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
	
	buffer_vzvrvzetazrsn = allocate_shared_float(nvz,nvr,nvzeta,nz,nr,nspecies_neutral)
	
	return scratch_dummy_arrays(dummy_sr,dummy_vpavperp,dummy_zr,
		buffer_z_1,buffer_z_2,buffer_z_3,buffer_z_4,
		buffer_r_1,buffer_r_2,buffer_r_3,buffer_r_4,
		buffer_vpavperpzs_1,buffer_vpavperpzs_2,buffer_vpavperpzs_3,buffer_vpavperpzs_4,buffer_vpavperpzs_5,buffer_vpavperpzs_6,
		buffer_vpavperprs_1,buffer_vpavperprs_2,buffer_vpavperprs_3,buffer_vpavperprs_4,buffer_vpavperprs_5,buffer_vpavperprs_6,
		buffer_vpavperpzrs_1,buffer_vpavperpzrs_2,
		buffer_vzvrvzetazsn_1,buffer_vzvrvzetazsn_2,buffer_vzvrvzetazsn_3,buffer_vzvrvzetazsn_4,buffer_vzvrvzetazsn_5,buffer_vzvrvzetazsn_6,
		buffer_vzvrvzetarsn_1,buffer_vzvrvzetarsn_2,buffer_vzvrvzetarsn_3,buffer_vzvrvzetarsn_4,buffer_vzvrvzetarsn_5,buffer_vzvrvzetarsn_6,
		buffer_vzvrvzetazrsn)

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
        temp_z_s_array = allocate_shared_float(moment_dims...)

        pdf_neutral_array = allocate_shared_float(pdf_neutral_dims...)
        density_neutral_array = allocate_shared_float(moment_neutral_dims...)


        scratch[istage] = scratch_pdf(pdf_array, density_array, upar_array,
                                      ppar_array, temp_z_s_array,
                                      pdf_neutral_array, density_neutral_array)
        @serial_region begin
            scratch[istage].pdf .= pdf_charged_in
            scratch[istage].density .= moments.charged.dens
            scratch[istage].upar .= moments.charged.upar
            scratch[istage].ppar .= moments.charged.ppar

            scratch[istage].pdf_neutral .= pdf_neutral_in
            scratch[istage].density_neutral .= moments.neutral.dens
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
function time_advance!(pdf, scratch, t, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions, 
           num_diss_params, advance, fp_arrays, scratch_dummy,
           manufactured_source_list, ascii_io, io_moments, io_dfns)

    @debug_detect_redundant_block_synchronize begin
        # Only want to check for redundant _block_synchronize() calls during the
        # time advance loop, so activate these checks here
        debug_detect_redundant_is_active[] = true
    end

    @serial_region begin
        if global_rank[] == 0
             println("beginning time advance   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # main time advance loop
    iwrite_moments = 2
    iwrite_dfns = 2
    for i ∈ 1:t_input.nstep

        time_advance_no_splitting!(pdf, scratch, t, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
                moments, fields, spectral_objects, advect_objects,
                composition, collisions, geometry, boundary_distributions,
                num_diss_params, advance, fp_arrays,  scratch_dummy, manufactured_source_list, i)
        # update the time
        t += t_input.dt
        # write moments data to file every nwrite_moments time steps
        if mod(i,t_input.nwrite_moments) == 0
            @debug_detect_redundant_block_synchronize begin
                # Skip check for redundant _block_synchronize() during file I/O because
                # it only runs infrequently
                debug_detect_redundant_is_active[] = false
            end
            begin_serial_region()
            @serial_region begin
                if global_rank[] == 0
                     println("finished time step ", i,"    ",
                                   Dates.format(now(), dateformat"H:MM:SS"))
                end
            end
           # @serial_region println("finished time step ", i,"    ",
           #                        Dates.format(now(), dateformat"H:MM:SS"))
            write_data_to_ascii(moments, fields, vpa, vperp, z, r, t,
                                composition.n_ion_species, composition.n_neutral_species,
                                ascii_io)
            write_moments_data_to_binary(moments, fields, t, composition.n_ion_species,
                                         composition.n_neutral_species, io_moments,
                                         iwrite_moments, r, z)
            iwrite_moments += 1
            begin_s_r_z_vperp_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end
        if mod(i,t_input.nwrite_dfns) == 0
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
                end
            end
            write_dfns_data_to_binary(pdf.charged.unnorm, pdf.neutral.unnorm, t,
                                      composition.n_ion_species,
                                      composition.n_neutral_species, io_dfns, iwrite_dfns,
                                      r, z, vperp, vpa, vzeta, vr, vz)
            iwrite_dfns += 1
            begin_s_r_z_vperp_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end
    end
    return nothing
end

"""
"""
function time_advance_no_splitting!(pdf, scratch, t, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions,
           num_diss_params, advance, fp_arrays, scratch_dummy, manufactured_source_list, istep)

    #pdf_in = pdf.unnorm #get input pdf values in case we wish to impose a constant-in-time boundary condition in r
    #use pdf.norm for this fn for now.

    if t_input.n_rk_stages > 1
        ssp_rk!(pdf, scratch, t, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
            moments, fields, spectral_objects, advect_objects,
            composition, collisions, geometry, boundary_distributions, num_diss_params,
            advance, fp_arrays, scratch_dummy, manufactured_source_list, istep)#pdf_in,
    else
        euler_time_advance!(scratch, scratch, pdf, fields, moments,
            advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t,
            t_input, spectral_objects, composition,
            collisions, geometry, boundary_distributions, scratch_dummy, manufactured_source_list, advance, 1)#pdf_in,
        # NB: this must be broken -- scratch is updated in euler_time_advance!,
        # but not the pdf or moments.  need to add update to these quantities here
    end
    return nothing
end

"""
"""
function rk_update!(scratch, pdf, moments, fields, vz, vr, vzeta, vpa, vperp, z, r, rk_coefs, istage, composition, z_spectral, r_spectral, scratch_dummy)
    begin_s_r_z_vperp_region()
    #nvpa = vpa.n
    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]

    ##
    # update the charged particle distribution and moments
    ##

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new_scratch.pdf[ivpa,ivperp,iz,ir,is] = rk_coefs[1]*pdf.charged.norm[ivpa,ivperp,iz,ir,is] + rk_coefs[2]*old_scratch.pdf[ivpa,ivperp,iz,ir,is] + rk_coefs[3]*new_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.charged.unnorm[ivpa,ivperp,iz,ir,is] = new_scratch.pdf[ivpa,ivperp,iz,ir,is]
    end
    update_density!(new_scratch.density, pdf.charged.unnorm, vpa, vperp, z, r, composition)

    update_upar!(new_scratch.upar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    # convert from particle particle flux to parallel flow
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        new_scratch.upar[iz,ir,is] /= new_scratch.density[iz,ir,is]
    end

    update_ppar!(new_scratch.ppar, pdf.charged.unnorm, vpa, vperp, z, r, composition)
    # update the thermal speed
    begin_s_r_z_region()
    try #below block causes DomainError if ppar < 0 or density, so exit cleanly if possible
		@loop_s_r_z is ir iz begin
			moments.charged.vth[iz,ir,is] = sqrt(2.0*new_scratch.ppar[iz,ir,is]/new_scratch.density[iz,ir,is])
		end
	catch e
		if global_size[] > 1
			println("ERROR: error at line 598 of time_advance.jl")
			println(e)
			display(stacktrace(catch_backtrace()))
            flush(stdout)
            flush(stderr)
			MPI.Abort(comm_world, 1)
		end
		rethrow(e)
	end
    # update the parallel heat flux
    update_qpar!(moments.charged.qpar, pdf.charged.unnorm, vpa, vperp, z, r, composition)

    ##
    # update the neutral particle distribution and moments
    ##

    if composition.n_neutral_species > 0
        begin_sn_r_z_vzeta_vr_vz_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            new_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = ( rk_coefs[1]*pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
             + rk_coefs[2]*old_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] + rk_coefs[3]*new_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn])
        end
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            pdf.neutral.unnorm[ivz,ivr,ivzeta,iz,ir,isn] = new_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
        end
        update_neutral_density!(new_scratch.density_neutral, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        # other neutral moments here if needed for individual Runga-Kutta steps
    end

    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], z, r, composition, z_spectral, r_spectral, scratch_dummy)
    #begin_s_r_z_vperp_region()
end

"""
"""
function ssp_rk!(pdf, scratch, t, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, boundary_distributions, num_diss_params,
           advance, fp_arrays, scratch_dummy, manufactured_source_list,  istep)#pdf_in,

    begin_s_r_z_region()

    n_rk_stages = t_input.n_rk_stages

    first_scratch = scratch[1]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        first_scratch.pdf[ivpa,ivperp,iz,ir,is] = pdf.charged.norm[ivpa,ivperp,iz,ir,is]
        # change norm -> unnorm if remove moment-based evolution?
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.charged.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.charged.upar[iz,ir,is]
        first_scratch.ppar[iz,ir,is] = moments.charged.ppar[iz,ir,is]
    end

    if composition.n_neutral_species > 0
        begin_sn_r_z_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            first_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
            # change norm -> unnorm if remove moment-based evolution?
        end
        @loop_sn_r_z isn ir iz begin
            first_scratch.density_neutral[iz,ir,isn] = moments.neutral.dens[iz,ir,isn]
            # other neutral moments here if required
        end
    end

    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage, composition, vpa, vperp, z, r)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments,
            advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t,
            t_input, spectral_objects, composition,
            collisions, geometry, boundary_distributions,
            scratch_dummy, manufactured_source_list, 
            num_diss_params, advance, fp_arrays, istage) #pdf_in,
        @views rk_update!(scratch, pdf, moments, fields, vz, vr, vzeta, vpa, vperp, z, r, advance.rk_coefs[:,istage], 
         istage, composition, spectral_objects.z_spectral, spectral_objects.r_spectral, scratch_dummy)
    end

    istage = n_rk_stages+1

    # update the pdf.norm and moments arrays as needed
    begin_s_r_z_region()
    final_scratch = scratch[istage]
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.charged.norm[ivpa,ivperp,iz,ir,is] = final_scratch.pdf[ivpa,ivperp,iz,ir,is]
        # change norm -> unnorm if remove moment-based evolution?
    end
    @loop_s_r_z is ir iz begin
        moments.charged.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
        moments.charged.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
        moments.charged.ppar[iz,ir,is] = final_scratch.ppar[iz,ir,is]
    end
    if composition.n_neutral_species > 0
        # No need to synchronize here as we only change neutral quantities and previous
        # region only changed plasma quantities.
        begin_sn_r_z_region(no_synchronize=true)
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn] = final_scratch.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]
            # change norm -> unnorm if remove moment-based evolution?
        end
        # for now update moments.neutral object directly for diagnostic moments
        # that are not used in Runga-Kutta steps
        update_neutral_qz!(moments.neutral.qz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_pz!(moments.neutral.pz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_pr!(moments.neutral.pr, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_pzeta!(moments.neutral.pzeta, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
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
        update_neutral_uz!(moments.neutral.uz, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_ur!(moments.neutral.ur, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
        update_neutral_uzeta!(moments.neutral.uzeta, pdf.neutral.unnorm, vz, vr, vzeta, z, r, composition)
		try #below loop can cause DomainError if ptot < 0 or density < 0, so exit cleanly if possible
			@loop_sn_r_z isn ir iz begin
				# update density using last density from Runga-Kutta stages
				moments.neutral.dens[iz,ir,isn] = final_scratch.density_neutral[iz,ir,isn]
				# convert from particle flux to mean velocity
				moments.neutral.uz[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
				moments.neutral.ur[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
				moments.neutral.uzeta[iz,ir,isn] /= moments.neutral.dens[iz,ir,isn]
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

    update_pdf_unnorm!(pdf, moments, scratch[istage].temp_z_s, composition, vpa, vperp)
    return nothing
end

"""
euler_time_advance! advances the vector equation dfvec/dt = G[f]
that includes the kinetic equation + any evolved moment equations
using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
with fvec_in an input and fvec_out the output
"""
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments,
    advect_objects, vz, vr, vzeta, vpa, vperp, gyrophase, z, r, t, t_input,
    spectral_objects, composition, collisions, geometry, boundary_distributions,
    scratch_dummy, manufactured_source_list, num_diss_params, advance, fp_arrays, istage) #pdf_in,
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa

    vpa_spectral, vperp_spectral, r_spectral, z_spectral = spectral_objects.vpa_spectral, spectral_objects.vperp_spectral, spectral_objects.r_spectral, spectral_objects.z_spectral
    vpa_advect, r_advect, z_advect = advect_objects.vpa_advect, advect_objects.r_advect, advect_objects.z_advect
    neutral_z_advect, neutral_r_advect = advect_objects.neutral_z_advect, advect_objects.neutral_r_advect

    if advance.vpa_advection
        vpa_advection!(fvec_out.pdf, fvec_in, fields, vpa_advect, vpa, vperp, z, r, dt,
            vpa_spectral, composition, geometry)
    end

    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to charged species

    if advance.z_advection
        z_advection!(fvec_out.pdf, fvec_in, fields, z_advect, z, vpa, vperp, r,
            dt, t, z_spectral, composition, geometry, scratch_dummy)
    end

    # r advection relies on derivatives in z to get ExB
    if advance.r_advection && r.n > 1
        r_advection!(fvec_out.pdf, fvec_in, fields, r_advect, r, z, vperp, vpa,
            dt, r_spectral, composition, geometry, scratch_dummy)
    end

    #if advance.vperp_advection
    # PLACEHOLDER
    #end

    if advance.neutral_z_advection
        neutral_advection_z!(fvec_out.pdf_neutral, fvec_in, neutral_z_advect,
            r, z, vzeta, vr, vz, dt, z_spectral, composition, geometry, scratch_dummy)
    end

    if advance.neutral_r_advection && r.n > 1
        neutral_advection_r!(fvec_out.pdf_neutral, fvec_in, neutral_r_advect,
            r, z, vzeta, vr, vz, dt, r_spectral, composition, geometry, scratch_dummy)
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
        charge_exchange_collisions_1V!(fvec_out.pdf, fvec_out.pdf_neutral, fvec_in, composition, vpa, vperp, z, r,
                                    collisions.charge_exchange, dt)
    elseif advance.cx_collisions
        charge_exchange_collisions_3V!(fvec_out.pdf, fvec_out.pdf_neutral, pdf.charged.buffer, pdf.neutral.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions.charge_exchange, dt)
    end
    # account for ionization collisions between ions and neutrals
    if advance.ionization_collisions_1V
        ionization_collisions_1V!(fvec_out.pdf, fvec_out.pdf_neutral, fvec_in, vpa, vperp, z, r, composition, collisions, dt)
    elseif advance.ionization_collisions
        ionization_collisions_3V!(fvec_out.pdf, fvec_out.pdf_neutral, pdf.charged.buffer, fvec_in, composition,
                                        vz, vr, vzeta, vpa, vperp, z, r, collisions, dt)
    end
    if advance.ionization_source
        constant_ionization_source!(fvec_out.pdf, vpa, vperp, z, r, composition, collisions, dt)
    end
    
    # add numerical dissipation
    if advance.numerical_dissipation
        vpa_dissipation!(fvec_out.pdf, fvec_in.pdf, vpa, vpa_spectral, dt,
                         num_diss_params)
        z_dissipation!(fvec_out.pdf, fvec_in.pdf, z, z_spectral, dt,
                       num_diss_params, scratch_dummy)
        r_dissipation!(fvec_out.pdf, fvec_in.pdf, r, r_spectral, dt,
                       num_diss_params, scratch_dummy)
    end
    
    if advance.explicit_fp_collisions
        explicit_fokker_planck_collisions!(fvec_out.pdf, fvec_in.pdf,composition,collisions,dt,fp_arrays,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral)
    end 

    # enforce boundary conditions in r, z and vpa on the charged particle distribution function
    enforce_boundary_conditions!(fvec_out.pdf, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance)
    # enforce boundary conditions in r and z on the neutral particle distribution function
    if n_neutral_species > 0
        enforce_neutral_boundary_conditions!(fvec_out.pdf_neutral, fvec_out.pdf, boundary_distributions,
         neutral_r_advect, neutral_z_advect, z_advect, vz, vr, vzeta, vpa, vperp, z, r, composition,
         scratch_dummy)
    end
    # End of advance for distribution function

    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    #reset_moments_status!(moments, composition, z)
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
            # other neutral moments here if needed
        end
    end
    return nothing
end

"""
if separately evolving the density via the continuity equation,
the evolved pdf has been normalised by the particle density
undo this normalisation to get the true particle distribution function
"""
function update_pdf_unnorm!(pdf, moments, scratch, composition, vpa, vperp)
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf.charged.unnorm[ivpa,ivperp,iz,ir,is] = pdf.charged.norm[ivpa,ivperp,iz,ir,is]
    end
    if composition.n_neutral_species > 0
        begin_sn_r_z_vzeta_vr_vz_region()
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            pdf.neutral.unnorm[ivz,ivr,ivzeta,iz,ir,isn] = pdf.neutral.norm[ivz,ivr,ivzeta,iz,ir,isn]
        end
    end
end

end
