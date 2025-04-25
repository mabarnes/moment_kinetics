"""
Module for including the Full-F Fokker-Planck Collision Operator.

The functions in this module are split into two groups. 

The first set of functions implement the weak-form
Collision operator using the Rosenbluth-MacDonald-Judd
formulation in a divergence form. The Green's functions
for the Rosenbluth potentials are used to obtain the Rosenbluth
potentials at the boundaries. To find the potentials
everywhere else elliptic solves of the PDEs for the
Rosenbluth potentials are performed with Dirichlet
boundary conditions. These routines provide the default collision operator
used in the code.

The second set of functions are used to set up the necessary arrays to 
compute the Rosenbluth potentials everywhere in vpa, vperp
by direct integration of the Green's functions. These functions are 
supported for the purposes of testing and debugging.

Lower-level routines are provided by functions from
[`moment_kinetics.fokker_planck_calculus`](@ref).

Parallelisation of the collision operator uses a special 'anyv' region type, see
[Collision operator and `anyv` region](@ref).
"""
module fokker_planck

# Import moment_kinetics so that we can refer to it in docstrings
import moment_kinetics

export init_fokker_planck_collisions, fokkerplanck_arrays_struct
export init_fokker_planck_collisions_weak_form
export explicit_fokker_planck_collisions_weak_form!
export explicit_fokker_planck_collisions!
export explicit_fp_collisions_weak_form_Maxwellian_cross_species!
export calculate_Maxwellian_Rosenbluth_coefficients
export get_local_Cssp_coefficients!, init_fokker_planck_collisions
# testing
export symmetric_matrix_inverse
export fokker_planck_collision_operator_weak_form!
export fokker_planck_self_collision_operator_weak_form!
# implicit advance
export setup_fp_nl_solve, implicit_ion_fokker_planck_self_collisions!,
       fokker_planck_self_collisions_backward_euler_step!

using SpecialFunctions: ellipk, ellipe, erf
using FastGaussQuadrature
using Dates
using LinearAlgebra: lu, ldiv!, mul!
using MPI
using OrderedCollections: OrderedDict
using ..type_definitions: mk_float, mk_int, OptionsDict
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication
using ..velocity_moments: integrate_over_vspace,
                          get_density, get_upar, get_ppar, 
                          get_pperp, get_qpar, get_pressure, get_rmom
using ..looping
using ..timer_utils
using ..input_structs: fkpl_collisions_input, set_defaults_and_check_section!
using ..input_structs: multipole_expansion, direct_integration
using ..reference_parameters: get_reference_collision_frequency_ii
using ..fokker_planck_calculus: init_Rosenbluth_potential_integration_weights!,
                                init_Rosenbluth_potential_boundary_integration_weights!,
                                allocate_boundary_integration_weights,
                                allocate_rosenbluth_potential_boundary_data,
                                fokkerplanck_arrays_direct_integration_struct, fokkerplanck_weakform_arrays_struct,
                                assemble_matrix_operators_dirichlet_bc,
                                assemble_matrix_operators_dirichlet_bc_sparse,
                                assemble_explicit_collision_operator_rhs_serial!,
                                assemble_explicit_collision_operator_rhs_parallel!,
                                assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!,
                                calculate_YY_arrays, enforce_vpavperp_BCs!,
                                calculate_rosenbluth_potential_boundary_data!,
                                elliptic_solve!, algebraic_solve!, allocate_preconditioner_matrix,
                                calculate_rosenbluth_potentials_via_elliptic_solve!,
                                calculate_rosenbluth_potentials_via_analytical_Maxwellian!,
                                calculate_test_particle_preconditioner!,
                                advance_linearised_test_particle_collisions!
using ..fokker_planck_test: Cssp_fully_expanded_form, calculate_collisional_fluxes,
                            d2Gdvpa2_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian,
                            F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
using ..reference_parameters: setup_reference_parameters
using ..nonlinear_solvers: setup_nonlinear_solve, newton_solve!

"""
Function for reading Fokker Planck collision operator input parameters. 
Structure the namelist as follows.

    [fokker_planck_collisions]
    use_fokker_planck = true
    nuii = 1.0
    frequency_option = "manual"
"""
function setup_fkpl_collisions_input(toml_input::AbstractDict, warn_unexpected::Bool)
    reference_params = setup_reference_parameters(toml_input, warn_unexpected)
    # get reference collision frequency (note factor of 1/2 due to definition choices)
    nuii_fkpl_default = 0.5*get_reference_collision_frequency_ii(reference_params)
    # read the input toml and specify a sensible default
    input_section = set_defaults_and_check_section!(
        toml_input, "fokker_planck_collisions", warn_unexpected;
        # begin default inputs (as kwargs)
        use_fokker_planck = false,
        nuii = -1.0,
        frequency_option = "reference_parameters",
        self_collisions = true,
        use_conserving_corrections = true,
        boundary_data_option = direct_integration,
        slowing_down_test = false,
        sd_density = 1.0,
        sd_temp = 0.01,
        sd_q = 1.0,
        sd_mi = 0.25,
        sd_me = 0.25/1836.0,
        Zi = 1.0)
    # ensure that the collision frequency is consistent with the input option
    frequency_option = input_section["frequency_option"]
    if frequency_option == "reference_parameters"
        input_section["nuii"] = nuii_fkpl_default
    elseif frequency_option == "manual" 
        # use the frequency from the input file
        # do nothing
    else
        error("Invalid option [fokker_planck_collisions] "
              * "frequency_option=$(frequency_option) passed")
    end
    # finally, ensure nuii < 0 if use_fokker_planck is false
    # so that nuii > 0 is the only check required in the rest of the code
    if !input_section["use_fokker_planck"]
        input_section["nuii"] = -1.0
    end
    input = OrderedDict(Symbol(k)=>v for (k,v) in input_section)
    #println(input)
    if input_section["slowing_down_test"]
        # calculate nu_alphae and vc3 (critical speed of slowing down)
        # as a diagnostic to aid timestep choice
        # nu_alphae/nuref
        Zalpha = input_section["Zi"]
        Zi = input_section["sd_q"]
        ni = input_section["sd_density"]
        ne = Zi*ni+Zalpha # assume unit density for alphas in initial state
        Te = input_section["sd_temp"]
        me = input_section["sd_me"]
        mi = input_section["sd_mi"]
        nu_alphae = (8.0/(3.0*sqrt(pi)))*ne*sqrt(me)*(Te^(-1.5))*(Zalpha^(2.0))
        vc3 = (3.0/4.0)*sqrt(pi)*(Zi^2)*(ni/ne)*(1.0/mi)*(1.0/sqrt(me))*(Te^(1.5))
        nu_alphaalpha = Zalpha^4
        nu_alphai = nu_alphae*vc3
        if global_rank[] == 0
            println("slowing_down_test = true")
            println("nu_alphaalpha/nuref = $nu_alphaalpha")
            println("nu_alphai/nuref = $nu_alphai")
            println("nu_alphae/nuref = $nu_alphae")
            println("vc3/cref^3 = $vc3")
            println("critical speed vc/cref = ",vc3^(1.0/3.0))
        end
    end
    return fkpl_collisions_input(; input...)
end

########################################################
# begin functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
########################################################

"""
Function that initialises the arrays needed for Fokker Planck collisions
using numerical integration to compute the Rosenbluth potentials only
at the boundary and using an elliptic solve to obtain the potentials 
in the rest of the velocity space domain.
"""
function init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; precompute_weights=false, test_dense_matrix_construction=false, print_to_screen=true)
    bwgt = allocate_boundary_integration_weights(vpa,vperp)
    if vperp.n > 1 && precompute_weights
        init_Rosenbluth_potential_boundary_integration_weights!(bwgt.G0_weights, bwgt.G1_weights, bwgt.H0_weights, bwgt.H1_weights,
                                        bwgt.H2_weights, bwgt.H3_weights, vpa, vperp, print_to_screen=print_to_screen)
    end
    rpbd = allocate_rosenbluth_potential_boundary_data(vpa,vperp)
    if test_dense_matrix_construction
        MM2D_sparse, KKpar2D_sparse, KKperp2D_sparse, 
        KKpar2D_with_BC_terms_sparse, KKperp2D_with_BC_terms_sparse,
        LP2D_sparse, LV2D_sparse, LB2D_sparse, KPperp2D_sparse,
        PUperp2D_sparse, PPparPUperp2D_sparse, PPpar2D_sparse,
        MMparMNperp2D_sparse = assemble_matrix_operators_dirichlet_bc(vpa,vperp,vpa_spectral,vperp_spectral,print_to_screen=print_to_screen)
    else
        MM2D_sparse, KKpar2D_sparse, KKperp2D_sparse,
        KKpar2D_with_BC_terms_sparse, KKperp2D_with_BC_terms_sparse,
        LP2D_sparse, LV2D_sparse, LB2D_sparse, KPperp2D_sparse,
        PUperp2D_sparse, PPparPUperp2D_sparse, PPpar2D_sparse,
        MMparMNperp2D_sparse = assemble_matrix_operators_dirichlet_bc_sparse(vpa,vperp,vpa_spectral,vperp_spectral,print_to_screen=print_to_screen)
    end
    lu_obj_MM = lu(MM2D_sparse)
    lu_obj_LP = lu(LP2D_sparse)
    lu_obj_LV = lu(LV2D_sparse)
    lu_obj_LB = lu(LB2D_sparse)
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("finished LU decomposition initialisation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    
    YY_arrays = calculate_YY_arrays(vpa,vperp,vpa_spectral,vperp_spectral)
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("finished YY array calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # The following velocity-space-sized buffer arrays are used to evaluate the
    # collision operator for a single species at a single spatial point. They are
    # shared-memory arrays. The `comm` argument to `allocate_shared_float()` is used to
    # set up the shared-memory arrays so that they are shared only by the processes on
    # `comm_anyv_subblock[]` rather than on the full `comm_block[]` (see also the
    # "Collision operator and anyv region" section of the "Developing" page of the docs.
    # This means that different subblocks that are calculating the collision operator at
    # different spatial points do not interfere with each others' buffer arrays.
    nvpa, nvperp = vpa.n, vperp.n
    S_dummy = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    Q_dummy = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    rhsvpavperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    rhsvpavperp_copy1 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    rhsvpavperp_copy2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    rhsvpavperp_copy3 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    
    CC = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    GG = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    HH = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dHdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dHdvperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dGdvperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2Gdvperp2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2Gdvpa2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2Gdvperpdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    
    FF = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dFdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dFdvperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    # preconditioner matrix
    CC2D_sparse, CC2D_sparse_constructor, lu_obj_CC2D = allocate_preconditioner_matrix(vpa,vperp,vpa_spectral,vperp_spectral)
    rhs_advection = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    # dummy arrays for JFNK
    Fnew = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    Fresidual = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    F_delta_x = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    F_rhs_delta = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    Fv = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    Fw = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    fka = fokkerplanck_weakform_arrays_struct(bwgt,rpbd,MM2D_sparse,KKpar2D_sparse,KKperp2D_sparse,
                                           KKpar2D_with_BC_terms_sparse,KKperp2D_with_BC_terms_sparse,
                                           LP2D_sparse,LV2D_sparse,LB2D_sparse,PUperp2D_sparse,PPparPUperp2D_sparse,
                                           PPpar2D_sparse,MMparMNperp2D_sparse,KPperp2D_sparse,
                                           lu_obj_MM,lu_obj_LP,lu_obj_LV,lu_obj_LB,
                                           YY_arrays, S_dummy, Q_dummy, rhsvpavperp, rhsvpavperp_copy1, rhsvpavperp_copy2, rhsvpavperp_copy3,
                                           CC, GG, HH, dHdvpa, dHdvperp, dGdvperp, d2Gdvperp2, d2Gdvpa2, d2Gdvperpdvpa,
                                           FF, dFdvpa, dFdvperp, 
                                           CC2D_sparse, CC2D_sparse_constructor, lu_obj_CC2D,
                                           rhs_advection, Fnew, Fresidual, F_delta_x, F_rhs_delta, Fv, Fw)
    return fka
end

"""
Function for advancing with the explicit, weak-form, self-collision operator
using the existing method for computing the Rosenbluth potentials, with
the addition of cross-species collisions against fixed Maxwellian distribution functions
where the Rosenbluth potentials are specified using analytical results.
"""
@timeit global_timer explicit_fp_collisions_weak_form_Maxwellian_cross_species!(
                         pdf_out, pdf_in, dSdt, composition, collisions, dt,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct, r, z, vperp,
                         vpa, vperp_spectral, vpa_spectral;
                         diagnose_entropy_production=false) = begin
    # N.B. only self-collisions are currently supported
    # This can be modified by adding a loop over s' below
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))
    
    use_conserving_corrections = collisions.fkpl.use_conserving_corrections
    fkin = collisions.fkpl
    # masses charge numbers and collision frequencies
    mref = 1.0 # generalise if multiple ions evolved
    Zs = fkin.Zi # generalise if multiple ions evolved
    nuref = fkin.nuii # generalise if multiple ions evolved
    msp = [fkin.sd_mi, fkin.sd_me]
    Zsp = [fkin.sd_q, -1.0]
    # assume here that ne = sum_i n_i and that initial condition
    # for beam ions has unit density
    ns = 1.0 # initial density of evolved ions (hardcode to be the same as initial conditions)
    # get electron density from quasineutrality ne = sum_s Zs ns
    densp = [fkin.sd_density, fkin.sd_q*fkin.sd_density+ns*Zs]
 
    uparsp = [0.0, 0.0]
    vthsp = [sqrt(fkin.sd_temp/msp[1]), sqrt(fkin.sd_temp/msp[2])]
    
    # N.B. parallelisation using special 'anyv' region
    @begin_s_r_z_anyv_region()
    @loop_s_r_z is ir iz begin
        # computes sum over s' of  C[Fs,Fs'] with Fs' an assumed Maxwellian 
        @views fokker_planck_collision_operator_weak_form_Maxwellian_Fsp!(pdf_in[:,:,iz,ir,is],
                                     nuref,mref,Zs,msp,Zsp,densp,uparsp,vthsp,
                                     fkpl_arrays,vperp,vpa,vperp_spectral,vpa_spectral)
        # enforce the boundary conditions on CC before it is used for timestepping
        enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral)
        # make sure that the cross-species terms conserve density
        if use_conserving_corrections
            density_conserving_correction!(fkpl_arrays.CC, pdf_in[:,:,iz,ir,is], vpa, vperp,
                                fkpl_arrays.S_dummy)
        end
        # advance this part of s,r,z with the resulting sum_s' C[Fs,Fs']
        @begin_anyv_vperp_vpa_region()
        CC = fkpl_arrays.CC
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] += dt*CC[ivpa,ivperp]
        end
        
    end
    return nothing
end


"""
Function for advancing with the explicit, weak-form, self-collision operator.
"""
@timeit global_timer explicit_fokker_planck_collisions_weak_form!(
                         pdf_out, pdf_in, dSdt, composition, collisions, dt,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct, r, z, vperp,
                         vpa, vperp_spectral, vpa_spectral, scratch_dummy;
                         diagnose_entropy_production=false) = begin
    # N.B. only self-collisions are currently supported
    # This can be modified by adding a loop over s' below
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))
    
    # masses and collision frequencies
    ms, msp = 1.0, 1.0 # generalise!
    nuref = collisions.fkpl.nuii # generalise!
    Zi = collisions.fkpl.Zi # generalise!
    nuss = nuref*(Zi^4) # include charge number factor for self collisions
    use_conserving_corrections = collisions.fkpl.use_conserving_corrections
    boundary_data_option = collisions.fkpl.boundary_data_option
    # N.B. parallelisation using special 'anyv' region
    @begin_s_r_z_anyv_region()
    @loop_s_r_z is ir iz begin
        # first argument is Fs, and second argument is Fs' in C[Fs,Fs'] 
        @views fokker_planck_self_collision_operator_weak_form!(
            pdf_in[:,:,iz,ir,is], ms, nuss, fkpl_arrays,
            vperp, vpa, vperp_spectral, vpa_spectral, 
            boundary_data_option = boundary_data_option,
            use_conserving_corrections = use_conserving_corrections)
        # advance this part of s,r,z with the resulting C[Fs,Fs]
        @begin_anyv_vperp_vpa_region()
        CC = fkpl_arrays.CC
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] += dt*CC[ivpa,ivperp]
        end
        if diagnose_entropy_production
            calculate_entropy_production!(dSdt, pdf_in, fkpl_arrays, vpa, vperp,
                                        iz, ir, is)
        end
    end
    return nothing
end

@timeit fokker_planck_self_collision_operator_weak_form!(
                         pdf_in, ms, nuss,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct, vperp, vpa,
                         vperp_spectral, vpa_spectral; 
                         boundary_data_option=direct_integration,
                         use_conserving_corrections=false) = begin
    # first argument is Fs, and second argument is Fs' in C[Fs,Fs'] 
    @views fokker_planck_collision_operator_weak_form!(
        pdf_in, pdf_in, ms, ms, nuss, fkpl_arrays,
        vperp, vpa, vperp_spectral, vpa_spectral,
        boundary_data_option = boundary_data_option)
    CC = fkpl_arrays.CC
    # enforce the boundary conditions on CC before it is used for timestepping
    enforce_vpavperp_BCs!(CC,vpa,vperp,vpa_spectral,vperp_spectral)
    # make ad-hoc conserving corrections appropriate only for the self operator
    if use_conserving_corrections
        conserving_corrections!(CC, pdf_in, vpa, vperp,
                            fkpl_arrays.S_dummy)
    end
    return nothing
end

"""
Function for evaluating \$C_{ss'} = C_{ss'}[F_s,F_{s'}]\$

The result is stored in the array `fkpl_arrays.CC`.

The normalised collision frequency for collisions between species s and s' is defined by
```math
\\tilde{\\nu}_{ss'} = \\frac{L_{\\mathrm{ref}}}{c_{\\mathrm{ref}}}\\frac{\\gamma_{ss'} n_\\mathrm{ref}}{m_s^2 c_\\mathrm{ref}^3}
```
with \$\\gamma_{ss'} = 2 \\pi (Z_s Z_{s'})^2 e^4 \\ln \\Lambda_{ss'} / (4 \\pi
\\epsilon_0)^2\$.
The input parameter to this code is 
```math
\\tilde{\\nu}_{ii} = \\frac{L_{\\mathrm{ref}}}{c_{\\mathrm{ref}}}\\frac{\\gamma_\\mathrm{ref} n_\\mathrm{ref}}{m_\\mathrm{ref}^2 c_\\mathrm{ref}^3}
```
with \$\\gamma_\\mathrm{ref} = 2 \\pi e^4 \\ln \\Lambda_{ii} / (4 \\pi
\\epsilon_0)^2\$. This means that \$\\tilde{\\nu}_{ss'} = (Z_s Z_{s'})^2\\tilde{\\nu}_\\mathrm{ref}\$ and this conversion is handled explicitly in the code with the charge number input provided by the user.
"""
@timeit global_timer fokker_planck_collision_operator_weak_form!(
                         ffs_in, ffsp_in, ms, msp, nussp,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct, vperp, vpa,
                         vperp_spectral, vpa_spectral; test_assembly_serial=false,
                         use_Maxwellian_Rosenbluth_coefficients=false,
                         use_Maxwellian_field_particle_distribution=false,
                         algebraic_solve_for_d2Gdvperp2 = false, calculate_GG=false,
                         calculate_dGdvperp=false,
                         boundary_data_option=direct_integration) = begin
    @boundscheck vpa.n == size(ffsp_in,1) || throw(BoundsError(ffsp_in))
    @boundscheck vperp.n == size(ffsp_in,2) || throw(BoundsError(ffsp_in))
    @boundscheck vpa.n == size(ffs_in,1) || throw(BoundsError(ffs_in))
    @boundscheck vperp.n == size(ffs_in,2) || throw(BoundsError(ffs_in))
    
    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    rhsvpavperp = fkpl_arrays.rhsvpavperp
    lu_obj_MM = fkpl_arrays.lu_obj_MM
    YY_arrays = fkpl_arrays.YY_arrays    
    
    CC = fkpl_arrays.CC
    GG = fkpl_arrays.GG
    HH = fkpl_arrays.HH
    dHdvpa = fkpl_arrays.dHdvpa
    dHdvperp = fkpl_arrays.dHdvperp
    dGdvperp = fkpl_arrays.dGdvperp
    d2Gdvperp2 = fkpl_arrays.d2Gdvperp2
    d2Gdvpa2 = fkpl_arrays.d2Gdvpa2
    d2Gdvperpdvpa = fkpl_arrays.d2Gdvperpdvpa
    FF = fkpl_arrays.FF
    dFdvpa = fkpl_arrays.dFdvpa
    dFdvperp = fkpl_arrays.dFdvperp
    
    if use_Maxwellian_Rosenbluth_coefficients
        calculate_rosenbluth_potentials_via_analytical_Maxwellian!(GG,HH,dHdvpa,dHdvperp,
                 d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,vpa,vperp)
    else
        calculate_rosenbluth_potentials_via_elliptic_solve!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays,
             algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
             calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp,
             boundary_data_option=boundary_data_option)
    end
    # assemble the RHS of the collision operator matrix eq
    if use_Maxwellian_field_particle_distribution
        @begin_anyv_region()
        dens = get_density(ffs_in,vpa,vperp)
        upar = get_upar(ffs_in, vpa, vperp, dens)
        ppar = get_ppar(ffs_in, vpa, vperp, upar)
        pperp = get_pperp(ffs_in, vpa, vperp)
        pressure = get_pressure(ppar,pperp)
        vth = sqrt(2.0*pressure/dens)
        @begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            FF[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dFdvpa[ivpa,ivperp] = dFdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dFdvperp[ivpa,ivperp] = dFdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
        # Need to synchronize as FF, dFdvpa, dFdvperp may be read outside the
        # locally-owned set of ivperp, ivpa indices in
        # assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!()
        @_anyv_subblock_synchronize()
        assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!(rhsvpavperp,
          FF,dFdvpa,dFdvperp,
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    elseif test_assembly_serial
        assemble_explicit_collision_operator_rhs_serial!(rhsvpavperp,ffs_in,
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    else
        @_anyv_subblock_synchronize()
        assemble_explicit_collision_operator_rhs_parallel!(rhsvpavperp,ffs_in,
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    end
    # solve the collision operator matrix eq
    @begin_anyv_region()
    @anyv_serial_region begin
        # sc and rhsc are 1D views of the data in CC and rhsc, created so that we can use
        # the 'matrix solve' functionality of ldiv!() from the LinearAlgebra package
        sc = vec(CC)
        rhsc = vec(rhsvpavperp)
        # invert mass matrix and fill fc
        ldiv!(sc, lu_obj_MM, rhsc)
    end
    return nothing
end

"""
Function for computing the collision operator
```math
\\sum_{s^\\prime} C[F_{s},F_{s^\\prime}]
```
when \$F_{s^\\prime}\$
is an analytically specified Maxwellian distribution and
the corresponding Rosenbluth potentials
are specified using analytical results.
"""
@timeit global_timer fokker_planck_collision_operator_weak_form_Maxwellian_Fsp!(
                         ffs_in, nuref::mk_float, ms::mk_float, Zs::mk_float,
                         msp::Array{mk_float,1}, Zsp::Array{mk_float,1},
                         densp::Array{mk_float,1}, uparsp::Array{mk_float,1},
                         vthsp::Array{mk_float,1},
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct, vperp, vpa,
                         vperp_spectral, vpa_spectral) = begin
    @boundscheck vpa.n == size(ffs_in,1) || throw(BoundsError(ffs_in))
    @boundscheck vperp.n == size(ffs_in,2) || throw(BoundsError(ffs_in))
    
    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    rhsvpavperp = fkpl_arrays.rhsvpavperp
    lu_obj_MM = fkpl_arrays.lu_obj_MM
    YY_arrays = fkpl_arrays.YY_arrays    
    
    CC = fkpl_arrays.CC
    GG = fkpl_arrays.GG
    HH = fkpl_arrays.HH
    dHdvpa = fkpl_arrays.dHdvpa
    dHdvperp = fkpl_arrays.dHdvperp
    dGdvperp = fkpl_arrays.dGdvperp
    d2Gdvperp2 = fkpl_arrays.d2Gdvperp2
    d2Gdvpa2 = fkpl_arrays.d2Gdvpa2
    d2Gdvperpdvpa = fkpl_arrays.d2Gdvperpdvpa
    FF = fkpl_arrays.FF
    dFdvpa = fkpl_arrays.dFdvpa
    dFdvperp = fkpl_arrays.dFdvperp
    
    # number of primed species
    nsp = size(msp,1)
    
    @begin_anyv_vperp_vpa_region()
    # fist set dummy arrays for coefficients to zero
    @loop_vperp_vpa ivperp ivpa begin
        d2Gdvpa2[ivpa,ivperp] = 0.0
        d2Gdvperp2[ivpa,ivperp] = 0.0
        d2Gdvperpdvpa[ivpa,ivperp] = 0.0
        dHdvpa[ivpa,ivperp] = 0.0
        dHdvperp[ivpa,ivperp] = 0.0
    end
    # sum the contributions from the potentials, including order unity factors that differ between species
    # making use of the Linearity of the operator in Fsp
    # note that here we absorb ms/msp and Zsp^2 into the definition of the potentials, and we pass
    # ms = msp = 1 to the collision operator assembly routine so that we can use a single array to include
    # the contribution to the summed Rosenbluth potential from all the species
    for isp in 1:nsp
        dens = densp[isp]
        upar = uparsp[isp]
        vth = vthsp[isp]
        ZZ = (Zsp[isp]*Zs)^2 # factor from gamma_ss'
        @loop_vperp_vpa ivperp ivpa begin
            d2Gdvpa2[ivpa,ivperp] += ZZ*d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperp2[ivpa,ivperp] += ZZ*d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperpdvpa[ivpa,ivperp] += ZZ*d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvpa[ivpa,ivperp] += ZZ*(ms/msp[isp])*dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvperp[ivpa,ivperp] += ZZ*(ms/msp[isp])*dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
    end
    # Need to synchronize as these arrays may be read outside the locally-owned set of
    # ivperp, ivpa indices in assemble_explicit_collision_operator_rhs_parallel!()
    # assemble the RHS of the collision operator matrix eq

    @_anyv_subblock_synchronize()
    assemble_explicit_collision_operator_rhs_parallel!(rhsvpavperp,ffs_in,
      d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
      dHdvpa,dHdvperp,1.0,1.0,nuref,
      vpa,vperp,YY_arrays)

    # solve the collision operator matrix eq
    @begin_anyv_region()
    @anyv_serial_region begin
        # sc and rhsc are 1D views of the data in CC and rhsc, created so that we can use
        # the 'matrix solve' functionality of ldiv!() from the LinearAlgebra package
        sc = vec(CC)
        rhsc = vec(rhsvpavperp)
        # invert mass matrix and fill fc
        ldiv!(sc, lu_obj_MM, rhsc)
    end
    return nothing
end

"""
Function that solves `A x = b` for a matrix of the form
```math
\\begin{array}{ccc}
A_{00} & 0 & A_{02} \\\\
0 & A_{11} & A_{12} \\\\
A_{02} & A_{12} & A_{22} \\\\
\\end{array}
```
appropriate for the moment numerical conserving terms used in
the Fokker-Planck collision operator.
"""
function symmetric_matrix_inverse(A00,A02,A11,A12,A22,b0,b1,b2)
    # matrix determinant
    detA = A00*(A11*A22 - A12^2) - A11*A02^2
    # cofactors C (also a symmetric matrix)
    C00 = A11*A22 - A12^2
    C01 = A12*A02
    C02 = -A11*A02
    C11 = A00*A22 - A02^2
    C12 = -A00*A12
    C22 = A00*A11
    x0 = ( C00*b0 + C01*b1 + C02*b2 )/detA
    x1 = ( C01*b0 + C11*b1 + C12*b2 )/detA
    x2 = ( C02*b0 + C12*b1 + C22*b2 )/detA
    #println("b0: ",b0," b1: ",b1," b2: ",b2)
    #println("A00: ",A00," A02: ",A02," A11: ",A11," A12: ",A12," A22: ",A22, " detA: ",detA)
    #println("C00: ",C00," C02: ",C02," C11: ",C11," C12: ",C12," C22: ",C22)
    #println("x0: ",x0," x1: ",x1," x2: ",x2)
    return x0, x1, x2
end

"""
Function that solves `A x = b` for a matrix of the form
```math
\\begin{array}{ccc}
A_{00} & A_{01} & A_{02} \\\\
A_{01} & A_{11} & A_{12} \\\\
A_{02} & A_{12} & A_{22} \\\\
\\end{array}
```
appropriate for moment numerical conserving terms.
 
"""
function symmetric_matrix_inverse(A00,A01,A02,A11,A12,A22,b0,b1,b2)
    # matrix determinant
    detA = A00*(A11*A22 - A12^2) - A01*(A01*A22 - A12*A02) + A02*(A01*A12 - A11*A02)
    # cofactors C (also a symmetric matrix)
    C00 = A11*A22 - A12^2
    C01 = A12*A02 - A01*A22
    C02 = A01*A12 -A11*A02
    C11 = A00*A22 - A02^2
    C12 = A01*A02 -A00*A12
    C22 = A00*A11 - A01^2
    x0 = ( C00*b0 + C01*b1 + C02*b2 )/detA
    x1 = ( C01*b0 + C11*b1 + C12*b2 )/detA
    x2 = ( C02*b0 + C12*b1 + C22*b2 )/detA
    #println("b0: ",b0," b1: ",b1," b2: ",b2)
    #println("A00: ",A00," A02: ",A02," A11: ",A11," A12: ",A12," A22: ",A22, " detA: ",detA)
    #println("C00: ",C00," C02: ",C02," C11: ",C11," C12: ",C12," C22: ",C22)
    #println("x0: ",x0," x1: ",x1," x2: ",x2)
    return x0, x1, x2
end

"""
Function that applies numerical-error correcting terms to ensure
numerical conservation of the moments `density, upar, pressure` in the self-collision operator.
Modifies the collision operator such that the operator becomes
```math
C_{ss} = C^\\ast_{ss}[F_s,F_{s}] - \\left(x_0 + x_1(v_{\\|}-u_{\\|})+ x_2(v_\\perp^2 +(v_{\\|}-u_{\\|})^2)\\right)F_s
```
where \$C^\\ast_{ss}[F_s,F_{s}]\$ is the weak-form self-collision operator computed using 
the finite-element implementation, \$u_{\\|}\$ is the parallel velocity of \$F_s\$,
and \$x_0,x_1,x_2\$ are parameters that are chosen so that \$C_{ss}\$
conserves density, parallel velocity and pressure of \$F_s\$.
"""
function conserving_corrections!(CC,pdf_in,vpa,vperp,dummy_vpavperp)
    @begin_anyv_region()
    x0, x1, x2, upar = 0.0, 0.0, 0.0, 0.0
    @anyv_serial_region begin
        # In principle the integrations here could be shared among the processes in the
        # 'anyv' subblock, but this block is not a significant part of the cost of the
        # collision operator, so probably not worth the complication.

        # compute moments of the input pdf
        dens =  get_density(pdf_in, vpa, vperp)
        upar = get_upar(pdf_in, vpa, vperp, dens)
        ppar = get_ppar(pdf_in, vpa, vperp, upar)
        pperp = get_pperp(pdf_in, vpa, vperp)
        pressure = get_pressure(ppar,pperp)
        qpar = get_qpar(pdf_in, vpa, vperp, upar, dummy_vpavperp)
        rmom = get_rmom(pdf_in, vpa, vperp, upar, dummy_vpavperp)

        # compute moments of the numerical collision operator
        dn = get_density(CC, vpa, vperp)
        du = get_upar(CC, vpa, vperp, 1.0)
        dppar = get_ppar(CC, vpa, vperp, upar)
        dpperp = get_pperp(CC, vpa, vperp)
        dp = get_pressure(dppar,dpperp)

        # form the appropriate matrix coefficients
        b0, b1, b2 = dn, du - upar*dn, 3.0*dp
        A00, A02, A11, A12, A22 = dens, 3.0*pressure, ppar, qpar, rmom

        # obtain the coefficients for the corrections
        (x0, x1, x2) = symmetric_matrix_inverse(A00,A02,A11,A12,A22,b0,b1,b2)
    end

    # Broadcast x0, x1, x2 to all processes in the 'anyv' subblock
    param_vec = [x0, x1, x2, upar]
    if comm_anyv_subblock[] != MPI.COMM_NULL
        MPI.Bcast!(param_vec, 0, comm_anyv_subblock[])
    end
    (x0, x1, x2, upar) = param_vec
    
    # correct CC
    @begin_anyv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        wpar = vpa.grid[ivpa] - upar
        CC[ivpa,ivperp] -= (x0 + x1*wpar + x2*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_in[ivpa,ivperp]
    end
end

"""
Function that applies a numerical-error correcting term to ensure
numerical conservation of the `density` in the collision operator.
```math
C_{ss^\\prime} = C^\\ast_{ss}[F_s,F_{s^\\prime}] - x_0 F_s
```
where \$C^\\ast_{ss}[F_s,F_{s^\\prime}]\$ is the weak-form collision operator computed using 
the finite-element implementation.
"""
function density_conserving_correction!(CC,pdf_in,vpa,vperp,dummy_vpavperp)
    @begin_anyv_region()
    x0 = 0.0
    @anyv_serial_region begin
        # In principle the integrations here could be shared among the processes in the
        # 'anyv' subblock, but this block is not a significant part of the cost of the
        # collision operator, so probably not worth the complication.

        # compute density of the input pdf
        dens =  get_density(pdf_in, vpa, vperp)
        
        # compute density of the numerical collision operator
        dn = get_density(CC, vpa, vperp)
        
        # obtain the coefficient for the correction
        x0 = dn/dens
    end

    # Broadcast x0 to all processes in the 'anyv' subblock
    param_vec = [x0]
    if comm_anyv_subblock[] != MPI.COMM_NULL
        MPI.Bcast!(param_vec, 0, comm_anyv_subblock[])
    end
    x0 = param_vec[1]
    
    # correct CC
    @begin_anyv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        CC[ivpa,ivperp] -= x0*pdf_in[ivpa,ivperp]
    end
end

"""
Function to calculate entropy production, in place.
"""
function calculate_entropy_production!(dSdt,pdf,fkpl_arrays,vpa,vperp,
                                iz::mk_int,ir::mk_int,is::mk_int)
    # Note that we pass spatial indices here to permit
    # use of the shared-memory parallelism to calculate
    # and return a float value in an array
    @begin_anyv_vperp_vpa_region()

    CC = fkpl_arrays.CC
    # assign dummy array
    lnfC = fkpl_arrays.rhsvpavperp
    @loop_vperp_vpa ivperp ivpa begin
        lnfC[ivpa,ivperp] = log(abs(pdf[ivpa,ivperp,iz,ir,is]) + 1.0e-15)*CC[ivpa,ivperp]
    end
    @begin_anyv_region()
    @anyv_serial_region begin
        dSdt[iz,ir,is] = -get_density(lnfC,vpa,vperp)
    end
    return nothing
end


######################################################
# end functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
######################################################



##########################################################
# begin functions associated with the direct integration
# method for computing the Rosenbluth potentials
##########################################################


"""
Function that allocates the required ancilliary arrays for direct integration routines.
"""
function allocate_fokkerplanck_arrays_direct_integration(vperp,vpa)
    nvpa = vpa.n
    nvperp = vperp.n
    
    G0_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    G1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H0_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H2_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H3_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)

    # The following velocity-space-sized buffer arrays are used to evaluate the
    # collision operator for a single species at a single spatial point. They are
    # shared-memory arrays. The `comm` argument to `allocate_shared_float()` is used to
    # set up the shared-memory arrays so that they are shared only by the processes on
    # `comm_anyv_subblock[]` rather than on the full `comm_block[]`. This means that
    # different subblocks that are calculating the collision operator at different
    # spatial points do not interfere with each others' buffer arrays.
    # Note that the 'weights' allocated above are read-only and therefore can be used
    # simultaneously by different subblocks. They are shared over the full
    # `comm_block[]` in order to save memory and setup time.
    GG = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2Gdvpa2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2Gdvperpdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2Gdvperp2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dGdvperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    HH = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dHdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dHdvperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    #Cflux_vpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    #Cflux_vperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    buffer_vpavperp_1 = allocate_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_float(nvpa,nvperp)
    Cssp_result_vpavperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dfdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2fdvpa2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2fdvperpdvpa = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    dfdvperp = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    d2fdvperp2 = allocate_shared_float(nvpa,nvperp; comm=comm_anyv_subblock[])
    
    return fokkerplanck_arrays_direct_integration_struct(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                               GG,d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,dGdvperp,
                               HH,dHdvpa,dHdvperp,buffer_vpavperp_1,buffer_vpavperp_2,
                               Cssp_result_vpavperp, dfdvpa, d2fdvpa2,
                               d2fdvperpdvpa, dfdvperp, d2fdvperp2)
end

"""
Function that initialises the arrays needed to calculate the Rosenbluth potentials
by direct integration. As this function is only supported to keep the testing
of the direct integration method, the struct 'fka' created here does not contain
all of the arrays necessary to compute the weak-form operator. This functionality
could be ported if necessary.
"""
function init_fokker_planck_collisions_direct_integration(vperp,vpa; precompute_weights=false, print_to_screen=false)
    fka = allocate_fokkerplanck_arrays_direct_integration(vperp,vpa)
    if vperp.n > 1 && precompute_weights
        init_Rosenbluth_potential_integration_weights!(fka.G0_weights, fka.G1_weights, fka.H0_weights, fka.H1_weights,
                                        fka.H2_weights, fka.H3_weights, vperp, vpa, print_to_screen=print_to_screen)
    end
    return fka
end

#################################################
# Functions associated with implicit timestepping
#################################################

"""
Function to setup nonlinear_solver struct for implicit
Fokker-Planck collisions. An input namelist of form

```
    [fokker_planck_collisions.nonlinear_solver]
    atol = 1.0e-10
    rtol = 0.0
    ...
```
is used, with the same defaults as the main
`[nonlinear_solver]` namelist, apart from `atol`
and `rtol`, which are set to their own defaults here.
"""
function setup_fp_nl_solve(implicit_ion_fp_collisions::Bool,
        # the main input namelist in dict format
        input_dict::OptionsDict,
        #coords = (vperp=vperp,vpa=vpa)
        coords)
    # section name in TOML input for nonlinear solver
    # options for FP collisions
    section_name = "fokker_planck_collisions_nonlinear_solver"
    # Default values of atol, rtol
    # for which implicit FP collisions have
    # been tested. Taking large timesteps
    # with smaller atol challenges the solver
    # with bc="zero".
    default_atol = 1.0e-10
    default_rtol = 0.0
    return setup_nonlinear_solve(
        implicit_ion_fp_collisions,
        input_dict,
        coords; serial_solve=false, anyv_region=true,
        section_name = section_name,
        default_atol=default_atol, default_rtol=default_rtol)
end

function implicit_ion_fokker_planck_self_collisions!(pdf_out, pdf_in, dSdt, 
                    composition, collisions, fkpl_arrays, 
                    vpa, vperp, z, r, delta_t, spectral_objects,
                    nl_solver_params; diagnose_entropy_production=false,
                    test_linearised_advance=false,
                    test_particle_preconditioner=false,
                    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
                    use_end_of_step_corrections=true)
    # bounds checking
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))

    # diagnostic parameter for user
    success = true

    # masses and collision frequencies
    ms, msp = 1.0, 1.0 # generalise!
    nuref = collisions.fkpl.nuii # generalise!
    Zi = collisions.fkpl.Zi # generalise!
    nuss = nuref*(Zi^4) # include charge number factor for self collisions
    use_conserving_corrections = collisions.fkpl.use_conserving_corrections
    boundary_data_option = collisions.fkpl.boundary_data_option

    # coords for vperp vpa newton_solve!
    coords = (vperp=vperp,vpa=vpa)
    # N.B. parallelisation using special 'anyv' region
    @begin_s_r_z_anyv_region()
    @loop_s_r_z is ir iz begin
        @views Fold = pdf_in[:,:,iz,ir,is]
        local_success = fokker_planck_self_collisions_backward_euler_step!(Fold, delta_t, ms, nuss, fkpl_arrays,
                            coords, spectral_objects,
                            nl_solver_params;
                            test_numerical_conserving_terms=use_conserving_corrections,
                            test_linearised_advance=test_linearised_advance,
                            test_particle_preconditioner=test_particle_preconditioner,
                            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
                            boundary_data_option=boundary_data_option)
        # global success true only if local_success true
        success = success && local_success
        Fnew = fkpl_arrays.Fnew
        if use_conserving_corrections && use_end_of_step_corrections
            # ad-hoc end-of-step corrections
            @begin_anyv_vperp_vpa_region()
            deltaF = fkpl_arrays.rhsvpavperp
            @loop_vperp_vpa ivperp ivpa begin
                deltaF[ivpa,ivperp] = Fnew[ivpa,ivperp] - Fold[ivpa,ivperp]
            end
            # correct deltaF = F^n+1 - F^n so it has no change in moments n, u, p
            # this introduces errors of the size of the distance between F^n+1 and the 
            # "correct" root that should have been found by the iterative solve, i.e.,
            # errors of size ~ atol.
            conserving_corrections!(deltaF, Fold, vpa, vperp,
                                fkpl_arrays.S_dummy)
            # update Fnew
            @loop_vperp_vpa ivperp ivpa begin
                Fnew[ivpa,ivperp] = deltaF[ivpa,ivperp] + Fold[ivpa,ivperp]
            end
        end
        
        # store Fnew = F^n+1 in the appropriate distribution function array
        @begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] = Fnew[ivpa,ivperp]
        end

        if diagnose_entropy_production
            calculate_entropy_production!(dSdt, pdf_out, fkpl_arrays, vpa, vperp,
                                        iz, ir, is)
        end
    end
    return success
end

function fokker_planck_self_collisions_backward_euler_step!(Fold, delta_t, ms, nuss, fkpl_arrays,
                        coords, spectral,
                        nl_solver_params;
                        test_numerical_conserving_terms=false,
                        test_linearised_advance=false,
                        test_particle_preconditioner=false,
                        use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
                        boundary_data_option=multipole_expansion)

    vperp, vperp_spectral = coords.vperp, spectral.vperp_spectral
    vpa, vpa_spectral = coords.vpa, spectral.vpa_spectral

    # residual function to be used for Newton-Krylov
    # residual(vpa, vperp) = F^(n+1) - F^n - dt * C[F^n+1,F^n+1]
    function residual_func!(Fresidual, Fnew; krylov=false)
        fokker_planck_self_collision_operator_weak_form!(
                        Fnew, ms, nuss,
                        fkpl_arrays, vperp, vpa,
                        vperp_spectral, vpa_spectral; 
                        boundary_data_option=boundary_data_option,
                        use_conserving_corrections=test_numerical_conserving_terms)

        @begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            Fresidual[ivpa,ivperp] = Fnew[ivpa,ivperp] - Fold[ivpa,ivperp] - delta_t * (fkpl_arrays.CC[ivpa,ivperp])
        end
        return nothing
    end

    if test_particle_preconditioner
        # test particle preconditioner CC2D_sparse is the matrix
        # K_ijkl = int phi_i(vpa)phi_j(vperp) ( phi_k(vpa)phi_l(vperp) - dt C[ phi_k(vpa)phi_l(vperp) , F^n(vpa,vperp) ])  vperp d vperp d vpa,
        # such that K * F^n+1 = M * F^n advances the linearised collision operator due
        # to test particle collisions only (differential piece of C).
        # CC2D_sparse is the approximate Jacobian for the residual Fresidual.
        calculate_test_particle_preconditioner!(Fold,delta_t,ms,ms,nuss,
            vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays, 
            use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
            boundary_data_option=boundary_data_option)
    
        function test_particle_precon!(x)
            # let K * dF = C[dF,F^n]
            # function to solve K * F^n+1 = M * F^n
            # and return F^n+1 in place in x
            pdf = x
            advance_linearised_test_particle_collisions!(pdf,fkpl_arrays,
                                vpa,vperp,vpa_spectral,vperp_spectral)
            return nothing
        end 
        right_preconditioner = test_particle_precon!
    else
        right_preconditioner = nothing
    end
    # initial condition for Fnew for JFNK or linearised advance below
    Fnew = fkpl_arrays.Fnew
    @begin_anyv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        Fnew[ivpa,ivperp] = Fold[ivpa,ivperp]
    end
    if test_linearised_advance
        test_particle_precon!(Fnew)
    else
        Fresidual = fkpl_arrays.Fresidual
        F_delta_x = fkpl_arrays.F_delta_x
        F_rhs_delta = fkpl_arrays.F_rhs_delta
        Fv = fkpl_arrays.Fv
        Fw = fkpl_arrays.Fw
        success = newton_solve!(Fnew, residual_func!, Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params;
                    coords, right_preconditioner=right_preconditioner)
        # apply BCs on result, if non-natural BCs are imposed
        # should only introduce error of order ~ atol
        enforce_vpavperp_BCs!(Fnew,vpa,vperp,vpa_spectral,vperp_spectral)
    end
    @_anyv_subblock_synchronize()
    return success
end

end
