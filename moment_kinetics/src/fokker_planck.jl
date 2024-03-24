"""
module for including the Full-F Fokker-Planck Collision Operator

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

"""
module fokker_planck


export init_fokker_planck_collisions, fokkerplanck_arrays_struct
export init_fokker_planck_collisions_weak_form
export explicit_fokker_planck_collisions_weak_form!
export explicit_fokker_planck_collisions!
export calculate_Maxwellian_Rosenbluth_coefficients
export get_local_Cssp_coefficients!, init_fokker_planck_collisions
# testing
export symmetric_matrix_inverse
export fokker_planck_collision_operator_weak_form!

using SpecialFunctions: ellipk, ellipe, erf
using FastGaussQuadrature
using Dates
using LinearAlgebra: lu, ldiv!
using MPI
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication
using ..velocity_moments: integrate_over_vspace
using ..velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_qpar, get_pressure, get_rmom
using ..looping
using ..fokker_planck_calculus: init_Rosenbluth_potential_integration_weights!
using ..fokker_planck_calculus: init_Rosenbluth_potential_boundary_integration_weights!
using ..fokker_planck_calculus: allocate_boundary_integration_weights
using ..fokker_planck_calculus: allocate_rosenbluth_potential_boundary_data
using ..fokker_planck_calculus: fokkerplanck_arrays_direct_integration_struct, fokkerplanck_weakform_arrays_struct
using ..fokker_planck_calculus: assemble_matrix_operators_dirichlet_bc
using ..fokker_planck_calculus: assemble_matrix_operators_dirichlet_bc_sparse
using ..fokker_planck_calculus: assemble_explicit_collision_operator_rhs_serial!
using ..fokker_planck_calculus: assemble_explicit_collision_operator_rhs_parallel!
using ..fokker_planck_calculus: assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!
using ..fokker_planck_calculus: calculate_YY_arrays, enforce_vpavperp_BCs!
using ..fokker_planck_calculus: calculate_rosenbluth_potential_boundary_data!
using ..fokker_planck_calculus: enforce_zero_bc!, elliptic_solve!, algebraic_solve!
using ..fokker_planck_calculus: calculate_rosenbluth_potentials_via_elliptic_solve!
using ..fokker_planck_test: Cssp_fully_expanded_form, calculate_collisional_fluxes, H_Maxwellian, dGdvperp_Maxwellian
using ..fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian
using ..fokker_planck_test: F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian

########################################################
# begin functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
########################################################

"""
function that initialises the arrays needed for Fokker Planck collisions
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
    
    fka = fokkerplanck_weakform_arrays_struct(bwgt,rpbd,MM2D_sparse,KKpar2D_sparse,KKperp2D_sparse,
                                           KKpar2D_with_BC_terms_sparse,KKperp2D_with_BC_terms_sparse,
                                           LP2D_sparse,LV2D_sparse,LB2D_sparse,PUperp2D_sparse,PPparPUperp2D_sparse,
                                           PPpar2D_sparse,MMparMNperp2D_sparse,KPperp2D_sparse,
                                           lu_obj_MM,lu_obj_LP,lu_obj_LV,lu_obj_LB,
                                           YY_arrays, S_dummy, Q_dummy, rhsvpavperp, rhsvpavperp_copy1, rhsvpavperp_copy2, rhsvpavperp_copy3,
                                           CC, GG, HH, dHdvpa, dHdvperp, dGdvperp, d2Gdvperp2, d2Gdvpa2, d2Gdvperpdvpa,
                                           FF, dFdvpa, dFdvperp)
    return fka
end

"""
Function for advancing with the explicit, weak-form, self-collision operator
"""
function explicit_fokker_planck_collisions_weak_form!(pdf_out,pdf_in,dSdt,composition,collisions,dt,
                                             fkpl_arrays::fokkerplanck_weakform_arrays_struct,
                                             r, z, vperp, vpa, vperp_spectral, vpa_spectral, scratch_dummy;
                                             test_assembly_serial=false,impose_zero_gradient_BC=false,
                                             diagnose_entropy_production=false)
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
    nussp = collisions.nuii # generalise!
    # N.B. parallelisation using special 'anyv' region
    begin_s_r_z_anyv_region()
    @loop_s_r_z is ir iz begin
        # first argument is Fs, and second argument is Fs' in C[Fs,Fs'] 
        @views fokker_planck_collision_operator_weak_form!(
            pdf_in[:,:,iz,ir,is], pdf_in[:,:,iz,ir,is], ms, msp, nussp, fkpl_arrays,
            vperp, vpa, vperp_spectral, vpa_spectral)
        # enforce the boundary conditions on CC before it is used for timestepping
        enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral)
        # make ad-hoc conserving corrections
        conserving_corrections!(fkpl_arrays.CC, pdf_in[:,:,iz,ir,is], vpa, vperp,
                                fkpl_arrays.S_dummy)
        
        # advance this part of s,r,z with the resulting C[Fs,Fs]
        begin_anyv_vperp_vpa_region()
        CC = fkpl_arrays.CC
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] += dt*CC[ivpa,ivperp]
        end
        if diagnose_entropy_production
            # assign dummy array
            lnfC = fkpl_arrays.rhsvpavperp
            @loop_vperp_vpa ivperp ivpa begin
                lnfC[ivpa,ivperp] = log(abs(pdf_in[ivpa,ivperp,iz,ir,is]) + 1.0e-15)*CC[ivpa,ivperp]
            end
            begin_anyv_region()
            @anyv_serial_region begin
                dSdt[iz,ir,is] = -get_density(lnfC,vpa,vperp)
            end
        end
    end
    return nothing
end


"""
Function for evaluating \$C_{ss'} = C_{ss'}[F_s,F_{s'}]\$

The result is stored in the array `fkpl_arrays.CC`.

The normalised collision frequency is defined by
```math
\\nu_{ss'} = \\frac{\\gamma_{ss'} n_\\mathrm{ref}}{2 m_s^2 c_\\mathrm{ref}^3}
```
with \$\\gamma_{ss'} = 2 \\pi (Z_s Z_{s'})^2 e^4 \\ln \\Lambda_{ss'} / (4 \\pi
\\epsilon_0)^2\$.
"""
function fokker_planck_collision_operator_weak_form!(ffs_in,ffsp_in,ms,msp,nussp,
                                             fkpl_arrays::fokkerplanck_weakform_arrays_struct,
                                             vperp, vpa, vperp_spectral, vpa_spectral;
                                             test_assembly_serial=false,
                                             use_Maxwellian_Rosenbluth_coefficients=false,
                                             use_Maxwellian_field_particle_distribution=false,
                                             algebraic_solve_for_d2Gdvperp2 = false,
                                             calculate_GG=false,
                                             calculate_dGdvperp=false)
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
        begin_anyv_region()
        dens = get_density(ffsp_in,vpa,vperp)
        upar = get_upar(ffsp_in, vpa, vperp, dens)
        ppar = get_ppar(ffsp_in, vpa, vperp, upar)
        pperp = get_pperp(ffsp_in, vpa, vperp)
        pressure = get_pressure(ppar,pperp)
        vth = sqrt(2.0*pressure/dens)
        begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            HH[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvpa2[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperp2[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dGdvperp[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperpdvpa[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvpa[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvperp[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
        # Need to synchronize as these arrays may be read outside the locally-owned set of
        # ivperp, ivpa indices in assemble_explicit_collision_operator_rhs_parallel!()
        _anyv_subblock_synchronize()
    else
        calculate_rosenbluth_potentials_via_elliptic_solve!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays,
             algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
             calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
    end
    # assemble the RHS of the collision operator matrix eq
    if use_Maxwellian_field_particle_distribution
        begin_anyv_region()
        dens = get_density(ffs_in,vpa,vperp)
        upar = get_upar(ffs_in, vpa, vperp, dens)
        ppar = get_ppar(ffs_in, vpa, vperp, upar)
        pperp = get_pperp(ffs_in, vpa, vperp)
        pressure = get_pressure(ppar,pperp)
        vth = sqrt(2.0*pressure/dens)
        begin_anyv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            FF[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dFdvpa[ivpa,ivperp] = dFdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dFdvperp[ivpa,ivperp] = dFdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
        # Need to synchronize as FF, dFdvpa, dFdvperp may be read outside the
        # locally-owned set of ivperp, ivpa indices in
        # assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!()
        _anyv_subblock_synchronize()
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
        _anyv_subblock_synchronize()
        assemble_explicit_collision_operator_rhs_parallel!(rhsvpavperp,ffs_in,
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    end
    # solve the collision operator matrix eq
    begin_anyv_region()
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

# solves A x = b for a matrix of the form
# A00  0    A02
# 0    A11  A12
# A02  A12  A22
# appropriate for the moment numerical conserving terms
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

# solves A x = b for a matrix of the form
# A00  A01  A02
# A01  A11  A12
# A02  A12  A22
# appropriate for the moment numerical conserving terms
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

function conserving_corrections!(CC,pdf_in,vpa,vperp,dummy_vpavperp)
    begin_anyv_region()
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
    MPI.Bcast!(param_vec, 0, comm_anyv_subblock[])
    (x0, x1, x2, upar) = param_vec
    
    # correct CC
    begin_anyv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        wpar = vpa.grid[ivpa] - upar
        CC[ivpa,ivperp] -= (x0 + x1*wpar + x2*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_in[ivpa,ivperp]
    end
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
allocate the required ancilliary arrays 
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
function that initialises the arrays needed to calculate the Rosenbluth potentials
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


end
