"""
module for including the Full-F Fokker-Planck Collision Operator
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
using LinearAlgebra: lu
using ..initial_conditions: enforce_boundary_conditions!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: MPISharedArray, global_rank
using ..velocity_moments: integrate_over_vspace
using ..velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_qpar, get_pressure, get_rmom
using ..calculus: derivative!, second_derivative!
using ..looping
using ..fokker_planck_calculus: init_Rosenbluth_potential_integration_weights!
using ..fokker_planck_calculus: init_Rosenbluth_potential_boundary_integration_weights!
using ..fokker_planck_calculus: allocate_boundary_integration_weights
using ..fokker_planck_calculus: allocate_rosenbluth_potential_boundary_data
using ..fokker_planck_calculus: fokkerplanck_arrays_struct, fokkerplanck_weakform_arrays_struct
using ..fokker_planck_calculus: assemble_matrix_operators_dirichlet_bc
using ..fokker_planck_calculus: assemble_matrix_operators_dirichlet_bc_sparse
using ..fokker_planck_calculus: assemble_explicit_collision_operator_rhs_serial!
using ..fokker_planck_calculus: assemble_explicit_collision_operator_rhs_parallel!
using ..fokker_planck_calculus: assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!
using ..fokker_planck_calculus: calculate_YY_arrays, enforce_vpavperp_BCs!
using ..fokker_planck_calculus: calculate_rosenbluth_potential_boundary_data!
using ..fokker_planck_calculus: enforce_zero_bc!, elliptic_solve!, algebraic_solve!, ravel_c_to_vpavperp_parallel!
using ..fokker_planck_test: Cssp_fully_expanded_form, calculate_collisional_fluxes, H_Maxwellian, dGdvperp_Maxwellian
using ..fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian
using ..fokker_planck_test: F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
"""
allocate the required ancilliary arrays 
"""

function allocate_fokkerplanck_arrays(vperp,vpa)
    nvpa = vpa.n
    nvperp = vperp.n
    
    G0_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    G1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H0_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H2_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H3_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    #Rosenbluth_G = allocate_float(nvpa,nvperp)
    d2Gdvpa2 = allocate_shared_float(nvpa,nvperp)
    d2Gdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    d2Gdvperp2 = allocate_shared_float(nvpa,nvperp)
    dGdvperp = allocate_shared_float(nvpa,nvperp)
    #Rosenbluth_H = allocate_float(nvpa,nvperp)
    dHdvpa = allocate_shared_float(nvpa,nvperp)
    dHdvperp = allocate_shared_float(nvpa,nvperp)
    #Cflux_vpa = allocate_shared_float(nvpa,nvperp)
    #Cflux_vperp = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_1 = allocate_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_float(nvpa,nvperp)
    Cssp_result_vpavperp = allocate_shared_float(nvpa,nvperp)
    dfdvpa = allocate_shared_float(nvpa,nvperp)
    d2fdvpa2 = allocate_shared_float(nvpa,nvperp)
    d2fdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    dfdvperp = allocate_shared_float(nvpa,nvperp)
    d2fdvperp2 = allocate_shared_float(nvpa,nvperp)
    
    return fokkerplanck_arrays_struct(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                               d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,dGdvperp,
                               dHdvpa,dHdvperp,buffer_vpavperp_1,buffer_vpavperp_2,
                               Cssp_result_vpavperp, dfdvpa, d2fdvpa2,
                               d2fdvperpdvpa, dfdvperp, d2fdvperp2)
end



"""
function that initialises the arrays needed for Fokker Planck collisions
using numerical integration to compute the Rosenbluth potentials only
at the boundary and using an elliptic solve to obtain the potentials 
in the rest of the velocity space domain.
"""

function init_fokker_planck_collisions_weak_form(vpa,vperp,vpa_spectral,vperp_spectral; precompute_weights=false, test_dense_matrix_construction=false)
    bwgt = allocate_boundary_integration_weights(vpa,vperp)
    if vperp.n > 1 && precompute_weights
        @views init_Rosenbluth_potential_boundary_integration_weights!(bwgt.G0_weights, bwgt.G1_weights, bwgt.H0_weights, bwgt.H1_weights,
                                        bwgt.H2_weights, bwgt.H3_weights, vpa, vperp)
    end
    rpbd = allocate_rosenbluth_potential_boundary_data(vpa,vperp)
    if test_dense_matrix_construction
        MM2D_sparse, KKpar2D_sparse, KKperp2D_sparse, 
        KKpar2D_with_BC_terms_sparse, KKperp2D_with_BC_terms_sparse,
        LP2D_sparse, LV2D_sparse, LB2D_sparse, KPperp2D_sparse,
        PUperp2D_sparse, PPparPUperp2D_sparse, PPpar2D_sparse,
        MMparMNperp2D_sparse = assemble_matrix_operators_dirichlet_bc(vpa,vperp,vpa_spectral,vperp_spectral)
    else
        MM2D_sparse, KKpar2D_sparse, KKperp2D_sparse,
        KKpar2D_with_BC_terms_sparse, KKperp2D_with_BC_terms_sparse,
        LP2D_sparse, LV2D_sparse, LB2D_sparse, KPperp2D_sparse,
        PUperp2D_sparse, PPparPUperp2D_sparse, PPpar2D_sparse,
        MMparMNperp2D_sparse = assemble_matrix_operators_dirichlet_bc_sparse(vpa,vperp,vpa_spectral,vperp_spectral)
    end
    lu_obj_MM = lu(MM2D_sparse)
    lu_obj_LP = lu(LP2D_sparse)
    lu_obj_LV = lu(LV2D_sparse)
    lu_obj_LB = lu(LB2D_sparse)
    @serial_region begin
        if global_rank[] == 0
            println("finished LU decomposition initialisation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    
    YY_arrays = calculate_YY_arrays(vpa,vperp,vpa_spectral,vperp_spectral)
    @serial_region begin
        if global_rank[] == 0
            println("finished YY array calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    nvpa, nvperp = vpa.n, vperp.n
    nc = nvpa*nvperp
    S_dummy = allocate_shared_float(nvpa,nvperp)
    Q_dummy = allocate_shared_float(nvpa,nvperp)
    rhsvpavperp = allocate_shared_float(nvpa,nvperp)
    rhsc = allocate_shared_float(nc)
    rhqc = allocate_shared_float(nc)
    sc = allocate_shared_float(nc)
    qc = allocate_shared_float(nc)
    
    CC = allocate_shared_float(nvpa,nvperp)
    HH = allocate_shared_float(nvpa,nvperp)
    dHdvpa = allocate_shared_float(nvpa,nvperp)
    dHdvperp = allocate_shared_float(nvpa,nvperp)
    dGdvperp = allocate_shared_float(nvpa,nvperp)
    d2Gdvperp2 = allocate_shared_float(nvpa,nvperp)
    d2Gdvpa2 = allocate_shared_float(nvpa,nvperp)
    d2Gdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    
    FF = allocate_shared_float(nvpa,nvperp)
    dFdvpa = allocate_shared_float(nvpa,nvperp)
    dFdvperp = allocate_shared_float(nvpa,nvperp)
    
    fka = fokkerplanck_weakform_arrays_struct(bwgt,rpbd,MM2D_sparse,KKpar2D_sparse,KKperp2D_sparse,
                                           KKpar2D_with_BC_terms_sparse,KKperp2D_with_BC_terms_sparse,
                                           LP2D_sparse,LV2D_sparse,LB2D_sparse,PUperp2D_sparse,PPparPUperp2D_sparse,
                                           PPpar2D_sparse,MMparMNperp2D_sparse,KPperp2D_sparse,
                                           lu_obj_MM,lu_obj_LP,lu_obj_LV,lu_obj_LB,
                                           YY_arrays, S_dummy, Q_dummy, rhsvpavperp, rhsc, rhqc, sc, qc,
                                           CC, HH, dHdvpa, dHdvperp, dGdvperp, d2Gdvperp2, d2Gdvpa2, d2Gdvperpdvpa,
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
    Css = scratch_dummy.buffer_vpavperp_1
    # N.B. parallelisation is only over vpa vperp
    # ensure s, r, z are local before initiating the s, r, z loop
    begin_vperp_vpa_region()
    @loop_s_r_z is ir iz begin
        # the functions within this loop will call
        # begin_vpa_region(), begin_vperp_region(), begin_vperp_vpa_region(), begin_serial_region() to synchronise the shared-memory arrays
        # first argument is Fs, and second argument is Fs' in C[Fs,Fs'] 
        @views fokker_planck_collision_operator_weak_form!(pdf_in[:,:,iz,ir,is],pdf_in[:,:,iz,ir,is],ms,msp,nussp,
                                             fkpl_arrays,vperp,vpa,vperp_spectral,vpa_spectral)        
        # enforce the boundary conditions on CC before it is used for timestepping
        enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp,vpa_spectral,vperp_spectral)
        # make ad-hoc conserving corrections
        conserving_corrections!(fkpl_arrays.CC,pdf_in[:,:,iz,ir,is],vpa,vperp,scratch_dummy.dummy_vpavperp)
        
        # advance this part of s,r,z with the resulting C[Fs,Fs]
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            Css[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
            pdf_out[ivpa,ivperp,iz,ir,is] += dt*Css[ivpa,ivperp]
        end
        if diagnose_entropy_production
            # assign dummy array
            lnfC = fkpl_arrays.rhsvpavperp
            @loop_vperp_vpa ivperp ivpa begin
                lnfC[ivpa,ivperp] = log(abs(pdf_in[ivpa,ivperp,iz,ir,is]) + 1.0e-15)*Css[ivpa,ivperp]
            end
            begin_serial_region()
            @serial_region begin
                dSdt[iz,ir,is] = -get_density(lnfC,vpa,vperp)
            end
        end
    end
    return nothing
end

"""
Function for evaluating Css' = Css'[Fs,Fs']
"""
function fokker_planck_collision_operator_weak_form!(ffs_in,ffsp_in,ms,msp,nussp,
                                             fkpl_arrays::fokkerplanck_weakform_arrays_struct,
                                             vperp, vpa, vperp_spectral, vpa_spectral;
                                             test_assembly_serial=false,
                                             use_Maxwellian_Rosenbluth_coefficients=false,
                                             use_Maxwellian_field_particle_distribution=false,
                                             algebraic_solve_for_d2Gdvperp2 = false)
    @boundscheck vpa.n == size(ffsp_in,1) || throw(BoundsError(ffsp_in))
    @boundscheck vperp.n == size(ffsp_in,2) || throw(BoundsError(ffsp_in))
    @boundscheck vpa.n == size(ffs_in,1) || throw(BoundsError(ffs_in))
    @boundscheck vperp.n == size(ffs_in,2) || throw(BoundsError(ffs_in))
    
    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    MM2D_sparse = fkpl_arrays.MM2D_sparse
    KKpar2D_sparse = fkpl_arrays.KKpar2D_sparse
    KKperp2D_sparse = fkpl_arrays.KKperp2D_sparse
    LP2D_sparse = fkpl_arrays.LP2D_sparse
    LV2D_sparse = fkpl_arrays.LV2D_sparse
    PUperp2D_sparse = fkpl_arrays.PUperp2D_sparse
    PPparPUperp2D_sparse = fkpl_arrays.PPparPUperp2D_sparse
    PPpar2D_sparse = fkpl_arrays.PPpar2D_sparse
    MMparMNperp2D_sparse = fkpl_arrays.MMparMNperp2D_sparse
    KPperp2D_sparse = fkpl_arrays.KPperp2D_sparse
    lu_obj_MM = fkpl_arrays.lu_obj_MM
    lu_obj_LP = fkpl_arrays.lu_obj_LP
    lu_obj_LV = fkpl_arrays.lu_obj_LV
    lu_obj_LB = fkpl_arrays.lu_obj_LB
    
    S_dummy = fkpl_arrays.S_dummy
    Q_dummy = fkpl_arrays.Q_dummy
    rhsc = fkpl_arrays.rhsc
    rhqc = fkpl_arrays.rhqc
    sc = fkpl_arrays.sc
    qc = fkpl_arrays.qc
    rhsvpavperp = fkpl_arrays.rhsvpavperp
    YY_arrays = fkpl_arrays.YY_arrays    
    bwgt = fkpl_arrays.bwgt
    rpbd = fkpl_arrays.rpbd
    
    CC = fkpl_arrays.CC
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
        begin_serial_region()
        dens = get_density(ffsp_in,vpa,vperp)
        upar = get_upar(ffsp_in, vpa, vperp, dens)
        ppar = get_ppar(ffsp_in, vpa, vperp, upar)
        pperp = get_pperp(ffsp_in, vpa, vperp)
        pressure = get_pressure(ppar,pperp)
        vth = sqrt(2.0*pressure/dens)
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            HH[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvpa2[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperp2[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dGdvperp[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            d2Gdvperpdvpa[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvpa[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dHdvperp[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
    else
        # the functions within this loop will call
        # begin_vpa_region(), begin_vperp_region(), begin_vperp_vpa_region(), begin_serial_region() to synchronise the shared-memory arrays
        # calculate the boundary data
        calculate_rosenbluth_potential_boundary_data!(rpbd,bwgt,@view(ffsp_in[:,:]),vpa,vperp,vpa_spectral,vperp_spectral)
        # carry out the elliptic solves required
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            S_dummy[ivpa,ivperp] = -(4.0/sqrt(pi))*ffsp_in[ivpa,ivperp]
        end
        elliptic_solve!(HH,S_dummy,rpbd.H_data,
                    lu_obj_LP,MM2D_sparse,rhsc,sc,vpa,vperp)
        elliptic_solve!(dHdvpa,S_dummy,rpbd.dHdvpa_data,
                    lu_obj_LP,PPpar2D_sparse,rhsc,sc,vpa,vperp)
        elliptic_solve!(dHdvperp,S_dummy,rpbd.dHdvperp_data,
                    lu_obj_LV,PUperp2D_sparse,rhsc,sc,vpa,vperp)
        
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp]
        
        end
        #elliptic_solve!(G_M_num,S_dummy,rpbd.G_data,
        #            lu_obj_LP,MM2D_sparse,rhsc,sc,vpa,vperp)
        elliptic_solve!(d2Gdvpa2,S_dummy,rpbd.d2Gdvpa2_data,
                    lu_obj_LP,KKpar2D_sparse,rhsc,sc,vpa,vperp)
        elliptic_solve!(dGdvperp,S_dummy,rpbd.dGdvperp_data,
                    lu_obj_LV,PUperp2D_sparse,rhsc,sc,vpa,vperp)
        elliptic_solve!(d2Gdvperpdvpa,S_dummy,rpbd.d2Gdvperpdvpa_data,
                    lu_obj_LV,PPparPUperp2D_sparse,rhsc,sc,vpa,vperp)
        
        if algebraic_solve_for_d2Gdvperp2
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp] - d2Gdvpa2[ivpa,ivperp]
                Q_dummy[ivpa,ivperp] = -dGdvperp[ivpa,ivperp]
            end
            # use the algebraic solve function to find
            # d2Gdvperp2 = 2H - d2Gdvpa2 - (1/vperp)dGdvperp
            # using a weak form
            algebraic_solve!(d2Gdvperp2,S_dummy,Q_dummy,rpbd.d2Gdvperp2_data,
                        lu_obj_MM,MM2D_sparse,MMparMNperp2D_sparse,
                        rhsc,rhqc,sc,qc,vpa,vperp)
        else
            # solve a weak-form PDE for d2Gdvperp2
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp]
                Q_dummy[ivpa,ivperp] = 2.0*d2Gdvpa2[ivpa,ivperp]
            end
            elliptic_solve!(d2Gdvperp2,S_dummy,Q_dummy,rpbd.d2Gdvperp2_data,
                        lu_obj_LB,KPperp2D_sparse,MMparMNperp2D_sparse,
                        rhsc,rhqc,sc,qc,vpa,vperp)
        end
        begin_serial_region()
    end
    # assemble the RHS of the collision operator matrix eq
    if use_Maxwellian_field_particle_distribution
        begin_serial_region()
        dens = get_density(ffs_in,vpa,vperp)
        upar = get_upar(ffs_in, vpa, vperp, dens)
        ppar = get_ppar(ffs_in, vpa, vperp, upar)
        pperp = get_pperp(ffs_in, vpa, vperp)
        pressure = get_pressure(ppar,pperp)
        vth = sqrt(2.0*pressure/dens)
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            FF[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dFdvpa[ivpa,ivperp] = dFdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            dFdvperp[ivpa,ivperp] = dFdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
        end
        assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!(rhsc,rhsvpavperp,
          FF,dFdvpa,dFdvperp,
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    elseif test_assembly_serial
        assemble_explicit_collision_operator_rhs_serial!(rhsc,@view(ffs_in[:,:]),
          d2Gdvpa,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    else
        assemble_explicit_collision_operator_rhs_parallel!(rhsc,rhsvpavperp,@view(ffs_in[:,:]),
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    end
    # solve the collision operator matrix eq
    begin_serial_region()
    @serial_region begin
        # invert mass matrix and fill fc
        sc .= lu_obj_MM \ rhsc
    end
    ravel_c_to_vpavperp_parallel!(CC,sc,vpa.n)
    return nothing
end

"""
function that initialises the arrays needed for Fokker Planck collisions
"""

function init_fokker_planck_collisions(vperp,vpa; precompute_weights=false)
    fka = allocate_fokkerplanck_arrays(vperp,vpa)
    if vperp.n > 1 && precompute_weights
        @views init_Rosenbluth_potential_integration_weights!(fka.G0_weights, fka.G1_weights, fka.H0_weights, fka.H1_weights,
                                        fka.H2_weights, fka.H3_weights, vperp, vpa)
    end
    return fka
end

"""
calculates the (normalised) Rosenbluth potential coefficients d2Gdvpa2, d2Gdvperpdvpa, ..., dHdvperp for a Maxwellian inputs.
"""
function calculate_Maxwellian_Rosenbluth_coefficients(dens,upar,vth,vpa,vperp,ivpa,ivperp,n_ion_species) # Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
    # zero coefficients prior to looping over s'
    Rosenbluth_d2Gdvpa2 = 0.0
    Rosenbluth_d2Gdvperpdvpa = 0.0
    Rosenbluth_d2Gdvperp2 = 0.0
    Rosenbluth_dHdvpa = 0.0
    Rosenbluth_dHdvperp = 0.0
    
    # fill in value at (ivpa,ivperp)
    for isp in 1:n_ion_species
        Rosenbluth_d2Gdvpa2 += d2Gdvpa2(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_d2Gdvperpdvpa += d2Gdvperpdvpa(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_d2Gdvperp2 += d2Gdvperp2(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_dHdvpa += dHdvpa(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_dHdvperp += dHdvperp(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
    end
    return Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp
end


"""
Function to carry out the integration of the revelant
distribution functions to form the required coefficients
for the full-F operator. We assume that the weights are
precalculated. The function takes as arguments the arrays
of coefficients (which we fill), the required distributions,
the precomputed weights, the indicies of the `field' velocities,
and the sizes of the primed vpa and vperp coordinates arrays.
"""
function get_local_Cssp_coefficients!(d2Gspdvpa2,dGspdvperp,d2Gspdvperpdvpa,
                                        d2Gspdvperp2,dHspdvpa,dHspdvperp,
                                        dfspdvpa,dfspdvperp,d2fspdvperpdvpa,
                                        G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                                        ivpa,ivperp,nvpa,nvperp)
    d2Gspdvpa2[ivpa,ivperp] = 0.0
    dGspdvperp[ivpa,ivperp] = 0.0
    d2Gspdvperpdvpa[ivpa,ivperp] = 0.0
    d2Gspdvperp2[ivpa,ivperp] = 0.0
    dHspdvpa[ivpa,ivperp] = 0.0
    dHspdvperp[ivpa,ivperp] = 0.0
    for ivperpp in 1:nvperp
        for ivpap in 1:nvpa
            #d2Gspdvpa2[ivpa,ivperp] += G_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvpa2[ivpap,ivperpp]
            d2Gspdvpa2[ivpa,ivperp] += H3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
            dGspdvperp[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            d2Gspdvperpdvpa[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperpdvpa[ivpap,ivperpp]
            #d2Gspdvperp2[ivpa,ivperp] += G2_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperp2[ivpap,ivperpp] + G3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            d2Gspdvperp2[ivpa,ivperp] += H2_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            dHspdvpa[ivpa,ivperp] += H0_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
            dHspdvperp[ivpa,ivperp] += H1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
        end
    end
    return nothing
end


"""
Evaluate the Fokker Planck collision Operator
using dummy arrays to store the 5 required derivatives.
For a single species, ir, and iz, this routine leaves 
in place the fokkerplanck_arrays struct with testable 
distributions function derivatives, Rosenbluth potentials,
and collision operator in place.
"""
#returns distribution function advanced by the (normalised)
#  C[F_s,F_s'] =  C[F_s,F_s'](vpa,vperp) given inputs
#collision frequency nu_{ss'} = gamma_{ss'} n_{ref} / 2 (m_s)^2 (c_{ref})^3
#with gamma_ss' = 2 pi (Z_s Z_s')^2 e^4 ln \Lambda_{ss'} / (4 pi \epsilon_0)^2 

function explicit_fokker_planck_collisions!(pdf_out,pdf_in,dSdt,composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral, boundary_distributions, advance,
                                             vpa_advect, z_advect, r_advect; diagnose_entropy_production = true)
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
    
    # setup species information
    mass = Array{mk_float,1}(undef,n_ion_species)
    mass[1] = 1.0 # generalise!
    nussp = Array{mk_float,2}(undef,n_ion_species,n_ion_species)
    nussp[1,1] = collisions.nuii # generalise!
    
    # assign Cssp to a dummy array
    Cssp = scratch_dummy.dummy_s
    
    # first, compute the require derivatives and store in the buffer arrays
    dfdvpa = scratch_dummy.buffer_vpavperpzrs_1
    d2fdvpa2 = scratch_dummy.buffer_vpavperpzrs_2
    d2fdvperpdvpa = scratch_dummy.buffer_vpavperpzrs_3
    dfdvperp = scratch_dummy.buffer_vpavperpzrs_4
    d2fdvperp2 = scratch_dummy.buffer_vpavperpzrs_5
    logfC = scratch_dummy.buffer_vpavperpzrs_6

    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. dfdvpa[:,ivperp,iz,ir,is] = vpa.scratch
        @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
        @. d2fdvpa2[:,ivperp,iz,ir,is] = vpa.scratch2
    end
    if vpa.discretization == "gausslegendre_pseudospectral"
        @loop_s_r_z_vperp is ir iz ivperp begin
           @views second_derivative!(vpa.scratch2, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
           @. d2fdvpa2[:,ivperp,iz,ir,is] = vpa.scratch2 
        end
    end

    begin_s_r_z_vpa_region()

    @loop_s_r_z_vpa is ir iz ivpa begin
        @views derivative!(vperp.scratch, pdf_in[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. dfdvperp[ivpa,:,iz,ir,is] = vperp.scratch
        @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
        @. d2fdvperp2[ivpa,:,iz,ir,is] = vperp.scratch2
        @views derivative!(vperp.scratch, dfdvpa[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. d2fdvperpdvpa[ivpa,:,iz,ir,is] = vperp.scratch
    end

    # to permit moment conservation, store the current moments of pdf_out
    # this involves imposing the boundary conditions to the present pre-collisions pdf_out
    if collisions.numerical_conserving_terms == "density+u||+T" || collisions.numerical_conserving_terms == "density"  
        store_moments_in_buffer!(pdf_out,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    end
    # now parallelise over all dimensions and calculate the 
    # collision operator coefficients and the collision operator
    # in one loop, noting that we only require data local to 
    # each ivpa,ivperp,iz,ir,is now that the derivatives are precomputed
    fka = fokkerplanck_arrays
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z is ir iz begin
        @loop_vperp_vpa ivperp ivpa begin
            for isp in 1:n_ion_species # make sure to sum over all ion species
                # get the local (in ivpa, ivperp) values of the coeffs
                @views get_local_Cssp_coefficients!(fka.d2Gdvpa2,fka.dGdvperp,fka.d2Gdvperpdvpa,
                                            fka.d2Gdvperp2,fka.dHdvpa,fka.dHdvperp,
                                            dfdvpa[:,:,iz,ir,isp],dfdvperp[:,:,iz,ir,isp],d2fdvperpdvpa[:,:,iz,ir,isp],
                                            fka.G1_weights,fka.H0_weights,fka.H1_weights,fka.H2_weights,fka.H3_weights,
                                            ivpa,ivperp,vpa.n,vperp.n)
                
                (Cssp[isp] = Cssp_fully_expanded_form(nussp[is,isp],mass[is],mass[isp],
                                                d2fdvpa2[ivpa,ivperp,iz,ir,is],d2fdvperp2[ivpa,ivperp,iz,ir,is],d2fdvperpdvpa[ivpa,ivperp,iz,ir,is],dfdvpa[ivpa,ivperp,iz,ir,is],dfdvperp[ivpa,ivperp,iz,ir,is],pdf_in[ivpa,ivperp,iz,ir,is],
                                                fka.d2Gdvpa2[ivpa,ivperp],fka.d2Gdvperp2[ivpa,ivperp],fka.d2Gdvperpdvpa[ivpa,ivperp],fka.dGdvperp[ivpa,ivperp],
                                                fka.dHdvpa[ivpa,ivperp],fka.dHdvperp[ivpa,ivperp],pdf_in[ivpa,ivperp,iz,ir,isp],vperp.grid[ivperp]) )
                pdf_out[ivpa,ivperp,iz,ir,is] += dt*Cssp[isp]
                # for testing
                fka.Cssp_result_vpavperp[ivpa,ivperp] = Cssp[isp]
                fka.dfdvpa[ivpa,ivperp] = dfdvpa[ivpa,ivperp,iz,ir,is]
                fka.d2fdvpa2[ivpa,ivperp] = d2fdvpa2[ivpa,ivperp,iz,ir,is]
                fka.d2fdvperpdvpa[ivpa,ivperp] = d2fdvperpdvpa[ivpa,ivperp,iz,ir,is]
                fka.dfdvperp[ivpa,ivperp] = dfdvperp[ivpa,ivperp,iz,ir,is]
                fka.d2fdvperp2[ivpa,ivperp] = d2fdvperp2[ivpa,ivperp,iz,ir,is]
            end
            # store the entropy production
            # we use ln|f| to avoid problems with f < 0. This is ok if C_s[f,f] is small where f ~< 0
            # + 1.0e-15 in case f = 0 exactly
            #println(Cssp[:])
            #println(pdf_in[ivpa,ivperp,iz,ir,is])
            logfC[ivpa,ivperp,iz,ir,is] = log(abs(pdf_in[ivpa,ivperp,iz,ir,is]) + 1.0e-15)*sum(Cssp[:])
            #println(dfdvpa[ivpa,ivperp,iz,ir,is])
       end
    end
    if diagnose_entropy_production
        # compute entropy production diagnostic
        begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
           @views dSdt[iz,ir,is] = -integrate_over_vspace(logfC[:,:,iz,ir,is], vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
    end
    if collisions.numerical_conserving_terms == "density+u||+T"
        # use an ad-hoc numerical model to conserve density, upar, vth
        # a different model is required for inter-species collisions
        # simply conserving particle density may be more appropriate in the multi-species case
        apply_numerical_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    elseif collisions.numerical_conserving_terms == "density"
        apply_density_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    elseif !(collisions.numerical_conserving_terms == "none")
        println("ERROR: collisions.numerical_conserving_terms = ",collisions.numerical_conserving_terms," NOT SUPPORTED")
    end
    return nothing 
end

function explicit_fokker_planck_collisions_Maxwellian_coefficients!(pdf_out,pdf_in,dens_in,upar_in,vth_in,
                                             composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral)
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
    
    mi = 1.0 # generalise this to an Array with size n_ion_species
    mip = 1.0 # generalise this to an Array with size n_ion_species
    cfreqii = collisions.nuii # generalise this to an Array with size (n_ion_species,n_ion_species)
    fk = fokkerplanck_arrays
    Cssp_result_vpavperp = scratch_dummy.dummy_vpavperp
    pdf_buffer_1 = scratch_dummy.buffer_vpavperpzrs_1
    pdf_buffer_2 = scratch_dummy.buffer_vpavperpzrs_2
    
    # precompute derivatives of the pdfs to benefit from parallelisation
    # d F / d vpa
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. pdf_buffer_1[:,ivperp,iz,ir,is] = vpa.scratch
    end
    # d F / d vperp
    begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views derivative!(vperp.scratch, pdf_in[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. pdf_buffer_2[ivpa,:,iz,ir,is] = vperp.scratch
    end
    
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z is ir iz begin
        @loop_vperp_vpa ivperp ivpa begin
            # first compute local (in z,r) Rosenbluth potential coefficients, summing over all s'
            ((Rosenbluth_d2Gdvpa2, Rosenbluth_d2Gdvperpdvpa, 
            Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,
            Rosenbluth_dHdvperp) = calculate_Maxwellian_Rosenbluth_coefficients(dens_in[iz,ir,:],
                 upar_in[iz,ir,:],vth_in[iz,ir,:],vpa,vperp,ivpa,ivperp,n_ion_species) )
                 
            # now form the collisional fluxes at this s,z,r
            ( (Cflux_vpa,Cflux_vperp) = calculate_collisional_fluxes(pdf_in[ivpa,ivperp,iz,ir,is],
                    pdf_buffer_1[ivpa,ivperp,iz,ir,is],pdf_buffer_2[ivpa,ivperp,iz,ir,is],
                    Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,
                    Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
                    mi,mip) )
            
            # now overwrite the buffer arrays with the local values as we no longer need dFdvpa or dFdvperp at s,r,z
            pdf_buffer_1[ivpa,ivperp,iz,ir,is] = Cflux_vpa
            pdf_buffer_2[ivpa,ivperp,iz,ir,is] = Cflux_vperp
        end
        
    end
    
    # now differentiate the fluxes to obtain the explicit operator 
    
    # d Cflux_vpa / d vpa
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_buffer_1[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. pdf_buffer_1[:,ivperp,iz,ir,is] = vpa.scratch
    end
    # (1/vperp) d Cflux_vperp / d vperp
    begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views @. vperp.scratch2 = vperp.grid*pdf_buffer_2[ivpa,:,iz,ir,is]
        @views derivative!(vperp.scratch, vperp.scratch2, vperp, vperp_spectral)
        @. pdf_buffer_2[ivpa,:,iz,ir,is] = vperp.scratch[:]/vperp.grid[:]
    end
    
    # now add the result to the outgoing pdf
    # d F / d t = nu_ii * ( d Cflux_vpa / d vpa + (1/vperp) d Cflux_vperp / d vperp)
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] += dt*cfreqii*(pdf_buffer_1[ivpa,ivperp,iz,ir,is] + pdf_buffer_1[ivpa,ivperp,iz,ir,is])
    end
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
    # compute moments of the input pdf
    dens =  get_density(@view(pdf_in[:,:]), vpa, vperp)
    upar = get_upar(@view(pdf_in[:,:,]), vpa, vperp, dens)
    ppar = get_ppar(@view(pdf_in[:,:,]), vpa, vperp, upar)
    pperp = get_pperp(@view(pdf_in[:,:,]), vpa, vperp)
    pressure = get_pressure(ppar,pperp)
    qpar = get_qpar(@view(pdf_in[:,:,]), vpa, vperp, upar, dummy_vpavperp)
    rmom = get_rmom(@view(pdf_in[:,:,]), vpa, vperp, upar, dummy_vpavperp)
    
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
    
    # correct CC
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        wpar = vpa.grid[ivpa] - upar
        CC[ivpa,ivperp] -= (x0 + x1*wpar + x2*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_in[ivpa,ivperp]
    end
end

# applies the numerical conservation to pdf_out, the advanced distribution function
# uses the low-level moment integration routines from velocity moments
# conserves n, upar, total pressure of each species
# only correct for the self collision operator
# multi-species cases requires conservation of  particle number and total momentum and total energy ( sum_s m_s upar_s, ... )
function apply_numerical_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # enforce bc prior to imposing conservation
    enforce_boundary_conditions!(pdf_out, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # buffer arrays
    buffer_pdf = scratch_dummy.buffer_vpavperpzrs_1
    dummy_vpavperp = scratch_dummy.dummy_vpavperp
    # data precalculated by store_moments_in_buffer!
    buffer_n = scratch_dummy.buffer_zrs_1
    buffer_upar = scratch_dummy.buffer_zrs_2
    buffer_pressure = scratch_dummy.buffer_zrs_3
    mass = 1.0
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        # get moments of incoming and outgoing distribution functions
        n_in = buffer_n[iz,ir,is]
        upar_in = buffer_upar[iz,ir,is]
        pressure_in = buffer_pressure[iz,ir,is]
        
        n_out = get_density(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp)
        upar_out = get_upar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, n_out)
        ppar_out = get_ppar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar_out, mass)
        pperp_out = get_pperp(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, mass)
        pressure_out = get_pressure(ppar_out,pperp_out)
        qpar_out = get_qpar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar_out, mass, dummy_vpavperp)
        rmom_out = get_rmom(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar_out, mass, dummy_vpavperp)
        
        # form the appropriate matrix coefficients
        b0, b1, b2 = n_in, n_in*(upar_in - upar_out), (3.0/2.0)*(pressure_in/mass) + n_in*(upar_in - upar_out)^2
        A00, A02, A11, A12, A22 = n_out, (3.0/2.0)*(pressure_out/mass), 0.5*ppar_out/mass, qpar_out/mass, rmom_out/mass
        
        # obtain the coefficients for the corrections 
        (x0, x1, x2) = symmetric_matrix_inverse(A00,A02,A11,A12,A22,b0,b1,b2)
        
        # fill the buffer with the corrected pdf 
        @loop_vperp_vpa ivperp ivpa begin
            wpar = vpa.grid[ivpa] - upar_out
            buffer_pdf[ivpa,ivperp,iz,ir,is] = (x0 + x1*wpar + x2*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_out[ivpa,ivperp,iz,ir,is]
        end
        
    end
    begin_s_r_z_vperp_vpa_region()
    # update pdf_out
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] = buffer_pdf[ivpa,ivperp,iz,ir,is]
    end
    return nothing
end

# function which corrects only for the loss of particles due to numerical error
# suitable for use with multiple species collisions
function apply_density_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # enforce bc prior to imposing conservation
    enforce_boundary_conditions!(pdf_out, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # buffer array
    buffer_pdf = scratch_dummy.buffer_vpavperpzrs_1
    # data precalculated by store_moments_in_buffer!
    buffer_n = scratch_dummy.buffer_zrs_1
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        # get density moment of incoming and outgoing distribution functions
        n_in = buffer_n[iz,ir,is]
        
        n_out = get_density(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp)
        
        # obtain the coefficient for the corrections 
        x0 = n_in/n_out
        
        # update pdf_out with the corrections 
        @loop_vperp_vpa ivperp ivpa begin
            buffer_pdf[ivpa,ivperp,iz,ir,is] = x0*pdf_out[ivpa,ivperp,iz,ir,is]
        end
        
    end
    begin_s_r_z_vperp_vpa_region()
    # update pdf_out
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] = buffer_pdf[ivpa,ivperp,iz,ir,is]
    end
    return nothing
end

function store_moments_in_buffer!(pdf_out,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # enforce bc prior to calculating the moments
    enforce_boundary_conditions!(pdf_out, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # buffer arrays
    density = scratch_dummy.buffer_zrs_1
    upar = scratch_dummy.buffer_zrs_2
    pressure = scratch_dummy.buffer_zrs_3
    # set the mass
    mass = 1.0
    
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        density[iz,ir,is] = get_density(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp)
        upar[iz,ir,is] = get_upar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, density[iz,ir,is])
        ppar = get_ppar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar[iz,ir,is], mass)
        pperp = get_pperp(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, mass)
        pressure[iz,ir,is] = get_pressure(ppar,pperp)
    end
    return nothing
end
end
