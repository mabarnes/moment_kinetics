"""
module for functions used 
in calculating the integrals and doing 
the numerical differentiation for 
the implementation of the 
the Full-F Fokker-Planck Collision Operator
"""
module fokker_planck_calculus

export assemble_matrix_operators_dirichlet_bc
export assemble_matrix_operators_dirichlet_bc_sparse
export assemble_explicit_collision_operator_rhs_serial!
export assemble_explicit_collision_operator_rhs_parallel!
export assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!
export YY_collision_operator_arrays, calculate_YY_arrays
export calculate_rosenbluth_potential_boundary_data!
export elliptic_solve!, algebraic_solve!
export fokkerplanck_arrays_direct_integration_struct
export fokkerplanck_weakform_arrays_struct
export enforce_vpavperp_BCs!
export calculate_rosenbluth_potentials_via_elliptic_solve!

# testing
export calculate_rosenbluth_potential_boundary_data_exact!
export ravel_c_to_vpavperp!, ravel_vpavperp_to_c!
export enforce_zero_bc!, ravel_c_to_vpavperp_parallel!
export allocate_rosenbluth_potential_boundary_data
export calculate_rosenbluth_potential_boundary_data_exact!
export test_rosenbluth_potential_boundary_data

using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..calculus: derivative!
using ..communication
using ..communication: MPISharedArray, global_rank
using ..looping
using moment_kinetics.gauss_legendre: get_QQ_local!
using Dates
using SpecialFunctions: ellipk, ellipe
using SparseArrays: sparse, AbstractSparseArray
using SuiteSparse
using LinearAlgebra: mul!, LU
using FastGaussQuadrature
using Printf

function print_matrix(matrix,name::String,n::mk_int,m::mk_int)
    println("\n ",name," \n")
    for i in 1:n
        for j in 1:m
            @printf("%.2f ", matrix[i,j])
        end
        println("")
    end
    println("\n")
end

function print_vector(vector,name::String,m::mk_int)
    println("\n ",name," \n")
    for j in 1:m
        @printf("%.3f ", vector[j])
    end
    println("")
    println("\n")
end

"""
a struct of dummy arrays and precalculated coefficients
for the strong-form Fokker-Planck collision operator 
"""

struct fokkerplanck_arrays_direct_integration_struct
    G0_weights::MPISharedArray{mk_float,4}
    G1_weights::MPISharedArray{mk_float,4}
    H0_weights::MPISharedArray{mk_float,4}
    H1_weights::MPISharedArray{mk_float,4}
    H2_weights::MPISharedArray{mk_float,4}
    H3_weights::MPISharedArray{mk_float,4}
    GG::MPISharedArray{mk_float,2}
    d2Gdvpa2::MPISharedArray{mk_float,2}
    d2Gdvperpdvpa::MPISharedArray{mk_float,2}
    d2Gdvperp2::MPISharedArray{mk_float,2}
    dGdvperp::MPISharedArray{mk_float,2}
    HH::MPISharedArray{mk_float,2}
    dHdvpa::MPISharedArray{mk_float,2}
    dHdvperp::MPISharedArray{mk_float,2}
    #Cflux_vpa::MPISharedArray{mk_float,2}
    #Cflux_vperp::MPISharedArray{mk_float,2}
    buffer_vpavperp_1::Array{mk_float,2}
    buffer_vpavperp_2::Array{mk_float,2}
    Cssp_result_vpavperp::MPISharedArray{mk_float,2}
    dfdvpa::MPISharedArray{mk_float,2}
    d2fdvpa2::MPISharedArray{mk_float,2}
    d2fdvperpdvpa::MPISharedArray{mk_float,2}
    dfdvperp::MPISharedArray{mk_float,2}
    d2fdvperp2::MPISharedArray{mk_float,2}
end

"""
a struct to contain the integration weights for the boundary points
in the (vpa,vperp) domain
"""
struct boundary_integration_weights_struct
    lower_vpa_boundary::MPISharedArray{mk_float,3}
    upper_vpa_boundary::MPISharedArray{mk_float,3}
    upper_vperp_boundary::MPISharedArray{mk_float,3}
end

"""
a struct used for calculating the integration weights for the 
boundary of the velocity space domain in (vpa,vperp) coordinates
"""
struct fokkerplanck_boundary_data_arrays_struct
    G0_weights::boundary_integration_weights_struct
    G1_weights::boundary_integration_weights_struct
    H0_weights::boundary_integration_weights_struct
    H1_weights::boundary_integration_weights_struct
    H2_weights::boundary_integration_weights_struct
    H3_weights::boundary_integration_weights_struct
    dfdvpa::MPISharedArray{mk_float,2}
    d2fdvperpdvpa::MPISharedArray{mk_float,2}
    dfdvperp::MPISharedArray{mk_float,2}    
end

struct vpa_vperp_boundary_data
    lower_boundary_vpa::MPISharedArray{mk_float,1}
    upper_boundary_vpa::MPISharedArray{mk_float,1}
    upper_boundary_vperp::MPISharedArray{mk_float,1}
end

struct rosenbluth_potential_boundary_data
    H_data::vpa_vperp_boundary_data
    dHdvpa_data::vpa_vperp_boundary_data
    dHdvperp_data::vpa_vperp_boundary_data
    G_data::vpa_vperp_boundary_data
    dGdvperp_data::vpa_vperp_boundary_data
    d2Gdvperp2_data::vpa_vperp_boundary_data
    d2Gdvperpdvpa_data::vpa_vperp_boundary_data
    d2Gdvpa2_data::vpa_vperp_boundary_data
end

struct YY_collision_operator_arrays
    # let phi_j(vperp) be the jth Lagrange basis function, 
    # and phi'_j(vperp) the first derivative of the Lagrange basis function
    # on the iel^th element. Then, the arrays are defined as follows.
    # YY0perp[i,j,k,iel] = \int phi_i(vperp) phi_j(vperp) phi_k(vperp) vperp d vperp
    YY0perp::Array{mk_float,4}
    # YY1perp[i,j,k,iel] = \int phi_i(vperp) phi_j(vperp) phi'_k(vperp) vperp d vperp
    YY1perp::Array{mk_float,4}
    # YY2perp[i,j,k,iel] = \int phi_i(vperp) phi'_j(vperp) phi'_k(vperp) vperp d vperp
    YY2perp::Array{mk_float,4}
    # YY3perp[i,j,k,iel] = \int phi_i(vperp) phi'_j(vperp) phi_k(vperp) vperp d vperp
    YY3perp::Array{mk_float,4}
    # YY0par[i,j,k,iel] = \int phi_i(vpa) phi_j(vpa) phi_k(vpa) vpa d vpa
    YY0par::Array{mk_float,4}
    # YY1par[i,j,k,iel] = \int phi_i(vpa) phi_j(vpa) phi'_k(vpa) vpa d vpa
    YY1par::Array{mk_float,4}
    # YY2par[i,j,k,iel] = \int phi_i(vpa) phi'_j(vpa) phi'_k(vpa) vpa d vpa
    YY2par::Array{mk_float,4}
    # YY3par[i,j,k,iel] = \int phi_i(vpa) phi'_j(vpa) phi_k(vpa) vpa d vpa
    YY3par::Array{mk_float,4}
end

"""
a struct of dummy arrays and precalculated coefficients
for the weak-form Fokker-Planck collision operator 
"""
struct fokkerplanck_weakform_arrays_struct{N}
    # boundary weights (Green's function) data
    bwgt::fokkerplanck_boundary_data_arrays_struct
    # dummy arrays for boundary data calculation
    rpbd::rosenbluth_potential_boundary_data
    # assembled 2D weak-form matrices
    MM2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    KKpar2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    KKperp2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    KKpar2D_with_BC_terms_sparse::AbstractSparseArray{mk_float,mk_int,N}
    KKperp2D_with_BC_terms_sparse::AbstractSparseArray{mk_float,mk_int,N}
    LP2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    LV2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    LB2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    PUperp2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    PPparPUperp2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    PPpar2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    MMparMNperp2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    KPperp2D_sparse::AbstractSparseArray{mk_float,mk_int,N}
    # lu decomposition objects
    lu_obj_MM::SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int}
    lu_obj_LP::SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int}
    lu_obj_LV::SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int}
    lu_obj_LB::SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int}
    # elemental matrices for the assembly of C[Fs,Fsp]
    YY_arrays::YY_collision_operator_arrays
    # dummy arrays for elliptic solvers
    S_dummy::MPISharedArray{mk_float,2}
    Q_dummy::MPISharedArray{mk_float,2}
    rhsvpavperp::MPISharedArray{mk_float,2}
    rhsc::MPISharedArray{mk_float,1}
    rhqc::MPISharedArray{mk_float,1}
    sc::MPISharedArray{mk_float,1}
    qc::MPISharedArray{mk_float,1}
    # dummy array for the result of the calculation
    CC::MPISharedArray{mk_float,2}
    # dummy arrays for storing Rosenbluth potentials
    GG::MPISharedArray{mk_float,2}
    HH::MPISharedArray{mk_float,2}
    dHdvpa::MPISharedArray{mk_float,2}
    dHdvperp::MPISharedArray{mk_float,2}
    dGdvperp::MPISharedArray{mk_float,2}
    d2Gdvperp2::MPISharedArray{mk_float,2}
    d2Gdvpa2::MPISharedArray{mk_float,2}
    d2Gdvperpdvpa::MPISharedArray{mk_float,2}
    FF::MPISharedArray{mk_float,2}
    dFdvpa::MPISharedArray{mk_float,2}
    dFdvperp::MPISharedArray{mk_float,2}
end

function allocate_boundary_integration_weight(vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    lower_vpa_boundary = allocate_shared_float(nvpa,nvperp,nvperp)
    upper_vpa_boundary = allocate_shared_float(nvpa,nvperp,nvperp)
    upper_vperp_boundary = allocate_shared_float(nvpa,nvperp,nvpa)
    return boundary_integration_weights_struct(lower_vpa_boundary,
            upper_vpa_boundary, upper_vperp_boundary)
end

function allocate_boundary_integration_weights(vpa,vperp)
    G0_weights = allocate_boundary_integration_weight(vpa,vperp)
    G1_weights = allocate_boundary_integration_weight(vpa,vperp)
    H0_weights = allocate_boundary_integration_weight(vpa,vperp)
    H1_weights = allocate_boundary_integration_weight(vpa,vperp)
    H2_weights = allocate_boundary_integration_weight(vpa,vperp)
    H3_weights = allocate_boundary_integration_weight(vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    dfdvpa = allocate_shared_float(nvpa,nvperp)
    d2fdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    dfdvperp = allocate_shared_float(nvpa,nvperp)
    return fokkerplanck_boundary_data_arrays_struct(G0_weights,
            G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
            dfdvpa,d2fdvperpdvpa,dfdvperp)
end


"""
function that precomputes the required integration weights
"""
function init_Rosenbluth_potential_integration_weights!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,vperp,vpa;print_to_screen=true)
    
    x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre = setup_basic_quadratures(vpa,vperp,print_to_screen=print_to_screen)
    
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # precalculated weights, integrating over Lagrange polynomials
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)
        
        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                G1_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H0_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H1_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights[:,:,ivpa,ivperp],G1_weights[:,:,ivpa,ivperp],
                H0_weights[:,:,ivpa,ivperp],H1_weights[:,:,ivpa,ivperp],
                H2_weights[:,:,ivpa,ivperp],H3_weights[:,:,ivpa,ivperp],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    
    
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("finished weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    return nothing
end

"""
function for getting the basic quadratures used for the 
numerical integration of the Lagrange polynomials and the 
Green's function.
"""
function setup_basic_quadratures(vpa,vperp;print_to_screen=true)
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    
    # get Gauss-Legendre points and weights on (-1,1)
    ngrid = max(vpa.ngrid,vperp.ngrid)
    nquad = 2*ngrid
    x_legendre, w_legendre = gausslegendre(nquad)
    #nlaguerre = min(9,nquad) # to prevent points to close to the boundaries
    nlaguerre = nquad
    x_laguerre, w_laguerre = gausslaguerre(nlaguerre)
    
    x_vpa, w_vpa = Array{mk_float,1}(undef,4*nquad), Array{mk_float,1}(undef,4*nquad)
    x_vperp, w_vperp = Array{mk_float,1}(undef,4*nquad), Array{mk_float,1}(undef,4*nquad)
  
    return x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre
end


"""
function for getting the indices used to choose the integration
quadrature 
"""
function get_element_limit_indices(ivpa,ivperp,vpa,vperp)
    nelement_vpa, ngrid_vpa = vpa.nelement_local, vpa.ngrid
    nelement_vperp, ngrid_vperp = vperp.nelement_local, vperp.ngrid
    #limits where checks required to determine which divergence-safe grid is needed
    igrid_vpa, ielement_vpa = vpa.igrid[ivpa], vpa.ielement[ivpa]
    ielement_vpa_low = ielement_vpa - ng_low(igrid_vpa,ngrid_vpa)*nel_low(ielement_vpa,nelement_vpa)
    ielement_vpa_hi = ielement_vpa + ng_hi(igrid_vpa,ngrid_vpa)*nel_hi(ielement_vpa,nelement_vpa)
    #println("igrid_vpa: ielement_vpa: ielement_vpa_low: ielement_vpa_hi:", igrid_vpa," ",ielement_vpa," ",ielement_vpa_low," ",ielement_vpa_hi)
    igrid_vperp, ielement_vperp = vperp.igrid[ivperp], vperp.ielement[ivperp]
    ielement_vperp_low = ielement_vperp - ng_low(igrid_vperp,ngrid_vperp)*nel_low(ielement_vperp,nelement_vperp)
    ielement_vperp_hi = ielement_vperp + ng_hi(igrid_vperp,ngrid_vperp)*nel_hi(ielement_vperp,nelement_vperp)
    #println("igrid_vperp: ielement_vperp: ielement_vperp_low: ielement_vperp_hi:", igrid_vperp," ",ielement_vperp," ",ielement_vperp_low," ",ielement_vperp_hi)
    return igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, 
            igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi
end
"""
function that precomputes the required integration weights
only along the velocity space boundaries
"""
function init_Rosenbluth_potential_boundary_integration_weights!(G0_weights,
      G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,vpa,vperp;print_to_screen=true)
    
    x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre = setup_basic_quadratures(vpa,vperp,print_to_screen=print_to_screen)
    
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("beginning (boundary) weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # precalculate weights, integrating over Lagrange polynomials
    # first compute weights along lower vpa boundary
    begin_vperp_region()
    ivpa = 1 # lower_vpa_boundary
    @loop_vperp ivperp begin
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)
        
        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                G1_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H0_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H1_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H2_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H3_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights.lower_vpa_boundary[:,:,ivperp],
                G1_weights.lower_vpa_boundary[:,:,ivperp],
                H0_weights.lower_vpa_boundary[:,:,ivperp],
                H1_weights.lower_vpa_boundary[:,:,ivperp],
                H2_weights.lower_vpa_boundary[:,:,ivperp],
                H3_weights.lower_vpa_boundary[:,:,ivperp],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    # second compute weights along upper vpa boundary
    ivpa = vpa.n # upper_vpa_boundary
    @loop_vperp ivperp begin
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)
        
        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                G1_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H0_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H1_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H2_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H3_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights.upper_vpa_boundary[:,:,ivperp],
                G1_weights.upper_vpa_boundary[:,:,ivperp],
                H0_weights.upper_vpa_boundary[:,:,ivperp],
                H1_weights.upper_vpa_boundary[:,:,ivperp],
                H2_weights.upper_vpa_boundary[:,:,ivperp],
                H3_weights.upper_vpa_boundary[:,:,ivperp],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    # finally compute weight along upper vperp boundary
    begin_vpa_region()
    ivperp = vperp.n # upper_vperp_boundary
    @loop_vpa ivpa begin
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)
        
        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0  
                G1_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0  
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H0_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0  
                H1_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0  
                H2_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0  
                H3_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0  
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights.upper_vperp_boundary[:,:,ivpa],
                G1_weights.upper_vperp_boundary[:,:,ivpa],
                H0_weights.upper_vperp_boundary[:,:,ivpa],
                H1_weights.upper_vperp_boundary[:,:,ivpa],
                H2_weights.upper_vperp_boundary[:,:,ivpa],
                H3_weights.upper_vperp_boundary[:,:,ivpa],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    # return the parallelisation status to serial
    begin_serial_region()
    @serial_region begin 
        if global_rank[] == 0 && print_to_screen
            println("finished (boundary) weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    return nothing
end

function get_imin_imax(coord,iel)
    j = iel
    if j > 1
        k = 1
    else
        k = 0
    end
    imin = coord.imin[j] - k
    imax = coord.imax[j]
    return imin, imax
end

function get_nodes(coord,iel)
    # get imin and imax of this element on full grid
    (imin, imax) = get_imin_imax(coord,iel)
    nodes = coord.grid[imin:imax]
    return nodes
end
"""
Lagrange polynomial
args: 
j - index of l_j from list of nodes
x_nodes - array of x node values
x - point where interpolated value is returned
"""
function lagrange_poly(j,x_nodes,x)
    # get number of nodes
    n = size(x_nodes,1)
    # location where l(x0) = 1
    x0 = x_nodes[j]
    # evaluate polynomial
    poly = 1.0
    for i in 1:j-1
            poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
    end
    for i in j+1:n
            poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
    end
    return poly
end

function get_scaled_x_w!(x_scaled, w_scaled, x_legendre, w_legendre, x_laguerre, w_laguerre, node_min, node_max, nodes, igrid_coord, coord_val)
    #println("nodes ",nodes)
    zero = 1.0e-10 
    @. x_scaled = 0.0
    @. w_scaled = 0.0
    nnodes = size(nodes,1)
    nquad_legendre = size(x_legendre,1)
    nquad_laguerre = size(x_laguerre,1)
    # assume x_scaled, w_scaled are arrays of length 2*nquad
    # use only nquad points for most elements, but use 2*nquad for
    # elements with interior divergences
    #println("coord: ",coord_val," node_max: ",node_max," node_min: ",node_min) 
    if abs(coord_val - node_max) < zero # divergence at upper endpoint 
        node_cut = (nodes[nnodes-1] + nodes[nnodes])/2.0
        
        n = nquad_laguerre + nquad_legendre
        shift = 0.5*(node_min + node_cut)
        scale = 0.5*(node_cut - node_min)
        @. x_scaled[1:nquad_legendre] = scale*x_legendre + shift
        @. w_scaled[1:nquad_legendre] = scale*w_legendre

        @. x_scaled[1+nquad_legendre:n] = node_max + (node_cut - node_max)*exp(-x_laguerre)
        @. w_scaled[1+nquad_legendre:n] = (node_max - node_cut)*w_laguerre
        
        nquad_coord = n
        #println("upper divergence")
    elseif abs(coord_val - node_min) < zero # divergence at lower endpoint
        n = nquad_laguerre + nquad_legendre
        nquad = size(x_laguerre,1)
        node_cut = (nodes[1] + nodes[2])/2.0
        for j in 1:nquad_laguerre
            x_scaled[nquad_laguerre+1-j] = node_min + (node_cut - node_min)*exp(-x_laguerre[j])
            w_scaled[nquad_laguerre+1-j] = (node_cut - node_min)*w_laguerre[j]
        end
        shift = 0.5*(node_max + node_cut)
        scale = 0.5*(node_max - node_cut)
        @. x_scaled[1+nquad_laguerre:n] = scale*x_legendre + shift
        @. w_scaled[1+nquad_laguerre:n] = scale*w_legendre

        nquad_coord = n
        #println("lower divergence")
    else #if (coord_val - node_min)*(coord_val - node_max) < - zero # interior divergence
        #println(nodes[igrid_coord]," ", coord_val)
        n = 2*nquad_laguerre
        node_cut_high = (nodes[igrid_coord+1] + nodes[igrid_coord])/2.0
        if igrid_coord == 1
            # exception for vperp coordinate near orgin
            k = 0
            node_cut_low = node_min
            nquad_coord = nquad_legendre + 2*nquad_laguerre
        else
            # fill in lower Gauss-Legendre points
            node_cut_low = (nodes[igrid_coord-1]+nodes[igrid_coord])/2.0
            shift = 0.5*(node_cut_low + node_min)
            scale = 0.5*(node_cut_low - node_min)
            @. x_scaled[1:nquad_legendre] = scale*x_legendre + shift
            @. w_scaled[1:nquad_legendre] = scale*w_legendre
            k = nquad_legendre
            nquad_coord = 2*(nquad_laguerre + nquad_legendre)
        end
        # lower half of domain  
        for j in 1:nquad_laguerre  
            x_scaled[k+j] = coord_val + (node_cut_low - coord_val)*exp(-x_laguerre[j])
            w_scaled[k+j] = (coord_val - node_cut_low)*w_laguerre[j]
        end  
        # upper half of domain
        for j in 1:nquad_laguerre
            x_scaled[k+n+1-j] = coord_val + (node_cut_high - coord_val)*exp(-x_laguerre[j])
            w_scaled[k+n+1-j] = (node_cut_high - coord_val)*w_laguerre[j]
        end
        # fill in upper Gauss-Legendre points
        shift = 0.5*(node_cut_high + node_max)
        scale = 0.5*(node_max - node_cut_high)
        @. x_scaled[k+n+1:nquad_coord] = scale*x_legendre + shift
        @. w_scaled[k+n+1:nquad_coord] = scale*w_legendre
        
        #println("intermediate divergence")
    #else # no divergences
    #    nquad = size(x_legendre,1) 
    #    shift = 0.5*(node_min + node_max)
    #    scale = 0.5*(node_max - node_min)
    #    @. x_scaled[1:nquad] = scale*x_legendre + shift
    #    @. w_scaled[1:nquad] = scale*w_legendre
    #    #println("no divergence")
    #    nquad_coord = nquad
    end
    #println("x_scaled",x_scaled)
    #println("w_scaled",w_scaled)
    return nquad_coord
end

function get_scaled_x_w_no_divergences!(x_scaled, w_scaled, x_legendre, w_legendre, node_min, node_max)
    @. x_scaled = 0.0
    @. w_scaled = 0.0
    #println("coord: ",coord_val," node_max: ",node_max," node_min: ",node_min) 
    nquad = size(x_legendre,1) 
    shift = 0.5*(node_min + node_max)
    scale = 0.5*(node_max - node_min)
    @. x_scaled[1:nquad] = scale*x_legendre + shift
    @. w_scaled[1:nquad] = scale*w_legendre
    #println("x_scaled",x_scaled)
    #println("w_scaled",w_scaled)
    return nquad
end

# function returns 1 if igrid = 1 or 0 if 1 < igrid <= ngrid
function ng_low(igrid,ngrid)
    return floor(mk_int, (ngrid - igrid)/(ngrid - 1))
end
# function returns 1 if igrid = ngrid or 0 if 1 =< igrid < ngrid
function ng_hi(igrid,ngrid)
    return floor(mk_int, igrid/ngrid)
end
# function returns 1 for nelement >= ielement > 1, 0 for ielement =1 
function nel_low(ielement,nelement)
    return floor(mk_int, (ielement - 2 + nelement)/nelement)
end
# function returns 1 for nelement > ielement >= 1, 0 for ielement =nelement 
function nel_hi(ielement,nelement)
    return 1- floor(mk_int, ielement/nelement)
end

# base level function for computing the Green's function weights
# note the definitions of ellipe & ellipk
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipe`
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipk`
# `ellipe(m) = \int^{\pi/2}\_0 \sqrt{ 1 - m \sin^2(\theta)} d \theta`
# `ellipe(k) = \int^{\pi/2}\_0 \frac{1}{\sqrt{ 1 - m \sin^2(\theta)}} d \theta`

function local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            nquad_vpa,ielement_vpa,vpa_nodes,vpa, # info about primed vpa grids
                            nquad_vperp,ielement_vperp,vperp_nodes,vperp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # points and weights for primed (source) grids
                            vpa_val, vperp_val) # values and indices for unprimed (field) grids
    for igrid_vperp in 1:vperp.ngrid
        for igrid_vpa in 1:vpa.ngrid
            # get grid index for point on full grid  
            ivpap = vpa.igrid_full[igrid_vpa,ielement_vpa]   
            ivperpp = vperp.igrid_full[igrid_vperp,ielement_vperp]   
            # carry out integration over Lagrange polynomial at this node, on this element
            for kvperp in 1:nquad_vperp
                for kvpa in 1:nquad_vpa 
                    x_kvpa = x_vpa[kvpa]
                    x_kvperp = x_vperp[kvperp]
                    w_kvperp = w_vperp[kvperp]
                    w_kvpa = w_vpa[kvpa]
                    denom = (vpa_val - x_kvpa)^2 + (vperp_val + x_kvperp)^2 
                    mm = min(4.0*vperp_val*x_kvperp/denom,1.0 - 1.0e-15)
                    #mm = 4.0*vperp_val*x_kvperp/denom/(1.0 + 10^-15)
                    #mm = 4.0*vperp_val*x_kvperp/denom
                    prefac = sqrt(denom)
                    ellipe_mm = ellipe(mm) 
                    ellipk_mm = ellipk(mm) 
                    #if mm_test > 1.0
                    #    println("mm: ",mm_test," ellipe: ",ellipe_mm," ellipk: ",ellipk_mm)
                    #end
                    G_elliptic_integral_factor = 2.0*ellipe_mm*prefac/pi
                    G1_elliptic_integral_factor = -(2.0*prefac/pi)*( (2.0 - mm)*ellipe_mm - 2.0*(1.0 - mm)*ellipk_mm )/(3.0*mm)
                    #G2_elliptic_integral_factor = (2.0*prefac/pi)*( (7.0*mm^2 + 8.0*mm - 8.0)*ellipe_mm + 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                    #G3_elliptic_integral_factor = (2.0*prefac/pi)*( 8.0*(mm^2 - mm + 1.0)*ellipe_mm - 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                    H_elliptic_integral_factor = 2.0*ellipk_mm/(pi*prefac)
                    H1_elliptic_integral_factor = -(2.0/(pi*prefac))*( (mm-2.0)*(ellipk_mm/mm) + (2.0*ellipe_mm/mm) )
                    H2_elliptic_integral_factor = (2.0/(pi*prefac))*( (3.0*mm^2 - 8.0*mm + 8.0)*(ellipk_mm/(3.0*mm^2)) + (4.0*mm - 8.0)*ellipe_mm/(3.0*mm^2) )
                    lagrange_poly_vpa = lagrange_poly(igrid_vpa,vpa_nodes,x_kvpa)
                    lagrange_poly_vperp = lagrange_poly(igrid_vperp,vperp_nodes,x_kvperp)
                    
                    (G0_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        G_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    (G1_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        G1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    #(G2_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    G2_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    #(G3_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    G3_elliptic_integral_factor*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    (H0_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        H_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                        
                    (H1_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        H1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                        
                    (H2_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        (H1_elliptic_integral_factor*vperp_val - H2_elliptic_integral_factor*x_kvperp)*
                        x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    (H3_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        H_elliptic_integral_factor*(vpa_val - x_kvpa)*
                        x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    #(n_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                end
            end
        end
    end
    return nothing
end

function loop_over_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                            vperp,ielement_vperpp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                            x_legendre,w_legendre,x_laguerre,w_laguerre,
                            igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    vperp_nodes = get_nodes(vperp,ielement_vperpp)
    vperp_max = vperp_nodes[end]
    vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
    nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
    for ielement_vpap in 1:ielement_vpa_low-1 
        # do integration over part of the domain with no divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        @views local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
    end
    nquad_vperp = get_scaled_x_w!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
    for ielement_vpap in ielement_vpa_low:ielement_vpa_hi
    #for ielement_vpap in 1:vpa.nelement_local
        # use general grid function that checks divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        #nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        nquad_vpa = get_scaled_x_w!(x_vpa, w_vpa, x_legendre, w_legendre, x_laguerre, w_laguerre, vpa_min, vpa_max, vpa_nodes, igrid_vpa, vpa_val)
        @views local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
    end
    nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
    for ielement_vpap in ielement_vpa_hi+1:vpa.nelement_local
        # do integration over part of the domain with no divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        @views local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
                    
    end
    return nothing
end

function loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                            x_legendre,w_legendre,
                            vpa_val, vperp_val)
    for ielement_vpap in 1:vpa.nelement_local
        # do integration over part of the domain with no divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        @views local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
                    
    end
    return nothing
end

function loop_over_vperp_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    for ielement_vperpp in 1:ielement_vperp_low-1
        
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        @views loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                vpa_val, vperp_val)
    end
    for ielement_vperpp in ielement_vperp_low:ielement_vperp_hi
        
        #vperp_nodes = get_nodes(vperp,ielement_vperpp)
        #vperp_max = vperp_nodes[end]
        #vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
        #nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        #nquad_vperp = get_scaled_x_w!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
        @views loop_over_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperpp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    for ielement_vperpp in ielement_vperp_hi+1:vperp.nelement_local
        
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        @views loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                vpa_val, vperp_val)
    end
    return nothing
end

function loop_over_vperp_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    for ielement_vperpp in 1:vperp.nelement_local
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,nelement_vperp) 
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        @views loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                vpa_val, vperp_val)
    end
    return nothing
end 


# Array indices for compound 1D form 
function ic_func(ivpa::mk_int,ivperp::mk_int,nvpa::mk_int)
    return ivpa + nvpa*(ivperp-1)
end
function ivperp_func(ic::mk_int,nvpa::mk_int)
    return floor(Int64,(ic-1)/nvpa) + 1
end
function ivpa_func(ic::mk_int,nvpa::mk_int)
    ivpa = ic - nvpa*(ivperp_func(ic,nvpa) - 1)
    return ivpa
end

function ravel_vpavperp_to_c!(pdf_c,pdf_vpavperp,nvpa::mk_int,nvperp::mk_int)
    for ivperp in 1:nvperp
        for ivpa in 1:nvpa
            ic = ic_func(ivpa,ivperp,nvpa)
            pdf_c[ic] = pdf_vpavperp[ivpa,ivperp]
        end
    end
    return nothing
end

function ravel_vpavperp_to_c_parallel!(pdf_c,pdf_vpavperp,nvpa::mk_int)
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin 
        ic = ic_func(ivpa,ivperp,nvpa)
        pdf_c[ic] = pdf_vpavperp[ivpa,ivperp]
    end
    return nothing
end

function ravel_c_to_vpavperp!(pdf_vpavperp,pdf_c,nc::mk_int,nvpa::mk_int)
    for ic in 1:nc
        ivpa = ivpa_func(ic,nvpa)
        ivperp = ivperp_func(ic,nvpa)
        pdf_vpavperp[ivpa,ivperp] = pdf_c[ic]
    end
    return nothing
end

function ravel_c_to_vpavperp_parallel!(pdf_vpavperp,pdf_c,nvpa::mk_int)
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        ic = ic_func(ivpa,ivperp,nvpa) 
        pdf_vpavperp[ivpa,ivperp] = pdf_c[ic]
    end
    return nothing
end

function ivpa_global_func(ivpa_local::mk_int,ielement_vpa::mk_int,ngrid_vpa::mk_int)
    ivpa_global = ivpa_local + (ielement_vpa - 1)*(ngrid_vpa - 1)
    return ivpa_global
end

# function that returns the sparse matrix index
# used to directly construct the nonzero entries
# of a 2D assembled sparse matrix
function icsc_func(ivpa_local::mk_int,ivpap_local::mk_int,
                   ielement_vpa::mk_int,
                   ngrid_vpa::mk_int,nelement_vpa::mk_int,
                   ivperp_local::mk_int,ivperpp_local::mk_int,
                   ielement_vperp::mk_int,
                   ngrid_vperp::mk_int,nelement_vperp::mk_int)
    ntot_vpa = (nelement_vpa - 1)*(ngrid_vpa^2 - 1) + ngrid_vpa^2
    #ntot_vperp = (nelement_vperp - 1)*(ngrid_vperp^2 - 1) + ngrid_vperp^2
    
    icsc_vpa = ((ivpap_local - 1) + (ivpa_local - 1)*ngrid_vpa +
                (ielement_vpa - 1)*(ngrid_vpa^2 - 1))
    icsc_vperp = ((ivperpp_local - 1) + (ivperp_local - 1)*ngrid_vperp + 
                    (ielement_vperp - 1)*(ngrid_vperp^2 - 1))
    icsc = 1 + icsc_vpa + ntot_vpa*icsc_vperp
    return icsc
end

struct sparse_matrix_constructor
    # the Ith row
    II::Array{mk_float,1}
    # the Jth column
    JJ::Array{mk_float,1}
    # the data S[I,J]
    SS::Array{mk_float,1}
end

function allocate_sparse_matrix_constructor(nsparse::mk_int)
    II = Array{mk_int,1}(undef,nsparse)
    @. II = 0
    JJ = Array{mk_int,1}(undef,nsparse)
    @. JJ = 0
    SS = Array{mk_float,1}(undef,nsparse)
    @. SS = 0.0
    return sparse_matrix_constructor(II,JJ,SS)
end

function assign_constructor_data!(data::sparse_matrix_constructor,icsc::mk_int,ii::mk_int,jj::mk_int,ss::mk_float)
    data.II[icsc] = ii
    data.JJ[icsc] = jj
    data.SS[icsc] = ss
    return nothing
end
function assemble_constructor_data!(data::sparse_matrix_constructor,icsc::mk_int,ii::mk_int,jj::mk_int,ss::mk_float)
    data.II[icsc] = ii
    data.JJ[icsc] = jj
    data.SS[icsc] += ss
    return nothing
end

function create_sparse_matrix(data::sparse_matrix_constructor)
    return sparse(data.II,data.JJ,data.SS)
end

function allocate_boundary_data(vpa,vperp)
    lower_boundary_vpa = allocate_shared_float(vperp.n)
    upper_boundary_vpa = allocate_shared_float(vperp.n)
    upper_boundary_vperp = allocate_shared_float(vpa.n)
    return vpa_vperp_boundary_data(lower_boundary_vpa,
            upper_boundary_vpa,upper_boundary_vperp)
end


function assign_exact_boundary_data!(func_data::vpa_vperp_boundary_data,
                                        func_exact,vpa,vperp)
    begin_serial_region()
    nvpa = vpa.n
    nvperp = vperp.n
    @serial_region begin
        for ivperp in 1:nvperp
            func_data.lower_boundary_vpa[ivperp] = func_exact[1,ivperp]
            func_data.upper_boundary_vpa[ivperp] = func_exact[nvpa,ivperp]
        end
        for ivpa in 1:nvpa
            func_data.upper_boundary_vperp[ivpa] = func_exact[ivpa,nvperp]
        end
    end
    return nothing
end
    
function allocate_rosenbluth_potential_boundary_data(vpa,vperp)
    H_data = allocate_boundary_data(vpa,vperp)
    dHdvpa_data = allocate_boundary_data(vpa,vperp)
    dHdvperp_data = allocate_boundary_data(vpa,vperp)
    G_data = allocate_boundary_data(vpa,vperp)
    dGdvperp_data = allocate_boundary_data(vpa,vperp)
    d2Gdvperp2_data = allocate_boundary_data(vpa,vperp)
    d2Gdvperpdvpa_data = allocate_boundary_data(vpa,vperp)
    d2Gdvpa2_data = allocate_boundary_data(vpa,vperp)
    return rosenbluth_potential_boundary_data(H_data,dHdvpa_data,
        dHdvperp_data,G_data,dGdvperp_data,d2Gdvperp2_data,
        d2Gdvperpdvpa_data,d2Gdvpa2_data)
end

function calculate_rosenbluth_potential_boundary_data_exact!(rpbd::rosenbluth_potential_boundary_data,
  H_exact,dHdvpa_exact,dHdvperp_exact,G_exact,dGdvperp_exact,
  d2Gdvperp2_exact,d2Gdvperpdvpa_exact,d2Gdvpa2_exact,
  vpa,vperp)
    assign_exact_boundary_data!(rpbd.H_data,H_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.dHdvpa_data,dHdvpa_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.dHdvperp_data,dHdvperp_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.G_data,G_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.dGdvperp_data,dGdvperp_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.d2Gdvperp2_data,d2Gdvperp2_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.d2Gdvperpdvpa_data,d2Gdvperpdvpa_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.d2Gdvpa2_data,d2Gdvpa2_exact,vpa,vperp)
    return nothing
end


function calculate_boundary_data!(func_data::vpa_vperp_boundary_data,
                                        weight::MPISharedArray{mk_float,4},func_input,vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    #for ivperp in 1:nvperp
    begin_vperp_region()
    @loop_vperp ivperp begin
        func_data.lower_boundary_vpa[ivperp] = 0.0
        func_data.upper_boundary_vpa[ivperp] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.lower_boundary_vpa[ivperp] += weight[ivpap,ivperpp,1,ivperp]*func_input[ivpap,ivperpp]
                func_data.upper_boundary_vpa[ivperp] += weight[ivpap,ivperpp,nvpa,ivperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    #for ivpa in 1:nvpa
    begin_vpa_region()
    @loop_vpa ivpa begin
        func_data.upper_boundary_vperp[ivpa] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.upper_boundary_vperp[ivpa] += weight[ivpap,ivperpp,ivpa,nvperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    # return to serial parallelisation
    begin_serial_region()
    return nothing
end

function calculate_boundary_data!(func_data::vpa_vperp_boundary_data,
                                  weight::boundary_integration_weights_struct,
                                  func_input,vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    #for ivperp in 1:nvperp
    begin_vperp_region()
    @loop_vperp ivperp begin
        func_data.lower_boundary_vpa[ivperp] = 0.0
        func_data.upper_boundary_vpa[ivperp] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.lower_boundary_vpa[ivperp] += weight.lower_vpa_boundary[ivpap,ivperpp,ivperp]*func_input[ivpap,ivperpp]
                func_data.upper_boundary_vpa[ivperp] += weight.upper_vpa_boundary[ivpap,ivperpp,ivperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    #for ivpa in 1:nvpa
    begin_vpa_region()
    @loop_vpa ivpa begin
        func_data.upper_boundary_vperp[ivpa] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.upper_boundary_vperp[ivpa] += weight.upper_vperp_boundary[ivpap,ivperpp,ivpa]*func_input[ivpap,ivperpp]
            end
        end
    end
    # return to serial parallelisation
    begin_serial_region()
    return nothing
end

function calculate_rosenbluth_potential_boundary_data!(rpbd::rosenbluth_potential_boundary_data,
    fkpl::Union{fokkerplanck_arrays_direct_integration_struct,fokkerplanck_boundary_data_arrays_struct},pdf,vpa,vperp,vpa_spectral,vperp_spectral;
    calculate_GG=false,calculate_dGdvperp=false)
    # get derivatives of pdf
    dfdvperp = fkpl.dfdvperp
    dfdvpa = fkpl.dfdvpa
    d2fdvperpdvpa = fkpl.d2fdvperpdvpa
    #for ivpa in 1:vpa.n
    begin_vpa_region()
    @loop_vpa ivpa begin
        @views derivative!(vperp.scratch, pdf[ivpa,:], vperp, vperp_spectral)
        @. dfdvperp[ivpa,:] = vperp.scratch
    end
    begin_vperp_region()
    @loop_vperp ivperp begin
    #for ivperp in 1:vperp.n
        @views derivative!(vpa.scratch, pdf[:,ivperp], vpa, vpa_spectral)
        @. dfdvpa[:,ivperp] = vpa.scratch
        @views derivative!(vpa.scratch, dfdvperp[:,ivperp], vpa, vpa_spectral)
        @. d2fdvperpdvpa[:,ivperp] = vpa.scratch
    end
    # ensure data is synchronized
    begin_serial_region()
    # carry out the numerical integration 
    calculate_boundary_data!(rpbd.H_data,fkpl.H0_weights,pdf,vpa,vperp)
    calculate_boundary_data!(rpbd.dHdvpa_data,fkpl.H0_weights,dfdvpa,vpa,vperp)
    calculate_boundary_data!(rpbd.dHdvperp_data,fkpl.H1_weights,dfdvperp,vpa,vperp)
    if calculate_GG
        calculate_boundary_data!(rpbd.G_data,fkpl.G0_weights,pdf,vpa,vperp)
    end
    if calculate_dGdvperp
        calculate_boundary_data!(rpbd.dGdvperp_data,fkpl.G1_weights,dfdvperp,vpa,vperp)
    end
    calculate_boundary_data!(rpbd.d2Gdvperp2_data,fkpl.H2_weights,dfdvperp,vpa,vperp)
    calculate_boundary_data!(rpbd.d2Gdvperpdvpa_data,fkpl.G1_weights,d2fdvperpdvpa,vpa,vperp)
    calculate_boundary_data!(rpbd.d2Gdvpa2_data,fkpl.H3_weights,dfdvpa,vpa,vperp)
    
    return nothing
end

function test_rosenbluth_potential_boundary_data(rpbd::rosenbluth_potential_boundary_data,
    rpbd_exact::rosenbluth_potential_boundary_data,vpa,vperp;print_to_screen=true)
    
    error_buffer_vpa = Array{mk_float,1}(undef,vpa.n)
    error_buffer_vperp_1 = Array{mk_float,1}(undef,vperp.n)
    error_buffer_vperp_2 = Array{mk_float,1}(undef,vperp.n)
    max_H_err = test_boundary_data(rpbd.H_data,rpbd_exact.H_data,"H",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_dHdvpa_err = test_boundary_data(rpbd.dHdvpa_data,rpbd_exact.dHdvpa_data,"dHdvpa",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_dHdvperp_err = test_boundary_data(rpbd.dHdvperp_data,rpbd_exact.dHdvperp_data,"dHdvperp",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_G_err = test_boundary_data(rpbd.G_data,rpbd_exact.G_data,"G",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_dGdvperp_err = test_boundary_data(rpbd.dGdvperp_data,rpbd_exact.dGdvperp_data,"dGdvperp",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_d2Gdvperp2_err = test_boundary_data(rpbd.d2Gdvperp2_data,rpbd_exact.d2Gdvperp2_data,"d2Gdvperp2",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_d2Gdvperpdvpa_err = test_boundary_data(rpbd.d2Gdvperpdvpa_data,rpbd_exact.d2Gdvperpdvpa_data,"d2Gdvperpdvpa",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  
    max_d2Gdvpa2_err = test_boundary_data(rpbd.d2Gdvpa2_data,rpbd_exact.d2Gdvpa2_data,"d2Gdvpa2",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)  

    return max_H_err, max_dHdvpa_err, max_dHdvperp_err, max_G_err, max_dGdvperp_err, max_d2Gdvperp2_err, max_d2Gdvperpdvpa_err, max_d2Gdvpa2_err
end

function test_boundary_data(func,func_exact,func_name,vpa,vperp,buffer_vpa,buffer_vperp_1,buffer_vperp_2,print_to_screen)
    nvpa = vpa.n
    nvperp = vperp.n
    for ivperp in 1:nvperp
        buffer_vperp_1 = abs(func.lower_boundary_vpa[ivperp] - func_exact.lower_boundary_vpa[ivperp])
        buffer_vperp_2 = abs(func.upper_boundary_vpa[ivperp] - func_exact.upper_boundary_vpa[ivperp])
    end
    for ivpa in 1:nvpa
        buffer_vpa = abs(func.upper_boundary_vperp[ivpa] - func_exact.upper_boundary_vperp[ivpa])
    end
    max_lower_vpa_err = maximum(buffer_vperp_1)
    max_upper_vpa_err = maximum(buffer_vperp_2)
    max_upper_vperp_err = maximum(buffer_vpa)
    if print_to_screen
        println(string(func_name*" boundary data:"))
        println("max(lower_vpa_err) = ",max_lower_vpa_err)
        println("max(upper_vpa_err) = ",max_upper_vpa_err)
        println("max(upper_vperp_err) = ",max_upper_vperp_err)
    end
    max_err = max(max_lower_vpa_err,max_upper_vpa_err,max_upper_vperp_err)
    return max_err
end

function get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
    # global indices on the grids
    ivpa_global = vpa.igrid_full[ivpa_local,ielement_vpa]
    ivperp_global = vperp.igrid_full[ivperp_local,ielement_vperp]
    # global compound index
    ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
    return ic_global, ivpa_global, ivperp_global
end
function enforce_zero_bc!(fc,vpa,vperp;impose_BC_at_zero_vperp=false)
    # lower vpa boundary
    ielement_vpa = 1
    ivpa_local = 1
    for ielement_vperp in 1:vperp.nelement_local
        for ivperp_local in 1:vperp.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = 0.0
        end
    end
    
    # upper vpa boundary
    ielement_vpa = vpa.nelement_local
    ivpa_local = vpa.ngrid
    for ielement_vperp in 1:vperp.nelement_local
        for ivperp_local in 1:vperp.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = 0.0
        end
    end
    
    if impose_BC_at_zero_vperp
        # lower vperp boundary
        ielement_vperp = 1
        ivperp_local = 1
        for ielement_vpa in 1:vpa.nelement_local
            for ivpa_local in 1:vpa.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = 0.0
            end
        end
    end
    
    # upper vperp boundary
    ielement_vperp = vperp.nelement_local
    ivperp_local = vperp.ngrid
    for ielement_vpa in 1:vpa.nelement_local
        for ivpa_local in 1:vpa.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = 0.0
        end
    end
end

function enforce_dirichlet_bc!(fc,vpa,vperp,f_bc;dirichlet_vperp_BC=false)
    # lower vpa boundary
    ielement_vpa = 1
    ivpa_local = 1
    for ielement_vperp in 1:vperp.nelement_local
        for ivperp_local in 1:vperp.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = f_bc[ivpa_global,ivperp_global]
        end
    end
    
    # upper vpa boundary
    ielement_vpa = vpa.nelement_local
    ivpa_local = vpa.ngrid
    for ielement_vperp in 1:vperp.nelement_local
        for ivperp_local in 1:vperp.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = f_bc[ivpa_global,ivperp_global]
        end
    end
    
    if dirichlet_vperp_BC
        # upper vperp boundary
        ielement_vperp = 1
        ivperp_local = 1
        for ielement_vpa in 1:vpa.nelement_local
            for ivpa_local in 1:vpa.ngrid
                ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                fc[ic_global] = f_bc[ivpa_global,ivperp_global]
            end
        end
    end
    
    # upper vperp boundary
    ielement_vperp = vperp.nelement_local
    ivperp_local = vperp.ngrid
    for ielement_vpa in 1:vpa.nelement_local
        for ivpa_local in 1:vpa.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = f_bc[ivpa_global,ivperp_global]
        end
    end
end

function enforce_dirichlet_bc!(fc,vpa,vperp,f_bc::vpa_vperp_boundary_data)
    # lower vpa boundary
    ielement_vpa = 1
    ivpa_local = 1
    for ielement_vperp in 1:vperp.nelement_local
        for ivperp_local in 1:vperp.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = f_bc.lower_boundary_vpa[ivperp_global]
        end
    end
    
    # upper vpa boundary
    ielement_vpa = vpa.nelement_local
    ivpa_local = vpa.ngrid
    for ielement_vperp in 1:vperp.nelement_local
        for ivperp_local in 1:vperp.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = f_bc.upper_boundary_vpa[ivperp_global]
        end
    end
            
    # upper vperp boundary
    ielement_vperp = vperp.nelement_local
    ivperp_local = vperp.ngrid
    for ielement_vpa in 1:vpa.nelement_local
        for ivpa_local in 1:vpa.ngrid
            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
            fc[ic_global] = f_bc.upper_boundary_vperp[ivpa_global]
        end
    end
    return nothing
end

function assemble_matrix_operators_dirichlet_bc(vpa,vperp,vpa_spectral,vperp_spectral;print_to_screen=true)
    nc_global = vpa.n*vperp.n
    # Assemble a 2D mass matrix in the global compound coordinate
    nc_global = vpa.n*vperp.n
    MM2D = Array{mk_float,2}(undef,nc_global,nc_global)
    MM2D .= 0.0
    KKpar2D = Array{mk_float,2}(undef,nc_global,nc_global)
    KKpar2D .= 0.0
    KKperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    KKperp2D .= 0.0
    KPperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    KPperp2D .= 0.0
    KKpar2D_with_BC_terms = Array{mk_float,2}(undef,nc_global,nc_global)
    KKpar2D_with_BC_terms .= 0.0
    KKperp2D_with_BC_terms = Array{mk_float,2}(undef,nc_global,nc_global)
    KKperp2D_with_BC_terms .= 0.0
    PUperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    PUperp2D .= 0.0
    PPparPUperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    PPparPUperp2D .= 0.0
    PPpar2D = Array{mk_float,2}(undef,nc_global,nc_global)
    PPpar2D .= 0.0
    MMparMNperp2D = Array{mk_float,2}(undef,nc_global,nc_global)
    MMparMNperp2D .= 0.0
    # Laplacian matrix
    LP2D = Array{mk_float,2}(undef,nc_global,nc_global)
    LP2D .= 0.0
    # Modified Laplacian matrix
    LV2D = Array{mk_float,2}(undef,nc_global,nc_global)
    LV2D .= 0.0
    # Modified Laplacian matrix
    LB2D = Array{mk_float,2}(undef,nc_global,nc_global)
    LB2D .= 0.0
    
    #print_matrix(MM2D,"MM2D",nc_global,nc_global)
    # local dummy arrays
    MMpar = Array{mk_float,2}(undef,vpa.ngrid,vpa.ngrid)
    MMperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    MNperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    MRperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    KKpar = Array{mk_float,2}(undef,vpa.ngrid,vpa.ngrid)
    KKperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    KKpar_with_BC_terms = Array{mk_float,2}(undef,vpa.ngrid,vpa.ngrid)
    KKperp_with_BC_terms = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    KJperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    LLperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    PPperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    PUperp = Array{mk_float,2}(undef,vperp.ngrid,vperp.ngrid)
    PPpar = Array{mk_float,2}(undef,vpa.ngrid,vpa.ngrid)
        
    impose_BC_at_zero_vperp = false
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("begin elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    for ielement_vperp in 1:vperp.nelement_local
        get_QQ_local!(MMperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"M")
        get_QQ_local!(MRperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"R")
        get_QQ_local!(MNperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"N")
        get_QQ_local!(KKperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"K")
        get_QQ_local!(KKperp_with_BC_terms,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"K_with_BC_terms")
        get_QQ_local!(KJperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"J")
        get_QQ_local!(LLperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"L")
        get_QQ_local!(PPperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"P")
        get_QQ_local!(PUperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"U")
        #print_matrix(MMperp,"MMperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(MRperp,"MRperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(MNperp,"MNperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(KKperp,"KKperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(KJperp,"KJperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(LLperp,"LLperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(PPperp,"PPperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(PUperp,"PUperp",vperp.ngrid,vperp.ngrid)
        
        for ielement_vpa in 1:vpa.nelement_local
            get_QQ_local!(MMpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"M")
            get_QQ_local!(KKpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"K")
            get_QQ_local!(KKpar_with_BC_terms,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"K_with_BC_terms")
            get_QQ_local!(PPpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"P")
            #print_matrix(MMpar,"MMpar",vpa.ngrid,vpa.ngrid)
            #print_matrix(KKpar,"KKpar",vpa.ngrid,vpa.ngrid)
            #print_matrix(PPpar,"PPpar",vpa.ngrid,vpa.ngrid)
            
            for ivperpp_local in 1:vperp.ngrid
                for ivperp_local in 1:vperp.ngrid
                    for ivpap_local in 1:vpa.ngrid
                        for ivpa_local in 1:vpa.ngrid
                            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                            icp_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpap_local,ivperpp_local) #get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivpap_local,ivperp_local,ivperpp_local)
                            #println("ielement_vpa: ",ielement_vpa," ielement_vperp: ",ielement_vperp)
                            #println("ivpa_local: ",ivpa_local," ivpap_local: ",ivpap_local)
                            #println("ivperp_local: ",ivperp_local," ivperpp_local: ",ivperpp_local)
                            #println("ic: ",ic_global," icp: ",icp_global)
                            # boundary condition possibilities
                            lower_boundary_row_vpa = (ielement_vpa == 1 && ivpa_local == 1)
                            upper_boundary_row_vpa = (ielement_vpa == vpa.nelement_local && ivpa_local == vpa.ngrid)
                            lower_boundary_row_vperp = (ielement_vperp == 1 && ivperp_local == 1)
                            upper_boundary_row_vperp = (ielement_vperp == vperp.nelement_local && ivperp_local == vperp.ngrid)
                            

                            if lower_boundary_row_vpa
                                if ivpap_local == 1 && ivperp_local == ivperpp_local
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                    LB2D[ic_global,icp_global] = 1.0
                                else 
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                    LB2D[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_vpa
                                if ivpap_local == vpa.ngrid && ivperp_local == ivperpp_local 
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                    LB2D[ic_global,icp_global] = 1.0
                                else 
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                    LB2D[ic_global,icp_global] = 0.0
                                end
                            elseif lower_boundary_row_vperp && impose_BC_at_zero_vperp
                                if ivperpp_local == 1 && ivpa_local == ivpap_local
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                    LB2D[ic_global,icp_global] = 1.0
                                else 
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                    LB2D[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_vperp
                                if ivperpp_local == vperp.ngrid && ivpa_local == ivpap_local
                                    LP2D[ic_global,icp_global] = 1.0
                                    LV2D[ic_global,icp_global] = 1.0
                                    LB2D[ic_global,icp_global] = 1.0
                                else 
                                    LP2D[ic_global,icp_global] = 0.0
                                    LV2D[ic_global,icp_global] = 0.0
                                    LB2D[ic_global,icp_global] = 0.0
                                end
                            else
                                # assign Laplacian and modified Laplacian matrix data
                                LP2D[ic_global,icp_global] += (KKpar[ivpa_local,ivpap_local]*
                                                                MMperp[ivperp_local,ivperpp_local] +
                                                               MMpar[ivpa_local,ivpap_local]*
                                                                LLperp[ivperp_local,ivperpp_local])
                                LV2D[ic_global,icp_global] += (KKpar[ivpa_local,ivpap_local]*
                                                                MRperp[ivperp_local,ivperpp_local] +
                                                               MMpar[ivpa_local,ivpap_local]*
                                                                (KJperp[ivperp_local,ivperpp_local] -
                                                                 PPperp[ivperp_local,ivperpp_local] - 
                                                                 MNperp[ivperp_local,ivperpp_local]))
                                LB2D[ic_global,icp_global] += (KKpar[ivpa_local,ivpap_local]*
                                                                MRperp[ivperp_local,ivperpp_local] +
                                                               MMpar[ivpa_local,ivpap_local]*
                                                                (KJperp[ivperp_local,ivperpp_local] -
                                                                 PPperp[ivperp_local,ivperpp_local] - 
                                                             4.0*MNperp[ivperp_local,ivperpp_local]))
                            end
                            # assign mass matrix data
                            MM2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                                MMperp[ivperp_local,ivperpp_local]
                            
                            # assign K matrices
                            KKpar2D[ic_global,icp_global] += KKpar[ivpa_local,ivpap_local]*
                                                            MMperp[ivperp_local,ivperpp_local]
                            KKperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            KKperp[ivperp_local,ivperpp_local]
                            KPperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                         (KJperp[ivperp_local,ivperpp_local] -
                                                      2.0*PPperp[ivperp_local,ivperpp_local] -
                                                      2.0*MNperp[ivperp_local,ivperpp_local])
                            # assign K matrices with explicit boundary terms from integration by parts
                            KKpar2D_with_BC_terms[ic_global,icp_global] += KKpar_with_BC_terms[ivpa_local,ivpap_local]*
                                                            MMperp[ivperp_local,ivperpp_local]
                            KKperp2D_with_BC_terms[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            KKperp_with_BC_terms[ivperp_local,ivperpp_local]
                            # assign PU matrix
                            PUperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            PUperp[ivperp_local,ivperpp_local]
                            PPparPUperp2D[ic_global,icp_global] += PPpar[ivpa_local,ivpap_local]*
                                                            PUperp[ivperp_local,ivperpp_local]
                            PPpar2D[ic_global,icp_global] += PPpar[ivpa_local,ivpap_local]*
                                                            MMperp[ivperp_local,ivperpp_local]
                            # assign RHS mass matrix for d2Gdvperp2
                            MMparMNperp2D[ic_global,icp_global] += MMpar[ivpa_local,ivpap_local]*
                                                            MNperp[ivperp_local,ivperpp_local]
                        end
                    end
                end
            end
        end
    end
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("finished elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        if nc_global < 60
            print_matrix(MM2D,"MM2D",nc_global,nc_global)
            #print_matrix(KKpar2D,"KKpar2D",nc_global,nc_global)
            #print_matrix(KKperp2D,"KKperp2D",nc_global,nc_global)
            #print_matrix(LP2D,"LP",nc_global,nc_global)
            #print_matrix(LV2D,"LV",nc_global,nc_global)
        end
        # convert these matrices to sparse matrices
        if global_rank[] == 0 && print_to_screen
            println("begin conversion to sparse matrices   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    MM2D_sparse = sparse(MM2D)
    KKpar2D_sparse = sparse(KKpar2D)
    KKperp2D_sparse = sparse(KKperp2D)
    KKpar2D_with_BC_terms_sparse = sparse(KKpar2D_with_BC_terms)
    KKperp2D_with_BC_terms_sparse = sparse(KKperp2D_with_BC_terms)
    LP2D_sparse = sparse(LP2D)
    LV2D_sparse = sparse(LV2D)
    LB2D_sparse = sparse(LB2D)
    KPperp2D_sparse = sparse(KPperp2D)
    PUperp2D_sparse = sparse(PUperp2D)
    PPparPUperp2D_sparse = sparse(PPparPUperp2D)
    PPpar2D_sparse = sparse(PPpar2D)
    MMparMNperp2D_sparse = sparse(MMparMNperp2D)
    return MM2D_sparse, KKpar2D_sparse, KKperp2D_sparse, 
           KKpar2D_with_BC_terms_sparse, KKperp2D_with_BC_terms_sparse,
           LP2D_sparse, LV2D_sparse, LB2D_sparse, 
           KPperp2D_sparse,PUperp2D_sparse, PPparPUperp2D_sparse,
           PPpar2D_sparse, MMparMNperp2D_sparse
end

function assemble_matrix_operators_dirichlet_bc_sparse(vpa,vperp,vpa_spectral,vperp_spectral;print_to_screen=true)
    # Assemble a 2D mass matrix in the global compound coordinate
    nc_global = vpa.n*vperp.n
    ntot_vpa = (vpa.nelement_local - 1)*(vpa.ngrid^2 - 1) + vpa.ngrid^2
    ntot_vperp = (vperp.nelement_local - 1)*(vperp.ngrid^2 - 1) + vperp.ngrid^2
    nsparse = ntot_vpa*ntot_vperp
    ngrid_vpa = vpa.ngrid
    nelement_vpa = vpa.nelement_local
    ngrid_vperp = vperp.ngrid
    nelement_vperp = vperp.nelement_local
    
    MM2D = allocate_sparse_matrix_constructor(nsparse)
    KKpar2D = allocate_sparse_matrix_constructor(nsparse)
    KKperp2D = allocate_sparse_matrix_constructor(nsparse)
    KKpar2D_with_BC_terms = allocate_sparse_matrix_constructor(nsparse)
    KKperp2D_with_BC_terms = allocate_sparse_matrix_constructor(nsparse)
    PUperp2D = allocate_sparse_matrix_constructor(nsparse)
    PPparPUperp2D = allocate_sparse_matrix_constructor(nsparse)
    PPpar2D = allocate_sparse_matrix_constructor(nsparse)
    MMparMNperp2D = allocate_sparse_matrix_constructor(nsparse)
    KPperp2D = allocate_sparse_matrix_constructor(nsparse)
    # Laplacian matrix
    LP2D = allocate_sparse_matrix_constructor(nsparse)
    # Modified Laplacian matrix (for d / d vperp potentials)
    LV2D = allocate_sparse_matrix_constructor(nsparse)
    # Modified Laplacian matrix (for d^2 / d vperp^2 potentials)
    LB2D = allocate_sparse_matrix_constructor(nsparse)
    
    # local dummy arrays
    MMpar = Array{mk_float,2}(undef,ngrid_vpa,ngrid_vpa)
    MMperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    MNperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    MRperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    KKpar = Array{mk_float,2}(undef,ngrid_vpa,ngrid_vpa)
    KKpar_with_BC_terms = Array{mk_float,2}(undef,ngrid_vpa,ngrid_vpa)
    KKperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    KKperp_with_BC_terms = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    KJperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    LLperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    PPperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    PUperp = Array{mk_float,2}(undef,ngrid_vperp,ngrid_vperp)
    PPpar = Array{mk_float,2}(undef,ngrid_vpa,ngrid_vpa)
        
    impose_BC_at_zero_vperp = false
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("begin elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    for ielement_vperp in 1:nelement_vperp
        get_QQ_local!(MMperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"M")
        get_QQ_local!(MRperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"R")
        get_QQ_local!(MNperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"N")
        get_QQ_local!(KKperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"K")
        get_QQ_local!(KKperp_with_BC_terms,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"K_with_BC_terms")
        get_QQ_local!(KJperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"J")
        get_QQ_local!(LLperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"L")
        get_QQ_local!(PPperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"P")
        get_QQ_local!(PUperp,ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"U")
        #print_matrix(MMperp,"MMperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(MRperp,"MRperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(MNperp,"MNperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(KKperp,"KKperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(KJperp,"KJperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(LLperp,"LLperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(PPperp,"PPperp",vperp.ngrid,vperp.ngrid)
        #print_matrix(PUperp,"PUperp",vperp.ngrid,vperp.ngrid)
        
        for ielement_vpa in 1:nelement_vpa
            get_QQ_local!(MMpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"M")
            get_QQ_local!(KKpar_with_BC_terms,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"K_with_BC_terms")
            get_QQ_local!(KKpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"K")
            get_QQ_local!(PPpar,ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"P")
            #print_matrix(MMpar,"MMpar",vpa.ngrid,vpa.ngrid)
            #print_matrix(KKpar,"KKpar",vpa.ngrid,vpa.ngrid)
            #print_matrix(PPpar,"PPpar",vpa.ngrid,vpa.ngrid)
            
            for ivperpp_local in 1:ngrid_vperp
                for ivperp_local in 1:ngrid_vperp
                    for ivpap_local in 1:ngrid_vpa
                        for ivpa_local in 1:ngrid_vpa
                            ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                            icp_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpap_local,ivperpp_local) #get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivpap_local,ivperp_local,ivperpp_local)
                            icsc = icsc_func(ivpa_local,ivpap_local,ielement_vpa::mk_int,
                                           ngrid_vpa,nelement_vpa,
                                           ivperp_local,ivperpp_local,
                                           ielement_vperp,
                                           ngrid_vperp,nelement_vperp)
                            #println("ielement_vpa: ",ielement_vpa," ielement_vperp: ",ielement_vperp)
                            #println("ivpa_local: ",ivpa_local," ivpap_local: ",ivpap_local)
                            #println("ivperp_local: ",ivperp_local," ivperpp_local: ",ivperpp_local)
                            #println("ic: ",ic_global," icp: ",icp_global)
                            # boundary condition possibilities
                            lower_boundary_row_vpa = (ielement_vpa == 1 && ivpa_local == 1)
                            upper_boundary_row_vpa = (ielement_vpa == vpa.nelement_local && ivpa_local == vpa.ngrid)
                            lower_boundary_row_vperp = (ielement_vperp == 1 && ivperp_local == 1)
                            upper_boundary_row_vperp = (ielement_vperp == vperp.nelement_local && ivperp_local == vperp.ngrid)
                            

                            if lower_boundary_row_vpa
                                if ivpap_local == 1 && ivperp_local == ivperpp_local
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,1.0)
                                else 
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,0.0)
                                end
                            elseif upper_boundary_row_vpa
                                if ivpap_local == vpa.ngrid && ivperp_local == ivperpp_local 
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,1.0)
                                else 
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,0.0)
                                end
                            elseif lower_boundary_row_vperp && impose_BC_at_zero_vperp
                                if ivperpp_local == 1 && ivpa_local == ivpap_local
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,1.0)
                                else 
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,0.0)
                                end
                            elseif upper_boundary_row_vperp
                                if ivperpp_local == vperp.ngrid && ivpa_local == ivpap_local
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,1.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,1.0)
                                else 
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LV2D,icsc,ic_global,icp_global,0.0)
                                    assign_constructor_data!(LB2D,icsc,ic_global,icp_global,0.0)
                                end
                            else
                                # assign Laplacian matrix data
                                assemble_constructor_data!(LP2D,icsc,ic_global,icp_global,
                                            (KKpar[ivpa_local,ivpap_local]*
                                             MMperp[ivperp_local,ivperpp_local] +
                                             MMpar[ivpa_local,ivpap_local]*
                                             LLperp[ivperp_local,ivperpp_local]))
                                assemble_constructor_data!(LV2D,icsc,ic_global,icp_global,
                                            (KKpar[ivpa_local,ivpap_local]*
                                             MRperp[ivperp_local,ivperpp_local] +
                                             MMpar[ivpa_local,ivpap_local]*
                                            (KJperp[ivperp_local,ivperpp_local] -
                                             PPperp[ivperp_local,ivperpp_local] - 
                                             MNperp[ivperp_local,ivperpp_local])))
                                assemble_constructor_data!(LB2D,icsc,ic_global,icp_global,
                                            (KKpar[ivpa_local,ivpap_local]*
                                             MRperp[ivperp_local,ivperpp_local] +
                                             MMpar[ivpa_local,ivpap_local]*
                                             (KJperp[ivperp_local,ivperpp_local] -
                                              PPperp[ivperp_local,ivperpp_local] -
                                          4.0*MNperp[ivperp_local,ivperpp_local])))
                            end
                            #assign mass matrix
                            assemble_constructor_data!(MM2D,icsc,ic_global,icp_global,
                                            (MMpar[ivpa_local,ivpap_local]*
                                             MMperp[ivperp_local,ivperpp_local]))
                                
                            # assign K matrices (no explicit boundary terms)
                            assemble_constructor_data!(KKpar2D,icsc,ic_global,icp_global,
                                            (KKpar[ivpa_local,ivpap_local]*
                                             MMperp[ivperp_local,ivperpp_local]))
                            assemble_constructor_data!(KKperp2D,icsc,ic_global,icp_global,
                                            (MMpar[ivpa_local,ivpap_local]*
                                             KKperp[ivperp_local,ivperpp_local]))
                            assemble_constructor_data!(KPperp2D,icsc,ic_global,icp_global,
                                            (MMpar[ivpa_local,ivpap_local]*
                                             (KJperp[ivperp_local,ivperpp_local] -
                                              2.0*PPperp[ivperp_local,ivperpp_local] -
                                              2.0*MNperp[ivperp_local,ivperpp_local])))
                                             
                            # assign K matrices (with explicit boundary terms from integration by parts)
                            assemble_constructor_data!(KKpar2D_with_BC_terms,icsc,ic_global,icp_global,
                                            (KKpar_with_BC_terms[ivpa_local,ivpap_local]*
                                             MMperp[ivperp_local,ivperpp_local]))
                            assemble_constructor_data!(KKperp2D_with_BC_terms,icsc,ic_global,icp_global,
                                            (MMpar[ivpa_local,ivpap_local]*
                                             KKperp_with_BC_terms[ivperp_local,ivperpp_local]))
                            # assign PU matrix
                            assemble_constructor_data!(PUperp2D,icsc,ic_global,icp_global,
                                            (MMpar[ivpa_local,ivpap_local]*
                                             PUperp[ivperp_local,ivperpp_local]))
                            assemble_constructor_data!(PPparPUperp2D,icsc,ic_global,icp_global,
                                            (PPpar[ivpa_local,ivpap_local]*
                                             PUperp[ivperp_local,ivperpp_local]))
                            assemble_constructor_data!(PPpar2D,icsc,ic_global,icp_global,
                                            (PPpar[ivpa_local,ivpap_local]*
                                             MMperp[ivperp_local,ivperpp_local]))
                            # assign RHS mass matrix for d2Gdvperp2
                            assemble_constructor_data!(MMparMNperp2D,icsc,ic_global,icp_global,
                                            (MMpar[ivpa_local,ivpap_local]*
                                             MNperp[ivperp_local,ivperpp_local]))
                        end
                    end
                end
            end
        end
    end
    MM2D_sparse = create_sparse_matrix(MM2D)
    KKpar2D_sparse = create_sparse_matrix(KKpar2D)
    KKperp2D_sparse = create_sparse_matrix(KKperp2D)
    KKpar2D_with_BC_terms_sparse = create_sparse_matrix(KKpar2D_with_BC_terms)
    KKperp2D_with_BC_terms_sparse = create_sparse_matrix(KKperp2D_with_BC_terms)
    LP2D_sparse = create_sparse_matrix(LP2D)
    LV2D_sparse = create_sparse_matrix(LV2D)
    LB2D_sparse = create_sparse_matrix(LB2D)
    KPperp2D_sparse = create_sparse_matrix(KPperp2D)
    PUperp2D_sparse = create_sparse_matrix(PUperp2D)
    PPparPUperp2D_sparse = create_sparse_matrix(PPparPUperp2D)
    PPpar2D_sparse = create_sparse_matrix(PPpar2D)
    MMparMNperp2D_sparse = create_sparse_matrix(MMparMNperp2D)
    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("finished elliptic operator constructor assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        #if nc_global < 60
        #    println("MM2D_sparse \n",MM2D_sparse)
        #    print_matrix(Array(MM2D_sparse),"MM2D_sparse",nc_global,nc_global)
        #    print_matrix(KKpar2D,"KKpar2D",nc_global,nc_global)
        #    print_matrix(KKperp2D,"KKperp2D",nc_global,nc_global)
        #    print_matrix(LP2D,"LP",nc_global,nc_global)
        #    print_matrix(LV2D,"LV",nc_global,nc_global)
        #end
    end
    return MM2D_sparse, KKpar2D_sparse, KKperp2D_sparse, 
           KKpar2D_with_BC_terms_sparse, KKperp2D_with_BC_terms_sparse, 
           LP2D_sparse, LV2D_sparse, LB2D_sparse, 
           KPperp2D_sparse, PUperp2D_sparse, PPparPUperp2D_sparse,
           PPpar2D_sparse, MMparMNperp2D_sparse
end

function calculate_YY_arrays(vpa,vperp,vpa_spectral,vperp_spectral)
    YY0perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY1perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY2perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY3perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY0par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    YY1par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    YY2par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    YY3par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    
    for ielement_vperp in 1:vperp.nelement_local
        @views get_QQ_local!(YY0perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY0")
        @views get_QQ_local!(YY1perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY1")
        @views get_QQ_local!(YY2perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY2")
        @views get_QQ_local!(YY3perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY3")
     end
     for ielement_vpa in 1:vpa.nelement_local
        @views get_QQ_local!(YY0par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY0")
        @views get_QQ_local!(YY1par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY1")
        @views get_QQ_local!(YY2par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY2")
        @views get_QQ_local!(YY3par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY3")
     end
    
    return YY_collision_operator_arrays(YY0perp,YY1perp,YY2perp,YY3perp,
                                        YY0par,YY1par,YY2par,YY3par)
end

function assemble_explicit_collision_operator_rhs_serial!(rhsc,pdfs,d2Gspdvpa2,d2Gspdvperpdvpa,
    d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    begin_serial_region()
    @serial_region begin
        # assemble RHS of collision operator
        @. rhsc = 0.0
        
        # loop over elements
        for ielement_vperp in 1:vperp.nelement_local
            YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            
            for ielement_vpa in 1:vpa.nelement_local
                YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                
                # loop over field positions in each element
                for ivperp_local in 1:vperp.ngrid
                    for ivpa_local in 1:vpa.ngrid
                        ic_global, ivpa_global, ivperp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                        # carry out the matrix sum on each 2D element
                        for jvperpp_local in 1:vperp.ngrid
                            jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                            for kvperpp_local in 1:vperp.ngrid
                                kvperpp = vperp.igrid_full[kvperpp_local,ielement_vperp]
                                for jvpap_local in 1:vpa.ngrid
                                    jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                                    pdfjj = pdfs[jvpap,jvperpp]
                                    for kvpap_local in 1:vpa.ngrid
                                        kvpap = vpa.igrid_full[kvpap_local,ielement_vpa]
                                        # first three lines represent parallel flux terms
                                        # second three lines represent perpendicular flux terms
                                        rhsc[ic_global] += (YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY2par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvpa2[kvpap,kvperpp] +
                                                            YY3perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] - 
                                                            2.0*(ms/msp)*YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvpa[kvpap,kvperpp] +
                                                            # end parallel flux, start of perpendicular flux
                                                            YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY3par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] + 
                                                            YY2perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperp2[kvpap,kvperpp] - 
                                                            2.0*(ms/msp)*YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvperp[kvpap,kvperpp])
                                    end
                                end
                            end
                        end
                    end
                end 
            end
        end
        # correct for minus sign due to integration by parts
        # and multiply by the normalised collision frequency
        @. rhsc *= -nussp
    end
    return nothing
end

function assemble_explicit_collision_operator_rhs_parallel!(rhsc,rhsvpavperp,pdfs,d2Gspdvpa2,d2Gspdvperpdvpa,
    d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    # assemble RHS of collision operator
    begin_vperp_vpa_region() 
    @loop_vperp_vpa ivperp ivpa begin
        rhsvpavperp[ivpa,ivperp] = 0.0
    end

    # loop over collocation points to benefit from shared-memory parallelism
    ngrid_vpa, ngrid_vperp = vpa.ngrid, vperp.ngrid
    @loop_vperp_vpa ivperp_global ivpa_global begin
        igrid_vpa, ielement_vpax, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperpx, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa_global,ivperp_global,vpa,vperp)
        # loop over elements belonging to this collocation point
        for ielement_vperp in ielement_vperp_low:ielement_vperp_hi
            # correct local ivperp in the case that we on a boundary point
            ivperp_local = igrid_vperp + (ielement_vperp - ielement_vperp_low)*(1-ngrid_vperp)
            @views YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            @views YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            @views YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            @views YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            
            for ielement_vpa in ielement_vpa_low:ielement_vpa_hi
                # correct local ivpa in the case that we on a boundary point
                ivpa_local = igrid_vpa + (ielement_vpa - ielement_vpa_low)*(1-ngrid_vpa)
                @views YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                @views YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                @views YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                @views YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                
                # carry out the matrix sum on each 2D element
                for jvperpp_local in 1:vperp.ngrid
                    jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                    for kvperpp_local in 1:vperp.ngrid
                        kvperpp = vperp.igrid_full[kvperpp_local,ielement_vperp]
                        for jvpap_local in 1:vpa.ngrid
                            jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                            pdfjj = pdfs[jvpap,jvperpp]
                            for kvpap_local in 1:vpa.ngrid
                                kvpap = vpa.igrid_full[kvpap_local,ielement_vpa]
                                # first three lines represent parallel flux terms
                                # second three lines represent perpendicular flux terms
                                rhsvpavperp[ivpa_global,ivperp_global] += -nussp*(YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY2par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvpa2[kvpap,kvperpp] +
                                                    YY3perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] - 
                                                    2.0*(ms/msp)*YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvpa[kvpap,kvperpp] +
                                                    # end parallel flux, start of perpendicular flux
                                                    YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY3par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperpdvpa[kvpap,kvperpp] + 
                                                    YY2perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*d2Gspdvperp2[kvpap,kvperpp] - 
                                                    2.0*(ms/msp)*YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfjj*dHspdvperp[kvpap,kvperpp])
                            end
                        end
                    end
                end
             end
        end
    end
    # ravel to compound index
    #begin_serial_region()
    #ravel_vpavperp_to_c!(rhsc,rhsvpavperp,vpa.n,vperp.n)
    ravel_vpavperp_to_c_parallel!(rhsc,rhsvpavperp,vpa.n)
    return nothing
end

function assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!(rhsc,rhsvpavperp,pdfs,dpdfsdvpa,dpdfsdvperp,d2Gspdvpa2,d2Gspdvperpdvpa,
    d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    # assemble RHS of collision operator
    begin_vperp_vpa_region() 
    @loop_vperp_vpa ivperp ivpa begin
        rhsvpavperp[ivpa,ivperp] = 0.0
    end

    # loop over collocation points to benefit from shared-memory parallelism
    ngrid_vpa, ngrid_vperp = vpa.ngrid, vperp.ngrid
    @loop_vperp_vpa ivperp_global ivpa_global begin
        igrid_vpa, ielement_vpax, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperpx, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa_global,ivperp_global,vpa,vperp)
        # loop over elements belonging to this collocation point
        for ielement_vperp in ielement_vperp_low:ielement_vperp_hi
            # correct local ivperp in the case that we on a boundary point
            ivperp_local = igrid_vperp + (ielement_vperp - ielement_vperp_low)*(1-ngrid_vperp)
            @views YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            @views YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            @views YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            @views YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            
            for ielement_vpa in ielement_vpa_low:ielement_vpa_hi
                # correct local ivpa in the case that we on a boundary point
                ivpa_local = igrid_vpa + (ielement_vpa - ielement_vpa_low)*(1-ngrid_vpa)
                @views YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                @views YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                @views YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                @views YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                
                # carry out the matrix sum on each 2D element
                for jvperpp_local in 1:vperp.ngrid
                    jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                    for kvperpp_local in 1:vperp.ngrid
                        kvperpp = vperp.igrid_full[kvperpp_local,ielement_vperp]
                        for jvpap_local in 1:vpa.ngrid
                            jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                            for kvpap_local in 1:vpa.ngrid
                                kvpap = vpa.igrid_full[kvpap_local,ielement_vpa]
                                # first three lines represent parallel flux terms
                                # second three lines represent perpendicular flux terms
                                rhsvpavperp[ivpa_global,ivperp_global] += -nussp*(YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*dpdfsdvpa[jvpap,jvperpp]*d2Gspdvpa2[kvpap,kvperpp] +
                                                    YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*dpdfsdvperp[jvpap,jvperpp]*d2Gspdvperpdvpa[kvpap,kvperpp] - 
                                                    2.0*(ms/msp)*YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*pdfs[jvpap,jvperpp]*dHspdvpa[kvpap,kvperpp] +
                                                    # end parallel flux, start of perpendicular flux
                                                    YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*dpdfsdvpa[jvpap,jvperpp]*d2Gspdvperpdvpa[kvpap,kvperpp] + 
                                                    YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*dpdfsdvperp[jvpap,jvperpp]*d2Gspdvperp2[kvpap,kvperpp] - 
                                                    2.0*(ms/msp)*YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*pdfs[jvpap,jvperpp]*dHspdvperp[kvpap,kvperpp])
                            end
                        end
                    end
                end
             end
        end
    end
    # ravel to compound index
    #begin_serial_region()
    #ravel_vpavperp_to_c!(rhsc,rhsvpavperp,vpa.n,vperp.n)
    ravel_vpavperp_to_c_parallel!(rhsc,rhsvpavperp,vpa.n)
    return nothing
end


# Elliptic solve function. 
# field: the solution
# source: the source function on the RHS
# boundary data: the known values of field at infinity
# lu_object_lhs: the object for the differential operator that defines field
# matrix_rhs: the weak matrix acting on the source vector
# rhsc, sc: dummy arrays in the compound index (assumed MPISharedArray or SubArray type)
# vpa, vperp: coordinate structs
function elliptic_solve!(field,source,boundary_data::vpa_vperp_boundary_data,
            lu_object_lhs,matrix_rhs,rhsc,sc,vpa,vperp)
    # get data into the compound index format
    begin_vperp_vpa_region()
    ravel_vpavperp_to_c_parallel!(sc,source,vpa.n)
    # assemble the rhs of the weak system
    begin_serial_region()
    @serial_region begin
        mul!(rhsc,matrix_rhs,sc)
        # enforce the boundary conditions
        enforce_dirichlet_bc!(rhsc,vpa,vperp,boundary_data)
        # solve the linear system
        sc .= lu_object_lhs \ rhsc
    end
    # get data into the vpa vperp indices format
    begin_vperp_vpa_region()
    ravel_c_to_vpavperp_parallel!(field,sc,vpa.n)
    return nothing
end
# same as above but source is made of two different terms
# with different weak matrices
function elliptic_solve!(field,source_1,source_2,boundary_data::vpa_vperp_boundary_data,
            lu_object_lhs,matrix_rhs_1,matrix_rhs_2,rhsc_1,rhsc_2,sc_1,sc_2,vpa,vperp)
    # get data into the compound index format
    begin_vperp_vpa_region()
    ravel_vpavperp_to_c_parallel!(sc_1,source_1,vpa.n)
    ravel_vpavperp_to_c_parallel!(sc_2,source_2,vpa.n)
    
    # assemble the rhs of the weak system
    begin_serial_region()
    @serial_region begin
        mul!(rhsc_1,matrix_rhs_1,sc_1)
        mul!(rhsc_2,matrix_rhs_2,sc_2)
    end
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        ic = ic_func(ivpa,ivperp,vpa.n)
        rhsc_1[ic] += rhsc_2[ic]
    end
    begin_serial_region()
    @serial_region begin
        # enforce the boundary conditions
        enforce_dirichlet_bc!(rhsc_1,vpa,vperp,boundary_data)
        # solve the linear system
        sc_1 .= lu_object_lhs \ rhsc_1
    end
    # get data into the vpa vperp indices format
    begin_vperp_vpa_region()
    ravel_c_to_vpavperp_parallel!(field,sc_1,vpa.n)
    return nothing
end

# same as above but source is made of two different terms
# with different weak matrices
function algebraic_solve!(field,source_1,source_2,boundary_data::vpa_vperp_boundary_data,
            lu_object_lhs,matrix_rhs_1,matrix_rhs_2,rhsc_1,rhsc_2,sc_1,sc_2,vpa,vperp)
    # get data into the compound index format
    begin_vperp_vpa_region()
    ravel_vpavperp_to_c_parallel!(sc_1,source_1,vpa.n)
    ravel_vpavperp_to_c_parallel!(sc_2,source_2,vpa.n)
    
    # assemble the rhs of the weak system
    begin_serial_region()
    @serial_region begin
        mul!(rhsc_1,matrix_rhs_1,sc_1)
        mul!(rhsc_2,matrix_rhs_2,sc_2)
    end
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        ic = ic_func(ivpa,ivperp,vpa.n)
        rhsc_1[ic] += rhsc_2[ic]
    end
    begin_serial_region()
    @serial_region begin
        # solve the linear system
        sc_1 .= lu_object_lhs \ rhsc_1
    end
    # get data into the vpa vperp indices format
    begin_vperp_vpa_region()
    ravel_c_to_vpavperp_parallel!(field,sc_1,vpa.n)
    return nothing
end

function calculate_rosenbluth_potentials_via_elliptic_solve!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays::fokkerplanck_weakform_arrays_struct;
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,calculate_dGdvperp=false)
    
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
    
    bwgt = fkpl_arrays.bwgt
    rpbd = fkpl_arrays.rpbd
    
    S_dummy = fkpl_arrays.S_dummy
    Q_dummy = fkpl_arrays.Q_dummy
    rhsc = fkpl_arrays.rhsc
    rhqc = fkpl_arrays.rhqc
    sc = fkpl_arrays.sc
    qc = fkpl_arrays.qc
    
    # calculate the boundary data
    calculate_rosenbluth_potential_boundary_data!(rpbd,bwgt,@view(ffsp_in[:,:]),vpa,vperp,vpa_spectral,vperp_spectral,
      calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
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
    if calculate_GG
        elliptic_solve!(GG,S_dummy,rpbd.G_data,
                    lu_obj_LP,MM2D_sparse,rhsc,sc,vpa,vperp)
    end
    if calculate_dGdvperp || algebraic_solve_for_d2Gdvperp2
        elliptic_solve!(dGdvperp,S_dummy,rpbd.dGdvperp_data,
                    lu_obj_LV,PUperp2D_sparse,rhsc,sc,vpa,vperp)
    end
    elliptic_solve!(d2Gdvpa2,S_dummy,rpbd.d2Gdvpa2_data,
                lu_obj_LP,KKpar2D_sparse,rhsc,sc,vpa,vperp)
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
    return nothing
end

"""
function to calculate Rosenbluth potentials by direct integration
"""

function calculate_rosenbluth_potentials_via_direct_integration!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays::fokkerplanck_arrays_direct_integration_struct)
    dfdvpa = fkpl_arrays.dfdvpa
    dfdvperp = fkpl_arrays.dfdvperp
    d2fdvperpdvpa = fkpl_arrays.d2fdvperpdvpa
    G0_weights = fkpl_arrays.G0_weights
    G1_weights = fkpl_arrays.G1_weights
    H0_weights = fkpl_arrays.H0_weights
    H1_weights = fkpl_arrays.H1_weights
    H2_weights = fkpl_arrays.H2_weights
    H3_weights = fkpl_arrays.H3_weights
    # first compute the derivatives of fs' (the integration weights assume d fs' dvpa and d fs' dvperp are known)
    begin_vperp_region()
    @loop_vperp ivperp begin
        @views derivative!(vpa.scratch, ffsp_in[:,ivperp], vpa, vpa_spectral)
        @. dfdvpa[:,ivperp] = vpa.scratch
    end
    begin_vpa_region()
    @loop_vpa ivpa begin
        @views derivative!(vperp.scratch, ffsp_in[ivpa,:], vperp, vperp_spectral)
        @. dfdvperp[ivpa,:] = vperp.scratch
        @views derivative!(vperp.scratch, dfdvpa[ivpa,:], vperp, vperp_spectral)
        @. d2fdvperpdvpa[ivpa,:] = vperp.scratch
    end
    # with the integrands calculated, compute the integrals
    calculate_rosenbluth_integrals!(GG,d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,
                                        d2Gdvperp2,HH,dHdvpa,dHdvperp,
                                        ffsp_in,dfdvpa,dfdvperp,d2fdvperpdvpa,
                                        G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                                        vpa.n,vperp.n)
    return nothing           
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
function calculate_rosenbluth_integrals!(GG,d2Gspdvpa2,dGspdvperp,d2Gspdvperpdvpa,
                                        d2Gspdvperp2,HH,dHspdvpa,dHspdvperp,
                                        fsp,dfspdvpa,dfspdvperp,d2fspdvperpdvpa,
                                        G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                                        nvpa,nvperp)
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        GG[ivpa,ivperp] = 0.0
        d2Gspdvpa2[ivpa,ivperp] = 0.0
        dGspdvperp[ivpa,ivperp] = 0.0
        d2Gspdvperpdvpa[ivpa,ivperp] = 0.0
        d2Gspdvperp2[ivpa,ivperp] = 0.0
        HH[ivpa,ivperp] = 0.0
        dHspdvpa[ivpa,ivperp] = 0.0
        dHspdvperp[ivpa,ivperp] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                GG[ivpa,ivperp] += G0_weights[ivpap,ivperpp,ivpa,ivperp]*fsp[ivpap,ivperpp]
                #d2Gspdvpa2[ivpa,ivperp] += G0_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvpa2[ivpap,ivperpp]
                d2Gspdvpa2[ivpa,ivperp] += H3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
                dGspdvperp[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                d2Gspdvperpdvpa[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperpdvpa[ivpap,ivperpp]
                #d2Gspdvperp2[ivpa,ivperp] += G2_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperp2[ivpap,ivperpp] + G3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                d2Gspdvperp2[ivpa,ivperp] += H2_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                HH[ivpa,ivperp] += H0_weights[ivpap,ivperpp,ivpa,ivperp]*fsp[ivpap,ivperpp]
                dHspdvpa[ivpa,ivperp] += H0_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
                dHspdvperp[ivpa,ivperp] += H1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            end
        end
    end
    return nothing
end

"""
function to enforce boundary conditions on the collision operator
result to be consistent with the boundary conditions imposed on the the pdf
"""
function enforce_vpavperp_BCs!(pdf,vpa,vperp,vpa_spectral,vperp_spectral)
    nvpa = vpa.n
    nvperp = vperp.n
    ngrid_vperp = vperp.ngrid
    D0 = vperp_spectral.radau.D0
    # vpa boundary conditions
    # zero at infinity
    begin_vperp_region()
    @loop_vperp ivperp begin
        pdf[1,ivperp] = 0.0
        pdf[nvpa,ivperp] = 0.0
    end
    # vperp boundary conditions
    # zero boundary condition at infinity
    # set regularity condition d F / d vperp = 0 at vperp = 0
    # adjust F(vperp = 0) so that d F / d vperp = 0 at vperp = 0
    begin_vpa_region()
    @loop_vpa ivpa begin
        pdf[ivpa,nvperp] = 0.0
        pdf[ivpa,1] = -sum(D0[2:ngrid_vperp].*pdf[ivpa,2:ngrid_vperp])/D0[1]
    end
end

end
