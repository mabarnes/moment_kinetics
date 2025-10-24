"""
Module for functions used
in calculating the integrals and doing
the numerical differentiation for
the implementation of the
the full-F Fokker-Planck collision operator [`moment_kinetics.fokker_planck`](@ref).

Parallelisation of the collision operator uses a special 'anysv' region type, see
[Collision operator and `anysv` region](@ref).
"""
module fokker_planck_calculus

export assemble_matrix_operators_dirichlet_bc
export assemble_matrix_operators_dirichlet_bc_sparse
export assemble_explicit_collision_operator_rhs_serial!
export assemble_explicit_collision_operator_rhs_parallel!
export assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!
export calculate_vpavperp_advection_terms!
export YY_collision_operator_arrays, calculate_YY_arrays
export calculate_rosenbluth_potential_boundary_data!
export calculate_rosenbluth_potential_boundary_data_multipole!
export calculate_rosenbluth_potential_boundary_data_delta_f_multipole!
export elliptic_solve!, algebraic_solve!
export fokkerplanck_arrays_direct_integration_struct
export fokkerplanck_weakform_arrays_struct
export enforce_vpavperp_BCs!
export calculate_rosenbluth_potentials_via_elliptic_solve!
export calculate_rosenbluth_potentials_via_analytical_Maxwellian!
export allocate_preconditioner_matrix
export calculate_test_particle_preconditioner!
export advance_linearised_test_particle_collisions!

# testing
export calculate_rosenbluth_potential_boundary_data_exact!
export allocate_rosenbluth_potential_boundary_data
export calculate_rosenbluth_potential_boundary_data_exact!
export test_rosenbluth_potential_boundary_data
export interpolate_2D_vspace!

# Import moment_kinetics so that we can refer to it in docstrings
import moment_kinetics

using ..type_definitions: mk_float, mk_int, MPISharedArray
using ..array_allocation: allocate_float, allocate_shared_float, allocate_shared_int
using ..calculus: derivative!, integral
using ..communication
using ..lagrange_polynomials: lagrange_poly, lagrange_poly_optimised
using ..looping
using ..velocity_moments: get_density, get_upar, get_p, get_ppar, get_pperp
using ..input_structs: direct_integration, multipole_expansion, delta_f_multipole
using ..fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian
using ..fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using moment_kinetics.gauss_legendre: get_QQ_local!
using Dates
using SpecialFunctions: ellipk, ellipe
using SparseArrays: sparse, AbstractSparseArray
using SuiteSparse
using LinearAlgebra: ldiv!, mul!, LU, ldiv, lu, lu!
using FastGaussQuadrature
using Printf
using MPI

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
Struct to contain data needed to create a sparse matrix.
"""
struct sparse_matrix_constructor{Ti <: AbstractVector{mk_int},Tf <: AbstractVector{mk_float}}
    # the Ith row
    II::Ti
    # the Jth column
    JJ::Ti
    # the data S[I,J]
    SS::Tf
end

"""
Function to allocate an instance of `sparse_matrix_constructor`.
"""
function allocate_sparse_matrix_constructor(nsparse::mk_int; sharedmem=false)
    if sharedmem
        II = allocate_shared_int(:vpa_and_vperp=>nsparse; comm=comm_anysv_subblock[])
        JJ = allocate_shared_int(:vpa_and_vperp=>nsparse; comm=comm_anysv_subblock[])
        SS = allocate_shared_float(:vpa_and_vperp=>nsparse; comm=comm_anysv_subblock[])
        @begin_r_z_anysv_region()
        if anysv_subblock_rank[] ≥ 0
            @begin_anysv_region()
            @anysv_serial_region begin
                @. II = 0
                @. JJ = 0
                @. SS = 0.0
            end
            @_anysv_subblock_synchronize()
        end
        return sparse_matrix_constructor(II,JJ,SS)
    else
        return sparse_matrix_constructor(zeros(mk_int, nsparse), zeros(mk_int, nsparse),
                                         zeros(mk_float, nsparse))
    end
end

"""
Function to assign data to an instance of `sparse_matrix_constructor`.
"""
function assign_constructor_data!(data::sparse_matrix_constructor,icsc::mk_int,ii::mk_int,jj::mk_int,ss::mk_float)
    data.II[icsc] = ii
    data.JJ[icsc] = jj
    data.SS[icsc] = ss
    return nothing
end

"""
Function to assemble data in an instance of `sparse_matrix_constructor`. Instead of
writing `data.SS[icsc] = ss`, as in `assign_constructor_data!()` we write `data.SS[icsc] += ss`.
"""
function assemble_constructor_data!(data::sparse_matrix_constructor,icsc::mk_int,ii::mk_int,jj::mk_int,ss::mk_float)
    data.II[icsc] = ii
    data.JJ[icsc] = jj
    data.SS[icsc] += ss
    return nothing
end

function assemble_constructor_value!(data::sparse_matrix_constructor,icsc::mk_int,ss::mk_float)
    data.SS[icsc] += ss
    return nothing
end

function assign_constructor_value!(data::sparse_matrix_constructor,icsc::mk_int,ss::mk_float)
    data.SS[icsc] = ss
    return nothing
end

"""
Wrapper function to create a sparse matrix with an instance of `sparse_matrix_constructor`
and `sparse()`.
"""
function create_sparse_matrix(data::sparse_matrix_constructor; sharedmem=false)
    if sharedmem
        unshared_sparse = sparse(data.II,data.JJ,data.SS)
    else
        return sparse(data.II,data.JJ,data.SS)
    end
end

"""
Struct of dummy arrays and precalculated coefficients
for the Fokker-Planck collision operator when the
Rosenbluth potentials are computed everywhere in `(vpa,vperp)`
by direct integration. Used for testing.
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
Struct to contain the integration weights for the boundary points
in the `(vpa,vperp)` domain.
"""
struct boundary_integration_weights_struct
    lower_vpa_boundary::MPISharedArray{mk_float,3}
    upper_vpa_boundary::MPISharedArray{mk_float,3}
    upper_vperp_boundary::MPISharedArray{mk_float,3}
end

"""
Struct used for storing the integration weights for the
boundary of the velocity space domain in `(vpa,vperp)` coordinates.
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

"""
Struct to store the `(vpa,vperp)` boundary data for an
individual Rosenbluth potential.
"""
struct vpa_vperp_boundary_data
    lower_boundary_vpa::MPISharedArray{mk_float,1}
    upper_boundary_vpa::MPISharedArray{mk_float,1}
    upper_boundary_vperp::MPISharedArray{mk_float,1}
end

"""
Struct to store the boundary data for all of the
Rosenbluth potentials required for the calculation.
"""
struct rosenbluth_potential_boundary_data{T <: AbstractVector{mk_float}}
    H_data::vpa_vperp_boundary_data
    dHdvpa_data::vpa_vperp_boundary_data
    dHdvperp_data::vpa_vperp_boundary_data
    G_data::vpa_vperp_boundary_data
    dGdvperp_data::vpa_vperp_boundary_data
    d2Gdvperp2_data::vpa_vperp_boundary_data
    d2Gdvperpdvpa_data::vpa_vperp_boundary_data
    d2Gdvpa2_data::vpa_vperp_boundary_data
    integrals_buffer::T
end

"""
Struct to store the elemental nonlinear stiffness matrices used
to express the finite-element weak form of the collision
operator. The arrays are indexed so that the contraction
in the assembly step is carried out over the fastest
accessed indices, i.e., for `YY0perp[i,j,k,iel]`, we contract
over `i` and `j` to give data for the field position index `k`,
all for the 1D element indexed by `iel`.
"""
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
    # MMpar[i,j,iel]
    MMpar::Array{mk_float,3}
    # MMperp[i,j,iel]
    MMperp::Array{mk_float,3}
    # PPpar[i,j,iel]
    PPpar::Array{mk_float,3}
    # PPperp[i,j,iel]
    PPperp::Array{mk_float,3}
end

"""
Struct of dummy arrays and precalculated coefficients
for the finite-element weak-form Fokker-Planck collision operator.
"""
struct fokkerplanck_weakform_arrays_struct{M <: AbstractSparseArray{mk_float,mk_int,N} where N, TLU}
    # boundary weights (Green's function) data
    bwgt::fokkerplanck_boundary_data_arrays_struct
    # dummy arrays for boundary data calculation
    rpbd::rosenbluth_potential_boundary_data
    # assembled 2D weak-form matrices
    MM2D_sparse::M
    KKpar2D_sparse::M
    KKperp2D_sparse::M
    KKpar2D_with_BC_terms_sparse::M
    KKperp2D_with_BC_terms_sparse::M
    LP2D_sparse::M
    LV2D_sparse::M
    LB2D_sparse::M
    PUperp2D_sparse::M
    PPparPUperp2D_sparse::M
    PPpar2D_sparse::M
    MMparMNperp2D_sparse::M
    KPperp2D_sparse::M
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
    rhsvpavperp_copy1::MPISharedArray{mk_float,2}
    rhsvpavperp_copy2::MPISharedArray{mk_float,2}
    rhsvpavperp_copy3::MPISharedArray{mk_float,2}
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
    # matrices for storing preconditioner
    # based on I - dt * C[delta F, F]
    CC2D_sparse::M
    CC2D_sparse_constructor::sparse_matrix_constructor
    lu_obj_CC2D::TLU
    # dummy array for vpa vperp advection contributions
    rhs_advection::MPISharedArray{mk_float,2}
    # dummy arrays for Jacobian-Free-Newton-Krylov solver
    Fnew::MPISharedArray{mk_float,2}
    Fresidual::MPISharedArray{mk_float,2}
    F_delta_x::MPISharedArray{mk_float,2}
    F_rhs_delta::MPISharedArray{mk_float,2}
    Fv::MPISharedArray{mk_float,2}
    Fw::MPISharedArray{mk_float,2}
    parallelised_2d_loop_vperp_indices::UnitRange{Int64}
    parallelised_2d_loop_vpa_indices::UnitRange{Int64}
end

"""
Function to allocate a `boundary_integration_weights_struct`.
"""
function allocate_boundary_integration_weight(vpa, vperp)
    lower_vpa_boundary = allocate_shared_float(vpa, vperp, vperp)
    upper_vpa_boundary = allocate_shared_float(vpa, vperp, vperp)
    upper_vperp_boundary = allocate_shared_float(vpa, vperp, vpa)
    return boundary_integration_weights_struct(lower_vpa_boundary,
            upper_vpa_boundary, upper_vperp_boundary)
end

"""
Function to allocate at `fokkerplanck_boundary_data_arrays_struct`.
"""
function allocate_boundary_integration_weights(vpa, vperp)
    G0_weights = allocate_boundary_integration_weight(vpa, vperp)
    G1_weights = allocate_boundary_integration_weight(vpa, vperp)
    H0_weights = allocate_boundary_integration_weight(vpa, vperp)
    H1_weights = allocate_boundary_integration_weight(vpa, vperp)
    H2_weights = allocate_boundary_integration_weight(vpa, vperp)
    H3_weights = allocate_boundary_integration_weight(vpa, vperp)

    # The following velocity-space-sized buffer arrays are used to evaluate the
    # collision operator for a single species at a single spatial point. They are
    # shared-memory arrays. The `comm` argument to `allocate_shared_float()` is used to
    # set up the shared-memory arrays so that they are shared only by the processes on
    # `comm_anysv_subblock[]` rather than on the full `comm_block[]`. This means that
    # different subblocks that are calculating the collision operator at different
    # spatial points do not interfere with each others' buffer arrays.
    # Note that the 'weights' allocated above are read-only and therefore can be used
    # simultaneously by different subblocks. They are shared over the full
    # `comm_block[]` in order to save memory and setup time.
    nvpa = vpa.n
    nvperp = vperp.n
    dfdvpa = allocate_shared_float(vpa, vperp; comm=comm_anysv_subblock[])
    d2fdvperpdvpa = allocate_shared_float(vpa, vperp; comm=comm_anysv_subblock[])
    dfdvperp = allocate_shared_float(vpa, vperp; comm=comm_anysv_subblock[])
    return fokkerplanck_boundary_data_arrays_struct(G0_weights,
            G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
            dfdvpa,d2fdvperpdvpa,dfdvperp)
end


"""
Function that precomputes the required integration weights in the whole of
`(vpa,vperp)` for the direct integration method of computing the Rosenbluth potentials.
"""
function init_Rosenbluth_potential_integration_weights!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,vperp,vpa;print_to_screen=true)

    x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre = setup_basic_quadratures(vpa,vperp,print_to_screen=print_to_screen)

    @serial_region begin
        if global_rank[] == 0 && print_to_screen
            println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # precalculated weights, integrating over Lagrange polynomials
    @begin_vperp_vpa_region()
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
Function for getting the basic quadratures used for the
numerical integration of the Lagrange polynomials and the
integration kernals.
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
Function for getting the indices used to choose the integration quadrature.
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
Function that precomputes the required integration weights only along the velocity space boundaries.
Used as the default option as part of the strategy to compute the Rosenbluth potentials
at the boundaries with direct integration and in the rest of `(vpa,vperp)` by solving elliptic PDEs.
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
    @begin_vperp_region()
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
    @begin_vpa_region()
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
    @begin_serial_region()
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
Function to get the local integration grid and quadrature weights
to integrate a 1D element in the 2D representation of the
velocity space distribution functions. This function assumes that
there is a divergence at the point `coord_val`, and splits the grid
and integration weights appropriately, using Gauss-Laguerre points
near the divergence and Gauss-Legendre points away from the divergence.
"""
function get_scaled_x_w_with_divergences!(x_scaled, w_scaled, x_legendre, w_legendre, x_laguerre, w_laguerre, node_min, node_max, nodes, igrid_coord, coord_val)
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

"""
Function to get the local grid and integration weights assuming
no divergences of the function on the 1D element. Gauss-Legendre
quadrature is used for the entire element.
"""
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

"""
Function returns `1` if `igrid = 1` or `0` if `1 < igrid <= ngrid`.
"""
function ng_low(igrid,ngrid)
    return floor(mk_int, (ngrid - igrid)/(ngrid - 1))
end

"""
Function returns `1` if `igrid = ngrid` or `0` if `1 =< igrid < ngrid`.
"""
function ng_hi(igrid,ngrid)
    return floor(mk_int, igrid/ngrid)
end

"""
Function returns `1` for `nelement >= ielement > 1`, `0` for `ielement = 1`.
"""
function nel_low(ielement,nelement)
    return floor(mk_int, (ielement - 2 + nelement)/nelement)
end

"""
Function returns `1` for `nelement > ielement >= 1`, `0` for `ielement = nelement`.
"""
function nel_hi(ielement,nelement)
    return 1- floor(mk_int, ielement/nelement)
end

"""
Base level function for computing the integration kernals for the Rosenbluth potential integration.
Note the definitions of `ellipe(m)` (\$E(m)\$) and `ellipk(m)` (\$K(m)\$).
`https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipe`
`https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipk`
```math
E(m) = \\int^{\\pi/2}_0 \\sqrt{ 1 - m \\sin^2(\\theta)} d \\theta
```
```math
K(m) = \\int^{\\pi/2}_0 \\frac{1}{\\sqrt{ 1 - m \\sin^2(\\theta)}} d \\theta
```
"""
function local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            nquad_vpa,ielement_vpa,vpa, # info about primed vpa grids
                            nquad_vperp,ielement_vperp,vperp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # points and weights for primed (source) grids
                            vpa_val, vperp_val) # values and indices for unprimed (field) grids
    @inbounds begin
        for igrid_vperp in 1:vperp.ngrid
            vperp_other_nodes = @view vperp.other_nodes[:,igrid_vperp,ielement_vperp]
            vperp_one_over_denominator = vperp.one_over_denominator[igrid_vperp,ielement_vperp]
            for igrid_vpa in 1:vpa.ngrid
                vpa_other_nodes = @view vpa.other_nodes[:,igrid_vpa,ielement_vpa]
                vpa_one_over_denominator = vpa.one_over_denominator[igrid_vpa,ielement_vpa]
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
                        G_elliptic_integral_factor = 2.0*ellipe_mm*prefac*sqrt(pi)
                        G1_elliptic_integral_factor = -(2.0*prefac*sqrt(pi))*( (2.0 - mm)*ellipe_mm - 2.0*(1.0 - mm)*ellipk_mm )/(3.0*mm)
                        #G2_elliptic_integral_factor = (2.0*prefac*sqrt(pi))*( (7.0*mm^2 + 8.0*mm - 8.0)*ellipe_mm + 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                        #G3_elliptic_integral_factor = (2.0*prefac*sqrt(pi))*( 8.0*(mm^2 - mm + 1.0)*ellipe_mm - 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                        H_elliptic_integral_factor = 2.0*ellipk_mm*sqrt(pi)/prefac
                        H1_elliptic_integral_factor = -(2.0*sqrt(pi)/prefac)*( (mm-2.0)*(ellipk_mm/mm) + (2.0*ellipe_mm/mm) )
                        H2_elliptic_integral_factor = (2.0*sqrt(pi)/prefac)*( (3.0*mm^2 - 8.0*mm + 8.0)*(ellipk_mm/(3.0*mm^2)) + (4.0*mm - 8.0)*ellipe_mm/(3.0*mm^2) )
                        lagrange_poly_vpa = lagrange_poly_optimised(vpa_other_nodes,
                                                                    vpa_one_over_denominator,
                                                                    x_kvpa)
                        lagrange_poly_vperp = lagrange_poly_optimised(vperp_other_nodes,
                                                                      vperp_one_over_denominator,
                                                                      x_kvperp)

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
end

"""
Function for computing the quadratures and carrying out the loop over the
primed `vpa` coordinate in doing the numerical integration. Splits the integrand
into three pieces -- two which use Gauss-Legendre quadrature assuming no divergences
in the integrand, and one which assumes a logarithmic divergence and uses a
Gauss-Laguerre quadrature with an (exponential) change of variables to mitigate this divergence.
"""
function loop_over_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                            vperp,ielement_vperpp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                            x_legendre,w_legendre,x_laguerre,w_laguerre,
                            igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    @inbounds begin
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local)
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        for ielement_vpap in 1:ielement_vpa_low-1
            # do integration over part of the domain with no divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)
        end
        nquad_vperp = get_scaled_x_w_with_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
        for ielement_vpap in ielement_vpa_low:ielement_vpa_hi
        #for ielement_vpap in 1:vpa.nelement_local
            # use general grid function that checks divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            #nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            nquad_vpa = get_scaled_x_w_with_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, x_laguerre, w_laguerre, vpa_min, vpa_max, vpa_nodes, igrid_vpa, vpa_val)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)
        end
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        for ielement_vpap in ielement_vpa_hi+1:vpa.nelement_local
            # do integration over part of the domain with no divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)

        end
        return nothing
    end
end

"""
Function for computing the quadratures and carrying out the loop over the
primed `vpa` coordinate in doing the numerical integration.
Uses a Gauss-Legendre quadrature assuming no divergences in the integrand.
"""
function loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                            x_legendre,w_legendre,
                            vpa_val, vperp_val)
    @inbounds begin
        for ielement_vpap in 1:vpa.nelement_local
            # do integration over part of the domain with no divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)

        end
        return nothing
    end
end

"""
Function for computing the quadratures and carrying out the loop over the
primed `vperp` coordinate in doing the numerical integration. Splits the integrand
into three pieces -- two which use Gauss-Legendre quadrature assuming no divergences
in the integrand, and one which assumes a logarithmic divergence and uses a
Gauss-Laguerre quadrature with an (exponential) change of variables to mitigate this divergence.
This function calls `loop_over_vpa_elements_no_divergences!()` and `loop_over_vpa_elements!()`
to carry out the primed `vpa` loop within the primed `vperp` loop.
"""
function loop_over_vperp_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    @inbounds begin
        for ielement_vperpp in 1:ielement_vperp_low-1

            vperp_nodes = get_nodes(vperp,ielement_vperpp)
            vperp_max = vperp_nodes[end]
            vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local)
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
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
            #nquad_vperp = get_scaled_x_w_with_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
            loop_over_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
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
            loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    x_legendre,w_legendre,
                    vpa_val, vperp_val)
        end
        return nothing
    end
end

"""
The function `loop_over_vperp_vpa_elements_no_divergences!()` was used for debugging.
By changing the source where `loop_over_vperp_vpa_elements!()` is called to
instead call this function we can verify that the Gauss-Legendre quadrature
is adequate for integrating a divergence-free integrand. This function should be
kept until we understand the problems preventing machine-precision accurary in the pure integration method of computing the
Rosenbluth potentials.
"""
function loop_over_vperp_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    @inbounds begin
        for ielement_vperpp in 1:vperp.nelement_local
            vperp_nodes = get_nodes(vperp,ielement_vperpp)
            vperp_max = vperp_nodes[end]
            vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,nelement_vperp)
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    x_legendre,w_legendre,
                    vpa_val, vperp_val)
        end
        return nothing
    end
end


"""
    ic_func(ivpa::mk_int,ivperp::mk_int,nvpa::mk_int)

Get the 'linear index' corresponding to `ivpa` and `ivperp`. Defined so that the linear
index corresponds to the underlying layout in memory of a 2d array indexed by
`[ivpa,ivperp]`, i.e. for a 2d array `f2d`:
* `size(f2d) == (vpa.n, vperp.n)`
* For a reference to `f2d` that is reshaped to a vector (a 1d array) `f1d = vec(f2d)` than
  for any `ivpa` and `ivperp` it is true that `f1d[ic_func(ivpa,ivperp)] ==
  f2d[ivpa,ivperp]`.
"""
function ic_func(ivpa::mk_int,ivperp::mk_int,nvpa::mk_int)
    return ivpa + nvpa*(ivperp-1)
end

"""
    ivperp_func(ic::mk_int,nvpa::mk_int)

Get the `vperp` index `ivperp` that corresponds to a 'linear index' `ic` that spans a 2d
velocity space.

Defined so that `ivperp_func(inc_func(ivpa,ivperp,nvpa), nvpa) == ivperp`.

See also [`ic_func`](@ref), [`ivpa_func`](@ref).
"""
function ivperp_func(ic::mk_int,nvpa::mk_int)
    return floor(Int64,(ic-1)/nvpa) + 1
end

"""
    ivpa_func(ic::mk_int,nvpa::mk_int)

Get the `vpa` index `ivpa` that corresponds to a 'linear index' `ic` that spans a 2d
velocity space.

Defined so that `ivpa_func(inc_func(ivpa,ivperp,nvpa), nvpa) == ivpa`.

See also [`ic_func`](@ref), [`ivperp_func`](@ref).
"""
function ivpa_func(ic::mk_int,nvpa::mk_int)
    ivpa = ic - nvpa*(ivperp_func(ic,nvpa) - 1)
    return ivpa
end

"""
Function that returns the sparse matrix index
used to directly construct the nonzero entries
of a 2D assembled sparse matrix.
"""
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

"""
Function to allocate an instance of `vpa_vperp_boundary_data`.
"""
function allocate_boundary_data(vpa, vperp)
    # The following velocity-space-sized buffer arrays are used to evaluate the
    # collision operator for a single species at a single spatial point. They are
    # shared-memory arrays. The `comm` argument to `allocate_shared_float()` is used to
    # set up the shared-memory arrays so that they are shared only by the processes on
    # `comm_anysv_subblock[]` rather than on the full `comm_block[]`. This means that
    # different subblocks that are calculating the collision operator at different
    # spatial points do not interfere with each others' buffer arrays.
    lower_boundary_vpa = allocate_shared_float(vperp; comm=comm_anysv_subblock[])
    upper_boundary_vpa = allocate_shared_float(vperp; comm=comm_anysv_subblock[])
    upper_boundary_vperp = allocate_shared_float(vpa; comm=comm_anysv_subblock[])
    return vpa_vperp_boundary_data(lower_boundary_vpa,
            upper_boundary_vpa,upper_boundary_vperp)
end

"""
Function to assign precomputed (exact) data to an instance
of `vpa_vperp_boundary_data`. Used in testing.
"""
function assign_exact_boundary_data!(func_data::vpa_vperp_boundary_data,
                                        func_exact,vpa,vperp)
    @begin_anysv_region()
    nvpa = vpa.n
    nvperp = vperp.n
    @anysv_serial_region begin
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

"""
Function to allocate an instance of `rosenbluth_potential_boundary_data`.
"""
function allocate_rosenbluth_potential_boundary_data(vpa, vperp)
    H_data = allocate_boundary_data(vpa, vperp)
    dHdvpa_data = allocate_boundary_data(vpa, vperp)
    dHdvperp_data = allocate_boundary_data(vpa, vperp)
    G_data = allocate_boundary_data(vpa, vperp)
    dGdvperp_data = allocate_boundary_data(vpa, vperp)
    d2Gdvperp2_data = allocate_boundary_data(vpa, vperp)
    d2Gdvperpdvpa_data = allocate_boundary_data(vpa, vperp)
    d2Gdvpa2_data = allocate_boundary_data(vpa, vperp)
    integrals_buffer = allocate_shared_float(:fp_integrals=>25; comm=comm_anysv_subblock[])
    return rosenbluth_potential_boundary_data(H_data,dHdvpa_data,
        dHdvperp_data,G_data,dGdvperp_data,d2Gdvperp2_data,
        d2Gdvperpdvpa_data,d2Gdvpa2_data, integrals_buffer)
end

"""
Function to assign data to an instance of `rosenbluth_potential_boundary_data`, in place,
without allocation. Used in testing.
"""
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

"""
Function to carry out the direct integration of a formal definition of one
of the Rosenbluth potentials, on the boundaries of the `(vpa,vperp)` domain,
using the precomputed integration weights with dimension 4.
The result is stored in an instance of `vpa_vperp_boundary_data`.
Used in testing.
"""
function calculate_boundary_data!(func_data::vpa_vperp_boundary_data,
                                  weight::MPISharedArray{mk_float,4},func_input,vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
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
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
        func_data.upper_boundary_vperp[ivpa] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.upper_boundary_vperp[ivpa] += weight[ivpap,ivperpp,ivpa,nvperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    return nothing
end

"""
Function to carry out the direct integration of a formal definition of one
of the Rosenbluth potentials, on the boundaries of the `(vpa,vperp)` domain,
using the precomputed integration weights with dimension 3.
The result is stored in an instance of `vpa_vperp_boundary_data`.
"""
function calculate_boundary_data!(func_data::vpa_vperp_boundary_data,
                                  weight::boundary_integration_weights_struct,
                                  func_input,vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
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
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
        func_data.upper_boundary_vperp[ivpa] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.upper_boundary_vperp[ivpa] += weight.upper_vperp_boundary[ivpap,ivperpp,ivpa]*func_input[ivpap,ivperpp]
            end
        end
    end
    # return to serial parallelisation
    return nothing
end

"""
Function to call direct integration function `calculate_boundary_data!()` and
assign data to an instance of `rosenbluth_potential_boundary_data`, in place,
without allocation.
"""
function calculate_rosenbluth_potential_boundary_data!(rpbd::rosenbluth_potential_boundary_data,
    fkpl::Union{fokkerplanck_arrays_direct_integration_struct,fokkerplanck_boundary_data_arrays_struct},pdf,vpa,vperp,vpa_spectral,vperp_spectral;
    calculate_GG=false,calculate_dGdvperp=false)
    # get derivatives of pdf
    dfdvperp = fkpl.dfdvperp
    dfdvpa = fkpl.dfdvpa
    d2fdvperpdvpa = fkpl.d2fdvperpdvpa
    #for ivpa in 1:vpa.n
    @begin_anysv_vpa_region()
    @loop_vpa ivpa begin
        @views derivative!(dfdvperp[ivpa,:], pdf[ivpa,:], vperp, vperp_spectral)
    end
    @begin_anysv_vperp_region()
    @loop_vperp ivperp begin
    #for ivperp in 1:vperp.n
        @views derivative!(dfdvpa[:,ivperp], pdf[:,ivperp], vpa, vpa_spectral)
        @views derivative!(d2fdvperpdvpa[:,ivperp], dfdvperp[:,ivperp], vpa, vpa_spectral)
    end
    # ensure data is synchronized
    @_anysv_subblock_synchronize()
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

function multipole_H(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   H_series = (I80*((128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8)/(128*(vpa^2 + vperp^2)^8))
             +I70*((vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
             +I62*((-7*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(64*(vpa^2 + vperp^2)^8))
             +I60*((16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6)/(16*(vpa^2 + vperp^2)^6))
             +I52*((21*vpa*(-16*vpa^6 + 168*vpa^4*vperp^2 - 210*vpa^2*vperp^4 + 35*vperp^6))/(32*(vpa^2 + vperp^2)^7))
             +I50*((8*vpa^5 - 40*vpa^3*vperp^2 + 15*vpa*vperp^4)/(8*(vpa^2 + vperp^2)^5))
             +I44*((105*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I42*((-15*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(32*(vpa^2 + vperp^2)^6))
             +I40*((8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4)/(8*(vpa^2 + vperp^2)^4))
             +I34*((105*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(128*(vpa^2 + vperp^2)^7))
             +I32*((-5*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^5))
             +I30*((vpa*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
             +I26*((-35*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I24*((45*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^6))
             +I22*((-3*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(8*(vpa^2 + vperp^2)^4))
             +I20*(-1/2*(-2*vpa^2 + vperp^2)/(vpa^2 + vperp^2)^2)
             +I16*((-35*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(256*(vpa^2 + vperp^2)^7))
             +I14*((15*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(64*(vpa^2 + vperp^2)^5))
             +I12*((-6*vpa^3 + 9*vpa*vperp^2)/(4*(vpa^2 + vperp^2)^3))
             +I10*(vpa/(vpa^2 + vperp^2))
             +I08*((35*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(16384*(vpa^2 + vperp^2)^8))
             +I06*((-5*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(256*(vpa^2 + vperp^2)^6))
             +I04*((3*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(64*(vpa^2 + vperp^2)^4))
             +I02*((-2*vpa^2 + vperp^2)/(4*(vpa^2 + vperp^2)^2))
             +I00*(1))
   # multiply by overall prefactor
   H_series *= ((vpa^2 + vperp^2)^(-1/2))
   return H_series
end

function multipole_dHdvpa(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   dHdvpa_series = (I80*((9*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                +I70*((128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8)/(16*(vpa^2 + vperp^2)^7))
                +I62*((-63*(128*vpa^9 - 2304*vpa^7*vperp^2 + 6048*vpa^5*vperp^4 - 3360*vpa^3*vperp^6 + 315*vpa*vperp^8))/(64*(vpa^2 + vperp^2)^8))
                +I60*((7*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                +I52*((-21*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(32*(vpa^2 + vperp^2)^7))
                +I50*((3*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                +I44*((945*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I42*((-105*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                +I40*((5*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I34*((105*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                +I32*((-15*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                +I30*((8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4)/(2*(vpa^2 + vperp^2)^3))
                +I26*((-315*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I24*((315*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(128*(vpa^2 + vperp^2)^6))
                +I22*((-15*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I20*((3*vpa*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^2))
                +I16*((-35*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                +I14*((45*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(64*(vpa^2 + vperp^2)^5))
                +I12*((-3*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(4*(vpa^2 + vperp^2)^3))
                +I10*(-1 + (3*vpa^2)/(vpa^2 + vperp^2))
                +I08*((315*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(16384*(vpa^2 + vperp^2)^8))
                +I06*((-35*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(256*(vpa^2 + vperp^2)^6))
                +I04*((15*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(64*(vpa^2 + vperp^2)^4))
                +I02*((-6*vpa^3 + 9*vpa*vperp^2)/(4*(vpa^2 + vperp^2)^2))
                +I00*(vpa))
   # multiply by overall prefactor
   dHdvpa_series *= -((vpa^2 + vperp^2)^(-3/2))
   return dHdvpa_series
end

function multipole_dHdvperp(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   dHdvperp_series = (I80*((45*vperp*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                +I70*((9*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
                +I62*((-315*(128*vpa^8*vperp - 896*vpa^6*vperp^3 + 1120*vpa^4*vperp^5 - 280*vpa^2*vperp^7 + 7*vperp^9))/(64*(vpa^2 + vperp^2)^8))
                +I60*((7*vperp*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                +I52*((-189*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(32*(vpa^2 + vperp^2)^7))
                +I50*((21*vpa*vperp*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                +I44*((4725*vperp*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I42*((105*vperp*(-64*vpa^6 + 240*vpa^4*vperp^2 - 120*vpa^2*vperp^4 + 5*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                +I40*((15*vperp*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I34*((945*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(128*(vpa^2 + vperp^2)^7))
                +I32*((-105*vpa*vperp*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                +I30*((5*vpa*vperp*(4*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
                +I26*((-1575*vperp*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I24*((315*vperp*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^6))
                +I22*((-45*vperp*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I20*((-3*vperp*(-4*vpa^2 + vperp^2))/(2*(vpa^2 + vperp^2)^2))
                +I16*((-315*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(256*(vpa^2 + vperp^2)^7))
                +I14*((315*vpa*vperp*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(64*(vpa^2 + vperp^2)^5))
                +I12*((-15*vpa*vperp*(4*vpa^2 - 3*vperp^2))/(4*(vpa^2 + vperp^2)^3))
                +I10*((3*vpa*vperp)/(vpa^2 + vperp^2))
                +I08*((1575*(128*vpa^8*vperp - 896*vpa^6*vperp^3 + 1120*vpa^4*vperp^5 - 280*vpa^2*vperp^7 + 7*vperp^9))/(16384*(vpa^2 + vperp^2)^8))
                +I06*((-35*(64*vpa^6*vperp - 240*vpa^4*vperp^3 + 120*vpa^2*vperp^5 - 5*vperp^7))/(256*(vpa^2 + vperp^2)^6))
                +I04*((45*(8*vpa^4*vperp - 12*vpa^2*vperp^3 + vperp^5))/(64*(vpa^2 + vperp^2)^4))
                +I02*((3*vperp*(-4*vpa^2 + vperp^2))/(4*(vpa^2 + vperp^2)^2))
                +I00*(vperp))
   # multiply by overall prefactor
   dHdvperp_series *= -((vpa^2 + vperp^2)^(-3/2))
   return dHdvperp_series
end

function multipole_G(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   G_series = (I80*((64*vpa^6*vperp^2 - 240*vpa^4*vperp^4 + 120*vpa^2*vperp^6 - 5*vperp^8)/(128*(vpa^2 + vperp^2)^8))
             +I70*((vpa*vperp^2*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(16*(vpa^2 + vperp^2)^7))
             +I62*((32*vpa^8 - 656*vpa^6*vperp^2 + 1620*vpa^4*vperp^4 - 670*vpa^2*vperp^6 + 25*vperp^8)/(64*(vpa^2 + vperp^2)^8))
             +I60*((vperp^2*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(16*(vpa^2 + vperp^2)^6))
             +I52*((vpa*(16*vpa^6 - 232*vpa^4*vperp^2 + 370*vpa^2*vperp^4 - 75*vperp^6))/(32*(vpa^2 + vperp^2)^7))
             +I50*((vpa*vperp^2*(4*vpa^2 - 3*vperp^2))/(8*(vpa^2 + vperp^2)^5))
             +I44*((-15*(64*vpa^8 - 864*vpa^6*vperp^2 + 1560*vpa^4*vperp^4 - 500*vpa^2*vperp^6 + 15*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I42*((16*vpa^6 - 152*vpa^4*vperp^2 + 138*vpa^2*vperp^4 - 9*vperp^6)/(32*(vpa^2 + vperp^2)^6))
             +I40*(-1/8*(vperp^2*(-4*vpa^2 + vperp^2))/(vpa^2 + vperp^2)^4)
             +I34*((5*vpa*(-32*vpa^6 + 296*vpa^4*vperp^2 - 320*vpa^2*vperp^4 + 45*vperp^6))/(128*(vpa^2 + vperp^2)^7))
             +I32*((vpa*(4*vpa^4 - 22*vpa^2*vperp^2 + 9*vperp^4))/(8*(vpa^2 + vperp^2)^5))
             +I30*((vpa*vperp^2)/(2*(vpa^2 + vperp^2)^3))
             +I26*((5*(96*vpa^8 - 1072*vpa^6*vperp^2 + 1500*vpa^4*vperp^4 - 330*vpa^2*vperp^6 + 5*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I24*((3*(-32*vpa^6 + 184*vpa^4*vperp^2 - 96*vpa^2*vperp^4 + 3*vperp^6))/(128*(vpa^2 + vperp^2)^6))
             +I22*((4*vpa^4 - 10*vpa^2*vperp^2 + vperp^4)/(8*(vpa^2 + vperp^2)^4))
             +I20*(vperp^2/(2*(vpa^2 + vperp^2)^2))
             +I16*((5*vpa*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(256*(vpa^2 + vperp^2)^7))
             +I14*((-3*vpa*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(64*(vpa^2 + vperp^2)^5))
             +I12*((vpa*(2*vpa^2 - vperp^2))/(4*(vpa^2 + vperp^2)^3))
             +I10*(-(vpa/(vpa^2 + vperp^2)))
             +I08*((5*(-128*vpa^8 + 1280*vpa^6*vperp^2 - 1440*vpa^4*vperp^4 + 160*vpa^2*vperp^6 + 5*vperp^8))/(16384*(vpa^2 + vperp^2)^8))
             +I06*((16*vpa^6 - 72*vpa^4*vperp^2 + 18*vpa^2*vperp^4 + vperp^6)/(256*(vpa^2 + vperp^2)^6))
             +I04*((-8*vpa^4 + 8*vpa^2*vperp^2 + vperp^4)/(64*(vpa^2 + vperp^2)^4))
             +I02*((2*vpa^2 + vperp^2)/(4*(vpa^2 + vperp^2)^2))
             +I00*(1))
   # multiply by overall prefactor
   G_series *= ((vpa^2 + vperp^2)^(1/2))
   return G_series
end

function multipole_dGdvperp(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   dGdvperp_series = (I80*((vperp*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                   +I70*((vpa*vperp*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
                   +I62*((-7*(256*vpa^8*vperp - 2144*vpa^6*vperp^3 + 3120*vpa^4*vperp^5 - 890*vpa^2*vperp^7 + 25*vperp^9))/(64*(vpa^2 + vperp^2)^8))
                   +I60*((vperp*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                   +I52*((21*vpa*vperp*(-32*vpa^6 + 192*vpa^4*vperp^2 - 180*vpa^2*vperp^4 + 25*vperp^6))/(32*(vpa^2 + vperp^2)^7))
                   +I50*((8*vpa^5*vperp - 40*vpa^3*vperp^3 + 15*vpa*vperp^5)/(8*(vpa^2 + vperp^2)^5))
                   +I44*((315*vperp*(128*vpa^8 - 832*vpa^6*vperp^2 + 960*vpa^4*vperp^4 - 220*vpa^2*vperp^6 + 5*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                   +I42*((15*vperp*(-32*vpa^6 + 128*vpa^4*vperp^2 - 68*vpa^2*vperp^4 + 3*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                   +I40*((vperp*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                   +I34*((315*vpa*vperp*(16*vpa^6 - 72*vpa^4*vperp^2 + 50*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^7))
                   +I32*((-5*vpa*vperp*(16*vpa^4 - 38*vpa^2*vperp^2 + 9*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                   +I30*((vpa*vperp*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
                   +I26*((-35*vperp*(512*vpa^8 - 2848*vpa^6*vperp^2 + 2640*vpa^4*vperp^4 - 430*vpa^2*vperp^6 + 5*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                   +I24*((-45*vperp*(-48*vpa^6 + 136*vpa^4*vperp^2 - 46*vpa^2*vperp^4 + vperp^6))/(128*(vpa^2 + vperp^2)^6))
                   +I22*((-3*vperp*(16*vpa^4 - 18*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                   +I20*(-1/2*(vperp*(-2*vpa^2 + vperp^2))/(vpa^2 + vperp^2)^2)
                   +I16*((-35*vpa*vperp*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(256*(vpa^2 + vperp^2)^7))
                   +I14*((45*vpa*vperp*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(64*(vpa^2 + vperp^2)^5))
                   +I12*((3*vpa*vperp*(-4*vpa^2 + vperp^2))/(4*(vpa^2 + vperp^2)^3))
                   +I10*((vpa*vperp)/(vpa^2 + vperp^2))
                   +I08*((175*(128*vpa^8*vperp - 640*vpa^6*vperp^3 + 480*vpa^4*vperp^5 - 40*vpa^2*vperp^7 - vperp^9))/(16384*(vpa^2 + vperp^2)^8))
                   +I06*((-5*(64*vpa^6*vperp - 144*vpa^4*vperp^3 + 24*vpa^2*vperp^5 + vperp^7))/(256*(vpa^2 + vperp^2)^6))
                   +I04*((3*(24*vpa^4*vperp - 12*vpa^2*vperp^3 - vperp^5))/(64*(vpa^2 + vperp^2)^4))
                   +I02*(-1/4*(vperp*(4*vpa^2 + vperp^2))/(vpa^2 + vperp^2)^2)
                   +I00*(vperp))
   # multiply by overall prefactor
   dGdvperp_series *= ((vpa^2 + vperp^2)^(-1/2))
   return dGdvperp_series
end

function multipole_d2Gdvperp2(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   d2Gdvperp2_series = (I80*((128*vpa^10 - 7424*vpa^8*vperp^2 + 41888*vpa^6*vperp^4 - 48160*vpa^4*vperp^6 + 11515*vpa^2*vperp^8 - 280*vperp^10)/(128*(vpa^2 + vperp^2)^8))
                   +I70*((16*vpa^9 - 728*vpa^7*vperp^2 + 3066*vpa^5*vperp^4 - 2345*vpa^3*vperp^6 + 280*vpa*vperp^8)/(16*(vpa^2 + vperp^2)^7))
                   +I62*((-7*(256*vpa^10 - 10528*vpa^8*vperp^2 + 45616*vpa^6*vperp^4 - 43670*vpa^4*vperp^6 + 9125*vpa^2*vperp^8 - 200*vperp^10))/(64*(vpa^2 + vperp^2)^8))
                   +I60*((16*vpa^8 - 552*vpa^6*vperp^2 + 1650*vpa^4*vperp^4 - 755*vpa^2*vperp^6 + 30*vperp^8)/(16*(vpa^2 + vperp^2)^6))
                   +I52*((-21*(32*vpa^9 - 1024*vpa^7*vperp^2 + 3204*vpa^5*vperp^4 - 1975*vpa^3*vperp^6 + 200*vpa*vperp^8))/(32*(vpa^2 + vperp^2)^7))
                   +I50*((8*vpa^7 - 200*vpa^5*vperp^2 + 395*vpa^3*vperp^4 - 90*vpa*vperp^6)/(8*(vpa^2 + vperp^2)^5))
                   +I44*((315*(128*vpa^10 - 4544*vpa^8*vperp^2 + 16448*vpa^6*vperp^4 - 13060*vpa^4*vperp^6 + 2245*vpa^2*vperp^8 - 40*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I42*((-15*(32*vpa^8 - 768*vpa^6*vperp^2 + 1620*vpa^4*vperp^4 - 565*vpa^2*vperp^6 + 18*vperp^8))/(32*(vpa^2 + vperp^2)^6))
                   +I40*((8*vpa^6 - 136*vpa^4*vperp^2 + 159*vpa^2*vperp^4 - 12*vperp^6)/(8*(vpa^2 + vperp^2)^4))
                   +I34*((315*vpa*(16*vpa^8 - 440*vpa^6*vperp^2 + 1114*vpa^4*vperp^4 - 535*vpa^2*vperp^6 + 40*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                   +I32*((5*vpa*(-16*vpa^6 + 274*vpa^4*vperp^2 - 349*vpa^2*vperp^4 + 54*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                   +I30*((vpa*(2*vpa^4 - 21*vpa^2*vperp^2 + 12*vperp^4))/(2*(vpa^2 + vperp^2)^3))
                   +I26*((-35*(512*vpa^10 - 16736*vpa^8*vperp^2 + 53072*vpa^6*vperp^4 - 34690*vpa^4*vperp^6 + 4345*vpa^2*vperp^8 - 40*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I24*((135*(16*vpa^8 - 328*vpa^6*vperp^2 + 530*vpa^4*vperp^4 - 125*vpa^2*vperp^6 + 2*vperp^8))/(128*(vpa^2 + vperp^2)^6))
                   +I22*((-3*(16*vpa^6 - 182*vpa^4*vperp^2 + 113*vpa^2*vperp^4 - 4*vperp^6))/(8*(vpa^2 + vperp^2)^4))
                   +I20*((2*vpa^4 - 11*vpa^2*vperp^2 + 2*vperp^4)/(2*(vpa^2 + vperp^2)^2))
                   +I16*((-35*vpa*(64*vpa^8 - 1616*vpa^6*vperp^2 + 3480*vpa^4*vperp^4 - 1235*vpa^2*vperp^6 + 40*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                   +I14*((45*vpa*(8*vpa^6 - 116*vpa^4*vperp^2 + 101*vpa^2*vperp^4 - 6*vperp^6))/(64*(vpa^2 + vperp^2)^5))
                   +I12*((-3*vpa*(4*vpa^4 - 27*vpa^2*vperp^2 + 4*vperp^4))/(4*(vpa^2 + vperp^2)^3))
                   +I10*(-2*vpa + (3*vpa^3)/(vpa^2 + vperp^2))
                   +I08*((175*(128*vpa^10 - 3968*vpa^8*vperp^2 + 11360*vpa^6*vperp^4 - 6040*vpa^4*vperp^6 + 391*vpa^2*vperp^8 + 8*vperp^10))/(16384*(vpa^2 + vperp^2)^8))
                   +I06*((-5*(64*vpa^8 - 1200*vpa^6*vperp^2 + 1560*vpa^4*vperp^4 - 185*vpa^2*vperp^6 - 6*vperp^8))/(256*(vpa^2 + vperp^2)^6))
                   +I04*((3*(24*vpa^6 - 228*vpa^4*vperp^2 + 67*vpa^2*vperp^4 + 4*vperp^6))/(64*(vpa^2 + vperp^2)^4))
                   +I02*((-4*vpa^4 + 13*vpa^2*vperp^2 + 2*vperp^4)/(4*(vpa^2 + vperp^2)^2))
                   +I00*(vpa^2))
   # multiply by overall prefactor
   d2Gdvperp2_series *= ((vpa^2 + vperp^2)^(-3/2))
   return d2Gdvperp2_series
end

function multipole_d2Gdvperpdvpa(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   d2Gdvperpdvpa_series = (I80*((9*vpa*vperp*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                      +I70*((vperp*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(16*(vpa^2 + vperp^2)^7))
                      +I62*((-63*(256*vpa^9*vperp - 2848*vpa^7*vperp^3 + 5936*vpa^5*vperp^5 - 2870*vpa^3*vperp^7 + 245*vpa*vperp^9))/(64*(vpa^2 + vperp^2)^8))
                      +I60*((7*vpa*vperp*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                      +I52*((-21*(256*vpa^8*vperp - 2144*vpa^6*vperp^3 + 3120*vpa^4*vperp^5 - 890*vpa^2*vperp^7 + 25*vperp^9))/(32*(vpa^2 + vperp^2)^7))
                      +I50*((3*vperp*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                      +I44*((945*vpa*vperp*(384*vpa^8 - 3392*vpa^6*vperp^2 + 5824*vpa^4*vperp^4 - 2380*vpa^2*vperp^6 + 175*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                      +I42*((-105*vpa*vperp*(32*vpa^6 - 192*vpa^4*vperp^2 + 180*vpa^2*vperp^4 - 25*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                      +I40*((5*vpa*vperp*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                      +I34*((315*vperp*(128*vpa^8 - 832*vpa^6*vperp^2 + 960*vpa^4*vperp^4 - 220*vpa^2*vperp^6 + 5*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                      +I32*((15*vperp*(-32*vpa^6 + 128*vpa^4*vperp^2 - 68*vpa^2*vperp^4 + 3*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                      +I30*((vperp*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(2*(vpa^2 + vperp^2)^3))
                      +I26*((-315*vpa*vperp*(512*vpa^8 - 3936*vpa^6*vperp^2 + 5712*vpa^4*vperp^4 - 1890*vpa^2*vperp^6 + 105*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                      +I24*((945*vpa*vperp*(16*vpa^6 - 72*vpa^4*vperp^2 + 50*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^6))
                      +I22*((-15*vpa*vperp*(16*vpa^4 - 38*vpa^2*vperp^2 + 9*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                      +I20*((3*vpa*vperp*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^2))
                      +I16*((-35*vperp*(512*vpa^8 - 2848*vpa^6*vperp^2 + 2640*vpa^4*vperp^4 - 430*vpa^2*vperp^6 + 5*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                      +I14*((-45*vperp*(-48*vpa^6 + 136*vpa^4*vperp^2 - 46*vpa^2*vperp^4 + vperp^6))/(64*(vpa^2 + vperp^2)^5))
                      +I12*((-3*vperp*(16*vpa^4 - 18*vpa^2*vperp^2 + vperp^4))/(4*(vpa^2 + vperp^2)^3))
                      +I10*(vperp*(-1 + (3*vpa^2)/(vpa^2 + vperp^2)))
                      +I08*((1575*vpa*(128*vpa^8*vperp - 896*vpa^6*vperp^3 + 1120*vpa^4*vperp^5 - 280*vpa^2*vperp^7 + 7*vperp^9))/(16384*(vpa^2 + vperp^2)^8))
                      +I06*((-35*vpa*(64*vpa^6*vperp - 240*vpa^4*vperp^3 + 120*vpa^2*vperp^5 - 5*vperp^7))/(256*(vpa^2 + vperp^2)^6))
                      +I04*((45*vpa*(8*vpa^4*vperp - 12*vpa^2*vperp^3 + vperp^5))/(64*(vpa^2 + vperp^2)^4))
                      +I02*((3*vpa*vperp*(-4*vpa^2 + vperp^2))/(4*(vpa^2 + vperp^2)^2))
                      +I00*(vpa*vperp))
   # multiply by overall prefactor
   d2Gdvperpdvpa_series *= -((vpa^2 + vperp^2)^(-3/2))
   return d2Gdvperpdvpa_series
end

function multipole_d2Gdvpa2(vpa::mk_float,vperp::mk_float,Inn_vec::AbstractVector{mk_float})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inn_vec
   # sum up terms in the multipole series
   d2Gdvpa2_series = (I80*((45*vperp^2*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                   +I70*((9*vpa*vperp^2*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
                   +I62*((7*(256*vpa^10 - 9088*vpa^8*vperp^2 + 43456*vpa^6*vperp^4 - 45920*vpa^4*vperp^6 + 10430*vpa^2*vperp^8 - 245*vperp^10))/(64*(vpa^2 + vperp^2)^8))
                   +I60*((7*vperp^2*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                   +I52*((21*vpa*(32*vpa^8 - 880*vpa^6*vperp^2 + 3108*vpa^4*vperp^4 - 2170*vpa^2*vperp^6 + 245*vperp^8))/(32*(vpa^2 + vperp^2)^7))
                   +I50*((21*vpa*vperp^2*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                   +I44*((105*(-512*vpa^10 + 12416*vpa^8*vperp^2 - 46592*vpa^6*vperp^4 + 41440*vpa^4*vperp^6 - 8260*vpa^2*vperp^8 + 175*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I42*((15*(32*vpa^8 - 656*vpa^6*vperp^2 + 1620*vpa^4*vperp^4 - 670*vpa^2*vperp^6 + 25*vperp^8))/(32*(vpa^2 + vperp^2)^6))
                   +I40*((15*vperp^2*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                   +I34*((-105*vpa*(64*vpa^8 - 1184*vpa^6*vperp^2 + 3192*vpa^4*vperp^4 - 1820*vpa^2*vperp^6 + 175*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                   +I32*((5*vpa*(16*vpa^6 - 232*vpa^4*vperp^2 + 370*vpa^2*vperp^4 - 75*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                   +I30*((5*vpa*vperp^2*(4*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
                   +I26*((105*(256*vpa^10 - 5248*vpa^8*vperp^2 + 16576*vpa^6*vperp^4 - 12320*vpa^4*vperp^6 + 2030*vpa^2*vperp^8 - 35*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I24*((-45*(64*vpa^8 - 864*vpa^6*vperp^2 + 1560*vpa^4*vperp^4 - 500*vpa^2*vperp^6 + 15*vperp^8))/(128*(vpa^2 + vperp^2)^6))
                   +I22*((3*(16*vpa^6 - 152*vpa^4*vperp^2 + 138*vpa^2*vperp^4 - 9*vperp^6))/(8*(vpa^2 + vperp^2)^4))
                   +I20*((-3*vperp^2*(-4*vpa^2 + vperp^2))/(2*(vpa^2 + vperp^2)^2))
                   +I16*((105*vpa*(32*vpa^8 - 496*vpa^6*vperp^2 + 1092*vpa^4*vperp^4 - 490*vpa^2*vperp^6 + 35*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                   +I14*((15*vpa*(-32*vpa^6 + 296*vpa^4*vperp^2 - 320*vpa^2*vperp^4 + 45*vperp^6))/(64*(vpa^2 + vperp^2)^5))
                   +I12*((3*vpa*(4*vpa^4 - 22*vpa^2*vperp^2 + 9*vperp^4))/(4*(vpa^2 + vperp^2)^3))
                   +I10*((3*vpa*vperp^2)/(vpa^2 + vperp^2))
                   +I08*((-35*(1024*vpa^10 - 19072*vpa^8*vperp^2 + 52864*vpa^6*vperp^4 - 32480*vpa^4*vperp^6 + 3920*vpa^2*vperp^8 - 35*vperp^10))/(16384*(vpa^2 + vperp^2)^8))
                   +I06*((5*(96*vpa^8 - 1072*vpa^6*vperp^2 + 1500*vpa^4*vperp^4 - 330*vpa^2*vperp^6 + 5*vperp^8))/(256*(vpa^2 + vperp^2)^6))
                   +I04*((-3*(32*vpa^6 - 184*vpa^4*vperp^2 + 96*vpa^2*vperp^4 - 3*vperp^6))/(64*(vpa^2 + vperp^2)^4))
                   +I02*((4*vpa^4 - 10*vpa^2*vperp^2 + vperp^4)/(4*(vpa^2 + vperp^2)^2))
                   +I00*(vperp^2))
   # multiply by overall prefactor
   d2Gdvpa2_series *= ((vpa^2 + vperp^2)^(-3/2))
   return d2Gdvpa2_series
end

"""
"""
function calculate_boundary_data_multipole_H!(func_data::vpa_vperp_boundary_data,vpa,vperp,
                                             Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_H(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_H(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_H(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_dHdvpa!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_dHdvpa(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_dHdvpa(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_dHdvpa(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_dHdvperp!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_dHdvperp(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_dHdvperp(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_dHdvperp(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_G!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_G(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_G(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_G(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_dGdvperp!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_dGdvperp(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_dGdvperp(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_dGdvperp(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_d2Gdvperp2!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_d2Gdvperp2(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_d2Gdvperp2(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_d2Gdvperp2(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_d2Gdvperpdvpa!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_d2Gdvperpdvpa(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_d2Gdvperpdvpa(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_d2Gdvperpdvpa(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
"""
function calculate_boundary_data_multipole_d2Gdvpa2!(func_data::vpa_vperp_boundary_data,vpa,vperp,Inn_vec)
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region(true)
    @loop_vperp ivperp begin
                func_data.lower_boundary_vpa[ivperp] = multipole_d2Gdvpa2(vpa.grid[1],vperp.grid[ivperp],Inn_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_d2Gdvpa2(vpa.grid[nvpa],vperp.grid[ivperp],Inn_vec)
    end
    @begin_anysv_vpa_region(true)
    @loop_vpa ivpa begin
                func_data.upper_boundary_vperp[ivpa] = multipole_d2Gdvpa2(vpa.grid[ivpa],vperp.grid[nvperp],Inn_vec)
    end
    # return to serial parallelisation
    return nothing
end

"""
Function to use the multipole expansion of the Rosenbluth potentials to calculate and
assign boundary data to an instance of `rosenbluth_potential_boundary_data`, in place,
without allocation.
"""
function calculate_rosenbluth_potential_boundary_data_multipole!(rpbd::rosenbluth_potential_boundary_data,
    pdf,vpa,vperp,vpa_spectral,vperp_spectral;
    calculate_GG=false,calculate_dGdvperp=false)
    @inbounds begin
        @_anysv_subblock_synchronize()

        # get required moments of pdf
        integrals_buffer = rpbd.integrals_buffer
        I00 = @view integrals_buffer[1:1]
        I10 = @view integrals_buffer[2:2]
        I20 = @view integrals_buffer[3:3]
        I30 = @view integrals_buffer[4:4]
        I40 = @view integrals_buffer[5:5]
        I50 = @view integrals_buffer[6:6]
        I60 = @view integrals_buffer[7:7]
        I70 = @view integrals_buffer[8:8]
        I80 = @view integrals_buffer[9:9]
        I02 = @view integrals_buffer[10:10]
        I12 = @view integrals_buffer[11:11]
        I22 = @view integrals_buffer[12:12]
        I32 = @view integrals_buffer[13:13]
        I42 = @view integrals_buffer[14:14]
        I52 = @view integrals_buffer[15:15]
        I62 = @view integrals_buffer[16:16]
        I04 = @view integrals_buffer[17:17]
        I14 = @view integrals_buffer[18:18]
        I24 = @view integrals_buffer[19:19]
        I34 = @view integrals_buffer[20:20]
        I44 = @view integrals_buffer[21:21]
        I06 = @view integrals_buffer[22:22]
        I16 = @view integrals_buffer[23:23]
        I26 = @view integrals_buffer[24:24]
        I08 = @view integrals_buffer[25:25]

        # Round-robin integrals among processes in the anysv subblock
        if anysv_subblock_rank[] == (0 % anysv_subblock_size[])
            I00[] = integral(pdf, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (1 % anysv_subblock_size[])
            I10[] = integral(pdf, vpa.grid, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (2 % anysv_subblock_size[])
            I20[] = integral(pdf, vpa.grid, 2, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (3 % anysv_subblock_size[])
            I30[] = integral(pdf, vpa.grid, 3, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (4 % anysv_subblock_size[])
            I40[] = integral(pdf, vpa.grid, 4, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (5 % anysv_subblock_size[])
            I50[] = integral(pdf, vpa.grid, 5, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (6 % anysv_subblock_size[])
            I60[] = integral(pdf, vpa.grid, 6, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (7 % anysv_subblock_size[])
            I70[] = integral(pdf, vpa.grid, 7, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (8 % anysv_subblock_size[])
            I80[] = integral(pdf, vpa.grid, 8, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        if anysv_subblock_rank[] == (9 % anysv_subblock_size[])
            I02[] = integral(pdf, vpa.grid, 0, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (10 % anysv_subblock_size[])
            I12[] = integral(pdf, vpa.grid, 1, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (11 % anysv_subblock_size[])
            I22[] = integral(pdf, vpa.grid, 2, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (12 % anysv_subblock_size[])
            I32[] = integral(pdf, vpa.grid, 3, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (13 % anysv_subblock_size[])
            I42[] = integral(pdf, vpa.grid, 4, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (14 % anysv_subblock_size[])
            I52[] = integral(pdf, vpa.grid, 5, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (15 % anysv_subblock_size[])
            I62[] = integral(pdf, vpa.grid, 6, vpa.wgts, vperp.grid, 2, vperp.wgts)
        end
        if anysv_subblock_rank[] == (16 % anysv_subblock_size[])
            I04[] = integral(pdf, vpa.grid, 0, vpa.wgts, vperp.grid, 4, vperp.wgts)
        end
        if anysv_subblock_rank[] == (17 % anysv_subblock_size[])
            I14[] = integral(pdf, vpa.grid, 1, vpa.wgts, vperp.grid, 4, vperp.wgts)
        end
        if anysv_subblock_rank[] == (18 % anysv_subblock_size[])
            I24[] = integral(pdf, vpa.grid, 2, vpa.wgts, vperp.grid, 4, vperp.wgts)
        end
        if anysv_subblock_rank[] == (19 % anysv_subblock_size[])
            I34[] = integral(pdf, vpa.grid, 3, vpa.wgts, vperp.grid, 4, vperp.wgts)
        end
        if anysv_subblock_rank[] == (20 % anysv_subblock_size[])
            I44[] = integral(pdf, vpa.grid, 4, vpa.wgts, vperp.grid, 4, vperp.wgts)
        end

        if anysv_subblock_rank[] == (21 % anysv_subblock_size[])
            I06[] = integral(pdf, vpa.grid, 0, vpa.wgts, vperp.grid, 6, vperp.wgts)
        end
        if anysv_subblock_rank[] == (22 % anysv_subblock_size[])
            I16[] = integral(pdf, vpa.grid, 1, vpa.wgts, vperp.grid, 6, vperp.wgts)
        end
        if anysv_subblock_rank[] == (23 % anysv_subblock_size[])
            I26[] = integral(pdf, vpa.grid, 2, vpa.wgts, vperp.grid, 6, vperp.wgts)
        end

        if anysv_subblock_rank[] == (24 % anysv_subblock_size[])
            I08[] = integral(pdf, vpa.grid, 0, vpa.wgts, vperp.grid, 8, vperp.wgts)
        end
        @_anysv_subblock_synchronize()

        # evaluate the multipole formulae
        calculate_boundary_data_multipole_H!(rpbd.H_data,vpa,vperp,integrals_buffer)
        calculate_boundary_data_multipole_dHdvpa!(rpbd.dHdvpa_data,vpa,vperp,integrals_buffer)
        calculate_boundary_data_multipole_dHdvperp!(rpbd.dHdvperp_data,vpa,vperp,integrals_buffer)
        if calculate_GG
            calculate_boundary_data_multipole_G!(rpbd.G_data,vpa,vperp,integrals_buffer)
        end
        if calculate_dGdvperp
            calculate_boundary_data_multipole_dGdvperp!(rpbd.dGdvperp_data,vpa,vperp,integrals_buffer)
        end
        calculate_boundary_data_multipole_d2Gdvperp2!(rpbd.d2Gdvperp2_data,vpa,vperp,integrals_buffer)
        calculate_boundary_data_multipole_d2Gdvperpdvpa!(rpbd.d2Gdvperpdvpa_data,vpa,vperp,integrals_buffer)
        calculate_boundary_data_multipole_d2Gdvpa2!(rpbd.d2Gdvpa2_data,vpa,vperp,integrals_buffer)

        return nothing
    end
end

"""
Function to use the multipole expansion of the Rosenbluth potentials to calculate and
assign boundary data to an instance of `rosenbluth_potential_boundary_data`, in place,
without allocation. Use the exact results for the part of F that can be described with
a Maxwellian, and the multipole expansion for the remainder.
"""
function calculate_rosenbluth_potential_boundary_data_delta_f_multipole!(rpbd::rosenbluth_potential_boundary_data,
    pdf,dummy_vpavperp,vpa,vperp,vpa_spectral,vperp_spectral;
    calculate_GG=false,calculate_dGdvperp=false)

    mass = 1.0
    dens, upar, vth = 0.0, 0.0, 0.0
    # first, compute the moments and delta f
    @begin_anysv_region()
    @anysv_serial_region begin
      dens = get_density(pdf, vpa, vperp)
      upar = get_upar(pdf, dens, vpa, vperp, false)
      pressure = get_p(pdf, dens, upar, vpa, vperp, false, false)
      vth = sqrt(2.0*pressure/(dens*mass))
      ppar = get_ppar(dens, upar, pressure, vth, pdf, vpa, vperp, false, false,
                      false)
      pperp = get_pperp(pressure, ppar)
      @loop_vperp_vpa ivperp ivpa begin
          dummy_vpavperp[ivpa,ivperp] = pdf[ivpa,ivperp] - F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
      end
    end
    # broadcast this information across cores
    param_vec = [dens, upar, vth]
    if comm_anysv_subblock[] != MPI.COMM_NULL
        MPI.Bcast!(param_vec, 0, comm_anysv_subblock[])
    end
    (dens, upar, vth) = param_vec
    # ensure data is synchronized
    @_anysv_subblock_synchronize()
    # now pass the delta f to the multipole function
    calculate_rosenbluth_potential_boundary_data_multipole!(rpbd,dummy_vpavperp,
      vpa,vperp,vpa_spectral,vperp_spectral,
      calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
    # now add on the contributions from the Maxwellian
    nvpa = vpa.n
    nvperp = vperp.n
    @begin_anysv_vperp_region()
    @loop_vperp ivperp begin
                rpbd.H_data.lower_boundary_vpa[ivperp] += H_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                rpbd.H_data.upper_boundary_vpa[ivperp] += H_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
                rpbd.dHdvpa_data.lower_boundary_vpa[ivperp] += dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                rpbd.dHdvpa_data.upper_boundary_vpa[ivperp] += dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
                rpbd.dHdvperp_data.lower_boundary_vpa[ivperp] += dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                rpbd.dHdvperp_data.upper_boundary_vpa[ivperp] += dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
                rpbd.d2Gdvpa2_data.lower_boundary_vpa[ivperp] += d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                rpbd.d2Gdvpa2_data.upper_boundary_vpa[ivperp] += d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
                rpbd.d2Gdvperpdvpa_data.lower_boundary_vpa[ivperp] += d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                rpbd.d2Gdvperpdvpa_data.upper_boundary_vpa[ivperp] += d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
                rpbd.d2Gdvperp2_data.lower_boundary_vpa[ivperp] += d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                rpbd.d2Gdvperp2_data.upper_boundary_vpa[ivperp] += d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
    end
    @begin_anysv_vpa_region()
    @loop_vpa ivpa begin
                rpbd.H_data.upper_boundary_vperp[ivpa] += H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
                rpbd.dHdvpa_data.upper_boundary_vperp[ivpa] += dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
                rpbd.dHdvperp_data.upper_boundary_vperp[ivpa] += dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
                rpbd.d2Gdvpa2_data.upper_boundary_vperp[ivpa] += d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
                rpbd.d2Gdvperpdvpa_data.upper_boundary_vperp[ivpa] += d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
                rpbd.d2Gdvperp2_data.upper_boundary_vperp[ivpa] += d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
    end
    if calculate_GG
       @begin_anysv_vperp_region()
       @loop_vperp ivperp begin
                   rpbd.G_data.lower_boundary_vpa[ivperp] += G_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                   rpbd.G_data.upper_boundary_vpa[ivperp] += G_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
       end
       @begin_anysv_vpa_region()
       @loop_vpa ivpa begin
                   rpbd.G_data.upper_boundary_vperp[ivpa] += G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
       end
    end
    if calculate_dGdvperp
       @begin_anysv_vperp_region()
       @loop_vperp ivperp begin
                   rpbd.dGdvperp_data.lower_boundary_vpa[ivperp] += dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,1,ivperp)
                   rpbd.dGdvperp_data.upper_boundary_vpa[ivperp] += dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,nvpa,ivperp)
       end
       @begin_anysv_vpa_region()
       @loop_vpa ivpa begin
                   rpbd.dGdvperp_data.upper_boundary_vperp[ivpa] += dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,nvperp)
       end
    end
    return nothing
end

"""
Function to compare two instances of `rosenbluth_potential_boundary_data` --
one assumed to contain exact results, and the other numerically computed results -- and compute
the maximum value of the error. Calls `test_boundary_data()`.
"""
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

"""
Function to compute the maximum error \${\\rm MAX}|f_{\\rm numerical}-f_{\\rm exact}|\$ for
instances of `vpa_vperp_boundary_data`.
"""
function test_boundary_data(func,func_exact,func_name,vpa,vperp,buffer_vpa,buffer_vperp_1,buffer_vperp_2,print_to_screen)
    nvpa = vpa.n
    nvperp = vperp.n
    for ivperp in 1:nvperp
        buffer_vperp_1[ivperp] = abs(func.lower_boundary_vpa[ivperp] - func_exact.lower_boundary_vpa[ivperp])
        buffer_vperp_2[ivperp] = abs(func.upper_boundary_vpa[ivperp] - func_exact.upper_boundary_vpa[ivperp])
    end
    for ivpa in 1:nvpa
        buffer_vpa[ivpa] = abs(func.upper_boundary_vperp[ivpa] - func_exact.upper_boundary_vperp[ivpa])
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

"""
    get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)

For local (within the single element specified by `ielement_vpa` and `ielement_vperp`)
indices `ivpa_local` and `ivperp_local`, get the global index in the 'linear-indexed' 2d
space of size `(vperp.n, vpa.n)` (as returned by [`ic_func`](@ref)).
"""
function get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
    # global indices on the grids
    ivpa_global = vpa.igrid_full[ivpa_local,ielement_vpa]
    ivperp_global = vperp.igrid_full[ivperp_local,ielement_vperp]
    # global compound index
    ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
    return ic_global
end

"""
Unused function. Sets `f(vpa,vperp)` to zero at the boundaries
in `(vpa,vperp)`.
"""
function enforce_zero_bc!(fvpavperp,vpa,vperp;impose_BC_at_zero_vperp=false)
    # lower vpa boundary
    @loop_vperp ivperp begin
        fvpavperp[1,ivperp] = 0.0
    end

    # upper vpa boundary
    @loop_vperp ivperp begin
        fvpavperp[end,ivperp] = 0.0
    end

    if impose_BC_at_zero_vperp
        # lower vperp boundary
        @loop_vpa ivpa begin
            fvpavperp[ivpa,1] = 0.0
        end
    end

    # upper vperp boundary
    @loop_vpa ivpa begin
        fvpavperp[ivpa,end] = 0.0
    end
end

"""
Sets `f(vpa,vperp)` to a specied value `f_bc` at the boundaries
in `(vpa,vperp)`. `f_bc` is a 2D array of `(vpa,vperp)` where
only boundary data is used. Used for testing.
"""
function enforce_dirichlet_bc!(fvpavperp,vpa,vperp,f_bc;dirichlet_vperp_lower_boundary=false)
    # lower vpa boundary
    for ivperp ∈ 1:vperp.n
        fvpavperp[1,ivperp] = f_bc[1,ivperp]
    end

    # upper vpa boundary
    for ivperp ∈ 1:vperp.n
        fvpavperp[end,ivperp] = f_bc[end,ivperp]
    end

    if dirichlet_vperp_lower_boundary
        # lower vperp boundary
        for ivpa ∈ 1:vpa.n
            fvpavperp[ivpa,1] = f_bc[ivpa,1]
        end
    end

    # upper vperp boundary
    for ivpa ∈ 1:vpa.n
        fvpavperp[ivpa,end] = f_bc[ivpa,end]
    end
end

"""
Sets `f(vpa,vperp)` to a specied value `f_bc` at the boundaries
in `(vpa,vperp)`. `f_bc` is an instance of `vpa_vperp_boundary_data`.
"""
function enforce_dirichlet_bc!(fvpavperp,vpa,vperp,f_bc::vpa_vperp_boundary_data)
    # lower vpa boundary
    for ivperp ∈ 1:vperp.n
        fvpavperp[1,ivperp] = f_bc.lower_boundary_vpa[ivperp]
    end

    # upper vpa boundary
    for ivperp ∈ 1:vperp.n
        fvpavperp[end,ivperp] = f_bc.upper_boundary_vpa[ivperp]
    end

    # upper vperp boundary
    for ivpa ∈ 1:vpa.n
        fvpavperp[ivpa,end] = f_bc.upper_boundary_vperp[ivpa]
    end
    return nothing
end

"""
Function to contruct the global sparse matrices used to solve
the elliptic PDEs for the Rosenbluth potentials. Uses a dense matrix
construction method. The matrices are 2D in the compound index `ic`
which indexes the velocity space labelled by `ivpa,ivperp`.
Dirichlet boundary conditions are imposed in the appropriate stiffness
matrices by setting the boundary row to be the Kronecker delta
(0 except where `ivpa = ivpap` and `ivperp = ivperpp`).
Used for testing.
"""
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
                            ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                            icp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpap_local,ivperpp_local) #get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivpap_local,ivperp_local,ivperpp_local)
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

"""
Function to contruct the global sparse matrices used to solve
the elliptic PDEs for the Rosenbluth potentials. Uses a sparse matrix
construction method. The matrices are 2D in the compound index `ic`
which indexes the velocity space labelled by `ivpa,ivperp`.
Dirichlet boundary conditions are imposed in the appropriate stiffness
matrices by setting the boundary row to be the Kronecker delta
(0 except where `ivpa = ivpap` and `ivperp = ivperpp`).
See also `assemble_matrix_operators_dirichlet_bc()`.
"""
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
                            ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                            icp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpap_local,ivperpp_local) #get_indices(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivpap_local,ivperp_local,ivperpp_local)
                            icsc = icsc_func(ivpa_local,ivpap_local,ielement_vpa,
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

function allocate_preconditioner_matrix(vpa,vperp,vpa_spectral,vperp_spectral)
    # Assemble a 2D mass matrix in the global compound coordinate
    ngrid_vpa = vpa.ngrid
    nelement_vpa = vpa.nelement_local
    ngrid_vperp = vperp.ngrid
    nelement_vperp = vperp.nelement_local
    ntot_vpa = (nelement_vpa - 1)*(ngrid_vpa^2 - 1) + ngrid_vpa^2
    ntot_vperp = (nelement_vperp - 1)*(ngrid_vperp^2 - 1) + ngrid_vperp^2
    nsparse = ntot_vpa*ntot_vperp

    CC2D = allocate_sparse_matrix_constructor(nsparse; sharedmem=true)
    @begin_r_z_anysv_region()
    if anysv_subblock_rank[] ≥ 0
        @begin_anysv_region()
        @anysv_serial_region begin
            for ielement_vperp in 1:nelement_vperp
                for ielement_vpa in 1:nelement_vpa
                    for ivperpp_local in 1:ngrid_vperp
                        for ivperp_local in 1:ngrid_vperp
                            for ivpap_local in 1:ngrid_vpa
                                for ivpa_local in 1:ngrid_vpa
                                    ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                                    icp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpap_local,ivperpp_local)
                                    icsc = icsc_func(ivpa_local,ivpap_local,ielement_vpa,
                                                   ngrid_vpa,nelement_vpa,
                                                   ivperp_local,ivperpp_local,
                                                   ielement_vperp,
                                                   ngrid_vperp,nelement_vperp)
                                    # assign placeholder matrix to be the identity
                                    if ic_global == icp_global
                                        # assign unit values
                                        assign_constructor_data!(CC2D,icsc,ic_global,icp_global,1.0)
                                    else
                                        # assign zero values
                                        assign_constructor_data!(CC2D,icsc,ic_global,icp_global,0.0)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        @_anysv_subblock_synchronize()
    end
    CC2D_sparse = create_sparse_matrix(CC2D)
    if anysv_subblock_rank[] == 0
        lu_obj_CC2D = lu(CC2D_sparse)
    else
        lu_obj_CC2D = nothing
    end
    return CC2D_sparse, CC2D, lu_obj_CC2D
end

function calculate_test_particle_preconditioner!(pdf,delta_t,ms,msp,nussp,
    vpa,vperp,vpa_spectral,vperp_spectral,
    fkpl_arrays::fokkerplanck_weakform_arrays_struct;
    use_Maxwellian_Rosenbluth_coefficients=false,
    algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
    calculate_dGdvperp=false,
    boundary_data_option=direct_integration)

    #Precon2D_sparse = fkpl_arrays.Precon2D_sparse
    #CC2D_sparse = fkpl_arrays.CC2D_sparse
    CC2D_sparse_constructor = fkpl_arrays.CC2D_sparse_constructor
    YY_arrays = fkpl_arrays.YY_arrays
    GG = fkpl_arrays.GG
    HH = fkpl_arrays.HH
    dHdvpa = fkpl_arrays.dHdvpa
    dHdvperp = fkpl_arrays.dHdvperp
    dGdvperp = fkpl_arrays.dGdvperp
    d2Gdvperp2 = fkpl_arrays.d2Gdvperp2
    d2Gdvpa2 = fkpl_arrays.d2Gdvpa2
    d2Gdvperpdvpa = fkpl_arrays.d2Gdvperpdvpa

    # consider making a wrapper function for the following block -- repeated in fokker_planck.jl
    if use_Maxwellian_Rosenbluth_coefficients
        calculate_rosenbluth_potentials_via_analytical_Maxwellian!(GG,HH,dHdvpa,dHdvperp,
                 d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,pdf,vpa,vperp,msp)
    else
        calculate_rosenbluth_potentials_via_elliptic_solve!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,pdf,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays,
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
             calculate_dGdvperp=false,
             boundary_data_option=boundary_data_option)
    end

    @begin_anysv_region()
    parallelised_2d_loop_vperp_indices = fkpl_arrays.parallelised_2d_loop_vperp_indices
    parallelised_2d_loop_vpa_indices = fkpl_arrays.parallelised_2d_loop_vpa_indices

    @anysv_serial_region begin
        # set the values of the matrix to zero before assembly
        CC2D_sparse_constructor.SS .= 0.0
    end
    @_anysv_subblock_synchronize()

    # assemble matrix for preconditioning collision operator
    # we form the linearised collision operator matrix
    # MM - dt * RHS_C
    # with MM the mass matrix
    # and RHS_C the operator such that RHS_C(pdf) * pdf is the usual RHS
    # of the collision operator.
    # loop over collocation points to benefit from shared-memory parallelism
    # to form matrix operator such that  RHS = dt * Precon2D * pdf
    ngrid_vpa, ngrid_vperp = vpa.ngrid, vperp.ngrid
    nelement_vpa, nelement_vperp = vpa.nelement_local, vperp.nelement_local
    vperp_igrid_full = vperp.igrid_full
    vpa_igrid_full = vpa.igrid_full

    function interior_loop(YY0perp, YY1perp, YY2perp, YY3perp, MMperp, YY0par, YY1par,
                           YY2par, YY3par, MMpar, PPpar, ivpa_local, ivperp_local,
                           ielement_vpa, ielement_vperp)
        @inbounds begin
            for jvperpp_local in 1:ngrid_vperp
                for jvpap_local in 1:ngrid_vpa
                    # carry out the matrix sum on each 2D element
                    # mass matrix contribution
                    # don't need these indices because we just overwrite
                    # the constructor values, not the indices
                    # ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                    # icp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,jvpap_local,jvperpp_local)
                    icsc = icsc_func(ivpa_local,jvpap_local,ielement_vpa,
                            ngrid_vpa,nelement_vpa,
                            ivperp_local,jvperpp_local,
                            ielement_vperp,
                            ngrid_vperp,nelement_vperp)
                    #assemble_constructor_data!(CC2D_sparse_constructor,
                    #                    icsc,ic_global,icp_global,
                    #                    (MMpar[ivpa_local,jvpap_local]*
                    #                    MMperp[ivperp_local,jvperpp_local]))
                    assemble_constructor_value!(CC2D_sparse_constructor,icsc,
                                        (MMpar[ivpa_local,jvpap_local]*
                                        MMperp[ivperp_local,jvperpp_local]))
                    # treat div ( dvpadt F) without integration by parts
                    #                    + delta_t * dvpadt * PPpar[ivpa_local,jvpap_local]*
                    #                       MMperp[ivperp_local,jvperpp_local]))
                end
                # collision operator contribution
                jvperpp = vperp_igrid_full[jvperpp_local,ielement_vperp]
                for kvperpp_local in 1:ngrid_vperp
                    kvperpp = vperp_igrid_full[kvperpp_local,ielement_vperp]
                    for jvpap_local in 1:ngrid_vpa
                        jvpap = vpa_igrid_full[jvpap_local,ielement_vpa]
                        icsc = icsc_func(ivpa_local,jvpap_local,ielement_vpa,
                                ngrid_vpa,nelement_vpa,
                                ivperp_local,jvperpp_local,
                                ielement_vperp,
                                ngrid_vperp,nelement_vperp)
                        for kvpap_local in 1:ngrid_vpa
                            kvpap = vpa_igrid_full[kvpap_local,ielement_vpa]
                            # first three lines represent parallel flux terms
                            # second three lines represent perpendicular flux terms
                            assemble_constructor_value!(CC2D_sparse_constructor,icsc,
                            -delta_t*(-nussp*(YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY2par[kvpap_local,jvpap_local,ivpa_local]*d2Gdvpa2[kvpap,kvperpp] +
                                                YY3perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*d2Gdvperpdvpa[kvpap,kvperpp] -
                                                2.0*(ms/msp)*YY0perp[kvperpp_local,jvperpp_local,ivperp_local]*YY1par[kvpap_local,jvpap_local,ivpa_local]*dHdvpa[kvpap,kvperpp] +
                                                # end parallel flux, start of perpendicular flux
                                                YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY3par[kvpap_local,jvpap_local,ivpa_local]*d2Gdvperpdvpa[kvpap,kvperpp] +
                                                YY2perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*d2Gdvperp2[kvpap,kvperpp] -
                                                2.0*(ms/msp)*YY1perp[kvperpp_local,jvperpp_local,ivperp_local]*YY0par[kvpap_local,jvpap_local,ivpa_local]*dHdvperp[kvpap,kvperpp]))
                                                )
                        end
                    end
                end
            end
        end
        return nothing
    end

    # loop over elements
    for ielement_vperp in parallelised_2d_loop_vperp_indices
        YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
        YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
        YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
        YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
        MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
        for ielement_vpa in parallelised_2d_loop_vpa_indices
            YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
            YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
            YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
            YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
            MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
            PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
            # loop over field positions in each element
            for ivperp_local in 1:vperp.ngrid-1
                for ivpa_local in 1:vpa.ngrid-1
                    interior_loop(YY0perp, YY1perp, YY2perp, YY3perp, MMperp, YY0par,
                                  YY1par, YY2par, YY3par, MMpar, PPpar, ivpa_local,
                                  ivperp_local, ielement_vpa, ielement_vperp)
                end
            end
        end
    end
    @_anysv_subblock_synchronize()
    # Add first part of second contribution to boundary points that belong to two elements
    for ielement_vperp in parallelised_2d_loop_vperp_indices
        YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
        YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
        YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
        YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
        MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
        for ielement_vpa in parallelised_2d_loop_vpa_indices
            YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
            YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
            YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
            YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
            MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
            PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
            # loop over field positions in each element
            ivperp_local = vperp.ngrid
            for ivpa_local in 1:vpa.ngrid-1
                interior_loop(YY0perp, YY1perp, YY2perp, YY3perp, MMperp, YY0par, YY1par,
                              YY2par, YY3par, MMpar, PPpar, ivpa_local, ivperp_local,
                              ielement_vpa, ielement_vperp)
            end
        end
    end
    @_anysv_subblock_synchronize()
    # Add second part of second contribution to boundary points that belong to two
    # elements
    for ielement_vperp in parallelised_2d_loop_vperp_indices
        YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
        YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
        YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
        YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
        MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
        for ielement_vpa in parallelised_2d_loop_vpa_indices
            YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
            YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
            YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
            YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
            MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
            PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
            # loop over field positions in each element
            ivpa_local = vpa.ngrid
            for ivperp_local in 1:vperp.ngrid-1
                interior_loop(YY0perp, YY1perp, YY2perp, YY3perp, MMperp, YY0par, YY1par,
                              YY2par, YY3par, MMpar, PPpar, ivpa_local, ivperp_local,
                              ielement_vpa, ielement_vperp)
            end
        end
    end
    @_anysv_subblock_synchronize()
    # Add third part of second contribution to boundary points that belong to two elements
    for ielement_vperp in parallelised_2d_loop_vperp_indices
        YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
        YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
        YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
        YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
        MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
        for ielement_vpa in parallelised_2d_loop_vpa_indices
            YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
            YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
            YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
            YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
            MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
            PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
            # loop over field positions in each element
            ivperp_local = vperp.ngrid
            ivpa_local = vpa.ngrid
            interior_loop(YY0perp, YY1perp, YY2perp, YY3perp, MMperp, YY0par, YY1par,
                          YY2par, YY3par, MMpar, PPpar, ivpa_local, ivperp_local,
                          ielement_vpa, ielement_vperp)
        end
    end
    @_anysv_subblock_synchronize()

    impose_BC_at_zero_vperp=false
    # only support zero bc
    if vpa.bc == "zero" || vperp.bc == "zero"
        function boundary_condition_interior_loop(YY0perp, YY1perp, YY2perp, YY3perp,
                                                  MMperp, YY0par, YY1par, YY2par, YY3par,
                                                  MMpar, PPpar, ivpa_local, ivperp_local,
                                                  ielement_vpa, ielement_vperp)
            vperp_bc = vperp.bc
            vpa_bc = vpa.bc
            for jvperpp_local in 1:ngrid_vperp
                for jvpap_local in 1:ngrid_vpa
                    #ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                    #icp_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,jvpap_local,jvperpp_local)
                    icsc = icsc_func(ivpa_local,jvpap_local,ielement_vpa,
                            ngrid_vpa,nelement_vpa,
                            ivperp_local,jvperpp_local,
                            ielement_vperp,
                            ngrid_vperp,nelement_vperp)

                    lower_boundary_row_vpa = (ielement_vpa == 1 && ivpa_local == 1)
                    upper_boundary_row_vpa = (ielement_vpa == nelement_vpa && ivpa_local == ngrid_vpa)
                    lower_boundary_row_vperp = (ielement_vperp == 1 && ivperp_local == 1)
                    upper_boundary_row_vperp = (ielement_vperp == nelement_vperp && ivperp_local == ngrid_vperp)

                    if lower_boundary_row_vpa && vpa_bc == "zero"
                        if jvpap_local == 1 && ivperp_local == jvperpp_local
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,1.0)
                        else
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,0.0)
                        end
                    elseif upper_boundary_row_vpa && vpa_bc == "zero"
                        if jvpap_local == ngrid_vpa && ivperp_local == jvperpp_local
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,1.0)
                        else
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,0.0)
                        end
                    elseif lower_boundary_row_vperp && impose_BC_at_zero_vperp
                        if jvperpp_local == 1 && ivpa_local == jvpap_local
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,1.0)
                        else
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,0.0)
                        end
                    elseif upper_boundary_row_vperp && vperp_bc == "zero"
                        if jvperpp_local == ngrid_vperp && ivpa_local == jvpap_local
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,1.0)
                        else
                            assign_constructor_value!(CC2D_sparse_constructor,icsc,0.0)
                        end
                    end
                end
            end
            return nothing
        end

        # loop over elements
        for ielement_vperp in parallelised_2d_loop_vperp_indices
            YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
            for ielement_vpa in parallelised_2d_loop_vpa_indices
                YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
                PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
                # loop over field positions in each element
                for ivperp_local in 1:ngrid_vperp-1
                    for ivpa_local in 1:ngrid_vpa-1
                        boundary_condition_interior_loop(YY0perp, YY1perp, YY2perp,
                                                         YY3perp, MMperp, YY0par, YY1par,
                                                         YY2par, YY3par, MMpar,
                                                         PPpar, ivpa_local, ivperp_local,
                                                         ielement_vpa, ielement_vperp)
                    end
                end
            end
        end
        @_anysv_subblock_synchronize()
        # Add first part of second contribution to boundary points that belong to two
        # elements
        for ielement_vperp in parallelised_2d_loop_vperp_indices
            YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
            for ielement_vpa in parallelised_2d_loop_vpa_indices
                YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
                PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
                # loop over field positions in each element
                ivperp_local = ngrid_vperp
                for ivpa_local in 1:ngrid_vpa-1
                    boundary_condition_interior_loop(YY0perp, YY1perp, YY2perp, YY3perp,
                                                     MMperp, YY0par, YY1par, YY2par,
                                                     YY3par, MMpar, PPpar, ivpa_local,
                                                     ivperp_local, ielement_vpa,
                                                     ielement_vperp)
                end
            end
        end
        @_anysv_subblock_synchronize()
        # Add second part of second contribution to boundary points that belong to two
        # elements
        for ielement_vperp in parallelised_2d_loop_vperp_indices
            YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
            for ielement_vpa in parallelised_2d_loop_vpa_indices
                YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
                PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
                # loop over field positions in each element
                ivpa_local = ngrid_vpa
                for ivperp_local in 1:ngrid_vperp-1
                    boundary_condition_interior_loop(YY0perp, YY1perp, YY2perp, YY3perp,
                                                     MMperp, YY0par, YY1par, YY2par,
                                                     YY3par, MMpar, PPpar, ivpa_local,
                                                     ivperp_local, ielement_vpa,
                                                     ielement_vperp)
                end
            end
        end
        @_anysv_subblock_synchronize()
        # Add third part of second contribution to boundary points that belong to two
        # elements
        for ielement_vperp in parallelised_2d_loop_vperp_indices
            YY0perp = YY_arrays.YY0perp[:,:,:,ielement_vperp]
            YY1perp = YY_arrays.YY1perp[:,:,:,ielement_vperp]
            YY2perp = YY_arrays.YY2perp[:,:,:,ielement_vperp]
            YY3perp = YY_arrays.YY3perp[:,:,:,ielement_vperp]
            MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
            for ielement_vpa in parallelised_2d_loop_vpa_indices
                YY0par = YY_arrays.YY0par[:,:,:,ielement_vpa]
                YY1par = YY_arrays.YY1par[:,:,:,ielement_vpa]
                YY2par = YY_arrays.YY2par[:,:,:,ielement_vpa]
                YY3par = YY_arrays.YY3par[:,:,:,ielement_vpa]
                MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
                PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
                # loop over field positions in each element
                ivperp_local = ngrid_vperp
                ivpa_local = ngrid_vpa
                boundary_condition_interior_loop(YY0perp, YY1perp, YY2perp, YY3perp,
                                                 MMperp, YY0par, YY1par, YY2par, YY3par,
                                                 MMpar, PPpar, ivpa_local, ivperp_local,
                                                 ielement_vpa, ielement_vperp)
            end
        end
        @_anysv_subblock_synchronize()
    end # end bc assignment

    @anysv_serial_region begin
        # should improve on this step to avoid recreating the sparse array if possible.
        fkpl_arrays.CC2D_sparse .= create_sparse_matrix(CC2D_sparse_constructor)
        lu!(fkpl_arrays.lu_obj_CC2D, fkpl_arrays.CC2D_sparse)
    end
    return nothing
end

function advance_linearised_test_particle_collisions!(pdf,fkpl_arrays,
                                vpa,vperp,vpa_spectral,vperp_spectral)
    # (the LU decomposition object for)
    # the backward Euler time advance matrix
    # for linearised test particle collisions K * dF = C[dF, F^n+1].
    # this is also the LU decomposition of the approximate Jacobian
    # for the nonlinear residual R = F^n+1 - F^n - C[F^n+1, F^n+1]
    lu_CC = fkpl_arrays.lu_obj_CC2D
    # function to solve K * F^n+1 = M * F^n
    # and return F^n+1 in place in pdf
    # enforce zero BCs on pdf in so that
    # these BCs are imposed via the unit boundary
    # values in CC2D_sparse, in the event BCs are used
    enforce_vpavperp_BCs!(pdf,vpa,vperp,vpa_spectral,vperp_spectral)
    # extra dummy arrays
    pdf_scratch = fkpl_arrays.rhsvpavperp
    pdf_dummy = fkpl_arrays.S_dummy
    # mass matrix for RHS
    MM2D_sparse = fkpl_arrays.MM2D_sparse
    @begin_anysv_region()
    @anysv_serial_region begin
        @views @. pdf_scratch = pdf
        pdf_c = vec(pdf)
        pdf_scratch_c = vec(pdf_scratch)
        pdf_dummy_c = vec(pdf_dummy)
        mul!(pdf_dummy_c, MM2D_sparse, pdf_scratch_c)
        ldiv!(pdf_c,lu_CC,pdf_dummy_c)
    end
    return nothing
end

function calculate_vpavperp_advection_terms!(pdfs,
    dvpadt,fkpl_arrays::fokkerplanck_weakform_arrays_struct,
    vpa,vperp)

    rhsvpavperp = fkpl_arrays.rhsvpavperp
    rhs_advection = fkpl_arrays.rhs_advection
    lu_obj_MM = fkpl_arrays.lu_obj_MM
    # compute the stiffness matrix for vpa vperp advection
    assemble_vpavperp_advection_terms!(rhsvpavperp,pdfs,dvpadt,
        vpa,vperp,fkpl_arrays.YY_arrays)
    # make 1D vector views of 2D arrays
    rhs_advection_c = vec(rhs_advection)
    rhsc = vec(rhsvpavperp)
    # solve mass matrix problem
    ldiv!(rhs_advection_c,lu_obj_MM,rhsc)
    return nothing
end

function assemble_vpavperp_advection_terms!(rhsvpavperp,pdfs,dvpadt,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    # assume below that dvpadt independent of vpa vperp
    @inbounds begin
        @begin_anysv_region()
        @anysv_serial_region begin
            # assemble RHS due to vpa vperp advection
            rhsc = vec(rhsvpavperp)
            @. rhsc = 0.0

            # loop over elements
            for ielement_vperp in 1:vperp.nelement_local
                MMperp = YY_arrays.MMperp[:,:,ielement_vperp]

                for ielement_vpa in 1:vpa.nelement_local
                    MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
                    PPpar = YY_arrays.PPpar[:,:,ielement_vpa]

                    # loop over field positions in each element
                    for ivperp_local in 1:vperp.ngrid
                        for ivpa_local in 1:vpa.ngrid
                            ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
                            # carry out the matrix sum on each 2D element
                            for jvperpp_local in 1:vperp.ngrid
                                jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                                for jvpap_local in 1:vpa.ngrid
                                    jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                                    pdfjj = pdfs[jvpap,jvperpp]

                                    # d  ( dvpadt F) dvpa, after integration by parts, assumming
                                    # dvpadt independent of vpa, vperp, and using the indexing
                                    # of PPpar to get derivatives in correct places.
                                    rhsc[ic_global] += ( - dvpadt * PPpar[ivpa_local,jvpap_local]*
                                                         MMperp[ivperp_local,jvperpp_local]*pdfjj)
                                end
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end
end

"""
Function to allocated an instance of `YY_collision_operator_arrays`.
Calls `get_QQ_local!()` from `gauss_legendre`. Definitions of these
nonlinear stiffness matrices can be found in the docs for `get_QQ_local!()`.
"""
function calculate_YY_arrays(vpa,vperp,vpa_spectral,vperp_spectral)
    YY0perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY1perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY2perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY3perp = Array{mk_float,4}(undef,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    YY0par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    YY1par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    YY2par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    YY3par = Array{mk_float,4}(undef,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    MMpar = Array{mk_float,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    MMperp = Array{mk_float,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement_local)
    PPpar = Array{mk_float,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement_local)
    PPperp = Array{mk_float,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement_local)

    for ielement_vperp in 1:vperp.nelement_local
        @views get_QQ_local!(YY0perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY0")
        @views get_QQ_local!(YY1perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY1")
        @views get_QQ_local!(YY2perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY2")
        @views get_QQ_local!(YY3perp[:,:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"YY3")
        @views get_QQ_local!(MMperp[:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"M")
        @views get_QQ_local!(PPperp[:,:,ielement_vperp],ielement_vperp,vperp_spectral.lobatto,vperp_spectral.radau,vperp,"P")

    end
     for ielement_vpa in 1:vpa.nelement_local
        @views get_QQ_local!(YY0par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY0")
        @views get_QQ_local!(YY1par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY1")
        @views get_QQ_local!(YY2par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY2")
        @views get_QQ_local!(YY3par[:,:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"YY3")
        @views get_QQ_local!(MMpar[:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"M")
        @views get_QQ_local!(PPpar[:,:,ielement_vpa],ielement_vpa,vpa_spectral.lobatto,vpa_spectral.radau,vpa,"P")
     end

    return YY_collision_operator_arrays(YY0perp,YY1perp,YY2perp,YY3perp,
                                        YY0par,YY1par,YY2par,YY3par,
                                        MMpar,MMperp, PPpar,PPperp)
end

"""
Function to assemble the RHS of the kinetic equation due to the collision operator,
in weak form. Once the array `rhsvpavperp` contains the assembled weak-form collision operator,
a mass matrix solve still must be carried out to find the time derivative of the distribution function
due to collisions. This function uses a purely serial algorithm for testing purposes.
"""
function assemble_explicit_collision_operator_rhs_serial!(rhsvpavperp,pdfs,d2Gspdvpa2,d2Gspdvperpdvpa,
    d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    @inbounds begin
        @begin_anysv_region()
        @anysv_serial_region begin
            # assemble RHS of collision operator
            rhsc = vec(rhsvpavperp)
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
                            ic_global = get_global_compound_index(vpa,vperp,ielement_vpa,ielement_vperp,ivpa_local,ivperp_local)
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
end

"""
Function to assemble the RHS of the kinetic equation due to the collision operator,
in weak form. Once the array `rhsvpavperp` contains the assembled weak-form collision operator,
a mass matrix solve still must be carried out to find the time derivative of the distribution function
due to collisions. This function uses a purely parallel algorithm and may be tested by comparing to
`assemble_explicit_collision_operator_rhs_serial!()`. The inner-most loop of the function is
in `assemble_explicit_collision_operator_rhs_parallel_inner_loop()`.
"""
function assemble_explicit_collision_operator_rhs_parallel!(rhsvpavperp,pdfs,d2Gspdvpa2,d2Gspdvperpdvpa,
    d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    # assemble RHS of collision operator
    @begin_anysv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        rhsvpavperp[ivpa,ivperp] = 0.0
    end

    # loop over collocation points to benefit from shared-memory parallelism
    ngrid_vpa, ngrid_vperp = vpa.ngrid, vperp.ngrid
    vperp_igrid_full = vperp.igrid_full
    vpa_igrid_full = vpa.igrid_full
    @loop_vperp_vpa ivperp_global ivpa_global begin
        igrid_vpa, ielement_vpax, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperpx, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa_global,ivperp_global,vpa,vperp)
        # loop over elements belonging to this collocation point
        for ielement_vperp in ielement_vperp_low:ielement_vperp_hi
            # correct local ivperp in the case that we on a boundary point
            ivperp_local = igrid_vperp + (ielement_vperp - ielement_vperp_low)*(1-ngrid_vperp)
            @views YY0perp = YY_arrays.YY0perp[:,:,ivperp_local,ielement_vperp]
            @views YY1perp = YY_arrays.YY1perp[:,:,ivperp_local,ielement_vperp]
            @views YY2perp = YY_arrays.YY2perp[:,:,ivperp_local,ielement_vperp]
            @views YY3perp = YY_arrays.YY3perp[:,:,ivperp_local,ielement_vperp]
            vperp_igrid_full_view = @view vperp_igrid_full[:,ielement_vperp]

            for ielement_vpa in ielement_vpa_low:ielement_vpa_hi
                # correct local ivpa in the case that we on a boundary point
                ivpa_local = igrid_vpa + (ielement_vpa - ielement_vpa_low)*(1-ngrid_vpa)
                @views YY0par = YY_arrays.YY0par[:,:,ivpa_local,ielement_vpa]
                @views YY1par = YY_arrays.YY1par[:,:,ivpa_local,ielement_vpa]
                @views YY2par = YY_arrays.YY2par[:,:,ivpa_local,ielement_vpa]
                @views YY3par = YY_arrays.YY3par[:,:,ivpa_local,ielement_vpa]
                vpa_igrid_full_view = @view vpa_igrid_full[:,ielement_vpa]

                # carry out the matrix sum on each 2D element
                rhsvpavperp[ivpa_global,ivperp_global] +=
                    assemble_explicit_collision_operator_rhs_parallel_inner_loop(
                        nussp, ms, msp, YY0perp, YY0par, YY1perp, YY1par, YY2perp, YY2par,
                        YY3perp, YY3par, pdfs, d2Gspdvpa2, d2Gspdvperpdvpa, d2Gspdvperp2,
                        dHspdvpa, dHspdvperp, ngrid_vperp, vperp_igrid_full_view,
                        ngrid_vpa, vpa_igrid_full_view)
            end
        end
    end
    return nothing
end

"""
The inner-most loop of the parallel collision operator assembly. Called in `assemble_explicit_collision_operator_rhs_parallel!()`.
"""
function assemble_explicit_collision_operator_rhs_parallel_inner_loop(
        nussp, ms, msp, YY0perp, YY0par, YY1perp, YY1par, YY2perp, YY2par, YY3perp,
        YY3par, pdfs, d2Gspdvpa2, d2Gspdvperpdvpa, d2Gspdvperp2, dHspdvpa, dHspdvperp,
        ngrid_vperp, vperp_igrid_full_view, ngrid_vpa, vpa_igrid_full_view)
    @inbounds begin
        # carry out the matrix sum on each 2D element
        result = 0.0
        for jvperpp_local in 1:ngrid_vperp
            jvperpp = vperp_igrid_full_view[jvperpp_local]
            for kvperpp_local in 1:ngrid_vperp
                kvperpp = vperp_igrid_full_view[kvperpp_local]
                YY0perp_kj = YY0perp[kvperpp_local,jvperpp_local]
                YY1perp_kj = YY1perp[kvperpp_local,jvperpp_local]
                YY2perp_kj = YY2perp[kvperpp_local,jvperpp_local]
                YY3perp_kj = YY3perp[kvperpp_local,jvperpp_local]
                for jvpap_local in 1:ngrid_vpa
                    jvpap = vpa_igrid_full_view[jvpap_local]
                    pdfjj = pdfs[jvpap,jvperpp]
                    temp = 0.0
                    for kvpap_local in 1:ngrid_vpa
                        kvpap = vpa_igrid_full_view[kvpap_local]
                        YY0par_kj = YY0par[kvpap_local,jvpap_local]
                        YY1par_kj = YY1par[kvpap_local,jvpap_local]
                        d2Gspdvperpdvpa_kk = d2Gspdvperpdvpa[kvpap,kvperpp]
                        # first three lines represent parallel flux terms
                        # second three lines represent perpendicular flux terms
                        temp += (YY0perp_kj*YY2par[kvpap_local,jvpap_local]*d2Gspdvpa2[kvpap,kvperpp] +
                                 YY3perp_kj*YY1par_kj*d2Gspdvperpdvpa_kk -
                                 2.0*(ms/msp)*YY0perp_kj*YY1par_kj*dHspdvpa[kvpap,kvperpp] +
                                 # end parallel flux, start of perpendicular flux
                                 YY1perp_kj*YY3par[kvpap_local,jvpap_local]*d2Gspdvperpdvpa_kk +
                                 YY2perp_kj*YY0par_kj*d2Gspdvperp2[kvpap,kvperpp] -
                                 2.0*(ms/msp)*YY1perp_kj*YY0par_kj*dHspdvperp[kvpap,kvperpp])
                    end
                    result += temp * pdfjj
                end
            end
        end
        result *= -nussp

        return result
    end
end

"""
Function to assemble the RHS of the kinetic equation due to the collision operator,
in weak form, when the distribution function appearing the derivatives is known analytically.
The inner-most loop of the function is
in `assemble_explicit_collision_operator_rhs_parallel_analytical_inputs_inner_loop()`.
"""
function assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!(rhsvpavperp,pdfs,dpdfsdvpa,dpdfsdvperp,d2Gspdvpa2,d2Gspdvperpdvpa,
    d2Gspdvperp2,dHspdvpa,dHspdvperp,ms,msp,nussp,
    vpa,vperp,YY_arrays::YY_collision_operator_arrays)
    # assemble RHS of collision operator
    @begin_anysv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        rhsvpavperp[ivpa,ivperp] = 0.0
    end

    # loop over collocation points to benefit from shared-memory parallelism
    ngrid_vpa, ngrid_vperp = vpa.ngrid, vperp.ngrid
    vperp_igrid_full = vperp.igrid_full
    vpa_igrid_full = vpa.igrid_full
    @loop_vperp_vpa ivperp_global ivpa_global begin
        igrid_vpa, ielement_vpax, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperpx, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa_global,ivperp_global,vpa,vperp)
        # loop over elements belonging to this collocation point
        for ielement_vperp in ielement_vperp_low:ielement_vperp_hi
            # correct local ivperp in the case that we on a boundary point
            ivperp_local = igrid_vperp + (ielement_vperp - ielement_vperp_low)*(1-ngrid_vperp)
            @views YY0perp = YY_arrays.YY0perp[:,:,ivperp_local,ielement_vperp]
            @views YY1perp = YY_arrays.YY1perp[:,:,ivperp_local,ielement_vperp]
            @views YY2perp = YY_arrays.YY2perp[:,:,ivperp_local,ielement_vperp]
            @views YY3perp = YY_arrays.YY3perp[:,:,ivperp_local,ielement_vperp]
            vperp_igrid_full_view = @view vperp_igrid_full[:,ielement_vperp]

            for ielement_vpa in ielement_vpa_low:ielement_vpa_hi
                # correct local ivpa in the case that we on a boundary point
                ivpa_local = igrid_vpa + (ielement_vpa - ielement_vpa_low)*(1-ngrid_vpa)
                @views YY0par = YY_arrays.YY0par[:,:,ivpa_local,ielement_vpa]
                @views YY1par = YY_arrays.YY1par[:,:,ivpa_local,ielement_vpa]
                @views YY2par = YY_arrays.YY2par[:,:,ivpa_local,ielement_vpa]
                @views YY3par = YY_arrays.YY3par[:,:,ivpa_local,ielement_vpa]
                vpa_igrid_full_view = @view vpa_igrid_full[:,ielement_vpa]

                # carry out the matrix sum on each 2D element
                rhsvpavperp[ivpa_global,ivperp_global] +=
                    assemble_explicit_collision_operator_rhs_parallel_analytical_inputs_inner_loop(
                        nussp, ms, msp, pdfs, dpdfsdvpa, dpdfsdvperp, d2Gspdvperp2,
                        d2Gspdvpa2, d2Gspdvperpdvpa, dHspdvperp, dHspdvpa, YY0perp,
                        YY0par, YY1perp, YY1par, ngrid_vperp, vperp_igrid_full_view,
                        ngrid_vpa, vpa_igrid_full_view)
            end
        end
    end
    return nothing
end

"""
The inner-most loop of `assemble_explicit_collision_operator_rhs_parallel_analytical_inputs!()`.
"""
# Separate function for inner loop, possible optimization??
function assemble_explicit_collision_operator_rhs_parallel_analytical_inputs_inner_loop(
        nussp, ms, msp, pdfs, dpdfsdvpa, dpdfsdvperp, d2Gspdvperp2,
        d2Gspdvpa2, d2Gspdvperpdvpa, dHspdvperp, dHspdvpa, YY0perp, YY0par, YY1perp,
        YY1par, ngrid_vperp, vperp_igrid_full_view, ngrid_vpa, vpa_igrid_full_view)

    @inbounds begin
        # carry out the matrix sum on each 2D element
        result = 0.0
        for jvperpp_local in 1:ngrid_vperp
            jvperpp = vperp_igrid_full_view[jvperpp_local]
            for kvperpp_local in 1:ngrid_vperp
                kvperpp = vperp_igrid_full_view[kvperpp_local]
                YY0perp_kj = YY0perp[kvperpp_local,jvperpp_local]
                YY1perp_kj = YY1perp[kvperpp_local,jvperpp_local]
                for jvpap_local in 1:ngrid_vpa
                    jvpap = vpa_igrid_full_view[jvpap_local]
                    pdfs_jj = pdfs[jvpap,jvperpp]
                    dpdfsdvperp_jj = dpdfsdvperp[jvpap,jvperpp]
                    dpdfsdvpa_jj = dpdfsdvpa[jvpap,jvperpp]
                    for kvpap_local in 1:ngrid_vpa
                        kvpap = vpa_igrid_full_view[kvpap_local]
                        YY0par_kj = YY0par[kvpap_local,jvpap_local]
                        YY1par_kj = YY1par[kvpap_local,jvpap_local]
                        d2Gspdvperpdvpa_kk = d2Gspdvperpdvpa[kvpap,kvperpp]
                        # first three lines represent parallel flux terms
                        # second three lines represent perpendicular flux terms
                        result +=
                            -nussp*(YY0perp_kj*YY1par_kj*dpdfsdvpa_jj*d2Gspdvpa2[kvpap,kvperpp] +
                                    YY0perp_kj*YY1par_kj*dpdfsdvperp_jj*d2Gspdvperpdvpa_kk -
                                    2.0*(ms/msp)*YY0perp_kj*YY1par_kj*pdfs_jj*dHspdvpa[kvpap,kvperpp] +
                                    # end parallel flux, start of perpendicular flux
                                    YY1perp_kj*YY0par_kj*dpdfsdvpa_jj*d2Gspdvperpdvpa_kk +
                                    YY1perp_kj*YY0par_kj*dpdfsdvperp_jj*d2Gspdvperp2[kvpap,kvperpp] -
                                    2.0*(ms/msp)*YY1perp_kj*YY0par_kj*pdfs_jj*dHspdvperp[kvpap,kvperpp])
                    end
                end
            end
        end

        return result
    end
end

"""
Elliptic solve function.

    field: the solution
    source: the source function on the RHS
    boundary data: the known values of field at infinity
    lu_object_lhs: the object for the differential operator that defines field
    matrix_rhs: the weak matrix acting on the source vector
    vpa, vperp: coordinate structs

Note: all variants of `elliptic_solve!()` run only in serial. They do not handle
shared-memory parallelism themselves. The calling site must ensure that
`elliptic_solve!()` is only called by one process in a shared-memory block.
"""
function elliptic_solve!(field,source,boundary_data::vpa_vperp_boundary_data,
            lu_object_lhs,matrix_rhs,rhsvpavperp,vpa,vperp)
    @inbounds begin
        # assemble the rhs of the weak system

        # get data into the compound index format
        sc = vec(source)
        fc = vec(field)
        rhsc = vec(rhsvpavperp)
        mul!(rhsc,matrix_rhs,sc)
        # enforce the boundary conditions
        enforce_dirichlet_bc!(rhsvpavperp,vpa,vperp,boundary_data)
        # solve the linear system
        ldiv!(fc, lu_object_lhs, rhsc)

        return nothing
    end
end
# same as above but source is made of two different terms
# with different weak matrices
function elliptic_solve!(field,source_1,source_2,boundary_data::vpa_vperp_boundary_data,
            lu_object_lhs,matrix_rhs_1,matrix_rhs_2,rhs,vpa,vperp)

    @inbounds begin
        # assemble the rhs of the weak system

        # get data into the compound index format
        sc_1 = vec(source_1)
        sc_2 = vec(source_2)
        rhsc = vec(rhs)
        fc = vec(field)

        # Do  rhsc = matrix_rhs_1*sc_1
        mul!(rhsc, matrix_rhs_1, sc_1)

        # Do rhsc = matrix_rhs_2*sc_2 + rhsc
        mul!(rhsc, matrix_rhs_2, sc_2, 1.0, 1.0)

        # enforce the boundary conditions
        enforce_dirichlet_bc!(rhs,vpa,vperp,boundary_data)
        # solve the linear system
        ldiv!(fc, lu_object_lhs, rhsc)

        return nothing
    end
end

"""
Same as `elliptic_solve!()` above but no Dirichlet boundary conditions are imposed,
because the function is only used where the `lu_object_lhs` is derived from a mass matrix.
The source is made of two different terms with different weak matrices
because of the form of the only algebraic equation that we consider.

Note: `algebraic_solve!()` run only in serial. They do not handle shared-memory
parallelism themselves. The calling site must ensure that `algebraic_solve!()` is only
called by one process in a shared-memory block.
"""
function algebraic_solve!(field,source_1,source_2,boundary_data::vpa_vperp_boundary_data,
            lu_object_lhs,matrix_rhs_1,matrix_rhs_2,rhs,vpa,vperp)

    @inbounds begin
        # assemble the rhs of the weak system

        # get data into the compound index format
        sc_1 = vec(source_1)
        sc_2 = vec(source_2)
        rhsc = vec(rhs)
        fc = vec(field)

        # Do  rhsc = matrix_rhs_1*sc_1
        mul!(rhsc, matrix_rhs_1, sc_1)

        # Do rhsc = matrix_rhs_2*sc_2 + rhsc
        mul!(rhsc, matrix_rhs_2, sc_2, 1.0, 1.0)

        # solve the linear system
        ldiv!(fc, lu_object_lhs, rhsc)

        return nothing
    end
end

"""
Function to solve the appropriate elliptic PDEs to find the
Rosenbluth potentials. First, we calculate the Rosenbluth potentials
at the boundary with the direct integration method. Then, we use this
data to solve the elliptic PDEs with the boundary data providing an
accurate Dirichlet boundary condition on the maximum `vpa` and `vperp`
of the domain. We use the sparse LU decomposition from the LinearAlgebra package
to solve the PDE matrix equations.
"""
function calculate_rosenbluth_potentials_via_elliptic_solve!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,
             vpa,vperp,vpa_spectral,vperp_spectral,fkpl_arrays::fokkerplanck_weakform_arrays_struct;
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,calculate_dGdvperp=false,
             boundary_data_option=direct_integration)

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
    rhsvpavperp = fkpl_arrays.rhsvpavperp
    rhsvpavperp_copy1 = fkpl_arrays.rhsvpavperp_copy1
    rhsvpavperp_copy2 = fkpl_arrays.rhsvpavperp_copy2
    rhsvpavperp_copy3 = fkpl_arrays.rhsvpavperp_copy3

    # calculate the boundary data
    if boundary_data_option == multipole_expansion
        calculate_rosenbluth_potential_boundary_data_multipole!(rpbd,ffsp_in,vpa,vperp,vpa_spectral,vperp_spectral,
          calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
    elseif boundary_data_option == delta_f_multipole # use a variant of the multipole method
        calculate_rosenbluth_potential_boundary_data_delta_f_multipole!(rpbd,ffsp_in,S_dummy,vpa,vperp,vpa_spectral,vperp_spectral,
          calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
    elseif boundary_data_option == direct_integration  # use direct integration on the boundary
        calculate_rosenbluth_potential_boundary_data!(rpbd,bwgt,ffsp_in,vpa,vperp,vpa_spectral,vperp_spectral,
         calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
    else
        error("No valid boundary_data_option specified. \n
              Pick  boundary_data_option='$multipole_expansion' \n
              or  boundary_data_option='$delta_f_multipole' \n
              or boundary_data_option='$direct_integration'")
    end
    # carry out the elliptic solves required
    @begin_anysv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        S_dummy[ivpa,ivperp] = -(4.0*pi)*ffsp_in[ivpa,ivperp]
    end

    # Can run the following three solves in parallel
    # The solves run on ranks 0, 1 and 2 of the subblock respectively, but modulo the size
    # of the subblock (to ensure that the ranks doing work are never outside the
    # subblock, if the size of the subblock is less than 3).
    @begin_anysv_region()
    if anysv_subblock_rank[] == 0 % anysv_subblock_size[]
        elliptic_solve!(HH, S_dummy, rpbd.H_data, lu_obj_LP, MM2D_sparse, rhsvpavperp,
                        vpa, vperp)
    end
    if anysv_subblock_rank[] == 1 % anysv_subblock_size[]
        elliptic_solve!(dHdvpa, S_dummy, rpbd.dHdvpa_data, lu_obj_LP, PPpar2D_sparse,
                        rhsvpavperp_copy1, vpa, vperp)
    end
    if anysv_subblock_rank[] == 2 % anysv_subblock_size[]
        elliptic_solve!(dHdvperp, S_dummy, rpbd.dHdvperp_data, lu_obj_LV, PUperp2D_sparse,
                        rhsvpavperp_copy2, vpa, vperp)
    end

    @begin_anysv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp]
    end

    # The following four solves can be done in parallel. Note: do the two that are always
    # done on ranks 0 and 1 of the subblock and the first optional one that actually needs
    # doing on rank 3 to maximise the chances that all solves get run on separate
    # processes (i.e. they will be on separate processes as long as there are at least 2
    # ranks in the subblock if both conditions calculate_GG and calculate_dGdvperp are
    # false; at least 3 ranks if only one of the conditions is true; and at least 4 ranks
    # if both conditions are true).
    @begin_anysv_region()
    if calculate_GG
        if anysv_subblock_rank[] == 2 % anysv_subblock_size[]
            elliptic_solve!(GG, S_dummy, rpbd.G_data, lu_obj_LP, MM2D_sparse,
                            rhsvpavperp_copy2, vpa, vperp)
        end
    end
    if calculate_dGdvperp || algebraic_solve_for_d2Gdvperp2
        if anysv_subblock_rank[] == (calculate_GG ? 3 : 2) % anysv_subblock_size[]
            elliptic_solve!(dGdvperp, S_dummy, rpbd.dGdvperp_data, lu_obj_LV,
                            PUperp2D_sparse, rhsvpavperp_copy3, vpa, vperp)
        end
    end
    if anysv_subblock_rank[] == 0 % anysv_subblock_size[]
        elliptic_solve!(d2Gdvpa2, S_dummy, rpbd.d2Gdvpa2_data, lu_obj_LP, KKpar2D_sparse,
                        rhsvpavperp, vpa, vperp)
    end
    if anysv_subblock_rank[] == 1 % anysv_subblock_size[]
        elliptic_solve!(d2Gdvperpdvpa, S_dummy, rpbd.d2Gdvperpdvpa_data, lu_obj_LV,
                        PPparPUperp2D_sparse, rhsvpavperp_copy1, vpa, vperp)
    end

    if algebraic_solve_for_d2Gdvperp2
        @begin_anysv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp] - d2Gdvpa2[ivpa,ivperp]
            Q_dummy[ivpa,ivperp] = -dGdvperp[ivpa,ivperp]
        end
        @begin_anysv_region()
        @anysv_serial_region begin
            # use the algebraic solve function to find
            # d2Gdvperp2 = 2H - d2Gdvpa2 - (1/vperp)dGdvperp
            # using a weak form
            algebraic_solve!(d2Gdvperp2, S_dummy, Q_dummy, rpbd.d2Gdvperp2_data,
                             lu_obj_MM, MM2D_sparse, MMparMNperp2D_sparse, rhsvpavperp,
                             vpa, vperp)
        end
    else
        # solve a weak-form PDE for d2Gdvperp2
        @begin_anysv_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            #S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp] # <- this is already the value of
                                                        #    S_dummy calculated above
            Q_dummy[ivpa,ivperp] = 2.0*d2Gdvpa2[ivpa,ivperp]
        end
        @begin_anysv_region()
        @anysv_serial_region begin
            elliptic_solve!(d2Gdvperp2, S_dummy, Q_dummy, rpbd.d2Gdvperp2_data, lu_obj_LB,
                            KPperp2D_sparse, MMparMNperp2D_sparse, rhsvpavperp, vpa,
                            vperp)
        end
    end
    @_anysv_subblock_synchronize
    return nothing
end

"""
Function to calculate Rosenbluth potentials in the entire
domain of `(vpa,vperp)` by direct integration.
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
    @begin_anysv_vperp_region()
    @loop_vperp ivperp begin
        @views derivative!(dfdvpa[:,ivperp], ffsp_in[:,ivperp], vpa, vpa_spectral)
    end
    @begin_anysv_vpa_region()
    @loop_vpa ivpa begin
        @views derivative!(dfdvperp[ivpa,:], ffsp_in[ivpa,:], vperp, vperp_spectral)
        @views derivative!(d2fdvperpdvpa[ivpa,:], dfdvpa[ivpa,:], vperp, vperp_spectral)
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
Function to calculate Rosenbluth potentials for shifted Maxwellians
using an analytical specification
"""
function calculate_rosenbluth_potentials_via_analytical_Maxwellian!(GG,HH,dHdvpa,dHdvperp,
    d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,vpa,vperp,mass)
    @begin_anysv_region()
    dens = get_density(ffsp_in, vpa, vperp)
        upar = get_upar(ffsp_in, dens, vpa, vperp, false)
        pressure = get_p(ffsp_in, dens, upar, vpa, vperp, false, false)
        vth = sqrt(2.0*pressure/(dens*mass))
    @begin_anysv_vperp_vpa_region()
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
    @_anysv_subblock_synchronize()
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
    @begin_anysv_vperp_vpa_region()
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
Function to enforce boundary conditions on the collision operator
result to be consistent with the boundary conditions imposed on the
distribution function.
"""
function enforce_vpavperp_BCs!(pdf,vpa,vperp,vpa_spectral,vperp_spectral;
        lower_wall=false, upper_wall=false)
    nvpa = vpa.n
    nvperp = vperp.n
    ngrid_vperp = vperp.ngrid
    D0 = vperp_spectral.radau.D0
    # vpa boundary conditions
    # zero at infinity
    if vpa.bc == "zero"
        @begin_anysv_vperp_region()
        @loop_vperp ivperp begin
            pdf[1,ivperp] = 0.0
            pdf[nvpa,ivperp] = 0.0
        end
    end
    # vperp boundary conditions
    # zero boundary condition at infinity
    # set regularity condition d F / d vperp = 0 at vperp = 0
    # adjust F(vperp = 0) so that d F / d vperp = 0 at vperp = 0
    @begin_anysv_vpa_region()
    if vperp.bc in ("zero", "zero-impose-regularity")
        @loop_vpa ivpa begin
            pdf[ivpa,nvperp] = 0.0
        end
    end
    if vperp.bc == "zero-impose-regularity"
        buffer = @view vperp.scratch[1:ngrid_vperp-1]
        @loop_vpa ivpa begin
            @views @. buffer = D0[2:ngrid_vperp] * pdf[ivpa,2:ngrid_vperp]
            pdf[ivpa,1] = -sum(buffer)/D0[1]
        end
    end
    if lower_wall
        # vpa_mask > 1 if -vpa.L < vpa.grid < 0+
        vpa_mask = vpa.mask_low
        @begin_anysv_vperp_region()
        @loop_vperp ivperp begin
            pdf[:,ivperp] .*= vpa_mask
        end
    end
    if upper_wall
        # vpa_mask > 1 if vpa.L > vpa.grid > -0
        vpa_mask = vpa.mask_up
        @begin_anysv_vperp_region()
        @loop_vperp ivperp begin
            pdf[:,ivperp] .*= vpa_mask
        end
    end
    return nothing
end

"""
Function to interpolate `f(vpa,vperp)` from one
velocity grid to another, assuming that both
grids are represented by `(vpa,vperp)` in normalised units,
but have different normalisation factors
defining the meaning of these grids in physical units. E.g.,

     vpai, vperpi = ci * vpa, ci * vperp
     vpae, vperpe = ce * vpa, ce * vperp

with `ci = sqrt(Ti/mi)`, `ce = sqrt(Te/mi)`

`scalefac = ci / ce` is the ratio of the
two reference speeds.
"""
function interpolate_2D_vspace!(pdf_out,pdf_in,vpa,vperp,scalefac)

    @begin_anysv_vperp_vpa_region()
    # loop over points in the output interpolated dataset
    @loop_vperp ivperp begin
        vperp_val = vperp.grid[ivperp]*scalefac
        # get element for interpolation data
        iel_vperp = ielement_loopup(vperp_val,vperp)
        if iel_vperp < 1 # vperp_interp outside of range of vperp.grid
            @loop_vpa ivpa begin
                pdf_out[ivpa,ivperp] = 0.0
            end
            continue
        else
            # get nodes for interpolation
            ivperpmin, ivperpmax = vperp.igrid_full[1,iel_vperp], vperp.igrid_full[vperp.ngrid,iel_vperp]
            vperp_nodes = vperp.grid[ivperpmin:ivperpmax]
            #print("vperp: ",iel_vperp, " ", vperp_nodes," ",vperp_val)

        end
        @loop_vpa ivpa begin
            vpa_val = vpa.grid[ivpa]*scalefac
            # get element for interpolation data
            iel_vpa = ielement_loopup(vpa_val,vpa)
            if iel_vpa < 1 # vpa_interp outside of range of vpa.grid
                pdf_out[ivpa,ivperp] = 0.0
                continue
            else
                # get nodes for interpolation
                ivpamin, ivpamax = vpa.igrid_full[1,iel_vpa], vpa.igrid_full[vpa.ngrid,iel_vpa]
                vpa_nodes = vpa.grid[ivpamin:ivpamax]
                #print("vpa: ", iel_vpa, " ", vpa_nodes," ",vpa_val)

                # do the interpolation
                pdf_out[ivpa,ivperp] = 0.0
                for ivperpgrid in 1:vperp.ngrid
                   # index for referencing pdf_in on orginal grid
                   ivperpp = vperp.igrid_full[ivperpgrid,iel_vperp]
                   # interpolating polynomial value at ivperpp for interpolation
                   vperppoly = lagrange_poly(ivperpgrid,vperp_nodes,vperp_val)
                   for ivpagrid in 1:vpa.ngrid
                       # index for referencing pdf_in on orginal grid
                       ivpap = vpa.igrid_full[ivpagrid,iel_vpa]
                       # interpolating polynomial value at ivpap for interpolation
                       vpapoly = lagrange_poly(ivpagrid,vpa_nodes,vpa_val)
                       pdf_out[ivpa,ivperp] += vpapoly*vperppoly*pdf_in[ivpap,ivperpp]
                   end
                end
            end
        end
    end
    return nothing
end
# Alternative version that should be faster - to be tested
#function interpolate_2D_vspace!(pdf_out, pdf_in, vpa, vpa_spectral, vperp, vperp_spectral,
#                                scalefac, pdf_buffer)
#    newgrid_vperp = vperp.scratch .= scalefac .* vperp.grid
#    newgrid_vpa = vpa.scratch .= scalefac .* vpa.grid
#
#    @begin_anysv_vpa_region()
#    @loop_vpa ivpa begin
#        @views interpolate_to_grid_1d!(pdf_buffer[ivpa,:], newgrid_vperp,
#                                       pdf_in[ivpa,:], vperp, vperp_spectral)
#    end
#
#    @begin_anysv_vperp_region()
#    @loop_vperp ivperp begin
#        @views interpolate_to_grid_1d!(pdf_out[:,ivperp], newgrid_vpa,
#                                       pdf_buffer[:,ivperp], vpa, vpa_spectral)

#    end
#end

"""
Function to find the element in which x sits.
"""
function ielement_loopup(x,coord)
    @inbounds begin
        xebs = coord.element_boundaries
        nelement = coord.nelement_global
        zero = 1.0e-14
        ielement = -1
        # find the element
        for j in 1:nelement
            # check for internal points
            if (x - xebs[j])*(xebs[j+1] - x) > zero
                ielement = j
                break
            # check for boundary points
            elseif (abs(x-xebs[j]) < 100*zero) || (abs(x-xebs[j+1]) < 100*zero && j == nelement)
                ielement = j
                break
            end
        end
        return ielement
    end
end

end
