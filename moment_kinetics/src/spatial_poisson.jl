"""
module for solving Poisson's equation on a spatial cylindrical domain
Take the approach of solving 1D radial equation in a cylinder, which is 
translationally symmetric in the axial direction, using 
Fourier series for the polar coordinate.
"""
module spatial_poisson

export init_spatial_poisson
export spatial_poisson_solve!
export init_spatial_poisson2D
export spatial_poisson2D_solve!

using ..type_definitions: mk_float, mk_int
using ..gauss_legendre: get_QQ_local!
using ..coordinates: coordinate
using ..array_allocation: allocate_float
using SparseArrays: sparse, AbstractSparseArray
using LinearAlgebra: ldiv!, mul!, LU, lu
using SuiteSparse
using ..fourier: fourier_info, fourier_forward_transform!, fourier_backward_transform! 
using ..moment_kinetics_structs: null_spatial_dimension_info
using ..sparse_matrix_functions: icsc_func, ic_func, allocate_sparse_matrix_constructor,
                                 assemble_constructor_data!, assign_constructor_data!,
                                 sparse_matrix_constructor, create_sparse_matrix,
                                 get_global_compound_index
using ..moment_kinetics_structs: weak_discretization_info
                                 
struct poisson_arrays 
   # an array of lu objects for solving the mth polar harmonic of Poisson's equation
   laplacian_lu_objs::Array{SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int},1}
   # an Array of 2D arrays (one 1D operator on r for
   # each Fourier component in the polar coordinate
   laplacian::Array{mk_float,3}
   # the matrix that needs to multiply the nodal values of the source function
   sourcevec::Array{mk_float,2}
   rhohat::Array{Complex{mk_float},2}
   rhs_dummy::Array{Complex{mk_float},1}
   phi_dummy::Array{Complex{mk_float},1}
end

"""
duplicate function, should be moved to coordinates
"""
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


function mwavenumber(polar,im)
   npolar = polar.n
   if mod(npolar,2) == 0
      imid = mk_int((npolar/2))
   else
      imid = mk_int(((npolar-1)/2))
   end
   mwvc = (2.0*pi/polar.L)
   if npolar > 1
      if im < imid+1
         mwvc *= (im-1)
      else
         mwvc *= ((im-1)-npolar)
      end
   else
      mwvc *= (im-1)
   end   
   return mwvc
end

"""
function to initialise the arrays needed for the weak-form
Poisson's equation problem
 radial = a coordinate struct of a cylindrical=true coordinate
 polar = a coordinate struct of a periodic coordinate with length = 2pi
"""
function init_spatial_poisson(radial::coordinate, polar::coordinate, radial_spectral)
   npolar = polar.n
   nrtot = radial.n
   nrgrid = radial.ngrid
   nrelement = radial.nelement_global
   laplacian = allocate_float(nrtot,nrtot,npolar)
   sourcevec = allocate_float(nrtot,nrtot)
   rhohat = allocate_float(npolar,nrtot)
   rhs_dummy = allocate_float(nrtot)
   phi_dummy = allocate_float(nrtot)
   MR = allocate_float(nrgrid,nrgrid)
   MN = allocate_float(nrgrid,nrgrid)
   KJ = allocate_float(nrgrid,nrgrid)
   PP = allocate_float(nrgrid,nrgrid)
   # initialise to zero
   @. laplacian = 0.0
   @. sourcevec = 0.0
   for im in 1:npolar
      mwn = mwavenumber(polar,im)
      for irel in 1:nrelement 
          imin, imax = get_imin_imax(radial,irel)
          get_QQ_local!(MN,irel,radial_spectral.lobatto,radial_spectral.radau,radial,"N")
          get_QQ_local!(KJ,irel,radial_spectral.lobatto,radial_spectral.radau,radial,"J")
          get_QQ_local!(PP,irel,radial_spectral.lobatto,radial_spectral.radau,radial,"P")
          # assemble the Laplacian matrix 
          @. laplacian[imin:imax,imin:imax,im] += KJ - PP - ((mwn)^2)*MN
      end
      # set rows for Dirichlet BCs on phi
      laplacian[nrtot,:,im] .= 0.0
      laplacian[nrtot,nrtot,im] = 1.0
   end
   for irel in 1:nrelement 
       imin, imax = get_imin_imax(radial,irel)
       get_QQ_local!(MR,irel,radial_spectral.lobatto,radial_spectral.radau,radial,"R")
       # assemble the weak-form source vector that should multiply the node values of the source
       @. sourcevec[imin:imax,imin:imax] += MR
   end
   
   # construct the sparse Laplacian matrix
   laplacian_sparse = Array{AbstractSparseArray{mk_float,mk_int,2},1}(undef,npolar)
   for im in 1:npolar
      laplacian_sparse[im] = sparse(laplacian[:,:,im])
   end
   # construct LU objects
   laplacian_lu_objs = Array{SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int},1}(undef,npolar)
   for im in 1:npolar
      laplacian_lu_objs[im] = lu(laplacian_sparse[im])
   end
   return poisson_arrays(laplacian_lu_objs,laplacian,sourcevec,rhohat,rhs_dummy,phi_dummy)
end

"""
Function to find the solution to 

nabla^2 phi = rho in cylindrical polar coordinates
nabla^2 phi = (1/r)d/dr(r dphi/dr) + (1/r^2)d^2 phi/dpolar^2

The arguments are 
 
 phi(polar,r) = the function solved for
 rho(polar,r) = the source evaluated at the nodal points
 poisson_arrays = precomputed arrays
 radial = coordinate
 polar = coordinate
 polar_spectral = fourier_info 

The function uses a 1D Fourier transform to convert the 2D Poisson's
equation into M 1D ODEs, which are solved using 1D elemental weak-form matrices.
The Fourier transform reconstructs the solution.
"""
# for now just support npolar = 1
# by skipping the FFT
function spatial_poisson_solve!(phi,rho,poisson_arrays,radial,polar,polar_spectral::Union{fourier_info,null_spatial_dimension_info})
   laplacian_lu_objs = poisson_arrays.laplacian_lu_objs
   sourcevec = poisson_arrays.sourcevec
   phi_dummy = poisson_arrays.phi_dummy
   rhs_dummy = poisson_arrays.rhs_dummy
   rhohat = poisson_arrays.rhohat
   #phihat = poisson_arrays.phihat
   
   npolar = polar.n
   nradial = radial.n
   if npolar > 1
      # first FFT rho to hat{rho} appropriate for using the 1D radial operators
      for irad in 1:nradial
         @views fourier_forward_transform!(rhohat[:,irad], polar_spectral.fext, rho[:,irad], polar_spectral.forward, polar_spectral.imidm, polar_spectral.imidp, polar.ngrid)
      end
   else
      @. rhohat = complex(rho,0.0)
   end
   
   for im in 1:npolar
      # solve the linear system
      # form the rhs vector
      mul!(rhs_dummy,sourcevec,rhohat[im,:])
      # set the Dirichlet BC phi = 0
      rhs_dummy[nradial] = 0.0
      lu_object_lhs = laplacian_lu_objs[im]
      ldiv!(phi_dummy, lu_object_lhs, rhs_dummy)
      rhohat[im,:] = phi_dummy
   end
   
   if npolar > 1
      # finally iFFT from hat{phi} to phi 
      for irad in 1:nradial
         @views fourier_backward_transform!(phi[:,irad], polar_spectral.fext, rhohat[:,irad], polar_spectral.backward, polar_spectral.imidm, polar_spectral.imidp, polar.ngrid)
      end
   else
      @. phi = real(rhohat)
   end
   return nothing
end

# functions associated with a 2D (polar,radial) solver
# using a 2D sparse matrix
struct poisson2D_arrays{M <: AbstractSparseArray{mk_float,mk_int,N} where N}
   # an array of lu objects for solving the mth polar harmonic of Poisson's equation
   laplacian_lu_obj::SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int}
   # an Array of 2D arrays (one 1D operator on r for
   # each Fourier component in the polar coordinate
   laplacian2D::M
   # the matrix that needs to multiply the nodal values of the source function
   massmatrix2D::M
   # dummy array for forming RHS in LU solve
   rhspr::Array{mk_float,2}
end

"""
function to initialise the arrays needed for the 2D weak-form
Poisson's equation problem
 radial = a coordinate struct of a cylindrical=true coordinate
 polar = a coordinate struct of a periodic polar coordinate with length = 2pi
"""
function init_spatial_poisson2D(radial::coordinate, polar::coordinate, 
                                radial_spectral::weak_discretization_info,
                                polar_spectral::weak_discretization_info;
                                use_sparse_init=false)
   npolar = polar.n
   nradial = radial.n
   if use_sparse_init
       LP2D_sparse, MR2D_sparse = init_sparse_laplacian2D(radial,polar,radial_spectral,polar_spectral)
   else
       LP2D_sparse, MR2D_sparse = init_laplacian2D(radial,polar,radial_spectral,polar_spectral)
   end
   laplacian_lu = lu(LP2D_sparse)
   rhspr = allocate_float(npolar,nradial)
   return poisson2D_arrays(laplacian_lu,LP2D_sparse,MR2D_sparse,rhspr)
end

"""
This function makes sparse matrices representing the 2D Laplacian operator.
The compound index runs over the radial and polar coordinates. 
We use sparse matrix constructor functions to build the matrix with minimal
memory usage, avoiding global matrices in the compound index.
We assume that the polar coordinate has Jacobian = 1 
so that we do not need to label elemental matrices when
assembling terms from the boundary conditions.

This function is currently broken due to incorrect sparse indexing functions
which do not handle the periodic boundary conditions correctly.
"""
function init_sparse_laplacian2D(radial::coordinate, polar::coordinate, 
                                radial_spectral::weak_discretization_info,
                                polar_spectral::weak_discretization_info)
    nradial = radial.n
    npolar = polar.n
    
    ngrid_radial = radial.ngrid
    ngrid_polar = polar.ngrid
    nelement_radial = radial.nelement_local
    nelement_polar = polar.nelement_local
    
    ntot_polar = (polar.nelement_local - 1)*(polar.ngrid^2 - 1) + polar.ngrid^2
    ntot_radial = (radial.nelement_local - 1)*(radial.ngrid^2 - 1) + radial.ngrid^2
    nsparse = ntot_polar*ntot_radial
    
    MR2D = allocate_sparse_matrix_constructor(nsparse)
    # Laplacian matrix
    LP2D = allocate_sparse_matrix_constructor(nsparse)
    # local dummy arrays
    MMpolar = Array{mk_float,2}(undef,ngrid_polar,ngrid_polar)
    KKpolar = Array{mk_float,2}(undef,ngrid_polar,ngrid_polar)
    MNradial = Array{mk_float,2}(undef,ngrid_radial,ngrid_radial)
    MRradial = Array{mk_float,2}(undef,ngrid_radial,ngrid_radial)
    KJradial = Array{mk_float,2}(undef,ngrid_radial,ngrid_radial)
    PPradial = Array{mk_float,2}(undef,ngrid_radial,ngrid_radial)
    # loop over elements to carry out assembly
    for ielement_radial in 1:nelement_radial
        # get local radial 1D matrices
        get_QQ_local!(MNradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"N")
        get_QQ_local!(MRradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"R")
        get_QQ_local!(KJradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"J")
        get_QQ_local!(PPradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"P")
        for ielement_polar in 1:nelement_polar
            # get local polar 1D matrices
            get_QQ_local!(MMpolar,ielement_polar,polar_spectral.lobatto,polar_spectral.radau,polar,"M")
            get_QQ_local!(KKpolar,ielement_polar,polar_spectral.lobatto,polar_spectral.radau,polar,"K")
            # loop over grid points
            for iradialp_local in 1:ngrid_radial
                for iradial_local in 1:ngrid_radial
                    for ipolarp_local in 1:ngrid_polar
                        for ipolar_local in 1:ngrid_polar
                            ic_global = get_global_compound_index(polar,radial,ielement_polar,ielement_radial,ipolar_local,iradial_local)
                            icp_global = get_global_compound_index(polar,radial,ielement_polar,ielement_radial,ipolarp_local,iradialp_local)
                            icsc = icsc_func(ipolar_local,ipolarp_local,ielement_polar,
                                           ngrid_polar,nelement_polar,
                                           iradial_local,iradialp_local,
                                           ielement_radial,
                                           ngrid_radial,nelement_radial)
                            # boundary condition possibilities
                            lower_boundary_row_polar = (ielement_polar == 1 && ipolar_local == 1)
                            upper_boundary_row_polar = (ielement_polar == polar.nelement_local && ipolar_local == polar.ngrid)
                            lower_boundary_row_radial = (ielement_radial == 1 && iradial_local == 1)
                            upper_boundary_row_radial = (ielement_radial == radial.nelement_local && iradial_local == radial.ngrid)
                            
                            # Laplacian assembly
                            if upper_boundary_row_radial
                                # Dirichlet boundary condition on outer boundary
                                if iradialp_local == radial.ngrid && ipolar_local == ipolarp_local
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,1.0)
                                else
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,0.0)
                                end
                            elseif upper_boundary_row_polar
                                # ensure periodicity by setting this row to 0 apart from appropriately placed 1, -1.
                                if ipolarp_local == ngrid_polar && iradial_local == iradialp_local
                                    # assign -1 to this point using loop icsc, ic, icp
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,-1.0)
                                    # find other end of polar domain for this ipolar_local
                                    # and set ipolarp_local = ielement_polar = 1.
                                    icp_global_bc = get_global_compound_index(polar,radial,1,ielement_radial,1,iradialp_local)
                                    icsc_bc = icsc_func(ipolar_local,1,1,
                                           ngrid_polar,nelement_polar,
                                           iradial_local,iradialp_local,
                                           ielement_radial,
                                           ngrid_radial,nelement_radial)
                                    # assign 1 to this point using ic from loop and updated icsc, icp
                                    assign_constructor_data!(LP2D,icsc_bc,ic_global,icp_global_bc,1.0)
                                else 
                                    assign_constructor_data!(LP2D,icsc,ic_global,icp_global,0.0)
                                end
                            else 
                                # carry out assembly
                                # of Laplacian matrix data
                                assemble_constructor_data!(LP2D,icsc,ic_global,icp_global,
                                            (MMpolar[ipolar_local,ipolarp_local]*
                                             (KJradial[iradial_local,iradialp_local] - 
                                              PPradial[iradial_local,iradialp_local]) +
                                             KKpolar[ipolar_local,ipolarp_local]*
                                             MNradial[iradial_local,iradialp_local]))
                                if lower_boundary_row_polar
                                    # assemble the contributions from the upper_boundary_row_polar location
                                    # take data from
                                    #     ielement_polar = nelement_polar
                                    #     ipolar_local = ngrid_polar
                                    # but ensure that data is put into correct target index (ic_global is unchanged from loop)
                                    icp_global_bc = get_global_compound_index(polar,radial,nelement_polar,ielement_radial,ipolarp_local,iradialp_local)
                                    icsc_bc = icsc_func(ipolar_local,ipolarp_local,nelement_polar,
                                           ngrid_polar,nelement_polar,
                                           iradial_local,iradialp_local,
                                           ielement_radial,
                                           ngrid_radial,nelement_radial)
                                    assemble_constructor_data!(LP2D,icsc_bc,ic_global,icp_global_bc,
                                            (MMpolar[ngrid_polar,ipolarp_local]*
                                             (KJradial[iradial_local,iradialp_local] - 
                                              PPradial[iradial_local,iradialp_local]) +
                                             KKpolar[ngrid_polar,ipolarp_local]*
                                             MNradial[iradial_local,iradialp_local]))    
                                end
                            end # Laplacian assembly
                            
                            # mass matrices for RHS of matrix equation (enforce periodicity only)
                            if upper_boundary_row_polar
                                # ensure periodicity by setting this row to 0 apart from appropriately placed 1, -1.
                                if ipolarp_local == ngrid_polar && iradial_local == iradialp_local
                                    # assign -1 to this point using loop icsc, ic, icp
                                    assign_constructor_data!(MR2D,icsc,ic_global,icp_global,-1.0)
                                    # find other end of polar domain for this ipolar_local
                                    # and set ipolarp_local = ielement_polar = 1.
                                    icp_global_bc = get_global_compound_index(polar,radial,1,ielement_radial,1,iradialp_local)
                                    icsc_bc = icsc_func(ipolar_local,1,1,
                                           ngrid_polar,nelement_polar,
                                           iradial_local,iradialp_local,
                                           ielement_radial,
                                           ngrid_radial,nelement_radial)
                                    # assign 1 to this point using ic from loop and updated icsc, icp
                                    assign_constructor_data!(MR2D,icsc_bc,ic_global,icp_global_bc,1.0)
                                else 
                                    assign_constructor_data!(MR2D,icsc,ic_global,icp_global,0.0)
                                end
                            else 
                                # carry out assembly
                                # of mass matrix data
                                assemble_constructor_data!(MR2D,icsc,ic_global,icp_global,
                                            (MMpolar[ipolar_local,ipolarp_local]*
                                             MRradial[iradial_local,iradialp_local]))
                                if lower_boundary_row_polar
                                    # assemble the contributions from the upper_boundary_row_polar location
                                    # take data from
                                    #     ielement_polar = nelement_polar
                                    #     ipolar_local = ngrid_polar
                                    # but ensure that data is put into correct target index (ic_global is unchanged from loop)
                                    icp_global_bc = get_global_compound_index(polar,radial,nelement_polar,ielement_radial,ipolarp_local,iradialp_local)
                                    icsc_bc = icsc_func(ipolar_local,ipolarp_local,nelement_polar,
                                           ngrid_polar,nelement_polar,
                                           iradial_local,iradialp_local,
                                           ielement_radial,
                                           ngrid_radial,nelement_radial)
                                    assemble_constructor_data!(MR2D,icsc_bc,ic_global,icp_global_bc,
                                            (MMpolar[ngrid_polar,ipolarp_local]*
                                             MRradial[iradial_local,iradialp_local]))    
                                end # mass matrix assembly
                            end
                        end                         
                    end                         
                end                         
            end                         
        end                         
    end                         
    LP2D_sparse = create_sparse_matrix(LP2D)
    MR2D_sparse = create_sparse_matrix(MR2D)
                            
    return LP2D_sparse, MR2D_sparse
end

"""
This function makes sparse matrices representing the 2D Laplacian operator.
The compound index runs over the radial and polar coordinates. 
We use global matrices in the compound index.
We assume that the polar coordinate has Jacobian = 1 
so that we do not need to label elemental matrices when
assembling terms from the boundary conditions.
"""
function init_laplacian2D(radial::coordinate, polar::coordinate, 
                                radial_spectral::weak_discretization_info,
                                polar_spectral::weak_discretization_info)
    nradial = radial.n
    npolar = polar.n
    
    nc_global = nradial*npolar
    ngrid_radial = radial.ngrid
    ngrid_polar = polar.ngrid
    nelement_radial = radial.nelement_local
    nelement_polar = polar.nelement_local
    
    MR2D = allocate_float(nc_global,nc_global)
    @. MR2D = 0.0
    # Laplacian matrix
    LP2D = allocate_float(nc_global,nc_global)
    @. LP2D = 0.0
    # local dummy arrays
    MMpolar = allocate_float(ngrid_polar,ngrid_polar)
    KKpolar = allocate_float(ngrid_polar,ngrid_polar)
    MNradial = allocate_float(ngrid_radial,ngrid_radial)
    MRradial = allocate_float(ngrid_radial,ngrid_radial)
    KJradial = allocate_float(ngrid_radial,ngrid_radial)
    PPradial = allocate_float(ngrid_radial,ngrid_radial)
    # loop over elements to carry out assembly
    for ielement_radial in 1:nelement_radial
        # get local radial 1D matrices
        get_QQ_local!(MNradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"N")
        get_QQ_local!(MRradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"R")
        get_QQ_local!(KJradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"J")
        get_QQ_local!(PPradial,ielement_radial,radial_spectral.lobatto,radial_spectral.radau,radial,"P")
        for ielement_polar in 1:nelement_polar
            # get local polar 1D matrices
            get_QQ_local!(MMpolar,ielement_polar,polar_spectral.lobatto,polar_spectral.radau,polar,"M")
            get_QQ_local!(KKpolar,ielement_polar,polar_spectral.lobatto,polar_spectral.radau,polar,"K")
            # loop over grid points
            for iradialp_local in 1:ngrid_radial
                for iradial_local in 1:ngrid_radial
                    for ipolarp_local in 1:ngrid_polar
                        for ipolar_local in 1:ngrid_polar
                            ic_global = get_global_compound_index(polar,radial,ielement_polar,ielement_radial,ipolar_local,iradial_local)
                            icp_global = get_global_compound_index(polar,radial,ielement_polar,ielement_radial,ipolarp_local,iradialp_local)
                            # boundary condition possibilities
                            lower_boundary_row_polar = (ielement_polar == 1 && ipolar_local == 1)
                            upper_boundary_row_polar = (ielement_polar == polar.nelement_local && ipolar_local == polar.ngrid)
                            lower_boundary_row_radial = (ielement_radial == 1 && iradial_local == 1)
                            upper_boundary_row_radial = (ielement_radial == radial.nelement_local && iradial_local == radial.ngrid)
                            
                            # Laplacian assembly
                            if upper_boundary_row_radial
                                # Dirichlet boundary condition on outer boundary
                                if iradialp_local == radial.ngrid && ipolar_local == ipolarp_local
                                    LP2D[ic_global,icp_global] = 1.0
                                else
                                    LP2D[ic_global,icp_global] = 0.0
                                end
                            elseif upper_boundary_row_polar
                                # ensure periodicity by setting this row to 0 apart from appropriately placed 1, -1.
                                if ipolarp_local == ngrid_polar && iradial_local == iradialp_local
                                    # assign -1 to this point using loop icsc, ic, icp
                                    LP2D[ic_global,icp_global] = -1.0
                                    # find other end of polar domain for this ipolar_local
                                    # and set ipolarp_local = ielement_polar = 1.
                                    icp_global_bc = get_global_compound_index(polar,radial,1,ielement_radial,1,iradialp_local)
                                    # assign 1 to this point using ic from loop and updated icsc, icp
                                    LP2D[ic_global,icp_global_bc] = 1.0
                                else 
                                    LP2D[ic_global,icp_global] = 0.0
                                end
                            else 
                                # carry out assembly
                                # of Laplacian matrix data
                                LP2D[ic_global,icp_global] +=(MMpolar[ipolar_local,ipolarp_local]*
                                                             (KJradial[iradial_local,iradialp_local] - 
                                                              PPradial[iradial_local,iradialp_local]) +
                                                             KKpolar[ipolar_local,ipolarp_local]*
                                                             MNradial[iradial_local,iradialp_local])
                                if lower_boundary_row_polar
                                    # assemble the contributions from the upper_boundary_row_polar location
                                    # take data from
                                    #     ielement_polar = nelement_polar
                                    #     ipolar_local = ngrid_polar
                                    # but ensure that data is put into correct target index (ic_global is unchanged from loop)
                                    # no elemental index on polar matrices here => assume elemental matrices independent of element index
                                    icp_global_bc = get_global_compound_index(polar,radial,nelement_polar,ielement_radial,ipolarp_local,iradialp_local)
                                    LP2D[ic_global,icp_global_bc] +=(MMpolar[ngrid_polar,ipolarp_local]*
                                                                     (KJradial[iradial_local,iradialp_local] - 
                                                                      PPradial[iradial_local,iradialp_local]) +
                                                                     KKpolar[ngrid_polar,ipolarp_local]*
                                                                     MNradial[iradial_local,iradialp_local])
                                end
                            end # Laplacian assembly
                            
                            # mass matrices for RHS of matrix equation (enforce periodicity only)
                            if upper_boundary_row_polar
                                # ensure periodicity by setting this row to 0 apart from appropriately placed 1, -1.
                                if ipolarp_local == ngrid_polar && iradial_local == iradialp_local
                                    # assign -1 to this point using loop icsc, ic, icp
                                    MR2D[ic_global,icp_global] = -1.0
                                    # find other end of polar domain for this ipolar_local
                                    # and set ipolarp_local = ielement_polar = 1.
                                    icp_global_bc = get_global_compound_index(polar,radial,1,ielement_radial,1,iradialp_local)
                                    # assign 1 to this point using ic from loop and updated icsc, icp
                                    MR2D[ic_global,icp_global_bc] = 1.0
                                else 
                                    MR2D[ic_global,icp_global] = 0.0
                                end
                            else 
                                # carry out assembly
                                # of mass matrix data
                                MR2D[ic_global,icp_global] +=(MMpolar[ipolar_local,ipolarp_local]*
                                                              MRradial[iradial_local,iradialp_local])
                                if lower_boundary_row_polar
                                    # assemble the contributions from the upper_boundary_row_polar location
                                    # take data from
                                    #     ielement_polar = nelement_polar
                                    #     ipolar_local = ngrid_polar
                                    # but ensure that data is put into correct target index (ic_global is unchanged from loop)
                                    # no elemental index on polar matrices here => assume elemental matrices independent of element index
                                    icp_global_bc = get_global_compound_index(polar,radial,nelement_polar,ielement_radial,ipolarp_local,iradialp_local)
                                    MR2D[ic_global,icp_global_bc] +=(MMpolar[ngrid_polar,ipolarp_local]*
                                                                     MRradial[iradial_local,iradialp_local])
                                end # mass matrix assembly
                            end
                        end                         
                    end                         
                end                         
            end                         
        end                         
    end                         
    LP2D_sparse = sparse(LP2D)
    MR2D_sparse = sparse(MR2D)
                            
    return LP2D_sparse, MR2D_sparse
end

"""
function for carrying out a 2D Poisson solve, solving the equation
(1/r) d( r d phi / d r) d r + (1/r^2)d^2 phi/ d polar^2 = rho
with 
 phi = an Array[polar,radial] representing phi(polar,r)
 rho = an Array[polar,radial] representing rho(polar,r)
 poisson = a struct of type poisson2D_arrays

The vec() function is used to get 1D views of the 2D arrays of shape Array{mk_float,2}(npolar,nradial)
phi has assumed zero boundary conditions at r = L, and we assume r spans (0,L].
"""
function spatial_poisson2D_solve!(phi::Array{mk_float,2},rho::Array{mk_float,2},
                                  poisson::poisson2D_arrays)
   laplacian_lu_obj = poisson.laplacian_lu_obj
   massmatrix2D = poisson.massmatrix2D
   rhspr = poisson.rhspr
   # get data into the compound index format
   # by making 1D views of the array with vec()
   rhoc = vec(rho)
   rhsc = vec(rhspr)
   phic = vec(phi)
   # form RHS of LP2D . phic = MR2D . rhoc
   mul!(rhsc,massmatrix2D,rhoc)
   # enforce the boundary conditions using the original 2D reference to the array
   # phi(r=L) = 0.0
   @. rhspr[:,end] = 0.0
   # solve system with LU decomposition
   ldiv!(phic,laplacian_lu_obj,rhsc)
   return nothing
end


end 