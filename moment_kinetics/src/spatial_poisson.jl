"""
module for solving Poisson's equation on a spatial cylindrical domain
Take the approach of solving 1D radial equation in a cylinder, which is 
translationally symmetric in the axial direction, using 
Fourier series for the polar coordinate.
"""
module spatial_poisson

export init_spatial_poisson
export spatial_poisson_solve!

using ..type_definitions: mk_float, mk_int
using ..gauss_legendre: get_QQ_local!
using ..coordinates: coordinate
using ..array_allocation: allocate_float
using SparseArrays: sparse, AbstractSparseArray
using LinearAlgebra: ldiv!, mul!, LU, lu
using SuiteSparse
using ..fourier: fourier_info, fourier_forward_transform!, fourier_backward_transform! 
using ..moment_kinetics_structs: null_spatial_dimension_info

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
   rhohat = allocate_float(nrtot,npolar)
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
function to find the solution to 
nabla^2 phi = rho in cylindrical polar coordinates
 phi(r,polar) = the function solved for
 rho(r,polar) = the source evaluated at the nodal points
 poisson_arrays = precomputed arrays
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
         @views fourier_forward_transform!(rhohat[irad,:], polar_spectral.fext, rho[irad,:], polar_spectral.forward, polar_spectral.imidm, polar_spectral.imidp, polar.ngrid)
      end
   else
      @. rhohat = complex(rho,0.0)
   end
   
   for im in 1:npolar
      # solve the linear system
      # form the rhs vector
      mul!(rhs_dummy,sourcevec,rhohat[:,im])
      # set the Dirichlet BC phi = 0
      rhs_dummy[nradial] = 0.0
      lu_object_lhs = laplacian_lu_objs[im]
      ldiv!(phi_dummy, lu_object_lhs, rhs_dummy)
      rhohat[:,im] = phi_dummy
   end
   
   if npolar > 1
      # finally iFFT from hat{phi} to phi 
      for irad in 1:nradial
         @views fourier_backward_transform!(phi[irad,:], polar_spectral.fext, rhohat[irad,:], polar_spectral.backward, polar_spectral.imidm, polar_spectral.imidp, polar.ngrid)
      end
   else
      @. phi = real(rhohat)
   end
   return nothing
end

end