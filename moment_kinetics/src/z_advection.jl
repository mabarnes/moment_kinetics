"""
"""
module z_advection

export z_advection!
export update_speed_z!
export init_z_advection_implicit

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..looping
using ..timer_utils
using ..derivatives: derivative_z!
using ..array_allocation: allocate_float, allocate_int
using ..type_definitions: mk_float, mk_int
using SparseArrays: sparse, AbstractSparseArray
using SuiteSparse
using LinearAlgebra: lu, mul!, ldiv!
"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer z_advection!(
                         f_out, fvec_in, moments, fields, advect, z, vpa, vperp, r, dt, t,
                         spectral, composition, geometry, scratch_dummy) = begin

    @begin_s_r_vperp_vpa_region()

    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fvec_in.upar[:,:,is],
                               moments.ion.vth[:,:,is], moments.evolve_upar,
                               moments.evolve_ppar, fields, vpa, vperp, z, r, t, geometry, is)
        # update adv_fac
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @views @. advect[is].adv_fac[:,ivpa,ivperp,ir] = -dt*advect[is].speed[:,ivpa,ivperp,ir]
        end
    end
    #calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, fvec_in.pdf, advect,
                  scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
                  scratch_dummy.buffer_vpavperprs_3, scratch_dummy.buffer_vpavperprs_4,
                  scratch_dummy.buffer_vpavperprs_5, scratch_dummy.buffer_vpavperprs_6,
                  spectral, z)

    # advance z-advection equation
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @. @views z.scratch = scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,:,ir,is], z.scratch,
                                         advect[is], ivpa, ivperp, ir, z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, upar, vth, evolve_upar, evolve_ppar, fields, vpa, vperp,
                         z, r, t, geometry, is)
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        # bzed = B_z/B only used for z.advection.option == "default"
        bzed = geometry.bzed
        Bmag = geometry.Bmag
        bzeta = geometry.bzeta
        jacobian = geometry.jacobian
        rhostar = geometry.rhostar
        ExBfac = -0.5*rhostar
        geofac = z.scratch
        cvdriftz = geometry.cvdriftz
        gbdriftz = geometry.gbdriftz
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                # vpa bzed
                @. @views advect.speed[:,ivpa,ivperp,ir] = vpa.grid[ivpa]*bzed[:,ir]
                # ExB drift
                @. @views advect.speed[:,ivpa,ivperp,ir] += ExBfac*bzeta[:,ir]*jacobian[:,ir]/Bmag[:,ir]*fields.gEr[ivperp,:,ir,is]
                # magnetic curvature drift
                @. @views advect.speed[:,ivpa,ivperp,ir] += rhostar*(vpa.grid[ivpa]^2)*cvdriftz[:,ir]
                # magnetic grad B drift
                @. @views advect.speed[:,ivpa,ivperp,ir] += 0.5*rhostar*(vperp.grid[ivperp]^2)*gbdriftz[:,ir]
            end
            if evolve_ppar
                @loop_r_vperp_vpa ir ivperp ivpa begin
                    @. @views advect.speed[:,ivpa,ivperp,ir] *= vth[:,ir]
                end
            end
            if evolve_upar
                @loop_r_vperp_vpa ir ivperp ivpa begin
                    @. @views advect.speed[:,ivpa,ivperp,ir] += upar[:,ir]
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @views advect.speed[:,ivpa,ivperp,ir] .= z.advection.constant_speed
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @. @views advect.speed[:,ivpa,ivperp,ir] = z.advection.constant_speed*(z.grid[i]+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @. @views advect.speed[:,ivpa,ivperp,ir] = z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    return nothing
end

struct z_advection_implicit_arrays
    mass_matrix_z::AbstractSparseArray{mk_float,mk_int,2}
    stream_matrices::Array{AbstractSparseArray{mk_float,mk_int,2},2}
    streaming_lu_objs::Array{SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int},2}
end

function init_z_advection_implicit(z,z_spectral,vperp,vpa,delta_t)
    nelement_z = z.nelement_local
    ngrid_z = z.ngrid
    z_igrid_full = z.igrid_full
    # number of entries in sparse matrix
    ntot_z = (nelement_z - 1)*(ngrid_z^2 - 1) + ngrid_z^2
    # arrays for creating sparse matrices
    II = allocate_int(ntot_z)
    JJ = allocate_int(ntot_z)
    VV = allocate_float(ntot_z)
    II .= 0
    JJ .= 0
    # data structures for storing results
    streaming_sparse = Array{AbstractSparseArray{mk_float,mk_int,2},2}(undef,vpa.n,vperp.n)
    streaming_lu_objs = Array{SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int},2}(undef,vpa.n,vperp.n)
    # the calculation
    # for now use that elemental matrices
    # are independent of ielement for z
    M0 = z_spectral.lobatto.M0
    P0 = z_spectral.lobatto.P0
    zerovpa = 1.0e-8

    # mass matrix
    VV .= 0.0
    for ielement in 1:nelement_z
        for iz_local in 1:ngrid_z
            for izp_local in 1:ngrid_z
                iz_global = z_igrid_full[iz_local,ielement]
                izp_global = z_igrid_full[izp_local,ielement]
                icsc_z = 1 + ((izp_local - 1) + (iz_local - 1)*ngrid_z +
                (ielement - 1)*(ngrid_z^2 - 1))

                II[icsc_z] = iz_global
                JJ[icsc_z] = izp_global

                # assemble matrix
                VV[icsc_z] += M0[iz_local,izp_local]
            end
        end
    end
    mass_matrix_z = sparse(II,JJ,VV)

    # make streaming matrices
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            VV .= 0.0
            # make the streaming matrix for this dzdt(vpa,vperp)
            # for now, use that dzdt independent of z
            for ielement in 1:nelement_z
                for iz_local in 1:ngrid_z
                    for izp_local in 1:ngrid_z
                        iz_global = z_igrid_full[iz_local,ielement]
                        izp_global = z_igrid_full[izp_local,ielement]
                        icsc_z = 1 + ((izp_local - 1) + (iz_local - 1)*ngrid_z +
                        (ielement - 1)*(ngrid_z^2 - 1))

                        II[icsc_z] = iz_global
                        JJ[icsc_z] = izp_global

                        dzdt = vpa.grid[ivpa]
                        lower_wall = (ielement == 1 && iz_local == 1 && dzdt > zerovpa)
                        upper_wall = (ielement == nelement_z && iz_local == ngrid_z && dzdt < -zerovpa)
                        if lower_wall
                            if (iz_local == izp_local)
                                # set bc row
                                VV[icsc_z] = 1.0
                            end
                        elseif upper_wall
                            if (iz_local == izp_local)
                                # set bc row
                                VV[icsc_z] = 1.0
                            end
                        else
                            # assemble matrix
                            VV[icsc_z] += M0[iz_local,izp_local] + delta_t * dzdt * P0[iz_local,izp_local]
                        end
                    end
                end
            end
            stream_matrix = sparse(II,JJ,VV)
            stream_lu = lu(stream_matrix)
            
            streaming_sparse[ivpa,ivperp] = stream_matrix
            streaming_lu_objs[ivpa,ivperp] = stream_lu
        end
    end
    return z_advection_implicit_arrays(mass_matrix_z, streaming_sparse, streaming_lu_objs)
end

function z_advection_implicit_advance!(pdf,z,vpa,vperp,streaming_arrays::z_advection_implicit_arrays)
    # use the implicit advance matrix to get the pdf at the next time level
    begin_vperp_vpa_region()
    mass_matrix_z = streaming_arrays.mass_matrix_z
    streaming_lu_objs = streaming_arrays.streaming_lu_objs
    zerovpa = 1.0e-8
    nz = z.n
    @loop_vperp_vpa ivperp ivpa begin
        # extract the source pdf and store it
        @views @. z.scratch = pdf[ivpa,ivperp,:]
        # apply mass matrix
        mul!(z.scratch2,mass_matrix_z,z.scratch)
        
        # impose bc
        dzdt = vpa.grid[ivpa]
        # lower wall
        if dzdt > zerovpa
            z.scratch2[1] = 0.0
        end
        # upper wall
        if dzdt < -zerovpa
            z.scratch2[nz] = 0.0
        end
        
        # solve matrix system
        lu_stream = streaming_lu_objs[ivpa,ivperp]
        ldiv!(z.scratch3,lu_stream,z.scratch2)
        
        # put result in pdf array
        @views @. pdf[ivpa,ivperp,:] = z.scratch3
    end
    return nothing
end
end # z_advection
