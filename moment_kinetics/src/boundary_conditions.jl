"""
Functions for applying boundary conditions
"""
module boundary_conditions

export enforce_boundary_conditions!
export enforce_neutral_boundary_conditions!

using SpecialFunctions: erfc

using ..calculus: reconcile_element_boundaries_MPI!
using ..coordinates: coordinate
using ..interpolation: interpolate_to_grid_1d!
using ..looping
using ..timer_utils
using ..moment_kinetics_structs: scratch_pdf, em_fields_struct
using ..type_definitions: mk_float, mk_int
using ..velocity_moments: integrate_over_vspace, integrate_over_neutral_vspace,
                          integrate_over_positive_vz, integrate_over_negative_vz

"""
enforce boundary conditions in vpa and z on the evolved pdf;
also enforce boundary conditions in z on all separately evolved velocity space moments of the pdf
"""
@timeit global_timer enforce_boundary_conditions!(
                         f, f_r_bc, density, upar, ppar, phi, moments, vpa_bc, z_bc, r_bc,
                         vpa, vperp, z, r, vpa_spectral, vperp_spectral, vpa_adv,
                         vperp_adv, z_adv, r_adv, composition, scratch_dummy, r_diffusion,
                         vpa_diffusion, vperp_diffusion) = begin
    if vpa.n > 1
        @begin_s_r_z_vperp_region()
        @loop_s_r_z_vperp is ir iz ivperp begin
            # enforce the vpa BC
            # use that adv.speed independent of vpa
            @views enforce_v_boundary_condition_local!(f[:,ivperp,iz,ir,is], vpa_bc,
                             vpa_adv[is].speed[:,ivperp,iz,ir], vpa_diffusion,
                             vpa, vpa_spectral)
        end
    end
    if vperp.n > 1
        @begin_s_r_z_vpa_region()
        enforce_vperp_boundary_condition!(f, vperp.bc, vperp, vperp_spectral,
                             vperp_adv, vperp_diffusion)
    end
    if z.n > 1
        @begin_s_r_vperp_vpa_region()
        # enforce the z BC on the evolved velocity space moments of the pdf
        enforce_z_boundary_condition_moments!(density, moments, z_bc)
        enforce_z_boundary_condition!(f, density, upar, ppar, phi, moments, z_bc, z_adv, z,
                                      vperp, vpa, composition,
                                      scratch_dummy.buffer_vpavperprs_1,
                                      scratch_dummy.buffer_vpavperprs_2,
                                      scratch_dummy.buffer_vpavperprs_3,
                                      scratch_dummy.buffer_vpavperprs_4)

    end
    if r.n > 1
        @begin_s_z_vperp_vpa_region()
        enforce_r_boundary_condition!(f, f_r_bc, r_bc, r_adv, vpa, vperp, z, r,
                                      composition, scratch_dummy.buffer_vpavperpzs_1,
                                      scratch_dummy.buffer_vpavperpzs_2,
                                      scratch_dummy.buffer_vpavperpzs_3,
                                      scratch_dummy.buffer_vpavperpzs_4, r_diffusion)
    end
end
function enforce_boundary_conditions!(fvec_out::scratch_pdf, moments, fields::em_fields_struct, f_r_bc, vpa_bc,
        z_bc, r_bc, vpa, vperp, z, r, vpa_spectral, vperp_spectral, vpa_adv, vperp_adv, z_adv, r_adv, composition, scratch_dummy,
        r_diffusion, vpa_diffusion, vperp_diffusion)
    enforce_boundary_conditions!(fvec_out.pdf, f_r_bc, fvec_out.density, fvec_out.upar,
        fvec_out.ppar, fields.phi, moments, vpa_bc, z_bc, r_bc, vpa, vperp, z, r,
        vpa_spectral, vperp_spectral, vpa_adv, vperp_adv, z_adv,
        r_adv, composition, scratch_dummy, r_diffusion, vpa_diffusion, vperp_diffusion)
end

"""
enforce boundary conditions on f in r
"""
function enforce_r_boundary_condition!(f::AbstractArray{mk_float,5}, f_r_bc, bc::String,
        adv, vpa, vperp, z, r, composition, end1::AbstractArray{mk_float,4},
        end2::AbstractArray{mk_float,4}, buffer1::AbstractArray{mk_float,4},
        buffer2::AbstractArray{mk_float,4}, r_diffusion::Bool)

    nr = r.n

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            end1[ivpa,ivperp,iz,is] = f[ivpa,ivperp,iz,1,is]
            end2[ivpa,ivperp,iz,is] = f[ivpa,ivperp,iz,nr,is]
        end
        reconcile_element_boundaries_MPI!(f, end1, end2, buffer1, buffer2, r)
    end

    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    # enforce the condition if r is local
    if bc == "periodic" && r.nelement_global == r.nelement_local
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            f[ivpa,ivperp,iz,1,is] = 0.5*(f[ivpa,ivperp,iz,nr,is]+f[ivpa,ivperp,iz,1,is])
            f[ivpa,ivperp,iz,nr,is] = f[ivpa,ivperp,iz,1,is]
        end
    end
    if bc == "Dirichlet"
        zero = 1.0e-10
        # use the old distribution to force the new distribution to have
        # consistant-in-time values at the boundary
        # with bc = "Dirichlet" and r_diffusion = false
        # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
        # with bc = "Dirichlet" and r_diffusion = true
        # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            ir = 1 # r = -L/2 -- check that the point is on lowest rank
            if r.irank == 0 && (r_diffusion || adv[is].speed[ir,ivpa,ivperp,iz] > zero)
                f[ivpa,ivperp,iz,ir,is] = f_r_bc[ivpa,ivperp,iz,1,is]
            end
            ir = r.n # r = L/2 -- check that the point is on highest rank
            if r.irank == r.nrank - 1 && (r_diffusion || adv[is].speed[ir,ivpa,ivperp,iz] < -zero)
                f[ivpa,ivperp,iz,ir,is] = f_r_bc[ivpa,ivperp,iz,end,is]
            end
        end
    end
end

"""
enforce boundary conditions on ion particle f in z
"""
function enforce_z_boundary_condition!(pdf, density, upar, ppar, phi, moments, bc::String, adv,
                                       z, vperp, vpa, composition, end1::AbstractArray{mk_float,4},
                                       end2::AbstractArray{mk_float,4}, buffer1::AbstractArray{mk_float,4},
                                       buffer2::AbstractArray{mk_float,4})
    # this block ensures periodic BC can be supported with distributed memory MPI
    if z.nelement_global > z.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        nz = z.n
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            end1[ivpa,ivperp,ir,is] = pdf[ivpa,ivperp,1,ir,is]
            end2[ivpa,ivperp,ir,is] = pdf[ivpa,ivperp,nz,ir,is]
        end
        # check on periodic bc happens inside this call below
        reconcile_element_boundaries_MPI!(pdf, end1, end2, buffer1, buffer2, z)
    end
    # define a zero that accounts for finite precision
    zero = 1.0e-14
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if bc == "constant"
        @begin_s_r_vperp_vpa_region()
        density_offset = 1.0
        vwidth = 1.0
        if z.irank == 0
            @loop_s is begin
                speed = adv[is].speed
                @loop_r ir begin
                    prefactor = density_offset
                    if moments.evolve_density
                        prefactor /= density[1,ir,is]
                    end
                    if moments.evolve_ppar
                        prefactor *= moments.ion.vth[1,ir,is]
                    end
                    @loop_vperp_vpa ivperp ivpa begin
                        if speed[1,ivpa,ivperp,ir] > 0.0
                            pdf[ivpa,ivperp,1,ir,is] = prefactor * exp(-(speed[1,ivpa,ivperp,ir]^2 + vperp.grid[ivperp]^2)/vwidth^2)
                        end
                    end
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_s is begin
                speed = adv[is].speed
                @loop_r ir begin
                    prefactor = density_offset
                    if moments.evolve_density
                        prefactor /= density[end,ir,is]
                    end
                    if moments.evolve_ppar
                        prefactor *= moments.ion.vth[end,ir,is]
                    end
                    @loop_vperp_vpa ivperp ivpa begin
                        if speed[end,ivpa,ivperp,ir] > 0.0
                            pdf[ivpa,ivperp,end,ir,is] = prefactor * exp(-(speed[end,ivpa,ivperp,ir]^2 + vperp.grid[ivperp]^2)/vwidth^2)
                        end
                    end
                end
            end
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif bc == "periodic" && z.nelement_global == z.nelement_local
        @begin_s_r_vperp_vpa_region()
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            pdf[ivpa,ivperp,1,ir,is] = 0.5*(pdf[ivpa,ivperp,z.n,ir,is]+pdf[ivpa,ivperp,1,ir,is])
            pdf[ivpa,ivperp,z.n,ir,is] = pdf[ivpa,ivperp,1,ir,is]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif bc == "wall"
        # Need integrals over vpa at wall boundaries in z, so cannot parallelize over z
        # or vpa.
        @begin_s_r_region()
        @loop_s is begin
            # zero incoming BC for ions, as they recombine at the wall
            if moments.evolve_upar
                @loop_r ir begin
                    @views enforce_zero_incoming_bc!(
                        pdf[:,:,:,ir,is], z, vpa, density[:,ir,is], upar[:,ir,is],
                        ppar[:,ir,is], moments.evolve_upar, moments.evolve_ppar, zero,
                        phi[:,ir])
                end
            else
                @loop_r ir begin
                    @views enforce_zero_incoming_bc!(pdf[:,:,:,ir,is],
                                                     adv[is].speed[:,:,:,ir], z, zero,
                                                     phi[:,ir],
                                                     z.boundary_parameters.epsz)
                end
            end
        end
    end
end

"""
enforce boundary conditions on neutral particle distribution function
"""
@timeit global_timer enforce_neutral_boundary_conditions!(
                         f_neutral, f_ion, boundary_distributions, density_neutral,
                         uz_neutral, pz_neutral, moments, density_ion, upar_ion, Er,
                         vzeta_spectral, vr_spectral, vz_spectral, r_adv, z_adv,
                         vzeta_adv, vr_adv, vz_adv, r, z, vzeta, vr, vz, composition,
                         geometry, scratch_dummy, r_diffusion, vz_diffusion) = begin

    # without acceleration of neutrals bc on vz vr vzeta should not be required as no
    # advection or diffusion in these coordinates

    if vzeta_adv !== nothing && vzeta.n_global > 1 && vzeta.bc != "none"
        @begin_sn_r_z_vr_vz_region()
        @loop_sn_r_z_vr_vz isn ir iz ivr ivz begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[ivz,ivr,:,iz,ir,isn],
                                                       vzeta.bc,
                                                       vzeta_adv[isn].speed[ivz,ivr,:,iz,ir],
                                                       false, vzeta, vzeta_spectral)
        end
    end
    if vr_adv !== nothing && vr.n_global > 1 && vr.bc != "none"
        @begin_sn_r_z_vzeta_vz_region()
        @loop_sn_r_z_vzeta_vz isn ir iz ivzeta ivz begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[ivz,:,ivzeta,iz,ir,isn],
                                                       vr.bc,
                                                       vr_adv[isn].speed[ivz,:,ivzeta,iz,ir],
                                                       false, vr, vr_spectral)
        end
    end
    if vz_adv !== nothing && vz.n_global > 1 && vz.bc != "none"
        @begin_sn_r_z_vzeta_vr_region()
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[:,ivr,ivzeta,iz,ir,isn],
                                                       vz.bc,
                                                       vz_adv[isn].speed[:,ivr,ivzeta,iz,ir],
                                                       vz_diffusion, vz, vz_spectral)
        end
    end
    # f_initial contains the initial condition for enforcing a fixed-boundary-value condition
    if z.n > 1
        @begin_sn_r_vzeta_vr_vz_region()
        enforce_neutral_z_boundary_condition!(f_neutral, density_neutral, uz_neutral,
            pz_neutral, moments, density_ion, upar_ion, Er, boundary_distributions,
            z_adv, z, vzeta, vr, vz, composition, geometry,
            scratch_dummy.buffer_vzvrvzetarsn_1, scratch_dummy.buffer_vzvrvzetarsn_2,
            scratch_dummy.buffer_vzvrvzetarsn_3, scratch_dummy.buffer_vzvrvzetarsn_4)
    end
    if r.n > 1
        @begin_sn_z_vzeta_vr_vz_region()
        enforce_neutral_r_boundary_condition!(f_neutral, boundary_distributions.pdf_rboundary_neutral,
                                    r_adv, vz, vr, vzeta, z, r, composition,
                                    scratch_dummy.buffer_vzvrvzetazsn_1, scratch_dummy.buffer_vzvrvzetazsn_2,
                                    scratch_dummy.buffer_vzvrvzetazsn_3, scratch_dummy.buffer_vzvrvzetazsn_4,
                                    r_diffusion)
    end
end

function enforce_neutral_r_boundary_condition!(f::AbstractArray{mk_float,6},
        f_r_bc::AbstractArray{mk_float,6}, adv, vz, vr, vzeta, z, r, composition,
        end1::AbstractArray{mk_float,5}, end2::AbstractArray{mk_float,5},
        buffer1::AbstractArray{mk_float,5}, buffer2::AbstractArray{mk_float,5},
        r_diffusion) #f_initial,

    bc = r.bc
    nr = r.n

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            end1[ivz,ivr,ivzeta,iz,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
            end2[ivz,ivr,ivzeta,iz,isn] = f[ivz,ivr,ivzeta,iz,nr,isn]
        end
        reconcile_element_boundaries_MPI!(f, end1, end2, buffer1, buffer2, r)
    end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    # local case only when no communication required
    if bc == "periodic" && r.nelement_global == r.nelement_local
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            f[ivz,ivr,ivzeta,iz,1,isn] = 0.5*(f[ivz,ivr,ivzeta,iz,1,isn]+f[ivz,ivr,ivzeta,iz,nr,isn])
            f[ivz,ivr,ivzeta,iz,nr,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
        end
    end
    # Dirichlet boundary condition for external endpoints
    if bc == "Dirichlet"
        zero = 1.0e-10
        # use the old distribution to force the new distribution to have
        # consistant-in-time values at the boundary
        # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            ir = 1 # r = -L/2
            # incoming particles and on lowest rank
            if r.irank == 0 && (r_diffusion || adv[isn].speed[ir,ivz,ivr,ivzeta,iz] > zero)
                f[ivz,ivr,ivzeta,iz,ir,isn] = f_r_bc[ivz,ivr,ivzeta,iz,1,isn]
            end
            ir = nr # r = L/2
            # incoming particles and on highest rank
            if r.irank == r.nrank - 1 && (r_diffusion || adv[isn].speed[ir,ivz,ivr,ivzeta,iz] < -zero)
                f[ivz,ivr,ivzeta,iz,ir,isn] = f_r_bc[ivz,ivr,ivzeta,iz,end,isn]
            end
        end
    end
end

"""
enforce boundary conditions on neutral particle f in z
"""
function enforce_neutral_z_boundary_condition!(pdf, density, uz, pz, moments, density_ion,
                                               upar_ion, Er, boundary_distributions, adv,
                                               z, vzeta, vr, vz, composition, geometry,
                                               end1::AbstractArray{mk_float,5}, end2::AbstractArray{mk_float,5},
                                               buffer1::AbstractArray{mk_float,5}, buffer2::AbstractArray{mk_float,5})


    if z.nelement_global > z.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        nz = z.n
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            end1[ivz,ivr,ivzeta,ir,isn] = pdf[ivz,ivr,ivzeta,1,ir,isn]
            end2[ivz,ivr,ivzeta,ir,isn] = pdf[ivz,ivr,ivzeta,nz,ir,isn]
        end
        # check on periodic bc occurs within this call below
        reconcile_element_boundaries_MPI!(pdf, end1, end2, buffer1, buffer2, z)
    end

    zero = 1.0e-14
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if z.bc == "constant"
        @begin_sn_r_vzeta_vr_vz_region()
        density_offset = 1.0
        vwidth = 1.0
        if z.irank == 0
            @loop_sn isn begin
                speed = adv[isn].speed
                @loop_r ir begin
                    prefactor = density_offset
                    if moments.evolve_density
                        prefactor /= density[1,ir,isn]
                    end
                    if moments.evolve_ppar
                        prefactor *= moments.neutral.vth[1,ir,isn]
                    end
                    @loop_vzeta_vr_vz ivzeta ivr ivz begin
                        if speed[1,ivz,ivr,ivzeta,ir] > 0.0
                            pdf[ivz,ivr,ivzeta,1,ir,isn] = prefactor *
                                exp(-(speed[1,ivz,ivr,ivzeta,ir]^2 + vr.grid[ivr] + vz.grid[ivz])/vwidth^2)
                        end
                    end
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_sn isn begin
                speed = adv[isn].speed
                @loop_r ir begin
                    prefactor = density_offset
                    if moments.evolve_density
                        prefactor /= density[end,ir,isn]
                    end
                    if moments.evolve_ppar
                        prefactor *= moments.neutral.vth[end,ir,isn]
                    end
                    @loop_vzeta_vr_vz ivzeta ivr ivz begin
                        if speed[end,ivz,ivr,ivzeta,ir] > 0.0
                            pdf[ivz,ivr,ivzeta,end,ir,isn] = prefactor *
                                exp(-(speed[end,ivz,ivr,ivzeta,ir][ivzeta]^2 + vr.grid[ivr] + vz.grid[ivz])/vwidth^2)
                        end
                    end
                end
            end
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif z.bc == "periodic" && z.nelement_global == z.nelement_local
        @begin_sn_r_vzeta_vr_vz_region()
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            pdf[ivz,ivr,ivzeta,1,ir,isn] = 0.5*(pdf[ivz,ivr,ivzeta,1,ir,isn] +
                                                pdf[ivz,ivr,ivzeta,end,ir,isn])
            pdf[ivz,ivr,ivzeta,end,ir,isn] = pdf[ivz,ivr,ivzeta,1,ir,isn]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif z.bc == "wall"
        # Need integrals over vpa at wall boundaries in z, so cannot parallelize over z
        # or vpa.
        @begin_sn_r_region()
        @loop_sn isn begin
            # BC for neutrals
            @loop_r ir begin
                # define vtfac to avoid repeated computation below
                vtfac = sqrt(composition.T_wall * composition.mn_over_mi)
                # Assume for now that the ion species index corresponding to this neutral
                # species is the same as the neutral species index.
                # Note, have already calculated moments of ion distribution function(s),
                # so can use the moments here to get the flux
                if z.irank == 0
                    ion_flux_0 = -density_ion[1,ir,isn] * (upar_ion[1,ir,isn]*geometry.bzed[1,ir] - 0.5*geometry.rhostar*Er[1,ir])
                else
                    ion_flux_0 = NaN
                end
                if z.irank == z.nrank - 1
                    ion_flux_L = density_ion[end,ir,isn] * (upar_ion[end,ir,isn]*geometry.bzed[end,ir] - 0.5*geometry.rhostar*Er[end,ir])
                else
                    ion_flux_L = NaN
                end
                # enforce boundary condition on the neutral pdf that all ions and neutrals
                # that leave the domain re-enter as neutrals
                @views enforce_neutral_wall_bc!(
                    pdf[:,:,:,:,ir,isn], z, vzeta, vr, vz, pz[:,ir,isn], uz[:,ir,isn],
                    density[:,ir,isn], ion_flux_0, ion_flux_L, boundary_distributions,
                    vtfac, composition.recycling_fraction, moments.evolve_ppar,
                    moments.evolve_upar, moments.evolve_density, zero, buffer1)
            end
        end
    end
end

"""
enforce a zero incoming BC in z for given species pdf at each radial location
"""
function enforce_zero_incoming_bc!(pdf, speed, z, zero, phi, epsz)
    nvpa = size(pdf,1)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa
    #
    # epsz is the ratio |z - z_wall|/|delta z|, with delta z the grid spacing at the wall
    # for epsz < 1, the cut off below would be imposed for particles travelling
    # out to a distance z = epsz * delta z from the wall before returning
    if z.irank == 0
        deltaphi = phi[2] - phi[1]
        vcut = deltaphi > 0 ? sqrt(deltaphi)*(epsz^0.25) : 0.0
        @loop_vperp_vpa ivperp ivpa begin
            # for left boundary in zed (z = -Lz/2), want
            # f(z=-Lz/2, v_parallel > 0) = 0
            if speed[1,ivpa,ivperp] > zero - vcut
                pdf[ivpa,ivperp,1] = 0.0
            end
        end
    end
    if z.irank == z.nrank - 1
        deltaphi = phi[end-1] - phi[end]
        vcut = deltaphi > 0 ? sqrt(deltaphi)*(epsz^0.25) : 0.0
        @loop_vperp_vpa ivperp ivpa begin
            # for right boundary in zed (z = Lz/2), want
            # f(z=Lz/2, v_parallel < 0) = 0
            if speed[end,ivpa,ivperp] < -zero + vcut
                pdf[ivpa,ivperp,end] = 0.0
            end
        end
    end
end
function get_ion_z_boundary_cutoff_indices(density, upar, ppar, evolve_upar, evolve_ppar,
                                           z, vpa, zero, phi)
    epsz = z.boundary_parameters.epsz
    if z.irank == 0
        deltaphi = phi[2] - phi[1]
        vcut = deltaphi > 0 ? sqrt(deltaphi)*(epsz^0.25) : 0.0
        vth = sqrt(2.0*(ppar[1]/density[1]))
        @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, vth,
                                         upar[1], evolve_ppar, evolve_upar)
        last_negative_vpa_ind = searchsortedlast(vpa.scratch, min(-zero, -vcut))
    else
        last_negative_vpa_ind = nothing
    end
    if z.irank == z.nrank - 1
        deltaphi = phi[end-1] - phi[end]
        vcut = deltaphi > 0 ? sqrt(deltaphi)*(epsz^0.25) : 0.0
        vth = sqrt(2.0*(ppar[end]/density[end]))
        @. vpa.scratch2 = vpagrid_to_dzdt(vpa.grid, vth,
                                          upar[end], evolve_ppar, evolve_upar)
        first_positive_vpa_ind = searchsortedfirst(vpa.scratch2, max(zero, vcut))
    else
        first_positive_vpa_ind = nothing
    end
    return last_negative_vpa_ind, first_positive_vpa_ind
end
function enforce_zero_incoming_bc!(pdf, z::coordinate, vpa::coordinate, density, upar,
                                   ppar, evolve_upar, evolve_ppar, zero, phi)
    if z.irank != 0 && z.irank != z.nrank - 1
        # No z-boundary in this block
        return nothing
    end
    nvpa, nvperp, nz = size(pdf)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    # absolute velocity at left boundary
    last_negative_vpa_ind, first_positive_vpa_ind =
        get_ion_z_boundary_cutoff_indices(density, upar, ppar, evolve_upar, evolve_ppar,
                                          z, vpa, zero, phi)
    if z.irank == 0
        pdf[last_negative_vpa_ind+1:end, :, 1] .= 0.0
    end
    # absolute velocity at right boundary
    if z.irank == z.nrank - 1
        pdf[1:first_positive_vpa_ind-1, :, end] .= 0.0
    end

    # Special constraint-forcing code that tries to keep the modifications smooth at
    # v_parallel=0.
    if z.irank == 0 && z.irank == z.nrank - 1
        # Both z-boundaries in this block
        z_range = (1,nz)
    elseif z.irank == 0
        z_range = (1,)
    elseif z.irank == z.nrank - 1
        z_range = (nz,)
    else
        error("No boundary in this block, should have returned already")
    end
    for iz ∈ z_range
        # moment-kinetic approach only implemented for 1V case so far
        @boundscheck size(pdf,2) == 1

        f = @view pdf[:,1,iz]
        if evolve_ppar && evolve_upar
            I0 = integrate_over_vspace(f, vpa.wgts)
            I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
            I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)

            # Store v_parallel with upar shift removed in vpa.scratch
            vth = sqrt(2.0*ppar[iz]/density[iz])
            @. vpa.scratch = vpa.grid + upar[iz]/vth
            # Introduce factors to ensure corrections go smoothly to zero near
            # v_parallel=0, and that there are no large corrections aw large w_parallel as
            # those can have a strong effect on the parallel heat flux and make
            # timestepping unstable when the cut-off point jumps from one grid point to
            # another.
            @. vpa.scratch2 = f * abs(vpa.scratch) / (1.0 + abs(vpa.scratch)) / (1.0 + (4.0 * vpa.scratch / vpa.L)^4)
            J1 = integrate_over_vspace(vpa.scratch2, vpa.grid, vpa.wgts)
            J2 = integrate_over_vspace(vpa.scratch2, vpa.grid, 2, vpa.wgts)
            J3 = integrate_over_vspace(vpa.scratch2, vpa.grid, 3, vpa.wgts)
            J4 = integrate_over_vspace(vpa.scratch2, vpa.grid, 4, vpa.wgts)

            A = (J3^2 - J2*J4 + 0.5*(J2^2 - J1*J3)) /
                (I0*(J3^2 - J2*J4) + I1*(J1*J4 - J2*J3) + I2*(J2^2 - J1*J3))
            B = (0.5*J3 + A*(I1*J4 - I2*J3)) / (J3^2 - J2*J4)
            C = (0.5 - A*I2 -B*J3) / J4

            @. f = A*f + B*vpa.grid*vpa.scratch2 + C*vpa.grid*vpa.grid*vpa.scratch2
        elseif evolve_upar
            I0 = integrate_over_vspace(f, vpa.wgts)
            I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)

            # Store v_parallel with upar shift removed in vpa.scratch
            @. vpa.scratch = vpa.grid + upar[iz]
            # Introduce factors to ensure corrections go smoothly to zero near
            # v_parallel=0, and that there are no large corrections aw large w_parallel as
            # those can have a strong effect on the parallel heat flux and make
            # timestepping unstable when the cut-off point jumps from one grid point to
            # another.
            @. vpa.scratch2 = f * abs(vpa.scratch) / (1.0 + abs(vpa.scratch)) / (1.0 + (4.0 * vpa.scratch / vpa.L)^4)
            J1 = integrate_over_vspace(vpa.scratch2, vpa.grid, vpa.wgts)
            J2 = integrate_over_vspace(vpa.scratch2, vpa.grid, 2, vpa.wgts)

            A = 1.0 / (I0 - I1*J1/J2)
            B = -A*I1/J2

            @. f = A*f + B*vpa.grid*vpa.scratch2
        elseif evolve_density
            I0 = integrate_over_vspace(f, vpa.wgts)
            @. f = f / I0
        end
    end
end

"""
Set up an initial condition that tries to be smoothly compatible with the sheath
boundary condition for ions, by setting f(±(v_parallel-u0)<0) where u0=0 at the sheath
boundaries and for z<0 increases linearly to u0=vpa.L at z=0, while for z>0 increases
from u0=-vpa.L at z=0 to zero at the z=z.L/2 sheath.

To be applied to 'full-f' distribution function on v_parallel grid (not w_parallel
grid).
"""
function enforce_initial_tapered_zero_incoming!(pdf, z::coordinate, vpa::coordinate)
    nvpa = size(pdf,1)
    zero = 1.0e-14
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    for iz ∈ 1:z.n
        u0 = (2.0*z.grid[iz]/z.L - sign(z.grid[iz])) * vpa.L / 2.0
        if z.grid[iz] < -zero
            for ivpa ∈ 1:nvpa
                if vpa.grid[ivpa] > u0 + zero
                    pdf[ivpa,iz] = 0.0
                end
            end
        elseif z.grid[iz] > zero
            for ivpa ∈ 1:nvpa
                if vpa.grid[ivpa] < u0 - zero
                    pdf[ivpa,iz] = 0.0
                end
            end
        end
    end
end

"""
enforce the wall boundary condition on neutrals;
i.e., the incoming flux of neutrals equals the sum of the ion/neutral outgoing fluxes
"""
function enforce_neutral_wall_bc!(pdf, z, vzeta, vr, vz, pz, uz, density, wall_flux_0,
                                  wall_flux_L, boundary_distributions, vtfac,
                                  recycling_fraction, evolve_ppar, evolve_upar,
                                  evolve_density, zero, buffer_vzvrvzetarsn)

    # Reduce the ion flux by `recycling_fraction` to account for ions absorbed by the
    # wall.
    wall_flux_0 *= recycling_fraction
    wall_flux_L *= recycling_fraction
    pdf_buffer = @view buffer_vzvrvzetarsn[:,:,:,1,1]

    if !evolve_density && !evolve_upar
        knudsen_cosine = boundary_distributions.knudsen

        if z.irank == 0
            ## treat z = -Lz/2 boundary ##

            # add the neutral species's contribution to the combined ion/neutral particle
            # flux out of the domain at z=-Lz/2
            @views @. pdf_buffer = abs(vz.grid) * pdf[:,:,:,1]
            wall_flux_0 += integrate_over_negative_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # for left boundary in zed (z = -Lz/2), want
            # f_n(z=-Lz/2, v_parallel > 0) = Γ_0 * f_KW(v_parallel)
            @loop_vz ivz begin
                if vz.grid[ivz] >= -zero
                    @views @. pdf[ivz,:,:,1] = wall_flux_0 * knudsen_cosine[ivz,:,:]
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##

            # add the neutral species's contribution to the combined ion/neutral particle
            # flux out of the domain at z=-Lz/2
            @views @. pdf_buffer = abs(vz.grid) * pdf[:,:,:,end]
            wall_flux_L += integrate_over_positive_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # for right boundary in zed (z = Lz/2), want
            # f_n(z=Lz/2, v_parallel < 0) = Γ_Lz * f_KW(v_parallel)
            @loop_vz ivz begin
                if vz.grid[ivz] <= zero
                    @views @. pdf[ivz,:,:,end] = wall_flux_L * knudsen_cosine[ivz,:,:]
                end
            end
        end
    elseif !evolve_upar
        # Evolving density case
        knudsen_cosine = boundary_distributions.knudsen

        if z.irank == 0
            ## treat z = -Lz/2 boundary ##

            # Note the numerical integrol of knudsen_cosine was forced to be 1 (to machine
            # precision) when it was initialised.
            @views pdf_integral_0 = integrate_over_negative_vz(pdf[:,:,:,1], vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,1]
            pdf_integral_1 = integrate_over_negative_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_0 = integrate_over_positive_vz(knudsen_cosine, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_1 = 1.0 # This is enforced in initialization

            # Calculate normalisation factors N_in for the incoming and N_out for the
            # Knudsen parts of the distirbution so that ∫dvpa F = 1 and ∫dvpa vpa F = uz
            # Note wall_flux_0 is the ion flux into the wall (reduced by the recycling
            # fraction), and the neutral flux should be out of the wall (i.e. uz>0) so
            # n*uz = |n*uz| = wall_flux_0
            # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
            #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = uz
            uz = wall_flux_0 / density[1]
            N_in = (1 - uz * knudsen_integral_0 / knudsen_integral_1) /
                   (pdf_integral_0
                    - pdf_integral_1 / knudsen_integral_1 * knudsen_integral_0)
            N_out = (uz - N_in * pdf_integral_1) / knudsen_integral_1

            @loop_vz ivz begin
                if vz.grid[ivz] >= -zero
                    @views @. pdf[ivz,:,:,1] = N_out * knudsen_cosine[ivz,:,:]
                else
                    @views @. pdf[ivz,:,:,1] *= N_in
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##

            # Note the numerical integrol of knudsen_cosine was forced to be 1 (to machine
            # precision) when it was initialised.
            @views pdf_integral_0 = integrate_over_positive_vz(pdf[:,:,:,end], vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,end]
            @views pdf_integral_1 = integrate_over_positive_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_0 = integrate_over_negative_vz(knudsen_cosine, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_1 = -1.0 # This is enforced in initialization

            # Calculate normalisation factors N_in for the incoming and N_out for the
            # Knudsen parts of the distirbution so that ∫dvpa F = 1 and ∫dvpa vpa F = uz
            # Note wall_flux_L is the ion flux into the wall (reduced by the recycling
            # fraction), and the neutral flux should be out of the wall (i.e. uz<0) so
            # -n*uz = |n*uz| = wall_flux_L
            # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
            #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = uz
            uz = -wall_flux_L / density[end]
            N_in = (1 - uz * knudsen_integral_0 / knudsen_integral_1) /
                   (pdf_integral_0
                    - pdf_integral_1 / knudsen_integral_1 * knudsen_integral_0)
            N_out = (uz - N_in * pdf_integral_1) / knudsen_integral_1

            @loop_vz ivz begin
                if vz.grid[ivz] <= zero
                    @views @. pdf[ivz,:,:,end] = N_out * knudsen_cosine[ivz,:,:]
                else
                    @views @. pdf[ivz,:,:,end] *= N_in
                end
            end
        end
    else
        if z.irank == 0
            ## treat z = -Lz/2 boundary ##
            # populate vz.scratch2 array with dz/dt values at z = -Lz/2
            if evolve_ppar
                vth = sqrt(2.0*pz[1]/density[1])
            else
                vth = nothing
            end
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[1], evolve_ppar, evolve_upar)

            # First apply boundary condition that total neutral outflux is equal to ion
            # influx to uz
            uz[1] = wall_flux_0 / density[1]
            #would setting density work better??
            #density[1] = - wall_flux_0 / uz[1]

            # Create normalised Knudsen cosine distribution, to use for positive v_parallel
            # at z = -Lz/2
            # Note this only makes sense for the 1V case with vr.n=vzeta.n=1
            @. vz.scratch = (3.0*pi/vtfac^3)*abs(vz.scratch2)*erfc(abs(vz.scratch2)/vtfac)

            # The v_parallel>0 part of the pdf is replaced by the Knudsen cosine
            # distribution. To ensure the constraints ∫dwpa wpa^m F = 0 are satisfied when
            # necessary, calculate a normalisation factor for the Knudsen distribution (in
            # vz.scratch) and correction terms for the incoming pdf similar to
            # enforce_moment_constraints!().
            #
            # Note that it seems to be important that this boundary condition not be
            # modified by the moment constraints, as that could cause numerical instability.
            # By ensuring that the constraints are satisfied already here,
            # enforce_moment_constraints!() will not change the pdf at the boundary. For
            # ions this is not an issue, because points set to 0 by the bc are not modified
            # from 0 by enforce_moment_constraints!().
            knudsen_integral_0 = integrate_over_positive_vz(vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @. vz.scratch4 = vz.grid * vz.scratch
            knudsen_integral_1 = integrate_over_positive_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            @views pdf_integral_0 = integrate_over_negative_vz(pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,1]
            pdf_integral_1 = integrate_over_negative_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            if !evolve_ppar
                # Calculate normalisation factors N_in for the incoming and N_out for the
                # Knudsen parts of the distirbution so that ∫dwpa F = 1 and ∫dwpa wpa F = 0
                # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = 0
                N_in = 1.0 / (pdf_integral_0 - pdf_integral_1/knudsen_integral_1*knudsen_integral_0)
                N_out = -N_in * pdf_integral_1 / knudsen_integral_1

                zero_vz_ind = 0
                for ivz ∈ 1:vz.n
                    if vz.scratch2[ivz] <= -zero
                        pdf[ivz,:,:,1] .= N_in*pdf[ivz,:,:,1]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_z = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,1] = 0.5*(N_in*pdf[ivz,:,:,1] + N_out*vz.scratch[ivz])
                        else
                            pdf[ivz,:,:,1] .= N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ zero_vz_ind+1:vz.n
                    pdf[ivz,:,:,1] .= N_out*vz.scratch[ivz]
                end
            else
                @. vz.scratch4 = vz.grid * vz.grid * vz.scratch
                knudsen_integral_2 = integrate_over_positive_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views @. pdf_buffer = vz.grid * vz.grid * pdf[:,:,:,1]
                pdf_integral_2 = integrate_over_negative_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                pdf_buffer .*= vz.grid
                @views pdf_integral_3 = integrate_over_negative_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                # Calculate normalisation factor N_out for the Knudsen part of the
                # distirbution and normalisation factor N_in and correction term C*wpa*F_in
                # for the incoming distribution so that ∫dwpa F = 1, ∫dwpa wpa F = 0, and
                # ∫dwpa wpa^2 F = 1/2
                # ⇒ N_in*pdf_integral_0 + C*pdf_integral_1 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + C*pdf_integral_2 + N_out*knudsen_integral_1 = 0
                #   N_in*pdf_integral_2 + C*pdf_integral_3 + N_out*knudsen_integral_2 = 1/2
                N_in = (0.5*knudsen_integral_0*pdf_integral_2 +
                        knudsen_integral_1*(pdf_integral_3 - 0.5*pdf_integral_1) -
                        knudsen_integral_2*pdf_integral_2) /
                       (knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) +
                        knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) +
                        knudsen_integral_2*(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2))
                N_out = -(N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2^2) + 0.5*pdf_integral_2) /
                         (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                C = (0.5 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)/pdf_integral_3

                zero_vz_ind = 0
                for ivz ∈ 1:vz.n
                    if vz.scratch2[ivz] <= -zero
                        @views @. pdf[ivz,:,:,1] = N_in*pdf[ivz,:,:,1] + C*vz.grid[ivz]*pdf[ivz,:,:,1]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,1] = 0.5*(N_in*pdf[ivz,:,:,1] +
                                                            C*vz.grid[ivz]*pdf[ivz,:,:,1] +
                                                            N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,1] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ zero_vz_ind+1:vz.n
                    @. pdf[ivz,:,:,1] = N_out*vz.scratch[ivz]
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##
            # populate vz.scratch2 array with dz/dt values at z = Lz/2
            if evolve_ppar
                vth = sqrt(2.0*pz[end]/density[end])
            else
                vth = nothing
            end
            @. vz.scratch2 = vpagrid_to_dzdt(vz.grid, vth, uz[end], evolve_ppar, evolve_upar)

            # First apply boundary condition that total neutral outflux is equal to ion
            # influx to uz
            uz[end] = - wall_flux_L / density[end]
            #would setting density work better??
            #density[end] = - wall_flux_L / upar[end]

            # obtain the Knudsen cosine distribution at z = Lz/2
            # the z-dependence is only introduced if the peculiiar velocity is used as vz
            # Note this only makes sense for the 1V case with vr.n=vzeta.n=1
            @. vz.scratch = (3.0*pi/vtfac^3)*abs(vz.scratch2)*erfc(abs(vz.scratch2)/vtfac)

            # The v_parallel<0 part of the pdf is replaced by the Knudsen cosine
            # distribution. To ensure the constraint ∫dwpa wpa F = 0 is satisfied, multiply
            # the Knudsen distribution (in vz.scratch) by a normalisation factor given by
            # the integral (over negative v_parallel) of the outgoing Knudsen distribution
            # and (over positive v_parallel) of the incoming pdf.
            knudsen_integral_0 = integrate_over_negative_vz(vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @. vz.scratch4 = vz.grid * vz.scratch
            knudsen_integral_1 = integrate_over_negative_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            @views pdf_integral_0 = integrate_over_positive_vz(pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,end]
            pdf_integral_1 = integrate_over_positive_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            if !evolve_ppar
                # Calculate normalisation factors N_in for the incoming and N_out for the
                # Knudsen parts of the distirbution so that ∫dwpa F = 1 and ∫dwpa wpa F = 0
                # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = 0
                N_in = 1.0 / (pdf_integral_0 - pdf_integral_1/knudsen_integral_1*knudsen_integral_0)
                N_out = -N_in * pdf_integral_1 / knudsen_integral_1

                zero_vz_ind = 0
                for ivz ∈ vz.n:-1:1
                    if vz.scratch2[ivz] >= zero
                        @views @. pdf[ivz,:,:,end] = N_in*pdf[ivz,:,:,end]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,end] = 0.5*(N_in*pdf[ivz,:,:,end] + N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ 1:zero_vz_ind-1
                    @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                end
            else
                @. vz.scratch4 = vz.grid * vz.grid * vz.scratch
                knudsen_integral_2 = integrate_over_negative_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views @. pdf_buffer = vz.grid * vz.grid * pdf[:,:,:,end]
                pdf_integral_2 = integrate_over_positive_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                pdf_buffer .*= vz.grid
                pdf_integral_3 = integrate_over_positive_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                # Calculate normalisation factor N_out for the Knudsen part of the
                # distirbution and normalisation factor N_in and correction term C*wpa*F_in
                # for the incoming distribution so that ∫dwpa F = 1, ∫dwpa wpa F = 0, and
                # ∫dwpa wpa^2 F = 1/2
                # ⇒ N_in*pdf_integral_0 + C*pdf_integral_1 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + C*pdf_integral_2 + N_out*knudsen_integral_1 = 0
                #   N_in*pdf_integral_2 + C*pdf_integral_3 + N_out*knudsen_integral_2 = 1/2
                N_in = (0.5*knudsen_integral_0*pdf_integral_2 +
                        knudsen_integral_1*(pdf_integral_3 - 0.5*pdf_integral_1) -
                        knudsen_integral_2*pdf_integral_2) /
                       (knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) +
                        knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) +
                        knudsen_integral_2*(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2))
                N_out = -(N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2^2) + 0.5*pdf_integral_2) /
                         (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                C = (0.5 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)/pdf_integral_3

                zero_vz_ind = 0
                for ivz ∈ vz.n:-1:1
                    if vz.scratch2[ivz] >= zero
                        @views @. pdf[ivz,:,:,end] = N_in*pdf[ivz,:,:,end] + C*vz.grid[ivz]*pdf[ivz,:,:,end]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,end] = 0.5*(N_in*pdf[ivz,:,:,end] +
                                                              C*vz.grid[ivz]*pdf[ivz,:,:,end] +
                                                              N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ 1:zero_vz_ind-1
                    @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                end
            end
        end
    end
end

"""
create an array of dz/dt values corresponding to the given vpagrid values
"""
function vpagrid_to_dzdt(vpagrid, vth, upar, evolve_ppar, evolve_upar)
    if evolve_ppar
        if evolve_upar
            return vpagrid .* vth .+ upar
        else
            return vpagrid .* vth
        end
    elseif evolve_upar
        return vpagrid .+ upar
    else
        return vpagrid
    end
end

"""
enforce the z boundary condition on the evolved velocity space moments of f
"""
function enforce_z_boundary_condition_moments!(density, moments, bc::String)
    ## TODO: parallelise
    #@begin_serial_region()
    #@serial_region begin
    #    # enforce z boundary condition on density if it is evolved separately from f
    #	if moments.evolve_density
    #        # TODO: extend to 'periodic' BC case, as this requires further code modifications to be consistent
    #        # with finite difference derivatives (should be fine for Chebyshev)
    #        if bc == "wall"
    #            @loop_s_r is ir begin
    #                density[1,ir,is] = 0.5*(density[1,ir,is] + density[end,ir,is])
    #                density[end,ir,is] = density[1,ir,is]
    #        	end
    #        end
    #    end
    #end
end

"""
"""
function enforce_v_boundary_condition_local!(f, bc, speed, v_diffusion, v, v_spectral)
    if bc == "zero"
        if v_diffusion || speed[1] > 0.0
            # 'upwind' boundary
            f[1] = 0.0
        end
        if v_diffusion || speed[end] < 0.0
            # 'upwind' boundary
            f[end] = 0.0
        end
    elseif bc == "both_zero"
        f[1] = 0.0
        f[end] = 0.0
    elseif bc == "zero_gradient"
        D0 = v_spectral.lobatto.Dmat[1,:]
        # adjust F(vpa = -L/2) so that d F / d vpa = 0 at vpa = -L/2
        f[1] = -sum(D0[2:v.ngrid].*f[2:v.ngrid])/D0[1]

        D0 = v_spectral.lobatto.Dmat[end,:]
        # adjust F(vpa = L/2) so that d F / d vpa = 0 at vpa = L/2
        f[end] = -sum(D0[1:v.ngrid-1].*f[end-v.ngrid+1:end-1])/D0[v.ngrid]
    elseif bc == "periodic"
        f[1] = 0.5*(f[1]+f[end])
        f[end] = f[1]
    elseif bc == "none"
        # Do nothing
    else
        error("Unsupported boundary condition option '$bc' for $(v.name)")
    end
    return nothing
end

"""
enforce zero boundary condition at vperp -> infinity
"""
function enforce_vperp_boundary_condition! end

function enforce_vperp_boundary_condition!(f::AbstractArray{mk_float,5}, bc, vperp, vperp_spectral, vperp_advect, diffusion)
    @loop_s is begin
        @views enforce_vperp_boundary_condition!(f[:,:,:,:,is], bc, vperp, vperp_spectral, vperp_advect[is], diffusion)
    end
    return nothing
end

function enforce_vperp_boundary_condition!(f::AbstractArray{mk_float,4}, bc, vperp, vperp_spectral, vperp_advect, diffusion)
    @loop_r ir begin
        @views enforce_vperp_boundary_condition!(f[:,:,:,ir], bc, vperp, vperp_spectral,
                                                 vperp_advect, diffusion, ir)
    end
    return nothing
end

function enforce_vperp_boundary_condition!(f::AbstractArray{mk_float,3}, bc, vperp,
                                           vperp_spectral, vperp_advect, diffusion, ir)
    if bc == "zero" || bc == "zero-impose-regularity"
        nvperp = vperp.n
        ngrid = vperp.ngrid
        # set zero boundary condition
        @loop_z_vpa iz ivpa begin
            if diffusion || vperp_advect.speed[nvperp,ivpa,iz,ir] < 0.0
                f[ivpa,nvperp,iz] = 0.0
            end
        end
        # set regularity condition d F / d vperp = 0 at vperp = 0
        if bc == "zero-impose-regularity" && (vperp.discretization == "gausslegendre_pseudospectral" || vperp.discretization == "chebyshev_pseudospectral")
            D0 = vperp_spectral.radau.D0
            buffer = @view vperp.scratch[1:ngrid-1]
            @loop_z_vpa iz ivpa begin
                if diffusion || vperp_advect.speed[1,ivpa,iz,ir] > 0.0
                    # adjust F(vperp = 0) so that d F / d vperp = 0 at vperp = 0
                    @views @. buffer = D0[2:ngrid] * f[ivpa,2:ngrid,iz]
                    f[ivpa,1,iz] = -sum(buffer)/D0[1]
                end
            end
        elseif bc == "zero"
            # do nothing
        else
            println("vperp.bc=\"$bc\" not supported by discretization "
                    * "$(vperp.discretization)")
        end
    elseif bc == "none"
        # Do nothing
    else
        error("Unsupported boundary condition option '$bc' for vperp")
    end
end

"""
    skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa)

This function returns `true` when the grid point specified by `iz`, `ivperp`, `ivpa` would
be set by the boundary conditions on the electron distribution function. When this
happens, the corresponding row should be skipped when adding contributions to the Jacobian
matrix, so that the row remains the same as a row of the identity matrix, so that the
Jacobian matrix does not modify those points. Returns `false` otherwise.
"""
function skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
    # z boundary condition
    # Treat as if using Dirichlet boundary condition for incoming part of the distribution
    # function on the block boundary, regardless of the actual boundary condition and
    # whether this is an internal boundary or an actual domain boundary. This prevents the
    # matrix evaluated for a single block (without coupling to neighbouring blocks) from
    # becoming singular
    if iz == 1 && z_speed[iz,ivpa,ivperp] ≥ 0.0
        return true
    end
    if iz == z.n && z_speed[iz,ivpa,ivperp] ≤ 0.0
        return true
    end

    # vperp boundary condition
    if vperp.n > 1 && ivperp == vperp.n
        return true
    end

    if ivpa == 1 || ivpa == vpa.n
        return true
    end

    return false
end

end # boundary_conditions
