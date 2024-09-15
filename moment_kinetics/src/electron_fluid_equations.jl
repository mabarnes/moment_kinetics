module electron_fluid_equations

export calculate_electron_density!
export calculate_electron_upar_from_charge_conservation!
export calculate_electron_moments!
export electron_energy_equation!
export calculate_electron_qpar!
export calculate_electron_parallel_friction_force!
export calculate_electron_qpar_from_pdf!
export update_electron_vth_temperature!

using ..communication
using ..derivatives: derivative_z!
using ..looping
using ..input_structs
using ..moment_kinetics_structs: electron_pdf_substruct, moments_electron_substruct
using ..nonlinear_solvers
using ..type_definitions: mk_float
using ..velocity_moments: integrate_over_vspace

using MPI

"""
use quasineutrality to obtain the electron density from the initial
densities of the various ion species:
    sum_i dens_i = dens_e
inputs:
    dens_e = electron density at previous time level
    updated = flag indicating if the electron density is updated
    dens_i = updated ion density
output:
    dens_e = updated electron density
    updated = flag indicating that the electron density has been updated
"""
function calculate_electron_density!(dens_e, updated, dens_i)
    # only update the electron density if it has not already been updated
    if !updated[]
        begin_r_z_region()
        # enforce quasineutrality
        @loop_r_z ir iz begin
            dens_e[iz,ir] = 0.0
            @loop_s is begin
                dens_e[iz,ir] += dens_i[iz,ir,is]
            end
        end
    end
    # set flag indicating that the electron density has been updated
    updated[] = true
    return nothing
end

"""
use charge conservation equation to solve for the electron parallel flow density:
    d/dz(sum_i n_i upar_i - n_e upar_e) = 0
    ==> [sum_i n_i upar_i](z) - [sum_i n_i upar_i](zbound) = [n_e upar_e](z) - [n_e upar_e](zbound)
inputs: 
    upar_e = should contain updated electron parallel flow at boundaries in zed
    updated = flag indicating whether the electron parallel flow is already updated
    dens_e = electron particle density
    upar_i = ion parallel flow density
    dens_i = ion particle density
output:
    upar_e = contains the updated electron parallel flow
"""
function calculate_electron_upar_from_charge_conservation!(upar_e, updated, dens_e, upar_i, dens_i, electron_model, r, z)
    # only calculate the electron parallel flow if it is not already updated
    if !updated[]
        begin_r_region()
        # get the number of zed grid points, nz
        nz = size(upar_e,1)
        # initialise the electron parallel flow density to zero
        @loop_r_z ir iz begin
            upar_e[iz,ir] = 0.0
        end
        # if using a simple logical sheath model, then the electron parallel current at the boundaries in zed
        # is equal and opposite to the ion parallel current
        if electron_model ∈ (boltzmann_electron_response_with_simple_sheath,
                             braginskii_fluid, kinetic_electrons,
                             kinetic_electrons_with_temperature_equation)
            boundary_flux = r.scratch_shared
            boundary_ion_flux = r.scratch_shared2
            if z.irank == 0
                @loop_r ir begin
                    boundary_flux[ir] = 0.0
                    boundary_ion_flux[ir] = 0.0
                    @loop_s is begin
                        boundary_flux[ir] += dens_i[1,ir,is] * upar_i[1,ir,is]
                        boundary_ion_flux[ir] += dens_i[1,ir,is] * upar_i[1,ir,is]
                    end
                end
            end
            begin_serial_region()
            @serial_region begin
                MPI.Bcast!(boundary_flux, 0, z.comm)
                MPI.Bcast!(boundary_ion_flux, 0, z.comm)
            end
            # loop over ion species, adding each species contribution to the
            # ion parallel particle flux at the boundaries in zed
            begin_r_z_region()
            @loop_r_z ir iz begin
                # initialise the electron particle flux to its value at the boundary in
                # zed and subtract the ion boundary flux - we want to calculate upar_e =
                # boundary_flux + (ion_flux - boundary_ion_flux) as at this intermediate
                # point upar is actually the electron particle flux
                upar_e[iz,ir] = boundary_flux[ir] - boundary_ion_flux[ir]
                # add the contributions to the electron particle flux from the various ion species
                # particle fluxes
                @loop_s is begin
                    upar_e[iz,ir] += dens_i[iz,ir,is] * upar_i[iz,ir,is]
                end
                # convert from parallel particle flux to parallel particle density
                upar_e[iz,ir] /= dens_e[iz,ir]
            end
        else
            begin_r_z_region()
            @loop_r_z ir iz begin
                upar_e[iz,ir] = upar_i[iz,ir,1]
            end
        end
        updated[] = true
    end
    return nothing
end

function calculate_electron_moments!(scratch, pdf, moments, composition, collisions, r, z,
                                     vpa)
    calculate_electron_density!(scratch.electron_density, moments.electron.dens_updated,
                                scratch.density)
    calculate_electron_upar_from_charge_conservation!(
        scratch.electron_upar, moments.electron.upar_updated, scratch.electron_density,
        scratch.upar, scratch.density, composition.electron_physics, r, z)
    if composition.electron_physics ∉ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        begin_r_z_region()
        @loop_r_z ir iz begin
            scratch.electron_ppar[iz,ir] = 0.5 * composition.me_over_mi *
                                           scratch.electron_density[iz,ir] *
                                           moments.electron.vth[iz,ir]^2
        end
        moments.electron.ppar_updated[] = true
    end
    update_electron_vth_temperature!(moments, scratch.electron_ppar,
                                     scratch.electron_density, composition)
    calculate_electron_qpar!(moments.electron, pdf.electron, scratch.electron_ppar,
                             scratch.electron_upar, scratch.upar, collisions.nu_ei,
                             composition.me_over_mi, composition.electron_physics, vpa)
    if composition.electron_physics == braginskii_fluid
        electron_fluid_qpar_boundary_condition!(scratch.electron_ppar,
                                                scratch.electron_upar,
                                                scratch.electron_density,
                                                moments.electron, z)
    end
    return nothing
end

"""
use the electron energy or temperature equation to evolve the electron temperature via an
explicit time advance.
NB: so far, this is only set up for 1D problem, where we can assume
an isotropic distribution in f_e so that p_e = n_e T_e = ppar_e
"""
function electron_energy_equation!(ppar_out, ppar_in, electron_density, electron_upar,
                                   ion_density, ion_upar, ion_ppar, density_neutral,
                                   uz_neutral, pz_neutral, moments, collisions, dt,
                                   composition, electron_source_settings, num_diss_params,
                                   z; conduction=true)
    if composition.electron_physics == kinetic_electrons_with_temperature_equation
        # Hacky way to implement temperature equation:
        #  - convert ppar to T by dividing by density
        #  - update T with a forward-Euler step using the temperature equation
        #  - multiply by density to get back to ppar (should this be new density rather than
        #    old density? For initial testing, only looking at the electron initialisation
        #    where density is not updated, this does not matter).

        begin_r_z_region()
        # define some abbreviated variables for convenient use in rest of function
        me_over_mi = composition.me_over_mi
        nu_ei = collisions.nu_ei
        T_in = moments.temp
        # calculate contribution to rhs of energy equation (formulated in terms of pressure)
        # arising from derivatives of ppar, qpar and upar
        @loop_r_z ir iz begin
            # Convert ppar_out to temperature for most of this function
            ppar_out[iz,ir] *= 2.0 / electron_density[iz,ir]
            ppar_out[iz,ir] -= dt*(electron_upar[iz,ir]*moments.dT_dz[iz,ir]
                                   + 2.0*T_in[iz,ir]*moments.dupar_dz[iz,ir])
        end
        if conduction
            @loop_r_z ir iz begin
                ppar_out[iz,ir] -= 2.0 * dt*moments.dqpar_dz[iz,ir] / electron_density[iz,ir]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from artificial diffusion
        diffusion_coefficient = num_diss_params.electron.moment_dissipation_coefficient
        if diffusion_coefficient > 0.0
            error("diffusion not implemented for electron temperature equation yet")
            @loop_r_z ir iz begin
                ppar_out[iz,ir] += dt*diffusion_coefficient*moments.d2T_dz2[iz,ir]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from electron-ion collisions
        if nu_ei > 0.0
            @loop_s_r_z is ir iz begin
                ppar_out[iz,ir] += dt * 2.0 * (2 * me_over_mi * nu_ei * (2.0*ion_ppar[iz,ir,is]/ion_density[iz,ir,is] - T_in[iz,ir]))
                ppar_out[iz,ir] += dt * 2.0 * ((2/3) * moments.parallel_friction[iz,ir]
                                               * (ion_upar[iz,ir,is]-electron_upar[iz,ir])) / electron_density[iz,ir]
            end
        end
        # add in contributions due to charge exchange/ionization collisions
        if composition.n_neutral_species > 0
            if abs(collisions.charge_exchange_electron) > 0.0
                @loop_sn_r_z isn ir iz begin
                    ppar_out[iz,ir] +=
                        dt * 2.0 * me_over_mi * collisions.charge_exchange_electron * (
                            2*(pz_neutral[iz,ir,isn] -
                               density_neutral[iz,ir,isn]*ppar_in[iz,ir]/electron_density[iz,ir]) +
                            (2/3)*density_neutral[iz,ir,isn] *
                            (uz_neutral[iz,ir,isn] - electron_upar[iz,ir])^2)
                end
            end
            if abs(collisions.ionization_electron) > 0.0
                @loop_sn_r_z isn ir iz begin
                    ppar_out[iz,ir] +=
                        dt * 2.0 * collisions.ionization_electron * density_neutral[iz,ir,isn] * (
                            ppar_in[iz,ir] / electron_density[iz,ir]  -
                            collisions.ionization_energy)
                end
            end
        end

        for index ∈ eachindex(electron_source_settings)
            if electron_source_settings[index].active
                @views pressure_source_amplitude = moments.external_source_pressure_amplitude[:, :, index]
                @views density_source_amplitude = moments.external_source_density_amplitude[:, :, index]
                @loop_r_z ir iz begin
                    ppar_out[iz,ir] += dt * (2.0 * pressure_source_amplitude[iz,ir]
                                            - T_in[iz,ir] * density_source_amplitude[iz,ir]) /
                                            electron_density[iz,ir]
                end
            end
        end

        # Now that forward-Euler step for temperature is finished, convert ppar_out back to
        # pressure.
        @loop_r_z ir iz begin
            ppar_out[iz,ir] *= 0.5 * electron_density[iz,ir]
        end
    else
        begin_r_z_region()
        # define some abbreviated variables for convenient use in rest of function
        me_over_mi = composition.me_over_mi
        nu_ei = collisions.nu_ei
        # calculate contribution to rhs of energy equation (formulated in terms of pressure)
        # arising from derivatives of ppar, qpar and upar
        @loop_r_z ir iz begin
            ppar_out[iz,ir] -= dt*(electron_upar[iz,ir]*moments.dppar_dz[iz,ir]
                                   + 3*ppar_in[iz,ir]*moments.dupar_dz[iz,ir])
        end
        if conduction
            @loop_r_z ir iz begin
                ppar_out[iz,ir] -= dt*moments.dqpar_dz[iz,ir]
            end
        end
        # @loop_r_z ir iz begin
        #     ppar_out[iz,ir] -= dt*(electron_upar[iz,ir]*moments.dppar_dz[iz,ir]
        #                 + (2/3)*moments.dqpar_dz[iz,ir]
        #                 + (5/3)*ppar_in[iz,ir]*moments.dupar_dz[iz,ir])
        # end
        # compute the contribution to the rhs of the energy equation
        # arising from artificial diffusion
        diffusion_coefficient = num_diss_params.electron.moment_dissipation_coefficient
        if diffusion_coefficient > 0.0
            @loop_r_z ir iz begin
                ppar_out[iz,ir] += dt*diffusion_coefficient*moments.d2ppar_dz2[iz,ir]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from electron-ion collisions
        if nu_ei > 0.0
            @loop_s_r_z is ir iz begin
                ppar_out[iz,ir] += dt * (2 * me_over_mi * nu_ei * (ion_ppar[iz,ir,is] - ppar_in[iz,ir]))
                ppar_out[iz,ir] += dt * ((2/3) * moments.parallel_friction[iz,ir]
                                         * (ion_upar[iz,ir,is]-electron_upar[iz,ir]))
            end
        end
        # add in contributions due to charge exchange/ionization collisions
        if composition.n_neutral_species > 0
            if abs(collisions.charge_exchange_electron) > 0.0
                @loop_sn_r_z isn ir iz begin
                    ppar_out[iz,ir] +=
                        dt * me_over_mi * collisions.charge_exchange_electron * (
                        2*(electron_density[iz,ir]*pz_neutral[iz,ir,isn] -
                        density_neutral[iz,ir,isn]*ppar_in[iz,ir]) +
                        (2/3)*electron_density[iz,ir]*density_neutral[iz,ir,isn] *
                        (uz_neutral[iz,ir,isn] - electron_upar[iz,ir])^2)
                end
            end
            if abs(collisions.ionization_electron) > 0.0
                # @loop_s_r_z is ir iz begin
                #     ppar_out[iz,ir] +=
                #         dt * collisions.ionization_electron * density_neutral[iz,ir,is] * (
                #         ppar_in[iz,ir] -
                #         (2/3)*electron_density[iz,ir] * collisions.ionization_energy)
                # end
                @loop_sn_r_z isn ir iz begin
                    ppar_out[iz,ir] +=
                        dt * collisions.ionization_electron * density_neutral[iz,ir,isn] * (
                        ppar_in[iz,ir] -
                        electron_density[iz,ir] * collisions.ionization_energy)
                end
            end
        end

        for index ∈ eachindex(electron_source_settings)
            if electron_source_settings[index].active
                @views source_amplitude = moments.external_source_pressure_amplitude[:, :, index]
                @loop_r_z ir iz begin
                    ppar_out[iz,ir] += dt * source_amplitude[iz,ir]
                end
            end
        end
    end

    return nothing
end

"""
    electron_energy_residual!(residual, electron_ppar_out, fvec_in, moments,
                              collisions, composition, external_source_settings,
                              num_diss_params, z, dt)

The residual is a function whose input is `electron_ppar`, so that when it's output
`residual` is zero, electron_ppar is the result of a backward-Euler timestep:
  (f_out - f_in) / dt = RHS(f_out)
⇒ (f_out - f_in) - dt*RHS(f_out) = 0

This function assumes any needed moment derivatives are already calculated using
`electron_ppar_out` and stored in `moments.electron`.
"""
function electron_energy_residual!(residual, electron_ppar_out, fvec_in, moments,
                                   collisions, composition, external_source_settings,
                                   num_diss_params, z, dt)
    begin_r_z_region()
    electron_ppar_in = fvec_in.electron_ppar
    @loop_r_z ir iz begin
        residual[iz,ir] = electron_ppar_in[iz,ir]
    end
    electron_energy_equation!(residual, electron_ppar_out,
                              fvec_in.density, fvec_in.electron_upar, fvec_in.density,
                              fvec_in.upar, fvec_in.ppar, fvec_in.density_neutral,
                              fvec_in.uz_neutral, fvec_in.pz_neutral, moments.electron,
                              collisions, dt, composition,
                              external_source_settings.electron, num_diss_params, z)
    # Now
    #   residual = f_in + dt*RHS(f_out)
    # so update to desired residual
    begin_r_z_region()
    @loop_r_z ir iz begin
        residual[iz,ir] = (electron_ppar_out[iz,ir] - residual[iz,ir])
    end
end

"""
Add just the braginskii conduction contribution to the electron pressure, and assume that
we have to calculate qpar and dqpar_dz from ppar within this function (they are not
pre-calculated).
"""
function electron_braginskii_conduction!(ppar_out::AbstractVector{mk_float},
                                         ppar_in::AbstractVector{mk_float},
                                         dens::AbstractVector{mk_float},
                                         upar_e::AbstractVector{mk_float},
                                         upar_i::AbstractVector{mk_float},
                                         electron_moments, collisions, composition, z,
                                         z_spectral, scratch_dummy, dt, ir)

    buffer_r_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[ir,1]

    temp = @view electron_moments.temp[:,ir]
    dT_dz = @view electron_moments.dT_dz[:,ir]
    qpar = @view electron_moments.qpar[:,ir]
    dqpar_dz = @view electron_moments.dqpar_dz[:,ir]

    update_electron_temperature!(temp, ppar_in, dens, composition)
    derivative_z!(dT_dz, temp, buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4, z_spectral,
                  z)
    electron_moments.qpar_updated[] = false
    calculate_electron_qpar!(electron_moments, nothing, ppar_in, upar_e, upar_i,
                             collisions.nu_ei, composition.me_over_mi,
                             composition.electron_physics, nothing)
    electron_fluid_qpar_boundary_condition!(ppar_in, upar_e, dens, electron_moments, z)
    derivative_z!(dqpar_dz, qpar, buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                  z_spectral, z)

    @loop_r_z ir iz begin
        ppar_out[iz,ir] -= dt*electron_moments.dqpar_dz[iz,ir]
    end

    return nothing
end

function implicit_braginskii_conduction!(fvec_out, fvec_in, moments, z, r, dt, z_spectral,
                                         composition, collisions, scratch_dummy,
                                         nl_solver_params)
    begin_z_region()

    for ir ∈ 1:r.n
        ppar_out = @view fvec_out.electron_ppar[:,ir]
        ppar_in = @view fvec_in.electron_ppar[:,ir]
        dens = @view fvec_in.electron_density[:,ir]
        upar_e = @view fvec_in.electron_upar[:,ir]
        upar_i = @view fvec_in.upar[:,ir]

        # Explicit timestep to give initial guess for implicit solve
        electron_braginskii_conduction!(ppar_out, ppar_in, dens, upar_e, upar_i,
                                        moments.electron, collisions, composition, z,
                                        z_spectral, scratch_dummy, dt, ir)

        # Define a function whose input is `electron_ppar`, so that when it's output
        # `residual` is zero, electron_ppar is the result of a backward-Euler timestep:
        #   (f_new - f_old) / dt = RHS(f_new)
        # ⇒ (f_new - f_old)/dt - RHS(f_new) = 0
        function residual_func!(residual, electron_ppar)
            begin_z_region()
            @loop_z iz begin
                residual[iz] = ppar_in[iz]
            end
            electron_braginskii_conduction!(residual, electron_ppar, dens, upar_e,
                                            upar_i, moments.electron, collisions,
                                            composition, z, z_spectral, scratch_dummy,
                                            dt, ir)
            # Now
            #   residual = f_old + dt*RHS(f_new)
            # so update to desired residual
            begin_z_region()
            @loop_z iz begin
                residual[iz] = (electron_ppar[iz] - residual[iz])
            end
        end

        # Shared-memory buffers
        residual = @view scratch_dummy.buffer_zs_1[:,1]
        delta_x = @view scratch_dummy.buffer_zs_2[:,1]
        rhs_delta = @view scratch_dummy.buffer_zs_3[:,1]
        v = @view scratch_dummy.buffer_zs_4[:,1]
        w = @view scratch_dummy.buffer_zrs_1[:,1,1]

        success = newton_solve!(ppar_out, residual_func!, residual, delta_x, rhs_delta, v,
                                w, nl_solver_params; left_preconditioner=nothing,
                                right_preconditioner=nothing, coords=(z=z,))
        if !success
            return success
        end
    end

    nl_solver_params.stage_counter[] += 1

    return true
end

"""
solve the electron force balance (parallel momentum) equation for the
parallel electric field, Epar:
    Epar = -dphi/dz = (2/n_e) * (-dppar_e/dz + friction_force + n_e * m_e  * n_n * R_en * (u_n - u_e))
    NB: in 1D only Epar is needed for update of ion pdf, so boundary phi is irrelevant
inputs:
    dens_e = electron density
    dppar_dz = zed derivative of electron parallel pressure
    nu_ei = electron-ion collision frequency
    friction = electron-ion parallel friction force
    n_neutral_species = number of evolved neutral species
    charge_exchange = electron-neutral charge exchange frequency
    me_over_mi = electron-ion mass ratio
    dens_n = neutral density
    upar_n = neutral parallel flow
    upar_e = electron parallel flow
output:
    Epar = parallel electric field
"""
function calculate_Epar_from_electron_force_balance!(Epar, dens_e, dppar_dz, nu_ei, friction, n_neutral_species,
                                                     charge_exchange, me_over_mi, dens_n, upar_n, upar_e)
    begin_r_z_region()
    # get the contribution to Epar from the parallel pressure
    @loop_r_z ir iz begin
        Epar[iz,ir] = -(2/dens_e[iz,ir]) * dppar_dz[iz,ir]
    end
    # if electron-ion collisions are taken into account, include electron-ion parallel friction
    if nu_ei > 0
        @loop_r_z ir iz begin
            Epar[iz,ir] += (2/dens_e[iz,ir]) * friction[iz,ir]
        end
    end
    # if there are neutral species evolved and accounting for charge exchange collisions with neutrals
    if n_neutral_species > 0 && charge_exchange > 0
        @loop_sn_r_z isn ir iz begin
            Epar[iz,ir] += 2 * me_over_mi * dens_n[iz,ir] * charge_exchange * (upar_n[iz,ir,isn] - upar_e[iz,ir]) 
        end
    end
    return nothing
end

"""
"""
function calculate_electron_parallel_friction_force!(friction, dens_e, upar_e, upar_i, dTe_dz,
                                                     me_over_mi, nu_ei, electron_model)
    begin_r_z_region()
    if electron_model == braginskii_fluid
        @loop_r_z ir iz begin
            friction[iz,ir] = -(1/2) * 0.71 * dens_e[iz,ir] * dTe_dz[iz,ir]
        end
        @loop_s_r_z is ir iz begin
            friction[iz,ir] += 0.51 * dens_e[iz,ir] * me_over_mi * nu_ei * (upar_i[iz,ir,is] - upar_e[iz,ir])
        end
    else
        @loop_r_z ir iz begin
            friction[iz,ir] = 0.0
        end
    end
    return nothing
end

"""
calculate the parallel component of the electron heat flux.
there are currently two supported options for the parallel heat flux:
    Braginskii collisional closure - qpar_e = -3.16*ppar_e/(m_e*nu_ei)*dT/dz - 0.71*ppar_e*(upar_i-upar_e)
    collisionless closure - d(qpar_e)/dz = 0 ==> qpar_e = constant
inputs:
    qpar_e = parallel electron heat flux at the previous time level
    qpar_updated = flag indicating whether qpar is updated already
    pdf = electron pdf
    ppar_e = electron parallel pressure
    upar_e = electron parallel flow
    vth_e = electron thermal speed
    dTe_dz = zed derivative of electron temperature
    upar_i = ion parallel flow
    nu_ei = electron-ion collision frequency
    me_over_mi = electron-to-ion mass ratio
    electron_model = choice of model for electron physics 
    vpa = struct containing information about the vpa coordinate
output:
    qpar_e = updated parallel electron heat flux
    qpar_updated = flag indicating that the parallel electron heat flux is updated
"""
function calculate_electron_qpar!(electron_moments, pdf, ppar_e, upar_e, upar_i, nu_ei,
                                  me_over_mi, electron_model, vpa)
    # only calculate qpar_e if needs updating
    qpar_updated = electron_moments.qpar_updated
    #if !qpar_updated[]
        qpar_e = electron_moments.qpar
        vth_e = electron_moments.vth
        dTe_dz = electron_moments.dT_dz
        if electron_model == braginskii_fluid
            begin_r_z_region()
            # use the classical Braginskii expression for the electron heat flux
            @loop_r_z ir iz begin
                qpar_e[iz,ir] = 0.0
                @loop_s is begin
                    qpar_e[iz,ir] -= 0.71 * ppar_e[iz,ir] * (upar_i[iz,ir,is] - upar_e[iz,ir])
                end
            end
            if nu_ei > 0.0
                @loop_r_z ir iz begin
                    qpar_e[iz,ir] -= (1/2) * 3.16 * ppar_e[iz,ir] / (me_over_mi * nu_ei) * dTe_dz[iz,ir] 
                end
            end
        elseif electron_model ∈ (kinetic_electrons,
                                 kinetic_electrons_with_temperature_equation)
            # use the modified electron pdf to calculate the electron heat flux
            if isa(pdf, electron_pdf_substruct)
                electron_pdf = pdf.norm
            else
                electron_pdf = pdf
            end
            calculate_electron_qpar_from_pdf!(qpar_e, ppar_e, vth_e, electron_pdf, vpa)
        else
            begin_r_z_region()
            # qpar_e is not used. Initialize to 0.0 to avoid failure of
            # @debug_track_initialized check
            @loop_r_z ir iz begin
                qpar_e[iz,ir] = 0.0
            end
        end
    #end
    # qpar has been updated
    qpar_updated[] = true
    return nothing
end

"""
calculate the parallel component of the electron heat flux,
defined as qpar = 2 * ppar * vth * int dwpa (pdf * wpa^3)
"""
function calculate_electron_qpar_from_pdf!(qpar, ppar, vth, pdf, vpa)
    # specialise to 1D for now
    begin_r_z_region()
    ivperp = 1
    @loop_r_z ir iz begin
        @views qpar[iz, ir] = 2*ppar[iz,ir]*vth[iz,ir]*integrate_over_vspace(pdf[:, ivperp, iz, ir], vpa.grid.^3, vpa.wgts)
    end
end

function calculate_electron_heat_source!(heat_source, ppar_e, dupar_dz, dens_n, ionization, ionization_energy,
                                         dens_e, ppar_i, nu_ei, me_over_mi, T_wall, z)
    begin_r_z_region()
    # heat_source currently only used for testing                                         
    # @loop_r_z ir iz begin
    #     heat_source[iz,ir] = (2/3) * ppar_e[iz,ir] * dupar_dz[iz,ir]
    # end
    # if nu_ei > 0.0
    #     @loop_s_r_z is ir iz begin
    #         heat_source[iz,ir] -= 2 * me_over_mi * nu_ei * (ppar_i[iz,ir,is] - ppar_e[iz,ir])
    #     end
    # end
    # n_neutral_species = size(dens_n, 3)
    # if n_neutral_species > 0 && ionization > 0.0
    #    @loop_s_r_z is ir iz begin
    #        heat_source[iz,ir] += (2/3) * dens_n[iz,ir,is] * dens_e[iz,ir] * ionization * ionization_energy
    #    end
    # end
    # @loop_r ir begin
    #     heat_source[1,ir] += 20 * 0.5 * (T_wall * dens_e[1,ir] - ppar_e[1,ir])
    #     heat_source[end,ir] += 20 * (0.5 * T_wall * dens_e[end,ir] - ppar_e[end,ir])
    # end
    # Gaussian heat deposition profile, with a decay of 5 e-foldings at the ends of the domain in z
    @loop_r_z ir iz begin
        heat_source[iz,ir] = 50*exp(-5*(2.0*z.grid[iz]/z.L)^2)
    end
    return nothing
end

#function enforce_parallel_BC_on_electron_pressure!(ppar, dens, T_wall, ppar_i)
#    # assume T_e = T_i at boundaries in z
#    @loop_r ir begin
#        #ppar[1,ir] = 0.5 * dens[1,ir] * T_wall
#        #ppar[end,ir] = 0.5 * dens[end,ir] * T_wall
#        ppar[1,ir] = ppar_i[1,ir,1]
#        ppar[end,ir] = ppar_i[end,ir,1]
#    end
#end

function update_electron_vth_temperature!(moments, ppar, dens, composition)
    begin_r_z_region()

    temp = moments.electron.temp
    vth = moments.electron.vth
    @loop_r_z ir iz begin
        p = max(ppar[iz,ir], 0.0)
        temp[iz,ir] = 2 * p / dens[iz,ir]
        vth[iz,ir] = sqrt(temp[iz,ir] / composition.me_over_mi)
    end
    moments.electron.temp_updated[] = true

    return nothing
end

function update_electron_temperature!(temp, ppar, dens, composition)
    begin_z_region()

    @loop_z iz begin
        p = max(ppar[iz], 0.0)
        temp[iz] = 2 * p / dens[iz]
    end

    return nothing
end

"""
    electron_fluid_qpar_boundary_condition!(electron_moments, z)

Impose fluid approximation to electron sheath boundary condition on the parallel heat
flux. See Stangeby textbook, equations (2.89) and (2.90).
"""
function electron_fluid_qpar_boundary_condition!(ppar, upar, dens, electron_moments, z)
    if z.bc == "periodic"
        # Nothing to do as z-derivative used to calculate qpar already imposed
        # periodicity.
        return nothing
    end

    begin_r_region()

    if z.irank == 0 && (z.irank == z.nrank - 1)
        z_indices = (1, z.n)
    elseif z.irank == 0
        z_indices = (1,)
    elseif z.irank == z.nrank - 1
        z_indices = (z.n,)
    else
        return nothing
    end

    @loop_r ir begin
        for iz ∈ z_indices
            this_ppar = ppar[iz,ir]
            this_upar = electron_moments.upar[iz,ir]
            this_dens = electron_moments.dens[iz,ir]
            particle_flux = this_dens * this_upar
            T_e = electron_moments.temp[iz,ir]

            # Stangeby (2.90)
            gamma_e = 5.5

            # Stangeby (2.89)
            total_heat_flux = gamma_e * T_e * particle_flux

            # E.g. Helander&Sigmar (2.14), neglecting electron viscosity and kinetic
            # energy fluxes due to small mass ratio
            conductive_heat_flux = total_heat_flux - 2.5 * this_ppar * this_upar

            electron_moments.qpar[iz,ir] = conductive_heat_flux
        end
    end

    return nothing
end

end
