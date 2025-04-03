module electron_fluid_equations

export calculate_electron_density!
export calculate_electron_upar_from_charge_conservation!
export calculate_electron_moments!
export calculate_electron_moments_no_r!
export electron_energy_equation!
export electron_energy_equation_no_r!
export add_electron_energy_equation_to_Jacobian!
export calculate_electron_qpar!
export calculate_electron_parallel_friction_force!
export calculate_electron_qpar_from_pdf!
export update_electron_vth_temperature!

using ..calculus: integral
using ..communication
using ..derivatives: derivative_z!
using ..looping
using ..input_structs
using ..timer_utils
using ..moment_kinetics_structs: electron_pdf_substruct
using ..nonlinear_solvers
using ..type_definitions: mk_float

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
        for ir ∈ 1:size(dens_e, 2)
            @views calculate_electron_density_no_r!(dens_e[:,ir], dens_i[:,ir,:], ir)
        end
    end
    # set flag indicating that the electron density has been updated
    updated[] = true
    return nothing
end
function calculate_electron_density_no_r!(dens_e, dens_i, ir)
    @begin_z_region()
    # enforce quasineutrality
    @loop_z iz begin
        dens_e[iz] = 0.0
        @loop_s is begin
            dens_e[iz] += dens_i[iz,is]
        end
    end
    return nothing
end

"""
use charge conservation equation to solve for the electron parallel flow density:
    d/dz(sum_i n_i upar_i - n_e upar_e) = 0
    ==> {sum_i n_i upar_i}(z) - {sum_i n_i upar_i}(zbound) = {n_e upar_e}(z) - {n_e upar_e}(zbound)
inputs: 
    upar_e = should contain updated electron parallel flow at boundaries in zed
    updated = flag indicating whether the electron parallel flow is already updated
    dens_e = electron particle density
    upar_i = ion parallel flow density
    dens_i = ion particle density
output:
    upar_e = contains the updated electron parallel flow
"""
function calculate_electron_upar_from_charge_conservation!(upar_e, updated, dens_e,
                                                           upar_i, dens_i, electron_model,
                                                           r, z)
    # only calculate the electron parallel flow if it is not already updated
    if !updated[]
        for ir ∈ 1:r.n
            @views calculate_electron_upar_from_charge_conservation_no_r!(
                       upar_e[:,ir], updated, dens_e[:,ir], upar_i[:,ir,:], dens_i[:,ir,:],
                       electron_model, r, z, ir)
        end
        updated[] = true
    end
    return nothing
end
function calculate_electron_upar_from_charge_conservation_no_r!(upar_e, updated, dens_e,
                                                                upar_i, dens_i,
                                                                electron_model, r, z, ir)
    @begin_serial_region()
    # initialise the electron parallel flow density to zero
    @loop_z iz begin
        upar_e[iz] = 0.0
    end
    # if using a simple logical sheath model, then the electron parallel current at the boundaries in zed
    # is equal and opposite to the ion parallel current
    if electron_model ∈ (boltzmann_electron_response_with_simple_sheath,
                         braginskii_fluid, kinetic_electrons,
                         kinetic_electrons_with_temperature_equation)
        boundary_flux = @view r.scratch_shared[ir:ir]
        boundary_ion_flux = @view r.scratch_shared2[ir:ir]
        @serial_region begin
            if z.irank == 0
                boundary_flux[] = 0.0
                boundary_ion_flux[] = 0.0
                for is ∈ 1:size(dens_i, 2)
                    boundary_flux[] += dens_i[1,is] * upar_i[1,is]
                    boundary_ion_flux[] += dens_i[1,is] * upar_i[1,is]
                end
            end
            MPI.Bcast!(boundary_flux, 0, z.comm)
            MPI.Bcast!(boundary_ion_flux, 0, z.comm)
        end
        # loop over ion species, adding each species contribution to the
        # ion parallel particle flux at the boundaries in zed
        @begin_z_region()
        @loop_z iz begin
            # initialise the electron particle flux to its value at the boundary in
            # zed and subtract the ion boundary flux - we want to calculate upar_e =
            # boundary_flux + (ion_flux - boundary_ion_flux) as at this intermediate
            # point upar is actually the electron particle flux
            upar_e[iz] = boundary_flux[] - boundary_ion_flux[]
            # add the contributions to the electron particle flux from the various ion species
            # particle fluxes
            @loop_s is begin
                upar_e[iz] += dens_i[iz,is] * upar_i[iz,is]
            end
            # convert from parallel particle flux to parallel particle density
            upar_e[iz] /= dens_e[iz]
        end
    else
        @begin_z_region()
        @loop_z iz begin
            upar_e[iz] = upar_i[iz,1]
        end
    end
    return nothing
end

function calculate_electron_moments!(scratch, pdf, moments, composition, collisions, r, z,
                                     vperp, vpa)
    @begin_z_region()

    if length(scratch.pdf_electron) > 0
        pdf_electron = scratch.pdf_electron
    elseif isa(pdf.electron, electron_pdf_substruct)
        pdf_electron = pdf.electron.norm
    else
        pdf_electron = nothing
    end

    for ir ∈ 1:r.n
        if pdf_electron === nothing
            this_pdf = nothing
        else
            this_pdf = @view pdf_electron[:,:,:,ir]
        end
        @views calculate_electron_moments_no_r!(this_pdf, scratch.electron_density[:,ir],
                                                scratch.electron_upar[:,ir],
                                                scratch.electron_p[:,ir],
                                                scratch.density[:,ir,:],
                                                scratch.upar[:,ir,:], moments,
                                                composition, collisions, r, z, vperp, vpa,
                                                ir)
    end
    return nothing
end

function calculate_electron_moments_no_r!(pdf_electron, electron_density, electron_upar,
                                          electron_p, ion_density, ion_upar, moments,
                                          composition, collisions, r, z, vperp, vpa, ir)
    calculate_electron_density_no_r!(electron_density, ion_density, ir)
    calculate_electron_upar_from_charge_conservation_no_r!(
        electron_upar, moments.electron.upar_updated, electron_density,
        ion_upar, ion_density, composition.electron_physics, r, z, ir)
    if composition.electron_physics ∉ (braginskii_fluid, kinetic_electrons,
                                       kinetic_electrons_with_temperature_equation)
        @begin_z_region()
        @loop_z iz begin
            electron_p[iz] = 0.5 * composition.me_over_mi *
                             electron_density[iz] * moments.electron.vth[iz]^2
        end
        moments.electron.p_updated[] = true
    end
    @views update_electron_vth_temperature_no_r!(moments.electron.vth[:,ir],
                                                 moments.electron.temp[:,ir], electron_p,
                                                 electron_density, composition)
    @views calculate_electron_ppar_no_r!(moments.electron.ppar[:,ir], electron_density,
                                         electron_upar, electron_p,
                                         moments.electron.vth[:,ir], pdf_electron, vpa,
                                         vperp, z, composition.me_over_mi)
    calculate_electron_qpar_no_r!(moments.electron, pdf_electron, electron_p,
                                  electron_density, electron_upar, ion_upar,
                                  collisions.electron_fluid.nu_ei, composition.me_over_mi,
                                  composition.electron_physics, vperp, vpa, ir)
    if composition.electron_physics == braginskii_fluid
        electron_fluid_qpar_boundary_condition!(electron_p, electron_upar,
                                                electron_density, moments.electron, z)
    end
    return nothing
end

"""
use the electron energy or temperature equation to evolve the electron temperature via an
explicit time advance.
NB: so far, this is only set up for 1D problem.
"""
function electron_energy_equation!(p_out, electron_density_out, p_in, electron_density_in,
                                   electron_upar, electron_ppar, ion_density, ion_upar,
                                   ion_p, density_neutral, uz_neutral, p_neutral, moments,
                                   collisions, dt, composition, electron_source_settings,
                                   num_diss_params, r, z; conduction=true)
    for ir ∈ 1:r.n
        @views electron_energy_equation_no_r!(p_out[:,ir], electron_density_out[:,ir],
                                              p_in[:,ir], electron_density_in[:,ir],
                                              electron_upar[:,ir], electron_ppar[:,ir],
                                              ion_density[:,ir,:], ion_upar[:,ir,:],
                                              ion_p[:,ir,:], density_neutral[:,ir,:],
                                              uz_neutral[:,ir,:], p_neutral[:,ir,:],
                                              moments, collisions, dt, composition,
                                              electron_source_settings, num_diss_params,
                                              z, ir; conduction=conduction)
    end
    return nothing
end

"""
"""
@timeit global_timer electron_energy_equation_no_r!(
                         p_out, electron_density_out, p_in, electron_density_in,
                         electron_upar, electron_ppar, ion_density, ion_upar, ion_p,
                         density_neutral, uz_neutral, p_neutral, moments, collisions, dt,
                         composition, electron_source_settings, num_diss_params, z, ir;
                         conduction=true, ion_dt=nothing) = begin
    if composition.electron_physics == kinetic_electrons_with_temperature_equation
        # Hacky way to implement temperature equation:
        #  - convert p to T by dividing by density
        #  - update T with a forward-Euler step using the temperature equation
        #  - multiply by density to get back to p (should this be new density rather than
        #    old density? For initial testing, only looking at the electron initialisation
        #    where density is not updated, this does not matter).

        @begin_z_region()
        # define some abbreviated variables for convenient use in rest of function
        me_over_mi = composition.me_over_mi
        nu_ei = collisions.electron_fluid.nu_ei
        T_in = @view moments.temp[:,ir]
        dT_dt = @view moments.dT_dt[:,ir]
        # calculate contribution to rhs of energy equation (formulated in terms of pressure)
        # arising from derivatives of p, qpar and upar
        @loop_z iz begin
            dT_dt[iz] = -(electron_upar[iz] * moments.dT_dz[iz,ir]
                          + 2.0 / 3.0 * electron_ppar[iz] / density_in[iz] *
                            moments.dupar_dz[iz,ir])
        end
        if conduction
            @loop_z iz begin
                dT_dt[iz] -= 2.0 / 3.0 * moments.dqpar_dz[iz,ir] / electron_density_in[iz]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from artificial diffusion
        diffusion_coefficient = num_diss_params.electron.moment_dissipation_coefficient
        if diffusion_coefficient > 0.0
            error("diffusion not implemented for electron temperature equation yet")
            @loop_z iz begin
                dT_dt[iz] += diffusion_coefficient*moments.d2T_dz2[iz,ir]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from electron-ion collisions
        if nu_ei > 0.0
            @loop_s_z is iz begin
                dT_dt[iz] += (2 * me_over_mi * nu_ei * (ion_p[iz,is]/ion_density[iz,is] - T_in[iz]))
                dT_dt[iz] += ((2/3) * moments.parallel_friction[iz,ir]
                                    * (ion_upar[iz,is]-electron_upar[iz])) / electron_density_in[iz]
            end
        end
        # add in contributions due to charge exchange/ionization collisions
        charge_exchange_electron = collisions.reactions.electron_charge_exchange_frequency
        ionization_electron = collisions.reactions.electron_ionization_frequency
        ionization_energy = collisions.reactions.ionization_energy
        if composition.n_neutral_species > 0
            if abs(charge_exchange_electron) > 0.0
                @loop_sn_z isn iz begin
                    dT_dt[iz] +=
                        me_over_mi * charge_exchange_electron * (
                            2*(p_neutral[iz,isn] -
                               density_neutral[iz,isn]*T_in[iz]) +
                            (2/3)*density_neutral[iz,isn] *
                            (uz_neutral[iz,isn] - electron_upar[iz])^2)
                end
            end
            if abs(ionization_electron) > 0.0
                @loop_sn_z isn iz begin
                    dT_dt[iz] +=
                        (2/3) * ionization_electron * density_neutral[iz,isn] * (
                            T_in[iz] - ionization_energy)
                end
            end
        end

        for index ∈ eachindex(electron_source_settings)
            if electron_source_settings[index].active
                pressure_source_amplitude = @view moments.external_source_pressure_amplitude[:, ir, index]
                density_source_amplitude = @view moments.external_source_density_amplitude[:, ir, index]
                @loop_z iz begin
                    dT_dt[iz] += (pressure_source_amplitude[iz]
                                  - T_in[iz] * density_source_amplitude[iz]) /
                                 electron_density_in[iz]
                end
            end
        end

        if ion_dt !== nothing
            # Add source term to turn steady state solution into a backward-Euler
            # update of electron_p with the ion timestep `ion_dt`.
            p_previous_ion_step = @view moments.electron.p[:,ir]
            @loop_z iz begin
                # At this point, p_out = p_in + dt*RHS(p_in). Here we add a
                # source/damping term so that in the steady state of the electron
                # pseudo-timestepping iteration,
                #   RHS(p) - (p - p_previous_ion_step) / ion_dt = 0,
                # resulting in a backward-Euler step (as long as the
                # pseudo-timestepping loop converges).
                dT_dt[iz] += -(p_in[iz] - p_previous_ion_step[iz]) / electron_density_in[iz] / ion_dt
            end
        end

        # Now that the time derivative for temperature is calculated, convert to an update
        # of pressure, p_out.
        @loop_z iz begin
            # The following is equivalent to converting to temperature, adding time
            # derivative, converting back to pressure, like
            # p_out[iz] *= 1.0 / electron_density_in[iz]
            # p_out[iz] += dt * dT_dt[iz]
            # p_out[iz] *= electron_density_out[iz]
            p_out[iz] += electron_density_out[iz] * dt * dT_dt[iz]
        end
    else
        @begin_z_region()
        # define some abbreviated variables for convenient use in rest of function
        me_over_mi = composition.me_over_mi
        nu_ei = collisions.electron_fluid.nu_ei
        dp_dt = @view moments.dp_dt[:,ir]
        # calculate contribution to rhs of energy equation (formulated in terms of pressure)
        # arising from derivatives of p, qpar and upar
        @loop_z iz begin
            dp_dt[iz] = -(electron_upar[iz]*moments.dp_dz[iz,ir]
                          + p_in[iz]*moments.dupar_dz[iz,ir]
                          + 2.0 / 3.0 * electron_ppar[iz]*moments.dupar_dz[iz,ir])
        end
        if conduction
            @loop_z iz begin
                dp_dt[iz] -= 2.0 / 3.0 * moments.dqpar_dz[iz,ir]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from artificial diffusion
        diffusion_coefficient = num_diss_params.electron.moment_dissipation_coefficient
        if diffusion_coefficient > 0.0
            @loop_z iz begin
                dp_dt[iz] += diffusion_coefficient*moments.d2p_dz2[iz,ir]
            end
        end
        # compute the contribution to the rhs of the energy equation
        # arising from electron-ion collisions
        if nu_ei > 0.0
            @loop_s_z is iz begin
                dp_dt[iz] += (2 * me_over_mi * nu_ei * (ion_p[iz,is] - p_in[iz]))
                dp_dt[iz] += ((2/3) * moments.parallel_friction[iz]
                              * (ion_upar[iz,is]-electron_upar[iz]))
            end
        end
        # add in contributions due to charge exchange/ionization collisions
        if composition.n_neutral_species > 0
            charge_exchange_electron = collisions.reactions.electron_charge_exchange_frequency
            ionization_electron = collisions.reactions.electron_ionization_frequency
            ionization_energy = collisions.reactions.ionization_energy
            if abs(charge_exchange_electron) > 0.0
                @loop_sn_z isn iz begin
                    dp_dt[iz] +=
                        me_over_mi * charge_exchange_electron * (
                        (electron_density_in[iz]*p_neutral[iz,isn] -
                        density_neutral[iz,isn]*p_in[iz]) +
                        (1/3)*me_over_mi*electron_density_in[iz]*density_neutral[iz,isn] *
                        (uz_neutral[iz,isn] - electron_upar[iz])^2)
                end
            end
            if abs(ionization_electron) > 0.0
                @loop_sn_z isn iz begin
                    dp_dt[iz] +=
                        -ionization_electron * density_neutral[iz,isn] *
                         electron_density_in[iz] * ionization_energy
                end
            end
        end

        for index ∈ eachindex(electron_source_settings)
            if electron_source_settings[index].active
                source_amplitude = @view moments.external_source_pressure_amplitude[:, ir, index]
                @loop_z iz begin
                    dp_dt[iz] += source_amplitude[iz]
                end
            end
        end

        if ion_dt !== nothing
            # Add source term to turn steady state solution into a backward-Euler
            # update of electron_p with the ion timestep `ion_dt`.
            p_previous_ion_step = @view moments.p[:,ir]
            @loop_z iz begin
                # At this point, p_out = p_in + dt*RHS(p_in). Here we add a
                # source/damping term so that in the steady state of the electron
                # pseudo-timestepping iteration,
                #   RHS(p) - (p - p_previous_ion_step) / ion_dt = 0,
                # resulting in a backward-Euler step (as long as the
                # pseudo-timestepping loop converges).
                dp_dt[iz] += -(p_in[iz] - p_previous_ion_step[iz]) / ion_dt
            end
        end

        @loop_z iz begin
            p_out[iz] += dt * dp_dt[iz]
        end
    end

    return nothing
end

function add_electron_energy_equation_to_Jacobian!(jacobian_matrix, f, dens, upar, p, ppar,
                                                   vth, third_moment, ddens_dz, dupar_dz,
                                                   dp_dz, dthird_moment_dz, collisions,
                                                   composition, z, vperp, vpa, z_spectral,
                                                   num_diss_params, dt, ir, include=:all;
                                                   f_offset=0, p_offset=0)
    if f_offset == p_offset
        error("Got f_offset=$f_offset the same as p_offset=$p_offset. f and p "
              * "cannot be in same place in state vector.")
    end
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @boundscheck size(jacobian_matrix, 1) ≥ p_offset + z.n || error("p_offset=$p_offset is too big")
    @boundscheck include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    if composition.electron_physics == kinetic_electrons_with_temperature_equation
        error("kinetic_electrons_with_temperature_equation not "
              * "supported yet in preconditioner")
    elseif composition.electron_physics != kinetic_electrons
        error("Unsupported electron_physics=$(composition.electron_physics) "
              * "in electron_backward_euler!() preconditioner.")
    end
    if num_diss_params.electron.moment_dissipation_coefficient > 0.0
        error("z-diffusion of electron_p not yet supported in "
              * "preconditioner")
    end
    if collisions.electron_fluid.nu_ei > 0.0
        error("electron-ion collision terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_charge_exchange_frequency > 0.0
        error("electron 'charge exchange' terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_ionization_frequency > 0.0
        error("electron ionization terms for electron_p not yet "
              * "supported in preconditioner")
    end

    me = composition.me_over_mi
    z_deriv_matrix = z_spectral.D_matrix_csr
    v_size = vperp.n * vpa.n

    @begin_z_region()
    @loop_z iz begin
        # Rows corresponding to electron_p
        row = p_offset + iz

        # Note that as
        #   q = 2 * p * vth * ∫dw_∥ w_∥^3 g
        #     = 2 * p^(3/2) * sqrt(2) / n^(1/2) / me^(1/2) * ∫dw_∥ w_∥^3 g
        # we have that
        #   d(q)/dz = 2 * p^(3/2) * sqrt(2) / n^(1/2) / me^(1/2) * ∫dw_∥ w_∥^3 d(g)/dz
        #             - p^(3/2) * sqrt(2) / n^(3/2) / me^(1/2) * ∫dw_∥ w_∥^3 g * d(n)/dz
        #             + 3 * p^(1/2) * sqrt(2) / n^(1/2) / me^(1/2) * ∫dw_∥ w_∥^3 g * d(p)/dz
        # so for the Jacobian
        #   d(d(q)/dz)[irowz])/d(p[icolz])
        #     = (3 * sqrt(2) * p^(1/2) / n^(1/2) / me^(1/2) * ∫dw_∥ w_∥^3 d(g)/dz
        #        - 3/2 * sqrt(2) * p^(1/2) / n^(3/2) / me^(1/2) * ∫dw_∥ w_∥^3 g * d(n)/dz
        #        + 3/2 * sqrt(2) / p^(1/2) / n^(1/2) / me^(1/2) * ∫dw_∥ w_∥^3 g * d(p)/dz)[irowz] * delta[irowz,icolz]
        #       + (3 * sqrt(2) * p^(1/2) / n^(1/2) / me^(1/2) * ∫dw_∥ w_∥^3 g)[irowz] * z_deriv_matrix[irowz,icolz]
        #   d(d(q)/dz)[irowz])/d(g[icolvpa,icolvperp,icolz])
        #     = (2 * sqrt(2) * p^(3/2) / n^(1/2) / me^(1/2))[irowz] * vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3 * z_deriv_matrix[irowz,icolz]
        #       + sqrt(2) * (-p^(3/2) / n^(3/2) / me^(1/2) * dn/dz + 3.0 * p^(1/2) / n^(1/2) / me^(1/2) * dp/dz)[irowz] * vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3 * delta[irowz,icolz]

        # upar*dp_dz
        z_deriv_row_startind = z_deriv_matrix.rowptr[iz]
        z_deriv_row_endind = z_deriv_matrix.rowptr[iz+1] - 1
        z_deriv_colinds = @view z_deriv_matrix.colval[z_deriv_row_startind:z_deriv_row_endind]
        z_deriv_row_nonzeros = @view z_deriv_matrix.nzval[z_deriv_row_startind:z_deriv_row_endind]
        if include ∈ (:all, :explicit_z)
            for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
                col = p_offset + icolz
                jacobian_matrix[row,col] +=
                    dt * upar[iz] * z_deriv_entry
            end
        end

        #   p*dupar_dz + 2/3*ppar*dupar_dz
        #   = p*dupar_dz + 2*p*dupar_dz
        #   = 3*p*dupar_dz
        # for the 1V case where ppar=3*p
        if include === :all
            jacobian_matrix[row,row] += 3.0 * dt * dupar_dz[iz]
        end

        # terms from d(qpar)/dz
        # dq/dz = 3/2*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #         - 1/2*p^(3/2)*sqrt(2/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #         + p*vth*∫dw_∥ w_∥^3 dg/dz
        # d(dq/dz)/d(p) = 3/4*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #                 - 3/4*sqrt(2*p/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #                 + 3/2*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 dg/dz
        #                 + 3/2*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 g * z_deriv_matrix
        # d(dq/dz)/d(g) = (3/2*sqrt(2*p/n/me) * dp/dz
        #                  - 1/2*p^(3/2)*sqrt(2/me)/n^(3/2) * dn/dz) * vpa.grid^3 * vpa.wgts
        #                 + p*vth * vpa.grid^3 * vpa.wgts * z_deriv_matrix
        if include === :all
            jacobian_matrix[row,row] +=
                dt * 2.0 / 3.0 * (0.75 * sqrt(2.0 / p[iz] / dens[iz] / me) * third_moment[iz] * dp_dz[iz]
                                  - 0.75 * sqrt(2.0 * p[iz] / me) / dens[iz]^1.5 * third_moment[iz] * ddens_dz[iz]
                                  + 1.5 * sqrt(2.0 * p[iz] / dens[iz] / me) * dthird_moment_dz[iz])
        end
        if include ∈ (:all, :explicit_z)
            for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
                col = p_offset + icolz
                jacobian_matrix[row,col] += dt * 2.0 / 3.0 * 1.5 * sqrt(2.0 * p[iz] / dens[iz] / me) * third_moment[iz] * z_deriv_entry
            end
        end
        if include ∈ (:all, :explicit_v)
            for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                col = (iz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                jacobian_matrix[row,col] += dt * 2.0 / 3.0 * (1.5*sqrt(2.0*p[iz]/dens[iz]/me)*dp_dz[iz]
                                                              - 0.5 * (p[iz]/dens[iz])^1.5*sqrt(2.0/me)*ddens_dz[iz]) *
                                                             vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3
            end
        end
        for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros), icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
            col = (icolz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
            jacobian_matrix[row,col] += dt * 2.0 / 3.0 * p[iz]^1.5*sqrt(2.0/dens[iz]/me) *
                                                         vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3 * z_deriv_entry
        end
    end

    return nothing
end

function add_electron_energy_equation_to_z_only_Jacobian!(
        jacobian_matrix, dens, upar, p, ppar, vth, third_moment, ddens_dz, dupar_dz,
        dp_dz, dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
        num_diss_params, dt, ir)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == z.n || error("Jacobian matrix size is wrong")

    if composition.electron_physics == kinetic_electrons_with_temperature_equation
        error("kinetic_electrons_with_temperature_equation not "
              * "supported yet in preconditioner")
    elseif composition.electron_physics != kinetic_electrons
        error("Unsupported electron_physics=$(composition.electron_physics) "
              * "in electron_backward_euler!() preconditioner.")
    end
    if num_diss_params.electron.moment_dissipation_coefficient > 0.0
        error("z-diffusion of electron_p not yet supported in "
              * "preconditioner")
    end
    if collisions.electron_fluid.nu_ei > 0.0
        error("electron-ion collision terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_charge_exchange_frequency > 0.0
        error("electron 'charge exchange' terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_ionization_frequency > 0.0
        error("electron ionization terms for electron_p not yet "
              * "supported in preconditioner")
    end

    me = composition.me_over_mi
    z_deriv_matrix = z_spectral.D_matrix_csr
    v_size = vperp.n * vpa.n

    @loop_z iz begin
        # Rows corresponding to electron_p
        row = iz

        z_deriv_row_startind = z_deriv_matrix.rowptr[iz]
        z_deriv_row_endind = z_deriv_matrix.rowptr[iz+1] - 1
        z_deriv_colinds = @view z_deriv_matrix.colval[z_deriv_row_startind:z_deriv_row_endind]
        z_deriv_row_nonzeros = @view z_deriv_matrix.nzval[z_deriv_row_startind:z_deriv_row_endind]
        for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
            col = icolz
            jacobian_matrix[row,col] +=
                dt * upar[iz] * z_deriv_entry
        end

        jacobian_matrix[row,row] += 3.0 * dt * dupar_dz[iz]

        jacobian_matrix[row,row] +=
            dt * 2.0 / 3.0 * (0.75 * sqrt(2.0 / p[iz] / dens[iz] / me) * third_moment[iz] * dp_dz[iz]
                              - 0.75 * sqrt(2.0 * p[iz] / me) / dens[iz]^1.5 * third_moment[iz] * ddens_dz[iz]
                              + 1.5 * sqrt(2.0 * p[iz] / dens[iz] / me) * dthird_moment_dz[iz])
        for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
            col = icolz
            jacobian_matrix[row,col] += dt * 2.0 / 3.0 * 1.5 * sqrt(2.0 * p[iz] / dens[iz] / me) * third_moment[iz] * z_deriv_entry
        end
    end

    return nothing
end

function add_electron_energy_equation_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, ppar, vth, third_moment, ddens_dz, dupar_dz,
        dp_dz, dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
        num_diss_params, dt, ir, iz)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    if composition.electron_physics == kinetic_electrons_with_temperature_equation
        error("kinetic_electrons_with_temperature_equation not "
              * "supported yet in preconditioner")
    elseif composition.electron_physics != kinetic_electrons
        error("Unsupported electron_physics=$(composition.electron_physics) "
              * "in electron_backward_euler!() preconditioner.")
    end
    if num_diss_params.electron.moment_dissipation_coefficient > 0.0
        error("z-diffusion of electron_p not yet supported in "
              * "preconditioner")
    end
    if collisions.electron_fluid.nu_ei > 0.0
        error("electron-ion collision terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_charge_exchange_frequency > 0.0
        error("electron 'charge exchange' terms for electron_p not yet "
              * "supported in preconditioner")
    end
    if composition.n_neutral_species > 0 && collisions.reactions.electron_ionization_frequency > 0.0
        error("electron ionization terms for electron_p not yet "
              * "supported in preconditioner")
    end

    me = composition.me_over_mi

    jacobian_matrix[end,end] += 3.0 * dt * dupar_dz

    jacobian_matrix[end,end] +=
    dt * 2.0 / 3.0 * (0.75 * sqrt(2.0 / p / dens / me) * third_moment * dp_dz
                      - 0.75 * sqrt(2.0 * p / me) / dens^1.5 * third_moment * ddens_dz
                      + 1.5 * sqrt(2.0 * p / dens / me) * dthird_moment_dz)
    for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
        col = (icolvperp - 1) * vpa.n + icolvpa
        jacobian_matrix[end,col] += dt * 2.0 / 3.0 * (1.5*sqrt(2.0*p/dens/me)*dp_dz
                                                      - 0.5 * (p/dens)^1.5*sqrt(2.0/me)*ddens_dz) *
                                         vpa.wgts[icolvpa] * vpa.grid[icolvpa]^3
    end

    return nothing
end

"""
    electron_energy_residual!(residual, electron_p_out, electron_p, in,
                              fvec_in, moments, collisions, composition,
                              external_source_settings, num_diss_params, z, dt, ir)

The residual is a function whose input is `electron_p`, so that when it's output
`residual` is zero, electron_p is the result of a backward-Euler timestep:
  (f_out - f_in) / dt = RHS(f_out)
⇒ (f_out - f_in) - dt*RHS(f_out) = 0

This function assumes any needed moment derivatives are already calculated using
`electron_p_out` and stored in `moments.electron`.

Note that this function operates on a single point in `r`, given by `ir`, and `residual`,
`electron_p_out`, and `electron_p_in` should have no r-dimension.
"""
function electron_energy_residual!(residual, electron_p_out, electron_p, in,
                                   fvec_in, moments, collisions, composition,
                                   external_source_settings, num_diss_params, z, dt, ir)
    @begin_z_region()
    @loop_z iz begin
        residual[iz] = electron_p_in[iz]
    end
    @views electron_energy_equation_no_r!(residual, fvec_in.electron_density[:,ir],
                                          electron_p_out, fvec_in.electron_density[:,ir],
                                          fvec_in.electron_upar[:,ir],
                                          moments.electron.ppar[:,ir],
                                          fvec_in.density[:,ir,:], fvec_in.upar[:,ir,:],
                                          fvec_in.p[:,ir,:],
                                          fvec_in.density_neutral[:,ir,:],
                                          fvec_in.uz_neutral[:,ir,:],
                                          fvec_in.p_neutral[:,ir,:], moments.electron,
                                          collisions, dt, composition,
                                          external_source_settings.electron,
                                          num_diss_params, z, ir)
    # Now
    #   residual = f_in + dt*RHS(f_out)
    # so update to desired residual
    @begin_z_region()
    @loop_z iz begin
        residual[iz] = (electron_p_out[iz] - residual[iz])
    end
end

"""
Add just the braginskii conduction contribution to the electron pressure, and assume that
we have to calculate qpar and dqpar_dz from p within this function (they are not
pre-calculated).
"""
function electron_braginskii_conduction!(p_out::AbstractVector{mk_float},
                                         p_in::AbstractVector{mk_float},
                                         dens::AbstractVector{mk_float},
                                         upar_e::AbstractVector{mk_float},
                                         upar_i::AbstractVector{mk_float},
                                         electron_moments, collisions, composition, z,
                                         z_spectral, scratch_dummy, dt, ir)

    buffer_r_1 = @view scratch_dummy.buffer_rs_1[ir,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[ir,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[ir,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[ir,1]

    vth = @view electron_moments.vth[:,ir]
    temp = @view electron_moments.temp[:,ir]
    dT_dz = @view electron_moments.dT_dz[:,ir]
    qpar = @view electron_moments.qpar[:,ir]
    dqpar_dz = @view electron_moments.dqpar_dz[:,ir]

    update_electron_vth_temperature_no_r!(vth, temp, p_in, dens, composition)
    derivative_z!(dT_dz, temp, buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4, z_spectral,
                  z)
    electron_moments.qpar_updated[] = false
    calculate_electron_qpar!(electron_moments, nothing, p_in, dens, upar_e, upar_i,
                             collisions.electron_fluid.nu_ei, composition.me_over_mi,
                             composition.electron_physics, nothing, nothing)
    electron_fluid_qpar_boundary_condition!(p_in, upar_e, dens, electron_moments, z)
    derivative_z!(dqpar_dz, qpar, buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                  z_spectral, z)

    @loop_r_z ir iz begin
        p_out[iz,ir] -= dt*electron_moments.dqpar_dz[iz,ir]
    end

    return nothing
end

@timeit global_timer implicit_braginskii_conduction!(
                         fvec_out, fvec_in, moments, z, r, dt, z_spectral, composition,
                         collisions, scratch_dummy, nl_solver_params) = begin
    @begin_z_region()

    for ir ∈ 1:r.n
        p_out = @view fvec_out.electron_p[:,ir]
        p_in = @view fvec_in.electron_p[:,ir]
        dens = @view fvec_in.electron_density[:,ir]
        upar_e = @view fvec_in.electron_upar[:,ir]
        upar_i = @view fvec_in.upar[:,ir]

        # Explicit timestep to give initial guess for implicit solve
        electron_braginskii_conduction!(p_out, p_in, dens, upar_e, upar_i,
                                        moments.electron, collisions, composition, z,
                                        z_spectral, scratch_dummy, dt, ir)

        # Define a function whose input is `electron_p`, so that when it's output
        # `residual` is zero, electron_p is the result of a backward-Euler timestep:
        #   (f_new - f_old) / dt = RHS(f_new)
        # ⇒ (f_new - f_old)/dt - RHS(f_new) = 0
        function residual_func!(residual, electron_p; krylov=false)
            @begin_z_region()
            @loop_z iz begin
                residual[iz] = p_in[iz]
            end
            electron_braginskii_conduction!(residual, electron_p, dens, upar_e,
                                            upar_i, moments.electron, collisions,
                                            composition, z, z_spectral, scratch_dummy,
                                            dt, ir)
            # Now
            #   residual = f_old + dt*RHS(f_new)
            # so update to desired residual
            @begin_z_region()
            @loop_z iz begin
                residual[iz] = (electron_p[iz] - residual[iz])
            end
        end

        # Shared-memory buffers
        residual = @view scratch_dummy.buffer_zs_1[:,1]
        delta_x = @view scratch_dummy.buffer_zs_2[:,1]
        rhs_delta = @view scratch_dummy.buffer_zs_3[:,1]
        v = @view scratch_dummy.buffer_zs_4[:,1]
        w = @view scratch_dummy.buffer_zrs_1[:,1,1]

        success = newton_solve!(p_out, residual_func!, residual, delta_x, rhs_delta, v,
                                w, nl_solver_params; left_preconditioner=nothing,
                                right_preconditioner=nothing, coords=(z=z,))
        if !success
            return success
        end
    end

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
    @begin_r_z_region()
    # get the contribution to Epar from the parallel pressure
    @loop_r_z ir iz begin
        Epar[iz,ir] = -(1.0 / dens_e[iz,ir]) * dppar_dz[iz,ir]
    end
    # if electron-ion collisions are taken into account, include electron-ion parallel friction
    if nu_ei > 0
        @loop_r_z ir iz begin
            Epar[iz,ir] += (1.0 / dens_e[iz,ir]) * friction[iz,ir]
        end
    end
    # if there are neutral species evolved and accounting for charge exchange collisions with neutrals
    if n_neutral_species > 0 && charge_exchange > 0
        @loop_sn_r_z isn ir iz begin
            Epar[iz,ir] += me_over_mi * dens_n[iz,ir] * charge_exchange * (upar_n[iz,ir,isn] - upar_e[iz,ir])
        end
    end
    return nothing
end

"""
"""
function calculate_electron_parallel_friction_force!(friction, dens_e, upar_e, upar_i, dTe_dz,
                                                     me_over_mi, nu_ei, electron_model)
    @begin_r_z_region()
    if electron_model == braginskii_fluid
        @loop_r_z ir iz begin
            friction[iz,ir] = -0.71 * dens_e[iz,ir] * dTe_dz[iz,ir]
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
calculate the electron parallel pressure
"""
function calculate_electron_ppar_no_r!(ppar, density, upar, p, vth, ff, vpa, vperp, z,
                                       me_over_mi)
    @boundscheck z.n == size(ppar, 1) || throw(BoundsError(ppar))

    # Only moment-kinetic electrons supported
    evolve_density = true
    evolve_upar = true
    evolve_p = true

    if evolve_p
        # this is the case where the pressure, parallel flow and density are evolved
        # separately from the shape function; the vpa coordinate
        # is <(v_∥ - upar_s) / vth_s> and so we set upar = 0 in the call to
        # get_vpa2_moment because the mean flow of the shape function is zero
        if vperp.n == 1
            @loop_z iz begin
                ppar[iz] = 3.0 * p[iz]
            end
        elseif ff !== nothing
            @loop_z iz begin
                ppar[iz] = me_over_mi * density[iz] * vth[iz]^2 *
                           integral((vperp,vpa)->vpa^2, @view(ff[:,:,iz]), vperp, vpa)
            end
        else
            # Must be fluid or Boltzmann electrons and the simulation is not 1V, so assume
            # ppar=pperp=p.
            @loop_z iz begin
                ppar[iz] = p[iz]
            end
        end
    elseif evolve_upar
        # this is the case where the parallel flow and density are evolved separately
        # from the normalized pdf; the vpa coordinate is <v_∥ - upar_s) / c_ref> and so we
        # set upar = 0 in the call to get_vpa2_moment because the mean flow of the
        # normalised ff is zero
        @loop_z iz begin
            ppar[iz] = me_over_mi * density[iz]*
                       integral((vperp,vpa)->vpa^2, @view(ff[:,:,iz]), vperp, vpa)
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf; the vpa coordinate is v_\parallel / cref.
        @loop_z iz begin
            ppar[iz] = me_over_mi * density[iz]*get_vpa2_moment(@view(ff[:,:,iz]), vpa, vperp, upar[iz])
                       integral((vperp,vpa)->(vpa-upar[iz])^2, @view(ff[:,:,iz]), vperp, vpa)
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vpa coordinate is v_∥ / cref.
        @loop_z iz begin
            ppar[iz] = me_over_mi *
                       integral((vperp,vpa)->(vpa-upar[iz])^2, @view(ff[:,:,iz]), vperp, vpa)
        end
    end
    return nothing
end

"""
calculate the parallel component of the electron heat flux.
there are currently two supported options for the parallel heat flux:
    Braginskii collisional closure - qpar_e = -3.16*p_e/(m_e*nu_ei)*dT/dz - 0.71*p_e*(upar_i-upar_e)
    collisionless closure - d(qpar_e)/dz = 0 ==> qpar_e = constant
inputs:
    qpar_e = parallel electron heat flux at the previous time level
    qpar_updated = flag indicating whether qpar is updated already
    pdf = electron pdf
    p_e = electron pressure
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
function calculate_electron_qpar!(electron_moments, pdf, p_e, dens_e, upar_e, upar_i,
                                  nu_ei, me_over_mi, electron_model, vperp, vpa)
    @begin_z_region()
    if isa(pdf, electron_pdf_substruct)
        electron_pdf = pdf.norm
    else
        electron_pdf = pdf
    end
    for ir ∈ 1:size(p_e,2)
        if electron_pdf === nothing
            this_pdf = nothing
        else
            this_pdf = @view electron_pdf[:,:,:,ir]
        end
        @views calculate_electron_qpar_no_r!(electron_moments, this_pdf, p_e[:,ir],
                                             dens_e[:,ir], upar_e[:,ir], upar_i[:,ir,:],
                                             nu_ei, me_over_mi, electron_model, vperp,
                                             vpa, ir)
    end
    return nothing
end

function calculate_electron_qpar_no_r!(electron_moments, pdf, p_e, dens_e, upar_e, upar_i,
                                       nu_ei, me_over_mi, electron_model, vperp, vpa, ir)
    # only calculate qpar_e if needs updating
    qpar_updated = electron_moments.qpar_updated
    #if !qpar_updated[]
        qpar_e = @view electron_moments.qpar[:,ir]
        vth_e = @view electron_moments.vth[:,ir]
        if electron_model == braginskii_fluid
            dTe_dz = @view electron_moments.dT_dz[:,ir]
            @begin_z_region()
            # use the classical Braginskii expression for the electron heat flux
            @loop_z iz begin
                qpar_e[iz] = 0.0
                @loop_s is begin
                    qpar_e[iz] -= 0.71 * p_e[iz] * (upar_i[iz,is] - upar_e[iz])
                end
            end
            if nu_ei > 0.0
                @loop_z iz begin
                    qpar_e[iz] -= 3.16 * p_e[iz] / (me_over_mi * nu_ei) * dTe_dz[iz]
                end
            end
        elseif electron_model ∈ (kinetic_electrons,
                                 kinetic_electrons_with_temperature_equation)
            # use the modified electron pdf to calculate the electron heat flux
            calculate_electron_qpar_from_pdf_no_r!(qpar_e, dens_e, vth_e, pdf, vperp, vpa,
                                                   me_over_mi, ir)
        else
            @begin_z_region()
            # qpar_e is not used. Initialize to 0.0 to avoid failure of
            # @debug_track_initialized check
            @loop_z iz begin
                qpar_e[iz] = 0.0
            end
        end
    #end
    # qpar has been updated
    qpar_updated[] = true
    return nothing
end

"""
calculate the parallel component of the electron heat flux,
defined as qpar = 0.5 * dens * vth^3 * int dwpa (pdf * wpa * (wpa^2 + wperp^2))
"""
function calculate_electron_qpar_from_pdf!(qpar, dens, vth, pdf, vperp, vpa, me)
    @begin_r_z_region()
    ivperp = 1
    @loop_r_z ir iz begin
        @views qpar[iz, ir] = 0.5*me*dens[iz,ir]*vth[iz,ir]^3*integral((vperp,vpa)->(vpa*(vpa^2+vperp^2)), pdf[:, :, iz, ir], vperp, vpa)
    end
end

"""
Calculate the parallel component of the electron heat flux, defined as qpar = 0.5 * dens *
vth^3 * int dwpa (pdf * wpa * (wpa^2 + wperp^2)). This version of the function does not
loop over `r`. `pdf` should have no r-dimension, while the moment variables are indexed at
`ir`.
"""
function calculate_electron_qpar_from_pdf_no_r!(qpar, dens, vth, pdf, vperp, vpa, me, ir)
    @begin_z_region()
    ivperp = 1
    @loop_z iz begin
        @views qpar[iz] = 0.5*me*dens[iz]*vth[iz]^3*integral((vperp,vpa)->(vpa*(vpa^2+vperp^2)), pdf[:, :, iz], vperp, vpa)
    end
end

function update_electron_vth_temperature!(moments, p, dens, composition)
    @begin_r_z_region()

    temp = moments.electron.temp
    vth = moments.electron.vth
    @loop_r_z ir iz begin
        this_p = max(p[iz,ir], 0.0)
        temp[iz,ir] = this_p / dens[iz,ir]
        vth[iz,ir] = sqrt(2.0 * temp[iz,ir] / composition.me_over_mi)
    end
    moments.electron.temp_updated[] = true

    return nothing
end

function update_electron_vth_temperature_no_r!(vth, temp, p, dens, composition)
    @begin_z_region()

    me = composition.me_over_mi
    @loop_z iz begin
        this_p = max(p[iz], 0.0)
        temp[iz] = this_p / dens[iz]
        vth[iz] = sqrt(2.0 * temp[iz] / me)
    end

    return nothing
end

"""
    electron_fluid_qpar_boundary_condition!(electron_moments, z)

Impose fluid approximation to electron sheath boundary condition on the parallel heat
flux. See Stangeby textbook, equations (2.89) and (2.90).
"""
function electron_fluid_qpar_boundary_condition!(p, upar, dens, electron_moments, z)
    if z.bc == "periodic"
        # Nothing to do as z-derivative used to calculate qpar already imposed
        # periodicity.
        return nothing
    end

    @begin_r_region()

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
            this_p = p[iz,ir]
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
            conductive_heat_flux = total_heat_flux - 2.5 * this_p * this_upar

            electron_moments.qpar[iz,ir] = conductive_heat_flux
        end
    end

    return nothing
end

end
