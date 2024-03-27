module electron_fluid_equations

export calculate_electron_density!
export calculate_electron_upar_from_charge_conservation!
export electron_energy_equation!
export calculate_electron_qpar!
export calculate_electron_parallel_friction_force!
export calculate_electron_qpar_from_pdf!
export update_electron_vth_temperature!

using ..communication
using ..looping
using ..input_structs: boltzmann_electron_response_with_simple_sheath
using ..input_structs: braginskii_fluid, kinetic_electrons
using ..moment_kinetics_structs: pdf_substruct
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
        if electron_model ∈ (boltzmann_electron_response_with_simple_sheath, braginskii_fluid, kinetic_electrons)
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
        end
        updated[] = true
    end
    return nothing
end

"""
use the electron energy equation to evolve the electron temperature via
an explicit time advance.
NB: so far, this is only set up for 1D problem, where we can assume
an isotropic distribution in f_e so that p_e = n_e T_e = ppar_e
"""
function electron_energy_equation!(ppar, fvec, moments, collisions, dt, composition,
                                   num_diss_params, z)
    begin_r_z_region()
    # define some abbreviated variables for convenient use in rest of function
    me_over_mi = composition.me_over_mi
    nu_ei = collisions.nu_ei
    # calculate contribution to rhs of energy equation (formulated in terms of pressure)
    # arising from derivatives of ppar, qpar and upar
    @loop_r_z ir iz begin
        ppar[iz,ir] -= dt*(fvec.electron_upar[iz,ir]*moments.dppar_dz[iz,ir]
                    + moments.dqpar_dz[iz,ir]
                    + 3*fvec.electron_ppar[iz,ir]*moments.dupar_dz[iz,ir])
    end
    # @loop_r_z ir iz begin
    #     ppar[iz,ir] -= dt*(fvec.electron_upar[iz,ir]*moments.dppar_dz[iz,ir]
    #                 + (2/3)*moments.dqpar_dz[iz,ir]
    #                 + (5/3)*fvec.electron_ppar[iz,ir]*moments.dupar_dz[iz,ir])
    # end
    # compute the contribution to the rhs of the energy equation
    # arising from artificial diffusion
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_r_z ir iz begin
            ppar[iz,ir] += dt*diffusion_coefficient*moments.d2ppar_dz2[iz,ir]
        end
    end
    # compute the contribution to the rhs of the energy equation
    # arising from electron-ion collisions
    if nu_ei > 0.0
        @loop_s_r_z is ir iz begin
            ppar[iz,ir] += dt * (2 * me_over_mi * nu_ei * (fvec.ppar[iz,ir,is] - fvec.electron_ppar[iz,ir]))
            ppar[iz,ir] += dt * ((2/3) * moments.parallel_friction[iz,ir]
                                * (fvec.upar[iz,ir,is]-fvec.electron_upar[iz,ir]))
        end
    end
    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        if abs(collisions.charge_exchange_electron) > 0.0
            @loop_sn_r_z isn ir iz begin
                ppar[iz,ir] +=
                    dt * me_over_mi * collisions.charge_exchange_electron * (
                    2*(fvec.electron_density[iz,ir]*fvec.pz_neutral[iz,ir,isn] -
                    fvec.density_neutral[iz,ir,isn]*fvec.electron_ppar[iz,ir]) +
                    (2/3)*fvec.electron_density[iz,ir]*fvec.density_neutral[iz,ir,isn] *
                    (fvec.uz_neutral[iz,ir,isn] - fvec.electron_upar[iz,ir])^2)
            end
        end
        if abs(collisions.ionization_electron) > 0.0
            # @loop_s_r_z is ir iz begin
            #     ppar[iz,ir] +=
            #         dt * collisions.ionization_electron * fvec.density_neutral[iz,ir,is] * (
            #         fvec.electron_ppar[iz,ir] -
            #         (2/3)*fvec.electron_density[iz,ir] * collisions.ionization_energy)
            # end
            @loop_sn_r_z isn ir iz begin
                ppar[iz,ir] +=
                    dt * collisions.ionization_electron * fvec.density_neutral[iz,ir,isn] * (
                    fvec.electron_ppar[iz,ir] -
                    fvec.electron_density[iz,ir] * collisions.ionization_energy)
            end
        end
    end
    # calculate the external electron heat source, if any
    calculate_electron_heat_source!(moments.heat_source, fvec.electron_ppar, moments.dupar_dz,
                                    fvec.density_neutral, collisions.ionization, collisions.ionization_energy,
                                    fvec.electron_density, fvec.ppar, collisions.nu_ei, composition.me_over_mi,
                                    composition.T_wall, z)
    # add the contribution from the electron heat source                                
    begin_r_z_region()
    @loop_r_z ir iz begin
        ppar[iz,ir] += dt * moments.heat_source[iz,ir]
    end
    return nothing
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
function calculate_electron_qpar!(qpar_e, qpar_updated, pdf, ppar_e, upar_e, vth_e, dTe_dz, upar_i, 
                                  nu_ei, me_over_mi, electron_model, vpa)
    # only calculate qpar_e if needs updating
    if !qpar_updated[]
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
        elseif electron_model == kinetic_electrons
            # use the modified electron pdf to calculate the electron heat flux
            if isa(pdf, pdf_substruct)
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
    end
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

function update_electron_vth_temperature!(moments, ppar, dens)
    begin_r_z_region()

    temp = moments.electron.temp
    vth = moments.electron.vth
    @loop_r_z ir iz begin
        temp[iz,ir] = 2 * ppar[iz,ir] / dens[iz,ir]
        vth[iz,ir] = sqrt(temp[iz,ir])
    end
    moments.electron.temp_updated[] = true

    return nothing
end

end
