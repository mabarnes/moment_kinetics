module electron_fluid_equations

export calculate_electron_density!
export calculate_electron_upar_from_charge_conservation!
export electron_energy_equation!
export calculate_electron_qpar!
export calculate_electron_parallel_friction_force!

using ..looping
using ..input_structs: boltzmann_electron_response_with_simple_sheath
using ..input_structs: braginskii_fluid

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
    if !updated
        dens_e .= 0.0
        # enforce quasineutrality
        @loop_s_r_z is ir iz begin
            dens_e[iz,ir] += dens_i[iz,ir,is]
        end
    end
    # set flag indicating that the electron density has been updated
    updated = true
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
function calculate_electron_upar_from_charge_conservation!(upar_e, updated, dens_e, upar_i, dens_i, electron_model)
    # only calculate the electron parallel flow if it is not already updated
    if !updated
        # get the number of zed grid points, nz
        nz = size(upar_e,1)
        # initialise the electron parallel flow density to zero
        upar_e .= 0.0
        # if using a simple logical sheath model, then the electron parallel current at the boundaries in zed
        # is equal and opposite to the ion parallel current
        if electron_model in (boltzmann_electron_response_with_simple_sheath, braginskii_fluid)
            # loop over ion species, adding each species contribution to the
            # ion parallel particle flux at the boundaries in zed
            @loop_s_r is ir begin
                # electron_upar at this intermediate stage is actually
                # the electron particle flux
                upar_e[1,ir] += dens_i[1,ir,is] * upar_i[1,ir,is]
                upar_e[end,ir] += dens_i[end,ir,is] * upar_i[end,ir,is]
            end
            @loop_r ir begin
                # fix the boundary flux
                boundary_flux = upar_e[1,ir]
                for iz in 2:nz-1
                    # initialise the electron particle flux to its value at the boundary in zed
                    upar_e[iz,ir] = boundary_flux
                    # add the contributions to the electron particle flux from the various ion species
                    # particle fluxes
                    @loop_s is begin
                        upar_e[iz,ir] += dens_i[iz,ir,is] * upar_i[iz,ir,is] - dens_i[1,ir,is] * upar_i[1,ir,is]
                    end
                    # convert from parallel particle flux to parallel particle density
                    upar_e[iz,ir] /= dens_e[iz,ir]
                end
                # convert from parallel particle flux to parallel particle density for boundary points
                upar_e[1,ir] /= dens_e[1,ir]
                upar_e[end,ir] /= dens_e[end,ir]
            end
        end
    end
    updated = true
    return nothing
end

"""
use the electron energy equation to evolve the electron temperature via
an explicit time advance.
NB: so far, this is only set up for 1D problem, where we can assume
an isotropic distribution in f_e so that p_e = n_e T_e = ppar_e
"""
function electron_energy_equation!(ppar, fvec, moments, collisions, dt, spectral, composition,
                                   num_diss_params, dens)
    begin_r_z_region()
    # define some abbreviated variables for convenient use in rest of function
    me_over_mi = composition.me_over_mi
    nu_ei = collisions.nu_ei
    # calculate contribution to rhs of energy equation (formulated in terms of pressure)
    # arising from derivatives of ppar, qpar and upar
    @loop_r_z ir iz begin
        ppar[iz,ir] -= dt*(fvec.electron_upar[iz,ir]*moments.electron.dppar_dz_upwind[iz,ir]
                    + (2/3)*moments.electron.dqpar_dz[iz,ir]
                    + (5/3)*fvec.electron_ppar[iz,ir]*moments.electron.dupar_dz[iz,ir])
    end
    # compute the contribution to the rhs of the energy equation
    # arising from artificial diffusion
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_r_z ir iz begin
            ppar[iz,ir] += dt*diffusion_coefficient*moments.electron.d2ppar_dz2[iz,ir]
        end
    end
    # compute the contribution to the rhs of the energy equation
    # arising from electron-ion collisions
    if nu_ei > 0.0
        @loop_r_z ir iz begin
            ppar[iz,ir] += dt*(2 * me_over_mi * nu_ei * (fvec.ppar[iz,ir]*fvec.electron_density[iz,ir]/fvec.density[iz,ir]
                                                     - fvec.electron_ppar[iz,ir])
                        + (2/3) * moments.electron.parallel_friction[iz,ir] * (fvec.upar[iz,ir]-fvec.electron_upar[iz,ir]))
        end
    end
    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        if abs(collisions.charge_exchange_electron) > 0.0
            @loop_s_r_z is ir iz begin
                ppar[iz,ir] +=
                    dt * me_over_mi * collisions.charge_exchange_electron * (
                    2*(fvec.electron_density[iz,ir]*fvec.pz_neutral[iz,ir,is] -
                    fvec.density_neutral[iz,ir,is]*fvec.electron_ppar[iz,ir]) +
                    (2/3)*fvec.electron_density[iz,ir]*fvec.density_neutral[iz,ir,is] *
                    (fvec.uz_neutral[iz,ir,is] - fvec.electron_upar[iz,ir])^2)
            end
        end
        if abs(collisions.ionization_electron) > 0.0
            @loop_s_r_z is ir iz begin
                ppar[iz,ir] -=
                    dt * collisions.ionization_electron * fvec.density_neutral[iz,ir,is] * (
                    fvec.electron_ppar[iz,ir] +
                    (2/3)*fvec.electron_density[iz,ir] * collisions.ionization_energy)
            end
        end
    end
    # calculate the external electron heat source, if any
    calculate_electron_heat_source!(moments.electron.heat_source, fvec.electron_ppar, moments.electron.dupar_dz,
                                    fvec.density_neutral, collisions.ionization,
                                    fvec.electron_upar, moments.electron.dppar_dz_upwind, fvec.electron_density,
                                    dens, dt)
    # add the contribution from the electron heat source                                
    @loop_r_z ir iz begin
        ppar[iz,ir] += dt * moments.electron.heat_source[iz,ir]
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
        @loop_r_z ir iz begin
            Epar[iz,ir] += 2 * me_over_mi * dens_n[iz,ir] * charge_exchange * (upar_n[iz,ir] - upar_e[iz,ir]) 
        end
    end
    return nothing
end

"""
"""
function calculate_electron_parallel_friction_force!(friction, dens_e, upar_e, upar_i, dTe_dz,
                                                     me_over_mi, nu_ei, electron_model)
    if electron_model == braginskii_fluid
        @loop_r_z ir iz begin
            friction[iz,ir] = -0.71 * dens_e[iz,ir] * dTe_dz[iz,ir]
        end
        @loop_r_z ir iz begin
            friction[iz,ir] += 0.51 * dens_e[iz,ir] * me_over_mi * nu_ei * (upar_i[iz,ir] - upar_e[iz,ir])
        end
    else
        @. friction = 0.0
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
    ppar_e = electron parallel pressure
    upar_e = electron parallel flow
    dTe_dz = zed derivative of electron temperature
    upar_i = ion parallel flow
    nu_ei = electron-ion collision frequency
    me_over_mi = electron-to-ion mass ratio
    electron_model = choice of model for electron physics 
output:
    qpar_e = updated parallel electron heat flux
    qpar_updated = flag indicating that the parallel electron heat flux is updated
"""
function calculate_electron_qpar!(qpar_e, qpar_updated, ppar_e, upar_e, dTe_dz, upar_i, nu_ei, me_over_mi, electron_model)
    # only calculate qpar_e if needs updating
    if qpar_updated == false
        if electron_model == braginskii_fluid
            # use the classical Braginskii expression for the electron heat flux
            @loop_r_z ir iz begin
                qpar_e[iz,ir] = - 0.71 * ppar_e[iz,ir] * (upar_i[iz,ir] - upar_e[iz,ir])
            end
            if nu_ei > 0.0
                @loop_r_z ir iz begin
                    qpar_e[iz,ir] -= 3.16 * ppar_e[iz,ir] / (me_over_mi * nu_ei) * dTe_dz[iz,ir] 
                end
            end
        else
            # if not using Braginskii fluid model, then assume for now
            # that we are in the collisionless limit, for which d/dz(qpar_e) = 0.
            # take qpar_e = 0
            @. qpar_e = 0.0
        end
    end
    # qpar has been updated
    qpar_updated = true
    return nothing
end

function calculate_electron_heat_source!(heat_source, ppar_e, dupar_dz, dens_n, ionization,
                                         upar_e, dppar_dz_upwind, dens_e, dens_new, dt)
    @loop_r_z ir iz begin
        heat_source[iz,ir] = (2/3) * ppar_e[iz,ir] * dupar_dz[iz,ir]
    end
    n_neutral_species = size(dens_n, 3)
    if n_neutral_species > 0 && ionization > 0.0
       @loop_s_r_z is ir iz begin
           heat_source[iz,ir] += ppar_e[iz,ir] * dens_n[iz,ir,is] * ionization
       end
    end
end

end