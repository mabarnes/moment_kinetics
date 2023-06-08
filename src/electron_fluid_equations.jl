module electron_fluid_equations

export calculate_electron_density!
export calculate_electron_upar_from_charge_conservation!
export calculate_electron_qpar!

using ..looping
using ..input_structs: boltzmann_electron_response_with_simple_sheath

"""
use quasineutrality to obtain the electron density from the initial
densities of the various ion species:
    sum_i dens_i = dens_e
inputs:
    dens_e - electron density at previous time level
    dens_i - updated ion density
output:
    dens_e - updated electron density
"""
function calculate_electron_density!(dens_e, dens_i)
    dens_e .= 0.0
    @loop_s_r_z is ir iz begin
        dens_e[iz,ir] += dens_i[iz,ir,is]
    end
    return nothing
end

"""
use charge conservation equation to solve for the electron parallel flow density:
    d/dz(sum_i n_i upar_i - n_e upar_e) = 0
    ==> [sum_i n_i upar_i](z) - [sum_i n_i upar_i](zbound) = [n_e upar_e](z) - [n_e upar_e](zbound)
inputs: 
    upar_e - should contain updated electron parallel flow density at boundaries in zed
    dens_e - electron particle density
    upar_i - ion parallel flow density
    dens_i - ion particle density
output:
    upar_e - contains the updated electron parallel flow density
"""
function calculate_electron_upar_from_charge_conservation!(upar_e, dens_e, upar_i, dens_i, electron_model)
    # get the number of zed grid points, nz
    nz = size(upar_e,1)
    # initialise the electron parallel flow density to zero
    upar_e .= 0.0
    # if using a simple logical sheath model, then the electron parallel current at the boundaries in zed
    # is equal and opposite to the ion parallel current
    if electron_model == boltzmann_electron_response_with_simple_sheath
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
    return nothing
end

"""
calculate the parallel component of the electron heat flux.
there are currently two supported options for the parallel heat flux:
    Braginskii collisional closure - qpar_e = -3.16*ppar_e/(m_e*nu_ei)*dT/dz - 0.71*ppar_e*(upar_i-upar_e)
    collisionless closure - d(qpar_e)/dz = 0 ==> qpar_e = 0
inputs:
    qpar_e - parallel electron heat flux at the previous time level
output:
    qpar_e - updated parallel electron heat flux
"""
function calculate_electron_qpar!(qpar_e, electron_model)
    if electron_model == "braginskii_fluid"
        # NEED TO FIX!
        qpar_e .= 0.0 
    else
        # if not using Braginskii fluid model, then assume for now
        # that we are in the collisionless limit, for which d/dz(qpar_e) = 0.
        # take qpar_e = 0
        qpar_e .= 0.0
    end
    return nothing
end

end