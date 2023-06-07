module charge_conservation

export calculate_electron_upar_from_charge_conservation!

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
function calculate_electron_upar_from_charge_conservation!(upar_e, dens_e, upar_i, dens_i)
    nr = size(upar_e, 2)
    nz = size(upar_e, 1)
    ns = size(upar_i, 3)
    for ir in 1:nr
        boundary_flux = dens_e[1,ir] * upar_e[1,ir]
        for iz in 2:nz-1
            # calculate the boundary value for the particle flux, and initialise 
            # the electron particle flux to it
            upar_e[iz,ir] = boundary_flux
            # add the contributions to the electron particle flux from the various ion species
            # particle fluxes
            for is in 1:ns
                upar_e[iz,ir] += dens_i[iz,ir,is] * upar_i[iz,ir,is] - dens_i[1,ir,is] * upar_i[1,ir,is]
            end
            # convert from parallel particle flux to parallel particle density
            upar_e[iz,ir] /= dens_e[iz,ir]
        end
    end
end

end