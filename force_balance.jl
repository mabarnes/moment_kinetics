module force_balance

export force_balance!

using calculus: derivative!

function force_balance!(pflx, fvec, fields, n_ion_species,
    n_neutral_species, CX_frequency, z, vpa, dt, spectral)
    # use the force balance equation d(nu)/dt + d(ppar + n*upar*upar)/dz =
    # -(dens/2)*dphi/dz + R*dens_i*dens_n*(uoar_n-upar_i)
    # to update the parallel particle flux dens*upar for each species
    n_species = size(fvec.upar,2)
    for is ∈ 1:n_ion_species
        @views force_balance_charged_species!(pflx[:,is], fields.phi,
            fvec.density[:,is], fvec.upar[:,is], fvec.ppar[:,is], z, dt, spectral)
    end
end

# use the force balance equation d(mnu)/dt + d(ppar + mnu * u)/dz = -n*Epar + ...
# to update mnu
function force_balance_charged_species!(pflx, phi, dens, upar, ppar, z, dt, spectral)
    # calculate the parallel flux of parallel momentum densitg at the previous time level/RK stage
    @. z.scratch = ppar + dens*upar^2
    # calculate d(nu)/dz, averaging the derivative values at element boundaries
    derivative!(z.scratch, z.scratch, z, spectral)
    # update the parallel momentum density to account for the parallel flux of parallel momentum
    @. pflx = dens*upar - dt*z.scratch
    # calculate the parallel electric field
    derivative!(z.scratch, -phi, z, spectral)
    # update the parallel momentum density to account for the force from the parallel electric field
    @. pflx += 0.5*dt*z.scratch*dens
end

end
