module energy_equation

export energy_equation!

using calculus: derivative!

function energy_equation!(ppar, fvec, moments, CX_frequency, z, dt, spectral, composition)
    for is ∈ 1:composition.n_species
        @views energy_equation_noCX!(ppar[:,is], fvec.upar[:,is], fvec.ppar[:,is],
                                     moments.qpar[:,is], dt, z, spectral)
    end
    #NB: need to add in contribution due to charge exchange
end
function energy_equation_noCX!(ppar_out, upar, ppar, qpar, dt, z, spectral)
    # calculate dppar/dz and store in z.scratch
    derivative!(z.scratch, ppar, z, spectral)
    # update ppar to account for contribution from parallel pressure gradient
    @. ppar_out -= dt*upar*z.scratch
    # calculate dqpar/dz and store in z.scratch
    derivative!(z.scratch, qpar, z, spectral)
    # update ppar to account for contribution from parallel heat flux gradient
    @. ppar_out -= dt*z.scratch
    # calculate dupar/dz and store in z.scratch
    derivative!(z.scratch, upar, z, spectral)
    # update ppar to account for contribution from parallel flow gradient
    @. ppar_out -= 3.0*dt*ppar*z.scratch
end

end
