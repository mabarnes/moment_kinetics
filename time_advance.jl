module time_advance

export rk_update_f!
export setup_time_advance!

using chebyshev: setup_chebyshev_pseudospectral
using chebyshev: chebyshev_derivative!
using velocity_moments: setup_moments
using initial_conditions: enforce_z_boundary_condition!
using initial_conditions: enforce_vpa_boundary_condition!
using advection: setup_source, update_boundary_indices!
using z_advection: update_speed_z!
using vpa_advection: update_speed_vpa!
using em_fields: setup_em_fields, update_phi!
using semi_lagrange: setup_semi_lagrange

# create arrays and do other work needed to setup
# the main time advance loop.
# this includes creating and populating structs
# for Chebyshev transforms, velocity space moments,
# EM fields, semi-Lagrange treatment, and source terms
function setup_time_advance!(ff, z, vpa, composition, drive_input)
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    # create structure z_source whose members are the arrays needed to compute
    # the source(s) appearing in the split part of the GK equation dealing
    # with advection in z
    z_source = setup_source(z, vpa, n_species)
    # initialise the z advection speed
    for is ∈ 1:n_species
        update_speed_z!(view(z_source,:,is), vpa, z, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(view(z_source,:,is))
        # enforce prescribed boundary condition in z on the distribution function f
        @views enforce_z_boundary_condition!(ff[:,:,is], z.bc, vpa, z_source[:,is])
    end
    if z.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        z_spectral = setup_chebyshev_pseudospectral(z)
        # obtain the local derivatives of the uniform z-grid with respect to the used z-grid
        chebyshev_derivative!(z.duniform_dgrid, z.uniform_grid, z_spectral, z)
    else
        # create dummy Bool variable to return in place of the above struct
        z_spectral = false
        z.duniform_dgrid .= 1.0
    end
    if vpa.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        vpa_spectral = setup_chebyshev_pseudospectral(vpa)
        # obtain the local derivatives of the uniform vpa-grid with respect to the used vpa-grid
        chebyshev_derivative!(vpa.duniform_dgrid, vpa.uniform_grid, vpa_spectral, vpa)
    else
        # create dummy Bool variable to return in place of the above struct
        vpa_spectral = false
        vpa.duniform_dgrid .= 1.0
    end
    # pass ff and allocate/initialize the velocity space moments needed for advancing
    # the kinetic equation coupled to fluid equations
    # the resulting moments are returned in the structure "moments"
    moments = setup_moments(ff, vpa, z.n)
    # pass a subarray of ff (its value at the previous time level)
    # and create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency)
    # initialize the electrostatic potential
    update_phi!(fields, moments, ff, vpa, z.n, composition, 0.0)
    # save the initial phi(z) for possible use later (e.g., if forcing phi)
    fields.phi0 .= fields.phi
    # create structure vpa_source whose members are the arrays needed to compute
    # the source(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_source = setup_source(vpa, z, n_ion_species)
    # initialise the vpa advection speed
    update_speed_vpa!(vpa_source, fields, moments, ff, vpa, z, composition, 0.0, z_spectral)
    for is ∈ 1:n_ion_species
        # initialise the upwind/downwind boundary indices in vpa
        update_boundary_indices!(view(vpa_source,:,is))
        # enforce prescribed boundary condition in vpa on the distribution function f
        @views enforce_vpa_boundary_condition!(ff[:,:,is], vpa.bc, vpa_source[:,is])
    end
    # create an array of structures containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n, vpa.n)
    vpa_SL = setup_semi_lagrange(vpa.n, z.n)
    return z_spectral, vpa_spectral, moments, fields, z_source, vpa_source, z_SL, vpa_SL
end

function rk_update_f!(ff, ff_rk, nz, nvpa, n_rk_stages)
    @boundscheck nz == size(ff_rk,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff_rk,2) || throw(BoundsError(ff_rk))
    @boundscheck n_rk_stages+1 == size(ff_rk,3) || throw(BoundsError(ff_rk))
    @boundscheck nz == size(ff,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff,2) || throw(BoundsError(ff_rk))
    if n_rk_stages == 1
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = ff_rk[iz,ivpa,2]
                end
            end
        end
    elseif n_rk_stages == 2
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = 0.5*(ff_rk[iz,ivpa,2] + ff_rk[iz,ivpa,3])
                end
            end
        end
    elseif n_rk_stages == 3
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = (2.0*(ff_rk[iz,ivpa,3] + ff_rk[iz,ivpa,4])-ff_rk[iz,ivpa,1])/3.0
                end
            end
        end
    end
end

end
