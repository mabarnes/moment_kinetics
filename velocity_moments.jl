module velocity_moments

export integrate_over_vspace!
export update_moments!
export update_density!
export reset_moments_status!

using type_definitions: mk_float
using array_allocation: allocate_float, allocate_bool

mutable struct moments
    # this is the particle density
    dens::Array{mk_float,2}
    # flag that keeps track of if the density needs updating before use
    dens_updated::Array{Bool,1}
    # this is the parallel flow
    upar::Array{mk_float,2}
    # flag that keeps track of whether or not upar needs updating before use
    upar_updated::Array{Bool,1}
    # this is the parallel pressure
    ppar::Array{mk_float,2}
    # flag that keeps track of whether or not ppar needs updating before use
    ppar_updated::Array{Bool,1}
end
# create and initialise arrays for the density and parallel pressure,
# as well as a scratch array used for intermediate calculations needed
# to later update these moments
function setup_moments(ff, vpa, nz)
    n_species = size(ff,3)
    # allocate array used for the particle density
    density = allocate_float(nz, n_species)
    # allocate array of Bools that indicate if the density is updated for each species
    density_updated = allocate_bool(n_species)
    # allocate array used for the parallel flow
    parallel_flow = allocate_float(nz, n_species)
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    parallel_flow_updated = allocate_bool(n_species)
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_float(nz, n_species)
    # allocate array of Bools that indicate if the parallel pressure is updated for each species
    parallel_pressure_updated = allocate_bool(n_species)
    # initialise the density and parallel_pressure arrays
    for is ∈ 1:n_species
        @views update_density!(density[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
        density_updated[is] = true
        @views update_upar!(parallel_flow[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
        parallel_flow_updated[is] = true
        @views update_ppar!(parallel_pressure[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
        parallel_pressure_updated[is] = true
    end
    # return struct containing arrays needed to update moments
    return moments(density, density_updated, parallel_flow, parallel_flow_updated,
        parallel_pressure, parallel_pressure_updated)
end
# calculate the updated density (dens) and parallel pressure (ppar) for all species
function update_moments!(moments, ff, vpa, nz)
    n_species = size(ff,3)
    @boundscheck n_species == size(moments.dens,2) || throw(BoundsError(moments))
    for is ∈ 1:n_species
        if moments.dens_updated[is] == false
            @views update_density!(moments.dens[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
            moments.dens_updated[is] = true
        end
        if moments.upar_updated[is] == false
            @views update_upar!(moments.upar[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
            moments.upar_updated[is] = true
        end
        if moments.ppar_updated[is] == false
            @views update_ppar!(moments.ppar[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
            moments.ppar_updated[is] = true
        end
    end
    return nothing
end
# calculate the updated density (dens)
function update_density!(dens, scratch, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck nz == length(dens) || throw(BoundsError(dens))
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:]
        dens[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# calculate the updated parallel flow (upar)
function update_upar!(upar, scratch, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck nz == length(upar) || throw(BoundsError(upar))
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:] * vpa.grid
        upar[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# calculate the updated parallel pressure (ppar)
function update_ppar!(ppar, scratch, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck nz == length(ppar) || throw(BoundsError(ppar))
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:] * 2.0*vpa.grid^2/3.0
        ppar[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# computes the integral over vpa of the integrand, using the input vpa_wgts
function integrate_over_vspace(integrand, vpa_wgts)
    # nvpa is the number of v_parallel grid points
    nvpa = length(vpa_wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck nvpa == length(integrand) || throw(BoundsError(integrand))
    @boundscheck nvpa == length(vpa_wgts) || throw(BoundsError(vpa_wgts))
    @inbounds for i ∈ 1:nvpa
        integral += integrand[i]*vpa_wgts[i]
    end
    integral /= sqrt(pi)
    return integral
end
function reset_moments_status!(moments)
    moments.dens_updated .= false
    moments.upar_updated .= false
    moments.ppar_updated .= false
end

end
