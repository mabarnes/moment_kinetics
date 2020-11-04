module velocity_moments

using array_allocation: allocate_float

export integrate_over_vspace!
export update_moments!
export update_density!

mutable struct moments
    # this is the particle density
    dens::Array{Float64,1}
    # flag that keeps track of if the density needs updating before use
    dens_updated::Bool
    # this is the parallel pressure
    ppar::Array{Float64,1}
    # flag that keeps track of whether or not ppar needs updating before use
    ppar_updated::Bool
end
# create and initialise arrays for the density and parallel pressure,
# as well as a scratch array used for intermediate calculations needed
# to later update these moments
function setup_moments(ff, vpa, nz)
    # allocate array used for the particle density
    density = allocate_float(nz)
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_float(nz)
    # initialise the density and parallel_pressure arrays
    update_density!(density, view(vpa.scratch,:,1), ff, vpa, nz)
    update_ppar!(parallel_pressure, view(vpa.scratch,:,1), ff, vpa, nz)
    # return a struct containing arrays/Bools needed to update moments
    return moments(density, true, parallel_pressure, true)
end
# calculate the updated density (dens) and parallel pressure (ppar)
function update_moments!(moments, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    update_density!(moments.dens, view(vpa.scratch,:,1), ff, vpa, nz)
    update_ppar!(moments.ppar, view(vpa.scratch,:,1), ff, vpa, nz)
    return nothing
end
# calculate the updated density (dens)
function update_density!(dens, scratch, ff, vpa, nz)
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:]
        dens[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# calculate the updated parallel pressure (ppar)
function update_ppar!(ppar, scratch, ff, vpa, nz)
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:] * vpa.grid^2
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
    @inbounds for i ∈ 1:nvpa
        integral += integrand[i]*vpa_wgts[i]
    end
    return integral
end

end
