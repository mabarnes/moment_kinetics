module velocity_moments

using array_allocation: allocate_float

export integrate_over_vspace!
export update_moments!

struct moment
    # this is the particle density
    dens::Array{Float64,1}
    # this is the parallel pressure
    ppar::Array{Float64,1}
end

function setup_moments(ff, vpa, nz)
    # allocate array used for the particle density
    density = allocate_float(nz)
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_float(nz)
    # NB: there will be a better way to do this to avoid memory usage
    tmp = allocate_float(vpa.n)
    # no need to check bounds, as array as declared above has correct size
    @inbounds for j ∈ 1:nz
        @views @. tmp = ff[j,:]
        density[j] = integrate_over_vspace(tmp, vpa.wgts)
        @views @. tmp = ff[j,:] * vpa.grid^2
        parallel_pressure[j] = integrate_over_vspace(tmp, vpa.wgts)
    end

    return moment(density, parallel_pressure)
end

function update_moments!(moments, ff, vpa, nz)
    # NB: there will be a better way to do this to avoid memory usage
    tmp = allocate_float(vpa.n)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    update_dens!(moments.dens, tmp, ff, vpa, nz)
    update_ppar!(moments.ppar, tmp, ff, vpa, nz)
#=
    @inbounds for j ∈ 1:nz
        @views @. tmp = ff[j,:]
        moments.dens[j] = integrate_over_vspace(tmp, vpa.wgts)
        @views @. tmp = ff[j,:] * vpa.grid^2
        moments.ppar[j] = integrate_over_vspace(tmp, vpa.wgts)
    end
=#
    return nothing
end

function update_dens(dens, tmp, ff, vpa, nz)
    @inbounds for j ∈ 1:nz
        @views @. tmp = ff[j,:]
        dens[j] = integrate_over_vspace(tmp, vpa.wgts)
    end
    return nothing
end

function update_ppar(ppar, tmp, ff, vpa, nz)
    @inbounds for j ∈ 1:nz
        @views @. tmp = ff[j,:] * vpa.grid^2
        ppar[j] = integrate_over_vspace(tmp, vpa.wgts)
    end
    return nothing
end

function integrate_over_vspace(integrand, vpa_wgts)
    # nvpa is the number of v_parallel grid points
    nvpa = length(vpa_wgts)
    # initialize 'integral' to zero before sum
    integral = 0
    @boundscheck nvpa == length(integrand) || throw(BoundsError(integrand))
    @inbounds for i ∈ 1:nvpa
        integral += integrand[i]*vpa_wgts[i]
    end
    return integral
end

end
