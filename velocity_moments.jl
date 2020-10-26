module velocity_moments

using array_allocation: allocate_float

export integrate_over_vspace!
export update_moments!
export update_density!

struct moments
    # this is the particle density
    dens::Array{Float64,1}
    # flag that keeps track of if the density needs updating before use
    dens_updated::Bool
    # this is the parallel pressure
    ppar::Array{Float64,1}
    # flag that keeps track of whether or not ppar needs updating before use
    ppar_updated::Bool
    # this is a scratch array that can be used for intermediate calculations
    # involving the moments; useful to avoid unneccesary allocation/garbage collection
    scratch::Array{Float64,1}
end

function setup_moments(ff, vpa, nz)
    # allocate array used for the particle density
    density = allocate_float(nz)
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_float(nz)
    # allocate arrary to be used for temporary storage
    scratch = allocate_float(vpa.n)
    # no need to check bounds, as array as declared above has correct size
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:]
        density[iz] = integrate_over_vspace(scratch, vpa.wgts)
        @views @. scratch = ff[iz,:] * vpa.grid^2
        parallel_pressure[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end

    return moments(density, true, parallel_pressure, true, scratch)
end

function update_moments!(moments, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    update_density!(moments.dens, moments.scratch, ff, vpa, nz)
    update_ppar!(moments.ppar, moments.scratch, ff, vpa, nz)
    return nothing
end

function update_density!(dens, scratch, ff, vpa, nz)
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:]
        dens[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end


function update_ppar!(ppar, scratch, ff, vpa, nz)
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:] * vpa.grid^2
        ppar[iz] = integrate_over_vspace(scratch, vpa.wgts)
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
