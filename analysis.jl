module analysis

export analyze_fields_data
export analyze_moments_data
export analyze_pdf_data

using array_allocation: allocate_float

function analyze_fields_data(phi, ntime, nz, z_wgts, Lz)
    print("Analyzing fields data...")
    # compute the z integration weights needed to do field line averages
    #z_wgts = composite_simpson_weights(z)
    # Lz = z box length
    #Lz = z[end]-z[1]
    phi_fldline_avg = allocate_float(ntime)
    for i ∈ 1:ntime
        phi_fldline_avg[i] = field_line_average(view(phi,:,i), z_wgts, Lz)
    end
    # delta_phi = phi - <phi> is the fluctuating phi
    delta_phi = allocate_float(nz,ntime)
    for iz ∈ 1:nz
        delta_phi[iz,:] .= phi[iz,:] - phi_fldline_avg
    end
    println("done.")
    return phi_fldline_avg, delta_phi
end

function analyze_moments_data(density, parallel_flow, parallel_pressure, ntime, n_species, nz, z_wgts, Lz)
    print("Analyzing velocity moments data...")
    density_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            density_fldline_avg[is,i] = field_line_average(view(density,:,is,i), z_wgts, Lz)
        end
    end
    upar_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            upar_fldline_avg[is,i] = field_line_average(view(parallel_flow,:,is,i), z_wgts, Lz)
        end
    end
    ppar_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            ppar_fldline_avg[is,i] = field_line_average(view(parallel_pressure,:,is,i), z_wgts, Lz)
        end
    end
    # delta_density = n_s - <n_s> is the fluctuating density
    delta_density = allocate_float(nz,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:nz
            @. delta_density[iz,is,:] = density[iz,is,:] - density_fldline_avg[is,:]
        end
    end
    # delta_upar = upar_s - <upar_s> is the fluctuating parallel flow
    delta_upar = allocate_float(nz,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:nz
            @. delta_upar[iz,is,:] = parallel_flow[iz,is,:] - upar_fldline_avg[is,:]
        end
    end
    # delta_ppar = ppar_s - <ppar_s> is the fluctuating parallel pressure
    delta_ppar = allocate_float(nz,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:nz
            @. delta_ppar[iz,is,:] = parallel_pressure[iz,is,:] - ppar_fldline_avg[is,:]
        end
    end
    println("done.")
    return density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, delta_density, delta_upar, delta_ppar
end

function analyze_pdf_data(ff, nz, nvpa, n_species, ntime, z_wgts, Lz, vpa_wgts)
    print("Analyzing distribution function data...")
    f_fldline_avg = allocate_float(nvpa,n_species,ntime)
    for i ∈ 1:ntime
        for is ∈ 1:n_species
            for ivpa ∈ 1:nvpa
                f_fldline_avg[ivpa,is,i] = field_line_average(view(ff,:,ivpa,is,i), z_wgts, Lz)
            end
        end
    end
    # delta_f = f - <f> is the fluctuating distribution function
    delta_f = allocate_float(nz,nvpa,n_species,ntime)
    for iz ∈ 1:nz
        @. delta_f[iz,:,:,:] = ff[iz,:,:,:] - f_fldline_avg
    end
    dens_moment = allocate_float(nz,n_species,ntime)
    for i ∈ 1:ntime
        for is ∈ 1:n_species
            for iz ∈ 1:nz
                @views dens_moment[iz,is,i] = integrate_over_vspace(ff[iz,:,is,i], vpa_wgts)
            end
        end
    end
    #@views advection_test_1d(ff[:,:,:,1], ff[:,:,:,end])
    println("done.")
    return f_fldline_avg, delta_f, dens_moment
end

function field_line_average(fld, wgts, L)
    n = length(fld)
    total = 0.0
    for i ∈ 1:n
        total += wgts[i]*fld[i]
    end
    return total/L
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

end
