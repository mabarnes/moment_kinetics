"""
"""
module analysis

export analyze_fields_data
export analyze_moments_data
export analyze_pdf_data

using ..array_allocation: allocate_float
using ..calculus: derivative!, integral
using ..velocity_moments: integrate_over_vspace

"""
"""
function analyze_fields_data(phi, ntime, z)
    print("Analyzing fields data...")
    phi_fldline_avg = allocate_float(ntime)
    for i ∈ 1:ntime
        phi_fldline_avg[i] = field_line_average(view(phi,:,i), z.wgts, z.L)
    end
    # delta_phi = phi - <phi> is the fluctuating phi
    delta_phi = allocate_float(z.n,ntime)
    for iz ∈ 1:z.n
        delta_phi[iz,:] .= phi[iz,:] - phi_fldline_avg
    end
    println("done.")
    return phi_fldline_avg, delta_phi
end

"""
Calculate E_parallel (from phi) as an array.

Only included in `analysis` as this function allocates a new array for the result, and
includes a loop over the whole grid.
"""
function calc_E_parallel(phi, z, z_spectral)
    ntime = size(phi, 3)
    nr = size(phi, 2)
    E_parallel = similar(phi)
    for it ∈ 1:ntime, ir ∈ 1:nr
        @views derivative!(z.scratch, phi[:,ir,it], z, z_spectral)
        @. E_parallel[:,ir,it] = -z.scratch
    end

    return E_parallel
end

"""
"""
function analyze_moments_data(density, parallel_flow, parallel_pressure, thermal_speed,
                              parallel_heat_flux, ntime, n_species, z)
    print("Analyzing velocity moments data...")
    density_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            density_fldline_avg[is,i] = field_line_average(view(density,:,is,i), z.wgts, z.L)
        end
    end
    upar_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            upar_fldline_avg[is,i] = field_line_average(view(parallel_flow,:,is,i), z.wgts, z.L)
        end
    end
    ppar_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            ppar_fldline_avg[is,i] = field_line_average(view(parallel_pressure,:,is,i), z.wgts, z.L)
        end
    end
    vth_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            vth_fldline_avg[is,i] = field_line_average(view(thermal_speed,:,is,i), z.wgts, z.L)
        end
    end
    qpar_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            qpar_fldline_avg[is,i] = field_line_average(view(parallel_heat_flux,:,is,i), z.wgts, z.L)
        end
    end
    # delta_density = n_s - <n_s> is the fluctuating density
    delta_density = allocate_float(z.n,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:z.n
            @. delta_density[iz,is,:] = density[iz,is,:] - density_fldline_avg[is,:]
        end
    end
    # delta_upar = upar_s - <upar_s> is the fluctuating parallel flow
    delta_upar = allocate_float(z.n,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:z.n
            @. delta_upar[iz,is,:] = parallel_flow[iz,is,:] - upar_fldline_avg[is,:]
        end
    end
    # delta_ppar = ppar_s - <ppar_s> is the fluctuating parallel pressure
    delta_ppar = allocate_float(z.n,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:z.n
            @. delta_ppar[iz,is,:] = parallel_pressure[iz,is,:] - ppar_fldline_avg[is,:]
        end
    end
    # delta_vth = vth_s - <vth_s> is the fluctuating thermal_speed
    delta_vth = allocate_float(z.n,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:z.n
            @. delta_vth[iz,is,:] = thermal_speed[iz,is,:] - vth_fldline_avg[is,:]
        end
    end
    # delta_qpar = qpar_s - <qpar_s> is the fluctuating parallel heat flux
    delta_qpar = allocate_float(z.n,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:z.n
            @. delta_qpar[iz,is,:] = parallel_heat_flux[iz,is,:] - qpar_fldline_avg[is,:]
        end
    end
    println("done.")
    return density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, vth_fldline_avg, qpar_fldline_avg,
           delta_density, delta_upar, delta_ppar, delta_vth, delta_qpar
end

"""
"""
function analyze_pdf_data(ff, vpa, z, n_species, ntime, vth, evolve_ppar)
    print("Analyzing distribution function data...")
    f_fldline_avg = allocate_float(vpa.n,n_species,ntime)
    for i ∈ 1:ntime
        for is ∈ 1:n_species
            for ivpa ∈ 1:vpa.n
                f_fldline_avg[ivpa,is,i] = field_line_average(view(ff,ivpa,:,is,i), z.wgts, z.L)
            end
        end
    end
    # delta_f = f - <f> is the fluctuating distribution function
    delta_f = allocate_float(vpa.n,z.n,n_species,ntime)
    for iz ∈ 1:z.n
        @. delta_f[:,iz,:,:] = ff[:,iz,:,:] - f_fldline_avg
    end
    dens_moment = allocate_float(z.n,n_species,ntime)
    upar_moment = allocate_float(z.n,n_species,ntime)
    ppar_moment = allocate_float(z.n,n_species,ntime)
    for i ∈ 1:ntime
        for is ∈ 1:n_species
            for iz ∈ 1:z.n
                @views dens_moment[iz,is,i] = integrate_over_vspace(ff[:,iz,is,i], vpa.wgts)
                @views upar_moment[iz,is,i] = integrate_over_vspace(ff[:,iz,is,i], vpa.grid, vpa.wgts)
                @views ppar_moment[iz,is,i] = integrate_over_vspace(ff[:,iz,is,i], vpa.grid, 2, vpa.wgts)
            end
        end
    end
    if evolve_ppar
        @. dens_moment *= vth
        @. upar_moment *= vth^2
        @. ppar_moment *= vth^3
    end
    #@views advection_test_1d(ff[:,:,:,1], ff[:,:,:,end])
    println("done.")
    return f_fldline_avg, delta_f, dens_moment, upar_moment, ppar_moment
end

"""
"""
function field_line_average(fld, wgts, L)
    return integral(fld, wgts)/L
end

end
