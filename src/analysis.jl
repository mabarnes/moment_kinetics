"""
"""
module analysis

export analyze_fields_data
export analyze_moments_data
export analyze_pdf_data

using ..array_allocation: allocate_float
using ..calculus: derivative!, integral
using ..initial_conditions: vpagrid_to_dzdt
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
Check the (kinetic) Chodura condition

Chodura condition is:
∫d^3v F/vpa^2 ≤ mi ne/Te

Return a tuple (whose first entry is the result for the lower boundary and second for the
upper) of the ratio which is 1 if the Chodura condition is satisfied (with equality):
Te/(mi ne) * ∫d^3v F/vpa^2

Currently only evaluates condition for the first species: is=1

The 1D1V code evolves the marginalised distribution function f = ∫d^2vperp F so the
Chodura condition becomes
∫dvpa f/vpa^2 ≤ mi ne/Te

In normalised form (normalised variables suffixed with 'N'):
vpa = cref vpaN
ne = nref neN
Te = Tref TeN
f = fN nref / cref sqrt(pi)
cref = sqrt(2 Tref / mi)

cref ∫dvpaN fN nref / cref sqrt(pi) cref^2 vpaN^2 ≤ mi nref neN / Tref TeN
nref / (sqrt(pi) cref^2) * ∫dvpaN fN / vpaN^2 ≤ mi nref neN / Tref TeN
mi nref / (sqrt(pi) 2 Tref) * ∫dvpaN fN / vpaN^2 ≤ mi nref neN / Tref TeN
1 / (2 sqrt(pi)) * ∫dvpaN fN / vpaN^2 ≤ neN / TeN
1 / (2 sqrt(pi)) * ∫dvpaN fN / vpaN^2 ≤ neN / TeN
TeN / (2 neN sqrt(pi)) * ∫dvpaN fN / vpaN^2 ≤ 1

Note that `integrate_over_vspace()` includes the 1/sqrt(pi) factor already.
"""
function check_Chodura_condition(f, vpa, dens, upar, vth, T_e, evolve_density,
                                 evolve_upar, evolve_ppar, z_bc)

    if z_bc != "wall"
        return nothing, nothing
    end

    ntime = size(f, 5)
    is = 1
    nr = size(f, 3)
    lower_result = zeros(nr, ntime)
    upper_result = zeros(nr, ntime)
    for it ∈ 1:ntime, ir ∈ 1:nr
        # create an array of dz/dt values at z = -Lz/2
        @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, vth[1,ir,is,it], upar[1,ir,is,it],
                                          evolve_ppar, evolve_upar)

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpa.scratch)
            if abs(vpa.scratch[ivpa]) < 1.e-14
                vpa.scratch[ivpa] = 1.0
            end
        end

        @views lower_result[ir,it] = integrate_over_vspace(f[:,1,ir,is,it], vpa.scratch,
                                                           -2, vpa.wgts)
        if evolve_ppar
            # vpagrid_to_dzdt already includes the factor of vth, so just need to multiply
            # by one for the d(wpa) in the integral
            lower_result[ir,it] *= vth[1,ir,is,it]
        end
        if !evolve_density
            # If evolve_dens=true, f is already normalised by dens
            lower_result[ir,it] /= dens[1,ir,is,it]
        end
        lower_result[ir,it] *= 0.5*T_e

        # create an array of dz/dt values at z = +Lz/2
        @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, vth[end,ir,is,it], upar[end,ir,is,it],
                                          evolve_ppar, evolve_upar)

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpa.scratch)
            if abs(vpa.scratch[ivpa]) < 1.e-14
                vpa.scratch[ivpa] = 1.0
            end
        end

        @views upper_result[ir,it] = integrate_over_vspace(f[:,end,ir,is,it], vpa.scratch,
                                                           -2, vpa.wgts)
        if evolve_ppar
            # vpagrid_to_dzdt already includes the factor of vth, so just need to multiply
            # by one for the d(wpa) in the integral
            upper_result[ir,it] *= vth[end,ir,is,it]
        end
        if !evolve_density
            # If evolve_dens=true, f is already normalised by dens
            upper_result[ir,it] /= dens[end,ir,is,it]
        end
        upper_result[ir,it] *= 0.5*T_e
    end

    return lower_result, upper_result
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
