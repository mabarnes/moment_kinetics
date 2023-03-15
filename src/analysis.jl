"""
"""
module analysis

export analyze_fields_data
export analyze_moments_data
export analyze_pdf_data

using ..array_allocation: allocate_float
using ..calculus: integral
using ..chebyshev: setup_chebyshev_pseudospectral, chebyshev_derivative!
using ..interpolation: interpolate_to_grid_1d
using ..load_data: open_readonly_output_file, get_nranks, load_pdf_data, load_rank_data
using ..velocity_moments: integrate_over_vspace

using FFTW
using Statistics

"""
"""
function analyze_fields_data(phi, ntime, nz, z_wgts, Lz)
    print("Analyzing fields data...")
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

"""
Check the (kinetic) Chodura condition

Chodura condition is:
∫d^3v F/vpa^2 ≤ mi ne/Te

Return a tuple (whose first entry is the result for the lower boundary and second for the
upper) of the ratio which is 1 if the Chodura condition is satisfied (with equality):
Te/(mi ne) * ∫d^3v F/vpa^2

Currently only evaluates condition for the first species: is=1

2D2V
----

In normalised form (normalised variables suffixed with 'N'):
vpa = cref vpaN
vperp = cref vperpN
ne = nref neN
Te = Tref TeN
F = FN nref / cref^3 pi^3/2
cref = sqrt(2 Tref / mi)

cref^3 ∫d^3vN FN nref / cref^3 pi^3/2 cref^2 vpaN^2 ≤ mi nref neN / Tref TeN
nref / (pi^3/2 cref^2) * ∫d^3vN FN / vpaN^2 ≤ mi nref neN / Tref TeN
mi nref / (pi^3/2 2 Tref) * ∫d^3vN FN / vpaN^2 ≤ mi nref neN / Tref TeN
1 / (2 pi^3/2) * ∫d^3vN FN / vpaN^2 ≤ neN / TeN
1 / (2 pi^3/2) * ∫d^3vN FN / vpaN^2 ≤ neN / TeN
TeN / (2 neN pi^3/2) * ∫d^3vN FN / vpaN^2 ≤ 1

Note that `integrate_over_vspace()` includes the 1/pi^3/2 factor already.

1D1V
----

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
function check_Chodura_condition(run_name, vpa_grid, vpa_wgts, vperp_grid, vperp_wgts,
                                 dens, T_e, Er, geometry, z_bc, nblocks)

    if z_bc != "wall"
        return nothing, nothing
    end

    ntime = size(Er, 3)
    is = 1
    nr = size(Er, 2)
    lower_result = zeros(nr, ntime)
    upper_result = zeros(nr, ntime)
    f_lower = nothing
    f_upper = nothing
    z_nrank, r_nrank = get_nranks(run_name, nblocks, "dfns")
    for iblock in 0:nblocks-1
        fid_pdfs = open_readonly_output_file(run_name,"dfns",iblock=iblock)
        z_irank, r_irank = load_rank_data(fid_pdfs)
        if z_irank == 0
            if f_lower === nothing
                f_lower = load_pdf_data(fid_pdfs)
            else
                # Concatenate along r-dimension
                f_lower = cat(f_lower, load_pdf_data(fid_pdfs); dims=4)
            end
        end
        if z_irank == z_nrank - 1
            if f_upper === nothing
                f_upper = load_pdf_data(fid_pdfs)
            else
                # Concatenate along r-dimension
                f_upper = cat(f_upper, load_pdf_data(fid_pdfs); dims=4)
            end
        end
    end
    for it ∈ 1:ntime, ir ∈ 1:nr
        vpabar = @. vpa_grid - 0.5 * geometry.rhostar * Er[1,ir,it] / geometry.bzed

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpabar)
            if abs(vpabar[ivpa]) < 1.e-14
                vpabar[ivpa] = 1.0
            end
        end

        @views lower_result[ir,it] =
            integrate_over_vspace(f_lower[:,:,1,ir,is,it], vpabar, -2, vpa_wgts,
                                  vperp_grid, 0, vperp_wgts)
        if it == ntime
            println("check vpabar lower", vpabar)
            println("result lower ", lower_result[ir,it])
        end

        lower_result[ir,it] *= 0.5 * T_e / dens[1,ir,is,it]

        vpabar = @. vpa_grid - 0.5 * geometry.rhostar * Er[end,ir,it] / geometry.bzed

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpabar)
            if abs(vpabar[ivpa]) < 1.e-14
                vpabar[ivpa] = 1.0
            end
        end

        @views upper_result[ir,it] =
            integrate_over_vspace(f_upper[:,:,end,ir,is,it], vpabar, -2, vpa_wgts,
                                  vperp_grid, 0, vperp_wgts)
        if it == ntime
            println("check vpabar upper ", vpabar)
            println("result upper ", upper_result[ir,it])
        end

        upper_result[ir,it] *= 0.5 * T_e / dens[end,ir,is,it]
    end

    println("final Chodura results result ", lower_result[1,end], " ", upper_result[1,end])

    return lower_result, upper_result
end

"""
"""
function analyze_moments_data(density, parallel_flow, parallel_pressure, parallel_heat_flux, ntime, n_species, nz, z_wgts, Lz)
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
    qpar_fldline_avg = allocate_float(n_species, ntime)
    for is ∈ 1:n_species
        for i ∈ 1:ntime
            qpar_fldline_avg[is,i] = field_line_average(view(parallel_heat_flux,:,is,i), z_wgts, Lz)
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
    # delta_qpar = qpar_s - <qpar_s> is the fluctuating parallel heat flux
    delta_qpar = allocate_float(nz,n_species,ntime)
    for is ∈ 1:n_species
        for iz ∈ 1:nz
            @. delta_qpar[iz,is,:] = parallel_heat_flux[iz,is,:] - qpar_fldline_avg[is,:]
        end
    end
    println("done.")
    return density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, qpar_fldline_avg,
           delta_density, delta_upar, delta_ppar, delta_qpar
end

"""
"""
function analyze_pdf_data(ff, vpa, nvpa, nz, n_species, ntime, vpa_wgts, z_wgts, Lz,
                          vth, evolve_ppar)
    print("Analyzing distribution function data...")
    f_fldline_avg = allocate_float(nvpa,n_species,ntime)
    for i ∈ 1:ntime
        for is ∈ 1:n_species
            for ivpa ∈ 1:nvpa
                f_fldline_avg[ivpa,is,i] = field_line_average(view(ff,ivpa,:,is,i), z_wgts, Lz)
            end
        end
    end
    # delta_f = f - <f> is the fluctuating distribution function
    delta_f = allocate_float(nvpa,nz,n_species,ntime)
    for iz ∈ 1:nz
        @. delta_f[:,iz,:,:] = ff[:,iz,:,:] - f_fldline_avg
    end
    dens_moment = allocate_float(nz,n_species,ntime)
    upar_moment = allocate_float(nz,n_species,ntime)
    ppar_moment = allocate_float(nz,n_species,ntime)
    for i ∈ 1:ntime
        for is ∈ 1:n_species
            for iz ∈ 1:nz
                @views dens_moment[iz,is,i] = integrate_over_vspace(ff[:,iz,is,i], vpa_wgts)
                @views upar_moment[iz,is,i] = integrate_over_vspace(ff[:,iz,is,i], vpa, vpa_wgts)
                @views ppar_moment[iz,is,i] = integrate_over_vspace(ff[:,iz,is,i], vpa, 2, vpa_wgts)
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

"""
"""
function analyze_2D_instability(phi, density, thermal_speed, r, z)
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

    if r.discretization == "chebyshev_pseudospectral" && r.n > 1
        # create arrays needed for explicit Chebyshev pseudospectral treatment in vpa
        # and create the plans for the forward and backward fast Chebyshev transforms
        r_spectral = setup_chebyshev_pseudospectral(r)
        # obtain the local derivatives of the uniform r-grid with respect to the used r-grid
        chebyshev_derivative!(r.duniform_dgrid, r.uniform_grid, r_spectral, r)
    else
        # create dummy Bool variable to return in place of the above struct
        r_spectral = false
        r.duniform_dgrid .= 1.0
    end

    nt = size(phi, 3)

    # Assume there is only one species for this test
    density = density[:,:,1,:]
    thermal_speed = thermal_speed[:,:,1,:]

    # NB normalisation removes the factor of 1/2
    temperature = thermal_speed.^2

    # Get background as r-average of initial condition, as the initial perturbation varies
    # sinusoidally in r
    background_phi = reshape(mean(phi, dims=2), (z.n, 1, nt))
    background_density = reshape(mean(density, dims=2), (z.n, 1, nt))
    background_temperature = reshape(mean(temperature, dims=2), (z.n, 1, nt))

    phi_perturbation = phi .- background_phi
    density_perturbation = density .- background_density
    temperature_perturbation = temperature .- background_temperature

    nt = size(phi, 3)

    function get_Fourier_modes(non_uniform_data, r, r_spectral, z, z_spectral)

        uniform_points_per_element_r = r.ngrid ÷ 4
        n_uniform_r = r.nelement_global * uniform_points_per_element_r
        uniform_spacing_r = r.L / n_uniform_r
        uniform_grid_r = collect(0:(n_uniform_r-1)).*uniform_spacing_r .+ 0.5.*uniform_spacing_r .- 0.5.*r.L

        uniform_points_per_element_z = z.ngrid ÷ 4
        n_uniform_z = z.nelement_global * uniform_points_per_element_z
        uniform_spacing_z = z.L / n_uniform_z
        uniform_grid_z = collect(0:(n_uniform_z-1)).*uniform_spacing_z .+ 0.5.*uniform_spacing_z .- 0.5.*z.L

        intermediate = allocate_float(n_uniform_z, r.n, nt)
        for it ∈ 1:nt, ir ∈ 1:r.n
            @views intermediate[:,ir,it] =
                interpolate_to_grid_1d(uniform_grid_z, non_uniform_data[:,ir,it], z,
                                       z_spectral)
        end

        uniform_data = allocate_float(n_uniform_z, n_uniform_r, nt)
        for it ∈ 1:nt, iz ∈ 1:n_uniform_z
            @views uniform_data[iz,:,it] =
                interpolate_to_grid_1d(uniform_grid_r, intermediate[iz,:,it], r,
                                       r_spectral)
        end

        fourier_data = fft(uniform_data, (1,2))

        return fourier_data
    end

    phi_Fourier = get_Fourier_modes(phi, r, r_spectral, z, z_spectral)
    density_Fourier = get_Fourier_modes(density, r, r_spectral, z, z_spectral)
    temperature_Fourier = get_Fourier_modes(temperature, r, r_spectral, z, z_spectral)

    return phi_perturbation, density_perturbation, temperature_perturbation,
           phi_Fourier, density_Fourier, temperature_Fourier
end

end
