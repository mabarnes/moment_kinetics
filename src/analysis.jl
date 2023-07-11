"""
"""
module analysis

export analyze_fields_data
export analyze_moments_data
export analyze_pdf_data

using ..array_allocation: allocate_float
using ..calculus: integral
using ..coordinates: coordinate
using ..interpolation: interpolate_to_grid_1d
using ..load_data: open_readonly_output_file, get_nranks, load_pdf_data, load_rank_data
using ..load_data: load_distributed_charged_pdf_slice
using ..velocity_moments: integrate_over_vspace

using FFTW
using Statistics
using StatsBase

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
function check_Chodura_condition(run_name, vperp_global, vpa_global, r, z, vperp, vpa,
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
    f_lower = load_distributed_charged_pdf_slice(run_name, nblocks, :, n_ion_species, r,
                                                 z, vperp, vpa; z=1)
    f_upper = load_distributed_charged_pdf_slice(run_name, nblocks, :, n_ion_species, r,
                                                 z, vperp, vpa; z=z.n_global)
    for it ∈ 1:ntime, ir ∈ 1:nr
        vpabar = @. vpa_global.grid - 0.5 * geometry.rhostar * Er[1,ir,it] / geometry.bzed

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpabar)
            if abs(vpabar[ivpa]) < 1.e-14
                vpabar[ivpa] = 1.0
            end
        end

        @views lower_result[ir,it] =
            integrate_over_vspace(f_lower[:,:,1,ir,is,it], vpabar, -2, vpa_global.wgts,
                                  vperp_global.grid, 0, vperp_global.wgts)
        if it == ntime
            println("check vpabar lower", vpabar)
            println("result lower ", lower_result[ir,it])
        end

        lower_result[ir,it] *= 0.5 * T_e / dens[1,ir,is,it]

        vpabar = @. vpa_global.grid - 0.5 * geometry.rhostar * Er[end,ir,it] / geometry.bzed

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpabar)
            if abs(vpabar[ivpa]) < 1.e-14
                vpabar[ivpa] = 1.0
            end
        end

        @views upper_result[ir,it] =
            integrate_over_vspace(f_upper[:,:,end,ir,is,it], vpabar, -2, vpa_global.wgts,
                                  vperp_global.grid, 0, vperp_global.wgts)
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
function analyze_pdf_data(ff, n_species, ntime, z, vpa, vth, evolve_ppar)
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

"""
Return (v - mean(v, dims=2))
"""
function get_r_perturbation(v::AbstractArray{T,3}) where T
    # Get background as r-average of the variable, assuming the background is constant
    # in r
    background = mean(v, dims=2)
    perturbation = v .- background
    return perturbation
end

"""
Get 2D Fourier transform (in r and z) of non_uniform_data

First interpolates to uniform grid, then uses FFT
"""
function get_Fourier_modes_2D(non_uniform_data::AbstractArray{T,3}, r::coordinate,
                              r_spectral, z::coordinate, z_spectral) where T
    nt = size(non_uniform_data, 3)

    uniform_points_per_element_r = r.ngrid ÷ 4
    n_uniform_r = r.nelement_global * uniform_points_per_element_r
    uniform_spacing_r = r.L / n_uniform_r
    uniform_grid_r = collect(1:n_uniform_r).*uniform_spacing_r .+ 0.5.*uniform_spacing_r .- 0.5.*r.L

    uniform_points_per_element_z = z.ngrid ÷ 4
    n_uniform_z = z.nelement_global * uniform_points_per_element_z
    uniform_spacing_z = z.L / n_uniform_z
    uniform_grid_z = collect(1:n_uniform_z).*uniform_spacing_z .+ 0.5.*uniform_spacing_z .- 0.5.*z.L

    intermediate = allocate_float(n_uniform_z, r.n, nt)
    for it ∈ 1:nt, ir ∈ 1:r.n
        @views intermediate[:,ir,it] =
        interpolate_to_grid_1d(uniform_grid_z, non_uniform_data[:,ir,it], z,
                               z_spectral)
    end

    uniform_data = allocate_float(n_uniform_z, n_uniform_r, nt)
    for it ∈ 1:nt, iz ∈ 1:n_uniform_z
        @views uniform_data[iz,:,it] =
        interpolate_to_grid_1d(uniform_grid_r, non_uniform_data[iz,:,it], r,
                               r_spectral)
    end

    fourier_data = fft(uniform_data, (1,2))

    return fourier_data
end

"""
Get 1D Fourier transform (in r) of non_uniform_data

First interpolates to uniform grid, then uses FFT.

If zind is not given, find the zind where mode seems to be growing most strongly.
"""
function get_Fourier_modes_1D(non_uniform_data::AbstractArray{T,3}, r::coordinate,
                              r_spectral, z; zind=nothing) where T

    nt = size(non_uniform_data, 3)

    if zind === nothing
        # Find a z-location where the mode seems to be growing most strongly to analyse
        ###############################################################################

        # Get difference between max and min over the r dimension as a measure of the mode
        # amplitude
        Delta_var = maximum(non_uniform_data; dims=2)[:,1,:] - minimum(non_uniform_data; dims=2)[:,1,:]
        max_Delta_var = maximum(Delta_var; dims=1)[1,:]

        # Start searching for mode position once amplitude has grown to twice initial
        # perturbation
        startind = findfirst(x -> x>max_Delta_var[1], max_Delta_var)
        if startind === nothing
            startind = 1
        end

        # Find the z-index of the maximum of Delta_var
        # Need the iterator thing to convert CartesianIndex structs returned by argmax into
        # Ints that we can do arithmetic with.
        zind_maximum = [i[1] for i ∈ argmax(Delta_var; dims=1)]
        zind_maximum = zind_maximum[startind:end]

        # Want the 'most common' value in zind_maximum, but maybe that is noisy?
        # First find the most common bin for some reasonable number of bins. The background is
        # a mode with one wave-period in the box, so 16 bins seems like plenty.
        if z.n > 16
            nbins = 16
            bin_size = (z.n - 1) ÷ 16
        else
            nbins = 1
            bin_size = z.n
        end
        binned_zind_maximum = @. (zind_maximum-1) ÷ bin_size
        most_common_bin = mode(binned_zind_maximum)
        bin_min = most_common_bin * bin_size + 1
        bin_max = (most_common_bin+1) * bin_size
        zinds_in_bin = [zind for zind in zind_maximum if bin_min ≤ zind ≤ bin_max]

        # Find the most common zind in the bin, which might have some noise but will be in
        # about the right region regardless as it is in the bin
        zind = mode(zinds_in_bin)
        println("Estimating average maximum mode amplitude at zind=$zind, z=", z.grid[zind])
    end

    # Analyse the Fourier modes at zind
    ###################################
    non_uniform_data = @view non_uniform_data[zind,:,:]
    uniform_points_per_element_r = r.ngrid ÷ 4
    n_uniform_r = r.nelement_global * uniform_points_per_element_r
    uniform_spacing_r = r.L / n_uniform_r
    uniform_grid_r = collect(0:(n_uniform_r-1)).*uniform_spacing_r .+ 0.5.*uniform_spacing_r .- 0.5.*r.L

    uniform_data = allocate_float(n_uniform_r, nt)
    for it ∈ 1:nt
        @views uniform_data[:,it] =
        interpolate_to_grid_1d(uniform_grid_r, non_uniform_data[:,it], r,
                               r_spectral)
    end

    fourier_data = fft(uniform_data, 1)

    return fourier_data, zind
end

"""
"""
function analyze_2D_instability(phi, density, thermal_speed, r, z, r_spectral, z_spectral;
                                do_1d=true, do_2d=true, do_perturbation=true)
    # Assume there is only one species for this test
    density = density[:,:,1,:]
    thermal_speed = thermal_speed[:,:,1,:]

    # NB normalisation removes the factor of 1/2
    temperature = thermal_speed.^2

    if do_perturbation
        phi_perturbation = get_r_perturbation(phi)
        density_perturbation = get_r_perturbation(density)
        temperature_perturbation = get_r_perturbation(temperature)
    else
        phi_perturbation = nothing
        density_perturbation = nothing
        temperature_perturbation = nothing
    end

    nt = size(phi, 3)

    if do_2d
        phi_Fourier_2D = get_Fourier_modes_2D(phi, r, r_spectral, z, z_spectral)
        density_Fourier_2D = get_Fourier_modes_2D(density, r, r_spectral, z, z_spectral)
        temperature_Fourier_2D = get_Fourier_modes_2D(temperature, r, r_spectral, z, z_spectral)
    else
        phi_Fourier_2D = nothing
        density_Fourier_2D = nothing
        temperature_Fourier_2D = nothing
    end

    if do_1d
        phi_Fourier_1D, zind = get_Fourier_modes_1D(phi, r, r_spectral, z)
        density_Fourier_1D, _ = get_Fourier_modes_1D(density, r, r_spectral, z, zind=zind)
        temperature_Fourier_1D, _ = get_Fourier_modes_1D(temperature, r, r_spectral, z, zind=zind)
    else
        phi_Fourier_1D = nothing
        density_Fourier_1D = nothing
        temperature_Fourier_1D = nothing
    end

    return phi_perturbation, density_perturbation, temperature_perturbation,
           phi_Fourier_2D, density_Fourier_2D, temperature_Fourier_2D,
           phi_Fourier_1D, density_Fourier_1D, temperature_Fourier_1D
end

end
