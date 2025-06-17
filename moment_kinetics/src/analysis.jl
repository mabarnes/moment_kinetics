"""
"""
module analysis

export analyze_fields_data
export analyze_moments_data
export analyze_pdf_data

using ..array_allocation: allocate_float, allocate_int
using ..calculus: integral
using ..communication
using ..coordinates: coordinate
using ..boundary_conditions: vpagrid_to_dzdt
using ..interpolation: interpolate_to_grid_1d
using ..load_data: open_readonly_output_file, get_nranks, load_pdf_data, load_rank_data
using ..load_data: load_distributed_ion_pdf_slice
using ..looping
using ..type_definitions: mk_int, mk_float

using FFTW
using LsqFit
using MPI
using OrderedCollections
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
F = FN nref / cref^3
cref = sqrt(Tref / mi)

cref^3 ∫d^3vN FN nref / cref^3 cref^2 vpaN^2 ≤ mi nref neN / Tref TeN
nref / (cref^2) * ∫d^3vN FN / vpaN^2 ≤ mi nref neN / Tref TeN
mi nref / (2 Tref) * ∫d^3vN FN / vpaN^2 ≤ mi nref neN / Tref TeN
1 / 2 * ∫d^3vN FN / vpaN^2 ≤ neN / TeN
1 / 2 * ∫d^3vN FN / vpaN^2 ≤ neN / TeN
TeN / (2 neN) * ∫d^3vN FN / vpaN^2 ≤ 1

1D1V
----

The 1D1V code evolves the marginalised distribution function f = ∫d^2vperp F so the
Chodura condition becomes
∫dvpa f/vpa^2 ≤ mi ne/Te

In normalised form (normalised variables suffixed with 'N'):
vpa = cref vpaN
ne = nref neN
Te = Tref TeN
f = fN nref / cref
cref = sqrt(2 Tref / mi)

cref ∫dvpaN fN nref / cref cref^2 vpaN^2 ≤ mi nref neN / Tref TeN
nref / cref^2 * ∫dvpaN fN / vpaN^2 ≤ mi nref neN / Tref TeN
mi nref / (2 Tref) * ∫dvpaN fN / vpaN^2 ≤ mi nref neN / Tref TeN
1 / 2 * ∫dvpaN fN / vpaN^2 ≤ neN / TeN
1 / 2 * ∫dvpaN fN / vpaN^2 ≤ neN / TeN
TeN / (2 neN) * ∫dvpaN fN / vpaN^2 ≤ 1

If `ir0` is passed, only load the data for as single r-point (to save memory).

If `find_extra_offset=true` is passed, calculates how many entries of `f_lower`/`f_upper`
adjacent to \$v_∥=0\$ would need to be zero-ed out in order for the condition to be
satisfied.
"""
function check_Chodura_condition(r, z, vperp, vpa, dens, upar, vth, temp_e, composition,
                                 Er, geometry, z_bc, nblocks, run_name=nothing,
                                 it0::Union{Nothing, mk_int}=nothing,
                                 ir0::Union{Nothing, mk_int}=nothing;
                                 f_lower=nothing, f_upper=nothing,
                                 evolve_density=false, evolve_upar=false, evolve_p=false,
                                 find_extra_offset=false)

    if z_bc != "wall"
        return nothing, nothing
    end

    zero = 1.0e-14

    if it0 === nothing
        ntime = size(Er, 3)
        t_range = 1:ntime
    else
        it_max = size(Er,3)
        dens = selectdim(dens, 4, it_max:it_max)
        Er = selectdim(Er, 3, it_max:it_max)
        ntime = 1
        t_range = it0:it0
    end
    is = 1
    if ir0 === nothing
        nr = size(Er, 2)
    else
        nr = 1
    end
    nvperp = size(f_lower,2)
    nvpa = size(f_lower,1)

    if temp_e === nothing
        # Assume this is from a Boltzmann electron response simulation
        temp_e = fill(composition.T_e, 2, nr, ntime)
    end

    lower_result = zeros(nr, ntime)
    upper_result = zeros(nr, ntime)
    if f_lower !== nothing || f_upper !== nothing
        if it0 !== nothing
            error("Using `it0` not compatible with passing `f_lower` or `f_upper` as "
                  * "arguments")
        end
        if ir0 !== nothing
            error("Using `ir0` not compatible with passing `f_lower` or `f_upper` as "
                  * "arguments")
        end
    end
    if f_lower === nothing
        f_lower = load_distributed_ion_pdf_slice(run_name, nblocks, t_range,
                                                     composition.n_ion_species, r, z,
                                                     vperp, vpa; iz=1, ir=ir0)
    end
    if f_upper === nothing
        f_upper = load_distributed_ion_pdf_slice(run_name, nblocks, t_range,
                                                     composition.n_ion_species, r, z,
                                                     vperp, vpa; iz=z.n_global, ir=ir0)
    end
    if ir0 !== nothing
        f_lower = reshape(f_lower,
                          (size(f_lower, 1), size(f_lower, 2), 1, size(f_lower, 3),
                           size(f_lower, 4), size(f_lower, 5)))
        f_upper = reshape(f_upper,
                          (size(f_upper, 1), size(f_upper, 2), 1, size(f_upper, 3),
                           size(f_upper, 4), size(f_upper, 5)))
    end

    f_lower = @views get_unnormalised_f_1d(f_lower, dens[1,:,:,:], vth[1,:,:,:],
                                           evolve_density, evolve_p)
    f_upper = @views get_unnormalised_f_1d(f_upper, dens[end,:,:,:], vth[end,:,:,:],
                                           evolve_density, evolve_p)
    if find_extra_offset
        # Allocate output arrays for the number of entries that would need to be zero-ed
        # out.
        extra_offset_lower = allocate_int(nr,ntime)
        extra_offset_upper = allocate_int(nr,ntime)
        cutoff_lower = allocate_float(nr,ntime)
        cutoff_upper = allocate_float(nr,ntime)
    end
    for it ∈ 1:ntime, ir ∈ 1:nr
        # Lower target
        ##############

        v_parallel = vpagrid_to_dzdt(vpa.grid, vth[1,ir,is,it], upar[1,ir,is,it],
                                     evolve_p, evolve_upar)
        vpabar = @. v_parallel - 0.5 * geometry.rhostar * Er[1,ir,it] / geometry.bzed[1,ir]

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpabar)
            if abs(vpabar[ivpa]) < zero
                vpabar[ivpa] = 1.0
            end
        end

        @views lower_result[ir,it] =
            integral(f_lower[:,:,ir,is,it], vpabar, -2, vpa.wgts, vperp.grid, 0,
                     vperp.wgts)
        if it == ntime
            println("check vpabar lower", vpabar)
            println("result lower ", lower_result[ir,it])
        end

        lower_result[ir,it] *= 0.5 * temp_e[1,ir,it] / dens[1,ir,is,it]

        if find_extra_offset
            if lower_result[ir,it] ≤ 1.0
                extra_offset_lower[ir,it] = 0
            else
                integrand = f_lower[:,:,ir,is,it]
                for ivperp ∈ 1:nvperp
                    @. integrand[:,ivperp] *= vpabar^(-2) * vpa.wgts * vperp.wgts[ivperp]
                end
                vperp_integral = @view sum(integrand; dims=2)[:,1]
                cumulative_vpa_integral = cumsum(vperp_integral)
                cutoff_index = searchsortedfirst(cumulative_vpa_integral, 2.0 * dens[1,ir,is,it] / temp_e[1,ir,it]) - 1
                cutoff_lower[ir,it] = mean(vpabar[cutoff_index:cutoff_index+1])
                vpa_before_zero_index = searchsortedfirst(vpabar, -zero) - 1
                extra_offset_lower[ir,it] = vpa_before_zero_index - cutoff_index
            end
        end

        # Upper target
        ##############

        v_parallel = vpagrid_to_dzdt(vpa.grid, vth[end,ir,is,it], upar[end,ir,is,it],
                                     evolve_p, evolve_upar)
        vpabar = @. v_parallel - 0.5 * geometry.rhostar * Er[end,ir,it] / geometry.bzed[end,ir]

        # Get rid of a zero if it is there to avoid a blow up - f should be zero at that
        # point anyway
        for ivpa ∈ eachindex(vpabar)
            if abs(vpabar[ivpa]) < zero
                vpabar[ivpa] = 1.0
            end
        end

        @views upper_result[ir,it] =
            integral(f_upper[:,:,ir,is,it], vpabar, -2, vpa.wgts, vperp.grid, 0,
                     vperp.wgts)
        if it == ntime
            println("check vpabar upper ", vpabar)
            println("result upper ", upper_result[ir,it])
        end

        upper_result[ir,it] *= 0.5 * temp_e[end,ir,it] / dens[end,ir,is,it]

        if find_extra_offset
            if upper_result[ir,it] ≤ 1.0
                extra_offset_upper[ir,it] = 0
            else
                integrand = f_upper[:,:,ir,is,it]
                for ivperp ∈ 1:nvperp
                    @. integrand[:,ivperp] *= vpabar^(-2) * vpa.wgts * vperp.wgts[ivperp]
                end
                vperp_integral = @view sum(integrand; dims=2)[:,1]
                cumulative_vpa_integral = reverse(cumsum(reverse(vperp_integral)))
                cutoff_index = searchsortedfirst(cumulative_vpa_integral, 2.0 * dens[end,ir,is,it] / temp_e[end,ir,it]; rev=true)
                cutoff_upper[ir,it] = mean(vpabar[cutoff_index-1:cutoff_index])
                vpa_after_zero_index = searchsortedlast(vpabar, zero) + 1
                extra_offset_upper[ir,it] = cutoff_index - vpa_after_zero_index
            end
        end
    end

    if find_extra_offset
        println("final Chodura results ", lower_result[1,end], " (", extra_offset_lower[1,end], ") ", upper_result[1,end], " (", extra_offset_upper[1,end], ")")
    else
        println("final Chodura results ", lower_result[1,end], " ", upper_result[1,end])
    end

    if it0 !== nothing && ir0 !== nothing
        lower_result = lower_result[1,1]
        upper_result = upper_result[1,1]
    elseif it0 !== nothing
        lower_result = @view lower_result[:,1]
        upper_result = @view upper_result[:,1]
    elseif ir0 !== nothing
        lower_result = @view lower_result[1,:]
        upper_result = @view upper_result[1,:]
    end

    if find_extra_offset
        return lower_result, upper_result, cutoff_lower, cutoff_upper
    else
        return lower_result, upper_result
    end
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
                @views dens_moment[iz,is,i] = integral(ff[:,iz,is,i], vpa.wgts)
                @views upar_moment[iz,is,i] = integral(ff[:,iz,is,i], vpa.grid, vpa.wgts)
                @views ppar_moment[iz,is,i] = integral(ff[:,iz,is,i], vpa.grid, 2, vpa.wgts)
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

const default_epsilon = 1.0e-4

"""
    steady_state_residuals(variable, variable_at_previous_time, dt;
                           epsilon=$default_epsilon, use_mpi=false,
                           only_max_abs=false)

Calculate how close a variable is to steady state.

Calculates several quantities. Define the 'squared absolute residual'
``r_\\mathrm{abs}(t)^2`` for a quantity ``a(t,x)`` as

``r_\\mathrm{abs}(t)^2 = \\left( a(t,x) - a(t - \\delta t,x) \\right)``

and the 'squared relative residual' ``r_\\mathrm{rel}(t)^2``

``r_\\mathrm{rel}(t)^2 = \\left( \\frac{a(t,x) - a(t - \\delta t,x)}{\\delta t \\left| a(t,x) + \\epsilon \\max_x(a(t,x)) \\right|} \\right)``

where ``x`` stands for any spatial and velocity coordinates, and the offset ``\\epsilon
\\max_x(a(t,x))`` is used to avoid points where ``a(t,x)`` happens to be very close to
zero from dominating the result in the 'squared relative residual', with ``max_x`` being
the maximum over the ``x`` coordinate(s). Returns an `OrderedDict` containing: the maximum
'absolute residual' ``\\max_x\\left( \\sqrt{r_\\mathrm{abs}(t)^2} \\right)``
(`"RMS absolute residual"`); the root-mean-square (RMS) 'absolute residual'
``\\left< \\sqrt{r_\\mathrm{abs}(t)^2} \\right>_x`` (`"max absolute residual"`); the
maximum 'relative residual' ``\\max_x\\left( \\sqrt{r_\\mathrm{rel}(t)^2} \\right)``
(`"RMS relative residual"`); the root-mean-square (RMS) 'relative residual'
``\\left< \\sqrt{r_\\mathrm{rel}(t)^2} \\right>_x`` (`"max relative residual"`).

`variable` gives the value of ``a(t,x)`` at the current time, `variable_at_previous_time`
the value ``a(t - \\delta t, x)`` at a previous time and `dt` gives the difference in
times ``\\delta t``. All three can be arrays with a time dimension of the same length, or
have no time dimension.

By default runs in serial, but if `use_mpi=true` is passed, assume MPI has been
initialised, and that `variable` has r and z dimensions but no species dimension, and use
`@loop_*` macros. In this case the result is returned only on global rank 0. When using
distributed-memory MPI, this routine will double-count the points on block boundaries.

If `only_max_abs=true` is passed, then only calculate the 'maxium absolute residual'. In
this case just returns the "max absolute residual", not an OrderedDict.
"""
function steady_state_residuals(variable, variable_at_previous_time, dt;
                                epsilon=default_epsilon, use_mpi=false,
                                only_max_abs=false)
    return steady_state_residuals(variable, variable_at_previous_time, dt, use_mpi,
                                  only_max_abs, epsilon)
end
function steady_state_residuals(variable, variable_at_previous_time, dt, use_mpi,
                                only_max_abs=false, epsilon=default_epsilon)
    square_residual_norms =
        steady_state_square_residuals(variable, variable_at_previous_time, dt, nothing,
                                      use_mpi, only_max_abs, epsilon)
    if global_rank[] == 0
        if only_max_abs
            # In this case as an optimisation the residual was not squared, so do not need
            # to square-root here
            return square_residual_norms
        else
            return OrderedDict{String,Vector{mk_float}}(k=>sqrt.(v) for (k,v) ∈ square_residual_norms)
        end
    else
        return nothing
    end
end

"""
    steady_state_square_residuals(variable, variable_at_previous_time, dt,
                                  variable_max=nothing, use_mpi=false,
                                  only_max_abs=false, epsilon=$default_epsilon)

Used to calculate the mean square residual for [`steady_state_residuals`](@ref).

Useful to define this separately as it can be called on (equally-sized) chunks of the
variable and then combined appropriately. If this is done, the global maximum of
`abs.(variable)` should be passed to `variable_max`.

See [`steady_state_residuals`](@ref) for documenation of the other arguments. The return
values of [`steady_state_residuals`](@ref) are the square-root of the return values of
this function.
"""
function steady_state_square_residuals(variable, variable_at_previous_time, dt,
                                       variable_max=nothing, use_mpi=false,
                                       only_max_abs=false, epsilon=default_epsilon)
    if ndims(dt) == 0
        t_dim = ndims(variable) + 1
    else
        t_dim = ndims(variable)
    end
    if use_mpi
        @begin_r_z_region()
        if !only_max_abs && variable_max === nothing
            local_max = 0.0
            @loop_r_z ir iz begin
                this_slice = selectdim(selectdim(variable, t_dim - 1, ir), t_dim - 2, iz)
                local_max = max(local_max, maximum(abs.(this_slice)))
            end
            variable_max = MPI.Allreduce(local_max, max, comm_world)
        end
        if isa(dt, Vector)
            reshaped_dt = reshape(dt, tuple((1 for _ ∈ 1:t_dim-1)..., size(dt)...))
        else
            reshaped_dt = dt
        end
        if only_max_abs
            if size(dt) == ()
                # local_max_absolute should always be at least a 1d array of size 1, not
                # a 0d array, so that the MPI.Gather() below works correctly.
                local_max_absolute = zeros(1)
            else
                local_max_absolute = zeros(size(dt))
            end
        else
            if size(dt) == ()
                # local_max_absolute should always be at least a 1d array of size 1, not
                # a 0d array, so that the MPI.Gather() below works correctly.
                local_total_absolute_square = zeros(1)
                local_max_absolute_square = zeros(1)
                local_total_relative_square = zeros(1)
                local_max_relative_square = zeros(1)
            else
                local_total_absolute_square = zeros(size(dt))
                local_max_absolute_square = zeros(size(dt))
                local_total_relative_square = zeros(size(dt))
                local_max_relative_square = zeros(size(dt))
            end
        end
        @loop_r_z ir iz begin
            this_slice = selectdim(selectdim(variable, t_dim - 1, ir), t_dim - 2, iz)
            this_slice_previous_time = selectdim(selectdim(variable_at_previous_time,
                                                           t_dim - 1, ir), t_dim - 2, iz)

            if only_max_abs
                absolute_residual =
                    _steady_state_absolute_residual(this_slice, this_slice_previous_time,
                                                    reshaped_dt)
                # Need to wrap the maximum(...) in a call to vec(...) so that we return a
                # Vector, not an N-dimensional array where the first (N-1) dimensions all
                # have size 1.
                this_dims = tuple((1:t_dim-3)...)
                if this_dims === ()
                    local_max_absolute = max.(local_max_absolute, [absolute_residual])
                else
                    local_max_absolute = max.(local_max_absolute,
                                              vec(maximum(absolute_residual,
                                                          dims=this_dims)))
                end
            else
                absolute_square_residual, relative_square_residual =
                    _steady_state_square_residual(this_slice, this_slice_previous_time,
                                                  reshaped_dt, epsilon, variable_max)
                # Need to wrap the sum(...) or maximum(...) in a call to vec(...) so that
                # we return a Vector, not an N-dimensional array where the first (N-1)
                # dimensions all have size 1.
                local_total_absolute_square .+= vec(sum(absolute_square_residual,
                                                        dims=tuple((1:t_dim-1)...)))
                local_max_absolute_square = max.(local_max_absolute_square,
                                                 vec(maximum(absolute_square_residual,
                                                             dims=tuple((1:t_dim-1)...))))
                local_total_relative_square .+= vec(sum(relative_square_residual,
                                                        dims=tuple((1:t_dim-1)...)))
                local_max_relative_square = max.(local_max_relative_square,
                                                 vec(maximum(relative_square_residual,
                                                             dims=tuple((1:t_dim-1)...))))
            end
        end

        # Pack results together so we only need one communication
        if only_max_abs
            packed_results = local_max_absolute
        else
            packed_results = hcat(local_total_absolute_square, local_max_absolute_square,
                                  local_total_relative_square, local_max_relative_square)
        end
        gathered_results = MPI.Gather(packed_results, 0, comm_block[])

        if block_rank[] == 0
            # MPI.Gather returns a flattened Vector, so reshape back into nice Array
            gathered_results = reshape(gathered_results,
                                       (size(packed_results)..., block_size[]))

            # Finish calculating block-local mean/max
            packed_block_results = similar(packed_results)
            if only_max_abs
                @boundscheck ndims(packed_results) == 1
                @boundscheck ndims(gathered_results) == 2
                packed_block_results .= maximum(gathered_results, dims=2)
            else
                @boundscheck ndims(packed_results) == 2
                @boundscheck ndims(gathered_results) == 3
                packed_block_results[:,1] = sum(@view(gathered_results[:,1,:]), dims=2)
                packed_block_results[:,2] = maximum(@view(gathered_results[:,2,:]), dims=2)
                packed_block_results[:,3] = sum(@view(gathered_results[:,3,:]), dims=2)
                packed_block_results[:,4] = maximum(@view(gathered_results[:,4,:]), dims=2)

                #block_mean_square = block_total_square / (prod(size(variable)) / prod(size(dt)))
                @boundscheck prod(size(variable)) % prod(size(dt)) == 0
                block_npoints = prod(size(variable)) ÷ prod(size(dt))
                packed_block_results[:,1] /= block_npoints
                packed_block_results[:,3] /= block_npoints
            end

            gathered_block_results = MPI.Gather(packed_block_results, 0, comm_inter_block[])
        end
        if global_rank[] == 0
            # MPI.Gather returns a flattened Vector, so reshape back into nice Array
            gathered_block_results = reshape(gathered_block_results,
                                             (size(packed_results)..., n_blocks[]))

            if only_max_abs
                return maximum(gathered_block_results, dims=2)
            else
                return OrderedDict{String,mk_float}(
                           "RMS absolute residual"=>mean(@view(gathered_block_results[:,1,:]), dims=2),
                           "max absolute residual"=>maximum(@view(gathered_block_results[:,2,:]), dims=2),
                           "RMS relative residual"=>mean(@view(gathered_block_results[:,3,:]), dims=2),
                           "max relative residual"=>maximum(@view(gathered_block_results[:,4,:]), dims=2))
            end
        else
            return nothing
        end
    else
        if !only_max_abs && variable_max === nothing
            variable_max = maximum(variable)
        end
        reshaped_dt = reshape(dt, tuple((1 for _ ∈ 1:t_dim-1)..., size(dt)...))

        if only_max_abs
            absolute_residual =
                _steady_state_absolute_residual(variable, variable_at_previous_time, reshaped_dt)
            # Need to wrap the maximum(...) in a call to vec(...) so that we return a
            # Vector, not an N-dimensional array where the first (N-1) dimensions all have
            # size 1.
            return vec(maximum(absolute_residual; dims=tuple((1:t_dim-1)...)))
        else
            absolute_square_residual, relative_square_residual =
                _steady_state_square_residual(variable, variable_at_previous_time,
                                              reshaped_dt, epsilon, variable_max)
            # Need to wrap the mean(...) or maximum(...) in a call to vec(...) so that we
            # return a Vector, not an N-dimensional array where the first (N-1) dimensions all
            # have size 1.
            return OrderedDict{String,Vector{mk_float}}(
                       "RMS absolute residual"=>vec(mean(absolute_square_residual;
                                                         dims=tuple((1:t_dim-1)...))),
                       "max absolute residual"=>vec(maximum(absolute_square_residual;
                                                            dims=tuple((1:t_dim-1)...))),
                       "RMS relative residual"=>vec(mean(relative_square_residual;
                                                         dims=tuple((1:t_dim-1)...))),
                       "max relative residual"=>vec(maximum(relative_square_residual;
                                                            dims=tuple((1:t_dim-1)...))))
        end
    end
end

# Utility function for the steady-state square residual to avoid code duplication in
# steady_state_square_residuals()
function _steady_state_square_residual(variable, variable_at_previous_time, reshaped_dt,
                                       epsilon, variable_max)
    absolute_square_residual = @. ((variable - variable_at_previous_time) / reshaped_dt)^2
    relative_square_residual = @. absolute_square_residual /
                                  ((abs(variable) + epsilon*variable_max))^2
    return absolute_square_residual, relative_square_residual
end

# Utility function for the steady-state absolute residual to avoid code duplication in
# steady_state_mean_square_residual(), used only when only_max_abs=true
function _steady_state_absolute_residual(variable, variable_at_previous_time, reshaped_dt)
    absolute_residual = @. abs((variable - variable_at_previous_time) / reshaped_dt)
    return absolute_residual
end

"""
Get the unnormalised distribution function and unnormalised ('lab space') dzdt
coordinate at a point in space.

Inputs should depend only on vpa.
"""
function get_unnormalised_f_dzdt_1d(f, vpa_grid, density, upar, vth, evolve_density,
                                    evolve_upar, evolve_ppar)

    dzdt = vpagrid_to_dzdt(vpa_grid, vth, upar, evolve_ppar, evolve_upar)

    f_unnorm = get_unnormalised_f_1d(f, density, vth, evolve_density, evolve_ppar)

    return f_unnorm, dzdt
end
function get_unnormalised_f_1d(f, density, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        f_unnorm = @. f * density / vth
    elseif evolve_density
        f_unnorm = @. f * density
    else
        f_unnorm = f
    end
    return f_unnorm
end

"""
Get the unnormalised distribution function and unnormalised ('lab space') coordinates.

Inputs should depend only on z and vpa.
"""
function get_unnormalised_f_coords_2d(f, z_grid, vpa_grid, density, upar, vth,
                                      evolve_density, evolve_upar, evolve_ppar)

    nvpa, nz = size(f)
    z2d = zeros(nvpa, nz)
    for iz ∈ 1:nz
        z2d[:,iz] .= z_grid[iz]
    end
    dzdt2d = vpagrid_to_dzdt_2d(vpa_grid, vth, upar, evolve_ppar, evolve_upar)
    f_unnorm = get_unnormalised_f_2d(f, density, vth, evolve_density, evolve_ppar)

    return f_unnorm, z2d, dzdt2d
end
function vpagrid_to_dzdt_2d(vpa_grid, vth, upar, evolve_ppar, evolve_upar)
    nvpa = length(vpa_grid)
    nz = length(vth)
    dzdt2d = zeros(nvpa, nz)
    for iz ∈ 1:nz
        @views dzdt2d[:,iz] .= vpagrid_to_dzdt(vpa_grid, vth[iz], upar[iz], evolve_ppar,
                                               evolve_upar)
    end
    return dzdt2d
end
function get_unnormalised_f_2d(f, density, vth, evolve_density, evolve_ppar)
    f_unnorm = similar(f)
    nz = size(f, 2)
    for iz ∈ 1:nz
        @views f_unnorm[:,iz] .= get_unnormalised_f_1d(f[:,iz], density[iz], vth[iz],
                                                       evolve_density, evolve_ppar)
    end
    return f_unnorm
end

"""
Calculate a moving average

```
result[i] = mean(v[i-n:i+n])
```
Except near the ends of the array where indices outside the range of v are skipped.
"""
function moving_average(v::AbstractVector, n::mk_int)
    if length(v) < 2*n+1
        error("Cannot take moving average with n=$n on vector of length=$(length(v))")
    end
    result = similar(v)
    for i ∈ 1:n
        result[i] = mean(v[begin:i+n])
    end
    for i ∈ n+1:length(v)-n-1
        result[i] = mean(v[i-n:i+n])
    end
    for i ∈ length(v)-n:length(v)
        result[i] = mean(v[i-n:end])
    end
    return result
end

"""
Fit delta_phi to get the frequency and growth rate.

Note, expect the input to be a standing wave (as simulations are initialised with just a
density perturbation), so need to extract both frequency and growth rate from the
time-variation of the amplitude.

The function assumes that if the amplitude does not cross zero, then the mode is
non-oscillatory and so fits just an exponential, not exp*cos. The simulation used as
input should be long enough to contain at least ~1 period of oscillation if the mode is
oscillatory or the fit will not work.

Arguments
---------
z : Array{mk_float, 1}
    1d array of the grid point positions
t : Array{mk_float, 1}
    1d array of the time points
delta_phi : Array{mk_float, 2}
    2d array of the values of delta_phi(z, t)

Returns
-------
phi_fit_result struct whose fields are:
    growth_rate : mk_float
        Fitted growth rate of the mode
    amplitude0 : mk_float
        Fitted amplitude at t=0
    frequency : mk_float
        Fitted frequency of the mode
    offset0 : mk_float
        Fitted offset at t=0
    amplitude_fit_error : mk_float
        RMS error in fit to ln(amplitude) - i.e. ln(A)
    offset_fit_error : mk_float
        RMS error in fit to offset - i.e. δ
    cosine_fit_error : mk_float
        Maximum of the RMS errors of the cosine fits at each time point
    amplitude : Array{mk_float, 1}
        Values of amplitude from which growth_rate fit was calculated
    offset : Array{mk_float, 1}
        Values of offset from which frequency fit was calculated
"""
function fit_delta_phi_mode(t, z, delta_phi)
    # First fit a cosine to each time slice
    results = allocate_float(3, size(delta_phi)[2])
    amplitude_guess = 1.0
    offset_guess = 0.0
    for (i, phi_z) in enumerate(eachcol(delta_phi))
        results[:, i] .= fit_cosine(z, phi_z, amplitude_guess, offset_guess)
        (amplitude_guess, offset_guess) = results[1:2, i]
    end

    amplitude = results[1, :]
    offset = results[2, :]
    cosine_fit_error = results[3, :]

    L = z[end] - z[begin]

    # Choose initial amplitude to be positive, for convenience.
    if amplitude[1] < 0
        # 'Wrong sign' of amplitude is equivalent to a phase shift by π
        amplitude .*= -1.0
        offset .+= L / 2.0
    end

    # model for linear fits
    @. model(t, p) = p[1] * t + p[2]

    # Fit offset vs. time
    # Would give phase velocity for a travelling wave, but we expect either a standing
    # wave or a zero-frequency decaying mode, so expect the time variation of the offset
    # to be ≈0
    offset_fit = curve_fit(model, t, offset, [1.0, 0.0])
    doffsetdt = offset_fit.param[1]
    offset0 = offset_fit.param[2]
    offset_error = sqrt(mean(offset_fit.resid .^ 2))
    offset_tol = 2.e-5
    if abs(doffsetdt) > offset_tol
        println("WARNING: d(offset)/dt=", doffsetdt, " is non-negligible (>", offset_tol,
              ") but fit_delta_phi_mode expected either a standing wave or a ",
              "zero-frequency decaying mode.")
    end

    growth_rate = 0.0
    amplitude0 = 0.0
    frequency = 0.0
    phase = 0.0
    fit_error = 0.0
    if all(amplitude .> 0.0)
        # No zero crossing, so assume the mode is non-oscillatory (i.e. purely
        # growing/decaying).

        # Fit ln(amplitude) vs. time so we don't give extra weight to early time points
        amplitude_fit = curve_fit(model, t, log.(amplitude), [-1.0, 1.0])
        growth_rate = amplitude_fit.param[1]
        amplitude0 = exp(amplitude_fit.param[2])
        fit_error = sqrt(mean(amplitude_fit.resid .^ 2))
        frequency = 0.0
        phase = 0.0
    else
        converged = false
        maxiter = 100
        for iter ∈ 1:maxiter
            @views growth_rate_change, frequency, phase, fit_error =
                fit_phi0_vs_time(exp.(-growth_rate*t) .* amplitude, t)
            growth_rate += growth_rate_change
            println("growth_rate: ", growth_rate, "  growth_rate_change/growth_rate: ", growth_rate_change/growth_rate, "  fit_error: ", fit_error)
            if abs(growth_rate_change/growth_rate) < 1.0e-12 || fit_error < 1.0e-11
                converged = true
                break
            end
        end
        if !converged
            println("WARNING: Iteration to find growth rate failed to converge in ", maxiter, " iterations")
        end
        amplitude0 = amplitude[1] / cos(phase)
    end

    return (growth_rate=growth_rate, frequency=frequency, phase=phase,
            amplitude0=amplitude0, offset0=offset0, amplitude_fit_error=fit_error,
            offset_fit_error=offset_error, cosine_fit_error=maximum(cosine_fit_error),
            amplitude=amplitude, offset=offset)
end

"""
Fit a cosine to a 1d array

Fit function is A*cos(2*π*n*(z + δ)/L)

The domain z is taken to be periodic, with the first and last points identified, so
L=z[end]-z[begin]

Arguments
---------
z : Array
    1d array with positions of the grid points - should have the same length as data
data : Array
    1d array of the data to be fit
amplitude_guess : Float
    Initial guess for the amplitude (the value from the previous time point might be a
    good choice)
offset_guess : Float
    Initial guess for the offset (the value from the previous time point might be a good
    choice)
n : Int, default 1
    The periodicity used for the fit

Returns
-------
amplitude : Float
    The amplitude A of the cosine fit
offset : Float
    The offset δ of the cosine fit
error : Float
    The RMS of the difference between data and the fit
"""
function fit_cosine(z, data, amplitude_guess, offset_guess, n=1)
    # Length of domain
    L = z[end] - z[begin]

    @. model(z, p) = p[1] * cos(2*π*n*(z + p[2])/L)
    fit = curve_fit(model, z, data, [amplitude_guess, offset_guess])

    # calculate error
    error = sqrt(mean(fit.resid .^ 2))

    return fit.param[1], fit.param[2], error
end

function fit_phi0_vs_time(phi0, tmod)
    # the model we are fitting to the data is given by the function 'model':
    # assume phi(z0,t) = exp(γt)cos(ωt+φ) so that
    # phi(z0,t)/phi(z0,t0) = exp((t-t₀)γ)*cos((t-t₀)*ω + phase)/cos(phase),
    # where tmod = t-t0 and phase = ωt₀-φ
    @. model(t, p) = exp(p[1]*t) * cos(p[2]*t + p[3]) / cos(p[3])
    model_params = allocate_float(3)
    model_params[1] = -0.1
    model_params[2] = 8.6
    model_params[3] = 0.0
    @views fit = curve_fit(model, tmod, phi0/phi0[1], model_params)
    # get the confidence interval at 10% level for each fit parameter
    #se = standard_error(fit)
    #standard_deviation = Array{Float64,1}
    #@. standard_deviation = se * sqrt(size(tmod))

    fitted_function = model(tmod, fit.param)
    norm = moving_average(@.((abs(phi0/phi0[1]) + abs(fitted_function))^2), 1)
    fit_error = sqrt(mean(@.((phi0/phi0[1] - fitted_function)^2 / norm)))

    return fit.param[1], fit.param[2], fit.param[3], fit_error
end

end
