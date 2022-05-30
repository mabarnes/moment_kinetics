"""
"""
module post_processing

export analyze_and_plot

# packages
using Plots
using IJulia
using LsqFit
using NCDatasets
using Statistics: mean
using SpecialFunctions: erfi
using LaTeXStrings
# modules
using ..post_processing_input: pp
using ..quadrature: composite_simpson_weights
using ..array_allocation: allocate_float
using ..file_io: open_output_file
using ..type_definitions: mk_float, mk_int
using ..load_data: open_netcdf_file
using ..load_data: load_coordinate_data, load_fields_data, load_moments_data, load_pdf_data
using ..analysis: analyze_fields_data, analyze_moments_data, analyze_pdf_data
using ..velocity_moments: integrate_over_vspace
using ..manufactured_solns: manufactured_solutions

using TOML
import Base: get

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
    L2_error_norm(a, b)

Calculate the L2 norm of the error between a and b: sqrt(mean((a-b)^2))
"""
function L2_error_norm(a, b)
    @assert size(a) == size(b)
    error = @. (a-b)
    return sqrt(mean(error.^2))
end

"""
    L_infinity_error_norm(a, b)

Calculate the L_infinity norm of the error between a and b: maximum(|a-b|)
"""
function L_infinity_error_norm(a, b)
    @assert size(a) == size(b)
    error = @. (a-b)
    return maximum(abs.(error))
end

"""
"""
function analyze_and_plot_data(path)
    # Create run_name from the path to the run directory
    path = realpath(path)
    run_name = joinpath(path, basename(path))
    input_filename = path * ".toml"
    scan_input = TOML.parsefile(input_filename)
        
    # open the netcdf file and give it the handle 'fid'
    fid = open_netcdf_file(run_name)
    # load space-time coordinate data
    nvpa, vpa, vpa_wgts, nvperp, vperp, vperp_wgts, nz, z, z_wgts, Lz, 
     nr, r, r_wgts, Lr, ntime, time = load_coordinate_data(fid)
    # initialise the post-processing input options
    nwrite_movie, itime_min, itime_max, ivpa0, ivperp0, iz0, ir0 = init_postprocessing_options(pp, nvpa, nvperp, nz, nr, ntime)
    # load full (z,r,t) fields data
    phi = load_fields_data(fid)
    # load full (z,r,species,t) velocity moments data
    density, parallel_flow, parallel_pressure, parallel_heat_flux,
        thermal_speed, n_species, evolve_ppar = load_moments_data(fid)
    # load full (vpa,vperp,z,r,species,t) particle distribution function (pdf) data
    ff = load_pdf_data(fid)
    
    #evaluate 1D-1V diagnostics at fixed ir0
    diagnostics_1d = false
    if diagnostics_1d
        plot_1D_1V_diagnostics(run_name, fid, nwrite_movie, itime_min, itime_max, ivpa0, iz0, ir0, r,
            phi[:,ir0,:], 
            density[:,ir0,:,:],
            parallel_flow[:,ir0,:,:],
            parallel_pressure[:,ir0,:,:],
            parallel_heat_flux[:,ir0,:,:],
            thermal_speed[:,ir0,:,:],
            ff[:,ivperp0,:,ir0,:,:],
            n_species, evolve_ppar, nvpa, vpa, vpa_wgts,
            nz, z, z_wgts, Lz, ntime, time)
    end 
    close(fid)
    
    # analyze the fields data
    phi_fldline_avg, delta_phi = analyze_fields_data(phi[iz0,:,:], ntime, nr, r_wgts, Lr)    
    plot_fields_rt(phi[iz0,:,:], delta_phi, time, itime_min, itime_max, nwrite_movie,
    r, ir0, run_name, delta_phi, pp)
    
    manufactured_solns_test = true
    # MRH hack condition on these plots for now
    # Plots compare density and density_symbolic at last timestep 
    if(manufactured_solns_test && nr > 1)
        r_bc = get(scan_input, "r_bc", "periodic")
        z_bc = get(scan_input, "z_bc", "periodic")
        dfni_func, densi_func = manufactured_solutions(Lr,Lz,r_bc,z_bc)
        
        is = 1
        spec_string = ""
        it = ntime
        heatmap(r, z, density[:,:,is,it], xlabel=L"r", ylabel=L"z", title=L"n_i/n_{ref}", c = :deep)
        outfile = string(run_name, "_dens_vs_r_z", spec_string, ".pdf")
        savefig(outfile)
        
        density_sym = copy(density[:,:,:,:])
        for ir in 1:nr
            for iz in 1:nz
                density_sym[iz,ir,is,it] = densi_func(z[iz],r[ir],time[it])
            end
        end
        heatmap(r, z, density_sym[:,:,is,it], xlabel=L"r", ylabel=L"z", title=L"n_i^{sym}/n_{ref}", c = :deep)
        outfile = string(run_name, "_dens_sym_vs_r_z", spec_string, ".pdf")
        savefig(outfile)
        
        density_norm = copy(density[1,1,1,:])
        for it in 1:ntime
            dummy = 0.0
            for ir in 1:nr
                for iz in 1:nz
                    dummy += (density[iz,ir,is,it] - densi_func(z[iz],r[ir],time[it]))^2
                end
            end
            density_norm[it] = dummy
        end
        println(density_norm)
        @views plot(time, density_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=L" \sum || n_i - n_i^{sym} ||^2") #, yaxis=:log)
        outfile = string(run_name, "_dens_norm_vs_t", spec_string, ".pdf")
        savefig(outfile)
    end 
    
    
end

"""
"""
function init_postprocessing_options(pp, nvpa, nvperp, nz, nr, ntime)
    print("Initializing the post-processing input options...")
    # nwrite_movie is the stride used when making animations
    nwrite_movie = pp.nwrite_movie
    # itime_min is the minimum time index at which to start animations
    if pp.itime_min > 0 && pp.itime_min <= ntime
        itime_min = pp.itime_min
    else
        itime_min = 1
    end
    # itime_max is the final time index at which to end animations
    # if itime_max < 0, the value used will be the total number of time slices
    if pp.itime_max > 0 && pp.itime_max <= ntime
        itime_max = pp.itime_max
    else
        itime_max = ntime
    end
    # ir0 is the ir index used when plotting data at a single r location
    # by default, it will be set to cld(nr,3) unless a non-negative value provided
    if pp.ir0 > 0
        ir0 = pp.ir0
    else
        ir0 = cld(nr,3)
    end
    # iz0 is the iz index used when plotting data at a single z location
    # by default, it will be set to cld(nz,3) unless a non-negative value provided
    if pp.iz0 > 0
        iz0 = pp.iz0
    else
        iz0 = cld(nz,3)
    end
    # ivperp0 is the iz index used when plotting data at a single vperp location
    # by default, it will be set to cld(nvperp,3) unless a non-negative value provided
    if pp.ivperp0 > 0
        ivperp0 = pp.ivperp0
    else
        ivperp0 = cld(nvperp,3)
    end
    # ivpa0 is the iz index used when plotting data at a single vpa location
    # by default, it will be set to cld(nvpa,3) unless a non-negative value provided
    if pp.ivpa0 > 0
        ivpa0 = pp.ivpa0
    else
        ivpa0 = cld(nvpa,3)
    end
    println("done.")
    return nwrite_movie, itime_min, itime_max, ivpa0, ivperp0, iz0, ir0
end

"""
"""
function plot_1D_1V_diagnostics(run_name, fid, nwrite_movie, itime_min, itime_max, ivpa0, iz0, ir0, r,
 phi, density, parallel_flow, parallel_pressure, parallel_heat_flux,
     thermal_speed, ff, n_species, evolve_ppar, nvpa, vpa, vpa_wgts,
                                nz, z, z_wgts, Lz, ntime, time)
    # analyze the fields data
    phi_fldline_avg, delta_phi = analyze_fields_data(phi, ntime, nz, z_wgts, Lz)
    # use a fit to calculate and write to file the damping rate and growth rate of the
    # perturbed electrostatic potential
    frequency, growth_rate, shifted_time, fitted_delta_phi =
        calculate_and_write_frequencies(fid, run_name, ntime, time, z, itime_min,
                                        itime_max, iz0, delta_phi, pp)
    # create the requested plots of the fields
    plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
                z, iz0, run_name, fitted_delta_phi, pp)
    # load velocity moments data
    # analyze the velocity moments data
    density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, qpar_fldline_avg,
        delta_density, delta_upar, delta_ppar, delta_qpar =
        analyze_moments_data(density, parallel_flow, parallel_pressure, parallel_heat_flux,
                             ntime, n_species, nz, z_wgts, Lz)
    # create the requested plots of the moments
    plot_moments(density, delta_density, density_fldline_avg,
        parallel_flow, delta_upar, upar_fldline_avg,
        parallel_pressure, delta_ppar, ppar_fldline_avg,
        parallel_heat_flux, delta_qpar, qpar_fldline_avg,
        pp, run_name, time, itime_min, itime_max,
        nwrite_movie, z, iz0, n_species)
    # load particle distribution function (pdf) data
    # analyze the pdf data
    f_fldline_avg, delta_f, dens_moment, upar_moment, ppar_moment =
        analyze_pdf_data(ff, vpa, nvpa, nz, n_species, ntime, vpa_wgts, z_wgts,
                         Lz, thermal_speed, evolve_ppar)

    println("Plotting distribution function data...")
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    for is ∈ 1:n_species
        if n_species > 1
            spec_string = string("_spec", string(is))
        else
            spec_string = ""
        end
        # plot difference between evolved density and ∫dvpa f; only possibly different if density removed from
        # normalised distribution function at run-time
        @views plot(time, density[iz0,is,:] .- dens_moment[iz0,is,:])
        outfile = string(run_name, "_intf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        # if evolve_upar = true, plot ∫dwpa wpa * f, which should equal zero
        # otherwise, this plots ∫dvpa vpa * f, which is dens*upar
        intwf0_max = maximum(abs.(upar_moment[iz0,is,:]))
        if intwf0_max < 1.0e-15
            @views plot(time, upar_moment[iz0,is,:], ylims = (-1.0e-15, 1.0e-15))
        else
            @views plot(time, upar_moment[iz0,is,:])
        end
        outfile = string(run_name, "_intwf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        # plot difference between evolved parallel pressure and ∫dvpa vpa^2 f;
        # only possibly different if density and thermal speed removed from
        # normalised distribution function at run-time
        @views plot(time, parallel_pressure[iz0,is,:] .- ppar_moment[iz0,is,:])
        outfile = string(run_name, "_intw2f0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        #fmin = minimum(ff[:,:,is,:])
        #fmax = maximum(ff[:,:,is,:])
        if pp.animate_f_vs_vpa_z
            # make a gif animation of ln f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                @views heatmap(z, vpa, log.(abs.(ff[:,:,is,i])), xlabel="z", ylabel="vpa", fillcolor = logdeep)
            end
            outfile = string(run_name, "_logf_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make a gif animation of f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                @views heatmap(z, vpa, ff[:,:,is,i], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_f_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make pdf of f(vpa,z,t_final) for each species
            str = string("spec ", string(is), " pdf")
            @views heatmap(z, vpa, ff[:,:,is,end], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic, title=str)
            outfile = string(run_name, "_f_vs_z_vpa_final", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.animate_deltaf_vs_vpa_z
            # make a gif animation of δf(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z, vpa, delta_f[:,:,is,i], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_deltaf_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(ff[ivpa0,:,is,:])
            fmax = maximum(ff[ivpa0,:,is,:])
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, ff[ivpa0,:,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_f_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(delta_f[ivpa0,:,is,:])
            fmax = maximum(delta_f[ivpa0,:,is,:])
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, delta_f[ivpa0,:,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_deltaf_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(ff[:,iz0,is,:])
            fmax = maximum(ff[:,iz0,is,:])

            # if is == 1
            #     tmp = copy(ff)
            #     @. tmp[:,1,1,:] /= vpa^2
            #     bohm_integral = copy(time)
            #     for i ∈ 1:ntime
            #         @views bohm_integral[i] = integrate_over_vspace(tmp[1:cld(nvpa,2)-1,1,1,i],vpa_wgts[1:cld(nvpa,2)-1])/2.0
            #     end
            #     plot(time, bohm_integral, xlabel="time", label="Bohm integral")
            #     plot!(time, density[1,1,:], label="nᵢ(zmin)")
            #     outfile = string(run_name, "_Bohm_criterion.pdf")
            #     savefig(outfile)
            #     println()
            #     if bohm_integral[end] <= density[1,1,end]
            #         println("Bohm criterion: ", bohm_integral[end], " <= ", density[1,1,end], " is satisfied!")
            #     else
            #         println("Bohm criterion: ", bohm_integral[end], " <= ", density[1,1,end], " is not satisfied!")
            #     end
            #     println()
            #     for j ∈ 0:10
            #         println("j: ", j, "  Bohm integral: ", integrate_over_vspace(tmp[1:cld(nvpa,2)-j,1,1,end],vpa_wgts[1:cld(nvpa,2)-j,end])/2.0)
            #     end
            # end
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #@views plot(vpa, ff[iz0,:,is,i], ylims = (fmin,fmax))
                @views plot(vpa, ff[:,iz0,is,i])
            end
            outfile = string(run_name, "_f_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(delta_f[:,iz0,is,:])
            fmax = maximum(delta_f[:,iz0,is,:])
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(vpa, delta_f[:,iz0,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_deltaf_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")

end 

"""
"""
function calculate_and_write_frequencies(fid, run_name, ntime, time, z, itime_min,
                                         itime_max, iz0, delta_phi, pp)
    if pp.calculate_frequencies
        println("Calculating the frequency and damping/growth rate...")
        # shifted_time = t - t0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        # assume phi(z0,t) = A*exp(growth_rate*t)*cos(ω*t + φ)
        # and fit phi(z0,t)/phi(z0,t0), which eliminates the constant A pre-factor
        @views phi_fit = fit_delta_phi_mode(shifted_time[itime_min:itime_max], z,
                                            delta_phi[:, itime_min:itime_max])
        frequency = phi_fit.frequency
        growth_rate = phi_fit.growth_rate

        # write info related to fit to file
        io = open_output_file(run_name, "frequency_fit.txt")
        println(io, "#growth_rate: ", phi_fit.growth_rate,
                "  frequency: ", phi_fit.frequency,
                " fit_errors: ", phi_fit.amplitude_fit_error, " ",
                phi_fit.offset_fit_error, " ", phi_fit.cosine_fit_error)
        println(io)

        # Calculate the fitted phi as a function of time at index iz0
        L = z[end] - z[begin]
        fitted_delta_phi =
            @. (phi_fit.amplitude0 * cos(2.0 * π * (z[iz0] + phi_fit.offset0) / L)
                * exp(phi_fit.growth_rate * shifted_time)
                * cos(phi_fit.frequency * shifted_time + phi_fit.phase))
        for i ∈ 1:ntime
            println(io, "time: ", time[i], "  delta_phi: ", delta_phi[iz0,i],
                    "  fitted_delta_phi: ", fitted_delta_phi[i])
        end
        close(io)
        # also save fit to NetCDF file
        function get_or_create(name, description, dims=())
            if name in fid
                return fid[name]
            else
                return defVar(fid, name, mk_float, dims,
                              attrib=Dict("description"=>description))
            end
        end
        var = get_or_create("growth_rate", "mode growth rate from fit")
        var[:] = phi_fit.growth_rate
        var = get_or_create("frequency","mode frequency from fit")
        var[:] = phi_fit.frequency
        var = get_or_create("delta_phi", "delta phi from simulation", ("nz", "ntime"))
        var[:,:] = delta_phi
        var = get_or_create("phi_amplitude", "amplitude of delta phi from fit over z",
                            ("ntime",))
        var[:,:] = phi_fit.amplitude
        var = get_or_create("phi_offset", "offset of delta phi from fit over z",
                            ("ntime",))
        var[:,:] = phi_fit.offset
        var = get_or_create("fitted_delta_phi","fit to delta phi", ("ntime",))
        var[:] = fitted_delta_phi
        var = get_or_create("amplitude_fit_error",
                            "RMS error on the fit of the ln(amplitude) of phi")
        var[:] = phi_fit.amplitude_fit_error
        var = get_or_create("offset_fit_error",
                            "RMS error on the fit of the offset of phi")
        var[:] = phi_fit.offset_fit_error
        var = get_or_create("cosine_fit_error",
                            "Maximum over time of the RMS error on the fit of a cosine "
                            * "to phi.")
        var[:] = phi_fit.cosine_fit_error
        println("done.")
    else
        frequency = 0.0
        growth_rate = 0.0
        phase = 0.0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        fitted_delta_phi = zeros(ntime)

    end
    return frequency, growth_rate, shifted_time, fitted_delta_phi
end

"""
"""
function plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
    z, iz0, run_name, fitted_delta_phi, pp)

    println("Plotting fields data...")
    phimin = minimum(phi)
    phimax = maximum(phi)
    if pp.plot_phi0_vs_t
        # plot the time trace of phi(z=z0)
        #plot(time, log.(phi[i,:]), yscale = :log10)
        @views plot(time, phi[iz0,:])
        outfile = string(run_name, "_phi0_vs_t.pdf")
        savefig(outfile)
        # plot the time trace of phi(z=z0)-phi_fldline_avg
        @views plot(time, abs.(delta_phi[iz0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log)
        if pp.calculate_frequencies
            plot!(time, abs.(fitted_delta_phi))
        end
        outfile = string(run_name, "_delta_phi0_vs_t.pdf")
        savefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(z,t)
        heatmap(time, z, phi, xlabel="time", ylabel="z", title="ϕ", c = :deep)
        outfile = string(run_name, "_phi_vs_z_t.pdf")
        savefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(z, phi[:,i], xlabel="z", ylabel="ϕ", ylims = (phimin,phimax))
        end
        outfile = string(run_name, "_phi_vs_z.gif")
        gif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(run_name, "_harrison_comparison.pdf")
    # savefig(outfile)
    plot(z, phi[:,end], xlabel="z/Lz", ylabel="eϕ/Te", label="", linewidth=2)
    outfile = string(run_name, "_phi_final.pdf")
    savefig(outfile)

    println("done.")
end

"""
"""
function plot_moments(density, delta_density, density_fldline_avg,
    parallel_flow, delta_upar, upar_fldline_avg,
    parallel_pressure, delta_ppar, ppar_fldline_avg,
    parallel_heat_flux, delta_qpar, qpar_fldline_avg,
    pp, run_name, time, itime_min, itime_max, nwrite_movie,
    z, iz0, n_species)
    println("Plotting velocity moments data...")
    # plot the species-summed, field-line averaged vs time
    denstot = copy(density_fldline_avg)
    denstot .= sum(density_fldline_avg,dims=1)
    @. denstot /= denstot[1,1]
    denstot_min = minimum(denstot[1,:]) - 0.1
    denstot_max = maximum(denstot[1,:]) + 0.1
    @views plot(time, denstot[1,:], ylims=(denstot_min,denstot_max), xlabel="time", ylabel="∑ⱼn̅ⱼ(t)/∑ⱼn̅ⱼ(0)", label="", linewidth=2)
    outfile = string(run_name, "_denstot_vs_t.pdf")
    savefig(outfile)
    for is ∈ 1:n_species
        spec_string = string(is)
        dens_min = minimum(density[:,is,:])
        dens_max = maximum(density[:,is,:])
        if pp.plot_dens0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, density[iz0,is,:])
            outfile = string(run_name, "_dens0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_density[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_dens0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of density_fldline_avg
            @views plot(time, density_fldline_avg[is,:], xlabel="time", ylabel="<ns/Nₑ>", ylims=(dens_min,dens_max))
            outfile = string(run_name, "_fldline_avg_dens_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the deviation from conservation of density_fldline_avg
            @views plot(time, density_fldline_avg[is,:] .- density_fldline_avg[is,1], xlabel="time", ylabel="<(ns-ns(0))/Nₑ>")
            outfile = string(run_name, "_conservation_dens_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        upar_min = minimum(parallel_flow[:,is,:])
        upar_max = maximum(parallel_flow[:,is,:])
        if pp.plot_upar0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, parallel_flow[iz0,is,:])
            outfile = string(run_name, "_upar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_upar[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_upar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            @views plot(time, upar_fldline_avg[is,:], xlabel="time", ylabel="<upars/sqrt(2Te/ms)>", ylims=(upar_min,upar_max))
            outfile = string(run_name, "_fldline_avg_upar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        ppar_min = minimum(parallel_pressure[:,is,:])
        ppar_max = maximum(parallel_pressure[:,is,:])
        if pp.plot_ppar0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, parallel_pressure[iz0,is,:])
            outfile = string(run_name, "_ppar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_ppar[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_ppar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            @views plot(time, ppar_fldline_avg[is,:], xlabel="time", ylabel="<ppars/NₑTₑ>", ylims=(ppar_min,ppar_max))
            outfile = string(run_name, "_fldline_avg_ppar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        qpar_min = minimum(parallel_heat_flux[:,is,:])
        qpar_max = maximum(parallel_heat_flux[:,is,:])
        if pp.plot_qpar0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, parallel_heat_flux[iz0,is,:])
            outfile = string(run_name, "_qpar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_qpar[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_qpar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            @views plot(time, qpar_fldline_avg[is,:], xlabel="time", ylabel="<qpars/NₑTₑvth>", ylims=(qpar_min,qpar_max))
            outfile = string(run_name, "_fldline_avg_qpar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_dens_vs_z_t
            # make a heatmap plot of n_s(z,t)
            heatmap(time, z, density[:,is,:], xlabel="time", ylabel="z", title="ns/Nₑ", c = :deep)
            outfile = string(run_name, "_dens_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_upar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            heatmap(time, z, parallel_flow[:,is,:], xlabel="time", ylabel="z", title="upars/vt", c = :deep)
            outfile = string(run_name, "_upar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_ppar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            heatmap(time, z, parallel_pressure[:,is,:], xlabel="time", ylabel="z", title="ppars/NₑTₑ", c = :deep)
            outfile = string(run_name, "_ppar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_qpar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            heatmap(time, z, parallel_heat_flux[:,is,:], xlabel="time", ylabel="z", title="qpars/NₑTₑvt", c = :deep)
            outfile = string(run_name, "_qpar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.animate_dens_vs_z
            # make a gif animation of ϕ(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, density[:,is,i], xlabel="z", ylabel="nᵢ/Nₑ", ylims = (dens_min,dens_max))
            end
            outfile = string(run_name, "_dens_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_upar_vs_z
            # make a gif animation of ϕ(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, parallel_flow[:,is,i], xlabel="z", ylabel="upars/vt", ylims = (upar_min,upar_max))
            end
            outfile = string(run_name, "_upar_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")
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
    growth_rate : mk_flaot
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

#function advection_test_1d(fstart, fend)
#    rmserr = sqrt(sum((fend .- fstart).^2))/(size(fend,1)*size(fend,2)*size(fend,3))
#    println("advection_test_1d rms error: ", rmserr)
#end

function plot_fields_rt(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
    r, ir0, run_name, fitted_delta_phi, pp)

    println("Plotting fields data...")
    phimin = minimum(phi)
    phimax = maximum(phi)
    if pp.plot_phi0_vs_t
        # plot the time trace of phi(r=r0)
        #plot(time, log.(phi[i,:]), yscale = :log10)
        @views plot(time, phi[ir0,:])
        outfile = string(run_name, "_phi(r0,z0)_vs_t.pdf")
        savefig(outfile)
        # plot the time trace of phi(r=r0)-phi_fldline_avg
        @views plot(time, abs.(delta_phi[ir0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log)
        if pp.calculate_frequencies
            plot!(time, abs.(fitted_delta_phi))
        end
        outfile = string(run_name, "_delta_phi(r0,z0)_vs_t.pdf")
        savefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(r,t)
        heatmap(time, r, phi, xlabel="time", ylabel="r", title="ϕ", c = :deep)
        outfile = string(run_name, "_phi_vs_r_t.pdf")
        savefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(r) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(r, phi[:,i], xlabel="r", ylabel="ϕ", ylims = (phimin,phimax))
        end
        outfile = string(run_name, "_phi_vs_r.gif")
        gif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(run_name, "_harrison_comparison.pdf")
    # savefig(outfile)
    plot(r, phi[:,end], xlabel="r/Lr", ylabel="eϕ/Te", label="", linewidth=2)
    outfile = string(run_name, "_phi(r)_final.pdf")
    savefig(outfile)

    println("done.")
end

end
