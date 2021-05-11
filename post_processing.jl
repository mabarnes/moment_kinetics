# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

module post_processing

export analyze_and_plot

# packages
using Plots
using LsqFit
# modules
using post_processing_input: pp
using quadrature: composite_simpson_weights
using array_allocation: allocate_float
using file_io: open_output_file
using type_definitions: mk_float
using load_data: open_netcdf_file
using load_data: load_coordinate_data, load_fields_data, load_moments_data, load_pdf_data
using analysis: analyze_fields_data, analyze_moments_data, analyze_pdf_data

function analyze_and_plot_data(path)
    # Create run_name from the path to the run directory
    path = realpath(path)
    run_name = joinpath(path, basename(path))
    # open the netcdf file and give it the handle 'fid'
    fid = open_netcdf_file(run_name)
    # load space-time coordinate data
    nz, z, z_wgts, Lz, nvpa, vpa, vpa_wgts, ntime, time = load_coordinate_data(fid)
    # initialise the post-processing input options
    nwrite_movie, itime_min, itime_max, iz0, ivpa0 = init_postprocessing_options(pp, nz, nvpa, ntime)
    # load fields data
    phi = load_fields_data(fid)
    # analyze the fields data
    phi_fldline_avg, delta_phi = analyze_fields_data(phi, ntime, nz, z_wgts, Lz)
    # use a fit to calculate and write to file the damping rate and growth rate of the
    # perturbed electrostatic potential
    frequency, growth_rate, phase, shifted_time =
        calculate_and_write_frequencies(fid, run_name, ntime, time, itime_min,
                                        itime_max, iz0, delta_phi, pp)
    # create the requested plots of the fields
    plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
                z, iz0, run_name, frequency, growth_rate, phase,
                shifted_time, pp)
    # load velocity moments data
    density, parallel_flow, parallel_pressure, n_species = load_moments_data(fid)
    # analyze the velocity moments data
    density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, delta_density, delta_upar, delta_ppar =
        analyze_moments_data(density, parallel_flow, parallel_pressure, ntime, n_species, nz, z_wgts, Lz)
    # create the requested plots of the moments
    plot_moments(density, delta_density, density_fldline_avg,
        parallel_flow, delta_upar, upar_fldline_avg,
        parallel_pressure, delta_ppar, ppar_fldline_avg,
        pp, run_name, time, itime_min, itime_max, nwrite_movie,
        z, iz0, n_species)
    # load particle distribution function (pdf) data
    ff = load_pdf_data(fid)
    # analyze the pdf data
    f_fldline_avg, delta_f, dens_moment = analyze_pdf_data(ff, nz, nvpa, n_species, ntime, z_wgts, Lz, vpa_wgts)
#=
    print("Loading distribution function data...")
    # define a handle for the distribution function
    cdfvar = fid["f"]
    # load the distribution function data
    ff = cdfvar.var[:,:,:,:]
    println("done.")
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
=#
    println("Plotting distribution function data...")
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    for is ∈ 1:n_species
        if n_species > 1
            spec_string = string("_spec", string(is))
        else
            spec_string = ""
        end
        # plot ∫dvpa f (= density unless density separately evolved, in which case, it should = 1)
        @views plot(time, 1.0 .- dens_moment[iz0,is,:])
        outfile = string(run_name, "_intf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        #fmin = minimum(ff[:,:,is,:])
        #fmax = maximum(ff[:,:,is,:])
        if pp.animate_f_vs_z_vpa
            # make a gif animation of ln f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(vpa, z, log.(abs.(ff[:,:,i])), xlabel="vpa", ylabel="z", clims = (fmin,fmax), c = :deep)
                @views heatmap(vpa, z, log.(abs.(ff[:,:,is,i])), xlabel="vpa", ylabel="z", fillcolor = logdeep)
            end
            outfile = string(run_name, "_logf_vs_z_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make a gif animation of f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(vpa, z, log.(abs.(ff[:,:,i])), xlabel="vpa", ylabel="z", clims = (fmin,fmax), c = :deep)
                @views heatmap(vpa, z, ff[:,:,is,i], xlabel="vpa", ylabel="z", c = :deep, interepolation = :cubic)
            end
            outfile = string(run_name, "_f_vs_z_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_z_vpa
            # make a gif animation of δf(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(vpa, z, delta_f[:,:,is,i], xlabel="vpa", ylabel="z", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_deltaf_vs_z_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_z_vpa0
            fmin = minimum(ff[:,ivpa0,is,:])
            fmax = maximum(ff[:,ivpa0,is,:])
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, ff[:,ivpa0,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_f_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_z_vpa0
            fmin = minimum(delta_f[:,ivpa0,is,:])
            fmax = maximum(delta_f[:,ivpa0,is,:])
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, delta_f[:,ivpa0,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_deltaf_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_z0_vpa
            fmin = minimum(ff[iz0,:,is,:])
            fmax = maximum(ff[iz0,:,is,:])
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(vpa, ff[iz0,:,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_f_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_z0_vpa
            fmin = minimum(delta_f[iz0,:,is,:])
            fmax = maximum(delta_f[iz0,:,is,:])
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(vpa, delta_f[iz0,:,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_deltaf_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")

    close(fid)

end
#=
function open_netcdf_file(run_name)
    # create the netcdf filename from the given run_name
    filename = string(run_name, ".cdf")

    print("Opening ", filename, " to read NetCDF data...")
    # open the netcdf file with given filename for reading
    fid = NCDataset(filename,"a")
    println("done.")

    return fid
end
function load_coordinate_data(fid)
    print("Loading coordinate data...")
    # define a handle for the z coordinate
    cdfvar = fid["z"]
    # get the number of z grid points
    nz = length(cdfvar)
    # load the data for z
    z = cdfvar.var[:]
    # get the weights associated with the z coordinate
    cdfvar = fid["z_wgts"]
    z_wgts = cdfvar.var[:]
    # Lz = z box length
    Lz = z[end]-z[1]

    # define a handle for the vpa coordinate
    cdfvar = fid["vpa"]
    # get the number of vpa grid points
    nvpa = length(cdfvar)
    # load the data for vpa
    vpa = cdfvar.var[:]
    # get the weights associated with the vpa coordinate
    cdfvar = fid["vpa_wgts"]
    vpa_wgts = cdfvar.var[:]

    # define a handle for the time coordinate
    cdfvar = fid["time"]
    # get the number of time grid points
    ntime = length(cdfvar)
    # load the data for time
    time = cdfvar.var[:]
    println("done.")

    return nz, z, z_wgts, Lz, nvpa, vpa, vpa_wgts, ntime, time
end
=#
function init_postprocessing_options(pp, nz, nvpa, ntime)
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
    # iz0 is the iz index used when plotting data at a single z location
    # by default, it will be set to cld(nz,3) unless a non-negative value provided
    if pp.iz0 > 0
        iz0 = pp.iz0
    else
        iz0 = cld(nz,3)
    end
    # ivpa0 is the iz index used when plotting data at a single vpa location
    # by default, it will be set to cld(nvpa,2) unless a non-negative value provided
    if pp.ivpa0 > 0
        ivpa0 = pp.ivpa0
    else
        ivpa0 = cld(nvpa,3)
    end
    println("done.")
    return nwrite_movie, itime_min, itime_max, iz0, ivpa0
end
#=
function load_fields_data(fid)
    print("Loading fields data...")
    # define a handle for the electrostatic potential
    cdfvar = fid["phi"]
    # load the electrostatic potential data
    phi = cdfvar.var[:,:]
    println("done.")
    return phi
end
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
=#
function calculate_and_write_frequencies(fid, run_name, ntime, time, itime_min, itime_max, iz0, delta_phi, pp)
    if pp.calculate_frequencies
        println("Calculating the frequency and damping/growth rate...")
        # shifted_time = t - t0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        # assume phi(z0,t) = A*exp(growth_rate*t)*cos(ω*t - φ)
        # and fit phi(z0,t)/phi(z0,t0), which eliminates the constant A pre-factor
        @views growth_rate, frequency, phase, fit_error =
            compute_frequencies(shifted_time[itime_min:itime_max], delta_phi[iz0,itime_min:itime_max])
        # estimate variation of frequency/growth rate over time_window by computing
        # values for each half of the time window and taking max/min
#=
        i1 = itime_min ; i2 = itime_min + cld(itime_max-itime_min,2)
        println("t1: ", time[i1], "  t2: ", time[i2])
        @views growth_rate_1, frequency_1, phase_1, _ =
            compute_frequencies(shifted_time[i1:i2], delta_phi[iz0,i1:i2])
        i1 = i2+1 ; i2 = itime_max
        println("t1: ", time[i1], "  t2: ", time[i2])
        @views growth_rate_2, frequency_2, phase_2, _ =
            compute_frequencies(shifted_time[i1:i2], delta_phi[iz0,i1:i2])
=#
        # use dummy values for growth_rate_x and frequency_x, as original calculation seems broken
        growth_rate_1 = growth_rate ; frequency_1 = frequency
        growth_rate_2 = growth_rate ; frequency_2 = frequency
        #
        growth_rate_max = max(growth_rate_1, growth_rate_2, growth_rate)
        growth_rate_min = min(growth_rate_1, growth_rate_2, growth_rate)
        frequency_max = max(frequency_1, frequency_2, frequency)
        frequency_min = min(frequency_1, frequency_2, frequency)
        # write info related to fit to file
        io = open_output_file(run_name, "frequency_fit.txt")
        println(io, "#growth_rate: ", growth_rate, "  min: ", growth_rate_min, "  max: ", growth_rate_max,
            "  frequency: ", frequency, "  min: ", frequency_min, "  max: ", frequency_max,
            "  phase: ", phase, " fit_error: ", fit_error)
        println(io)
        fit_phi = (delta_phi[iz0,itime_min]./cos(phase) .* exp.(growth_rate.*shifted_time)
                   .* cos.(frequency*shifted_time .+ phase))
        for i ∈ 1:ntime
            println(io, "time: ", time[i], "  delta_phi: ", delta_phi[iz0,i],
                    "  fit_phi: ", fit_phi[i])
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
        var[:] = growth_rate
        var = get_or_create("growth_rate_min","min growth rate from different fits")
        var[:] = growth_rate_min
        var = get_or_create("growth_rate_max","max growth rate from different fits")
        var[:] = growth_rate_max
        var = get_or_create("frequency","mode frequency from fit")
        var[:] = frequency
        var = get_or_create("frequency_min","min frequency from different fits")
        var[:] = frequency_min
        var = get_or_create("frequency_max","max frequency from different fits")
        var[:] = frequency_max
        var = get_or_create("phase","mode phase from fit")
        var[:] = phase
        var = get_or_create("delta_phi", "delta phi from simulation", ("nz", "ntime"))
        var[:,:] = delta_phi
        var = get_or_create("fit_phi","fit to delta phi", ("ntime",))
        var[:] = fit_phi
        var = get_or_create("fit_error","RMS error on the fit of phi")
        var[:] = fit_error
        println("done.")
    else
        frequency = 0.0
        growth_rate = 0.0
        phase = 0.0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
    end
    return frequency, growth_rate, phase, shifted_time
end
function plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
    z, iz0, run_name, frequency, growth_rate, phase, shifted_time, pp)

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
            plot!(time, abs.(delta_phi[iz0,itime_min]/cos(phase) * exp.(growth_rate*shifted_time)
                .* cos.(frequency*shifted_time .+ phase)))
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
    println("done.")
end
#=
function load_moments_data(fid)
    print("Loading velocity moments data...")
    # define a handle for the species density
    cdfvar = fid["density"]
    # load the species density data
    density = cdfvar.var[:,:,:]
    # define a handle for the species parallel flow
    cdfvar = fid["parallel_flow"]
    # load the species density data
    parallel_flow = cdfvar.var[:,:,:]
    # define a handle for the species parallel pressure
    cdfvar = fid["parallel_pressure"]
    # load the species density data
    parallel_pressure = cdfvar.var[:,:,:]
    # define the number of species
    n_species = size(cdfvar,2)
    println("done.")
    return density, parallel_flow, parallel_pressure, n_species
end
=#
#=
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
=#
function plot_moments(density, delta_density, density_fldline_avg,
    parallel_flow, delta_upar, upar_fldline_avg,
    parallel_pressure, delta_ppar, ppar_fldline_avg,
    pp, run_name, time, itime_min, itime_max, nwrite_movie, z, iz0, n_species)
    println("Plotting velocity moments data...")
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
#=
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
=#

function compute_frequencies(time_window, dphi)
    growth_rate = 0.0 ; frequency = 0.0 ; phase = 0.0 ; fit_error = 0.0
    for iter ∈ 1:10
        @views growth_rate_change, frequency, phase, fit_error =
            fit_phi0_vs_time(exp.(-growth_rate*time_window) .* dphi, time_window)
        growth_rate += growth_rate_change
        println("growth_rate: ", growth_rate, "  growth_rate_change/growth_rate: ", growth_rate_change/growth_rate)
        if abs(growth_rate_change/growth_rate) < 1.0e-8
            break
        end
    end
    return growth_rate, frequency, phase, fit_error
end

function fit_phi0_vs_time(phi0, tmod)
    # the model we are fitting to the data is given by the function 'model':
    # assume phi(z0,t) = exp(γt)cos(ωt-φ) so that
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
    fit_error = sqrt(sum((phi0/phi0[1] - fitted_function).^2
                         / max.(phi0/phi0[1], fitted_function).^2) / length(tmod))

    return fit.param[1], fit.param[2], fit.param[3], fit_error
end

#function advection_test_1d(fstart, fend)
#    rmserr = sqrt(sum((fend .- fstart).^2))/(size(fend,1)*size(fend,2)*size(fend,3))
#    println("advection_test_1d rms error: ", rmserr)
#end

end

if abspath(PROGRAM_FILE) == @__FILE__
    post_processing.analyze_and_plot_data(ARGS[1])
end
