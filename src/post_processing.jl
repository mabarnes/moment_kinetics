"""
"""
module post_processing

export analyze_and_plot_data

# Next three lines only used for workaround needed by plot_unnormalised()
using PyCall
import PyPlot

# packages
using Plots
using IJulia
using LsqFit
using NCDatasets
using Statistics: mean
using SpecialFunctions: erfi
#using LaTeXStrings
# modules
using ..post_processing_input: pp
using ..quadrature: composite_simpson_weights
using ..array_allocation: allocate_float
using ..file_io: open_output_file
using ..type_definitions: mk_float, mk_int
using ..initial_conditions: vpagrid_to_dzdt
using ..load_data: open_netcdf_file
using ..load_data: load_coordinate_data, load_fields_data, load_moments_data, load_pdf_data
using ..analysis: analyze_fields_data, analyze_moments_data, analyze_pdf_data
using ..velocity_moments: integrate_over_vspace

const default_compare_prefix = "comparison_plots/compare"

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
    get_tuple_of_return_values(func, arg_tuples...)

Suppose `func(args...)` returns a tuple of return values, then
`get_tuple_of_return_values(func, arg_tuples...)` returns a tuple (with an entry for each
return value of `func`) of tuples (one for each argument in each of `arg_tuples...`) of
return values.
"""
function get_tuple_of_return_values(func, arg_tuples...)

    if isempty(arg_tuples)
        return ()
    end
    n_args_tuple = Tuple(length(a) for a ∈ arg_tuples)
    if !all(n==n_args_tuple[1] for n ∈ n_args_tuple)
        error("All argument tuples passed to `get_tuple_of_return_values()` must have "
              * "the same length")
    end
    n_args = n_args_tuple[1]

    collected_args = Tuple(Tuple(a[i] for a ∈ arg_tuples) for i ∈ 1:n_args)

    wrong_way_tuple = Tuple(func(args...) for args ∈ collected_args)

    n_return_values = length(wrong_way_tuple[1])

    return Tuple(Tuple(wrong_way_tuple[i][j] for i ∈ 1:n_args)
                 for j ∈ 1:n_return_values)
end

"""
    analyze_and_plot_data(path...)

Make some plots for the simulation at `path`. If more than one argument is passed to
`path`, plot all the simulations together.

If a single value is passed for `path` the plots/movies are created in the same
directory as the run, and given names based on the name of the run. If multiple values
are passed, the plots/movies are given names beginning with `compare_` and are created
in the current directory.
"""
function analyze_and_plot_data(path...)
    # Create run_name from the path to the run directory
    run_names = Vector{String}(undef,0)
    for p ∈ path
        p = realpath(p)
        if isdir(p)
            push!(run_names, joinpath(p, basename(p)))
        else
            push!(run_names, splitext(p)[1])
        end
    end

    # open the netcdf file and give it the handle 'fid'
    nc_files = Tuple(open_netcdf_file(x) for x ∈ run_names)

    # Truncate to just the file names to make better titles
    run_labels = Tuple(basename(x) for x ∈ run_names)

    # load space-time coordinate data
    nvpa, vpa, vpa_wgts, nz, z, z_wgts, Lz, nr, r, r_wgts, Lr, ntime, time =
        get_tuple_of_return_values(load_coordinate_data, nc_files)

    # initialise the post-processing input options
    nwrite_movie, itime_min, itime_max, ivpa0, iz0, ir0 =
        init_postprocessing_options(pp, minimum(nvpa), minimum(nz), minimum(nr),
                                    minimum(ntime))
    # load full (z,r,t) fields data
    phi = Tuple(load_fields_data(f)[:,ir0,:] for f ∈ nc_files)

    # load full (z,r,species,t) velocity moments data
    density, parallel_flow, parallel_pressure, parallel_heat_flux,
        thermal_speed, n_species, evolve_density, evolve_upar, evolve_ppar =
        get_tuple_of_return_values(load_moments_data, nc_files)
    density = Tuple(n[:,ir0,:,:] for n ∈ density)
    parallel_flow = Tuple(upar[:,ir0,:,:] for upar ∈ parallel_flow)
    parallel_pressure = Tuple(ppar[:,ir0,:,:] for ppar ∈ parallel_pressure)
    parallel_heat_flux = Tuple(qpar[:,ir0,:,:] for qpar ∈ parallel_heat_flux)
    thermal_speed = Tuple(vth[:,ir0,:,:] for vth ∈ thermal_speed)

    # load full (vpa,z,r,species,t) particle distribution function (pdf) data
    ff = Tuple(load_pdf_data(f)[:,:,ir0,:,:] for f ∈ nc_files)

    #evaluate 1D-1V diagnostics at fixed ir0
    plot_1D_1V_diagnostics(run_names, run_labels, nc_files, nwrite_movie, itime_min,
        itime_max, ivpa0, iz0, ir0, r, phi, density, parallel_flow, parallel_pressure,
        parallel_heat_flux, thermal_speed, ff, n_species, evolve_density, evolve_upar,
        evolve_ppar, nvpa, vpa, vpa_wgts, nz, z, z_wgts, Lz, ntime, time)

    for f ∈ nc_files
        close(f)
    end

end

"""
"""
function init_postprocessing_options(pp, nvpa, nz, nr, ntime)
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
    # ivpa0 is the iz index used when plotting data at a single vpa location
    # by default, it will be set to cld(nvpa,3) unless a non-negative value provided
    if pp.ivpa0 > 0
        ivpa0 = pp.ivpa0
    else
        ivpa0 = cld(nvpa,3)
    end
    println("done.")
    return nwrite_movie, itime_min, itime_max, ivpa0, iz0, ir0
end

"""
"""
function plot_1D_1V_diagnostics(run_names, run_labels, nc_files, nwrite_movie,
        itime_min, itime_max, ivpa0, iz0, ir0, r, phi, density, parallel_flow,
        parallel_pressure, parallel_heat_flux, thermal_speed, ff, n_species,
        evolve_density, evolve_upar, evolve_ppar, nvpa, vpa, vpa_wgts, nz, z, z_wgts,
        Lz, ntime, time)

    n_runs = length(run_names)

    # plot_unnormalised() requires PyPlot, so ensure it is used for all plots for
    # consistency
    pyplot()

    # analyze the fields data
    phi_fldline_avg, delta_phi = get_tuple_of_return_values(analyze_fields_data, phi,
                                                            ntime, nz, z_wgts, Lz)

    function as_tuple(x)
        return Tuple(x for _ ∈ 1:n_runs)
    end

    # use a fit to calculate and write to file the damping rate and growth rate of the
    # perturbed electrostatic potential
    frequency, growth_rate, shifted_time, fitted_delta_phi =
        get_tuple_of_return_values(calculate_and_write_frequencies, nc_files, run_names,
            ntime, time, z, as_tuple(itime_min), as_tuple(itime_max), as_tuple(iz0),
            delta_phi, as_tuple(pp))
    # create the requested plots of the fields
    plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
                z, iz0, run_names, run_labels, fitted_delta_phi, pp)
    # load velocity moments data
    # analyze the velocity moments data
    density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, vth_fldline_avg, qpar_fldline_avg,
        delta_density, delta_upar, delta_ppar, delta_vth, delta_qpar =
        get_tuple_of_return_values(analyze_moments_data, density, parallel_flow,
            parallel_pressure, thermal_speed, parallel_heat_flux, ntime, n_species, nz,
            z_wgts, Lz)
    # create the requested plots of the moments
    plot_moments(density, delta_density, density_fldline_avg,
        parallel_flow, delta_upar, upar_fldline_avg,
        parallel_pressure, delta_ppar, ppar_fldline_avg,
        thermal_speed, delta_vth, vth_fldline_avg,
        parallel_heat_flux, delta_qpar, qpar_fldline_avg,
        pp, run_names, run_labels, time, itime_min, itime_max,
        nwrite_movie, z, iz0, n_species)
    # load particle distribution function (pdf) data
    # analyze the pdf data
    f_fldline_avg, delta_f, dens_moment, upar_moment, ppar_moment =
        get_tuple_of_return_values(analyze_pdf_data, ff, vpa, nvpa, nz, n_species,
            ntime, vpa_wgts, z_wgts, Lz, thermal_speed, evolve_ppar)

    println("Plotting distribution function data...")
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    n_species_max = maximum(n_species)
    if n_runs == 1
        prefix = run_names[1]
        legend = false
    else
        prefix = default_compare_prefix
        legend = true
    end
    for is ∈ 1:n_species_max
        if n_species_max > 1
            spec_string = string("_spec", string(is))
        else
            spec_string = ""
        end
        # plot difference between evolved density and ∫dvpa f; only possibly different if density removed from
        # normalised distribution function at run-time
        plot(legend=legend)
        for (t, n, n_int, run_label) ∈ zip(time, density, dens_moment, run_labels)
            @views plot!(t, n[iz0,is,:] .- n_int[iz0,is,:], label=run_label)
        end
        outfile = string(prefix, "_intf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        # if evolve_upar = true, plot ∫dwpa wpa * f, which should equal zero
        # otherwise, this plots ∫dvpa vpa * f, which is dens*upar
        plot(legend=legend)
        for (t, upar_int, run_label) ∈ zip(time, upar_moment, run_labels)
            intwf0_max = maximum(abs.(upar_int[iz0,is,:]))
            if intwf0_max < 1.0e-15
                @views plot!(t, upar_int[iz0,is,:], ylims = (-1.0e-15, 1.0e-15), label=run_label)
            else
                @views plot!(t, upar_int[iz0,is,:], label=run_label)
            end
        end
        outfile = string(prefix, "_intwf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        # plot difference between evolved parallel pressure and ∫dvpa vpa^2 f;
        # only possibly different if density and thermal speed removed from
        # normalised distribution function at run-time
        plot(legend=legend)
        for (t, ppar, ppar_int, run_label) ∈ zip(time, parallel_pressure, ppar_moment, run_labels)
            @views plot(t, ppar[iz0,is,:] .- ppar_int[iz0,is,:], label=run_label)
        end
        outfile = string(prefix, "_intw2f0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        #fmin = minimum(ff[:,:,is,:])
        #fmax = maximum(ff[:,:,is,:])
        if pp.animate_f_vs_vpa_z
            # make a gif animation of ln f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                subplots = (@views heatmap(this_z, this_vpa, log.(abs.(f[:,:,is,i])),
                                           xlabel="z", ylabel="vpa", fillcolor =
                                           logdeep, title=run_label)
                            for (f, this_z, this_vpa, run_label) ∈ zip(ff, z, vpa, run_labels))
                plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            end
            outfile = string(prefix, "_logf_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make a gif animation of f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                subplots = (@views heatmap(this_z, this_vpa, f[:,:,is,i], xlabel="z",
                                           ylabel="vpa", c = :deep, interpolation =
                                           :cubic, title=run_label)
                            for (f, this_z, this_vpa, run_label) ∈ zip(ff, z, vpa, run_labels))
                plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            end
            outfile = string(prefix, "_f_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make pdf of f(vpa,z,t_final) for each species
            str = string("spec ", string(is), " pdf")
            subplots = (@views heatmap(this_z, this_vpa, f[:,:,is,end], xlabel="vpa", ylabel="z",
                                       c = :deep, interpolation = :cubic,
                                       title=string(run_label, str))
                        for (f, this_z, this_vpa, run_label) ∈ zip(ff, z, vpa, run_labels))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_f_vs_z_vpa_final", spec_string, ".pdf")
            savefig(outfile)

            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (f, this_vpa, run_label) ∈ zip(ff, vpa, run_labels)
                    @views plot!(this_vpa, f[:,1,is,i], xlabel="vpa", ylabel="f(z=0)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_f0_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)

            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (f, this_vpa, run_label) ∈ zip(ff, vpa, run_labels)
                    @views plot!(this_vpa, f[:,end,is,i], xlabel="vpa", ylabel="f(z=L)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_fL_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_unnormalized
            ## The nice, commented out version will only work when plot_unnormalised can
            ## use Plots.jl...
            ## make a gif animation of f_unnorm(v_parallel_unnorm,z,t)
            #anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            #    subplots = (@views plot_unnormalised(f[:,:,is,i], this_z, this_vpa,
            #                           n[:,is,i], upar[:,is,i], vth[:,is,i], ev_n, ev_u,
            #                           ev_p, title=run_label)
            #                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_z,
            #                     this_vpa, run_label) ∈
            #                zip(ff, density, parallel_flow, thermal_speed,
            #                    evolve_density, evolve_upar, evolve_ppar, z, vpa,
            #                    run_labels))
            #    plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            #end
            #outfile = string(prefix, "_f_unnorm_vs_vpa_z", spec_string, ".gif")
            #gif(anim, outfile, fps=5)
            ## make a gif animation of log(f_unnorm)(v_parallel_unnorm,z,t)
            #anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            #    subplots = (@views plot_unnormalised(f[:,:,is,i], this_z, this_vpa, n[:,is,i],
            #                           upar[:,is,i], vth[:,is,i], ev_n, ev_u, ev_p,
            #                           plot_log=true, title=run_label)
            #                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_z,
            #                     this_vpa, run_label) ∈
            #                    zip(ff, density, parallel_flow, thermal_speed,
            #                        evolve_density, evolve_upar, evolve_ppar, z,
            #                        vpa, run_labels))
            #    plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            #end
            #outfile = string(prefix, "_logf_unnorm_vs_vpa_z", spec_string, ".gif")
            #gif(anim, outfile, fps=5)

            matplotlib = pyimport("matplotlib")
            matplotlib.use("agg")
            matplotlib_animation = pyimport("matplotlib.animation")
            iframes = collect(itime_min:nwrite_movie:itime_max)
            nframes = length(iframes)
            function make_frame(i)
                PyPlot.clf()
                iframe = iframes[i+1]
                # i counts from 0, Python-style
                for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                     run_label) ∈ zip(1:n_runs, ff, density, parallel_flow,
                                      thermal_speed, evolve_density, evolve_upar,
                                      evolve_ppar, z, vpa, run_labels)

                    PyPlot.subplot(1, n_runs, run_ind)
                    @views f_unnorm, z2d, dzdt2d = get_unnormalised_f_coords_2d(
                        f[:,:,is,iframe], this_z, this_vpa, n[:,is,iframe],
                        upar[:,is,iframe], vth[:,is,iframe], ev_n, ev_u, ev_p)
                    plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; title=run_label,
                                          plot_log=false)
                end
            end
            fig = PyPlot.figure(1, figsize=(6*n_runs,4))
            myanim = matplotlib_animation.FuncAnimation(fig, make_frame, frames=nframes)
            outfile = string(prefix, "_f_unnorm_vs_vpa_z", spec_string, ".gif")
            myanim.save(outfile, writer=matplotlib_animation.PillowWriter(fps=30))
            PyPlot.clf()

            function make_frame_log(i)
                PyPlot.clf()
                iframe = iframes[i+1]
                # i counts from 0, Python-style
                for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                     run_label) ∈ zip(1:n_runs, ff, density, parallel_flow,
                                      thermal_speed, evolve_density, evolve_upar,
                                      evolve_ppar, z, vpa, run_labels)

                    PyPlot.subplot(1, n_runs, run_ind)
                    @views f_unnorm, z2d, dzdt2d = get_unnormalised_f_coords_2d(
                        f[:,:,is,iframe], this_z, this_vpa, n[:,is,iframe],
                        upar[:,is,iframe], vth[:,is,iframe], ev_n, ev_u, ev_p)
                    plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; title=run_label,
                                          plot_log=true)
                end
            end
            fig = PyPlot.figure(figsize=(6*n_runs,4))
            myanim = matplotlib_animation.FuncAnimation(fig, make_frame_log, frames=nframes)
            outfile = string(prefix, "_logf_unnorm_vs_vpa_z", spec_string, ".gif")
            myanim.save(outfile, writer=matplotlib_animation.PillowWriter(fps=30))

            # Ensure PyPlot figure is cleared
            closeall()

            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_vpa, run_label) ∈
                    zip(ff, density, parallel_flow, thermal_speed, evolve_density,
                        evolve_upar, evolve_ppar, vpa, run_labels)
                    @views f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(
                        f[:,1,is,i], this_vpa, n[1,is,i], upar[1,is,i], vth[1,is,i],
                        ev_n, ev_u, ev_p)
                    @views plot!(dzdt, f_unnorm, xlabel="vpa", ylabel="f_unnorm(z=0)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_f0_unnorm_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)

            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_vpa, run_label) ∈
                    zip(ff, density, parallel_flow, thermal_speed, evolve_density,
                        evolve_upar, evolve_ppar, vpa, run_labels)
                    @views f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(
                        f[:,end,is,i], this_vpa, n[end,is,i], upar[end,is,i],
                        vth[end,is,i], ev_n, ev_u, ev_p)
                    @views plot!(dzdt, f_unnorm, xlabel="vpa", ylabel="f_unnorm(z=L)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_fL_unnorm_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z
            # make a gif animation of δf(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                subplots = (@views heatmap(this_z, this_vpa, delta_f[:,:,is,i],
                                       xlabel="z", ylabel="vpa", c = :deep,
                                       interpolation = :cubic, title=run_label)
                            for (df, this_z, this_vpa, run_label) ∈
                                zip(delta_f, z, vpa, run_labels))
                plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            end
            outfile = string(prefix, "_deltaf_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(minimum(f[ivpa0,:,is,:]) for f ∈ ff)
            fmax = maximum(maximum(f[ivpa0,:,is,:]) for f ∈ ff)
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (f, this_z, run_label) ∈ zip(ff, z, run_labels)
                    @views plot!(this_z, f[ivpa0,:,is,i], ylims = (fmin,fmax),
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_f_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(minimum(df[ivpa0,:,is,:]) for df ∈ delta_f)
            fmax = maximum(maximum(df[ivpa0,:,is,:]) for df ∈ delta_f)
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (df, this_z, run_label) ∈ zip(delta_f, z, run_labels)
                    @views plot!(this_z, df[ivpa0,:,is,i], ylims = (fmin,fmax),
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_deltaf_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(minimum(f[:,iz0,is,:]) for f ∈ ff)
            fmax = maximum(maximum(f[:,iz0,is,:]) for f ∈ ff)

            # if is == 1
            #     tmp = copy(ff)
            #     @. tmp[:,1,1,:] /= vpa^2
            #     bohm_integral = copy(time)
            #     for i ∈ 1:ntime
            #         @views bohm_integral[i] = integrate_over_vspace(tmp[1:cld(nvpa,2)-1,1,1,i],vpa_wgts[1:cld(nvpa,2)-1])/2.0
            #     end
            #     plot(time, bohm_integral, xlabel="time", label="Bohm integral")
            #     plot!(time, density[1,1,:], label="nᵢ(zmin)")
            #     outfile = string(prefix, "_Bohm_criterion.pdf")
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
                plot(legend=legend)
                for (f, this_vpa, run_label) ∈ zip(ff, vpa, run_labels)
                    @views plot!(this_vpa, f[:,iz0,is,i], label=run_label)
                end
            end
            outfile = string(prefix, "_f_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(minimum(df[:,iz0,is,:]) for df ∈ delta_f)
            fmax = maximum(maximum(df[:,iz0,is,:]) for df ∈ delta_f)
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (df, this_vpa, fn, fx, run_label) ∈
                        zip(delta_f, vpa, fmin, fmax, run_labels)
                    @views plot!(this_vpa, delta_f[:,iz0,is,i], ylims = (fn,fx),
                                label=run_label)
                end
            end
            outfile = string(prefix, "_deltaf_vs_vpa", spec_string, ".gif")
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
    z, iz0, run_names, run_labels, fitted_delta_phi, pp)

    println("Plotting fields data...")

    n_runs = length(run_names)
    if n_runs == 1
        prefix = run_names[1]
        legend = false
    else
        prefix = default_compare_prefix
        legend = true
    end

    phimin = minimum(minimum(p) for p ∈ phi)
    phimax = maximum(maximum(p) for p ∈ phi)

    if pp.plot_phi0_vs_t
        # plot the time trace of phi(z=z0)
        #plot(time, log.(phi[i,:]), yscale = :log10)
        plot(legend=legend)
        for (t, p, run_label) ∈ zip(time, phi, run_labels)
            @views plot!(t, p[iz0,:], label=run_label)
        end
        outfile = string(prefix, "_phi0_vs_t.pdf")
        savefig(outfile)
        plot(legend=legend)
        for (t, dp, fit, run_label) ∈ zip(time, delta_phi, fitted_delta_phi, run_labels)
            # plot the time trace of phi(z=z0)-phi_fldline_avg
            @views plot!(t, abs.(dp[iz0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log,
                         label="$run_label δϕ")
            if pp.calculate_frequencies
                plot!(t, abs.(fit), linestyle=:dash, label="$run_label fit")
            end
        end
        outfile = string(prefix, "_delta_phi0_vs_t.pdf")
        savefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(z,t)
        subplots = (heatmap(t, this_z, p, xlabel="time", ylabel="z", title=run_label, c =
                            :deep)
                    for (t, this_z, p, run_label) ∈ zip(time, z, phi, run_labels))
        plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
        outfile = string(prefix, "_phi_vs_z_t.pdf")
        savefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            plot(legend=legend)
            for (t, this_z, p, run_label) ∈ zip(time, z, phi, run_labels)
                @views plot!(this_z, p[:,i], xlabel="z", ylabel="ϕ",
                             ylims=(phimin, phimax), label=run_label)
            end
        end
        outfile = string(prefix, "_phi_vs_z.gif")
        gif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(prefix, "_harrison_comparison.pdf")
    # savefig(outfile)
    plot(legend=legend)
    for (t, this_z, p, run_label) ∈ zip(time, z, phi, run_labels)
        plot!(this_z, p[:,end], xlabel="z/Lz", ylabel="eϕ/Te", label=run_label,
             linewidth=2)
    end
    outfile = string(prefix, "_phi_final.pdf")
    savefig(outfile)

    println("done.")
end

"""
"""
function plot_moments(density, delta_density, density_fldline_avg,
    parallel_flow, delta_upar, upar_fldline_avg,
    parallel_pressure, delta_ppar, ppar_fldline_avg,
    thermal_speed, delta_vth, vth_fldline_avg,
    parallel_heat_flux, delta_qpar, qpar_fldline_avg,
    pp, run_names, run_labels, time, itime_min, itime_max, nwrite_movie,
    z, iz0, n_species)

    println("Plotting velocity moments data...")

    n_runs = length(run_names)
    if n_runs == 1
        prefix = run_names[1]
        legend = false
    else
        prefix = default_compare_prefix
        legend = true
    end

    # plot the species-summed, field-line averaged vs time
    denstot = Tuple(sum(n_fldline_avg,dims=1)[1,:]
                    for n_fldline_avg ∈ density_fldline_avg)
    for d in denstot
        d ./= d[1]
    end
    denstot_min = minimum(minimum(dtot) for dtot in denstot) - 0.1
    denstot_max = maximum(maximum(dtot) for dtot in denstot) + 0.1
    plot(legend=legend)
    for (t, dtot, run_label) ∈ zip(time, denstot, run_labels)
        @views plot!(t, dtot[1,:], ylims=(denstot_min,denstot_max), xlabel="time",
                     ylabel="∑ⱼn̅ⱼ(t)/∑ⱼn̅ⱼ(0)", linewidth=2, label=run_label)
    end
    outfile = string(prefix, "_denstot_vs_t.pdf")
    savefig(outfile)
    for is ∈ 1:maximum(n_species)
        spec_string = string(is)
        dens_min = minimum(minimum(n[:,is,:]) for n ∈ density)
        dens_max = maximum(maximum(n[:,is,:]) for n ∈ density)
        if pp.plot_dens0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, n, run_label) ∈ zip(time, density, run_labels)
                @views plot!(t, n[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_dens0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dn, run_label) ∈ zip(time, delta_density, run_labels)
                @views plot!(t, abs.(dn[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_delta_dens0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of density_fldline_avg
            plot(legend=legend)
            for (t, n_avg, run_label) ∈ zip(time, density_fldline_avg, run_labels)
                @views plot!(t, n_avg[is,:], xlabel="time", ylabel="<ns/Nₑ>",
                             ylims=(dens_min,dens_max), label=run_label)
            end
            outfile = string(prefix, "_fldline_avg_dens_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the deviation from conservation of density_fldline_avg
            plot(legend=legend)
            for (t, n_avg, run_label) ∈ zip(time, density_fldline_avg, run_labels)
                @views plot!(t, n_avg[is,:] .- n_avg[is,1], xlabel="time",
                             ylabel="<(ns-ns(0))/Nₑ>", label=run_label)
            end
            outfile = string(prefix, "_conservation_dens_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        upar_min = minimum(minimum(upar[:,is,:]) for upar ∈ parallel_flow)
        upar_max = maximum(maximum(upar[:,is,:]) for upar ∈ parallel_flow)
        if pp.plot_upar0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, upar, run_label) ∈ zip(time, parallel_flow, run_labels)
                @views plot!(t, upar[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_upar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dupar, run_label) ∈ zip(time, delta_upar, run_labels)
                @views plot!(t, abs.(du[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_delta_upar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            plot(legend=legend)
            for (t, upar_avg, run_label) ∈ zip(time, upar_fldline_avg, run_labels)
                @views plot!(t, upar_avg[is,:], xlabel="time",
                             ylabel="<upars/sqrt(2Te/ms)>", ylims=(upar_min,upar_max),
                             label=run_label)
            end
            outfile = string(prefix, "_fldline_avg_upar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        ppar_min = minimum(minimum(ppar[:,is,:]) for ppar ∈ parallel_pressure)
        ppar_max = maximum(maximum(ppar[:,is,:]) for ppar ∈ parallel_pressure)
        if pp.plot_ppar0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, ppar, run_label) ∈ zip(time, parallel_pressure, run_labels)
                @views plot!(t, ppar[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_ppar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dppar, run_label) ∈ zip(time, delta_ppar, run_labels)
                @views plot!(t, abs.(dppar[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_delta_ppar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            plot(legend=legend)
            for (t, ppar_avg, run_label) ∈ zip(time, ppar_fldline_avg, run_labels)
                @views plot!(t, ppar_avg[is,:], xlabel="time", ylabel="<ppars/NₑTₑ>",
                             ylims=(ppar_min,ppar_max), label=run_label)
            end
            outfile = string(prefix, "_fldline_avg_ppar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        vth_min = minimum(minimum(vth[:,is,:]) for vth ∈ thermal_speed)
        vth_max = maximum(maximum(vth[:,is,:]) for vth ∈ thermal_speed)
        if pp.plot_vth0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, vth, run_label) ∈ zip(time, thermal_speed, run_labels)
                @views plot!(t, vth[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_vth0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dvth, run_label) ∈ zip(time, delta_vth, run_labels)
                @views plot!(t, abs.(dvth[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_delta_vth0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of vth_fldline_avg
            plot(legend=legend)
            for (t, vth_avg, run_label) ∈ zip(time, vth_fldline_avg, run_labels)
                @views plot!(t, vth_avg[is,:], xlabel="time", ylabel="<vths/cₛ₀>",
                             ylims=(vth_min,vth_max), label=run_label)
            end
            outfile = string(prefix, "_fldline_avg_vth_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        qpar_min = minimum(minimum(qpar[:,is,:]) for qpar ∈ parallel_heat_flux)
        qpar_max = maximum(maximum(qpar[:,is,:]) for qpar ∈ parallel_heat_flux)
        if pp.plot_qpar0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, qpar, run_label) ∈ zip(time, parallel_heat_flux, run_labels)
                @views plot!(t, qpar[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_qpar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dqpar, run_label) ∈ zip(time, delta_qpar, run_labels)
                @views plot!(t, abs.(dqpar[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_delta_qpar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            plot(legend=legend)
            for (t, qpar_avg, run_label) ∈ zip(time, qpar_fldline_avg, run_labels)
                @views plot!(t, qpar_avg[is,:], xlabel="time", ylabel="<qpars/NₑTₑvth>",
                             ylims=(qpar_min,qpar_max), label=run_label)
            end
            outfile = string(prefix, "_fldline_avg_qpar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_dens_vs_z_t
            # make a heatmap plot of n_s(z,t)
            subplots = (heatmap(t, this_z, n[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, n, run_label) ∈ zip(time, z, density, run_labels))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_dens_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_upar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z, upar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, upar, run_label) ∈ zip(time, z, parallel_flow,
                                                              run_labels))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_upar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_ppar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z, ppar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, ppar, run_label) ∈
                            zip(time, z, parallel_pressure, run_labels))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_ppar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_qpar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z, qpar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, qpar, run_label) ∈
                            zip(time, z, parallel_heat_flux, run_labels))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_qpar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.animate_dens_vs_z
            # make a gif animation of ϕ(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, n, run_label) ∈ zip(time, z, density, run_labels)
                    @views plot!(this_z, n[:,is,i], xlabel="z", ylabel="nᵢ/Nₑ",
                                 ylims=(dens_min, dens_max), label=run_label)
                end
            end
            outfile = string(prefix, "_dens_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_upar_vs_z
            # make a gif animation of upar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, upar, run_label) ∈ zip(time, z, parallel_flow, run_labels)
                    @views plot!(this_z, upar[:,is,i], xlabel="z", ylabel="upars/vt",
                                 ylims=(upar_min, upar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_upar_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_ppar_vs_z
            # make a gif animation of ppar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, ppar, run_label) ∈ zip(time, z, parallel_pressure, run_labels)
                    @views plot!(this_z, ppar[:,is,i], xlabel="z", ylabel="ppars",
                                 ylims=(ppar_min, ppar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_ppar_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_vth_vs_z
            # make a gif animation of vth(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, vth, run_label) ∈ zip(time, z, thermal_speed, run_labels)
                    @views plot!(this_z, vth[:,is,i], xlabel="z", ylabel="vths",
                                 ylims=(vth_min, vth_max), label=run_label)
                end
            end
            outfile = string(prefix, "_vth_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_qpar_vs_z
            # make a gif animation of ppar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, qpar, run_label) ∈ zip(time, z, parallel_heat_flux,
                                                      run_labels)
                    @views plot!(this_z, qpar[:,is,i], xlabel="z", ylabel="qpars",
                                 ylims=(qpar_min, qpar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_qpar_vs_z_spec", spec_string, ".gif")
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

"""
Add a thin, red, dashed line showing v_parallel=(vth*w_parallel+upar)=0 to a 2d plot
with axes (z,vpa).
"""
function draw_v_parallel_zero!(plt::Plots.Plot, z::AbstractVector, upar, vth,
                               evolve_upar::Bool, evolve_ppar::Bool)
    if evolve_ppar && evolve_upar
        zero_value = @. -upar/vth
    elseif evolve_upar
        zero_value = @. -upar
    else
        zero_value = zeros(size(upar))
    end
    plot!(plt, z, zero_value, color=:red, linestyle=:dash, linewidth=1,
          xlims=xlims(plt), ylims=ylims(plt), label="")
end
function draw_v_parallel_zero!(z::AbstractVector, upar, vth, evolve_upar::Bool,
                               evolve_ppar::Bool)
    draw_v_parallel_zero!(Plots.CURRENT_PLOT, z, upar, vth, evolve_upar, evolve_ppar)
end

"""
Get the unnormalised distribution function and unnormalised ('lab space') dzdt
coordinate at a point in space.

Inputs should depend only on vpa.
"""
function get_unnormalised_f_dzdt_1d(f, vpa_grid, density, upar, vth, evolve_density,
                                    evolve_upar, evolve_ppar)

    dzdt = vpagrid_to_dzdt(vpa_grid, vth, upar, evolve_ppar, evolve_upar)

    if evolve_ppar
        f_unnorm = @. f * density / vth
    elseif evolve_density
        f_unnorm = @. f * density
    else
        f_unnorm = copy(f)
    end

    return f_unnorm, dzdt
end

"""
Get the unnormalised distribution function and unnormalised ('lab space') coordinates.

Inputs should depend only on z and vpa.
"""
function get_unnormalised_f_coords_2d(f, z_grid, vpa_grid, density, upar, vth,
                                      evolve_density, evolve_upar, evolve_ppar)

    nvpa, nz = size(f)
    z2d = zeros(nvpa, nz)
    dzdt2d = zeros(nvpa, nz)
    f_unnorm = similar(f)
    for iz ∈ 1:nz
        @views z2d[:,iz] .= z_grid[iz]
        f_unnorm[:,iz], dzdt2d[:,iz] =
            get_unnormalised_f_dzdt_1d(f[:,iz], vpa_grid, density[iz], upar[iz],
                                       vth[iz], evolve_density, evolve_upar,
                                       evolve_ppar)
    end

    return f_unnorm, z2d, dzdt2d
end

"""
Make a 2d plot of an unnormalised f on unnormalised coordinates, as returned from
get_unnormalised_f_coords()

Note this function requires using the PyPlot backend to support 2d coordinates being
passed to `heatmap()`.
"""
function plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; plot_log=false, kwargs...)

    if backend_name() != :pyplot
        error("PyPlot backend is required for plot_unnormalised(). Call pyplot() "
              * "first.")
    end

    ## The following commented out section does not work at the moment because
    ## Plots.heatmap() does not support 2d coordinates.
    ## https://github.com/JuliaPlots/Plots.jl/pull/4298 would add this feature...
    #if plot_log
    #    @. f_unnorm = log(abs(f_unnorm))
    #    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    #    cmap = cgrad(:deep, scale=:log) |> cmlog
    #else
    #    cmap = :deep
    #end

    #p = plot(; xlabel="z", ylabel="vpa", c=cmap, kwargs...)

    #heatmap(z2d, dzdt2d, f_unnorm)

    # Use PyPlot directly instead. Unfortunately makes animation a pain...
    if plot_log
        vmin = minimum(x for x in f_unnorm if x > 0.0)
        norm = PyPlot.matplotlib.colors.LogNorm(vmin=vmin, vmax=maximum(f_unnorm))
    else
        norm = nothing
    end
    p = PyPlot.pcolormesh(z2d, dzdt2d, f_unnorm; norm=norm, cmap="viridis_r")
    PyPlot.xlabel("z")
    PyPlot.ylabel("vpa")
    PyPlot.colorbar()

    return p
end

end
