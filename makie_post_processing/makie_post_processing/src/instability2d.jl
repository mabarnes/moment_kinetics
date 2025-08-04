using moment_kinetics.analysis: get_r_perturbation, get_Fourier_modes_2D,
                                get_Fourier_modes_1D
using moment_kinetics.interpolation: interpolate_to_grid_1d!

using LsqFit

"""
    instability_plots(run_info; plot_prefix)

Make all 2D instability plots.
"""
function instability2D_plots(run_info_moments, run_info_dfns; plot_prefix)
    has_rdim = any(ri !== nothing && ri.r.n > 1 for ri ∈ run_info_moments)

    if has_rdim
        # Plots for 2D instability do not make sense for 1D simulations
        instability_input = input_dict["instability2D"]
        if any((instability_input["plot_1d"], instability_input["plot_2d"],
                instability_input["animate_perturbations"]))
            # Get zind from the first variable in the loop (phi), and use the same one for
            # all subseqeunt variables.
            zind = Union{mk_int,Nothing}[nothing for _ ∈ run_info_moments]
            for variable_name ∈ ("phi", "density", "temperature")
                zind = instability2D_plots_for_variable(run_info_moments, variable_name;
                                                        plot_prefix=plot_prefix,
                                                        zind=zind)
            end
        end

        if instability_input["compare_perturbed_Maxwellian"]
            instability2D_compare_perturbed_Maxwellian(run_info_dfns;
                                                       plot_prefix=plot_prefix)
        end
    end

    return nothing
end

"""
    instability2D_plots_for_variable(run_info::Vector{Any}, variable_name; plot_prefix,
                                     zind=nothing)
    instability2D_plots_for_variable(run_info, variable_name; plot_prefix, zind=nothing,
                                     axes_and_observables=nothing)

Make plots of `variable_name` for analysis of 2D instability.

The information for the runs to analyse and plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, make plots comparing the runs, shown in
a horizontal row..

Settings are read from the `[instability2D]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

When `run_info` is not a Vector, `axes_and_observables` can be passed to add plots and
animations to existing figures, although this is not very convenient - see the use of this
argument when called from the `run_info::Vector{Any}` method.

If `zind` is not passed, it is calculated as the z-index where the mode seems to have
the maximum growth rate for this variable.
Returns `zind`.
"""
function instability2D_plots_for_variable end

function instability2D_plots_for_variable(run_info::Vector{Any}, variable_name;
                                          plot_prefix, zind=nothing)
    println("2D instability plots for $variable_name")
    flush(stdout)

    n_runs = length(run_info)
    var_symbol = get_variable_symbol(variable_name)
    instability2D_options = Dict_to_NamedTuple(input_dict["instability2D"])

    if zind === nothing
        zind = [nothing for _ in 1:n_runs]
    end

    if n_runs == 1
        # Don't need to set up for comparison plots, or include run_name in subplot titles
        zi = instability2D_plots_for_variable(run_info[1], variable_name,
                                              plot_prefix=plot_prefix, zind=zind[1])
        return Union{mk_int,Nothing}[zi]
    end

    figs = []
    axes_and_observables = [[] for _ ∈ 1:n_runs]
    if instability2D_options.plot_1d
        fig, ax = get_1d_ax(n_runs; title="$var_symbol 1D Fourier components", yscale=log10)
        push!(figs, fig)
        for (i, a) ∈ enumerate(ax)
            push!(axes_and_observables[i], a)
        end
        fig, ax = get_1d_ax(n_runs; title="phase of n_r=1 mode for $var_symbol")
        push!(figs, fig)
        for (i, a) ∈ enumerate(ax)
            push!(axes_and_observables[i], a)
        end
    else
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
    end
    if instability2D_options.plot_2d
        fig, ax = get_1d_ax(n_runs; title="$var_symbol Fourier components", yscale=log10)
        push!(figs, fig)
        for (i, a) ∈ enumerate(ax)
            push!(axes_and_observables[i], a)
        end
        frame_index = Observable(1)
        fig, ax, colorbar_places = get_2d_ax(n_runs; title="$var_symbol Fourier components")
        push!(figs, fig)
        for (i, (a, cb)) ∈ enumerate(zip(ax, colorbar_places))
            push!(axes_and_observables[i], (a, cb, frame_index))
        end

        # Delete any existing mode stats file so we can append to an empty file
        mode_stats_file_name = string(plot_prefix, "mode_$variable_name.txt")
        if isfile(mode_stats_file_name)
            rm(mode_stats_file_name)
        end
    else
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
    end
    if instability2D_options.animate_perturbations
        frame_index = Observable(1)
        fig, ax, colorbar_places = get_2d_ax(n_runs; title="$var_symbol perturbation")
        push!(figs, fig)
        for (i, (a, cb)) ∈ enumerate(zip(ax, colorbar_places))
            push!(axes_and_observables[i], (a, cb, frame_index))
        end
    else
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
    end

    for (i, (ri, ax_ob, zi)) ∈ enumerate(zip(run_info, axes_and_observables, zind))
        zi = instability2D_plots_for_variable(ri, variable_name, plot_prefix=plot_prefix,
                                              zind=zi, axes_and_observables=ax_ob)
        zind[i] = zi
    end

    fig = figs[1]
    if fig !== nothing
        outfile = string(plot_prefix, "$(variable_name)_1D_Fourier_components.pdf")
        save(outfile, fig)
    end

    fig = figs[2]
    if fig !== nothing
        outfile = string(plot_prefix, "$(variable_name)_1D_phase.pdf")
        save(outfile, fig)
    end

    fig = figs[3]
    if fig !== nothing
        outfile = string(plot_prefix, "$(variable_name)_Fourier_components.pdf")
        save(outfile, fig)
    end

    fig = figs[4]
    if fig !== nothing
        frame_index = axes_and_observables[1][4][3]
        nt = minimum(ri.nt for ri ∈ run_info)
        outfile = plot_prefix * variable_name * "_Fourier." *
                  instability2D_options.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    fig = figs[5]
    if fig !== nothing
        frame_index = axes_and_observables[1][5][3]
        nt = minimum(ri.nt for ri ∈ run_info)
        outfile = plot_prefix * variable_name * "_perturbation." *
                  instability2D_options.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    return zind
end

function instability2D_plots_for_variable(run_info, variable_name; plot_prefix,
                                          zind=nothing, axes_and_observables=nothing)
    instability2D_options = Dict_to_NamedTuple(input_dict["instability2D"])

    time = run_info.time

    variable = get_variable(run_info, variable_name)

    if ndims(variable) == 4
        # Only support single species runs in this routine, so pick is=1
        variable = @view variable[:,:,1,:]
    elseif ndims(variable) > 4
        error("Variables with velocity space dimensions not supported in "
              * "instability2D_plots_for_variable.")
    end

    if instability2D_options.plot_1d
        function unravel_phase!(phase::AbstractVector)
            # Remove jumps in phase where it crosses from -π to π
            for i ∈ 2:length(phase)
                if phase[i] - phase[i-1] > π
                    @views phase[i:end] .-= 2.0*π
                elseif phase[i] - phase[i-1] < -π
                    @views phase[i:end] .+= 2.0*π
                end
            end
        end
        function get_real_frequency(phase, time, amplitude)
            # Assume that once the amplitude reaches 2x initial amplitude that the mode is
            # well established, so will be able to measure phase velocity
            startind = findfirst(x -> x>2*amplitude[1], amplitude)
            if startind === nothing
                startind = 1
            end

            # Linear fit to phase after startind
            linear_model(x, param) = @. param[1]*x+param[2]
            fit = @views curve_fit(linear_model, time[startind:end], phase[startind:end],
                                   [0.0, 0.0])
            real_frequency = fit.param[1]
            phase_offset = fit.param[2]

            return real_frequency, phase_offset, startind
        end
        function get_growth_rate(amplitude, time)
            # Assume that once the amplitude reaches 2x initial amplitude that the mode is
            # well established, so will be able to measure phase velocity
            startind = findfirst(x -> x>2*amplitude[1], amplitude)
            if startind === nothing
                startind = 1
            end

            # Linear fit to log(amplitude) after startind
            growth_rate = 0.0
            initial_fit_amplitude = 1.0
            try
                linear_model(x, param) = @. param[1]*x+param[2]
                fit = @views curve_fit(linear_model, time[startind:end],
                                       log.(amplitude[startind:end]), [0.0, 0.0])
                growth_rate = fit.param[1]
                initial_fit_amplitude = exp(fit.param[2])
            catch e
                println("Warning: error $e when fitting growth rate")
            end

            return growth_rate, initial_fit_amplitude, startind
        end

        function plot_Fourier_1D(var, symbol, name)
            # File to save growth rate and frequency to
            if axes_and_observables === nothing
                mode_stats_file = open(string(plot_prefix, "mode_$name.txt"), "w")
            else
                # Processing multiple runs, so any existing mode_stats_file should have
                # already been deleted so that we can append in this function.
                mode_stats_file = open(string(plot_prefix, "mode_$name.txt"), "a")
                println(mode_stats_file, run_info.run_name)
                println(mode_stats_file, "-" ^ length(run_info.run_name))
            end

            amplitude = abs.(var)

            @views growth_rate, initial_fit_amplitude, startind =
                get_growth_rate(amplitude[2,:], time)

            # ikr=2 is the n_r=1 mode, so...
            kr_2 = 2.0*π/run_info.r.L
            println("for $symbol, kr=$kr_2, growth rate is $growth_rate")
            println(mode_stats_file, "kr = $kr_2")
            println(mode_stats_file, "growth_rate = $growth_rate")

            if axes_and_observables === nothing
                fig, ax = get_1d_ax(title="$symbol 1D Fourier components", xlabel="time",
                                    ylabel="amplitude", yscale=log10)
            else
                fig = nothing
                ax = axes_and_observables[1]
                ax.title = run_info.run_name
            end

            n_kr, nt = size(amplitude)

            # Drop constant mode (ikr=1) and aliased (?) modes >n_kr/2
            for ikr ∈ 2:n_kr÷2
                data = amplitude[ikr,:]
                data[data.==0.0] .= NaN
                plot_1d(time, data, ax=ax)
                text!(ax, position=(time[end], data[end]), "ikr=$ikr", fontsize=6,
                      justification=:right)
            end

            plot_1d(time, initial_fit_amplitude.*exp.(growth_rate.*time), ax=ax)
            vlines!(ax, [time[startind]], linestyle=:dot)

            if axes_and_observables === nothing
                outfile = string(plot_prefix, "$(name)_1D_Fourier_components.pdf")
                save(outfile, fig)
            end

            # Plot phase of n_r=1 mode
            phase = angle.(var[2,:])
            unravel_phase!(phase)

            # ikr=2 is the n_r=1 mode, so...
            omega_2, phase_offset, startind =
                get_real_frequency(phase, time, @view amplitude[2,:])

            phase_velocity_2 = omega_2 / kr_2

            println("for $symbol, kr=$kr_2, omega=$omega_2, phase velocity is $phase_velocity_2")
            println(mode_stats_file, "omega = $omega_2")

            if axes_and_observables === nothing
                fig, ax = get_1d_ax(title="phase of n_r=1 mode", xlabel="time",
                                    ylabel="phase")
            else
                fig = nothing
                ax = axes_and_observables[2]
                ax.title = run_info.run_name
            end

            plot_1d(time, phase, ax=ax, label="phase")
            plot_1d(time, phase_offset.+omega_2.*time, ax=ax, label="fit")
            vlines!(ax, [time[startind]], linestyle=:dot)
            axislegend(ax)

            if axes_and_observables === nothing
                outfile = string(plot_prefix, "$(name)_1D_phase.pdf")
                save(outfile, fig)
            end

            if axes_and_observables === nothing
                println(mode_stats_file, "")
            end
            close(mode_stats_file)
        end
        try
            variable_Fourier_1D, zind = get_Fourier_modes_1D(variable, run_info.r,
                                                             run_info.r_spectral, run_info.z,
                                                             zind=zind)
            plot_Fourier_1D(variable_Fourier_1D, get_variable_symbol(variable_name),
                            variable_name)
        catch e
            return makie_post_processing_error_handler(
                       e,
                       "Warning: error in 1D Fourier analysis for $variable_name.")
        end

        # Do this to allow memory to be garbage-collected.
        variable_Fourier_1D = nothing
    end

    if instability2D_options.plot_2d
        function plot_Fourier_2D(var, symbol, name)
            if axes_and_observables === nothing
                fig, ax = get_1d_ax(title="$symbol Fourier components", xlabel="time",
                                    ylabel="amplitude", yscale=log10)
            else
                fig = nothing
                ax = axes_and_observables[3]
                ax.title = run_info.run_name
            end

            n_kz, n_kr, nt = size(var)
            for ikr ∈ 1:n_kr, ikz ∈ 1:n_kz
                ikr!=2 && continue
                data = abs.(var[ikz,ikr,:])
                data[data.==0.0] .= NaN
                plot_1d(time, data, ax=ax)
                text!(ax, position=(time[end], data[end]), "ikr=$ikr, ikz=$ikz", fontsize=6,
                      justification=:right)
            end

            if axes_and_observables === nothing
                outfile = string(plot_prefix, "$(name)_Fourier_components.pdf")
                save(outfile, fig)
            end

            # make a gif animation of Fourier components
            if axes_and_observables === nothing
                ax = nothing
                colorbar_place = nothing
                frame_index = nothing
                outfile = plot_prefix * name * "_Fourier." * instability2D_options.animation_ext
                title = "$symbol Fourier components"
            else
                ax, colorbar_place, frame_index = axes_and_observables[4]
                outfile = nothing
                title = run_info.run_name
            end
            kr = collect(0:n_kr-1) * 2 * π / run_info.r.L
            kz = collect(0:n_kz-1) * 2 * π / run_info.z.L
            animate_2d(kz, kr, abs.(var), xlabel="kz", ylabel="kr",
                       title=title,
                       colormap=instability2D_options.colormap, colorscale=log10, ax=ax,
                       colorbar_place=colorbar_place, frame_index=frame_index,
                       outfile=outfile)
        end
        variable_Fourier = get_Fourier_modes_2D(variable, run_info.r, run_info.r_spectral,
                                                run_info.z, run_info.z_spectral)
        try
            plot_Fourier_2D(variable_Fourier, get_variable_symbol(variable_name),
                            variable_name)
        catch e
            return makie_post_processing_error_handler(
                       e,
                       "Warning: error in 2D Fourier analysis for $variable_name.")
        end

        # Do this to allow memory to be garbage-collected.
        variable_Fourier = nothing
    end

    if instability2D_options.animate_perturbations
        try
            _, perturbation = get_r_perturbation(variable)
            # make animation of perturbation
            if axes_and_observables === nothing
                ax = nothing
                colorbar_place = nothing
                frame_index = nothing
                outfile = plot_prefix*variable_name*"_perturbation." * instability2D_options.animation_ext
                title = "$(get_variable_symbol(variable_name)) perturbation"
            else
                ax, colorbar_place, frame_index = axes_and_observables[5]
                outfile = nothing
                title = run_info.run_name
            end
            animate_2d(run_info.z.grid, run_info.r.grid, perturbation, xlabel="z", ylabel="r",
                       title=title,
                       colormap=instability2D_options.colormap, ax=ax,
                       colorbar_place=colorbar_place, frame_index=frame_index,
                       outfile=outfile)
        catch e
            return makie_post_processing_error_handler(
                       e,
                       "Warning: error in perturbation animation for $variable_name.")
        end

        # Do this to allow memory to be garbage-collected (although this is redundant
        # here as this is the last thing in the function).
        perturbation = nothing
    end

    return zind
end

"""
    instability2D_compare_perturbed_Maxwellian(run_info_dfns;
                                               plot_prefix=plot_prefix)

Compare \$\\delta f\$ of the 2D instability, calculated by subtracting off the
\$r\$-average of the distribution function, to the perturbation to the local Maxwellian
given by the \$\\delta n\$, \$\\delta u_\\parallel\$, and \$\\delta T\$ of the
instability.
"""
function instability2D_compare_perturbed_Maxwellian end

function instability2D_compare_perturbed_Maxwellian(run_info_dfns::Vector{Any};
                                                    plot_prefix=plot_prefix)
    # Comparison plots not yet supported for this function.
    if length(run_info_dfns) > 1
        error("Comparison plots not yet supported in "
              * "instability2D_compare_perturbed_Maxwellian()")
    end
    return instability2D_compare_perturbed_Maxwellian(run_info_dfns[1];
                                                      plot_prefix=plot_prefix)
end

function instability2D_compare_perturbed_Maxwellian(run_info_dfns;
                                                    plot_prefix=plot_prefix)
    f_input = Dict_to_NamedTuple(input_dict["f"])
    is_1V = run_info_dfns.vperp.n == 1

    f = get_variable(run_info_dfns, "f_unnorm")

    vpa_unnorm = get_variable(run_info_dfns, "vpa_unnorm")
    vperp_unnorm = get_variable(run_info_dfns, "vperp_unnorm")

    vpamin, vpamax = extrema(vpa_unnorm)
    nvpa = run_info_dfns.vpa.n * 4
    vperpmax = maximum(vperp_unnorm)

    vpa_uniform = collect(LinRange(vpamin, vpamax, nvpa))
    if is_1V
        nvperp = 1
        vperp_uniform = zeros(mk_float, 1)
    else
        nvperp = run_info_dfns.vperp.n * 4
        vperp_uniform = collect(LinRange(0.0, vperpmax, nvperp))
    end

    n = get_variable(run_info_dfns, "density")
    n0, delta_n = get_r_perturbation(n)
    u = get_variable(run_info_dfns, "parallel_flow")
    u0, delta_u = get_r_perturbation(u)
    T = get_variable(run_info_dfns, "temperature")
    if is_1V
        # Want to work with parallel temperature for 1V case.
        T .*= 3.0
    end
    T0, delta_T = get_r_perturbation(T)

    # Interpolate to a high resolution (uniformly spaced) grid in unnormalised v_parallel
    vth = get_variable(run_info_dfns, "thermal_speed")
    size_f = size(f)
    if !is_1V
        newf = allocate_float(size_f[1], nvperp, size_f[3:end]...)
        for it ∈ 1:size_f[6], is ∈ 1:size_f[5], ir ∈ 1:size_f[4], iz ∈ 1:size_f[3]
            wperp_of_vperp_uniform = @. vperp_uniform / vth[iz,ir,is,it]
            for ivpa ∈ 1:size_f[1]
                @views interpolate_to_grid_1d!(newf[ivpa,:,iz,ir,is,it], wperp_of_vperp_uniform,
                                               f[ivpa,:,iz,ir,is,it], run_info_dfns.vperp,
                                               run_info_dfns.vperp_spectral)
            end
        end
        f = newf
    end
    newf = allocate_float(nvpa, size_f[2:end]...)
    for it ∈ 1:size_f[6], is ∈ 1:size_f[5], ir ∈ 1:size_f[4], iz ∈ 1:size_f[3]
        wpa_of_vpa_uniform = @. (vpa_uniform - u[iz,ir,is,it]) / vth[iz,ir,is,it]
        for ivperp ∈ 1:size_f[2]
            @views interpolate_to_grid_1d!(newf[:,ivperp,iz,ir,is,it], wpa_of_vpa_uniform,
                                           f[:,ivperp,iz,ir,is,it], run_info_dfns.vpa,
                                           run_info_dfns.vpa_spectral)
        end
    end
    f = newf

    f0, delta_f = get_r_perturbation(f)

    function add_vdims(x)
        return reshape(x, 1, 1, size(x)...)
    end
    n0 = add_vdims(n0)
    delta_n = add_vdims(delta_n)
    u0 = add_vdims(u0)
    delta_u = add_vdims(delta_u)
    T0 = add_vdims(T0)
    delta_T = add_vdims(delta_T)

    vT0 = @. sqrt(2.0 * T0)

    vperp_uniform_expanded = reshape(vperp_uniform, 1, nvperp, 1, 1, 1, 1)
    vpa_uniform_expanded = reshape(vpa_uniform, nvpa, 1, 1, 1, 1, 1)
    if is_1V
        f_M0 = @. n0 / sqrt(π) / vT0 * exp(-(vpa_uniform_expanded - u0)^2 / vT0^2)
        delta_f_M = @. f_M0 * (delta_n / n0
                               + 2.0 * (vpa_uniform_expanded - u0) * delta_u  / vT0^2
                               + ((vpa_uniform_expanded - u0)^2 / vT0^2 - 0.5) * delta_T / T0)
    else
        f_M0 = @. n0 / π^1.5 / vT0^3 * exp(-((vpa_uniform_expanded - u0)^2 + vperp_uniform_expanded^2) / vT0^2)
        delta_f_M = @. f_M0 * (delta_n / n0
                               + 2.0 * (vpa_uniform_expanded - u0) * delta_u  / vT0^2
                               + ((vpa_uniform_expanded - u0)^2 / vT0^2 - 1.5) * delta_T / T0)
    end

    it0 = f_input.it0
    ir0 = f_input.ir0
    iz0 = f_input.iz0
    ivperp0 = f_input.ivperp0

    fig, ax, colorbar_place = get_2d_ax(3; xlabel="vpa", ylabel="z")

    plot_2d(vpa_uniform, run_info_dfns.z.grid, delta_f[:,ivperp0,:,ir0,1,it0];
            title="delta_f", xlabel="vpa", ylabel="z", ax=ax[1],
            colorbar_place=colorbar_place[1])
    plot_2d(vpa_uniform, run_info_dfns.z.grid, delta_f_M[:,ivperp0,:,ir0,1,it0];
            title="delta_f_M", xlabel="vpa", ylabel="z", ax=ax[2],
            colorbar_place=colorbar_place[2])
    plot_2d(vpa_uniform, run_info_dfns.z.grid,
            delta_f[:,ivperp0,:,ir0,1,it0].-delta_f_M[:,ivperp0,:,ir0,1,it0];
            title="delta_f - delta_f_M", xlabel="vpa", ylabel="z", ax=ax[3],
            colorbar_place=colorbar_place[3])

    save(plot_prefix * "instability_perturbation_Maxwellian_difference_vs_vpa_z.pdf", fig)

    fig, ax, colorbar_place = get_2d_ax(3; xlabel="vpa", ylabel="z")

    plot_2d(vpa_uniform, run_info_dfns.z.grid, abs.(delta_f[:,ivperp0,:,ir0,1,it0]);
            title="delta_f", xlabel="vpa", ylabel="z", ax=ax[1],
            colorbar_place=colorbar_place[1], colorscale=log10,
            transform=x->positive_or_nan(x; epsilon=1.e-16))
    plot_2d(vpa_uniform, run_info_dfns.z.grid, abs.(delta_f_M[:,ivperp0,:,ir0,1,it0]);
            title="delta_f_M", xlabel="vpa", ylabel="z", ax=ax[2],
            colorbar_place=colorbar_place[2], colorscale=log10,
            transform=x->positive_or_nan(x; epsilon=1.e-16))
    plot_2d(vpa_uniform, run_info_dfns.z.grid,
            abs.(delta_f[:,ivperp0,:,ir0,1,it0].-delta_f_M[:,ivperp0,:,ir0,1,it0]);
            title="delta_f - delta_f_M", xlabel="vpa", ylabel="z", ax=ax[3],
            colorbar_place=colorbar_place[3], colorscale=log10,
            transform=x->positive_or_nan(x; epsilon=1.e-16))

    save(plot_prefix * "instability_perturbation_Maxwellian_difference_log_vs_vpa_z.pdf", fig)

    fig, ax, legend_place = get_1d_ax(; xlabel="vpa", get_legend_place=:below)

    plot_1d(vpa_uniform, delta_f[:,ivperp0,iz0,ir0,1,it0]; label="delta_f", ax=ax)
    plot_1d(vpa_uniform, delta_f_M[:,ivperp0,iz0,ir0,1,it0]; label="delta_f_M", ax=ax)
    plot_1d(vpa_uniform,
            delta_f[:,ivperp0,iz0,ir0,1,it0].-delta_f_M[:,ivperp0,iz0,ir0,1,it0];
            label="delta_f - delta_f_M", ax=ax)
    ax.ylabel = "f"
    Legend(legend_place, ax; tellheight=true, tellwidth=false)

    save(plot_prefix * "instability_perturbation_Maxwellian_difference_vs_vpa.pdf", fig)

    fig, ax, legend_place = get_1d_ax(; xlabel="vpa", get_legend_place=:below)

    plot_1d(vpa_uniform, abs.(delta_f[:,ivperp0,iz0,ir0,1,it0]); label="delta_f", ax=ax,
            yscale=log10, transform=x->positive_or_nan(x; epsilon=1.e-16))
    plot_1d(vpa_uniform, abs.(delta_f_M[:,ivperp0,iz0,ir0,1,it0]); label="delta_f_M",
            ax=ax, yscale=log10, transform=x->positive_or_nan(x; epsilon=1.e-16))
    plot_1d(vpa_uniform,
            abs.(delta_f[:,ivperp0,iz0,ir0,1,it0].-delta_f_M[:,ivperp0,iz0,ir0,1,it0]);
            label="delta_f - delta_f_M", ax=ax, yscale=log10,
            transform=x->positive_or_nan(x; epsilon=1.e-16))
    ax.ylabel = "f"
    Legend(legend_place, ax; tellheight=true, tellwidth=false)

    save(plot_prefix * "instability_perturbation_Maxwellian_difference_log_vs_vpa.pdf", fig)

    fig, ax, colorbar_place = get_2d_ax(3; xlabel="vpa", ylabel="z")
    frame_index = Observable(1)

    animate_2d(vpa_uniform, run_info_dfns.z.grid, delta_f[:,ivperp0,:,ir0,1,:];
               title="delta_f", xlabel="vpa", ylabel="z", ax=ax[1],
               frame_index=frame_index, colorbar_place=colorbar_place[1])
    animate_2d(vpa_uniform, run_info_dfns.z.grid,
               delta_f_M[:,ivperp0,:,ir0,1,:]; title="delta_f_M", xlabel="vpa",
               ylabel="z", ax=ax[2], frame_index=frame_index,
               colorbar_place=colorbar_place[2])
    animate_2d(vpa_uniform, run_info_dfns.z.grid,
               delta_f[:,ivperp0,:,ir0,1,:].-delta_f_M[:,ivperp0,:,ir0,1,:];
               title="delta_f - delta_f_M", xlabel="vpa", ylabel="z", ax=ax[3],
               frame_index=frame_index, colorbar_place=colorbar_place[3])

    save_animation(fig, frame_index, run_info_dfns.nt,
                   plot_prefix * "instability_perturbation_Maxwellian_difference_vs_vpa_z.gif")

    fig, ax, colorbar_place = get_2d_ax(3; xlabel="vpa", ylabel="z")
    frame_index = Observable(1)

    animate_2d(vpa_uniform, run_info_dfns.z.grid, abs.(delta_f[:,ivperp0,:,ir0,1,:]);
               title="delta_f", xlabel="vpa", ylabel="z", ax=ax[1],
               frame_index=frame_index, colorbar_place=colorbar_place[1],
               colorscale=log10, transform=x->positive_or_nan(x; epsilon=1.e-16))
    animate_2d(vpa_uniform, run_info_dfns.z.grid,
               abs.(delta_f_M[:,ivperp0,:,ir0,1,:]); title="delta_f_M", xlabel="vpa",
               ylabel="z", ax=ax[2], frame_index=frame_index,
               colorbar_place=colorbar_place[2], colorscale=log10,
               transform=x->positive_or_nan(x; epsilon=1.e-16))
    animate_2d(vpa_uniform, run_info_dfns.z.grid,
               abs.(delta_f[:,ivperp0,:,ir0,1,:].-delta_f_M[:,ivperp0,:,ir0,1,:]);
               title="delta_f - delta_f_M", xlabel="vpa", ylabel="z", ax=ax[3],
               frame_index=frame_index, colorbar_place=colorbar_place[3],
               colorscale=log10, transform=x->positive_or_nan(x; epsilon=1.e-16))

    save_animation(fig, frame_index, run_info_dfns.nt,
                   plot_prefix * "instability_perturbation_Maxwellian_difference_log_vs_vpa_z.gif")

    this_delta_f = delta_f[:,ivperp0,iz0,ir0,1,:]
    this_delta_f_M = delta_f_M[:,ivperp0,iz0,ir0,1,:]
    this_diff = this_delta_f .- this_delta_f_M

    fig, ax, legend_place = get_1d_ax(; xlabel="vpa", get_legend_place=:below)
    frame_index = Observable(1)

    plot_min = min.(vec(minimum(this_delta_f, dims=1)),
                    vec(minimum(this_delta_f_M, dims=1)),
                    vec(minimum(this_diff, dims=1)))
    plot_max = max.(vec(maximum(this_delta_f, dims=1)),
                    vec(maximum(this_delta_f_M, dims=1)),
                    vec(maximum(this_diff, dims=1)))

    animate_1d(vpa_uniform, this_delta_f; label="delta_f", ax=ax, frame_index=frame_index)
    animate_1d(vpa_uniform, this_delta_f_M; label="delta_f_M", ax=ax,
               frame_index=frame_index)
    animate_1d(vpa_uniform, this_diff; label="delta_f - delta_f_M",
               ylims=(plot_min, plot_max), ax=ax, frame_index=frame_index)
    ax.ylabel = "f"
    Legend(legend_place, ax; tellheight=true, tellwidth=false)

    save_animation(fig, frame_index, run_info_dfns.nt,
                   plot_prefix * "instability_perturbation_Maxwellian_difference_vs_vpa.gif")

    fig, ax, legend_place = get_1d_ax(; xlabel="vpa", get_legend_place=:below, yscale=log10)
    frame_index = Observable(1)

    plot_min = fill(mk_float(1.0e-16), run_info_dfns.nt)
    plot_max = max.(vec(maximum(this_delta_f, dims=1)),
                    vec(maximum(this_delta_f_M, dims=1)),
                    vec(maximum(this_diff, dims=1)))

    animate_1d(vpa_uniform, abs.(this_delta_f); label="delta_f", ax=ax,
               frame_index=frame_index, transform=x->positive_or_nan(x; epsilon=1.e-16))
    animate_1d(vpa_uniform, abs.(this_delta_f_M); label="delta_f_M", ax=ax,
               frame_index=frame_index, transform=x->positive_or_nan(x; epsilon=1.e-16))
    animate_1d(vpa_uniform, abs.(this_diff); label="delta_f - delta_f_M",
               ylims=(plot_min, plot_max), ax=ax, frame_index=frame_index,
               transform=x->positive_or_nan(x; epsilon=1.e-16))
    ax.ylabel = "f"
    Legend(legend_place, ax; tellheight=true, tellwidth=false)

    save_animation(fig, frame_index, run_info_dfns.nt,
                   plot_prefix * "instability_perturbation_Maxwellian_difference_log_vs_vpa.gif")

    return nothing
end
