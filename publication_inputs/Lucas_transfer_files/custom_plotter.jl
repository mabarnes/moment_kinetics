"""
Custom plotter for anything you want on demand, using the plotting functions in makie_post_processing, without having to 
plot everything else in the post_processing_input.toml file.
"""

module custom_plotter

import makie_post_processing
import moment_kinetics

using moment_kinetics.load_data: get_z_derivative
using makie_post_processing: get_variable, get_1d_ax, plot_1d
using Makie: save, Legend
using Statistics: mean
using CairoMakie

export custom_plot

function calculate_f_i1(f_M, run_info, run_info_dfns, index)
    ppar = get_variable(run_info, "parallel_pressure")[index]
    temp = get_variable(run_info, "temperature")[index]
    vth = get_variable(run_info, "thermal_speed")[index]
    nu_ii = get_variable(run_info, "collision_frequency_ii")[index]
    dq_dz = get_z_derivative(run_info, "parallel_heat_flux")[index]
    dT_dz = get_variable(run_info, "dT_dz")[index]
    vpa_grid = run_info_dfns.vpa.grid
    f_i1 = zeros(size(vpa_grid))
    #@. f_i1 = - (f_M/nu_ii) * ( ((vpa_grid)/((vth^2)/2)) * dT_dz * ((vpa_grid)^2/(vth^2) - 3/2) + 2/1 * 1/ppar * dq_dz * (1/2 - (vpa_grid)^2/(vth^2)))
    @. f_i1 = - (f_M/nu_ii) * ( ((vpa_grid*vth)*2/(vth^2)) * (1/2*dT_dz) * ((vpa_grid*vth)^2/(vth^2) - 3/2) + 2/1 * 1/ppar * dq_dz * (1/2 - (vpa_grid*vth)^2/(vth^2)))
    return f_i1
end

function custom_plot(run_directory_with_runs::String)
    # make plot directory
    custom_plot_dir = "custom_plots"
    mkpath(custom_plot_dir)

    run_directory = run_directory_with_runs[6:end]
    run_directory_krook = "coll_krook" * run_directory[7:end]
    
    # get run info and variables to plot
    run_info_dkions = makie_post_processing.get_run_info_no_setup("runs/"*run_directory)
    run_info_dkions = (run_info_dkions,)
    if run_directory_krook in readdir("runs")
        run_info_krook = makie_post_processing.get_run_info_no_setup("runs/"*run_directory_krook)
        run_info_krook = (run_info_krook,)
    else
        println("warning: $run_directory_krook is not in this directory")
    end
    #println(run_info_dkions[1])

    external_source_amplitude = get_variable(run_info_dkions, "external_source_amplitude")
    mfp = get_variable(run_info_dkions, "mfp")[1]
    L_T = get_variable(run_info_dkions, "L_T")[1]
    L_n = get_variable(run_info_dkions, "L_n")[1]
    L_upar = get_variable(run_info_dkions, "L_upar")[1]
    println(run_directory, " is the plot prefix")
    upper_lim = 4000
    collisionality_T = L_T./mfp
    collisionality_n = L_n./mfp
    collisionality_upar = L_upar./mfp
    collisionality_T[collisionality_T .> upper_lim] .= NaN
    collisionality_n[collisionality_n .> upper_lim] .= NaN
    collisionality_upar[collisionality_upar .> upper_lim] .= NaN

    mean_nu_T = mean(replace(collisionality_T[:,1,1,end],NaN=>0.0))
    mean_nu_n = mean(replace(collisionality_n[:,1,1,end],NaN=>0.0))
    mean_nu_upar = mean(replace(collisionality_upar[:,1,1,end],NaN=>0.0))

    plot_sources = false
    plot_sources_own = true
    plot_collisionalities = false
    plot_density_comparisons = false
    plot_temperature_comparisons = false
    plot_f = true
    plot_individual_fs_along_z = false
    plot_qpar = false
    plot_temperature_residuals = false
    animate_f_diff_and_sources = true

    # plot variables
    if plot_sources
        fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
        plot_1d(run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,1,end], xlabel="z/m",
                    ylabel="amplitude/au", label="energy source", ax=ax[1], title = "Source Amplitudes")
        plot_1d(run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,2,end], label="particle source", ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
            orientation=:horizontal)
        outfile =  custom_plot_dir * "/" * run_directory * "external_source_amplitude_vs_z.pdf"
        save(outfile, fig)
    end
    if plot_sources_own
        fig = Figure(size = (500, 350))
        ax = Axis(fig[1, 1], title = "Source Amplitudes", ylabel="amplitude/au", xlabel="z/m", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        #ax = Axis(fig[1, 1], title = L"\nu^* \approx 10", titlesize = 28, xlabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        source1 = external_source_amplitude[1][:,1,1,end]
        source2 = external_source_amplitude[1][:,1,2,end]
        source1_line = lines!(fig[1, 1], run_info_dkions[1].z.grid*10, source1, label="energy source"; linewidth = 1.8)
        source2_line = lines!(fig[1, 1], run_info_dkions[1].z.grid*10, source2, label="particle source"; linewidth = 1.8)

        #legend = Legend(fig[1,2], [kinetic_line, fluid_line], ["Kinetic run", "Fluid run"], orientation = :horizontal)
        #legend.width = 200
        #legend.height = 35

        outfile =  custom_plot_dir * "/" * "both_sources.pdf"
        save(outfile, fig)
    end


    if plot_collisionalities
        fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
        plot_1d(run_info_dkions[1].z.grid*10, collisionality_T[:,1,1,end], xlabel="z/m",
                    ylabel="amplitude/au", label="collisionality_T", ax=ax[1], title = "Collisionalities")
        plot_1d(run_info_dkions[1].z.grid*10, collisionality_n[:,1,1,end], label="collisionality_n", ax=ax[1])
        plot_1d(run_info_dkions[1].z.grid*10, collisionality_upar[:,1,1,end], label="collisionality_upar", ax=ax[1])
        plot_1d(run_info_dkions[1].z.grid*10, mean_nu_T*ones(size(run_info_dkions[1].z.grid)), label="mean nu_T", ax=ax[1])
        plot_1d(run_info_dkions[1].z.grid*10, mean_nu_n*ones(size(run_info_dkions[1].z.grid)), label="mean nu_n", ax=ax[1])
        plot_1d(run_info_dkions[1].z.grid*10, mean_nu_upar*ones(size(run_info_dkions[1].z.grid)), label="mean nu_upar", ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
            orientation=:horizontal)
        outfile =  custom_plot_dir * "/" * run_directory * "collisionality_vs_z.pdf"
        save(outfile, fig)
    end


    if plot_density_comparisons
        fig = Figure(size = (500, 350))
        ax = Axis(fig[1, 1], title = L"\nu^* \approx 3", ylabel=L"\mathrm{n}/10^{19} \mathrm{m}^{-3}", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        #ax = Axis(fig[1, 1], title = L"\nu^* \approx 10", titlesize = 28, xlabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        density = get_variable(run_info_dkions, "density")
        kinetic_line = lines!(fig[1, 1], run_info_dkions[1].z.grid*10, density[1][:,1,1,end], label="Kinetic run"; color = :blue, linewidth = 1.8)
        density = get_variable(run_info_krook, "density")
        fluid_line = lines!(fig[1, 1], run_info_krook[1].z.grid*10, density[1][:,1,1,end], label="Fluid run"; color = :red, linewidth = 1.8)

        #legend = Legend(fig[1,2], [kinetic_line, fluid_line], ["Kinetic run", "Fluid run"], orientation = :horizontal)
        #legend.width = 200
        #legend.height = 35

        outfile =  custom_plot_dir * "/" * run_directory[7:end] * "coll_dkions_density_profile_comparison_vs_z_completed_plot.pdf"
        save(outfile, fig)
    end

    if plot_temperature_comparisons  
        fig = Figure(size = (500, 350))
        ax = Axis(fig[1, 1], xlabel=L"\mathrm{z/m}", ylabel=L"\mathrm{T}/100eV", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        #ax = Axis(fig[1, 1], xlabel=L"\mathrm{z/m}", titlesize = 28, xlabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        temperature = get_variable(run_info_dkions, "temperature")
        kinetic_line = lines!(fig[1, 1], run_info_dkions[1].z.grid*10, temperature[1][:,1,1,end], label="Kinetic run"; color = :blue, linewidth = 1.8)
        temperature = get_variable(run_info_krook, "temperature")
        fluid_line = lines!(fig[1, 1], run_info_krook[1].z.grid*10, temperature[1][:,1,1,end], label="Fluid run"; color = :red, linewidth = 1.8)

        outfile =  custom_plot_dir * "/" * run_directory[7:end] * "coll_dkions_temperature_profile_comparison_vs_z_completed_plot.pdf"
        save(outfile, fig)
    end

    if plot_f
        run_info_dfns = makie_post_processing.get_run_info_no_setup(run_directory_with_runs, dfns = true)

        f = get_variable(run_info_dfns, "f")
        f_65 = f[:,1,65,1,1,end]
        println("f size is ", size(f))

        fig = Figure(size = (500, 350))
        ax = Axis(fig[1, 1], xlabel=L"v_{∥}", ylabel=L"f", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20)
        #ax = Axis(fig[1, 1], xlabel=L"\mathrm{z/m}", titlesize = 28, xlabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        kinetic_line = lines!(fig[1, 1], run_info_dfns.vpa.grid*10, f_65, label="Kinetic run"; color = :blue, linewidth = 1.8)

        outfile =  custom_plot_dir * "/" * run_directory[7:end] * "dkions_f_65_profile.pdf"
        save(outfile, fig)
        if plot_individual_fs_along_z
            for z in 1:2:65

                vpa_squared = run_info_dfns.vpa.grid.^2
                f_M = exp.(-(vpa_squared))
                f_diff = f[:,1,z,1,1,end] .- f_M
                f_i1 = calculate_f_i1(f_M, run_info_dkions[1], run_info_dfns, z)

                fig = Figure(size = (1400, 700))
                #, ylabel=L"f"
                ax = Axis(fig[1, 1], xlabel=L"v_{∥}/v_{th}", ylabel=L"f", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, title = L"\nu^* \approx 1", limits = (-8.5, 8.5, nothing, nothing), xticks = -5:5:5)
                #ax = Axis(fig[1, 1], xlabel=L"\mathrm{z/m}", titlesize = 28, xlabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
                kinetic_line1 = lines!(fig[1, 1], run_info_dfns.vpa.grid, f_diff, label=L"\text{Actual } f_i - f_{Mi}"; color = :black, linewidth = 1.8)
                kinetic_line2 = lines!(fig[1, 1], run_info_dfns.vpa.grid, f_i1, label=L"\text{Calculated } f_{i1}"; color = :lightseagreen, linewidth = 1.8)
                #axislegend()
                #axislegend("Titled Legend", position = :rt)
                #axislegend(ax, [kinetic_line1, kinetic_line2], ["Actual f_i - f_{Mi}", "Calculated f_{i1}"], orientation = :horizontal)
                if plot_sources
                    ax = Axis(fig[1, 2], xlabel=L"v_{∥}/v_{th}", ylabel="External source amplitude/au", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, title = L"\nu^* \approx 1")
                    #plot_1d(run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,1,end], xlabel="z/m",
                    #            ylabel="amplitude/au", label="energy source", ax=ax[1], title = "Source Amplitudes")
                    #plot_1d(run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,2,end], label="particle source", ax=ax[1])
                    kinetic_line3 = lines!(fig[1, 2], run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,1,end], label=L"\text{Actual } f_i - f_{Mi}"; linewidth = 1.8)
                    kinetic_line4 = lines!(fig[1, 2], run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,2,end], label=L"\text{Actual } f_i - f_{Mi}"; linewidth = 1.8)
                    z_position = run_info_dkions[1].z.grid[z]*10
                    arrow = arrows!(fig[1,2], [z_position], [-1.4], [0.0], [3.0], arrowsize = 10, lengthscale = 0.3, arrowcolor = :black)
                    #Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                    #    orientation=:horizontal)

                end
                #Legend(fig[1,2], [kinetic_line1, kinetic_line2], [L"\text{Actual } f_i - f_{Mi}", L"\text{Calculated } f_{i1}"], orientation = :horizontal)

                outfile =  custom_plot_dir * "/" * run_directory * "dkions_f_diff_and_f_i1_profile_and_sources_$z.pdf"
                save(outfile, fig)
            end
        end

        if animate_f_diff_and_sources
            z = Observable(1)
            framerate = 5
            timestamps = range(1, 65, step=1)
            vpa_squared = run_info_dfns.vpa.grid.^2
            f_M = exp.(-(vpa_squared))
            f_diff = @lift(f[:,1,$z,1,1,end] .- f_M)
            f_i1 = @lift(calculate_f_i1(f_M, run_info_dkions[1], run_info_dfns, $z))
            z_values = run_info_dkions[1].z.grid*10
            z_position = @lift([z_values[$z]])
            #println(z_position)
            #println(z_position[1])
            fig = Figure(size = (950, 475))
            #, ylabel=L"f"
            ax = Axis(fig[1, 1], xlabel=L"v_{∥}/v_{th}", ylabel=L"f", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, xticks = -5:5:5, limits = (-8.5, 8.5, -0.015, 0.015))
            #ax = Axis(fig[1, 1], xlabel=L"\mathrm{z/m}", titlesize = 28, xlabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
            kinetic_line1 = lines!(fig[1, 1], run_info_dfns.vpa.grid, f_diff, label=L"\text{Actual } f_i - f_{Mi}"; color = :black, linewidth = 1.8)
            kinetic_line2 = lines!(fig[1, 1], run_info_dfns.vpa.grid, f_i1, label=L"\text{Collisional } f_{i1}"; color = :lightseagreen, linewidth = 1.8)
            axislegend()
            ax = Axis(fig[1, 2], xlabel="z/m", ylabel="External source amplitude/au", titlesize = 28, xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20)

            kinetic_line3 = lines!(fig[1, 2], run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,1,5]*20.0, label="Energy source"; linewidth = 1.8)
            kinetic_line4 = lines!(fig[1, 2], run_info_dkions[1].z.grid*10, external_source_amplitude[1][:,1,2,5], label="Particle source"; linewidth = 1.8)
            axislegend(position = :ct)
            arrow = arrows!(fig[1,2], z_position, [-200.4], [0.0], [350.0], arrowsize = 10, lengthscale = 0.3, arrowcolor = :black)

            record(fig, custom_plot_dir * "/" * run_directory * "dkions_f_diff_and_f_i1_with_sources.mp4", timestamps;
                    framerate = framerate) do t
                z[] = t
                autolimits!(ax)
            end
        end
    end


    if plot_qpar
        qpar_conversion_from_ref_to_Wm2e6 = 15.68463642

        qpar = get_variable(run_info_dkions, "parallel_heat_flux")[1][:,1,1,end] * qpar_conversion_from_ref_to_Wm2e6
        coll_krook_heat_flux = get_variable(run_info_dkions, "coll_krook_heat_flux")[1][:,1,1,end] * qpar_conversion_from_ref_to_Wm2e6
        fig = Figure(size = (500, 400))
        #ylabel=L"\mathrm{q_{\parallel}}/ \mathrm{M Wm}^{-2}",
        ax = Axis(fig[1, 1], xlabel=L"\mathrm{z/m}", titlesize = 28, ylabel=L"\mathrm{q_{\parallel}}/ \mathrm{M Wm}^{-2}", xlabelsize = 28, title = L"\nu^* \approx 500", ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, limits = (-5.1, 5.1, nothing, nothing), xticks = -5:5:5)
        qpar_line = lines!(fig[1, 1], run_info_dkions[1].z.grid*10, qpar, label=L"\text{Kinetic } q_∥"; color = :blue, linewidth = 1.8)
        coll_krook_line = lines!(fig[1, 1], run_info_dkions[1].z.grid*10, coll_krook_heat_flux, label=L"\text{Coll_krook } q_∥"; color = :red, linewidth = 1.8)
        #Legend(fig[1,2], [qpar_line, coll_krook_line], ["Kinetic heat flux", "Fluid closure heat flux"], orientation = :horizontal)
        outfile =  custom_plot_dir * "/" * run_directory * "dkions_qpar_inWatts_profile_comparison_vs_z_completed_plot.pdf"
        save(outfile, fig)
    
    end

    if plot_temperature_residuals
        base_name_coll_krook = run_directory_krook[1:43]
        base_name_dkions = run_directory[1:39]
        density_list = []
        temperature_residual_list = []
        
        for n in 5:5:100
            this_run_directory_krook = base_name_coll_krook * "$n.0_final"
            this_run_directory = base_name_dkions * "$n.0_final"
            this_run_info_dkions = makie_post_processing.get_run_info_no_setup("runs/"*this_run_directory)
            this_run_info_dkions = (this_run_info_dkions,)
            this_run_info_krook = makie_post_processing.get_run_info_no_setup("runs/"*this_run_directory_krook)
            this_run_info_krook = (this_run_info_krook,)
            density_midpoint = get_variable(this_run_info_dkions, "density")[1][65,1,1,end]

            temperature_dkions = get_variable(this_run_info_dkions, "temperature")
            temperature_krook = get_variable(this_run_info_krook, "temperature")
            temperature_residuals = temperature_dkions[1][:,1,1,end] - temperature_krook[1][:,1,1,end]
            temp_residual_sum = sum(abs.(temperature_residuals))
            push!(temperature_residual_list, temp_residual_sum)
            push!(density_list, density_midpoint)
        end

        for n in 500:500:3000
            this_run_directory_krook = base_name_coll_krook * "$n.0_final"
            this_run_directory = base_name_dkions * "$n.0_final"
            this_run_info_dkions = makie_post_processing.get_run_info_no_setup("runs/"*this_run_directory)
            this_run_info_dkions = (this_run_info_dkions,)
            this_run_info_krook = makie_post_processing.get_run_info_no_setup("runs/"*this_run_directory_krook)
            this_run_info_krook = (this_run_info_krook,)
            density_midpoint = get_variable(this_run_info_dkions, "density")[1][65,1,1,end]

            temperature_dkions = get_variable(this_run_info_dkions, "temperature")
            temperature_krook = get_variable(this_run_info_krook, "temperature")
            temperature_residuals = temperature_dkions[1][:,1,1,end] - temperature_krook[1][:,1,1,end]
            temp_residual_sum = sum(abs.(temperature_residuals))
            push!(temperature_residual_list, temp_residual_sum)
            push!(density_list, density_midpoint)
        end

        fig = Figure(size = (500, 400))
        temperature_residual_list /= length(density_list)
        ax = Axis(fig[1, 1], xlabel=L"\text{midpoint density } /10^{19}\mathrm{m}^{-3}", ylabel=L"\mathrm{mean (|T_{\text{residual}}|) }/100\mathrm{eV}", titlesize = 28, title = "Mean residuals in T", xlabelsize = 28, ylabelsize = 28, xticklabelsize = 20, yticklabelsize = 20, xscale = log10, yscale = log10)
        lines!(fig[1, 1], density_list, temperature_residual_list, label="Temperature residuals"; color = :crimson, linewidth = 1.8)
        scatter!(fig[1, 1], density_list, temperature_residual_list, label="Temperature residuals"; marker = :x, markersize = 10, color = :black)
        outfile =  custom_plot_dir * "/coll_dkions_temperature_residuals_vs_n.pdf"
        save(outfile, fig)
    end


end

end