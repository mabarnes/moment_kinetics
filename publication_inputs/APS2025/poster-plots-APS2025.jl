using CairoMakie
using makie_post_processing
using makie_post_processing.Printf
using StatsBase

plots_dir = "publication_inputs/APS2025"

# 1D backgrounds
for (sim_dir, label) ∈ (("runs/1D1V-instability-test/", "background_1D"),
                        ("runs/1D1V-instability-test_no-Krook/", "background_1D_no-Krook")
                       )
    dir_1d = mkpath(joinpath(plots_dir, label))

    # load data
    ri_1d = get_run_info(sim_dir; dfns=true)

    parallel_coordinate = ri_1d.z.grid ./ ri_1d.geometry.input.pitch

    # Plot sources
    source_amplitude = get_variable(ri_1d, "external_source_amplitude"; it=ri_1d.nt, is=1, ir=1)[:,1]
    source_T = get_variable(ri_1d, "external_source_T_array"; it=ri_1d.nt, is=1, ir=1)[:,1]

    fig_1d, ax_1d, l = lines(parallel_coordinate, source_amplitude; label="source amplitude")
    lines!(ax_1d, parallel_coordinate, source_T; label="source temperature")
    ax_1d.xlabel = "parallel distance"
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_source.png"), fig_1d; px_per_unit=16.0)

    # Plot moment profiles
    n = get_variable(ri_1d, "density"; it=ri_1d.nt, is=1, ir=1)
    T = get_variable(ri_1d, "temperature"; it=ri_1d.nt, is=1, ir=1)

    fig_1d, ax_1d, l = lines(parallel_coordinate, n; label="n")
    lines!(ax_1d, parallel_coordinate, T; label="T")
    ax_1d.xlabel = "parallel distance"
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_nT_profiles.png"), fig_1d; px_per_unit=16.0)

    u = get_variable(ri_1d, "parallel_flow"; it=ri_1d.nt, is=1, ir=1)

    fig_1d, ax_1d, l = lines(parallel_coordinate, u; label="u")
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "u_∥"
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_upar_profiles.png"), fig_1d; px_per_unit=16.0)

    if label == "background_1D"
        nu_ii_Krook = get_variable(ri_1d, "Krook_collision_frequency_ii"; it=ri_1d.nt,
                                   is=1, ir=1)

        fig_1d, ax_1d, l = lines(parallel_coordinate, nu_ii_Krook; label="nu_ii_Krook")
        ax_1d.xlabel = "parallel distance"
        ax_1d.ylabel = "nu_ii"
        Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

        save(joinpath(dir_1d, label * "_nu_ii_Krook_profiles.png"), fig_1d; px_per_unit=16.0)
    end

    # Plot distribution function
    f = get_variable(ri_1d, "f"; it=ri_1d.nt, is=1, ir=1, ivperp=1)

    fig_1d, ax_1d, hm = heatmap(ri_1d.vpa.grid, parallel_coordinate, f)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "v_∥"
    Colorbar(fig_1d[1,2], hm)

    save(joinpath(dir_1d, label * "_f.png"), fig_1d; px_per_unit=16.0)

    # Classical particle diffusivity and classical heat diffusivity.
    nu_ii = get_variable(ri_1d, "Krook_collision_frequency_ii"; it=ri_1d.nt, is=1, ir=1)
    nu_ei = get_variable(ri_1d, "collision_frequency_ei"; it=ri_1d.nt, is=1, ir=1)
    # Dimensionless Ω_i = Ω_{i,ref} (as B=B_ref is constant) is 1/rhostar due to reference
    # parameter definitions (see
    # https://mabarnes.github.io/moment_kinetics/dev/moment_kinetic_equations/#Dimensionless-equations-for-code).
    Omega_i = 1.0 / ri_1d.geometry.rhostar
    Omega_e = Omega_i / ri_1d.composition.me_over_mi
    vth_i = get_variable(ri_1d, "thermal_speed"; it=ri_1d.nt, is=1, ir=1)
    vth_e = get_variable(ri_1d, "electron_thermal_speed"; it=ri_1d.nt, is=1, ir=1)
    rho_i = vth_i ./ Omega_i
    rho_e = vth_e ./ Omega_e
    D_classical = @. rho_e^2 * nu_ei
    chi_i_classical = @. rho_i^2 * nu_ii

    fig_1d, ax_1d, l = lines(parallel_coordinate, D_classical)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "D_classical"

    save(joinpath(dir_1d, label * "_D_classical.png"), fig_1d; px_per_unit=16.0)

    fig_1d, ax_1d, l = lines(parallel_coordinate, chi_i_classical)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "chi_i_classical"

    save(joinpath(dir_1d, label * "_chi_i_classical.png"), fig_1d; px_per_unit=16.0)

    fig_1d, ax_1d, l = lines(parallel_coordinate, rho_i)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "rho_i"

    save(joinpath(dir_1d, label * "_rho_i.png"), fig_1d; px_per_unit=16.0)
end

# Instability analysis - scan in radial resolution
dir_r_nelement_scan = mkpath(joinpath(plots_dir, "instability_2D_r-resolution-scan"))
r_r_nelement_scan_run_dirs = ("runs/2D1V-instability-test_Lr1cm/",
                            "runs/2D1V-instability-test_Lr1cm-rnelement8/",
                            "runs/2D1V-instability-test_Lr1cm-rnelement16/",
                            "runs/2D1V-instability-test_Lr1cm-rnelement32/",
                           )
ri_r_nelement_scan = get_run_info(r_r_nelement_scan_run_dirs...; dfns=true)

function plot_mode_amplitude(this_ri, phi, ax, irun)
    # Take a crude but hopefully robust estimate of the mode amplitude.
    # Take the maximum over z of the the RMS over r of the variable.
    # Pass `corrected=false` to `StatsBase.std` so that it just calculates the RMS,
    # not the unbiased standard deviation.
    amplitude = maximum(std(phi; corrected=false, dims=2); dims=1)[1,1,:]

    # Fit A*exp(γ*time) to exponentially-growing interval.
    γ, A, tmin, tmax = makie_post_processing.partial_fit_exponential_growth(this_ri.time, amplitude)
    println(this_ri.run_name, " γ = $γ")

    lines!(ax, this_ri.time, amplitude; color=Cycled(irun), linestyle=:dot)

    # Plot the fitted exponential growth
    fit_t = this_ri.time[@. this_ri.time > tmin && this_ri.time < tmax]
    fit_amplitude = @. A * exp(γ * fit_t)
    lines!(ax, fit_t, fit_amplitude; color=Cycled(irun),
           label="r_nelement = $(this_ri.r.nelement_global)")
    label_t = 0.5 * (fit_t[1] + fit_t[end])
    gamma_string = @sprintf("%.5g", γ)
    # For now setting the text color with Cycled(irun) doesn't work in Makie.jl.
    #text!(ax, Point2f(label_t, A * exp(γ * label_t)); text="γ = $gamma_string",
    #      align=(:left, :top), color=Cycled(irun))
    text!(ax, Point2f(label_t, A * exp(γ * label_t)); text="γ = $gamma_string",
          align=(:left, :top))
end

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)

for (irun, this_ri) ∈ enumerate(ri_r_nelement_scan)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_r_nelement_scan, "phi_perturbation_r-nelement$(this_ri.r.nelement_global).gif")
    title = "r_nelement = $(this_ri.r.nelement_global)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = $(this_ri.r.nelement_global)"
    save(joinpath(dir_r_nelement_scan,
                  "final_phi_perturbation_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig; px_per_unit=16.0)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
save(joinpath(dir_r_nelement_scan, "growth_rate_r-resolution-scan.png"), fig; px_per_unit=16.0)

# Check convergence with r_nelement of stabilisation at Dr=1e-8.
stabilised_run_dirs = ("runs/2D1V-instability-test_Lr1cm_rdiss1e-8/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss1e-8/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement32-rdiss1e-8/",
                      )
ri_stabilised = get_run_info(stabilised_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)

for (irun, this_ri) ∈ enumerate(ri_stabilised)
    phi = get_variable(this_ri, "phi")
    plot_mode_amplitude(this_ri, phi, ax, irun)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
save(joinpath(dir_r_nelement_scan, "stabilised-resolution-scan.png"), fig; px_per_unit=16.0)

# Show apparently converged mode with Dr=7e-9
converged_rdiss7em9_run_dirs = ("runs/2D1V-instability-test_Lr1cm_rdiss7e-9/",
                                "runs/2D1V-instability-test_Lr1cm-rnelement8-rdiss7e-9/",
                                "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss7e-9/",
                                #"runs/2D1V-instability-test_Lr1cm-rnelement32-rdiss7e-9/",
                               )
ri_converged = get_run_info(converged_rdiss7em9_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)

for (irun, this_ri) ∈ enumerate(ri_converged)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_r_nelement_scan, "phi_perturbation_rdiss7e-9_r-nelement$(this_ri.r.nelement_global).gif")
    title = "r_nelement = $(this_ri.r.nelement_global), Dr = 7e-9"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = $(this_ri.r.nelement_global), Dr = 7e-9"
    save(joinpath(dir_r_nelement_scan,
                  "final_phi_perturbation_rdiss7e-9_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig; px_per_unit=16.0)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
save(joinpath(dir_r_nelement_scan, "converged-rdiss7e-9-resolution-scan.png"), fig; px_per_unit=16.0)

# Scan dissipation between Dr=1e-9 (unsure if the instability is radial-grid-scale) and Dr=7e-9 (instability seems well resolved).
dir_rdiss_scan = mkpath(joinpath(plots_dir, "instability_2D_rdiss-scan"))
rdiss_scan_run_dirs = ("runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss1e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss2e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss3e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss4e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss5e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss6e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss7e-9/",
                      )
ri_rdiss_scan = get_run_info(rdiss_scan_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)

for (irun, this_ri) ∈ enumerate(ri_rdiss_scan)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_rdiss_scan, "phi_perturbation_rdiss$(irun)e-9.gif")
    title = "r_nelement = 16, Dr = $(irun)e-9"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = 16, Dr = $(irun)e-9"
    save(joinpath(dir_rdiss_scan,
                  "final_phi_perturbation_rdiss$(irun)e-9.png"),
                  final_fig; px_per_unit=16.0)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
save(joinpath(dir_rdiss_scan, "rdiss-scan.png"), fig; px_per_unit=16.0)

# Compare case with Krook collisions switched off
dir_no_Krook = mkpath(joinpath(plots_dir, "instability_2D_no-Krook"))
no_Krook_run_dirs = ("runs/2D1V-instability-test_Lr1cm-no-Krook/",
                     "runs/2D1V-instability-test_Lr1cm-rnelement8-no-Krook/",
                     "runs/2D1V-instability-test_Lr1cm-rnelement16-no-Krook/",
                     "runs/2D1V-instability-test_Lr1cm-rnelement32-no-Krook/",
                    )
ri_no_Krook = get_run_info(no_Krook_run_dirs...; dfns=true)
fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)

for (irun, this_ri) ∈ enumerate(ri_no_Krook)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_no_Krook, "phi_perturbation_no-Krook_r-nelement$(this_ri.r.nelement_global).gif")
    title = "no Krook, r_nelement = $(this_ri.r.nelement_global)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "no Krook, r_nelement = $(this_ri.r.nelement_global)"
    save(joinpath(dir_no_Krook,
                  "final_phi_perturbation_no-Krook_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig; px_per_unit=16.0)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
save(joinpath(dir_no_Krook, "growth_rate_no-Krook_r-resolution-scan.png"), fig; px_per_unit=16.0)
