using DelimitedFiles: readdlm
using Glob
using NaturalSort
using NCDatasets: NCDataset
using OrderedCollections: OrderedDict
using Plots

# Which simulation results to plot?
ni_array = [0.0001, 0.25, 0.5, 0.75, 0.9999]
nn_array = 1.0 .- ni_array

scan_basename_n = "CXscan1"

T_array = [0.25, 0.5, 1.0, 2.0, 4.0]

scan_basename_T = "CXscan2"

# Read simulation results
function get_sim_results(ni, nn)
    # Use 'natural' sort to get CX frequencies sorted numerically despite this being a list of strings
    run_directories = sort(glob(string("runs/", scan_basename_n, "_initial_density1-", ni, "_initial_density2-", nn, "_charge_exchange_frequency-*")), lt=natural)
    n = length(run_directories)
    CX_freq = Vector{Float64}(undef, n)
    real_frequency = Vector{Float64}(undef, n)
    growth_rate = Vector{Float64}(undef, n)
    fit_error = Vector{Float64}(undef, n)
    for (i, run) ∈ enumerate(run_directories)
        filename = string(joinpath(run, basename(run)), ".cdf")
        try
            fid = NCDataset(filename)
            # Divide by 2π to get 'Hz' instead of 'radians/s'
            CX_freq[i] = fid["charge_exchange_frequency"][:] / 2 / π
            growth_rate[i] = fid["growth_rate"][:] / 2 / π
            real_frequency[i] = fid["frequency"][:] / 2 / π
            fit_error[i] = fid["fit_error"][:]
            close(fid)
        catch LoadError
            CX_freq[i] = NaN
            growth_rate[i] = NaN
            real_frequency[i] = NaN
            fit_error[i] = NaN
            println(filename, " failed")
        end
    end
    return CX_freq, real_frequency, growth_rate, fit_error
end

function get_sim_results_T(T_e)
    # Use 'natural' sort to get CX frequencies sorted numerically despite this being a list of strings
    run_directories = sort(glob(string("runs/", scan_basename_T, "_T_e-", T_e, "_charge_exchange_frequency-*")), lt=natural)
    n = length(run_directories)
    CX_freq = Vector{Float64}(undef, n)
    real_frequency = Vector{Float64}(undef, n)
    growth_rate = Vector{Float64}(undef, n)
    fit_error = Vector{Float64}(undef, n)
    for (i, run) ∈ enumerate(run_directories)
        filename = string(joinpath(run, basename(run)), ".cdf")
        try
            fid = NCDataset(filename)
            # Divide by 2π to get 'Hz' instead of 'radians/s'
            CX_freq[i] = fid["charge_exchange_frequency"][:] / 2 / π
            growth_rate[i] = fid["growth_rate"][:] / 2 / π
            real_frequency[i] = fid["frequency"][:] / 2 / π
            fit_error[i] = fid["fit_error"][:]
            close(fid)
        catch LoadError
            CX_freq[i] = NaN
            growth_rate[i] = NaN
            real_frequency[i] = NaN
            fit_error[i] = NaN
            println(filename, " failed")
        end
    end
    return CX_freq, real_frequency, growth_rate, fit_error
end

# Read analytical results
# First column is normalised CX collision frequency
# Second column is real frequency
# Third column is growth rate of the finite frequency mode
# Fourth column is growth rate of the zero frequency mode
analytical_results_ni = OrderedDict(
     "ni000"=>readdlm("runs/analytic_results/ni000_mod.txt", comments=true),
     "ni025"=>readdlm("runs/analytic_results/ni025.txt", comments=true),
     "ni050"=>readdlm("runs/analytic_results/ni050.txt", comments=true),
     "ni075"=>readdlm("runs/analytic_results/ni075.txt", comments=true),
     "ni100"=>readdlm("runs/analytic_results/ni100.txt", comments=true),
    )
analytical_results_Te = OrderedDict(
     "Te025"=>readdlm("runs/analytic_results/Te025.txt", comments=true),
     "Te005"=>readdlm("runs/analytic_results/Te05.txt", comments=true),
     "Te100"=>readdlm("runs/analytic_results/Te100.txt", comments=true),
     "Te200"=>readdlm("runs/analytic_results/Te200.txt", comments=true),
     "Te400"=>readdlm("runs/analytic_results/Te400.txt", comments=true),
    )

# Plot analytical results

real_frequency_plot = plot(legend=:outertopright, size=(900,400))
for (i, (k,v)) ∈ enumerate(analytical_results_ni)
    plot!(real_frequency_plot,
          v[:, 1], v[:, 2],
          label=k,
          xlabel="CX collision frequency",
          ylabel="Mode frequency",
          color=i,
         )
end

growth_rate_plot = plot(legend=:outertopright, size=(900,400))
for (i, (k,v)) ∈ enumerate(analytical_results_ni)
    plot!(growth_rate_plot,
          v[2:end, 1], v[2:end, 3],
          label=k,
          xlabel="CX collision frequency",
          ylabel="Growth rate",
          color=i,
         )
    plot!(growth_rate_plot,
          v[2:end, 1], v[2:end, 4],
          label="",
          color=i,
          linestyle=:dash,
         )
end

# fix plot limits so funny simulation results don't mess them up
plot!(real_frequency_plot,
      xlims=xlims(real_frequency_plot),
      ylims=ylims(real_frequency_plot),
     )
plot!(growth_rate_plot,
      xlims=xlims(growth_rate_plot),
      ylims=ylims(growth_rate_plot),
     )

function  split_results(array, fit_error)
    threshold = 0.07
    return ([x for (i, x) in enumerate(array) if fit_error[i] < threshold],
            [x for (i, x) in enumerate(array) if fit_error[i] >= threshold])
end

# Plot simulation results
for (i, (ni, nn)) ∈ enumerate(zip(ni_array, nn_array))
    CX_freq, real_frequency, growth_rate, fit_error = get_sim_results(ni, nn)
    cx_good, cx_bad = split_results(CX_freq, fit_error)
    rf_good, rf_bad = split_results(real_frequency, fit_error)
    scatter!(real_frequency_plot,
             cx_good, rf_good,
             label=string("sim, ni=", ni, " nn=", nn),
             color=i,
             markerstrokecolor=0,
            )
    scatter!(real_frequency_plot,
             cx_bad, rf_bad,
             label="",
             color=i,
             markershape=:xcross,
             markerstrokecolor=0,
            )
    gr_good, gr_bad = split_results(growth_rate, fit_error)
    scatter!(growth_rate_plot,
             cx_good, gr_good,
             label=string("sim, ni=", ni, " nn=", nn),
             color=i,
             markerstrokecolor=0,
            )
    scatter!(growth_rate_plot,
             cx_bad, gr_bad,
             label="",
             color=i,
             markershape=:xcross,
             markerstrokecolor=0,
            )
end

savefig(real_frequency_plot, "runs/comparison_plots/real_frequency.pdf")
savefig(growth_rate_plot, "runs/comparison_plots/growth_rate.pdf")


# Plot T scan

# Plot analytical results

real_frequency_plot = plot(legend=:outertopright, size=(900,400))
for (i, (k,v)) ∈ enumerate(analytical_results_Te)
    plot!(real_frequency_plot,
          v[:, 1], v[:, 2],
          label=k,
          xlabel="CX collision frequency",
          ylabel="Mode frequency",
          color=i,
         )
end

growth_rate_plot = plot(legend=:outertopright, size=(900,400))
for (i, (k,v)) ∈ enumerate(analytical_results_Te)
    plot!(growth_rate_plot,
          v[2:end, 1], v[2:end, 3],
          label=k,
          xlabel="CX collision frequency",
          ylabel="Growth rate",
          color=i,
         )
    plot!(growth_rate_plot,
          v[2:end, 1], v[2:end, 4],
          label="",
          color=i,
          linestyle=:dash,
         )
end

# fix plot limits so funny simulation results don't mess them up
plot!(real_frequency_plot,
      xlims=xlims(real_frequency_plot),
      #ylims=ylims(real_frequency_plot),
      ylims=(0.0, ylims(real_frequency_plot)[2]),
     )
plot!(growth_rate_plot,
      xlims=xlims(growth_rate_plot),
      ylims=ylims(growth_rate_plot),
     )

# Plot simulation results
for (i, (T_e)) ∈ enumerate(T_array)
    CX_freq, real_frequency, growth_rate, fit_error = get_sim_results_T(T_e)
    cx_good, cx_bad = split_results(CX_freq, fit_error)
    rf_good, rf_bad = split_results(real_frequency, fit_error)
    scatter!(real_frequency_plot,
             cx_good, rf_good,
             label=string("sim, T_e=", T_e),
             color=i,
             markerstrokecolor=0,
            )
    scatter!(real_frequency_plot,
             cx_bad, rf_bad,
             label="",
             color=i,
             markershape=:xcross,
             markerstrokecolor=0,
            )
    gr_good, gr_bad = split_results(growth_rate, fit_error)
    scatter!(growth_rate_plot,
             cx_good, gr_good,
             label=string("sim, T_e=", T_e),
             color=i,
             markerstrokecolor=0,
            )
    scatter!(growth_rate_plot,
             cx_bad, gr_bad,
             label="",
             color=i,
             markershape=:xcross,
             markerstrokecolor=0,
            )
end

savefig(real_frequency_plot, "runs/comparison_plots/real_frequency_T.pdf")
savefig(growth_rate_plot, "runs/comparison_plots/growth_rate_T.pdf")
