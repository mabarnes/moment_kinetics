using DelimitedFiles: readdlm
using Glob
using NaturalSort
using OrderedCollections: OrderedDict
using Plots

# Which simulation results to plot?
ni_array = [0.0001, 0.25, 0.5, 0.75, 0.9999]
nn_array = 1.0 .- ni_array

scan_basename = "CXscan1"

# Read simulation results
function get_sim_results(ni, nn)
    # Use 'natural' sort to get CX frequencies sorted numerically despite this being a list of strings
    run_directories = sort(glob(string("runs/", scan_basename, "_initial_density2-", nn, "_charge_exchange_frequency-*_initial_density1-", ni)), lt=natural)
    n = length(run_directories)
    CX_freq = Vector{Float64}(undef, n)
    real_frequency = Vector{Float64}(undef, n)
    growth_rate = Vector{Float64}(undef, n)
    for (i, run) ∈ enumerate(run_directories)
        try
            # Divide by 2π to get 'Hz' instead of 'radians/s'
            CX_freq[i] = parse(Float64, split(split(run, "_")[6], "-")[2]) / 2 / π
            info = split(readline(string(joinpath(run, basename(run)), ".frequency_fit.txt")))
            growth_rate[i] = parse(Float64, info[2]) / 2 / π
            real_frequency[i] = parse(Float64, info[8]) / 2 / π
        catch LoadError
            CX_freq[i] = NaN
            growth_rate[i] = NaN
            real_frequency[i] = NaN
        end
    end
    return CX_freq, real_frequency, growth_rate
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
     "Te005"=>readdlm("runs/analytic_results/Te025.txt", comments=true),
     "Te100"=>readdlm("runs/analytic_results/Te025.txt", comments=true),
     "Te200"=>readdlm("runs/analytic_results/Te025.txt", comments=true),
     "Te400"=>readdlm("runs/analytic_results/Te025.txt", comments=true),
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

# Plot simulation results
for (i, (ni, nn)) ∈ enumerate(zip(ni_array, nn_array))
    CX_freq, real_frequency, growth_rate = get_sim_results(ni, nn)
    scatter!(real_frequency_plot,
             CX_freq, real_frequency,
             label=string("sim, ni=", ni, " nn=", nn),
             color=i,
            )
    scatter!(growth_rate_plot,
             CX_freq, growth_rate,
             label=string("sim, ni=", ni, " nn=", nn),
             color=i,
            )
end

savefig(real_frequency_plot, "runs/comparison_plots/real_frequency.pdf")
savefig(growth_rate_plot, "runs/comparison_plots/growth_rate.pdf")
