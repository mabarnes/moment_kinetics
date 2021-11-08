module MKPlotPerformance

include("utils.jl")
using .PerformanceTestUtils

export plot_performance_history

using Dates
using CSV
using ElasticArrays
using Plots

"""
Loads performance data from a file formatted by PerformanceTestUtils.upload_result()

Returns
-------
CSV.File
"""
function load_run(filename)
    data = CSV.File(filename; header=false, datarow=2, delim=" ", ignorerepeated=true,
                    dateformat=PerformanceTestUtils.date_format, type=Float64,
                    types=Dict(1=>String, 2=>String, 3=>DateTime))
    return data
end

struct MachineData
    ntests::Int
    commits::ElasticArray{String}
    dates::ElasticArray{DateTime}
    data::Vector{ElasticArray{Float64}}
    MachineData(ntests) = new(ntests, ElasticArray{String}(undef, 0),
                              ElasticArray{DateTime}(undef, 0),
                              [ElasticArray{Float64}(undef, 4, 0) for _ in 1:ntests])
end
"""
Parse the data returned by load_run and extract data for each machine into a separate struct

Returns
-------
MachineData
"""
function split_machines(data)
    split_data = Dict{String, MachineData}()
    for row ∈ data
        ntests = (length(row) - 3) ÷ 4
        commit = row[1]
        machine = row[2]
        date = row[3]

        if !haskey(split_data, machine)
            split_data[machine] = MachineData(ntests)
        end

        machine_data = split_data[machine]
        push!(machine_data.commits, commit)
        push!(machine_data.dates, date)
        for i ∈ 1:ntests
            append!(machine_data.data[i], [row[j] for j in 4 + 4*(i-1):4 + 4*i - 1])
        end
    end
    return split_data
end

"""
Plot performance data

* Plots memory usage and run time for saved test results.
* Results are labelled by commit hash (first 6 characters) and date.
* In the run time plot, the line and crosses show the minimum time from the benchmark
    set (which should be most representative of the raw performance, unaffected by
    machine conditions), while the error bar shows the median run time from the
    benchmark set.
* Different lines show different test cases - for their meaning, refer to the test file
    that generated the results.

Arguments
---------
filename : String
    Name of a file, filled with data for a certain set of tests by
    PerformanceTestUtils.upload_result().
machine : String, optional
    Name of the machine to plot performace data from. If not given, the name is read
    from `config.toml`.
show : Bool, default false
    Show the plots in a window.
save : Bool, default true
    Save the plots to a file
start_from : Int, default -100
    Plot a number of results determined by `start_from`. If the value is negative, plot
    the most recent `abs(start_from)+1` recorded results. If the value is positive,
    start plotting from result number `start_from`.
"""
function plot_performance_history(filename, machine=nothing; show=false, save=true, start_from=-100)
    if machine == nothing
        config = get_config()
        machine = config["machine"]
    end

    data = load_run(filename)
    split_data = split_machines(data)
    machine_data = split_data[machine]

    start_index = (start_from <= 0 ? max(length(machine_data.commits) + start_from, 1)
                                   : min(start_from, length(machine_data.commits)))

    commits = [x[1:6] for x in machine_data.commits[start_index:end]]

    dates = machine_data.dates[start_index:end]

    ntests = machine_data.ntests

    plots = plot(;layout=(2, 1), link=:x, yaxis=:log, grid=true, minorgrid=true, legend=:outertopright)
    memplot = plots[1]
    timeplot = plots[2]
    ylabel!(memplot, "Memory usage (B)")
    ylabel!(timeplot, "Run time (s)")
    xlabel!(timeplot, "commit")

    # truncate commit hashes to first 6 characters and trim the time from dates
    xticks = (1:length(commits), ["$(a[1:6]) $(Dates.format(b, "Y-m-d"))"
                                  for (a, b) in zip(commits, dates)])
    blankticks = (1:length(commits), ["" for _ ∈ 1:length(commits)])
    for i ∈ 1:ntests
        perf_data = machine_data.data[i]
        plot!(memplot, perf_data[1, start_index:end];
              marker=:x, xticks=blankticks, label="test $i")
        plot!(timeplot, perf_data[2, start_index:end];
              marker=:x, xticks=xticks, xrotation=45,
              yerror=(0.0, perf_data[3, start_index:end]
                           - perf_data[2, start_index:end]),
              label="test $i")
    end

    if show
        gui()
    end

    if save
        run_type = split(basename(filename), ".")[1]
        savefig(plots, string(run_type, ".pdf"))
    end

    return plots
end

end # MKPlotPerformance

using .MKPlotPerformance

if abspath(PROGRAM_FILE) == @__FILE__
    filename = undef
    if length(ARGS) > 0
        prefix = ARGS[1]
    else
        prefix = "results/sound_wave"
    end
    filename = string(prefix, "_1procs.txt")
    if length(ARGS) > 1
        machine = ARGS[2]
        plot_performance_history(filename, machine)
        plot_performance_history(string(prefix, "_1procs_initialization.txt"), machine)
    else
        plot_performance_history(filename)
        plot_performance_history(string(prefix, "_1procs_initialization.txt"))
    end
end
