module MKPlotPerformance

include("utils.jl")
using .PerformanceTestUtils

export plot_performance_history, plot_strong_scaling_history

using Dates
using CSV
using ElasticArrays
using Glob
using Plots

# include the test cases so that we can get inputs from them
include("sound_wave.jl")
include("sound_wave-2xres.jl")

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
Parse the data returned by `load_run()` and extract data for each machine into a
separate struct.

Parameters
----------
data : Array returned by `load_run()`
remove_duplicates : Bool, default false
    If `true`, only keep the first entry for each commit hash, discarding any
    duplicates.

Returns
-------
MachineData
"""
function split_machines(data; remove_duplicates::Bool=false)
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
        if !remove_duplicates || !(commit in machine_data.commits)
            push!(machine_data.commits, commit)
            push!(machine_data.dates, date)
            for i ∈ 1:ntests
                append!(machine_data.data[i], [row[j] for j in 4 + 4*(i-1):4 + 4*i - 1])
            end
        end
    end
    return split_data
end

"""
Clean up axis limits for logscale axis

Ensures that there are at least two 'major tick' on an axis. If there are not then the
axis labels seem to get messed up.

Does make the y-axis longer than it needs to be - maybe future versions of Plots.jl will
make this unnecessary (see e.g.  https://github.com/JuliaPlots/Plots.jl/issues/3918).
"""
function cleanup_log_ylims!(plot)
    ymin, ymax = ylims(plot)

    logmin = log10(ymin)
    logmax = log10(ymax)

    ymin = 10.0^(floor(logmin))
    ymax = 10.0^(ceil(logmax))

    ylims!(plot, (ymin, ymax))
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

    # Make sure yaxes include at least one major tick
    cleanup_log_ylims!(memplot)
    cleanup_log_ylims!(timeplot)

    if show
        gui()
    end

    if save
        run_type = split(basename(filename), ".")[1]
        savefig(plots, string(run_type, ".pdf"))
    end

    return plots
end

"""
Find all files corresponding to 'prefix'

Arguments
---------
prefix : String
    Prefix of files in the format "\$(prefix)_\$(n)procs.txt", filled with data for a
    certain set of tests by PerformanceTestUtils.upload_result().

Returns
-------
filenames : Vector{String}
    Names of the files, ordered by number of processors
nprocs : Vector{Int}
    Number of processes for each file
"""
function get_strong_scaling_filenames(prefix)
    filepaths = Dict{Int, String}()

    more_paths = glob("$(prefix)_*procs.txt")
    pattern = Regex("$(prefix)_([0-9]*)procs.txt")
    for path ∈ more_paths
        m = match(pattern, path)
        nproc = parse(Int, m.captures[])
        filepaths[nproc] = m.match
    end

    nprocs = sort(collect(keys(filepaths)))

    return [filepaths[n] for n ∈ nprocs], nprocs
end

"""
Merge two lists of commits

Modifies the first argument `base`, merging in the entries in `extra`.
Both arguments are assumed to have no duplicates.
"""
function merge_commit_lists!(base, extra)
    index_to_insert = 1
    for commit ∈ extra
        i = findfirst(x->x==commit, base)
        if i == nothing
            # `commit` was not in `base`, so insert
            insert!(base, index_to_insert, commit)
            index_to_insert += 1
        else
            # `commit` was in base, no need to insert
            index_to_insert = i + 1
        end
    end
    return nothing
end

"""
Convert MachineData to a Dict of data indexed by the commit hash

If there are duplicate commits in `machine_data`, result will contain the data for the
last duplicate.

Returns
-------
Dict{String, Matrix{Float64}}
    Indices are commits. Values are Matrices of size (4, machine_data.ntests) containing
    the memory/timing information.
"""
function MachineData_as_Dict(machine_data::MachineData)
    result = Dict{String, Matrix{Float64}}()
    for i ∈ 1:length(machine_data.commits)
        data = Matrix{Float64}(undef, 4, machine_data.ntests)
        for j ∈ 1:machine_data.ntests
            data[:, j] .= machine_data.data[j][:, i]
        end
        result[machine_data.commits[i]] = data
    end
    return result
end

"""
Get grid sizes

Look up the test script that produced the output from prefix, get its input
Dicts and use them to calculate the grid sizes.

Arguments
---------
prefix : String
    prefix indicating which test type to use (corresponds to 'test_name' in each test
    script).

Returns
-------
grid_sizes : Vector{Int}
    Vector of numbers giving the grid size over which (most) loops are parallelized,
    e.g. nz for 1D1V cases, one for each test case.
"""
function get_grid_sizes(prefix)
    prefix = splitpath(prefix)[end]
    if prefix == "sound_wave"
        inputs_list = SoundWavePerformance.inputs_list
        # Use nz as the 'grid size' because that is the outer-loop in most of the 1D1V
        # code (basically everything apart from z_advection!()).
        grid_sizes = [input["z_nelement"] * (input["z_ngrid"] - 1) + 1
                      for input in inputs_list]
    elseif prefix == "sound_wave-2xres"
        inputs_list = SoundWave2xResPerformance.inputs_list
        # Use nz as the 'grid size' because that is the outer-loop in most of the 1D1V
        # code (basically everything apart from z_advection!()).
        grid_sizes = [input["z_nelement"] * (input["z_ngrid"] - 1) + 1
                      for input in inputs_list]
    else
        error("Unrecognized prefix $prefix")
    end

    return grid_sizes
end

"""
Plot strong scaling performance data

* Plots strong scaling performance for saved test results.
* Results are labelled by commit hash (first 6 characters) and date.
* In the run time plot, the line and crosses show the minimum time from the benchmark
    set (which should be most representative of the raw performance, unaffected by
    machine conditions), while the error bar shows the median run time from the
    benchmark set.
* Different test cases are shown in separate plots - for their meaning, refer to the
    test file that generated the results.

Arguments
---------
prefix : String
    Prefix of files in the format "\$(prefix)_\$(n)procs.txt", filled with data for a
    certain set of tests by PerformanceTestUtils.upload_result().
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
function plot_strong_scaling_history(prefix, machine=nothing; show=false, save=true, start_from=-100)
    if machine == nothing
        config = get_config()
        machine = config["machine"]
    end

    filenames, nprocs = get_strong_scaling_filenames(prefix)
    grid_sizes = get_grid_sizes(prefix)

    data_for_nproc = Dict{Int, Dict}()
    all_commits = Vector{String}(undef, 0)
    ntests = -1
    for (filename, nproc) ∈ zip(filenames, nprocs)
        data = load_run(filename)
        split_data = split_machines(data, remove_duplicates=true)
        machine_data = split_data[machine]

        if ntests < 0
            ntests = machine_data.ntests
        elseif ntests != machine_data.ntests
            error("Inconsistent number of tests in $filename: got "
                  * "$(machine_data.ntests), expected $ntests.")
        end

        merge_commit_lists!(all_commits, machine_data.commits)

        data_for_nproc[nproc] = MachineData_as_Dict(machine_data)
    end

    start_index = (start_from <= 0 ? max(length(all_commits) + start_from, 1)
                                   : min(start_from, length(all_commits)))

    commits = all_commits[start_index:end]

    strong_scaling_plots = plot(;layout=(1, ntests), xaxis=:log, yaxis=:log,
                                 grid=true, minorgrid=true, legend=false,
                                 size=(2400, 400))
    efficiency_plots = plot(;layout=(1, ntests), grid=true, minorgrid=true,
                             legend=false, size=(2400, 400))
    actual_LB_plots = plot(;layout=(1, ntests), grid=true, minorgrid=true,
                           legend=false, size=(2400, 400))
    function add_plot(commit; linewidth=1, show_ideal=false)
        to_plot = Array{Union{Float64,Missing},3}(undef, 4, ntests, length(nprocs))
        for (i, nproc) in enumerate(nprocs)
            data = data_for_nproc[nproc]
            if commit ∈ keys(data)
                to_plot[:,:,i] .= data[commit]
            else
                to_plot[:,:,i] .= missing
            end
        end
        for i ∈ 1:ntests
            this_nprocs = [n for (i,n) ∈ enumerate(nprocs) if !ismissing(to_plot[1,1,i])]
            this_to_plot = Array{Float64,3}(undef, 4, ntests, length(this_nprocs))
            counter = 1
            for i in 1:size(to_plot,3)
                if !ismissing(to_plot[1,1,i])
                    this_to_plot[:,:,counter] .= to_plot[:,:,i]
                    counter += 1
                end
            end
            ideal = (this_to_plot[2,i,1] * this_nprocs[1]) ./ this_nprocs
            # ideal_with_actual_load_balance is the run time expected with no
            # communication overhead for the process with the largest number of points
            # to loop over.
            ideal_with_actual_load_balance =
                this_to_plot[2,i,1] .*
                [ceil(grid_sizes[i] / n) / grid_sizes[i] for n in this_nprocs]
            plot!(strong_scaling_plots[i], this_nprocs, this_to_plot[2,i,:];
                  marker=:x, linewidth=linewidth,
                  yerror=(zeros(length(this_nprocs)), this_to_plot[3,i,:] - this_to_plot[2,i,:]),
                  label=commit[1:6])
            if show_ideal
                # Ideal scaling is time proportional to 1/nproc, with the fit line
                # passing through the run with the lowest number of processors
                # (presumably the serial run usually).
                plot!(strong_scaling_plots[i], this_nprocs, ideal; linestyle=:dash, label="ideal")
                # 'actual LB' is the time expected with the 'actual load balance', i.e.
                # for the process with the largest number of grid points in the actual
                # run n_max = ceil(n/nproc), it is n_max/n times the time for the serial
                # run
                plot!(strong_scaling_plots[i], this_nprocs,
                      ideal_with_actual_load_balance; linestyle=:dash, label="actual LB")
            end
            plot!(efficiency_plots[i], this_nprocs, ideal ./ this_to_plot[2,i,:];
                  marker=:x, linewidth=linewidth,
                  yerror=(- ideal ./ this_to_plot[3,i,:] + ideal ./ this_to_plot[2,i,:],
                          zeros(length(this_nprocs))),
                  label=commit[1:6])
            plot!(actual_LB_plots[i], this_nprocs,
                  ideal_with_actual_load_balance ./ this_to_plot[2,i,:];
                  marker=:x, linewidth=linewidth,
                  yerror=(- ideal_with_actual_load_balance ./ this_to_plot[3,i,:]
                          + ideal_with_actual_load_balance ./ this_to_plot[2,i,:],
                          zeros(length(this_nprocs))),
                  label=commit[1:6])
        end
    end

    for commit ∈ commits[begin:end-1]
        add_plot(commit)
    end
    # Emphasise the most recent version
    add_plot(commits[end], linewidth=2, show_ideal=true)

    plot!(strong_scaling_plots[end]; legend=:true)
    plot!(efficiency_plots[end]; legend=:true)
    plot!(actual_LB_plots[end]; legend=:true)

    for i ∈ 1:length(strong_scaling_plots)
        ylabel!(strong_scaling_plots[i], "Run time (s)")
        xlabel!(strong_scaling_plots[i], "n_proc")
        ylabel!(efficiency_plots[i], "Efficiency")
        xlabel!(efficiency_plots[i], "n_proc")
        ylabel!(actual_LB_plots[i], "Efficiency*")
        xlabel!(actual_LB_plots[i], "n_proc")
    end

    if show
        gui()
    end

    if save
        savefig(strong_scaling_plots, string(basename(prefix), "_strong_scaling.pdf"))
        savefig(efficiency_plots, string(basename(prefix), "_efficiency.pdf"))
        savefig(actual_LB_plots, string(basename(prefix), "_actual_LB.pdf"))
    end

    return strong_scaling_plots, efficiency_plots
end

end # MKPlotPerformance

using .MKPlotPerformance
using moment_kinetics: options as mk_options

if abspath(PROGRAM_FILE) == @__FILE__
    filename = undef
    if mk_options["inputfile"] == nothing
        prefix = "results/sound_wave"
    else
        prefix = mk_options["inputfile"]
    end
    filename = string(prefix, "_1procs.txt")

    machine = mk_options["machine-name"]
    plot_performance_history(filename, machine)
    plot_performance_history(string(prefix, "_1procs_initialization.txt"), machine)
    plot_strong_scaling_history(prefix, machine)
end
