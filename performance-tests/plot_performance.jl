module MKPlotPerformance

include("utils.jl")
using .PerformanceTestUtils

export plot_performance_history
export compare_nthreads_performance_history

using Dates
using CSV
using Glob
using Plots
using StatsBase: countmap

default_start_from = -100

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

mutable struct RunData
    memory::Float64
    min::Float64
    median::Float64
    max::Float64
end
struct MachineDataRow
    commit::String
    date::DateTime
    data::Vector{RunData}
end
struct MachineData
    ntests::Int
    list::Vector{MachineDataRow}
    MachineData(ntests) = new(ntests, Vector{MachineDataRow}(undef, 0))
end
import Base: push!
function push!(data::MachineData, commit::String, date::DateTime, run_data)
    rd = [RunData(x...) for x ∈ run_data]
    @assert length(rd) == data.ntests
    row = MachineDataRow(commit, date, rd)
    push!(data.list, row)
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
        push!(machine_data, commit, date,
              [[row[j] for j ∈ 4 + 4*(i-1):4 + 4*i - 1] for i ∈ 1:ntests])
    end
    return split_data
end

"""
Get nicely formatted tick labels for the x-axis

Arguments
---------
commits : ElasticArray{String}
    Commits to put in the tick labels
dates : ElasticArray{DateTime}
    Dates to put in the tick labels

Returns
-------
xticks : Tuple(UnitRange{Int64}, Vector{String})
    Positions and labels for the x-ticks
blankticks : Tuple(UnitRange{Int64, Vector{String})
    Positions and empty labels for x-axes that don't need tick labels
"""
function get_x_ticks(commits, dates)
    # truncate commit hashes to first 6 characters and trim the time from dates
    xticks = (1:length(commits), ["$(a[1:6]) $(Dates.format(b, "Y-m-d"))"
                                  for (a, b) ∈ zip(commits, dates)])
    blankticks = (1:length(commits), ["" for _ ∈ 1:length(commits)])

    return xticks, blankticks
end

"""
Get nicely formatted tick labels for the x-axis

Arguments
---------
commits : ElasticArray{String}
    Commits to put in the tick labels

Returns
-------
xticks : Tuple(UnitRange{Int64}, Vector{String})
    Positions and labels for the x-ticks
blankticks : Tuple(UnitRange{Int64, Vector{String})
    Positions and empty labels for x-axes that don't need tick labels
"""
function get_x_ticks(commits)
    # truncate commit hashes to first 6 characters and trim the time from dates
    xticks = (1:length(commits), ["$(c[1:6])" for c ∈ commits])
    blankticks = (1:length(commits), ["" for _ ∈ 1:length(commits)])

    return xticks, blankticks
end

"""
Generate integer grid sizes to make a nearly-square grid

Arguments
---------
n : Integer
    Number of things to fit in the grid

Returns
-------
nx, ny : Int64
    Number of elements in horizontal and vertical directions in the
    nearly-square grid
"""
function square_grid_sizes(n)
    nx = ceil(Int64, sqrt(n))
    ny = ceil(Int64, n/nx)

    return nx, ny
end

"""
Is this grid entry the last in a column?
"""
function is_bottom(i, n, nx)
    # index of the next plot down the column
    inext = i + nx
    return inext > n
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
function plot_performance_history(filename, machine=nothing; show=false,
                                  save=true, start_from=default_start_from)
    if machine == nothing
        config = get_config()
        machine = config["machine"]
    end

    all_data = load_run(filename)
    split_data = split_machines(all_data)
    machine_data = split_data[machine]

    l = length(machine_data.list)
    start_index = (start_from <= 0 ? max(l + start_from, 1) : min(start_from, l))
    data = @view machine_data.list[start_index:end]

    commits = [r.commit[1:6] for r ∈ data]

    dates = [r.date for r ∈ data]

    ntests = machine_data.ntests

    plots = plot(;layout=(2, 1), link=:x, yaxis=:log, grid=true, minorgrid=:y, legend=:outertopright)
    memplot = plots[1]
    timeplot = plots[2]
    ylabel!(memplot, "Memory usage (B)")
    ylabel!(timeplot, "Run time (s)")
    xlabel!(timeplot, "commit")

    xticks, blankticks = get_x_ticks(commits, dates)
    for i ∈ 1:ntests
        perf_data = [r.data[i] for r ∈ data]
        plot!(memplot, [x.memory for x ∈ perf_data];
              marker=:x, xticks=blankticks, label="test $i")
        mintime = [x.min for x ∈ perf_data]
        mediantime = [x.median for x ∈ perf_data]
        plot!(timeplot, mintime;
              marker=:x, xticks=xticks, xrotation=45,
              yerror=(0.0, mediantime - mintime),
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

"""
Remove duplicates of any commits that appear more than once in a MachineData struct,
checking that the memory usage is identical for all entries and taking the minimum
run-time.
"""
function remove_duplicates!(d::MachineData)
    commits = [r.commit for r ∈ d.list]
    # First find any duplicated commits
    n_instances = countmap(commits)
    duplicates = [commit for (commit,count) ∈ n_instances if count > 1]

    # Remove duplicates
    for commit ∈ duplicates
        positions = findall(c -> c == commit, commits)

        for i ∈ 1:d.ntests
            # All values of memory usage should be the same for the same code version
            memvalues = [r.data[i].memory for r ∈ d.list[positions]]
            # Check all entries of memvalues are the same
            @assert all(isequal(first(memvalues)), memvalues)

            # Take the minimum runtime in case some entries were affected by how busy the
            # machine was or other external factors
            runtimes = [x.data[i].min for x ∈ d.list[positions]]
            d.list[positions[1]].data[i].min = minimum(runtimes)
        end

        # Remove duplicate entries
        deleteat!(d.list, positions[2:end])

        commits = [r.commit for r ∈ d.list]
    end
end

"""
Given a MachineData `source`, ensure all commits are present in `destination`, adding
anything missing in `destination` with `nothing` for the entries in `destination.data`.
Assumes that neither source nor destination have duplicated commits (use
remove_duplicates!() on both before calling).
"""
function add_missing_commits!(source, destination)
    last_index = 0
    dest_commits = [r.commit for r in destination.list]
    for (commit, date) ∈ collect((r.commit, r.date) for r in source.list)
        if commit ∈ dest_commits
            last_index = findfirst(isequal(commit), dest_commits)
        else
            new_entry = MachineDataRow(commit, date,
                                       [RunData([NaN for j ∈ 1:4]...)
                                        for i ∈ 1:destination.ntests])
            insert!(destination.list, last_index+1, new_entry)
            dest_commits = [r.commit for r in destination.list]
            last_index += 1
        end
    end
end

"""
Compare performance data for different numbers of threads

* Plots memory usage and run time for saved test results, comparing different
    numbers of threads used for the test.
* Results are labelled by commit hash (first 6 characters) and date.
* In the run time plot, the line and crosses show the minimum time from the benchmark
    set (which should be most representative of the raw performance, unaffected by
    machine conditions), while the error bar shows the median run time from the
    benchmark set.
* Different subplots show different test cases - for their meaning, refer to the test file
    that generated the results. Different lines show results for different
    numbers of threads used.
* Unlike plot_performance(), removes duplicate entries for the same commit, taking the
    minimum run-time.

Bugs
----
* Markers do not show in legend if the first entry for that series is NaN. Guess this is
    a bug in Plots.jl or the backend, but does not seem important enough to chase down.

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
function compare_nthreads_performance_history(filename, machine=nothing;
        show=false, save=true, start_from=default_start_from)

    if machine == nothing
        config = get_config()
        machine = config["machine"]
    end

    # Find filenames for different possible numbers of threads
    base, ext = splitext(filename)
    filenames_vec = [filename; glob("$(base)_*threads$ext")]

    # Load data for each number of threads present
    machine_data = Dict{Int, MachineData}()
    nthreads_vec = []
    for f ∈ filenames_vec
        if f == filename
            nthreads = 1
        else
            # Get number of threads from the filename...
            # remove suffix
            name, _ = splitext(f)
            # separate out part of name with number of threads, e.g. "2threads"
            threadsuffix = split(name, "_")[end]
            # remove "threads" and convert to an Int
            nthreads = parse(Int, split(threadsuffix, "threads")[begin])
        end

        push!(nthreads_vec, nthreads)

        all_data = load_run(f)
        split_data = split_machines(all_data)
        this_data = split_data[machine]
        remove_duplicates!(this_data)
        machine_data[nthreads] = this_data
    end

    n = length(nthreads_vec)
    for i ∈ 1:n
        for j ∈ 1:n
            if i ≠ j
                add_missing_commits!(machine_data[nthreads_vec[i]],
                                     machine_data[nthreads_vec[j]])
            end
        end
    end

    commits = [r.commit[1:6] for r ∈ machine_data[nthreads_vec[1]].list]

    ncommits = length(commits)
    start_index = (start_from <= 0 ? max(ncommits + start_from, 1)
                                   : min(start_from, ncommits))

    commits = commits[start_index:end]

    ntests = machine_data[nthreads_vec[1]].ntests

    nx, ny = square_grid_sizes(ntests)
    fontsize = 4
    markersize = 3
    #yaxis = :log
    yaxis = ((0.0, Inf),)
    memplots = plot(;layout=(nx, ny), yaxis=yaxis, grid=true,
                     minorgrid=:y, legend=:outertopright,
                     xtickfontsize=fontsize, ytickfontsize=fontsize,
                     legendfontsize=fontsize)
    timeplots = plot(;layout=(nx, ny), yaxis=yaxis, grid=true,
                      minorgrid=:y, legend=:outertopright,
                      xtickfontsize=fontsize, ytickfontsize=fontsize,
                      legendfontsize=fontsize)
    # Don't know how to make nice labels outside a grid of subplots
    #ylabel!(memplots, "Memory usage (B)")
    #xlabel!(memplots, "commit")
    #ylabel!(timeplots, "Run time (s)")
    #xlabel!(timeplots, "commit")

    # Make extra subplots blank - see
    # https://discourse.julialang.org/t/blank-subplot-in-plots-jl/16453
    for i ∈ ntests+1:nx*ny
        p = memplots[i]
        plot!(p, legend=false,grid=false,foreground_color_subplot=:white)
        p = timeplots[i]
        plot!(p, legend=false,grid=false,foreground_color_subplot=:white)
    end

    xticks, blankticks = get_x_ticks(commits)
    for i ∈ 1:ntests
        p = memplots[i]
        for nthreads ∈ nthreads_vec
            mem_data = [r.data[i].memory
                        for r ∈ machine_data[nthreads].list[start_index:end]]
            plot!(p, mem_data; marker=:x, markersize=markersize, label="$nthreads")
        end
        if is_bottom(i, ntests, nx)
            xticks!(p, xticks, xrotation=45)
        else
            xticks!(p, blankticks)
        end
        p = timeplots[i]
        for nthreads ∈ nthreads_vec
            min_data = [r.data[i].min
                        for r ∈ machine_data[nthreads].list[start_index:end]]
            med_data = [r.data[i].median
                        for r ∈ machine_data[nthreads].list[start_index:end]]
            plot!(p, min_data; marker=:x, markersize=markersize,
                  yerror=[(0.0, e) for e ∈ (med_data - min_data)], label="$nthreads")
        end
        if is_bottom(i, ntests, nx)
            xticks!(p, xticks, xrotation=45)
        else
            xticks!(p, blankticks)
        end
    end

    if show
        gui()
    end

    if save
        run_type = split(basename(filename), ".")[1]
        savefig(memplots, string(run_type, "_compare_threads_memory.pdf"))
        savefig(timeplots, string(run_type, "_compare_threads_runtime.pdf"))
    end

    return memplots, timeplots
end

end # MKPlotPerformance

using .MKPlotPerformance

if abspath(PROGRAM_FILE) == @__FILE__
    filename = undef
    if length(ARGS) > 0
        filename = ARGS[1]
    else
        filename = "results/sound_wave.txt"
    end
    if length(ARGS) > 2
        start_from = parse(Int, ARGS[3])
    else
        start_from = MKPlotPerformance.default_start_from
    end
    if length(ARGS) > 1
        machine = ARGS[2]
        plot_performance_history(filename, machine, start_from=start_from)
    else
        plot_performance_history(filename, start_from=start_from)
    end
end
