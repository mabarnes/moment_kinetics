include("plot_performance.jl")

filename = undef
if length(ARGS) > 1
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
    compare_nthreads_performance_history(filename, machine, start_from=start_from)
else
    compare_nthreads_performance_history(filename, start_from=start_from)
end
