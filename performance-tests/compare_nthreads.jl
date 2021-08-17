include("plot_performance.jl")

filename = undef
if length(ARGS) > 1
    filename = ARGS[1]
else
    filename = "results/sound_wave.txt"
end
machine = undef
if length(ARGS) > 1
    compare_nthreads_performance_history(filename, machine)
else
    compare_nthreads_performance_history(filename)
end
