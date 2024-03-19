# This script can be used to run a benchmark (using `benchmark_samples` repetitions) of an
# arbitrary input file. Only the time taken for `time_advance!()` is measured. The results
# (from the rank-0 process, which should be representative of overall timings as the
# processes must synchronize fairly often) are printed in the default format of
# BenchmarkTools.@benchmark.
#
# The name of the input file should be given as the first command line argument to the
# script.

using BenchmarkTools, moment_kinetics
using moment_kinetics: time_advance!, setup_moment_kinetics, cleanup_moment_kinetics!

const benchmark_seconds = Inf
const benchmark_samples = 10
const benchmark_evals = 1

function run_benchmark(input_filename)
    rank0 = (moment_kinetics.communication.global_rank[] == 0)

    input_dict = moment_kinetics.moment_kinetics_input.read_input_file(input_filename)

    result = @benchmark(time_advance!(mk_state...),
                        setup=(mk_state = setup_moment_kinetics($input_dict)),
                        teardown=cleanup_moment_kinetics!(mk_state[end-2:end]...),
                        seconds=benchmark_seconds,
                        samples=benchmark_samples,
                        evals=benchmark_evals)

    if rank0
        display(result)
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    input_filename = ARGS[1]
    run_benchmark(input_filename)
end
