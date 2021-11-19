module PerformanceTestUtils

export upload_result, extract_summary, check_config, get_config, run_test

using BenchmarkTools
using Dates
using DelimitedFiles
using LibGit2
using Printf
using Statistics
using TOML

using moment_kinetics.communication: block_rank, block_size
using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!, time_advance!

const date_format = "Y-m-d_HH:MM:SS"

# check we are running with bounds checking disabled
test_boundscheck() = @boundscheck error("Bounds checking is enabled - disable to run "
                                        * "performance tests using '--check-bounds=no'")
test_boundscheck()

# check optimization level 3 is enabled
if Base.JLOptions().opt_level != 3
    error("Found optimization level of $(Base.JLOptions().opt_level). Performance "
          * "tests should always be run with `-O3`.")
end

"""
Read configuration information from `config.toml`

Returns
-------
Dict{String, Any}
"""
function get_config()
    return TOML.parsefile("config.toml")
end

const results_directory = joinpath(@__DIR__, "results")
const results_url = "git@github.com:johnomotani/moment_kinetics_performance_results.git"

"""
Get the git repo where results are saved, and pull so that it is up to date

Returns
-------
LibGit2.GitRepo
"""
function get_updated_results_repo(upload::Bool)
    if isdir(results_directory)
        repo = GitRepo(results_directory)
        if upload
            LibGit2.fetch(repo)
            success = LibGit2.merge!(repo)
            if !success
                error("Merging results repo failed")
            end
        end
        return repo
    else
        return LibGit2.clone(results_url, results_directory)
    end
end

"""
Check that the `machine` set in the config is one of the 'known machines' in the
`results/known_machines` file.

Results from a single machine should always be labelled with the same name, so that the
plotting script can collect them together. `results/known_machines` lists the allowed
names, to avoid typos. If running on a new machine, add a name for it to
`results/known_machines` and commit the change to the `results` repo.
"""
function check_machine(config)
    machine_file = joinpath(results_directory, "known_machines")
    known_machines = readdlm(machine_file, String)
    config = get_config()
    machine_name = config["machine"]
    if ! (machine_name in known_machines)
        error("Machine name '$machine_name' is not present in $machine_file. "
              * "Check for typos or update $machine_file and commit the change.")
    end
end

"""
Run checks on the configuration in `config.toml`
"""
function check_config()
    config = get_config()
    if config["commit"]
        # If the data is not going to be committed, doesn't matter if the machine name
        # is in the known list
        check_machine(config)
    end
end

"""
Get the commit hash of the moment_kinetics repo

Checks that the repo is not 'dirty', i.e. there are no uncommitted changes. This ensures
that saved performance data can be linked to a specific version of the code.

Returns
-------
git_hash : String
    String containing the 40-character hexadecimal git hash.
"""
function get_mk_commit()
    repo = GitRepo("..")
    if LibGit2.isdirty(repo)
        error("moment_kinetics repo is dirty - commit changes and re-run")
    end
    return string(LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))))
end

"""
Upload performance test data

If `upload = true` is set in `config.toml`, writes the performance data along with some
metadata (commit hash of moment_kinetics, name of the machine where the test was run,
and date/time when the test was run).

Arguments
---------
testtype : String
    Name for the test that produced the results. Used as the filename for the results,
    with `.txt` appended.
results : Vector{Float64}
    Results of the test, a vector with concatenated results of several test cases.
    Results from each test case should be formatted by extract_summary()
"""
function upload_result(testtype::AbstractString,
                       initialization_results::Vector{Float64},
                       results::Vector{Float64})
    if block_rank[] == 0
        config = get_config()
        if config["commit"]
            date = Dates.format(now(), date_format)
            mk_commit = get_mk_commit()

            function make_result_string(r)
                return_string = @sprintf "%40s %32s %18s" mk_commit config["machine"] date
                for x ∈ r
                    return_string *= @sprintf " %22.17g" x
                end
                return_string *= "\n"
                return return_string
            end
            initialization_results_string = make_result_string(initialization_results)
            results_string = make_result_string(results)

            repo = get_updated_results_repo(config["upload"])

            # append results to file
            function append_to_file(filename, line, nresults)
                header_string = "Commit                                  | Machine                        | Date             "
                for i ∈ 1:(nresults÷4)
                    header_string *= "| Memory usage $i (B)   | Minimum runtime $i (s)| Median runtime $i (s) | Maximum runtime $i (s)"
                end
                header_string *= "\n"
                if !isfile(filename)
                    open(filename, "w") do io
                        write(io, header_string)
                        write(io, line)
                    end
                else
                    open(filename, "a") do io
                        write(io, line)
                    end
                end
            end
            results_file = string(testtype, "_", block_size[], "procs.txt")
            initialization_results_file = string(testtype, "_", block_size[],
                                                 "procs_initialization.txt")
            initialization_results_path = joinpath(results_directory,
                                                   initialization_results_file)
            results_path = joinpath(results_directory, results_file)
            append_to_file(results_path, results_string, length(results))
            append_to_file(initialization_results_path, initialization_results_string, length(results))

            # Commit results
            LibGit2.add!(repo, initialization_results_file)
            LibGit2.add!(repo, results_file)
            LibGit2.commit(repo, "Update $results_file")
            if config["upload"]
                # refspecs argument seems to be needed, even though apparently it
                # shouldn't be according to
                # https://github.com/JuliaLang/julia/issues/20741
                LibGit2.push(repo, refspecs=["refs/heads/master"])
            end
        end
    end
end

"""
Extract results from test and save to a 1d array

Arguments
---------
result : BenchmarkTools.Trial
    Result of a benchmark test

Returns
-------
result : Vector{Float64}
    Vector containing [memory usage, minimum runtime, median runtime, maximum runtime]
"""
function extract_summary(result)
    times = result.times
    # Convert times from ns to s
    return [result.memory, minimum(times) * 1.e-9, median(times) * 1.e-9,
            maximum(times) * 1.e-9]
end

# Wrap the setup and cleanup functions so we can keep the state in an external
# variable when benchmarking initialization with setup_moment_kinetics(). Necessary
# because it doesn't seem to be possible to assign the output of the function being
# benchmarked to a variable that can be passed to the teardown function.
const mk_state_ref = Ref{Any}()
function setup_wrapper!(input)
    mk_state_ref[] = setup_moment_kinetics(input)
end
function cleanup_wrapper!()
    cleanup_moment_kinetics!(mk_state_ref[][end-1:end]...)
end

_println0(s="") = block_rank[] == 0 && println(s)
_display0(s="") = block_rank[] == 0 && display(s)
const initialization_seconds = Inf
const initialization_samples = 40
const initialization_evals = 1
const benchmark_seconds = Inf
const benchmark_samples = 100
const benchmark_evals = 1
"""
Benchmark for one set of parameters

Returns
-------
[minimum time, median time, maximum time]
"""
function run_test(input)
    message = input["run_name"] * " ($(block_size[]) procs)"
    _println0(message)
    _println0("=" ^ length(message))
    _println0()
    flush(stdout)

    result = @benchmark(time_advance!(mk_state...),
                        setup=(mk_state = setup_moment_kinetics($input)),
                        teardown=cleanup_moment_kinetics!(mk_state[end-1:end]...),
                        seconds=benchmark_seconds,
                        samples=benchmark_samples,
                        evals=benchmark_evals)

    message = "Time advance ($(block_size[]) procs)"
    _println0(message)
    _println0("-" ^ length(message))
    _display0(result)
    _println0()
    _println0()
    flush(stdout)

    initialization_result = @benchmark(setup_wrapper!($input),
                                       teardown=cleanup_wrapper!(),
                                       seconds=initialization_seconds,
                                       samples=initialization_samples,
                                       evals=initialization_evals)
    message = "Initialization ($(block_size[]) procs)"
    _println0(message)
    _println0("-" ^ length(message))
    _display0(initialization_result)
    _println0()
    _println0()
    _println0()
    flush(stdout)

    return extract_summary(initialization_result), extract_summary(result)
end


end # PerformanceTestUtils
