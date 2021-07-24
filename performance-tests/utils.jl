module PerformanceTestUtils

export upload_result, extract_summary, check_config, get_config

using Dates
using DelimitedFiles
using LibGit2
using Printf
using Statistics
using TOML

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
function get_updated_results_repo()
    if isdir(results_directory)
        repo = GitRepo(results_directory)
        LibGit2.fetch(repo)
        success = LibGit2.merge!(repo)
        if !success
            error("Merging results repo failed")
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
    if config["upload"]
        # If the data is not going to be uploaded, doesn't matter if the machine name is
        # in the known list
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
function upload_result(testtype::AbstractString, results::Vector{Float64})
    config = get_config()
    if config["upload"]
        date = Dates.format(now(), date_format)
        mk_commit = get_mk_commit()

        results_string = @sprintf "%40s %32s %18s" mk_commit config["machine"] date
        for r ∈ results
            results_string *= @sprintf " %22.17g" r
        end
        results_string *= "\n"

        repo = get_updated_results_repo()

        # append results to file
        results_file = string(testtype, ".txt")
        results_path = joinpath(results_directory, results_file)
        open(results_path, "a") do io
            write(io, results_string)
        end
        LibGit2.add!(repo, results_file)
        LibGit2.commit(repo, "Update $results_file")
        # refspecs argument seems to be needed, even though apparently it shouldn't be
        # according to https://github.com/JuliaLang/julia/issues/20741
        LibGit2.push(repo, refspecs=["refs/heads/master"])
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

end # PerformanceTestUtils
