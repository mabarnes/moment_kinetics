A `config.toml` file, containing a bool `upload` (whether to upload results to the
`moment_kinetics_performance_results` repo) and a string `machine` giving the name of
the machine where the tests are being run, should be present in the current directory.
You can copy `config.toml.example` and update `machine`.

The `machine` setting should be one of the values present in `results/known_machines`.
If you are running on a new machine, add its name to `results/known_machines` and commit
the changes to the repo in `results/`. You will first need to either run a test (which
should work, but fail to save the results to a file because the machine is unknown - in
the process it will clone the results repo) or first do
```
$ git clone git@github.com:johnomotani/moment_kinetics_performance_results.git results
```

Historical performance test results are saved by adding to files in the `results/` repo,
and automatically committing/pushing when the test finishes.

To run, for example, the sound wave tests, either in the REPL (started with
`julia -O3 --project --check-bounds=no`) run
```
julia> include("sound_wave.jl")
```
or from the command line
```
julia -O3 --project --check-bounds=no sound_wave.jl
```
Note that julia should be run with full optimization and with all bounds checking
disabled, since this is a performance test. Running without disabling bounds checking
will be an error.
