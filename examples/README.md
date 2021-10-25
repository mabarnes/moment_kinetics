The input files in this directory are examples of different cases that can be
run with `moment_kinetics`. Many of them are based on the inputs used for the
tests.

To run, go to the top-level directory and make sure there is a subdirectory called
`runs`. Then to run, e.g., one of the sound-wave examples either from the
command line:
```
$ julia --project -O3 run_moment_kinetics.jl examples/sound-wave/sound-wave_cheb.toml
```
or using the Julia REPL:
```
$ julia --project -O3
julia> using moment_kinetics
julia> run_moment_kinetics("examples/sound-wave/sound-wave_cheb.toml")
```
