# provide option of running from command line via 'julia moment_kinetics.jl'
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    using TimerOutputs
    using moment_kinetics
    using moment_kinetics.moment_kinetics_input: mk_input

    to = TimerOutput
    input = mk_input()
    run_moment_kinetics(to, input)
end
