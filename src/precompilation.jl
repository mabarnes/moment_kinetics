using PrecompileTools

@compile_workload begin
    moment_kinetics.communication.__init__()
    run_moment_kinetics("minimal.toml")
end
