using Pkg, TOML


repo_dir = dirname(dirname(dirname(@__FILE__)))
local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
local_preferences = TOML.parsefile(local_preferences_filename)
mk_preferences = local_preferences["moment_kinetics"]

batch_system = mk_preferences["batch_system"]

if mk_preferences["use_plots"] == "y"
    python_venv_path = joinpath(repo_dir, "machines", "artifacts", "mk_venv")
    activate_path = joinpath(python_venv_path, "bin", "activate")
    run(`bash -c "python -m venv --system-site-packages $python_venv_path; source $activate_path; PYTHONNOUSERSITE=1 pip install matplotlib"`)
    if batch_system
        open("julia.env", "a") do io
            println(io, "source $activate_path")
        end
    else
        bin_path = joinpath(repo_dir, "bin", "julia")
        contents = readlines(bin_path)
        open(bin_path, "w") do io
            println(io, contents[1])
            println(io, "source $activate_path")
            for line ∈ contents[2:end]
                println(io, line)
            end
        end
    end
end


if batch_system
  # Make symlinks to batch job submission scripts
  symlink("precompile-submit.sh", joinpath("machines", "shared", "precompile-submit.sh"))
  symlink("submit-run.sh", joinpath("machines", "shared", "submit-run.sh"))
  symlink("submit-restart.sh", joinpath("machines", "shared", "submit-restart.sh"))
  if mk_preferences["use_makie"]
      symlink("precompile-makie-post-processing-submit.sh",
              joinpath("machines", "shared",
                       "precompile-makie-post-processing-submit.sh"))
  end
  if mk_preferences["use_plots"]
      symlink("precompile-plots-post-processing-submit.sh",
              joinpath("machines", "shared",
                       "precompile-plots-post-processing-submit.sh"))
  end

  if mk_preferences["submit_precompilation"] == "y"
      run(`precompile-submit.sh`)
  end
end


# Force exit so Julia must be restarted
#######################################

println()
println("************************************************************")
println("Julia must be restarted to use the updated MPI, exiting now.")
println("************************************************************")
exit(0)
