using Pkg, TOML


repo_dir = dirname(Pkg.project().path)
local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
local_preferences = TOML.parsefile(local_preferences_filename)
mk_preferences = local_preferences["moment_kinetics"]

batch_system = mk_preferences["batch_system"]

if mk_preferences["use_plots"] == "y"
    python_venv_path = joinpath(repo_dir, "machines", "artifacts", "mk_venv")
    activate_path = joinpath(python_venv_path, "bin", "activate")
    run(`bash -c "/usr/bin/env python3 -m venv --system-site-packages $python_venv_path; source $activate_path; PYTHONNOUSERSITE=1 pip install matplotlib"`)
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
  function make_batch_symlink(script_name)
      if islink(script_name)
          rm(script_name)
      end
      symlink(joinpath("machines", "shared", script_name), script_name)
  end
  make_batch_symlink("precompile-submit.sh")
  make_batch_symlink("submit-run.sh")
  make_batch_symlink("submit-restart.sh")
  make_batch_symlink("submit-precompile-and-run.sh")
  make_batch_symlink("submit-precompile-and-restart.sh")
  if mk_preferences["use_makie"] == "y"
      make_batch_symlink("precompile-makie-post-processing-submit.sh")
  end
  if mk_preferences["use_plots"] == "y"
      make_batch_symlink("precompile-plots-post-processing-submit.sh")
  end

  if mk_preferences["submit_precompilation"] == "y"
      run(`./precompile-submit.sh -o`)
  end
end
