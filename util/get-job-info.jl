#!bin/julia --project
# Note the shebang above uses a relative path. This path is relative to the location
# where the script is run from, not the location of the script itself, so this is a bit
# fragile, but should be OK as this script should only ever be called from the top level
# of the moment_kinetics repo

"""
Get information needed for submitting a run
"""

using TOML


function failure()
    println("Missing `$name` preference. Need to run `setup_moment_kinetics()` before "
            * "using this script.")
    exit(1)
end

settings_string = ""

if !isfile("LocalPreferences.toml")
    failure()
end
local_preferences = TOML.parsefile("LocalPreferences.toml")

if "moment_kinetics" ∉ keys(local_preferences)
    failure()
end
mk_preferences = local_preferences["moment_kinetics"]

function check_and_get_pref(name)
    if name ∉ keys(mk_preferences)
        failure()
    end
    return string(mk_preferences[name]) * " "
end
settings_string *= check_and_get_pref("machine")
settings_string *= check_and_get_pref("account")
settings_string *= check_and_get_pref("default_run_time")
settings_string *= check_and_get_pref("default_nodes")
settings_string *= check_and_get_pref("default_postproc_time")
settings_string *= check_and_get_pref("default_postproc_memory")
settings_string *= check_and_get_pref("default_partition")
settings_string *= check_and_get_pref("default_qos")
settings_string *= check_and_get_pref("use_makie")
settings_string *= check_and_get_pref("use_plots")

println(settings_string)
exit(0)
