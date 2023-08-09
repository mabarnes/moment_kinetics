#!bin/julia --project
# Note the shebang above uses a relative path. This path is relative to the location
# where the script is run from, not the location of the script itself, so this is a bit
# fragile, but should be OK as this script should only ever be called from the top level
# of the moment_kinetics repo

"""
Get information needed for submitting a run
"""

using Preferences
using UUIDs

settings_string = ""

# Use the UUID to get preferences to avoid having to import moment_kinetics here, which
# takes a while
moment_kinetics_uuid = UUID("b5ff72cc-06fc-4161-ad14-dba1c22ed34e")

function check_and_get_pref(name)
    if !has_preference(moment_kinetics_uuid, name)
        println("Missing `$name` preference. Need to run `setup_moment_kinetics()` before "
                * "using this script.")
        exit(1)
    end
    return string(load_preference(moment_kinetics_uuid, name)) * " "
end
settings_string *= check_and_get_pref("machine")
settings_string *= check_and_get_pref("account")
settings_string *= check_and_get_pref("default_run_time")
settings_string *= check_and_get_pref("default_nodes")
settings_string *= check_and_get_pref("default_postproc_time")
settings_string *= check_and_get_pref("default_postproc_memory")
settings_string *= check_and_get_pref("default_partition")
settings_string *= check_and_get_pref("default_qos")

println(settings_string)
exit(0)
