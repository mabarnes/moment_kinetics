# Set a string-valued preference for moment_kinetics in LocalPreferences.toml

using TOML

preference_name = ARGS[1]
preference_value = ARGS[2]

top_level_directory = dirname(dirname(dirname(@__FILE__)))
local_preferences_filename = joinpath(top_level_directory, "LocalPreferences.toml")
if ispath(local_preferences_filename)
    local_preferences = TOML.parsefile(local_preferences_filename)
else
    local_preferences = Dict{String,Any}()
end
mk_section = get(local_preferences, "moment_kinetics", Dict{String,Any}())
mk_section[preference_name] = preference_value
open(local_preferences_filename, "w") do io
    TOML.print(io, local_preferences, sorted=true)
end
