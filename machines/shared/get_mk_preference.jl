# Get string-valued integer-valued preference for moment_kinetics from LocalPreferences.toml

using TOML

preference_name = ARGS[1]
if length(ARGS) > 1
    default = ARGS[2]
else
    default = "n"
end

top_level_directory = dirname(dirname(dirname(@__FILE__)))
local_preferences_filename = joinpath(top_level_directory, "LocalPreferences.toml")
if ispath(local_preferences_filename)
    local_preferences = TOML.parsefile(local_preferences_filename)
else
    local_preferences = Dict{String,Any}()
end
mk_section = get(local_preferences, "moment_kinetics", Dict{String,Any}())
preference = get(mk_section, preference_name, default)

println(preference)
