using TOML

export duplicate_toml
# script that takes in a TOML file, some strings, and custom ranges, and duplicates
# the TOML file with the strings replaced by the custom ranges, and the title of the 
# file also reflects this change.

function duplicate_toml(toml_file::String, strings::Array{String,1}, lower_limit::Float64, upper_limit::Float64, step_size::Float64)
    # read the TOML file
    toml_data = TOML.parsefile(toml_file)
    
    # get title without .toml extension
    title = join(split(toml_file, ".")[1:end-1], ".")

    # loop through the custom range
    for j in lower_limit:step_size:upper_limit
        # create a new TOML file
        new_toml_file = title * "_n$j.toml"
        
        # loop through the keys in the TOML file
        for (section, dict) in toml_data
            for (key, value) in dict
                if key ∈ strings
                    toml_data[section][key] = j
                end
            end
        end
        open(new_toml_file, "w") do file
            TOML.print(file, toml_data)
        end

    end
end

function main()
    toml_file = ARGS[1]
    strings = ARGS[2:end-3]
    lower_limit = parse(Float64, ARGS[end-2])
    upper_limit = parse(Float64, ARGS[end-1])
    step_size = parse(Float64, ARGS[end])
    duplicate_toml(toml_file, strings, lower_limit, upper_limit, step_size)
end

main()