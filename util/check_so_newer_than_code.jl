function check_so_newer_than_code(so_filename = "moment_kinetics.so")
    is_makie = occursin("makie", so_filename)
    is_plots = occursin("plots", so_filename)

    if !isfile(so_filename)
        error("Trying to check age of $so_filename, but $(realpath(so_filename)) does "
              * "not exist")
    end

    # Get modification time of *.so system image
    so_mtime = mtime(so_filename)

    # Get newest modification time of julia source file
    newest_jl_mtime = 0.0

    function get_newest_jl_mtime(directory)
        for (root, dirs, files) ∈ walkdir(directory; follow_symlinks=true)
            for f ∈ files
                mt = mtime(joinpath(root, f))
                newest_jl_mtime = max(mt, newest_jl_mtime)
            end
        end
    end
    get_newest_jl_mtime("moment_kinetics/src/")
    if is_makie
        get_newest_jl_mtime("makie_post_processing/makie_post_processing/src/")
    end
    if is_plots
        get_newest_jl_mtime("plots_post_processing/plots_post_processing/src/")
    end

    so_is_newer = so_mtime > newest_jl_mtime

    if !so_is_newer
        error_message =
            "WARNING: source code files have been modified more recently than\n" *
             "'$so_filename'. It is likely that you need to re-compile your system image\n" *
             "(so that the code changes take effect) by re-running "
        if is_makie
            error_message *= "'precompile-makie-post-processing-submit.sh'."
        elseif is_plots
            error_message *= "'precompile-plots-post-processing-submit.sh'."
        else
            error_message *= "'precompile-submit.sh'."
        end
        println(error_message)
    end

    return so_is_newer
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_so_newer_than_code(ARGS...)
end
