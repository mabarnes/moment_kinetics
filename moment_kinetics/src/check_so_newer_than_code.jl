"""
    check_so_newer_than_code(so_filename=nothing)

Utility function that checks if `so_filename` is newer than the source code in
`moment_kinetics/src`. If it is, prints an error message and returns `false`; otherwise
returns `true`.

If `so_filename` is `nothing`, use the name of the system image of the current julia
session for `so_filename`.

If `so_filename` is `"makie_postproc.so"`, also checks against the source code in
`makie_post_processing/makie_post_processing/src/`.

If `so_filename` is `"plots_postproc.so"`, also checks against the source code in
`plots_post_processing/plots_post_processing/src/`.
"""
function check_so_newer_than_code(so_filename=nothing)
    if so_filename === nothing
        # Get filename of system image currently being used.
        # https://discourse.julialang.org/t/get-path-of-system-image-from-within-julia/108257/2
        so_filename = unsafe_string(Base.JLOptions().image_file)
    end

    if basename(so_filename) ∉ ("moment_kinetics.so", "makie_postproc.so", "plots_postproc.so")
        # Not using a custom system image that includes moment_kinetics, so no need to
        # check.
        return nothing
    end

    is_makie = (basename(so_filename) == "makie_postproc.so")
    is_plots = (basename(so_filename) == "plots_postproc.so")

    if !isfile(so_filename)
        error("Trying to check age of $so_filename, but $(realpath(so_filename)) does "
              * "not exist")
    end

    # Get modification time of *.so system image
    so_mtime = mtime(so_filename)

    repo_dir = dirname(dirname(dirname(@__FILE__)))

    # Get newest modification time of julia source file
    so_is_newer = true

    function check_file_mtimes(directory)
        for (root, dirs, files) ∈ walkdir(directory; follow_symlinks=true)
            for f ∈ files
                mt = mtime(joinpath(root, f))
                if mt > so_mtime
                    # Found a file that is newer than the .so
                    so_is_newer = false
                    break
                end
            end
            if !so_is_newer
                # Already found a file newer than the .so. No need to continue searching.
                break
            end
        end
    end
    check_file_mtimes(joinpath(repo_dir, "moment_kinetics/src/"))

    # If we already found a code file newer than the .so, no need to keep checking more
    # files, so only keep checking if so_is_newer=true.
    if so_is_newer && is_makie
        check_file_mtimes(joinpath(repo_dir, "makie_post_processing/makie_post_processing/src/"))
    end
    if so_is_newer && is_plots
        check_file_mtimes(joinpath(repo_dir, "plots_post_processing/plots_post_processing/src/"))
    end

    if !so_is_newer
        error_message =
            "\n************************ WARNING ************************\n" *
            " source code files have been modified more recently than\n" *
            "'$so_filename'.\n" *
            "It is likely that you need to re-compile your system image\n" *
            "(so that the code changes take effect) by re-running\n"
        if is_makie
            error_message *= "'precompile-makie-post-processing-submit.sh'" *
                             "\n(or 'precompile_submit.sh')."
        elseif is_plots
            error_message *= "'precompile-plots-post-processing-submit.sh'" *
                             "\n(or 'precompile_submit.sh')."
        else
            error_message *= "'precompile-submit.sh'."
        end
        error_message *= "\n"
        println(error_message)
    end

    return so_is_newer
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_so_newer_than_code(ARGS...)
end
