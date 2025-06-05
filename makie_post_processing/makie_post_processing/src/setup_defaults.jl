using moment_kinetics: check_so_newer_than_code
using moment_kinetics.input_structs: set_defaults_and_check_top_level!,
                                     set_defaults_and_check_section!, check_sections!

using TOML

"""
    setup_makie_post_processing_input!(input_file::Union{AbstractString,Nothing}=nothing;
                                       run_info_moments=nothing, run_info_dfns=nothing,
                                       allow_missing_input_file=false)
    setup_makie_post_processing_input!(new_input_dict::AbstractDict{String,Any};
                                       run_info_moments=nothing,
                                       run_info_dfns=nothing)

Pass `input_file` to read the input from an input file other than
`$default_input_file_name`. You can also pass a `Dict{String,Any}` of options.

Set up input, storing in the global [`input_dict`](@ref) and [`input_dict_dfns`](@ref) to
be used in the various plotting and analysis functions.

The `run_info` that you are using (as returned by
[`get_run_info`](@ref)) should be passed to `run_info_moments` (if it contains only the
moments), or `run_info_dfns` (if it also contains the distributions functions), or both
(if you have loaded both sets of output).  This allows default values to be set based on
the grid sizes and number of time points read from the output files. Note that
`setup_makie_post_processing_input!()` is called by default at the end of
`get_run_info()`, for conveinence in interactive use.

By default an error is raised if `input_file` does not exist. To continue anyway, using
default options, pass `allow_missing_input_file=true`.
"""
function setup_makie_post_processing_input! end

function setup_makie_post_processing_input!(
        input_file::Union{AbstractString,Nothing}=nothing; run_info_moments=nothing,
        run_info_dfns=nothing, allow_missing_input_file=false)

    if input_file === nothing
        input_file = default_input_file_name
    end

    if isfile(input_file)
        new_input_dict = TOML.parsefile(input_file)
    elseif allow_missing_input_file
        println("Warning: $input_file does not exist, using default post-processing "
                * "options")
        new_input_dict = OrderedDict{String,Any}()
    else
        error("$input_file does not exist")
    end
    setup_makie_post_processing_input!(new_input_dict, run_info_moments=run_info_moments,
                                       run_info_dfns=run_info_dfns)

    return nothing
end

function setup_makie_post_processing_input!(new_input_dict::AbstractDict{String,Any};
                                            run_info_moments=nothing,
                                            run_info_dfns=nothing)

    # Check that, if we are using a custom compiled system image that includes
    # moment_kinetics, the system image is newer than the source code files (if there are
    # changes made to the source code since the system image was compiled, they will not
    # affect the current run). Prints a warning if any code files are newer than the
    # system image.
    check_so_newer_than_code()

    convert_to_OrderedDicts!(new_input_dict)

    if isa(run_info_moments, AbstractVector)
        has_moments = any(ri !== nothing for ri ∈ run_info_moments)
    else
        has_moments = run_info_moments !== nothing
    end
    if isa(run_info_dfns, AbstractVector)
        has_dfns = any(ri !== nothing for ri ∈ run_info_dfns)
    else
        has_dfns = run_info_dfns !== nothing
    end

    if !has_moments && !has_dfns
        println("Neither `run_info_moments` nor `run_info_dfns` passed. Setting "
                * "defaults without using grid sizes")
    elseif !has_moments
        println("No run_info_moments, using run_info_dfns to set defaults")
        run_info_moments = run_info_dfns
        has_moments = true
    elseif !has_dfns
        println("No run_info_dfns, defaults for distribution function coordinate sizes "
                * "will be set to 1.")
    end

    _setup_single_input!(input_dict, new_input_dict, run_info_moments, false)
    _setup_single_input!(input_dict_dfns, new_input_dict, run_info_dfns, true)

    return nothing
end

# Utility function to reduce code duplication in setup_makie_post_processing_input!()
function _setup_single_input!(this_input_dict::OrderedDict{String,Any},
                              new_input_dict::AbstractDict{String,Any}, run_info,
                              dfns::Bool)
    # Error on unexpected options, so that the user can fix them.
    warn_unexpected = false

    # Remove all existing entries from this_input_dict
    clear_Dict!(this_input_dict)

    # Put entries from new_input_dict into this_input_dict
    merge!(this_input_dict, deepcopy(new_input_dict))

    if !isa(run_info, AbstractVector)
        # Make sure run_info is a Vector
        run_info= Any[run_info]
    end
    has_run_info = any(ri !== nothing for ri ∈ run_info)

    if has_run_info
        nt_unskipped_min = minimum(ri.nt_unskipped for ri in run_info
                                                   if ri !== nothing)
        nt_min = minimum(ri.nt for ri in run_info if ri !== nothing)
        nr_min = minimum(ri.r.n for ri in run_info if ri !== nothing)
        nz_min = minimum(ri.z.n for ri in run_info if ri !== nothing)
    else
        nt_unskipped_min = 1
        nt_min = 1
        nr_min = 1
        nz_min = 1
    end
    if dfns && has_run_info
        if any(ri.vperp !== nothing for ri ∈ run_info)
            nvperp_min = minimum(ri.vperp.n for ri in run_info
                                 if ri !== nothing && ri.vperp !== nothing)
        else
            nvperp_min = 1
        end
        if any(ri.vpa !== nothing for ri ∈ run_info)
            nvpa_min = minimum(ri.vpa.n for ri in run_info
                               if ri !== nothing && ri.vpa !== nothing)
        else
            nvpa_min = 1
        end
        if any(ri.vzeta !== nothing for ri ∈ run_info)
            nvzeta_min = minimum(ri.vzeta.n for ri in run_info
                                 if ri !== nothing && ri.vzeta !== nothing)
        else
            nvzeta_min = 1
        end
        if any(ri.vr !== nothing for ri ∈ run_info)
            nvr_min = minimum(ri.vr.n for ri in run_info
                              if ri !== nothing && ri.vr !== nothing)
        else
            nvr_min = 1
        end
        if any(ri.vz !== nothing for ri ∈ run_info)
            nvz_min = minimum(ri.vz.n for ri in run_info
                              if ri !== nothing && ri.vz !== nothing)
        else
            nvz_min = 1
        end
    else
        nvperp_min = 1
        nvpa_min = 1
        nvzeta_min = 1
        nvr_min = 1
        nvz_min = 1
    end

    # Whitelist of options that only apply at the global level, and should not be used
    # as defaults for per-variable options.
    # Notes:
    # - Don't allow setting "itime_*" and "itime_*_dfns" per-variable because we
    #   load time and time_dfns in run_info and these must use the same
    #   "itime_*"/"itime_*_dfns" setting as each variable.
    only_global_options = ("itime_min", "itime_max", "itime_skip", "itime_min_dfns",
                           "itime_max_dfns", "itime_skip_dfns", "handle_errors")

    set_defaults_and_check_top_level!(this_input_dict, warn_unexpected;
       # Options that only apply at the global level (not per-variable)
       ################################################################
       # Options that provide the defaults for per-variable settings
       #############################################################
       colormap="reverse_deep",
       animation_ext="gif",
       # Slice t to this value when making time-independent plots
       it0=nt_min,
       it0_dfns=nt_min,
       # Choose this species index when not otherwise specified
       is0=1,
       # Slice r to this value when making reduced dimensionality plots
       ir0=max(cld(nr_min, 3), 1),
       # Slice z to this value when making reduced dimensionality plots
       iz0=max(cld(nz_min, 3), 1),
       # Slice vperp to this value when making reduced dimensionality plots
       ivperp0=max(cld(nvperp_min, 3), 1),
       # Slice vpa to this value when making reduced dimensionality plots
       ivpa0=max(cld(nvpa_min, 3), 1),
       # Slice vzeta to this value when making reduced dimensionality plots
       ivzeta0=max(cld(nvzeta_min, 3), 1),
       # Slice vr to this value when making reduced dimensionality plots
       ivr0=max(cld(nvr_min, 3), 1),
       # Slice vz to this value when making reduced dimensionality plots
       ivz0=max(cld(nvz_min, 3), 1),
       # Time index to start from
       itime_min=1,
       # Time index to end at
       itime_max=nt_unskipped_min,
       # Load every `time_skip` time points for EM and moment variables, to save memory
       itime_skip=1,
       # Time index to start from for distribution functions
       itime_min_dfns=1,
       # Time index to end at for distribution functions
       itime_max_dfns=nt_unskipped_min,
       # Load every `time_skip` time points for distribution function variables, to save
       # memory
       itime_skip_dfns=1,
       plot_vs_r=true,
       plot_vs_z=true,
       plot_vs_r_t=true,
       plot_vs_z_t=true,
       plot_vs_z_r=true,
       animate_vs_z=false,
       animate_vs_r=false,
       animate_vs_z_r=false,
       show_element_boundaries=false,
       steady_state_residual=false,
       # By default, errors are caught so that later plots can still be made. For
       # debugging it can be useful to turn this off.
       handle_errors=true,
      )

    section_defaults = OrderedDict(k=>v for (k,v) ∈ this_input_dict
                                   if !isa(v, AbstractDict) &&
                                      !(k ∈ only_global_options))
    for variable_name ∈ tuple(all_moment_variables..., timestep_diagnostic_variables...)
        set_defaults_and_check_section!(
            this_input_dict, variable_name, warn_unexpected;
            OrderedDict(Symbol(k)=>v for (k,v) ∈ section_defaults)...)
    end

    plot_options_1d = Tuple(Symbol(:plot_vs_, d) for d ∈ one_dimension_combinations)
    plot_log_options_1d = Tuple(Symbol(:plot_log_vs_, d) for d ∈ one_dimension_combinations)
    plot_options_2d = Tuple(Symbol(:plot_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations)
    plot_log_options_2d = Tuple(Symbol(:plot_log_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations)
    animate_options_1d = Tuple(Symbol(:animate_vs_, d) for d ∈ one_dimension_combinations_no_t)
    animate_log_options_1d = Tuple(Symbol(:animate_log_vs_, d) for d ∈ one_dimension_combinations_no_t)
    animate_options_2d = Tuple(Symbol(:animate_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations_no_t)
    animate_log_options_2d = Tuple(Symbol(:animate_log_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations_no_t)
    for variable_name ∈ all_dfn_variables
        set_defaults_and_check_section!(
            this_input_dict, variable_name, warn_unexpected;
            check_moments=false,
            (o=>false for o ∈ plot_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ plot_log_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ plot_options_2d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ plot_log_options_2d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_log_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_options_2d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_log_options_2d if String(o) ∉ keys(section_defaults))...,
            plot_unnorm_vs_vpa=false,
            plot_unnorm_vs_vz=false,
            plot_unnorm_vs_vpa_z=false,
            plot_unnorm_vs_vz_z=false,
            plot_log_unnorm_vs_vpa=false,
            plot_log_unnorm_vs_vz=false,
            plot_log_unnorm_vs_vpa_z=false,
            plot_log_unnorm_vs_vz_z=false,
            animate_unnorm_vs_vpa=false,
            animate_unnorm_vs_vz=false,
            animate_unnorm_vs_vpa_z=false,
            animate_unnorm_vs_vz_z=false,
            animate_log_unnorm_vs_vpa=false,
            animate_log_unnorm_vs_vz=false,
            animate_log_unnorm_vs_vpa_z=false,
            animate_log_unnorm_vs_vz_z=false,
            OrderedDict(Symbol(k)=>v for (k,v) ∈ section_defaults)...)
        # Sort keys to make dict easier to read
        sort!(this_input_dict[variable_name])
    end

    set_defaults_and_check_section!(
        this_input_dict, "compare_runs", warn_unexpected;
        enable=false,
        interpolate_to_other_grid=false,
       )

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf", warn_unexpected;
        plot=false,
        animate=false,
        advection_velocity=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
        n_points_near_wall=4,
       )

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf_electron", warn_unexpected;
        plot=false,
        animate=false,
        advection_velocity=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
        n_points_near_wall=4,
       )

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf_neutral", warn_unexpected;
        plot=false,
        animate=false,
        advection_velocity=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
        n_points_near_wall=4,
       )

    set_defaults_and_check_section!(
        this_input_dict, "constraints", warn_unexpected;
        plot=false,
        animate=false,
        it0=this_input_dict["it0"],
        ir0=this_input_dict["ir0"],
        iz0=this_input_dict["iz0"],
        ivperp0=this_input_dict["ivperp0"],
        ivpa0=this_input_dict["ivpa0"],
        ivzeta0=this_input_dict["ivzeta0"],
        ivr0=this_input_dict["ivr0"],
        ivz0=this_input_dict["ivz0"],
        animation_ext=this_input_dict["animation_ext"],
        show_element_boundaries=this_input_dict["show_element_boundaries"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "Chodura_condition", warn_unexpected;
        plot_vs_t=false,
        plot_vs_r=false,
        plot_vs_r_t=false,
        plot_f_over_vpa2=false,
        animate_f_over_vpa2=false,
        it0=this_input_dict["it0"],
        ir0=this_input_dict["ir0"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "instability2D", warn_unexpected;
        plot_1d=false,
        plot_2d=false,
        animate_perturbations=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "sound_wave_fit", warn_unexpected;
        calculate_frequency=false,
        plot=false,
        ir0=this_input_dict["ir0"],
        iz0=this_input_dict["iz0"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "manufactured_solns", warn_unexpected;
        calculate_error_norms=true,
        wall_plots=false,
        (o=>false for o ∈ plot_options_1d)...,
        (o=>false for o ∈ plot_log_options_1d)...,
        (o=>false for o ∈ plot_options_2d)...,
        (o=>false for o ∈ plot_log_options_2d)...,
        (o=>false for o ∈ animate_options_1d)...,
        (o=>false for o ∈ animate_log_options_1d)...,
        (o=>false for o ∈ animate_options_2d)...,
        (o=>false for o ∈ animate_log_options_2d if String(o) ∉ keys(section_defaults))...,
        (o=>section_defaults[String(o)] for o ∈ (:it0, :ir0, :iz0, :ivperp0, :ivpa0, :ivzeta0, :ivr0, :ivz0))...,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
        show_element_boundaries=this_input_dict["show_element_boundaries"],
       )
    sort!(this_input_dict["manufactured_solns"])

    set_defaults_and_check_section!(
        this_input_dict, "timestep_diagnostics", warn_unexpected;
        plot=true,
        animate_CFL=false,
        plot_timestep_residual=false,
        animate_timestep_residual=false,
        plot_timestep_error=false,
        animate_timestep_error=false,
        plot_steady_state_residual=false,
        animate_steady_state_residual=false,
       )
    
    set_defaults_and_check_section!(
        this_input_dict, "collisionality_plots", warn_unexpected;
        plot=true,
        plot_dT_dz_vs_z=false,
        animate_dT_dz_vs_z=false,
        plot_mfp_vs_z=false,
        animate_mfp_vs_z=false,
        plot_nu_ii_vth_mfp_vs_z = false,
        plot_LT_mfp_vs_z = false,
        animate_LT_mfp_vs_z = false,
        plot_LT_dT_dz_temp_vs_z = false,
        plot_Ln_mfp_vs_z = false,
        animate_Ln_mfp_vs_z = false,
        plot_Lupar_mfp_vs_z = false,
        animate_Lupar_mfp_vs_z = false,
        plot_Lupar_Ln_LT_mfp_vs_z = false,
        animate_Lupar_Ln_LT_mfp_vs_z = false,
        plot_overlay_coll_krook_heat_flux = false,
        animate_overlay_coll_krook_heat_flux = false,
        animation_ext = "gif"
       )
    # set_defaults_and_check_section!(
    #     this_input_dict, "mk_1D1V_term_size_diagnostics", warn_unexpected;
    #     plot=true)
       
    set_defaults_and_check_section!(
        this_input_dict, "timing_data", warn_unexpected;
        plot=false,
        threshold=1.0e-2,
        include_patterns=String[],
        exclude_patterns=String[],
        ranks=mk_int[],
        figsize=[600,800]
       )

    # We allow top-level options in the post-processing input file
    check_sections!(this_input_dict; check_no_top_level_options=false)

    return nothing
end
