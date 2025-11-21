"""
Post processing functions using Makie.jl

Options are read by default from a file `post_processing_input.toml`, if it exists.

The plots can be generated from the command line by running
```
julia --project run_makie_post_processing.jl dir1 [dir2 [dir3 ...]]
```
"""
module makie_post_processing

export makie_post_process, generate_example_input_file, get_variable,
       setup_makie_post_processing_input!, get_run_info, close_run_info
export animate_f_unnorm_vs_vpa, animate_f_unnorm_vs_vpa_z, get_1d_ax, get_2d_ax,
       irregular_heatmap, irregular_heatmap!, plot_f_unnorm_vs_vpa,
       plot_f_unnorm_vs_vpa_z, positive_or_nan, get_variable, positive_or_nan,
       put_legend_above, put_legend_below, put_legend_left, put_legend_right
export timing_data, parallel_scaling

# Need this import just to allow links in the docstrings to be understood by Documenter.jl
import moment_kinetics
using moment_kinetics.input_structs: Dict_to_NamedTuple
using moment_kinetics.looping: all_dimensions, ion_dimensions, neutral_dimensions
using moment_kinetics.load_data: get_variable, timestep_diagnostic_variables,
                                 em_variables, ion_moment_variables,
                                 electron_moment_variables, neutral_moment_variables,
                                 all_source_variables, all_moment_variables,
                                 ion_dfn_variables, electron_dfn_variables,
                                 neutral_dfn_variables, all_dfn_variables, ion_variables,
                                 neutral_variables, all_variables, ion_source_variables,
                                 neutral_source_variables, electron_source_variables
using moment_kinetics.type_definitions

using LaTeXStrings
using NaNMath
using OrderedCollections
using StatsBase

using CairoMakie
using Makie

const default_input_file_name = "post_processing_input.toml"

"""
Global dict containing settings for makie_post_processing. Can be re-loaded at any time
to change settings.

Is an OrderedDict so the order of sections is nicer if `input_dict` is written out as a
TOML file.
"""
const input_dict = OrderedDict{String,Any}()

"""
Global dict containing settings for makie_post_processing for files with distribution
function output. Can be re-loaded at any time to change settings.

Is an OrderedDict so the order of sections is nicer if `input_dict_dfns` is written out as
a TOML file.
"""
const input_dict_dfns = OrderedDict{String,Any}()

include("high_level_interface.jl")
include("setup_defaults.jl")
include("variable_cache.jl")
include("plots_for_variable.jl")
include("generic_plot_functions.jl")
include("slicing.jl")
include("error_handler.jl")
include("utils.jl")

include("compare_runs.jl")
include("moment_constraints.jl")
include("steady_state_residual.jl")
include("wall_plots.jl")
include("chodura_condition.jl")
include("sound_wave.jl")
include("instability2d.jl")
include("mms.jl")
include("timestep_diagnostics.jl")
include("collisionality.jl")
include("performance_analysis.jl")

end
