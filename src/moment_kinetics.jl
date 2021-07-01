module moment_kinetics

export mk_input
export run_moment_kinetics

include("type_definitions.jl")
include("chebyshev.jl")
include("input_structs.jl")

include("advection.jl")
include("analysis.jl")
include("array_allocation.jl")
include("bgk.jl")
include("calculus.jl")
include("charge_exchange.jl")
include("continuity.jl")
include("coordinates.jl")
#include("driver.jl")
include("em_fields.jl")
include("energy_equation.jl")
include("file_io.jl")
include("finite_differences.jl")
include("force_balance.jl")
include("initial_conditions.jl")
include("load_data.jl")
include("moment_kinetics_input.jl")
include("plot_comparison.jl")
#include("post_processing_driver.jl")
include("post_processing_input.jl")
include("post_processing.jl")
#include("precompile.jl")
include("quadrature.jl")
include("run_moment_kinetics.jl")
include("scan_input.jl")
include("semi_lagrange.jl")
include("source_terms.jl")
include("time_advance.jl")
include("velocity_moments.jl")
include("vpa_advection.jl")
include("z_advection.jl")

end

# provide option of running from command line via 'julia moment_kinetics.jl'
if abspath(PROGRAM_FILE) == @__FILE__
    using TimerOutputs

    to = TimerOutput
    input = moment_kinetics.mk_input()
    moment_kinetics.run_moment_kinetics(to, input)
end
