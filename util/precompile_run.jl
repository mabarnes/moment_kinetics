# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using TimerOutputs
using moment_kinetics

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

base_input = Dict("nstep"=>1,
                  "run_name"=>"precompilation",
                  "base_directory" => test_output_directory,
                  "dt" => 0.0,
                  "z_ngrid" => 5,
                  "z_nelement" => 1,
                  "z_bc" => "periodic",
                  "z_discretization" => "finite_difference",
                  "vpa_ngrid" => 5,
                  "vpa_nelement" => 1,
                  "vpa_bc" => "periodic",
                  "vpa_L" => 4.0,
                  "vpa_discretization" => "finite_difference")
cheb_input = merge(base_input, Dict("z_discretization" => "chebyshev_pseudospectral",
                                    "vpa_discretization" => "chebyshev_pseudospectral"))
wall_bc_input = merge(base_input, Dict("z_bc" => "wall"))
wall_bc_cheb_input = merge(cheb_input, Dict("z_bc" => "wall"))

inputs_list = Vector{Dict{String, Any}}(undef, 0)
for input ∈ [base_input, cheb_input, wall_bc_input, wall_bc_cheb_input]
    push!(inputs_list, input)
    x = merge(input, Dict("evolve_moments_density" => true, "ionization_frequency" => 0.0))
    push!(inputs_list, x)
    x = merge(x, Dict("evolve_moments_parallel_flow" => true))
    push!(inputs_list, x)
    x = merge(x, Dict("evolve_moments_parallel_pressure" => true))
    push!(inputs_list, x)
end

for input in inputs_list
    run_moment_kinetics(input)
end
