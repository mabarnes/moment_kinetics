# Hard-code arguments because PackageCompiler does not allow passing them
push!(ARGS, "--long")

# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict
using moment_kinetics.utils: recursive_merge

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

base_input = OptionsDict("output" => OptionsDict("run_name"=>"precompilation",
                                                 "base_directory" => test_output_directory),
                         "z" => OptionsDict("ngrid" => 5,
                                            "nelement" => 1,
                                            "bc" => "periodic",
                                            "discretization" => "finite_difference"),
                         "vpa" => OptionsDict("ngrid" => 5,
                                              "nelement" => 1,
                                              "bc" => "periodic",
                                              "discretization" => "finite_difference"),
                         "timestepping" => OptionsDict("nstep" => 1, "dt" => 2.0e-11))
cheb_input = recursive_merge(base_input, OptionsDict("z" => OptionsDict("discretization" => "chebyshev_pseudospectral"),
                                                     "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral")))
wall_bc_input = recursive_merge(base_input, OptionsDict("z" => OptionsDict("bc" => "wall")))
wall_bc_cheb_input = recursive_merge(cheb_input, OptionsDict("z" => OptionsDict("bc" => "wall")))

inputs_list = Vector{OptionsDict}(undef, 0)
for input ∈ [base_input, cheb_input, wall_bc_input, wall_bc_cheb_input]
    push!(inputs_list, input)
    x = recursive_merge(input, OptionsDict("evolve_moments" => OptionsDict("density" => true),
                                           "reactions" => OptionsDict("ionization_frequency" => 0.0)))
    push!(inputs_list, x)
    x = recursive_merge(x, OptionsDict("evolve_moments" => OptionsDict("parallel_flow" => true)))
    push!(inputs_list, x)
    x = recursive_merge(x, OptionsDict("evolve_moments" => OptionsDict("pressure" => true)))
    push!(inputs_list, x)
end

for input in inputs_list
    run_moment_kinetics(input)
end
