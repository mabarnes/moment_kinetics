using moment_kinetics
using makie_post_processing

# Create a temporary directory for test output
test_output_directory = tempname()
run_name = "precompilation"
mkpath(test_output_directory)

input_dict = Dict("run_name"=>run_name,
                  "base_directory" => test_output_directory,
                  "dt" => 0.0,
                  "r_ngrid" => 5,
                  "r_nelement" => 1,
                  "r_bc" => "periodic",
                  "r_discretization" => "chebyshev_pseudospectral",
                  "z_ngrid" => 5,
                  "z_nelement" => 1,
                  "z_bc" => "wall",
                  "z_discretization" => "chebyshev_pseudospectral",
                  "vperp_ngrid" => 5,
                  "vperp_nelement" => 1,
                  #"vperp_bc" => "periodic",
                  "vperp_L" => 4.0,
                  "vperp_discretization" => "chebyshev_pseudospectral",
                  "vpa_ngrid" => 7,
                  "vpa_nelement" => 1,
                  "vpa_bc" => "periodic",
                  "vpa_L" => 4.0,
                  "vpa_discretization" => "chebyshev_pseudospectral",
                  "vzeta_ngrid" => 7,
                  "vzeta_nelement" => 1,
                  "vzeta_bc" => "periodic",
                  "vzeta_L" => 4.0,
                  "vzeta_discretization" => "chebyshev_pseudospectral",
                  "vr_ngrid" => 7,
                  "vr_nelement" => 1,
                  "vr_bc" => "periodic",
                  "vr_L" => 4.0,
                  "vr_discretization" => "chebyshev_pseudospectral",
                  "vz_ngrid" => 7,
                  "vz_nelement" => 1,
                  "vz_bc" => "periodic",
                  "vz_L" => 4.0,
                  "vz_discretization" => "chebyshev_pseudospectral",
                  "timestepping" => Dict{String,Any}("nstep" => 1))

run_moment_kinetics(input_dict)

precompile_postproc_options = makie_post_processing.generate_example_input_Dict()

# Try to activate all plot types to get as much compiled as possible
for (k,v) ∈ precompile_postproc_options
    if v === false
        precompile_postproc_options[k] = true
    end
end

makie_post_process(joinpath(test_output_directory, run_name), precompile_postproc_options)
