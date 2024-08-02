# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

base_input = Dict("run_name" => "precompilation",
                  "base_directory" => test_output_directory,
                  "dt" => 0.0,
                  "r_ngrid" => 5,
                  "r_nelement" => 3,
                  "r_bc" => "periodic",
                  "r_discretization" => "finite_difference",
                  "z_ngrid" => 5,
                  "z_nelement" => 3,
                  "z_bc" => "periodic",
                  "z_discretization" => "finite_difference",
                  "vperp_ngrid" => 5,
                  "vperp_nelement" => 3,
                  "vperp_bc" => "zero",
                  "vperp_L" => 4.0,
                  "vperp_discretization" => "finite_difference",
                  "vpa_ngrid" => 7,
                  "vpa_nelement" => 3,
                  "vpa_bc" => "zero",
                  "vpa_L" => 8.0,
                  "vpa_discretization" => "finite_difference",
                  "vzeta_ngrid" => 5,
                  "vzeta_nelement" => 3,
                  "vzeta_bc" => "zero",
                  "vzeta_L" => 4.0,
                  "vzeta_discretization" => "finite_difference",
                  "vr_ngrid" => 5,
                  "vr_nelement" => 3,
                  "vr_bc" => "zero",
                  "vr_L" => 4.0,
                  "vr_discretization" => "finite_difference",
                  "vz_ngrid" => 7,
                  "vz_nelement" => 3,
                  "vz_bc" => "zero",
                  "vz_L" => 8.0,
                  "vz_discretization" => "finite_difference",
                  "timestepping" => Dict{String,Any}("nstep" => 1))
cheb_input = merge(base_input, Dict("r_discretization" => "chebyshev_pseudospectral",
                                    "z_discretization" => "chebyshev_pseudospectral",
                                    "vperp_discretization" => "chebyshev_pseudospectral",
                                    "vpa_discretization" => "chebyshev_pseudospectral"))
wall_bc_input = merge(base_input, Dict("z_bc" => "wall"))
wall_bc_cheb_input = merge(cheb_input, Dict("z_bc" => "wall"))

inputs_list = Vector{Dict{String, Any}}(undef, 0)
for input ∈ [base_input, cheb_input, wall_bc_input, wall_bc_cheb_input]
    push!(inputs_list, input)
    x = merge(input, Dict("evolve_moments_density" => true, "ionization_frequency" => 0.0,
                          "r_ngrid" => 1, "r_nelement" => 1, "vperp_ngrid" => 1,
                          "vperp_nelement" => 1, "vzeta_ngrid" => 1,
                          "vzeta_nelement" => 1, "vr_ngrid" => 1, "vr_nelement" => 1))
    push!(inputs_list, x)
    x = merge(x, Dict("evolve_moments_parallel_flow" => true))
    push!(inputs_list, x)
    x = merge(x, Dict("evolve_moments_parallel_pressure" => true))
    push!(inputs_list, x)
end

collisions_input = merge(wall_bc_cheb_input, Dict("n_neutral_species" => 0,
                                                  "krook_collisions" => Dict{String,Any}("use_krook" => true),
                                                  "fokker_planck_collisions" => Dict{String,Any}("use_fokker_planck" => true, "self_collisions" => true, "slowing_down_test" => true),
                                                  "vperp_discretization" => "gausslegendre_pseudospectral",
                                                  "vpa_discretization" => "gausslegendre_pseudospectral",
                                                 ))
# add an additional input for every geometry option available in addition to the default
geo_input1 = merge(wall_bc_cheb_input, Dict("n_neutral_species" => 0,
                                            "geometry" => Dict{String,Any}("option" => "1D-mirror", "DeltaB" => 0.5, "pitch" => 0.5, "rhostar" => 1.0))) 

kinetic_electron_input = merge(cheb_input, Dict("evolve_moments_density" => true,
                                                "evolve_moments_parallel_flow" => true,
                                                "evolve_moments_parallel_pressure" => true,
                                                "r_ngrid" => 1,
                                                "r_nelement" => 1,
                                                "vperp_ngrid" => 1,
                                                "vperp_nelement" => 1,
                                                "vzeta_ngrid" => 1,
                                                "vzeta_nelement" => 1,
                                                "vr_ngrid" => 1,
                                                "vr_nelement" => 1,
                                                "electron_physics" => "kinetic_electrons",
                                                "electron_timestepping" => Dict{String,Any}("nstep" => 1,
                                                                                            "dt" => 2.0e-11,
                                                                                            "initialization_residual_value" => 1.0e10,
                                                                                            "converged_residual_value" => 1.0e10,
                                                                                            "rtol" => 1.0e10,
                                                                                            "no_restart" => true),
                                               ))

push!(inputs_list, collisions_input, geo_input1, kinetic_electron_input)

for input in inputs_list
    run_moment_kinetics(input)
end
