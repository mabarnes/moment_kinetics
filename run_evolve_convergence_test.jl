if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk
                  #"runs/1D-wall_evolve_ngrid_5_nel_r_1_z_2_vpa_8_vperp_1","runs/1D-wall_evolve_ngrid_5_nel_r_1_z_4_vpa_16_vperp_1",
    path_list = ["runs/1D-wall_evolve_ngrid_5_nel_r_1_z_8_vpa_32_vperp_1","runs/1D-wall_evolve_ngrid_5_nel_r_1_z_16_vpa_64_vperp_1", 
                "runs/1D-wall_evolve_ngrid_5_nel_r_1_z_32_vpa_128_vperp_1","runs/1D-wall_evolve_ngrid_5_nel_r_1_z_64_vpa_256_vperp_1", 
               ]

    mk.plot_sequence.plot_sequence_fields_data(path_list)
end