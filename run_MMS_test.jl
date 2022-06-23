if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk


    path_list = ["runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_2_vperp_16","runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_4_vperp_16",
                    "runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_8_vperp_16","runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_16_vperp_16"]
    scan_type = "vpa_nelement"


    mk.plot_MMS_sequence.get_MMS_error_data(path_list,scan_type)
end
