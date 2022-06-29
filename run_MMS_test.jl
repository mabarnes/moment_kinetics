if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk

   test_option = "collisional_soundwaves"

    if test_option == "collisional_soundwaves"
        # collisional 
        path_list = ["runs/2D-sound-wave_cheb_cxiz_nel_r_2_z_2_vpa_2_vperp_2","runs/2D-sound-wave_cheb_cxiz_nel_r_2_z_2_vpa_4_vperp_4",
                       "runs/2D-sound-wave_cheb_cxiz_nel_r_2_z_2_vpa_8_vperp_8","runs/2D-sound-wave_cheb_cxiz_nel_r_2_z_2_vpa_16_vperp_16"]
        scan_type = "velocity_nelement"
        scan_name = "2D-sound-wave_cheb_cxiz"
  
    elseif test_option == "collisionless_soundwaves"
        # collisionless
        path_list = ["runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_2_vperp_2","runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_4_vperp_4",
                        "runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_8_vperp_8","runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_16_vperp_16"]
        scan_type = "velocity_nelement"
        scan_name = "2D-sound-wave_cheb"
    end
    mk.plot_MMS_sequence.get_MMS_error_data(path_list,scan_type,scan_name)
end
