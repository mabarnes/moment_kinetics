if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk

   #test_option = "collisional_soundwaves"
   #test_option = "collisionless_soundwaves"
   test_option = "collisionless_wall-1D-3V-updated"
   #test_option = "collisionless_wall-1D-3V"
   #test_option = "collisionless_wall-2D-3V"
   #test_option = "collisionless_wall-2D-3V-Er-zero-at-plate"
   #test_option = "collisionless_biased_wall-2D-3V"
   #test_option = "collisionless_wall-1D-3V-with-sheath"

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
    elseif test_option == "collisionless_wall-2D-3V-Er-zero-at-plate"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_2_vperp_2",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_4_z_4_vpa_4_vperp_4",#"runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_12_vperp_12",
                        #"runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_16_vperp_16" 
						]
        scan_type = "velocity_nelement"
        scan_name = "2D-3V-wall_cheb"
    elseif test_option == "collisionless_wall-2D-3V"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_4_vperp_4",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_8_vperp_8","runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_12_vperp_12",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_16_vperp_16"                       ]
        scan_type = "velocity_nelement"
        scan_name = "2D-3V-wall_cheb"
    elseif test_option == "collisionless_biased_wall-2D-3V"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall-Dirichlet_nel_r_2_z_2_vpa_2","runs/2D-wall-Dirichlet_nel_r_2_z_2_vpa_4",
                     "runs/2D-wall-Dirichlet_nel_r_2_z_2_vpa_8","runs/2D-wall-Dirichlet_nel_r_2_z_2_vpa_16"]
        scan_type = "velocity_nelement"
        scan_name = "2D-3V-biased_wall_cheb"
    
    elseif test_option == "collisionless_wall-1D-3V"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_2_vperp_2","runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_4_vperp_4",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_8_vperp_8",#"runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_12_vperp_12",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_16_vperp_16"#,"runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_24_vperp_24"
                       ]
        scan_type = "velocity_nelement"
        scan_name = "1D-3V-wall_cheb"
    
    elseif test_option == "collisionless_wall-1D-3V-updated"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_2_vperp_2","runs/2D-wall_cheb-with-neutrals_nel_r_1_z_4_vpa_4_vperp_4",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_1_z_8_vpa_8_vperp_8",#"runs/2D-wall_cheb-with-neutrals_nel_r_1_z_12_vpa_12_vperp_12",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_1_z_16_vpa_16_vperp_16"#,"runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_24_vperp_24"
                       ]
        scan_type = "velocity_nelement"
        scan_name = "1D-3V-wall_cheb-updated"
    
    elseif test_option == "collisionless_wall-1D-3V-with-sheath"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_2_vperp_2","runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_4_vperp_4",
                        "runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_8_vperp_8",#"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_12_vperp_12",
                        "runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_16_vperp_16"#,"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_24_vperp_24"
                       ]
        scan_type = "velocity_nelement"
        scan_name = "1D-3V-wall-sheath_cheb"
    end
    mk.plot_MMS_sequence.get_MMS_error_data(path_list,scan_type,scan_name)
end
