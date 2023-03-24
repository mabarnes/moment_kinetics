if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk

   #test_option = "collisional_soundwaves"
   #test_option = "collisionless_soundwaves_ions_only"
   #test_option = "collisionless_soundwaves"
   #test_option = "collisionless_wall-1D-3V-new-dfni-Er"
   #test_option = "collisionless_wall-2D-1V-Er-nonzero-at-plate"
   #test_option = "collisionless_wall-2D-1V-Er-zero-at-plate"
   #test_option = "collisionless_wall-1D-1V"
   #test_option = "collisionless_wall-1D-1V-constant-Er"
   #test_option = "collisionless_wall-1D-1V-constant-Er-zngrid-5"
   #test_option = "collisionless_wall-1D-1V-constant-Er-ngrid-5"
   test_option = "collisionless_wall-1D-1V-constant-Er-ngrid-5-opt"
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
        path_list = ["runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_2_vperp_2","runs/2D-sound-wave_cheb_nel_r_4_z_4_vpa_4_vperp_4",# "runs/2D-sound-wave_cheb_nel_r_6_z_6_vpa_6_vperp_6",
                     "runs/2D-sound-wave_cheb_nel_r_8_z_8_vpa_8_vperp_8","runs/2D-sound-wave_cheb_nel_r_16_z_16_vpa_16_vperp_16"
                     ]
        scan_type = "nelement"
        scan_name = "2D-sound-wave_cheb"
    elseif test_option == "collisionless_soundwaves_ions_only"
        # collisionless
        path_list = ["runs/2D-sound-wave_cheb_ion_only_nel_r_2_z_2_vpa_2_vperp_2","runs/2D-sound-wave_cheb_ion_only_nel_r_4_z_4_vpa_4_vperp_4", "runs/2D-sound-wave_cheb_ion_only_nel_r_8_z_8_vpa_8_vperp_8",
#                        "runs/2D-sound-wave_cheb_nel_r_8_z_8_vpa_8_vperp_8"
#,"runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_16_vperp_16"
                     ]
        scan_type = "nelement"
        scan_name = "2D-sound-wave_cheb_ions_only"
    elseif test_option == "collisionless_wall-2D-3V-Er-zero-at-plate"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_2_vperp_2",
                        "runs/2D-wall_cheb-with-neutrals_nel_r_4_z_4_vpa_4_vperp_4",
						"runs/2D-wall_cheb-with-neutrals_nel_r_6_z_6_vpa_6_vperp_6",
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
           #             "runs/2D-wall_cheb-with-neutrals_nel_r_1_z_16_vpa_16_vperp_16"#,"runs/2D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_24_vperp_24"
                       ]
        scan_type = "nelement"
        scan_name = "1D-3V-wall_cheb-updated"
    elseif test_option == "collisionless_wall-1D-3V-new-dfni"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-new-MMS-dfni-ngrid-9_nel_r_1_z_2_vpa_2_vperp_2",
                     "runs/2D-wall_cheb-new-MMS-dfni-ngrid-9_nel_r_1_z_4_vpa_4_vperp_4",
                     "runs/2D-wall_cheb-new-MMS-dfni-ngrid-9_nel_r_1_z_8_vpa_8_vperp_8",
                     "runs/2D-wall_cheb-new-MMS-dfni-ngrid-9_nel_r_1_z_16_vpa_16_vperp_16",
                     "runs/2D-wall_cheb-new-MMS-dfni-ngrid-9_nel_r_1_z_32_vpa_32_vperp_32"
                       ]
        scan_type = "nelement"
        scan_name = "1D-3V-wall_cheb-new-dfni"
    elseif test_option == "collisionless_wall-1D-3V-new-dfni-Er"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-new-MMS-dfni-Er-ngrid-9_nel_r_1_z_2_vpa_2_vperp_2",
                     "runs/2D-wall_cheb-new-MMS-dfni-Er-ngrid-9_nel_r_1_z_4_vpa_4_vperp_4",
                     "runs/2D-wall_cheb-new-MMS-dfni-Er-ngrid-9_nel_r_1_z_8_vpa_8_vperp_8",
                     "runs/2D-wall_cheb-new-MMS-dfni-Er-ngrid-9_nel_r_1_z_16_vpa_16_vperp_16",
                     "runs/2D-wall_cheb-new-MMS-dfni-Er-ngrid-9_nel_r_1_z_32_vpa_32_vperp_32"
                       ]
        scan_type = "nelement"
        scan_name = "1D-3V-wall_cheb-new-dfni-Er"
    elseif test_option == "collisionless_wall-1D-3V-with-sheath"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_2_vperp_2","runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_4_vperp_4",
                        "runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_8_vperp_8",#"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_12_vperp_12",
                        "runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_16_vperp_16"#,"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_24_vperp_24"
                       ]
        scan_type = "velocity_nelement"
        scan_name = "1D-3V-wall-sheath_cheb"
    elseif test_option == "collisionless_wall-2D-1V-Er-zero-at-plate"
        #path_list = ["runs/2D-wall_MMS_nel_r_2_z_2_vpa_16_vperp_1_diss","runs/2D-wall_MMS_nel_r_4_z_4_vpa_16_vperp_1_diss",
        #                "runs/2D-wall_MMS_nel_r_8_z_8_vpa_16_vperp_1_diss",#"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_12_vperp_12",
        #                "runs/2D-wall_MMS_nel_r_16_z_16_vpa_16_vperp_1_diss",
        #                "runs/2D-wall_MMS_nel_r_32_z_32_vpa_16_vperp_1_diss5"#,"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_24_vperp_24"
        #               ]
        #scan_type = "zr_nelement"
        path_list = ["runs/2D-wall_MMS_ngrid_5_nel_r_2_z_2_vpa_8_vperp_1_diss","runs/2D-wall_MMS_ngrid_5_nel_r_4_z_4_vpa_16_vperp_1_diss",
                        "runs/2D-wall_MMS_ngrid_5_nel_r_8_z_8_vpa_32_vperp_1_diss","runs/2D-wall_MMS_ngrid_5_nel_r_16_z_16_vpa_64_vperp_1_diss"]
        scan_type = "vpazr_nelement0.25"
        scan_name = "2D-1V-wall_cheb"
    elseif test_option == "collisionless_wall-2D-1V-Er-nonzero-at-plate"
        path_list = ["runs/2D-wall_MMSEr_ngrid_5_nel_r_2_z_2_vpa_8_vperp_1_diss","runs/2D-wall_MMSEr_ngrid_5_nel_r_4_z_4_vpa_16_vperp_1_diss","runs/2D-wall_MMSEr_ngrid_5_nel_r_8_z_8_vpa_32_vperp_1_diss","runs/2D-wall_MMSEr_ngrid_5_nel_r_16_z_16_vpa_64_vperp_1_diss",
                    ]
        #path_list = ["runs/2D-wall_MMSEr_nel_r_2_z_2_vpa_2_vperp_1_diss","runs/2D-wall_MMSEr_nel_r_4_z_4_vpa_4_vperp_1_diss",
        #                "runs/2D-wall_MMSEr_nel_r_8_z_8_vpa_8_vperp_1_diss","runs/2D-wall_MMSEr_nel_r_16_z_16_vpa_16_vperp_1_diss"
        #                #"runs/2D-wall_MMSEr_nel_r_32_z_32_vpa_16_vperp_1_diss"
        #            ]
        #path_list = ["runs/2D-wall_MMSEr_nel_r_2_z_2_vpa_16_vperp_1_diss","runs/2D-wall_MMSEr_nel_r_4_z_4_vpa_16_vperp_1_diss",
        #                "runs/2D-wall_MMSEr_nel_r_8_z_8_vpa_16_vperp_1_diss",#"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_12_vperp_12",
        #                "runs/2D-wall_MMSEr_nel_r_16_z_16_vpa_16_vperp_1_diss",
        #                "runs/2D-wall_MMSEr_nel_r_32_z_32_vpa_16_vperp_1_diss"#,"runs/2D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_24_vperp_24"
        #               ]
        scan_type = "vpazr_nelement0.25"
        scan_name = "2D-1V-wall_cheb-nonzero-Er"
    elseif test_option == "collisionless_wall-1D-1V-constant-Er-ngrid-5-opt"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_8_vpa_32_vperp_1","runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_16_vpa_64_vperp_1",
                        "runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_32_vpa_128_vperp_1","runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_64_vpa_256_vperp_1"
                    ]
        scan_type = "vpaz_nelement0.25"
        scan_name = "1D-1V-wall_cheb-constant-Er-ngrid-5-opt"
    elseif test_option == "collisionless_wall-1D-1V-constant-Er-ngrid-5"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_8_vpa_8_vperp_1","runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_16_vpa_16_vperp_1",
                        "runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_32_vpa_32_vperp_1","runs/1D-wall_MMSEr_ngrid_5_nel_r_1_z_64_vpa_64_vperp_1"
                    ]
        scan_type = "vpaz_nelement"
        scan_name = "1D-1V-wall_cheb-constant-Er-ngrid-5"
    elseif test_option == "collisionless_wall-1D-1V-constant-Er-zngrid-5"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/1D-wall_MMSEr_zngrid_5_nel_r_1_z_8_vpa_2_vperp_1","runs/1D-wall_MMSEr_zngrid_5_nel_r_1_z_16_vpa_4_vperp_1",
                        "runs/1D-wall_MMSEr_zngrid_5_nel_r_1_z_32_vpa_8_vperp_1","runs/1D-wall_MMSEr_zngrid_5_nel_r_1_z_64_vpa_16_vperp_1"
                    ]
        scan_type = "vpaz_nelement4"
        scan_name = "1D-1V-wall_cheb-constant-Er-zngrid-5"
    elseif test_option == "collisionless_wall-1D-1V-constant-Er"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/1D-wall_MMSEr_nel_r_1_z_2_vpa_2_vperp_1","runs/1D-wall_MMSEr_nel_r_1_z_4_vpa_4_vperp_1",
                        "runs/1D-wall_MMSEr_nel_r_1_z_8_vpa_8_vperp_1","runs/1D-wall_MMSEr_nel_r_1_z_16_vpa_16_vperp_1"
                    ]
        scan_type = "vpaz_nelement"
        scan_name = "1D-1V-wall_cheb-constant-Er"
    elseif test_option == "collisionless_wall-1D-1V"
        # collisionless wall test, no sheath for electrons, no radial coordinate
        path_list = ["runs/1D-wall_MMS_nel_r_1_z_2_vpa_2_vperp_1","runs/1D-wall_MMS_nel_r_1_z_4_vpa_4_vperp_1",
                        "runs/1D-wall_MMS_nel_r_1_z_8_vpa_8_vperp_1","runs/1D-wall_MMS_nel_r_1_z_16_vpa_16_vperp_1"
                    ]
        scan_type = "vpaz_nelement"
        scan_name = "1D-1V-wall_cheb"
    end
    mk.plot_MMS_sequence.get_MMS_error_data(path_list,scan_type,scan_name)
end
