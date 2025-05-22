using Distributed

@everywhere using makie_post_processing

@everywhere const soundwave_postproc_input = Dict{String,Any}(
    "plot_vs_r" => false,
    "plot_vs_z" => false,
    "plot_vs_r_t" => false,
    "plot_vs_z_t" => false,
    "plot_vs_z_r" => false,
    "sound_wave_fit" => Dict{String,Any}("calculate_frequency"=>true, "plot"=>true),
    "itime_min" => 50,
    "timestep_diagnostics" => Dict{String,Any}("plot" => false),
    "collisionality_plots" => Dict{String,Any}("plot" => false),
   )

# get the run_names from the command-line
function post_process_parameter_scan(scan_dir)
    run_directories = Tuple(d for d ∈ readdir(scan_dir, join=true) if isdir(d))
    @sync @distributed for d ∈ run_directories
        println("post-processing ", d)
        try
            this_input = copy(soundwave_postproc_input)
            if (occursin("nratio", d) && occursin("ini_1.0_ini_1.0", d)) || occursin("T1", d)
                this_input["itime_min"] = 20
                this_input["itime_max"] = 70
                if occursin("cha_1.5", d)
                    this_input["itime_min"] = 20
                    this_input["itime_max"] = 65
                elseif occursin("cha_1.8", d)
                    this_input["itime_min"] = 20
                    this_input["itime_max"] = 60
                elseif occursin("cha_2.1", d)
                    this_input["itime_min"] = 20
                    this_input["itime_max"] = 50
                elseif occursin("cha_2.4", d)
                    this_input["itime_min"] = 20
                    this_input["itime_max"] = 60
                elseif occursin("cha_2.7", d)
                    this_input["itime_min"] = 25
                    this_input["itime_max"] = 65
                elseif occursin("cha_3.0", d)
                    continue
                elseif occursin("cha_3.3", d)
                    continue
                elseif occursin("cha_3.6", d)
                    continue
                elseif occursin("cha_3.9", d)
                    continue
                elseif occursin("cha_4.2", d)
                    continue
                elseif occursin("cha_4.5", d)
                    continue
                elseif occursin("cha_4.8", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 75
                elseif occursin("cha_5.1", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 75
                elseif occursin("cha_5.4", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 75
                elseif occursin("cha_5.7", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 75
                elseif occursin("cha_6.0", d)
                    this_input["itime_min"] = 55
                    this_input["itime_max"] = 75
                elseif occursin("cha_6.3", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 75
                elseif occursin("cha_6.6", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 80
                elseif occursin("cha_6.9", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 80
                elseif occursin("cha_7.2", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 80
                elseif occursin("cha_7.5", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 80
                elseif occursin("cha_7.8", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 85
                elseif occursin("cha_8.1", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 85
                elseif occursin("cha_8.4", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 85
                elseif occursin("cha_8.7", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 90
                elseif occursin("cha_9.0", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 90
                elseif occursin("cha_9.3", d)
                    this_input["itime_min"] = 50
                    this_input["itime_max"] = 90
                end
            elseif occursin("nratio", d)
                this_input["itime_min"] = 20
                this_input["itime_max"] = 70
                if occursin("ini_1.5_ini_0.5", d)
                    if occursin("cha_2.1", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 65
                    elseif occursin("cha_2.4", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 65
                    elseif occursin("cha_2.7", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_3.0", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_3.3", d)
                        continue
                    elseif occursin("cha_3.6", d)
                        continue
                    elseif occursin("cha_3.9", d)
                        continue
                    elseif occursin("cha_4.2", d)
                        continue
                    elseif occursin("cha_4.5", d)
                        continue
                    elseif occursin("cha_4.8", d)
                        continue
                    elseif occursin("cha_5.1", d)
                        continue
                    elseif occursin("cha_5.4", d)
                        continue
                    elseif occursin("cha_5.7", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 75
                    elseif occursin("cha_6.0", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 75
                    elseif occursin("cha_6.3", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 75
                    elseif occursin("cha_6.6", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_6.9", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_7.2", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_7.5", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_7.8", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_8.1", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_8.4", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_8.7", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    elseif occursin("cha_9.0", d)
                        this_input["itime_min"] = 60
                        this_input["itime_max"] = 80
                    end
                elseif occursin("ini_0.5_ini_1.5", d)
                    this_input["itime_min"] = 35
                    this_input["itime_max"] = 75
                    if occursin("cha_0.0", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 70
                    elseif occursin("cha_0.3", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 70
                    elseif occursin("cha_0.6", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 65
                    elseif occursin("cha_0.9", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_1.2", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_1.5", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_1.8", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 60
                    elseif occursin("cha_2.1", d)
                        this_input["itime_min"] = 30
                        this_input["itime_max"] = 60
                    elseif occursin("cha_2.4", d)
                        this_input["itime_min"] = 30
                        this_input["itime_max"] = 60
                    elseif occursin("cha_2.7", d)
                        this_input["itime_min"] = 30
                        this_input["itime_max"] = 60
                    elseif occursin("cha_3.0", d)
                        continue
                    elseif occursin("cha_3.3", d)
                        continue
                    elseif occursin("cha_3.6", d)
                        continue
                    elseif occursin("cha_3.9", d)
                        continue
                        #this_input["itime_min"] = 1
                        #this_input["itime_max"] = 120
                    elseif occursin("cha_4.2", d)
                        this_input["itime_min"] = 40
                        this_input["itime_max"] = 70
                    elseif occursin("cha_4.5", d)
                        this_input["itime_min"] = 50
                        this_input["itime_max"] = 70
                    elseif occursin("cha_4.8", d)
                        this_input["itime_min"] = 45
                        this_input["itime_max"] = 70
                    elseif occursin("cha_5.1", d)
                        this_input["itime_min"] = 45
                        this_input["itime_max"] = 70
                    elseif occursin("cha_5.4", d)
                        this_input["itime_min"] = 40
                        this_input["itime_max"] = 70
                    elseif occursin("cha_5.7", d)
                        this_input["itime_min"] = 40
                        this_input["itime_max"] = 70
                    elseif occursin("cha_6.0", d)
                        this_input["itime_min"] = 40
                        this_input["itime_max"] = 75
                    elseif occursin("cha_6.3", d)
                        this_input["itime_min"] = 40
                        this_input["itime_max"] = 75
                    elseif occursin("cha_6.6", d)
                        this_input["itime_min"] = 40
                        this_input["itime_max"] = 75
                    end
                elseif occursin("ini_1.0e-5_ini_1.99999", d)
                    this_input["itime_min"] = 25
                    this_input["itime_max"] = 70
                    if occursin("cha_0.0", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 70
                    elseif occursin("cha_0.3", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 70
                    elseif occursin("cha_0.6", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 65
                    elseif occursin("cha_0.9", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 65
                    elseif occursin("cha_1.28", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 65
                    elseif occursin("cha_1.5", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_1.8", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_2.1", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 60
                    elseif occursin("cha_2.4", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 55
                    elseif occursin("cha_2.7", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 55
                    elseif occursin("cha_3.0", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 50
                    elseif occursin("cha_3.3", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 50
                    elseif occursin("cha_3.6", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 45
                    elseif occursin("cha_3.9", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 45
                    elseif occursin("cha_4.2", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 50
                    elseif occursin("cha_4.5", d)
                        this_input["itime_min"] = 20
                        this_input["itime_max"] = 50
                    elseif occursin("cha_4.8", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 55
                    elseif occursin("cha_5.1", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 55
                    elseif occursin("cha_5.4", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 60
                    elseif occursin("cha_5.7", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 65
                    elseif occursin("cha_6.0", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 70
                    elseif occursin("cha_6.3", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 70
                    elseif occursin("cha_6.6", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 70
                    elseif occursin("cha_6.9", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 70
                    elseif occursin("cha_7.2", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 70
                    elseif occursin("cha_7.5", d)
                        this_input["itime_min"] = 25
                        this_input["itime_max"] = 70
                    end
                end
            elseif occursin("T0.25", d) || occursin("ini_0.25", d)
                this_input["itime_min"] = 20
                if occursin("cha_4.2", d)
                    continue
                elseif occursin("cha_4.5", d)
                    continue
                end
            elseif occursin("T0.5", d) || occursin("ini_0.5", d)
                this_input["itime_min"] = 20
                this_input["itime_max"] = 76
                if occursin("cha_3.3", d)
                    this_input["itime_min"] = 30
                    this_input["itime_max"] = 70
                elseif occursin("cha_3.6", d)
                    continue
                elseif occursin("cha_3.9", d)
                    continue
                elseif occursin("cha_4.2", d)
                    continue
                elseif occursin("cha_4.5", d)
                    continue
                elseif occursin("cha_4.8", d)
                    continue
                elseif occursin("cha_5.1", d)
                    continue
                elseif occursin("cha_5.4", d)
                    this_input["itime_min"] = 80
                    this_input["itime_max"] = 90
                elseif occursin("cha_5.7", d)
                    this_input["itime_min"] = 75
                    this_input["itime_max"] = 90
                elseif occursin("cha_6.0", d)
                    this_input["itime_min"] = 75
                    this_input["itime_max"] = 90
                elseif occursin("cha_6.3", d)
                    this_input["itime_min"] = 79
                    this_input["itime_max"] = 91
                elseif occursin("cha_6.6", d)
                    continue
                end
            elseif occursin("T2", d) || occursin("ini_2.0", d)
                this_input["itime_min"] = 5
                this_input["itime_max"] = 75
                if occursin("cha_0.0", d)
                    this_input["itime_min"] = 10
                    this_input["itime_max"] = 70
                elseif occursin("cha_0.4", d)
                    this_input["itime_min"] = 10
                    this_input["itime_max"] = 70
                elseif occursin("cha_0.8", d)
                    this_input["itime_min"] = 10
                    this_input["itime_max"] = 70
                elseif occursin("cha_1.2", d)
                    this_input["itime_min"] = 20
                    this_input["itime_max"] = 65
                elseif occursin("cha_1.6", d)
                    this_input["itime_min"] = 30
                    this_input["itime_max"] = 60
                elseif occursin("cha_2.0", d)
                    # Can't really get a good fit here. The decay rates of the two modes
                    # are too close, so just get beating for the whole length of the
                    # simulation.
                    continue
                elseif occursin("cha_2.4", d)
                    # Can't really get a good fit here. The decay rates of the two modes
                    # are too close, so just get beating for the whole length of the
                    # simulation.
                    continue
                elseif occursin("cha_2.8", d)
                    # Can't really get a good fit here. The decay rates of the two modes
                    # are too close, so just get beating for the whole length of the
                    # simulation.
                    continue
                elseif occursin("cha_3.2", d)
                    # Can't really get a good fit here. The decay rates of the two modes
                    # are too close, so just get beating for the whole length of the
                    # simulation.
                    continue
                elseif occursin("cha_3.6", d)
                    # Can't really get a good fit here. The decay rates of the two modes
                    # are too close, so just get beating for the whole length of the
                    # simulation.
                    continue
                elseif occursin("cha_4.0", d)
                    this_input["itime_min"] = 55
                elseif occursin("cha_4.4", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_4.8", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_5.2", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_5.6", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_6.0", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_6.4", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_6.8", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_7.2", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_7.6", d)
                    this_input["itime_min"] = 45
                elseif occursin("cha_8.0", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 80
                elseif occursin("cha_8.4", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_8.8", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_9.2", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_9.6", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_10.0", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_10.4", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_10.8", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_11.2", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_11.6", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                elseif occursin("cha_12.0", d)
                    this_input["itime_min"] = 45
                    this_input["itime_max"] = 91
                end
            elseif occursin("T4", d) || occursin("ini_4.0", d)
                this_input["itime_min"] = 10
                this_input["itime_max"] = 50
                if occursin("cha_2.4", d)
                    continue
                elseif occursin("cha_3.0", d)
                    continue
                elseif occursin("cha_3.6", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 60
                elseif occursin("cha_4.2", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 65
                elseif occursin("cha_4.8", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 65
                elseif occursin("cha_5.4", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 65
                elseif occursin("cha_6.0", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 65
                elseif occursin("cha_6.6", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 70
                elseif occursin("cha_7.2", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 70
                elseif occursin("cha_7.8", d)
                    this_input["itime_min"] = 35
                    this_input["itime_max"] = 70
                elseif occursin("cha_8.4", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 70
                elseif occursin("cha_9.0", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 70
                elseif occursin("cha_9.6", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 70
                elseif occursin("cha_10.2", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 75
                elseif occursin("cha_10.8", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 75
                elseif occursin("cha_11.4", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 75
                elseif occursin("cha_12.0", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 75
                elseif occursin("cha_12.6", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_13.2", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_13.8", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_14.4", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_15.0", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_15.6", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_16.2", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_16.8", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_17.4", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_18.0", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                elseif occursin("cha_18.6", d)
                    this_input["itime_min"] = 40
                    this_input["itime_max"] = 80
                end
            end
            makie_post_process(d, this_input)
        catch e
            println(d, " failed with ", e)
        end
    end
end

function post_process_all_scans()
    post_process_parameter_scan("../../runs/scan_sound-wave_nratio/")
    post_process_parameter_scan("../../runs/scan_sound-wave_T0.25/")
    post_process_parameter_scan("../../runs/scan_sound-wave_T0.5/")
    post_process_parameter_scan("../../runs/scan_sound-wave_T1/")
    post_process_parameter_scan("../../runs/scan_sound-wave_T2/")
    post_process_parameter_scan("../../runs/scan_sound-wave_T4/")
    for i ∈ 1:3
        post_process_parameter_scan("../../runs/scan_sound-wave_nratio_split$i/")
        post_process_parameter_scan("../../runs/scan_sound-wave_T0.25_split$i/")
        post_process_parameter_scan("../../runs/scan_sound-wave_T0.5_split$i/")
        post_process_parameter_scan("../../runs/scan_sound-wave_T1_split$i/")
        post_process_parameter_scan("../../runs/scan_sound-wave_T2_split$i/")
        post_process_parameter_scan("../../runs/scan_sound-wave_T4_split$i/")
    end

    #post_process_parameter_scan("../../runs/scan_sound-wave_lowres/")
    #post_process_parameter_scan("../../runs/scan_sound-wave_T/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    post_process_all_scans()
end
