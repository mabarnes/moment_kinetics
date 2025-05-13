# Note, run this from top-level moment-kinetics directory

include(joinpath("..", "..", "..", "run_parameter_scan.jl"))

sound_wave_dir = joinpath("publication_inputs", "1D1V-paper", "sound-wave")
input_files = (
    "scan_sound-wave_T0.25.toml",
    "scan_sound-wave_T0.25_split1.toml",
    "scan_sound-wave_T0.25_split2.toml",
    "scan_sound-wave_T0.25_split3.toml",
    "scan_sound-wave_T0.5.toml",
    "scan_sound-wave_T0.5_split1.toml",
    "scan_sound-wave_T0.5_split2.toml",
    "scan_sound-wave_T0.5_split3.toml",
    "scan_sound-wave_T1.toml",
    "scan_sound-wave_T1_split1.toml",
    "scan_sound-wave_T1_split2.toml",
    "scan_sound-wave_T1_split3.toml",
    "scan_sound-wave_T2.toml",
    "scan_sound-wave_T2_split1.toml",
    "scan_sound-wave_T2_split2.toml",
    "scan_sound-wave_T2_split3.toml",
    "scan_sound-wave_T4.toml",
    "scan_sound-wave_T4_split1.toml",
    "scan_sound-wave_T4_split2.toml",
    "scan_sound-wave_T4_split3.toml",
    "scan_sound-wave_nratio.toml",
    "scan_sound-wave_nratio_split1.toml",
    "scan_sound-wave_nratio_split2.toml",
    "scan_sound-wave_nratio_split3.toml",
)

for f ∈ input_files
    println("running scan from $f")
    run_parameter_scan(joinpath(sound_wave_dir, f))
end
