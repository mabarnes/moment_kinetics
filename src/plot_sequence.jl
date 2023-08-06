"""
"""
module plot_sequence

# packages
using Plots
using IJulia
using LsqFit
using NCDatasets
using Statistics: mean
using SpecialFunctions: erfi
using LaTeXStrings
# modules
using ..post_processing: read_distributed_zr_data!, construct_global_zr_coords
using ..post_processing: allocate_global_zr_neutral_moments, allocate_global_zr_charged_moments
using ..post_processing: allocate_global_zr_fields#, get_coords_nelement
using ..array_allocation: allocate_float
using ..type_definitions: mk_float, mk_int
using ..load_data: open_readonly_output_file
using ..load_data: load_fields_data
using ..load_data: load_time_data, load_species_data
using ..load_data: load_block_data, load_coordinate_data
using ..moment_kinetics_input: mk_input, read_input_file

function plot_sequence_fields_data(path_list)
    
    nsimulation = length(path_list)
    nelement_sequence = zeros(mk_int,nsimulation)
    
    phi_z_list = []
    Ez_z_list = []
    z_list = []
    phi_sheath_entrance_list = []
    Ez_sheath_entrance_list = []
    
    for isim in 1:nsimulation
        path = path_list[isim]
        # Create run_name from the path to the run directory
        path = realpath(path)
        run_name = joinpath(path, basename(path))
        input_filename = path * ".toml"
        scan_input = read_input_file(input_filename)
        z_nelement, r_nelement, vpa_nelement, vperp_nelement, 
          vz_nelement, vr_nelement, vzeta_nelement = get_coords_nelement(scan_input)
        
        nelement_sequence[isim] = z_nelement
        
        # open the netcdf file and give it the handle 'fid'
        fid = open_readonly_output_file(run_name,"moments")
        # load block data on iblock=0
        nblocks, iblock = load_block_data(fid)
             
        # load global sizes of grids that are distributed in memory
        # load local sizes of grids stored on each netCDF file 
        # z z_wgts r r_wgts may take different values on different blocks
        # we need to construct the global grid below
        z_local, z_local_spectral, _ = load_coordinate_data(fid, "z")
        r_local, r_local_spectral, _ = load_coordinate_data(fid, "r")
        # load time data 
        ntime, time, _ = load_time_data(fid)
        # load species data 
        n_ion_species, n_neutral_species = load_species_data(fid)
        close(fid)
        
        # allocate arrays to contain the global fields as a function of (z,r,t)
        phi, Ez, Er = allocate_global_zr_fields(nz_global,nr_global,ntime)
        # read in the data from different block netcdf files
        # grids 
        r, r_spectral, z, z_spectral = construct_global_zr_coords(r_local, z_local)
        # fields 
        read_distributed_zr_data!(phi,"phi",run_name,"moments",nblocks,z_local.n,r_local.n)
        read_distributed_zr_data!(Ez,"Ez",run_name,"moments",nblocks,z_local.n,r_local.n)
        read_distributed_zr_data!(Er,"Er",run_name,"moments",nblocks,z_local.n,r_local.n)
        
        # store phi, Ez at ir = 1 and itime = end
        push!(phi_z_list,phi[:,1,end])
        push!(Ez_z_list,Ez[:,1,end])
        push!(z_list,z.grid[:])
        push!(phi_sheath_entrance_list,phi[1,1,end])
        push!(Ez_sheath_entrance_list,Ez[1,1,end])
    end
    #println(z_list)
    outprefix = "z_nelement_scan"
    xlabel_z = L"z/L_z"
    xlabel_nelement = L"N_{element}(z)"
    fontsize = 10
    legend_labels = Matrix{String}(undef, 1, nsimulation)
    for isim in 1:nsimulation
        legend_labels[isim] = string(nelement_sequence[isim])
    end
    # use GR backend
    gr()
    
    # plot phi(z_sheath_entrance) vs nelement 
    plot(nelement_sequence, phi_sheath_entrance_list, xlabel=xlabel_nelement, 
        label="", ylabel=L"\widetilde{\phi}(z_{sheath})", linewidth=2, xscale=:log10, xticks = (nelement_sequence, nelement_sequence), shape =:circle,
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*"_phi_sheath_entrance.pdf"
    savefig(outfile)
    println(outfile)
    
    # plot Ez(z_sheath_entrance) vs nelement 
    plot(nelement_sequence, Ez_sheath_entrance_list, xlabel=xlabel_nelement, 
        label="", ylabel=L"\widetilde{E}_z(z_{sheath})", linewidth=2, xscale=:log10, xticks = (nelement_sequence, nelement_sequence), shape =:circle,
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*"_Ez_sheath_entrance.pdf"
    savefig(outfile)
    println(outfile)
    
    # plot phi(z)
    plot(z_list, phi_z_list, xlabel=xlabel_z, 
        label=legend_labels, ylabel=L"\widetilde{\phi}(z)", linewidth=2, lengendtitle = L"N_{element}(z)",
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*"_phi_vs_z.pdf"
    savefig(outfile)
    println(outfile)
    # plot Ez(z)
    plot(z_list, Ez_z_list, xlabel=xlabel_z, 
        label=legend_labels, ylabel=L"\widetilde{E}_z(z)", linewidth=2, lengendtitle = L"N_{element}(z)",
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*"_Ez_vs_z.pdf"
    savefig(outfile)
    println(outfile)
end

end
