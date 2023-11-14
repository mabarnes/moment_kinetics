"""
"""
module plot_MMS_sequence

export get_MMS_error_data

# packages
using Plots
using IJulia
using LsqFit
using NCDatasets
using Statistics: mean
using SpecialFunctions: erfi
using LaTeXStrings
# modules
using ..post_processing_input: pp
using ..post_processing: compare_charged_pdf_symbolic_test, compare_fields_symbolic_test
using ..post_processing: compare_moments_symbolic_test, compare_neutral_pdf_symbolic_test
using ..post_processing: read_distributed_zr_data!, construct_global_zr_coords
using ..post_processing: allocate_global_zr_neutral_moments, allocate_global_zr_charged_moments
using ..post_processing: allocate_global_zr_fields, get_geometry_and_composition, get_coords_nelement
using ..array_allocation: allocate_float
using ..type_definitions: mk_float, mk_int
using ..load_data: open_readonly_output_file
using ..load_data: load_fields_data, load_pdf_data
using ..load_data: load_charged_particle_moments_data, load_neutral_particle_moments_data
using ..load_data: load_neutral_pdf_data, load_time_data, load_species_data
using ..load_data: load_block_data, load_coordinate_data, load_input
using ..velocity_moments: integrate_over_vspace
using ..manufactured_solns: manufactured_solutions, manufactured_electric_fields
using ..moment_kinetics_input: mk_input, read_input_file

import Base: get

# assume in function below that we have a list of simulations 
# where only a single nelement parameter is varied
# we plot the MMS error measurements as a fn of nelement
function get_MMS_error_data(path_list,scan_type,scan_name)
    Plots.font(family=:serif)
    
    nsimulation = length(path_list)
    phi_error_sequence = zeros(mk_float,nsimulation)
    Er_error_sequence = zeros(mk_float,nsimulation)
    Ez_error_sequence = zeros(mk_float,nsimulation)
    ion_density_error_sequence = zeros(mk_float,nsimulation)
    ion_pdf_error_sequence = zeros(mk_float,nsimulation)
    neutral_density_error_sequence = zeros(mk_float,nsimulation)
    neutral_pdf_error_sequence = zeros(mk_float,nsimulation)
    nelement_sequence = zeros(mk_int,nsimulation)
    
    # declare local variables that are needed outside "nsimulation" loop below
    local n_neutral_species
    
    for isim in 1:nsimulation
        path = path_list[isim]
        # Create run_name from the path to the run directory
        path = realpath(path)
        run_name = joinpath(path, basename(path))
        input_filename = path * ".toml"

        # open the netcdf file and give it the handle 'fid'
        fid = open_readonly_output_file(run_name,"moments")

        scan_input = load_input(fid)
        # get run-time input/composition/geometry/collisions/species info for convenience
        #io_input, evolve_moments, t_input, z, z_spectral, r, r_spectral, vpa, vpa_spectral,
        #    vperp, vperp_spectral, gyrophase, gyrophase_spectral, vz, vz_spectral, vr,
        #    vr_spectral, vzeta, vzeta_spectral, composition, species, collisions, geometry,
        #    drive_input, external_source_settings, num_diss_params,
        #    manufactured_solns_input = mk_input(scan_input)
        z_nelement, r_nelement, vpa_nelement, vperp_nelement, 
          vz_nelement, vr_nelement, vzeta_nelement = get_coords_nelement(scan_input)
        if scan_type == "vpa_nelement"
            # get the number of elements for plot
            nelement_sequence[isim] = vpa_nelement
        elseif scan_type == "nelement"
            nelement = vpa_nelement
            if  nelement == r_nelement &&nelement == z_nelement && nelement == vperp_nelement && nelement == vz_nelement && nelement == vr_nelement && nelement == vzeta_nelement
                nelement_sequence[isim] = nelement
            elseif  1 == r_nelement && nelement == z_nelement && nelement == vperp_nelement && nelement == vz_nelement && nelement == vr_nelement && nelement == vzeta_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires element number to be equal in all dimensions")
            end
        elseif scan_type == "velocity_nelement"
            nelement = vpa_nelement
            if nelement == vperp_nelement && nelement == vz_nelement && nelement == vr_nelement && nelement == vzeta_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires velocity elements equal in all dimensions")
            end
        elseif scan_type == "zr_nelement"
            nelement = z_nelement
            if nelement == r_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires z & r elements equal")
            end
        elseif scan_type == "vpaz_nelement0.25"
            nelement = z_nelement
            if nelement*4 == vpa_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires vpa_nelement = z_nelement*4")
            end
        elseif scan_type == "vpazr_nelement0.25"
            nelement = z_nelement
            if nelement*4 == vpa_nelement && nelement == r_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires vpa_nelement = z_nelement*4 = r_nelement*4")
            end
        elseif scan_type == "vpaz_nelement4"
            nelement = z_nelement
            if nelement/4 == vpa_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires vpa_nelement = z_nelement/4")
            end
        elseif scan_type == "vpaz_nelement"
            nelement = z_nelement
            if nelement == vpa_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires vpa & z elements equal")
            end
        elseif scan_type == "vpazr_nelement"
            nelement = z_nelement
            if nelement == r_nelement && nelement == vpa_nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires vpa & z & r elements equal")
            end
        else 
            println("ERROR: scan_type = ",scan_type," is unsupported")
        end

        # load block data on iblock=0
        nblocks, iblock = load_block_data(fid)
             
        # load global sizes of grids that are distributed in memory
        # load local sizes of grids stored on each netCDF file 
        # z z_wgts r r_wgts may take different values on different blocks
        # we need to construct the global grid below
        z, z_spectral, _ = load_coordinate_data(fid, "z")
        r, r_spectral, _ = load_coordinate_data(fid, "r")
        # load time data 
        ntime, time, _ = load_time_data(fid)
        # load species data 
        n_ion_species, n_neutral_species = load_species_data(fid)
        close(fid)
        
        # load local velocity coordinate data from `moments' cdf
        # these values are currently the same for all blocks 
        fid = open_readonly_output_file(run_name,"dfns")
        vpa, _, _ = load_coordinate_data(fid, "vpa")
        vperp, _, _ = load_coordinate_data(fid, "vperp")
        #vz, _, _ = load_coordinate_data(fid, "vz")
        #vr, _, _ = load_coordinate_data(fid, "vr")
        #vzeta, _, _ = load_coordinate_data(fid, "vzeta")
        close(fid)
        
        
        # allocate arrays to contain the global fields as a function of (z,r,t)
        phi, Ez, Er = allocate_global_zr_fields(z.n_global,r.n_global,ntime)
        density, parallel_flow, parallel_pressure, parallel_heat_flux,
            thermal_speed = allocate_global_zr_charged_moments(z.n_global,r.n_global,n_ion_species,ntime)
        if n_neutral_species > 0
            neutral_density, neutral_uz, neutral_pz, 
             neutral_qz, neutral_thermal_speed = allocate_global_zr_neutral_moments(z.n_global,r.n_global,n_neutral_species,ntime)
        end 
        # read in the data from different block netcdf files
        # grids 
        r_global, z_global = construct_global_zr_coords(r, z)
        #println("z: ",z)
        #println("r: ",r)
        
        # fields 
        read_distributed_zr_data!(phi,"phi",run_name,"moments",nblocks,z.n,r.n) 
        read_distributed_zr_data!(Ez,"Ez",run_name,"moments",nblocks,z.n,r.n) 
        read_distributed_zr_data!(Er,"Er",run_name,"moments",nblocks,z.n,r.n) 
        # charged particle moments
        read_distributed_zr_data!(density,"density",run_name,"moments",nblocks,z.n,r.n) 
        read_distributed_zr_data!(parallel_flow,"parallel_flow",run_name,"moments",nblocks,z.n,r.n) 
        read_distributed_zr_data!(parallel_pressure,"parallel_pressure",run_name,"moments",nblocks,z.n,r.n) 
        read_distributed_zr_data!(parallel_heat_flux,"parallel_heat_flux",run_name,"moments",nblocks,z.n,r.n) 
        read_distributed_zr_data!(thermal_speed,"thermal_speed",run_name,"moments",nblocks,z.n,r.n) 
        # neutral particle moments 
        if n_neutral_species > 0
            read_distributed_zr_data!(neutral_density,"density_neutral",run_name,"moments",nblocks,z.n,r.n) 
            read_distributed_zr_data!(neutral_uz,"uz_neutral",run_name,"moments",nblocks,z.n,r.n) 
            read_distributed_zr_data!(neutral_pz,"pz_neutral",run_name,"moments",nblocks,z.n,r.n) 
            read_distributed_zr_data!(neutral_qz,"qz_neutral",run_name,"moments",nblocks,z.n,r.n) 
            read_distributed_zr_data!(neutral_thermal_speed,"thermal_speed_neutral",run_name,"moments",nblocks,z.n,r.n) 
        end
        
        
        r_bc = get(scan_input, "r_bc", "periodic")
        z_bc = get(scan_input, "z_bc", "periodic")
        # avoid passing Lr = 0 into manufactured_solns functions 
        if r.n > 1
            Lr_in = r.L 
        else 
            Lr_in = 1.0
        end
        geometry, composition = get_geometry_and_composition(scan_input,n_ion_species,n_neutral_species)
        
        manufactured_solns_list = manufactured_solutions(Lr_in,z.L,r_bc,z_bc,geometry,composition,r.n) 
        dfni_func = manufactured_solns_list.dfni_func
        densi_func = manufactured_solns_list.densi_func
        dfnn_func = manufactured_solns_list.dfnn_func
        densn_func = manufactured_solns_list.densn_func
        
        manufactured_E_fields = manufactured_electric_fields(Lr_in,z.L,r_bc,z_bc,composition,r.n)
        Er_func = manufactured_E_fields.Er_func
        Ez_func = manufactured_E_fields.Ez_func
        phi_func = manufactured_E_fields.phi_func
        
        # phi, Er, Ez test
        phi_sym = copy(phi[:,:,:])
        Er_sym = copy(phi[:,:,:])
        Ez_sym = copy(phi[:,:,:])
        for it in 1:ntime
            for ir in 1:r.n_global
                for iz in 1:z.n_global
                    phi_sym[iz,ir,it] = phi_func(z.grid[iz],r.grid[ir],time[it])
                    Ez_sym[iz,ir,it] = Ez_func(z.grid[iz],r.grid[ir],time[it])
                    Er_sym[iz,ir,it] = Er_func(z.grid[iz],r.grid[ir],time[it])
                end
            end
        end
        phi_error_t = compare_fields_symbolic_test(run_name,phi,phi_sym,z.grid,r.grid,time,z.n_global,r.n_global,ntime,
         L"\widetilde{\phi}",L"\widetilde{\phi}^{sym}",L"\sqrt{\sum || \widetilde{\phi} - \widetilde{\phi}^{sym} ||^2 / N} ","phi")
        phi_error_sequence[isim] = phi_error_t[end]
        Er_error_t = compare_fields_symbolic_test(run_name,Er,Er_sym,z.grid,r.grid,time,z.n_global,r.n_global,ntime,
         L"\widetilde{E_r}",L"\widetilde{E_r}^{sym}",L"\sqrt{\sum || \widetilde{E_r} - \widetilde{E_r}^{sym} ||^2 /N} ","Er")
        Er_error_sequence[isim] = Er_error_t[end]
        Ez_error_t = compare_fields_symbolic_test(run_name,Ez,Ez_sym,z.grid,r.grid,time,z.n_global,r.n_global,ntime,
         L"\widetilde{E_z}",L"\widetilde{E_z}^{sym}",L"\sqrt{\sum || \widetilde{E_z} - \widetilde{E_z}^{sym} ||^2 /N} ","Ez")
        Ez_error_sequence[isim] = Ez_error_t[end]
        
        # ion test
        density_sym = copy(density[:,:,:,:])
        is = 1
        for it in 1:ntime
            for ir in 1:r.n_global
                for iz in 1:z.n_global
                    density_sym[iz,ir,is,it] = densi_func(z.grid[iz],r.grid[ir],time[it])
                end
            end
        end
        ion_density_error_t = compare_moments_symbolic_test(run_name,density,density_sym,"ion",z.grid,r.grid,time,z.n_global,r.n_global,ntime,
         L"\widetilde{n}_i",L"\widetilde{n}_i^{sym}",L"\sum || \widetilde{n}_i - \widetilde{n}_i^{sym} ||^2 ","dens")
        # use final time point for analysis
        ion_density_error_sequence[isim] = ion_density_error_t[end] 
        
        ion_pdf_error_t = compare_charged_pdf_symbolic_test(run_name,manufactured_solns_list,"ion",
         L"\widetilde{f}_i",L"\widetilde{f}^{sym}_i",L"\sum || \widetilde{f}_i - \widetilde{f}_i^{sym} ||^2","pdf")
        ion_pdf_error_sequence[isim] = ion_pdf_error_t[end]
        
        if n_neutral_species > 0
            # neutral test
            neutral_density_sym = copy(density[:,:,:,:])
            is = 1
            for it in 1:ntime
                for ir in 1:r.n_global
                    for iz in 1:z.n_global
                        neutral_density_sym[iz,ir,is,it] = densn_func(z.grid[iz],r.grid[ir],time[it])
                    end
                end
            end
            neutral_density_error_t = compare_moments_symbolic_test(run_name,neutral_density,neutral_density_sym,"neutral",z.grid,r.grid,time,z.n_global,r.n_global,ntime,
             L"\widetilde{n}_n",L"\widetilde{n}_n^{sym}",L"\sum || \widetilde{n}_n - \widetilde{n}_n^{sym} ||^2 ","dens")
            neutral_density_error_sequence[isim] = neutral_density_error_t[end]
        
            neutral_pdf_error_t = compare_neutral_pdf_symbolic_test(run_name,manufactured_solns_list,"neutral",
             L"\widetilde{f}_n",L"\widetilde{f}^{sym}_n",L"\sum || \widetilde{f}_n - \widetilde{f}_n^{sym} ||^2","pdf")
            neutral_pdf_error_sequence[isim] = neutral_pdf_error_t[end]
        end
    
    end
    
    # set plot labels 
    ylabel_ion_density = L"\varepsilon(\widetilde{n}_i)"#L"\sum || \widetilde{n}_i - \widetilde{n}^{sym}_i ||^2"
    ylabel_ion_pdf =  L"\varepsilon(\widetilde{F}_i)"#L"\sum || \widetilde{f}_i - \widetilde{f}^{sym}_i ||^2"
    ylabel_neutral_density =  L"\varepsilon(\widetilde{n}_n)"#L"\sum || \widetilde{n}_n - \widetilde{n}^{sym}_n ||^2"
    ylabel_neutral_pdf =  L"\varepsilon(\widetilde{F}_n)"#L"\sum || \widetilde{f}_n - \widetilde{f}^{sym}_n ||^2"
    ylabel_phi = L"\varepsilon(\widetilde{\phi})"
    ylabel_Er = L"\varepsilon(\widetilde{E}_r)"
    ylabel_Ez = L"\varepsilon(\widetilde{E}_z)"
	if scan_type == "vpa_nelement"
        xlabel = L"v_{||}"*" "*L"N_{element}"
    elseif scan_type == "vperp_nelement"
        xlabel = L"v_{\perp}"*" "*L"N_{element}"
    elseif scan_type == "vzeta_nelement"
        xlabel = L"v_{\zeta}"*" "*L"N_{element}"
    elseif scan_type == "vr_nelement"
        xlabel = L"v_{r}"*" "*L"N_{element}"
    elseif scan_type == "vz_nelement"
        xlabel = L"v_{z}"*" "*L"N_{element}"
    elseif scan_type == "r_nelement"
        xlabel = L"r"*" "*L"N_{element}"
    elseif scan_type == "z_nelement"
        xlabel = L"z"*" "*L"N_{element}"
    elseif scan_type == "zr_nelement"
        xlabel = L"z "*" & "*L"r "*" "*L"N_{element}"
    elseif scan_type == "vpazr_nelement0.25"
        xlabel = L"N_{element}(z) = N_{element}(r) = N_{element}(v_{||})/4"
    elseif scan_type == "vpaz_nelement0.25"
        xlabel = L"N_{element}(z) = N_{element}(v_{||})/4"
    elseif scan_type == "vpaz_nelement4"
        xlabel = L"N_{element}(z) = 4 N_{element}(v_{||})"
    elseif scan_type == "velocity_nelement"
        xlabel = L"N_{element}"
    elseif scan_type == "nelement" || scan_type == "vpazr_nelement" || scan_type == "vpaz_nelement"
        xlabel = L"N_{element}"
    else 
        println("ERROR: scan_type = ",scan_type," is unsupported")
    end
    
    outprefix = "MMS_test_"*scan_type*"_"*scan_name
    nelmin = nelement_sequence[1]
    nelmax = nelement_sequence[end]
    ymax = 1.0e1
    ymin = 1.0e-7
    fontsize = 10
    if scan_name == "2D-3V-wall_cheb" 
        ytick_sequence = Array([1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    elseif scan_name == "1D-3V-wall_cheb" || scan_name == "1D-3V-wall-sheath_cheb"  || scan_name == "1D-1V-wall_cheb-constant-Er"  || scan_name == "1D-1V-wall_cheb-constant-Er-zngrid-5"  || scan_name == "1D-1V-wall_cheb-constant-Er-ngrid-5" 
        ytick_sequence = Array([1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1])
    elseif scan_name == "2D-sound-wave_cheb_cxiz"
        ytick_sequence = Array([1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    elseif scan_name == "2D-sound-wave_cheb_cxiz" 
        ytick_sequence = Array([1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    elseif scan_name == "1D-1V-wall_cheb"
        ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    elseif scan_name == "1D-3V-wall_cheb-updated" || scan_name == "1D-3V-wall_cheb-new-dfni-Er" || scan_name == "1D-3V-wall_cheb-new-dfni" || scan_name == "2D-sound-wave_cheb"
        ytick_sequence = Array([1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    elseif scan_name == "2D-1V-wall_cheb" || scan_name == "2D-1V-wall_cheb-nonzero-Er" || scan_name == "1D-1V-wall_cheb-constant-Er-ngrid-5-opt" 
        ytick_sequence = Array([1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0])
        #ytick_sequence = Array([1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0])
    else
        ytick_sequence = Array([1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    end
    
    plot(nelement_sequence, [ion_density_error_sequence,ion_pdf_error_sequence], xlabel=xlabel, label=[ylabel_ion_density ylabel_ion_pdf], ylabel="",
     shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
      xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*".pdf"
    savefig(outfile)
    println(outfile)
    
    try
        plot(nelement_sequence, [ion_density_error_sequence,phi_error_sequence,Er_error_sequence,Ez_error_sequence], xlabel=xlabel,
             label=[ylabel_ion_density ylabel_phi ylabel_Er ylabel_Ez], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
             xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
        outfile = outprefix*"_fields.pdf"
        savefig(outfile)
        println(outfile)
    catch LoadError
        # This plot will fail when the Er error is zero, e.g. for a 1d case
        # Delete .pdf file as it will contain junk
        rm(outprefix*"_fields.pdf")
    end
    
	plot(nelement_sequence, [ion_density_error_sequence,phi_error_sequence,Ez_error_sequence], xlabel=xlabel,
	label=[ylabel_ion_density ylabel_phi ylabel_Ez], ylabel="",
     shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
      xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*"_fields_no_Er.pdf"
    savefig(outfile)
    println(outfile)
    
    plot(nelement_sequence, [ion_density_error_sequence,phi_error_sequence,Ez_error_sequence,ion_pdf_error_sequence], xlabel=xlabel,
	label=[ylabel_ion_density ylabel_phi ylabel_Ez ylabel_ion_pdf], ylabel="",
     shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
      xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*"_fields_and_ion_pdf_no_Er.pdf"
    savefig(outfile)
    println(outfile)
    
    try
        plot(nelement_sequence, [ion_density_error_sequence,phi_error_sequence,Ez_error_sequence,Er_error_sequence,ion_pdf_error_sequence], xlabel=xlabel,
             label=[ylabel_ion_density ylabel_phi ylabel_Ez ylabel_Er ylabel_ion_pdf], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
             xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
        outfile = outprefix*"_fields_and_ion_pdf.pdf"
        savefig(outfile)
        println(outfile)
    catch LoadError
        # This plot will fail when the Er error is zero, e.g. for a 1d case
        # Delete .pdf file as it will contain junk
        rm(outprefix*"_fields_and_ion_pdf.pdf")
    end
    
    plot(nelement_sequence, ion_density_error_sequence, xlabel=xlabel, ylabel=ylabel_ion_density, label="",
     shape =:circle, color =:black, yscale=:log10, xticks = (nelmin:nelmax, nelmin:nelmax), markersize = 5, linewidth=2)
    outfile = outprefix*"_ion_density.pdf"
    savefig(outfile)
    println(outfile)
    
    plot(nelement_sequence, ion_pdf_error_sequence, xlabel=xlabel, ylabel=ylabel_ion_pdf, label="", 
     shape =:circle, color =:black, yscale=:log10, xticks = (nelmin:nelmax, nelmin:nelmax), markersize = 5, linewidth=2)
    outfile = outprefix*"_ion_pdf.pdf"
    savefig(outfile)
    println(outfile)

    if n_neutral_species > 0
        plot(nelement_sequence, [ion_density_error_sequence, ion_pdf_error_sequence, neutral_density_error_sequence, neutral_pdf_error_sequence], xlabel=xlabel, 
        label=[ylabel_ion_density ylabel_ion_pdf ylabel_neutral_density ylabel_neutral_pdf], ylabel="",
         shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2,
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
        outfile = outprefix*".pdf"
        savefig(outfile)
        println(outfile)
        
		plot(nelement_sequence, [ion_density_error_sequence, neutral_density_error_sequence, phi_error_sequence, Er_error_sequence, Ez_error_sequence], xlabel=xlabel, 
        label=[ylabel_ion_density ylabel_neutral_density ylabel_phi ylabel_Er ylabel_Ez], ylabel="",
         shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2,
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
        outfile = outprefix*"_fields.pdf"
        savefig(outfile)
        println(outfile)

		plot(nelement_sequence, [ion_density_error_sequence, neutral_density_error_sequence, phi_error_sequence, Ez_error_sequence], xlabel=xlabel, 
        label=[ylabel_ion_density ylabel_neutral_density ylabel_phi ylabel_Ez], ylabel="",
         shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2,
         xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
        outfile = outprefix*"_fields_no_Er.pdf"
        savefig(outfile)
        println(outfile)
        
        plot(nelement_sequence, neutral_density_error_sequence, xlabel=xlabel, ylabel=ylabel_neutral_density, label="",
         shape =:circle, color =:black, yscale=:log10, xticks = (nelmin:nelmax, nelmin:nelmax), markersize = 5, linewidth=2)
        outfile = outprefix*"_neutral_density.pdf"
        savefig(outfile)    
        println(outfile)
        
        plot(nelement_sequence, neutral_pdf_error_sequence, xlabel=xlabel, ylabel=ylabel_neutral_pdf, label="",
         shape =:circle, color =:black, yscale=:log10, xticks = (nelmin:nelmax, nelmin:nelmax), markersize = 5, linewidth=2)
        outfile = outprefix*"_neutral_pdf.pdf"
        savefig(outfile)
        println(outfile)
    end

end

end
