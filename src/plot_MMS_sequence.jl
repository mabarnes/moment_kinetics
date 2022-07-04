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
using ..post_processing:  compare_moments_symbolic_test, compare_neutral_pdf_symbolic_test
using ..array_allocation: allocate_float
using ..file_io: open_output_file
using ..type_definitions: mk_float, mk_int
using ..load_data: open_netcdf_file
using ..load_data: load_coordinate_data, load_fields_data, load_pdf_data
using ..load_data: load_charged_particle_moments_data, load_neutral_particle_moments_data
using ..load_data: load_neutral_pdf_data, load_neutral_coordinate_data
using ..velocity_moments: integrate_over_vspace
using ..manufactured_solns: manufactured_solutions, manufactured_electric_fields
using ..moment_kinetics_input: mk_input

using TOML
import Base: get

# assume in function below that we have a list of simulations 
# where only a single nelement parameter is varied
# we plot the MMS error measurements as a fn of nelement
function get_MMS_error_data(path_list,scan_type,scan_name)
    
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
        scan_input = TOML.parsefile(input_filename)
        # get run-time input/composition/geometry/collisions/species info for convenience
        run_name_internal, output_dir, evolve_moments, 
            t_input, z_input, r_input, 
            vpa_input, vperp_input, gyrophase_input,
            vz_input, vr_input, vzeta_input, 
            composition, species, collisions, geometry, drive_input = mk_input(scan_input)
        
        if scan_type == "vpa_nelement"
            # get the number of elements for plot
            nelement_sequence[isim] = vpa_input.nelement
        elseif scan_type == "velocity_nelement"
            nelement = vpa_input.nelement
            if nelement == vperp_input.nelement && nelement == vz_input.nelement && nelement == vr_input.nelement && nelement == vzeta_input.nelement
                nelement_sequence[isim] = nelement
            else 
                println("ERROR: scan_type = ",scan_type," requires velocity elements equal in all dimensions")
            end
        else 
            println("ERROR: scan_type = ",scan_type," is unsupported")
        end

        # open the netcdf file and give it the handle 'fid'
        fid = open_netcdf_file(run_name)
        # load space-time coordinate data
        nvpa, vpa, vpa_wgts, nvperp, vperp, vperp_wgts, nz, z, z_wgts, Lz, 
         nr, r, r_wgts, Lr, ntime, time, n_ion_species, n_neutral_species = load_coordinate_data(fid)
        #println("\n Info: n_neutral_species = ",n_neutral_species,", n_ion_species = ",n_ion_species,"\n")
        if n_neutral_species > 0
            nvz, vz, vz_wgts, nvr, vr, vr_wgts, nvzeta, vzeta, vzeta_wgts = load_neutral_coordinate_data(fid)
        end
        # load full (z,r,t) fields data
        phi, Er, Ez = load_fields_data(fid)
        # load full (z,r,species,t) charged particle velocity moments data
        density, parallel_flow, parallel_pressure, parallel_heat_flux,
            thermal_speed, evolve_ppar = load_charged_particle_moments_data(fid)
        # load full (vpa,vperp,z,r,species,t) charged particle distribution function (pdf) data
        ff = load_pdf_data(fid)
        # load neutral particle data
        if n_neutral_species > 0
            neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed = load_neutral_particle_moments_data(fid)
            neutral_ff = load_neutral_pdf_data(fid)
        end
        close(fid)
        
        
        r_bc = get(scan_input, "r_bc", "periodic")
        z_bc = get(scan_input, "z_bc", "periodic")
        # avoid passing Lr = 0 into manufactured_solns functions 
        if nr > 1
            Lr_in = Lr 
        else 
            Lr_in = 1.0
        end
        manufactured_solns_list = manufactured_solutions(Lr_in,Lz,r_bc,z_bc,geometry,composition,nr) 
        dfni_func = manufactured_solns_list.dfni_func
        densi_func = manufactured_solns_list.densi_func
        dfnn_func = manufactured_solns_list.dfnn_func
        densn_func = manufactured_solns_list.densn_func
        
        manufactured_E_fields = manufactured_electric_fields(Lr_in,Lz,r_bc,z_bc,composition,nr)
        Er_func = manufactured_E_fields.Er_func
        Ez_func = manufactured_E_fields.Ez_func
        phi_func = manufactured_E_fields.phi_func
        
        # phi, Er, Ez test
        phi_sym = copy(phi[:,:,:])
        Er_sym = copy(phi[:,:,:])
        Ez_sym = copy(phi[:,:,:])
        for it in 1:ntime
            for ir in 1:nr
                for iz in 1:nz
                    phi_sym[iz,ir,it] = phi_func(z[iz],r[ir],time[it])
                    Ez_sym[iz,ir,it] = Ez_func(z[iz],r[ir],time[it])
                    Er_sym[iz,ir,it] = Er_func(z[iz],r[ir],time[it])
                end
            end
        end
        phi_error_t = compare_fields_symbolic_test(run_name,phi,phi_sym,z,r,time,nz,nr,ntime,
         L"\widetilde{\phi}",L"\widetilde{\phi}^{sym}",L"\sqrt{\sum || \widetilde{\phi} - \widetilde{\phi}^{sym} ||^2 / N} ","phi")
        phi_error_sequence[isim] = phi_error_t[end]
        Er_error_t = compare_fields_symbolic_test(run_name,Er,Er_sym,z,r,time,nz,nr,ntime,
         L"\widetilde{E_r}",L"\widetilde{E_r}^{sym}",L"\sqrt{\sum || \widetilde{E_r} - \widetilde{E_r}^{sym} ||^2 /N} ","Er")
        Er_error_sequence[isim] = Er_error_t[end]
        Ez_error_t = compare_fields_symbolic_test(run_name,Ez,Ez_sym,z,r,time,nz,nr,ntime,
         L"\widetilde{E_z}",L"\widetilde{E_z}^{sym}",L"\sqrt{\sum || \widetilde{E_z} - \widetilde{E_z}^{sym} ||^2 /N} ","Ez")
        Ez_error_sequence[isim] = Ez_error_t[end]
        
        # ion test
        density_sym = copy(density[:,:,:,:])
        is = 1
        for it in 1:ntime
            for ir in 1:nr
                for iz in 1:nz
                    density_sym[iz,ir,is,it] = densi_func(z[iz],r[ir],time[it])
                end
            end
        end
        ion_density_error_t = compare_moments_symbolic_test(run_name,density,density_sym,"ion",z,r,time,nz,nr,ntime,
         L"\widetilde{n}_i",L"\widetilde{n}_i^{sym}",L"\sum || \widetilde{n}_i - \widetilde{n}_i^{sym} ||^2 ","dens")
        # use final time point for analysis
        ion_density_error_sequence[isim] = ion_density_error_t[end] 
        
        ff_sym = copy(ff)
        is = 1
        for it in 1:ntime
            for ir in 1:nr
                for iz in 1:nz
                    for ivperp in 1:nvperp
                        for ivpa in 1:nvpa
                            ff_sym[ivpa,ivperp,iz,ir,is,it] = dfni_func(vpa[ivpa],vperp[ivperp],z[iz],r[ir],time[it])
                        end
                    end
                end
            end
        end
        ion_pdf_error_t = compare_charged_pdf_symbolic_test(run_name,ff,ff_sym,"ion",vpa,vperp,z,r,time,nvpa,nvperp,nz,nr,ntime,
         L"\widetilde{f}_i",L"\widetilde{f}^{sym}_i",L"\sum || \widetilde{f}_i - \widetilde{f}_i^{sym} ||^2","pdf")
        ion_pdf_error_sequence[isim] = ion_pdf_error_t[end]
        
        if n_neutral_species > 0
            # neutral test
            neutral_density_sym = copy(density[:,:,:,:])
            is = 1
            for it in 1:ntime
                for ir in 1:nr
                    for iz in 1:nz
                        neutral_density_sym[iz,ir,is,it] = densn_func(z[iz],r[ir],time[it])
                    end
                end
            end
            neutral_density_error_t = compare_moments_symbolic_test(run_name,neutral_density,neutral_density_sym,"neutral",z,r,time,nz,nr,ntime,
             L"\widetilde{n}_n",L"\widetilde{n}_n^{sym}",L"\sum || \widetilde{n}_n - \widetilde{n}_n^{sym} ||^2 ","dens")
            neutral_density_error_sequence[isim] = neutral_density_error_t[end]
        
            neutral_ff_sym = copy(neutral_ff)
            is = 1
            for it in 1:ntime
                for ir in 1:nr
                    for iz in 1:nz
                        for ivzeta in 1:nvzeta
                            for ivr in 1:nvr
                                for ivz in 1:nvz
                                    neutral_ff_sym[ivz,ivr,ivzeta,iz,ir,is,it] = dfnn_func(vz[ivz],vr[ivr],vzeta[ivzeta],z[iz],r[ir],time[it])
                                end
                            end
                        end
                    end
                end
            end
            neutral_pdf_error_t = compare_neutral_pdf_symbolic_test(run_name,neutral_ff,neutral_ff_sym,"neutral",vz,vr,vzeta,z,r,time,nvz,nvr,nvzeta,nz,nr,ntime,
             L"\widetilde{f}_n",L"\widetilde{f}^{sym}_n",L"\sum || \widetilde{f}_n - \widetilde{f}_n^{sym} ||^2","pdf")
            neutral_pdf_error_sequence[isim] = neutral_pdf_error_t[end]
        end
    
    end
    
    # set plot labels 
    ylabel_ion_density = L"\varepsilon(\widetilde{n}_i)"#L"\sum || \widetilde{n}_i - \widetilde{n}^{sym}_i ||^2"
    ylabel_ion_pdf =  L"\varepsilon(\widetilde{F}_i)"#L"\sum || \widetilde{f}_i - \widetilde{f}^{sym}_i ||^2"
    ylabel_neutral_density =  L"\varepsilon(\widetilde{n}_n)"#L"\sum || \widetilde{n}_n - \widetilde{n}^{sym}_n ||^2"
    ylabel_neutral_pdf =  L"\varepsilon(\widetilde{F}_n)"#L"\sum || \widetilde{f}_n - \widetilde{f}^{sym}_n ||^2"
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
    elseif scan_type == "velocity_nelement"
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
    if scan_name == "2D-3V-wall_cheb" || scan_name == "1D-3V-wall_cheb" || scan_name == "1D-3V-wall-sheath_cheb"
        ytick_sequence = Array([1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    elseif scan_name == "2D-sound-wave_cheb_cxiz" || scan_name == "2D-sound-wave_cheb"
        ytick_sequence = Array([1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    else
        ytick_sequence = Array([1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
    end
    
    plot(nelement_sequence, [ion_density_error_sequence,ion_pdf_error_sequence], xlabel=xlabel, label=[ylabel_ion_density ylabel_ion_pdf], ylabel="",
     shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_sequence, nelement_sequence), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
      xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize)
    outfile = outprefix*".pdf"
    savefig(outfile)
    println(outfile)
    
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
