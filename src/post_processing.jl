"""
"""
module post_processing

export analyze_and_plot_data
export compare_charged_pdf_symbolic_test
export compare_moments_symbolic_test
export compare_neutral_pdf_symbolic_test
export compare_fields_symbolic_test
export read_distributed_zr_data!
export construct_global_zr_grids
export allocate_global_zr_charged_moments
export allocate_global_zr_neutral_moments
export allocate_global_zr_fields

# packages
using Plots
using IJulia
using LsqFit
using NCDatasets
using Statistics: mean
using SpecialFunctions: erfi
using LaTeXStrings
using Measures
# modules
using ..post_processing_input: pp
using ..quadrature: composite_simpson_weights
using ..array_allocation: allocate_float
using ..file_io: open_ascii_output_file
using ..type_definitions: mk_float, mk_int
using ..load_data: open_readonly_output_file, get_group, load_time_data
using ..load_data: get_nranks
using ..load_data: load_fields_data, load_pdf_data
using ..load_data: load_charged_particle_moments_data, load_neutral_particle_moments_data
using ..load_data: load_neutral_pdf_data
using ..load_data: load_variable
using ..load_data: load_coordinate_data, load_block_data, load_rank_data, load_species_data
using ..analysis: analyze_fields_data, analyze_moments_data, analyze_pdf_data,
                  check_Chodura_condition
using ..velocity_moments: integrate_over_vspace
using ..manufactured_solns: manufactured_solutions, manufactured_electric_fields
using ..moment_kinetics_input: mk_input, get
using ..input_structs: geometry_input, species_composition
using ..input_structs: electron_physics_type, boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath
using TOML
import Base: get

"""
Calculate a moving average

```
result[i] = mean(v[i-n:i+n])
```
Except near the ends of the array where indices outside the range of v are skipped.
"""
function moving_average(v::AbstractVector, n::mk_int)
    if length(v) < 2*n+1
        error("Cannot take moving average with n=$n on vector of length=$(length(v))")
    end
    result = similar(v)
    for i ∈ 1:n
        result[i] = mean(v[begin:i+n])
    end
    for i ∈ n+1:length(v)-n-1
        result[i] = mean(v[i-n:i+n])
    end
    for i ∈ length(v)-n:length(v)
        result[i] = mean(v[i-n:end])
    end
    return result
end

"""
"""
function read_distributed_zr_data!(var::Array{mk_float,3}, var_name::String,
   run_name::String, file_key::String, nblocks::mk_int, nz_local::mk_int,nr_local::mk_int)
    # dimension of var is [z,r,t]
    
    for iblock in 0:nblocks-1
        fid = open_readonly_output_file(run_name,file_key,iblock=iblock,printout=false)
        group = get_group(fid, "dynamic_data")
        var_local = load_variable(group, var_name)
        
        z_irank, r_irank = load_rank_data(fid)
        
        # min index set to avoid double assignment of repeated points 
        # 1 if irank = 0, 2 otherwise
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        for ir_local in imin_r:nr_local
            for iz_local in imin_z:nz_local
                ir_global = iglobal_func(ir_local,r_irank,nr_local)
                iz_global = iglobal_func(iz_local,z_irank,nz_local)
                var[iz_global,ir_global,:] .= var_local[iz_local,ir_local,:]
            end
        end
        close(fid)
    end
end

function read_distributed_zr_data!(var::Array{mk_float,4}, var_name::String,
   run_name::String, file_key::String, nblocks::mk_int, nz_local::mk_int,nr_local::mk_int)
    # dimension of var is [z,r,species,t]
    for iblock in 0:nblocks-1
        fid = open_readonly_output_file(run_name,file_key,iblock=iblock,printout=false)
        group = get_group(fid, "dynamic_data")
        var_local = load_variable(group, var_name)
        
        z_irank, r_irank = load_rank_data(fid)
        
        
        # min index set to avoid double assignment of repeated points 
        # 1 if irank = 0, 2 otherwise
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        for ir_local in imin_r:nr_local
            for iz_local in imin_z:nz_local
                ir_global = iglobal_func(ir_local,r_irank,nr_local)
                iz_global = iglobal_func(iz_local,z_irank,nz_local)
                var[iz_global,ir_global,:,:] .= var_local[iz_local,ir_local,:,:]
            end
        end
        close(fid)
    end
end

function iglobal_func(ilocal,irank,nlocal)
    if irank == 0
        iglobal = ilocal
    elseif irank > 0 && ilocal > 1 
        iglobal = ilocal + irank*(nlocal - 1)
    else
        println("ERROR: Invalid call to iglobal_func") 
    end
    return iglobal
end

function construct_global_zr_grids(run_name::String,file_key::String,nz_global::mk_int,nr_global::mk_int,nblocks::mk_int)
    zgrid = allocate_float(nz_global)
    zwgts = allocate_float(nz_global)
    rgrid = allocate_float(nr_global)
    rwgts = allocate_float(nr_global)

    # current routine loops over blocks 
    # whereas the optimal routine would loop over a single z/r group
    for iblock in 0:nblocks-1
        fid = open_readonly_output_file(run_name,file_key,iblock=iblock,printout=false)
        nz_local, nz_global, z_local, z_local_wgts, Lz = load_coordinate_data(fid, "z")
        nr_local, nr_global, r_local, r_local_wgts, Lr = load_coordinate_data(fid, "r")
    
        z_irank, r_irank = load_rank_data(fid)
        # MRH should wgts at element boundaries be doubled 
        # MRH in the main code duplicated points have half the integration wgt
        if z_irank == 0
            zgrid[1] = z_local[1]
            zwgts[1] = z_local_wgts[1]
        end
        for iz_local in 2:nz_local
            iz_global = iglobal_func(iz_local,z_irank,nz_local)
            zgrid[iz_global] = z_local[iz_local]  
            zwgts[iz_global] = z_local_wgts[iz_local]  
        end

        if r_irank == 0
            rgrid[1] = r_local[1]
            rwgts[1] = r_local_wgts[1]
        end
        for ir_local in 2:nr_local
            ir_global = iglobal_func(ir_local,r_irank,nr_local)
            rgrid[ir_global] = r_local[ir_local]  
            rwgts[ir_global] = r_local_wgts[ir_local]  
        end
    end 
    return zgrid, zwgts, rgrid, rwgts
end 


"""
functions to allocate arrays that are used at run-time to postprocess
data that is stored in the netcdf files 
"""
function allocate_global_zr_fields(nz_global,nr_global,ntime)
    Er = allocate_float(nz_global,nr_global,ntime)
    Ez = allocate_float(nz_global,nr_global,ntime)
    phi = allocate_float(nz_global,nr_global,ntime)
    return phi, Ez, Er 
end

function allocate_global_zr_charged_moments(nz_global,nr_global,n_ion_species,ntime)
    density = allocate_float(nz_global,nr_global,n_ion_species,ntime)
    parallel_flow = allocate_float(nz_global,nr_global,n_ion_species,ntime)
    parallel_pressure = allocate_float(nz_global,nr_global,n_ion_species,ntime)
    parallel_heat_flux = allocate_float(nz_global,nr_global,n_ion_species,ntime)
    thermal_speed = allocate_float(nz_global,nr_global,n_ion_species,ntime)
    return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed
end

function allocate_global_zr_neutral_moments(nz_global,nr_global,n_neutral_species,ntime)
    neutral_density = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_uz = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_pz = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_qz = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_thermal_speed = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    return neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed
end


"""
Function below duplicates code from moment_kinetics_input.jl
 -- need to make better functions there that can be called here
"""

function get_geometry_and_composition(scan_input,n_ion_species,n_neutral_species)
    # set geometry_input
    # MRH need to get this in way that does not duplicate code
    # MRH from moment_kinetics_input.jl
    Bzed = get(scan_input, "Bzed", 1.0)
    Bmag = get(scan_input, "Bmag", 1.0)
    bzed = Bzed/Bmag
    bzeta = sqrt(1.0 - bzed^2.0)
    Bzeta = Bmag*bzeta
    rhostar = get(scan_input, "rhostar", 0.0)
    geometry = geometry_input(Bzed,Bmag,bzed,bzeta,Bzeta,rhostar)

    # set composition input
    # MRH need to get this in way that does not duplicate code
    # MRH from moment_kinetics_input.jl
    electron_physics = get(scan_input, "electron_physics", boltzmann_electron_response)

    if electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
        n_species = n_ion_species + n_neutral_species
    else
        n_species = n_ion_species + n_neutral_species + 1
    end
    T_e = get(scan_input, "T_e", 1.0)
    # set wall temperature T_wall = Tw/Te
    T_wall = get(scan_input, "T_wall", 1.0)
    # set initial neutral temperature Tn/Tₑ = 1
    # set initial nᵢ/Nₑ = 1.0
    # set phi_wall at z = 0
    phi_wall = get(scan_input, "phi_wall", 0.0)
    # if false use true Knudsen cosine for neutral wall bc
    use_test_neutral_wall_pdf = get(scan_input, "use_test_neutral_wall_pdf", false)
    # constant to be used to test nonzero Er in wall boundary condition
    Er_constant = get(scan_input, "Er_constant", 0.0)
    # constant to be used to control Ez divergences
    epsilon_offset = get(scan_input, "epsilon_offset", 0.001)
    # ratio of the neutral particle mass to the ion particle mass
    mn_over_mi = 1.0
    # ratio of the electron particle mass to the ion particle mass
    me_over_mi = 1.0/1836.0
    composition = species_composition(n_species, n_ion_species, n_neutral_species,
        electron_physics, use_test_neutral_wall_pdf, 1:n_ion_species, n_ion_species+1:n_species, T_e, T_wall,
        phi_wall, Er_constant, epsilon_offset, mn_over_mi, me_over_mi, allocate_float(n_species))
    return geometry, composition

end
"""
"""
function analyze_and_plot_data(path)
    # Create run_name from the path to the run directory
    path = realpath(path)
    run_name = joinpath(path, basename(path))
    input_filename = path * ".toml"
    scan_input = TOML.parsefile(input_filename)
    # get run-time input/composition/geometry/collisions/species info for convenience
    # run_name_internal, output_dir, evolve_moments,
    #    t_input, z_input, r_input,
    #    vpa_input, vperp_input, gyrophase_input,
    #    vz_input, vr_input, vzeta_input,
    #    composition, species, collisions, geometry, drive_input = mk_input(scan_input)

    # open the output file and give it the handle 'fid'
    fid = open_readonly_output_file(run_name,"moments")
    # load block data on iblock=0
    nblocks, iblock = load_block_data(fid)
         
    # load global and local sizes of grids stored on each output file 
    # z z_wgts r r_wgts may take different values on different blocks
    # we need to construct the global grid below
    nz_local, nz_global, z_local, z_local_wgts, Lz = load_coordinate_data(fid, "z")
    nr_local, nr_global, r_local, r_local_wgts, Lr = load_coordinate_data(fid, "r")
    # load time data 
    ntime, time = load_time_data(fid)
    # load species data 
    n_ion_species, n_neutral_species = load_species_data(fid)

    close(fid)
    
    
    # allocate arrays to contain the global fields as a function of (z,r,t)
    phi, Ez, Er = allocate_global_zr_fields(nz_global,nr_global,ntime)
    density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed =
        allocate_global_zr_charged_moments(nz_global,nr_global,n_ion_species,ntime)
    if n_neutral_species > 0
        neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed =
            allocate_global_zr_neutral_moments(nz_global,nr_global,n_neutral_species,ntime)
    end 
    # read in the data from different block netcdf files
    # grids 
    z, z_wgts, r, r_wgts = construct_global_zr_grids(run_name,
       "moments",nz_global,nr_global,nblocks)
    #println("z: ",z)
    #println("r: ",r)
    
    # fields 
    read_distributed_zr_data!(phi,"phi",run_name,"moments",nblocks,nz_local,nr_local)
    read_distributed_zr_data!(Ez,"Ez",run_name,"moments",nblocks,nz_local,nr_local)
    read_distributed_zr_data!(Er,"Er",run_name,"moments",nblocks,nz_local,nr_local)
    # charged particle moments
    read_distributed_zr_data!(density,"density",run_name,"moments",nblocks,nz_local,nr_local)
    read_distributed_zr_data!(parallel_flow,"parallel_flow",run_name,"moments",nblocks,nz_local,nr_local)
    read_distributed_zr_data!(parallel_pressure,"parallel_pressure",run_name,"moments",nblocks,nz_local,nr_local)
    read_distributed_zr_data!(parallel_heat_flux,"parallel_heat_flux",run_name,"moments",nblocks,nz_local,nr_local)
    read_distributed_zr_data!(thermal_speed,"thermal_speed",run_name,"moments",nblocks,nz_local,nr_local)
    # neutral particle moments 
    if n_neutral_species > 0
        read_distributed_zr_data!(neutral_density,"density_neutral",run_name,"moments",nblocks,nz_local,nr_local)
        read_distributed_zr_data!(neutral_uz,"uz_neutral",run_name,"moments",nblocks,nz_local,nr_local)
        read_distributed_zr_data!(neutral_pz,"pz_neutral",run_name,"moments",nblocks,nz_local,nr_local)
        read_distributed_zr_data!(neutral_qz,"qz_neutral",run_name,"moments",nblocks,nz_local,nr_local)
        read_distributed_zr_data!(neutral_thermal_speed,"thermal_speed_neutral",run_name,"moments",nblocks,nz_local,nr_local)
    end 
    # load time data from `dfns' cdf
    fid_pdfs = open_readonly_output_file(run_name,"dfns")
    # note that ntime may differ in these output files
     
    ntime_pdfs, time_pdfs = load_time_data(fid_pdfs)
    
    # load local velocity coordinate data from `dfns' cdf
    # these values are currently the same for all blocks 
    nvpa, nvpa_global, vpa_local, vpa_local_wgts, Lvpa = load_coordinate_data(fid_pdfs, "vpa")
    nvperp, nvperp_global, vperp_local, vperp_local_wgts, Lvperp = load_coordinate_data(fid_pdfs, "vperp")
    if n_neutral_species > 0
        nvzeta, nvzeta_global, vzeta_local, vzeta_local_wgts, Lvzeta = load_coordinate_data(fid_pdfs, "vzeta")
        nvr, nvr_global, vr_local, vr_local_wgts, Lvr = load_coordinate_data(fid_pdfs, "vr")
        nvz, nvz_global, vz_local, vz_local_wgts, Lvz = load_coordinate_data(fid_pdfs, "vz")
    else # define nvz nvr nvzeta to avoid errors below
        nvz = 1
        nvr = 1
        nvzeta = 1
    end

    geometry, composition = get_geometry_and_composition(scan_input,n_ion_species,n_neutral_species)
	
	# initialise the post-processing input options
    nwrite_movie, itime_min, itime_max, ivpa0, ivperp0, iz0, ir0,
        ivz0, ivr0, ivzeta0 = init_postprocessing_options(pp, nvpa, nvperp, nz_global, nr_global, nvz, nvr, nvzeta, ntime)
    # load full (z,r,t) fields data
    #phi, Er, Ez = load_fields_data(fid)
    # load full (z,r,species,t) charged particle velocity moments data
    #density, parallel_flow, parallel_pressure, parallel_heat_flux,
    #    thermal_speed, evolve_ppar = load_charged_particle_moments_data(fid)
    
    # load full (vpa,vperp,z,r,species,t) charged particle distribution function (pdf) data
    ff = load_pdf_data(fid_pdfs)
    # load neutral particle data
    #if n_neutral_species > 0
    #    neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed = load_neutral_particle_moments_data(fid)
    #    neutral_ff = load_neutral_pdf_data(fid_pdfs)
    #end

    #evaluate 1D-1V diagnostics at fixed ir0
    diagnostics_1d = false
    if diagnostics_1d
        plot_1D_1V_diagnostics(run_name, fid, nwrite_movie, itime_min, itime_max, ivpa0, iz0, ir0, r,
            phi[:,ir0,:],
            density[:,ir0,:,:],
            parallel_flow[:,ir0,:,:],
            parallel_pressure[:,ir0,:,:],
            parallel_heat_flux[:,ir0,:,:],
            thermal_speed[:,ir0,:,:],
            ff[:,ivperp0,:,ir0,:,:],
            n_ion_species, evolve_ppar, nvpa, vpa, vpa_wgts,
            nz, z, z_wgts, Lz, ntime, time)
    end
    close(fid_pdfs)

    diagnostics_2d = false
    if diagnostics_2d
        # analyze the fields data
        phi_fldline_avg, delta_phi = analyze_fields_data(phi[iz0,:,:], ntime, nr, r_wgts, Lr)
        plot_fields_rt(phi[iz0,:,:], delta_phi, time, itime_min, itime_max, nwrite_movie,
        r, ir0, run_name, delta_phi, pp)
    end

    Chodura_ratio_lower, Chodura_ratio_upper =
        check_Chodura_condition(run_name, vpa_local, vpa_local_wgts, vperp_local,
                                vperp_local_wgts,
                                moment_at_pdf_times(density, ntime, ntime_pdfs),
                                composition.T_e,
                                moment_at_pdf_times(Er, ntime, ntime_pdfs),
                                geometry, "wall", nblocks)

    plot(legend=true)
    for ir ∈ 1:nr_global
        plot!(time_pdfs, Chodura_ratio_lower[ir,:], xlabel="time",
              ylabel="Chodura ratio at z=-L/2", label="ir=$ir")
    end
    outfile = string(run_name, "_Chodura_ratio_lower.pdf")
    savefig(outfile)
    plot(legend=true)
    for ir ∈ 1:nr_global
        plot(time_pdfs, Chodura_ratio_upper[ir,:], xlabel="time",
             ylabel="Chodura ratio at z=+L/2", label="ir=$ir")
    end
    outfile = string(run_name, "_Chodura_ratio_upper.pdf")
    savefig(outfile)

    # make plots and animations of the phi, Ez and Er 
    plot_charged_moments_2D(density, parallel_flow, parallel_pressure, time, z, r, iz0, ir0, n_ion_species,
     itime_min, itime_max, nwrite_movie, run_name, pp)
    # make plots and animations of the phi, Ez and Er 
    plot_fields_2D(phi, Ez, Er, time, z, r, iz0, ir0,
     itime_min, itime_max, nwrite_movie, run_name, pp, "")
    # make plots and animations of the ion pdf
    # only if ntime == ntime_pdfs
    if ntime == ntime_pdfs && false 
        spec_type = "ion"
        plot_charged_pdf(ff, vpa, vperp, z, r, ivpa0, ivperp0, iz0, ir0,
            spec_type, n_ion_species,
            itime_min, itime_max, nwrite_movie, run_name, pp)
        # make plots and animations of the neutral pdf
        if n_neutral_species > 0
            spec_type = "neutral"
            plot_neutral_pdf(neutral_ff, vz, vr, vzeta, z, r,
                ivz0, ivr0, ivzeta0, iz0, ir0,
                spec_type, n_neutral_species,
                itime_min, itime_max, nwrite_movie, run_name, pp)
        end 
    end
    # plot ion pdf data near the wall boundary
    if pp.plot_wall_pdf
        plot_charged_pdf_2D_at_wall(run_name)
    end
    # MRH need to get some run-time data here without copy-paste from mk_input
    
    manufactured_solns_test = use_manufactured_solns_for_advance = get(scan_input, "use_manufactured_solns_for_advance", false)
    # Plots compare density and density_symbolic at last timestep
    #if(manufactured_solns_test && nr > 1)
    if(manufactured_solns_test)
        r_bc = get(scan_input, "r_bc", "periodic")
        z_bc = get(scan_input, "z_bc", "periodic")
        # avoid passing Lr = 0 into manufactured_solns functions
        if nr_local > 1
            Lr_in = Lr
        else
            Lr_in = 1.0
        end
        
        #geometry, composition = get_geometry_and_composition(scan_input,n_ion_species,n_neutral_species)

        manufactured_solns_list = manufactured_solutions(Lr_in,Lz,r_bc,z_bc,geometry,composition,nr_local)
        dfni_func = manufactured_solns_list.dfni_func
        densi_func = manufactured_solns_list.densi_func
        dfnn_func = manufactured_solns_list.dfnn_func
        densn_func = manufactured_solns_list.densn_func
        manufactured_E_fields = manufactured_electric_fields(Lr_in,Lz,r_bc,z_bc,composition,nr_local)
        Er_func = manufactured_E_fields.Er_func
        Ez_func = manufactured_E_fields.Ez_func
        phi_func = manufactured_E_fields.phi_func

        # phi, Er, Ez test
        phi_sym = copy(phi[:,:,:])
        Er_sym = copy(phi[:,:,:])
        Ez_sym = copy(phi[:,:,:])
        for it in 1:ntime
            for ir in 1:nr_global
                for iz in 1:nz_global
                    phi_sym[iz,ir,it] = phi_func(z[iz],r[ir],time[it])
                    Ez_sym[iz,ir,it] = Ez_func(z[iz],r[ir],time[it])
                    Er_sym[iz,ir,it] = Er_func(z[iz],r[ir],time[it])
                end
            end
        end
        # make plots and animations of the phi, Ez and Er 
        #plot_fields_2D(phi_sym, Ez_sym, Er_sym, time, z, r, iz0, ir0,
        #    itime_min, itime_max, nwrite_movie, run_name, pp, "_sym")
    
        compare_fields_symbolic_test(run_name,phi,phi_sym,z,r,time,nz_global,nr_global,ntime,
         L"\widetilde{\phi}",L"\widetilde{\phi}^{sym}",L"\sqrt{\sum || \widetilde{\phi} - \widetilde{\phi}^{sym} ||^2 / N} ","phi")
        compare_fields_symbolic_test(run_name,Er,Er_sym,z,r,time,nz_global,nr_global,ntime,
         L"\widetilde{E_r}",L"\widetilde{E_r}^{sym}",L"\sqrt{\sum || \widetilde{E_r} - \widetilde{E_r}^{sym} ||^2 /N} ","Er")
        compare_fields_symbolic_test(run_name,Ez,Ez_sym,z,r,time,nz_global,nr_global,ntime,
         L"\widetilde{E_z}",L"\widetilde{E_z}^{sym}",L"\sqrt{\sum || \widetilde{E_z} - \widetilde{E_z}^{sym} ||^2 /N} ","Ez")

        # ion test
        density_sym = copy(density[:,:,:,:])
        is = 1
        for it in 1:ntime
            for ir in 1:nr_global
                for iz in 1:nz_global
                    density_sym[iz,ir,is,it] = densi_func(z[iz],r[ir],time[it])
                end
            end
        end
        compare_moments_symbolic_test(run_name,density,density_sym,"ion",z,r,time,nz_global,nr_global,ntime,
         L"\widetilde{n}_i",L"\widetilde{n}_i^{sym}",L"\sqrt{\sum || \widetilde{n}_i - \widetilde{n}_i^{sym} ||^2 / N }","dens")

        compare_charged_pdf_symbolic_test(run_name,manufactured_solns_list,"ion",
          L"\widetilde{f}_i",L"\widetilde{f}^{sym}_i",L"\sqrt{ \sum || \widetilde{f}_i - \widetilde{f}_i^{sym} ||^2 / N}","pdf")
        if n_neutral_species > 0
            # neutral test
            neutral_density_sym = copy(density[:,:,:,:])
            is = 1
            for it in 1:ntime
                for ir in 1:nr_global
                    for iz in 1:nz_global
                        neutral_density_sym[iz,ir,is,it] = densn_func(z[iz],r[ir],time[it])
                    end
                end
            end
            compare_moments_symbolic_test(run_name,neutral_density,neutral_density_sym,"neutral",z,r,time,nz_global,nr_global,ntime,
             L"\widetilde{n}_n",L"\widetilde{n}_n^{sym}",L"\sqrt{ \sum || \widetilde{n}_n - \widetilde{n}_n^{sym} ||^2 /N}","dens")
            
            compare_neutral_pdf_symbolic_test(run_name,manufactured_solns_list,"neutral",
             L"\widetilde{f}_n",L"\widetilde{f}^{sym}_n",L"\sqrt{\sum || \widetilde{f}_n - \widetilde{f}_n^{sym} ||^2 /N}","pdf")
        end
    end


end

"""
"""

function compare_fields_symbolic_test(run_name,field,field_sym,z,r,time,nz,nr,ntime,field_label,field_sym_label,norm_label,file_string)
	
	# plot last timestep field vs z at r0 
	if nr > 1 
		ir0 = div(nr,2)
	else
		ir0 = 1
	end
	fieldmin = minimum(field[:,ir0,end])
    fieldmax = maximum(field[:,ir0,end])
	@views plot(z, [field[:,ir0,end], field_sym[:,ir0,end] ], xlabel=L"z/L_z", ylabel=field_label, label=["num" "sym"], ylims = (fieldmin,fieldmax))
    outfile = string(run_name, "_"*file_string*"(r0,z)_vs_z.pdf")
    savefig(outfile)    
    
	if nr > 1
		# plot last timestep field vs r at z_wall 
		fieldmin = minimum(field[end,:,end])
		fieldmax = maximum(field[end,:,end])
		@views plot(r, [field[end,:,end], field_sym[end,:,end]], xlabel=L"r/L_r", ylabel=field_label, label=["num" "sym"], ylims = (fieldmin,fieldmax))
		outfile = string(run_name, "_"*file_string*"(r,z_wall)_vs_r.pdf")
		savefig(outfile)

        it = ntime
        fontsize = 20
        ticksfontsize = 10
        heatmap(r, z, field[:,:,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=field_label, c = :deep,
         #xtickfontsize = ticksfontsize, xguidefontsize = fontsize, ytickfontsize = ticksfontsize, yguidefontsize = fontsize, titlefontsize = fontsize)
         windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_vs_r_z.pdf")
        savefig(outfile)

        heatmap(r, z, field_sym[:,:,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=field_sym_label, c = :deep,
        #xtickfontsize = ticksfontsize, xguidefontsize = fontsize, ytickfontsize = ticksfontsize, yguidefontsize = fontsize, titlefontsize = fontsize)
        windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_sym_vs_r_z.pdf")
        savefig(outfile)
    end	
    
    field_norm = zeros(mk_float,ntime)
    for it in 1:ntime
        dummy = 0.0
        dummy_N = 0.0
        for ir in 1:nr
            for iz in 1:nz
                dummy += (field[iz,ir,it] - field_sym[iz,ir,it])^2
                dummy_N +=  (field_sym[iz,ir,it])^2
            end
        end
        #field_norm[it] = dummy/dummy_N
        field_norm[it] = sqrt(dummy/(nr*nz))
    end
    println("test: ",file_string,": ",field_norm)
    @views plot(time, field_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=norm_label) #, yaxis=:log)
    outfile = string(run_name, "_"*file_string*"_norm_vs_t.pdf")
    savefig(outfile)

    return field_norm

end

function compare_moments_symbolic_test(run_name,moment,moment_sym,spec_string,z,r,time,nz,nr,ntime,moment_label,moment_sym_label,norm_label,file_string)
    
    is = 1
    # plot last timestep moment vs z at r0 
	if nr > 1 
		ir0 = div(nr,2)
	else
		ir0 = 1
	end
	momentmin = minimum(moment[:,ir0,is,end])
    momentmax = maximum(moment[:,ir0,is,end])
	@views plot(z, [moment[:,ir0,is,end], moment_sym[:,ir0,is,end] ], xlabel=L"z/L_z", ylabel=moment_label, label=["num" "sym"], ylims = (momentmin,momentmax))
    outfile = string(run_name, "_"*file_string*"(r0,z)_vs_z_", spec_string, ".pdf")
    savefig(outfile)
    
    
    if nr > 1
        it = ntime
        fontsize = 20
        heatmap(r, z, moment[:,:,is,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=moment_label, c = :deep,
        #xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, titlefontsize = fontsize
        windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_vs_r_z_", spec_string, ".pdf")
        savefig(outfile)

        heatmap(r, z, moment_sym[:,:,is,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=moment_sym_label, c = :deep,
        #xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, titlefontsize = fontsize
        windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_sym_vs_r_z_", spec_string, ".pdf")
        savefig(outfile)
    end 
    
    moment_norm = zeros(mk_float,ntime)
    for it in 1:ntime
        dummy = 0.0
        dummy_N = 0.0
        for ir in 1:nr
            for iz in 1:nz
                dummy += (moment[iz,ir,is,it] - moment_sym[iz,ir,is,it])^2
                dummy_N +=  (moment_sym[iz,ir,is,it])^2
            end
        end
        #moment_norm[it] = dummy/dummy_N
        moment_norm[it] = sqrt(dummy/(nr*nz))
    end
    println("test: ",file_string,": ",spec_string," ",moment_norm)
    @views plot(time, moment_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=norm_label) #, yaxis=:log)
    outfile = string(run_name, "_"*file_string*"_norm_vs_t_", spec_string, ".pdf")
    savefig(outfile)

    return moment_norm

end

function compare_charged_pdf_symbolic_test(run_name,manufactured_solns_list,spec_string,
    pdf_label,pdf_sym_label,norm_label,file_string)
    fid = open_readonly_output_file(run_name,"dfns", printout=false)
    # load block data on iblock=0
    nblocks, iblock = load_block_data(fid, printout=false)
    # load global sizes of grids that are distributed in memory
    nz_local, nz_global, z_local, z_wgts_local, Lz = load_coordinate_data(fid, "z", printout=false)
    nr_local, nr_global, r_local, r_wgts_local, Lr = load_coordinate_data(fid, "r", printout=false)
    # velocity grid data on iblock=0 (same for all blocks)
    nvpa, _, vpa, vpa_wgts, Lvpa = load_coordinate_data(fid, "vpa", printout=false)
    nvperp, _, vperp, vperp_wgts, Lvperp = load_coordinate_data(fid, "vperp", printout=false)
    # load time data (unique to pdf, may differ to moment values depending on user nwrite_dfns value)
    ntime, time = load_time_data(fid, printout=false)
    close(fid)
    # get the charged particle pdf
    dfni_func = manufactured_solns_list.dfni_func
    is = 1 # only one species supported currently
    
    pdf_norm = zeros(mk_float,ntime)
    for iblock in 0:nblocks-1
        fid_pdfs = open_readonly_output_file(run_name,"dfns",iblock=iblock, printout=false)
        z_irank, r_irank = load_rank_data(fid_pdfs,printout=false)
        pdf = load_pdf_data(fid_pdfs, printout=false)
        # local local grid data on iblock=0
        nz_local, _, z_local, z_wgts_local, Lz = load_coordinate_data(fid_pdfs, "z")
        nr_local, _, r_local, r_wgts_local, Lr = load_coordinate_data(fid_pdfs, "r")
        close(fid_pdfs)
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        for it in 1:ntime, ir in imin_r:nr_local, iz in imin_z:nz_local,
                ivperp in 1:nvperp, ivpa in 1:nvpa

            pdf_sym = dfni_func(vpa[ivpa],vperp[ivperp],z_local[iz],r_local[ir],time[it])
            pdf_norm[it] += (pdf[ivpa,ivperp,iz,ir,is,it] - pdf_sym)^2
        end
    end
    for it in 1:ntime
        pdf_norm[it] = sqrt(pdf_norm[it]/(nr_global*nz_global*nvpa*nvperp))
    end
    println("test: ",file_string,": ",spec_string," ",pdf_norm)
    @views plot(time, pdf_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=norm_label) #, yaxis=:log)
    outfile = string(run_name, "_"*file_string*"_norm_vs_t_", spec_string, ".pdf")
    savefig(outfile)
    
    # plot distribution at lower wall boundary
    # find the number of ranks
    z_nrank, r_nrank = get_nranks(run_name,nblocks,"dfns")
    for iblock in 0:nblocks-1
        fid_pdfs = open_readonly_output_file(run_name,"dfns",iblock=iblock, printout=false)
        z_irank, r_irank = load_rank_data(fid_pdfs,printout=false)
        if (z_irank == 0 || z_irank == z_nrank - 1) && r_irank == 0
            pdf = load_pdf_data(fid_pdfs, printout=false)
            # local local grid data on iblock=0
            nz_local, _, z_local, z_wgts_local, Lz = load_coordinate_data(fid_pdfs, "z")
            nr_local, _, r_local, r_wgts_local, Lr = load_coordinate_data(fid_pdfs, "r")
            pdf_sym_array = copy(vpa)
            # plot a thermal vperp on line plots
            ivperp0 = floor(mk_int,nvpa/3)
            # plot a typical r on line plots
            ir0 = 1
            # plot at the wall boundary 
            if z_irank == 0
                iz0 = 1
                zlabel="wall-"
            elseif z_irank == z_nrank - 1
                iz0 = nz_local
                zlabel="wall+"
            end
            for ivpa in 1:nvpa
                pdf_sym_array[ivpa] = dfni_func(vpa[ivpa],vperp[ivperp0],z_local[iz0],r_local[ir0],time[ntime])
            end
            # plot f(vpa,ivperp0,iz_wall,ir0,is,itime) at the wall
            @views plot(vpa, [pdf[:,ivperp0,iz0,ir0,is,ntime], pdf_sym_array], xlabel=L"v_{\|\|}/L_{v_{\|\|}}", ylabel=L"f_i", label=["num" "sym"])
            outfile = string(run_name, "_pdf(vpa,vperp0,iz_"*zlabel*",ir0)_sym_vs_vpa.pdf")
            savefig(outfile) 
        end
        close(fid_pdfs)
    end    
    return pdf_norm
end

function compare_neutral_pdf_symbolic_test(run_name,manufactured_solns_list,spec_string,
    pdf_label,pdf_sym_label,norm_label,file_string)
    fid = open_readonly_output_file(run_name,"dfns", printout=false)
    # load block data on iblock=0
    nblocks, iblock = load_block_data(fid, printout=false)
    # load global sizes of grids that are distributed in memory
    nz_local, nz_global, z_local, z_wgts_local, Lz = load_coordinate_data(fid, "z", printout=false)
    nr_local, nr_global, r_local, r_wgts_local, Lr = load_coordinate_data(fid, "r", printout=false)
    # velocity grid data on iblock=0 (same for all blocks)
    nvz, _, vz, vz_wgts, Lvz = load_coordinate_data(fid, "vz", printout=false)
    nvr, _, vr, vr_wgts, Lvr = load_coordinate_data(fid, "vr", printout=false)
    nvzeta, _, vzeta, vzeta_wgts, Lvzeta = load_coordinate_data(fid, "vzeta", printout=false)
    # load time data (unique to pdf, may differ to moment values depending on user nwrite_dfns value)
    ntime, time = load_time_data(fid, printout=false)
    close(fid)
    # get the charged particle pdf
    dfnn_func = manufactured_solns_list.dfnn_func
    is = 1 # only one species supported currently
    
    pdf_norm = zeros(mk_float,ntime)
    for iblock in 0:nblocks-1
        fid_pdfs = open_readonly_output_file(run_name,"dfns",iblock=iblock, printout=false)
        z_irank, r_irank = load_rank_data(fid_pdfs,printout=false)
        pdf = load_neutral_pdf_data(fid_pdfs, printout=false)
        # load local grid data
        nz_local, _, z_local, z_wgts_local, Lz = load_coordinate_data(fid_pdfs, "z", printout=false)
        nr_local, _, r_local, r_wgts_local, Lr = load_coordinate_data(fid_pdfs, "r", printout=false)
        close(fid_pdfs)
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        for it in 1:ntime, ir in imin_r:nr_local, iz in imin_z:nz_local,
                ivzeta in 1:nvzeta, ivr in 1:nvr, ivz in 1:nvz

            pdf_sym = dfnn_func(vz[ivz],vr[ivr],vzeta[ivzeta],z_local[iz],r_local[ir],time[it])
            pdf_norm[it] += (pdf[ivz,ivr,ivzeta,iz,ir,is,it] - pdf_sym)^2
        end
    end
    for it in 1:ntime
        pdf_norm[it] = sqrt(pdf_norm[it]/(nr_global*nz_global*nvz*nvr*nvzeta))
    end
    println("test: ",file_string,": ",spec_string," ",pdf_norm)
    @views plot(time, pdf_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=norm_label) #, yaxis=:log)
    outfile = string(run_name, "_"*file_string*"_norm_vs_t_", spec_string, ".pdf")
    savefig(outfile)

    return pdf_norm
end

function init_postprocessing_options(pp, nvpa, nvperp, nz, nr, nvz, nvr, nvzeta, ntime)
    print("Initializing the post-processing input options...")
    # nwrite_movie is the stride used when making animations
    nwrite_movie = pp.nwrite_movie
    # itime_min is the minimum time index at which to start animations
    if pp.itime_min > 0 && pp.itime_min <= ntime
        itime_min = pp.itime_min
    else
        itime_min = 1
    end
    # itime_max is the final time index at which to end animations
    # if itime_max < 0, the value used will be the total number of time slices
    if pp.itime_max > 0 && pp.itime_max <= ntime
        itime_max = pp.itime_max
    else
        itime_max = ntime
    end
    # ir0 is the ir index used when plotting data at a single r location
    # by default, it will be set to cld(nr,3) unless a non-negative value provided
    if pp.ir0 > 0
        ir0 = pp.ir0
    else
        ir0 = cld(nr,3)
    end
    # iz0 is the iz index used when plotting data at a single z location
    # by default, it will be set to cld(nz,3) unless a non-negative value provided
    if pp.iz0 > 0
        iz0 = pp.iz0
    else
        iz0 = cld(nz,3)
    end
    # ivperp0 is the iz index used when plotting data at a single vperp location
    # by default, it will be set to cld(nvperp,3) unless a non-negative value provided
    if pp.ivperp0 > 0
        ivperp0 = pp.ivperp0
    else
        ivperp0 = cld(nvperp,3)
    end
    # ivpa0 is the iz index used when plotting data at a single vpa location
    # by default, it will be set to cld(nvpa,3) unless a non-negative value provided
    if pp.ivpa0 > 0
        ivpa0 = pp.ivpa0
    else
        ivpa0 = cld(nvpa,3)
    end
    # ivz0 is the ivr index used when plotting data at a single vz location
    # by default, it will be set to cld(nvz,3) unless a non-negative value provided
    if pp.ivz0 > 0
        ivz0 = pp.ivz0
    else
        ivz0 = cld(nvz,3)
    end
    # ivr0 is the ivr index used when plotting data at a single vr location
    # by default, it will be set to cld(nvr,3) unless a non-negative value provided
    if pp.ivr0 > 0
        ivr0 = pp.ivr0
    else
        ivr0 = cld(nvr,3)
    end
    # ivzeta0 is the ivzeta index used when plotting data at a single vzeta location
    # by default, it will be set to cld(nvr,3) unless a non-negative value provided
    if pp.ivzeta0 > 0
        ivzeta0 = pp.ivzeta0
    else
        ivzeta0 = cld(nvzeta,3)
    end
    println("done.")
    return nwrite_movie, itime_min, itime_max, ivpa0, ivperp0, iz0, ir0, ivz0, ivr0, ivzeta0
end

"""
"""
function plot_1D_1V_diagnostics(run_name, fid, nwrite_movie, itime_min, itime_max, ivpa0, iz0, ir0, r,
 phi, density, parallel_flow, parallel_pressure, parallel_heat_flux,
     thermal_speed, ff, n_species, evolve_ppar, nvpa, vpa, vpa_wgts,
                                nz, z, z_wgts, Lz, ntime, time)
    # analyze the fields data
    phi_fldline_avg, delta_phi = analyze_fields_data(phi, ntime, nz, z_wgts, Lz)
    # use a fit to calculate and write to file the damping rate and growth rate of the
    # perturbed electrostatic potential
    frequency, growth_rate, shifted_time, fitted_delta_phi =
        calculate_and_write_frequencies(fid, run_name, ntime, time, z, itime_min,
                                        itime_max, iz0, delta_phi, pp)
    # create the requested plots of the fields
    plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
                z, iz0, run_name, fitted_delta_phi, pp)
    # load velocity moments data
    # analyze the velocity moments data
    density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, qpar_fldline_avg,
        delta_density, delta_upar, delta_ppar, delta_qpar =
        analyze_moments_data(density, parallel_flow, parallel_pressure, parallel_heat_flux,
                             ntime, n_species, nz, z_wgts, Lz)
    # create the requested plots of the moments
    plot_moments(density, delta_density, density_fldline_avg,
        parallel_flow, delta_upar, upar_fldline_avg,
        parallel_pressure, delta_ppar, ppar_fldline_avg,
        parallel_heat_flux, delta_qpar, qpar_fldline_avg,
        pp, run_name, time, itime_min, itime_max,
        nwrite_movie, z, iz0, n_species)
    # load particle distribution function (pdf) data
    # analyze the pdf data
    f_fldline_avg, delta_f, dens_moment, upar_moment, ppar_moment =
        analyze_pdf_data(ff, vpa, nvpa, nz, n_species, ntime, vpa_wgts, z_wgts,
                         Lz, thermal_speed, evolve_ppar)

    println("Plotting distribution function data...")
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    for is ∈ 1:n_species
        if n_species > 1
            spec_string = string("_spec", string(is))
        else
            spec_string = ""
        end
        # plot difference between evolved density and ∫dvpa f; only possibly different if density removed from
        # normalised distribution function at run-time
        @views plot(time, density[iz0,is,:] .- dens_moment[iz0,is,:])
        outfile = string(run_name, "_intf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        # if evolve_upar = true, plot ∫dwpa wpa * f, which should equal zero
        # otherwise, this plots ∫dvpa vpa * f, which is dens*upar
        intwf0_max = maximum(abs.(upar_moment[iz0,is,:]))
        if intwf0_max < 1.0e-15
            @views plot(time, upar_moment[iz0,is,:], ylims = (-1.0e-15, 1.0e-15))
        else
            @views plot(time, upar_moment[iz0,is,:])
        end
        outfile = string(run_name, "_intwf0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        # plot difference between evolved parallel pressure and ∫dvpa vpa^2 f;
        # only possibly different if density and thermal speed removed from
        # normalised distribution function at run-time
        @views plot(time, parallel_pressure[iz0,is,:] .- ppar_moment[iz0,is,:])
        outfile = string(run_name, "_intw2f0_vs_t", spec_string, ".pdf")
        savefig(outfile)
        #fmin = minimum(ff[:,:,is,:])
        #fmax = maximum(ff[:,:,is,:])
        if pp.animate_f_vs_vpa_z
            # make a gif animation of ln f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                @views heatmap(z, vpa, log.(abs.(ff[:,:,is,i])), xlabel="z", ylabel="vpa", fillcolor = logdeep)
            end
            outfile = string(run_name, "_logf_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make a gif animation of f(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                @views heatmap(z, vpa, ff[:,:,is,i], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_f_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
            # make pdf of f(vpa,z,t_final) for each species
            str = string("spec ", string(is), " pdf")
            @views heatmap(z, vpa, ff[:,:,is,end], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic, title=str)
            outfile = string(run_name, "_f_vs_z_vpa_final", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.animate_deltaf_vs_vpa_z
            # make a gif animation of δf(vpa,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z, vpa, delta_f[:,:,is,i], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_deltaf_vs_vpa_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(ff[ivpa0,:,is,:])
            fmax = maximum(ff[ivpa0,:,is,:])
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, ff[ivpa0,:,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_f_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(delta_f[ivpa0,:,is,:])
            fmax = maximum(delta_f[ivpa0,:,is,:])
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, delta_f[ivpa0,:,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_deltaf_vs_z", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(ff[:,iz0,is,:])
            fmax = maximum(ff[:,iz0,is,:])

            # if is == 1
            #     tmp = copy(ff)
            #     @. tmp[:,1,1,:] /= vpa^2
            #     bohm_integral = copy(time)
            #     for i ∈ 1:ntime
            #         @views bohm_integral[i] = integrate_over_vspace(tmp[1:cld(nvpa,2)-1,1,1,i],vpa_wgts[1:cld(nvpa,2)-1])/2.0
            #     end
            #     plot(time, bohm_integral, xlabel="time", label="Bohm integral")
            #     plot!(time, density[1,1,:], label="nᵢ(zmin)")
            #     outfile = string(run_name, "_Bohm_criterion.pdf")
            #     savefig(outfile)
            #     println()
            #     if bohm_integral[end] <= density[1,1,end]
            #         println("Bohm criterion: ", bohm_integral[end], " <= ", density[1,1,end], " is satisfied!")
            #     else
            #         println("Bohm criterion: ", bohm_integral[end], " <= ", density[1,1,end], " is not satisfied!")
            #     end
            #     println()
            #     for j ∈ 0:10
            #         println("j: ", j, "  Bohm integral: ", integrate_over_vspace(tmp[1:cld(nvpa,2)-j,1,1,end],vpa_wgts[1:cld(nvpa,2)-j,end])/2.0)
            #     end
            # end
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                #@views plot(vpa, ff[iz0,:,is,i], ylims = (fmin,fmax))
                @views plot(vpa, ff[:,iz0,is,i])
            end
            outfile = string(run_name, "_f_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(delta_f[:,iz0,is,:])
            fmax = maximum(delta_f[:,iz0,is,:])
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(vpa, delta_f[:,iz0,is,i], ylims = (fmin,fmax))
            end
            outfile = string(run_name, "_deltaf_vs_vpa", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")

end

"""
"""
function calculate_and_write_frequencies(fid, run_name, ntime, time, z, itime_min,
                                         itime_max, iz0, delta_phi, pp)
    if pp.calculate_frequencies
        println("Calculating the frequency and damping/growth rate...")
        # shifted_time = t - t0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        # assume phi(z0,t) = A*exp(growth_rate*t)*cos(ω*t + φ)
        # and fit phi(z0,t)/phi(z0,t0), which eliminates the constant A pre-factor
        @views phi_fit = fit_delta_phi_mode(shifted_time[itime_min:itime_max], z,
                                            delta_phi[:, itime_min:itime_max])
        frequency = phi_fit.frequency
        growth_rate = phi_fit.growth_rate

        # write info related to fit to file
        io = open_ascii_output_file(run_name, "frequency_fit.txt")
        println(io, "#growth_rate: ", phi_fit.growth_rate,
                "  frequency: ", phi_fit.frequency,
                " fit_errors: ", phi_fit.amplitude_fit_error, " ",
                phi_fit.offset_fit_error, " ", phi_fit.cosine_fit_error)
        println(io)

        # Calculate the fitted phi as a function of time at index iz0
        L = z[end] - z[begin]
        fitted_delta_phi =
            @. (phi_fit.amplitude0 * cos(2.0 * π * (z[iz0] + phi_fit.offset0) / L)
                * exp(phi_fit.growth_rate * shifted_time)
                * cos(phi_fit.frequency * shifted_time + phi_fit.phase))
        for i ∈ 1:ntime
            println(io, "time: ", time[i], "  delta_phi: ", delta_phi[iz0,i],
                    "  fitted_delta_phi: ", fitted_delta_phi[i])
        end
        close(io)
        # also save fit to NetCDF file
        function get_or_create(name, description, dims=())
            if name in fid
                return fid[name]
            else
                return defVar(fid, name, mk_float, dims,
                              attrib=Dict("description"=>description))
            end
        end
        var = get_or_create("growth_rate", "mode growth rate from fit")
        var[:] = phi_fit.growth_rate
        var = get_or_create("frequency","mode frequency from fit")
        var[:] = phi_fit.frequency
        var = get_or_create("delta_phi", "delta phi from simulation", ("nz", "ntime"))
        var[:,:] = delta_phi
        var = get_or_create("phi_amplitude", "amplitude of delta phi from fit over z",
                            ("ntime",))
        var[:,:] = phi_fit.amplitude
        var = get_or_create("phi_offset", "offset of delta phi from fit over z",
                            ("ntime",))
        var[:,:] = phi_fit.offset
        var = get_or_create("fitted_delta_phi","fit to delta phi", ("ntime",))
        var[:] = fitted_delta_phi
        var = get_or_create("amplitude_fit_error",
                            "RMS error on the fit of the ln(amplitude) of phi")
        var[:] = phi_fit.amplitude_fit_error
        var = get_or_create("offset_fit_error",
                            "RMS error on the fit of the offset of phi")
        var[:] = phi_fit.offset_fit_error
        var = get_or_create("cosine_fit_error",
                            "Maximum over time of the RMS error on the fit of a cosine "
                            * "to phi.")
        var[:] = phi_fit.cosine_fit_error
        println("done.")
    else
        frequency = 0.0
        growth_rate = 0.0
        phase = 0.0
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        fitted_delta_phi = zeros(ntime)

    end
    return frequency, growth_rate, shifted_time, fitted_delta_phi
end

"""
"""
function plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
    z, iz0, run_name, fitted_delta_phi, pp)

    println("Plotting fields data...")
    phimin = minimum(phi)
    phimax = maximum(phi)
    if pp.plot_phi0_vs_t
        # plot the time trace of phi(z=z0)
        #plot(time, log.(phi[i,:]), yscale = :log10)
        @views plot(time, phi[iz0,:])
        outfile = string(run_name, "_phi0_vs_t.pdf")
        savefig(outfile)
        # plot the time trace of phi(z=z0)-phi_fldline_avg
        @views plot(time, abs.(delta_phi[iz0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log)
        if pp.calculate_frequencies
            plot!(time, abs.(fitted_delta_phi))
        end
        outfile = string(run_name, "_delta_phi0_vs_t.pdf")
        savefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(z,t)
        heatmap(time, z, phi, xlabel="time", ylabel="z", title="ϕ", c = :deep)
        outfile = string(run_name, "_phi_vs_z_t.pdf")
        savefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(z, phi[:,i], xlabel="z", ylabel="ϕ", ylims = (phimin,phimax))
        end
        outfile = string(run_name, "_phi_vs_z.gif")
        gif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(run_name, "_harrison_comparison.pdf")
    # savefig(outfile)
    plot(z, phi[:,end], xlabel="z/Lz", ylabel="eϕ/Te", label="", linewidth=2)
    outfile = string(run_name, "_phi_final.pdf")
    savefig(outfile)

    println("done.")
end

"""
"""
function plot_moments(density, delta_density, density_fldline_avg,
    parallel_flow, delta_upar, upar_fldline_avg,
    parallel_pressure, delta_ppar, ppar_fldline_avg,
    parallel_heat_flux, delta_qpar, qpar_fldline_avg,
    pp, run_name, time, itime_min, itime_max, nwrite_movie,
    z, iz0, n_species)
    println("Plotting velocity moments data...")
    # plot the species-summed, field-line averaged vs time
    denstot = copy(density_fldline_avg)
    denstot .= sum(density_fldline_avg,dims=1)
    @. denstot /= denstot[1,1]
    denstot_min = minimum(denstot[1,:]) - 0.1
    denstot_max = maximum(denstot[1,:]) + 0.1
    @views plot(time, denstot[1,:], ylims=(denstot_min,denstot_max), xlabel="time", ylabel="∑ⱼn̅ⱼ(t)/∑ⱼn̅ⱼ(0)", label="", linewidth=2)
    outfile = string(run_name, "_denstot_vs_t.pdf")
    savefig(outfile)
    for is ∈ 1:n_species
        spec_string = string(is)
        dens_min = minimum(density[:,is,:])
        dens_max = maximum(density[:,is,:])
        if pp.plot_dens0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, density[iz0,is,:])
            outfile = string(run_name, "_dens0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_density[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_dens0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of density_fldline_avg
            @views plot(time, density_fldline_avg[is,:], xlabel="time", ylabel="<ns/Nₑ>", ylims=(dens_min,dens_max))
            outfile = string(run_name, "_fldline_avg_dens_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the deviation from conservation of density_fldline_avg
            @views plot(time, density_fldline_avg[is,:] .- density_fldline_avg[is,1], xlabel="time", ylabel="<(ns-ns(0))/Nₑ>")
            outfile = string(run_name, "_conservation_dens_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        upar_min = minimum(parallel_flow[:,is,:])
        upar_max = maximum(parallel_flow[:,is,:])
        if pp.plot_upar0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, parallel_flow[iz0,is,:])
            outfile = string(run_name, "_upar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_upar[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_upar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            @views plot(time, upar_fldline_avg[is,:], xlabel="time", ylabel="<upars/sqrt(2Te/ms)>", ylims=(upar_min,upar_max))
            outfile = string(run_name, "_fldline_avg_upar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        ppar_min = minimum(parallel_pressure[:,is,:])
        ppar_max = maximum(parallel_pressure[:,is,:])
        if pp.plot_ppar0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, parallel_pressure[iz0,is,:])
            outfile = string(run_name, "_ppar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_ppar[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_ppar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            @views plot(time, ppar_fldline_avg[is,:], xlabel="time", ylabel="<ppars/NₑTₑ>", ylims=(ppar_min,ppar_max))
            outfile = string(run_name, "_fldline_avg_ppar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        qpar_min = minimum(parallel_heat_flux[:,is,:])
        qpar_max = maximum(parallel_heat_flux[:,is,:])
        if pp.plot_qpar0_vs_t
            # plot the time trace of n_s(z=z0)
            @views plot(time, parallel_heat_flux[iz0,is,:])
            outfile = string(run_name, "_qpar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            @views plot(time, abs.(delta_qpar[iz0,is,:]), yaxis=:log)
            outfile = string(run_name, "_delta_qpar0_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
            # plot the time trace of ppar_fldline_avg
            @views plot(time, qpar_fldline_avg[is,:], xlabel="time", ylabel="<qpars/NₑTₑvth>", ylims=(qpar_min,qpar_max))
            outfile = string(run_name, "_fldline_avg_qpar_vs_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_dens_vs_z_t
            # make a heatmap plot of n_s(z,t)
            heatmap(time, z, density[:,is,:], xlabel="time", ylabel="z", title="ns/Nₑ", c = :deep)
            outfile = string(run_name, "_dens_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_upar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            heatmap(time, z, parallel_flow[:,is,:], xlabel="time", ylabel="z", title="upars/vt", c = :deep)
            outfile = string(run_name, "_upar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_ppar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            heatmap(time, z, parallel_pressure[:,is,:], xlabel="time", ylabel="z", title="ppars/NₑTₑ", c = :deep)
            outfile = string(run_name, "_ppar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.plot_qpar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            heatmap(time, z, parallel_heat_flux[:,is,:], xlabel="time", ylabel="z", title="qpars/NₑTₑvt", c = :deep)
            outfile = string(run_name, "_qpar_vs_z_t_spec", spec_string, ".pdf")
            savefig(outfile)
        end
        if pp.animate_dens_vs_z
            # make a gif animation of ϕ(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, density[:,is,i], xlabel="z", ylabel="nᵢ/Nₑ", ylims = (dens_min,dens_max))
            end
            outfile = string(run_name, "_dens_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        if pp.animate_upar_vs_z
            # make a gif animation of ϕ(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views plot(z, parallel_flow[:,is,i], xlabel="z", ylabel="upars/vt", ylims = (upar_min,upar_max))
            end
            outfile = string(run_name, "_upar_vs_z_spec", spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")
end

"""
Fit delta_phi to get the frequency and growth rate.

Note, expect the input to be a standing wave (as simulations are initialised with just a
density perturbation), so need to extract both frequency and growth rate from the
time-variation of the amplitude.

The function assumes that if the amplitude does not cross zero, then the mode is
non-oscillatory and so fits just an exponential, not exp*cos. The simulation used as
input should be long enough to contain at least ~1 period of oscillation if the mode is
oscillatory or the fit will not work.

Arguments
---------
z : Array{mk_float, 1}
    1d array of the grid point positions
t : Array{mk_float, 1}
    1d array of the time points
delta_phi : Array{mk_float, 2}
    2d array of the values of delta_phi(z, t)

Returns
-------
phi_fit_result struct whose fields are:
    growth_rate : mk_float
        Fitted growth rate of the mode
    amplitude0 : mk_float
        Fitted amplitude at t=0
    frequency : mk_float
        Fitted frequency of the mode
    offset0 : mk_float
        Fitted offset at t=0
    amplitude_fit_error : mk_float
        RMS error in fit to ln(amplitude) - i.e. ln(A)
    offset_fit_error : mk_float
        RMS error in fit to offset - i.e. δ
    cosine_fit_error : mk_float
        Maximum of the RMS errors of the cosine fits at each time point
    amplitude : Array{mk_float, 1}
        Values of amplitude from which growth_rate fit was calculated
    offset : Array{mk_float, 1}
        Values of offset from which frequency fit was calculated
"""
function fit_delta_phi_mode(t, z, delta_phi)
    # First fit a cosine to each time slice
    results = allocate_float(3, size(delta_phi)[2])
    amplitude_guess = 1.0
    offset_guess = 0.0
    for (i, phi_z) in enumerate(eachcol(delta_phi))
        results[:, i] .= fit_cosine(z, phi_z, amplitude_guess, offset_guess)
        (amplitude_guess, offset_guess) = results[1:2, i]
    end

    amplitude = results[1, :]
    offset = results[2, :]
    cosine_fit_error = results[3, :]

    L = z[end] - z[begin]

    # Choose initial amplitude to be positive, for convenience.
    if amplitude[1] < 0
        # 'Wrong sign' of amplitude is equivalent to a phase shift by π
        amplitude .*= -1.0
        offset .+= L / 2.0
    end

    # model for linear fits
    @. model(t, p) = p[1] * t + p[2]

    # Fit offset vs. time
    # Would give phase velocity for a travelling wave, but we expect either a standing
    # wave or a zero-frequency decaying mode, so expect the time variation of the offset
    # to be ≈0
    offset_fit = curve_fit(model, t, offset, [1.0, 0.0])
    doffsetdt = offset_fit.param[1]
    offset0 = offset_fit.param[2]
    offset_error = sqrt(mean(offset_fit.resid .^ 2))
    offset_tol = 2.e-5
    if abs(doffsetdt) > offset_tol
        println("WARNING: d(offset)/dt=", doffsetdt, " is non-negligible (>", offset_tol,
              ") but fit_delta_phi_mode expected either a standing wave or a ",
              "zero-frequency decaying mode.")
    end

    growth_rate = 0.0
    amplitude0 = 0.0
    frequency = 0.0
    phase = 0.0
    fit_error = 0.0
    if all(amplitude .> 0.0)
        # No zero crossing, so assume the mode is non-oscillatory (i.e. purely
        # growing/decaying).

        # Fit ln(amplitude) vs. time so we don't give extra weight to early time points
        amplitude_fit = curve_fit(model, t, log.(amplitude), [-1.0, 1.0])
        growth_rate = amplitude_fit.param[1]
        amplitude0 = exp(amplitude_fit.param[2])
        fit_error = sqrt(mean(amplitude_fit.resid .^ 2))
        frequency = 0.0
        phase = 0.0
    else
        converged = false
        maxiter = 100
        for iter ∈ 1:maxiter
            @views growth_rate_change, frequency, phase, fit_error =
                fit_phi0_vs_time(exp.(-growth_rate*t) .* amplitude, t)
            growth_rate += growth_rate_change
            println("growth_rate: ", growth_rate, "  growth_rate_change/growth_rate: ", growth_rate_change/growth_rate, "  fit_error: ", fit_error)
            if abs(growth_rate_change/growth_rate) < 1.0e-12 || fit_error < 1.0e-11
                converged = true
                break
            end
        end
        if !converged
            println("WARNING: Iteration to find growth rate failed to converge in ", maxiter, " iterations")
        end
        amplitude0 = amplitude[1] / cos(phase)
    end

    return (growth_rate=growth_rate, frequency=frequency, phase=phase,
            amplitude0=amplitude0, offset0=offset0, amplitude_fit_error=fit_error,
            offset_fit_error=offset_error, cosine_fit_error=maximum(cosine_fit_error),
            amplitude=amplitude, offset=offset)
end

function fit_phi0_vs_time(phi0, tmod)
    # the model we are fitting to the data is given by the function 'model':
    # assume phi(z0,t) = exp(γt)cos(ωt+φ) so that
    # phi(z0,t)/phi(z0,t0) = exp((t-t₀)γ)*cos((t-t₀)*ω + phase)/cos(phase),
    # where tmod = t-t0 and phase = ωt₀-φ
    @. model(t, p) = exp(p[1]*t) * cos(p[2]*t + p[3]) / cos(p[3])
    model_params = allocate_float(3)
    model_params[1] = -0.1
    model_params[2] = 8.6
    model_params[3] = 0.0
    @views fit = curve_fit(model, tmod, phi0/phi0[1], model_params)
    # get the confidence interval at 10% level for each fit parameter
    #se = standard_error(fit)
    #standard_deviation = Array{Float64,1}
    #@. standard_deviation = se * sqrt(size(tmod))

    fitted_function = model(tmod, fit.param)
    norm = moving_average(@.((abs(phi0/phi0[1]) + abs(fitted_function))^2), 1)
    fit_error = sqrt(mean(@.((phi0/phi0[1] - fitted_function)^2 / norm)))

    return fit.param[1], fit.param[2], fit.param[3], fit_error
end

"""
Fit a cosine to a 1d array

Fit function is A*cos(2*π*n*(z + δ)/L)

The domain z is taken to be periodic, with the first and last points identified, so
L=z[end]-z[begin]

Arguments
---------
z : Array
    1d array with positions of the grid points - should have the same length as data
data : Array
    1d array of the data to be fit
amplitude_guess : Float
    Initial guess for the amplitude (the value from the previous time point might be a
    good choice)
offset_guess : Float
    Initial guess for the offset (the value from the previous time point might be a good
    choice)
n : Int, default 1
    The periodicity used for the fit

Returns
-------
amplitude : Float
    The amplitude A of the cosine fit
offset : Float
    The offset δ of the cosine fit
error : Float
    The RMS of the difference between data and the fit
"""
function fit_cosine(z, data, amplitude_guess, offset_guess, n=1)
    # Length of domain
    L = z[end] - z[begin]

    @. model(z, p) = p[1] * cos(2*π*n*(z + p[2])/L)
    fit = curve_fit(model, z, data, [amplitude_guess, offset_guess])

    # calculate error
    error = sqrt(mean(fit.resid .^ 2))

    return fit.param[1], fit.param[2], error
end

#function advection_test_1d(fstart, fend)
#    rmserr = sqrt(sum((fend .- fstart).^2))/(size(fend,1)*size(fend,2)*size(fend,3))
#    println("advection_test_1d rms error: ", rmserr)
#end

function plot_fields_rt(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
    r, ir0, run_name, fitted_delta_phi, pp)

    println("Plotting fields data...")
    phimin = minimum(phi)
    phimax = maximum(phi)
    if pp.plot_phi0_vs_t
        # plot the time trace of phi(r=r0)
        #plot(time, log.(phi[i,:]), yscale = :log10)
        @views plot(time, phi[ir0,:])
        outfile = string(run_name, "_phi(r0,z0)_vs_t.pdf")
        savefig(outfile)
        # plot the time trace of phi(r=r0)-phi_fldline_avg
        @views plot(time, abs.(delta_phi[ir0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log)
        if pp.calculate_frequencies
            plot!(time, abs.(fitted_delta_phi))
        end
        outfile = string(run_name, "_delta_phi(r0,z0)_vs_t.pdf")
        savefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(r,t)
        heatmap(time, r, phi, xlabel="time", ylabel="r", title="ϕ", c = :deep)
        outfile = string(run_name, "_phi_vs_r_t.pdf")
        savefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(r) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(r, phi[:,i], xlabel="r", ylabel="ϕ", ylims = (phimin,phimax))
        end
        outfile = string(run_name, "_phi_vs_r.gif")
        gif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(run_name, "_harrison_comparison.pdf")
    # savefig(outfile)
    plot(r, phi[:,end], xlabel="r/Lr", ylabel="eϕ/Te", label="", linewidth=2)
    outfile = string(run_name, "_phi(r)_final.pdf")
    savefig(outfile)

    println("done.")
end

"""
plots various slices of the ion pdf (1d and 2d, stills and animations)
"""
function plot_charged_pdf(pdf, vpa, vperp, z, r,
    ivpa0, ivperp0, iz0, ir0,
    spec_type, n_species,
    itime_min, itime_max, nwrite_movie, run_name, pp)

    print("Plotting ion distribution function data...")

    # set up a color scheme for heat maps
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    # create strings to help identify phase space location and species
    # in file names
    ivpa0_string = string("_ivpa0", string(ivpa0))
    ivperp0_string = string("_ivperp0", string(ivperp0))
    iz0_string = string("_iz0", string(iz0))
    ir0_string = string("_ir0", string(ir0))
    # create animations of the ion pdf
    for is ∈ 1:n_species
        if n_species > 1
            spec_string = string("_", spec_type, "_spec", string(is))
        else
            spec_string = string("_", spec_type)
        end
        # make a gif animation of f(vpa,z,t) at a given (vperp,r) location
        if pp.animate_f_vs_vpa_z
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z, vpa, pdf[:,ivperp0,:,ir0,is,i], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vpa_z", ivperp0_string, ir0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(vpa,r,t) at a given (vperp,z) location
        if pp.animate_f_vs_vpa_r
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r, vpa, pdf[:,ivperp0,iz0,:,is,i], xlabel="r", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vpa_r", ivperp0_string, iz0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(vperp,z,t) at a given (vpa,r) location
        if pp.animate_f_vs_vperp_z
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z, vperp, pdf[ivpa0,:,:,ir0,is,i], xlabel="z", ylabel="vperp", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vperp_z", ivpa0_string, ir0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(vperp,r,t) at a given (vpa,z) location
        if pp.animate_f_vs_vperp_r
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r, vperp, pdf[ivpa0,:,iz0,:,is,i], xlabel="r", ylabel="vperp", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vperp_r", ivperp0_string, iz0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(vpa,vperp,t) at a given (z,r) location
        if pp.animate_f_vs_vperp_vpa
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(vperp, vpa, pdf[:,:,iz0,ir0,is,i], xlabel="vperp", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vperp_vpa", iz0_string, ir0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(z,r,t) at a given (vpa,vperp) location
        if pp.animate_f_vs_r_z
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r, z, pdf[ivpa0,ivperp0,:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_r_z", ivpa0_string, ivperp0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")
end

"""
plots various slices of the neutral pdf (1d and 2d, stills and animations)
"""
function plot_neutral_pdf(pdf, vz, vr, vzeta, z, r,
    ivz0, ivr0, ivzeta0, iz0, ir0,
    spec_type, n_species,
    itime_min, itime_max, nwrite_movie, run_name, pp)

    print("Plotting neutral distribution function data...")

    # set up a color scheme for heat maps
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    # create strings to help identify phase space location and species
    # in file names
    ivz0_string = string("_ivz0", string(ivz0))
    ivr0_string = string("_ivr0", string(ivr0))
    ivzeta0_string = string("_ivzeta0", string(ivzeta0))
    iz0_string = string("_iz0", string(iz0))
    ir0_string = string("_ir0", string(ir0))
    # create animations of the neutral pdf
    for is ∈ 1:n_species
        if n_species > 1
            spec_string = string("_", spec_type, "_spec", string(is))
        else
            spec_string = string("_", spec_type)
        end
        # make a gif animation of f(vz,z,t) at a given (vr,vzeta,r) location
        if pp.animate_f_vs_vz_z
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z, vz, pdf[:,ivr0,ivzeta0,:,ir0,is,i], xlabel="z", ylabel="vz", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vz_z", ivr0_string, ivzeta0_string, ir0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(vr,r,t) at a given (vz,vzeta,z) location
        if pp.animate_f_vs_vr_r
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r, vr, pdf[ivz0,:,ivzeta0,iz0,:,is,i], xlabel="r", ylabel="vr", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vr_r", ivz0_string, ivzeta0_string, iz0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
        # make a gif animation of f(z,r,t) at a given (vz,vr,vzeta) location
        if pp.animate_f_vs_r_z
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r, z, pdf[ivz0,ivr0,ivzeta0,:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_z_r", ivz0_string, ivr0_string, ivzeta0_string, spec_string, ".gif")
            gif(anim, outfile, fps=5)
        end
    end
    println("done.")
end

function plot_fields_2D(phi, Ez, Er, time, z, r, iz0, ir0,
    itime_min, itime_max, nwrite_movie, run_name, pp, description)
    nr = size(r,1)
    print("Plotting fields data...")
    phimin = minimum(phi)
    phimax = maximum(phi)
    if pp.plot_phi_vs_r0_z # plot last timestep phi[z,ir0]
        @views plot(z, phi[:,ir0,end], xlabel=L"z/L_z", ylabel=L"\phi")
        outfile = string(run_name, "_phi"*description*"(r0,z)_vs_z.pdf")
        savefig(outfile)    
    end
    if pp.animate_phi_vs_r_z && nr > 1
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views heatmap(r, z, phi[:,:,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
        end
        outfile = string(run_name, "_phi"*description*"_vs_r_z.gif")
        gif(anim, outfile, fps=5)
    end
    Ezmin = minimum(Ez)
    Ezmax = maximum(Ez)
    if pp.plot_Ez_vs_r0_z # plot last timestep Ez[z,ir0]
        @views plot(z, Ez[:,ir0,end], xlabel=L"z/L_z", ylabel=L"E_z")
        outfile = string(run_name, "_Ez"*description*"(r0,z)_vs_z.pdf")
        savefig(outfile)    
    end
    if pp.plot_wall_Ez_vs_r && nr > 1 # plot last timestep Ez[z_wall,r]
        @views plot(r, Ez[end,:,end], xlabel=L"r/L_r", ylabel=L"E_z")
        outfile = string(run_name, "_Ez"*description*"(r,z_wall)_vs_r.pdf")
        savefig(outfile)
    end
    if pp.animate_Ez_vs_r_z && nr > 1
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views heatmap(r, z, Ez[:,:,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
        end
        outfile = string(run_name, "_Ez"*description*"_vs_r_z.gif")
        gif(anim, outfile, fps=5)
    end
    Ermin = minimum(Er)
    Ermax = maximum(Er)
    if pp.plot_Er_vs_r0_z # plot last timestep Er[z,ir0]
        @views plot(z, Er[:,ir0,end], xlabel=L"z/L_z", ylabel=L"E_r")
        outfile = string(run_name, "_Er"*description*"(r0,z)_vs_z.pdf")
        savefig(outfile)    
    end
    if pp.plot_wall_Er_vs_r && nr > 1 # plot last timestep Er[z_wall,r]
        @views plot(r, Er[end,:,end], xlabel=L"r/L_r", ylabel=L"E_r")
        outfile = string(run_name, "_Er"*description*"(r,z_wall)_vs_r.pdf")
        savefig(outfile)
    end
    if pp.animate_Er_vs_r_z && nr > 1
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views heatmap(r, z, Er[:,:,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
        end
        outfile = string(run_name, "_Er"*description*"_vs_r_z.gif")
        gif(anim, outfile, fps=5)
    end
    println("done.")
end

function plot_charged_moments_2D(density, parallel_flow, parallel_pressure, time, z, r, iz0, ir0, n_ion_species,
    itime_min, itime_max, nwrite_movie, run_name, pp)
    nr = size(r,1)
    print("Plotting charged moments data...")
    for is in 1:n_ion_species
		description = "_ion_spec"*string(is)*"_"
		# the density
		densitymin = minimum(density[:,:,is,:])
		densitymax = maximum(density)
		if pp.plot_density_vs_r0_z # plot last timestep density[z,ir0]
			@views plot(z, density[:,ir0,is,end], xlabel=L"z/L_z", ylabel=L"n_i")
			outfile = string(run_name, "_density"*description*"(r0,z)_vs_z.pdf")
			savefig(outfile)    
		end
		if pp.plot_wall_density_vs_r && nr > 1 # plot last timestep density[z_wall,r]
			@views plot(r, density[end,:,is,end], xlabel=L"r/L_r", ylabel=L"n_i")
			outfile = string(run_name, "_density"*description*"(r,z_wall)_vs_r.pdf")
			savefig(outfile)
		end
		if pp.animate_density_vs_r_z && nr > 1
			# make a gif animation of ϕ(z) at different times
			anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
				@views heatmap(r, z, density[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
			end
			outfile = string(run_name, "_density"*description*"_vs_r_z.gif")
			gif(anim, outfile, fps=5)
		end
		if pp.plot_density_vs_r_z && nr > 1
			@views heatmap(r, z, density[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
			windowsize = (360,240), margin = 15pt)
			outfile = string(run_name, "_density"*description*"_vs_r_z.pdf")
			savefig(outfile)
		end
		
		# the parallel flow
		parallel_flowmin = minimum(parallel_flow[:,:,is,:])
		parallel_flowmax = maximum(parallel_flow)
		if pp.plot_parallel_flow_vs_r0_z # plot last timestep parallel_flow[z,ir0]
			@views plot(z, parallel_flow[:,ir0,is,end], xlabel=L"z/L_z", ylabel=L"u_{i\|\|}")
			outfile = string(run_name, "_parallel_flow"*description*"(r0,z)_vs_z.pdf")
			savefig(outfile)    
		end
		if pp.plot_wall_parallel_flow_vs_r && nr > 1 # plot last timestep parallel_flow[z_wall,r]
			@views plot(r, parallel_flow[end,:,is,end], xlabel=L"r/L_r", ylabel=L"u_{i\|\|}")
			outfile = string(run_name, "_parallel_flow"*description*"(r,z_wall)_vs_r.pdf")
			savefig(outfile)
		end
		if pp.animate_parallel_flow_vs_r_z && nr > 1
			# make a gif animation of ϕ(z) at different times
			anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
				@views heatmap(r, z, parallel_flow[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
			end
			outfile = string(run_name, "_parallel_flow"*description*"_vs_r_z.gif")
			gif(anim, outfile, fps=5)
		end
		if pp.plot_parallel_flow_vs_r_z && nr > 1
			@views heatmap(r, z, parallel_flow[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
			windowsize = (360,240), margin = 15pt)
			outfile = string(run_name, "_parallel_flow"*description*"_vs_r_z.pdf")
			savefig(outfile)
		end
		
		# the parallel pressure
		parallel_pressuremin = minimum(parallel_pressure[:,:,is,:])
		parallel_pressuremax = maximum(parallel_pressure)
		if pp.plot_parallel_pressure_vs_r0_z # plot last timestep parallel_pressure[z,ir0]
			@views plot(z, parallel_pressure[:,ir0,is,end], xlabel=L"z/L_z", ylabel=L"p_{i\|\|}")
			outfile = string(run_name, "_parallel_pressure"*description*"(r0,z)_vs_z.pdf")
			savefig(outfile)    
		end
		if pp.plot_wall_parallel_pressure_vs_r && nr > 1 # plot last timestep parallel_pressure[z_wall,r]
			@views plot(r, parallel_pressure[end,:,is,end], xlabel=L"r/L_r", ylabel=L"p_{i\|\|}")
			outfile = string(run_name, "_parallel_pressure"*description*"(r,z_wall)_vs_r.pdf")
			savefig(outfile)
		end
		if pp.animate_parallel_pressure_vs_r_z && nr > 1
			# make a gif animation of ϕ(z) at different times
			anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
				@views heatmap(r, z, parallel_pressure[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
			end
			outfile = string(run_name, "_parallel_pressure"*description*"_vs_r_z.gif")
			gif(anim, outfile, fps=5)
		end
		if pp.plot_parallel_pressure_vs_r_z && nr > 1
			@views heatmap(r, z, parallel_pressure[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
			windowsize = (360,240), margin = 15pt)
			outfile = string(run_name, "_parallel_pressure"*description*"_vs_r_z.pdf")
			savefig(outfile)
		end
	end
    println("done.")
end

function plot_charged_pdf_2D_at_wall(run_name)
    print("Plotting charged pdf data at wall boundaries...")
    # open a dfn file
    fid = open_readonly_output_file(run_name,"dfns", printout=false)
    # load block data on iblock=0
    nblocks, iblock = load_block_data(fid, printout=false)
    # velocity grid data on iblock=0 (same for all blocks)
    nvpa, _, vpa, vpa_wgts, Lvpa = load_coordinate_data(fid, "vpa", printout=false)
    nvperp, _, vperp, vperp_wgts, Lvperp = load_coordinate_data(fid, "vperp", printout=false)
    # load time data (unique to pdf, may differ to moment values depending on user nwrite_dfns value)
    ntime, time = load_time_data(fid, printout=false)
    # load species data 
    n_ion_species, n_neutral_species = load_species_data(fid)
    close(fid)
    
    # plot only data at last timestep
    itime0 = ntime
    # plot a thermal vpa on line plots
    ivpa0 = floor(mk_int,nvpa/3)
    # plot a thermal vperp on line plots
    ivperp0 = floor(mk_int,nvpa/3)
    # plot a typical r on line plots
    ir0 = 1
    
    # find the number of ranks
    z_nrank, r_nrank = get_nranks(run_name,nblocks,"dfns")
    
    # find the relevant dfn data that includes the wall boundaries
    for iblock in 0:nblocks-1
        fid_pdfs = open_readonly_output_file(run_name,"dfns",iblock=iblock, printout=false)
        z_irank, r_irank = load_rank_data(fid_pdfs,printout=false)
        if (z_irank == 0 || z_irank == z_nrank-1) && r_irank == 0
            # plot data from lower wall boundary near z = -L/2
            # load local grid data
            nz_local, _, z_local, z_wgts_local, Lz = load_coordinate_data(fid_pdfs, "z", printout=false)
            nr_local, _, r_local, r_wgts_local, Lr = load_coordinate_data(fid_pdfs, "r", printout=false)
            
            if z_irank == 0 
                iz_wall = 1
                #print("z_local[iz_wall-]: ",z_local[iz_wall])
                zlabel = "wall-"
            elseif z_irank == z_nrank-1
                iz_wall = nz_local
                #print("z_local[iz_wall+]: ",z_local[iz_wall])
                zlabel = "wall+"
            end
            # load local pdf data 
            pdf = load_pdf_data(fid_pdfs, printout=false)
            for is in 1:n_ion_species
                description = "_ion_spec"*string(is)*"_"
                
                # plot f(vpa,ivperp0,iz_wall,ir0,is,itime) at the wall
                @views plot(vpa, pdf[:,ivperp0,iz_wall,ir0,is,itime0], xlabel=L"v_{\|\|}/L_{v_{\|\|}}", ylabel=L"f_i")
                outfile = string(run_name, "_pdf(vpa,vperp0,iz_"*zlabel*",ir0)"*description*"_vs_vpa.pdf")
                savefig(outfile) 
                
                # plot f(vpa,vperp,iz_wall,ir0,is,itime) at the wall                
                @views heatmap(vperp, vpa, pdf[:,:,iz_wall,ir0,is,itime0], xlabel=L"vperp", ylabel=L"vpa", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(run_name, "_pdf(vpa,vperp,iz_"*zlabel*",ir0)"*description*"_vs_vperp_vpa.pdf")
                savefig(outfile)
                
                # plot f(vpa,ivperp0,z,ir0,is,itime) near the wall                
                @views heatmap(z_local, vpa, pdf[:,ivperp0,:,ir0,is,itime0], xlabel=L"z", ylabel=L"vpa", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(run_name, "_pdf(vpa,ivperp0,z_"*zlabel*",ir0)"*description*"_vs_z_vpa.pdf")
                savefig(outfile)

                # plot f(ivpa0,ivperp0,z,r,is,itime) near the wall                  
                if nr_local > 1
                    @views heatmap(r_local, z_local, pdf[ivpa0,ivperp0,:,:,is,itime0], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                    outfile = string(run_name, "_pdf(ivpa0,ivperp0,z_"*zlabel*",r)"*description*"_vs_r_z.pdf")
                    savefig(outfile)
                end
            end
        end
        close(fid_pdfs)
    end
    println("done.")
end

end
