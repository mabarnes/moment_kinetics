"""
"""
module post_processing

export analyze_and_plot_data
export compare_charged_pdf_symbolic_test
export compare_moments_symbolic_test
export compare_neutral_pdf_symbolic_test
export compare_fields_symbolic_test
export construct_global_zr_coords
export allocate_global_zr_charged_moments
export allocate_global_zr_neutral_moments
export allocate_global_zr_fields
export get_coords_nelement

# Next three lines only used for workaround needed by plot_unnormalised()
using PyCall
import PyPlot

# packages
using Plots
using IJulia
using LsqFit
using NCDatasets
using Statistics: mean
using SpecialFunctions: erfi
using LaTeXStrings
using Measures
using MPI
# modules
using ..post_processing_input: pp
using ..quadrature: composite_simpson_weights
using ..array_allocation: allocate_float
using ..coordinates: coordinate, define_coordinate
using ..file_io: open_ascii_output_file
using ..type_definitions: mk_float, mk_int
using ..initial_conditions: vpagrid_to_dzdt
using ..load_data: open_readonly_output_file, get_group, load_input, load_time_data
using ..load_data: get_nranks
using ..load_data: load_fields_data, load_pdf_data
using ..load_data: load_charged_particle_moments_data, load_neutral_particle_moments_data
using ..load_data: load_neutral_pdf_data
using ..load_data: load_variable
using ..load_data: load_coordinate_data, load_block_data, load_rank_data,
                   load_species_data, load_mk_options
using ..analysis: analyze_fields_data, analyze_moments_data, analyze_pdf_data,
                  check_Chodura_condition, analyze_2D_instability
using ..velocity_moments: integrate_over_vspace
using ..manufactured_solns: manufactured_solutions, manufactured_electric_fields
using ..moment_kinetics_input: mk_input, get
using ..input_structs: geometry_input, grid_input, species_composition
using ..input_structs: electron_physics_type, boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath
using TOML
import Base: get

const default_compare_prefix = "comparison_plots/compare"

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
Call savefig, but catch the exception if there is an error
"""
function trysavefig(outfile)
    try
        savefig(outfile)
    catch
        println("Failed to make plot $outfile")
    end
    return nothing
end

"""
Call gif, but catch the exception if there is an error
"""
function trygif(anim, outfile; kwargs...)
    try
        gif(anim, outfile; kwargs...)
    catch
        println("Failed to make animation $outfile")
    end
    return nothing
end

"""
"""
function read_distributed_zr_data!(var::Array{mk_float,3}, var_name::String,
   run_name::String, file_key::String, nblocks::mk_int,
   nz_local::mk_int,nr_local::mk_int,iskip::mk_int)
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
                var[iz_global,ir_global,:] .= var_local[iz_local,ir_local,begin:iskip:end]
            end
        end
        close(fid)
    end
end

function read_distributed_zr_data!(var::Array{mk_float,4}, var_name::String,
   run_name::String, file_key::String, nblocks::mk_int,
   nz_local::mk_int,nr_local::mk_int,iskip::mk_int)
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
                var[iz_global,ir_global,:,:] .= var_local[iz_local,ir_local,:,begin:iskip:end]
            end
        end
        close(fid)
    end
end

function load_distributed_charged_pdf_slice(run_name, nblocks::mk_int, t_range,
                                            n_species::mk_int, r::coordinate,
                                            z::coordinate, vperp::coordinate,
                                            vpa::coordinate; is=nothing, ir=nothing,
                                            iz=nothing, ivperp=nothing, ivpa=nothing)
    result_dims = mk_int[]
    if ivpa === nothing
        push!(result_dims, vpa.n_global)
    end
    if ivperp === nothing
        push!(result_dims, vperp.n_global)
    end
    if iz === nothing
        push!(result_dims, z.n_global)
    else
        push!(result_dims, 1)
    end
    if ir === nothing
        push!(result_dims, r.n_global)
    else
        push!(result_dims, 1)
    end
    if is === nothing
        push!(result_dims, n_species)
    else
        push!(result_dims, 1)
    end
    push!(result_dims, length(t_range))

    f_global = allocate_float(result_dims...)

    # dimension of pdf is [vpa,vperp,z,r,species,t]
    for iblock in 0:nblocks-1
        fid = open_readonly_output_file(run_name, "dfns", iblock=iblock, printout=false)

        z_irank, r_irank = load_rank_data(fid)

        # min index set to avoid double assignment of repeated points
        # 1 if irank = 0, 2 otherwise
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        local_r_range = imin_r:r.n
        local_z_range = imin_z:z.n
        global_r_range = iglobal_func(imin_r, r_irank, r.n):iglobal_func(r.n, r_irank, r.n)
        global_z_range = iglobal_func(imin_z, z_irank, z.n):iglobal_func(z.n, z_irank, z.n)

        if ir !== nothing && !(ir ∈ global_r_range)
            # No data for the slice on this rank
            continue
        end
        if iz !== nothing && !(iz ∈ global_z_range)
            # No data for the slice on this rank
            continue
        end

        f_local_slice = load_pdf_data(fid)
        f_global_slice = f_global

        # Note: use selectdim() and get the dimension from thisdim because the actual
        # number of dimensions in f_global_slice, f_local_slice is different depending
        # on which combination of ivpa, ivperp, iz, ir, and is was passed.
        thisdim = ndims(f_local_slice) - 5
        if ivpa !== nothing
            f_local_slice = selectdim(f_local_slice, thisdim, ivpa)
        end

        thisdim = ndims(f_local_slice) - 4
        if ivperp !== nothing
            f_local_slice = selectdim(f_local_slice, thisdim, ivperp)
        end

        thisdim = ndims(f_local_slice) - 3
        if iz === nothing
            f_global_slice = selectdim(f_global_slice, thisdim, global_z_range)
            f_local_slice = selectdim(f_local_slice, thisdim, local_z_range)
        else
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
            f_local_slice = selectdim(f_local_slice, thisdim,
                                      ilocal_func(iz, z_irank, z.n))
        end

        thisdim = ndims(f_local_slice) - 2
        if ir === nothing
            f_global_slice = selectdim(f_global_slice, thisdim, global_r_range)
            f_local_slice = selectdim(f_local_slice, thisdim, local_r_range)
        else
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
            f_local_slice = selectdim(f_local_slice, thisdim,
                                      ilocal_func(ir, r_irank, r.n))
        end

        thisdim = ndims(f_local_slice) - 1
        if is !== nothing
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
            f_local_slice = selectdim(f_local_slice, thisdim, is)
        end

        # Select time slice
        thisdim = ndims(f_local_slice)
        f_local_slice = selectdim(f_local_slice, thisdim, t_range)

        f_global_slice .= f_local_slice
        close(fid)
    end

    if iz !== nothing
        thisdim = ndims(f_global) - 3
        f_global = selectdim(f_global, thisdim, 1)
    end
    if ir !== nothing
        thisdim = ndims(f_global) - 2
        f_global = selectdim(f_global, thisdim, 1)
    end
    if is !== nothing
        thisdim = ndims(f_global) - 1
        f_global = selectdim(f_global, thisdim, 1)
    end

    return f_global
end

function load_distributed_neutral_pdf_slice(run_name, nblocks::mk_int, t_range,
                                            n_species::mk_int, r::coordinate,
                                            z::coordinate, vzeta::coordinate,
                                            vr::coordinate, vz::coordinate; isn=nothing,
                                            ir=nothing, iz=nothing, ivzeta=nothing,
                                            ivr=nothing, ivz=nothing)
    result_dims = mk_int[]
    if ivz === nothing
        push!(result_dims, vz.n_global)
    end
    if ivr === nothing
        push!(result_dims, vr.n_global)
    end
    if ivzeta === nothing
        push!(result_dims, vzeta.n_global)
    end
    if iz === nothing
        push!(result_dims, z.n_global)
    else
        push!(result_dims, 1)
    end
    if ir === nothing
        push!(result_dims, r.n_global)
    else
        push!(result_dims, 1)
    end
    if isn === nothing
        push!(result_dims, n_species)
    else
        push!(result_dims, 1)
    end
    push!(result_dims, length(t_range))

    f_global = allocate_float(result_dims...)

    # dimension of pdf is [vpa,vperp,z,r,species,t]
    for iblock in 0:nblocks-1
        fid = open_readonly_output_file(run_name, "dfns", iblock=iblock, printout=false)

        z_irank, r_irank = load_rank_data(fid)

        # min index set to avoid double assignment of repeated points
        # 1 if irank = 0, 2 otherwise
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        local_r_range = imin_r:r.n
        local_z_range = imin_z:z.n
        global_r_range = iglobal_func(imin_r, r_irank, r.n):iglobal_func(r.n, r_irank, r.n)
        global_z_range = iglobal_func(imin_z, z_irank, z.n):iglobal_func(z.n, z_irank, z.n)

        if ir !== nothing && !(ir ∈ global_r_range)
            # No data for the slice on this rank
            continue
        end
        if iz !== nothing && !(iz ∈ global_z_range)
            # No data for the slice on this rank
            continue
        end

        f_local_slice = load_neutral_pdf_data(fid)
        f_global_slice = f_global

        # Note: use selectdim() and get the dimension from thisdim because the actual
        # number of dimensions in f_global_slice, f_local_slice is different depending
        # on which combination of ivpa, ivperp, iz, ir, and is was passed.
        thisdim = ndims(f_local_slice) - 6
        if ivz !== nothing
            f_local_slice = selectdim(f_local_slice, thisdim, ivz)
        end

        thisdim = ndims(f_local_slice) - 5
        if ivr !== nothing
            f_local_slice = selectdim(f_local_slice, thisdim, ivr)
        end

        thisdim = ndims(f_local_slice) - 4
        if ivzeta !== nothing
            f_local_slice = selectdim(f_local_slice, thisdim, ivzeta)
        end

        thisdim = ndims(f_local_slice) - 3
        if iz === nothing
            f_global_slice = selectdim(f_global_slice, thisdim, global_z_range)
            f_local_slice = selectdim(f_local_slice, thisdim, local_z_range)
        else
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
            f_local_slice = selectdim(f_local_slice, thisdim,
                                      ilocal_func(iz, z_irank, z.n))
        end

        thisdim = ndims(f_local_slice) - 2
        if ir === nothing
            f_global_slice = selectdim(f_global_slice, thisdim, global_r_range)
            f_local_slice = selectdim(f_local_slice, thisdim, local_r_range)
        else
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
            f_local_slice = selectdim(f_local_slice, thisdim,
                                      ilocal_func(ir, r_irank, r.n))
        end

        thisdim = ndims(f_local_slice) - 1
        if isn !== nothing
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
            f_local_slice = selectdim(f_local_slice, thisdim, isn)
        end

        # Select time slice
        thisdim = ndims(f_local_slice)
        f_local_slice = selectdim(f_local_slice, thisdim, t_range)

        f_global_slice .= f_local_slice
        close(fid)
    end

    if iz !== nothing
        thisdim = ndims(f_global) - 3
        f_global = selectdim(f_global, thisdim, 1)
    end
    if ir !== nothing
        thisdim = ndims(f_global) - 2
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isn !== nothing
        thisdim = ndims(f_global) - 1
        f_global = selectdim(f_global, thisdim, 1)
    end

    return f_global
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

function ilocal_func(iglobal,irank,nlocal)
    return iglobal - irank*(nlocal - 1)
end

function construct_global_zr_coords(r_local, z_local)

    function make_global_input(coord_local)
        return grid_input(coord_local.name, coord_local.ngrid,
            coord_local.nelement_global, coord_local.nelement_global, 1, 0, coord_local.L,
            coord_local.discretization, coord_local.fd_option, coord_local.bc,
            coord_local.advection, MPI.COMM_NULL)
    end

    r_global, r_global_spectral = define_coordinate(make_global_input(r_local))
    z_global, z_global_spectral = define_coordinate(make_global_input(z_local))

    return r_global, r_global_spectral, z_global, z_global_spectral
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

function allocate_global_zr_charged_dfns(nvpa_global, nvperp_global, nz_global, nr_global,
                                         n_ion_species, ntime)
    f = allocate_float(nvpa_global, nvperp_global, nz_global, nr_global, n_ion_species,
                       ntime)
    return f
end

function allocate_global_zr_neutral_dfns(nvz_global, nvr_global, nvzeta_global, nz_global, nr_global,
                                         n_ion_species, ntime)
    f = allocate_float(nvz_global, nvr_global, nvzeta_global, nz_global, nr_global,
                       n_ion_species, ntime)
    return f
end

function allocate_global_zr_neutral_moments(nz_global,nr_global,n_neutral_species,ntime)
    neutral_density = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_uz = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_pz = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_qz = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    neutral_thermal_speed = allocate_float(nz_global,nr_global,n_neutral_species,ntime)
    return neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed
end

function get_coords_nelement(scan_input)
    # use 1 as default because these values should be set in input .toml
    z_nelement = get(scan_input, "z_nelement", 1)
    r_nelement = get(scan_input, "r_nelement", 1)
    vpa_nelement = get(scan_input, "vpa_nelement", 1)
    vperp_nelement = get(scan_input, "vperp_nelement", 1)
    vz_nelement = get(scan_input, "vz_nelement", 1)
    vr_nelement = get(scan_input, "vr_nelement", 1)
    vzeta_nelement = get(scan_input, "vzeta_nelement", 1)
    return z_nelement, r_nelement, vpa_nelement, vperp_nelement, vz_nelement, vr_nelement, vzeta_nelement
end

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
    # bool to control if dfni is a function of vpa or vpabar in MMS test
    use_vpabar_in_mms_dfni = get(scan_input, "use_vpabar_in_mms_dfni", true)
    if use_vpabar_in_mms_dfni
        alpha_switch = 1.0
    else
        alpha_switch = 0.0
    end
    # ratio of the neutral particle mass to the ion particle mass
    mn_over_mi = 1.0
    # ratio of the electron particle mass to the ion particle mass
    me_over_mi = 1.0/1836.0
    composition = species_composition(n_species, n_ion_species, n_neutral_species,
        electron_physics, use_test_neutral_wall_pdf, T_e, T_wall, phi_wall, Er_constant,
        epsilon_offset, use_vpabar_in_mms_dfni, alpha_switch, mn_over_mi, me_over_mi,
        allocate_float(n_species))
    return geometry, composition

end

"""
    get_tuple_of_return_values(func, arg_tuples...)

Suppose `func(args...)` returns a tuple of return values, then
`get_tuple_of_return_values(func, arg_tuples...)` returns a tuple (with an entry for each
return value of `func`) of tuples (one for each argument in each of `arg_tuples...`) of
return values.
"""
function get_tuple_of_return_values(func, arg_tuples...)

    if isempty(arg_tuples)
        return ()
    end
    n_args_tuple = Tuple(length(a) for a ∈ arg_tuples if isa(a, Tuple))
    if length(n_args_tuple) == 0
        error("At least one of `arg_tuples` must actually be a tuple")
    end
    if !all(n==n_args_tuple[1] for n ∈ n_args_tuple)
        error("All argument tuples passed to `get_tuple_of_return_values()` must have "
              * "the same length")
    end
    n_args = n_args_tuple[1]

    # Convert any non-tuple arguments (i.e. single values) to tuples
    arg_tuples = Tuple(isa(a, Tuple) ? a : Tuple(a for _ ∈ 1:n_args) for a ∈ arg_tuples)

    collected_args = Tuple(Tuple(a[i] for a ∈ arg_tuples) for i ∈ 1:n_args)

    wrong_way_tuple = Tuple(func(args...) for args ∈ collected_args)

    if isa(wrong_way_tuple[1], Tuple)
        n_return_values = length(wrong_way_tuple[1])
        return Tuple(Tuple(wrong_way_tuple[i][j] for i ∈ 1:n_args)
                     for j ∈ 1:n_return_values)
    else
        # Return values from func are not a tuple
        return wrong_way_tuple
    end
end

"""
    analyze_and_plot_data(prefix...)

Make some plots for the simulation at `prefix`. If more than one argument is passed to
`prefix`, plot all the simulations together.

The strings passed to `prefix` should be either a directory (which contains run output) or
the prefix of output files, like `<directory>/<prefix>` where the output files are
`<directory>/<prefix>.moments.h5` and `<directory>/<prefix>.dfns.h5`.

If a single value is passed for `prefix` the plots/movies are created in the same
directory as the run, and given names based on the name of the run. If multiple values
are passed, the plots/movies are given names beginning with `compare_` and are created
in the `comparison_plots/` subdirectory.
"""
function analyze_and_plot_data(prefix...)
    # Create run_names from the paths to the run directory
    run_names = Vector{String}(undef,0)
    for p ∈ prefix
        p = realpath(p)
        if isdir(p)
            push!(run_names, joinpath(p, basename(p)))
        else
            push!(run_names, p)
        end
    end
    run_names = tuple(run_names...)

    # open the output file and give it the handle 'fid'
    moments_files0 = get_tuple_of_return_values(open_readonly_output_file, run_names,
                                                "moments")
    # load block data on iblock=0
    nblocks, iblock = get_tuple_of_return_values(load_block_data, moments_files0)

    # load input used for the run(s)
    scan_input = get_tuple_of_return_values(load_input, moments_files0)

    # load global and local sizes of grids stored on each output file
    # z z_wgts r r_wgts may take different values on different blocks
    # we need to construct the global grid below
    z, z_spectral = get_tuple_of_return_values(load_coordinate_data, moments_files0, "z")
    r, r_spectral = get_tuple_of_return_values(load_coordinate_data, moments_files0, "r")
    # load time data
    ntime, time = get_tuple_of_return_values(load_time_data, moments_files0)
    # load species data
    n_ion_species, n_neutral_species =
        get_tuple_of_return_values(load_species_data, moments_files0)
    evolve_density, evolve_upar, evolve_ppar =
        get_tuple_of_return_values(load_mk_options, moments_files0)

    for f in moments_files0
        close(f)
    end

    iskip = pp.itime_skip
    ntime = Tuple((nt + iskip - 1) ÷ iskip for nt ∈ ntime)
    time = Tuple(t[begin:iskip:end] for t ∈ time)

    # allocate arrays to contain the global fields as a function of (z,r,t)
    phi, Ez, Er = get_tuple_of_return_values(allocate_global_zr_fields,
                                             Tuple(this_z.n_global for this_z ∈ z),
                                             Tuple(this_r.n_global for this_r ∈ r),
                                             ntime)
    density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed =
        get_tuple_of_return_values(allocate_global_zr_charged_moments,
                                   Tuple(this_z.n_global for this_z ∈ z),
                                   Tuple(this_r.n_global for this_r ∈ r),
                                   n_ion_species, ntime)
    if any(n_neutral_species .> 0)
        neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed =
            get_tuple_of_return_values(allocate_global_zr_neutral_moments,
                                       Tuple(this_z.n_global for this_z ∈ z),
                                       Tuple(this_r.n_global for this_r ∈ r),
                                       n_neutral_species, ntime)
    end
    # read in the data from different block files
    # grids
    r_global, r_global_spectral, z_global, z_global_spectral =
        get_tuple_of_return_values(construct_global_zr_coords, r, z)

    # fields
    get_tuple_of_return_values(read_distributed_zr_data!, phi, "phi", run_names, "moments",
                               nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    get_tuple_of_return_values(read_distributed_zr_data!, Ez, "Ez", run_names, "moments",
                               nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    get_tuple_of_return_values(read_distributed_zr_data!, Er, "Er", run_names, "moments",
                               nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    # charged particle moments
    get_tuple_of_return_values(read_distributed_zr_data!, density, "density", run_names,
                               "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_flow, "parallel_flow",
                               run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_pressure,
                               "parallel_pressure", run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_heat_flux,
                               "parallel_heat_flux", run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    get_tuple_of_return_values(read_distributed_zr_data!, thermal_speed, "thermal_speed",
                               run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip)
    # neutral particle moments
    if any(n_neutral_species .> 0)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_density,
                                   "density_neutral", run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_uz, "uz_neutral",
                                   run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_pz, "pz_neutral",
                                   run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_qz, "qz_neutral",
                                   run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_thermal_speed,
                                   "thermal_speed_neutral", run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip)
    end

    # load time data from `dfns' cdf
    dfns_files0 = get_tuple_of_return_values(open_readonly_output_file, run_names, "dfns")
    # note that ntime may differ in these output files

    iskip_pdfs = pp.itime_skip_pdfs
    ntime_pdfs, time_pdfs = get_tuple_of_return_values(load_time_data, dfns_files0)
    ntime_pdfs = Tuple((nt + iskip_pdfs - 1) ÷ iskip_pdfs for nt ∈ ntime_pdfs)
    time_pdfs = Tuple(t[begin:iskip_pdfs:end] for t ∈ time_pdfs)

    # load local velocity coordinate data from `dfns' cdf
    # these values are currently the same for all blocks
    vpa, vpa_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vpa")
    vperp, vperp_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vperp")
    if any(n_neutral_species .> 0)
        vzeta, vzeta_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vzeta")
        vr, vr_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vr")
        vz, vz_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vz")
    else # define nvz nvr nvzeta to avoid errors below
        vzeta = vzeta_spectral = nothing
        vr = vr_spectral = nothing
        vz = vz_spectral = nothing
    end

    phi_at_pdf_times, Ez_at_pdf_times, Er_at_pdf_times =
        get_tuple_of_return_values(allocate_global_zr_fields,
                                   Tuple(this_z.n_global for this_z ∈ z),
                                   Tuple(this_r.n_global for this_r ∈ r), ntime_pdfs)
    density_at_pdf_times, parallel_flow_at_pdf_times, parallel_pressure_at_pdf_times,
    parallel_heat_flux_at_pdf_times, thermal_speed_at_pdf_times =
        get_tuple_of_return_values(allocate_global_zr_charged_moments,
                                   Tuple(this_z.n_global for this_z ∈ z),
                                   Tuple(this_r.n_global for this_r ∈ r), n_ion_species,
                                   ntime_pdfs)
    if any(n_neutral_species .> 0)
        neutral_density_at_pdf_times, neutral_uz_at_pdf_times, neutral_pz_at_pdf_times,
        neutral_qz_at_pdf_times, neutral_thermal_speed_at_pdf_times =
            get_tuple_of_return_values(allocate_global_zr_neutral_moments,
                                       Tuple(this_z.n_global for this_z ∈ z),
                                       Tuple(this_r.n_global for this_r ∈ r),
                                       n_neutral_species, ntime_pdfs)
    end
    # fields
    get_tuple_of_return_values(read_distributed_zr_data!, phi_at_pdf_times, "phi",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    get_tuple_of_return_values(read_distributed_zr_data!, Ez_at_pdf_times, "Ez",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    get_tuple_of_return_values(read_distributed_zr_data!, Er_at_pdf_times, "Er",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    # charged particle moments
    get_tuple_of_return_values(read_distributed_zr_data!, density_at_pdf_times, "density",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_flow_at_pdf_times,
                               "parallel_flow", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_pressure_at_pdf_times,
                               "parallel_pressure", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_heat_flux_at_pdf_times,
                               "parallel_heat_flux", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    get_tuple_of_return_values(read_distributed_zr_data!, thermal_speed_at_pdf_times,
                               "thermal_speed", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    # neutral particle moments
    if any(n_neutral_species .> 0)
        get_tuple_of_return_values(read_distributed_zr_data!,
                                   neutral_density_at_pdf_times, "density_neutral",
                                   run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_uz_at_pdf_times,
                                   "uz_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_pz_at_pdf_times,
                                   "pz_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_qz_at_pdf_times,
                                   "qz_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
        get_tuple_of_return_values(read_distributed_zr_data!,
                                   neutral_thermal_speed_at_pdf_times,
                                   "thermal_speed_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r), iskip_pdfs)
    end

    for f in dfns_files0
        close(f)
    end

    geometry, composition =
        get_tuple_of_return_values(get_geometry_and_composition, scan_input,
                                   n_ion_species, n_neutral_species)

    # initialise the post-processing input options
    nwrite_movie, itime_min, itime_max, nwrite_movie_pdfs, itime_min_pdfs, itime_max_pdfs,
    ivpa0, ivperp0, iz0, ir0, ivz0, ivr0, ivzeta0 =
        init_postprocessing_options(pp, minimum(this_vpa.n_global for this_vpa ∈ vpa),
            minimum(this_vperp.n_global for this_vperp ∈ vperp),
            minimum(this_z.n_global for this_z ∈ z),
            minimum(this_r.n_global for this_r ∈ r),
            vz === nothing ? 0 : minimum(this_vz.n_global for this_vz ∈ vz),
            vr === nothing ? 0 : minimum(this_vr.n_global for this_vr ∈ vr),
            vzeta === nothing ? 0 : minimum(this_vzeta.n_global for this_vzeta ∈ vzeta),
            minimum(ntime), minimum(ntime_pdfs))

    is_1D1V = all([(this_r.n == 1 for this_r ∈ r_global)...,
                   (this_vperp.n == 1 for this_vperp ∈ vperp)...,
                   (vzeta === nothing ? true : (this_vzeta.n == 1 for this_vzeta ∈ vzeta))...,
                   (vr === nothing ? true : (this_vr.n == 1 for this_vr ∈ vr))...])
    if is_1D1V
        # load full (vpa,z,r,species,t) particle distribution function (pdf) data
        ff = get_tuple_of_return_values(load_distributed_charged_pdf_slice, run_names,
                                        nblocks, itime_min_pdfs:iskip_pdfs:itime_max_pdfs,
                                        n_ion_species, r, z, vperp, vpa)
        if maximum(n_neutral_species) > 0
            neutral_ff = get_tuple_of_return_values(load_distributed_neutral_pdf_slice,
                                                    run_names, nblocks,
                                                    itime_min_pdfs:iskip_pdfs:itime_max_pdfs,
                                                    n_neutral_species, r, z, vzeta, vr,
                                                    vz)
        else
            neutral_ff = nothing
        end

        #evaluate 1D-1V diagnostics at fixed ir0
        plot_1D_1V_diagnostics(run_names, nwrite_movie, itime_min, itime_max,
            nwrite_movie_pdfs, itime_min_pdfs, itime_max_pdfs, ivpa0, iz0,
            ir0, r_global,
            Tuple(p[:,ir0,:] for p ∈ phi),
            Tuple(n[:,ir0,:,:] for n ∈ density),
            Tuple(upar[:,ir0,:,:] for upar ∈ parallel_flow),
            Tuple(ppar[:,ir0,:,:] for ppar ∈ parallel_pressure),
            Tuple(qpar[:,ir0,:,:] for qpar ∈ parallel_heat_flux),
            Tuple(vth[:,ir0,:,:] for vth ∈ thermal_speed),
            Tuple(p[:,ir0,:] for p ∈ phi_at_pdf_times),
            Tuple(n[:,ir0,:,:] for n ∈ density_at_pdf_times),
            Tuple(upar[:,ir0,:,:] for upar ∈ parallel_flow_at_pdf_times),
            Tuple(ppar[:,ir0,:,:] for ppar ∈ parallel_pressure_at_pdf_times),
            Tuple(qpar[:,ir0,:,:] for qpar ∈ parallel_heat_flux_at_pdf_times),
            Tuple(vth[:,ir0,:,:] for vth ∈ thermal_speed_at_pdf_times),
            Tuple(f[:,ivperp0,:,ir0,:,:] for f ∈ ff),
            Tuple(neutral_n[:,ir0,:,:] for neutral_n ∈ neutral_density),
            Tuple(uz[:,ir0,:,:] for uz ∈ neutral_uz),
            Tuple(pz[:,ir0,:,:] for pz ∈ neutral_pz),
            Tuple(qz[:,ir0,:,:] for qz ∈ neutral_qz),
            Tuple(neutral_vth[:,ir0,:,:] for neutral_vth ∈ neutral_thermal_speed),
            Tuple(neutral_n[:,ir0,:,:] for neutral_n ∈ neutral_density_at_pdf_times),
            Tuple(uz[:,ir0,:,:] for uz ∈ neutral_uz_at_pdf_times),
            Tuple(pz[:,ir0,:,:] for pz ∈ neutral_pz_at_pdf_times),
            Tuple(qz[:,ir0,:,:] for qz ∈ neutral_qz_at_pdf_times),
            Tuple(neutral_vth[:,ir0,:,:] for neutral_vth ∈ neutral_thermal_speed_at_pdf_times),
            Tuple(neutral_f[:,ivr0,ivzeta0,:,ir0,:,:] for neutral_f ∈ neutral_ff),
            n_ion_species, n_neutral_species, evolve_density, evolve_upar, evolve_ppar,
            vz, vpa, z_global, ntime, time, ntime_pdfs, time_pdfs)
    end

    if !is_1D1V
        # analyze the fields data
        phi_iz0 = Tuple(p[iz0,:,:] for p ∈ phi)
        phi_fldline_avg, delta_phi =
            get_tuple_of_return_values(analyze_fields_data, phi_iz0, ntime, r_global)
        get_tuple_of_return_values(plot_fields_rt, phi_iz0, delta_phi, time, itime_min,
                                   itime_max, nwrite_movie, r_global, ir0, run_names, delta_phi,
                                   pp)
    end

    if pp.diagnostics_chodura
        n_runs = length(run_names)
        if n_runs == 1
            prefix = run_names[1]
            legend = false
        else
            prefix = default_compare_prefix
            legend = true
        end
        Chodura_ratio_lower, Chodura_ratio_upper =
            get_tuple_of_return_values(check_Chodura_condition, run_names, vpa, vperp,
                                       density_at_pdf_times, composition.T_e,
                                       Er_at_pdf_times, geometry, "wall", nblocks)

        plot(legend=legend)
        for (t, cr, run_label) ∈ zip(time_pdfs, Chodura_ratio_lower, run_names)
            plot!(t, cr[ir0,:], xlabel="time", ylabel="Chodura ratio at z=-L/2",
                  label=run_label)
        end
        outfile = string(prefix, "_Chodura_ratio_lower.pdf")
        trysavefig(outfile)
        plot(legend=legend)
        for (t, cr, run_label) ∈ zip(time_pdfs, Chodura_ratio_upper, run_names)
            plot!(t, cr[ir0,:], xlabel="time", ylabel="Chodura ratio at z=+L/2",
                  label=run_label)
        end
        outfile = string(prefix, "_Chodura_ratio_upper.pdf")
        trysavefig(outfile)
    end

    # For now, don't support multi-run comparison in remaining 2D and MMS diagnostics
    scan_input = scan_input[1]
    density = density[1]
    parallel_flow = parallel_flow[1]
    parallel_pressure = parallel_pressure[1]
    parallel_heat_flux = parallel_heat_flux[1]
    thermal_speed = thermal_speed[1]
    time = time[1]
    ntime = ntime[1]
    time_pdfs = time_pdfs[1]
    ntime_pdfs = ntime_pdfs[1]
    nblocks = nblocks[1]
    z = z[1]
    r = r[1]
    z_global = z_global[1]
    r_global = r_global[1]
    z_global_spectral = z_global_spectral[1]
    r_global_spectral = r_global_spectral[1]
    n_ion_species = n_ion_species[1]
    n_neutral_species = n_neutral_species[1]
    run_name = run_names[1]
    phi = phi[1]
    Ez = Ez[1]
    Er = Er[1]
    vpa = vpa[1]
    vperp = vperp[1]
    geometry = geometry[1]
    composition = composition[1]
    if n_neutral_species > 0
        neutral_density = neutral_density[1]
        vz = vz[1]
        vr = vr[1]
        vzeta = vzeta[1]
    end

    if !is_1D1V
        # make plots and animations of the phi, Ez and Er
        plot_charged_moments_2D(density, parallel_flow, parallel_pressure, time,
                                z_global.grid, r_global.grid, iz0, ir0, n_ion_species,
                                itime_min, itime_max, nwrite_movie, run_name, pp)
        # make plots and animations of the phi, Ez and Er
        plot_fields_2D(phi, Ez, Er, time, z_global.grid, r_global.grid, iz0, ir0, itime_min,
                       itime_max, nwrite_movie, run_name, pp, "")

        # load full (vpa,z,r,species,t) particle distribution function (pdf) data
        spec_type = "ion"
        plot_charged_pdf(run_name, vpa, vperp, z_global, r_global, z, r, ivpa0, ivperp0, iz0,
                         ir0, spec_type, n_ion_species, ntime_pdfs, nblocks, itime_min_pdfs,
                         itime_max_pdfs, iskip_pdfs, nwrite_movie_pdfs, pp)
        # make plots and animations of the neutral pdf
        if n_neutral_species > 0
            spec_type = "neutral"
            plot_neutral_pdf(run_name, vz, vr, vzeta, z_global, r_global, z, r, ivz0, ivr0,
                             ivzeta0, iz0, ir0, spec_type, n_neutral_species, ntime_pdfs,
                             nblocks, itime_min_pdfs, iskip_pdfs, itime_max_pdfs,
                             nwrite_movie_pdfs, pp)
        end
        # plot ion pdf data near the wall boundary
        if pp.plot_wall_pdf
            plot_charged_pdf_2D_at_wall(run_name)
        end
    end

    if pp.instability2D
        phi_perturbation, density_perturbation, temperature_perturbation, phi_Fourier,
        density_Fourier, temperature_Fourier =
            analyze_2D_instability(phi, density, thermal_speed, r_global, z_global,
                                   r_global_spectral, z_global_spectral)

        n_kz, n_kr, nt = size(phi_Fourier)

        cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
        logdeep = cgrad(:deep, scale=:log) |> cmlog
        function plot_Fourier_2D(var, symbol, name)
            plot(title="$symbol Fourier components", xlabel="time", ylabel="amplitude",
                 legend=false, yscale=:log)
            for ikr ∈ 1:n_kr, ikz ∈ 1:n_kz
                ikr!=2 && continue
                data = abs.(var[ikz,ikr,:])
                data[data.==0.0] .= NaN
                plot!(time, data, annotations=(time[end], data[end], "ikr=$ikr, ikz=$ikz"),
                      annotationhalign=:right, annotationfontsize=6)
            end
            outfile = string(run_name, "_$(name)_Fourier_components.pdf")
            trysavefig(outfile)

            # make a gif animation of Fourier components
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(log.(abs.(var[:,:,i])), xlabel="kr", ylabel="kz",
                               title=symbol, fillcolor = logdeep)
            end
            outfile = string(run_name, "_$(name)_Fourier.gif")
            trygif(anim, outfile, fps=5)
        end
        plot_Fourier_2D(phi_Fourier, "ϕ", "phi")
        plot_Fourier_2D(density_Fourier, "n", "density")
        plot_Fourier_2D(temperature_Fourier, "T", "temperature")

        function animate_perturbation(var, name)
            # make a gif animation of perturbation
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r_global.grid, z_global.grid, var[:,:,i], xlabel="r",
                               ylabel="z", fillcolor = :deep)
            end
            outfile = string(run_name, "_$name_perturbation.gif")
            trygif(anim, outfile, fps=5)
        end
        animate_perturbation(phi_perturbation, "phi")
        animate_perturbation(density_perturbation, "density")
        animate_perturbation(temperature_perturbation, "temperature")

    end

    manufactured_solns_section = get(scan_input, "manufactured_solns", Dict{String,Any}())
    use_manufactured_solns_for_advance = get(manufactured_solns_section, "use_for_advance", false)
    use_manufactured_solns_for_init = get(manufactured_solns_section, "use_for_init", false)
    manufactured_solns_test = use_manufactured_solns_for_advance && use_manufactured_solns_for_init
    # Plots compare density and density_symbolic at last timestep
    #if(manufactured_solns_test && nr > 1)
    if(manufactured_solns_test)
        r_bc = get(scan_input, "r_bc", "periodic")
        z_bc = get(scan_input, "z_bc", "periodic")
        # avoid passing Lr = 0 into manufactured_solns functions
        if r_global.n > 1
            Lr_in = r_global.L
        else
            Lr_in = 1.0
        end

        manufactured_solns_list = manufactured_solutions(Lr_in,z_global.L,r_bc,z_bc,geometry,composition,r_global.n)
        dfni_func = manufactured_solns_list.dfni_func
        densi_func = manufactured_solns_list.densi_func
        dfnn_func = manufactured_solns_list.dfnn_func
        densn_func = manufactured_solns_list.densn_func
        manufactured_E_fields = manufactured_electric_fields(Lr_in,z_global.L,r_bc,z_bc,composition,r_global.n)
        Er_func = manufactured_E_fields.Er_func
        Ez_func = manufactured_E_fields.Ez_func
        phi_func = manufactured_E_fields.phi_func

        # phi, Er, Ez test
        phi_sym = copy(phi[:,:,:])
        Er_sym = copy(phi[:,:,:])
        Ez_sym = copy(phi[:,:,:])
        for it in 1:ntime
            for ir in 1:r_global.n
                for iz in 1:z_global.n
                    phi_sym[iz,ir,it] = phi_func(z_global.grid[iz],r_global.grid[ir],time[it])
                    Ez_sym[iz,ir,it] = Ez_func(z_global.grid[iz],r_global.grid[ir],time[it])
                    Er_sym[iz,ir,it] = Er_func(z_global.grid[iz],r_global.grid[ir],time[it])
                end
            end
        end
        # make plots and animations of the phi, Ez and Er
        #plot_fields_2D(phi_sym, Ez_sym, Er_sym, time, z_global.grid, r_global.grid, iz0, ir0,
        #    itime_min, itime_max, nwrite_movie, run_name, pp, "_sym")
        println("time/ (Lref/cref): ", time)
        compare_fields_symbolic_test(run_name,phi,phi_sym,z_global.grid,r_global.grid,time,z_global.n,r_global.n,ntime,
         L"\widetilde{\phi}",L"\widetilde{\phi}^{sym}",L"\varepsilon(\widetilde{\phi})","phi")
        compare_fields_symbolic_test(run_name,Er,Er_sym,z_global.grid,r_global.grid,time,z_global.n,r_global.n,ntime,
         L"\widetilde{E}_r",L"\widetilde{E}^{sym}_r",L"\varepsilon(\widetilde{E}_r)","Er")
        compare_fields_symbolic_test(run_name,Ez,Ez_sym,z_global.grid,r_global.grid,time,z_global.n,r_global.n,ntime,
         L"\widetilde{E}_z",L"\widetilde{E}^{sym}_z",L"\varepsilon(\widetilde{E}_z)","Ez")

        # ion test
        density_sym = copy(density[:,:,:,:])
        is = 1
        for it in 1:ntime
            for ir in 1:r_global.n
                for iz in 1:z_global.n
                    density_sym[iz,ir,is,it] = densi_func(z_global.grid[iz],r_global.grid[ir],time[it])
                end
            end
        end
        compare_moments_symbolic_test(run_name,density,density_sym,"ion",z_global.grid,r_global.grid,time,z_global.n,r_global.n,ntime,
         L"\widetilde{n}_i",L"\widetilde{n}_i^{sym}",L"\varepsilon(\widetilde{n}_i)","dens")

        compare_charged_pdf_symbolic_test(run_name,manufactured_solns_list,"ion",
          L"\widetilde{f}_i",L"\widetilde{f}^{sym}_i",L"\varepsilon(\widetilde{f}_i)","pdf")
        if n_neutral_species > 0
            # neutral test
            neutral_density_sym = copy(density[:,:,:,:])
            is = 1
            for it in 1:ntime
                for ir in 1:nr_global
                    for iz in 1:nz_global
                        neutral_density_sym[iz,ir,is,it] = densn_func(z_global.grid[iz],r_global.grid[ir],time[it])
                    end
                end
            end
            compare_moments_symbolic_test(run_name,neutral_density,neutral_density_sym,"neutral",z_global.grid,r_global.grid,time,z_global.n,r_global.n,ntime,
             L"\widetilde{n}_n",L"\widetilde{n}_n^{sym}",L"\varepsilon(\widetilde{n}_n)","dens")

            compare_neutral_pdf_symbolic_test(run_name,manufactured_solns_list,"neutral",
             L"\widetilde{f}_n",L"\widetilde{f}^{sym}_n",L"\varepsilon(\widetilde{f}_n)","pdf")
        end
    end
end

"""
Find the maximum difference, as a function of time, between two or more outputs
for each variable.
"""
function calculate_differences(prefix...)
    # Create run_names from the paths to the run directory
    run_names = Vector{String}(undef,0)
    for p ∈ prefix
        p = realpath(p)
        if isdir(p)
            push!(run_names, joinpath(p, basename(p)))
        else
            push!(run_names, p)
        end
    end
    run_names = tuple(run_names...)

    # open the output file and give it the handle 'fid'
    moments_files0 = get_tuple_of_return_values(open_readonly_output_file, run_names,
                                                "moments")
    # load block data on iblock=0
    nblocks, iblock = get_tuple_of_return_values(load_block_data, moments_files0)

    # load input used for the run(s)
    scan_input = get_tuple_of_return_values(load_input, moments_files0)

    # load global and local sizes of grids stored on each output file
    # z z_wgts r r_wgts may take different values on different blocks
    # we need to construct the global grid below
    z, z_spectral = get_tuple_of_return_values(load_coordinate_data, moments_files0, "z")
    r, r_spectral = get_tuple_of_return_values(load_coordinate_data, moments_files0, "r")
    # load time data
    ntime, time = get_tuple_of_return_values(load_time_data, moments_files0)
    # load species data
    n_ion_species, n_neutral_species =
        get_tuple_of_return_values(load_species_data, moments_files0)
    evolve_density, evolve_upar, evolve_ppar =
        get_tuple_of_return_values(load_mk_options, moments_files0)

    for f in moments_files0
        close(f)
    end

    # allocate arrays to contain the global fields as a function of (z,r,t)
    phi, Ez, Er = get_tuple_of_return_values(allocate_global_zr_fields,
                                             Tuple(this_z.n_global for this_z ∈ z),
                                             Tuple(this_r.n_global for this_r ∈ r),
                                             ntime)
    density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed =
        get_tuple_of_return_values(allocate_global_zr_charged_moments,
                                   Tuple(this_z.n_global for this_z ∈ z),
                                   Tuple(this_r.n_global for this_r ∈ r),
                                   n_ion_species, ntime)
    if any(n_neutral_species .> 0)
        neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed =
            get_tuple_of_return_values(allocate_global_zr_neutral_moments,
                                       Tuple(this_z.n_global for this_z ∈ z),
                                       Tuple(this_r.n_global for this_r ∈ r),
                                       n_neutral_species, ntime)
    end
    # read in the data from different block files
    # grids
    r_global, r_global_spectral, z_global, z_global_spectral =
        get_tuple_of_return_values(construct_global_zr_coords, r, z)

    # fields
    get_tuple_of_return_values(read_distributed_zr_data!, phi, "phi", run_names, "moments",
                               nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, Ez, "Ez", run_names, "moments",
                               nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, Er, "Er", run_names, "moments",
                               nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    # charged particle moments
    get_tuple_of_return_values(read_distributed_zr_data!, density, "density", run_names,
                               "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_flow, "parallel_flow",
                               run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_pressure,
                               "parallel_pressure", run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_heat_flux,
                               "parallel_heat_flux", run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, thermal_speed, "thermal_speed",
                               run_names, "moments", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    # neutral particle moments
    if any(n_neutral_species .> 0)
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_density,
                                   "density_neutral", run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_uz, "uz_neutral",
                                   run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_pz, "pz_neutral",
                                   run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_qz, "qz_neutral",
                                   run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_thermal_speed,
                                   "thermal_speed_neutral", run_names, "moments", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
    end

    # load time data from `dfns' cdf
    dfns_files0 = get_tuple_of_return_values(open_readonly_output_file, run_names, "dfns")
    # note that ntime may differ in these output files

    ntime_pdfs, time_pdfs = get_tuple_of_return_values(load_time_data, dfns_files0)

    # load local velocity coordinate data from `dfns' cdf
    # these values are currently the same for all blocks
    vpa, vpa_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vpa")
    vperp, vperp_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vperp")
    if any(n_neutral_species .> 0)
        vzeta, vzeta_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vzeta")
        vr, vr_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vr")
        vz, vz_spectral = get_tuple_of_return_values(load_coordinate_data, dfns_files0, "vz")
    else # define nvz nvr nvzeta to avoid errors below
        vzeta = vzeta_spectral = nothing
        vr = vr_spectral = nothing
        vz = vz_spectral = nothing
    end

    for f in dfns_files0
        close(f)
    end

    phi_at_pdf_times, Ez_at_pdf_times, Er_at_pdf_times =
        get_tuple_of_return_values(allocate_global_zr_fields,
                                   Tuple(this_z.n_global for this_z ∈ z),
                                   Tuple(this_r.n_global for this_r ∈ r), ntime_pdfs)
    density_at_pdf_times, parallel_flow_at_pdf_times, parallel_pressure_at_pdf_times,
    parallel_heat_flux_at_pdf_times, thermal_speed_at_pdf_times =
        get_tuple_of_return_values(allocate_global_zr_charged_moments,
                                   Tuple(this_z.n_global for this_z ∈ z),
                                   Tuple(this_r.n_global for this_r ∈ r), n_ion_species,
                                   ntime_pdfs)
    if any(n_neutral_species .> 0)
        neutral_density_at_pdf_times, neutral_uz_at_pdf_times, neutral_pz_at_pdf_times,
        neutral_qz_at_pdf_times, neutral_thermal_speed_at_pdf_times =
            get_tuple_of_return_values(allocate_global_zr_neutral_moments,
                                       Tuple(this_z.n_global for this_z ∈ z),
                                       Tuple(this_r.n_global for this_r ∈ r),
                                       n_neutral_species, ntime_pdfs)
    end
    # fields
    get_tuple_of_return_values(read_distributed_zr_data!, phi_at_pdf_times, "phi",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, Ez_at_pdf_times, "Ez",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, Er_at_pdf_times, "Er",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    # charged particle moments
    get_tuple_of_return_values(read_distributed_zr_data!, density_at_pdf_times, "density",
                               run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_flow_at_pdf_times,
                               "parallel_flow", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_pressure_at_pdf_times,
                               "parallel_pressure", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, parallel_heat_flux_at_pdf_times,
                               "parallel_heat_flux", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    get_tuple_of_return_values(read_distributed_zr_data!, thermal_speed_at_pdf_times,
                               "thermal_speed", run_names, "dfns", nblocks,
                               Tuple(this_z.n for this_z ∈ z),
                               Tuple(this_r.n for this_r ∈ r))
    # neutral particle moments
    if any(n_neutral_species .> 0)
        get_tuple_of_return_values(read_distributed_zr_data!,
                                   neutral_density_at_pdf_times, "density_neutral",
                                   run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_uz_at_pdf_times,
                                   "uz_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_pz_at_pdf_times,
                                   "pz_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!, neutral_qz_at_pdf_times,
                                   "qz_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
        get_tuple_of_return_values(read_distributed_zr_data!,
                                   neutral_thermal_speed_at_pdf_times,
                                   "thermal_speed_neutral", run_names, "dfns", nblocks,
                                   Tuple(this_z.n for this_z ∈ z),
                                   Tuple(this_r.n for this_r ∈ r))
    end

    # load full (vpa,z,r,species,t) particle distribution function (pdf) data
    ff = get_tuple_of_return_values(load_distributed_charged_pdf_slice, run_names,
                                    nblocks, Tuple(1:nt for nt ∈ ntime_pdfs),
                                    n_ion_species, r, z, vperp, vpa)
    if maximum(n_neutral_species) > 0
        neutral_ff = get_tuple_of_return_values(load_distributed_neutral_pdf_slice,
                                                run_names, nblocks,
                                                Tuple(1:nt for nt ∈ ntime_pdfs),
                                                n_neutral_species, r, z, vzeta, vr, vz)
    else
        neutral_ff = nothing
    end

    function get_error_vs_run0(var, name)
        var0 = var[1]
        nd = ndims(var0)
        for (i,v) in enumerate(var[2:end])
            diff = abs.(v - var0)
            # Leave maxdiff as a function of time
            maxdiff = maximum(diff, dims=1:nd-1)

            println("$i $name $maxdiff")
        end
        return nothing
    end

    get_error_vs_run0(phi, "phi")
    get_error_vs_run0(Er, "Er")
    get_error_vs_run0(Ez, "Ez")
    get_error_vs_run0(density, "density")
    get_error_vs_run0(parallel_flow, "parallel_flow")
    get_error_vs_run0(parallel_pressure, "parallel_pressure")
    get_error_vs_run0(parallel_heat_flux, "parallel_heat_flux")
    get_error_vs_run0(thermal_speed, "thermal_speed")
    get_error_vs_run0(phi_at_pdf_times, "phi_at_pdf_times")
    get_error_vs_run0(Er_at_pdf_times, "Er_at_pdf_times")
    get_error_vs_run0(Ez_at_pdf_times, "Ez_at_pdf_times")
    get_error_vs_run0(density_at_pdf_times, "density_at_pdf_times")
    get_error_vs_run0(parallel_flow_at_pdf_times, "parallel_flow_at_pdf_times")
    get_error_vs_run0(parallel_pressure_at_pdf_times, "parallel_pressure_at_pdf_times")
    get_error_vs_run0(parallel_heat_flux_at_pdf_times, "parallel_heat_flux_at_pdf_times")
    get_error_vs_run0(thermal_speed_at_pdf_times, "thermal_speed_at_pdf_times")
    get_error_vs_run0(ff, "ff")

    if any(n_neutral_species .> 0)
        get_error_vs_run0(neutral_density, "neutral_density")
        get_error_vs_run0(neutral_uz, "neutral_uz")
        get_error_vs_run0(neutral_pz, "neutral_pz")
        get_error_vs_run0(neutral_qz, "neutral_qz")
        get_error_vs_run0(neutral_thermal_speed, "neutral_thermal_speed")
        get_error_vs_run0(neutral_density_at_pdf_times, "neutral_density_at_pdf_times")
        get_error_vs_run0(neutral_uz_at_pdf_times, "neutral_uz_at_pdf_times")
        get_error_vs_run0(neutral_pz_at_pdf_times, "neutral_pz_at_pdf_times")
        get_error_vs_run0(neutral_qz_at_pdf_times, "neutral_qz_at_pdf_times")
        get_error_vs_run0(neutral_thermal_speed_at_pdf_times, "neutral_thermal_speed_at_pdf_times")
        get_error_vs_run0(neutral_ff, "neutral_ff")
    end
end

"""
"""
function init_postprocessing_options(pp, nvpa, nvperp, nz, nr, nvz, nvr, nvzeta, ntime,
                                     ntime_pdfs)
    print("Initializing the post-processing input options...")
    # nwrite_movie is the stride used when making animations of moments
    nwrite_movie = pp.nwrite_movie
    # itime_min is the minimum time index at which to start animations of moments
    if pp.itime_min > 0 && pp.itime_min <= ntime
        itime_min = pp.itime_min
    else
        itime_min = 1
    end
    # itime_max is the final time index at which to end animations of moments
    # if itime_max < 0, the value used will be the total number of time slices
    if pp.itime_max > 0 && pp.itime_max <= ntime
        itime_max = pp.itime_max
    else
        itime_max = ntime
    end
    # nwrite_movie_pdfs is the stride used when making animations of pdfs
    nwrite_movie_pdfs = pp.nwrite_movie_pdfs
    # itime_min_pdfs is the minimum time index at which to start animations of pdfs
    if pp.itime_min_pdfs > 0 && pp.itime_min_pdfs <= ntime_pdfs
        itime_min_pdfs = pp.itime_min_pdfs
    else
        itime_min_pdfs = 1
    end
    # itime_max is the final time index at which to end animations of pdfs
    # if itime_max_pdfs < 0, the value used will be the total number of time slices
    if pp.itime_max_pdfs > 0 && pp.itime_max_pdfs <= ntime_pdfs
        itime_max_pdfs = pp.itime_max_pdfs
    else
        itime_max_pdfs = ntime_pdfs
    end
    # ir0 is the ir index used when plotting data at a single r location
    # by default, it will be set to cld(nr,3) unless a non-negative value provided
    if pp.ir0 > 0
        ir0 = pp.ir0
    else
        ir0 = max(cld(nr,3), 1)
    end
    # iz0 is the iz index used when plotting data at a single z location
    # by default, it will be set to cld(nz,3) unless a non-negative value provided
    if pp.iz0 > 0
        iz0 = pp.iz0
    else
        iz0 = max(cld(nz,3), 1)
    end
    # ivperp0 is the iz index used when plotting data at a single vperp location
    # by default, it will be set to cld(nvperp,3) unless a non-negative value provided
    if pp.ivperp0 > 0
        ivperp0 = pp.ivperp0
    else
        ivperp0 = max(cld(nvperp,3), 1)
    end
    # ivpa0 is the iz index used when plotting data at a single vpa location
    # by default, it will be set to cld(nvpa,3) unless a non-negative value provided
    if pp.ivpa0 > 0
        ivpa0 = pp.ivpa0
    else
        ivpa0 = max(cld(nvpa,3), 1)
    end
    # ivz0 is the ivr index used when plotting data at a single vz location
    # by default, it will be set to cld(nvz,3) unless a non-negative value provided
    if pp.ivz0 > 0
        ivz0 = pp.ivz0
    else
        ivz0 = max(cld(nvz,3), 1)
    end
    # ivr0 is the ivr index used when plotting data at a single vr location
    # by default, it will be set to cld(nvr,3) unless a non-negative value provided
    if pp.ivr0 > 0
        ivr0 = pp.ivr0
    else
        ivr0 = max(cld(nvr,3), 1)
    end
    # ivzeta0 is the ivzeta index used when plotting data at a single vzeta location
    # by default, it will be set to cld(nvr,3) unless a non-negative value provided
    if pp.ivzeta0 > 0
        ivzeta0 = pp.ivzeta0
    else
        ivzeta0 = max(cld(nvzeta,3), 1)
    end
    println("done.")
    return nwrite_movie, itime_min, itime_max, nwrite_movie_pdfs, itime_min_pdfs,
           itime_max_pdfs, ivpa0, ivperp0, iz0, ir0, ivz0, ivr0, ivzeta0
end

"""
"""
function plot_1D_1V_diagnostics(run_names, nwrite_movie, itime_min, itime_max,
        nwrite_movie_pdfs, itime_min_pdfs, itime_max_pdfs, ivpa0, iz0, ir0, r, phi,
        density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed,
        phi_at_pdf_times, density_at_pdf_times, parallel_flow_at_pdf_times,
        parallel_pressure_at_pdf_times, parallel_heat_flux_at_pdf_times,
        thermal_speed_at_pdf_times, ff, neutral_density, neutral_uz, neutral_pz,
        neutral_qz, neutral_thermal_speed, neutral_density_at_pdf_times,
        neutral_uz_at_pdf_times, neutral_pz_at_pdf_times, neutral_qz_at_pdf_times,
        neutral_thermal_speed_at_pdf_times, neutral_ff, n_ion_species, n_neutral_species,
        evolve_density, evolve_upar, evolve_ppar, vz, vpa, z, ntime, time, ntime_pdfs,
        time_pdfs)

    n_runs = length(run_names)

    # plot_unnormalised() requires PyPlot, so ensure it is used for all plots for
    # consistency
    pyplot()

    # analyze the fields data
    phi_fldline_avg, delta_phi = get_tuple_of_return_values(analyze_fields_data, phi,
                                                            ntime, z)

    # use a fit to calculate and write to file the damping rate and growth rate of the
    # perturbed electrostatic potential
    frequency, growth_rate, shifted_time, fitted_delta_phi =
        get_tuple_of_return_values(calculate_and_write_frequencies, run_names, ntime,
                                   time, z, itime_min, itime_max, iz0, delta_phi, pp)
    # create the requested plots of the fields
    plot_fields(phi, delta_phi, time, itime_min, itime_max, nwrite_movie,
                z, iz0, run_names, fitted_delta_phi, pp)
    # analyze the velocity moments data
    density_fldline_avg, upar_fldline_avg, ppar_fldline_avg, vth_fldline_avg, qpar_fldline_avg,
        delta_density, delta_upar, delta_ppar, delta_vth, delta_qpar =
        get_tuple_of_return_values(analyze_moments_data, density, parallel_flow,
            parallel_pressure, thermal_speed, parallel_heat_flux, ntime, n_ion_species, z)
    # create the requested plots of the moments
    plot_moments(density, delta_density, density_fldline_avg,
        parallel_flow, delta_upar, upar_fldline_avg,
        parallel_pressure, delta_ppar, ppar_fldline_avg,
        thermal_speed, delta_vth, vth_fldline_avg,
        parallel_heat_flux, delta_qpar, qpar_fldline_avg,
        pp, run_names, time, itime_min, itime_max, nwrite_movie, z, iz0, n_ion_species,
        "ion")
    if maximum(n_neutral_species) > 0
        # analyze the velocity neutral moments data
        neutral_density_fldline_avg, neutral_uz_fldline_avg, neutral_pz_fldline_avg,
        neutral_vth_fldline_avg, neutral_qz_fldline_avg, delta_neutral_density,
        delta_neutral_uz, delta_neutral_pz, delta_neutral_vth, delta_neutral_qz =
            get_tuple_of_return_values(analyze_moments_data, neutral_density, neutral_uz,
                neutral_pz, neutral_thermal_speed, neutral_qz, ntime, n_neutral_species, z)
        # create the requested plots of the neutral moments
        plot_moments(neutral_density, delta_neutral_density, neutral_density_fldline_avg,
            neutral_uz, delta_neutral_uz, neutral_uz_fldline_avg, neutral_pz,
            delta_neutral_pz, neutral_pz_fldline_avg, neutral_thermal_speed,
            delta_neutral_vth, neutral_vth_fldline_avg, neutral_qz, delta_neutral_qz,
            neutral_qz_fldline_avg, pp, run_names, time, itime_min, itime_max,
            nwrite_movie, z, iz0, n_ion_species, "neutral")
    end

    # analyze the pdf data
    f_fldline_avg, delta_f, dens_moment, upar_moment, ppar_moment =
        get_tuple_of_return_values(analyze_pdf_data, ff, n_ion_species, ntime_pdfs, z,
                                   vpa, thermal_speed_at_pdf_times, evolve_ppar)

    plot_dfns(density_at_pdf_times, parallel_flow_at_pdf_times,
        parallel_pressure_at_pdf_times, thermal_speed_at_pdf_times, ff, dens_moment,
        upar_moment, ppar_moment, time_pdfs, n_ion_species, z, vpa, evolve_density,
        evolve_upar, evolve_ppar, run_names, itime_min_pdfs, itime_max_pdfs,
        nwrite_movie_pdfs, iz0, "ion")

    if maximum(n_neutral_species) > 0
        # analyze the neutral pdf data
        neutral_f_fldline_avg, delta_neutral_f, neutral_dens_moment, neutral_uz_moment,
        neutral_pz_moment =
            get_tuple_of_return_values(analyze_pdf_data, neutral_ff, n_neutral_species,
                                       ntime_pdfs, z, vpa, neutral_thermal_speed_at_pdf_times,
                                       evolve_ppar)

        plot_dfns(neutral_density_at_pdf_times, neutral_uz_at_pdf_times,
            neutral_pz_at_pdf_times, neutral_thermal_speed_at_pdf_times, neutral_ff,
            neutral_dens_moment, neutral_uz_moment, neutral_pz_moment, time_pdfs,
            n_neutral_species, z, vz, evolve_density, evolve_upar, evolve_ppar, run_names,
            itime_min_pdfs, itime_max_pdfs, nwrite_movie_pdfs, iz0, "neutral")
    end

    println("done.")
end

"""
"""
function calculate_and_write_frequencies(run_name, ntime, time, z, itime_min, itime_max,
                                         iz0, delta_phi, pp)
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
    z, iz0, run_names, fitted_delta_phi, pp)

    println("Plotting fields data...")

    n_runs = length(run_names)
    if n_runs == 1
        prefix = run_names[1]
        legend = false
    else
        prefix = default_compare_prefix
        legend = true
    end

    phimin = minimum(minimum(p) for p ∈ phi)
    phimax = maximum(maximum(p) for p ∈ phi)

    if pp.plot_phi0_vs_t
        # plot the time trace of phi(z=z0)
        #plot(time, log.(phi[i,:]), yscale = :log10)
        plot(legend=legend)
        for (t, p, run_label) ∈ zip(time, phi, run_names)
            @views plot!(t, p[iz0,:], label=run_label)
        end
        outfile = string(prefix, "_phi0_vs_t.pdf")
        trysavefig(outfile)
        plot(legend=legend)
        for (t, dp, fit, run_label) ∈ zip(time, delta_phi, fitted_delta_phi, run_names)
            # plot the time trace of phi(z=z0)-phi_fldline_avg
            @views plot!(t, abs.(dp[iz0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log,
                         label="$run_label δϕ")
            if pp.calculate_frequencies
                plot!(t, abs.(fit), linestyle=:dash, label="$run_label fit")
            end
        end
        outfile = string(prefix, "_delta_phi0_vs_t.pdf")
        trysavefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(z,t)
        subplots = (heatmap(t, this_z.grid, p, xlabel="time", ylabel="z", title=run_label,
                            c = :deep)
                    for (t, this_z, p, run_label) ∈ zip(time, z, phi, run_names))
        plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
        outfile = string(prefix, "_phi_vs_z_t.pdf")
        trysavefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            plot(legend=legend)
            for (t, this_z, p, run_label) ∈ zip(time, z, phi, run_names)
                @views plot!(this_z.grid, p[:,i], xlabel="z", ylabel="ϕ",
                             ylims=(phimin, phimax), label=run_label)
            end
        end
        outfile = string(prefix, "_phi_vs_z.gif")
        trygif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(prefix, "_harrison_comparison.pdf")
    # trysavefig(outfile)
    plot(legend=legend)
    for (t, this_z, p, run_label) ∈ zip(time, z, phi, run_names)
        plot!(this_z.grid, p[:,end], xlabel="z/Lz", ylabel="eϕ/Te", label=run_label,
              linewidth=2)
    end
    outfile = string(prefix, "_phi_final.pdf")
    trysavefig(outfile)

    println("done.")
end

"""
"""
function plot_moments(density, delta_density, density_fldline_avg,
    parallel_flow, delta_upar, upar_fldline_avg,
    parallel_pressure, delta_ppar, ppar_fldline_avg,
    thermal_speed, delta_vth, vth_fldline_avg,
    parallel_heat_flux, delta_qpar, qpar_fldline_avg,
    pp, run_names, time, itime_min, itime_max, nwrite_movie, z, iz0, n_species, label)

    println("Plotting velocity moments data...")

    n_runs = length(run_names)
    if n_runs == 1
        prefix = run_names[1]
        legend = false
    else
        prefix = default_compare_prefix
        legend = true
    end

    # plot the species-summed, field-line averaged vs time
    denstot = Tuple(sum(n_fldline_avg,dims=1)[1,:]
                    for n_fldline_avg ∈ density_fldline_avg)
    for d in denstot
        d ./= d[1]
    end
    denstot_min = minimum(minimum(dtot) for dtot in denstot) - 0.1
    denstot_max = maximum(maximum(dtot) for dtot in denstot) + 0.1
    plot(legend=legend)
    for (t, dtot, run_label) ∈ zip(time, denstot, run_names)
        @views plot!(t, dtot, ylims=(denstot_min,denstot_max), xlabel="time",
                     ylabel="∑ⱼn̅ⱼ(t)/∑ⱼn̅ⱼ(0)", linewidth=2, label=run_label)
    end
    outfile = string(prefix, "_$(label)_denstot_vs_t.pdf")
    trysavefig(outfile)
    for is ∈ 1:maximum(n_species)
        spec_string = string(is)
        dens_min = minimum(minimum(n[:,is,:]) for n ∈ density)
        dens_max = maximum(maximum(n[:,is,:]) for n ∈ density)
        if pp.plot_dens0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, n, run_label) ∈ zip(time, density, run_names)
                @views plot!(t, n[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_$(label)_dens0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dn, run_label) ∈ zip(time, delta_density, run_names)
                @views plot!(t, abs.(dn[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_$(label)_delta_dens0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of density_fldline_avg
            plot(legend=legend)
            for (t, n_avg, run_label) ∈ zip(time, density_fldline_avg, run_names)
                @views plot!(t, n_avg[is,:], xlabel="time", ylabel="<ns/Nₑ>",
                             ylims=(dens_min,dens_max), label=run_label)
            end
            outfile = string(prefix, "_$(label)_fldline_avg_dens_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the deviation from conservation of density_fldline_avg
            plot(legend=legend)
            for (t, n_avg, run_label) ∈ zip(time, density_fldline_avg, run_names)
                @views plot!(t, n_avg[is,:] .- n_avg[is,1], xlabel="time",
                             ylabel="<(ns-ns(0))/Nₑ>", label=run_label)
            end
            outfile = string(prefix, "_$(label)_conservation_dens_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        upar_min = minimum(minimum(upar[:,is,:]) for upar ∈ parallel_flow)
        upar_max = maximum(maximum(upar[:,is,:]) for upar ∈ parallel_flow)
        if pp.plot_upar0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, upar, run_label) ∈ zip(time, parallel_flow, run_names)
                @views plot!(t, upar[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_$(label)_upar0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dupar, run_label) ∈ zip(time, delta_upar, run_names)
                @views plot!(t, abs.(du[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_$(label)_delta_upar0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of ppar_fldline_avg
            plot(legend=legend)
            for (t, upar_avg, run_label) ∈ zip(time, upar_fldline_avg, run_names)
                @views plot!(t, upar_avg[is,:], xlabel="time",
                             ylabel="<upars/sqrt(2Te/ms)>", ylims=(upar_min,upar_max),
                             label=run_label)
            end
            outfile = string(prefix, "_$(label)_fldline_avg_upar_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        ppar_min = minimum(minimum(ppar[:,is,:]) for ppar ∈ parallel_pressure)
        ppar_max = maximum(maximum(ppar[:,is,:]) for ppar ∈ parallel_pressure)
        if pp.plot_ppar0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, ppar, run_label) ∈ zip(time, parallel_pressure, run_names)
                @views plot!(t, ppar[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_$(label)_ppar0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dppar, run_label) ∈ zip(time, delta_ppar, run_names)
                @views plot!(t, abs.(dppar[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_$(label)_delta_ppar0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of ppar_fldline_avg
            plot(legend=legend)
            for (t, ppar_avg, run_label) ∈ zip(time, ppar_fldline_avg, run_names)
                @views plot!(t, ppar_avg[is,:], xlabel="time", ylabel="<ppars/NₑTₑ>",
                             ylims=(ppar_min,ppar_max), label=run_label)
            end
            outfile = string(prefix, "_$(label)_fldline_avg_ppar_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        vth_min = minimum(minimum(vth[:,is,:]) for vth ∈ thermal_speed)
        vth_max = maximum(maximum(vth[:,is,:]) for vth ∈ thermal_speed)
        if pp.plot_vth0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, vth, run_label) ∈ zip(time, thermal_speed, run_names)
                @views plot!(t, vth[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_$(label)_vth0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dvth, run_label) ∈ zip(time, delta_vth, run_names)
                @views plot!(t, abs.(dvth[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_$(label)_delta_vth0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of vth_fldline_avg
            plot(legend=legend)
            for (t, vth_avg, run_label) ∈ zip(time, vth_fldline_avg, run_names)
                @views plot!(t, vth_avg[is,:], xlabel="time", ylabel="<vths/cₛ₀>",
                             ylims=(vth_min,vth_max), label=run_label)
            end
            outfile = string(prefix, "_$(label)_fldline_avg_vth_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        qpar_min = minimum(minimum(qpar[:,is,:]) for qpar ∈ parallel_heat_flux)
        qpar_max = maximum(maximum(qpar[:,is,:]) for qpar ∈ parallel_heat_flux)
        if pp.plot_qpar0_vs_t
            # plot the time trace of n_s(z=z0)
            plot(legend=legend)
            for (t, qpar, run_label) ∈ zip(time, parallel_heat_flux, run_names)
                @views plot!(t, qpar[iz0,is,:], label=run_label)
            end
            outfile = string(prefix, "_$(label)_qpar0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of n_s(z=z0)-density_fldline_avg
            plot(legend=legend)
            for (t, dqpar, run_label) ∈ zip(time, delta_qpar, run_names)
                @views plot!(t, abs.(dqpar[iz0,is,:]), yaxis=:log, label=run_label)
            end
            outfile = string(prefix, "_$(label)_delta_qpar0_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
            # plot the time trace of ppar_fldline_avg
            plot(legend=legend)
            for (t, qpar_avg, run_label) ∈ zip(time, qpar_fldline_avg, run_names)
                @views plot!(t, qpar_avg[is,:], xlabel="time", ylabel="<qpars/NₑTₑvth>",
                             ylims=(qpar_min,qpar_max), label=run_label)
            end
            outfile = string(prefix, "_$(label)_fldline_avg_qpar_vs_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        if pp.plot_dens_vs_z_t
            # make a heatmap plot of n_s(z,t)
            subplots = (heatmap(t, this_z.grid, n[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, n, run_label) ∈ zip(time, z, density, run_names))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_$(label)_dens_vs_z_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        if pp.plot_upar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z.grid, upar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, upar, run_label) ∈ zip(time, z, parallel_flow,
                                                              run_names))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_$(label)_upar_vs_z_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        if pp.plot_ppar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z.grid, ppar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, ppar, run_label) ∈
                            zip(time, z, parallel_pressure, run_names))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_$(label)_ppar_vs_z_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        # Note factor of 2 here because currently temperatures are normalised by Tref,
        # while pressures are normalised by m*nref*c_ref^2=2*nref*Tref
        parallel_temperature = (2.0 .* ppar ./ n
                                for (n, ppar) ∈ zip(density, parallel_pressure))
        Tpar_min = minimum(minimum(Tpar[:,is,:]) for Tpar ∈ parallel_temperature)
        Tpar_max = maximum(maximum(Tpar[:,is,:]) for Tpar ∈ parallel_temperature)
        if pp.plot_Tpar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z.grid, Tpar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, Tpar, run_label) ∈
                            zip(time, z, parallel_temperature, run_names))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_$(label)_Tpar_vs_z_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        if pp.plot_qpar_vs_z_t
            # make a heatmap plot of upar_s(z,t)
            subplots = (heatmap(t, this_z.grid, qpar[:,is,:], xlabel="time", ylabel="z",
                                title=run_label, c = :deep)
                        for (t, this_z, qpar, run_label) ∈
                            zip(time, z, parallel_heat_flux, run_names))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_$(label)_qpar_vs_z_t_spec", spec_string, ".pdf")
            trysavefig(outfile)
        end
        if pp.animate_dens_vs_z
            # make a gif animation of ϕ(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, n, run_label) ∈ zip(time, z, density, run_names)
                    @views plot!(this_z.grid, n[:,is,i], xlabel="z", ylabel="nᵢ/Nₑ",
                                 ylims=(dens_min, dens_max), label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_dens_vs_z_spec", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_upar_vs_z
            # make a gif animation of upar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, upar, run_label) ∈ zip(time, z, parallel_flow, run_names)
                    @views plot!(this_z.grid, upar[:,is,i], xlabel="z", ylabel="upars/vt",
                                 ylims=(upar_min, upar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_upar_vs_z_spec", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_ppar_vs_z
            # make a gif animation of ppar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, ppar, run_label) ∈ zip(time, z, parallel_pressure, run_names)
                    @views plot!(this_z.grid, ppar[:,is,i], xlabel="z", ylabel="ppars",
                                 ylims=(ppar_min, ppar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_ppar_vs_z_spec", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_Tpar_vs_z
            # make a gif animation of Tpar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, Tpar, run_label) ∈ zip(time, z, parallel_temperature,
                                                       run_names)
                    @views plot!(this_z.grid, Tpar[:,is,i], xlabel="z", ylabel="Tpars",
                                 ylims=(Tpar_min, Tpar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_ppar_vs_z_spec", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_vth_vs_z
            # make a gif animation of vth(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, vth, run_label) ∈ zip(time, z, thermal_speed, run_names)
                    @views plot!(this_z.grid, vth[:,is,i], xlabel="z", ylabel="vths",
                                 ylims=(vth_min, vth_max), label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_vth_vs_z_spec", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_qpar_vs_z
            # make a gif animation of ppar(z) at different times
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                plot(legend=legend)
                for (t, this_z, qpar, run_label) ∈ zip(time, z, parallel_heat_flux,
                                                       run_names)
                    @views plot!(this_z.grid, qpar[:,is,i], xlabel="z", ylabel="qpars",
                                 ylims=(qpar_min, qpar_max), label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_qpar_vs_z_spec", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    println("done.")
end

"""
"""
function plot_dfns(density, parallel_flow, parallel_pressure, thermal_speed, ff,
                   dens_moment, upar_moment, ppar_moment, time_pdfs, n_species, z, vpa,
                   evolve_density, evolve_upar, evolve_ppar, run_names, itime_min_pdfs,
                   itime_max_pdfs, nwrite_movie_pdfs, iz0, label)
    println("Plotting distribution function data...")
    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    logdeep = cgrad(:deep, scale=:log) |> cmlog
    n_species_max = maximum(n_species)
    n_runs = length(run_names)
    if n_runs == 1
        prefix = run_names[1]
        legend = false
    else
        prefix = default_compare_prefix
        legend = true
    end
    for is ∈ 1:n_species_max
        if n_species_max > 1
            spec_string = string("_spec", string(is))
        else
            spec_string = ""
        end
        # plot difference between evolved density and ∫dvpa f; only possibly different if density removed from
        # normalised distribution function at run-time
        plot(legend=legend)
        for (t, n, n_int, run_label) ∈ zip(time_pdfs, density, dens_moment, run_names)
            @views plot!(t, n[iz0,is,:] .- n_int[iz0,is,:], label=run_label)
        end
        outfile = string(prefix, "_$(label)_intf0_vs_t", spec_string, ".pdf")
        trysavefig(outfile)
        # if evolve_upar = true, plot ∫dwpa wpa * f, which should equal zero
        # otherwise, this plots ∫dvpa vpa * f, which is dens*upar
        plot(legend=legend)
        for (t, upar_int, run_label) ∈ zip(time_pdfs, upar_moment, run_names)
            intwf0_max = maximum(abs.(upar_int[iz0,is,:]))
            if intwf0_max < 1.0e-15
                @views plot!(t, upar_int[iz0,is,:], ylims = (-1.0e-15, 1.0e-15), label=run_label)
            else
                @views plot!(t, upar_int[iz0,is,:], label=run_label)
            end
        end
        outfile = string(prefix, "_$(label)_intwf0_vs_t", spec_string, ".pdf")
        trysavefig(outfile)
        # plot difference between evolved parallel pressure and ∫dvpa vpa^2 f;
        # only possibly different if density and thermal speed removed from
        # normalised distribution function at run-time
        plot(legend=legend)
        for (t, ppar, ppar_int, run_label) ∈ zip(time_pdfs, parallel_pressure, ppar_moment, run_names)
            @views plot(t, ppar[iz0,is,:] .- ppar_int[iz0,is,:], label=run_label)
        end
        outfile = string(prefix, "_$(label)_intw2f0_vs_t", spec_string, ".pdf")
        trysavefig(outfile)
        #fmin = minimum(ff[:,:,is,:])
        #fmax = maximum(ff[:,:,is,:])
        if pp.animate_f_vs_vpa_z
            # make a gif animation of ln f(vpa,z,t)
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                subplots = (@views heatmap(this_z.grid, this_vpa.grid, log.(abs.(f[:,:,is,i])),
                                           xlabel="z", ylabel="vpa", fillcolor = logdeep,
                                           title=run_label)
                            for (f, this_z, this_vpa, run_label) ∈ zip(ff, z, vpa, run_names))
                plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            end
            outfile = string(prefix, "_$(label)_logf_vs_vpa_z", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
            # make a gif animation of f(vpa,z,t)
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                #heatmap(z, vpa, log.(abs.(ff[:,:,i])), xlabel="z", ylabel="vpa", clims = (fmin,fmax), c = :deep)
                subplots = (@views heatmap(this_z.grid, this_vpa.grid, f[:,:,is,i], xlabel="z",
                                           ylabel="vpa", c = :deep,
                                           interpolation = :cubic, title=run_label)
                            for (f, this_z, this_vpa, run_label) ∈ zip(ff, z, vpa, run_names))
                plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            end
            outfile = string(prefix, "_$(label)_f_vs_vpa_z", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
            # make pdf of f(vpa,z,t_final) for each species
            str = string("spec ", string(is), " pdf")
            subplots = (@views heatmap(this_z.grid, this_vpa.grid, f[:,:,is,end],
                                       xlabel="vpa", ylabel="z", c = :deep, interpolation
                                       = :cubic, title=string(run_label, str))
                        for (f, this_z, this_vpa, run_label) ∈ zip(ff, z, vpa, run_names))
            plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            outfile = string(prefix, "_$(label)_f_vs_z_vpa_final", spec_string, ".pdf")
            trysavefig(outfile)

            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (f, this_vpa, run_label) ∈ zip(ff, vpa, run_names)
                    @views plot!(this_vpa.grid, f[:,1,is,i], xlabel="vpa", ylabel="f(z=0)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_f0_vs_vpa", spec_string, ".gif")
            trygif(anim, outfile, fps=5)

            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (f, this_vpa, run_label) ∈ zip(ff, vpa, run_names)
                    @views plot!(this_vpa.grid, f[:,end,is,i], xlabel="vpa", ylabel="f(z=L)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_fL_vs_vpa", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.plot_f_unnormalized_vs_vpa_z
            PyPlot.clf()
            fig = PyPlot.figure(1, figsize=(6*n_runs,4))
            it = itime_max_pdfs
            # i counts from 0, Python-style
            for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                 run_label) ∈ zip(1:n_runs, ff, density, parallel_flow, thermal_speed,
                                  evolve_density, evolve_upar, evolve_ppar, z, vpa,
                                  run_names)

                PyPlot.subplot(1, n_runs, run_ind)
                @views f_unnorm, z2d, dzdt2d = get_unnormalised_f_coords_2d(
                    f[:,:,is,it], this_z.grid, this_vpa.grid, n[:,is,it],
                    upar[:,is,it], vth[:,is,it], ev_n, ev_u, ev_p)
                plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; title=run_label,
                                      plot_log=false)
            end
            outfile = string(prefix, "_f_unnorm_vs_vpa_z", spec_string, ".pdf")
            try
                PyPlot.savefig(outfile)
            catch
                println("Failed to make plot $outfile")
            end
            PyPlot.clf()

            plot(legend=legend)
            it = itime_max_pdfs
            # i counts from 0, Python-style
            for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                 run_label) ∈ zip(1:n_runs, ff, density, parallel_flow, thermal_speed,
                                  evolve_density, evolve_upar, evolve_ppar, z, vpa,
                                  run_names)

                @views f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(
                    f[:,1,is,it], this_vpa.grid, n[1,is,it], upar[1,is,it], vth[1,is,it],
                    ev_n, ev_u, ev_p)
                @views plot!(dzdt, f_unnorm, xlabel="vpa", ylabel="f_unnorm(z=0)",
                             label=run_label)
            end
            trysavefig(string(prefix, "_f0_unnorm_vs_vpa", spec_string, ".pdf"))

            plot(legend=legend)
            it = itime_max_pdfs
            # i counts from 0, Python-style
            for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                 run_label) ∈ zip(1:n_runs, ff, density, parallel_flow, thermal_speed,
                                  evolve_density, evolve_upar, evolve_ppar, z, vpa,
                                  run_names)

                @views f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(
                    f[:,end,is,it], this_vpa.grid, n[end,is,it], upar[end,is,it], vth[end,is,it],
                    ev_n, ev_u, ev_p)
                @views plot!(dzdt, f_unnorm, xlabel="vpa", ylabel="f_unnorm(z=L)",
                             label=run_label)
            end
            trysavefig(string(prefix, "_fL_unnorm_vs_vpa", spec_string, ".pdf"))
        end
        if pp.animate_f_unnormalized
            ## The nice, commented out version will only work when plot_unnormalised can
            ## use Plots.jl...
            ## make a gif animation of f_unnorm(v_parallel_unnorm,z,t)
            #anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
            #    subplots = (@views plot_unnormalised(f[:,:,is,i], this_z.grid, this_vpa.grid,
            #                           n[:,is,i], upar[:,is,i], vth[:,is,i], ev_n, ev_u,
            #                           ev_p, title=run_label)
            #                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_z,
            #                     this_vpa, run_label) ∈
            #                zip(ff, density, parallel_flow, thermal_speed,
            #                    evolve_density, evolve_upar, evolve_ppar, z, vpa,
            #                    run_names))
            #    plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            #end
            #outfile = string(prefix, "_$(label)_f_unnorm_vs_vpa_z", spec_string, ".gif")
            #trygif(anim, outfile, fps=5)
            ## make a gif animation of log(f_unnorm)(v_parallel_unnorm,z,t)
            #anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
            #    subplots = (@views plot_unnormalised(f[:,:,is,i], this_z.grid, this_vpa.grid, n[:,is,i],
            #                           upar[:,is,i], vth[:,is,i], ev_n, ev_u, ev_p,
            #                           plot_log=true, title=run_label)
            #                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_z,
            #                     this_vpa, run_label) ∈
            #                    zip(ff, density, parallel_flow, thermal_speed,
            #                        evolve_density, evolve_upar, evolve_ppar, z,
            #                        vpa, run_names))
            #    plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            #end
            #outfile = string(prefix, "_$(label)_logf_unnorm_vs_vpa_z", spec_string, ".gif")
            #trygif(anim, outfile, fps=5)

            matplotlib = pyimport("matplotlib")
            matplotlib.use("agg")
            matplotlib_animation = pyimport("matplotlib.animation")
            iframes = collect(itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs)
            nframes = length(iframes)
            function make_frame(i)
                PyPlot.clf()
                iframe = iframes[i+1]
                # i counts from 0, Python-style
                for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                     run_label) ∈ zip(1:n_runs, ff, density, parallel_flow, thermal_speed,
                                      evolve_density, evolve_upar, evolve_ppar, z, vpa,
                                      run_names)

                    PyPlot.subplot(1, n_runs, run_ind)
                    @views f_unnorm, z2d, dzdt2d = get_unnormalised_f_coords_2d(
                        f[:,:,is,iframe], this_z.grid, this_vpa.grid, n[:,is,iframe],
                        upar[:,is,iframe], vth[:,is,iframe], ev_n, ev_u, ev_p)
                    plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; title=run_label,
                                          plot_log=false)
                end
            end
            fig = PyPlot.figure(1, figsize=(6*n_runs,4))
            myanim = matplotlib_animation.FuncAnimation(fig, make_frame, frames=nframes)
            outfile = string(prefix, "_$(label)_f_unnorm_vs_vpa_z", spec_string, ".gif")
            try
                myanim.save(outfile, writer=matplotlib_animation.PillowWriter(fps=30))
            catch
                println("Failed to make animation $outfile")
            end
            PyPlot.clf()

            function make_frame_log(i)
                PyPlot.clf()
                iframe = iframes[i+1]
                # i counts from 0, Python-style
                for (run_ind, f, n, upar, vth, ev_n, ev_u, ev_p, this_z, this_vpa,
                     run_label) ∈ zip(1:n_runs, ff, density, parallel_flow, thermal_speed,
                                      evolve_density, evolve_upar, evolve_ppar, z, vpa,
                                      run_names)

                    PyPlot.subplot(1, n_runs, run_ind)
                    @views f_unnorm, z2d, dzdt2d = get_unnormalised_f_coords_2d(
                        f[:,:,is,iframe], this_z.grid, this_vpa.grid, n[:,is,iframe],
                        upar[:,is,iframe], vth[:,is,iframe], ev_n, ev_u, ev_p)
                    plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; title=run_label,
                                          plot_log=true)
                end
            end
            fig = PyPlot.figure(figsize=(6*n_runs,4))
            myanim = matplotlib_animation.FuncAnimation(fig, make_frame_log, frames=nframes)
            outfile = string(prefix, "_$(label)_logf_unnorm_vs_vpa_z", spec_string, ".gif")
            try
                myanim.save(outfile, writer=matplotlib_animation.PillowWriter(fps=30))
            catch
                println("Failed to make animation $outfile")
            end

            # Ensure PyPlot figure is cleared
            closeall()

            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_vpa, run_label) ∈
                    zip(ff, density, parallel_flow, thermal_speed, evolve_density,
                        evolve_upar, evolve_ppar, vpa, run_names)
                    @views f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(
                        f[:,1,is,i], this_vpa.grid, n[1,is,i], upar[1,is,i], vth[1,is,i],
                        ev_n, ev_u, ev_p)
                    @views plot!(dzdt, f_unnorm, xlabel="vpa", ylabel="f_unnorm(z=0)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_f0_unnorm_vs_vpa", spec_string, ".gif")
            trygif(anim, outfile, fps=5)

            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (f, n, upar, vth, ev_n, ev_u, ev_p, this_vpa, run_label) ∈
                    zip(ff, density, parallel_flow, thermal_speed, evolve_density,
                        evolve_upar, evolve_ppar, vpa, run_names)
                    @views f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(
                        f[:,end,is,i], this_vpa.grid, n[end,is,i], upar[end,is,i],
                        vth[end,is,i], ev_n, ev_u, ev_p)
                    @views plot!(dzdt, f_unnorm, xlabel="vpa", ylabel="f_unnorm(z=L)",
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_fL_unnorm_vs_vpa", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z
            # make a gif animation of δf(vpa,z,t)
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                subplots = (@views heatmap(this_z, this_vpa.grid, delta_f[:,:,is,i],
                                       xlabel="z", ylabel="vpa", c = :deep,
                                       interpolation = :cubic, title=run_label)
                            for (df, this_z, this_vpa, run_label) ∈
                                zip(delta_f, z, vpa, run_names))
                plot(subplots..., layout=(1,n_runs), size=(600*n_runs, 400))
            end
            outfile = string(prefix, "_$(label)_deltaf_vs_vpa_z", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(minimum(f[ivpa0,:,is,:]) for f ∈ ff)
            fmax = maximum(maximum(f[ivpa0,:,is,:]) for f ∈ ff)
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (f, this_z, run_label) ∈ zip(ff, z, run_names)
                    @views plot!(this_z, f[ivpa0,:,is,i], ylims = (fmin,fmax),
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_f_vs_z", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(minimum(df[ivpa0,:,is,:]) for df ∈ delta_f)
            fmax = maximum(maximum(df[ivpa0,:,is,:]) for df ∈ delta_f)
            # make a gif animation of f(vpa0,z,t)
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (df, this_z, run_label) ∈ zip(delta_f, z, run_names)
                    @views plot!(this_z, df[ivpa0,:,is,i], ylims = (fmin,fmax),
                                 label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_deltaf_vs_z", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_f_vs_vpa_z0
            fmin = minimum(minimum(f[:,iz0,is,:]) for f ∈ ff)
            fmax = maximum(maximum(f[:,iz0,is,:]) for f ∈ ff)

            # if is == 1
            #     tmp = copy(ff)
            #     @. tmp[:,1,1,:] /= vpa^2
            #     bohm_integral = copy(time)
            #     for i ∈ 1:ntime
            #         @views bohm_integral[i] = integrate_over_vspace(tmp[1:cld(nvpa,2)-1,1,1,i],vpa_wgts[1:cld(nvpa,2)-1])/2.0
            #     end
            #     plot(time, bohm_integral, xlabel="time", label="Bohm integral")
            #     plot!(time, density[1,1,:], label="nᵢ(zmin)")
            #     outfile = string(prefix, "_$(label)_Bohm_criterion.pdf")
            #     trysavefig(outfile)
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
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                #@views plot(vpa, ff[iz0,:,is,i], ylims = (fmin,fmax))
                plot(legend=legend)
                for (f, this_vpa, run_label) ∈ zip(ff, vpa, run_names)
                    @views plot!(this_vpa.grid, f[:,iz0,is,i], label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_f_vs_vpa", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
        if pp.animate_deltaf_vs_vpa_z0
            fmin = minimum(minimum(df[:,iz0,is,:]) for df ∈ delta_f)
            fmax = maximum(maximum(df[:,iz0,is,:]) for df ∈ delta_f)
            # make a gif animation of f(vpa,z0,t)
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                plot(legend=legend)
                for (df, this_vpa, fn, fx, run_label) ∈
                        zip(delta_f, vpa, fmin, fmax, run_names)
                    @views plot!(this_vpa.grid, delta_f[:,iz0,is,i], ylims = (fn,fx),
                                label=run_label)
                end
            end
            outfile = string(prefix, "_$(label)_deltaf_vs_vpa", spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    return nothing
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

"""
Add a thin, red, dashed line showing v_parallel=(vth*w_parallel+upar)=0 to a 2d plot
with axes (z,vpa).
"""
function draw_v_parallel_zero!(plt::Plots.Plot, z::AbstractVector, upar, vth,
                               evolve_upar::Bool, evolve_ppar::Bool)
    if evolve_ppar && evolve_upar
        zero_value = @. -upar/vth
    elseif evolve_upar
        zero_value = @. -upar
    else
        zero_value = zeros(size(upar))
    end
    plot!(plt, z, zero_value, color=:red, linestyle=:dash, linewidth=1,
          xlims=xlims(plt), ylims=ylims(plt), label="")
end
function draw_v_parallel_zero!(z::AbstractVector, upar, vth, evolve_upar::Bool,
                               evolve_ppar::Bool)
    draw_v_parallel_zero!(Plots.CURRENT_PLOT, z, upar, vth, evolve_upar, evolve_ppar)
end

"""
Get the unnormalised distribution function and unnormalised ('lab space') dzdt
coordinate at a point in space.

Inputs should depend only on vpa.
"""
function get_unnormalised_f_dzdt_1d(f, vpa_grid, density, upar, vth, evolve_density,
                                    evolve_upar, evolve_ppar)

    dzdt = vpagrid_to_dzdt(vpa_grid, vth, upar, evolve_ppar, evolve_upar)

    if evolve_ppar
        f_unnorm = @. f * density / vth
    elseif evolve_density
        f_unnorm = @. f * density
    else
        f_unnorm = copy(f)
    end

    return f_unnorm, dzdt
end

"""
Get the unnormalised distribution function and unnormalised ('lab space') coordinates.

Inputs should depend only on z and vpa.
"""
function get_unnormalised_f_coords_2d(f, z_grid, vpa_grid, density, upar, vth,
                                      evolve_density, evolve_upar, evolve_ppar)

    nvpa, nz = size(f)
    z2d = zeros(nvpa, nz)
    dzdt2d = zeros(nvpa, nz)
    f_unnorm = similar(f)
    for iz ∈ 1:nz
        @views z2d[:,iz] .= z_grid[iz]
        f_unnorm[:,iz], dzdt2d[:,iz] =
            get_unnormalised_f_dzdt_1d(f[:,iz], vpa_grid, density[iz], upar[iz],
                                       vth[iz], evolve_density, evolve_upar,
                                       evolve_ppar)
    end

    return f_unnorm, z2d, dzdt2d
end

"""
Make a 2d plot of an unnormalised f on unnormalised coordinates, as returned from
get_unnormalised_f_coords()

Note this function requires using the PyPlot backend to support 2d coordinates being
passed to `heatmap()`.
"""
function plot_unnormalised_f2d(f_unnorm, z2d, dzdt2d; plot_log=false, kwargs...)

    if backend_name() != :pyplot
        error("PyPlot backend is required for plot_unnormalised(). Call pyplot() "
              * "first.")
    end

    ## The following commented out section does not work at the moment because
    ## Plots.heatmap() does not support 2d coordinates.
    ## https://github.com/JuliaPlots/Plots.jl/pull/4298 would add this feature...
    #if plot_log
    #    @. f_unnorm = log(abs(f_unnorm))
    #    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
    #    cmap = cgrad(:deep, scale=:log) |> cmlog
    #else
    #    cmap = :deep
    #end

    #p = plot(; xlabel="z", ylabel="vpa", c=cmap, kwargs...)

    #heatmap(z2d, dzdt2d, f_unnorm)

    # Use PyPlot directly instead. Unfortunately makes animation a pain...
    if plot_log
        vmin = minimum(x for x in f_unnorm if x > 0.0)
        norm = PyPlot.matplotlib.colors.LogNorm(vmin=vmin, vmax=maximum(f_unnorm))
    else
        norm = nothing
    end
    p = PyPlot.pcolormesh(z2d, dzdt2d, f_unnorm; norm=norm, cmap="viridis_r")
    PyPlot.xlabel("z")
    PyPlot.ylabel("vpa")
    PyPlot.colorbar()

    return p
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
    trysavefig(outfile)

	if nr > 1
		# plot last timestep field vs r at z_wall
		fieldmin = minimum(field[end,:,end])
		fieldmax = maximum(field[end,:,end])
		@views plot(r, [field[end,:,end], field_sym[end,:,end]], xlabel=L"r/L_r", ylabel=field_label, label=["num" "sym"], ylims = (fieldmin,fieldmax))
		outfile = string(run_name, "_"*file_string*"(r,z_wall)_vs_r.pdf")
		trysavefig(outfile)

        it = ntime
        fontsize = 20
        ticksfontsize = 10
        heatmap(r, z, field[:,:,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=field_label, c = :deep,
         #xtickfontsize = ticksfontsize, xguidefontsize = fontsize, ytickfontsize = ticksfontsize, yguidefontsize = fontsize, titlefontsize = fontsize)
         windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_vs_r_z.pdf")
        trysavefig(outfile)

        heatmap(r, z, field_sym[:,:,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=field_sym_label, c = :deep,
        #xtickfontsize = ticksfontsize, xguidefontsize = fontsize, ytickfontsize = ticksfontsize, yguidefontsize = fontsize, titlefontsize = fontsize)
        windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_sym_vs_r_z.pdf")
        trysavefig(outfile)
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
    trysavefig(outfile)

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
    trysavefig(outfile)

    if nr > 1
        it = ntime
        fontsize = 20
        heatmap(r, z, moment[:,:,is,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=moment_label, c = :deep,
        #xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, titlefontsize = fontsize
        windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_vs_r_z_", spec_string, ".pdf")
        trysavefig(outfile)

        heatmap(r, z, moment_sym[:,:,is,it], xlabel=L"r / L_r", ylabel=L"z / L_z", title=moment_sym_label, c = :deep,
        #xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, titlefontsize = fontsize
        windowsize = (360,240), margin = 15pt)
        outfile = string(run_name, "_"*file_string*"_sym_vs_r_z_", spec_string, ".pdf")
        trysavefig(outfile)
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
    trysavefig(outfile)

    return moment_norm

end

function compare_charged_pdf_symbolic_test(run_name,manufactured_solns_list,spec_string,
    pdf_label,pdf_sym_label,norm_label,file_string)
    fid = open_readonly_output_file(run_name,"dfns", printout=false)
    # load block data on iblock=0
    nblocks, iblock = load_block_data(fid, printout=false)
    # load global sizes of grids that are distributed in memory
    z0, _ = load_coordinate_data(fid, "z", printout=false)
    r0, _ = load_coordinate_data(fid, "r", printout=false)
    # velocity grid data on iblock=0 (same for all blocks)
    vpa, _ = load_coordinate_data(fid, "vpa", printout=false)
    vperp, _ = load_coordinate_data(fid, "vperp", printout=false)
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
        z_local, _ = load_coordinate_data(fid_pdfs, "z")
        r_local, _ = load_coordinate_data(fid_pdfs, "r")
        close(fid_pdfs)
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        for it in 1:ntime, ir in imin_r:r_local.n, iz in imin_z:z_local.n,
                ivperp in 1:vperp.n, ivpa in 1:vpa.n

            pdf_sym = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp],z_local.grid[iz],r_local.grid[ir],time[it])
            pdf_norm[it] += (pdf[ivpa,ivperp,iz,ir,is,it] - pdf_sym)^2
        end
    end
    for it in 1:ntime
        pdf_norm[it] = sqrt(pdf_norm[it]/(r0.n_global*z0.n_global*vpa.n*vperp.n))
    end
    println("test: ",file_string,": ",spec_string," ",pdf_norm)
    @views plot(time, pdf_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=norm_label) #, yaxis=:log)
    outfile = string(run_name, "_"*file_string*"_norm_vs_t_", spec_string, ".pdf")
    trysavefig(outfile)

    # plot distribution at lower wall boundary
    # find the number of ranks
    z_nrank, r_nrank = get_nranks(run_name,nblocks,"dfns")
    for iblock in 0:nblocks-1
        fid_pdfs = open_readonly_output_file(run_name,"dfns",iblock=iblock, printout=false)
        z_irank, r_irank = load_rank_data(fid_pdfs,printout=false)
        if (z_irank == 0 || z_irank == z_nrank - 1) && r_irank == 0
            pdf = load_pdf_data(fid_pdfs, printout=false)
            # local local grid data on iblock=0
            z_local, _ = load_coordinate_data(fid_pdfs, "z")
            r_local, _ = load_coordinate_data(fid_pdfs, "r")
            pdf_sym_array = copy(vpa.grid)
            # plot a thermal vperp on line plots
            ivperp0 = max(floor(mk_int,vperp.n/3),1)
            # plot a typical r on line plots
            ir0 = 1
            # plot at the wall boundary
            if z_irank == 0
                iz0 = 1
                zlabel="wall-"
            elseif z_irank == z_nrank - 1
                iz0 = z_local.n
                zlabel="wall+"
            end
            for ivpa in 1:vpa.n
                pdf_sym_array[ivpa] = dfni_func(vpa.grid[ivpa],vperp.grid[ivperp0],z_local.grid[iz0],r_local.grid[ir0],time[ntime])
            end
            # plot f(vpa,ivperp0,iz_wall,ir0,is,itime) at the wall
            @views plot(vpa.grid, [pdf[:,ivperp0,iz0,ir0,is,ntime], pdf_sym_array], xlabel=L"v_{\|\|}/L_{v_{\|\|}}", ylabel=L"f_i", label=["num" "sym"])
            outfile = string(run_name, "_pdf(vpa,vperp0,iz_"*zlabel*",ir0)_sym_vs_vpa.pdf")
            trysavefig(outfile)
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
    z0, _ = load_coordinate_data(fid, "z", printout=false)
    r0, _ = load_coordinate_data(fid, "r", printout=false)
    # velocity grid data on iblock=0 (same for all blocks)
    vz, _ = load_coordinate_data(fid, "vz", printout=false)
    vr, _ = load_coordinate_data(fid, "vr", printout=false)
    vzeta, _ = load_coordinate_data(fid, "vzeta", printout=false)
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
        z_local, _ = load_coordinate_data(fid_pdfs, "z", printout=false)
        r_local, _ = load_coordinate_data(fid_pdfs, "r", printout=false)
        close(fid_pdfs)
        imin_r = min(1,r_irank) + 1
        imin_z = min(1,z_irank) + 1
        for it in 1:ntime, ir in imin_r:r_local.n, iz in imin_z:z_local.n,
                ivzeta in 1:vzeta.n, ivr in 1:vr.n, ivz in 1:vz.n

            pdf_sym = dfnn_func(vz.grid[ivz],vr.grid[ivr],vzeta.grid[ivzeta],z_local.grid[iz],r_local.grid[ir],time[it])
            pdf_norm[it] += (pdf[ivz,ivr,ivzeta,iz,ir,is,it] - pdf_sym)^2
        end
    end
    for it in 1:ntime
        pdf_norm[it] = sqrt(pdf_norm[it]/(r0.n_global*z0.n_global*vz.n*vr.n*vzeta.n))
    end
    println("test: ",file_string,": ",spec_string," ",pdf_norm)
    @views plot(time, pdf_norm[:], xlabel=L"t L_z/v_{ti}", ylabel=norm_label) #, yaxis=:log)
    outfile = string(run_name, "_"*file_string*"_norm_vs_t_", spec_string, ".pdf")
    trysavefig(outfile)

    return pdf_norm
end

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
        trysavefig(outfile)
        # plot the time trace of phi(r=r0)-phi_fldline_avg
        @views plot(time, abs.(delta_phi[ir0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log)
        if pp.calculate_frequencies
            plot!(time, abs.(fitted_delta_phi))
        end
        outfile = string(run_name, "_delta_phi(r0,z0)_vs_t.pdf")
        trysavefig(outfile)
    end
    if pp.plot_phi_vs_z_t
        # make a heatmap plot of ϕ(r,t)
        heatmap(time, r.grid, phi, xlabel="time", ylabel="r", title="ϕ", c = :deep)
        outfile = string(run_name, "_phi_vs_r_t.pdf")
        trysavefig(outfile)
    end
    if pp.animate_phi_vs_z
        # make a gif animation of ϕ(r) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(r.grid, phi[:,i], xlabel="r", ylabel="ϕ", ylims = (phimin,phimax))
        end
        outfile = string(run_name, "_phi_vs_r.gif")
        trygif(anim, outfile, fps=5)
    end
    # nz = length(z)
    # izmid = cld(nz,2)
    # plot(z[izmid:end], phi[izmid:end,end] .- phi[izmid,end], xlabel="z/Lz - 1/2", ylabel="eϕ/Te", label = "data", linewidth=2)
    # plot!(exp.(-(phi[cld(nz,2),end] .- phi[izmid:end,end])) .* erfi.(sqrt.(abs.(phi[cld(nz,2),end] .- phi[izmid:end,end])))/sqrt(pi)/0.688, phi[izmid:end,end] .- phi[izmid,end], label = "analytical", linewidth=2)
    # outfile = string(run_name, "_harrison_comparison.pdf")
    # trysavefig(outfile)
    plot(r.grid, phi[:,end], xlabel="r/Lr", ylabel="eϕ/Te", label="", linewidth=2)
    outfile = string(run_name, "_phi(r)_final.pdf")
    trysavefig(outfile)

    println("done.")
end

"""
plots various slices of the ion pdf (1d and 2d, stills and animations)
"""
function plot_charged_pdf(run_name, vpa, vperp, z, r, z_local, r_local, ivpa0, ivperp0,
                          iz0, ir0, spec_type, n_species, n_time_pdfs, nblocks, itime_min,
                          itime_max, iskip, nwrite_movie, pp)

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
    if n_species > 1
        spec_string = [string("_", spec_type, "_spec", string(is)) for is ∈ 1:n_species]
    else
        spec_string = [string("_", spec_type)]
    end
    # make a gif animation of f(vpa,z,t) at a given (vperp,r) location
    if pp.animate_f_vs_vpa_z
        pdf = load_distributed_charged_pdf_slice(run_name, nblocks,
                                                 itime_min:iskip:itime_max, n_species,
                                                 r_local, z_local, vperp, vpa;
                                                 ivperp=ivperp0, ir=ir0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z.grid, vpa.grid, pdf[:,:,is,i], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vpa_z", ivperp0_string, ir0_string, spec_string[is], ".gif")
            trygif(anim, outfile, fps=5)

            @views heatmap(z.grid, vpa.grid, pdf[:,:,is,itime_max], xlabel="z", ylabel="vpa", c = :deep, interpolation = :cubic)
            outfile = string(run_name, "_pdf_vs_vpa_z", ivperp0_string, ir0_string, spec_string[is], ".pdf")
            trysavefig(outfile)
        end
    end
    # make a gif animation of f(vpa,r,t) at a given (vperp,z) location
    if pp.animate_f_vs_vpa_r
        pdf = load_distributed_charged_pdf_slice(run_name, nblocks,
                                                 itime_min:iskip:itime_max, n_species,
                                                 r_local, z_local, vperp, vpa;
                                                 ivperp=ivperp0, iz=iz0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r.grid, vpa.grid, pdf[:,:,is,i], xlabel="r", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vpa_r", ivperp0_string, iz0_string, spec_string[is], ".gif")
            trygif(anim, outfile, fps=5)

            @views heatmap(r.grid, vpa.grid, pdf[:,:,is,itime_max], xlabel="r", ylabel="vpa", c = :deep, interpolation = :cubic)
            outfile = string(run_name, "_pdf_vs_vpa_r", ivperp0_string, iz0_string, spec_string[is], ".pdf")
            trysavefig(outfile)
        end
    end
    # make a gif animation of f(vperp,z,t) at a given (vpa,r) location
    if pp.animate_f_vs_vperp_z
        pdf = load_distributed_charged_pdf_slice(run_name, nblocks,
                                                 itime_min:iskip:itime_max, n_species,
                                                 r_local, z_local, vperp, vpa; ivpa=ivpa0,
                                                 ir=ir0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(z.grid, vperp.grid, pdf[:,:,is,i], xlabel="z", ylabel="vperp", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vperp_z", ivpa0_string, ir0_string, spec_string[is], ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    # make a gif animation of f(vperp,r,t) at a given (vpa,z) location
    if pp.animate_f_vs_vperp_r
        pdf = load_distributed_charged_pdf_slice(run_name, nblocks,
                                                 itime_min:iskip:itime_max, n_species,
                                                 r_local, z_local, vperp, vpa; ivpa=ivpa0,
                                                 iz=iz0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r.grid, vperp.grid, pdf[:,:,is,i], xlabel="r", ylabel="vperp", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vperp_r", ivperp0_string, iz0_string, spec_string[is], ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    # make a gif animation of f(vpa,vperp,t) at a given (z,r) location
    if pp.animate_f_vs_vperp_vpa
        pdf = load_distributed_charged_pdf_slice(run_name, nblocks,
                                                 itime_min:iskip:itime_max, n_species,
                                                 r_local, z_local, vperp, vpa; iz=iz0,
                                                 ir=ir0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(vperp.grid, vpa.grid, pdf[:,:,is,i], xlabel="vperp", ylabel="vpa", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vperp_vpa", iz0_string, ir0_string, spec_string[is], ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    # make a gif animation of f(z,r,t) at a given (vpa,vperp) location
    if pp.animate_f_vs_r_z
        pdf = load_distributed_charged_pdf_slice(run_name, nblocks,
                                                 itime_min:iskip:itime_max, n_species,
                                                 r_local, z_local, vperp, vpa; ivpa=ivpa0,
                                                 ivperp=ivperp0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                @views heatmap(r.grid, z.grid, pdf[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_r_z", ivpa0_string, ivperp0_string, spec_string[is], ".gif")
            trygif(anim, outfile, fps=5)

            @views heatmap(r.grid, z.grid, pdf[:,:,is,itime_max], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
            outfile = string(run_name, "_pdf_vs_r_z", ivpa0_string, ivperp0_string, spec_string[is], ".pdf")
            trysavefig(outfile)
        end
    end
    println("done.")
end

"""
plots various slices of the neutral pdf (1d and 2d, stills and animations)
"""
function plot_neutral_pdf(run_name, vz, vr, vzeta, z, r, z_local, r_local, ivz0, ivr0,
                          ivzeta0, iz0, ir0, spec_type, n_species, n_time_pdfs, nblocks,
                          itime_min_pdfs, itime_max_pdfs, iskip_pdfs, nwrite_movie_pdfs,
                          pp)

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
    if n_species > 1
        spec_string = string("_", spec_type, "_spec", string(is))
    else
        spec_string = string("_", spec_type)
    end
    # make a gif animation of f(vz,z,t) at a given (vr,vzeta,r) location
    if pp.animate_f_vs_vz_z
        pdf = load_distributed_neutral_pdf_slice(run_name, nblocks,
                                                 itime_min_pdfs:iskip_pdfs:itime_max_pdfs,
                                                 n_species, r_local, z_local, vzeta, vr,
                                                 vz; ivr=ivr0, ivzeta=ivzeta0, ir=ir0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                @views heatmap(z.grid, vz.grid, pdf[:,:,is,i], xlabel="z", ylabel="vz",
                               c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vz_z", ivr0_string, ivzeta0_string, ir0_string, spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    # make a gif animation of f(vr,r,t) at a given (vz,vzeta,z) location
    if pp.animate_f_vs_vr_r
        pdf = load_distributed_neutral_pdf_slice(run_name, nblocks,
                                                 itime_min_pdfs:iskip_pdfs:itime_max_pdfs,
                                                 n_species, r_local, z_local, vzeta, vr,
                                                 vz; ivz=ivz0, ivzeta=ivzeta0, ir=ir0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                @views heatmap(r.grid, vr.grid, pdf[:,:,is,i], xlabel="r", ylabel="vr",
                               c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_vr_r", ivz0_string, ivzeta0_string, iz0_string, spec_string, ".gif")
            trygif(anim, outfile, fps=5)
        end
    end
    # make a gif animation of f(z,r,t) at a given (vz,vr,vzeta) location
    if pp.animate_f_vs_r_z
        pdf = load_distributed_neutral_pdf_slice(run_name, nblocks,
                                                 itime_min_pdfs:iskip_pdfs:itime_max_pdfs,
                                                 n_species, r_local, z_local, vzeta, vr,
                                                 vz; ivz=ivz0, ivr=ivr0, ivzeta=ivzeta0)
        for is ∈ 1:n_species
            anim = @animate for i ∈ itime_min_pdfs:nwrite_movie_pdfs:itime_max_pdfs
                @views heatmap(r.grid, z.grid, pdf[:,:,is,i], xlabel="r", ylabel="z",
                               c = :deep, interpolation = :cubic)
            end
            outfile = string(run_name, "_pdf_vs_z_r", ivz0_string, ivr0_string, ivzeta0_string, spec_string, ".gif")
            trygif(anim, outfile, fps=5)
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
        trysavefig(outfile)
    end
    if pp.animate_phi_vs_r_z && nr > 1
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views heatmap(r, z, phi[:,:,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
        end
        outfile = string(run_name, "_phi"*description*"_vs_r_z.gif")
        trygif(anim, outfile, fps=5)
    elseif pp.animate_phi_vs_r_z && nr == 1 # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(z, phi[:,1,i], xlabel="z", ylabel=L"\widetilde{\phi}", ylims = (phimin,phimax))
        end
        outfile = string(run_name, "_phi_vs_z.gif")
        trygif(anim, outfile, fps=5)
    end
    Ezmin = minimum(Ez)
    Ezmax = maximum(Ez)
    if pp.plot_Ez_vs_r0_z # plot last timestep Ez[z,ir0]
        @views plot(z, Ez[:,ir0,end], xlabel=L"z/L_z", ylabel=L"E_z")
        outfile = string(run_name, "_Ez"*description*"(r0,z)_vs_z.pdf")
        trysavefig(outfile)
    end
    if pp.plot_wall_Ez_vs_r && nr > 1 # plot last timestep Ez[z_wall,r]
        @views plot(r, Ez[end,:,end], xlabel=L"r/L_r", ylabel=L"E_z")
        outfile = string(run_name, "_Ez"*description*"(r,z_wall)_vs_r.pdf")
        trysavefig(outfile)
    end
    if pp.animate_Ez_vs_r_z && nr > 1
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views heatmap(r, z, Ez[:,:,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
        end
        outfile = string(run_name, "_Ez"*description*"_vs_r_z.gif")
        trygif(anim, outfile, fps=5)
    elseif pp.animate_Ez_vs_r_z && nr == 1
        Ezmin = minimum(Ez)
        Ezmax = maximum(Ez)
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views plot(z, Ez[:,1,i], xlabel="z", ylabel=L"\widetilde{E}_z", ylims = (Ezmin,Ezmax))
        end
        outfile = string(run_name, "_Ez_vs_z.gif")
        trygif(anim, outfile, fps=5)
    end
    Ermin = minimum(Er)
    Ermax = maximum(Er)
    if pp.plot_Er_vs_r0_z # plot last timestep Er[z,ir0]
        @views plot(z, Er[:,ir0,end], xlabel=L"z/L_z", ylabel=L"E_r")
        outfile = string(run_name, "_Er"*description*"(r0,z)_vs_z.pdf")
        trysavefig(outfile)
    end
    if pp.plot_wall_Er_vs_r && nr > 1 # plot last timestep Er[z_wall,r]
        @views plot(r, Er[end,:,end], xlabel=L"r/L_r", ylabel=L"E_r")
        outfile = string(run_name, "_Er"*description*"(r,z_wall)_vs_r.pdf")
        trysavefig(outfile)
    end
    if pp.animate_Er_vs_r_z && nr > 1
        # make a gif animation of ϕ(z) at different times
        anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
            @views heatmap(r, z, Er[:,:,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
        end
        outfile = string(run_name, "_Er"*description*"_vs_r_z.gif")
        trygif(anim, outfile, fps=5)
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
			trysavefig(outfile)
		end
		if pp.plot_wall_density_vs_r && nr > 1 # plot last timestep density[z_wall,r]
			@views plot(r, density[end,:,is,end], xlabel=L"r/L_r", ylabel=L"n_i")
			outfile = string(run_name, "_density"*description*"(r,z_wall)_vs_r.pdf")
			trysavefig(outfile)
		end
		if pp.animate_density_vs_r_z && nr > 1
			# make a gif animation of ϕ(z) at different times
			anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
				@views heatmap(r, z, density[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
			end
			outfile = string(run_name, "_density"*description*"_vs_r_z.gif")
			trygif(anim, outfile, fps=5)
		end
		if pp.plot_density_vs_r_z && nr > 1
			@views heatmap(r, z, density[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
			windowsize = (360,240), margin = 15pt)
			outfile = string(run_name, "_density"*description*"_vs_r_z.pdf")
			trysavefig(outfile)
		end
		
		# the parallel flow
		parallel_flowmin = minimum(parallel_flow[:,:,is,:])
		parallel_flowmax = maximum(parallel_flow)
		if pp.plot_parallel_flow_vs_r0_z # plot last timestep parallel_flow[z,ir0]
			@views plot(z, parallel_flow[:,ir0,is,end], xlabel=L"z/L_z", ylabel=L"u_{i\|\|}")
			outfile = string(run_name, "_parallel_flow"*description*"(r0,z)_vs_z.pdf")
			trysavefig(outfile)
		end
		if pp.plot_wall_parallel_flow_vs_r && nr > 1 # plot last timestep parallel_flow[z_wall,r]
			@views plot(r, parallel_flow[end,:,is,end], xlabel=L"r/L_r", ylabel=L"u_{i\|\|}")
			outfile = string(run_name, "_parallel_flow"*description*"(r,z_wall)_vs_r.pdf")
			trysavefig(outfile)
		end
		if pp.animate_parallel_flow_vs_r_z && nr > 1
			# make a gif animation of ϕ(z) at different times
			anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
				@views heatmap(r, z, parallel_flow[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
			end
			outfile = string(run_name, "_parallel_flow"*description*"_vs_r_z.gif")
			trygif(anim, outfile, fps=5)
		end
		if pp.plot_parallel_flow_vs_r_z && nr > 1
			@views heatmap(r, z, parallel_flow[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
			windowsize = (360,240), margin = 15pt)
			outfile = string(run_name, "_parallel_flow"*description*"_vs_r_z.pdf")
			trysavefig(outfile)
		end
		
		# the parallel pressure
		parallel_pressuremin = minimum(parallel_pressure[:,:,is,:])
		parallel_pressuremax = maximum(parallel_pressure)
		if pp.plot_parallel_pressure_vs_r0_z # plot last timestep parallel_pressure[z,ir0]
			@views plot(z, parallel_pressure[:,ir0,is,end], xlabel=L"z/L_z", ylabel=L"p_{i\|\|}")
			outfile = string(run_name, "_parallel_pressure"*description*"(r0,z)_vs_z.pdf")
			trysavefig(outfile)
		end
		if pp.plot_wall_parallel_pressure_vs_r && nr > 1 # plot last timestep parallel_pressure[z_wall,r]
			@views plot(r, parallel_pressure[end,:,is,end], xlabel=L"r/L_r", ylabel=L"p_{i\|\|}")
			outfile = string(run_name, "_parallel_pressure"*description*"(r,z_wall)_vs_r.pdf")
			trysavefig(outfile)
		end
		if pp.animate_parallel_pressure_vs_r_z && nr > 1
			# make a gif animation of ϕ(z) at different times
			anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
				@views heatmap(r, z, parallel_pressure[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
			end
			outfile = string(run_name, "_parallel_pressure"*description*"_vs_r_z.gif")
			trygif(anim, outfile, fps=5)
		end
		if pp.plot_parallel_pressure_vs_r_z && nr > 1
			@views heatmap(r, z, parallel_pressure[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
			windowsize = (360,240), margin = 15pt)
			outfile = string(run_name, "_parallel_pressure"*description*"_vs_r_z.pdf")
			trysavefig(outfile)
		end
                # the parallel temperature
                # Note factor of 2 here because currently temperatures are normalised by
                # Tref, while pressures are normalised by m*nref*c_ref^2=2*nref*Tref
                temperature = 2.0 * parallel_pressure ./ density
                if pp.plot_parallel_temperature_vs_r0_z # plot last timestep parallel_temperature[z,ir0]
                    @views plot(z, temperature[:,ir0,is,end], xlabel=L"z/L_z", ylabel=L"T_i")
                    outfile = string(run_name, "_temperature"*description*"(r0,z)_vs_z.pdf")
                    trysavefig(outfile)
                end
                if pp.plot_wall_parallel_temperature_vs_r && nr > 1 # plot last timestep parallel_temperature[z_wall,r]
                    @views plot(r, temperature[end,:,is,end], xlabel=L"r/L_r", ylabel=L"T_i")
                    outfile = string(run_name, "_temperature"*description*"(r,z_wall)_vs_r.pdf")
                    trysavefig(outfile)
                end
                if pp.animate_parallel_temperature_vs_r_z && nr > 1
                    # make a gif animation of T_||(z) at different times
                    anim = @animate for i ∈ itime_min:nwrite_movie:itime_max
                        @views heatmap(r, z, temperature[:,:,is,i], xlabel="r", ylabel="z", c = :deep, interpolation = :cubic)
                    end
                    outfile = string(run_name, "_temperature"*description*"_vs_r_z.gif")
                    trygif(anim, outfile, fps=5)
                end
                if pp.plot_parallel_temperature_vs_r_z && nr > 1
                    @views heatmap(r, z, temperature[:,:,is,end], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
                                  windowsize = (360,240), margin = 15pt)
                    outfile = string(run_name, "_temperature"*description*"_vs_r_z.pdf")
                    trysavefig(outfile)
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
    vpa, _ = load_coordinate_data(fid, "vpa", printout=false)
    vperp, _ = load_coordinate_data(fid, "vperp", printout=false)
    # load time data (unique to pdf, may differ to moment values depending on user nwrite_dfns value)
    ntime, time = load_time_data(fid, printout=false)
    # load species data
    n_ion_species, n_neutral_species = load_species_data(fid)
    close(fid)

    # plot only data at last timestep
    itime0 = ntime
    # plot a thermal vpa on line plots
    ivpa0 = floor(mk_int,vpa.n/3)
    # plot a thermal vperp on line plots
    ivperp0 = max(1,floor(mk_int,vperp.n/3))
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
            z, _ = load_coordinate_data(fid_pdfs, "z", printout=false)
            r, _ = load_coordinate_data(fid_pdfs, "r", printout=false)

            if z_irank == 0
                iz_wall = 1
                #print("z.grid[iz_wall-]: ",z.grid[iz_wall])
                zlabel = "wall-"
            elseif z_irank == z_nrank-1
                iz_wall = z.n
                #print("z.grid[iz_wall+]: ",z.grid[iz_wall])
                zlabel = "wall+"
            end
            # load local pdf data
            pdf = load_pdf_data(fid_pdfs, printout=false)
            for is in 1:n_ion_species
                description = "_ion_spec"*string(is)*"_"

                # plot f(vpa,ivperp0,iz_wall,ir0,is,itime) at the wall
                @views plot(vpa.grid, pdf[:,ivperp0,iz_wall,ir0,is,itime0], xlabel=L"v_{\|\|}/L_{v_{\|\|}}", ylabel=L"f_i")
                outfile = string(run_name, "_pdf(vpa,vperp0,iz_"*zlabel*",ir0)"*description*"_vs_vpa.pdf")
                trysavefig(outfile)

                # plot f(vpa,vperp,iz_wall,ir0,is,itime) at the wall
                @views heatmap(vperp.grid, vpa.grid, pdf[:,:,iz_wall,ir0,is,itime0], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(run_name, "_pdf(vpa,vperp,iz_"*zlabel*",ir0)"*description*"_vs_vperp_vpa.pdf")
                trysavefig(outfile)

                # plot f(vpa,ivperp0,z,ir0,is,itime) near the wall
                @views heatmap(z.grid, vpa.grid, pdf[:,ivperp0,:,ir0,is,itime0], xlabel=L"z", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(run_name, "_pdf(vpa,ivperp0,z_"*zlabel*",ir0)"*description*"_vs_z_vpa.pdf")
                trysavefig(outfile)

                # plot f(ivpa0,ivperp0,z,r,is,itime) near the wall
                if r.n > 1
                    @views heatmap(r.grid, z.grid, pdf[ivpa0,ivperp0,:,:,is,itime0], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                    outfile = string(run_name, "_pdf(ivpa0,ivperp0,z_"*zlabel*",r)"*description*"_vs_r_z.pdf")
                    trysavefig(outfile)
                    @views heatmap(r_local, vpa.grid, pdf[:,ivperp0,iz_wall,:,is,itime0], xlabel=L"r", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                    outfile = string(run_name, "_pdf(vpa,ivperp0,z_"*zlabel*",r)"*description*"_vs_r_vpa.pdf")
                    trysavefig(outfile)
                end
            end
        end
        close(fid_pdfs)
    end
    println("done.")
end

end
