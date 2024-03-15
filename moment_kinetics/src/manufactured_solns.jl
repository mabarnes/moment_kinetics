"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_electric_fields
export manufactured_geometry

using ..array_allocation: allocate_shared_float
using ..looping
using ..type_definitions: mk_float, mk_int

function manufactured_solutions end
function manufactured_sources_setup end
function manufactured_electric_fields end
function manufactured_geometry end

function __init__()
    try
        # Try to load the Symbolics package so we can use it for manufactured solutions.
        # If the package is not installed, then manufactured solutions will not be
        # available.
        Base.require(Main, :Symbolics)
    catch
        # Do nothing
    end
end

# This function is defined here rather than in the extension, because the looping macros
# break if used outside the moment_kinetics module.
function manufactured_sources(manufactured_solns_input, r_coord, z_coord, vperp_coord,
            vpa_coord, vzeta_coord, vr_coord, vz_coord, composition, geometry, collisions,
            num_diss_params, species)

    time_independent_sources, Source_i_func, Source_n_func =
        manufactured_sources_setup(manufactured_solns_input, r_coord, z_coord,
            vperp_coord, vpa_coord, vzeta_coord, vr_coord, vz_coord, composition,
            geometry, collisions, num_diss_params, species)

    if time_independent_sources
        # Time independent, so store arrays instead of functions

        Source_i_array = allocate_shared_float(vpa_coord.n,vperp_coord.n,z_coord.n,r_coord.n)
        begin_s_r_z_region()
        #println("here loop thing ", looping.loop_ranges[].s)
        @loop_s is begin
            if is == 1
                @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                    Source_i_array[ivpa,ivperp,iz,ir,is] = Source_i_func(vpa_coord.grid[ivpa],vperp_coord.grid[ivperp],z_coord.grid[iz],r_coord.grid[ir],0.0)
                end
            end
        end

        if composition.n_neutral_species > 0
            Source_n_array = allocate_shared_float(vz_coord.n,vr_coord.n,vzeta_coord.n,z_coord.n,r_coord.n)
            begin_sn_r_z_region()
            @loop_sn isn begin
                if isn == 1
                    @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
                        Source_n_array[ivz,ivr,ivzeta,iz,ir,isn] = Source_n_func(vz_coord.grid[ivz],vr_coord.grid[ivr],vzeta_coord.grid[ivzeta],z_coord.grid[iz],r_coord.grid[ir],0.0)
                    end
                end
            end
        else
            Source_n_array = zeros(mk_float,0)
        end

        manufactured_sources_list = (time_independent_sources = true, Source_i_array = Source_i_array, Source_n_array = Source_n_array)
    else
        manufactured_sources_list = (time_independent_sources = false, Source_i_func = Source_i_func, Source_n_func = Source_n_func)
    end

    return manufactured_sources_list
end

end
