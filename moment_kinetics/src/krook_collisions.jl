"""
"""
module krook_collisions

export setup_krook_collisions_input, get_collision_frequency, krook_collisions!

using ..constants: epsilon0, proton_charge
using ..looping
using ..input_structs: krook_collisions_input
using ..reference_parameters: get_reference_collision_frequency

function setup_krook_collisions_input(toml_input::Dict, reference_params)
    # get reference collision frequency
    nuii_krook_default = get_reference_collision_frequency(reference_params)
    # read the input toml and specify a sensible default    
    default_dict = Dict{String,Any}("use_krook" => false, "krook_collision_frequency_prefactor" => -1.0,
          "frequency_option" => "reference_parameters" )
    input_section = get(toml_input, "krook_collisions", default_dict)
    # ensure defaults are carried over from the default dict if only a partial dict is given as an input
    # as the default Dict here is entirely ignored if the namelist is present    
    for (k,v) in default_dict
        if !( k in keys(input_section))
            input_section[k] = v
        end
    end
    # ensure that the collision frequency is consistent with the input option
    frequency_option = input_section["frequency_option"]
    if frequency_option == "reference_parameters"
        input_section["krook_collision_frequency_prefactor"] = nuii_krook_default
    elseif frequency_option == "manual" 
        # use the frequency from the input file
        # do nothing
    else
        error("Invalid option [krook_collisions] "
              * "frequency_option=$(frequency_option) passed")
    end
    # finally, ensure prefactor < 0 if use_krook is false
    # so that prefactor > 0 is the only check required in the rest of the code
    if !input_section["use_krook"]
        input_section["krook_collision_frequency_prefactor"] = -1.0
    end
    input = Dict(Symbol(k)=>v for (k,v) in input_section)
    #println(input)
    return krook_collisions_input(; input...)
end

"""
    get_collision_frequency(collisions, n, vth)

Calculate the collision frequency, depending on the settings/parameters in `collisions`,
for the given density `n` and thermal speed `vth`.

`n` and `vth` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency(collisions, n, vth)
    # extract krook options from collisions struct
    colk = collisions.krook
    krook_collision_frequency_prefactor = colk.krook_collision_frequency_prefactor
    frequency_option = colk.frequency_option
    if frequency_option == "reference_parameters"
        return @. krook_collision_frequency_prefactor * n * vth^(-3)
    elseif frequency_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. krook_collision_frequency_prefactor + 0.0 * n
    else
        error("Unrecognised option [krook_collisions] "
              * "frequency_option=$(frequency_option)")
    end
end

"""
Add collision operator

Currently Krook collisions
"""
function krook_collisions!(pdf_out, fvec_in, moments, composition, collisions, vperp, vpa, dt)
    begin_s_r_z_region()

    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Krook collisions not implemented for 2V moment-kinetic cases yet")
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).
    if moments.evolve_ppar && moments.evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2))
            end
        end
    elseif moments.evolve_ppar
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/vth)^2
                           - (vperp.grid[ivperp]/vth)^2))
            end
        end
    elseif moments.evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - 1.0 / vth * exp(-(vpa.grid[ivpa] / vth)^2
                                       - (vperp.grid[ivperp] / vth)^2))
            end
        end
    elseif moments.evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                 - 1.0 / vth
                 * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is]) / vth)^2
                           - (vperp.grid[ivperp]/vth)^2))
            end
        end
    else
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.charged.vth[iz,ir,is]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ii = get_collision_frequency(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - n * vth_prefactor
                     * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/vth)^2
                           - (vperp.grid[ivperp]/vth)^2))
            end
        end
    end

    return nothing
end

end # krook_collisions
