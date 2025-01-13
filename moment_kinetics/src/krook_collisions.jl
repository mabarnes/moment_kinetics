"""
"""
module krook_collisions

export setup_krook_collisions_input, krook_collisions!, electron_krook_collisions!,
       add_electron_krook_collisions_to_Jacobian!

using ..looping
using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..input_structs: krook_collisions_input, set_defaults_and_check_section!
using ..timer_utils 
using ..collision_frequencies
using ..reference_parameters
using OrderedCollections: OrderedDict

"""
Function for reading Krook collision operator input parameters. 
Structure the namelist as follows.

[krook_collisions]
use_krook = true
nuii0 = 1.0
frequency_option = "manual"
"""
function setup_krook_collisions_input(toml_input::AbstractDict)
    reference_params = setup_reference_parameters(toml_input)
    # get reference collision frequency
    nuii_krook_default = get_reference_collision_frequency_ii(reference_params)
    nuee_krook_default = get_reference_collision_frequency_ee(reference_params)
    nuei_krook_default = get_reference_collision_frequency_ei(reference_params)
    # read the input toml and specify a sensible default    
    input_section = input_section = set_defaults_and_check_section!(toml_input, "krook_collisions",
       # begin default inputs (as kwargs)
       use_krook = false,
       nuii0 = -1.0,
       nuee0 = -1.0,
       nuei0 = -1.0,
       frequency_option = "reference_parameters")
       
    # ensure that the collision frequency is consistent with the input option
    frequency_option = input_section["frequency_option"]
    if frequency_option == "reference_parameters"
        input_section["nuii0"] = nuii_krook_default
        input_section["nuee0"] = nuee_krook_default
        input_section["nuei0"] = nuei_krook_default
    elseif frequency_option == "collisionality_scan"
        input_section["nuii0"] *= nuii_krook_default
        input_section["nuee0"] *= nuee_krook_default
        input_section["nuei0"] *= nuei_krook_default
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
        input_section["nuii0"] = -1.0
        input_section["nuee0"] = -1.0
        input_section["nuei0"] = -1.0
    end
    input = OrderedDict(Symbol(k)=>v for (k,v) in input_section)
    #println(input)
    return krook_collisions_input(; input...)
end


"""
Add collision operator

Currently Krook collisions
"""
@timeit global_timer krook_collisions!(
                         pdf_out, fvec_in, moments, composition, collisions, vperp, vpa,
                         dt) = begin
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
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
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
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is]))^2
                           - (vperp.grid[ivperp])^2))
            end
        end
    elseif moments.evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
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
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
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
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
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

"""
Add Krook collision operator for electrons

Note that this function operates on a single point in `r`, so `pdf_out`, `pdf_in`,
`dens_in`, `upar_in`, `upar_ion_in`, and `vth_in` should have no r-dimension.
"""
@timeit global_timer electron_krook_collisions!(
                         pdf_out, pdf_in, dens_in, upar_in, upar_ion_in, vth_in,
                         collisions, vperp, vpa, dt) = begin
    begin_z_region()

    # For now, electrons are always fully moment-kinetic
    evolve_density = true
    evolve_upar = true
    evolve_ppar = true

    if vperp.n > 1 && (evolve_density || evolve_upar || evolve_ppar)
        error("Krook collisions not implemented for 2V moment-kinetic cases yet")
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).
    if evolve_ppar && evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            # e-i collisions push electrons towards a Maxwellian drifting at the ion
            # parallel flow, so need a corresponding normalised parallel velocity
            # coordinate.
            # For now, assume there is only one ion species rather than bothering to
            # calculate an average ion flow speed, or sum over ion species here.
            @. vpa.scratch = vpa.grid + (upar_ion_in[iz,1] - upar_in[iz]) / vth

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2))
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - exp(-vpa.scratch[ivpa]^2 - vperp.grid[ivperp]^2))
                   )
            end
        end
    elseif evolve_ppar
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - exp(-((vpa.grid[ivpa] - upar_in[iz])/vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - exp(-((vpa.grid[ivpa] - upar_ion_in[iz,1])/vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    elseif evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            # e-i collisions push electrons towards a Maxwellian drifting at the ion
            # parallel flow, so need a corresponding normalised parallel velocity
            # coordinate.
            # For now, assume there is only one ion species rather than bothering to
            # calculate an average ion flow speed, or sum over ion species here.
            @. vpa.scratch = vpa.grid + (upar_ion_in[iz,1] - upar_in[iz])

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - 1.0 / vth * exp(-(vpa.grid[ivpa] / vth)^2
                                               - (vperp.grid[ivperp] / vth)^2))
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - 1.0 / vth * exp(-(vpa.scratch[ivpa] / vth)^2
                                                 - (vperp.grid[ivperp] / vth)^2))
                   )
            end
        end
    elseif evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - 1.0 / vth
                             * exp(-((vpa.grid[ivpa] - upar_in[iz]) / vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - 1.0 / vth
                               * exp(-((vpa.grid[ivpa] - upar_ion_in[iz,1]) / vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    else
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - n * vth_prefactor
                             * exp(-((vpa.grid[ivpa] - upar_in[iz])/vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ee * (pdf_in[ivpa,ivperp,iz]
                               - n * vth_prefactor
                               * exp(-((vpa.grid[ivpa] - upar_ion_in[iz,1])/vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    end

    return nothing
end

function add_electron_krook_collisions_to_Jacobian!(jacobian_matrix, f, dens, upar, ppar,
                                                    vth, upar_ion, collisions, z, vperp,
                                                    vpa, z_speed, dt, ir, include=:all;
                                                    f_offset=0, ppar_offset)
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @boundscheck include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    if collisions.krook.nuee0 ≤ 0.0 && collisions.krook.nuei0 ≤ 0.0
        return nothing
    end

    v_size = vperp.n * vpa.n

    using_reference_parameters = (collisions.krook.frequency_option == "reference_parameters")

    begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

        # Contribution from electron_krook_collisions!()
        nu_ee = get_collision_frequency_ee(collisions, dens[iz], vth[iz])
        nu_ei = get_collision_frequency_ei(collisions, dens[iz], vth[iz])
        if include === :all
            jacobian_matrix[row,row] += dt * (nu_ee + nu_ei)
        end

        if include ∈ (:all, :explicit_v)
            fM_i = exp(-(vpa.grid[ivpa] + (upar_ion[iz] - upar[iz])/vth[iz])^2 - vperp.grid[ivperp]^2)
            #   d(f_M(u_i)[irowz])/d(ppar[icolz])
            #       = -2*(vpa.grid+(upar_ion-upar)/vth)*(upar_ion-upar)*(-1/2/vth/ppar)*f_M(u_i) * delta(irow,icolz)
            #       = (vpa.grid+(upar_ion-upar)/vth)*(upar_ion-upar)/vth/ppar*f_M(u_i) * delta(irow,icolz)
            jacobian_matrix[row,ppar_offset+iz] +=
                -dt * nu_ei * (vpa.grid[ivpa]+(upar_ion[iz]-upar[iz])/vth[iz])*(upar_ion[iz]-upar[iz])/vth[iz]/ppar[iz]*fM_i

            if using_reference_parameters
                # Both collision frequencies are proportional to n/vth^3=n^(5/2)*(me/2/p)^3/2,
                # so
                #   d(nu[irowz])/d(ppar[icolz]) = -3/2*nu/ppar * delta(irowz,icolz)
                #   d(-(vpa.grid+(upar_ion-upar)/vth)^2[irowz])/d(ppar[icoliz]
                #       = -(vpa.grid+(upar_ion-upar)/vth)*(upar_ion-upar)/vth/ppar * delta(irow,icolz)
                jacobian_matrix[row,ppar_offset+iz] +=
                    -dt * 1.5 / ppar[iz] *
                          (nu_ee * (f[ivpa,ivperp,iz] - exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2))
                           + nu_ei * (f[ivpa,ivperp,iz] - fM_i))
            end
        end
    end

    return nothing
end

function add_electron_krook_collisions_to_z_only_Jacobian!(
        jacobian_matrix, f, dens, upar, ppar, vth, upar_ion, collisions, z, vperp, vpa,
        z_speed, dt, ir, ivperp, ivpa)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == z.n || error("Jacobian matrix size is wrong")

    if collisions.krook.nuee0 ≤ 0.0 && collisions.krook.nuei0 ≤ 0.0
        return nothing
    end

    @loop_z iz begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = iz

        # Contribution from electron_krook_collisions!()
        nu_ee = get_collision_frequency_ee(collisions, dens[iz], vth[iz])
        nu_ei = get_collision_frequency_ei(collisions, dens[iz], vth[iz])
        jacobian_matrix[row,row] += dt * (nu_ee + nu_ei)
    end

    return nothing
end

function add_electron_krook_collisions_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, ppar, vth, upar_ion, collisions, z, vperp, vpa,
        z_speed, dt, ir, iz)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    if collisions.krook.nuee0 ≤ 0.0 && collisions.krook.nuei0 ≤ 0.0
        return nothing
    end

    using_reference_parameters = (collisions.krook.frequency_option == "reference_parameters")

    @loop_vperp_vpa ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (ivperp - 1) * vpa.n + ivpa

        # Contribution from electron_krook_collisions!()
        nu_ee = get_collision_frequency_ee(collisions, dens, vth)
        nu_ei = get_collision_frequency_ei(collisions, dens, vth)
        jacobian_matrix[row,row] += dt * (nu_ee + nu_ei)

        fM_i = exp(-(vpa.grid[ivpa] + (upar_ion - upar)/vth)^2 - vperp.grid[ivperp]^2)
        jacobian_matrix[row,end] +=
            -dt * nu_ei * (vpa.grid[ivpa]+(upar_ion-upar)/vth)*(upar_ion-upar)/vth/ppar*fM_i

        if using_reference_parameters
            jacobian_matrix[row,end] +=
                -dt * 1.5 / ppar *
                      (nu_ee * (f[ivpa,ivperp] - exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2))
                       + nu_ei * (f[ivpa,ivperp] - fM_i))
        end
    end

    return nothing
end

end # krook_collisions
