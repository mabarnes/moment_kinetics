"""
Model Diffusion operator - this will act to push the distribution towards
Maxwellian, but purely based on velocity gradients. So need the second derivative of 
the distribution function. Integration constants should not be a problem, i.e. pushing 
the second derivatives of a Maxwellian and our function together should also move their 
values towards each other, as the boundary conditions of both would be the same. 

This operator will mostly come into effect in places where there is ringing in the 
distribution, which can be expected in the grid points near the walls. Fortunately this
region is also where the plasma is most collisional, so having such an operator is also
most valid here.
"""

module maxwell_diffusion

export setup_mxwl_diff_collisions_input, ion_vpa_maxwell_diffusion!, neutral_vz_maxwell_diffusion!, implicit_ion_maxwell_diffusion!

using ..looping
using ..input_structs: mxwl_diff_collisions_input, set_defaults_and_check_section!
using ..array_allocation: allocate_float
using ..boundary_conditions: enforce_v_boundary_condition_local!, vpagrid_to_dzdt
using ..calculus: derivative!, second_derivative!
using ..moment_constraints: moment_constraints_on_residual!
using ..moment_kinetics_structs: scratch_pdf
using ..nonlinear_solvers: newton_solve!
using ..reference_parameters: get_reference_collision_frequency_ii
using ..velocity_moments: update_derived_moments!, calculate_ion_moment_derivatives!

using LinearAlgebra
using SparseArrays

"""
Function for reading Maxwell diffusion operator input parameters. 
Structure the namelist as follows.

[maxwell_diffusion_collisions]
use_maxwell_diffusion = true
D_ii = 1.0
diffusion_coefficient_option = "manual"
"""
function setup_mxwl_diff_collisions_input(toml_input::Dict, reference_params)
    # get reference diffusion coefficient, made up of collision frequency and 
    # thermal speed for now. NOTE THAT THIS CONSTANT PRODUCES ERRORS. DO NOT USE
    D_ii_mxwl_diff_default = get_reference_collision_frequency_ii(reference_params)# *
                             #2 * reference_params.Tref/reference_params.mref
    D_nn_mxwl_diff_default = D_ii_mxwl_diff_default
    # read the input toml and specify a sensible default    
    input_section = set_defaults_and_check_section!(toml_input, "maxwell_diffusion_collisions",
       # begin default inputs (as kwargs)
       use_maxwell_diffusion = false,
       D_ii = -1.0,
       D_nn = -1.0,
       diffusion_coefficient_option = "reference_parameters")
       
    # ensure that the diffusion coefficient is consistent with the input option
    diffusion_coefficient_option = input_section["diffusion_coefficient_option"]
    if diffusion_coefficient_option == "reference_parameters"
        input_section["D_ii"] = D_ii_mxwl_diff_default
        input_section["D_nn"] = -1.0 #D_nn_mxwl_diff_default
    elseif diffusion_coefficient_option == "manual" 
        # use the diffusion coefficient from the input file
        # do nothing
    else
        error("Invalid option [maxwell_diffusion_collisions] "
              * "diffusion_coefficient_option=$(diffusion_coefficient_option) passed")
    end
    # finally, ensure prefactor < 0 if use_maxwell_diffusion is false
    # so that prefactor > 0 is the only check required in the rest of the code
    if !input_section["use_maxwell_diffusion"]
        input_section["D_ii"] = -1.0
        input_section["D_nn"] = -1.0
    end
    input = Dict(Symbol(k)=>v for (k,v) in input_section)

    return mxwl_diff_collisions_input(; input...)
end

function ion_vpa_maxwell_diffusion_inner!(f_out, f_in, n, upar, vth, vpa, spectral,
                                          diffusion_coefficient, dt, ::Val{false},
                                          ::Val{false}, ::Val{false})
    second_derivative!(vpa.scratch2, f_in, vpa, spectral)
    @. vpa.scratch = (vpa.grid - upar) * f_in
    derivative!(vpa.scratch3, vpa.scratch, vpa, spectral)
    @. f_out += dt * diffusion_coefficient * n / vth^3 * (0.5 * vth^2 * vpa.scratch2 + vpa.scratch3)
end

function ion_vpa_maxwell_diffusion_inner!(f_out, f_in, n, upar, vth, vpa, spectral,
                                          diffusion_coefficient, dt, ::Val{true},
                                          ::Val{true}, ::Val{true})
    second_derivative!(vpa.scratch2, f_in, vpa, spectral)
    @. vpa.scratch = vpa.grid * f_in
    derivative!(vpa.scratch3, vpa.scratch, vpa, spectral)
    @. f_out += dt * diffusion_coefficient * n / vth^3 * (vpa.scratch2 + vpa.scratch3)
end

"""
Calculate the Maxwellian associated with the current ion pdf moments, and then 
subtract this from current pdf. Then take second derivative of this function
to act as the diffusion operator. 
"""
function ion_vpa_maxwell_diffusion!(f_out, f_in, moments, vpa, vperp, spectral, dt,
                                    diffusion_coefficient)
    
    # If negative input (should be -1.0), then none of this diffusion will happen. 
    # This number can be put in as some parameter in the input file called something
    # like 'maxwellian_diffusion_coefficient'
    if diffusion_coefficient <= 0.0 || vpa.n == 1
        return nothing
    end

    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Maxwell diffusion not implemented for 2V moment-kinetic cases yet")
    end

    evolve_density = Val(moments.evolve_density)
    evolve_upar = Val(moments.evolve_upar)
    evolve_ppar = Val(moments.evolve_ppar)

    # Otherwise, build the maxwellian function (which is going to be subtracted from 
    # the current distribution) using the moments of the distribution (so that the 
    # operator itself conserves the moments), and then this result will be the one 
    # whose second derivative will be added to the RHS (i.e. subtracted from the current)
    begin_s_r_z_vperp_region()

    # In what follows, there are eight combinations of booleans (though not all have been
    # fully implemented yet). In line with moment kinetics, the Maxwellian is normalised
    # in the relevant ways:
    # - density: normalise by n 
    # - upar: working in peculiar velocity space, so no upar subtraction from vpa 
    # - ppar: normalisation by vth, in 1D is 1/vth prefactor, and grid is normalised by vth,
    # hence no 1/vth^2 term in the exponent.
    @loop_s_r_z_vperp is ir iz ivperp begin
        #@views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
        #                exp(-((vpa.grid[:])^2 + (vperp.grid[ivperp])^2) )
        #second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
        #@views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        @views ion_vpa_maxwell_diffusion_inner!(f_out[:,ivperp,iz,ir,is],
                                                f_in.pdf[:,ivperp,iz,ir,is],
                                                f_in.density[iz,ir,is],
                                                f_in.upar[iz,ir,is],
                                                moments.ion.vth[iz,ir,is], vpa, spectral,
                                                diffusion_coefficient, dt, evolve_density,
                                                evolve_upar, evolve_ppar)
    end
    #elseif moments.evolve_density && moments.evolve_upar
    #    @loop_s_r_z_vperp is ir iz ivperp begin
    #        vth = moments.ion.vth[iz,ir,is]
    #        @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
    #                        1.0 / vth * exp(- ((vpa.grid[:])^2 + (vperp.grid[ivperp])^2)/(vth^2) )
    #        second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
    #        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
    #    end
    #    error("hack not implemented for this case")
    #elseif moments.evolve_density && moments.evolve_ppar
    #    @loop_s_r_z_vperp is ir iz ivperp begin
    #        upar = f_in.upar[iz,ir,is]
    #        @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
    #                        exp(- ((vpa.grid[:] - upar)^2 + (vperp.grid[ivperp])^2))
    #        second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
    #        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
    #    end
    #    error("hack not implemented for this case")
    #elseif moments.evolve_upar && moments.evolve_ppar
    #    @loop_s_r_z_vperp is ir iz ivperp begin
    #        n = f_in.density[iz,ir,is]
    #        @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
    #                        n * exp(- ((vpa.grid[:])^2 + (vperp.grid[ivperp])^2) )
    #        second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
    #        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
    #    end
    #    error("hack not implemented for this case")
    #elseif moments.evolve_density
    #    @loop_s_r_z_vperp is ir iz ivperp begin
    #        vth = moments.ion.vth[iz,ir,is]
    #        upar = f_in.upar[iz,ir,is]
    #        @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
    #                        1.0 / vth * exp(- ((vpa.grid[:] - upar)^2 + (vperp.grid[ivperp])^2)/(vth^2) )
    #        second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
    #        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
    #    end
    #    error("hack not implemented for this case")
    #elseif moments.evolve_upar
    #    @loop_s_r_z_vperp is ir iz ivperp begin
    #        vth = moments.ion.vth[iz,ir,is]
    #        n = f_in.density[iz,ir,is]
    #        @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
    #                        n / vth * exp(- ((vpa.grid[:])^2 + (vperp.grid[ivperp])^2)/(vth^2) )
    #        second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
    #        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
    #    end
    #    error("hack not implemented for this case")
    #elseif moments.evolve_ppar
    #    @loop_s_r_z_vperp is ir iz ivperp begin
    #        n = f_in.density[iz,ir,is]
    #        upar = f_in.upar[iz,ir,is]
    #        @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
    #                        n * exp(- ((vpa.grid[:] - upar)^2 + (vperp.grid[ivperp])^2) )
    #        second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
    #        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
    #    end
    #    error("hack not implemented for this case")
    return nothing
end

"""
Calculate the Maxwellian associated with the current neutral pdf moments, and then 
subtract this from current pdf. Then take second derivative of this function
to act as the diffusion operator. 
"""
function neutral_vz_maxwell_diffusion!(f_out, f_in, moments, vzeta, vr, vz, spectral::T_spectral, 
                                       dt, diffusion_coefficient) where T_spectral
    
    # If negative input (should be -1.0), then none of this diffusion will happen. 
    # This number can be put in as some parameter in the input file called something
    # like 'maxwellian_diffusion_coefficient'
    if diffusion_coefficient <= 0.0 || vz.n == 1
        return nothing
    end

    if vr.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Maxwell diffusion not implemented for 2V moment-kinetic cases yet")
    end

    # Otherwise, build the maxwellian function (which is going to be subtracted from 
    # the current distribution) using the moments of the distribution (so that the 
    # operator itself conserves the moments), and then this result will be the one 
    # whose second derivative will be added to the RHS (i.e. subtracted from the current pdf)
    begin_sn_r_z_vzeta_vr_region()


    if moments.evolve_ppar && moments.evolve_upar
        # See similar comments in krook_collisions! function. 
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            exp(-((vz.grid[:])^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    elseif moments.evolve_ppar
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            vth = moments.neutral.vth[iz,ir,isn]
            uz = f_in.uz_neutral[iz,ir,isn]
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            exp(- ((vz.grid[:] - uz)^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    elseif moments.evolve_upar
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            vth = moments.neutral.vth[iz,ir,isn]
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            1.0 / vth * exp(- ((vz.grid[:])^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    elseif moments.evolve_density
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            vth = moments.neutral.vth[iz,ir,isn]
            uz = f_in.uz_neutral[iz,ir,isn]
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            1.0 / vth * exp(- ((vz.grid[:] - uz)^2 + 
                                            (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    else 
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            n = f_in.density_neutral[iz,ir,isn]
            uz = f_in.uz_neutral[iz,ir,isn]
            vth = moments.neutral.vth[iz,ir,isn]
            if vr.n == 1 && vzeta.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - n * vth_prefactor * 
                            exp(-( (vz.grid[:] - uz)^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    end
    return nothing
end

"""
"""
function implicit_ion_maxwell_diffusion!(f_out, fvec_in, moments, z_advect, vpa, vperp, z,
                                         r, dt, r_spectral, vpa_spectral, composition,
                                         collisions, geometry, nl_solver_params, gyroavs,
                                         scratch_dummy)
    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Moment constraints in implicit_maxwell_diffusion!() do not support 2V runs yet")
    end

    # Ensure moments are consistent with f_new
    new_scratch = scratch_pdf(f_out, fvec_in.density, fvec_in.upar, fvec_in.ppar,
                              fvec_in.pperp, fvec_in.temp_z_s, fvec_in.electron_density,
                              fvec_in.electron_upar, fvec_in.electron_ppar,
                              fvec_in.electron_pperp, fvec_in.electron_temp,
                              fvec_in.pdf_neutral, fvec_in.density_neutral,
                              fvec_in.uz_neutral, fvec_in.pz_neutral)
    update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition,
                            r_spectral, geometry, gyroavs, scratch_dummy, z_advect, false)

    begin_s_r_z_vperp_region()

    evolve_density = Val(moments.evolve_density)
    evolve_upar = Val(moments.evolve_upar)
    evolve_ppar = Val(moments.evolve_ppar)
    coords = (vpa=vpa,)
    vpa_bc = vpa.bc
    D_ii = collisions.mxwl_diff.D_ii
    zero = 1.0e-14

    function get_precon(prefactor; icut_lower=2, icut_upper=vpa.n-1)
        if !(isa(evolve_density, Val{true}) && isa(evolve_upar, Val{true}) &&
             isa(evolve_ppar, Val{true}))
            error("maxwell_diffusion preconditioner is only implemented for fully "
                  * "moment-kinetic case")
        end
        if icut_lower < 2
            icut_lower = 2
        end
        if icut_upper > vpa.n - 1
            icut_upper = vpa.n - 1
        end

        # Dirichlet boundary conditions set the first and last values of the solution
        # to zero, so can remove the first/last rows/columns of the matrix.
        # When there is a 'cutoff index' because we are imposing sheath-edge boundary
        # conditions, more values (all those outside the `icut` index) are
        # zero-ed out, and so removed from the matrix system.
        precon_matrix = allocate_float(vpa.n-2, vpa.n-2)
        precon_matrix .= 0.0
        for i ∈ 1:vpa.nelement_local
            imin = vpa.imin[i] - (i != 1)
            if i != 1
                # Remove first row/column
                imin -= 1
            end
            imax = vpa.imax[i]
            if i < vpa.nelement_local
                # Remove first row/column
                imax -= 1
            else
                # Remove first and last row/column
                imax -= 2
            end
            if i == 1 && i == vpa.nelement_local
                @. precon_matrix += vpa_spectral.lobatto.Dmat[2:end-1,2:end-1] / vpa.element_scale[i]
            elseif i == 1
                @. precon_matrix[imin:imax-1,imin:imax] += vpa_spectral.lobatto.Dmat[2:end-1,2:end] / vpa.element_scale[i]
                @. precon_matrix[imax,imin:imax] += 0.5 * vpa_spectral.lobatto.Dmat[end,2:end] / vpa.element_scale[i]
            elseif i == vpa.nelement_local
                @. precon_matrix[imin,imin:imax] += 0.5 .* vpa_spectral.lobatto.Dmat[1,1:end-1] / vpa.element_scale[i]
                @. precon_matrix[imin+1:imax,imin:imax] += vpa_spectral.lobatto.Dmat[2:end-1,1:end-1] / vpa.element_scale[i]
            else
                @. precon_matrix[imin,imin:imax] += 0.5 * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i]
                @. precon_matrix[imin+1:imax-1,imin:imax] += vpa_spectral.lobatto.Dmat[2:end-1,:] / vpa.element_scale[i]
                @. precon_matrix[imax,imin:imax] += 0.5 * vpa_spectral.lobatto.Dmat[end,:] / vpa.element_scale[i]
            end
        end
        # Right-multiply by w_∥
        for i ∈ 1:vpa.n-2
            precon_matrix[:,i] .*= vpa.grid[i+1]
        end

        ## This allocates a new matrix - to avoid this would need to pre-allocate a
        ## suitable buffer somewhere.
        #precon_matrix .+= inv(@view vpa_spectral.mass_matrix[2:end-1,2:end-1]) *
        #                  vpa_spectral.K_matrix[2:end-1,2:end-1]

        #precon_matrix = @view precon_matrix[icut_lower-1:icut_upper-1,icut_lower-1:icut_upper-1]

        #precon_matrix .= Diagonal(ones(icut_upper - icut_lower + 1)) .- prefactor .* precon_matrix

        #precon_lu = lu(precon_matrix)

        # This allocates a new matrix - to avoid this would need to pre-allocate a
        # suitable buffer somewhere.
        precon_matrix .= vpa_spectral.mass_matrix[2:end-1,2:end-1] * precon_matrix
        precon_matrix .+= vpa_spectral.K_matrix[2:end-1,2:end-1]

        precon_matrix = @view precon_matrix[icut_lower-1:icut_upper-1,icut_lower-1:icut_upper-1]

        @views precon_matrix .=
            vpa_spectral.mass_matrix[icut_lower:icut_upper,icut_lower:icut_upper] .-
            prefactor .* precon_matrix

        precon_lu = lu(sparse(precon_matrix))

        return precon_lu
    end

    @loop_s is begin
        @loop_r_z_vperp ir iz ivperp begin
            f_old_no_bc = @view fvec_in.pdf[:,ivperp,iz,ir,is]
            this_f_out = @view f_out[:,ivperp,iz,ir,is]
            n = fvec_in.density[iz,ir,is]
            upar = fvec_in.upar[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]

            if z.irank == 0 && iz == 1
                @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, moments.ion.vth[iz,ir,is],
                                                 fvec_in.upar[iz,ir,is],
                                                 moments.evolve_ppar,
                                                 moments.evolve_upar)
                icut_lower_z = vpa.n
                for ivpa ∈ vpa.n:-1:1
                    # for left boundary in zed (z = -Lz/2), want
                    # f(z=-Lz/2, v_parallel > 0) = 0
                    if vpa.scratch[ivpa] < -zero
                        icut_lower_z = ivpa + 1
                        break
                    end
                end
            end
            if z.irank == z.nrank - 1 && iz == z.n
                @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, moments.ion.vth[iz,ir,is],
                                                 fvec_in.upar[iz,ir,is],
                                                 moments.evolve_ppar,
                                                 moments.evolve_upar)
                icut_upper_z = 0
                for ivpa ∈ 1:vpa.n
                    # for right boundary in zed (z = Lz/2), want
                    # f(z=Lz/2, v_parallel < 0) = 0
                    if vpa.scratch[ivpa] > zero
                        icut_upper_z = ivpa - 1
                        break
                    end
                end
            end

            function apply_bc!(x)
                # Boundary condition
                enforce_v_boundary_condition_local!(x, vpa_bc, nothing, true,
                                                    vpa, vpa_spectral)

                if z.bc == "wall"
                    # Wall boundary conditions. Note that as density, upar, ppar do not
                    # change in this implicit step, f_new, f_old, and residual should all
                    # be zero at exactly the same set of grid points, so it is reasonable
                    # to zero-out `residual` to impose the boundary condition. We impose
                    # this after subtracting f_old in case rounding errors, etc. mean that
                    # at some point f_old had a different boundary condition cut-off
                    # index.
                    if z.irank == 0 && iz == 1
                        x[icut_lower_z:end] .= 0.0
                    end
                    # absolute velocity at right boundary
                    if z.irank == z.nrank - 1 && iz == z.n
                        x[1:icut_upper_z] .= 0.0
                    end
                end
            end

            # Need to apply 'new' boundary conditions to `f_old`, so that by imposing them
            # on `residual`, they are automatically imposed on `f_new`.
            f_old = vpa.scratch9 .= f_old_no_bc
            apply_bc!(f_old)
            apply_bc!(this_f_out)

            if nl_solver_params.stage_counter[] % nl_solver_params.preconditioner_update_interval == 0
                if z.irank == 0 && iz == 1
                    nl_solver_params.preconditioners[ivperp,iz,ir,is] =
                        (get_precon(dt * D_ii * n / vth^3; icut_upper=icut_lower_z-1),
                         2, icut_lower_z-1)
                elseif z.irank == z.nrank - 1 && iz == z.n
                    nl_solver_params.preconditioners[ivperp,iz,ir,is] =
                        (get_precon(dt * D_ii * n / vth^3; icut_lower=icut_upper_z+1),
                         icut_upper_z+1, vpa.n-1)
                else
                    nl_solver_params.preconditioners[ivperp,iz,ir,is] =
                        (get_precon(dt * D_ii * n / vth^3),
                         2, vpa.n-1)
                end
            end

            function preconditioner(x)
                precon_lu, icut_lower, icut_upper =
                    nl_solver_params.preconditioners[ivperp,iz,ir,is]
                @views mul!(vpa.scratch[icut_lower:icut_upper],
                            vpa_spectral.mass_matrix[icut_lower:icut_upper,icut_lower:icut_upper],
                            x[icut_lower:icut_upper])
                @views ldiv!(x[icut_lower:icut_upper], precon_lu,
                             vpa.scratch[icut_lower:icut_upper])
                return nothing
            end

            left_preconditioner = identity
            right_preconditioner = preconditioner

            # Define a function whose input is `f_new`, so that when it's output
            # `residual` is zero, f_new is the result of a backward-Euler timestep:
            #   (f_new - f_old) / dt = RHS(f_new)
            # ⇒ f_new - f_old - dt*RHS(f_new) = 0
            function residual_func!(residual, f_new)
                apply_bc!(f_new)
                residual .= f_old
                ion_vpa_maxwell_diffusion_inner!(residual, f_new, n, upar, vth, vpa,
                                                 vpa_spectral, D_ii, dt, evolve_density,
                                                 evolve_upar, evolve_ppar)

                # Now
                #   residual = f_old + dt*RHS(f_new)
                # so update to desired residual
                @. residual = f_new - residual

                apply_bc!(residual)
                #moment_constraints_on_residual!(residual, f_new, moments, vpa)
            end

            # Buffers
            # Note vpa.scratch, vpa.scratch2 and vpa.scratch3 are used by advance_f!, so
            # we cannot use it here.
            residual = vpa.scratch4
            delta_x = vpa.scratch5
            rhs_delta = vpa.scratch6
            v = vpa.scratch7
            w = vpa.scratch8

            # Use forward-Euler step for initial guess
            # By passing this_f_out, which is equal to f_old at this point, the 'residual'
            # is
            #   f_new - f_old - dt*RHS(f_old) = -dt*RHS(f_old)
            # so to get a forward-Euler step we have to subtract this 'residual'
            residual_func!(residual, this_f_out)
            this_f_out .-= residual

            success = newton_solve!(this_f_out, residual_func!, residual, delta_x,
                                    rhs_delta, v, w, nl_solver_params, coords=coords,
                                    left_preconditioner=left_preconditioner,
                                    right_preconditioner=right_preconditioner)
            if !success
                return success
            end
        end
    end

    nl_solver_params.stage_counter[] += 1

    return true
end

end # maxwell_diffusion
