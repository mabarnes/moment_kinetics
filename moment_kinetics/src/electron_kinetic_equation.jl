module electron_kinetic_equation

using Dates
using LinearAlgebra
using MPI

export get_electron_critical_velocities

using ..looping
using ..derivatives: derivative_z!
using ..boundary_conditions: enforce_v_boundary_condition_local!,
                             enforce_vperp_boundary_condition!
using ..calculus: derivative!, second_derivative!, integral
using ..communication
using ..interpolation: interpolate_to_grid_1d!
using ..type_definitions: mk_float
using ..array_allocation: allocate_float
using ..electron_fluid_equations: calculate_electron_qpar_from_pdf!
using ..electron_fluid_equations: electron_energy_equation!
using ..electron_z_advection: electron_z_advection!
using ..electron_vpa_advection: electron_vpa_advection!
using ..file_io: write_initial_electron_state, finish_initial_electron_io
using ..krook_collisions: electron_krook_collisions!
using ..moment_constraints: hard_force_moment_constraints!
using ..velocity_moments: integrate_over_vspace

"""
update_electron_pdf is a function that uses the electron kinetic equation 
to solve for the updated electron pdf

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor

    INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    ppar = electron parallel pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
    dt = time step size
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf!(fvec, pdf, moments, dens, vthe, ppar, qpar, qpar_updated,
        phi, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, r, z, vperp, vpa, z_spectral,
        vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy, dt, collisions, composition,
        num_diss_params, max_electron_pdf_iterations; io_initial_electron=nothing,
        initial_time=0.0)

    # set the method to use to solve the electron kinetic equation
    solution_method = "artificial_time_derivative"
    #solution_method = "shooting_method"
    #solution_method = "picard_iteration"
    # solve the electron kinetic equation using the specified method
    if solution_method == "artificial_time_derivative"
        return update_electron_pdf_with_time_advance!(fvec, pdf, qpar, qpar_updated, 
            moments, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, phi, collisions, composition, 
            r, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy, dt,
            num_diss_params, max_electron_pdf_iterations;
            io_initial_electron=io_initial_electron, initial_time=initial_time)
    elseif solution_method == "shooting_method"
        return update_electron_pdf_with_shooting_method!(pdf, dens, vthe, ppar, qpar,
            qpar_updated, phi, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, z, vpa,
            vpa_spectral, scratch_dummy, composition)
    elseif solution_method == "picard_iteration"
        return update_electron_pdf_with_picard_iteration!(pdf, dens, vthe, ppar, ddens_dz,
            dppar_dz, dqpar_dz, dvth_dz, z, vpa, vpa_spectral, scratch_dummy,
            max_electron_pdf_iterations)
    else
        error("!!! invalid solution method specified !!!")
    end
    return nothing    
end

"""
update_electron_pdf_with_time_advance is a function that introduces an artifical time derivative to advance 
the electron kinetic equation until a steady-state solution is reached.

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor

    INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    ppar = electron parallel pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
    dt = time step size
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
    io_initial_electron = info struct for binary file I/O
    initial_time = initial value for the (pseudo-)time
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_time_advance!(fvec, pdf, qpar, qpar_updated, 
    moments, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, phi, collisions, composition,
    r, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy, dt,
    num_diss_params, max_electron_pdf_iterations; io_initial_electron=nothing, initial_time=0.0)

    begin_r_z_region()

    #println("TMP FOR TESTING: SETTING UPAR_I = UPAR_E = 0!!!")
    #moments.electron.upar .= 0.0

    # there will be a better way of doing this
    # store the incoming ppar
    #ppar_old = allocate_float(z.n,1)
    #ppar_old .= ppar

    upar = moments.electron.upar
    println("WARNING: when time-evolving the ions, this should really be something like fvec_in.upar, so that it is defined at the correct time level...")
    upar_ion = moments.ion.upar

    # create a (z,r) dimension dummy array for use in taking derivatives
    dummy_zr = @view scratch_dummy.dummy_zrs[:,:,1]
    # create several (z) dimension dummy arrays for use in taking derivatives
    buffer_r_1 = @view scratch_dummy.buffer_rs_1[:,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[:,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[:,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[:,1]
    buffer_r_5 = @view scratch_dummy.buffer_rs_5[:,1]
    buffer_r_6 = @view scratch_dummy.buffer_rs_6[:,1]

    # compute the z-derivative of the input electron parallel flow, needed for the electron kinetic equation
    @views derivative_z!(moments.electron.dupar_dz, upar, buffer_r_1, buffer_r_2, buffer_r_3,
                         buffer_r_4, z_spectral, z)

    # compute the z-derivative of the input electron parallel pressure, needed for the electron kinetic equation
    @views derivative_z!(dppar_dz, ppar, buffer_r_1, buffer_r_2, buffer_r_3,
                         buffer_r_4, z_spectral, z)

    #println("TMP FOR TESTING -- dens and ddens_dz artificially set!!!")
    #@. dens = 1.0
    #@. ddens_dz = (dppar_dz / ppar)*0.9

    @loop_r_z ir iz begin
        # update the electron thermal speed using the updated electron parallel pressure
        vthe[iz,ir] = sqrt(abs(2.0 * ppar[iz,ir] / (dens[iz,ir] * composition.me_over_mi)))
        # update the z-derivative of the electron thermal speed from the z-derivatives of the electron density
        # and parallel pressure
        dvth_dz[iz,ir] = 0.5 * vthe[iz,ir] * (dppar_dz[iz,ir] / ppar[iz,ir] - ddens_dz[iz,ir] / dens[iz,ir])
        fvec.electron_ppar[iz,ir] = ppar[iz,ir]
    end

    # compute the z-derivative of the input electron parallel heat flux, needed for the electron kinetic equation
    @views derivative_z!(dqpar_dz, qpar, buffer_r_1, buffer_r_2, buffer_r_3,
                         buffer_r_4, z_spectral, z)

    #dt_electron = dt * sqrt(composition.me_over_mi)
    #dt_max = 3.0e-8
    dt_max = 2.0e-8
    #dt_max = 2.5e-9

    #dt_energy = 1.0e-7
    dt_energy = 2.0e-8
    #dt_energy = 2.5e-9
    #n_ppar_subcycles = 1000
    #n_ppar_subcycles = 200
    n_ppar_subcycles = 1
    time = initial_time

    #z_speedup_fac = 20.0
    #z_speedup_fac = 5.0
    z_speedup_fac = 1.0

    # define residual to point to a dummy array;
    # to be filled with the contributions to the electron kinetic equation; i.e., d(pdf)/dt = -residual
    residual = scratch_dummy.buffer_vpavperpzr_6
    max_term = scratch_dummy.buffer_vpavperpzr_5
    single_term = scratch_dummy.buffer_vpavperpzr_4
    # initialise the number of iterations in the solution of the electron kinetic equation to be 1
    iteration = 1
    # initialise the electron pdf convergence flag to false
    electron_pdf_converged = false
    # calculate the residual of the electron kinetic equation for the initial guess of the electron pdf
    dt_electron = electron_kinetic_equation_residual!(residual, max_term, single_term, pdf, dens, upar, vthe, ppar, upar_ion,
                                        ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
                                        z, vperp, vpa, z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                                        collisions, num_diss_params, dt_max)
    speedup_hack_residual!(residual, z_speedup_fac, z, vpa)
    if n_blocks[] == 1
        text_output_suffix = ""
    else
        text_output_suffix = "$(iblock_index[])"
    end
    begin_serial_region()
    @serial_region begin
        # open files to write the electron heat flux and pdf to file                                
        io_upar = open("upar$text_output_suffix.txt", "w")
        io_qpar = open("qpar$text_output_suffix.txt", "w")
        io_ppar = open("ppar$text_output_suffix.txt", "w")
        io_pdf = open("pdf$text_output_suffix.txt", "w")
        io_vth = open("vth$text_output_suffix.txt", "w")
        if !electron_pdf_converged
            # need to exit or handle this appropriately
            @loop_vpa ivpa begin
                @loop_z iz begin
                    println(io_pdf, "z: ", z.grid[iz], " wpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " time: ", time, " residual: ", residual[ivpa, 1, iz, 1])
                end
                println(io_pdf,"")
            end
            @loop_z iz begin
                println(io_upar, "z: ", z.grid[iz], " upar: ", upar[iz,1], " dupar_dz: ", moments.electron.dupar_dz[iz,1], " time: ", time, " iteration: ", iteration)
                println(io_qpar, "z: ", z.grid[iz], " qpar: ", qpar[iz,1], " dqpar_dz: ", dqpar_dz[iz,1], " time: ", time, " iteration: ", iteration)
                println(io_ppar, "z: ", z.grid[iz], " ppar: ", ppar[iz,1], " dppar_dz: ", dppar_dz[iz,1], " time: ", time, " iteration: ", iteration)
                println(io_vth, "z: ", z.grid[iz], " vthe: ", vthe[iz,1], " dvth_dz: ", dvth_dz[iz,1], " time: ", time, " iteration: ", iteration, " dens: ", dens[iz,1])
            end
            println(io_upar,"")
            println(io_qpar,"")
            println(io_ppar,"")
            println(io_vth,"")
        end
        io_pdf_stages = open("pdf_zright$text_output_suffix.txt", "w")
    end

    # check to see if the electron pdf satisfies the electron kinetic equation to within the specified tolerance
    #average_residual, electron_pdf_converged = check_electron_pdf_convergence(residual, max_term)
    average_residual, electron_pdf_converged = check_electron_pdf_convergence(residual, pdf, upar, vthe, z, vpa)

    output_interval = 1000
    output_counter = 0
    begin_serial_region()
    @serial_region begin
        output_counter += 1
        if io_initial_electron !== nothing
            write_initial_electron_state(pdf, moments, time, io_initial_electron,
                                         output_counter, r, z, vperp, vpa)
        end
    end
    # evolve (artificially) in time until the residual is less than the tolerance
    while !electron_pdf_converged && (iteration <= max_electron_pdf_iterations)
        #dt_energy = dt_electron * 10.0

        # get an updated iterate of the electron parallel pressure
        begin_r_z_region()
        #ppar .= ppar_old
        wpa3_moment = @view scratch_dummy.buffer_zrs_1[:,:,1]
        @loop_r_z ir iz begin
            wpa3_moment[iz,ir] = qpar[iz,ir] / vthe[iz,ir]^3
        end
        for i in 1:n_ppar_subcycles
            @loop_r_z ir iz begin
                qpar[iz,ir] = vthe[iz,ir]^3 * wpa3_moment[iz,ir]
                dummy_zr[iz,ir] = -upar[iz,ir]
            end
            @views derivative_z!(dqpar_dz, qpar, buffer_r_1, buffer_r_2, buffer_r_3,
                                 buffer_r_4, z_spectral, z)
            # Compute the upwinded z-derivative of the electron parallel pressure for the
            # electron energy equation
            @views derivative_z!(moments.electron.dppar_dz_upwind, ppar, dummy_zr,
                             buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                             buffer_r_5, buffer_r_6, z_spectral, z)
            # centred second derivative for dissipation
            if num_diss_params.electron.moment_dissipation_coefficient > 0.0
                @views derivative_z!(moments.electron.d2ppar_dz2, dppar_dz, buffer_r_1,
                                 buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
                begin_serial_region()
                @serial_region begin
                    if z.irank == 0
                        moments.electron.d2ppar_dz2[1,:] .= 0.0
                    end
                    if z.irank == z.nrank - 1
                        moments.electron.d2ppar_dz2[end,:] .= 0.0
                    end
                end
            end

            # TMP FOR TESTING -- MAB
            #dt_energy = dt_electron
            electron_energy_equation!(ppar, dens, fvec, moments, collisions, dt_energy, composition, num_diss_params, z)

            # Apply same 'speed up' hack to ppar that we do to the distribution function,
            # but without the wpa dependence.
            @loop_r_z ir iz begin
                zval = z.grid[iz]
                znorm = 2.0*zval/z.L
                ppar[iz,ir] = fvec.electron_ppar[iz,ir] +
                              (ppar[iz,ir] - fvec.electron_ppar[iz,ir]) *
                              (1.0 + z_speedup_fac*(1.0 - znorm^2))
            end
            begin_r_z_region()
            @loop_r_z ir iz begin
                fvec.electron_ppar[iz,ir] = ppar[iz,ir]
            end

            # compute the z-derivative of the updated electron parallel pressure
            @views derivative_z!(dppar_dz, ppar, buffer_r_1, buffer_r_2, buffer_r_3,
                                 buffer_r_4, z_spectral, z)
            begin_r_z_region()
            @loop_r_z ir iz begin
                # update the electron thermal speed using the updated electron parallel pressure
                vthe[iz,ir] = sqrt(abs(2.0 * ppar[iz,ir] / (dens[iz,ir] * composition.me_over_mi)))
                # update the z-derivative of the electron thermal speed from the z-derivatives of the electron density
                # and parallel pressure
                dvth_dz[iz,ir] = 0.5 * vthe[iz,ir] * (dppar_dz[iz,ir] / ppar[iz,ir] - ddens_dz[iz,ir] / dens[iz,ir])
            end
        end

        begin_r_z_region()
        # d(pdf)/dt = -kinetic_eqn_terms, so pdf_new = pdf - dt * kinetic_eqn_terms
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            pdf[ivpa,ivperp,iz,ir] -= dt_electron * residual[ivpa,ivperp,iz,ir]
        end

        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            pdf[ivpa,ivperp,iz,ir] = max(pdf[ivpa,ivperp,iz,ir], 0.0)
        end

        # enforce the boundary condition(s) on the electron pdf
        enforce_boundary_condition_on_electron_pdf!(pdf, phi, vthe, upar, vperp, vpa,
                                                    vperp_spectral, vpa_spectral,
                                                    vpa_advect, moments,
                                                    num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                    composition.me_over_mi)
        #println("A pdf 1 ", pdf[:,1,1,1])
        #println("A pdf end ", pdf[:,1,end,1])

        begin_r_z_region()
        A = moments.electron.constraints_A_coefficient
        B = moments.electron.constraints_B_coefficient
        C = moments.electron.constraints_C_coefficient
        @loop_r_z ir iz begin
            if (iz == 1 && z.irank == 0) || (iz == z.n && z.irank == z.nrank - 1)
                continue
            end
            (A[iz,ir], B[iz,ir], C[iz,ir]) =
                #@views hard_force_moment_constraints!(pdf[:,:,iz,ir], (evolve_density=true, evolve_upar=false, evolve_ppar=true), vpa)
                @views hard_force_moment_constraints!(pdf[:,:,iz,ir], (evolve_density=true, evolve_upar=true, evolve_ppar=true), vpa)
        end
        #println("B pdf 1 ", pdf[:,1,1,1])
        #println("B pdf end ", pdf[:,1,end,1])
        #error("foo")
        
        if (mod(iteration,output_interval)==1)
            begin_serial_region()
            @serial_region begin
                @loop_vpa ivpa begin
                    println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa,1,end,1], " iteration: ", iteration, " flag: ", 1)
                end
                println(io_pdf_stages,"")
            end
        end

        # update the time following the pdf update
        time += dt_electron

        qpar_updated = false
        # update the electron heat flux
        calculate_electron_qpar_from_pdf!(qpar, ppar, vthe, pdf, vpa)
        qpar_updated = true
        
        # compute the z-derivative of the parallel electron heat flux
        @views derivative_z!(dqpar_dz, qpar, buffer_r_1,
            buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)

        # Compute the upwinded z-derivative of the electron parallel pressure for the 
        # electron energy equation
        @loop_r_z ir iz begin
            dummy_zr[iz,ir] = -upar[iz,ir]
        end
        @views derivative_z!(moments.electron.dppar_dz_upwind, ppar, dummy_zr,
                             buffer_r_1, buffer_r_2, buffer_r_3, buffer_r_4,
                             buffer_r_5, buffer_r_6, z_spectral, z)
         # centred second derivative for dissipation
        if num_diss_params.electron.moment_dissipation_coefficient > 0.0
            @views derivative_z!(moments.electron.d2ppar_dz2, dppar_dz, buffer_r_1,
                                 buffer_r_2, buffer_r_3, buffer_r_4, z_spectral, z)
        end

        if (mod(iteration,100) == 0)
            begin_serial_region()
            @serial_region begin
                println("iteration: ", iteration, " time: ", time, " dt_electron: ", dt_electron, " phi_boundary: ", phi[[1,end],1], " average_residual: ", average_residual)
            end
        end
        if (mod(iteration,output_interval) == 0)
            begin_serial_region()
            @serial_region begin
                if (mod(iteration,100*output_interval) == 0)
                    @loop_vpa ivpa begin
                        @loop_z iz begin
                            println(io_pdf, "z: ", z.grid[iz], " wpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " time: ", time, " residual: ", residual[ivpa, 1, iz, 1])
                        end
                        println(io_pdf,"")
                    end
                    println(io_pdf,"")
                end
                @loop_z iz begin
                    println(io_upar, "z: ", z.grid[iz], " upar: ", upar[iz,1], " dupar_dz: ", moments.electron.dupar_dz[iz,1], " time: ", time, " iteration: ", iteration)
                    println(io_qpar, "z: ", z.grid[iz], " qpar: ", qpar[iz,1], " dqpar_dz: ", dqpar_dz[iz,1], " time: ", time, " iteration: ", iteration)
                    println(io_ppar, "z: ", z.grid[iz], " ppar: ", ppar[iz,1], " dppar_dz: ", dppar_dz[iz,1], " time: ", time, " iteration: ", iteration)
                    println(io_vth, "z: ", z.grid[iz], " vthe: ", vthe[iz,1], " dvth_dz: ", dvth_dz[iz,1], " time: ", time, " iteration: ", iteration, " dens: ", dens[iz,1])
                end
                println(io_upar,"")
                println(io_qpar,"")
                println(io_ppar,"")
                println(io_vth,"")
                output_counter += 1
                if io_initial_electron !== nothing
                    write_initial_electron_state(pdf, moments, time, io_initial_electron,
                                                 output_counter, r, z, vperp, vpa)
                end
            end
        end

        #@views derivative_z!(dqpar_dz, qpar, 
        #        scratch_dummy.buffer_rs_1[:,1], scratch_dummy.buffer_rs_2[:,1], scratch_dummy.buffer_rs_3[:,1],
        #        scratch_dummy.buffer_rs_4[:,1], z_spectral, z)
        # TMP FOR TESTING
        #dqpar_dz .= 0.0
        # calculate the residual of the electron kinetic equation for the updated electron pdf
        dt_electron = electron_kinetic_equation_residual!(residual, max_term, single_term, pdf, dens, upar, vthe, ppar, upar_ion, ddens_dz,
                                            dppar_dz, dqpar_dz, dvth_dz, 
                                            z, vperp, vpa, z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                                            collisions, num_diss_params, dt_max)
        # check to see if the electron pdf satisfies the electron kinetic equation to within the specified tolerance
        #average_residual, electron_pdf_converged = check_electron_pdf_convergence(residual, max_term)
        average_residual, electron_pdf_converged = check_electron_pdf_convergence(residual, pdf, upar, vthe, z, vpa)

        speedup_hack_residual!(residual, z_speedup_fac, z, vpa)
        if electron_pdf_converged || any(isnan.(ppar)) || any(isnan.(pdf))
            break
        end
        iteration += 1
    end
    # Update the 'scratch' arrays with the final result
    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        fvec.pdf_electron[ivpa,ivperp,iz,ir] = pdf[ivpa,ivperp,iz,ir]
    end
    begin_serial_region()
    @serial_region begin
        if !electron_pdf_converged
            @loop_vpa ivpa begin
                @loop_z iz begin
                    println(io_pdf, "z: ", z.grid[iz], " wpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " time: ", time, " residual: ", residual[ivpa, 1, iz, 1])
                end
                println(io_pdf,"")
            end
        end
        close(io_upar)
        close(io_qpar)
        close(io_ppar)
        close(io_vth)
        close(io_pdf)
        close(io_pdf_stages)
        if !electron_pdf_converged
            # need to exit or handle this appropriately

            if io_initial_electron !== nothing
                output_counter += 1
                write_initial_electron_state(pdf, moments, time, io_initial_electron,
                                             output_counter, r, z, vperp, vpa)
                finish_initial_electron_io(io_initial_electron)
            end

        end
    end
    if !electron_pdf_converged
        error("!!!max number of iterations for electron pdf update exceeded!!!\n"
              * "Stopping at $(Dates.format(now(), dateformat"H:MM:SS"))")
    end
    return time, output_counter
end

function speedup_hack_residual!(residual, z_speedup_fac, z, vpa)
    # Divide by wpa to relax CFL condition at large wpa - only looking for steady
    # state here, so does not matter that this makes time evolution incorrect.
    # Also increase the effective timestep for z-values far from the sheath boundary -
    # these have a less-limited timestep so letting them evolve faster speeds up
    # convergence to the steady state.

    # Actually modify so that large wpa does go faster (to allow some phase mixing - maybe
    # this makes things more stable?), but not by so much.
    #vpa_fudge_factor = 1.0
    #vpa_fudge_factor = 0.8
    vpa_fudge_factor = 0.0

    Lz = z.L
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        zval = z.grid[iz]
        znorm = 2.0*zval/Lz
        residual[ivpa,ivperp,iz,ir] *=
            (1.0 + z_speedup_fac*(1.0 - znorm^2)) /
            sqrt(1.0 + vpa_fudge_factor * vpa.grid[ivpa]^2)
    end
    return nothing
end

function enforce_boundary_condition_on_electron_pdf!(pdf, phi, vthe, upar, vperp, vpa,
                                                     vperp_spectral, vpa_spectral,
                                                     vpa_adv, moments, vpa_diffusion,
                                                     me_over_mi)
    # Enforce velocity-space boundary conditions
    if vpa.n > 1
        begin_r_z_vperp_region()
        @loop_r_z_vperp ir iz ivperp begin
            # enforce the vpa BC
            # use that adv.speed independent of vpa
            @views enforce_v_boundary_condition_local!(pdf[:,ivperp,iz,ir], vpa.bc,
                                                       vpa_adv[1].speed[:,ivperp,iz,ir],
                                                       vpa_diffusion, vpa, vpa_spectral)
        end
    end
    if vperp.n > 1
        begin_r_z_vpa_region()
        @views enforce_vperp_boundary_condition!(pdf, vperp.bc, vperp, vperp_spectral)
    end

    # first enforce the boundary condition at z_min.
    # this involves forcing the pdf to be zero for electrons travelling faster than the max speed
    # they could attain by accelerating in the electric field between the wall and the simulation boundary;
    # for electrons with positive velocities less than this critical value, they must have the same
    # pdf as electrons with negative velocities of the same magnitude.
    # the electrostatic potential at the boundary, which determines the critical speed, is unknown a priori;
    # use the constraint that the first moment of the normalised pdf be zero to choose the potential.

    begin_r_region()

    # pdf_adjustment_option determines the velocity-dependent pre-factor for the
    # corrections to the pdf needed to ensure moment constraints are satisfied
    #pdf_adjustment_option = "vpa4"
    #pdf_adjustment_option = "vpa4_gaussian"
    pdf_adjustment_option = "no1st_vpa2"

    cutoff_step_width = 0.1

    # wpa_values will be used to store the wpa = (vpa - upar)/vthe values corresponding to a vpa grid symmetric about vpa=0
    #wpa_values = vpa.scratch
    # interpolated_pdf will be used to store the pdf interpolated onto the vpa-symmetric grid
    #interpolated_pdf = vpa.scratch2
    reversed_pdf = vpa.scratch

    # ivpa_zero is the index of the interpolated_pdf corresponding to vpa = 0
    #ivpa_zero = (vpa.n+1)÷2

    @loop_r ir begin
        # construct a grid of wpa = (vpa - upar)/vthe values corresponding to a vpa-symmetric grid
        #@. wpa_values = vpa.grid #- upar[1,ir] / vthe[1,ir]
        #wpa_of_minus_vpa = @. vpa.scratch3 = -vpa.grid - upar[1,ir] / vthe[1,ir]

        # Want to construct the w-grid corresponding to -vpa.
        #   wpa(vpa) = (vpa - upar)/vth
        #   ⇒ vpa = vth*wpa(vpa) + upar
        #   wpa(-vpa) = (-vpa - upar)/vth
        #             = (-(vth*wpa(vpa) + upar) - upar)/vth
        #             = (-vth*wpa - 2*upar)/vth
        #             = -wpa - 2*upar/vth
        # [Note that `vpa.grid` is slightly mis-named here - it contains the values of
        #  wpa(+vpa) as we are using a 'moment kinetic' approach.]
        # Need to reverse vpa.grid because the grid passed as the second argument of
        # interpolate_to_grid_1d!() needs to be sorted in increasing order.
        #reversed_wpa_of_minus_vpa = vpa.scratch2 .= .-reverse(vpa.grid) .- 2.0 * upar[1,ir] / vthe[1,ir]
        #reversed_wpa_of_minus_vpa = vpa.scratch2 .= .-reverse(vpa.grid) .- 1.5 * upar[1,ir] / vthe[1,ir]
        reversed_wpa_of_minus_vpa = vpa.scratch2 .= .-reverse(vpa.grid)
        # interpolate the pdf onto this grid
        #@views interpolate_to_grid_1d!(interpolated_pdf, wpa_values, pdf[:,1,1,ir], vpa, vpa_spectral)
        @views interpolate_to_grid_1d!(reversed_pdf, reversed_wpa_of_minus_vpa, pdf[:,1,1,ir], vpa, vpa_spectral) # Could make this more efficient by only interpolating to the points needed below, by taking an appropriate view of wpa_of_minus_vpa. Also, in the element containing vpa=0, this interpolation depends on the values that will be replaced by the reflected, interpolated values, which is not ideal (maybe this element should be treated specially first?).
        reverse!(reversed_pdf)
        # fill in the vpa > 0 points of the pdf by mirroring the vpa < 0 points
        #@. interpolated_pdf[ivpa_zero+1:end] = interpolated_pdf[ivpa_zero-1:-1:1]
        # construct a grid of vpa/vthe = wpa + upar/vthe values corresponding to the wpa-symmetric grid
        #@. wpa_values = vpa.grid #+ upar[1,ir] / vthe[1,ir]
        # interpolate back onto the original wpa grid
        #@views interpolate_to_grid_1d!(pdf[:,1,1,ir], wpa_values, interpolated_pdf, vpa, vpa_spectral)
        # construct wpa * pdf
        #@. vpa.scratch3 = pdf[:,1,1,ir] * vpa.grid
        # calculate the first moment of the normalised pdf
        #first_vspace_moment = 0.0
        #first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # the initial max vpa index is the largest one possible; this will be reduced if the first moment is positive
        ivpa = 0
        ivpa_max = vpa.n + 1
        # adjust the critical (cutoff) speed until the first moment is as close to zero as possible
        # if the first moment is positive, then the cutoff speed needs to be reduced
        upar_over_vth = upar[1,ir] / vthe[1,ir]
        #println("upar=", upar[1,ir], " vthe=", vthe[1,ir])
        #println("$first_vspace_moment, u/vth=$upar_over_vth")
        vpa_unnorm = @. vpa.scratch3 = vthe[1,ir] * vpa.grid + upar[1,ir]
        upar_integral = 0.0
        #while first_vspace_moment > upar_over_vth # > 0.0
        #    # zero out the pdf at the current cutoff velocity
        #    pdf[ivpa_max,1,1,ir] = 0.0
        #    # update wpa * pdf
        #    vpa.scratch3[ivpa_max] = 0.0
        #    # calculate the updated first moment of the normalised pdf
        #    first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        #    #println("truncating pdf $ivpa_max, $first_vspace_moment, u/vth=$upar_over_vth")
        #    if first_vspace_moment > upar_over_vth #0.0
        #        ivpa_max -= 1
        #    end
        #end
        upar0 = upar[1,ir]
        #println("before pdf left ", pdf[:,1,1,ir])
        while upar_integral > upar0 && ivpa_max > 1
            ivpa += 1
            ivpa_max -= 1
            # zero out the reversed pdf at the current cutoff velocity
            #reversed_pdf[ivpa_max] = 0.0
            # calculate the updated first moment of the normalised pdf
            upar_integral += vpa_unnorm[ivpa] * pdf[ivpa,1,1,ir] * vpa.wgts[ivpa]
            #println("left ", ivpa, " ", upar_integral, " ", upar0)
        end
        integral_excess = upar_integral - upar0
        fraction_of_pdf = integral_excess / (vpa_unnorm[ivpa] * vpa.wgts[ivpa]) / pdf[ivpa,1,1,ir]
        #println("fraction_of_pdf=", fraction_of_pdf)

        # Define so that when fraction_of_pdf=1 (when all of the contribution to the
        # integral from the point at ivpa is required to make upar_integral<upar0) the
        # cut-off velocity is half way between ivpa and ivpa+1, while when
        # fraction_of_pdf=0 (none of the contribution to the integral from the point at
        # ivpa is required to make upar_integral<upar0) the cut-off is half way between
        # ivpa-1 and ivpa.
        vmax = 0.5 * (vpa_unnorm[ivpa] + vpa_unnorm[ivpa+1]) +
               0.5 * fraction_of_pdf*(vpa_unnorm[ivpa-1] - vpa_unnorm[ivpa+1])

        #println("vmax=$vmax, v-no-interp=", vpa_unnorm[ivpa])
        #wmax = (-vmax - upar[1,ir]) / vthe[1,ir]
        #println("wmax=$wmax, w-no-interp", (vpa_unnorm[ivpa] - upar0)/vthe[1,ir])
        #@loop_vpa ivpa begin
        #    reversed_pdf[ivpa] *= 0.5*(1.0 - tanh((vpa.grid[ivpa] - wmax) / cutoff_step_width))
        #end
        reversed_pdf[ivpa_max+1:end] .= 0
        reversed_pdf[ivpa_max] *= fraction_of_pdf
        #println("first_vspace_moment=$first_vspace_moment, ivpa_max=$ivpa_max")
        #println("done first cutoff loop")
        # update the electrostatic potential at the boundary to be the value corresponding to the updated cutoff velocity
        #phi[1,ir] = me_over_mi * vthe[1,ir]^2 * vpa.grid[ivpa_max]^2
        phi[1,ir] = me_over_mi * vmax^2
        iv0 = findfirst(x -> x>0.0, vpa_unnorm)
        pdf[iv0:end,1,1,ir] .= reversed_pdf[iv0:end]
        #println("check reversed change ", reversed_pdf[iv0:end])
        #println("reversed_pdf ", reversed_pdf)
        #println("after pdf left ", pdf[:,1,1,ir])
        # obtain the normalisation constants needed to ensure the zeroth, first and second moments
        # of the modified pdf are 1, 0 and 1/2 respectively
        # will need vpa / vthe = wpa + upar/vthe
        @. vpa.scratch2 = vpa.grid + upar[1,ir] / vthe[1,ir]
        # first need to calculate int dwpa pdf = zeroth_moment
        zeroth_moment = integrate_over_vspace(pdf[:,1,1,ir], vpa.wgts)
        # calculate int dwpa wpa^2 * pdf = wpa2_moment
        @. vpa.scratch3 = pdf[:,1,1,ir] * vpa.grid^2
        wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # calculate int dwpa vpa^2 * pdf = vpa2_moment
        @. vpa.scratch3 = pdf[:,1,1,ir] * vpa.scratch2^2
        vpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # calculate int dwpa vpa^2 * wpa * pdf = vpa2_wpa_moment
        @. vpa.scratch3 = vpa.grid * vpa.scratch2^2 * pdf[:,1,1,ir]
        vpa2_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # calculate int dwpa wpa^2 * vpa^2 * pdf = vpa2_wpa2_moment
        @. vpa.scratch3 = vpa.grid^2 * vpa.scratch2^2 * pdf[:,1,1,ir]
        vpa2_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        
        normalisation_constant_A = 0.0
        normalisation_constant_B = 0.0
        normalisation_constant_C = 0.0
        if pdf_adjustment_option == "absvpa"
            # calculate int dwpa |vpa| * pdf = absvpa_moment
            @. vpa.scratch3 = pdf[:,1,1,ir] * abs(vpa.scratch2)
            absvpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa |vpa| * wpa * pdf = absvpa_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            absvpa_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa |vpa| * wpa^2 * pdf = absvpa_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            absvpa_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # assuming pdf_updated = pdf * (normalisation_constant_A + |vpa| * normalisation_constant_B + vpa^2 * normalisation_constant_C)
            # calculate the 'B' normalisation constant
            normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) / 
                (absvpa_wpa2_moment  - absvpa_wpa_moment * vpa2_wpa2_moment / vpa2_wpa_moment
                + wpa2_moment / zeroth_moment * (absvpa_wpa_moment * vpa2_moment / vpa2_wpa_moment - absvpa_moment))
            # calculate the 'A' normalisation constant
            normalisation_constant_A = (1 + normalisation_constant_B * 
                (vpa2_moment * absvpa_wpa_moment / vpa2_wpa_moment - absvpa_moment)) / zeroth_moment
            # calculate the 'C' normalisation constant
            normalisation_constant_C = -normalisation_constant_B * absvpa_wpa_moment / vpa2_wpa_moment
            # updated pdf is old pdf * (normalisation_constant_A + |vpa| * normalisation_constant_B + vpa^2 * normalisation_constant_C)
            @. pdf[:,1,1,ir] *= (normalisation_constant_A + abs(vpa.scratch2) * normalisation_constant_B 
                                + vpa.scratch2^2 * normalisation_constant_C)
        elseif pdf_adjustment_option == "vpa4"
            # calculate int dwpa vpa^4 * pdf = vpa4_moment
            @. vpa.scratch3 = vpa.scratch2^4 * pdf[:,1,1,ir]
            vpa4_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa * pf = vpa4_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa^2 * pdf = vpa4_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # assuming pdf_updated = pdf * (normalisation_constant_A + vpa^2 * normalisation_constant_B + vpa^4 * normalisation_constant_C)
            normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) /
                 (vpa2_wpa2_moment - vpa2_wpa_moment * vpa4_wpa2_moment / vpa4_wpa_moment
                 + wpa2_moment / zeroth_moment * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment))
            normalisation_constant_A = (1 + normalisation_constant_B 
                * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment)) / zeroth_moment
            normalisation_constant_C = -normalisation_constant_B * vpa2_wpa_moment / vpa4_wpa_moment
            @. pdf[:,1,1,ir] *= (normalisation_constant_A + vpa.scratch2^2 * normalisation_constant_B 
                                + vpa.scratch2^4 * normalisation_constant_C)
        elseif pdf_adjustment_option == "vpa4_gaussian"
            afac = 0.1
            bfac = 0.2
            # calculate int dwpa vpa^2 * exp(-vpa^2) * pdf = vpa2_moment
            @. vpa.scratch3 = pdf[:,1,1,ir] * vpa.scratch2^2 * exp(-afac * vpa.scratch2^2)
            vpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^2 * exp(-vpa^2) * wpa * pdf = vpa2_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            vpa2_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa wpa^2 * vpa^2 * exp(-vpa^2) * pdf = vpa2_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            vpa2_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * exp(-vpa^2) * pdf = vpa4_moment
            @. vpa.scratch3 = vpa.scratch2^4 * exp(-bfac * vpa.scratch2^2) * pdf[:,1,1,ir]
            vpa4_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa * pf = vpa4_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa^2 * pdf = vpa4_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # assuming pdf_updated = pdf * (normalisation_constant_A + vpa^2 * normalisation_constant_B + exp(-vpa^2) * vpa^4 * normalisation_constant_C)
            normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) /
                 (vpa2_wpa2_moment - vpa2_wpa_moment * vpa4_wpa2_moment / vpa4_wpa_moment
                 + wpa2_moment / zeroth_moment * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment))
            normalisation_constant_A = (1 + normalisation_constant_B 
                * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment)) / zeroth_moment
            normalisation_constant_C = -normalisation_constant_B * vpa2_wpa_moment / vpa4_wpa_moment
            #normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) /
            #                           (vpa2_wpa2_moment
            #                            - wpa2_moment / zeroth_moment * vpa2_moment)
            #normalisation_constant_A = (1 - normalisation_constant_B 
            #                                * vpa2_moment) / zeroth_moment
            #normalisation_constant_C = 0.0
            @. pdf[:,1,1,ir] *= (normalisation_constant_A + exp(-afac * vpa.scratch2^2) * vpa.scratch2^2 * normalisation_constant_B 
                                + exp(-bfac * vpa.scratch2^2) * vpa.scratch2^4 * normalisation_constant_C)
        elseif pdf_adjustment_option == "no1st_vpa2"
            normalisation_constant_C = (1.0 - 0.5*zeroth_moment/wpa2_moment) /
                                       (vpa2_moment - zeroth_moment*vpa2_wpa2_moment / wpa2_moment)
            normalisation_constant_A = (0.5 - normalisation_constant_C*vpa2_wpa2_moment) / wpa2_moment
            @. pdf[:,1,1,ir] *= (normalisation_constant_A + vpa.scratch2^2 * normalisation_constant_C)
        else
            println("pdf_adjustment_option not recognised")
            stop()
        end

        moments.electron.constraints_A_coefficient[1,ir] = normalisation_constant_A
        moments.electron.constraints_B_coefficient[1,ir] = normalisation_constant_B
        moments.electron.constraints_C_coefficient[1,ir] = normalisation_constant_C

        # smooth the pdf at the boundary
        #for ivpa ∈ 2:ivpa_max-1
        #    pdf[ivpa,1,1,ir] = (pdf[ivpa-1,1,1,ir] + pdf[ivpa+1,1,1,ir]) / 2.0
        #end
    end

    # next enforce the boundary condition at z_max.
    # this involves forcing the pdf to be zero for electrons travelling faster than the max speed
    # they could attain by accelerating in the electric field between the wall and the simulation boundary;
    # for electrons with negative velocities less than this critical value, they must have the same
    # pdf as electrons with positive velocities of the same magnitude.
    # the electrostatic potential at the boundary, which determines the critical speed, is unknown a priori;
    # use the constraint that the first moment of the normalised pdf be zero to choose the potential.
    
    # io_pdf_stages = open("pdf_stages.txt", "w")
    # zeroth_vspace_moment = integrate_over_vspace(pdf[:,1,end,1], vpa.wgts)
    # @. vpa.scratch3 = pdf[:,1,end,1] * vpa.grid
    # first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
    # @. vpa.scratch3 = pdf[:,1,end,1] * vpa.grid^2
    # second_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
    # @loop_vpa ivpa begin
    #     println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa,1,end,1], " zeroth_vspace_moment: ", zeroth_vspace_moment, 
    #         " first_vspace_moment: ", first_vspace_moment, " second_vspace_moment: ", second_vspace_moment, " stage: ", 0)
    # end
    # println(io_pdf_stages,"")

    @loop_r ir begin
        # construct a grid of wpa = (vpa - upar)/vthe values corresponding to a vpa-symmetric grid
        #@. wpa_values = vpa.grid # - upar[end,ir] / vthe[end,ir]
        # Need to reverse vpa.grid because the grid passed as the second argument of
        # interpolate_to_grid_1d!() needs to be sorted in increasing order.

        # [Note that `vpa.grid` is slightly mis-named here - it contains the values of
        #  wpa(+vpa) as we are using a 'moment kinetic' approach.]
        # Need to reverse vpa.grid because the grid passed as the second argument of
        # interpolate_to_grid_1d!() needs to be sorted in increasing order.
        #reversed_wpa_of_minus_vpa = vpa.scratch2 .= .-reverse(vpa.grid) .- 2.0 * upar[end,ir] / vthe[end,ir]
        #reversed_wpa_of_minus_vpa = vpa.scratch2 .= .-reverse(vpa.grid) .- 1.5 * upar[end,ir] / vthe[end,ir]
        reversed_wpa_of_minus_vpa = vpa.scratch2 .= .-reverse(vpa.grid)
        # interpolate the pdf onto this grid
        #@views interpolate_to_grid_1d!(interpolated_pdf, wpa_values, pdf[:,1,end,ir], vpa, vpa_spectral)
        @views interpolate_to_grid_1d!(reversed_pdf, reversed_wpa_of_minus_vpa, pdf[:,1,end,ir], vpa, vpa_spectral) # Could make this more efficient by only interpolating to the points needed below, by taking an appropriate view of wpa_of_minus_vpa. Also, in the element containing vpa=0, this interpolation depends on the values that will be replaced by the reflected, interpolated values, which is not ideal (maybe this element should be treated specially first?).
        reverse!(reversed_pdf)
        # fill in the vpa < 0 points of the pdf by mirroring the vpa > 0 points
        #@. interpolated_pdf[ivpa_zero-1:-1:1] = interpolated_pdf[ivpa_zero+1:end]
        # construct a grid of vpa/vthe = wpa + upar/vthe values corresponding to the wpa-symmetric grid
        #@. wpa_values = vpa.grid #+ upar[end,ir] / vthe[end,ir]
        # interpolate back onto the original wpa grid
        #@views interpolate_to_grid_1d!(pdf[:,1,end,ir], wpa_values, interpolated_pdf, vpa, vpa_spectral)

        # zeroth_vspace_moment = integrate_over_vspace(pdf[:,1,end,1], vpa.wgts)
        # @. vpa.scratch3 = pdf[:,1,end,1] * vpa.grid
        # first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # @. vpa.scratch3 *= vpa.grid
        # second_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # @loop_vpa ivpa begin
        #     println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa,1,end,ir], " zeroth_vspace_moment: ", zeroth_vspace_moment, 
        #         " first_vspace_moment: ", first_vspace_moment, " second_vspace_moment: ", second_vspace_moment, " stage: ", 1)
        # end
        # println(io_pdf_stages,"")

        # construct wpa * pdf
        #@. vpa.scratch3 = pdf[:,1,end,ir] * vpa.grid
        # calculate the first moment of the normalised pdf
        #first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # the initial min vpa index is the smallest one possible; this will be increased if the first moment is negative
        ivpa = vpa.n+1
        ivpa_min = 0
        # adjust the critical (cutoff) speed until the first moment is as close to zero as possible
        # if the first moment is negative, then the magnitude of the cutoff speed needs to be reduced
        upar_over_vth = upar[end,ir] / vthe[end,ir]
        #println("$first_vspace_moment, u/vth=$upar_over_vth")
        vpa_unnorm = @. vpa.scratch3 = vthe[end,ir] * vpa.grid + upar[end,ir]
        upar_integral = 0.0
        #while first_vspace_moment < upar_over_vth # < 0.0
        #    # zero out the pdf at the current cutoff velocity
        #    pdf[ivpa_min,1,end,ir] = 0.0
        #    # update wpa * pdf
        #    vpa.scratch3[ivpa_min] = 0.0
        #    # calculate the updated first moment of the normalised pdf
        #    first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        #    if first_vspace_moment < upar_over_vth
        #        ivpa_min += 1
        #    end
        #end
        upar_end = upar[end,ir]
        #println("before pdf ", pdf[:,1,end,ir])
        while upar_integral < upar_end && ivpa > 1
            ivpa -= 1
            ivpa_min += 1
            # zero out the reversed pdf at the current cutoff velocity
            #reversed_pdf[ivpa_min] = 0.0
            # calculate the updated first moment of the normalised pdf
            upar_integral += vpa_unnorm[ivpa] * pdf[ivpa,1,end,ir] * vpa.wgts[ivpa]
            #println("right ", ivpa, " ", upar_integral, " ", upar_end)
        end
        integral_excess = upar_integral - upar_end
        fraction_of_pdf = integral_excess / (vpa_unnorm[ivpa] * vpa.wgts[ivpa]) / pdf[ivpa,1,end,ir]
        #println("B fraction_of_pdf=", fraction_of_pdf)

        # Define so that when fraction_of_pdf=1 (when all of the contribution to the
        # integral from the point at ivpa is required to make upar_integral>upar_end) the
        # cut-off velocity is half way between ivpa-1 and ivpa, while when
        # fraction_of_pdf=0 (none of the contribution to the integral from the point at
        # ivpa is required to make upar_integral>upar_end) the cut-off is half way between
        # ivpa and ivpa+1.
        vmin = 0.5 * (vpa_unnorm[ivpa-1] + vpa_unnorm[ivpa]) +
               0.5 * fraction_of_pdf*(vpa_unnorm[ivpa+1] - vpa_unnorm[ivpa-1])

        #println("vmin=$vmin, v-no-interp=", vpa_unnorm[ivpa])
        #wmin = (-vmin - upar[end,ir]) / vthe[end,ir]
        #@loop_vpa ivpa begin
        #    reversed_pdf[ivpa] *= 0.5*(1.0 + tanh((vpa.grid[ivpa] - wmin) / cutoff_step_width))
        #end
        reversed_pdf[1:ivpa_min-1] .= 0
        reversed_pdf[ivpa_min] *= fraction_of_pdf
        #println("done second cutoff loop")

        # zeroth_vspace_moment = integrate_over_vspace(pdf[:,1,end,1], vpa.wgts)
        # @. vpa.scratch3 = pdf[:,1,end,1] * vpa.grid
        # first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # @. vpa.scratch3 *= vpa.grid
        # second_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # @loop_vpa ivpa begin
        #     println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa,1,end,ir], " zeroth_vspace_moment: ", zeroth_vspace_moment, 
        #         " first_vspace_moment: ", first_vspace_moment, " second_vspace_moment: ", second_vspace_moment, " stage: ", 2)
        # end
        # println(io_pdf_stages,"")

        # update the electrostatic potential at the boundary to be the value corresponding to the updated cutoff velocity
        #phi[end,ir] = me_over_mi * vthe[end,ir]^2 * vpa.grid[ivpa_min]^2
        phi[end,ir] = me_over_mi * vmin^2
        iv0 = findlast(x -> x<0.0, vpa_unnorm)
        pdf[1:iv0,1,end,ir] .= reversed_pdf[1:iv0]
        #println("after pdf ", pdf[:,1,end,ir])
        # obtain the normalisation constants needed to ensure the zeroth, first and second moments
        # of the modified pdf are 1, 0 and 1/2 respectively
        # will need vpa / vthe = wpa + upar/vthe
        @. vpa.scratch2 = vpa.grid + upar[end,ir] / vthe[end,ir]
        # first need to calculate int dwpa pdf = zeroth_moment
        zeroth_moment = integrate_over_vspace(pdf[:,1,end,ir], vpa.wgts)
        # calculate int dwpa wpa^2 * pdf = wpa2_moment
        @. vpa.scratch3 = pdf[:,1,end,ir] * vpa.grid^2
        wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # calculate int dwpa vpa^2 * pdf = vpa2_moment
        @. vpa.scratch3 = pdf[:,1,end,ir] * vpa.scratch2^2
        vpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # calculate int dwpa vpa^2 * wpa * pdf = vpa2_wpa_moment
        @. vpa.scratch3 = vpa.grid * vpa.scratch2^2 * pdf[:,1,end,ir]
        vpa2_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # calculate int dwpa wpa^2 * vpa^2 * pdf = vpa2_wpa2_moment
        @. vpa.scratch3 = vpa.grid^2 * vpa.scratch2^2 * pdf[:,1,end,ir]
        vpa2_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        
        normalisation_constant_A = 0.0
        normalisation_constant_B = 0.0
        normalisation_constant_C = 0.0
        if pdf_adjustment_option == "absvpa"
            # calculate int dwpa |vpa| * pdf = absvpa_moment
            @. vpa.scratch3 = pdf[:,1,end,ir] * abs(vpa.scratch2)
            absvpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa |vpa| * wpa * pdf = absvpa_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            absvpa_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa |vpa| * wpa^2 * pdf = absvpa_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            absvpa_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # assuming pdf_updated = pdf * (normalisation_constant_A + |vpa| * normalisation_constant_B + vpa^2 * normalisation_constant_C)
            # calculate the 'B' normalisation constant
            normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) / 
                (absvpa_wpa2_moment  - absvpa_wpa_moment * vpa2_wpa2_moment / vpa2_wpa_moment
                + wpa2_moment / zeroth_moment * (absvpa_wpa_moment * vpa2_moment / vpa2_wpa_moment - absvpa_moment))
            # calculate the 'A' normalisation constant
            normalisation_constant_A = (1 + normalisation_constant_B * 
                (vpa2_moment * absvpa_wpa_moment / vpa2_wpa_moment - absvpa_moment)) / zeroth_moment
            # calculate the 'C' normalisation constant
            normalisation_constant_C = -normalisation_constant_B * absvpa_wpa_moment / vpa2_wpa_moment
            # updated pdf is old pdf * (normalisation_constant_A + |vpa| * normalisation_constant_B + vpa^2 * normalisation_constant_C)
            @. pdf[:,1,end,ir] *= (normalisation_constant_A + abs(vpa.scratch2) * normalisation_constant_B 
                                + vpa.scratch2^2 * normalisation_constant_C)
        elseif pdf_adjustment_option == "vpa4"
            # calculate int dwpa vpa^4 * pdf = vpa4_moment
            @. vpa.scratch3 = vpa.scratch2^4 * pdf[:,1,end,ir]
            vpa4_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa * pf = vpa4_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa^2 * pdf = vpa4_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # assuming pdf_updated = pdf * (normalisation_constant_A + vpa^2 * normalisation_constant_B + vpa^4 * normalisation_constant_C)
            normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) /
                 (vpa2_wpa2_moment - vpa2_wpa_moment * vpa4_wpa2_moment / vpa4_wpa_moment
                 + wpa2_moment / zeroth_moment * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment))
            normalisation_constant_A = (1 + normalisation_constant_B 
                * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment)) / zeroth_moment
            normalisation_constant_C = -normalisation_constant_B * vpa2_wpa_moment / vpa4_wpa_moment
            @. pdf[:,1,end,ir] *= (normalisation_constant_A + vpa.scratch2^2 * normalisation_constant_B 
                                + vpa.scratch2^4 * normalisation_constant_C)
        elseif pdf_adjustment_option == "vpa4_gaussian"
            afac = 0.1
            bfac = 0.2
            # calculate int dwpa vpa^2 * exp(-vpa^2) * pdf = vpa2_moment
            @. vpa.scratch3 = pdf[:,1,end,ir] * vpa.scratch2^2 * exp(-afac * vpa.scratch2^2)
            vpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^2 * exp(-vpa^2) * wpa * pdf = vpa2_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            vpa2_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa wpa^2 * vpa^2 * exp(-vpa^2) * pdf = vpa2_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            vpa2_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * exp(-vpa^2) * pdf = vpa4_moment
            @. vpa.scratch3 = vpa.scratch2^4 * exp(-bfac * vpa.scratch2^2) * pdf[:,1,end,ir]
            vpa4_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa * pf = vpa4_wpa_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # calculate int dwpa vpa^4 * wpa^2 * pdf = vpa4_wpa2_moment
            @. vpa.scratch3 *= vpa.grid
            vpa4_wpa2_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
            # assuming pdf_updated = pdf * (normalisation_constant_A + vpa^2 * normalisation_constant_B + exp(-vpa^2) * vpa^4 * normalisation_constant_C)
            normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) /
                 (vpa2_wpa2_moment - vpa2_wpa_moment * vpa4_wpa2_moment / vpa4_wpa_moment
                 + wpa2_moment / zeroth_moment * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment))
            normalisation_constant_A = (1 + normalisation_constant_B 
                * (vpa2_wpa_moment * vpa4_moment / vpa4_wpa_moment - vpa2_moment)) / zeroth_moment
            normalisation_constant_C = -normalisation_constant_B * vpa2_wpa_moment / vpa4_wpa_moment
            #normalisation_constant_B = (0.5 - wpa2_moment / zeroth_moment) /
            #                           (vpa2_wpa2_moment
            #                            - wpa2_moment / zeroth_moment * vpa2_moment)
            #normalisation_constant_A = (1 - normalisation_constant_B 
            #                                * vpa2_moment) / zeroth_moment
            #normalisation_constant_C = 0.0
            @. pdf[:,1,end,ir] *= (normalisation_constant_A + exp(-afac * vpa.scratch2^2) * vpa.scratch2^2 * normalisation_constant_B 
                                + exp(-bfac * vpa.scratch2^2) * vpa.scratch2^4 * normalisation_constant_C)
        elseif pdf_adjustment_option == "no1st_vpa2"
            normalisation_constant_C = (1.0 - 0.5*zeroth_moment/wpa2_moment) /
                                       (vpa2_moment - zeroth_moment*vpa2_wpa2_moment / wpa2_moment)
            normalisation_constant_A = (0.5 - normalisation_constant_C*vpa2_wpa2_moment) / wpa2_moment
            @. pdf[:,1,end,ir] *= (normalisation_constant_A + vpa.scratch2^2 * normalisation_constant_C)
        else
            println("pdf_adjustment_option not recognised")
            stop()
        end

        moments.electron.constraints_A_coefficient[end,ir] = normalisation_constant_A
        moments.electron.constraints_B_coefficient[end,ir] = normalisation_constant_B
        moments.electron.constraints_C_coefficient[end,ir] = normalisation_constant_C

        # smooth the pdf at the boundary
        #for ivpa ∈ ivpa_min+1:vpa.n-1
        #    pdf[ivpa,1,end,ir] = (pdf[ivpa-1,1,end,ir] + pdf[ivpa+1,1,end,ir]) / 2.0
        #end

        # zeroth_vspace_moment = integrate_over_vspace(pdf[:,1,end,1], vpa.wgts)
        # @. vpa.scratch3 = pdf[:,1,end,1] * vpa.grid
        # first_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # @. vpa.scratch3 *= vpa.grid
        # second_vspace_moment = integrate_over_vspace(vpa.scratch3, vpa.wgts)
        # @loop_vpa ivpa begin
        #     println(io_pdf_stages, "vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa,1,end,ir], " zeroth_vspace_moment: ", zeroth_vspace_moment, 
        #         " first_vspace_moment: ", first_vspace_moment, " second_vspace_moment: ", second_vspace_moment, " stage: ", 3)
        # end
        # println(io_pdf_stages,"")
    end

    # # initialise the electron pdf for positive vpa to be the mirror reflection about vpa = 0
    # ivpa_zero = (vpa.n+1)÷2
    # ivpa_max = vpa.n
    # @. pdf[ivpa_zero+1:end,:,1,:] = pdf[ivpa_zero-1:-1:1,:,1,:]
    # # calculate the zeroth v-space moment of the normalised electron pdf:
    # # if unity to within specified tolerance, then the boundary condition is satisfied;
    # # otherwise, modify the cutoff velocity and repeat
    # @loop_r ir begin
    #     unity = 2.0
    #     while unity > 1.0
    #         unity = integrate_over_vspace(pdf[:,1,1,ir], vpa.wgts)
    #         # if unity > 1.0, then the cutoff velocity is too high so reduce it
    #         if unity > 1.0
    #             pdf[ivpa_max,1,1,ir] = 0.0
    #             ivpa_max -= 1
    #         end
    #     end
    #     phi[1,ir] = vthe[1,ir]^2 * vpa.grid[ivpa_max]^2
    # end
    # # repeat the above procedure for the boundary at z_max
    # @. pdf[ivpa_zero-1:-1:1,:,end,:] = pdf[ivpa_zero+1:end,:,end,:]
    # ivpa_max = 1
    # @loop_r ir begin
    #     unity = 2.0
    #     while unity > 1.0
    #         unity = integrate_over_vspace(pdf[:,1,end,ir], vpa.wgts)
    #         if unity > 1.0
    #             pdf[ivpa_max,1,end,ir] = 0.0
    #             ivpa_max += 1
    #         end
    #         phi[end,ir] = vthe[end,ir]^2 * vpa.grid[ivpa_max]^2
    #     end
    #     #println("unity: ", unity)
    # end
end


"""
update_electron_pdf_with_shooting_method is a function that using a shooting method to solve for the electron pdf

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor
The shooting method is 'explicit' in z, solving
    zdot_i * (pdf_{i+1} - pdf_{i})/dz_{i} + wpadot_{i} * d(pdf_{i})/dwpa = pdf_{i} * pre_factor_{i}

    INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    ppar = electron parallel pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_shooting_method!(pdf, dens, vthe, ppar, qpar, qpar_updated, phi, 
    ddens_dz, dppar_dz, dqpar_dz, dvth_dz, z, vpa, vpa_spectral, scratch_dummy, composition)

    # it is convenient to create a pointer to a scratch array to store the RHS of the linear 
    # system that includes all terms evaluated at the previous z location in the shooting method
    rhs = scratch_dummy.buffer_vpavperpzr_1
    # initialise the RHS to zero
    rhs .= 0.0
    # get critical velocities beyond which electrons are lost to the wall
    crit_speed_zmin, crit_speed_zmax = get_electron_critical_velocities(phi, vthe, composition.me_over_mi, z)
    # add the contribution to rhs from the term proportional to the pdf (rather than its derivatives)
    add_contribution_from_pdf_term!(rhs, pdf, ppar, vthe, dens, ddens_dz, dvth_dz, dqpar_dz, vpa.grid, z)
    # add the contribution to rhs from the wpa advection term
    add_contribution_from_wpa_advection!(rhs, pdf, vthe, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
    # shoot in z from incoming boundary (using sign of zdot to determine direction)
    # pdf_{i+1} = pdf_{i} - dz_{i} * rhs_{i}
    @loop_r_vperp ir ivperp begin
        for ivpa ∈ 1:vpa.n
            # deal with case of positive z-advection speed
            if vpa.grid[ivpa] > eps(mk_float)
                for iz ∈ 1:z.n-1
                    pdf[ivpa, ivperp, iz + 1, ir] = pdf[ivpa, ivperp, iz, ir] - (z.cell_width[iz] * rhs[ivpa, ivperp, iz, ir] 
                                                              / (vpa.grid[ivpa] * vthe[iz, ir]))
                end
            # deal with case of negative z-advection speed
            elseif vpa.grid[ivpa] < -eps(mk_float)
                for iz ∈ z.n:-1:2
                    pdf[ivpa, ivperp, iz - 1, ir] = pdf[ivpa, ivperp, iz, ir] + (z.cell_width[iz-1] * rhs[ivpa, ivperp, iz, ir] 
                                                              / (vpa.grid[ivpa] * vthe[iz, ir]))
                end
            # deal with case of zero z-advection speed
            else
                # hack for now until I figure out how to deal with the singular case
                @. pdf[ivpa, ivperp, :, ir] = 0.5 * (pdf[ivpa+1, ivperp, :, ir] + pdf[ivpa-1, ivperp, :, ir])
            end
        end
        # enforce the boundary condition at the walls
        for ivpa ∈ 1:vpa.n
            # find the ivpa index corresponding to the same magnitude 
            # of vpa but with opposite sign
            ivpamod = vpa.n - ivpa + 1
            # deal with wall at z_min
            if vpa.grid[ivpa] > eps(mk_float)
                iz = 1
                # electrons travelling sufficiently fast in the negative z-direction are lost to the wall
                # the maximum posative z-velocity that can be had is the amount by which electrons
                # can be accelerated by the electric field between the wall and the simulation boundary
                if vpa.grid[ivpa] > crit_speed_zmin
                    pdf[ivpa, ivperp, iz, ir] = 0.0
                else
                    pdf[ivpa, ivperp, iz, ir] = pdf[ivpamod, ivperp, iz, ir]
                end
            # deal with wall at z_max
            elseif vpa.grid[ivpa] < -eps(mk_float)
                iz = z.n
                # electrons travelling sufficiently fast in the positive z-direction are lost to the wall
                # the maximum negative z-velocity that can be had is the amount by which electrons
                # can be accelerated by the electric field between the wall and the simulation boundary
                if vpa.grid[ivpa] < crit_speed_zmax
                    pdf[ivpa, ivperp, iz, ir] = 0.0
                else
                    pdf[ivpa, ivperp, iz, ir] = pdf[ivpamod, ivperp, iz, ir]
                end
            end
        end
    end
    # the electron parallel heat flux is no longer consistent with the electron pdf
    qpar_updated = false
    # calculate the updated electron parallel heat flux
    calculate_electron_qpar_from_pdf!(qpar, ppar, vthe, pdf, vpa)
    for iz ∈ 1:z.n
        for ivpa ∈ 1:vpa.n
            println("z: ", z.grid[iz], " vpa: ", vpa.grid[ivpa], " pdf: ", pdf[ivpa, 1, iz, 1], " qpar: ", qpar[iz, 1], 
                " dppardz: ", dppar_dz[iz,1], " vth: ", vthe[iz,1], " ppar: ", ppar[iz,1], " dqpar_dz: ", dqpar_dz[iz,1],
                " dvth_dz: ", dvth_dz[iz,1], " ddens_dz: ", ddens_dz[iz,1])
        end
        println()
    end
    return nothing    
end

"""
use Picard iteration to solve the electron kinetic equation

The electron kinetic equation is:
    zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa = pdf * pre_factor
Picard iteration uses the previous iteration of the electron pdf to calculate the next iteration:
    zdot * d(pdf^{i+1})/dz + wpadot^{i} * d(pdf^{i})/dwpa = pdf^{i} * prefactor^{i}

INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
    dens = electron density
    vthe = electron thermal speed
    ppar = electron parallel pressure
    ddens_dz = z-derivative of the electron density
    dppar_dz = z-derivative of the electron parallel pressure
    dqpar_dz = z-derivative of the electron parallel heat flux
    dvth_dz = z-derivative of the electron thermal speed
    z = struct containing z-coordinate information
    vpa = struct containing vpa-coordinate information
    z_spectral = struct containing spectral information for the z-coordinate
    vpa_spectral = struct containing spectral information for the vpa-coordinate
    scratch_dummy = dummy arrays to be used for temporary storage
    max_electron_pdf_iterations = maximum number of iterations to use in the solution of the electron kinetic equation
OUTPUT:
    pdf = updated (modified) electron pdf
"""
function update_electron_pdf_with_picard_iteration!(pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz,
    z, vpa, vpa_spectral, scratch_dummy, max_electron_pdf_iterations)

    # it is convenient to create a pointer to a scratch array to store the RHS of the linear 
    # system that will be solved iteratively
    rhs = scratch_dummy.buffer_vpavperpzr_1
    # define residual to point to a scratch array;
    # to be filled with the difference between successive iterations of the electron pdf
    residual = scratch_dummy.buffer_vpavperpzr_2
    # define pdf_new to point to a scratch array;
    # to be filled with the updated electron pdf
    pdf_new = scratch_dummy.buffer_vpavperpzr_3
    # initialise the electron pdf convergence flag to false
    electron_pdf_converged = false
    # initialise the number of iterations in the solution of the electron kinetic equation to be 1
    iteration = 1
    while !electron_pdf_converged && (iteration < max_electron_pdf_iterations)
        # calculate the RHS of the linear system, consisting of all terms in which the
        # pdf is evaluated at the previous iteration.
        
        # initialise the RHS to zero
        rhs .= 0.0
        # add the contribution to rhs from the term proportional to the pdf (rather than its derivatives)
        add_contribution_from_pdf_term!(rhs, pdf, ppar, vthe, dens, ddens_dz, dvth_dz, dqpar_dz, vpa.grid, z)
        # add the contribution to rhs from the wpa advection term
        #add_contribution_from_wpa_advection!(rhs, pdf, vthe, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
        # loop over wpa locations, solving the linear system at each location;
        # care must be taken when wpa = 0, as the linear system is singular in this case
        @loop_r_vperp_vpa ir ivperp ivpa begin
            # modify the rhs vector to account for the pre-factor multiplying d(pdf)/dz
            if abs(vpa.grid[ivpa]) < eps()
                # hack for now until I figure out how to deal with the singular linear system
                rhs[ivpa, ivperp, :, ir] .= 0.0
            else
                @loop_z iz begin
                    #rhs[ivpa, ivperp, iz, ir] /= -(vpa.grid[ivpa] * vthe[iz, ir])
                    rhs[ivpa, ivperp, iz, ir] = -rhs[ivpa, ivperp, iz, ir]
                end
            end
            # solve the linear system at this (wpa, wperp, r) location
            pdf_new[ivpa, ivperp, :, ir] = z.differentiation_matrix \ rhs[ivpa, ivperp, :, ir]
        end
        # calculate the difference between successive iterations of the electron pdf
        @. residual = pdf_new - pdf
        # check to see if the electron pdf has converged to within the specified tolerance
        average_residual, electron_pdf_converged = check_electron_pdf_convergence(residual)
        # prepare for the next iteration by updating the pdf
        @. pdf = pdf_new
        # if converged, exit the loop; otherwise, increment the iteration counter
        if electron_pdf_converged
            break
        else
            iteration +=1
        end
    end
    # if the maximum number of iterations has been exceeded, print a warning
    if !electron_pdf_converged
        # need to exit or handle this appropriately
        println("!!! max number of iterations for electron pdf update exceeded !!!")
        @loop_z iz begin
            println("z: ", z.grid[iz], " pdf: ", pdf[10, 1, iz, 1])
        end
    end
    
    return nothing
end


"""
electron_kinetic_equation_residual! calculates the residual of the (time-independent) electron kinetic equation
INPUTS:
    residual = dummy array to be filled with the residual of the electron kinetic equation
OUTPUT:
    residual = updated residual of the electron kinetic equation
"""
function electron_kinetic_equation_residual!(residual, max_term, single_term, pdf, dens, upar, vth, ppar, upar_ion,
                                             ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
                                             z, vperp, vpa, z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                                             collisions, num_diss_params, dt_electron)

    # initialise the residual to zero                                             
    begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        residual[ivpa,ivperp,iz,ir] = 0.0
    end
    # calculate the contribution to the residual from the z advection term
    electron_z_advection!(residual, pdf, upar, vth, z_advect, z, vpa.grid, z_spectral, scratch_dummy)
    #dt_max_zadv = simple_z_advection!(residual, pdf, vth, z, vpa.grid, dt_electron)
    #single_term .= residual
    #max_term .= abs.(residual)
    #println("z_adv residual = ", maximum(abs.(single_term)))
    #println("z_advection: ", sum(residual), " dqpar_dz: ", sum(abs.(dqpar_dz)))
    #calculate_contribution_from_z_advection!(residual, pdf, vth, z, vpa.grid, z_spectral, scratch_dummy)
    # add in the contribution to the residual from the wpa advection term
    electron_vpa_advection!(residual, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, 
                            vpa_advect, vpa, vpa_spectral, scratch_dummy)#, z)
    #dt_max_vadv = simple_vpa_advection!(residual, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, vpa, dt_electron)
    #@. single_term = residual - single_term
    #max_term .= max.(max_term, abs.(single_term))
    #@. single_term = residual
    #println("v_adv residual = ", maximum(abs.(single_term)))
    #add_contribution_from_wpa_advection!(residual, pdf, vth, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
    # add in the contribution to the residual from the term proportional to the pdf
    add_contribution_from_pdf_term!(residual, pdf, ppar, vth, dens, ddens_dz, dvth_dz, dqpar_dz, vpa.grid, z)
    #@. single_term = residual - single_term
    #max_term .= max.(max_term, abs.(single_term))
    #@. single_term = residual
    #println("pdf_term residual = ", maximum(abs.(single_term)))
    # @loop_vpa ivpa begin
    #     @loop_z iz begin
    #         println("LHS: ", residual[ivpa,1,iz,1], " vpa: ", vpa.grid[ivpa], " z: ", z.grid[iz], " dvth_dz: ", dvth_dz[iz,1], " type: ", 1) 
    #     end
    #     println("")
    # end
    # println("")
    # add in numerical dissipation terms
    add_dissipation_term!(residual, pdf, scratch_dummy, z_spectral, z, vpa, vpa_spectral, num_diss_params)
    #@. single_term = residual - single_term
    #println("dissipation residual = ", maximum(abs.(single_term)))
    #max_term .= max.(max_term, abs.(single_term))
    # add in particle and heat source term(s)
    #@. single_term = residual
    #add_source_term!(residual, vpa.grid, z.grid, dvth_dz)
    #@. single_term = residual - single_term
    #max_term .= max.(max_term, abs.(single_term))
    #stop()
    # @loop_vpa ivpa begin
    #     @loop_z iz begin
    #         println("total_residual: ", residual[ivpa,1,iz,1], " vpa: ", vpa.grid[ivpa], " z: ", z.grid[iz], " dvth_dz: ", dvth_dz[iz,1], " type: ", 2) 
    #     end
    #     println("")
    # end
    # stop()
    #dt_max = min(dt_max_zadv, dt_max_vadv)

    if collisions.krook_collision_frequency_prefactor_ee > 0.0
        # Add a Krook collision operator
        # Set dt=-1 as we update the residual here rather than adding an update to
        # 'fvec_out'.
        electron_krook_collisions!(residual, pdf, dens, upar, upar_ion, vth,
                                   collisions, vperp, vpa, -1.0)
    end

    dt_max = dt_electron
    #println("dt_max: ", dt_max, " dt_max_zadv: ", dt_max_zadv, " dt_max_vadv: ", dt_max_vadv)
    return dt_max
end

function simple_z_advection!(advection_term, pdf, vth, z, vpa, dt_max_in)
    dt_max = dt_max_in
    # take the z derivative of the input pdf
    begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        speed = vth[iz,ir] * vpa[ivpa]
        dt_max = min(dt_max, 0.5*z.cell_width[iz]/(max(abs(speed),1e-3)))
        if speed > 0
            if iz == 1
                #dpdf_dz  = pdf[ivpa,ivperp,iz,ir]/z.cell_width[1]
                dpdf_dz = 0.0
            else
                dpdf_dz = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa,ivperp,iz-1,ir])/z.cell_width[iz-1]
            end
        elseif speed < 0
            if iz == z.n
                #dpdf_dz = -pdf[ivpa,ivperp,iz,ir]/z.cell_width[z.n-1]
                dpdf_dz = 0.0
            else
                dpdf_dz = (pdf[ivpa,ivperp,iz+1,ir] - pdf[ivpa,ivperp,iz,ir])/z.cell_width[iz]
            end
        else
            if iz == 1
                dpdf_dz = (pdf[ivpa,ivperp,iz+1,ir] - pdf[ivpa,ivperp,iz,ir])/z.cell_width[1]
            elseif iz == z.n
                dpdf_dz = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa,ivperp,iz-1,ir])/z.cell_width[z.n-1]
            else
                dpdf_dz = (pdf[ivpa,ivperp,iz+1,ir] - pdf[ivpa,ivperp,iz,ir])/z.cell_width[1]
            end
        end
        advection_term[ivpa,ivperp,iz,ir] += speed * dpdf_dz
    end
    return dt_max
end

#################
# NEED TO ADD PARTICLE/HEAT SOURCE INTO THE ELECTRON KINETIC EQUATION TO FIX amplitude
#################


function simple_vpa_advection!(advection_term, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, vpa, dt_max_in)
    dt_max = dt_max_in
    # take the vpa derivative in the input pdf
    begin_r_z_vperp_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        speed = ((vth[iz,ir] * dppar_dz[iz,ir] + vpa.grid[ivpa] * dqpar_dz[iz,ir]) 
                / (2 * ppar[iz,ir]) - vpa.grid[ivpa]^2 * dvth_dz[iz,ir])
        dt_max = min(dt_max, 0.5*vpa.cell_width[ivpa]/max(abs(speed),1e-3))
        if speed > 0
            if ivpa == 1
                dpdf_dvpa  = pdf[ivpa,ivperp,iz,ir]/vpa.cell_width[1]
            else
                dpdf_dvpa = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa-1,ivperp,iz,ir])/vpa.cell_width[ivpa-1]
            end
        elseif speed < 0
            if ivpa == vpa.n
                dpdf_dvpa = -pdf[ivpa,ivperp,iz,ir]/vpa.cell_width[vpa.n-1]
            else
                dpdf_dvpa = (pdf[ivpa+1,ivperp,iz,ir] - pdf[ivpa,ivperp,iz,ir])/vpa.cell_width[ivpa]
            end
        else
            if ivpa == 1
                dpdf_dvpa = (pdf[ivpa+1,ivperp,iz,ir] - pdf[ivpa,ivperp,iz,ir])/vpa.cell_width[1]
            elseif ivpa == vpa.n
                dpdf_dvpa = (pdf[ivpa,ivperp,iz,ir] - pdf[ivpa-1,ivperp,iz,ir])/vpa.cell_width[vpa.n-1]
            else
                dpdf_dvpa = (pdf[ivpa+1,ivperp,iz,ir] - pdf[ivpa,ivperp,iz,ir])/vpa.cell_width[1]
            end
        end
        advection_term[ivpa,ivperp,iz,ir] += speed * dpdf_dvpa
    end
    return dt_max
end

function add_source_term!(source_term, vpa, z, dvth_dz)
    # add in particle and heat source term(s)
    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
    #    source_term[ivpa,ivperp,iz,ir] -= 40*exp(-vpa[ivpa]^2)*exp(-z[iz]^2)
    #    source_term[ivpa,ivperp,iz,ir] -= vpa[ivpa]*exp(-vpa[ivpa]^2)*(2*vpa[ivpa]^2-3)*dvth_dz[iz,ir]
    end
    return nothing
end

function add_dissipation_term!(residual, pdf, scratch_dummy, z_spectral, z, vpa, vpa_spectral, num_diss_params)
    dummy_zr1 = @view scratch_dummy.dummy_zrs[:,:,1]
    dummy_zr2 = @view scratch_dummy.buffer_vpavperpzr_1[1,1,:,:]
    buffer_r_1 = @view scratch_dummy.buffer_rs_1[:,1]
    buffer_r_2 = @view scratch_dummy.buffer_rs_2[:,1]
    buffer_r_3 = @view scratch_dummy.buffer_rs_3[:,1]
    buffer_r_4 = @view scratch_dummy.buffer_rs_4[:,1]
    # add in numerical dissipation terms
    #@loop_vperp_vpa ivperp ivpa begin
    #    @views derivative_z!(dummy_zr1, pdf[ivpa,ivperp,:,:], buffer_r_1, buffer_r_2, buffer_r_3,
    #                         buffer_r_4, z_spectral, z)
    #    @views derivative_z!(dummy_zr2, dummy_zr1, buffer_r_1, buffer_r_2, buffer_r_3,
    #                         buffer_r_4, z_spectral, z)
    #    @. residual[ivpa,ivperp,:,:] -= num_diss_params.electron.z_dissipation_coefficient * dummy_zr2
    #end
    begin_r_z_vperp_region()
    @loop_r_z_vperp ir iz ivperp begin
        #@views derivative!(vpa.scratch, pdf[:,ivperp,iz,ir], vpa, false)
        #@views derivative!(vpa.scratch2, vpa.scratch, vpa, false)
        #@. residual[:,ivperp,iz,ir] -= num_diss_params.electron.vpa_dissipation_coefficient * vpa.scratch2
        @views second_derivative!(vpa.scratch, pdf[:,ivperp,iz,ir], vpa, vpa_spectral)
        @. residual[:,ivperp,iz,ir] -= num_diss_params.electron.vpa_dissipation_coefficient * vpa.scratch
    end
    #stop()
    return nothing
end

"""
update_electron_pdf! iterates to find a solution for the electron pdf
from the electron kinetic equation:
    zdot * d(pdf^{i})/dz + wpadot^{i} * d(pdf^{i})/dwpa + wpedot^{i} * d(pdf^{i})/dwpe  = pdf^{i+1} * pre_factor^{i}
NB: zdot, wpadot, wpedot and pre-factor contain electron fluid quantities, that will have been updated
independently of the electron pdf;
NB: wpadot, wpedot and pre-factor all contain the electron parallel heat flux,
which itself depends on the electron pdf.

INPUTS:
    pdf = modified electron pdf @ previous time level = (true electron pdf / dens_e) * vth_e
OUTPUT:
    pdf = updated (modified) electron pdf
"""
# function update_electron_pdf!(pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                               max_electron_pdf_iterations, z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#     # iterate the electron kinetic equation until the electron pdf is converged
#     # or the specified maximum number of iterations is exceeded
#     electron_pdf_converged = false
#     iteration = 1

#     println("pdf_update_electron_pdf: ", sum(abs.(pdf)))
#     while !electron_pdf_converged && (iteration < max_electron_pdf_iterations)
#         # calculate the contribution to the kinetic equation due to z advection
#         # and store in the dummy array scratch_dummy.buffer_vpavperpzr_1
#         @views calculate_contribution_from_z_advection!(scratch_dummy.buffer_vpavperpzr_1, pdf, vthe, z, vpa.grid, z_spectral, scratch_dummy)
#         # calculate the contribution to the kinetic equation due to w_parallel advection and add to the z advection term
#         @views add_contribution_from_wpa_advection!(scratch_dummy.buffer_vpavperpzr_1, pdf, vthe, ppar, dppar_dz, dqpar_dz, dvth_dz, vpa, vpa_spectral)
#         # calculate the pre-factor multiplying the modified electron pdf
#         # and store in the dummy array scratch_dummy.buffer_vpavperpzr_2
#         @views calculate_pdf_dot_prefactor!(scratch_dummy.buffer_vpavperpzr_2, ppar, vthe, dens, ddens_dz, dvth_dz, dqpar_dz, vpa.grid)
#         # update the electron pdf
#         #pdf_new = scratch_dummy.buffer_vpavperpzr_1
#         #println("advection_terms: ", sum(abs.(advection_terms)), "pdf_dot_prefactor: ", sum(abs.(pdf_dot_prefactor)))
#         @. scratch_dummy.buffer_vpavperpzr_1 /= scratch_dummy.buffer_vpavperpzr_2
#         # check to see if the electron pdf has converged to within the specified tolerance
#         check_electron_pdf_convergence!(electron_pdf_converged, scratch_dummy.buffer_vpavperpzr_1, pdf)
#         # prepare for the next iteration
#         @. pdf = scratch_dummy.buffer_vpavperpzr_1
#         iteration += 1
#     end
#     if !electron_pdf_converged
#         # need to exit or handle this appropriately
#         println("!!!max number of iterations for electron pdf update exceeded!!!")
#     end
#     return nothing
# end



# """
# use the biconjugate gradient stabilized method to solve the electron kinetic equation
# """
# function update_electron_pdf_bicgstab!(pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#     max_electron_pdf_iterations, z, vpa, z_spectral, vpa_spectral, scratch_dummy)
 
#     for iz in 1:z.n
#         println("z: ", z.grid[iz], " pdf: ", pdf[10, 1, iz, 1], " iteration: ", 1, " z.n: ", z.n)
#     end

#     # set various arrays to point to a corresponding dummy array scratch_dummy.buffer_vpavperpzr_X
#     # so that it is easier to understand what is going on
#     residual = scratch_dummy.buffer_vpavperpzr_1
#     residual_hat = scratch_dummy.buffer_vpavperpzr_2
#     pvec = scratch_dummy.buffer_vpavperpzr_3
#     vvec = scratch_dummy.buffer_vpavperpzr_4
#     tvec = scratch_dummy.buffer_vpavperpzr_5

#     # calculate the residual of the electron kinetic equation for the initial guess of the electron pdf
#     electron_kinetic_equation_residual!(residual, pdf, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                         z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#     @. residual = -residual

#     for iz in 1:z.n
#         println("z: ", z.grid[iz], " residual: ", residual[10, 1, iz, 1], " iteration: ", -1, " z.n: ", z.n)
#     end

#     residual_hat .= residual + pdf

#     # calculate the inner product of the residual and residual_hat
#     rho = dot(residual_hat, residual)
#     #@views rho = integral(residual_hat[10, 1, :, 1], residual[10,1,:,1], z.wgts)

#     println("rho: ", rho)

#     pvec .= residual

#     # iterate the electron kinetic equation until the electron pdf is converged
#     # or the specified maximum number of iterations is exceeded
#     electron_pdf_converged = false
#     iteration = 1

#     while !electron_pdf_converged && (iteration < max_electron_pdf_iterations)
#         # calculate the residual of the electron kinetic equation with the residual of the previous iteration
#         # as the input electron pdf
#         electron_kinetic_equation_residual!(vvec, pvec, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                             z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#         alpha = rho / dot(residual_hat, vvec)
#         #alpha = rho / integral(residual_hat[10,1,:,1], vvec[10,1,:,1], z.wgts)
#         println("alpha: ", alpha)
#         @. pdf += alpha * pvec
#         @. residual -= alpha * vvec
#         # check to see if the electron pdf has converged to within the specified tolerance
#         check_electron_pdf_convergence!(electron_pdf_converged, residual)
#         if electron_pdf_converged
#             break
#         end
#         electron_kinetic_equation_residual!(tvec, residual, dens, vthe, ppar, ddens_dz, dppar_dz, dqpar_dz, dvth_dz, 
#                                             z, vpa, z_spectral, vpa_spectral, scratch_dummy)
#         omega = dot(tvec, residual) / dot(tvec, tvec)
#         #omega = integral(tvec[10,1,:,1],residual[10,1,:,1],z.wgts)/integral(tvec[10,1,:,1],tvec[10,1,:,1],z.wgts)
#         @. pdf += omega * residual
#         @. residual -= omega * tvec
#         check_electron_pdf_convergence!(electron_pdf_converged, residual)
#         if electron_pdf_converged
#             break
#         end
#         #rho_new = integral(residual_hat[10,1,:,1],residual[10,1,:,1],z.wgts)
#         rho_new = dot(residual_hat, residual)
#         beta = (rho_new / rho) * (alpha / omega)
#         println("rho_new: ", rho_new, " beta: ", beta)
#         #@. pvec -= omega * vvec
#         #@. pvec *= beta
#         #@. pvec += residual
#         @. pvec = residual + beta * (pvec - omega * vvec)
#         # prepare for the next iteration
#         rho = rho_new
#         iteration += 1
#         println()
#         if iteration == 2
#             for iz in 1:z.n
#                 println("z: ", z.grid[iz], " pvec: ", pvec[10, 1, iz, 1], " iteration: ", -2, " z.n: ", z.n)
#             end
#         end
#         for iz in 1:z.n
#             println("z: ", z.grid[iz], " pdf: ", pdf[10, 1, iz, 1], " iteration: ", iteration, " z.n: ", z.n)
#         end
#     end
#     if !electron_pdf_converged
#         # need to exit or handle this appropriately
#         println("!!!max number of iterations for electron pdf update exceeded!!!")
#     end
#     return nothing
# end

"""
calculates the contribution to the residual of the electron kinetic equation from the z advection term:
residual = zdot * d(pdf)/dz + wpadot * d(pdf)/dwpa - pdf * pre_factor
INPUTS:
    z_advection_term = dummy array to be filled with the contribution to the residual from the z advection term
    pdf = modified electron pdf used in the kinetic equation = (true electron pdf / dens_e) * vth_e
    vthe = electron thermal speed
    z = z grid
    vpa = v_parallel grid
    z_spectral = spectral representation of the z grid
    scratch_dummy = dummy array to be used for temporary storage
OUTPUT:
    z_advection_term = updated contribution to the residual from the z advection term
"""
function calculate_contribution_from_z_advection!(z_advection_term, pdf, vthe, z, vpa, spectral, scratch_dummy)
    # calculate the z-derivative of the electron pdf and store in the dummy array
    # scratch_dummy.buffer_vpavperpzr_1
    derivative_z!(z_advection_term, pdf,
                  scratch_dummy.buffer_vpavperpr_1,
                  scratch_dummy.buffer_vpavperpr_2, scratch_dummy.buffer_vpavperpr_3,
                  scratch_dummy.buffer_vpavperpr_4, spectral, z)
    # contribution from the z-advection term is wpa * vthe * d(pdf)/dz
    begin_r_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @. z_advection_term[ivpa,ivperp,iz,ir] = z_advection_term[ivpa,ivperp,iz,ir] #* vpa[ivpa] * vthe[:, :]
    end
    return nothing
end

function add_contribution_from_wpa_advection!(residual, pdf, vth, ppar, dppar_dz, dqpar_dz, dvth_dz,
                                                    vpa, vpa_spectral)
    begin_r_z_vperp_region()
    @loop_r_z_vperp ir iz ivperp begin
        # calculate the wpa-derivative of the pdf and store in the scratch array vpa.scratch
        @views derivative!(vpa.scratch, pdf[:, ivperp, iz, ir], vpa, vpa_spectral)
        # contribution from the wpa-advection term is ( ) * d(pdf)/dwpa
        @. residual[:,ivperp,iz,ir] += vpa.scratch * (0.5 * vth[iz,ir] / ppar[iz,ir] * dppar_dz[iz,ir]
                + 0.5 * vpa.grid / ppar[iz,ir] * dqpar_dz[iz,ir] - vpa.grid^2 * dvth_dz[iz,ir])
    end
    return nothing
end

# calculate the pre-factor multiplying the modified electron pdf
function calculate_pdf_dot_prefactor!(pdf_dot_prefactor, ppar, vth, dens, ddens_dz, dvth_dz, dqpar_dz, vpa)
    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @. pdf_dot_prefactor[ivpa,ivperp,iz,ir] = 0.5 * dqpar_dz[iz,ir] / ppar[iz,ir] - vpa[ivpa] * vth[iz,ir] * 
        (ddens_dz[iz,ir] / dens[iz,ir] + dvth_dz[iz,ir] / vth[iz,ir])
    end
    return nothing
end

# add contribution to the residual coming from the term proporational to the pdf
function add_contribution_from_pdf_term!(residual, pdf, ppar, vth, dens, ddens_dz, dvth_dz, dqpar_dz, vpa, z)
    begin_r_z_vperp_vpa_region()
    @loop_r_z ir iz begin
        this_dqpar_dz = dqpar_dz[iz,ir]
        this_ppar = ppar[iz,ir]
        this_vth = vth[iz,ir]
        this_ddens_dz = ddens_dz[iz,ir]
        this_dens = dens[iz,ir]
        this_dvth_dz = dvth_dz[iz,ir]
        this_vth = vth[iz,ir]
        @loop_vperp_vpa ivperp ivpa begin
            residual[ivpa,ivperp,iz,ir] -= (-0.5 * this_dqpar_dz / this_ppar - vpa[ivpa] * this_vth *
                                    (this_ddens_dz / this_dens - this_dvth_dz / this_vth)) * pdf[ivpa,ivperp,iz,ir]
            #residual[ivpa, ivperp, :, :] -= (-0.5 * dqpar_dz[:, :] / ppar[:, :]) * pdf[ivpa, ivperp, :, :]
        end
    end
    return nothing
end

# function check_electron_pdf_convergence!(electron_pdf_converged, pdf_new, pdf)
#     # check to see if the electron pdf has converged to within the specified tolerance
#     # NB: the convergence criterion is based on the average relative difference between the
#     # new and old electron pdfs
#     println("sum(abs(pdf)): ", sum(abs.(pdf)), " sum(abs(pdf_new)): ", sum(abs.(pdf_new)))
#     println("sum(abs(pdfdiff)): ", sum(abs.(pdf_new - pdf)) / sum(abs.(pdf)), " length: ", length(pdf))
#     average_relative_error = sum(abs.(pdf_new - pdf)) / sum(abs.(pdf))
#     electron_pdf_converged = (average_relative_error < 1e-3)
#     println("average_relative_error: ", average_relative_error, " electron_pdf_converged: ", electron_pdf_converged)
#     return nothing
# end

function get_electron_critical_velocities(phi, vthe, me_over_mi, z)
    # get the critical velocities beyond which electrons are lost to the wall
    if z.irank == 0 && block_rank[] == 0
        crit_speed_zmin = sqrt(phi[1, 1] / (me_over_mi * vthe[1, 1]^2))
    else
        crit_speed_zmin = 0.0
    end
    @serial_region begin
        crit_speed_zmin = MPI.Bcast(crit_speed_zmin, 0, z.comm)
    end
    crit_speed_zmin = MPI.Bcast(crit_speed_zmin, 0, comm_block[])

    if z.irank == z.nrank - 1 && block_rank[] == 0
        crit_speed_zmax = -sqrt(max(phi[end, 1],0.0) / (me_over_mi * vthe[end, 1]^2))
    else
        crit_speed_zmin = 0.0
    end
    @serial_region begin
        crit_speed_zmax = MPI.Bcast(crit_speed_zmax, z.nrank-1, z.comm)
    end
    crit_speed_zmax = MPI.Bcast(crit_speed_zmax, 0, comm_block[])

    return crit_speed_zmin, crit_speed_zmax
end

function check_electron_pdf_convergence(residual, pdf, upar, vthe, z, vpa)
    #average_residual = 0.0
    # @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
    #     if pdf[ivpa, ivperp, iz, ir] > eps(mk_float)
    #         average_residual += abs(residual[ivpa, ivperp, iz, ir]) / pdf[ivpa, ivperp, iz, ir]
    #     else
    #         average_residual += abs(residual[ivpa, ivperp, iz, ir])
    #     end
    #     average_residual /= length(residual)
    # end

    # Only sum residual over points that are not set by the sheath boundary condition, as
    # those that are set by the sheath boundary condition are not used by the time
    # advance, and so might not converge to 0.
    # First, sum the contributions from the bulk of the domain
    begin_r_z_vperp_region()
    sum_residual = 0.0
    sum_pdf = 0.0
    @loop_r_z ir iz begin
        if z.irank == 0 && iz == 1
            vpa_unnorm_lower = @. vpa.scratch3 = vthe[1,ir] * vpa.grid + upar[1,ir]
            iv0_end = findfirst(x -> x>0.0, vpa_unnorm_lower) - 1
        else
            iv0_end = vpa.n
        end
        if z.irank == z.nrank-1 && iz == z.n
            vpa_unnorm_upper = @. vpa.scratch3 = vthe[end,ir] * vpa.grid + upar[end,ir]
            iv0_start = findlast(x -> x>0.0, vpa_unnorm_upper) + 1
        else
            iv0_start = 1
        end
        @loop_vperp ivperp begin
            sum_residual += sum(abs.(@view residual[iv0_start:iv0_end,ivperp,iz,ir]))
            # account for the fact that we want dg/dt << vthe/L * g, but 
            # residual is normalized by c_ref/L_ref * g
            sum_pdf += sum(abs.(@view pdf[iv0_start:iv0_end,ivperp,iz,ir]) * vthe[iz,ir])
        end
    end
    sum_residual, sum_pdf = MPI.Allreduce([sum_residual, sum_pdf], +, comm_world)

    average_residual = sum_residual / sum_pdf

    electron_pdf_converged = (average_residual < 1e-3)

    return average_residual, electron_pdf_converged
end

function check_electron_pdf_convergence(residual)
    begin_r_z_vperp_region()
    sum_residual = 0.0
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        sum_residual += abs.(residual[ivpa,ivperp,iz,ir])
    end
    sum_residual, sum_length = MPI.Allreduce((sum_residual, length(residual) / block_size[]), +, comm_world)
    average_residual = sum_residual / sum_length
    electron_pdf_converged = (average_residual < 1e-3)
    return average_residual, electron_pdf_converged
end

end
