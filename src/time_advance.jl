"""
"""
module time_advance

export setup_time_advance!
export time_advance!

using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: _block_synchronize
using ..debugging
using ..file_io: write_data_to_ascii, write_data_to_binary, debug_dump
using ..looping
using ..moment_constraints: enforce_moment_constraints!, hard_force_moment_constraints!
using ..moment_kinetics_structs: scratch_pdf
using ..velocity_moments: update_moments!, reset_moments_status!
using ..velocity_moments: update_density!, update_upar!, update_ppar!, update_qpar!
using ..initial_conditions: enforce_z_boundary_condition!, enforce_boundary_conditions!
using ..initial_conditions: enforce_z_boundary_condition_moments!
using ..initial_conditions: enforce_vpa_boundary_condition!
using ..advection: setup_advection, update_boundary_indices!
using ..z_advection: update_speed_z!, z_advection!
using ..r_advection: update_speed_r!, r_advection!
using ..vpa_advection: update_speed_vpa!, vpa_advection!
using ..charge_exchange: charge_exchange_collisions!
using ..ionization: ionization_collisions!
using ..numerical_dissipation: vpa_boundary_buffer_decay!,
                               vpa_boundary_buffer_diffusion!, vpa_dissipation!,
                               z_dissipation!, vpa_boundary_force_decreasing!,
                               force_minimum_pdf_value!
using ..source_terms: source_terms!
using ..continuity: continuity_equation!
using ..force_balance: force_balance!
using ..energy_equation: energy_equation!
using ..em_fields: setup_em_fields, update_phi!
using ..semi_lagrange: setup_semi_lagrange

@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

using Dates
using Plots
using ..post_processing: draw_v_parallel_zero!

"""
"""
mutable struct advance_info
    vpa_advection::Bool
    z_advection::Bool
    cx_collisions::Bool
    ionization_collisions::Bool
    source_terms::Bool
    numerical_dissipation::Bool
    continuity::Bool
    force_balance::Bool
    energy::Bool
    rk_coefs::Array{mk_float,2}
end

"""
create arrays and do other work needed to setup
the main time advance loop.
this includes creating and populating structs
for Chebyshev transforms, velocity space moments,
EM fields, semi-Lagrange treatment, and advection terms
"""
function setup_time_advance!(pdf, vpa, z, r, z_spectral, composition, drive_input,
                             moments, t_input, collisions, species, num_diss_params)
    # define some local variables for convenience/tidiness
    n_species = composition.n_species
    n_ion_species = composition.n_ion_species
    # create array containing coefficients needed for the Runge Kutta time advance
    rk_coefs = setup_runge_kutta_coefficients(t_input.n_rk_stages)
    # create the 'advance' struct to be used in later Euler advance to
    # indicate which parts of the equations are to be advanced concurrently.
    # if no splitting of operators, all terms advanced concurrently;
    # else, will advance one term at a time.
    advance = setup_advance_flags(moments, composition, t_input.split_operators, collisions, rk_coefs)
    # create structure r_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in r
    begin_serial_region()
    r_advect = setup_advection(n_species, r, vpa, z)
    # initialise the r advection speed
    begin_s_z_vpa_region()
    @loop_s is begin
        @views update_speed_r!(r_advect[is], moments.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(r_advect[is], loop_ranges[].vpa, loop_ranges[].z)
    end
    # enforce prescribed boundary condition in r on the distribution function f
    # PLACEHOLDER
    #@views enforce_r_boundary_condition!(pdf.unnorm, r.bc, r_advect, vpa, z, composition)


    # create structure z_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in z
    begin_serial_region()
    z_advect = setup_advection(n_species, z, vpa, r)
    # initialise the z advection speed
    begin_s_r_vpa_region()
    @loop_s is begin
        @views update_speed_z!(z_advect[is], moments.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, 0.0)
        # initialise the upwind/downwind boundary indices in z
        update_boundary_indices!(z_advect[is], loop_ranges[].vpa, loop_ranges[].r)
    end

    begin_serial_region()

    # create an array of structs containing scratch arrays for the pdf and low-order moments
    # that may be evolved separately via fluid equations
    scratch = setup_scratch_arrays(moments, pdf.norm, t_input.n_rk_stages)
    # setup dummy arrays
    scratch_dummy_sr = allocate_float(r.n, composition.n_species)
    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and eventually the electromagnetic fields
    fields = setup_em_fields(z.n, r.n, drive_input.force_phi, drive_input.amplitude, drive_input.frequency)
    # initialize the electrostatic potential
    begin_s_r_z_region()
    update_phi!(fields, scratch[1], z, r, composition)
    begin_serial_region()
    @serial_region begin
        # save the initial phi(z) for possible use later (e.g., if forcing phi)
        fields.phi0 .= fields.phi
    end
    # create structure vpa_advect whose members are the arrays needed to compute
    # the advection term(s) appearing in the split part of the GK equation dealing
    # with advection in vpa
    vpa_advect = setup_advection(n_species, vpa, z, r)
    # initialise the vpa advection speed
    begin_s_r_z_region()
    update_speed_vpa!(vpa_advect, fields, scratch[1], moments, vpa, z, r, composition,
                      collisions, 0.0, z_spectral)
    if moments.evolve_upar
        nspec = n_species
    else
        nspec = n_ion_species
    end
    begin_serial_region()
    @serial_region begin
        for is ∈ 1:nspec
            # initialise the upwind/downwind boundary indices in vpa
            update_boundary_indices!(vpa_advect[is], 1:z.n, 1:r.n)
        end
    end

    # ensure initial pdf has no negative values
    force_minimum_pdf_value!(pdf.norm, num_diss_params)
    # enforce boundary conditions and moment constraints to ensure a consistent initial
    # condition
    enforce_boundary_conditions!(pdf.norm, moments.dens, moments.upar, moments.ppar,
        moments, vpa.bc, z.bc, vpa, z, r, vpa_advect, z_advect, composition)
    # Ensure normalised pdf exactly obeys integral constraints if evolving moments
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        @views hard_force_moment_constraints!(pdf.norm[:,iz,ir,is], moments, vpa)
    end
    # update unnormalised pdf, moments and phi in case they were affected by applying
    # boundary conditions or constraints to the pdf
    update_pdf_unnorm!(pdf, moments, scratch[1].temp_z_s, composition, vpa)
    update_moments!(moments, pdf.norm, vpa, z, r, composition)
    update_phi!(fields, scratch[1], z, r, composition)

    # create an array of structures containing the arrays needed for the semi-Lagrange
    # solve and initialize the characteristic speed and departure indices
    # so that the code can gracefully run without using the semi-Lagrange
    # method if the user specifies this
    z_SL = setup_semi_lagrange(z.n, vpa.n, r.n)
    vpa_SL = setup_semi_lagrange(vpa.n, z.n, r.n)
    r_SL = setup_semi_lagrange(r.n, vpa.n, z.n)

    # Ensure all processes are synchronized at the end of the setup
    _block_synchronize()

    return moments, fields, vpa_advect, z_advect, r_advect, vpa_SL, z_SL, r_SL, scratch,
           advance, scratch_dummy_sr
end

"""
create the 'advance_info' struct to be used in later Euler advance to
indicate which parts of the equations are to be advanced concurrently.
if no splitting of operators, all terms advanced concurrently;
else, will advance one term at a time.
"""
function setup_advance_flags(moments, composition, split_operators, collisions, rk_coefs)
    # default is not to concurrently advance different operators
    advance_vpa_advection = false
    advance_z_advection = false
    advance_cx = false
    advance_ionization = false
    advance_sources = false
    advance_numerical_dissipation = false
    advance_continuity = false
    advance_force_balance = false
    advance_energy = false
    # all advance flags remain false if using operator-splitting
    # otherwise, check to see if the flags need to be set to true
    if !split_operators
        # default for non-split operators is to include both vpa and z advection together
        advance_vpa_advection = true
        advance_z_advection = true
        # if neutrals present, check to see if different ion-neutral
        # collisions are enabled
        if composition.n_neutral_species > 0
            # if charge exchange collision frequency non-zero,
            # account for charge exchange collisions
            if abs(collisions.charge_exchange) > 0.0
                advance_cx = true
            end
            # if ionization collision frequency non-zero,
            # account for charge exchange collisions
            if abs(collisions.ionization) > 0.0
                advance_ionization = true
            end
        elseif collisions.constant_ionization_rate && collisions.ionization > 0.0
            advance_ionization = true
        end
        advance_numerical_dissipation = true
        # if evolving the density, must advance the continuity equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_density
            advance_sources = true
            advance_continuity = true
        end
        # if evolving the parallel flow, must advance the force balance equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_upar
            advance_sources = true
            advance_force_balance = true
        end
        # if evolving the parallel pressure, must advance the energy equation,
        # in addition to including sources arising from the use of a modified distribution
        # function in the kinetic equation
        if moments.evolve_ppar
            advance_sources = true
            advance_energy = true
        end
    end
    return advance_info(advance_vpa_advection, advance_z_advection, advance_cx,
                        advance_ionization, advance_sources,
                        advance_numerical_dissipation, advance_continuity,
                        advance_force_balance, advance_energy, rk_coefs)
end

"""
if evolving the density via continuity equation, redefine the normalised f → f/n
if evolving the parallel pressure via energy equation, redefine f -> f * vth / n
'scratch' should be a (nz,nspecies) array
"""
function normalize_pdf!(pdf, moments, scratch)
    error("Function normalise_pdf() has not been updated to be parallelized. Does not "
          * "seem to be used at the moment.")
    if moments.evolve_ppar
        @. scratch = moments.vth/moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    elseif moments.evolve_density
        @. scatch = 1.0 / moments.dens
        nvpa, nz, nspecies = size(pdf)
        for is ∈ 1:nspecies, iz ∈ 1:nz, ivpa ∈ 1:nvpa
            pdf[ivpa,iz,is] *= scratch[iz, is]
        end
    end
    return nothing
end

"""
create an array of structs containing scratch arrays for the normalised pdf and low-order moments
that may be evolved separately via fluid equations
"""
function setup_scratch_arrays(moments, pdf_in, n_rk_stages)
    # create n_rk_stages+1 structs, each of which will contain one pdf,
    # one density, and one parallel flow array
    scratch = Vector{scratch_pdf{4,3}}(undef, n_rk_stages+1)
    pdf_dims = size(pdf_in)
    moment_dims = size(moments.dens)
    # populate each of the structs
    for istage ∈ 1:n_rk_stages+1
        # Allocate arrays in temporary variables so that we can identify them
        # by source line when using @debug_shared_array
        pdf_array = allocate_shared_float(pdf_dims...)
        density_array = allocate_shared_float(moment_dims...)
        upar_array = allocate_shared_float(moment_dims...)
        ppar_array = allocate_shared_float(moment_dims...)
        temp_z_s_array = allocate_shared_float(moment_dims...)
        scratch[istage] = scratch_pdf(pdf_array, density_array, upar_array,
                                      ppar_array, temp_z_s_array)
        @serial_region begin
            scratch[istage].pdf .= pdf_in
            scratch[istage].density .= moments.dens
            scratch[istage].upar .= moments.upar
            scratch[istage].ppar .= moments.ppar
        end
    end
    return scratch
end

"""
given the number of Runge Kutta stages that are requested,
returns the needed Runge Kutta coefficients;
e.g., if f is the function to be updated, then
f^{n+1}[stage+1] = rk_coef[1,stage]*f^{n} + rk_coef[2,stage]*f^{n+1}[stage] + rk_coef[3,stage]*(f^{n}+dt*G[f^{n+1}[stage]]
"""
function setup_runge_kutta_coefficients(n_rk_stages)
    rk_coefs = allocate_float(3,n_rk_stages)
    rk_coefs .= 0.0
    if n_rk_stages == 4
        rk_coefs[1,1] = 0.5
        rk_coefs[3,1] = 0.5
        rk_coefs[2,2] = 0.5
        rk_coefs[3,2] = 0.5
        rk_coefs[1,3] = 2.0/3.0
        rk_coefs[2,3] = 1.0/6.0
        rk_coefs[3,3] = 1.0/6.0
        rk_coefs[2,4] = 0.5
        rk_coefs[3,4] = 0.5
    elseif n_rk_stages == 3
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.75
        rk_coefs[3,2] = 0.25
        rk_coefs[1,3] = 1.0/3.0
        rk_coefs[3,3] = 2.0/3.0
    elseif n_rk_stages == 2
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.5
        rk_coefs[3,2] = 0.5
    else
        rk_coefs[3,1] = 1.0
    end
    return rk_coefs
end

"""
solve ∂f/∂t + v(z,t)⋅∂f/∂z + dvpa/dt ⋅ ∂f/∂vpa= 0
define approximate characteristic velocity
v₀(z)=vⁿ(z) and take time derivative along this characteristic
df/dt + δv⋅∂f/∂z = 0, with δv(z,t)=v(z,t)-v₀(z)
for prudent choice of v₀, expect δv≪v so that explicit
time integrator can be used without severe CFL condition
"""
function time_advance!(pdf, scratch, t, t_input, vpa, z, r, vpa_spectral, z_spectral, r_spectral,
    moments, fields, vpa_advect, z_advect, r_advect, vpa_SL, z_SL, r_SL, composition,
    collisions, num_diss_params, advance, scratch_dummy_sr, io, cdf)

    @debug_detect_redundant_block_synchronize begin
        # Only want to check for redundant _block_synchronize() calls during the
        # time advance loop, so activate these checks here
        debug_detect_redundant_is_active[] = true
    end

    @serial_region begin
        println("beginning time advance...", Dates.format(now(), dateformat"H:MM:SS"))
    end

    # main time advance loop
    iwrite = 2
    for i ∈ 1:t_input.nstep
        if t_input.split_operators
            # MRH NOT SUPPORTED
            time_advance_split_operators!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, i)
        else
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z, r,
                vpa_spectral, z_spectral, r_spectral, moments, fields, vpa_advect,
                z_advect, r_advect, vpa_SL, z_SL, r_SL, composition, collisions,
                num_diss_params, advance,  scratch_dummy_sr, i)
        end
        # update the time
        t += t_input.dt
        # write data to file every nwrite time steps
        if mod(i,t_input.nwrite) == 0
            @debug_detect_redundant_block_synchronize begin
                # Skip check for redundant _block_synchronize() during file I/O because
                # it only runs infrequently
                debug_detect_redundant_is_active[] = false
            end
            begin_serial_region()
            @serial_region println("finished time step ", i, "  ",
                                   Dates.format(now(), dateformat"H:MM:SS"))
            write_data_to_ascii(pdf.norm, moments, fields, vpa, z, r, t, composition.n_species, io)
            # write initial data to binary file (netcdf)
            write_data_to_binary(pdf.norm, moments, fields, t, composition.n_species, cdf, iwrite)
            # Hack to save *.pdf of current pdf
            if t_input.runtime_plots
                @serial_region begin
                    #pyplot()
                    cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
                    logdeep = cgrad(:deep, scale=:log) |> cmlog
                    f_plots = [
                        heatmap(z.grid, vpa.grid, pdf.norm[:,:,1,is],
                                xlim=(z.grid[1] - z.L / 100.0, z.grid[end] + z.L / 100.0),
                                ylim=(vpa.grid[1] - vpa.L / 100.0, vpa.grid[end] + vpa.L / 100.0),
                                xlabel="z", ylabel="vpa", c=:deep, colorbar=false)
                        for is ∈ 1:composition.n_species]
                    for (is, p) in enumerate(f_plots)
                        @views draw_v_parallel_zero!(p, z.grid, moments.upar[:,1,is],
                                                     moments.vth[:,1,is],
                                                     moments.evolve_upar,
                                                     moments.evolve_ppar)
                    end
                    logf_plots = [
                        heatmap(z.grid, vpa.grid, log.(abs.(pdf.norm[:,:,1,is])),
                                xlim=(z.grid[1] - z.L / 100.0, z.grid[end] + z.L / 100.0),
                                ylim=(vpa.grid[1] - vpa.L / 100.0, vpa.grid[end] + vpa.L / 100.0),
                                xlabel="z", ylabel="vpa", fillcolor=logdeep, colorbar=false)
                        for is ∈ 1:composition.n_species]
                    for (is, p) in enumerate(logf_plots)
                        @views draw_v_parallel_zero!(p, z.grid, moments.upar[:,1,is],
                                                     moments.vth[:,1,is],
                                                     moments.evolve_upar,
                                                     moments.evolve_ppar)
                    end
                    f0_plots = [
                        plot(vpa.grid, pdf.norm[:,1,1,is], xlabel="vpa", ylabel="f0", legend=false)
                        for is ∈ 1:composition.n_species]
                    fL_plots = [
                        plot(vpa.grid, pdf.norm[:,end,1,is], xlabel="vpa", ylabel="fL", legend=false)
                        for is ∈ 1:composition.n_species]
                    density_plots = [
                        plot(z.grid, moments.dens[:,1,is], xlabel="z", ylabel="density", legend=false)
                        for is ∈ 1:composition.n_species]
                    upar_plots = [
                        plot(z.grid, moments.upar[:,1,is], xlabel="z", ylabel="upar", legend=false)
                        for is ∈ 1:composition.n_species]
                    ppar_plots = [
                        plot(z.grid, moments.ppar[:,1,is], xlabel="z", ylabel="ppar", legend=false)
                        for is ∈ 1:composition.n_species]
                    vth_plots = [
                        plot(z.grid, moments.vth[:,1,is], xlabel="z", ylabel="vth", legend=false)
                        for is ∈ 1:composition.n_species]
                    qpar_plots = [
                        plot(z.grid, moments.qpar[:,1,is], xlabel="z", ylabel="qpar", legend=false)
                        for is ∈ 1:composition.n_species]
                    # Put all plots into subplots of a single figure
                    plot(f_plots..., logf_plots..., f0_plots..., fL_plots...,
                         density_plots..., upar_plots..., ppar_plots..., vth_plots...,
                         qpar_plots...,
                         layout=(9,composition.n_species), size=(800,3600), plot_title="$t")
                    savefig("latest_plots.png")
                end
            end
            iwrite += 1

            # Restore to same region type as in rk_update!() so that execution after a
            # write is the same as after no write
            begin_s_r_z_region()
            @debug_detect_redundant_block_synchronize begin
                # Reactivate check for redundant _block_synchronize()
                debug_detect_redundant_is_active[] = true
            end
        end
    end
    return nothing
end

"""
"""
function time_advance_split_operators!(pdf, scratch, t, t_input, vpa, z,
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
    vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)

    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    dt = t_input.dt
    n_rk_stages = t_input.n_rk_stages
    use_semi_lagrange = t_input.use_semi_lagrange
    # to ensure 2nd order accuracy in time for operator-split advance,
    # have to reverse order of operations every other time step
    flipflop = (mod(istep,2)==0)
    if flipflop
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
        advance.vpa_advection = false
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
        advance.z_advection = false
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, num_diss_params, advance,
                    istep)
                advance.cx_collisions = false
            end
            if collisions.ionization > 0.0
                advance.ionization_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, collisions, num_diss_params, advance,
                    istep)
                advance.ionization_collisions = false
            end
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.source_terms = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.continuity = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.force_balance = false
        end
        # use the energy equation to update the parallel pressure
        if moments.evolve_ppar
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.energy = false
        end
    else
        # use the energy equation to update the parallel pressure
        if moments.evolve_ppar
            advance.energy = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.energy = false
        end
        # use force balance to update the parallel flow
        if moments.evolve_upar
            advance.force_balance = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.force_balance = false
        end
        # use the continuity equation to update the density
        if moments.evolve_density
            advance.continuity = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.continuity = false
        end
        # and add the source terms associated with redefining g = pdf/density or pdf*vth/density
        # to the kinetic equation
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
            advance.source_terms = true
            time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
            advance.source_terms = false
        end
        # account for charge exchange collisions between ions and neutrals
        if composition.n_neutral_species > 0
            if collisions.ionization > 0.0
                advance.ionization = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, z, vpa,
                    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
                    z_SL, vpa_SL, composition, collisions, num_diss_params, advance,
                    istep)
                advance.ionization = false
            end
            if collisions.charge_exchange > 0.0
                advance.cx_collisions = true
                time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
                    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
                    vpa_SL, z_SL, composition, collisions, num_diss_params, advance,
                    istep)
                advance.cx_collisions = false
            end
        end
        # z_advection! advances the operator-split 1D advection equation in z
        # apply z-advection operation to all species (charged and neutral)
        advance.z_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
        advance.z_advection = false
        # advance the operator-split 1D advection equation in vpa
        # vpa-advection only applies for charged species
        advance.vpa_advection = true
        time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
            vpa_SL, z_SL, composition, collisions, num_diss_params, advance, istep)
        advance.vpa_advection = false
    end
    return nothing
end

"""
"""
function time_advance_no_splitting!(pdf, scratch, t, t_input, vpa, z, r,
    vpa_spectral, z_spectral, r_spectral, moments, fields, vpa_advect, z_advect, r_advect,
    vpa_SL, z_SL, r_SL, composition, collisions, num_diss_params, advance,
    scratch_dummy_sr, istep)

    if t_input.n_rk_stages > 1
        ssp_rk!(pdf, scratch, t, t_input, vpa, z, r,
            vpa_spectral, z_spectral, r_spectral, moments, fields, vpa_advect, z_advect, r_advect,
            vpa_SL, z_SL, r_SL, composition, collisions, num_diss_params, advance,
            scratch_dummy_sr, istep)
    else
        euler_time_advance!(scratch, scratch, pdf, fields, moments, vpa_SL, z_SL, r_SL,
            vpa_advect, z_advect, r_advect, vpa, z, r, t,
            t_input, vpa_spectral, z_spectral, r_spectral, composition,
            collisions, num_diss_params, advance, 1)
        # NB: this must be broken -- scratch is updated in euler_time_advance!,
        # but not the pdf or moments. need to add update to these quantities here. Also
        # need to apply boundary conditions, possibly other things that are taken care
        # of in rk_update!() for the ssp_rk!() method.
    end
    return nothing
end

"""
use information obtained from the Runge-Kutta stages to compute the updated pdf;
for the quantities (density, upar, ppar, vth, qpar and phi) that are derived
from the 'true', un-modified pdf, either: update them using info from Runge Kutta
stages, if the quantities are evolved separately from the modified pdf;
or update them by taking the appropriate velocity moment of the evolved pdf
"""
function rk_update!(scratch, pdf, moments, fields, vpa, z, r, vpa_advect, z_advect,
                    rk_coefs, istage, composition, num_diss_params)
    begin_s_r_z_region()
    nvpa = size(pdf.unnorm, 1)
    new_scratch = scratch[istage+1]
    old_scratch = scratch[istage]
    # use Runge Kutta to update the evolved pdf
    @loop_s_r_z_vpa is ir iz ivpa begin
        new_scratch.pdf[ivpa,iz,ir,is] = rk_coefs[1]*pdf.norm[ivpa,iz,ir,is] + rk_coefs[2]*old_scratch.pdf[ivpa,iz,ir,is] + rk_coefs[3]*new_scratch.pdf[ivpa,iz,ir,is]
    end
    # use Runge Kutta to update any velocity moments evolved separately from the pdf
    rk_update_evolved_moments!(new_scratch, old_scratch, moments, rk_coefs)

    # Ensure there are no negative values in the pdf before applying boundary
    # conditions, so that negative deviations do not mess up the integral-constraint
    # corrections in the sheath boundary conditions.
    force_minimum_pdf_value!(new_scratch.pdf, num_diss_params)

    # Enforce boundary conditions in z and vpa on the distribution function.
    # Must be done after Runge Kutta update so that the boundary condition applied to
    # the updated pdf is consistent with the updated moments - otherwise different upar
    # between 'pdf', 'old_scratch' and 'new_scratch' might mean a point that should be
    # set to zero at the sheath boundary according to the final upar has a non-zero
    # contribution from one or more of the terms.
    # NB: probably need to do the same for the evolved moments
    enforce_boundary_conditions!(new_scratch, moments, vpa.bc, z.bc, vpa, z, r,
                                 vpa_advect, z_advect, composition)

    if moments.evolve_density && moments.enforce_conservation
        begin_s_r_z_region()
        #enforce_moment_constraints!(new_scratch, scratch[1], vpa, z, r, composition, moments, scratch_dummy_sr)
        @loop_s_r_z is ir iz begin
            @views hard_force_moment_constraints!(new_scratch.pdf[:,iz,ir,is], moments, vpa)
        end
    end

    # update remaining velocity moments that are calculable from the evolved pdf
    update_derived_moments!(new_scratch, moments, vpa, z, r, composition)
    # update the thermal speed from the updated pressure and density
    @loop_s_r_z is ir iz begin
        moments.vth[iz,ir,is] = sqrt(2.0*new_scratch.ppar[iz,ir,is]/new_scratch.density[iz,ir,is])
    end
    # update the parallel heat flux
    update_qpar!(moments.qpar, moments.qpar_updated, new_scratch.density,
                 new_scratch.upar, moments.vth, new_scratch.pdf, vpa, z, r, composition,
                 moments.evolve_density, moments.evolve_upar, moments.evolve_ppar)
    # update the 'true', un-normalized pdf
    update_unnormalized_pdf!(pdf.unnorm, new_scratch, moments)
    # update the electrostatic potential phi
    update_phi!(fields, scratch[istage+1], z, r, composition)
    if !(( moments.evolve_upar || moments.evolve_ppar) &&
              istage == length(scratch)-1)
        # _block_synchronize() here because phi needs to be read on different ranks than
        # it was written on, even though the loop-type does not change here. However,
        # after the final RK stage can skip if:
        #  * evolving upar or ppar as synchronization will be triggered after moments
        #    updates at the beginning of the next RK step
        _block_synchronize()
    end
end

"""
use Runge Kutta to update any velocity moments evolved separately from the pdf
"""
function rk_update_evolved_moments!(new_scratch, old_scratch, moments, rk_coefs)
    # if separately evolving the particle density, update using RK
    if moments.evolve_density
        @loop_s_r_z is ir iz begin
            new_scratch.density[iz,ir,is] = rk_coefs[1]*moments.dens[iz,ir,is] + rk_coefs[2]*old_scratch.density[iz,ir,is] + rk_coefs[3]*new_scratch.density[iz,ir,is]
        end
    end
    # if separately evolving the parallel flow, update using RK
    if moments.evolve_upar
        @loop_s_r_z is ir iz begin
            new_scratch.upar[iz,ir,is] = rk_coefs[1]*moments.upar[iz,ir,is] + rk_coefs[2]*old_scratch.upar[iz,ir,is] + rk_coefs[3]*new_scratch.upar[iz,ir,is]
        end
    end
    # if separately evolving the parallel pressure, update using RK;
    if moments.evolve_ppar
        @loop_s_r_z is ir iz begin
            new_scratch.ppar[iz,ir,is] = rk_coefs[1]*moments.ppar[iz,ir,is] + rk_coefs[2]*old_scratch.ppar[iz,ir,is] + rk_coefs[3]*new_scratch.ppar[iz,ir,is]
        end
    end
end

"""
update velocity moments that are calculable from the evolved pdf
"""
function update_derived_moments!(new_scratch, moments, vpa, z, r, composition)
    if !moments.evolve_density
        update_density!(new_scratch.density, moments.dens_updated, new_scratch.pdf, vpa, z, r, composition)
    end
    if !moments.evolve_upar
        update_upar!(new_scratch.upar, moments.upar_updated, new_scratch.density,
                     new_scratch.ppar, new_scratch.pdf, vpa, z, r, composition,
                     moments.evolve_density, moments.evolve_ppar)
    end
    if !moments.evolve_ppar
        # update_ppar! calculates (p_parallel/m_s N_e c_s^2) + (n_s/N_e)*(upar_s/c_s)^2 = (1/√π)∫d(vpa/c_s) (vpa/c_s)^2 * (√π f_s c_s / N_e)
        update_ppar!(new_scratch.ppar, moments.ppar_updated, new_scratch.density,
                     new_scratch.upar, new_scratch.pdf, vpa, z, r, composition,
                     moments.evolve_density, moments.evolve_upar)
    end
end

"""
update the 'true', un-normalized pdf
"""
function update_unnormalized_pdf!(pdf_unnorm, new_scratch, moments)
    # if no moments are evolved separately from the pdf, then the
    # evolved pdf is the 'true', un-normalized pdf;
    # initialize to this value and modify below if necessary
    @loop_s_r_z_vpa is ir iz ivpa begin
        pdf_unnorm[ivpa,iz,ir,is] = new_scratch.pdf[ivpa,iz,ir,is]
    end
    # if separately evolving the particle density, the evolved
    # pdf is the 'true' pdf divided by the particle density
    if moments.evolve_density
        @loop_s_r_z_vpa is ir iz ivpa begin
            pdf_unnorm[ivpa,iz,ir,is] = new_scratch.pdf[ivpa,iz,ir,is] * new_scratch.density[iz,ir,is]
        end
    end
    # if separately evolving the parallel pressure, the evolved
    # pdf is the 'true' pdf multiplied by the thermal speed
    if moments.evolve_ppar
        @loop_s_r_z_vpa is ir iz ivpa begin
            pdf_unnorm[ivpa,iz,ir,is] /= moments.vth[iz,ir,is]
        end
    end
end

"""
"""
function ssp_rk!(pdf, scratch, t, t_input, vpa, z, r,
    vpa_spectral, z_spectral, r_spectral, moments, fields, vpa_advect, z_advect, r_advect,
    vpa_SL, z_SL, r_SL, composition, collisions, num_diss_params, advance,
    scratch_dummy_sr, istep)

    n_rk_stages = t_input.n_rk_stages

    first_scratch = scratch[1]
    @loop_s_r_z_vpa is ir iz ivpa begin
        first_scratch.pdf[ivpa,iz,ir,is] = pdf.norm[ivpa,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        first_scratch.density[iz,ir,is] = moments.dens[iz,ir,is]
        first_scratch.upar[iz,ir,is] = moments.upar[iz,ir,is]
        first_scratch.ppar[iz,ir,is] = moments.ppar[iz,ir,is]
    end
    if moments.evolve_upar
        # moments may be read on all ranks, even though loop type is z_s, so need to
        # synchronize here
        _block_synchronize()
    end

    for istage ∈ 1:n_rk_stages
        # do an Euler time advance, with scratch[2] containing the advanced quantities
        # and scratch[1] containing quantities at time level n
        update_solution_vector!(scratch, moments, istage, composition, vpa, z, r)
        # calculate f^{(1)} = fⁿ + Δt*G[fⁿ] = scratch[2].pdf
        euler_time_advance!(scratch[istage+1], scratch[istage],
            pdf, fields, moments, vpa_SL, z_SL, r_SL, vpa_advect, z_advect, r_advect, vpa, z, r, t,
            t_input, vpa_spectral, z_spectral, r_spectral, composition,
            collisions, num_diss_params, advance, istage)
        @views rk_update!(scratch, pdf, moments, fields, vpa, z, r, vpa_advect,
                          z_advect, advance.rk_coefs[:,istage], istage, composition,
                          num_diss_params)
    end

    istage = n_rk_stages+1
    final_scratch = scratch[istage]

    # update the pdf.norm and moments arrays as needed
    @loop_s_r_z_vpa is ir iz ivpa begin
        pdf.norm[ivpa,iz,ir,is] = final_scratch.pdf[ivpa,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        moments.dens[iz,ir,is] = final_scratch.density[iz,ir,is]
        moments.upar[iz,ir,is] = final_scratch.upar[iz,ir,is]
        moments.ppar[iz,ir,is] = final_scratch.ppar[iz,ir,is]
    end
    update_pdf_unnorm!(pdf, moments, scratch[istage].temp_z_s, composition, vpa)
    return nothing
end

"""
euler_time_advance! advances the vector equation dfvec/dt = G[f]
that includes the kinetic equation + any evolved moment equations
using the forward Euler method: fvec_out = fvec_in + dt*fvec_in,
with fvec_in an input and fvec_out the output
"""
function euler_time_advance!(fvec_out, fvec_in, pdf, fields, moments, vpa_SL, z_SL, r_SL,
    vpa_advect, z_advect, r_advect, vpa, z, r, t, t_input, vpa_spectral, z_spectral, r_spectral,
    composition, collisions, num_diss_params, advance, istage)
    # define some abbreviated variables for tidiness
    n_ion_species = composition.n_ion_species
    dt = t_input.dt
    use_semi_lagrange = t_input.use_semi_lagrange
    # vpa_advection! advances the 1D advection equation in vpa.
    # only charged species have a force accelerating them in vpa;
    # however, neutral species do have non-zero d(wpa)/dt, so there is advection in wpa

    if advance.vpa_advection
        vpa_advection!(fvec_out.pdf, fvec_in, pdf.norm, fields, moments,
            vpa_SL, vpa_advect, vpa, z, r, use_semi_lagrange, dt, t,
            vpa_spectral, z_spectral, composition, collisions, istage)
    end

    # z_advection! advances 1D advection equation in z
    # apply z-advection operation to all species (charged and neutral)

    if advance.z_advection
        z_advection!(fvec_out.pdf, fvec_in, pdf.norm, moments, z_SL, z_advect, z, vpa, r,
            use_semi_lagrange, dt, t, z_spectral, composition, istage)
    end

    if advance.source_terms
        source_terms!(fvec_out.pdf, fvec_in, moments, vpa, z, r, dt, z_spectral,
                      composition, collisions)
    end
    # account for charge exchange collisions between ions and neutrals
    if advance.cx_collisions
        charge_exchange_collisions!(fvec_out.pdf, fvec_in, moments, composition, vpa, z, r,
                                    collisions.charge_exchange, vpa_spectral, dt)
    end
    # account for ionization collisions between ions and neutrals
    if advance.ionization_collisions
        ionization_collisions!(fvec_out.pdf, fvec_in, moments, n_ion_species,
            composition.n_neutral_species, vpa, z, r, vpa_spectral, composition,
            collisions, z.n, dt)
    end
    # add numerical dissipation
    if advance.numerical_dissipation
        vpa_boundary_buffer_decay!(fvec_out.pdf, fvec_in, moments, vpa, dt,
                                   num_diss_params)
        vpa_boundary_buffer_diffusion!(fvec_out.pdf, fvec_in, vpa, vpa_spectral, dt,
                                       num_diss_params)
        vpa_dissipation!(fvec_out.pdf, fvec_in, moments, vpa, vpa_spectral, dt,
                         num_diss_params)
        z_dissipation!(fvec_out.pdf, fvec_in, moments, z, vpa, z_spectral, dt,
                       num_diss_params)
    end
    # End of advance of distribution function

    # Start advancing moments
    if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
        # Only need to change region type if moment evolution equations will be used.
        # Exept when using wall boundary conditions, do not actually need to synchronize
        # here because above we only modify the distribution function and below we only
        # modify the moments, so there is no possibility of race conditions.
        begin_s_r_region(no_synchronize=true)
    end
    if advance.continuity
        continuity_equation!(fvec_out.density, fvec_in, moments, composition, vpa, z, r,
                             dt, z_spectral, collisions.ionization, num_diss_params)
    end
    if advance.force_balance
        # fvec_out.upar is over-written in force_balance! and contains the particle flux
        force_balance!(fvec_out.upar, fvec_out.density, fvec_in, fields, collisions,
                       vpa, z, r, dt, z_spectral, composition, num_diss_params)
    end
    if advance.energy
        energy_equation!(fvec_out.ppar, fvec_in, moments, collisions, z,
                         r, dt, z_spectral, composition, num_diss_params)
    end
    # reset "xx.updated" flags to false since ff has been updated
    # and the corresponding moments have not
    reset_moments_status!(moments, composition, z)
    return nothing
end

"""
update the vector containing the pdf and any evolved moments of the pdf
for use in the Runge-Kutta time advance
"""
function update_solution_vector!(evolved, moments, istage, composition, vpa, z, r)
    new_evolved = evolved[istage+1]
    old_evolved = evolved[istage]
    @loop_s_r_z_vpa is ir iz ivpa begin
        new_evolved.pdf[ivpa,iz,ir,is] = old_evolved.pdf[ivpa,iz,ir,is]
    end
    @loop_s_r_z is ir iz begin
        new_evolved.density[iz,ir,is] = old_evolved.density[iz,ir,is]
        new_evolved.upar[iz,ir,is] = old_evolved.upar[iz,ir,is]
        new_evolved.ppar[iz,ir,is] = old_evolved.ppar[iz,ir,is]
    end
    return nothing
end

"""
if separately evolving the density via the continuity equation,
the evolved pdf has been normalised by the particle density
undo this normalisation to get the true particle distribution function

scratch should be a (nz,nspecies) array
"""
function update_pdf_unnorm!(pdf, moments, scratch, composition, vpa)
    nvpa = size(pdf.unnorm, 1)
    if moments.evolve_ppar
        @loop_s_r_z is ir iz begin
            scratch[iz,ir,is] = moments.dens[iz,ir,is]/moments.vth[iz,ir,is]
        end
        @loop_s_r_z_vpa is ir iz ivpa begin
            pdf.unnorm[ivpa,iz,ir,is] = pdf.norm[ivpa,iz,ir,is]*scratch[iz,ir,is]
        end
    elseif moments.evolve_density
        @loop_s_r_z_vpa is ir iz ivpa begin
            pdf.unnorm[ivpa,iz,ir,is] = pdf.norm[ivpa,iz,ir,is] * moments.dens[iz,ir,is]
        end
    else
        @loop_s_r_z_vpa is ir iz ivpa begin
            pdf.unnorm[ivpa,iz,ir,is] = pdf.norm[ivpa,iz,ir,is]
        end
    end
end

end
