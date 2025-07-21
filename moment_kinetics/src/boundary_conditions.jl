"""
Functions for applying boundary conditions
"""
module boundary_conditions

export create_boundary_info
export enforce_boundary_conditions!
export enforce_neutral_boundary_conditions!

using LinearAlgebra: dot
using SpecialFunctions: erfc

using ..array_allocation: allocate_shared_float
using ..calculus: integral, reconcile_element_boundaries_MPI!
using ..communication
using ..coordinates: coordinate
using ..input_structs
using ..interpolation: interpolate_to_grid_1d!
using ..looping
using ..timer_utils
using ..moment_kinetics_structs
using ..type_definitions
using ..velocity_moments: integrate_over_positive_vz, integrate_over_negative_vz

function create_boundary_info(input_dict, pdf, moments, r, z, vperp, vpa, vzeta, vr, vz,
                              r_spectral, composition, zero; warn_unexpected)
    return boundary_info(create_r_boundary_info(input_dict, pdf, moments, r, z, vperp,
                                                vpa, vzeta, vr, vz, r_spectral,
                                                composition;
                                                warn_unexpected=warn_unexpected),
                         create_z_boundary_info(r, z, vperp, vpa, vzeta, vr, vz,
                                                composition, zero))
end

function read_r_sections_input(input_dict, warn_unexpected)
    inner_sections_counter = 1
    inner_input = OptionsDict[]
    element_lower = 1
    single_section = false
    section_name = "inner_r_bc_$inner_sections_counter"
    while true
        this_input = deepcopy(set_defaults_and_check_section!(
            input_dict, section_name, warn_unexpected;
            nelement=-1,
            bc="Neumann",
            ion_bc="default",
            electron_bc="default",
            neutral_bc="default",
           ))
        this_input["section_name"] = section_name
        this_input["element_lower"] = element_lower
        if this_input["nelement"] == -1
            single_section = true
            this_input["element_upper"] = -1
        else
            this_input["element_upper"] = element_lower + this_input["nelement"] - 1
            element_lower = this_input["element_upper"] + 1
        end
        this_input["element_lower"] = element_lower
        push!(inner_input, this_input)

        inner_sections_counter += 1
        section_name = "inner_r_bc_$inner_sections_counter"
        if section_name ∉ keys(input_dict)
            break
        elseif single_section
            error("When nelement=-1 is set, there can only be a single r-boundary "
                  * "section, but found section $section_name")
        end
    end

    outer_sections_counter = 1
    outer_input = OptionsDict[]
    element_lower = 1
    single_section = false
    section_name = "outer_r_bc_$outer_sections_counter"
    while true
        this_input = deepcopy(set_defaults_and_check_section!(
            input_dict, section_name, warn_unexpected;
            nelement=-1,
            bc="Neumann",
            ion_bc="default",
            electron_bc="default",
            neutral_bc="default",
           ))
        this_input["section_name"] = section_name
        this_input["element_lower"] = element_lower
        if this_input["nelement"] == -1
            single_section = true
            this_input["element_upper"] = -1
        else
            this_input["element_upper"] = element_lower + this_input["nelement"] - 1
            element_lower = this_input["element_upper"] + 1
        end
        this_input["element_lower"] = element_lower
        push!(outer_input, this_input)

        outer_sections_counter += 1
        section_name = "outer_r_bc_$outer_sections_counter"
        if section_name ∉ keys(input_dict)
            break
        elseif single_section
            error("When nelement=-1 is set, there can only be a single r-boundary "
                  * "section, but found section $section_name")
        end
    end

    return inner_input, outer_input
end

function create_r_boundary_info(input_dict, pdf, moments, r, z, vperp, vpa, vzeta, vr, vz,
                                r_spectral, composition; warn_unexpected)
    inner_input, outer_input = read_r_sections_input(input_dict, warn_unexpected)

    if pdf !== nothing && moments !== nothing
        ion_pdf_inner = @view pdf.ion.norm[:,:,:,1,:]
        ion_density_inner = @view moments.ion.dens[:,1,:]
        ion_upar_inner = @view moments.ion.upar[:,1,:]
        ion_p_inner = @view moments.ion.p[:,1,:]
        electron_pdf_inner = pdf.electron === nothing ? nothing : @view pdf.electron.norm[:,:,:,1]
        electron_density_inner = @view moments.electron.dens[:,1]
        electron_upar_inner = @view moments.electron.upar[:,1]
        electron_p_inner = @view moments.electron.p[:,1]
        neutral_pdf_inner = @view pdf.neutral.norm[:,:,:,:,1,:]
        neutral_density_inner = @view moments.neutral.dens[:,1,:]
        neutral_uz_inner = @view moments.neutral.uz[:,1,:]
        neutral_p_inner = @view moments.neutral.p[:,1,:]
    else
        ion_pdf_inner = nothing
        ion_density_inner = nothing
        ion_upar_inner = nothing
        ion_p_inner = nothing
        electron_pdf_inner = nothing
        electron_density_inner = nothing
        electron_upar_inner = nothing
        electron_p_inner = nothing
        neutral_pdf_inner = nothing
        neutral_density_inner = nothing
        neutral_uz_inner = nothing
        neutral_p_inner = nothing
    end

    # Only inner-r shared-memory block needs to apply r-boundary conditions.
    # Note that 1D simulations cannot have r-boundaries
    if r.n > 1 && r.irank == 0
        inner_sections = collect(create_r_section(inp, ion_pdf_inner, ion_density_inner,
                                                  ion_upar_inner, ion_p_inner,
                                                  electron_pdf_inner,
                                                  electron_density_inner,
                                                  electron_upar_inner, electron_p_inner,
                                                  neutral_pdf_inner,
                                                  neutral_density_inner, neutral_uz_inner,
                                                  neutral_p_inner, r, z, vperp, vpa,
                                                  vzeta, vr, vz, r_spectral, moments,
                                                  composition, true)
                                 for inp in inner_input)
    else
        inner_sections = ()
    end

    if pdf !== nothing && moments !== nothing
        ion_pdf_outer = @view pdf.ion.norm[:,:,:,end,:]
        ion_density_outer = @view moments.ion.dens[:,end,:]
        ion_upar_outer = @view moments.ion.upar[:,end,:]
        ion_p_outer = @view moments.ion.p[:,end,:]
        electron_pdf_outer = pdf.electron === nothing ? nothing : @view pdf.electron.norm[:,:,:,end]
        electron_density_outer = @view moments.electron.dens[:,end]
        electron_upar_outer = @view moments.electron.upar[:,end]
        electron_p_outer = @view moments.electron.p[:,end]
        neutral_pdf_outer = @view pdf.neutral.norm[:,:,:,:,end,:]
        neutral_density_outer = @view moments.neutral.dens[:,end,:]
        neutral_uz_outer = @view moments.neutral.uz[:,end,:]
        neutral_p_outer = @view moments.neutral.p[:,end,:]
    else
        ion_pdf_outer = nothing
        ion_density_outer = nothing
        ion_upar_outer = nothing
        ion_p_outer = nothing
        electron_pdf_outer = nothing
        electron_density_outer = nothing
        electron_upar_outer = nothing
        electron_p_outer = nothing
        neutral_pdf_outer = nothing
        neutral_density_outer = nothing
        neutral_uz_outer = nothing
        neutral_p_outer = nothing
    end

    # Only outer-r shared-memory block needs to apply r-boundary conditions.
    # Note that 1D simulations cannot have r-boundaries
    if r.n > 1 && r.irank == r.nrank - 1
        outer_sections = collect(create_r_section(inp, ion_pdf_outer, ion_density_outer,
                                                  ion_upar_outer, ion_p_outer,
                                                  electron_pdf_outer,
                                                  electron_density_outer,
                                                  electron_upar_outer, electron_p_outer,
                                                  neutral_pdf_outer,
                                                  neutral_density_outer, neutral_uz_outer,
                                                  neutral_p_outer, r, z, vperp, vpa,
                                                  vzeta, vr, vz, r_spectral, moments,
                                                  composition, false)
                                 for inp in outer_input)
    else
        outer_sections = ()
    end

    for s_inner ∈ inner_sections
        if isa(s_inner.ion, ion_r_boundary_section_periodic)
            if r.nelement_local != r.nelement_global
                error("Periodic r-boundary-conditions are not supported when using "
                      * "distributed-MPI in the r-direction")
            end
            # Check there is a corresponding periodic section in the outer boundary.
            if !any(isa(s_outer.ion, ion_r_boundary_section_periodic)
                    && s_outer.z_range == s_inner.z_range
                    for s_outer ∈ outer_sections)
                error("Periodic boundary conditions requested for ions on inner "
                      * "r-boundary for iz=$(s_inner.z_range) but no "
                      * "corresponding section found on the outer r-boundary.")
            end
        end
        if isa(s_inner.electron, electron_r_boundary_section_periodic)
            if r.nelement_local != r.nelement_global
                error("Periodic r-boundary-conditions are not supported when using "
                      * "distributed-MPI in the r-direction")
            end
            # Check there is a corresponding periodic section in the outer boundary.
            if !any(isa(s_outer.electron, electron_r_boundary_section_periodic)
                    && s_outer.z_range == s_inner.z_range
                    for s_outer ∈ outer_sections)
                error("Periodic boundary conditions requested for electrons on inner "
                      * "r-boundary for iz=$(s_inner.z_range) but no "
                      * "corresponding section found on the outer r-boundary.")
            end
        end
        if isa(s_inner.neutral, neutral_r_boundary_section_periodic)
            if r.nelement_local != r.nelement_global
                error("Periodic r-boundary-conditions are not supported when using "
                      * "distributed-MPI in the r-direction")
            end
            # Check there is a corresponding periodic section in the outer boundary.
            if !any(isa(s_outer.neutral, neutral_r_boundary_section_periodic)
                    && s_outer.z_range == s_inner.z_range
                    for s_outer ∈ outer_sections)
                error("Periodic boundary conditions requested for neutral on inner "
                      * "r-boundary for iz=$(s_inner.z_range) but no "
                      * "corresponding section found on the outer r-boundary.")
            end
        end
    end

    r_boundaries = r_boundary_info(Tuple(s for s ∈ inner_sections if s !== nothing),
                                   Tuple(s for s ∈ outer_sections if s !== nothing))

    # Check that all boundary points are handled
    if r.irank == 0 && r.n > 1
        covered_points = vcat([b.z_range for b ∈ r_boundaries.inner_sections]...)
        sort!(covered_points)
        if covered_points != collect(1:z.n)
            error("On inner r-boundary, boundary condition not specified for all points, "
                  * "or multiply specified for some points. Points specified are "
                  * "$covered_points.")
        end
    end
    if r.irank == r.nrank - 1 && r.n > 1
        covered_points = vcat([b.z_range for b ∈ r_boundaries.outer_sections]...)
        sort!(covered_points)
        if covered_points != collect(1:z.n)
            error("On outer r-boundary, boundary condition not specified for all points, "
                  * "or multiply specified for some points. Points specified are "
                  * "$covered_points.")
        end
    end

    return r_boundaries
end

function create_r_section(this_input, ion_pdf, ion_density, ion_upar, ion_p, electron_pdf,
                          electron_density, electron_upar, electron_p, neutral_pdf,
                          neutral_density, neutral_uz, neutral_p, r, z, vperp, vpa, vzeta,
                          vr, vz, r_spectral, moments, composition, is_inner)
    element_lower = this_input["element_lower"]
    if this_input["element_upper"] < 0
        element_upper = z.nelement_global
    else
        element_upper = this_input["element_upper"]
    end

    # element_lower and element_upper are global element indices. If we are using
    # distributed-MPI, we need to convert them to local element indices.
    offset = z.irank * z.nelement_local
    element_lower -= offset
    element_upper -= offset

    if element_upper < 0 || element_lower > z.nelement_local + 1
        # This section falls entirely outside this shared-memory block.
        return nothing
    end
    has_element_lower_boundary = true
    if element_lower ≤ 0
        # This section is only partially contained in this shared-memory block.
        element_lower = 1
        has_element_lower_boundary = false
    end
    has_element_upper_boundary = true
    if element_upper > z.nelement_local
        # This section is only partially contained in this shared-memory block.
        element_upper = z.nelement_local
        has_element_upper_boundary = false
    end

    # At the meeting point of two sections, apply the boundary condition from whichever
    # section is closer to z=0. This is an arbitrary choice, made to allow up-down
    # symmetric simulations.
    if (z.irank == 0 && element_lower == 1)
        # First section, always include lower point
        first_index = 1
    elseif element_upper == 0
        # Only upper edge of section meets this shared-memory block
        if z.grid[1] > 0
            first_index = 1
        else
            # Don't need to include this section in things imposed on this shared-memory
            # block.
            return nothing
        end
    else
        first_index = (element_lower - 1) * (z.ngrid - 1) + 1
        if z.grid[first_index] ≥ 0.0 && has_element_lower_boundary
            # First point is included in previous section.
            first_index += 1
        end
    end
    if (z.irank == z.nrank - 1 && element_upper == z.nelement_local)
        # Last section, always include upper point
        last_index = z.n
    elseif element_lower == z.nelement_local + 1
        # Only lower edge of section meets this shared-memory block
        if z.grid[end] < 0
            last_index = z.n
        else
            # Don't need to include this section in things imposed on this shared-memory
            # block.
            return nothing
        end
    else
        last_index = element_upper * (z.ngrid - 1) + 1
        if z.grid[last_index] < 0.0 && has_element_upper_boundary
            # Last point is included in next section.
            last_index -= 1
        end
    end

    z_range = first_index:last_index

    function get_section(species)
        if this_input["$(species)_bc"] == "default"
            this_input["$(species)_bc"] = this_input["bc"]
        end
        if this_input["$(species)_bc"] == "periodic"
            # No boundary condition to impose, periodicity is imposed by communication
            # when calculating derivatives.
            if species == "ion"
                this_section = ion_r_boundary_section_periodic()
            elseif species == "electron"
                this_section = electron_r_boundary_section_periodic()
            elseif species == "neutral"
                this_section = neutral_r_boundary_section_periodic()
            else
                error("Unrecognised species=$species")
            end
        elseif startswith(this_input["$(species)_bc"], "Neumann")
            boundary_value_string = split(this_input["$(species)_bc"], "Neumann")[2]
            if boundary_value_string == ""
                boundary_value = 0.0
            else
                boundary_value = parse(mk_float, boundary_value_string)
            end

            # Logarithmic derivative at boundary is equal to 'boundary value'. If `x` is
            # the vector of ngrid values from the first/last element, D is the
            # derivative matrix row corresponding to the inner/outer point, xb = x[1], Db
            # = D[1] for the inner boundary or xb = x[end], Db = D[end] for the outer
            # boundary, and xo, Do are the other x, D entries then
            #   D.x / xb = boundary_value
            #   ⇒ Do.xo = (boundary_value - Db) * xb
            #        xb = Do.xo / (boundary_value - Db)
            if :lobatto ∉ fieldnames(typeof(r_spectral)) || :Dmat ∉ fieldnames(typeof(r_spectral.lobatto))
                error("$(typeof(r_spectral)) discretization does not support Neumann bc.")
            end
            if is_inner
                Dmat_row = @view r_spectral.lobatto.Dmat[1,:]
                derivative_coefficients = Dmat_row[2:end]
                one_over_boundary_value_minus_Db = 1.0 / (boundary_value - Dmat_row[1])
            else
                Dmat_row = @view r_spectral.lobatto.Dmat[end,:]
                derivative_coefficients = Dmat_row[1:end-1]
                one_over_boundary_value_minus_Db = 1.0 / (boundary_value - Dmat_row[end])
            end

            if species == "ion"
                this_section = ion_r_boundary_section_Neumann(is_inner,
                                                              one_over_boundary_value_minus_Db,
                                                              derivative_coefficients)
            elseif species == "electron"
                this_section = electron_r_boundary_section_Neumann(is_inner,
                                                                   one_over_boundary_value_minus_Db,
                                                                   derivative_coefficients)
            elseif species == "neutral"
                this_section = neutral_r_boundary_section_Neumann(is_inner,
                                                                  one_over_boundary_value_minus_Db,
                                                                  derivative_coefficients)
            else
                error("Unrecognised species=$species")
            end
        elseif startswith(this_input["$(species)_bc"], "Dirichlet")
            bc_options = split(this_input["$(species)_bc"], "Dirichlet"; limit=2)[2]
            bc_options_list = split(bc_options, "_")
            n_val = 1.0
            u_val = 0.0
            T_val = 1.0

            function get_Dirichlet_val(optionstring)
                try
                    val = parse(mk_float, optionstring[2:end])
                catch e
                    if isa(e, ArgumentError)
                        error("Invalid Dirichlet option '$optionstring' found in "
                              * "$(species)_bc in [$(this_input["section_name"])].")
                    else
                        rethrow()
                    end
                end
            end

            for o ∈ bc_options_list
                if startswith(o, "n")
                    n_val = get_Dirichlet_val(o)
                elseif startswith(o, "u")
                    u_val = get_Dirichlet_val(o)
                elseif startswith(o, "T")
                    T_val = get_Dirichlet_val(o)
                end
            end
            vth = sqrt(2.0 * T_val)
            vperp_grid = vperp.grid
            vpa_grid = vpa.grid
            if vperp.n == 1
                Maxwellian_prefactor = 1.0 / sqrt(π)
                vth_power = 0.5
                fudge_factor_1V = sqrt(3.0)
            else
                Maxwellian_prefactor = 1.0 / π^1.5
                vth_power = 1.5
                fudge_factor_1V = 1.0
            end

            if species == "ion"
                bc_ion_pdf = allocate_shared_float(vpa.n, vperp.n, length(z_range),
                                                   composition.n_ion_species)
                bc_ion_density = allocate_shared_float(length(z_range), composition.n_ion_species)
                bc_ion_upar = allocate_shared_float(length(z_range), composition.n_ion_species)
                bc_ion_p = allocate_shared_float(length(z_range), composition.n_ion_species)

                @begin_s_region()
                @loop_s is begin
                    bc_ion_density[z_range,is] = n_val
                    bc_ion_upar[z_range,is] = u_val
                    bc_ion_p[z_range,is] = n_val * T_val
                    if moments.evolve_p
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_ion_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor / fudge_factor_1V *
                                exp(-(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2) / fudge_factor_1V^2)
                        end
                    elseif moments.evolve_upar
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_ion_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor / vth^vth_power *
                                exp(-(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    elseif moments.evolve_density
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_ion_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor / vth^vth_power *
                                exp(-((vpa_grid[ivpa] - u_val)^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    else
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_ion_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor * n_val / vth^vth_power *
                                exp(-((vpa_grid[ivpa] - u_val)^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    end
                end

                this_section = ion_r_boundary_section_Dirichlet(bc_ion_pdf,
                                                                bc_ion_density,
                                                                bc_ion_upar, bc_ion_p)
            elseif species == "electron"
                vth /= sqrt(composition.me_over_mi)

                bc_electron_pdf = allocate_shared_float(vpa.n, vperp.n, length(z_range))
                bc_electron_density = allocate_shared_float(length(z_range))
                bc_electron_upar = allocate_shared_float(length(z_range))
                bc_electron_p = allocate_shared_float(length(z_range))

                @begin_serial_region()
                @serial_region begin
                    bc_electron_density[z_range,is] = n_val
                    bc_electron_upar[z_range,is] = u_val
                    bc_electron_p[z_range,is] = n_val * T_val
                    if moments.evolve_p
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_electron_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor / fudge_factor_1V *
                                exp(-(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2) / fudge_factor_1V^2)
                        end
                    elseif moments.evolve_upar
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_electron_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor / vth^vth_power *
                                exp(-(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    elseif moments.evolve_density
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_electron_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor / vth^vth_power *
                                exp(-((vpa_grid[ivpa] - u_val)^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    else
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_electron_pdf[ivperp,ivpa,z_range,is] =
                                Maxwellian_prefactor * n_val / vth^vth_power *
                                exp(-((vpa_grid[ivpa] - u_val)^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    end
                end

                this_section = electron_r_boundary_section_Dirichlet(bc_electron_pdf,
                                                                     bc_electron_density,
                                                                     bc_electron_upar,
                                                                     bc_electron_p)
            elseif species == "neutral"
                bc_neutral_pdf = allocate_shared_float(vz.n, vr.n, vzeta.n,
                                                       length(z_range),
                                                       composition.n_neutral_species)
                bc_neutral_density = allocate_shared_float(length(z_range),
                                                           composition.n_neutral_species)
                bc_neutral_upar = allocate_shared_float(length(z_range),
                                                        composition.n_neutral_species)
                bc_neutral_p = allocate_shared_float(length(z_range),
                                                     composition.n_neutral_species)

                @begin_sn_region()
                @loop_sn isn begin
                    bc_neutral_density[z_range,isn] = n_val
                    bc_neutral_upar[z_range,isn] = u_val
                    bc_neutral_p[z_range,isn] = n_val * T_val
                    if moments.evolve_p
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_neutral_pdf[ivperp,ivpa,z_range,isn] =
                                Maxwellian_prefactor / fudge_factor_1V *
                                exp(-(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2) / fudge_factor_1V^2)
                        end
                    elseif moments.evolve_upar
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_neutral_pdf[ivperp,ivpa,z_range,isn] =
                                Maxwellian_prefactor / vth^vth_power *
                                exp(-(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    elseif moments.evolve_density
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_neutral_pdf[ivperp,ivpa,z_range,isn] =
                                Maxwellian_prefactor / vth^vth_power *
                                exp(-((vpa_grid[ivpa] - u_val)^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    else
                        @loop_vperp_vpa ivperp ivpa begin
                            bc_neutral_pdf[ivperp,ivpa,z_range,isn] =
                                Maxwellian_prefactor * n_val / vth^vth_power *
                                exp(-((vpa_grid[ivpa] - u_val)^2 + vperp_grid[ivperp]^2) / vth^2)
                        end
                    end
                end

                this_section = neutral_r_boundary_section_Dirichlet(bc_neutral_pdf,
                                                                    bc_neutral_density,
                                                                    bc_neutral_upar,
                                                                    bc_neutral_p)
            end
        elseif this_input["$(species)_bc"] == "pin_initial"
            if species == "ion"
                if ion_pdf === nothing
                    bc_ion_pdf = allocate_shared_float(vpa.n, vperp.n, length(z_range),
                                                       composition.n_ion_species)
                    bc_ion_density = allocate_shared_float(length(z_range), composition.n_ion_species)
                    bc_ion_upar = allocate_shared_float(length(z_range), composition.n_ion_species)
                    bc_ion_p = allocate_shared_float(length(z_range), composition.n_ion_species)

                    @begin_s_region()
                    @loop_s is begin
                        @views bc_ion_pdf[:,:,z_range,is] = ion_pdf[:,:,z_range,is]
                        @views bc_ion_density[z_range,is] = ion_density[z_range,is]
                        @views bc_ion_upar[z_range,is] = ion_upar[z_range,is]
                        @views bc_ion_p[z_range,is] = ion_p[z_range,is]
                    end

                    this_section = ion_r_boundary_section_Dirichlet(bc_ion_pdf,
                                                                    bc_ion_density,
                                                                    bc_ion_upar, bc_ion_p)
                else
                    this_section = ion_r_boundary_section_Dirichlet(nothing, nothing,
                                                                    nothing, nothing)
                end
            elseif species == "electron"
                if electron_pdf === nothing
                    bc_electron_pdf = zeros(0, 0, 0)
                else
                    bc_electron_pdf = allocate_shared_float(vpa.n, vperp.n, length(z_range))
                end
                if electron_pdf === nothing && electron_density === nothing
                    bc_electron_density = allocate_shared_float(length(z_range))
                    bc_electron_upar = allocate_shared_float(length(z_range))
                    bc_electron_p = allocate_shared_float(length(z_range))

                    @begin_serial_region()
                    @serial_region begin
                        if electron_pdf !== nothing
                            @views bc_electron_pdf[:,:,z_range] = electron_pdf[:,:,z_range]
                        end
                        @views bc_electron_density[z_range] = electron_density[z_range]
                        @views bc_electron_upar[z_range] = electron_upar[z_range]
                        @views bc_electron_p[z_range] = electron_p[z_range]
                    end
                else
                    bc_electron_density = nothing
                    bc_electron_upar = nothing
                    bc_electron_p = nothing
                end

                this_section = electron_r_boundary_section_pin_initial(bc_electron_pdf,
                                                                       bc_electron_density,
                                                                       bc_electron_upar,
                                                                       bc_electron_p)
            elseif species == "neutral"
                if neutral_pdf === nothing
                    bc_neutral_pdf = allocate_shared_float(vz.n, vr.n, vzeta.n,
                                                           length(z_range),
                                                           composition.n_neutral_species)
                    bc_neutral_density = allocate_shared_float(length(z_range),
                                                               composition.n_neutral_species)
                    bc_neutral_upar = allocate_shared_float(length(z_range),
                                                            composition.n_neutral_species)
                    bc_neutral_p = allocate_shared_float(length(z_range),
                                                         composition.n_neutral_species)

                    @begin_sn_region()
                    @loop_sn isn begin
                        @views bc_neutral_pdf[:,:,z_range,isn] = neutral_pdf[:,:,z_range,isn]
                        @views bc_neutral_density[z_range,isn] = neutral_density[z_range,isn]
                        @views bc_neutral_uz[z_range,isn] = neutral_uz[z_range,isn]
                        @views bc_neutral_p[z_range,isn] = neutral_p[z_range,isn]
                    end

                    this_section = neutral_r_boundary_section_pin_initial(bc_neutral_pdf,
                                                                          bc_neutral_density,
                                                                          bc_neutral_uz,
                                                                          bc_neutral_p)
                else
                    this_section = neutral_r_boundary_section_pin_initial(nothing,
                                                                          nothing,
                                                                          nothing,
                                                                          nothing)
                end
            else
                error("Unrecognised species=$species")
            end
        else
            error("Unrecognised option bc=$bc in section $(this_this_input["section_name"])")
        end

        return this_section
    end

    return r_boundary_section(z_range, get_section("ion"), get_section("electron"),
                              get_section("neutral"))
end

function create_z_boundary_info(r, z, vperp, vpa, vzeta, vr, vz, composition, zero)
    knudsen_cosine = init_knudsen_cosine(vzeta, vr, vz, composition, zero)
    return z_boundary_info(knudsen_cosine)
end

function init_knudsen_cosine(vzeta, vr, vz, composition, zero)

    knudsen_cosine = allocate_shared_float(vz.n, vr.n, vzeta.n)

    @begin_serial_region()
    @serial_region begin
        integrand = zeros(mk_float, vz.n, vr.n, vzeta.n)

        T_wall_over_m = composition.T_wall / composition.mn_over_mi

        if vzeta.n > 1 && vr.n > 1
            # 3V specification of neutral wall emission distribution for boundary condition
            if composition.use_test_neutral_wall_pdf
                # use test distribution that is easy for integration scheme to handle
                for ivzeta in 1:vzeta.n
                    for ivr in 1:vr.n
                        for ivz in 1:vz.n
                            v_transverse = sqrt(vzeta.grid[ivzeta]^2 + vr.grid[ivr]^2)
                            v_normal = abs(vz.grid[ivz])
                            knudsen_cosine[ivz,ivr,ivzeta] = (1.0/π/T_wall_over_m^2.5)*v_normal*exp( - 0.5 * (v_normal^2 + v_transverse^2) / T_wall_over_m)
                            integrand[ivz,ivr,ivzeta] = vz.grid[ivz]*knudsen_cosine[ivz,ivr,ivzeta]
                        end
                    end
                end
            else # get the true Knudsen cosine distribution for neutral particle wall emission
                for ivzeta in 1:vzeta.n
                    for ivr in 1:vr.n
                        for ivz in 1:vz.n
                            v_transverse = sqrt(vzeta.grid[ivzeta]^2 + vr.grid[ivr]^2)
                            v_normal = abs(vz.grid[ivz])
                            v_tot = sqrt(v_normal^2 + v_transverse^2)
                            if v_tot > zero
                                prefac = v_normal/v_tot
                            else
                                prefac = 0.0
                            end
                            knudsen_cosine[ivz,ivr,ivzeta] = (0.75/π/T_wall_over_m^2)*prefac*exp( - 0.5 * (v_normal^2 + v_transverse^2) / T_wall_over_m )
                            integrand[ivz,ivr,ivzeta] = vz.grid[ivz]*knudsen_cosine[ivz,ivr,ivzeta]
                        end
                    end
                end
            end
            normalisation = integrate_over_positive_vz(integrand, vz.grid, vz.wgts,
                                                       vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            # uncomment this line to test:
            #println("normalisation should be 1, it is = ", normalisation)
            #correct knudsen_cosine to conserve particle fluxes numerically
            @. knudsen_cosine /= normalisation

        elseif vzeta.n == 1 && vr.n == 1
            # get the marginalised Knudsen cosine distribution after integrating over vperp
            # appropriate for 1V model

            # Knudsen cosine distribution does not have separate T_∥ and T_⟂, so is
            # marginalised rather than setting T_⟂=0, therefore no need to convert to a
            # thermal speed defined with the parallel temperature in 1V case.

            @. vz.scratch = 3.0 * sqrt(π) * (0.5 / T_wall_over_m)^1.5 * abs(vz.grid) * erfc(sqrt(0.5 / T_wall_over_m) * abs(vz.grid))
            normalisation = integrate_over_positive_vz(vz.grid .* vz.scratch, vz.grid, vz.wgts, vz.scratch2,
                                                       vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            # uncomment this line to test:
            #println("normalisation should be 1, it is = ", normalisation)
            #correct knudsen_cosine to conserve particle fluxes numerically
            @. vz.scratch /= normalisation
            @. knudsen_cosine[:,1,1] = vz.scratch[:]

        end
    end
    return knudsen_cosine
end

"""
enforce boundary conditions in vpa and z on the evolved pdf;
also enforce boundary conditions in z on all separately evolved velocity space moments of the pdf
"""
@timeit global_timer enforce_boundary_conditions!(
                         f, density, upar, p, fields, boundaries::boundary_info, moments,
                         vpa, vperp, z, r, vpa_spectral, vperp_spectral, vpa_adv,
                         vperp_adv, z_adv, r_adv, composition, geometry, scratch_dummy,
                         r_diffusion, vpa_diffusion, vperp_diffusion) = begin

    if vpa.n > 1
        @begin_s_r_z_vperp_region()
        @loop_s_r_z_vperp is ir iz ivperp begin
            # enforce the vpa BC
            # use that adv.speed independent of vpa
            @views enforce_v_boundary_condition_local!(f[:,ivperp,iz,ir,is], vpa.bc,
                             vpa_adv[is].speed[:,ivperp,iz,ir], vpa_diffusion,
                             vpa, vpa_spectral)
        end
    end
    if vperp.n > 1
        @begin_s_r_z_vpa_region()
        enforce_vperp_boundary_condition!(f, vperp.bc, vperp, vperp_spectral,
                             vperp_adv, vperp_diffusion)
    end
    if z.n > 1
        @begin_s_r_vperp_vpa_region()
        # enforce the z BC on the evolved velocity space moments of the pdf
        enforce_z_boundary_condition_moments!(density, moments, z.bc)
        enforce_z_boundary_condition!(f, density, upar, p, fields, moments, z.bc, z_adv,
                                      z, vperp, vpa, composition, geometry,
                                      scratch_dummy.buffer_vpavperprs_1,
                                      scratch_dummy.buffer_vpavperprs_2,
                                      scratch_dummy.buffer_vpavperprs_3,
                                      scratch_dummy.buffer_vpavperprs_4,
                                      scratch_dummy.dummy_vpavperp)

    end
    if r.n > 1
        moment_r_speed = fields.vEr # For now, allow only ExB drift of ion moments
        enforce_r_boundary_condition!(f, density, upar, p, moment_r_speed, boundaries.r,
                                      r_adv, vpa, vperp, z, r, composition,
                                      scratch_dummy.buffer_vpavperpzs_1,
                                      scratch_dummy.buffer_vpavperpzs_2,
                                      scratch_dummy.buffer_vpavperpzs_3,
                                      scratch_dummy.buffer_vpavperpzs_4, r_diffusion)
    end
end
function enforce_boundary_conditions!(fvec_out::scratch_pdf, moments,
                                      fields::em_fields_struct, boundaries::boundary_info,
                                      vpa, vperp, z, r, vpa_spectral, vperp_spectral,
                                      vpa_adv, vperp_adv, z_adv, r_adv, composition,
                                      geometry, scratch_dummy, r_diffusion, vpa_diffusion,
                                      vperp_diffusion)
    enforce_boundary_conditions!(fvec_out.pdf, fvec_out.density, fvec_out.upar,
                                 fvec_out.p, fields, boundaries, moments, vpa, vperp, z,
                                 r, vpa_spectral, vperp_spectral, vpa_adv, vperp_adv,
                                 z_adv, r_adv, composition, geometry, scratch_dummy,
                                 r_diffusion, vpa_diffusion, vperp_diffusion)
end

"""
enforce boundary conditions on ions in r
"""
function enforce_r_boundary_condition!(f::AbstractArray{mk_float,ndim_pdf_ion},
        density::AbstractArray{mk_float,ndim_moment},
        upar::AbstractArray{mk_float,ndim_moment}, p::AbstractArray{mk_float,ndim_moment},
        moment_r_speed::AbstractArray{mk_float,ndim_field},
        r_boundaries::r_boundary_info, adv, vpa, vperp, z, r, composition,
        end1::AbstractArray{mk_float,ndim_pdf_ion_boundary},
        end2::AbstractArray{mk_float,ndim_pdf_ion_boundary},
        buffer1::AbstractArray{mk_float,ndim_pdf_ion_boundary},
        buffer2::AbstractArray{mk_float,ndim_pdf_ion_boundary}, r_diffusion::Bool)

    nr = r.n
    zero = 1.0e-10

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @begin_s_z_vperp_vpa_region()
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            end1[ivpa,ivperp,iz,is] = f[ivpa,ivperp,iz,1,is]
            end2[ivpa,ivperp,iz,is] = f[ivpa,ivperp,iz,nr,is]
        end
        reconcile_element_boundaries_MPI!(f, end1, end2, buffer1, buffer2, r)

        # Hacky way to get buffers to use for moment communication, to avoid having to
        # pass in even more buffers. We know that the distribution function buffers are
        # bigger than those needed for moments, so we can just take a chunk at the
        # beginning of the distribution function buffers.
        moment_end1 = reshape(@view(end1[1:z.n*composition.n_ion_species]), z.n,
                              composition.n_ion_species)
        moment_end2 = reshape(@view(end2[1:z.n*composition.n_ion_species]), z.n,
                              composition.n_ion_species)
        moment_buffer1 = reshape(@view(buffer1[1:z.n*composition.n_ion_species]), z.n,
                                 composition.n_ion_species)
        moment_buffer2 = reshape(@view(buffer2[1:z.n*composition.n_ion_species]), z.n,
                                 composition.n_ion_species)
        @begin_s_z_region()
        @loop_s_z is iz begin
            moment_end1[iz,is] = density[iz,1,is]
            moment_end2[iz,is] = density[iz,nr,is]
        end
        reconcile_element_boundaries_MPI!(density, end1, end2, buffer1, buffer2, r)
        @loop_s_z is iz begin
            moment_end1[iz,is] = upar[iz,1,is]
            moment_end2[iz,is] = upar[iz,nr,is]
        end
        reconcile_element_boundaries_MPI!(upar, end1, end2, buffer1, buffer2, r)
        @loop_s_z is iz begin
            moment_end1[iz,is] = p[iz,1,is]
            moment_end2[iz,is] = p[iz,nr,is]
        end
        reconcile_element_boundaries_MPI!(p, end1, end2, buffer1, buffer2, r)
    end

    for (ir, sections_tuple) ∈ ((1, r_boundaries.inner_sections),
                                (nr, r_boundaries.outer_sections))
        for section ∈ sections_tuple
            ion_section = section.ion
            # 'periodic' BC enforces periodicity by taking the average of the boundary points
            # enforce the condition if r is local
            if isa(ion_section, ion_r_boundary_section_periodic)
                if r.nelement_global != r.nelement_local
                    error("Periodic r-boundaries not yet implemented when r-dimension "
                          * " is distributed")
                end
                if ir > 1
                    # Both r-boundaries for periodic sections handled by inner boundary.
                    # We have checked that there is a corresponding periodic
                    # outer-boundary section defined.
                    continue
                end
                @begin_s_vperp_vpa_region()
                @loop_s is begin
                    for iz ∈ section.z_range
                        @loop_vperp_vpa ivperp ivpa begin
                            f[ivpa,ivperp,iz,1,is] = 0.5 * (f[ivpa,ivperp,iz,nr,is] +
                                                            f[ivpa,ivperp,iz,1,is])
                            f[ivpa,ivperp,iz,nr,is] = f[ivpa,ivperp,iz,1,is]
                        end
                    end
                end
                @begin_s_region()
                @loop_s is begin
                    for iz ∈ section.z_range
                        density[iz,1,is] = 0.5 * (density[iz,nr,is] + density[iz,1,is])
                        density[iz,nr,is] = density[iz,1,is]
                        upar[iz,1,is] = 0.5 * (upar[iz,nr,is] + upar[iz,1,is])
                        upar[iz,nr,is] = upar[iz,1,is]
                        p[iz,1,is] = 0.5 * (p[iz,nr,is] + p[iz,1,is])
                        p[iz,nr,is] = p[iz,1,is]
                    end
                end
            elseif isa(ion_section, ion_r_boundary_section_Dirichlet)
                # use the old distribution to force the new distribution to have
                # consistant-in-time values at the boundary
                # with bc = "Dirichlet" and r_diffusion = false
                # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
                # with bc = "Dirichlet" and r_diffusion = true
                # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
                f_boundary = ion_section.pdf
                @begin_s_vperp_vpa_region()
                @loop_s is begin
                    adv_speed = adv[is].speed
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        @loop_vperp_vpa ivperp ivpa begin
                            if r_diffusion || (ir == 1 ? adv[is].speed[ir,ivpa,ivperp,iz] > zero
                                                       : adv[is].speed[ir,ivpa,ivperp,iz] < -zero)
                                f[ivpa,ivperp,iz,ir,is] = f_boundary[ivpa,ivperp,iz_section,is]
                            end
                        end
                    end
                end
                n_boundary = ion_section.density
                u_boundary = ion_section.upar
                p_boundary = ion_section.p
                @begin_s_region()
                @loop_s is begin
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        if r_diffusion || (ir == 1 ? moment_r_speed[iz,ir] > zero
                                                   : moment_r_speed[iz,ir] < -zero)
                            density[iz,ir,is] = n_boundary[iz_section,is]
                            upar[iz,ir,is] = u_boundary[iz_section,is]
                            p[iz,ir,is] = p[iz_section,is]
                        end
                    end
                end
            elseif isa(ion_section, ion_r_boundary_section_Neumann)
                # with bc = "Neumann" and r_diffusion = false
                # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
                # with bc = "Neumann" and r_diffusion = true
                # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
                one_over_logarithmic_gradient_value_minus_Db =
                    ion_section.one_over_logarithmic_gradient_value_minus_Db
                derivative_coefficients = ion_section.derivative_coefficients
                if ion_section.is_inner
                    other_elements_range = 2:r.ngrid
                else
                    other_elements_range = r.n-r.ngrid+1:r.n-1
                end
                @begin_s_vperp_vpa_region()
                @loop_s is begin
                    adv_speed = adv[is].speed
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        @loop_vperp_vpa ivperp ivpa begin
                            if r_diffusion || (ir == 1 ? adv[is].speed[ir,ivpa,ivperp,iz] > zero
                                                       : adv[is].speed[ir,ivpa,ivperp,iz] < -zero)
                                @views f[ivpa,ivperp,iz,ir,is] =
                                    dot(derivative_coefficients,
                                        f[ivpa,ivperp,iz,other_elements_range,is]) *
                                    one_over_logarithmic_gradient_value_minus_Db
                            end
                        end
                    end
                end
                @begin_s_region()
                @loop_s is begin
                    for iz ∈ section.z_range
                        if r_diffusion || (ir == 1 ? moment_r_speed[iz,ir] > zero
                                                   : moment_r_speed[iz,ir] < -zero)
                            @views density[iz,ir,is] =
                                dot(derivative_coefficients,
                                    density[iz,other_elements_range,is]) *
                                one_over_logarithmic_gradient_value_minus_Db
                            @views upar[iz,ir,is] =
                                dot(derivative_coefficients,
                                    upar[iz,other_elements_range,is]) *
                                one_over_logarithmic_gradient_value_minus_Db
                            @views p[iz,ir,is] =
                                dot(derivative_coefficients,
                                    p[iz,other_elements_range,is]) *
                                one_over_logarithmic_gradient_value_minus_Db
                        end
                    end
                end
            else
                error("Unsupported section type $(typeof(ion_section)).")
            end
        end
    end
end

"""
enforce boundary conditions on ion particle f in z

`vpavperp_buffer` should be an unshared array, as it is used inside a
shared-memory-parallelised loop.
"""
function enforce_z_boundary_condition!(pdf, density, upar, p, fields, moments, bc::String,
                                       adv, z, vperp, vpa, composition, geometry,
                                       end1::AbstractArray{mk_float,4},
                                       end2::AbstractArray{mk_float,4},
                                       buffer1::AbstractArray{mk_float,4},
                                       buffer2::AbstractArray{mk_float,4},
                                       vpavperp_buffer::AbstractArray{mk_float,2})
    # this block ensures periodic BC can be supported with distributed memory MPI
    if z.nelement_global > z.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        nz = z.n
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            end1[ivpa,ivperp,ir,is] = pdf[ivpa,ivperp,1,ir,is]
            end2[ivpa,ivperp,ir,is] = pdf[ivpa,ivperp,nz,ir,is]
        end
        # check on periodic bc happens inside this call below
        reconcile_element_boundaries_MPI!(pdf, end1, end2, buffer1, buffer2, z)
    end
    # define a zero that accounts for finite precision
    zero = 1.0e-14
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if bc == "constant"
        @begin_s_r_vperp_vpa_region()
        density_offset = 1.0
        vwidth = 1.0
        if vperp.n == 1
            constant_prefactor = density_offset / sqrt(π)
        else
            constant_prefactor = density_offset / π^1.5
        end
        if z.irank == 0
            @loop_s is begin
                speed = adv[is].speed
                @loop_r ir begin
                    prefactor = constant_prefactor
                    if moments.evolve_density
                        prefactor /= density[1,ir,is]
                    end
                    if moments.evolve_p
                        prefactor *= moments.ion.vth[1,ir,is]
                    end
                    @loop_vperp_vpa ivperp ivpa begin
                        if speed[1,ivpa,ivperp,ir] > 0.0
                            pdf[ivpa,ivperp,1,ir,is] = prefactor * exp(-(speed[1,ivpa,ivperp,ir]^2 + vperp.grid[ivperp]^2)/vwidth^2)
                        end
                    end
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_s is begin
                speed = adv[is].speed
                @loop_r ir begin
                    prefactor = constant_prefactor
                    if moments.evolve_density
                        prefactor /= density[end,ir,is]
                    end
                    if moments.evolve_p
                        prefactor *= moments.ion.vth[end,ir,is]
                    end
                    @loop_vperp_vpa ivperp ivpa begin
                        if speed[end,ivpa,ivperp,ir] > 0.0
                            pdf[ivpa,ivperp,end,ir,is] = prefactor * exp(-(speed[end,ivpa,ivperp,ir]^2 + vperp.grid[ivperp]^2)/vwidth^2)
                        end
                    end
                end
            end
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif bc == "periodic" && z.nelement_global == z.nelement_local
        @begin_s_r_vperp_vpa_region()
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            pdf[ivpa,ivperp,1,ir,is] = 0.5*(pdf[ivpa,ivperp,z.n,ir,is]+pdf[ivpa,ivperp,1,ir,is])
            pdf[ivpa,ivperp,z.n,ir,is] = pdf[ivpa,ivperp,1,ir,is]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif bc == "wall"
        # Need integrals over vpa at wall boundaries in z, so cannot parallelize over z
        # or vpa.
        @begin_s_r_region()
        @loop_s is begin
            # zero incoming BC for ions, as they recombine at the wall
            if moments.evolve_upar
                @loop_r ir begin
                    @views enforce_zero_incoming_bc!(
                        pdf[:,:,:,ir,is], z, vperp, vpa, density[:,ir,is], upar[:,ir,is],
                        p[:,ir,is], fields.vEz[:,ir,is], geometry.bzed[:,ir],
                        moments.evolve_upar, moments.evolve_p, zero, fields.phi[:,ir],
                        vpavperp_buffer)
                end
            else
                @loop_r ir begin
                    @views enforce_zero_incoming_bc!(pdf[:,:,:,ir,is],
                                                     adv[is].speed[:,:,:,ir], z, zero,
                                                     fields.phi[:,ir],
                                                     z.boundary_parameters.epsz)
                end
            end
        end
    end
end

"""
enforce boundary conditions on electrons in r
"""
function enforce_electron_r_boundary_condition!(f::AbstractArray{mk_float,ndim_pdf_electron},
        density::AbstractArray{mk_float,ndim_field},
        upar::AbstractArray{mk_float,ndim_field}, p::AbstractArray{mk_float,ndim_field},
        moment_r_speed::AbstractArray{mk_float,ndim_field},
        r_boundaries::r_boundary_info, adv, vpa, vperp, z, r, composition,
        end1::AbstractArray{mk_float,ndim_pdf_electron_boundary},
        end2::AbstractArray{mk_float,ndim_pdf_electron_boundary},
        buffer1::AbstractArray{mk_float,ndim_pdf_electron_boundary},
        buffer2::AbstractArray{mk_float,ndim_pdf_electron_boundary}, r_diffusion::Bool)

    nr = r.n
    zero = 1.0e-10

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @begin_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            end1[ivpa,ivperp,iz] = f[ivpa,ivperp,iz,1]
            end2[ivpa,ivperp,iz] = f[ivpa,ivperp,iz,nr]
        end
        reconcile_element_boundaries_MPI!(f, end1, end2, buffer1, buffer2, r)

        # Hacky way to get buffers to use for moment communication, to avoid having to
        # pass in even more buffers. We know that the distribution function buffers are
        # bigger than those needed for moments, so we can just take a chunk at the
        # beginning of the distribution function buffers.
        moment_end1 = @view end1[1:z.n]
        moment_end2 = @view end2[1:z.n]
        moment_buffer1 = @view buffer1[1:z.n]
        moment_buffer2 = @view buffer2[1:z.n]
        @begin_z_region()
        @loop_z iz begin
            moment_end1[iz] = density[iz,1]
            moment_end2[iz] = density[iz,nr]
        end
        reconcile_element_boundaries_MPI!(density, end1, end2, buffer1, buffer2, r)
        @loop_z iz begin
            moment_end1[iz] = upar[iz,1]
            moment_end2[iz] = upar[iz,nr]
        end
        reconcile_element_boundaries_MPI!(upar, end1, end2, buffer1, buffer2, r)
        @loop_z iz begin
            moment_end1[iz] = p[iz,1]
            moment_end2[iz] = p[iz,nr]
        end
        reconcile_element_boundaries_MPI!(p, end1, end2, buffer1, buffer2, r)
    end

    for (ir, sections_tuple) ∈ ((1, r_boundaries.inner_sections),
                                (nr, r_boundaries.outer_sections))
        for section ∈ sections_tuple
            electron_section = section.electron
            # 'periodic' BC enforces periodicity by taking the average of the boundary points
            # enforce the condition if r is local
            if isa(electron_section, electron_r_boundary_section_periodic)
                if r.nelement_global != r.nelement_local
                    error("Periodic r-boundaries not yet implemented when r-dimension "
                          * " is distributed")
                end
                if ir > 1
                    # Both r-boundaries for periodic sections handled by inner boundary.
                    # We have checked that there is a corresponding periodic
                    # outer-boundary section defined.
                    continue
                end
                @begin_vperp_vpa_region()
                for iz ∈ section.z_range
                    @loop_vperp_vpa ivperp ivpa begin
                        f[ivpa,ivperp,iz,1] = 0.5 * (f[ivpa,ivperp,iz,nr] +
                                                     f[ivpa,ivperp,iz,1])
                        f[ivpa,ivperp,iz,nr] = f[ivpa,ivperp,iz,1]
                    end
                end
                @begin_serial_region()
                @serial_region begin
                    for iz ∈ section.z_range
                        density[iz,1] = 0.5 * (density[iz,nr] + density[iz,1])
                        density[iz,nr] = density[iz,1]
                        upar[iz,1] = 0.5 * (upar[iz,nr] + upar[iz,1])
                        upar[iz,nr] = upar[iz,1]
                        p[iz,1] = 0.5 * (p[iz,nr] + p[iz,1])
                        p[iz,nr] = p[iz,1]
                    end
                end
            elseif isa(electron_section, electron_r_boundary_section_Dirichlet)
                # use the old distribution to force the new distribution to have
                # consistant-in-time values at the boundary
                # with bc = "Dirichlet" and r_diffusion = false
                # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
                # with bc = "Dirichlet" and r_diffusion = true
                # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
                f_boundary = electron_section.pdf
                @begin_vperp_vpa_region()
                for (iz_section, iz) ∈ enumerate(section.z_range)
                    @loop_vperp_vpa ivperp ivpa begin
                        if r_diffusion || (ir == 1 ? adv[1].speed[ir,ivpa,ivperp,iz] > zero
                                                   : adv[1].speed[ir,ivpa,ivperp,iz] < -zero)
                            f[ivpa,ivperp,iz,ir] = f_boundary[ivpa,ivperp,iz_section]
                        end
                    end
                end
                n_boundary = ion_section.density
                u_boundary = ion_section.upar
                p_boundary = ion_section.p
                @begin_serial_region()
                @serial_region begin
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        if r_diffusion || (ir == 1 ? moment_r_speed[iz,ir] > zero
                                                   : moment_r_speed[iz,ir] < -zero)
                            density[iz,ir] = n_boundary[iz_section]
                            upar[iz,ir] = u_boundary[iz_section]
                            p[iz,ir] = p[iz_section]
                        end
                    end
                end
            elseif isa(electron_section, electron_r_boundary_section_Neumann)
                # with bc = "Neumann" and r_diffusion = false
                # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
                # with bc = "Neumann" and r_diffusion = true
                # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
                one_over_logarithmic_gradient_value_minus_Db =
                    electron_section.one_over_logarithmic_gradient_value_minus_Db
                derivative_coefficients = electron_section.derivative_coefficients
                if electron_section.is_inner
                    other_elements_range = 2:r.ngrid
                else
                    other_elements_range = r.n-r.ngrid+1:r.n-1
                end
                @begin_vperp_vpa_region()
                for (iz_section, iz) ∈ enumerate(section.z_range)
                    @loop_vperp_vpa ivperp ivpa begin
                        if r_diffusion || (ir == 1 ? adv[1].speed[ir,ivpa,ivperp,iz] > zero
                                                   : adv[1].speed[ir,ivpa,ivperp,iz] < -zero)
                            @views f[ivpa,ivperp,iz,ir,is] =
                                dot(derivative_coefficients,
                                    f[ivpa,ivperp,iz,other_elements_range,is]) *
                                one_over_logarithmic_gradient_value_minus_Db
                        end
                    end
                end
                @begin_serial_region()
                @serial_region begin
                    for iz ∈ section.z_range
                        if r_diffusion || (ir == 1 ? moment_r_speed[iz,ir] > zero
                                                   : moment_r_speed[iz,ir] < -zero)
                            @views density[iz,ir] =
                                dot(derivative_coefficients,
                                    density[iz,other_elements_range]) *
                                one_over_logarithmic_gradient_value_minus_Db
                            @views upar[iz,ir] =
                                dot(derivative_coefficients,
                                    upar[iz,other_elements_range]) *
                                one_over_logarithmic_gradient_value_minus_Db
                            @views p[iz,ir] =
                                dot(derivative_coefficients,
                                    p[iz,other_elements_range]) *
                                one_over_logarithmic_gradient_value_minus_Db
                        end
                    end
                end
            else
                error("Unsupported section type $(typeof(electron_section)).")
            end
        end
    end
end

"""
enforce boundary conditions on neutral particle distribution function
"""
@timeit global_timer enforce_neutral_boundary_conditions!(
                         f_neutral, f_ion, density_neutral, uz_neutral, ur_neutral,
                         p_neutral, boundaries::boundary_info, moments, density_ion,
                         upar_ion, Er, vzeta_spectral, vr_spectral, vz_spectral, r_adv,
                         z_adv, vzeta_adv, vr_adv, vz_adv, r, z, vzeta, vr, vz,
                         composition, geometry, scratch_dummy, r_diffusion,
                         vz_diffusion) = begin

    # without acceleration of neutrals bc on vz vr vzeta should not be required as no
    # advection or diffusion in these coordinates

    if vzeta_adv !== nothing && vzeta.n_global > 1 && vzeta.bc != "none"
        @begin_sn_r_z_vr_vz_region()
        @loop_sn_r_z_vr_vz isn ir iz ivr ivz begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[ivz,ivr,:,iz,ir,isn],
                                                       vzeta.bc,
                                                       vzeta_adv[isn].speed[ivz,ivr,:,iz,ir],
                                                       false, vzeta, vzeta_spectral)
        end
    end
    if vr_adv !== nothing && vr.n_global > 1 && vr.bc != "none"
        @begin_sn_r_z_vzeta_vz_region()
        @loop_sn_r_z_vzeta_vz isn ir iz ivzeta ivz begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[ivz,:,ivzeta,iz,ir,isn],
                                                       vr.bc,
                                                       vr_adv[isn].speed[ivz,:,ivzeta,iz,ir],
                                                       false, vr, vr_spectral)
        end
    end
    if vz_adv !== nothing && vz.n_global > 1 && vz.bc != "none"
        @begin_sn_r_z_vzeta_vr_region()
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            # enforce the vz BC
            @views enforce_v_boundary_condition_local!(f_neutral[:,ivr,ivzeta,iz,ir,isn],
                                                       vz.bc,
                                                       vz_adv[isn].speed[:,ivr,ivzeta,iz,ir],
                                                       vz_diffusion, vz, vz_spectral)
        end
    end
    # f_initial contains the initial condition for enforcing a fixed-boundary-value condition
    if z.n > 1
        @begin_sn_r_vzeta_vr_vz_region()
        enforce_neutral_z_boundary_condition!(f_neutral, density_neutral, uz_neutral,
            p_neutral, moments, density_ion, upar_ion, Er, boundaries.z, z_adv, z, vzeta,
            vr, vz, composition, geometry, scratch_dummy.buffer_vzvrvzetarsn_1,
            scratch_dummy.buffer_vzvrvzetarsn_2, scratch_dummy.buffer_vzvrvzetarsn_3,
            scratch_dummy.buffer_vzvrvzetarsn_4,
            scratch_dummy.buffer_vzvrvzetarsn_5)
    end
    if r.n > 1
        enforce_neutral_r_boundary_condition!(f_neutral, density_neutral, uz_neutral,
            ur_neutral, p_neutral, boundaries.r, r_adv, vz, vr, vzeta, z, r, composition,
            scratch_dummy.buffer_vzvrvzetazsn_1, scratch_dummy.buffer_vzvrvzetazsn_2,
            scratch_dummy.buffer_vzvrvzetazsn_3, scratch_dummy.buffer_vzvrvzetazsn_4,
            r_diffusion)
    end
end

"""
enforce boundary conditions on neutrals in r
"""
function enforce_neutral_r_boundary_condition!(
        f::AbstractArray{mk_float,ndim_pdf_neutral},
        density::AbstractArray{mk_float,ndim_moment},
        uz::AbstractArray{mk_float,ndim_moment}, ur::AbstractArray{mk_float,ndim_moment},
        p::AbstractArray{mk_float,ndim_moment}, r_boundaries::r_boundary_info, adv, vz,
        vr, vzeta, z, r, composition,
        end1::AbstractArray{mk_float,ndim_pdf_neutral_boundary},
        end2::AbstractArray{mk_float,ndim_pdf_neutral_boundary},
        buffer1::AbstractArray{mk_float,ndim_pdf_neutral_boundary},
        buffer2::AbstractArray{mk_float,ndim_pdf_neutral_boundary}, r_diffusion::Bool)

    nr = r.n
    zero = 1.0e-10

    if r.nelement_global > r.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        @begin_sn_z_vzeta_vr_vz_region()
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            end1[ivz,ivr,ivzeta,iz,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
            end2[ivz,ivr,ivzeta,iz,isn] = f[ivz,ivr,ivzeta,iz,nr,isn]
        end
        reconcile_element_boundaries_MPI!(f, end1, end2, buffer1, buffer2, r)

        # Hacky way to get buffers to use for moment communication, to avoid having to
        # pass in even more buffers. We know that the distribution function buffers are
        # bigger than those needed for moments, so we can just take a chunk at the
        # beginning of the distribution function buffers.
        moment_end1 = reshape(@view(end1[1:z.n*composition.n_neutral_species]), z.n,
                              composition.n_neutral_species)
        moment_end2 = reshape(@view(end2[1:z.n*composition.n_neutral_species]), z.n,
                              composition.n_neutral_species)
        moment_buffer1 = reshape(@view(buffer1[1:z.n*composition.n_neutral_species]), z.n,
                                 composition.n_neutral_species)
        moment_buffer2 = reshape(@view(buffer2[1:z.n*composition.n_neutral_species]), z.n,
                                 composition.n_neutral_species)
        @begin_sn_z_region()
        @loop_sn_z isn iz begin
            moment_end1[iz,isn] = density[iz,1,isn]
            moment_end2[iz,isn] = density[iz,nr,isn]
        end
        reconcile_element_boundaries_MPI!(density, end1, end2, buffer1, buffer2, r)
        @loop_sn_z isn iz begin
            moment_end1[iz,isn] = uz[iz,1,isn]
            moment_end2[iz,isn] = uz[iz,nr,isn]
        end
        reconcile_element_boundaries_MPI!(uz, end1, end2, buffer1, buffer2, r)
        @loop_sn_z isn iz begin
            moment_end1[iz,isn] = p[iz,1,isn]
            moment_end2[iz,isn] = p[iz,nr,isn]
        end
        reconcile_element_boundaries_MPI!(p, end1, end2, buffer1, buffer2, r)
    end

    for (ir, sections_tuple) ∈ ((1, r_boundaries.inner_sections),
                                (nr, r_boundaries.outer_sections))
        for section ∈ sections_tuple
            neutral_section = section.neutral
            # 'periodic' BC enforces periodicity by taking the average of the boundary points
            # enforce the condition if r is local
            if isa(neutral_section, neutral_r_boundary_section_periodic)
                if r.nelement_global != r.nelement_local
                    error("Periodic r-boundaries not yet implemented when r-dimension "
                          * " is distributed")
                end
                if ir > 1
                    # Both r-boundaries for periodic sections handled by inner boundary.
                    # We have checked that there is a corresponding periodic
                    # outer-boundary section defined.
                    continue
                end
                @begin_sn_vzeta_vr_vz_region()
                @loop_sn isn begin
                    for iz ∈ section.z_range
                        @loop_vzeta_vr_vz ivzeta ivr ivz begin
                            f[ivz,ivr,ivzeta,iz,1,isn] = 0.5 * (f[ivz,ivr,ivzeta,iz,nr,isn] +
                                                               f[ivz,ivr,ivzeta,iz,1,isn])
                            f[ivz,ivr,ivzeta,iz,nr,isn] = f[ivz,ivr,ivzeta,iz,1,isn]
                        end
                    end
                end
                @begin_sn_region()
                @loop_sn isn begin
                    for iz ∈ section.z_range
                        density[iz,1,isn] = 0.5 * (density[iz,nr,isn] + density[iz,1,isn])
                        density[iz,nr,isn] = density[iz,1,isn]
                        uz[iz,1,isn] = 0.5 * (uz[iz,nr,isn] + uz[iz,1,isn])
                        uz[iz,nr,isn] = uz[iz,1,isn]
                        p[iz,1,isn] = 0.5 * (p[iz,nr,isn] + p[iz,1,isn])
                        p[iz,nr,isn] = p[iz,1,isn]
                    end
                end
            elseif isa(neutral_section, neutral_r_boundary_section_Dirichlet)
                # use the old distribution to force the new distribution to have
                # consistant-in-time values at the boundary
                # with bc = "Dirichlet" and r_diffusion = false
                # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
                # with bc = "Dirichlet" and r_diffusion = true
                # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
                f_boundary = ion_section.pdf
                @begin_sn_vzeta_vr_vz_region()
                @loop_sn isn begin
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        @loop_vzeta_vr_vz ivzeta ivr ivz begin
                            if r_diffusion || (ir == 1 ? adv[isn].speed[ir,ivz,ivr,ivzeta,iz] > zero
                                                       : adv[isn].speed[ir,ivz,ivr,ivzeta,iz] < -zero)
                                f[ivz,ivr,ivzeta,iz,ir,isn] = f_boundary[ivz,ivr,ivzeta,iz_section,isn]
                            end
                        end
                    end
                end
                n_boundary = neutral_section.density
                u_boundary = neutral_section.uz
                p_boundary = neutral_section.p
                @begin_sn_region()
                @loop_sn isn begin
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        if r_diffusion || (ir == 1 ? ur[iz,ir,isn] > zero
                                                   : ur[iz,ir,isn] < -zero)
                            density[iz,ir,isn] = n_boundary[iz_section,isn]
                            uz[iz,ir,isn] = u_boundary[iz_section,isn]
                            p[iz,ir,isn] = p[iz_section,isn]
                        end
                    end
                end
            elseif isa(neutral_section, neutral_r_boundary_section_Neumann)
                # with bc = "Neumann" and r_diffusion = false
                # impose bc for incoming parts of velocity space only (Hyperbolic PDE)
                # with bc = "Neumann" and r_diffusion = true
                # impose bc on both sides of the domain to accomodate a diffusion operator d^2 / d r^2
                one_over_logarithmic_gradient_value_minus_Db =
                    neutral_section.one_over_logarithmic_gradient_value_minus_Db
                derivative_coefficients = neutral_section.derivative_coefficients
                if neutral_section.is_inner
                    other_elements_range = 2:r.ngrid
                else
                    other_elements_range = r.n-r.ngrid+1:r.n-1
                end
                @begin_sn_vzeta_vr_vz_region()
                @loop_sn isn begin
                    for (iz_section, iz) ∈ enumerate(section.z_range)
                        @loop_vzeta_vr_vz ivzeta ivr ivz begin
                            if r_diffusion || (ir == 1 ? adv[isn].speed[ir,ivz,ivr,ivzeta,iz] > zero
                                                       : adv[isn].speed[ir,ivz,ivr,ivzeta,iz] < -zero)
                                @views f[ivz,ivr,ivzeta,iz,ir,isn] =
                                    dot(derivative_coefficients,
                                        f[ivz,ivr,ivzeta,iz,other_elements_range,isn]) *
                                    one_over_logarithmic_gradient_value_minus_Db
                            end
                        end
                    end
                end
                @begin_sn_region()
                @loop_sn isn begin
                    for iz ∈ section.z_range
                        if r_diffusion || (ir == 1 ? ur[iz,ir,isn] > zero
                                                   : ur[iz,ir,isn] < -zero)
                            @views density[iz,ir,isn] =
                                dot(derivative_coefficients,
                                    density[iz,other_elements_range,isn]) *
                                one_over_logarithmic_gradient_value_minus_Db
                            @views uz[iz,ir,isn] =
                                dot(derivative_coefficients,
                                    uz[iz,other_elements_range,isn]) *
                                one_over_logarithmic_gradient_value_minus_Db
                            @views p[iz,ir,isn] =
                                dot(derivative_coefficients,
                                    p[iz,other_elements_range,isn]) *
                                one_over_logarithmic_gradient_value_minus_Db
                        end
                    end
                end
            else
                error("Unsupported section type $(typeof(neutral_section)).")
            end
        end
    end
end

"""
enforce boundary conditions on neutral particle f in z
"""
function enforce_neutral_z_boundary_condition!(pdf, density, uz, pz, moments, density_ion,
                                               upar_ion, Er, z_boundaries, adv,
                                               z, vzeta, vr, vz, composition, geometry,
                                               end1::AbstractArray{mk_float,5}, end2::AbstractArray{mk_float,5},
                                               buffer1::AbstractArray{mk_float,5}, buffer2::AbstractArray{mk_float,5},
                                               buffer3::AbstractArray{mk_float,5})


    if z.nelement_global > z.nelement_local
        # reconcile internal element boundaries across processes
        # & enforce periodicity and external boundaries if needed
        nz = z.n
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            end1[ivz,ivr,ivzeta,ir,isn] = pdf[ivz,ivr,ivzeta,1,ir,isn]
            end2[ivz,ivr,ivzeta,ir,isn] = pdf[ivz,ivr,ivzeta,nz,ir,isn]
        end
        # check on periodic bc occurs within this call below
        reconcile_element_boundaries_MPI!(pdf, end1, end2, buffer1, buffer2, z)
    end

    zero = 1.0e-14
    # 'constant' BC is time-independent f at upwind boundary
    # and constant f beyond boundary
    if z.bc == "constant"
        @begin_sn_r_vzeta_vr_vz_region()
        density_offset = 1.0
        vwidth = 1.0
        if vzeta.n == 1 && vr.n == 1
            constant_prefactor = density_offset / sqrt(π)
        else
            constant_prefactor = density_offset / π^1.5
        end
        if z.irank == 0
            @loop_sn isn begin
                speed = adv[isn].speed
                @loop_r ir begin
                    prefactor = constant_prefactor
                    if moments.evolve_density
                        prefactor /= density[1,ir,isn]
                    end
                    if moments.evolve_p
                        prefactor *= moments.neutral.vth[1,ir,isn]
                    end
                    @loop_vzeta_vr_vz ivzeta ivr ivz begin
                        if speed[1,ivz,ivr,ivzeta,ir] > 0.0
                            pdf[ivz,ivr,ivzeta,1,ir,isn] = prefactor *
                                exp(-(speed[1,ivz,ivr,ivzeta,ir]^2 + vr.grid[ivr] + vz.grid[ivz])/vwidth^2)
                        end
                    end
                end
            end
        end
        if z.irank == z.nrank - 1
            @loop_sn isn begin
                speed = adv[isn].speed
                @loop_r ir begin
                    prefactor = constant_prefactor
                    if moments.evolve_density
                        prefactor /= density[end,ir,isn]
                    end
                    if moments.evolve_p
                        prefactor *= moments.neutral.vth[end,ir,isn]
                    end
                    @loop_vzeta_vr_vz ivzeta ivr ivz begin
                        if speed[end,ivz,ivr,ivzeta,ir] > 0.0
                            pdf[ivz,ivr,ivzeta,end,ir,isn] = prefactor *
                                exp(-(speed[end,ivz,ivr,ivzeta,ir][ivzeta]^2 + vr.grid[ivr] + vz.grid[ivz])/vwidth^2)
                        end
                    end
                end
            end
        end
    # 'periodic' BC enforces periodicity by taking the average of the boundary points
    elseif z.bc == "periodic" && z.nelement_global == z.nelement_local
        @begin_sn_r_vzeta_vr_vz_region()
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            pdf[ivz,ivr,ivzeta,1,ir,isn] = 0.5*(pdf[ivz,ivr,ivzeta,1,ir,isn] +
                                                pdf[ivz,ivr,ivzeta,end,ir,isn])
            pdf[ivz,ivr,ivzeta,end,ir,isn] = pdf[ivz,ivr,ivzeta,1,ir,isn]
        end
    # 'wall' BC enforces wall boundary conditions
    elseif z.bc == "wall"
        # Need integrals over vpa at wall boundaries in z, so cannot parallelize over z
        # or vpa.
        @begin_sn_r_region()
        @loop_sn isn begin
            # BC for neutrals
            @loop_r ir begin
                # define T_wall_over_m to avoid repeated computation below
                T_wall_over_m = composition.T_wall / composition.mn_over_mi
                # Assume for now that the ion species index corresponding to this neutral
                # species is the same as the neutral species index.
                # Note, have already calculated moments of ion distribution function(s),
                # so can use the moments here to get the flux
                if z.irank == 0
                    ion_flux_0 = -density_ion[1,ir,isn] * (upar_ion[1,ir,isn]*geometry.bzed[1,ir] - geometry.rhostar*Er[1,ir])
                else
                    ion_flux_0 = NaN
                end
                if z.irank == z.nrank - 1
                    ion_flux_L = density_ion[end,ir,isn] * (upar_ion[end,ir,isn]*geometry.bzed[end,ir] - geometry.rhostar*Er[end,ir])
                else
                    ion_flux_L = NaN
                end
                # enforce boundary condition on the neutral pdf that all ions and neutrals
                # that leave the domain re-enter as neutrals
                @views enforce_neutral_wall_bc!(
                    pdf[:,:,:,:,ir,isn], z, vzeta, vr, vz, pz[:,ir,isn], uz[:,ir,isn],
                    density[:,ir,isn], ion_flux_0, ion_flux_L, z_boundaries,
                    T_wall_over_m, composition.recycling_fraction, moments.evolve_p,
                    moments.evolve_upar, moments.evolve_density, zero,
                    buffer3[:,:,:,ir,isn])
            end
        end
    end
end

"""
enforce a zero incoming BC in z for given species pdf at each radial location
"""
function enforce_zero_incoming_bc!(pdf, speed, z, zero, phi, epsz)
    nvpa = size(pdf,1)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa
    #
    # epsz is the ratio |z - z_wall|/|delta z|, with delta z the grid spacing at the wall
    # for epsz < 1, the cut off below would be imposed for particles travelling
    # out to a distance z = epsz * delta z from the wall before returning, as the cutoff
    # is imposed when v_∥<vcut, where 0.5*m*vcut^2 = sqrt(epsz)*delta_phi and assuming
    # that phi ∝ sqrt(z) near the wall, so that z = epsz * delta_z corresponds to
    # phi = sqrt(epsz) * delta_phi.
    if z.irank == 0
        deltaphi = phi[2] - phi[1]
        vcut = deltaphi > 0 ? sqrt(2.0 * deltaphi)*(epsz^0.25) : 0.0
        @loop_vperp_vpa ivperp ivpa begin
            # for left boundary in zed (z = -Lz/2), want
            # f(z=-Lz/2, v_parallel > 0) = 0
            if speed[1,ivpa,ivperp] > zero - vcut
                pdf[ivpa,ivperp,1] = 0.0
            end
        end
    end
    if z.irank == z.nrank - 1
        deltaphi = phi[end-1] - phi[end]
        vcut = deltaphi > 0 ? sqrt(2.0 * deltaphi)*(epsz^0.25) : 0.0
        @loop_vperp_vpa ivperp ivpa begin
            # for right boundary in zed (z = Lz/2), want
            # f(z=Lz/2, v_parallel < 0) = 0
            if speed[end,ivpa,ivperp] < -zero + vcut
                pdf[ivpa,ivperp,end] = 0.0
            end
        end
    end
end
function get_ion_z_boundary_cutoff_indices(density, upar, vEz, bz, vth0, vthL,
                                           evolve_upar, evolve_p, z, vpa, zero, phi)
    epsz = z.boundary_parameters.epsz
    if z.irank == 0
        deltaphi = phi[2] - phi[1]
        vz_cut = deltaphi > 0 ? bz[1] * sqrt(2.0 * deltaphi)*(epsz^0.25) : 0.0
        vpa_cut = (vz_cut - vEz[1]) / bz[1]
        wpa_cut_lower = vpa_to_wpa(vpa_cut, vth0, upar[1], evolve_p, evolve_upar)
        last_negative_vpa_ind = searchsortedlast(vpa.grid, wpa_cut_lower)
    else
        wpa_cut_lower = nothing
        last_negative_vpa_ind = nothing
    end
    if z.irank == z.nrank - 1
        deltaphi = phi[end-1] - phi[end]
        vz_cut = deltaphi > 0 ? bz[end] * sqrt(2.0 * deltaphi)*(epsz^0.25) : 0.0
        vpa_cut = (vz_cut - vEz[end]) / bz[end]
        wpa_cut_upper = vpa_to_wpa(vpa_cut, vthL, upar[end], evolve_p, evolve_upar)
        first_positive_vpa_ind = searchsortedfirst(vpa.grid, wpa_cut_upper)
    else
        wpa_cut_upper = nothing
        first_positive_vpa_ind = nothing
    end
    return last_negative_vpa_ind, wpa_cut_lower, first_positive_vpa_ind, wpa_cut_upper
end
function enforce_zero_incoming_bc!(pdf, z::coordinate, vperp::coordinate, vpa::coordinate,
                                   density, upar, p, vEz, bz, evolve_upar, evolve_p, zero,
                                   phi, vpavperp_buffer)
    if z.irank != 0 && z.irank != z.nrank - 1
        # No z-boundary in this block
        return nothing
    end
    nvpa, nvperp, nz = size(pdf)
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    vth0 = sqrt(2.0 * p[1] / density[1])
    vthL = sqrt(2.0 * p[end] / density[end])

    # absolute velocity at left boundary
    last_negative_vpa_ind, wpa_cut_lower, first_positive_vpa_ind, wpa_cut_upper =
        get_ion_z_boundary_cutoff_indices(density, upar, vEz, bz, vth0, vthL, evolve_upar,
                                          evolve_p, z, vpa, zero, phi)
    if z.irank == 0
        pdf[last_negative_vpa_ind+1:end,:,1] .= 0.0
        # Limit last non-zero grid point to be less than the linear fit between the
        # second-last non-zero point and zero at the cut-off velocity. This ensures that
        # there is no jump when the cut-off velocity passes this grid point, but the
        # boundary condition can also be re-applied giving the exact same result.
        @. pdf[last_negative_vpa_ind,:,1] = min(pdf[last_negative_vpa_ind,:,1],
                                                pdf[last_negative_vpa_ind-1,:,1] *
                                                (wpa_cut_lower - vpa.grid[last_negative_vpa_ind]) /
                                                (vpa.grid[last_negative_vpa_ind+1] - vpa.grid[last_negative_vpa_ind]))
    end
    # absolute velocity at right boundary
    if z.irank == z.nrank - 1
        pdf[1:first_positive_vpa_ind-1,:,end] .= 0.0
        # Limit first non-zero grid point to be less than the linear fit between the
        # second non-zero point and zero at the cut-off velocity. This ensures that
        # there is no jump when the cut-off velocity passes this grid point, but the
        # boundary condition can also be re-applied giving the exact same result.
        @. pdf[first_positive_vpa_ind,:,end] = min(pdf[first_positive_vpa_ind,:,end],
                                                   pdf[first_positive_vpa_ind+1,:,end] *
                                                 (vpa.grid[first_positive_vpa_ind] - wpa_cut_upper) /
                                                 (vpa.grid[first_positive_vpa_ind] - vpa.grid[first_positive_vpa_ind-1]))
    end

    # Special constraint-forcing code that tries to keep the modifications smooth at
    # v_parallel=0.
    if z.irank == 0 && z.irank == z.nrank - 1
        # Both z-boundaries in this block
        z_range = (1,nz)
        vth_list = (vth0, vthL)
    elseif z.irank == 0
        z_range = (1,)
        vth_list = (vth0,)
    elseif z.irank == z.nrank - 1
        z_range = (nz,)
        vth_list = (vthL,)
    else
        error("No boundary in this block, should have returned already")
    end
    for (iz, vth) ∈ zip(z_range, vth_list)
        @boundscheck size(pdf,2) == 1
        f = @view pdf[:,:,iz]
        if evolve_p && evolve_upar
            I0 = integral((vperp,vpa)->(1), f, vperp, vpa)
            I1 = integral((vperp,vpa)->(vpa), f, vperp, vpa)
            I2 = integral((vperp,vpa)->(vpa^2 + vperp^2), f, vperp, vpa)

            # The velocities where ions do not come back from the wall are (see TN-07
            # report "2D drift kinetic model with wall boundary conditions")
            # v^z ≶ 0
            # b^z v_∥ + vEz ≶ 0
            # b^z (vth*w_∥ + u_∥) + vEz ≶ 0
            # Store v_parallel/vth with cutoff shift removed (which is equal to
            # v^z/vth/b^z) in vpa.scratch
            @. vpa.scratch = vpa.grid + (upar[iz] + vEz[iz] / bz[iz])/vth
            # Introduce factors to ensure corrections go smoothly to zero near
            # v_parallel=0, and that there are no large corrections aw large w_parallel as
            # those can have a strong effect on the parallel heat flux and make
            # timestepping unstable when the cut-off point jumps from one grid point to
            # another.
            if vperp.n == 1
                # Scale relative to thermal speed calculated with parallel temperature
                # rather than total temperature.
                one_over_scale_factor = sqrt(3.0)
            else
                one_over_scale_factor = 1.0
            end

            @. vpa.scratch2 = abs(vpa.scratch) / (one_over_scale_factor + abs(vpa.scratch)) / (1.0 + (4.0 * vpa.scratch / vpa.L)^4)
            # vpa is the first index of vpavperp_buffer, so this broadcast operation can handle vpa.scratch being copied along
            # the columns of vpavperp_buffer. Even if vpa.n = vperp.n, the operation is unambiguous.
            @. vpavperp_buffer .= vpa.scratch2 * f

            J1 = integral((vperp,vpa)->(vpa), vpavperp_buffer, vperp, vpa)
            J2 = integral((vperp,vpa)->((vpa^2+vperp^2)), vpavperp_buffer, vperp, vpa)
            J3 = integral((vperp,vpa)->(vpa*(vpa^2+vperp^2)), vpavperp_buffer, vperp, vpa)
            J4 = integral((vperp,vpa)->((vpa^2+vperp^2)^2), vpavperp_buffer, vperp, vpa)
            J5 = integral((vperp,vpa)->(vpa^2), vpavperp_buffer, vperp, vpa)
            # Given a corrected distribution function
            #   F = A * Fhat + (B*wpa + C*wpa^2) * s*|vpa/vth| / (1 + s*|vpa/vth|) / (1 +(4*vpa/vth/Lvpa)^4) * Fhat
            # calling the prefactor in the second term on the RHS (coefficient of (B*wpa + C*wpa^2)*Fhat) as P(vpa,vperp,z,t),
            #
            # the constraints 
            #   ∫d^3w F = 1
            #   ∫d^3w wpa F = 0
            #   ∫d^3w (wpa + wperp)^2 F = 3/2
            #
            # and defining the integrals
            # I0 = ∫d^3w F
            # I1 = ∫d^3w wpa F
            # I2 = ∫d^3w (wpa^2 + wperp^2) F
            # J1 = ∫d^3w wpa * P * F
            # J2 = ∫d^3w (wpa^2 + wperp^2) * P * F
            # J3 = ∫d^3w wpa * (wpa^2 + wperp^2) * P * F
            # J4 = ∫d^3w (wpa^2 + wperp^2)^2 * P * F
            # J5 = ∫d^3w wpa^2 * P * F
            #
            #
            # we can substitute F into the constraint equations and solve for A, B, and C
            #   A I0 + B J1 + C J2 = 1
            #   A I1 + B J5 + C J3 = 0
            #   A I2 + B J3 + C J4 = 3/2
            # ⇒
            # inverting 3x3 matrix to get A B and C as functions of the coefficients: 
            # 
            # determinant = I0*(J5*J4 - J3^2) - J1 * (I1*J4 - I2*J3) + J2*(I1*J3 - I2*J5)
            # A = (J4*J5 - J3^2 + 1.5*(J1*J3 - J2*J5)) / determinant
            # B = (J3*I2 - I1*J4 + 1.5*(J2*I1 - I0*J3)) / determinant
            # C = (I1*J3 - J5*I2 + 1.5*(I0*J5 - I1*J1)) / determinant

            determinant = I0*(J5*J4 - J3^2) - J1 * (I1*J4 - I2*J3) + J2*(I1*J3 - I2*J5)
            A = (J4*J5 - J3^2 + 1.5*(J1*J3 - J2*J5)) / determinant
            B = (J3*I2 - I1*J4 + 1.5*(J2*I1 - I0*J3)) / determinant
            C = (I1*J3 - J5*I2 + 1.5*(I0*J5 - I1*J1)) / determinant

            @. f = A*f + B*vpa.grid*vpa.scratch2*f + C*vpa.grid*vpa.grid*vpa.scratch2*f
        elseif evolve_upar
            I0 = integral((vperp,vpa)->(1), f, vperp, vpa)
            I1 = integral((vperp,vpa)->(vpa), f, vperp, vpa)

            # The velocities where ions do not come back from the wall are (see TN-07
            # report "2D drift kinetic model with wall boundary conditions")
            # v^z ≶ 0
            # b^z v_∥ + vEz ≶ 0
            # b^z (w_∥ + u_∥) + vEz ≶ 0
            # Store v_parallel with cutoff shift removed (which is equal to
            # v^z/b^z) in vpa.scratch
            @. vpa.scratch = vpa.grid + upar[iz] + vEz[iz] / bz[iz]
            # Introduce factors to ensure corrections go smoothly to zero near
            # v_parallel=0, and that there are no large corrections aw large w_parallel as
            # those can have a strong effect on the parallel heat flux and make
            # timestepping unstable when the cut-off point jumps from one grid point to
            # another.
            # Factor sqrt(2) below is chosen so that the transition happens at ~vth when
            # T/Tref = 1, or for the 1V case at ~sqrt(2 T_∥ / m_i) when T_∥/Tref = 1.

            @. vpa.scratch2 = abs(vpa.scratch) / (sqrt(2.0) + abs(vpa.scratch)) / (1.0 + (4.0 * vpa.scratch / vpa.L)^4)
            # vpa is the first index of vpavperp_buffer, so this broadcast operation can handle vpa.scratch being copied along
            # the columns of vpavperp_buffer. Even if vpa.n = vperp.n, the operation is unambiguous.
            @. vpavperp_buffer .= vpa.scratch2 * f

            J1 = integral((vperp,vpa)->(vpa), vpavperp_buffer, vperp, vpa)
            J2 = integral((vperp,vpa)->(vpa^2), vpavperp_buffer, vperp, vpa)

            # Given a corrected distribution function
            #   F = A * Fhat + B*wpa * s*vpa / (1 + s*|vpa|) / (1 +(4*vpa/Lvpa)^4) * Fhat
            # the constraints 
            #   ∫d^3w F = 1
            #   ∫d^3w wpa F = 0
            # and defining the integrals
            #   In = ∫d^3w wpa^n * F
            #   Jn = ∫d^3w wpa^n * s*vpa / (1 + s*|vpa|) / (1 +(4*vpa/Lvpa)^4) * F
            # we can substitute F into the constraint equations and solve for A and B
            #   A I0 + B J1 = 1
            #   A I1 + B J2 = 0
            # ⇒
            #   B = -A I1 / J2
            #   A I0 = 1 - B J1
            #   A I0 J2 = J2 + A I1 J1
            #   A = J2 / (I0 J2 - I1 J1)
            #   A = 1 / (I0 - I1 J1 / J2)

            A = 1.0 / (I0 - I1*J1/J2)
            B = -A*I1/J2
            @. f = A*f + B*vpa.grid*vpa.scratch2*f
        elseif evolve_density
            I0 = integral((vperp,vpa)->(1), f, vperp, vpa)
            @. f = f / I0
        end
    end
end

"""
Set up an initial condition that tries to be smoothly compatible with the sheath
boundary condition for ions, by setting f(±(v_parallel-u0)<0) where u0=0 at the sheath
boundaries and for z<0 increases linearly to u0=vpa.L at z=0, while for z>0 increases
from u0=-vpa.L at z=0 to zero at the z=z.L/2 sheath.

To be applied to 'full-f' distribution function on v_parallel grid (not w_parallel
grid).
"""
function enforce_initial_tapered_zero_incoming!(pdf, z::coordinate, vpa::coordinate)
    nvpa = size(pdf,1)
    zero = 1.0e-14
    # no parallel BC should be enforced for dz/dt = 0
    # note that the parallel velocity coordinate vpa may be dz/dt or
    # some version of the peculiar velocity (dz/dt - upar),
    # so use advection speed below instead of vpa

    for iz ∈ 1:z.n
        u0 = (2.0*z.grid[iz]/z.L - sign(z.grid[iz])) * vpa.L / 2.0
        if z.grid[iz] < -zero
            for ivpa ∈ 1:nvpa
                if vpa.grid[ivpa] > u0 + zero
                    pdf[ivpa,iz] = 0.0
                end
            end
        elseif z.grid[iz] > zero
            for ivpa ∈ 1:nvpa
                if vpa.grid[ivpa] < u0 - zero
                    pdf[ivpa,iz] = 0.0
                end
            end
        end
    end
end

"""
enforce the wall boundary condition on neutrals;
i.e., the incoming flux of neutrals equals the sum of the ion/neutral outgoing fluxes
"""
function enforce_neutral_wall_bc!(pdf, z, vzeta, vr, vz, pz, uz, density, wall_flux_0,
                                  wall_flux_L, z_boundaries, T_wall_over_m,
                                  recycling_fraction, evolve_p, evolve_upar,
                                  evolve_density, zero, pdf_buffer)

    # Reduce the ion flux by `recycling_fraction` to account for ions absorbed by the
    # wall.
    wall_flux_0 *= recycling_fraction
    wall_flux_L *= recycling_fraction

    if !evolve_density && !evolve_upar
        knudsen_cosine = z_boundaries.knudsen_cosine

        if z.irank == 0
            ## treat z = -Lz/2 boundary ##

            # add the neutral species's contribution to the combined ion/neutral particle
            # flux out of the domain at z=-Lz/2
            @views @. pdf_buffer = abs(vz.grid) * pdf[:,:,:,1]
            wall_flux_0 += integrate_over_negative_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # for left boundary in zed (z = -Lz/2), want
            # f_n(z=-Lz/2, v_parallel > 0) = Γ_0 * f_KW(v_parallel)
            @loop_vz ivz begin
                if vz.grid[ivz] >= -zero
                    @views @. pdf[ivz,:,:,1] = wall_flux_0 * knudsen_cosine[ivz,:,:]
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##

            # add the neutral species's contribution to the combined ion/neutral particle
            # flux out of the domain at z=-Lz/2
            @views @. pdf_buffer = abs(vz.grid) * pdf[:,:,:,end]
            wall_flux_L += integrate_over_positive_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            # for right boundary in zed (z = Lz/2), want
            # f_n(z=Lz/2, v_parallel < 0) = Γ_Lz * f_KW(v_parallel)
            @loop_vz ivz begin
                if vz.grid[ivz] <= zero
                    @views @. pdf[ivz,:,:,end] = wall_flux_L * knudsen_cosine[ivz,:,:]
                end
            end
        end
    elseif !evolve_upar
        # Evolving density case
        knudsen_cosine = z_boundaries.knudsen_cosine

        if z.irank == 0
            ## treat z = -Lz/2 boundary ##

            # Note the numerical integrol of knudsen_cosine was forced to be 1 (to machine
            # precision) when it was initialised.
            @views pdf_integral_0 = integrate_over_negative_vz(pdf[:,:,:,1], vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,1]
            pdf_integral_1 = integrate_over_negative_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_0 = integrate_over_positive_vz(knudsen_cosine, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_1 = 1.0 # This is enforced in initialization

            # Calculate normalisation factors N_in for the incoming and N_out for the
            # Knudsen parts of the distirbution so that ∫dvpa F = 1 and ∫dvpa vpa F = uz
            # Note wall_flux_0 is the ion flux into the wall (reduced by the recycling
            # fraction), and the neutral flux should be out of the wall (i.e. uz>0) so
            # n*uz = |n*uz| = wall_flux_0
            # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
            #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = uz
            uz = wall_flux_0 / density[1]
            N_in = (1 - uz * knudsen_integral_0 / knudsen_integral_1) /
                   (pdf_integral_0
                    - pdf_integral_1 / knudsen_integral_1 * knudsen_integral_0)
            N_out = (uz - N_in * pdf_integral_1) / knudsen_integral_1

            @loop_vz ivz begin
                if vz.grid[ivz] >= -zero
                    @views @. pdf[ivz,:,:,1] = N_out * knudsen_cosine[ivz,:,:]
                else
                    @views @. pdf[ivz,:,:,1] *= N_in
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##

            # Note the numerical integrol of knudsen_cosine was forced to be 1 (to machine
            # precision) when it was initialised.
            @views pdf_integral_0 = integrate_over_positive_vz(pdf[:,:,:,end], vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,end]
            @views pdf_integral_1 = integrate_over_positive_vz(pdf_buffer, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_0 = integrate_over_negative_vz(knudsen_cosine, vz.grid, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            knudsen_integral_1 = -1.0 # This is enforced in initialization

            # Calculate normalisation factors N_in for the incoming and N_out for the
            # Knudsen parts of the distirbution so that ∫dvpa F = 1 and ∫dvpa vpa F = uz
            # Note wall_flux_L is the ion flux into the wall (reduced by the recycling
            # fraction), and the neutral flux should be out of the wall (i.e. uz<0) so
            # -n*uz = |n*uz| = wall_flux_L
            # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
            #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = uz
            uz = -wall_flux_L / density[end]
            N_in = (1 - uz * knudsen_integral_0 / knudsen_integral_1) /
                   (pdf_integral_0
                    - pdf_integral_1 / knudsen_integral_1 * knudsen_integral_0)
            N_out = (uz - N_in * pdf_integral_1) / knudsen_integral_1

            @loop_vz ivz begin
                if vz.grid[ivz] <= zero
                    @views @. pdf[ivz,:,:,end] = N_out * knudsen_cosine[ivz,:,:]
                else
                    @views @. pdf[ivz,:,:,end] *= N_in
                end
            end
        end
    else
        if z.irank == 0
            ## treat z = -Lz/2 boundary ##
            # populate vz.scratch2 array with dz/dt values at z = -Lz/2
            if evolve_p
                vth = sqrt(2.0*pz[1]/density[1])
            else
                vth = nothing
            end
            @. vz.scratch2 = vpagrid_to_vpa(vz.grid, vth, uz[1], evolve_p, evolve_upar)

            # First apply boundary condition that total neutral outflux is equal to ion
            # influx to uz
            uz[1] = wall_flux_0 / density[1]
            #would setting density work better??
            #density[1] = - wall_flux_0 / uz[1]

            # Create normalised Knudsen cosine distribution, to use for positive v_parallel
            # at z = -Lz/2
            # Note this only makes sense for the 1V case with vr.n=vzeta.n=1
            @. vz.scratch = 3.0 * sqrt(π) * (0.5 / T_wall_over_m)^1.5 * abs(vz.scratch2) * erfc(sqrt(0.5 / T_wall_over_m) * abs(vz.scratch2))

            # The v_parallel>0 part of the pdf is replaced by the Knudsen cosine
            # distribution. To ensure the constraints ∫dwpa wpa^m F = 0 are satisfied when
            # necessary, calculate a normalisation factor for the Knudsen distribution (in
            # vz.scratch) and correction terms for the incoming pdf similar to
            # enforce_moment_constraints!().
            #
            # Note that it seems to be important that this boundary condition not be
            # modified by the moment constraints, as that could cause numerical instability.
            # By ensuring that the constraints are satisfied already here,
            # enforce_moment_constraints!() will not change the pdf at the boundary. For
            # ions this is not an issue, because points set to 0 by the bc are not modified
            # from 0 by enforce_moment_constraints!().
            knudsen_integral_0 = integrate_over_positive_vz(vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @. vz.scratch4 = vz.grid * vz.scratch
            knudsen_integral_1 = integrate_over_positive_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            @views pdf_integral_0 = integrate_over_negative_vz(pdf[:,:,:,1], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,1]
            pdf_integral_1 = integrate_over_negative_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            if !evolve_p
                # Calculate normalisation factors N_in for the incoming and N_out for the
                # Knudsen parts of the distirbution so that ∫dwpa F = 1 and ∫dwpa wpa F = 0
                # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = 0
                N_in = 1.0 / (pdf_integral_0 - pdf_integral_1/knudsen_integral_1*knudsen_integral_0)
                N_out = -N_in * pdf_integral_1 / knudsen_integral_1

                zero_vz_ind = 0
                for ivz ∈ 1:vz.n
                    if vz.scratch2[ivz] <= -zero
                        pdf[ivz,:,:,1] .= N_in*pdf[ivz,:,:,1]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_z = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,1] = 0.5*(N_in*pdf[ivz,:,:,1] + N_out*vz.scratch[ivz])
                        else
                            pdf[ivz,:,:,1] .= N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ zero_vz_ind+1:vz.n
                    pdf[ivz,:,:,1] .= N_out*vz.scratch[ivz]
                end
            else
                @. vz.scratch4 = vz.grid * vz.grid * vz.scratch
                knudsen_integral_2 = integrate_over_positive_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views @. pdf_buffer = vz.grid * vz.grid * pdf[:,:,:,1]
                pdf_integral_2 = integrate_over_negative_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                pdf_buffer .*= vz.grid
                @views pdf_integral_3 = integrate_over_negative_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                # Calculate normalisation factor N_out for the Knudsen part of the
                # distirbution and normalisation factor N_in and correction term C*wpa*F_in
                # for the incoming distribution so that ∫d^3w F = 1, ∫d^3w wpa F = 0, and
                # ∫d^3w w^2 F = ∫d^3w wpa^2 F = 3/2
                # ⇒ N_in*pdf_integral_0 + C*pdf_integral_1 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + C*pdf_integral_2 + N_out*knudsen_integral_1 = 0
                #   N_in*pdf_integral_2 + C*pdf_integral_3 + N_out*knudsen_integral_2 = 3/2
                # ⇒
                #   C = (3/2 - N_out*knudsen_integral_2 - N_in*pdf_integral_2) / pdf_integral_3
                #   N_out*knudsen_integral_1 = - N_in*pdf_integral_1 - C*pdf_integral_2
                #   N_out*knudsen_integral_1*pdf_integral_3 = - N_in*pdf_integral_1*pdf_integral_3 - (3/2 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)*pdf_integral_2
                #   N_out*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) = -N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2) - 3/2*pdf_integral_2
                #   N_out = [-N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2) - 3/2*pdf_integral_2] / (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                #   N_in*pdf_integral_0 = 1 - C*pdf_integral_1 - N_out*knudsen_integral_0
                #   N_in*pdf_integral_0*pdf_integral_3 = pdf_integral_3 - (3/2 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)*pdf_integral_1 - N_out*knudsen_integral_0*pdf_integral_3
                #   N_in*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) = pdf_integral_3 - 3/2*pdf_integral_1 + N_out*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)
                #   N_in*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) = (pdf_integral_3 - 3/2*pdf_integral_1)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) + [-N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2) - 3/2*pdf_integral_2]*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)
                #   N_in*[(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) + (pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2)*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)] = (pdf_integral_3 - 3/2*pdf_integral_1)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) - 3/2*pdf_integral_2*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)

                #   N_in*[knudsen_integral_1*pdf_integral_0*pdf_integral_3^2 - knudsen_integral_2*pdf_integral_0*pdf_integral_2*pdf_integral_3 - knudsen_integral_1*pdf_integral_1*pdf_integral_2*pdf_integral_3 + knudsen_integral_2*pdf_integral_1*pdf_integral_2^2 + knudsen_integral_2*pdf_integral_1^2*pdf_integral_3 - knudsen_integral_0*pdf_integral_1*pdf_integral3^2 - knudsen_integral_2*pdf_integral_1*pdf_integral_2^2 + knudsen_integral_0*pdf_integral_2^2*pdf_integral_3] = knudsen_integral_1*pdf_integral_3*pdf_integral_3 - knudsen_integral_2*pdf_integral_3*pdf_integral_2 - 3/2*knudsen_integral_1*pdf_integral_1*pdf_integral_3 + 3/2*knudsen_integral_2*pdf_integral_1*pdf_integral_2 - 3/2*knudsen_integral_2*pdf_integral_1*pdf_integral_2 + 3/2*knudsen_integral_0*pdf_integral_2*pdf_integral_3

                #   N_in*[knudsen_integral_0*(pdf_integral_2^2*pdf_integral_3 - pdf_integral_1*pdf_integral_3^2) + knudsen_integral_1*(pdf_integral_0*pdf_integral_3^2 - pdf_integral_1*pdf_integral_2*pdf_integral_3) + knudsen_integral_2(pdf_integral_1*pdf_integral_2^2 - pdf_integral_0*pdf_integral_2*pdf_integral_3 + pdf_integral_1^2*pdf_integral_3 - pdf_integral_1*pdf_integral_2^2)] = 3/2*knudsen_integral_0*pdf_integral_2*pdf_integral_3 + knudsen_integral_1*(pdf_integral_3*pdf_integral_3 - 3/2*pdf_integral_1*pdf_integral_3) + knudsen_integral_2*(3/2*pdf_integral_1*pdf_integral_2 - pdf_integral_3*pdf_integral_2 - 3/2*pdf_integral_1*pdf_integral_2)
                #   N_in*[knudsen_integral_0*(pdf_integral_2^2*pdf_integral_3 - pdf_integral_1*pdf_integral_3^2) + knudsen_integral_1*(pdf_integral_0*pdf_integral_3^2 - pdf_integral_1*pdf_integral_2*pdf_integral_3) + knudsen_integral_2(pdf_integral_1^2*pdf_integral_3 - pdf_integral_0*pdf_integral_2*pdf_integral_3)] = 3/2*knudsen_integral_0*pdf_integral_2*pdf_integral_3 + knudsen_integral_1*(pdf_integral_3*pdf_integral_3 - 3/2*pdf_integral_1*pdf_integral_3) - knudsen_integral_2*pdf_integral_3*pdf_integral_2
                #   N_in*[knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) + knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) + knudsen_integral_2(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2)] = 3/2*knudsen_integral_0*pdf_integral_2 + knudsen_integral_1*(pdf_integral_3 - 3/2*pdf_integral_1) - knudsen_integral_2*pdf_integral_2
                N_in = (1.5*knudsen_integral_0*pdf_integral_2 +
                        knudsen_integral_1*(pdf_integral_3 - 1.5*pdf_integral_1) -
                        knudsen_integral_2*pdf_integral_2) /
                       (knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) +
                        knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) +
                        knudsen_integral_2*(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2))
                N_out = -(N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2^2) + 1.5*pdf_integral_2) /
                         (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                C = (1.5 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)/pdf_integral_3

                zero_vz_ind = 0
                for ivz ∈ 1:vz.n
                    if vz.scratch2[ivz] <= -zero
                        @views @. pdf[ivz,:,:,1] = N_in*pdf[ivz,:,:,1] + C*vz.grid[ivz]*pdf[ivz,:,:,1]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,1] = 0.5*(N_in*pdf[ivz,:,:,1] +
                                                            C*vz.grid[ivz]*pdf[ivz,:,:,1] +
                                                            N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,1] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ zero_vz_ind+1:vz.n
                    @. pdf[ivz,:,:,1] = N_out*vz.scratch[ivz]
                end
            end
        end

        if z.irank == z.nrank - 1
            ## treat the right boundary at z = Lz/2 ##
            # populate vz.scratch2 array with dz/dt values at z = Lz/2
            if evolve_p
                vth = sqrt(2.0*pz[end]/density[end])
            else
                vth = nothing
            end
            @. vz.scratch2 = vpagrid_to_vpa(vz.grid, vth, uz[end], evolve_p, evolve_upar)

            # First apply boundary condition that total neutral outflux is equal to ion
            # influx to uz
            uz[end] = - wall_flux_L / density[end]
            #would setting density work better??
            #density[end] = - wall_flux_L / upar[end]

            # obtain the Knudsen cosine distribution at z = Lz/2
            # the z-dependence is only introduced if the peculiiar velocity is used as vz
            # Note this only makes sense for the 1V case with vr.n=vzeta.n=1
            @. vz.scratch = 3.0 * sqrt(π) * (0.5 / T_wall_over_m)^1.5 * abs(vz.scratch2) * erfc(sqrt(0.5 / T_wall_over_m) * abs(vz.scratch2))

            # The v_parallel<0 part of the pdf is replaced by the Knudsen cosine
            # distribution. To ensure the constraint ∫dwpa wpa F = 0 is satisfied, multiply
            # the Knudsen distribution (in vz.scratch) by a normalisation factor given by
            # the integral (over negative v_parallel) of the outgoing Knudsen distribution
            # and (over positive v_parallel) of the incoming pdf.
            knudsen_integral_0 = integrate_over_negative_vz(vz.scratch, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @. vz.scratch4 = vz.grid * vz.scratch
            knudsen_integral_1 = integrate_over_negative_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            @views pdf_integral_0 = integrate_over_positive_vz(pdf[:,:,:,end], vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
            @views @. pdf_buffer = vz.grid * pdf[:,:,:,end]
            pdf_integral_1 = integrate_over_positive_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)

            if !evolve_p
                # Calculate normalisation factors N_in for the incoming and N_out for the
                # Knudsen parts of the distirbution so that ∫dwpa F = 1 and ∫dwpa wpa F = 0
                # ⇒ N_in*pdf_integral_0 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + N_out*knudsen_integral_1 = 0
                N_in = 1.0 / (pdf_integral_0 - pdf_integral_1/knudsen_integral_1*knudsen_integral_0)
                N_out = -N_in * pdf_integral_1 / knudsen_integral_1

                zero_vz_ind = 0
                for ivz ∈ vz.n:-1:1
                    if vz.scratch2[ivz] >= zero
                        @views @. pdf[ivz,:,:,end] = N_in*pdf[ivz,:,:,end]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,end] = 0.5*(N_in*pdf[ivz,:,:,end] + N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ 1:zero_vz_ind-1
                    @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                end
            else
                @. vz.scratch4 = vz.grid * vz.grid * vz.scratch
                knudsen_integral_2 = integrate_over_negative_vz(vz.scratch4, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                @views @. pdf_buffer = vz.grid * vz.grid * pdf[:,:,:,end]
                pdf_integral_2 = integrate_over_positive_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                pdf_buffer .*= vz.grid
                pdf_integral_3 = integrate_over_positive_vz(pdf_buffer, vz.scratch2, vz.wgts, vz.scratch3, vr.grid, vr.wgts, vzeta.grid, vzeta.wgts)
                # Calculate normalisation factor N_out for the Knudsen part of the
                # distirbution and normalisation factor N_in and correction term C*wpa*F_in
                # for the incoming distribution so that ∫dwpa F = 1, ∫dwpa wpa F = 0, and
                # ∫dwpa wpa^2 F = 1/2
                # ⇒ N_in*pdf_integral_0 + C*pdf_integral_1 + N_out*knudsen_integral_0 = 1
                #   N_in*pdf_integral_1 + C*pdf_integral_2 + N_out*knudsen_integral_1 = 0
                #   N_in*pdf_integral_2 + C*pdf_integral_3 + N_out*knudsen_integral_2 = 3/2
                # ⇒
                #   C = (3/2 - N_out*knudsen_integral_2 - N_in*pdf_integral_2) / pdf_integral_3
                #   N_out*knudsen_integral_1 = - N_in*pdf_integral_1 - C*pdf_integral_2
                #   N_out*knudsen_integral_1*pdf_integral_3 = - N_in*pdf_integral_1*pdf_integral_3 - (3/2 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)*pdf_integral_2
                #   N_out*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) = -N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2) - 3/2*pdf_integral_2
                #   N_out = [-N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2) - 3/2*pdf_integral_2] / (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                #   N_in*pdf_integral_0 = 1 - C*pdf_integral_1 - N_out*knudsen_integral_0
                #   N_in*pdf_integral_0*pdf_integral_3 = pdf_integral_3 - (3/2 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)*pdf_integral_1 - N_out*knudsen_integral_0*pdf_integral_3
                #   N_in*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) = pdf_integral_3 - 3/2*pdf_integral_1 + N_out*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)
                #   N_in*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) = (pdf_integral_3 - 3/2*pdf_integral_1)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) + [-N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2) - 3/2*pdf_integral_2]*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)
                #   N_in*[(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) + (pdf_integral_1*pdf_integral_3 - pdf_integral_2*pdf_integral_2)*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)] = (pdf_integral_3 - 3/2*pdf_integral_1)*(knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2) - 3/2*pdf_integral_2*(knudsen_integral_2*pdf_integral_1 - knudsen_integral_0*pdf_integral_3)

                #   N_in*[knudsen_integral_1*pdf_integral_0*pdf_integral_3^2 - knudsen_integral_2*pdf_integral_0*pdf_integral_2*pdf_integral_3 - knudsen_integral_1*pdf_integral_1*pdf_integral_2*pdf_integral_3 + knudsen_integral_2*pdf_integral_1*pdf_integral_2^2 + knudsen_integral_2*pdf_integral_1^2*pdf_integral_3 - knudsen_integral_0*pdf_integral_1*pdf_integral3^2 - knudsen_integral_2*pdf_integral_1*pdf_integral_2^2 + knudsen_integral_0*pdf_integral_2^2*pdf_integral_3] = knudsen_integral_1*pdf_integral_3*pdf_integral_3 - knudsen_integral_2*pdf_integral_3*pdf_integral_2 - 3/2*knudsen_integral_1*pdf_integral_1*pdf_integral_3 + 3/2*knudsen_integral_2*pdf_integral_1*pdf_integral_2 - 3/2*knudsen_integral_2*pdf_integral_1*pdf_integral_2 + 3/2*knudsen_integral_0*pdf_integral_2*pdf_integral_3

                #   N_in*[knudsen_integral_0*(pdf_integral_2^2*pdf_integral_3 - pdf_integral_1*pdf_integral_3^2) + knudsen_integral_1*(pdf_integral_0*pdf_integral_3^2 - pdf_integral_1*pdf_integral_2*pdf_integral_3) + knudsen_integral_2(pdf_integral_1*pdf_integral_2^2 - pdf_integral_0*pdf_integral_2*pdf_integral_3 + pdf_integral_1^2*pdf_integral_3 - pdf_integral_1*pdf_integral_2^2)] = 3/2*knudsen_integral_0*pdf_integral_2*pdf_integral_3 + knudsen_integral_1*(pdf_integral_3*pdf_integral_3 - 3/2*pdf_integral_1*pdf_integral_3) + knudsen_integral_2*(3/2*pdf_integral_1*pdf_integral_2 - pdf_integral_3*pdf_integral_2 - 3/2*pdf_integral_1*pdf_integral_2)
                #   N_in*[knudsen_integral_0*(pdf_integral_2^2*pdf_integral_3 - pdf_integral_1*pdf_integral_3^2) + knudsen_integral_1*(pdf_integral_0*pdf_integral_3^2 - pdf_integral_1*pdf_integral_2*pdf_integral_3) + knudsen_integral_2(pdf_integral_1^2*pdf_integral_3 - pdf_integral_0*pdf_integral_2*pdf_integral_3)] = 3/2*knudsen_integral_0*pdf_integral_2*pdf_integral_3 + knudsen_integral_1*(pdf_integral_3*pdf_integral_3 - 3/2*pdf_integral_1*pdf_integral_3) - knudsen_integral_2*pdf_integral_3*pdf_integral_2
                #   N_in*[knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) + knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) + knudsen_integral_2(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2)] = 3/2*knudsen_integral_0*pdf_integral_2 + knudsen_integral_1*(pdf_integral_3 - 3/2*pdf_integral_1) - knudsen_integral_2*pdf_integral_2
                N_in = (1.5*knudsen_integral_0*pdf_integral_2 +
                        knudsen_integral_1*(pdf_integral_3 - 1.5*pdf_integral_1) -
                        knudsen_integral_2*pdf_integral_2) /
                       (knudsen_integral_0*(pdf_integral_2^2 - pdf_integral_1*pdf_integral_3) +
                        knudsen_integral_1*(pdf_integral_0*pdf_integral_3 - pdf_integral_1*pdf_integral_2) +
                        knudsen_integral_2*(pdf_integral_1^2 - pdf_integral_0*pdf_integral_2))
                N_out = -(N_in*(pdf_integral_1*pdf_integral_3 - pdf_integral_2^2) + 1.5*pdf_integral_2) /
                         (knudsen_integral_1*pdf_integral_3 - knudsen_integral_2*pdf_integral_2)
                C = (1.5 - N_out*knudsen_integral_2 - N_in*pdf_integral_2)/pdf_integral_3

                zero_vz_ind = 0
                for ivz ∈ vz.n:-1:1
                    if vz.scratch2[ivz] >= zero
                        @views @. pdf[ivz,:,:,end] = N_in*pdf[ivz,:,:,end] + C*vz.grid[ivz]*pdf[ivz,:,:,end]
                    else
                        zero_vz_ind = ivz
                        if abs(vz.scratch2[ivz]) < zero
                            # v_parallel = 0 point, half contribution from original pdf and half
                            # from Knudsen cosine distribution, to be consistent with weights
                            # used in
                            # integrate_over_positive_vz()/integrate_over_negative_vz().
                            @views @. pdf[ivz,:,:,end] = 0.5*(N_in*pdf[ivz,:,:,end] +
                                                              C*vz.grid[ivz]*pdf[ivz,:,:,end] +
                                                              N_out*vz.scratch[ivz])
                        else
                            @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                        end
                        break
                    end
                end
                for ivz ∈ 1:zero_vz_ind-1
                    @. pdf[ivz,:,:,end] = N_out*vz.scratch[ivz]
                end
            end
        end
    end
end

"""
create an array of v_∥ values corresponding to the given vpagrid values
"""
function vpagrid_to_vpa(vpagrid, vth, upar, evolve_p, evolve_upar)
    if evolve_p
        if evolve_upar
            return @. vpagrid * vth + upar
        else
            return @. vpagrid * vth
        end
    elseif evolve_upar
        return @. vpagrid + upar
    else
        return vpagrid
    end
end

"""
create an array of w_∥ values corresponding to the given vpa values
"""
function vpa_to_wpa(vpa, vth, upar, evolve_p, evolve_upar)
    if evolve_p
        if evolve_upar
            return @. (vpa - upar) / vth
        else
            return @. vpa / vth
        end
    elseif evolve_upar
        return @. vpa - upar
    else
        return vpa
    end
end

"""
enforce the z boundary condition on the evolved velocity space moments of f
"""
function enforce_z_boundary_condition_moments!(density, moments, bc::String)
    ## TODO: parallelise
    #@begin_serial_region()
    #@serial_region begin
    #    # enforce z boundary condition on density if it is evolved separately from f
    #	if moments.evolve_density
    #        # TODO: extend to 'periodic' BC case, as this requires further code modifications to be consistent
    #        # with finite difference derivatives (should be fine for Chebyshev)
    #        if bc == "wall"
    #            @loop_s_r is ir begin
    #                density[1,ir,is] = 0.5*(density[1,ir,is] + density[end,ir,is])
    #                density[end,ir,is] = density[1,ir,is]
    #        	end
    #        end
    #    end
    #end
end

"""
"""
function enforce_v_boundary_condition_local!(f, bc, speed, v_diffusion, v, v_spectral)
    if bc == "zero"
        if v_diffusion || speed[1] > 0.0
            # 'upwind' boundary
            f[1] = 0.0
        end
        if v_diffusion || speed[end] < 0.0
            # 'upwind' boundary
            f[end] = 0.0
        end
    elseif bc == "both_zero"
        f[1] = 0.0
        f[end] = 0.0
    elseif bc == "zero_gradient"
        D0 = v_spectral.lobatto.Dmat[1,:]
        # adjust F(vpa = -L/2) so that d F / d vpa = 0 at vpa = -L/2
        f[1] = -sum(D0[2:v.ngrid].*f[2:v.ngrid])/D0[1]

        D0 = v_spectral.lobatto.Dmat[end,:]
        # adjust F(vpa = L/2) so that d F / d vpa = 0 at vpa = L/2
        f[end] = -sum(D0[1:v.ngrid-1].*f[end-v.ngrid+1:end-1])/D0[v.ngrid]
    elseif bc == "periodic"
        f[1] = 0.5*(f[1]+f[end])
        f[end] = f[1]
    elseif bc == "none"
        # Do nothing
    else
        error("Unsupported boundary condition option '$bc' for $(v.name)")
    end
    return nothing
end

"""
enforce zero boundary condition at vperp -> infinity
"""
function enforce_vperp_boundary_condition! end

function enforce_vperp_boundary_condition!(f::AbstractArray{mk_float,5}, bc, vperp, vperp_spectral, vperp_advect, diffusion)
    @loop_s is begin
        @views enforce_vperp_boundary_condition!(f[:,:,:,:,is], bc, vperp, vperp_spectral, vperp_advect[is], diffusion)
    end
    return nothing
end

function enforce_vperp_boundary_condition!(f::AbstractArray{mk_float,4}, bc, vperp, vperp_spectral, vperp_advect, diffusion)
    @loop_r ir begin
        @views enforce_vperp_boundary_condition!(f[:,:,:,ir], bc, vperp, vperp_spectral,
                                                 vperp_advect, diffusion, ir)
    end
    return nothing
end

function enforce_vperp_boundary_condition!(f::AbstractArray{mk_float,3}, bc, vperp,
                                           vperp_spectral, vperp_advect, diffusion, ir)
    if bc == "zero" || bc == "zero-impose-regularity"
        nvperp = vperp.n
        ngrid = vperp.ngrid
        # set zero boundary condition
        @loop_z_vpa iz ivpa begin
            if diffusion || vperp_advect.speed[nvperp,ivpa,iz,ir] < 0.0
                f[ivpa,nvperp,iz] = 0.0
            end
        end
        # set regularity condition d F / d vperp = 0 at vperp = 0
        if bc == "zero-impose-regularity" && (vperp.discretization == "gausslegendre_pseudospectral" || vperp.discretization == "chebyshev_pseudospectral")
            D0 = vperp_spectral.radau.D0
            buffer = @view vperp.scratch[1:ngrid-1]
            @loop_z_vpa iz ivpa begin
                if diffusion || vperp_advect.speed[1,ivpa,iz,ir] > 0.0
                    # adjust F(vperp = 0) so that d F / d vperp = 0 at vperp = 0
                    @views @. buffer = D0[2:ngrid] * f[ivpa,2:ngrid,iz]
                    f[ivpa,1,iz] = -sum(buffer)/D0[1]
                end
            end
        elseif bc == "zero"
            # do nothing
        else
            println("vperp.bc=\"$bc\" not supported by discretization "
                    * "$(vperp.discretization)")
        end
    elseif bc == "none"
        # Do nothing
    else
        error("Unsupported boundary condition option '$bc' for vperp")
    end
end

"""
    skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa)

This function returns `true` when the grid point specified by `iz`, `ivperp`, `ivpa` would
be set by the boundary conditions on the electron distribution function. When this
happens, the corresponding row should be skipped when adding contributions to the Jacobian
matrix, so that the row remains the same as a row of the identity matrix, so that the
Jacobian matrix does not modify those points. Returns `false` otherwise.
"""
function skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
    # z boundary condition
    # Treat as if using Dirichlet boundary condition for incoming part of the distribution
    # function on the block boundary, regardless of the actual boundary condition and
    # whether this is an internal boundary or an actual domain boundary. This prevents the
    # matrix evaluated for a single block (without coupling to neighbouring blocks) from
    # becoming singular
    if iz == 1 && z_speed[iz,ivpa,ivperp] ≥ 0.0
        return true
    end
    if iz == z.n && z_speed[iz,ivpa,ivperp] ≤ 0.0
        return true
    end

    # vperp boundary condition
    if vperp.n > 1 && ivperp == vperp.n
        return true
    end

    if ivpa == 1 || ivpa == vpa.n
        return true
    end

    return false
end

end # boundary_conditions
