module initial_conditions

export init_f
export enforce_z_boundary_condition!
export enforce_vpa_boundary_condition!

using type_definitions: mk_float
using array_allocation: allocate_float
using moment_kinetics_input: z_initialization_option, vpa_initialization_option
using moment_kinetics_input: z_width, vpa_width
using moment_kinetics_input: z_monomial_degree, vpa_monomial_degree
using moment_kinetics_input: z_amplitude, z_wavenumber, vpa_amplitude, vpa_wavenumber
using moment_kinetics_input: density_offset

# creates f and specifies its initial condition
# all initial conditions are of the form f = F(z)*G(vpa)
function init_f(z, vpa)
    f = allocate_float(z.n, vpa.n)
    f_scratch = allocate_float(z.n, vpa.n, 3)
    # initialize F(z) and return in z.scratch
    init_fz(z, z_initialization_option)
    # initialize G(vpa) and return in vpa.scratch
    init_fvpa(vpa, vpa_initialization_option)
    # calculate f = F(z)*G(vpa)
    for ivpa ∈ 1:vpa.n
        for iz ∈ 1:z.n
            f[iz,ivpa] = z.scratch[iz]*vpa.scratch[ivpa]
        end
    end
    return f, f_scratch
end
# init_fz iniitializes F(z)
function init_fz(z, init_option)
    @inbounds begin
        if init_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for i ∈ 1:z.n
                z.scratch[i] = density_offset + exp(-(z.grid[i]/z_width)^2)
            end
        elseif init_option == "sinusoid"
            # initial condition is sinusoid in z
            for i ∈ 1:z.n
                z.scratch[i] = density_offset + z_amplitude*cospi(2.0*z_wavenumber*z.grid[i]/z.L)
            end
        elseif init_option == "monomial"
            # linear variation in z, with offset so that
            # function passes through zero at upwind boundary
            for i ∈ 1:z.n
                z.scratch[i] = (z.grid[i] + 0.5*z.L)^z_monomial_degree
            end
        end
    end
    return nothing
end
# init_fvpa initializes G(vpa)
function init_fvpa(vpa, init_option)
    @inbounds begin
        if init_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for j ∈ 1:vpa.n
                vpa.scratch[j] = exp(-(vpa.grid[j]/vpa_width)^2)
            end
        elseif initialization_option == "sinusoid"
            # initial condition is sinusoid in vpa
            for j ∈ 1:vpa.n
                vpa.scratch[j] = vpa_amplitude*cospi(2.0*vpa_wavenumber*vpa.grid[j]/vpa.L)
            end
        elseif initialization_option == "monomial"
            # linear variation in vpa, with offset so that
            # function passes through zero at upwind boundary
            for j ∈ 1:z.n
                vpa.scratch[j] = (vpa.grid[j] + 0.5*vpa.L)^vpa_monomial_degree
            end
        end
    end
    return nothing
end
# impose the prescribed z boundary condition on f
# at every vpa grid point
function enforce_z_boundary_condition!(f::Array{mk_float,2}, bc::String, vpa, src::T) where T
    for ivpa ∈ 1:size(src,1)
        enforce_z_boundary_condition!(view(f,:,ivpa), bc,
            src[ivpa].upwind_idx, src[ivpa].downwind_idx, vpa.grid[ivpa])
    end
end
# impose the prescribed z boundary conditin on f
# at a single vpa grid point
function enforce_z_boundary_condition!(f, bc, upwind_idx, downwind_idx, v)
    if bc == "constant"
        # BC is time-independent f at upwind boundary
        # and constant f beyond boundary
        f[upwind_idx] = density_offset * exp(-(v/vpawidth)^2) / sqrt(pi)
    elseif bc == "periodic"
        # impose periodicity
        f[downwind_idx] = f[upwind_idx]
        #f[upwind_idx] = f[downwind_idx]
    end
end
# impose the prescribed vpa boundary condition on f
# at every z grid point
function enforce_vpa_boundary_condition!(f, bc, src::T) where T
    nz = size(f,1)
    for iz ∈ 1:nz
        enforce_vpa_boundary_condition_local!(view(f,iz,:), bc, src[iz].upwind_idx,
            src[iz].downwind_idx)
    end
end
function enforce_vpa_boundary_condition_local!(f::T, bc, upwind_idx, downwind_idx) where T
    if bc == "zero"
        f[upwind_idx] = 0.0
    elseif bc == "periodic"
        f[downwind_idx] = f[upwind_idx]
    end
end

end
