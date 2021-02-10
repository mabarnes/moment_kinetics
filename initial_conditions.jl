module initial_conditions

export init_f
export enforce_z_boundary_condition!
export enforce_vpa_boundary_condition!

using type_definitions: mk_float
using array_allocation: allocate_float

# creates f and specifies its initial condition
# all initial conditions are of the form f = F(z)*G(vpa)
function init_f(z, vpa, composition, species, n_rk_stages)
    n_species = composition.n_species
    f = allocate_float(z.n, vpa.n, n_species)
    f_scratch = allocate_float(z.n, vpa.n, n_species, n_rk_stages+1)
    for is ∈ 1:n_species
        # initialize F(z) and return in z.scratch
        init_fz(z, species[is])
        # initialize G(vpa) and return in vpa.scratch
        init_fvpa(vpa, species[is])
        # calculate f = F(z)*G(vpa)
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                f[iz,ivpa,is] = z.scratch[iz]*vpa.scratch[ivpa]
            end
        end
    end
    return f, f_scratch
end
# init_fz iniitializes F(z)
function init_fz(z, spec)
    @inbounds begin
        if spec.z_IC.initialization_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for i ∈ 1:z.n
                z.scratch[i] = spec.initial_density + exp(-(z.grid[i]/spec.z_IC.width)^2)
            end
        elseif spec.z_IC.initialization_option == "sinusoid"
            # initial condition is sinusoid in z
            for i ∈ 1:z.n
                z.scratch[i] = spec.initial_density*(1.0 + spec.z_IC.amplitude
                    *cospi(2.0*spec.z_IC.wavenumber*z.grid[i]/z.L))
            end
        elseif spec.z_IC.inititalization_option == "monomial"
            # linear variation in z, with offset so that
            # function passes through zero at upwind boundary
            for i ∈ 1:z.n
                z.scratch[i] = (z.grid[i] + 0.5*z.L)^spec.z_IC.monomial_degree
            end
        end
    end
    return nothing
end
# init_fvpa initializes G(vpa)
function init_fvpa(vpa, spec)
    @inbounds begin
        if spec.vpa_IC.initialization_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for j ∈ 1:vpa.n
                vpa.scratch[j] = exp(-(vpa.grid[j]/spec.vpa_IC.width)^2/spec.initial_temperature)
            end
        elseif spec.vpa_IC.initialization_option == "sinusoid"
            # initial condition is sinusoid in vpa
            for j ∈ 1:vpa.n
                vpa.scratch[j] = spec.vpa_IC.amplitude*cospi(2.0*spec.vpa_IC.wavenumber*vpa.grid[j]/vpa.L)
            end
        elseif spec.vpa_IC.initialization_option == "monomial"
            # linear variation in vpa, with offset so that
            # function passes through zero at upwind boundary
            for j ∈ 1:z.n
                vpa.scratch[j] = (vpa.grid[j] + 0.5*vpa.L)^spec.vpa_IC.monomial_degree
            end
        end
    end
    return nothing
end
# impose the prescribed z boundary condition on f
# at every vpa grid point
function enforce_z_boundary_condition!(f, bc::String, vpa, src::T) where T
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
