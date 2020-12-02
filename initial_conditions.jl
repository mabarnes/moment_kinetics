module initial_conditions

export init_f
export enforce_z_boundary_condition!
export enforce_vpa_boundary_condition!

using type_definitions: mk_float
using array_allocation: allocate_float
using moment_kinetics_input: initialization_option
using moment_kinetics_input: monomial_degree
using moment_kinetics_input: zwidth, vpawidth
using moment_kinetics_input: density_offset

# creates ff and specifies its initial condition
function init_f(z, vpa)
    f = allocate_float(z.n, vpa.n)
    f_scratch = allocate_float(z.n, vpa.n, 3)
    @inbounds begin
        if initialization_option == "gaussian"
            # initial condition is an unshifted Gaussian
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    f[i,j] = ((density_offset + exp(-(z.grid[i]/zwidth)^2))
                     * exp(-(vpa.grid[j]/vpawidth)^2) / sqrt(pi))
                end
            end
        elseif initialization_option == "monomial"
            # linear variation in z, with offset so that
            # function passes through zero at upwind boundary
            for i ∈ 1:z.n
                f[i,j] = ((z.grid[i] + 0.5*z.L)^monomial_degree
                    * (vpa.grid[j] + 0.5*vpa.L)^monomial_degree)
            end
        end
        #enforce_vpa_boundary_condition!(f, vpa.bc)
        #if vpa.bc == "zero"
        #    #f[:,1] .= 0.0
        #    f[:,end] .= 0.0
        #end
    end
    return f, f_scratch
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
