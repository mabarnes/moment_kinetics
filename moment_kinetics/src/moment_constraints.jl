"""
Functions for enforcing integral constraints on the normalised distribution function.
Ensures consistency of evolution split into moments and normalised distribution
function.
"""
module moment_constraints

using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..communication: _block_synchronize
using ..looping
using ..timer_utils
using ..type_definitions: mk_float
using ..velocity_moments: integrate_over_vspace, update_ion_qpar!

export hard_force_moment_constraints!, hard_force_moment_constraints_neutral!,
       electron_implicit_constraint_forcing!,
       add_electron_implicit_constraint_forcing_to_Jacobian!

"""
    hard_force_moment_constraints!(f, moments, vpa)

Force the moment constraints needed for the system being evolved to be applied to `f`.
Not guaranteed to be a small correction, if `f` does not approximately obey the
constraints to start with, but can be useful at initialisation to ensure a consistent
initial state, and when applying boundary conditions.

Note this function assumes the input is given at a single spatial position.
"""
function hard_force_moment_constraints!(f, moments, vpa)
    #if moments.evolve_ppar
    #    I0 = integrate_over_vspace(f, vpa.wgts)
    #    I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
    #    I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
    #    I3 = integrate_over_vspace(f, vpa.grid, 3, vpa.wgts)
    #    I4 = integrate_over_vspace(f, vpa.grid, 4, vpa.wgts)
    #    A = ((1.0 - 0.5*I2/I4)*(I2 - I3^2/I4) + 0.5*I3/I4*(I1-I2*I3/I4)) /
    #        ((I0 - I2^2/I4)*(I2 - I3^2/I4) - (I1 - I2*I3/I4)^2)
    #    B = -(0.5*I3/I4 + A*(I1 - I2*I3/I4)) / (I2 - I3^2/I4)
    #    C = -(A*I1 + B*I2) / I3
    #    @. f = A*f + B*vpa.grid*f + C*vpa.grid*vpa.grid*f
    #elseif moments.evolve_upar
    #    I0 = integrate_over_vspace(f, vpa.wgts)
    #    I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
    #    I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
    #    A = 1.0 / (I0 + I1*I1/I2)
    #    B = -I1*A/I2
    #    @. f = A*f + B*vpa.grid*f
    #elseif moments.evolve_density
    #    I0 = integrate_over_vspace(f, vpa.wgts)
    #    @. f = f / I0
    #end

    f1d = @view f[:,1]
    if moments.evolve_ppar
        I0 = integrate_over_vspace(f1d, vpa.wgts)
        I1 = integrate_over_vspace(f1d, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f1d, vpa.grid, 2, vpa.wgts)
        I3 = integrate_over_vspace(f1d, vpa.grid, 3, vpa.wgts)
        I4 = integrate_over_vspace(f1d, vpa.grid, 4, vpa.wgts)

        A = (I3^2 - I2*I4 + 0.5*(I2^2 - I1*I3)) /
            (I0*(I3^2 - I2*I4) + I1*I1*I4 - 2.0*I1*I2*I3 + I2^3)
        B = (0.5*I3 + A*(I1*I4 - I2*I3)) / (I3^2 - I2*I4)
        C = (0.5 - A*I2 -B*I3) / I4

        @. f1d = A*f1d + B*vpa.grid*f1d + C*vpa.grid*vpa.grid*f1d
    elseif moments.evolve_upar
        I0 = integrate_over_vspace(f1d, vpa.wgts)
        I1 = integrate_over_vspace(f1d, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f1d, vpa.grid, 2, vpa.wgts)

        A = 1.0 / (I0 - I1^2/I2)
        B = -A*I1/I2

        @. f1d = A*f1d + B*vpa.grid*f1d

        C = NaN
    elseif moments.evolve_density
        I0 = integrate_over_vspace(f1d, vpa.wgts)
        A = 1.0 / I0
        @. f1d = A * f1d

        B = NaN
        C = NaN
    else
        A = NaN
        B = NaN
        C = NaN
    end

    return A, B, C
end
@timeit global_timer hard_force_moment_constraints!(
                         f::AbstractArray{mk_float,4}, moments, vpa) = begin
    A = moments.electron.constraints_A_coefficient
    B = moments.electron.constraints_B_coefficient
    C = moments.electron.constraints_C_coefficient
    begin_r_z_region()
    @loop_r_z ir iz begin
        A[iz,ir], B[iz,ir], C[iz,ir] =
            hard_force_moment_constraints!(@view(f[:,:,iz,ir]), moments, vpa)
    end
end
@timeit global_timer hard_force_moment_constraints!(
                         f::AbstractArray{mk_float,5}, moments, vpa) = begin
    A = moments.ion.constraints_A_coefficient
    B = moments.ion.constraints_B_coefficient
    C = moments.ion.constraints_C_coefficient
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        A[iz,ir,is], B[iz,ir,is], C[iz,ir,is] =
            hard_force_moment_constraints!(@view(f[:,:,iz,ir,is]), moments, vpa)
    end
end

"""
    hard_force_moment_constraints_neutral!(f, moments, vz)

Force the moment constraints needed for the system being evolved to be applied to `f`.
Not guaranteed to be a small correction, if `f` does not approximately obey the
constraints to start with, but can be useful at initialisation to ensure a consistent
initial state, and when applying boundary conditions.

Notes:
* this function assumes the input is given at a single spatial position.
* currently only works with '1V' runs, where vz is the only velocity-space dimension
"""
function hard_force_moment_constraints_neutral!(f, moments, vz)
    f1d = @view f[:,1,1]
    if moments.evolve_ppar
        I0 = integrate_over_vspace(f1d, vz.wgts)
        I1 = integrate_over_vspace(f1d, vz.grid, vz.wgts)
        I2 = integrate_over_vspace(f1d, vz.grid, 2, vz.wgts)
        I3 = integrate_over_vspace(f1d, vz.grid, 3, vz.wgts)
        I4 = integrate_over_vspace(f1d, vz.grid, 4, vz.wgts)

        A = (I3^2 - I2*I4 + 0.5*(I2^2 - I1*I3)) /
            (I0*(I3^2 - I2*I4) + I1*I1*I4 - 2.0*I1*I2*I3 + I2^3)
        B = (0.5*I3 + A*(I1*I4 - I2*I3)) / (I3^2 - I2*I4)
        C = (0.5 - A*I2 -B*I3) / I4

        @. f1d = A*f1d + B*vz.grid*f1d + C*vz.grid*vz.grid*f1d
    elseif moments.evolve_upar
        I0 = integrate_over_vspace(f1d, vz.wgts)
        I1 = integrate_over_vspace(f1d, vz.grid, vz.wgts)
        I2 = integrate_over_vspace(f1d, vz.grid, 2, vz.wgts)

        A = 1.0 / (I0 - I1^2/I2)
        B = -A*I1/I2

        @. f1d = A*f1d + B*vz.grid*f1d

        C = NaN
    elseif moments.evolve_density
        I0 = integrate_over_vspace(f1d, vz.wgts)
        A = 1.0 / I0
        @. f1d = A * f1d

        B = NaN
        C = NaN
    else
        A = NaN
        B = NaN
        C = NaN
    end

    return A, B, C
end
@timeit global_timer hard_force_moment_constraints_neutral!(
                         f::AbstractArray{mk_float,6}, moments, vz) = begin
    A = moments.neutral.constraints_A_coefficient
    B = moments.neutral.constraints_B_coefficient
    C = moments.neutral.constraints_C_coefficient
    begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        A[iz,ir,isn], B[iz,ir,isn], C[iz,ir,isn] =
            hard_force_moment_constraints_neutral!(@view(f[:,:,:,iz,ir,is]), moments, vz)
    end
end

"""
    moment_constraints_on_residual!(residual, f, moments, vpa)

A 'residual' (used in implicit timestepping) is an update to the distribution function
\$f_\\mathrm{new} = f_\\mathrm{old} + \\mathtt{residual}\$. \$f_\\mathrm{new}\$ should
obey the moment constraints ([Constraints on normalized distribution function](@ref)), and
\$f_\\mathrm{old}\$ already obeys the constraints, which means that the first 3 moments of
`residual` should be zero. We impose this constraint by adding corrections proportional to
`f`.
```math
r = \\hat{r} + (A + B w_{\\|} + C w_{\\|}^2) f
```

Note this function assumes the input is given at a single spatial position.
"""
function moment_constraints_on_residual!(residual::AbstractArray{T,N},
                                         f::AbstractArray{T,N}, moments, vpa) where {T,N}
    if N == 2
        f = @view f[:,1]
        residual = @view residual[:,1]
    end
    if moments.evolve_ppar
        I0 = integrate_over_vspace(f, vpa.wgts)
        I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
        I3 = integrate_over_vspace(f, vpa.grid, 3, vpa.wgts)
        I4 = integrate_over_vspace(f, vpa.grid, 4, vpa.wgts)
        J0 = integrate_over_vspace(residual, vpa.wgts)
        J1 = integrate_over_vspace(residual, vpa.grid, vpa.wgts)
        J2 = integrate_over_vspace(residual, vpa.grid, 2, vpa.wgts)

        A = ((I2*J2 - J0*I4)*(I2*I4 - I3^2) + (I2*I3 - I1*I4)*(J2*I3 - J1*I4)) /
            ((I0*I4 - I2^2)*(I2*I4 - I3^2) - (I2*I3 - I1*I4)^2)
        B = (J2*I3 - J1*I4 + (I2*I3 - I1*I4)*A) / (I2*I4 - I3^2)
        C = -(J2 + I2*A + I3*B) / I4

        @. residual = residual + (A + B*vpa.grid + C*vpa.grid*vpa.grid) * f
    elseif moments.evolve_upar
        I0 = integrate_over_vspace(f, vpa.wgts)
        I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
        J0 = integrate_over_vspace(residual, vpa.wgts)
        J1 = integrate_over_vspace(residual, vpa.grid, vpa.wgts)

        A = (I1*J1 - J0*I2) / (I0*I2 - I1^2)
        B = -(J1 + I1*A) / I2

        @. residual = residual + (A + B*vpa.grid) * f

        C = NaN
    elseif moments.evolve_density
        I0 = integrate_over_vspace(f, vpa.wgts)
        J0 = integrate_over_vspace(residual, vpa.wgts)
        A = -J0 / I0
        @. f = A * f
        @. residual = residual + A * f

        B = NaN
        C = NaN
    else
        A = NaN
        B = NaN
        C = NaN
    end

    return A, B, C
end

"""
    electron_implicit_constraint_forcing!(f_out, f_in, constraint_forcing_rate, vpa,
                                          dt, ir)

Add terms to the electron kinetic equation that force the moment constraints to be
approximately satisfied. Needed to avoid large errors when taking large, implicit
timesteps that do not guarantee accurate time evolution.
"""
@timeit global_timer electron_implicit_constraint_forcing!(
                         f_out, f_in, constraint_forcing_rate, vpa, dt, ir) = begin
    begin_z_region()
    vpa_grid = vpa.grid
    @loop_z iz begin
        @views zeroth_moment = integrate_over_vspace(f_in[:,1,iz], vpa.wgts)
        @views first_moment = integrate_over_vspace(f_in[:,1,iz], vpa.grid, vpa.wgts)
        @views second_moment = integrate_over_vspace(f_in[:,1,iz], vpa.grid, 2, vpa.wgts)

        @loop_vperp_vpa ivperp ivpa begin
            f_out[ivpa,ivperp,iz] +=
                dt * constraint_forcing_rate *
                ((1.0 - zeroth_moment)
                 - first_moment*vpa_grid[ivpa]
                 + (0.5 - second_moment)*vpa_grid[ivpa]^2) * f_in[ivpa,ivperp,iz]
        end
    end

    return nothing
end

"""
    add_electron_implicit_constraint_forcing_to_Jacobian!(jacobian_matrix, f,
                                                          z_speed, z, vperp, vpa,
                                                          constraint_forcing_rate,
                                                          dt, ir; f_offset=0)

Add the contributions corresponding to [`electron_implicit_constraint_forcing!`](@ref) to
`jacobian_matrix`.
"""
function add_electron_implicit_constraint_forcing_to_Jacobian!(jacobian_matrix, f,
                                                               z_speed, z, vperp, vpa,
                                                               constraint_forcing_rate,
                                                               dt, ir; f_offset=0)
    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts
    v_size = vperp.n * vpa.n

    zeroth_moment = z.scratch_shared
    first_moment = z.scratch_shared2
    second_moment = z.scratch_shared3
    begin_z_region()
    @loop_z iz begin
        @views zeroth_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_wgts)
        @views first_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_grid, vpa_wgts)
        @views second_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_grid, 2, vpa_wgts)
    end

    begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

        # Diagonal terms
        jacobian_matrix[row,row] += -dt * constraint_forcing_rate *
                                          ((1.0 - zeroth_moment[iz])
                                           - first_moment[iz]*vpa_grid[ivpa]
                                           + (0.5 - second_moment[iz])*vpa_grid[ivpa]^2)

        # Integral terms
        # d(∫dw_∥ w_∥^n g[irow])/d(g[icol]) = vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^n
        for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
            col = (iz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
            jacobian_matrix[row,col] += dt * constraint_forcing_rate *
                                             (1.0
                                              + vpa_grid[icolvpa]*vpa_grid[ivpa]
                                              + vpa_grid[icolvpa]^2*vpa_grid[ivpa]^2) *
                                             vpa_wgts[icolvpa]/sqrt(π) * f[ivpa,ivperp,iz]
        end
    end

    return nothing
end

end
