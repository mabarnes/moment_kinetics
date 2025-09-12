"""
Functions for enforcing integral constraints on the normalised distribution function.
Ensures consistency of evolution split into moments and normalised distribution
function.
"""
module moment_constraints

using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..calculus: integral
using ..debugging
using ..looping
using ..timer_utils
using ..type_definitions: mk_float

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
function hard_force_moment_constraints!(f, moments, vpa, vperp)
    if moments.evolve_p
        # fnew = (A + B*wpa + C*(wpa^2 + wperp^2))*f
        # Constraints:
        #   1 = ∫fnew d^3w
        #   0 = ∫wpa*fnew d^3w
        #   3/2 = ∫(wpa^2 + wperp^2)*fnew d^3w
        #
        # Define 
        # I0 = ∫f dwpa
        # I1 = ∫wpa*f dwpa
        # I2 = ∫(wpa^2 + wperp^2)*f dwpa
        # I3 = ∫(wpa^2 + wperp^2)*wpa*f dwpa
        # I4 = ∫(wpa^2 + wperp^2)^2*f dwpa
        # I5 = ∫wpa^2 * f dwpa

        # gives 3 simultaneous equations
        #   A*I0 + B*I1 + C*I2 = 1
        #   A*I1 + B*I5 + C*I3 = 0
        #   A*I2 + B*I3 + C*I4 = 3/2
        # 
        # Using an inverted matrix to solve for A, B, C, the result is:
        #
        # determinant = I0*I5*I4 - I0*I3^2 - I1^2*I4 + 2*I1*I2*I3 - I2^2*I5
        # A = (I5*I4 - I3^2 + 1.5*(I1*I3 - I2*I5)) / determinant
        # B = (I3*I2 - I1*I4 + 1.5*(I1*I2 - I0*I3)) / determinant
        # C = (I1*I3 - I2*I5 + 1.5*(I0*I5 - I1^2)) / determinant

        I0 = integral((vperp,vpa)->(1), @view(f[:,:]), vperp, vpa)
        I1 = integral((vperp,vpa)->(vpa), @view(f[:,:]), vperp, vpa)
        I2 = integral((vperp,vpa)->(vpa^2 + vperp^2), @view(f[:,:]), vperp, vpa)
        I3 = integral((vperp,vpa)->((vpa^2 + vperp^2)*vpa), @view(f[:,:]), vperp, vpa)
        I4 = integral((vperp,vpa)->((vpa^2 + vperp^2)^2), @view(f[:,:]), vperp, vpa)
        I5 = integral((vperp,vpa)->(vpa^2), @view(f[:,:]), vperp, vpa)

        determinant = I0*I5*I4 - I0*I3^2 - I1^2*I4 + 2*I1*I2*I3 - I2^2*I5
        A = (I5*I4 - I3^2 + 1.5*(I1*I3 - I2*I5)) / determinant
        B = (I3*I2 - I1*I4 + 1.5*(I1*I2 - I0*I3)) / determinant
        C = (I1*I3 - I2*I5 + 1.5*(I0*I5 - I1^2)) / determinant
        @. f = (A + B*vpa.grid + C*vpa.grid*vpa.grid)*f
    elseif moments.evolve_upar
        # fnew = (A + B*wpa)*f
        # Constraints:
        #   1 = ∫fnew d^3w
        #   0 = ∫wpa*fnew d^3w
        #
        # Define 
        # I0 = ∫f dwpa
        # I1 = ∫wpa*f dwpa
        # I2 = ∫wpa^2 * f dwpa
        # (note that I2 is not the same in the evolve_p = true case, where it is instead I5)
        #
        # gives 2 simultaneous equations
        #   A*I0 + B*I1 = 1
        #   A*I1 + B*I2 = 0
        # 
        # Using an inverted matrix to solve for A, B, the result is:
        #
        # determinant = I0*I2 - I1^2
        # A = I2 / determinant
        # B = -I1 / determinant
        
        I0 = integral((vperp,vpa)->(1), @view(f[:,:]), vperp, vpa)
        I1 = integral((vperp,vpa)->(vpa), @view(f[:,:]), vperp, vpa)
        I2 = integral((vperp,vpa)->(vpa^2), @view(f[:,:]), vperp, vpa)
        determinant = I0*I2 - I1^2
        A = I2 / determinant
        B = -I1 / determinant

        @. f = A*f + B*vpa.grid*f

        C = NaN
    elseif moments.evolve_density
        I0 = integral((vperp,vpa)->(1), @view(f[:,:]), vperp, vpa)
        A = 1.0 / I0
        @. f = A * f

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
                         f::AbstractArray{mk_float,4}, moments, vpa, vperp) = begin
    A = moments.electron.constraints_A_coefficient
    B = moments.electron.constraints_B_coefficient
    C = moments.electron.constraints_C_coefficient
    @begin_r_z_region()
    @loop_r_z ir iz begin
        A[iz,ir], B[iz,ir], C[iz,ir] =
            hard_force_moment_constraints!(@view(f[:,:,iz,ir]), moments, vpa, vperp)
    end
end
@timeit global_timer hard_force_moment_constraints!(
                         f::AbstractArray{mk_float,5}, moments, vpa, vperp) = begin
    A = moments.ion.constraints_A_coefficient
    B = moments.ion.constraints_B_coefficient
    C = moments.ion.constraints_C_coefficient
    @begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        A[iz,ir,is], B[iz,ir,is], C[iz,ir,is] =
            hard_force_moment_constraints!(@view(f[:,:,iz,ir,is]), moments, vpa, vperp)
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
    if moments.evolve_p
        # fnew = (A + B*wz + C*wz^2)*f
        # Constraints:
        #   1 = ∫fnew dwpa
        #   0 = ∫wz*fnew dwpa
        #   3/2 = ∫wz^2*fnew dwpa
        #
        # Define In = ∫wz^n*f dwpa
        # gives 3 simultaneous equations
        #   1 = A*I0 + B*I1 + C*I2
        #   0 = A*I1 + B*I2 + C*I3
        #   3/2 = A*I2 + B*I3 + C*I4
        # which we can solve for
        #   C = (3/2 - A*I2 - B*I3) / I4
        #
        #   B*I2 = -A*I1 - C*I3
        #        = -A*I1 - (3/2 - A*I2 - B*I3)/I4 * I3
        #   B = (3/2*I3 + A*(I1*I4 - I2*I3)) / (I3^2 - I2*I4)
        #
        #   A*I0 = 1 - B*I1 - C*I2
        #        = 1 - B*I1 - (3/2 - A*I2 - B*I3) / I4 * I2
        #   A*I0*I4 = I4 - B*I1*I4 - 3/2*I2 + A*I2^2 + B*I3*I2
        #   A*(I0*I4 - I2^2) = I4 - 3/2*I2 + B*(I2*I3 - I1*I4)
        #   A*(I0*I4 - I2^2) = I4 - 3/2*I2 + (3/2*I3 + A*(I1*I4 - I2*I3))*(I2*I3 - I1*I4) / (I3^2 - I2*I4)
        #   A*(I0*I4 - I2^2)*(I3^2 - I2*I4) = (I4 - 3/2*I2)*(I3^2 - I2*I4) + (3/2*I3 + A*(I1*I4 - I2*I3))*(I2*I3 - I1*I4)
        #   A*((I0*I4 - I2^2)*(I3^2 - I2*I4) - (I1*I4 - I2*I3)*(I2*I3 - I1*I4) = (I4 - 3/2*I2)*(I3^2 - I2*I4) + 3/2*I3*(I2*I3 - I1*I4)
        #   A*(I0*I3^2*I4 - I0*I2*I4^2 - I2^2*I3^2 + I2^3*I4 - I1*I2*I3*I4 + I1^2*I4^2 + I2^2*I3^2 - I1*I2*I3*I4) = I3^2*I4 - I2*I4^2 - 3/2*I2*I3^2 + 3/2*I2^2*I4 + 3/2*I2*I3^2 - 3/2*I1*I3*I4
        #   A*(I0*I3^2*I4 - I0*I2*I4^2 + I2^3*I4 - 2*I1*I2*I3*I4 + I1^2*I4^2) = I3^2*I4 - I2*I4^2 + 3/2*I2^2*I4 - 3/2*I1*I3*I4
        #   A*(I0*I3^2 - I0*I2*I4 + I2^3 - 2*I1*I2*I3 + I1^2*I4) = I3^2 - I2*I4 + 3/2*I2^2 - 3/2*I1*I3

        I0 = integral(f1d, vz.wgts)
        I1 = integral(f1d, vz.grid, vz.wgts)
        I2 = integral(f1d, vz.grid, 2, vz.wgts)
        I3 = integral(f1d, vz.grid, 3, vz.wgts)
        I4 = integral(f1d, vz.grid, 4, vz.wgts)

        A = (I3^2 - I2*I4 + 1.5*(I2^2 - I1*I3)) /
            (I0*(I3^2 - I2*I4) + I1*I1*I4 - 2.0*I1*I2*I3 + I2^3)
        B = (1.5*I3 + A*(I1*I4 - I2*I3)) / (I3^2 - I2*I4)
        C = (1.5 - A*I2 -B*I3) / I4

        @. f1d = A*f1d + B*vz.grid*f1d + C*vz.grid*vz.grid*f1d
    elseif moments.evolve_upar
        # fnew = (A + B*wz)*f
        # Constraints:
        #   1 = ∫fnew dwpa
        #   0 = ∫wz*fnew dwpa
        #
        # Define In = ∫wz^n*f dwpa
        # gives 3 simultaneous equations
        #   1 = A*I0 + B*I1
        #   0 = A*I1 + B*I2
        # which we can solve for
        #   B = -A*I1/I2
        #
        #   A*I0 = 1 - B*I1
        #   A*I0 = 1 + A*I1/I2*I1
        #   A*(I0 - I1^2/I2) = 1

        I0 = integral(f1d, vz.wgts)
        I1 = integral(f1d, vz.grid, vz.wgts)
        I2 = integral(f1d, vz.grid, 2, vz.wgts)

        A = 1.0 / (I0 - I1^2/I2)
        B = -A*I1/I2

        @. f1d = A*f1d + B*vz.grid*f1d

        C = NaN
    elseif moments.evolve_density
        I0 = integral(f1d, vz.wgts)
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
    @begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        A[iz,ir,isn], B[iz,ir,isn], C[iz,ir,isn] =
            hard_force_moment_constraints_neutral!(@view(f[:,:,:,iz,ir,isn]), moments, vz)
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
    if moments.evolve_p
        # rnew = r + (A + B*wpa + C*wpa^2)*f
        # Constraints:
        #   0 = ∫rnew dwpa
        #   0 = ∫wpa*fnew dwpa
        #   0 = ∫wpa^2*rnew dwpa
        #
        # Define In = ∫wpa^n*f dwpa, Jn = ∫wpa^n*r dwpa
        # gives 3 simultaneous equations
        #   0 = J0 + A*I0 + B*I1 + C*I2
        #   0 = J1 + A*I1 + B*I2 + C*I3
        #   0 = J2 + A*I2 + B*I3 + C*I4
        # which we can solve for
        #   C = -(J2 + A*I2 + B*I3) / I4
        #
        #   B*I2 = -J1 - A*I1 - C*I3
        #        = -J1 - A*I1 + (J2 + A*I2 + B*I3)/I4 * I3
        #   B = (J2*I3 - J1*I4 - A*(I1*I4 - I2*I3)) / (I2*I4 - I3^2)
        #
        #   A*I0 = -J0 - B*I1 - C*I2
        #        = -J0 - B*I1 + (J2 + A*I2 + B*I3) / I4 * I2
        #   A*I0*I4 = -J0*I4 - B*I1*I4 + J2*I2 + A*I2^2 + B*I3*I2
        #   A*(I0*I4 - I2^2) = J2*I2 - J0*I4 + B*(I2*I3 - I1*I4)
        #   A*(I0*I4 - I2^2) = J2*I2 - J0*I4 + (J2*I3 - J1*I4 - A*(I1*I4 - I2*I3))*(I2*I3 - I1*I4) / (I2*I4 - I3^2)
        #   A*(I0*I4 - I2^2)*(I2*I4 - I3^2) = (J2*I2 - J0*I4)*(I2*I4 - I3^2) + (J2*I3 - J1*I4 - A*(I1*I4 - I2*I3))*(I2*I3 - I1*I4)
        #   A*((I0*I4 - I2^2)*(I2*I4 - I3^2) + (I1*I4 - I2*I3)*(I2*I3 - I1*I4)) = (J2*I2 - J0*I4)*(I2*I4 - I3^2) + (J2*I3 - J1*I4)*(I2*I3 - I1*I4)
        #   A*((I0*I4 - I2^2)*(I2*I4 - I3^2) - (I2*I3 - I1*I4)^2) = (J2*I2 - J0*I4)*(I2*I4 - I3^2) + (J2*I3 - J1*I4)*(I2*I3 - I1*I4)

        I0 = integral(f, vpa.wgts)
        I1 = integral(f, vpa.grid, vpa.wgts)
        I2 = integral(f, vpa.grid, 2, vpa.wgts)
        I3 = integral(f, vpa.grid, 3, vpa.wgts)
        I4 = integral(f, vpa.grid, 4, vpa.wgts)
        J0 = integral(residual, vpa.wgts)
        J1 = integral(residual, vpa.grid, vpa.wgts)
        J2 = integral(residual, vpa.grid, 2, vpa.wgts)

        A = ((I2*J2 - J0*I4)*(I2*I4 - I3^2) + (I2*I3 - I1*I4)*(J2*I3 - J1*I4)) /
            ((I0*I4 - I2^2)*(I2*I4 - I3^2) - (I2*I3 - I1*I4)^2)
        B = (J2*I3 - J1*I4 + (I2*I3 - I1*I4)*A) / (I2*I4 - I3^2)
        C = -(J2 + I2*A + I3*B) / I4

        @. residual = residual + (A + B*vpa.grid + C*vpa.grid*vpa.grid) * f
    elseif moments.evolve_upar
        # rnew = r + (A + B*wpa + C*wpa^2)*f
        # Constraints:
        #   0 = ∫rnew dwpa
        #   0 = ∫wpa*fnew dwpa
        #
        # Define In = ∫wpa^n*f dwpa, Jn = ∫wpa^n*r dwpa
        # gives 3 simultaneous equations
        #   0 = J0 + A*I0 + B*I1
        #   0 = J1 + A*I1 + B*I2
        # which we can solve for
        #   B = -(J1 + A*I1)/I2
        #   A*I0 = -J0 - B*I1
        #   A*I0 = -J0 + I1*(J1 + A*I1)/I2
        #   A*I0*I2 = -J0*I2 + I1*(J1 + A*I1)
        #   A*(I0*I2 - I1^2) = J1*I1 - J0*I2

        I0 = integral(f, vpa.wgts)
        I1 = integral(f, vpa.grid, vpa.wgts)
        I2 = integral(f, vpa.grid, 2, vpa.wgts)
        J0 = integral(residual, vpa.wgts)
        J1 = integral(residual, vpa.grid, vpa.wgts)

        A = (I1*J1 - J0*I2) / (I0*I2 - I1^2)
        B = -(J1 + I1*A) / I2

        @. residual = residual + (A + B*vpa.grid) * f

        C = NaN
    elseif moments.evolve_density
        I0 = integral(f, vpa.wgts)
        J0 = integral(residual, vpa.wgts)
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
    electron_implicit_constraint_forcing!(f_out, f_in, constraint_forcing_rate, vperp,
                                          vpa, dt, ir)

Add terms to the electron kinetic equation that force the moment constraints to be
approximately satisfied. Needed to avoid large errors when taking large, implicit
timesteps that do not guarantee accurate time evolution.
"""
@timeit global_timer electron_implicit_constraint_forcing!(
                         f_out, f_in, constraint_forcing_rate, vperp, vpa, dt, ir) = begin
    @begin_anyzv_z_region()
    vpa_grid = vpa.grid
    vperp_grid = vperp.grid
    @loop_z iz begin
        @views zeroth_moment = integral(f_in[:,:,iz], vpa.grid, 0, vpa.wgts, vperp.grid,
                                        0, vperp.wgts)
        @views first_moment = integral(f_in[:,:,iz], vpa.grid, 1, vpa.wgts, vperp.grid, 0,
                                       vperp.wgts)
        @views second_moment = integral((vperp,vpa)->(vperp^2 + vpa^2), f_in[:,:,iz],
                                        vperp, vpa)

        @loop_vperp_vpa ivperp ivpa begin
            f_out[ivpa,ivperp,iz] +=
                dt * constraint_forcing_rate *
                ((1.0 - zeroth_moment)
                 - first_moment*vpa_grid[ivpa]
                 + (1.5 - second_moment)*(vpa_grid[ivpa]^2 + vperp_grid[ivperp]^2)
                ) * f_in[ivpa,ivperp,iz]
        end
    end

    return nothing
end

"""
    add_electron_implicit_constraint_forcing_to_Jacobian!(jacobian_matrix, f,
                                                          z_speed, z, vperp, vpa,
                                                          constraint_forcing_rate,
                                                          dt, ir, include=:all;
                                                          f_offset=0)

Add the contributions corresponding to [`electron_implicit_constraint_forcing!`](@ref) to
`jacobian_matrix`.
"""
function add_electron_implicit_constraint_forcing_to_Jacobian!(
        jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z, vperp,
        vpa, constraint_forcing_rate, dt, ir, include=:all; f_offset=0)

    @debug_consistency_checks size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @debug_consistency_checks size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @debug_consistency_checks include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts
    v_size = vperp.n * vpa.n

    @begin_anyzv_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

        # Diagonal terms
        if include === :all
            jacobian_matrix[row,row] += -dt * constraint_forcing_rate *
                                              ((1.0 - zeroth_moment[iz])
                                               - first_moment[iz]*vpa_grid[ivpa]
                                               + (1.5 - second_moment[iz])*vpa_grid[ivpa]^2)
        end

        if include ∈ (:all, :explicit_v)
            # Integral terms
            # d(∫dw_∥ w_∥^n g[irow])/d(g[icol]) = vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^n
            for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                col = (iz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                jacobian_matrix[row,col] += dt * constraint_forcing_rate *
                                                 (1.0
                                                  + vpa_grid[icolvpa]*vpa_grid[ivpa]
                                                  + vpa_grid[icolvpa]^2*vpa_grid[ivpa]^2) *
                                                 vpa_wgts[icolvpa] * f[ivpa,ivperp,iz]
            end
        end
    end

    return nothing
end

function add_electron_implicit_constraint_forcing_to_z_only_Jacobian!(
        jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z, vperp,
        vpa, constraint_forcing_rate, dt, ir, ivperp, ivpa)

    @debug_consistency_checks size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @debug_consistency_checks size(jacobian_matrix, 1) == z.n || error("Jacobian matrix size is wrong")

    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts

    @loop_z iz begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = iz

        # Diagonal terms
        jacobian_matrix[row,row] += -dt * constraint_forcing_rate *
                                          ((1.0 - zeroth_moment[iz])
                                           - first_moment[iz]*vpa_grid[ivpa]
                                           + (1.5 - second_moment[iz])*vpa_grid[ivpa]^2)
    end

    return nothing
end

function add_electron_implicit_constraint_forcing_to_v_only_Jacobian!(
        jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z, vperp,
        vpa, constraint_forcing_rate, dt, ir, iz)

    @debug_consistency_checks size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @debug_consistency_checks size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts
    v_size = vperp.n * vpa.n

    @loop_vperp_vpa ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (ivperp - 1) * vpa.n + ivpa

        # Diagonal terms
        jacobian_matrix[row,row] += -dt * constraint_forcing_rate *
                                          ((1.0 - zeroth_moment)
                                           - first_moment*vpa_grid[ivpa]
                                           + (1.5 - second_moment)*vpa_grid[ivpa]^2)

        # Integral terms
        # d(∫dw_∥ w_∥^n g[irow])/d(g[icol]) = vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^n
        for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
            col = (icolvperp - 1) * vpa.n + icolvpa
            jacobian_matrix[row,col] += dt * constraint_forcing_rate *
                                             (1.0
                                              + vpa_grid[icolvpa]*vpa_grid[ivpa]
                                              + vpa_grid[icolvpa]^2*vpa_grid[ivpa]^2) *
                                             vpa_wgts[icolvpa] * f[ivpa,ivperp]
        end
    end

    return nothing
end

end
