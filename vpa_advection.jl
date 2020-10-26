module vpa_advection

export vpa_advection!
export update_speed_vpa!

import moment_kinetics_input: advection_speed, advection_speed_option
import semi_lagrange: find_approximate_characteristic!
import source_terms: update_advection_factor!
import source_terms: calculate_explicit_source!
import source_terms: update_f!
import chebyshev: update_fcheby!
import chebyshev: update_df_chebyshev!
using em_fields: update_phi!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function vpa_advection!(ff, phi, moments, SL, source, vpa, z, use_semi_lagrange, dt, chebyshev)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
    @boundscheck size(ff,3) == 3 || throw(BoundsError(ff))
    # get the updated speed along the vpa direction
    update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,1), vpa, z)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for iz ∈ 1:z.n
            find_approximate_characteristic!(SL[iz], view(source.speed,iz,:), vpa, dt)
        end
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
    for j ∈ 1:jend
        for iz ∈ 1:z.n
            # calculate the factor appearing in front of df/dvpa in the advection
            # term at time level n in the frame moving with the approximate
            # characteristic
            update_advection_factor!(view(source.adv_fac,iz,:),
                view(source.speed,iz,:), SL[iz], vpa.n, dt, j)
            # Chebyshev transform f to get Chebyshev spectral coefficients
            # and use them to calculate f'
            update_fcheby!(chebyshev, view(ff,iz,:,j), vpa)
            update_df_chebyshev!(view(source.df,iz,:), chebyshev, vpa)
            # calculate the explicit source terms on the rhs of the equation;
            # i.e., -Δt⋅δv⋅f'
            calculate_explicit_source!(view(source.rhs,iz,:), view(source.df,iz,:),
                view(source.adv_fac,iz,:), SL[iz].dep_idx, vpa.n, j)
            # update ff at time level n+1 using an explicit Runge-Kutta method
            # along approximate characteristics
            update_f!(view(ff,iz,:,:), view(source.rhs,iz,:), SL[iz].dep_idx, vpa.n, j)
        end
        #moments.dens_updated = false ; moments.ppar_updated = false
        # calculate the advection speed corresponding to current f
        if j != jend
            update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,j+1), vpa, z)
        end
    end
    @inbounds begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa,1] = 0.5*(ff[iz,ivpa,2] + ff[iz,ivpa,3])
            end
        end
    end
end
# for use with finite difference scheme
# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function vpa_advection!(ff, phi, moments, SL, source, vpa, z, use_semi_lagrange, dt)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(f,1) == z.n || throw(BoundsError(f))
    @boundscheck size(f,2) == vpa.n || throw(BoundsError(f))
    @boundscheck size(f,3) == 3 || throw(BoundsError(f))
    # get the updated speed along the vpa direction
    update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,1), vpa, z)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for iz ∈ 1:z.n
            find_approximate_characteristic!(SL[iz], view(source.speed,iz,:), vpa, dt)
        end
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
    for j ∈ 1:jend
        for iz ∈ 1:z.n
            # calculate the factor appearing in front of df/dz in the advection
            # term at time level n in the frame moving with the approximate
            # characteristic
            update_advection_factor!(view(source.adv_fac,iz,:),
                view(source.speed,iz,:), SL[iz], vpa.n, dt, j)
            # calculate the derivative of f
            update_df_finite_difference!(view(source.df,iz,:), view(ff,iz,:,:),
                vpa.cell_width, j, view(source.adv_fac,iz,:), vpa.bc)
            # calculate the explicit source terms on the rhs of the equation;
            # i.e., -Δt⋅δv⋅f'
            calculate_explicit_source!(view(source.rhs,iz,:), view(source.df,iz,:),
                view(source.adv_fac,iz,:), SL[iz].dep_idx, vpa.n, j)
            # update ff at time level n+1 using an explicit Runge-Kutta method
            # along approximate characteristics
            update_f!(view(ff,iz,:,:), view(source.rhs,iz,:), SL[iz].dep_idx, vpa.n, j)
        end
        #moments.dens_updated = false ; moments.ppar_updated = false
        # calculate the advection speed corresponding to current f
	if j != jend
	   update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,j+1), vpa, z)
	end
    end
    @inbounds begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa,1] = 0.5*(ff[iz,ivpa,2] + ff[iz,ivpa,3])
            end
        end
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(speed, phi, moments, ff, vpa, z)
    @boundscheck z.n == size(speed,1) || throw(BoundsError(speed))
    @boundscheck vpa.n == size(speed,2) || throw(BoundsError(speed))
    if advection_speed_option == "default"
        update_phi!(phi, moments, ff, vpa, z.n)
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    speed[i,j] = advection_speed
                end
            end
        end
    elseif advection_speed_option == "constant"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    speed[i,j] = advection_speed
                end
            end
        end
    elseif advection_speed_option == "linear"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    speed[i,j] = advection_speed*(vpa.grid[i]+0.5*vpa.L)
                end
            end
        end
    end
    return nothing
end

function update_df_finite_difference!(df, f, del, j, adv_fac, bc)
    n = length(del)
    @boundscheck n == length(df) || throw(BoundsError(df))
    @boundscheck n == length(del) || throw(BoundsError(del))
    @inbounds begin
        if bc == "zero"
            df[1] = 0.
        elseif bc == "periodic"
            df[1] = (f[1,j]-f[n-1,j])/del[1]
        end
        for i ∈ 3:n-2
            if adv_fac[i] < 0
                df[i] =  (3*f[i,j]-4*f[i-1,j]+f[i-2,j])/(2*del[i])
            else
                df[i] = (-f[i+2,j]+4*f[i+1,j]-3*f[i,j])/(2*del[i+1])
            end
        end
        if adv_fac[1] > 0
            df[1] = (-f[3,j]+4*f[2,j]-3*f[1,j])/(2*del[2])
        end
        if adv_fac[2] > 0
            df[2] = (-f[4,j]+4*f[3,j]-3*f[2,j])/(2*del[3])
        else
            # have to modify for periodic
            df[2] = (3*f[2,j]-4*f[1,j])/(2*del[2])
        end
        if adv_fac[n] < 0
            df[n] = (3*f[n,j]-4*f[n-1,j]+f[n-2,j])/(2*del[n])
        end
        if adv_fac[n-1] < 0
            df[n-1] = (3*f[n-1,j]-4*f[n-2,j]+f[n-3,j])/(2*del[n-1])
        else
            # have to modify for periodic
            df[n-1] = (4*f[n,j]-3*f[n-1,j])/(2*del[n])
        end
    end
end

end
