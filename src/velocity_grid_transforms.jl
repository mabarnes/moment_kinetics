"""
"""
module velocity_grid_transforms

export vzvrvzeta_to_vpamu!
export vpamu_to_vzvrvzeta!

using Interpolations
using ..looping
"""
"""

function vzvrvzeta_to_vpamu!(f_out,f_in,vz,vr,vzeta,vpa,mu,gyrophase,z,r,geometry,composition)
    @boundscheck vz.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck vr.n == size(f_in, 2) || throw(BoundsError(f_in))
    @boundscheck vzeta.n == size(f_in, 3) || throw(BoundsError(f_in))
    @boundscheck z.n == size(f_in, 4) || throw(BoundsError(f_in))
    @boundscheck r.n == size(f_in, 5) || throw(BoundsError(f_in))
    @boundscheck composition.n_neutral_species == size(f_in, 6) || throw(BoundsError(f_in))
    @boundscheck vpa.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck mu.n == size(f_out, 2) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out, 3) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out, 4) || throw(BoundsError(f_out))
    @boundscheck composition.n_neutral_species == size(f_out, 5) || throw(BoundsError(f_out))

    begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        @views vzvrvzeta_to_vpamu_species!(f_out[:,:,iz,ir,isn],f_in[:,:,:,iz,ir,isn],
                                vz,vr,vzeta,vpa,mu,gyrophase,geometry)
    end

end

function vzvrvzeta_to_vpamu_species!(f_out,f_in,vz,vr,vzeta,vpa,mu,gyrophase,geometry)
    @boundscheck vz.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck vr.n == size(f_in, 2) || throw(BoundsError(f_in))
    @boundscheck vzeta.n == size(f_in, 3) || throw(BoundsError(f_in))
    @boundscheck vpa.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck mu.n == size(f_out, 2) || throw(BoundsError(f_out))

    pdf_interp = LinearInterpolation((vz.grid,vr.grid,vzeta.grid),f_in,extrapolation_bc = 0.0)
    # pdf_interp( vz_val, vr_val, vzeta_val) is interpolated value of f_in
    # extrapolation_bc = 0.0 makes pdf_interp = 0.0 for |vx| > vx.L/2 (x = z,r,zeta)
    
    bzed = geometry.bzed
    bzeta = geometry.bzeta
    Bmag = geometry.Bmag
    
    @loop_mu_vpa imu ivpa begin
        # for each ivpa, imu, compute gyroaverage of f_in
        # use 
        # vz = vpa b_zed - sqrt(2 Bmag mu) sin gyrophase b_zeta
        # vr = sqrt(2 Bmag mu) cos gyrophase
        # vzeta = vpa b_zeta + sqrt(2 Bmag mu) sin gyrophase b_zed
        gyroaverage = 0.0
        for igyro in 1:gyrophase.n
            vz_val = vpa.grid[ivpa]*bzed  - sqrt(2.0*Bmag*mu.grid[imu])*sin(gyrophase.grid[igyro])*bzeta
            vr_val = sqrt(2.0*Bmag*mu.grid[imu])*cos(gyrophase.grid[igyro])
            vzeta_val = vpa.grid[ivpa]*bzeta + sqrt(2.0*Bmag*mu.grid[imu])*sin(gyrophase.grid[igyro])*bzed
            
            gyroaverage += gyrophase.wgts[igyro] * pdf_interp(vz_val, vr_val, vzeta_val) / (2.0*pi)
        end
        f_out[ivpa,imu] = gyroaverage
    end
end 

function vpamu_to_vzvrvzeta!(f_out,f_in,vz,vr,vzeta,vpa,mu,z,r,geometry,composition)
    @boundscheck vpa.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck mu.n == size(f_in, 2) || throw(BoundsError(f_in))
    @boundscheck z.n == size(f_in, 3) || throw(BoundsError(f_in))
    @boundscheck r.n == size(f_in, 4) || throw(BoundsError(f_in))
    @boundscheck composition.n_ion_species == size(f_in, 5) || throw(BoundsError(f_in))
    @boundscheck vz.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck vr.n == size(f_out, 2) || throw(BoundsError(f_out))
    @boundscheck vzeta.n == size(f_out, 3) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out, 4) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out, 5) || throw(BoundsError(f_out))
    @boundscheck composition.n_ion_species == size(f_out, 6) || throw(BoundsError(f_out))
    
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        @views vpamu_to_vzvrvzeta_species!(f_out[:,:,:,iz,ir,is],f_in[:,:,iz,ir,is],
                                vz,vr,vzeta,vpa,mu,geometry)
    end

end


function vpamu_to_vzvrvzeta_species!(f_out,f_in,vz,vr,vzeta,vpa,mu,geometry)
    @boundscheck vz.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck vr.n == size(f_out, 2) || throw(BoundsError(f_out))
    @boundscheck vzeta.n == size(f_out, 3) || throw(BoundsError(f_out))
    @boundscheck vpa.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck mu.n == size(f_in, 2) || throw(BoundsError(f_in))

    pdf_interp = LinearInterpolation((vpa.grid,mu.grid),f_in,extrapolation_bc = 0.0)
    # pdf_interp( vz_val, vr_val, vzeta_val) is interpolated value of f_in
    # extrapolation_bc = 0.0 makes pdf_interp = 0.0 for |vpa| > vpa.L/2 and mu > mu.L
    
    bzed = geometry.bzed
    bzeta = geometry.bzeta
    Bmag = geometry.Bmag
    
    @loop_vzeta_vr_vz ivzeta ivr ivz begin
        # for each ivzeta, ivr, ivz interpolate f_in onto f_out
        # use 
        # vz = vpa b_zed - sqrt(2 Bmag mu) sin gyrophase b_zeta
        # vr = sqrt(2 Bmag mu) cos gyrophase
        # vzeta = vpa b_zeta + sqrt(2 Bmag mu) sin gyrophase b_zed
        vpa_val = bzed*vz.grid[ivz] + bzeta*vzeta.grid[ivzeta]
        mu_val =  ( vr.grid[ivr]^2.0 + (bzed*vzeta.grid[ivzeta] - bzeta*vz.grid[ivz])^2.0 )/(2.0*Bmag)
        
        f_out[ivz,ivr,ivzeta] = pdf_interp(vpa_val,mu_val)
    end
end 

end
