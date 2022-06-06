"""
"""
module velocity_grid_transforms

export vzvrvzeta_to_vpavperp!
export vpavperp_to_vzvrvzeta!

using Interpolations
using ..looping
"""
"""

function vzvrvzeta_to_vpavperp!(f_out,f_in,vz,vr,vzeta,vpa,vperp,gyrophase,z,r,geometry,composition)
    @boundscheck vz.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck vr.n == size(f_in, 2) || throw(BoundsError(f_in))
    @boundscheck vzeta.n == size(f_in, 3) || throw(BoundsError(f_in))
    @boundscheck z.n == size(f_in, 4) || throw(BoundsError(f_in))
    @boundscheck r.n == size(f_in, 5) || throw(BoundsError(f_in))
    @boundscheck composition.n_neutral_species == size(f_in, 6) || throw(BoundsError(f_in))
    @boundscheck vpa.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck vperp.n == size(f_out, 2) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out, 3) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out, 4) || throw(BoundsError(f_out))
    @boundscheck composition.n_neutral_species == size(f_out, 5) || throw(BoundsError(f_out))

    begin_s_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        @views vzvrvzeta_to_vpavperp_species!(f_out[:,:,iz,ir,isn],f_in[:,:,:,iz,ir,isn],
                                vz,vr,vzeta,vpa,vperp,gyrophase,geometry)
    end

end

function vzvrvzeta_to_vpavperp_species!(f_out,f_in,vz,vr,vzeta,vpa,vperp,gyrophase,geometry)
    @boundscheck vz.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck vr.n == size(f_in, 2) || throw(BoundsError(f_in))
    @boundscheck vzeta.n == size(f_in, 3) || throw(BoundsError(f_in))
    @boundscheck vpa.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck vperp.n == size(f_out, 2) || throw(BoundsError(f_out))

    pdf_interp = LinearInterpolation((vz.grid,vr.grid,vzeta.grid),f_in,extrapolation_bc = 0.0)
    # pdf_interp( vz_val, vr_val, vzeta_val) is interpolated value of f_in
    # extrapolation_bc = 0.0 makes pdf_interp = 0.0 for |vx| > vx.L/2 (x = z,r,zeta)
    
    bzed = geometry.bzed
    bzeta = geometry.bzeta
    
    @loop_vperp_vpa ivperp ivpa begin
        # for each ivpa, ivperp, compute gyroaverage of f_in
        # use 
        # vz = vpa b_zed - vperp sin gyrophase b_zeta
        # vr = vperp cos gyrophase
        # vzeta = vpa b_zeta + vperp sin gyrophase b_zed
        gyroaverage = 0.0
        for igyro in 1:gyrophase.n
            vz_val = vpa.grid[ivpa]*bzed  - vperp.grid[ivperp]*sin(gyrophase.grid[igyro])*bzeta
            vr_val = vperp.grid[ivperp]*cos(gyrophase.grid[igyro])
            vzeta_val = vpa.grid[ivpa]*bzeta + vperp.grid[ivperp]*sin(gyrophase.grid[igyro])*bzed
            
            gyroaverage += gyrophase.wgts[igyro] * pdf_interp(vz_val, vr_val, vzeta_val) / (2.0*pi)
        end
        f_out[ivpa,ivperp] = gyroaverage
    end
end 

function vpavperp_to_vzvrvzeta!(f_out,f_in,vz,vr,vzeta,vpa,vperp,z,r,geometry,composition)
    @boundscheck vpa.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck vperp.n == size(f_in, 2) || throw(BoundsError(f_in))
    @boundscheck z.n == size(f_in, 3) || throw(BoundsError(f_in))
    @boundscheck r.n == size(f_in, 4) || throw(BoundsError(f_in))
    @boundscheck composition.n_ion_species == size(f_in, 5) || throw(BoundsError(f_in))
    @boundscheck vz.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck vr.n == size(f_out, 2) || throw(BoundsError(f_out))
    @boundscheck vzeta.n == size(f_out, 3) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out, 4) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out, 5) || throw(BoundsError(f_out))
    @boundscheck composition.n_ion_species == size(f_out, 6) || throw(BoundsError(f_out))
    
    begin_s_sn_r_z_region()
    @loop_s_r_z is ir iz begin
        @views vpavperp_to_vzvrvzeta_species!(f_out[:,:,:,iz,ir,is],f_in[:,:,iz,ir,is],
                                vz,vr,vzeta,vpa,vperp,geometry)
    end

end


function vpavperp_to_vzvrvzeta_species!(f_out,f_in,vz,vr,vzeta,vpa,vperp,geometry)
    @boundscheck vz.n == size(f_out, 1) || throw(BoundsError(f_out))
    @boundscheck vr.n == size(f_out, 2) || throw(BoundsError(f_out))
    @boundscheck vzeta.n == size(f_out, 3) || throw(BoundsError(f_out))
    @boundscheck vpa.n == size(f_in, 1) || throw(BoundsError(f_in))
    @boundscheck vperp.n == size(f_in, 2) || throw(BoundsError(f_in))

    pdf_interp = LinearInterpolation((vpa.grid,vperp.grid),f_in,extrapolation_bc = 0.0)
    # pdf_interp( vz_val, vr_val, vzeta_val) is interpolated value of f_in
    # extrapolation_bc = 0.0 makes pdf_interp = 0.0 for |vpa| > vpa.L/2 and vperp > vperp.L
    
    bzed = geometry.bzed
    bzeta = geometry.bzeta
    
    @loop_vzeta_vr_vz ivzeta ivr ivz begin
        # for each ivpa, ivperp, compute gyroaverage of f_in
        # use 
        # vz = vpa b_zed - vperp sin gyrophase b_zeta
        # vr = vperp cos gyrophase
        # vzeta = vpa b_zeta + vperp sin gyrophase b_zed
        vpa_val = bzed*vz.grid[ivz] + bzeta*vzeta.grid[ivzeta]
        vperp_val = sqrt( vr.grid[ivr]^2.0 + (bzed*vzeta.grid[ivzeta] - bzeta*vz.grid[ivzeta])^2.0)
        
        f_out[ivz,ivr,ivzeta] = pdf_interp(vpa_val,vperp_val)
    end
end 

end