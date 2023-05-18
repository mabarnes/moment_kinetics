"""
module containing Krook collision operators and other model operators
for charged particle collisions
"""
module collision_models

using ..looping

"""
krook self-collision operator
Ckrook[F] = nu * ( FMaxwellian - F)
"""
function explicit_krook_collisions!(pdf_out, pdf_in, dens_in, upar_in, vth_in, 
                                                composition, collisions, dt,
                                                r, z, vperp, vpa)
    @boundscheck vpa.n == size(pdf_in, 1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in, 2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in, 3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in, 4) || throw(BoundsError(pdf_in))
    @boundscheck composition.n_ion_species == size(pdf_in, 5) || throw(BoundsError(pdf_in))
    @boundscheck vpa.n == size(pdf_out, 1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out, 2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out, 3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out, 4) || throw(BoundsError(pdf_out))
    @boundscheck composition.n_ion_species == size(pdf_out, 5) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(dens_in, 1) || throw(BoundsError(dens_in))
    @boundscheck r.n == size(dens_in, 2) || throw(BoundsError(dens_in))
    @boundscheck composition.n_ion_species == size(dens_in, 3) || throw(BoundsError(dens_in))
    @boundscheck z.n == size(upar_in, 1) || throw(BoundsError(upar_in))
    @boundscheck r.n == size(upar_in, 2) || throw(BoundsError(upar_in))
    @boundscheck composition.n_ion_species == size(upar_in, 3) || throw(BoundsError(upar_in))
    @boundscheck z.n == size(vth_in, 1) || throw(BoundsError(vth_in))
    @boundscheck r.n == size(vth_in, 2) || throw(BoundsError(vth_in))
    @boundscheck composition.n_ion_species == size(vth_in, 3) || throw(BoundsError(vth_in))
    nu_krook = collisions.nuii_krook
    if vperp.n > 1
        pvth = 3
    else
        pvth = 1
    end
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        # form the Maxwellian from the moments of the current distribution
        F_Maxwellian = ( (dens_in[iz,ir,is]/vth_in[iz,ir,is]^pvth)*
                      exp( - (((vpa.grid[ivpa] - upar_in[iz,ir,is])^2) 
                       + (vperp.grid[ivperp]^2) )/(vth_in[iz,ir,is]^2) ) )
        Ckrook = nu_krook*(F_Maxwellian - pdf_in[ivpa,ivperp,iz,ir,is])
        pdf_out[ivpa,ivperp,iz,ir,is] += dt*Ckrook
    end
end

end