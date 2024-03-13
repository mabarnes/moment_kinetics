"""
module for supporting gyroaverages at 
fixed guiding centre R and fixed position r
"""
module gyroaverages

export gyro_operators
export init_gyro_operators
export gyroaverage_field!
export gyroaverage_pdf!

using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..array_allocation: allocate_int
using ..looping
using ..communication: MPISharedArray
#using ..communication

struct gyro_operators
    # matrix for applying a gyroaverage to a function F(r,vpa,vperp) at fixed r, with R = r - rhovec and rhovec = b x v / Omega
    # the matrix can also be used ring-averaging a function F(R,vpa,vperp) at fixed R, since F(R,vpa,vperp) is independent of gyrophase
    # other matrices are required for gyroaveraging functions that depend on gyrophase.
    gyromatrix::MPISharedArray{mk_float,6}
end

"""
initialise the gyroaverage matrix
currently the matrix for single field (z,r)
is supported. Gyroaverages are assumed to have 
no contribution from outside of the domain 
 -- this should be modified if we support periodic or
    other nonzero boundary conditions for the z and r domains,
    but might be appropriate for domains with zero boundary conditions   
"""

function init_gyro_operators(vperp,z,r,gyrophase,geometry,composition;print_info=false)
    gkions = composition.gyrokinetic_ions
    if !gkions
        gyromatrix =  allocate_shared_float(1,1,1,1,1,1)
    else
       if print_info
           println("Begin: init_gyro_operators")
       end
       gyromatrix = allocate_shared_float(z.n,r.n,vperp.n,z.n,r.n,composition.n_ion_species)
       
       # init the matrix!
       # the first two indices are to be summed over
       # the other indices are the "field" positions of the resulting gyroaveraged quantity
       begin_serial_region()
       @serial_region begin
           zlist = allocate_float(gyrophase.n)
           rlist = allocate_float(gyrophase.n)
           zelementlist = allocate_int(gyrophase.n)
           relementlist = allocate_int(gyrophase.n)
           
           @loop_s_r_z_vperp is ir iz ivperp begin
               #println("ivperp, iz, ir: ",ivperp," ",iz," ",ir)
               r_val = r.grid[ir]
               z_val = z.grid[iz]
               vperp_val = vperp.grid[ivperp]
               rhostar = geometry.rhostar
               # Bmag at the centre of the gyroaveraged path
               Bmag = geometry.Bmag[iz,ir]
               # bzeta at the centre of the gyroaveraged path
               bzeta = geometry.bzeta[iz,ir]
               # rho at the centre of the gyroaveraged path 
               # (modify to include different mass or reference temperatures)
               rho_val = vperp_val*rhostar/Bmag
               # convert these angles to a list of z'(gphase) r'(gphase)
               zrcoordinatelist!(gyrophase,zlist,rlist,rho_val,r_val,z_val,bzeta)
               # determine which elements contain these z', r'
               elementlist!(zelementlist,zlist,z)
               elementlist!(relementlist,rlist,r)
               #println(z_val,zlist)
               #println(r_val,rlist)
               #println(zelementlist)
               #println(relementlist)
               # initialise matrix to zero
               @. gyromatrix[:,:,ivperp,iz,ir,is] = 0.0
               for igyro in 1:gyrophase.n
                   # integration weight from gyroaverage (1/2pi)* int d gyrophase
                   gyrowgt = gyrophase.wgts[igyro]/(2.0*pi)
                   # get information about contributing element
                   
                   izel = zelementlist[igyro]
                   if izel < 1
                       # z' point is outside of the grid, skip this point
                       # if simply ignore contributions from outside of the domain
                       #continue
                       # if set to zero any <field> where the path exits the domain
                       @. gyromatrix[:,:,ivperp,iz,ir,is] = 0.0
                       if z.bc == "periodic"
                           print("ERROR: -1 in zelementlist")
                       end
                       break
                   end
                   izmin, izmax = z.igrid_full[1,izel], z.igrid_full[z.ngrid,izel]
                   znodes = z.grid[izmin:izmax]
                   
                   irel = relementlist[igyro]
                   if irel < 1
                       # r' point is outside of the grid, skip this point
                       # if simply ignore contributions from outside of the domain
                       #continue
                       # if set to zero any <field> where the path exits the domain
                       @. gyromatrix[:,:,ivperp,iz,ir,is] = 0.0
                       if r.bc == "periodic"
                           print("ERROR: -1 in relementlist")
                       end
                       break
                   end
                   irmin, irmax = r.igrid_full[1,irel], r.igrid_full[r.ngrid,irel]
                   rnodes = r.grid[irmin:irmax]
                   
                   #println("igyro ",igyro)
                   #println("izel ",izel," znodes ",znodes)
                   #println("irel ",irel," rnodes ",rnodes)
                   # sum over all contributing Lagrange polynomials from each
                   # collocation point in the element
                   icounter = 0
                   for irgrid in 1:r.ngrid
                       irp = r.igrid_full[irgrid,irel]
                       rpoly = lagrange_poly(irgrid,rnodes,rlist[igyro])
                       for izgrid in 1:z.ngrid
                           izp = z.igrid_full[izgrid,izel]
                           zpoly = lagrange_poly(izgrid,znodes,zlist[igyro])
                           # add the contribution from this z',r'
                           gyromatrix[izp,irp,ivperp,iz,ir,is] += gyrowgt*rpoly*zpoly
                           icounter +=1
                       end
                   end
                   #println("counter: ",icounter)
               end
           end
           if print_info
               println("Finished: init_gyro_operators")
           end
        end
    end
    gyro = gyro_operators(gyromatrix)
    return gyro
end

function zrcoordinatelist!(gyrophase,zlist,rlist,rho_val,r_val,z_val,bzeta)
    ngyro = gyrophase.n
    for i in 1:ngyro
        gphase = gyrophase.grid[i]
        zlist[i] = z_val + rho_val*cos(gphase)*bzeta
        rlist[i] = r_val + rho_val*sin(gphase)
    end
end

"""
for a given list of coordinate values, determine in which elements they are found
-1 indicates that the required element would be outside of the existing grid

-- assume here that the coordinates are fully local in memory

"""
function elementlist!(elist,coordlist,coord)
    zero = 1.0e-14
    ngyro = size(coordlist,1)
    nelement = coord.nelement_global
    xebs = coord.element_boundaries
    bc = coord.bc
    L = coord.L
    for i in 1:ngyro
        if bc=="periodic"
            x = coordlist[i]
            # if x is outside the domain, shift x to the appropriate periodic copy within the domain
            x0 = xebs[1] # the lower endpoint
            y = x-x0
            r = rem(y,L) + 0.5*L*(1.0 - sign(y)) # get the remainder of x - x0 w.r.t. the domain length L, noting that r should be r > 0
            x = r + x0 # shift so that x is bounded below by x0
            coordlist[i] = x # update coordlist for later use
        end
        # determine which element contains the position x
        x = coordlist[i]
        elist[i] = -1
        for j in 1:nelement
            # check internal locations
            if (x - xebs[j])*(xebs[j+1] - x) > zero
                elist[i] = j
                break
            # check element boundary 
            # (lower or upper, force a choice of element for boundary values)
            elseif (abs(x-xebs[j]) < 100*zero) || (abs(x-xebs[j+1]) < 100*zero && j == nelement)
                elist[i] = j
                break
            end            
        end
    end
    return 
end

"""
Copy of function in fokker_planck_calculus.jl
Lagrange polynomial
args: 
j - index of l_j from list of nodes
x_nodes - array of x node values
x - point where interpolated value is returned
"""
function lagrange_poly(j,x_nodes,x)
    # get number of nodes
    n = size(x_nodes,1)
    # location where l(x0) = 1
    x0 = x_nodes[j]
    # evaluate polynomial
    poly = 1.0
    for i in 1:j-1
            poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
    end
    for i in j+1:n
            poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
    end
    return poly
end

function gyroaverage_field!(gfield_out,field_in,gyro,vperp,z,r,composition)
    @boundscheck z.n == size(field_in, 1) || throw(BoundsError(field_in))
    @boundscheck r.n == size(field_in, 2) || throw(BoundsError(field_in))
    @boundscheck vperp.n == size(gfield_out, 1) || throw(BoundsError(gfield))
    @boundscheck z.n == size(gfield_out, 2) || throw(BoundsError(gfield))
    @boundscheck r.n == size(gfield_out, 3) || throw(BoundsError(gfield))
    @boundscheck composition.n_ion_species == size(gfield_out, 4) || throw(BoundsError(gfield))
    
    nr = r.n
    nz = z.n
    gyromatrix = gyro.gyromatrix
    
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        gfield_out[ivperp,iz,ir] = 0.0
        # sum over all the contributions in the gyroaverage
        for irp in 1:nr
            for izp in 1:nz
                gfield_out[ivperp,iz,ir,is] += gyromatrix[izp,irp,ivperp,iz,ir,is]*field_in[izp,irp]
            end
        end
    end

end

function gyroaverage_pdf!(gpdf_out,pdf_in,gyro,vpa,vperp,z,r,composition)
    @boundscheck vpa.n == size(pdf_in, 1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in, 2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in, 3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in, 4) || throw(BoundsError(pdf_in))
    @boundscheck composition.n_ion_species == size(pdf_in, 5) || throw(BoundsError(pdf_in))
    @boundscheck vpa.n == size(gpdf_out, 1) || throw(BoundsError(gpdf_out))
    @boundscheck vperp.n == size(gpdf_out, 2) || throw(BoundsError(gpdf_out))
    @boundscheck z.n == size(gpdf_out, 3) || throw(BoundsError(gpdf_out))
    @boundscheck r.n == size(gpdf_out, 4) || throw(BoundsError(gpdf_out))
    @boundscheck composition.n_ion_species == size(gpdf_out, 5) || throw(BoundsError(gpdf_out))
    
    nr = r.n
    nz = z.n
    gyromatrix = gyro.gyromatrix
    
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        gpdf_out[ivpa,ivperp,iz,ir,is] = 0.0
        # sum over all the contributions in the gyroaverage
        for irp in 1:nr
            for izp in 1:nz
                gpdf_out[ivpa,ivperp,iz,ir,is] += gyromatrix[izp,irp,ivperp,iz,ir,is]*pdf_in[ivpa,ivperp,izp,irp,is]
            end
        end
    end

end


end
