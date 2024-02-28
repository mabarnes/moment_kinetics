"""
module for supporting gyroaverages at 
fixed guiding centre R and fixed position r
"""
module gyroaverages

using ..type_definitions: mk_float
using ..array_allocation: allocate_float, allocate_shared_float
using ..array_allocation: allocate_int
using ..looping

struct gyro_operators
    gyromatrix::Array{mk_float,5}
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

function init_gyro_operators(vperp,z,r,gyrophase,geometry,composition)
    gyromatrix = allocate_float(z.n,r.n,vperp.n,z.n,r.n)
    
    # init the matrix!
    # the first two indices are to be summed over
    # the other indices are the "field" positions of the resulting gyroaveraged quantity
    begin_serial_region()
    zlist = allocate_float(gyrophase.n)
    rlist = allocate_float(gyrophase.n)
    zelementlist = allocate_int(gyrophase.n)
    relementlist = allocate_int(gyrophase.n)
    
    @loop_r_z_vperp ir iz ivperp begin
        println("ivperp, iz, ir: ",ivperp," ",iz," ",ir)
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
        println(z_val,zlist)
        println(r_val,rlist)
        println(zelementlist)
        println(relementlist)
        # initialise matrix to zero
        @. gyromatrix[:,:,ivperp,iz,ir] = 0.0
        for igyro in 1:gyrophase.n
            # integration weight from gyroaverage (1/2pi)* int d gyrophase
            gyrowgt = gyrophase.wgts[igyro]/(2.0*pi)
            # get information about contributing element
            
            izel = zelementlist[igyro]
            if izel < 1
                # z' point is outside of the grid, skip this point
                continue
            end
            izmin, izmax = z.igrid_full[1,izel], z.igrid_full[z.ngrid,izel]
            znodes = z.grid[izmin:izmax]
            
            irel = relementlist[igyro]
            if irel < 1
                # r' point is outside of the grid, skip this point
                continue
            end
            irmin, irmax = r.igrid_full[1,irel], r.igrid_full[r.ngrid,irel]
            rnodes = r.grid[irmin:irmax]
            
            println("igyro ",igyro)
            println("izel ",izel," znodes ",znodes)
            println("irel ",irel," rnodes ",rnodes)
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
                    gyromatrix[izp,irp,ivperp,iz,ir] += gyrowgt*rpoly*zpoly
                    icounter +=1
                end
            end
            println("counter: ",icounter)
        end
    end
    
    gyro = gyro_operators(gyromatrix)
    return gyro
end

function zrcoordinatelist!(gyrophase,zlist,rlist,rho_val,r_val,z_val,bzeta)
    ngyro = gyrophase.n
    for i in 1:ngyro
        gphase = gyrophase.grid[i]
        zlist[i] = z_val + rho_val*sin(gphase)
        rlist[i] = r_val + rho_val*cos(gphase)*bzeta
    end
end

"""
for a given list of coordinate values, determine in which elements they are found
-1 indicates that the required element would be outside of the existing grid

-- assume here that the coordinates are fully local in memory

"""
function elementlist!(elist,coordlist,coord)
    zero = 1.0e-8
    ngyro = size(coordlist,1)
    nelement = coord.nelement_global
    xebs = coord.element_boundaries
    for i in 1:ngyro
        x = coordlist[i]
        elist[i] = -1
        for j in 1:nelement
            # check internal locations
            if (x - xebs[j])*(xebs[j+1] - x) > zero
                elist[i] = j
                break
            end
            # check element boundary 
            # (lower or upper, force a choice of element for boundary values)
            if (abs(x-xebs[j]) < zero) || (abs(x-xebs[j+1]) < zero && j == nelement)
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

function gyroaverage_field!(gfield_out,field_in,gyro,vperp,z,r)
    @boundscheck z.n == size(field_in, 1) || throw(BoundsError(field_in))
    @boundscheck r.n == size(field_in, 2) || throw(BoundsError(field_in))
    @boundscheck vperp.n == size(gfield_out, 1) || throw(BoundsError(gfield))
    @boundscheck z.n == size(gfield_out, 2) || throw(BoundsError(gfield))
    @boundscheck r.n == size(gfield_out, 3) || throw(BoundsError(gfield))
    
    nr = r.n
    nz = z.n
    gyromatrix = gyro.gyromatrix
    
    begin_serial_region()
    @loop_r_z_vperp ir iz ivperp begin
        gfield_out[ivperp,iz,ir] = 0.0
        # sum over all the contributions in the gyroaverage
        for irp in 1:nr
            for izp in 1:nz
                gfield_out[ivperp,iz,ir] += gyromatrix[izp,irp,ivperp,iz,ir]*field_in[izp,irp]
            end
        end
    end

end


end
