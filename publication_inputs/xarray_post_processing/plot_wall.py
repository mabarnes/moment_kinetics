# -*- coding: utf-8 -*-
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # this line allows plots to be made without using a display environment variable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from xarray_mk_utils import grid_data
from xarray_mk_utils import dynamic_data
from plot_mk_utils import plot_1d_list_pdf, plot_1d_loglog_list_pdf

def get_wall_plot_data(filename):
    # return grid, moments, and pdf at the last timestep of the simulation
    # where the pdf is written
    # assume that there is single output file, from a simulation
    # using parallel HDF5 or a single shared-memory region
    
    nz_global, nz_local, zgrid = grid_data(filename,"z")
    nr_global, nr_local, rgrid = grid_data(filename,"r")
    nvpa_global, nvpa_local, vpagrid = grid_data(filename,"vpa")
    nvperp_global, nvperp_local, vperpgrid = grid_data(filename,"vperp")

    time, time_present = dynamic_data(filename,"time")
    ff, ff_present = dynamic_data(filename,"f")
    Ez, Ez_present = dynamic_data(filename,"Ez")
    phi, phi_present = dynamic_data(filename,"phi")
    cil, cil_present = dynamic_data(filename,"chodura_integral_lower")
    
    ntime = time.size
    
    #print("z: ",zgrid)
    #print("r: ",rgrid)
    #print("vpa: ",vpagrid)
    #print("vperp: ",vperpgrid)

    # return the data of interest only
    it = ntime - 1
    ivperp = 0
    ir = 0
    iz = 0 # lower wall plate
    ispec = 0 # single species
    return zgrid, vpagrid, ff[it,ispec,ir,iz,ivperp,:], Ez[it,ir,:], phi[it,ir,:], cil[it,ir]

workdir = ""
filename = workdir + "moment_kinetics_newgeo/runs/wall-bc_cheb/wall-bc_cheb.moments.0.h5"
filename_dfns_list = ["moment_kinetics_newgeo/runs/wall-bc_cheb_epsz1/wall-bc_cheb_epsz1.dfns.0.h5",
                      "moment_kinetics_newgeo/runs/wall-bc_cheb_epsz0.1/wall-bc_cheb_epsz0.1.dfns.0.h5",
                      "moment_kinetics_newgeo/runs/wall-bc_cheb_epsz0.01/wall-bc_cheb_epsz0.01.dfns.0.h5",
                      "moment_kinetics_newgeo/runs/wall-bc_cheb_epsz0.001/wall-bc_cheb_epsz0.001.dfns.0.h5",
                      "moment_kinetics_newgeo/runs/wall-bc_cheb_epsz0/wall-bc_cheb_epsz0.dfns.0.h5"]

zgrid_list = []
vpagrid_list = []
ff_list = []
ff_over_vpa2_list = []
Ez_list = []
phi_list = []
cil_list = []
logphi_list = []
logEz_list = []
logz_list = []
for filename_dfn in filename_dfns_list:
    zgrid, vpagrid, ff, Ez, phi, cil = get_wall_plot_data(workdir+filename_dfn)
    nz = zgrid.size
    zgrid_list.append(zgrid)
    vpagrid_list.append(vpagrid)
    ff_list.append(ff)
    Ez_list.append(Ez)
    phi_list.append(phi)
    cil_list.append(cil)
    nzlog = nz//12
    #print(zgrid[-1-nzlog:-1])
    logz_list.append(np.log(0.5-zgrid[-1-nzlog:-1]))
    logphi_list.append(np.log(phi[-1-nzlog:-1]-phi[-1]))
    logEz_list.append(np.log(Ez[-1-nzlog:-1]))
    nvpa = vpagrid.size
    vpafunc = np.zeros(nvpa)
    deltavpa = np.amin(vpagrid[1:nvpa]-vpagrid[:nvpa-1])
    #print(deltavpa)
    for ivpa in range(0,nvpa):
        if np.abs(vpagrid[ivpa]) > 0.5*deltavpa:
            vpafunc[ivpa] = 1.0/(vpagrid[ivpa]**2)    
    ff_over_vpa2_list.append(ff*vpafunc)

#print(logz_list)    
#print(logEz_list)    
#print(logphi_list)    
#print(ff_over_vpa2_list)
epsz_values = [1.0,0.1,0.01,0.001,0.0]
marker_list = ['k','b','r','g','c','y']
ylab_list = [str(epsz) for epsz in epsz_values]
file = workdir + "moment_kinetics_newgeo/wall_boundary_cutoff_scan.pdf"
pdf = PdfPages(file)

# plot ff
plot_1d_list_pdf (vpagrid_list,ff_list,marker_list,"$v_{\\|\\|}$", pdf,
  title='$f(z_{\\rm wall-},v_{\\|\\|})$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="$\\epsilon_z$", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)
# plot ff/vpa2
plot_1d_list_pdf (vpagrid_list,ff_over_vpa2_list,marker_list,"$v_{\\|\\|}$", pdf,
  title='$f(z_{\\rm wall-},v_{\\|\\|})/v_{\\|\\|}^2$',ylab='',xlims=None,ylims=[-0.1,5.0],aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="$\\epsilon_z$", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)
# plot Bohm condition
ylist = [np.array(cil_list)]
xlist = [np.array(epsz_values)]
plot_1d_list_pdf (xlist,ylist,["kx--"],"$\\epsilon_z$", pdf,
  title='$(T_{\\rm e}/2 n) \\int (f/v_{\\|\\|}^2) d v_{\\|\\|}/\\sqrt{\\pi} $',ylab='',
  xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=15, legend_title="", use_legend=False,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.25, 1.0), legend_fontsize=15, ncol_opt=1, hlines = [[1.0,"","r","--"]])
# plot phi
plot_1d_list_pdf (zgrid_list,phi_list,marker_list,"$z/L_z$", pdf,
  title='$\\phi(z)$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="$\\epsilon_z$", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.55, 0.85), legend_fontsize=15, ncol_opt=1)
# plot Ez
plot_1d_list_pdf (zgrid_list,Ez_list,marker_list,"$z/L_z$", pdf,
  title='$E_z(z)$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="$\\epsilon_z$", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.25, 1.0), legend_fontsize=15, ncol_opt=1)
# plot log phi
plot_1d_list_pdf (logz_list,logphi_list,marker_list,"$\\ln(0.5 - z/L_z)$", pdf,
  title='$\\ln (\\phi(z)-\\phi_{\\rm wall})$',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1,
  legend_shadow=False,legend_frame=False,slines=[[0.5,0.0,'k--'," 0.5 log dz"],[0.6666, 1.5, 'b--',"2 log dz/3 + 1.5"]])
# plot log Ez
plot_1d_list_pdf (logz_list,logEz_list,marker_list,"$\\ln(0.5 - z/L_z)$", pdf,
  title='$\\ln E_z(z)$',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1,
  legend_shadow=False,legend_frame=False,slines=[[-0.5,0.0,'k--'," -0.5 log dz"],[-0.333, 0.9,'b--',"0.9 - log dz/3"]])

# plot log phi
#plot_1d_loglog_list_pdf (logz_list,logphi_list,marker_list,"$0.5 - z/L_z$", pdf,
#  title='$\\phi(z)-\\phi_{\\rm wall}$',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, yticks = None,
#  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
#  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1,
#  legend_shadow=False,legend_frame=False)
# plot log Ez
#plot_1d_loglog_list_pdf (logz_list,logEz_list,marker_list,"$0.5 - z/L_z$", pdf,
#  title='$E_z(z)$',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, yticks = None,
#  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
#  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1,
#  legend_shadow=False,legend_frame=False)

pdf.close()
print("Saving figure: "+file)
