# -*- coding: utf-8 -*-
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # this line allows plots to be made without using a display environment variable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import toml as tml
from xarray_mk_utils import grid_data
from xarray_mk_utils import dynamic_data
from plot_mk_utils import plot_1d_list_pdf, plot_1d_loglog_list_pdf

def get_sd_plot_data(filename):
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
    
    ntime = time.size
    
    #print("z: ",zgrid)
    #print("r: ",rgrid)
    #print("vpa: ",vpagrid)
    #print("vperp: ",vperpgrid)

    # return the data of interest only
    it = ntime - 1
    ivpa = nvpa_global//2+1
    ivperp = 0
    ir = 0
    iz = 0 # lower wall plate
    ispec = 0 # single species
    
    return vperpgrid, ff[it,ispec,ir,iz,:,ivpa], vpagrid, ff[it,ispec,ir,iz,ivperp,:]

def get_sd_input_data(filename):
    print(filename)
    with open(filename, 'r') as file:
        inputdata = tml.load(file)
    key = "fokker_planck_collisions"
    print(inputdata[key])
    ni = inputdata[key]["sd_density"]
    Ti = inputdata[key]["sd_temp"]
    Te = Ti
    # only a single charge ion is evolved, hence 
    # we interpret species i as alpha here
    Zalpha = inputdata[key]["Zi"]
    # Zi must be absolute / relative to proton charge
    # whereas the fixed Maxwellian ion charge number is given here
    Zi = inputdata[key]["sd_q"]
    ne = Zi*ni + 1.0*Zalpha # quasineutrality to determine electron density
    # initial alpha density must be unity
    mi = inputdata[key]["sd_mi"]
    me = inputdata[key]["sd_me"]
    nuref = inputdata[key]["nuii"]
    # compute critical speed vc3/cref^3, cref = sqrt(2 Tref/mref) -> factor of 1/ 2 sqrt 2 
    vc3 = (np.sqrt(2.0)/4.0)*3.0*np.sqrt(np.pi/2.0)*(Zi**2)*((Te)**1.5)*(ni/ne)/(np.sqrt(me)*mi)

    key = "ion_source"
    print(inputdata[key])
    v0 = inputdata[key]["source_v0"]
    Salpha = inputdata[key]["source_strength"]
    # use that nuref = gamma_alphaalpha nref / mref^2 cref^3, with cref = sqrt(2Tref/mref) and alphas the reference species
    # gamma_alphaalpha = 2 pi Zalpha^4 e^4 ln Lambda/ (4pi epsilon0)^2
    # and nu_alphae = (4/(3sqrt(2pi))) gamma_alphae ne Te^(-3/2) sqrt(me)/m_alpha
    nualphae = nuref*(8.0/3.0)*(1.0/np.sqrt(np.pi))*ne*np.sqrt(me)*(Te**(-1.5))*(Zalpha**(2))
    amplitude = (np.sqrt(np.pi)/4.0)*Salpha/nualphae # pi^3/2 * (1/4 pi) factor had pi^3/2 due to normalisation of integration and pdf
    return v0, vc3, amplitude

workdir = ""
input_filename_list = [workdir+"/excalibur/moment_kinetics_gyro/runs/fokker-planck-relaxation-example-4.toml"]
filename_dfns_list = [workdir+"/excalibur/moment_kinetics_gyro/runs/fokker-planck-relaxation-example-4/fokker-planck-relaxation-example-4.dfns.0.h5",]
identity = "example-4"

vpagrid_list = []
vperpgrid_list = []
ff_list = []
logff_list = []
ffvpa_list = []
logffvpa_list = []

for ifile, filename_dfn in enumerate(filename_dfns_list):

    v0, vc3, amplitude = get_sd_input_data(input_filename_list[ifile])

    vperpgrid, ff, vpagrid, ffvpa = get_sd_plot_data(filename_dfn)
    vperpgrid_list.append(vperpgrid)
    ff_list.append(ff)
    logff_list.append(np.log(np.abs(ff)+1.0e-15))
    vpagrid_list.append(vpagrid)
    ffvpa_list.append(ffvpa)
    logffvpa_list.append(np.log(np.abs(ffvpa)+1.0e-15))

    # compute a slowing down distribution for comparison from an analytical formula

    ff_sd = np.copy(vperpgrid)
    nvperp = vperpgrid.size
    vc3test = 3.0*np.sqrt(np.pi/2.0)*((0.01)**1.5)*(1.0/(np.sqrt(2.7e-4)*0.5))*(0.5*0.5*1.0/2.0)
    #print(vc3test," ",vc3test**(1.0/3.0))
    print("vc3: ", vc3," vc: ",vc3**(1.0/3.0))
    for ivperp in range(0,nvperp):
        vperp = vperpgrid[ivperp]
        if vperp < v0:
            ff_sd[ivperp] = 1.0/(vc3 + vperp**3.0)
            #print(ivperp)
        else:
            ff_sd[ivperp] = 0.0
    # pick a point to normalise by
    ivperp = 32 #nvperp//3 + 1
    amplitude_test=ff[ivperp]/ff_sd[ivperp]
    print(amplitude_test," ",amplitude, " ", amplitude/amplitude_test)
    ff_sd = ff_sd*amplitude
#    ff_sd = ff_sd*amplitude_test

    vperpgrid_list.append(vperpgrid)
    ff_list.append(ff_sd)
    logff_list.append(np.log(np.abs(ff_sd)+1.0e-15))

    ff_sd_vpa = np.copy(vpagrid)
    nvpa = vpagrid.size
    for ivpa in range(0,nvpa):
        vpa = vpagrid[ivpa]
        if np.abs(vpa) < v0:
            ff_sd_vpa[ivpa] = 1.0/(vc3 + np.abs(vpa)**3.0)
            #print(ivpa)
        else:
            ff_sd_vpa[ivpa] = 0.0
    # pick a point to normalise by
    ivpa = 96# nvpa//2 + nvpa//6 + 1
    amplitude_test=ffvpa[ivperp]/ff_sd_vpa[ivperp]
    print(amplitude_test," ",amplitude, " ", amplitude/amplitude_test)
    ff_sd_vpa = ff_sd_vpa*amplitude
#    ff_sd_vpa = ff_sd_vpa*amplitude_test

    vpagrid_list.append(vpagrid)
    ffvpa_list.append(ff_sd_vpa)
    logffvpa_list.append(np.log(np.abs(ff_sd_vpa)+1.0e-15))

marker_list = ['k','b','r','g','c','y']
ylab_list = ["Num","SD"]
file = workdir + "excalibur/moment_kinetics_gyro/sd_scan_"+str(identity)+".pdf"
pdf = PdfPages(file)

# plot ff
plot_1d_list_pdf (vperpgrid_list,ff_list,marker_list,"$v_{\\perp}$", pdf,
  title='$f(v_{\\|}=0,v_{\\perp})$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)
# plot logff
plot_1d_list_pdf (vperpgrid_list,logff_list,marker_list,"$v_{\\perp}$", pdf,
  title='$\\ln|f(v_{\\|}=0,v_{\\perp})|$',ylab='',xlims=None,ylims=[-10.0,1.1*np.max(logff_list[0])],aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)

# plot ffvpa
#print(vpagrid_list,ffvpa_list)
plot_1d_list_pdf (vpagrid_list,ffvpa_list,marker_list,"$v_{\\|}$", pdf,
  title='$f(v_{\\|},v_{\\perp}=0)$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)
# plot logffvpa
plot_1d_list_pdf (vpagrid_list,logffvpa_list,marker_list,"$v_{\\|}$", pdf,
  title='$\\ln|f(v_{\\|},v_{\\perp}=0)|$',ylab='',xlims=None,ylims=[-10.0,1.1*np.max(logffvpa_list[0])],aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)


pdf.close()
print("Saving figure: "+file)
