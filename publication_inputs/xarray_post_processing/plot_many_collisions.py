# -*- coding: utf-8 -*-
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # this line allows plots to be made without using a display environment variable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import toml as tml
import h5py
from xarray_mk_utils import grid_data, wgts_data
from xarray_mk_utils import dynamic_data
from plot_mk_utils import plot_1d_list_pdf, plot_1d_loglog_list_pdf
from plot_mk_utils import plot_1d_semilog_list_pdf, plot_2d_pdf


def plot_ff_norms_with_vspace(filename,ff,ffm,vpagrid,vperpgrid):
    # plot infinity norm
    pdffile = filename+".ffplots.pdf"
    pdf = PdfPages(pdffile)
    ffplot = np.abs(ff[:,:]-ffm[:,:])
    plot_2d_pdf(vpagrid,vperpgrid,ffplot,pdf,title="$|F-F_M|$",ylab="$v_{\\perp}$",xlab="$v_{||}$")
    pdf.close()
    print("Saving figures: "+pdffile)
    return None

def get_time_evolving_data(filename):
    print(filename)
    nz_global, nz_local, zgrid = grid_data(filename,"z")
    nr_global, nr_local, rgrid = grid_data(filename,"r")
    nvpa_global, nvpa_local, vpagrid = grid_data(filename,"vpa")
    nvperp_global, nvperp_local, vperpgrid = grid_data(filename,"vperp")
    vpawgts = wgts_data(filename,"vpa")
    vperpwgts = wgts_data(filename,"vperp")
    time, time_present = dynamic_data(filename,"time")
    ff, ff_present = dynamic_data(filename,"f")
    dsdt, dsdt_present = dynamic_data(filename,"entropy_production") 
    density, density_present = dynamic_data(filename,"density") 
    parallel_flow, parallel_flow_present = dynamic_data(filename,"parallel_flow") 
    parallel_pressure, parallel_pressure_present = dynamic_data(filename,"parallel_pressure") 
    perpendicular_pressure, perpendicular_pressure_present = dynamic_data(filename,"perpendicular_pressure") 
    if parallel_flow_present and perpendicular_pressure_present:
        pressure = (2.0*perpendicular_pressure + parallel_pressure)/3.0
    else:
        pressure = None
    thermal_speed, thermal_speed_present = dynamic_data(filename,"thermal_speed") 
    ntime = time.size
    nvperp = vperpgrid.size
    nvpa = vpagrid.size
    #print(dsdt)
    #print(density)
    #print(parallel_flow)
    #print(np.shape(thermal_speed))
    ffm = np.copy(ff)
    for it in range(0,ntime):
       	for ivperp in range(0,nvperp):
            for ivpa in range(0,nvpa):
                vth = thermal_speed[it,0,0,0]
                v2 = ((vpagrid[ivpa]-parallel_flow[it,0,0,0])/vth)**2 + (vperpgrid[ivperp]/vth)**2
                ffm[it,0,0,0,ivperp,ivpa] = density[it,0,0,0]*np.exp(-v2)/(vth**3)

    L2fm = np.copy(dsdt) 
    L2denom = np.sum(vperpwgts[:])*np.sum(vpawgts[:])
    #print(np.shape(L2fm))
    #print(np.shape(ff))
    for it in range(0,ntime):
        L2fm[it,0,0,0] = 0.0 
       	for ivperp in range(0,nvperp):
            for ivpa in range(0,nvpa):
                L2fm[it,0,0,0] += vperpwgts[ivperp]*vpawgts[ivpa]*(ff[it,0,0,0,ivperp,ivpa]-ffm[it,0,0,0,ivperp,ivpa])**2
                #continue
        L2fm[it,0,0,0] = np.sqrt(L2fm[it,0,0,0]/L2denom)
        
    Inffm = np.copy(dsdt) 
    for it in range(0,ntime):
        Inffm[it,0,0,0] = 0.0 
        Inffm[it,0,0,0] = np.max(np.abs(ff[it,0,0,0,:,:]-ffm[it,0,0,0,:,:]))
    #plot_ff_norms_with_vspace(filename,ff[it,0,0,0,:,:],ffm[it,0,0,0,:,:],vpagrid,vperpgrid)
    print("delta n: ", density[-1,0,0,0]-density[0,0,0,0])
    print("delta u: ", parallel_flow[-1,0,0,0]-parallel_flow[0,0,0,0])
    print("delta vth: ", thermal_speed[-1,0,0,0]-thermal_speed[0,0,0,0])
    #print("L2fm(t): ",L2fm[::50,0,0,0]," time: ",time[::50])
    print("L2fm: ", L2fm[0,0,0,0]," ",L2fm[-1,0,0,0])
    #print("Inffm(t): ",Inffm[::50,0,0,0]," time: ",time[::50])
    print("Inffm: ", Inffm[0,0,0,0]," ",Inffm[-1,0,0,0])
    return time, dsdt[:,0,0,0], L2fm[:,0,0,0], Inffm[:,0,0,0], density[:,0,0,0], parallel_flow[:,0,0,0], thermal_speed[:,0,0,0], pressure[:,0,0,0], vpagrid, vperpgrid, ff[-1,0,0,0,:,:], ffm[-1,0,0,0,:,:]

def save_plot_data(filename, time, dSdt, L2norm, Infnorm, dens, upar, vth, pres, 
        vpagrid, vperpgrid, ff, ffm):
        f = h5py.File(filename+".hdf5", "w")
        f.create_dataset("time",data=time)
        f.create_dataset("dSdt",data=dSdt)
        f.create_dataset("L2norm",data=L2norm)
        f.create_dataset("Infnorm",data=Infnorm)
        f.create_dataset("dens",data=dens)
        f.create_dataset("upar",data=upar)
        f.create_dataset("vth",data=vth)
        f.create_dataset("pres",data=pres)
        f.create_dataset("vpagrid",data=vpagrid)
        f.create_dataset("vperpgrid",data=vperpgrid)
        f.create_dataset("ff",data=ff)
        f.create_dataset("ffm",data=ffm)
        f.close()
        return None
        
def load_plot_data(filename):
        f = h5py.File(filename+".hdf5", "r")
        time = np.copy(f["time"][:])
        dSdt = np.copy(f["dSdt"][:])
        L2norm = np.copy(f["L2norm"][:])
        Infnorm = np.copy(f["Infnorm"][:])
        dens = np.copy(f["dens"][:])
        upar = np.copy(f["upar"][:])
        vth = np.copy(f["vth"][:])
        pres = np.copy(f["pres"][:])
        vpagrid = np.copy(f["vpagrid"][:])
        vperpgrid = np.copy(f["vperpgrid"][:])
        ff = np.copy(f["ff"][:,:])
        ffm = np.copy(f["ffm"][:,:])
        f.close()
        return time, dSdt, L2norm, Infnorm, dens, upar, vth, pres, vpagrid, vperpgrid, ff, ffm

time_list = []
Stime_list = []
Mtime_list = []
Mnoupar_time_list = []
dSdt_list = []
L2norm_list = []
Infnorm_list = []
dens_list = []
upar_list = []
vth_list = []
p_list = []
M_list = []
Mnoupar_list = []
M2_list = []
M2noupar_list = []

#input_raw_names = ["fokker-planck-relaxation-beam-init1",
#                   "fokker-planck-relaxation-beam-init2",
#                   "fokker-planck-relaxation-beam-init3"]
#input_raw_names = ["fokker-planck-relaxation-beam-init1long",
#                   "fokker-planck-relaxation-beam-init2long",
#                   "fokker-planck-relaxation-beam-init3long"]
#input_raw_names = ["fokker-planck-relaxation-no-dfdvperp1",
#                   "fokker-planck-relaxation-no-dfdvperp2",
#                   "fokker-planck-relaxation-no-dfdvperp3"]
#input_raw_names = ["fokker-planck-relaxation-no-dfdvperp-no-conserve1",
#                   "fokker-planck-relaxation-no-dfdvperp-no-conserve2",
#                   "fokker-planck-relaxation-no-dfdvperp-no-conserve3"]
workdir = ""
input_raw_names = ["fokker-planck-relaxation-flux-bc-only1",
                   "fokker-planck-relaxation-flux-bc-only2",
                   "fokker-planck-relaxation-flux-bc-only3"]
inputname_list = [workdir+instr+".toml" for instr in input_raw_names]
outfilename_list = [workdir+instr+"/"+instr+".dfns.0.h5" for instr in input_raw_names]
process_raw_data = True

for outfilename in outfilename_list:
    savefilename = outfilename[:-10]+".processed.h5"
    if process_raw_data:
        time, dSdt, L2norm, Infnorm, dens, upar, vth, pres, vpagrid, vperpgrid, ff, ffm = get_time_evolving_data(outfilename)
        print("Saving processed data: ",savefilename)
        save_plot_data(savefilename, time, dSdt, L2norm, Infnorm, dens, upar, vth, pres,
                        vpagrid, vperpgrid, ff, ffm)
    else:
        print("Loading pre-processed data: ",savefilename)
        time, dSdt, L2norm, Infnorm, dens, upar, vth, pres, vpagrid, vperpgrid, ff, ffm = load_plot_data(savefilename)
    
    plot_ff_norms_with_vspace(outfilename,ff,ffm,vpagrid,vperpgrid)
    time_list.append(time)
    Stime_list.append(time[1:])
    dSdt_list.append(dSdt[1:])
    L2norm_list.append(L2norm)
    Infnorm_list.append(Infnorm)
    dens_list.append(dens-dens[0])
    upar_list.append(upar-upar[0])
    vth_list.append(vth-vth[0])
    p_list.append(pres-pres[0])
    Mtime_list.append(time[:])
    Mtime_list.append(time[:])
    Mtime_list.append(time[:])
    M_list.append(np.abs(dens-dens[0]))
    M_list.append(np.abs(upar-upar[0]))
    M_list.append(np.abs(vth-vth[0]))
    Mnoupar_time_list.append(time[:])
    Mnoupar_time_list.append(time[:])
    Mnoupar_list.append(np.abs(dens-dens[0]))
    Mnoupar_list.append(np.abs(vth-vth[0]))
    M2_list.append(np.abs(dens-dens[0]))
    M2_list.append(np.abs(upar-upar[0]))
    M2_list.append(np.abs(pres-pres[0]))
    M2noupar_list.append(np.abs(dens-dens[0]))
    M2noupar_list.append(np.abs(pres-pres[0]))

file = workdir + "collisions_plots_many.pdf"
pdf = PdfPages(file)
tlabel = "$ \\nu_{ss} t $"
marker_list = ['k','r-.','b--']
ylab_list = ["#1", "#2", "#3"]
plot_1d_semilog_list_pdf (Stime_list,dSdt_list,marker_list,tlabel, pdf,
  title='$\\dot{S}$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="Resolutions", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)
plot_1d_semilog_list_pdf (time_list,L2norm_list,marker_list,tlabel, pdf,
  title='$L_2(F-F_M)$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="Resolutions", use_legend=True,loc_opt='lower right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.05), legend_fontsize=15, ncol_opt=1)
plot_1d_semilog_list_pdf (time_list,Infnorm_list,marker_list,tlabel, pdf,
  title='$L_{\infty}(F-F_M)$',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="Resolutions", use_legend=True,loc_opt='lower right', ylab_list = ylab_list,
  bbox_to_anchor_opt=(0.95, 0.05), legend_fontsize=15, ncol_opt=1)

Mylab_list = ["$|n(t)-n(0)|$ #1","$|u_{||}(t)- u_{||}(0)|$ #1","$|v_{\\rm th}(t) - v_{\\rm th}(0)|$ #1",
              "$|n(t)-n(0)|$ #2","$|u_{||}(t)- u_{||}(0)|$ #2","$|v_{\\rm th}(t) - v_{\\rm th}(0)|$ #2",
              "$|n(t)-n(0)|$ #3","$|u_{||}(t)- u_{||}(0)|$ #3","$|v_{\\rm th}(t) - v_{\\rm th}(0)|$ #3",]
marker_list = ['k','k-.','k--','r','r-.','r--','b','b-.','b--',]
print(len(Mtime_list))
print(len(M_list))
print(len(marker_list))
print(len(Mylab_list))
plot_1d_semilog_list_pdf (Mtime_list,M_list,marker_list,tlabel, pdf,
  title='',ylab='',xlims=None,ylims=[None,10**(0)],aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = Mylab_list,
  bbox_to_anchor_opt=(0.975, 0.975), legend_fontsize=15, ncol_opt=3)
  
Mylab_list = ["$|n(t)-n(0)|$ #1","$|v_{\\rm th}(t) - v_{\\rm th}(0)|$ #1",
              "$|n(t)-n(0)|$ #2","$|v_{\\rm th}(t) - v_{\\rm th}(0)|$ #2",
              "$|n(t)-n(0)|$ #3","$|v_{\\rm th}(t) - v_{\\rm th}(0)|$ #3",]
marker_list = ['k','k--','r','r--','b','b--',]
print(len(Mnoupar_time_list))
print(len(Mnoupar_list))
print(len(marker_list))
print(len(Mylab_list))
plot_1d_semilog_list_pdf (Mnoupar_time_list,Mnoupar_list,marker_list,tlabel, pdf,
  title='',ylab='',xlims=None,ylims=[None,10**(-1)],aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = Mylab_list,
  bbox_to_anchor_opt=(0.975, 0.975), legend_fontsize=15, ncol_opt=3)

Mylab_list = ["$|\\Delta n(t)|$ #1","$|\\Delta u_{||}(t)|$ #1","$|\\Delta p(t)|$ #1",
              "$|\\Delta n(t)|$ #2","$|\\Delta u_{||}(t)|$ #2","$|\\Delta p(t)|$ #2",
              "$|\\Delta n(t)|$ #3","$|\\Delta u_{||}(t)|$ #3","$|\\Delta p(t)|$ #3",]
marker_list = ['k','k-.','k--','r','r-.','r--','b','b-.','b--',]
print(len(Mtime_list))
print(len(M2_list))
print(len(marker_list))
print(len(Mylab_list))
#ylims = [None,10**(-7)]
#ylims = [10**(-14),10**(0)]
ylims = [None,10**(0)]
plot_1d_semilog_list_pdf (Mtime_list,M2_list,marker_list,tlabel, pdf,
  title='',ylab='',xlims=None,ylims=ylims,aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = Mylab_list,
  bbox_to_anchor_opt=(0.825, 1.0), legend_fontsize=15, ncol_opt=3)
  
Mylab_list = ["$|\\Delta n(t)|$ #1","$|\\Delta p(t)|$ #1",
              "$|\\Delta n(t)|$ #2","$|\\Delta p(t)|$ #2",
              "$|\\Delta n(t)|$ #3","$|\\Delta p(t)|$ #3",]
marker_list = ['k','k--','r','r--','b','b--',]
print(len(Mnoupar_time_list))
print(len(M2noupar_list))
print(len(marker_list))
print(len(Mylab_list))
plot_1d_semilog_list_pdf (Mnoupar_time_list,M2noupar_list,marker_list,tlabel, pdf,
  title='',ylab='',xlims=None,ylims=[None,10**(-1)],aspx=9,aspy=6, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = Mylab_list,
  bbox_to_anchor_opt=(0.8, 0.975), legend_fontsize=15, ncol_opt=3)
pdf.close()
print("Saving figure: "+file)
