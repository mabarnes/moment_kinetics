# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

using NCDatasets
using Plots

# get the run_name from the command-line
run_name = ARGS[1]

# create the netcdf filename from the given run_name
filename = string(run_name, ".cdf")
# open the netcdf file with given filename for reading
fid = NCDataset(filename,"r")

# define a handle for the z coordinate
cdfvar = fid["z"]
# get the number of z grid points
nz = length(cdfvar)
# load the data for z
z = cdfvar.var[:]

# define a handle for the vpa coordinate
cdfvar = fid["vpa"]
# get the number of vpa grid points
nvpa = length(cdfvar)
# load the data for vpa
vpa = cdfvar.var[:]

# define a handle for the time coordinate
cdfvar = fid["time"]
# get the number of time grid points
ntime = length(cdfvar)
# load the data for time
time = cdfvar.var[:]

# define a handle for the electrostatic potential
cdfvar = fid["phi"]
# load the electrostatic potential data
phi = cdfvar.var[:,:]

# make a heatmap plot of ϕ(z,t)
heatmap(time, z, phi, xlabel="time", ylabel="z", title="ϕ", c = :deep)
outfile = string(run_name, "_phi_vs_z_t.pdf")
savefig(outfile)
# make a gif animation of ϕ(z,t)
anim = @animate for i ∈ 1:ntime
    plot(z, phi[:,i], xlabel="z", ylabel="ϕ")
end
outfile = string(run_name, "_phi_vs_z.gif")
gif(anim, outfile, fps=5)

# define a handle for the distribution function
cdfvar = fid["f"]
# load the distribution function data
ff = cdfvar.var[:,:,:]

# make a gif animation of f(vpa,z,t)
anim = @animate for i ∈ 1:ntime
    heatmap(vpa, z, ff[:,:,i], xlabel="vpa", ylabel="z", clims = (0,1), c = :deep)
end
outfile = string(run_name, "_f_vs_z_vpa.gif")
gif(anim, outfile, fps=5)

close(fid)
