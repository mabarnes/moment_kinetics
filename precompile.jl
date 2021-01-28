using Pkg
using PackageCompiler

# if "tmpdir" exists, then remove it to make way for clean creation
isdir("tmpdir") && rm("tmpdir", recursive=true)
# create a temporary directory called tmpdir, where a new environment will be
# activated and the new sysimage will be initially created
mkpath("tmpdir")
# activate a new environment for the temporary project directory
Pkg.activate("tmpdir")
# add the packages we wish to pre-compile
Pkg.add("NCDatasets")
Pkg.add("Plots")
Pkg.add("LsqFit")
packages = [:NCDatasets, :Plots, :LsqFit]
# create the sysimage 'moment_kinetics.so' in the base moment_kinetics source directory
# with the above pre-compiled packages
create_sysimage(packages; sysimage_path="moment_kinetics.so")
# remove the temporary directory
rm("tmpdir", recursive=true)
