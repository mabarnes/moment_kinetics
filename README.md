# moment_kinetics
0) Ensure that the Julia version is >= 1.6.1 by doing `julia --version` at command line. 
1) To install dependencies, run 'julia -e "import Pkg; Pkg.add(\"PackageCompiler\"); Pkg.add(\"TimerOutputs\"); Pkg.add(\"NCDatasets\"); Pkg.add(\"FFTW\"); Pkg.add(\"Plots\"); Pkg.add(\"LsqFit\"); Pkg.add(\"OrderedCollections\"); Pkg.add(\"Glob\"); Pkg.add(\"NaturalSort\"); Pkg.add(\"SpecialFunctions\"); Pkg.add(\"Roots\")"'.
2) To pre-compile a static image that includes a few of the external packages required for post-processing, run 'julia precompile.jl'.
3) Create a subdirectory to store run output, 'mkdir runs'.
4) To run julia with optimization, type 'julia -O3 --package -Jmoment_kinetics.so run_moment_kinetics.jl'.  Input options can be specified in moment_kinetics_input.jl.
5) To make plots and calculate frequencies/growth rates, type 'julia --package -Jmoment_kinetics.so run_post_processing.jl'. Pass the directory to process as a command line argument. Input options for post-processing can be specified in post_processing_input.jl.
4b) Parameter scans or performance tests can be performed by running driver.jl. If running a scan, it can be parallelised by passing the number of processors as an argument. Scan options are set in scan_inputs.jl.
5b) Post processing can be done for several directories at once using 'julia --package -Jmoment_kinetics.so post_processing_driver.jl'. Pass the directories to process as command line arguments. Optionally pass a number as the first argument to parallelise post processing of different directories. Input options for post-processing can be specified in post_processing_input.jl.

## Running parameter scans
Parameter scans can be run, and can (optionally) use multiple processors. Short summary of implementation and usage:
1) mk_input() now takes a Dict argument, which can modify values. So mk_input() sets the 'defaults' (for a scan), which are overridden by any key/value pairs in the Dict.
2) mk_scan_inputs() (in scan_input.jl) creates an Array of Dicts that can be passed to mk_input(). It first creates a Dict of parameters to scan over (keys are the names of the variable, values are an Array to scan over), then assembles an Array of Dicts (where each entry in the Array is a Dict with a single value for each variable being scanned). Most variables are combined as an 'inner product', e.g. {:ni=>[0.5, 1.], :nn=>[0.5, 0.]} gives [{:ni=>0.5, :nn=>0.5}, {ni=>1., nn=>0.}]. Any special variables specified in the 'combine_outer' array are instead combined with the rest as an 'outer product', i.e. an entry is created for every value of those variables for each entry in the 'inner-producted' list. [This was just complicated enough to run the scans I've done so far without wasted simulations.]
3) The code in 'driver.jl' picks between a single run (normal case), a performance_test, or creating a scan by calling mk_scan_input() and then looping over the returned array, calling mk_input() and running a simulation for each entry. This loop is parallelised (with the set of simulations dispatched over several processes - each simulation is still running serially). Running a scan (on 12 processes - actually 13 but the 'master' process doesn't run any of the loop bodies, so there are 12 'workers'):
```
julia -O3 --package -Jmoment_kinetics.so driver.jl 12
```
(runs in serial if no argument is given)
4) The scan puts each run in a separate directory, named with a prefix specified by 'base_name' in scan_input.jl and the rest the names and values of the scanned-over parameters (the names are created in mk_scan_input() too, and passed as the :run_name entry of the returned Dicts).
5) To run post_processing.analyze_and_plot_data() over a bunch of directories (again parallelized trivially, and the number of processes to use is an optional argument, serial if omitted):
```
julia -O3 --package -Jmoment_kinetics.so post_processing_driver.jl 12 runs/scan_name_*
```
6) Plotting the scan is not so general, plot_comparison.jl does it, but is only set up for the particular scans I ran - everything except the charge exchange frequencies is hard-coded in.
```
julia -O3 --package -Jmoment_kinetics.so plot_comparison.jl
```
