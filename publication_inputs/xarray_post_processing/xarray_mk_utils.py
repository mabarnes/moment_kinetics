import xarray as xr


def read_variable(dataset,varstring):
    try:
        var=dataset[varstring].data
        var_present=True
    except KeyError:
            print('INFO: '+varstring+' not found in data file')
            var=None
            var_present = False
    return var, var_present
    
def grid_data(filename,coord):
    dataset = xr.open_dataset(filename,group ="coords/"+coord)
    n_global = dataset["n_global"].data
    n_local = dataset["n_local"].data
    grid = dataset["grid"].data
    dataset.close()
    return n_global, n_local, grid

def wgts_data(filename,coord):
    dataset = xr.open_dataset(filename,group ="coords/"+coord)
    wgts = dataset["wgts"].data
    dataset.close()
    return wgts
    
def dynamic_data(filename,varstring):
    dataset = xr.open_dataset(filename,group ="dynamic_data/")
    var, var_present = read_variable(dataset,varstring)
    return var, var_present
    
    
