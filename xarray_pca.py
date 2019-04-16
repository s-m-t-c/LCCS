import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

def xarray_pca(dataset, variables):

    """
    An xarray wrapper for scitkit learn PCA
    
    Parameters
    dataset: an xarray dataset with more than one data variable
    variables: a list of the variables you wish to conduct the PCA on
    """

    # Drop unecessary variables
    to_drop = []
    
    for var in list(dataset.data_vars):
        if var not in variables:
            to_drop.append(var)
    a = dataset.drop(to_drop)

    # Drop time dimension
    if len(a.time)==1:
        a = a.squeeze()

    # Convert dataset to dataarray with each variable broadcast against each other
    a = a.to_array()

    # # Stack the dimensions
    a = a.stack(features=['x', 'y'])#.values

    # Extract the underlying numpy array
    b = a.data

    # Number of samples
    n_samples = b.shape[0]

    # Number of features / attributes
    grid_shape = b.shape[1:]
    n_grids = np.prod(grid_shape)

    # Reshape array to samples, features
    b = b.reshape((n_samples, n_grids))

    # Boolean mask to remove Nans
    valid_grids = ~np.isnan(b[0,:])
    b = b[:,valid_grids]

    # Set PCA variables
    pca = PCA(n_components=1)

    # Fit PCA
    c = pca.fit(b)

    ### reshape the coefficient array back ###

    # Create empty array with the dimensions of the PCA output
    arr = np.empty((c.n_components_, n_grids)) * np.nan
    # Assign components from PCA to each grid location in empty array
    arr[:, valid_grids] = c.components_
    # Reshape array 
    arr = arr.reshape((c.n_components_,) + grid_shape)

    ### Reshape the mean_ ###

    # Create an empty array the size of the number of grids
    mean_ = np.empty(n_grids) * np.nan
    # Populate empty array with values
    mean_[valid_grids] = c.mean_
    # Resahpe to match grid size
    mean_ = mean_.reshape(grid_shape)

    ### wrap regression coefficient into DataArray ###

    # extract dimension names from input DataArray
    grid_dims = a.dims[1:]
    arr_dims = a.dims
    # Extract the coords for the features dimension
    grid_coords = {dim: a[dim] for dim in grid_dims}
    # Assign these to the output coords variable 
    arr_coords = grid_coords.copy()
    # Add an array to the output coords with the size of the out PCA dimensions and the key of the dimensions
    arr_coords[arr_dims[0]] = np.arange(c.n_components_)
    # Add new DataArray of components (directions of maximum variance int he data) to the PCA output
    c.components_da = xr.DataArray(arr,
        dims=arr_dims, coords=arr_coords)
    #A Add new Dataarray of per feature empirical mean_ to the PCA output
    c.mean_da = xr.DataArray(mean_,
        dims=grid_dims, coords=grid_coords)
    # Extract DataArray output
    d = c.mean_da.unstack()

    # Convert DataArray to dataset
    d = d.to_dataset(name='temp').transpose()
    d = d.to_array().squeeze()

#     e = c.components_da.unstack().isel(variable=0)
#     e = e.to_dataset(name='temp').transpose()
#     e = e.to_array().squeeze()
    return d
