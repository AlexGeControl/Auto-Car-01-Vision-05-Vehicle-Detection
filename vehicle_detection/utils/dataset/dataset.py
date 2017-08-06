## Set up session:
import h5py
import numpy as np

def to_hdf5(features, labels, hdf5_filename, dataset_name, write_method="w"):
    """ Save dataset as HDF5 file
    """

    db = h5py.File(hdf5_filename, write_method)
    dataset = db.create_dataset(
        dataset_name,
        (len(features), len(features[0]) + 1),
        dtype = "f"
    )
    # Features come first, then labels:
    dataset[0:len(features)] = np.c_[features, labels]
    db.close()

def read_hdf5(hdf5_filename, dataset_name):
    """ Load dataset from HDF5 file
    """
    db = h5py.File(hdf5_filename, 'r')
    dataset = db[dataset_name]
    (features, labels) = dataset[:, :-1], dataset[:, -1]

    return (features, labels)
