"""
Script to adapt a .mat files containing a dataset from another .mat file to be used in the toolbox

Usage:
    1) activate environment
    2) run the following command from project root
    3) run from terminal
    $ python utils/adapt_data_for_hysupp.py --name_origin_data Toto

It assumes that Toto is located in the $PROJECT_ROOT/data folder

Ex: for Urban4 must be in $PROJECT_ROOT/data/Urban4/ 

"""
import scipy.io as sio

import os
import argparse
import numpy as np


def check_objects_shape_and_reshape(Y, E, A, H, W, p, L):
    """
        Check shape of objects.
    """
    assert len(E.shape) == 2
    if E.shape[0] < E.shape[1]:  # p x L
        E = E.T  # L x p

    L, p = E.shape

    if len(Y.shape) == 3:

        if Y.shape[0] == L:
            if H is not None:
                assert Y.shape[1] == H
            if W is not None:
                assert Y.shape[2] == W
            H, W = Y.shape[1], Y.shape[2]
            N = H * W
            Y = Y.reshape(L, N)
        
        elif Y.shape[2] == L:
            if H is not None:
                assert Y.shape[0] == H
            if W is not None:
                assert Y.shape[1] == W
            H, W = Y.shape[0], Y.shape[1]
            N = H * W
            Y = Y.reshape(N, L).T

        else:
            raise ValueError

    elif len(Y.shape) == 2:
        if Y.shape[0] != L:  # N x L
            N = Y.shape[0]
            Y = Y.T  # L x N
    else:
        raise ValueError("Invalid shape for Y...")

    if len(A.shape) == 3:

        if A.shape[0] == p:
            assert A.shape[1] * A.shape[2] == N
            A = A.reshape(p, A.shape[1] * A.shape[2])

        elif A.shape[2] == p:
            A = A.transpose((2, 0, 1))
            A = A.reshape(p, N)

        else:
            raise ValueError("Corner case not handled...")

    elif len(A.shape) == 2:

        if A.shape[0] != p:  # N x p
            A = A.T  # p x N
    
    else:
        raise ValueError("Invalid shape for A...")

    return Y, E, A, H, W, p, L


def get_and_adapt_samson(base_dir, origin_dataset_name):

    img_filename = "samson_1.mat"
    gt_filename  = "end3.mat"

    img = sio.loadmat(os.path.join(base_dir + img_filename))
    gt  = sio.loadmat(os.path.join(base_dir + gt_filename ))

    n_rows      = int(img["nRow"][0][0])
    n_cols      = int(img["nCol"][0][0])
    n_samples   = n_rows * n_cols
    n_bands     = int(img["nBand"][0][0])
    n_sources   = gt["A"].shape[0]

    # Spectra matrix
    Y = np.reshape(img["V"].T, (n_rows, n_cols, n_bands))
    # Abundance matrix
    A = gt["A"] 
    # Endmembers matrix
    E = gt["M"]

    H = n_cols
    W = n_rows

    p = n_sources
    L = n_bands

    Y, E, A, H, W, p, L = check_objects_shape_and_reshape(Y, E, A, H, W, p, L)

    data = {
        # observations - shape ( n_bands x n_samples )
        "Y": Y,
        # Endmembers - shape ( n_bands x n_ems )
        "E": E,
        # Abundances - shape ( n_ems x n_samples )
        "A": A,
        # Dim of image
        "H": n_cols,
        "W": n_rows,
        # nb ems
        "p": n_sources,
        # bands
        "L": n_bands,
    }

    sio.savemat(os.path.join("./data/", as_matlab("adapted_for_hysupp_" + origin_dataset_name)), data)


def get_and_adapt_urban(base_dir, origin_dataset_name):

    img_filename = "Urban_R162.mat"
    
    if   "4" in origin_dataset_name:
        gt_filename  = "end4_groundTruth.mat"

    elif "5" in origin_dataset_name:
        gt_filename  = "end5_groundTruth.mat"

    elif "6" in origin_dataset_name:
        gt_filename  = "end6_groundTruth.mat"

    else:
        raise ValueError

    img = sio.loadmat(os.path.join((base_dir + img_filename)))
    gt  = sio.loadmat(os.path.join((base_dir + gt_filename )))

    H = int(img["nCol"][0][0])
    W = int(img["nRow"][0][0])
    L = int(img["Y"].shape[0])
    p = int(gt["nEnd"][0][0])
    
    # Normalize observations
    max_value = img["maxValue"][0]
    Y = img["Y"].T.reshape((W, H, L)) / max_value
    # Abundance matrix
    A = gt["A"] 
    # Endmembers matrix
    E = gt["M"]


    Y, E, A, H, W, p, L = check_objects_shape(Y, E, A, H, W, p, L)

    data = {
        # observations - shape ( n_bands x n_samples )
        "Y": Y,
        # Endmembers - shape ( n_bands x n_ems )
        "E": E,
        # Abundances - shape ( n_ems x n_samples )
        "A": A,
        # Dim of image
        "H": H,
        "W": W,
        # nb ems
        "p": p,
        # nb bands
        "L": L,
    }

    sio.savemat(os.path.join("./data/", as_matlab("adapted_for_hysupp_" + origin_dataset_name)), data)


def as_matlab(key):
    return f"{key}.mat"


def main(args):

    origin_dataset_name = args.name_origin_data

    base_dir = f"./data/{origin_dataset_name}/"


    if origin_dataset_name == "Samson":
        get_and_adapt_samson(base_dir, origin_dataset_name)

    elif "Urban" in origin_dataset_name:
        get_and_adapt_urban(base_dir, origin_dataset_name)

    else:
        raise NotImplementedError
    


if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description="Data bundler")
    parser.add_argument("--name_origin_data",  required=True, help="Name of the file to adapt. Available options: Samson, Urban4, Urban5, Urban6.")

    args = parser.parse_args()

    main(args)
