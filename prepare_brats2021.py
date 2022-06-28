import os
import argparse
from certifi import where
import numpy as np

# to fix a bug with numpy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from parse_config import ConfigParser
import nibabel as nib

import matplotlib
import matplotlib.pyplot as plt  # we need to import this to make use of matplotlib.image.imsave

import multiprocessing
from functools import partial
from os.path import exists


def unpack_and_store_all(cases, raw_data_dir, processed_data_dir, n_slices_total):
    # constants
    modalities = ['flair', 't1', 't1ce', 't2']
    seg = 'seg'

    # if cases is not a list, make it a list
    if not isinstance(cases, list):
        cases = [cases]

    # loop over all cases
    for caseidx, case in enumerate(cases):
        # loop over all modalities + segmentation mask    
        for file_idx, s_type in enumerate(modalities + [seg]):   # enumerate([seg]): 
            nii = f'{raw_data_dir}/{case}/{case}_{s_type}.nii.gz'
            print(f'Processing {nii}')

            # use nibabel to unzip and load all 155 slices
            image_stack = nib.load(nii)
            image_stack_np = image_stack.get_fdata() # numpy array of image data
            image_affine = image_stack.affine
            n_slices = image_stack_np.shape[2]

            # loop over all 155 slices
            for slice_idx in range(n_slices):
                # save the image as .nii.gz (see https://www.programcreek.com/python/example/98178/nibabel.save)
                slice_img_np = image_stack_np[:, :, slice_idx]
                nft_img = nib.Nifti1Image(slice_img_np, image_affine)
                image_path = os.path.join(processed_data_dir, f'{case}_{slice_idx:03d}_{s_type}.nii.gz')
                nib.save(nft_img, image_path)

# we will store the images as PNG's
def main(config):
    data_raw_dir = config['prepare_dataset_location']['raw_data']
    data_processed_dir = config['prepare_dataset_location']['processed_data']
    training_split = config['prepare_dataset_location']['training_split']
    test_split = config['prepare_dataset_location']['test_split']
    validation_split = config['prepare_dataset_location']['validation_split']
    start_slice = 0
    end_slice = 154 # for now we use all slices
    slices_per_case = end_slice - start_slice + 1

    # assert sum of splits=1
    assert sum([training_split, test_split, validation_split]) == 1, 'Sum of splits must be 1'
    print("data_raw_dir:", data_raw_dir)
    print("data_processed_dir:", data_processed_dir)

    # get the list of all subfolders in the directory, keep only folders that start with BRaTS2021_0
    dir_names = sorted([name for name in os.listdir(data_raw_dir) if os.path.isdir(os.path.join(data_raw_dir, name))])
    cases = sorted([case for case in dir_names if case.startswith('BraTS2021_0')])

    n_cases = len(cases)
    total_slices = n_cases * slices_per_case
    print("Number of total slices: ", total_slices)

    # create train/test/validation folders for processed data
    if not os.path.exists(data_processed_dir):
        os.makedirs(data_processed_dir)
    for path in ['train/', 'test/', 'validation/']:
        t_path = os.path.join(data_processed_dir, path)

        if not os.path.exists(t_path):
            os.makedirs(t_path) 

    # create random indices for training, test, validation
    indices = []

    # if indices.txt does not exist
    if not exists(os.path.join('data/', 'indices.txt')):
        print("Could not find indices.txt, creating new one")
        indices = np.random.permutation(n_cases)

        # create indices.txt
        with open('data/indices.txt', 'w') as f:
            for idx in indices:
                f.write(f'{idx}\n')
    else:
        print("Found indices.txt, using it")
        # read indices.txt
        with open('data/indices.txt', 'r') as f:
            indices = [int(line.strip()) for line in f]

    # indices
    training_indices = indices[:int(n_cases * training_split)]
    test_indices = indices[int(n_cases * training_split):int(n_cases * (training_split + test_split))]
    validation_indices = indices[int(n_cases * (training_split + test_split)):]

    assert len(training_indices) + len(test_indices) + len(validation_indices) == n_cases, 'N indices train, test, valid (%d) + (%d) + (%d) != n_cases = (%d)' % (len(training_indices), len(test_indices), len(validation_indices), n_cases)

    # split the cases
    training_cases = [cases[i] for i in training_indices]
    test_cases = [cases[i] for i in test_indices]
    validation_cases = [cases[i] for i in validation_indices]

    # create multiprocessing pool
    n_processes = os.cpu_count()//2
    print("Number of processes: ", n_processes)
    pool = multiprocessing.Pool(processes=n_processes)

    # training data
    print("------------------ processing training data ------------------")
    train_part = partial(unpack_and_store_all, raw_data_dir=data_raw_dir, 
                        processed_data_dir=os.path.join(data_processed_dir, 'train/'),
                        n_slices_total=slices_per_case*len(training_cases)) 

    pool.map(train_part, training_cases)

    # test data
    print("------------------ processing test data ------------------")
    test_part = partial(unpack_and_store_all, raw_data_dir=data_raw_dir,
                        processed_data_dir=os.path.join(data_processed_dir, 'test/'),
                        n_slices_total=slices_per_case*len(test_cases))

    pool.map(test_part, test_cases)

    # validation data
    print("------------------ processing validation data ------------------")
    validation_part = partial(unpack_and_store_all, raw_data_dir=data_raw_dir,
                        processed_data_dir=os.path.join(data_processed_dir, 'validation/'),
                        n_slices_total=slices_per_case*len(validation_cases))

    pool.map(validation_part, validation_cases)


if __name__ == "__main__":
    print("Unpacking and storing the dataset files, this may take a minute...")

    # define command line arguments
    args = argparse.ArgumentParser(description='Prepare BRaTS2021 dataset')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    # we don't use these options in this script but otherwise the parser will complain
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

    print("Preparing dataset completed.")
