from tracemalloc import start
import torch
import nibabel as nib
from torch.utils import data
import os
import numpy as np
import torch.nn.functional as F
import cv2

# from transformations.transformations import normalize_01
import torchvision.transforms as transforms

from time import sleep

class Bats2021Dataset(data.Dataset):
    def __init__(self,          
                data_dir: str,
                split: str='train',
                transform=None,
                start_slice=0,
                end_slice=154
                ):

        assert split in ['train', 'test', 'validation'], 'Only train, test, validation are supported options.'
        assert os.path.exists(data_dir), 'data_dir path does not exist: {}'.format(data_dir)

        print('Loading dataset from: {}'.format(data_dir+'/'+split+'/'))

        self.data_dir = data_dir
        self.split = split
        self.data_dir_img = os.path.join(data_dir, split)
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

        # MRI modalities
        self.modalities = ['t1', 't1ce', 't2', 'flair'] 

        # slicing
        self.start_slice = start_slice
        self.end_slice = end_slice
        self.slices_per_case = self.end_slice - self.start_slice + 1

        # filenames
        self.file_names = sorted(os.listdir(self.data_dir_img))
        self.n_files = len(self.file_names)
        self.n_cases = self.n_files // (155*5)
        print('Number of cases: {}'.format(self.n_cases))
        self.n_total_slices = self.n_cases * self.slices_per_case     # total number of slices that we actually use
        print('Total number of slices: {}'.format(self.n_total_slices))

        # get the list of all the cases (example=BraTS2021_00021_flair_0) we only need 'BraTS2021_xxxxx'
        self.cases = sorted([self.file_names[i][:15] for i in range(0, self.n_files, 155*5)])
        assert len(self.cases) == self.n_cases, 'Number of cases (%d) does not match the number of casefiles (%d)' % (len(self.cases), self.n_cases)

        print('cases: {}'.format(self.cases))

    def __len__(self):
        return self.n_total_slices

    def __getitem__(self,
                    index: int):     
        # convert index to case lookup
        case_idx = index // self.slices_per_case
        slice_idx = index % self.slices_per_case + self.start_slice # add start_slice because slices are in range (start_slice, end_slice)

        # assert that the index is in range (start_slice, end_slice) and case is in range (0, n_cases) 
        assert slice_idx >= self.start_slice and slice_idx <= self.end_slice, 'Index out of range'
        assert case_idx >= 0 and case_idx < self.n_cases, 'Case index out of range'

        # get file names for the case
        case_filename = self.cases[case_idx]

        # placeholders 
        modality_dict = {}

        ### load all 4 MRI modalities: t1, t1ce, t2, FLAIR ###
        for modality in self.modalities:
            # nii (Nifti) format
            filename = f'{case_filename}_{slice_idx:03d}_{modality}.nii.gz'
            img_path = os.path.join(self.data_dir_img, filename)
            img = nib.load(img_path).get_fdata()
            img = torch.from_numpy(img)

            # store in dict
            modality_dict.update({modality: img})

        # load seg mask png (groundtruth)   
        seg_filename = f'{case_filename}_{slice_idx:03d}_seg.nii.gz'

        seg_path = os.path.join(self.data_dir_img, seg_filename)
        seg = nib.load(seg_path).get_fdata()
        seg = torch.from_numpy(seg)
        seg[seg == 4] = 3  # convert from 0, 1, 2, 4 --> 0, 1, 2, 3
        modality_dict.update({'seg': seg})
        
        # preprocessing
        if self.transform is not None:
            data_dict = self.transform(modality_dict)

            seg = data_dict['seg']

            # write data_dict to modality_dict excluding seg
            del data_dict['seg']
            modality_dict = data_dict

        # convert the segmentation to one-hot encoding, 4 classes: healthy/BG, necrotic, edema, enhancing
        y_in = seg.to(torch.int64)
        y_one_hot = F.one_hot(y_in, num_classes=4).squeeze(0)
        y = torch.transpose(y_one_hot, 0, 2)  

        # concatenate all 4 modalities
        x = torch.cat((modality_dict['flair'], 
                       modality_dict['t1'], 
                       modality_dict['t1ce'], 
                       modality_dict['t2']), dim=0)

        # assert that x and y have the same shape
        assert x.shape == y.shape, 'x and y have different shapes'        

        return x, y
