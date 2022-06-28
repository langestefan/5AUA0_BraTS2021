
from torchvision import transforms
from base import BaseDataLoader
from data_loader.bats2021_dataset import Bats2021Dataset

from transformations.transformations import brats_train_transform, brats_validation_transform



class BaTS2021DataLoader(BaseDataLoader):
    """
    BaTS 2021 Data Loader.
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, split='train',
                 prefetch_factor=2, start_slice=0, end_slice=154):

        train_transform = transforms.Compose([
            brats_train_transform(image_keys=['t1', 't1ce', 't2', 'flair'], 
                                  all_keys=['t1', 't1ce', 't2', 'flair', 'seg'],)
        ])

        validation_transform = transforms.Compose([
            brats_validation_transform(image_keys=['t1', 't1ce', 't2', 'flair'], 
                                       all_keys=['t1', 't1ce', 't2', 'flair', 'seg'],)
        ])

        # Store the train/test/validation dataset location
        self.data_dir = data_dir

        # train dataset
        if split == 'train':
            self.dataset = Bats2021Dataset(self.data_dir, start_slice=start_slice, end_slice=end_slice,
                                           split=split, transform=train_transform)
        # test/validation dataset
        elif split == 'test' or split == 'validation':
            self.dataset = Bats2021Dataset(self.data_dir, start_slice=start_slice, end_slice=end_slice,
                                           split=split, transform=validation_transform)
        else:
            raise ValueError('Invalid split name: {}'.format(split))

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, prefetch_factor=prefetch_factor)