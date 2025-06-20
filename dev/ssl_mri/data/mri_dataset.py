# coding=utf-8
import os
from typing import Callable, Optional, Sequence, Union
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
# from mmap_ninja.ragged import RaggedMmap
import glob
import nibabel as nib
from tqdm import tqdm
from monai.data.dataset import Dataset as monaiDataset
import random

def read_csv(filename, num_classes=3, return_dicts=False):
    print('------------read csv-------------')
    print('filename: ', filename)
    
    df = pd.read_csv(filename)

    filenames = [''.join([row['path'],row['filename']]) for _,row in df.iterrows()]
    if num_classes == 3:
        print('getting labels for 3way classification')
        labels = torch.LongTensor([0 if int(row['NC']) else (1 if int(row['MCI']) else 2) for _,row in df.iterrows()])
    else:
        print('getting labels for 2way classification')
        labels = torch.LongTensor([0 if int(row['NC']) else 2 for _,row in df.iterrows()])
    print('-> uniques: ', torch.unique(labels,sorted=True))

    if return_dicts:
        return [{'image':f, 'label':l} for (f,l) in zip(filenames, labels.tolist())]
    return filenames, labels

def read_df(filename, return_dicts=True, labels=None, mri_seq='T1', multilabel=False, stripped=False):
    df = pd.read_csv(filename)
    fnames = []
    label_list = []
    # ic(df.columns)
    # df = df.iloc[:20]
    for idx, row in tqdm(df.iterrows()):
        # print('filename: ', row['filename'])
        # print('path: ', row['path'])
        # print('mri_zip: ', row['mri_zip'])
        # print('filename_vit_emb: ', row['filename_vit_emb'])
        if not pd.isna(row['mri_zip']):
            if os.uname().nodename == 'echo':
                mri_path = f"/SeaExpCIFS/NACC_updated/raw/{mri_seq}/{row['mri_zip'][:-4]}ni"
            else:
                mri_path = f"/projectnb/ivc-ml/dlteif/NACC_raw/{mri_seq}/{row['mri_zip'][:-4]}ni"
            if os.path.exists(mri_path):
                files = glob.glob(f"{mri_path}/**/*.nii", recursive=True)
                # ic(files)
                for f in files:
                    mri = nib.load(f)
                    if len(np.squeeze(mri.get_fdata()).shape) == 3:
                        # raw_mris.append(mri[None,...])
                        # mri = mri[None,...]
                        fnames.append(f.replace(".nii", "_stripped.nii") if stripped else f)
                        if labels is not None:
                            # ic(labels)
                            if multilabel:
                                label_list.append({l: row[l] for l in labels})
                            else:
                                for i in range(len(labels)):
                                    if row[labels[i]]:
                                        label_list.append(i)
                                        break
                        break

    if return_dicts:
        return [{'image':f, 'label':l} for (f,l) in zip(fnames, label_list)]
    return filenames, labels


def get_fpaths(data_dir, stripped=False, modality='t1ce'):
    """
    Get file paths for BraTS2020 dataset
    Args:
        data_dir: Path to BraTS2020_TrainingData directory
        stripped: Not used for BraTS (kept for compatibility)
        modality: Which modality to use ('t1', 't1ce', 't2', 'flair', 'all')
    Returns:
        List of dictionaries with image paths
    """
    fpaths_file = os.path.join(data_dir, f'fpaths_brats_{modality}.txt')

    if os.path.exists(fpaths_file):
        print(f"Loading cached file paths from {fpaths_file}")
        fpaths = list(open(fpaths_file, 'r').read().split('\n'))
        if modality == 'all':
            # For multi-modal, each line contains 4 paths separated by ','
            fpaths = [{"image": f.split(',')} for f in fpaths if f.strip()]
        else:
            fpaths = [{"image": f} for f in fpaths if f.strip()]
    else:
        print(f"Scanning BraTS2020 directory: {data_dir}")
        fpaths = []

        # BraTS2020 directory structure: BraTS20_Training_XXX/
        for subject_dir in sorted(os.listdir(data_dir)):
            subject_path = os.path.join(data_dir, subject_dir)
            if os.path.isdir(subject_path) and subject_dir.startswith('BraTS20_Training_'):

                if modality == 'all':
                    # Get all 4 modalities for multi-modal training
                    t1_path = os.path.join(subject_path, f"{subject_dir}_t1.nii")
                    t1ce_path = os.path.join(subject_path, f"{subject_dir}_t1ce.nii")
                    t2_path = os.path.join(subject_path, f"{subject_dir}_t2.nii")
                    flair_path = os.path.join(subject_path, f"{subject_dir}_flair.nii")

                    # Check if all modalities exist
                    if all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path]):
                        fpaths.append({"image": [t1_path, t1ce_path, t2_path, flair_path]})
                        print(f"Found multi-modal sample: {subject_dir}")
                else:
                    # Get single modality
                    modality_path = os.path.join(subject_path, f"{subject_dir}_{modality}.nii")
                    if os.path.exists(modality_path):
                        fpaths.append({"image": modality_path})
                        print(f"Found {modality} file: {modality_path}")

        # Save the file paths for future use
        if fpaths:
            with open(fpaths_file, 'w') as f:
                if modality == 'all':
                    f.write('\n'.join([','.join(f["image"]) for f in fpaths]))
                else:
                    f.write('\n'.join([f["image"] for f in fpaths]))
            print(f"Saved {len(fpaths)} file paths to {fpaths_file}")

    print(f"Total {modality} samples found: {len(fpaths)}")
    return fpaths
                


class ImageDataset(Dataset):
    def __init__(self, dataset, labels=None, transform=None, target_transform=None, indices=None, filename=None, num_classes=3):
        self.domain_num = 0
        self.dataset = dataset
        self.filename = filename
        print('filename: ', filename)
        self.fnames, labels = read_csv(filename, num_classes=num_classes)    
        self.x = self.fnames
        # self.x = np.asarray([np.load(f.replace('.nii', '.npy'), mmap_mode='r+') for f in self.fnames])
        # mmap_gen = RaggedMmap.from_generator(out_dir='/data_1/dlteif/images_mmap',
        #                                    sample_generator=map(np.load, self.fnames),
        #                                    batch_size=1024,
        #                                    verbose=True)
        # self.x = RaggedMmap('/data_1/dlteif/images_mmap')
        
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(self.fnames))
        else:
            self.indices = indices

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        # print(len(self.x), index)
        data = np.load(self.x[index].replace('.nii', '.npy'))
        # data = np.memmap(self.x[index].replace('.nii', '.npy'), mode='r', dtype='float32', shape=(182,218,182))
        # data = self.x[index]
        ic(data.shape)
        data = np.expand_dims(data, axis=0)
        ic(data.shape)
        img = self.input_trans(torch.from_numpy(data))
        # img = self.input_trans(data[0,:,:,None])
    
        target = self.target_trans(self.labels[index])
        return self.fnames[index].replace('.nii', '.npy'), img, target

    def __len__(self):
        return len(self.indices)

    def get_sample_weights(self):
        print('------------def get_sample_weights--------------')
        label_list = self.labels.tolist()
        count = float(len(label_list))
        print('total count: ', count)
        print(sorted(list(set(label_list))))
            
        uniques = sorted(list(set(label_list)))
        print('uniques: ',  uniques)
        counts = [float(label_list.count(i)) for i in uniques]
        print('counts: ', counts)
        
        weights = [count / counts[i] for i in label_list]
        # print('weights: ', weights)
        return weights, counts

   
class MonaiDataset(monaiDataset):
    def __init__(self, data, transform):
        super().__init__(data, transform)

    def _transform(self, index):
        """
        Fetch single data item from `self.data`.
        """
        try:
            return super()._transform(index) #["image"]
        except (RuntimeError, TypeError) as e:
            ic(e)
            ic(type(self.data[index]))
            # exit()
            # index = random.randint(0,len(self.data)-1)
            # ic(index)
            # return self._transform(index)
            return None 