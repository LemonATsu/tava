import os

import cv2 
import numpy as np
import torch
from tava.datasets.abstract import CachedIterDataset
import tava.datasets.zju_loader as zju_loader

from tava.datasets.h36m_zju_parser import SubjectParser


def _dataset_view_split(parser, split):
    _train_camera_ids = [0, 1, 2]
    if split == "all":
        camera_ids = parser.camera_ids
    elif split == "train":
        camera_ids = _train_camera_ids
    elif split in ["val_ind", "val_ood", "val_view"]:
        camera_ids = list(set(parser.camera_ids) - set(_train_camera_ids))
    elif split == "test":
        camera_ids = [3]
    return camera_ids

def _dataset_frame_split(parser, split):
    if split in ['train']:
        splits_fp = os.path.join(parser.root_dir, 'train.txt')
    else:
        raise NotImplementedError('error')
    with open(splits_fp, mode='r') as fp:
        frame_list = np.loadtxt(fp, dtype=int).tolist()
    return frame_list

def _dataset_index_list(parser, split):
    camera_ids = _dataset_view_split(parser, split)
    frame_list = _dataset_frame_split(parser, split)
    index_list = []
    for frame_id in frame_list:
        index_list.extend([(frame_id, camera_id) for camera_id in camera_ids])
    return index_list

class SubjectLoader(zju_loader.SubjectLoader):
    SPLIT = ['train', 'test']

    def __init__(
        self, 
        subject_id: str, 
        root_fp: str, 
        split: str, 
        resize_factor: float = 1, 
        color_bkgd_aug: str = None, 
        num_rays: int = None, 
        cache_n_repeat: int = 0, 
        near: float = None, 
        far: float = None, 
        legacy: bool = False,
        **kwargs
    ):
        assert split in self.SPLIT, '%s' % split
        assert color_bkgd_aug in ['white', 'black', 'random']
        self.resize_factor = resize_factor
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.legacy = legacy
        self.training = (num_rays is not None) and (split in ["train", "all"])
        self.color_bkgd_aug = color_bkgd_aug
        self.parser = SubjectParser(subject_id=subject_id, root_fp=root_fp)
        self.index_list = _dataset_index_list(self.parser, split)
        self.dtype = torch.get_default_dtype()
        CachedIterDataset.__init__(self, self.training, cache_n_repeat)