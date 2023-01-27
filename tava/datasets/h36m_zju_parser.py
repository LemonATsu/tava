import os

import cv2
import imageio
import numpy as np
import torch
import tava.datasets.zju_parser as zju_parser


class SubjectParser(zju_parser.SubjectParser):

    WIDTH = 1000
    HEIGHT = 1000

    def __init__(self, subject_id: str, root_fp: str, frame_interval: int):
        if not root_fp.startswith("/"):
            # allow relative path. e.g., "./data/zju/"
            root_fp = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "..", "..",
                root_fp,
            )
        self.frame_interval = frame_interval
        self.subject_id = subject_id
        self.root_fp = root_fp
        self.root_dir = os.path.join(root_fp, subject_id, "Posing")
        self.mask_dir = os.path.join(self.root_dir, "mask_cihp")
        self.smpl_dir = os.path.join(self.root_dir, "new_params")
        self.splits_dir = self.root_dir

        annots_fp = os.path.join(self.root_dir, "annots.npy")
        annots_data = np.load(annots_fp, allow_pickle=True).item()
        self.cameras = self._parse_camera(annots_data)

        self.image_files = np.array(
            #[[os.path.join(self.root_dir, fp) for fp in fps["ims"]] for fps in annots_data["ims"]], dtype=str
            [[fp for fp in fps["ims"]] for fps in annots_data["ims"][::frame_interval]], dtype=str
        )
        #self._frame_ids = list(range(self.image_files.shape[0]))
        self._frame_ids = list(np.arange(len(self.image_files)))
        self._camera_ids = list(range(self.image_files.shape[1]))

    def load_image(self, frame_id: int, camera_id: int):
        fp = os.path.join(self.root_dir, self.image_files[frame_id, camera_id])
        img = imageio.imread(fp)
        H, W, C = img.shape
        assert H == self.HEIGHT and W == self.WIDTH, f'{frame_id}:{camera_id} breaks'
        if H == 1001: 
            img = img[1:-1]
        elif W == 1001:
            img = img[:, 1:-1]
        return img
    
    def load_mask(self, frame_id, camera_id, trimap=True):
        path = os.path.join(
            self.mask_dir,
            self.image_files[frame_id, camera_id].replace(".jpg", ".png")
        )
        mask = (imageio.imread(path) != 0).astype(np.uint8) * 255
        H, W = mask.shape[:2]
        if H == 1001: 
            mask = mask[1:-1]
        elif W == 1001:
            mask = mask[:, 1:-1]
        if trimap:
            mask = self._process_mask(mask)
        return mask

    def load_meta_data(self, frame_ids=None):
        data = torch.load(os.path.join(self.root_dir, "pose_data.pt"))
        keys = [
            "lbs_weights",
            "rest_verts",
            "rest_joints",
            "rest_tfs",
            "rest_tfs_bone",
            "verts",
            "joints",
            "tfs",
            "tf_bones",
            "params",
        ]
        return {
            key: (
                data[key][frame_ids].numpy()
                if (
                    frame_ids is not None
                    and key in ["verts", "joints", "tfs", "tf_bones", "params"]
                )
                else data[key].numpy()
            )
            for key in keys
        }