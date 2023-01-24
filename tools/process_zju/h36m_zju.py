import os
import numpy as np
import torch
import tqdm

from body_model import SMPLlayer

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
DATA_CONFIG = {
    'S1': (150, 49),
    'S5': (250, 127),
    'S6': (150, 83),
    'S7': (200, 300),
    'S8': (250, 87),
    'S9': (260, 133),
    'S11': (200, 82),
}

def load_rest_pose_info(subject_id: str, body_model):
    data_dir = os.path.join(
        PROJECT_DIR, "data", "h36m_zju", "%s" % subject_id, "Posing", "new_params",
    )
    fp = os.path.join(data_dir, "0.npy")
    smpl_data = np.load(fp, allow_pickle=True).item()
    vertices, joints, joints_transform, bones_transform = body_model(
        poses=np.zeros((1, 72), dtype=np.float32),
        shapes=smpl_data["shapes"],
        Rh=np.zeros((1, 3), dtype=np.float32),
        Th=np.zeros((1, 3), dtype=np.float32),
        scale=1,
        new_params=True,
    )
    return (
        vertices.squeeze(0), 
        joints.squeeze(0), 
        joints_transform.squeeze(0),
        bones_transform.squeeze(0),
    )

def load_pose_info(subject_id: str, frame_id: int, body_model):
    data_dir = os.path.join(
        PROJECT_DIR, "data", "h36m_zju", "%s" % subject_id, "Posing", "new_params",
    )
    fp = os.path.join(data_dir, "%d.npy" % frame_id)
    smpl_data = np.load(fp, allow_pickle=True).item()
    vertices, joints, joints_transform, bones_transform = body_model(
        poses=smpl_data["poses"],
        shapes=smpl_data["shapes"],
        Rh=smpl_data["Rh"],
        Th=smpl_data["Th"],
        scale=1,
        new_params=True,
    )
    pose_params = torch.cat(
        [
            torch.tensor(smpl_data['poses']),
            torch.tensor(smpl_data['Rh']),
            torch.tensor(smpl_data['Th']),
        ], dim=-1
    ).float()
    return (
        vertices.squeeze(0),
        joints.squeeze(0),
        joints_transform.squeeze(0),
        pose_params.squeeze(0),
        bones_transform.squeeze(0),
    )

def cli(subect_id: str, frame_interval=5):
    print(f'processing subject {subject_id}')
    # smpl body model
    body_model = SMPLlayer(
        model_path=os.path.join(PROJECT_DIR, "data"), gender="neutral", 
    )

    n_train = DATA_CONFIG[subject_id][0]
    n_eval = DATA_CONFIG[subect_id][1]

    i_intv = frame_interval
    i = 0
    ni = n_train + n_eval

    # parsing frame ids
    meta_fp = os.path.join(
        PROJECT_DIR, "data", "h36m_zju", "%s" % subject_id, "Posing", "annots.npy"
    )
    meta_data = np.load(meta_fp, allow_pickle=True).item()
    frame_ids = list(
        int(img_data['ims'][0].split('/')[-1].split('.')[0])
        for img_data in meta_data['ims'][i:i + ni * i_intv][::i_intv]
    )


    # rest state info
    rest_verts, rest_joints, rest_tfs, rest_tfs_bone = (
        load_rest_pose_info(subject_id, body_model)
    )
    lbs_weights = body_model.weights.float()

    # pose state info
    verts, joints, tfs, params, tf_bones = [], [], [], [], []
    for frame_id in tqdm.tqdm(frame_ids):
        _verts, _joints, _tfs, _params, _tfs_bone = (
            load_pose_info(subject_id, frame_id, body_model)
        )
        verts.append(_verts)
        joints.append(_joints)
        tfs.append(_tfs)
        params.append(_params)
        tf_bones.append(_tfs_bone)
    verts = torch.stack(verts)
    joints = torch.stack(joints)
    tfs = torch.stack(tfs)
    params = torch.stack(params)
    tf_bones = torch.stack(tf_bones)

    data = {
        "lbs_weights": lbs_weights,  # [6890, 24]
        "rest_verts": rest_verts,  # [6890, 3]
        "rest_joints": rest_joints,  # [24, 3]
        "rest_tfs": rest_tfs,  # [24, 4, 4]
        "rest_tfs_bone": rest_tfs_bone, # [24, 4, 4]
        "verts": verts,  # [1470, 6890, 3]
        "joints": joints,  # [1470, 24, 3]
        "tfs": tfs,  # [1470, 24, 4, 4]
        "tf_bones": tf_bones,  # [1470, 24, 4, 4]
        "params": params  # [1470, 72 + 3 + 3]
    }
    save_path = os.path.join(
        PROJECT_DIR, "data", "h36m_zju", "%s" % subject_id, "Posing", "pose_data.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)


if __name__ == "__main__":
    for subject_id in DATA_CONFIG.keys():
        cli(subject_id)