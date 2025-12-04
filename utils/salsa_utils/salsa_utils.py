# Todo: Implemet data loader preprocess and Salsa SMPL to HumanML3D representation.
from email.base64mime import body_encode
from glob import glob

from smplx import SMPLX
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import HumanMl3D_functions as HM3D_F

import lmdb
import pyarrow
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import torchaudio
import argparse
import math
from tqdm import tqdm
print(os.getcwd())
# Optional import for HumorRenderer (only used in sanity_check_vide)
HumorRenderer = None
try:
    # Try absolute import (when run from project root)
    from utils.salsa_utils.libs.MotionScript.stmc_renderer.humor import HumorRenderer
except ImportError:
    try:
        # Try relative import (when run from Salsa-Agent directory)
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to Salsa-Agent/utils/salsa_utils, then to Salsa-Agent
        salsa_agent_dir = os.path.dirname(os.path.dirname(current_dir))
        if salsa_agent_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(salsa_agent_dir))  # Add parent (New_2025)
        from utils.salsa_utils.libs.MotionScript.stmc_renderer.humor import HumorRenderer
    except ImportError:
        print("Warning: HumorRenderer not available (only needed for sanity_check_vide)")

def _get_smplx_path():
    """Helper function to find SMPLX model path"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'body_model', 'smplx'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))), 'Salsa-Agent', 'body_model', 'smplx'),
        'body_model/smplx',
        'body_model/models_lockedhead/smplx',
        'SMPLX_DEP/models_lockedhead/smplx',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"SMPLX model not found. Tried: {possible_paths}")

def salsa_smplx_to_pos3d(data):
    smplx = None
    joints_num = 22 # to be consistent with HumanML3D
    frames = data['poses'].shape[0]
    b = np.repeat(data['betas'][:10], frames).reshape((frames, 10))
    smplx_path = _get_smplx_path()
    smplx = SMPLX(model_path=smplx_path, betas=b,
                  gender=np.array2string(data['gender'])[1:-1], \
                  batch_size=len(b), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)



    keypoints3d = smplx.forward(
        # global_orient=torch.from_numpy(data['global_orient']).float(),

        global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
        body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
        jaw_pose=torch.from_numpy(data['poses'][:, 66:69]).float(),
        leye_pose=torch.from_numpy(data['poses'][:, 69:72]).float(),
        reye_pose=torch.from_numpy(data['poses'][:, 72:75]).float(),
        left_hand_pose=torch.from_numpy(data['poses'][:, 75:120]).float(),
        right_hand_pose=torch.from_numpy(data['poses'][:, 120:]).float(),
        transl=torch.from_numpy(data['trans']).float(),  # transl=torch.from_numpy(data['transl']).float(),
        # betas=torch.from_numpy(data['betas'][:10]).float()
        betas=torch.from_numpy(b).float()
    ).joints.detach().numpy()[:, :joints_num]
    # ours is (N, 144, 3)
    #
    # nframes = keypoints3d.shape[0]
    # keypoints3d = keypoints3d.reshape(nframes, -1)
    #
    # # Calculate relative offset with respect to root
    # root = keypoints3d[:, :3]  # the root
    # keypoints3d = keypoints3d - np.tile(root, (1, joints_num))
    # keypoints3d[:, :3] = root

    # 1. after getting Jtr Todo: why we need this?
    # trans_matrix = np.array([[1.0, 0.0, 0.0],
    #                          [0.0, 0.0, 1.0],
    #                          [0.0, 1.0, 0.0]])
    # keypoints3d = np.dot(keypoints3d, trans_matrix)

    keypoints3d = interp1d(np.linspace(0, 1, len(keypoints3d)), keypoints3d, axis=0)(
        np.linspace(0, 1, int(len(keypoints3d) * 20 / 30)))


    # Expected shape is (N, 52, 3) or (N, 24, 3)
    # sanity_check_vide(data )

    return keypoints3d


def sanity_check_vide(data):
    if HumorRenderer is None:
        print("Error: HumorRenderer not available. Cannot perform sanity check.")
        return

    frames = data['poses'].shape[0]
    b = np.repeat(data['betas'][:10], frames).reshape((frames, 10))
    smplx_path = _get_smplx_path()
    smplx = SMPLX(model_path=smplx_path, betas=b,
                  gender=np.array2string(data['gender'])[1:-1], \
                  batch_size=len(b), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)

    smplx_forwarded = smplx.forward(
        # global_orient=torch.from_numpy(data['global_orient']).float(),

        global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
        body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
        jaw_pose=torch.from_numpy(data['poses'][:, 66:69]).float(),
        leye_pose=torch.from_numpy(data['poses'][:, 69:72]).float(),
        reye_pose=torch.from_numpy(data['poses'][:, 72:75]).float(),
        left_hand_pose=torch.from_numpy(data['poses'][:, 75:120]).float(),
        right_hand_pose=torch.from_numpy(data['poses'][:, 120:]).float(),
        transl=torch.from_numpy(data['trans']).float(),  # transl=torch.from_numpy(data['transl']).float(),
        # betas=torch.from_numpy(data['betas'][:10]).float()
        betas=torch.from_numpy(b).float()
    )

    # Extract vertices

    vert = smplx_forwarded.vertices.detach().cpu().numpy()
    faces = smplx_forwarded.v_shaped.detach().cpu().numpy()
    vert, faces = vert[:200], faces[:200]


    # Redner animation
    smpl_renderer = HumorRenderer(20, imw=720, imh=720)
    smpl_renderer(
        vert, smplx.faces.astype(float),
        output= data['file_name']+'.mp4', # 'smpl_video_path.mp4',
        progress_bar=tqdm,
    )

rotX = lambda theta: torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])
rotY = lambda theta: torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]])
rotZ = lambda theta: torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])
def transf(rotMat, theta_deg, values):
    theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
    return rotMat(theta_rad).mm(values.t()).t()

def Salsa_smplx_body_shape(data):

    body_shape_dic = {'betas': None,
                      'smplx_vertices': None,
                      'smplx_faces': None,
                      'smplx_gender': None}
    return body_shape_dic

    frames = data['poses'].shape[0]
    b = np.repeat(data['betas'][:10], frames).reshape((frames, 10))
    smplx_path = _get_smplx_path()
    smplx = SMPLX(model_path=smplx_path, betas=b,
                  gender=np.array2string(data['gender'])[1:-1], \
                  batch_size=len(b), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)

    smplx_forwarded = smplx.forward(
        # global_orient=torch.from_numpy(data['global_orient']).float(),

        global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
        body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
        jaw_pose=torch.from_numpy(data['poses'][:, 66:69]).float(),
        leye_pose=torch.from_numpy(data['poses'][:, 69:72]).float(),
        reye_pose=torch.from_numpy(data['poses'][:, 72:75]).float(),
        left_hand_pose=torch.from_numpy(data['poses'][:, 75:120]).float(),
        right_hand_pose=torch.from_numpy(data['poses'][:, 120:]).float(),
        transl=torch.from_numpy(data['trans']).float(),  # transl=torch.from_numpy(data['transl']).float(),
        # betas=torch.from_numpy(data['betas'][:10]).float()
        betas=torch.from_numpy(b).float()
    )

    # Extract vertices


    vert = smplx_forwarded.vertices.detach().cpu().numpy()
    vert_20fps = interp1d(np.linspace(0, 1, len(vert)), vert, axis=0)(
        np.linspace(0, 1, int(len(vert) * 20 / 30)))
    faces = smplx.faces.astype(float)

    body_shape_dic = {'betas': data['betas'],
                        'smplx_vertices': vert_20fps,
                        'smplx_faces': faces,
                        'smplx_gender': data['gender'].astype(str).tolist()}

    return body_shape_dic

def salsa_smplx_to_rotmat(data):
    smpl_poses, smpl_trans = data['poses'], data['trans']

    nframes = smpl_poses.shape[0]
    njoints = 55

    r = R.from_rotvec(smpl_poses.reshape([nframes * njoints, 3]))
    rotmat = r.as_matrix().reshape([nframes, njoints, 3, 3])

    rotmat = np.concatenate([
        smpl_trans,
        rotmat.reshape([nframes, njoints * 3 * 3])
    ], axis=-1)

    rotmat = interp1d(np.linspace(0, 1, len(rotmat)), rotmat, axis=0)(
        np.linspace(0, 1, int(len(rotmat) * 20 / 30)))

    nframes = rotmat.shape[0]
    return rotmat.reshape(nframes, -1)


def Salsa2HM3D():
    sample_path = ''

    salsa_smplx_to_pos3d()

def audio_from_mp4(vide_path, audio_output_path):

    try:
        if not os.path.exists(audio_output_path):
            video = VideoFileClip(vide_path)
            video.audio.write_audiofile(audio_output_path)
            print(f"Extracted audio from {vide_path} -> {audio_output_path}")

        y, sr = torchaudio.load(audio_output_path)
        return y.cpu().numpy(), sr
    except Exception as e:
        print(f"Failed to process {vide_path}: {e}")


joints_num = 22
def read_all_salsa(base_path):

    out_path = os.path.join(base_path, 'lmdb_Salsa')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 12  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    # all_poses = []
    all_keypoints3d = []
    all_rotmat = []


    smpl_root = 'S:\Payam\Dance_Salsa_SFU\delivery_241121\delivery_241121'
    pos3d_root = './salsa_data/motion/pos3d'
    rotmat_root = './salsa_data/motion/rotmat'

    mp4_root = 'S:\Payam\Dance_Salsa_SFU\salsa project\salsa project\Animations'
    mp4_files = glob(os.path.join(mp4_root, '*.mp4'))
    smplx2mp4_map = {os.path.basename(f).replace('.mp4', ''): f for f in mp4_files}

    os.makedirs(pos3d_root, exist_ok=True)
    os.makedirs(rotmat_root, exist_ok=True)
    v_i = 0
    ex_fps = 20
    fps = 30

    for folder in os.listdir(smpl_root):
        # if v_i>2:
        #     break
        print(folder)
        smplx_folder = os.path.join(smpl_root, folder)
        pos3d_folder = os.path.join(pos3d_root, folder)
        rotmat_folder = os.path.join(rotmat_root, folder)
        if not os.path.exists(pos3d_folder):
            os.mkdir(pos3d_folder)
        if not os.path.exists(rotmat_folder):
            os.mkdir(rotmat_folder)
        for takes_folder in os.listdir(smplx_folder):
            # if v_i > 2 : break
            for file in os.scandir(os.path.join(smplx_folder, takes_folder)):
                if not file.name.endswith('.npz'): continue
                # if "Pair2" not in file.name: continue



                mp4_path = smplx2mp4_map['_'.join(file.name.split('_')[:5])]
                audio_path = mp4_path.replace('mp4', 'wav')
                audio_y, audio_sr = audio_from_mp4(mp4_path, audio_path)


                loaded = np.load(file.path, allow_pickle=True)
                # Todo: HumanML3D/ raw_pose_processing.ipynb --> Done!
                dict_loaded = dict(loaded)
                dict_loaded['file_name'] = file.name # for sanity check.
                keypoints3d = salsa_smplx_to_pos3d(dict_loaded)
                rotmat = salsa_smplx_to_rotmat(loaded)
                body_shape =Salsa_smplx_body_shape(loaded)
                # HumanML3D Representation
                (data, ground_positions,
                 positions, l_velocity) = HM3D_F.process_file(keypoints3d,
                                                              0.002)
                rec_ric_data = HM3D_F.recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
                HML3D_New_Joints = rec_ric_data.squeeze().numpy() # N, 22, 3
                HML3D_New_Joints_Vec = data

                raw_euler_poses = loaded['poses']
                raw_euler_poses = (interp1d(np.linspace(0, 1, len(raw_euler_poses)),
                                        raw_euler_poses, axis=0)
                               (np.linspace(0, 1, int(len(raw_euler_poses) * 20 / 30))))
                raw_trans = loaded['trans']
                raw_trans = (interp1d(np.linspace(0, 1, len(raw_trans)),
                                            raw_trans, axis=0)
                                   (np.linspace(0, 1, int(len(raw_trans) * 20 / 30))))

                # test:
                # import os
                # pjoin = os.path.join
                # np.save(pjoin(out_path, 'SFU_SALSA_EXAMPLE.npy'), HML3D_New_Joints_Vec[:1000])

                # np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
                # np.save(pjoin(save_dir2, source_file), data)
                # Todo: motion_representation.ipynb
                # Done!

                # # Sanity Check:
                # save_path = '1.gif'
                # # HM3D_F.plot_3d_motion(save_path, kinematic_chain, New_Joints[:20], title="None", fps=20, radius=4)
                # ARGUS = HML3D_New_Joints[:400], save_path, 'title'
                # HM3D_F.plot_3d_motion_Payam(ARGUS)

                # for v_i, bvh_file in enumerate(bvh_files):
                name = os.path.split(file.name)[1][:-4]
                print(name)

                # process
                clips = [{'vid': name, 'clips': []},  # train
                         {'vid': name, 'clips': []}]  # validation

                # split
                if v_i == 0:
                    dataset_idx = 1  # validation
                else:
                    dataset_idx = 0  # train

                # save subtitles and skeletons

                #Todo: I used to use the following for sanity check
                # which increases the size drastically:
                body_shape['betas'] = None
                body_shape['smplx_vertices'] = None
                body_shape['smplx_faces'] = None
                body_shape['smplx_gender'] = None

                poses = np.asarray(rotmat, dtype=np.float16)
                clips[dataset_idx]['clips'].append(
                    {'raw_euler_poses': raw_euler_poses,
                     'raw_trans': raw_trans,
                     'keypoints3d': keypoints3d,
                     'rotmat': rotmat,
                     'HML3D_joints': HML3D_New_Joints,
                     'HML3D_joints_vec': HML3D_New_Joints_Vec,
                     'audio_raw': audio_y,
                     'audio_sr': audio_sr,
                     'body_betas': body_shape['betas'],
                     'body_vertices': body_shape['smplx_vertices'],
                     'body_faces': body_shape['smplx_faces'],
                     'body_gender': body_shape['smplx_gender']
                     # Todo: add motioncodes here? No, we add it after windowing.
                     })

                # write to db
                for i in range(2):
                    with db[i].begin(write=True) as txn:
                        if len(clips[i]['clips']) > 0:
                            k = '{:010}'.format(v_i).encode('ascii')
                            v = pyarrow.serialize(clips[i]).to_buffer()
                            txn.put(k, v)

                # # all_poses.append(poses)
                # all_keypoints3d.append(keypoints3d)
                # all_rotmat.append(rotmat)
                v_i += 1

        # close db
    for i in range(2):
        db[i].sync()
        db[i].close()



def read_all_salsa_pairs_old(salsa_data_root, save_path):

    out_path = os.path.join(save_path, 'lmdb_Salsa_pair')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 12  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    # all_poses = []
    all_keypoints3d = []
    all_rotmat = []


    smpl_root = salsa_data_root#
    pos3d_root = os.path.join(save_path, './salsa_data/motion/pos3d')
    rotmat_root = os.path.join(save_path, './salsa_data/motion/rotmat')

    mp4_root = 'S:\Payam\Dance_Salsa_SFU\salsa project\salsa project\Animations' # os.path.join(salsa_data_root, 'synced_music')
    mp4_files = glob(os.path.join(mp4_root, '*.mp4'))
    smplx2mp4_map = {os.path.basename(f).replace('.mp4', ''): f for f in mp4_files}

    os.makedirs(pos3d_root, exist_ok=True)
    os.makedirs(rotmat_root, exist_ok=True)
    v_i = 0
    ex_fps = 20
    fps = 30

    for folder in os.listdir(smpl_root):
        if v_i>2:
            break
        print(folder)
        smplx_folder = os.path.join(smpl_root, folder)
        pos3d_folder = os.path.join(pos3d_root, folder)
        rotmat_folder = os.path.join(rotmat_root, folder)
        if not os.path.exists(pos3d_folder):
            os.mkdir(pos3d_folder)
        if not os.path.exists(rotmat_folder):
            os.mkdir(rotmat_folder)
        for takes_folder in os.listdir(smplx_folder):
            # if v_i > 2 : break
            for file1 in os.scandir(os.path.join(smplx_folder, takes_folder)):
                if not file1.name.endswith('.npz'): continue
                # if "Pair2" not in file.name: continue

                if 'leader' not in file1.name: continue
                leader_file = file1

                # find the follower file:
                for file2 in os.scandir(os.path.join(smplx_folder, takes_folder)):
                    if 'follower' in file2.name:
                        follower_file = file2



                mp4_path = smplx2mp4_map['_'.join(file1.name.split('_')[:5])]
                audio_path = mp4_path.replace('mp4', 'wav')
                audio_y, audio_sr = audio_from_mp4(mp4_path, audio_path)


                loaded_leader = np.load(leader_file.path, allow_pickle=True)
                loaded_follower = np.load(follower_file.path, allow_pickle=True)

                # Todo: HumanML3D/ raw_pose_processing.ipynb --> Done!
                dict_loaded_L = dict(loaded_leader)
                dict_loaded_F = dict(loaded_follower)

                dict_loaded_L['file_name'] = leader_file.name # for sanity check.
                dict_loaded_F['file_name'] = follower_file.name

                keypoints3d_L = salsa_smplx_to_pos3d(dict_loaded_L)
                keypoints3d_F = salsa_smplx_to_pos3d(dict_loaded_F)

                # rotate -90 def around X-axisto be consistent with common datasets e.g., HumanML3D
                keypoints3d_L = torch.tensor(keypoints3d_L).float()
                keypoints3d_F = torch.tensor(keypoints3d_F).float()
                for frame_i in range(keypoints3d_F.shape[0]):
                    keypoints3d_L[frame_i] = transf(rotX, -90, keypoints3d_L[frame_i])
                    keypoints3d_F[frame_i] = transf(rotX, -90, keypoints3d_F[frame_i])
                keypoints3d_L = keypoints3d_L.numpy()
                keypoints3d_F = keypoints3d_F.numpy()



                rotmat_L = salsa_smplx_to_rotmat(loaded_leader)
                rotmat_F = salsa_smplx_to_rotmat(loaded_follower)

                # Todo: do we need to keep this here or comment it out?
                body_shape_L =Salsa_smplx_body_shape(loaded_leader)
                body_shape_F = Salsa_smplx_body_shape(loaded_follower)

                # HumanML3D Representation
                (data_L, ground_positions_L,
                 positions_L, l_velocity_L) = HM3D_F.process_file(keypoints3d_L,
                                                              0.002)
                (data_F, ground_positions_F,
                 positions_F, l_velocity_F) = HM3D_F.process_file(keypoints3d_F,
                                                              0.002)
                rec_ric_data_L = HM3D_F.recover_from_ric(torch.from_numpy(data_L).unsqueeze(0).float(), joints_num)
                rec_ric_data_F = HM3D_F.recover_from_ric(torch.from_numpy(data_F).unsqueeze(0).float(), joints_num)

                HML3D_New_Joints_L = rec_ric_data_L.squeeze().numpy() # N, 22, 3
                HML3D_New_Joints_F = rec_ric_data_F.squeeze().numpy()  # N, 22, 3

                HML3D_New_Joints_Vec_L = data_L
                HML3D_New_Joints_Vec_F = data_F


                # ----- raw data
                raw_euler_poses_L = loaded_leader['poses']
                raw_euler_poses_F = loaded_follower['poses']

                raw_euler_poses_L = (interp1d(np.linspace(0, 1, len(raw_euler_poses_L)),
                                        raw_euler_poses_L, axis=0)
                               (np.linspace(0, 1, int(len(raw_euler_poses_L) * 20 / 30))))

                raw_euler_poses_F = (interp1d(np.linspace(0, 1, len(raw_euler_poses_F)),
                                              raw_euler_poses_F, axis=0)
                                     (np.linspace(0, 1, int(len(raw_euler_poses_F) * 20 / 30))))

                raw_trans_L = loaded_leader['trans']
                raw_trans_F = loaded_follower['trans']

                raw_trans_L = (interp1d(np.linspace(0, 1, len(raw_trans_L)),
                                            raw_trans_L, axis=0)
                                   (np.linspace(0, 1, int(len(raw_trans_L) * 20 / 30))))

                raw_trans_F = (interp1d(np.linspace(0, 1, len(raw_trans_F)),
                                      raw_trans_F, axis=0)
                             (np.linspace(0, 1, int(len(raw_trans_F) * 20 / 30))))

                # test:
                # import os
                # pjoin = os.path.join
                # np.save(pjoin(out_path, 'SFU_SALSA_EXAMPLE.npy'), HML3D_New_Joints_Vec[:1000])

                # np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
                # np.save(pjoin(save_dir2, source_file), data)
                # Todo: motion_representation.ipynb --> Done!


                # # Sanity Check:
                #Todo: based on the sanity check, I noticed that we need to rotate the keypoints -90 degree to make
                # the Salsa data consistent with common motion datasets such as HumanML3D
                # save_path = '2.gif'
                # # HM3D_F.plot_3d_motion(save_path, kinematic_chain, New_Joints[:20], title="None", fps=20, radius=4)
                # goto_plot = torch.tensor(keypoints3d_L[:50]).float()
                # for frame_i in range(goto_plot.shape[0]):
                #     goto_plot[frame_i] = transf(rotX, -90, goto_plot[frame_i])
                # ARGUS = goto_plot[:50].numpy(), save_path, 'title'
                # HM3D_F.plot_3d_motion_Payam(ARGUS)

                # for v_i, bvh_file in enumerate(bvh_files):
                name = os.path.split(leader_file.name)[1][:-4] + ',' + \
                       os.path.split(follower_file.name)[1][:-4]
                print(name)

                # process
                clips = [{'vid': name, 'clips': []},  # train
                         {'vid': name, 'clips': []}]  # validation

                # split
                if v_i == 0:
                    dataset_idx = 1  # validation
                else:
                    dataset_idx = 0  # train

                # save subtitles and skeletons

                #Todo: I used to use the following for sanity check
                # which increases the size drastically:
                body_shape_L['betas'] = None
                body_shape_L['smplx_vertices'] = None
                body_shape_L['smplx_faces'] = None
                body_shape_L['smplx_gender'] = None

                body_shape_F['betas'] = None
                body_shape_F['smplx_vertices'] = None
                body_shape_F['smplx_faces'] = None
                body_shape_F['smplx_gender'] = None

                poses = np.asarray(rotmat_L, dtype=np.float16)
                clips[dataset_idx]['clips'].append(
                    {'raw_euler_poses_L': raw_euler_poses_L,
                     'raw_trans_L': raw_trans_L,
                     'keypoints3d_L': keypoints3d_L,
                     'rotmat_L': rotmat_L,
                     'HML3D_joints_L': HML3D_New_Joints_L,
                     'HML3D_joints_vec_L': HML3D_New_Joints_Vec_L,

                     'raw_euler_poses_F': raw_euler_poses_F,
                     'raw_trans_F': raw_trans_F,
                     'keypoints3d_F': keypoints3d_F,
                     'rotmat_F': rotmat_F,
                     'HML3D_joints_F': HML3D_New_Joints_F,
                     'HML3D_joints_vec_F': HML3D_New_Joints_Vec_F,

                     'audio_raw': audio_y,
                     'audio_sr': audio_sr,



                     # Todo: this was for sanity check.
                     'body_betas': body_shape_L['betas'],
                     'body_vertices': body_shape_L['smplx_vertices'],
                     'body_faces': body_shape_L['smplx_faces'],
                     'body_gender': body_shape_L['smplx_gender']
                     # Todo: add motioncodes here? No, we add it after windowing.
                     })

                # write to db
                for i in range(2):
                    with db[i].begin(write=True) as txn:
                        if len(clips[i]['clips']) > 0:
                            k = '{:010}'.format(v_i).encode('ascii')
                            v = pyarrow.serialize(clips[i]).to_buffer()
                            txn.put(k, v)

                # # all_poses.append(poses)
                # all_keypoints3d.append(keypoints3d)
                # all_rotmat.append(rotmat)
                v_i += 1

        # close db
    for i in range(2):
        db[i].sync()
        db[i].close()


def read_all_salsa_pairs(salsa_data_root, save_path):
    """
    Process CoMPAS3D dataset with improved path handling, proper splits, and error handling.
    Supports both Dataset/compas3d and Dataset root paths.
    """
    # Handle flexible path - support both Dataset/compas3d and Dataset root
    if os.path.basename(salsa_data_root) != 'compas3d':
        compas3d_path = os.path.join(salsa_data_root, 'compas3d')
        if os.path.exists(compas3d_path):
            smpl_root = compas3d_path
        else:
            smpl_root = salsa_data_root
    else:
        smpl_root = salsa_data_root

    out_path = os.path.join(save_path, 'lmdb_Salsa_pair')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 12  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_val'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(3):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    pos3d_root = os.path.join(save_path, './salsa_data/motion/pos3d')
    rotmat_root = os.path.join(save_path, './salsa_data/motion/rotmat')

    os.makedirs(pos3d_root, exist_ok=True)
    os.makedirs(rotmat_root, exist_ok=True)
    v_i = 0
    ex_fps = 20
    fps = 30

    # Define train/val/test splits based on README
    splits_map = {
        'Pair1': {'val': 'Song2_Take2', 'test': 'Song1_Take1'},
        'Pair2': {'val': 'Song3_Take1', 'test': 'Song1_Take2'},
        'Pair3': {'val': None, 'test': 'Song2_Take1'},
        'Pair4': {'val': None, 'test': 'Song2_Take2'},
        'Pair5': {'val': 'Song1_Take1', 'test': 'Song3_Take1'},
        'Pair6': {'val': None, 'test': 'Song3_Take2'},
        'Pair7': {'val': None, 'test': 'Song4_Take1'},
        'Pair8': {'val': None, 'test': 'Song4_Take2'},
        'Pair9': {'val': None, 'test': 'Song1_Take1'},
    }
    
    # Track samples by split for summary report
    split_samples = {'train': [], 'val': [], 'test': []}

    for folder in os.listdir(smpl_root):
        # if v_i > 1:
        #     break
        if not folder.startswith('Pair'):
            continue
        
        print(f"Processing {folder}")
        smplx_folder = os.path.join(smpl_root, folder)
        if not os.path.isdir(smplx_folder):
            continue
            
        pos3d_folder = os.path.join(pos3d_root, folder)
        rotmat_folder = os.path.join(rotmat_root, folder)
        if not os.path.exists(pos3d_folder):
            os.mkdir(pos3d_folder)
        if not os.path.exists(rotmat_folder):
            os.mkdir(rotmat_folder)
            
        pair_splits = splits_map.get(folder, {'val': None, 'test': None})
        
        for takes_folder in os.listdir(smplx_folder):
            takes_path = os.path.join(smplx_folder, takes_folder)
            if not os.path.isdir(takes_path):
                continue
                
            leader_file = None
            follower_file = None
            mp4_file = None
            txt_file = None
            
            # Find all relevant files in the take folder
            for file_entry in os.scandir(takes_path):
                if file_entry.name.endswith('_leader.npz'):
                    leader_file = file_entry
                elif file_entry.name.endswith('_follower.npz'):
                    follower_file = file_entry
                elif file_entry.name.endswith('.mp4'):
                    mp4_file = file_entry
                elif file_entry.name.endswith('.txt'):
                    txt_file = file_entry
            
            # Skip if missing required files
            if leader_file is None:
                print(f"  Warning: No leader file found in {takes_folder}, skipping")
                continue
            if follower_file is None:
                print(f"  Warning: No follower file found in {takes_folder}, skipping")
                continue
            if mp4_file is None:
                print(f"  Warning: No mp4 file found in {takes_folder}, skipping")
                continue

            # Determine dataset split
            # Extract song and take from folder name (e.g., "Pair1_song1_take1" -> "Song1_Take1")
            folder_parts = takes_folder.split('_')
            if len(folder_parts) >= 3:
                song_num = folder_parts[1].replace('song', 'Song')
                take_num = folder_parts[2].replace('take', 'Take')
                song_take = f"{song_num}_{take_num}"
                
                if pair_splits['test'] == song_take:
                    dataset_idx = 2  # test
                elif pair_splits['val'] == song_take:
                    dataset_idx = 1  # validation
                else:
                    dataset_idx = 0  # train
            else:
                # # Fallback to old logic if naming doesn't match
                # if v_i == 0:
                #     dataset_idx = 1  # validation
                # else:
                #     dataset_idx = 0  # train
                v_i = None
            # Extract audio from mp4
            audio_path = mp4_file.path.replace('.mp4', '.wav')
            try:
                audio_y, audio_sr = audio_from_mp4(mp4_file.path, audio_path)
            except Exception as e:
                print(f"  Error extracting audio from {mp4_file.path}: {e}")
                continue

            # Load motion data
            try:
                loaded_leader = np.load(leader_file.path, allow_pickle=True)
                loaded_follower = np.load(follower_file.path, allow_pickle=True)
            except Exception as e:
                print(f"  Error loading npz files: {e}")
                continue

            dict_loaded_L = dict(loaded_leader)
            dict_loaded_F = dict(loaded_follower)
            dict_loaded_L['file_name'] = leader_file.name
            dict_loaded_F['file_name'] = follower_file.name

            # Process motion data
            try:
                keypoints3d_L = salsa_smplx_to_pos3d(dict_loaded_L)
                keypoints3d_F = salsa_smplx_to_pos3d(dict_loaded_F)
                
                # Store original keypoints for sanity check (before rotation)
                original_keypoints_L = keypoints3d_L.copy()
                original_keypoints_F = keypoints3d_F.copy()

                # rotate -90 deg around X-axis to be consistent with common datasets e.g., HumanML3D
                keypoints3d_L = torch.tensor(keypoints3d_L).float()
                keypoints3d_F = torch.tensor(keypoints3d_F).float()
                for frame_i in range(keypoints3d_F.shape[0]):
                    keypoints3d_L[frame_i] = transf(rotX, -90, keypoints3d_L[frame_i])
                    keypoints3d_F[frame_i] = transf(rotX, -90, keypoints3d_F[frame_i])
                keypoints3d_L = keypoints3d_L.numpy()
                keypoints3d_F = keypoints3d_F.numpy()

                rotmat_L = salsa_smplx_to_rotmat(loaded_leader)
                rotmat_F = salsa_smplx_to_rotmat(loaded_follower)

                body_shape_L = Salsa_smplx_body_shape(loaded_leader)
                body_shape_F = Salsa_smplx_body_shape(loaded_follower)

                # HumanML3D Representation
                (data_L, ground_positions_L,
                 positions_L, l_velocity_L) = HM3D_F.process_file(keypoints3d_L, 0.002)
                (data_F, ground_positions_F,
                 positions_F, l_velocity_F) = HM3D_F.process_file(keypoints3d_F, 0.002)
                rec_ric_data_L = HM3D_F.recover_from_ric(torch.from_numpy(data_L).unsqueeze(0).float(), joints_num)
                rec_ric_data_F = HM3D_F.recover_from_ric(torch.from_numpy(data_F).unsqueeze(0).float(), joints_num)

                HML3D_New_Joints_L = rec_ric_data_L.squeeze().numpy()
                HML3D_New_Joints_F = rec_ric_data_F.squeeze().numpy()

                HML3D_New_Joints_Vec_L = data_L
                HML3D_New_Joints_Vec_F = data_F

                # # Sanity Check:
                # # Todo: based on the sanity check, I noticed that we need to rotate the keypoints -90 degree to make
                # # the Salsa data consistent with common motion datasets such as HumanML3D
                if False:
                    sanity_check_dir = os.path.join(save_path, 'sanity_check_gifs')
                    os.makedirs(sanity_check_dir, exist_ok=True)
                    
                    # Create subfolder for this sample (use name or index)
                    sample_name = os.path.split(leader_file.name)[1][:-4] + '_' + os.path.split(follower_file.name)[1][:-4]
                    sample_dir = os.path.join(sanity_check_dir, sample_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Test different rotation angles
                    rotation_angles = [0, 90, -90, 180, -180, 45, -45, 270, -270]
                    
                    for angle in rotation_angles:
                        # Rotate keypoints from original (before the -90 rotation applied above)
                        goto_plot_L = torch.tensor(original_keypoints_L[:50]).float()
                        goto_plot_F = torch.tensor(original_keypoints_F[:50]).float()
                        for frame_i in range(goto_plot_L.shape[0]):
                            goto_plot_L[frame_i] = transf(rotX, angle, goto_plot_L[frame_i])
                            goto_plot_F[frame_i] = transf(rotX, angle, goto_plot_F[frame_i])
                        
                        # Save leader animation
                        save_path_L = os.path.join(sample_dir, f'leader_rot{angle:+.0f}.gif')
                        ARGUS_L = goto_plot_L[:50].numpy(), save_path_L, f'Leader Rot {angle}°'
                        HM3D_F.plot_3d_motion_Payam(ARGUS_L)
                        
                        # Save follower animation
                        save_path_F = os.path.join(sample_dir, f'follower_rot{angle:+.0f}.gif')
                        ARGUS_F = goto_plot_F[:50].numpy(), save_path_F, f'Follower Rot {angle}°'
                        HM3D_F.plot_3d_motion_Payam(ARGUS_F)
                    
                    print(f"  Sanity check GIFs saved to: {sample_dir}")

                # Raw data
                raw_euler_poses_L = loaded_leader['poses']
                raw_euler_poses_F = loaded_follower['poses']

                raw_euler_poses_L = (interp1d(np.linspace(0, 1, len(raw_euler_poses_L)),
                                        raw_euler_poses_L, axis=0)
                               (np.linspace(0, 1, int(len(raw_euler_poses_L) * 20 / 30))))

                raw_euler_poses_F = (interp1d(np.linspace(0, 1, len(raw_euler_poses_F)),
                                              raw_euler_poses_F, axis=0)
                                     (np.linspace(0, 1, int(len(raw_euler_poses_F) * 20 / 30))))

                raw_trans_L = loaded_leader['trans']
                raw_trans_F = loaded_follower['trans']

                raw_trans_L = (interp1d(np.linspace(0, 1, len(raw_trans_L)),
                                            raw_trans_L, axis=0)
                                   (np.linspace(0, 1, int(len(raw_trans_L) * 20 / 30))))

                raw_trans_F = (interp1d(np.linspace(0, 1, len(raw_trans_F)),
                                      raw_trans_F, axis=0)
                             (np.linspace(0, 1, int(len(raw_trans_F) * 20 / 30))))

            except Exception as e:
                print(f"  Error processing motion data: {e}")
                continue

            name = os.path.split(leader_file.name)[1][:-4] + ',' + \
                   os.path.split(follower_file.name)[1][:-4]
            split_name = ['train', 'val', 'test'][dataset_idx]
            print(f"  Processing: {name} -> split: {split_name}")
            
            # Track sample for summary
            split_samples[split_name].append(f"{folder}/{takes_folder}")

            # Load annotation if available
            annotation = None
            if txt_file is not None:
                try:
                    with open(txt_file.path, 'r') as f:
                        annotation = f.read()
                except Exception as e:
                    print(f"  Warning: Could not load annotation file: {e}")

            # Prepare clips
            clips = [{'vid': name, 'clips': []},  # train
                     {'vid': name, 'clips': []},  # validation
                     {'vid': name, 'clips': []}]  # test

            body_shape_L['betas'] = None
            body_shape_L['smplx_vertices'] = None
            body_shape_L['smplx_faces'] = None
            body_shape_L['smplx_gender'] = None

            body_shape_F['betas'] = None
            body_shape_F['smplx_vertices'] = None
            body_shape_F['smplx_faces'] = None
            body_shape_F['smplx_gender'] = None

            clips[dataset_idx]['clips'].append(
                {'raw_euler_poses_L': raw_euler_poses_L,
                 'raw_trans_L': raw_trans_L,
                 'keypoints3d_L': keypoints3d_L,
                 'rotmat_L': rotmat_L,
                 'HML3D_joints_L': HML3D_New_Joints_L,
                 'HML3D_joints_vec_L': HML3D_New_Joints_Vec_L,

                 'raw_euler_poses_F': raw_euler_poses_F,
                 'raw_trans_F': raw_trans_F,
                 'keypoints3d_F': keypoints3d_F,
                 'rotmat_F': rotmat_F,
                 'HML3D_joints_F': HML3D_New_Joints_F,
                 'HML3D_joints_vec_F': HML3D_New_Joints_Vec_F,

                 'audio_raw': audio_y,
                 'audio_sr': audio_sr,
                 'annotation': annotation,  # Optional annotation

                 'body_betas': body_shape_L['betas'],
                 'body_vertices': body_shape_L['smplx_vertices'],
                 'body_faces': body_shape_L['smplx_faces'],
                 'body_gender': body_shape_L['smplx_gender']
                 })

            # Write to db
            for i in range(3):
                with db[i].begin(write=True) as txn:
                    if len(clips[i]['clips']) > 0:
                        k = '{:010}'.format(v_i).encode('ascii')
                        v = pyarrow.serialize(clips[i]).to_buffer()
                        txn.put(k, v)

            v_i += 1
            # if v_i > 1:
            #     break

    # Close db
    for i in range(3):
        db[i].sync()
        db[i].close()
    
    # Print split summary
    print("\n" + "=" * 80)
    print("DATASET SPLIT SUMMARY")
    print("=" * 80)
    split_names = ['train', 'val', 'test']
    split_labels = ['Training', 'Validation', 'Test']
    for split, label in zip(split_names, split_labels):
        count = len(split_samples[split])
        print(f"\n[{label}] ({count} samples):")
        if count > 0:
            for sample in sorted(split_samples[split]):
                print(f"  - {sample}")
        else:
            print("  (no samples)")
    print("\n" + "=" * 80)


def sanity_check_lmdb(save_path, show_example=True):
    """
    Sanity check function to verify LMDB databases were created correctly.
    Reports entry counts and optionally displays example data from each split.
    
    Args:
        save_path: Path where LMDB databases were saved
        show_example: If True, display one example from each database
    """
    
    lmdb_path = os.path.join(save_path, 'lmdb_Salsa_pair')
    splits = ['train', 'val', 'test']
    split_names = ['Training', 'Validation', 'Test']
    
    print("=" * 80)
    print("LMDB SANITY CHECK")
    print("=" * 80)
    
    results = {}
    
    for split_idx, (split, split_name) in enumerate(zip(splits, split_names)):
        db_path = os.path.join(lmdb_path, f'lmdb_{split}')
        
        if not os.path.exists(db_path):
            print(f"\n[{split_name}] Database not found: {db_path}")
            results[split] = {'count': 0, 'example': None}
            continue
        
        db = lmdb.open(db_path, readonly=True, lock=False)
        
        # Count entries
        with db.begin() as txn:
            cursor = txn.cursor()
            count = sum(1 for _ in cursor)
        
        results[split] = {'count': count, 'example': None}
        
        print(f"\n[{split_name}] Database: {db_path}")
        print(f"  Total entries: {count}")
        
        # Get example if requested and entries exist
        if show_example and count > 0:
            with db.begin() as txn:
                cursor = txn.cursor()
                cursor.first()  # Get first entry
                key = cursor.key()
                value = cursor.value()
                
                # Deserialize
                try:
                    example_data = pyarrow.deserialize(value)
                    results[split]['example'] = example_data
                    
                    print(f"\n  Example Entry (key: {key.decode('ascii')}):")
                    print(f"  {'-' * 76}")
                    
                    if 'vid' in example_data:
                        print(f"  Video ID: {example_data['vid']}")
                    
                    if 'clips' in example_data and len(example_data['clips']) > 0:
                        clip = example_data['clips'][0]
                        print(f"  Number of clips: {len(example_data['clips'])}")
                        print(f"\n  Clip Data Structure:")
                        
                        # Leader data
                        if 'raw_euler_poses_L' in clip:
                            print(f"    [Leader Motion]")
                            print(f"      - raw_euler_poses_L: shape {np.array(clip['raw_euler_poses_L']).shape}, dtype {np.array(clip['raw_euler_poses_L']).dtype}")
                            print(f"      - raw_trans_L: shape {np.array(clip['raw_trans_L']).shape}, dtype {np.array(clip['raw_trans_L']).dtype}")
                            print(f"      - keypoints3d_L: shape {np.array(clip['keypoints3d_L']).shape}, dtype {np.array(clip['keypoints3d_L']).dtype}")
                            print(f"      - rotmat_L: shape {np.array(clip['rotmat_L']).shape}, dtype {np.array(clip['rotmat_L']).dtype}")
                            print(f"      - HML3D_joints_L: shape {np.array(clip['HML3D_joints_L']).shape}, dtype {np.array(clip['HML3D_joints_L']).dtype}")
                            print(f"      - HML3D_joints_vec_L: shape {np.array(clip['HML3D_joints_vec_L']).shape}, dtype {np.array(clip['HML3D_joints_vec_L']).dtype}")
                        
                        # Follower data
                        if 'raw_euler_poses_F' in clip:
                            print(f"    [Follower Motion]")
                            print(f"      - raw_euler_poses_F: shape {np.array(clip['raw_euler_poses_F']).shape}, dtype {np.array(clip['raw_euler_poses_F']).dtype}")
                            print(f"      - raw_trans_F: shape {np.array(clip['raw_trans_F']).shape}, dtype {np.array(clip['raw_trans_F']).dtype}")
                            print(f"      - keypoints3d_F: shape {np.array(clip['keypoints3d_F']).shape}, dtype {np.array(clip['keypoints3d_F']).dtype}")
                            print(f"      - rotmat_F: shape {np.array(clip['rotmat_F']).shape}, dtype {np.array(clip['rotmat_F']).dtype}")
                            print(f"      - HML3D_joints_F: shape {np.array(clip['HML3D_joints_F']).shape}, dtype {np.array(clip['HML3D_joints_F']).dtype}")
                            print(f"      - HML3D_joints_vec_F: shape {np.array(clip['HML3D_joints_vec_F']).shape}, dtype {np.array(clip['HML3D_joints_vec_F']).dtype}")
                        
                        # Audio data
                        if 'audio_raw' in clip:
                            audio_shape = np.array(clip['audio_raw']).shape if clip['audio_raw'] is not None else None
                            print(f"    [Audio]")
                            print(f"      - audio_raw: shape {audio_shape}, dtype {np.array(clip['audio_raw']).dtype if clip['audio_raw'] is not None else 'None'}")
                            print(f"      - audio_sr: {clip['audio_sr']}")
                        
                        # Annotation
                        if 'annotation' in clip:
                            annotation_len = len(clip['annotation']) if clip['annotation'] else 0
                            print(f"    [Annotation]")
                            print(f"      - annotation: {annotation_len} characters" + (f" (preview: {clip['annotation'][:100]}...)" if clip['annotation'] and len(clip['annotation']) > 100 else ""))
                        
                        # Body shape (should be None)
                        print(f"    [Body Shape]")
                        print(f"      - body_betas: {clip.get('body_betas', None)}")
                        print(f"      - body_vertices: {type(clip.get('body_vertices', None)).__name__ if clip.get('body_vertices') is not None else 'None'}")
                        print(f"      - body_faces: {type(clip.get('body_faces', None)).__name__ if clip.get('body_faces') is not None else 'None'}")
                        print(f"      - body_gender: {clip.get('body_gender', None)}")
                        
                        # Show sample values
                        print(f"\n    [Sample Values]")
                        if 'keypoints3d_L' in clip:
                            sample_kp = np.array(clip['keypoints3d_L'])[0, :3, :]  # First frame, first 3 joints
                            print(f"      Leader keypoints (frame 0, joints 0-2):")
                            for j in range(3):
                                print(f"        Joint {j}: [{sample_kp[j, 0]:.4f}, {sample_kp[j, 1]:.4f}, {sample_kp[j, 2]:.4f}]")
                        
                        if 'keypoints3d_F' in clip:
                            sample_kp = np.array(clip['keypoints3d_F'])[0, :3, :]  # First frame, first 3 joints
                            print(f"      Follower keypoints (frame 0, joints 0-2):")
                            for j in range(3):
                                print(f"        Joint {j}: [{sample_kp[j, 0]:.4f}, {sample_kp[j, 1]:.4f}, {sample_kp[j, 2]:.4f}]")
                
                except Exception as e:
                    print(f"  Error loading example: {e}")
                    import traceback
                    traceback.print_exc()
        
        db.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = sum(r['count'] for r in results.values())
    print(f"Total entries across all splits: {total}")
    for split, split_name in zip(splits, split_names):
        count = results[split]['count']
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {split_name:12s}: {count:4d} entries ({percentage:5.1f}%)")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SALSA dataset or check LMDB databases.")
    parser.add_argument("--salsa_data_root", type=str, default=None, help="Path to the SALSA data directory (required for processing).")
    parser.add_argument("--save_path", type=str, default='dataset_processed_New', help="Path to save processed LMDB databases.")
    parser.add_argument("--sanity_check", action='store_true', help="Run sanity check on existing LMDB databases instead of processing.")
    parser.add_argument("--check_path", type=str, default=None, help="Path to LMDB databases for sanity check (defaults to save_path).")
    args = parser.parse_args()

    if args.sanity_check:
        # Run sanity check
        check_path = args.check_path if args.check_path else args.save_path
        print(f"Running sanity check on LMDB databases in: {check_path}")
        sanity_check_lmdb(check_path, show_example=True)
    else:
        # Process dataset
        if args.salsa_data_root is None:
            parser.error("--salsa_data_root is required for processing dataset")
        print(f"SALSA data root: {args.salsa_data_root}")
        read_all_salsa_pairs(args.salsa_data_root, args.save_path)
        print("\nProcessing complete! Running sanity check...")
        sanity_check_lmdb(args.save_path, show_example=True)



