"""create data samples
"""
import os
os.chdir(r'S:\Payam\Dance_Salsa_SFU\Motion-Agent-Salsa')
import sys
sys.path.append(r'S:\Payam\Dance_Salsa_SFU\Motion-Agent-Salsa')
import math
import pickle
import os
from typing import Tuple

import lmdb
import numpy as np
import pyarrow
import torch
import librosa
from tqdm import tqdm
# from configargparse import argparse
# from model.vocab import Vocab
import random
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import models.vqvae as vqvae




from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.motion_utils import recover_from_ric, plot_3d_motion
import time
from utils.paramUtil import t2m_kinematic_chain
from torch.utils.data import Dataset


# Path to the FFmpeg bin directory
ffmpeg_path = r'S:\Payam\LAMMA\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin'
import matplotlib as mpl
import shutil

ffmpeg_path = shutil.which("ffmpeg")
mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path


# Check if the path is already in the PATH variable
if ffmpeg_path not in os.environ['PATH']:
    # Add the path to the PATH environment variable
    os.environ['PATH'] += os.pathsep + ffmpeg_path


class Motion_tokenizer:

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        # self.args.nb_joints = 22
        self.args.dataname = 't2m'
        if not args.is_MDM:
            self.args.vq_path = "ckpt/vqvae.pth"
        if args.is_MDM:
            self.args.vq_path = os.path.join(args.parent_dir, "ckpt/vqvae.pth")
        self.net = vqvae.HumanVQVAE(self.args, ## use args to define different parameters in different quantizers
                           self.args.nb_code,
                           self.args.code_dim,
                           self.args.output_emb_width,
                           self.args.down_t,
                           self.args.stride_t,
                           self.args.width,
                           self.args.depth,
                           self.args.dilation_growth_rate,
                           self.args.vq_act,
                           self.args.vq_norm)
        print ('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone)
        self.nb_text_tokens = len(self.tokenizer)
        if not args.is_MDM:
            self.mean = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
            self.std = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')
        if args.is_MDM:
            self.mean = np.load(os.path.join(args.parent_dir, 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy'))
            self.std = np.load(os.path.join(args.parent_dir, 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy'))
        print('Loading the HumanVQVAE model is completed!')
    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std

    def npy263_tokenizer(self, input_motion):
        # Todo: High! We need to polish the code such that it only returns index in VQ space, not the LLM.
        motion = self.normalize(input_motion)
        motion = torch.from_numpy(motion).float().to(self.device).unsqueeze(0)
        motion_tokens = self.net.encode(motion).squeeze(0)
        # We skip the following reindex since we later use tokenizer to figure it out.
        # motion_tokens = motion_tokens + self.nb_text_tokens + 2  # reindex the motion tokens
        # print(motion_tokens)
        return motion_tokens


    def sanity_check(self, motion_tokens, org_263):
        self.save_dir = 'demo'
        motion_tokens = motion_tokens - self.nb_text_tokens - 2  # reindex the motion tokens
        motion = self.net.forward_decoder(motion_tokens)
        motion = self.denormalize(motion.detach().cpu().numpy())
        motion = recover_from_ric(torch.from_numpy(motion).float().to(self.device), 22)
        filename = f"{self.save_dir}/motion_{int(time.time())}.mp4"
        print('Plotting motion...')
        message = "Sanity Check..!"
        plot_3d_motion(filename, t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=message, fps=20,
                       radius=4)
        np.save(f"{self.save_dir}/motion_{int(time.time())}.npy", motion.squeeze().detach().cpu().numpy())
        print(f"Motion saved to {filename}")

        org_motion = recover_from_ric(torch.from_numpy(org_263).float().to(self.device), 22)

        plot_3d_motion(filename.replace('motion', 'org_motion'),
                       t2m_kinematic_chain, org_motion.squeeze().detach().cpu().numpy(),
                       title=message, fps=20, radius=4)
        np.save(f"{self.save_dir}/org_motion_{int(time.time())}.npy", motion.squeeze().detach().cpu().numpy())
        print(f"Org Motion saved to {filename}")


from WavTokenizer.encoder.utils import convert_audio

from WavTokenizer.decoder.pretrained import WavTokenizer

class Audio_tokenizer:

    def __init__(self, args):
        if not args.is_MDM:
            WavTokenizer_relativeroot = 'utils/salsa_utils/libs/WavTokenizer'
        if args.is_MDM:
            WavTokenizer_relativeroot = os.path.join(args.parent_dir, 'utils/salsa_utils/libs/WavTokenizer')

        config_path = os.path.join(WavTokenizer_relativeroot,'configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml')
        model_path = os.path.join(WavTokenizer_relativeroot, 'results/train/wavtokenizer_large_unify_600_24k.ckpt')
        self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.wavtokenizer = self.wavtokenizer.to(device)
    def normalize(self, input_audio, sr):
        wav = convert_audio(input_audio, sr, 24000, 1)
        return wav, 24000
    def tokenize(self, input_audio24k):
        bandwidth_id = torch.tensor([0])
        input_audio24k = input_audio24k.to(device)
        features, discrete_code = self.wavtokenizer.encode_infer(input_audio24k,
                                                                 bandwidth_id=bandwidth_id)
        return discrete_code



    def normalize(self, sample_audio, sr):
        wav = convert_audio(sample_audio, sr, 24000, 1)
        return wav, 24000

# from MotionScript.captioning_motion_Salsa import MotionScript_Forward_Salsa
import MotionScript.captioning_motion_Salsa as MS_Salsa
# class MotionScript:
#
#     def __init__(self, args):
#         self.nothing = None
#     def normalize(self, input_audio, sr):
#
#     def tokenize(self, input_audio24k):
#
#
#
#
#     def normalize(self, sample_audio, sr):
#         wav = convert_audio(sample_audio, sr, 24000, 1)
#         return wav, 24000




class DataPreprocessor:
    """Loads and extracts skeleton, audio and video data from Lmdb files and writes those entires into a separate new Lmdb file.

    Attributes:
        src_lmdb_env: A Lmdb object containing the origin database environment (similar to PostgreSQL schema).
        dst_lmdb_env: A Lmdb object containing the destination database environment.
        n_videos: An integer number of entries in the database (equal to the number of videos in the training set).
        n_poses: An integer number of frames in each clip in the dataset (normally 30 (in 30 fps)).
        subdivision_stride: An integer number of frames between the start of one clip and the start of the next clip (clips can overlap).
        skeleton_resampling_fps: An integer frames per second of clip to use for training (usually downsampled to 20 fps, clips are normally 30 fps).
        audio_sample_length: An integer length of the audio clip in hertz (sampled at 16,000 Hz).
        n_out_samples: An integer total number of database entries (audio, video and skeleton) that has been extracted from the original videos.
        sentence_frame_length: An integer number of frames in each clip but for sentences rather than gestures.
        audio_sampling_rate: An integer sampling rate for an audio signal.
        DAE_frame_level: A DAE model only if args.name in the initialization method is not 'DAE'.
        rnn_representation: A VQVAE model only if 'sentence_level' is True else None.
        ckpt_path_DAE: A string filepath to a saved 'DAE' checkpoint model.
        ckpt_path_Autoencode: A string filepath to a saved VQVAE checkpoint model.
    """
    def __init__(self, args, clip_lmdb_dir: str, out_lmdb_dir: str, n_poses: int, subdivision_stride: int, pose_resampling_fps: int, sentence_level: bool = False):

        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.sentence_level = sentence_level
        self.src_lmdb_env: lmdb.Environment = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        self.out_lmdb_dir = out_lmdb_dir
        self.args = args
        with self.src_lmdb_env.begin() as txn:
            self.n_videos: int = txn.stat()['entries']

        # self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # self.ckpt_path_DAE: str = args.rep_learning_checkpoint
        # self.ckpt_path_Autoencode: str = args.autoencoder_checkpoint
        self.motion_tokenizer = Motion_tokenizer(args)
        self.audio_tokenizer = Audio_tokenizer(args)
        self.pair_dancer = True
        #Todo if args.name != "Frame_Level":
        #     self.DAE_frame_level: Tuple[argparse.Namespace, torch.nn.Module, torch.nn.MSELoss, Vocab, int] = utils.train_utils.load_checkpoint_and_model(
        #         self.ckpt_path_DAE, device,'DAE')
        #
        # if self.sentence_level:
        #     self.rnn_representation: Tuple[argparse.Namespace, torch.nn.Module, torch.nn.MSELoss, Vocab, int] = utils.train_utils.load_checkpoint_and_model(
        #         self.ckpt_path_Autoencode, device, 'autoencoder_vq')
        #     self.n_out_samples_has_move = 0

        # create db for samples
        map_size = 1024 * 30  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env: lmdb.Environment = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        # self.sentence_frame_length = args.sentence_frame_length
        self.audio_sampling_rate = 24000

        # self.Dance_moves_annot = pickle.load(open('Bermet/processed_dance_annotations.pk', 'rb'))

    def run(self) -> None:
        """Extract skeleton, audio, word data from source and write entries into a destination Lmdb file.

        Closes both src_lmdb_env and dst_lmdb_env database connections upon completion.
        Does not return any values. Modifies internal state of the object (Close db connection).
        """
        src_txn = self.src_lmdb_env.begin(write=False)
        total_count = src_txn.stat()['entries']

        # sampling and normalization
        cursor = src_txn.cursor()
        counter = 0
        for key, value in tqdm(cursor):
            print("video ", counter, "of", total_count, '\n')
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)
                counter = counter + 1
            # if counter > 2: break

        # print number of samples
        with self.dst_lmdb_env.begin() as txn:
            print("Sample_counter", txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid: str, clip: dict) -> None:
        """Internal function to extract and write skeleton, audio and word data from provided clip.

        Modifies internal state of the object (n_out_samples, n_poses, audio_sample_length).
        #TODO

        Args:
            vid: A string representing the name or id of the clip.
            clip: A dictionary containing the following string keys:
                'poses': A Numpy array of pose/gesture data.
                'audio_raw': A Numpy array of audio data.
                'words': A list of lists. Each internal list contains 3 elements:
                    index 0: A float start time.
                    index 1: A float end time.
                    index 2: A string word.
        """
        if self.pair_dancer == True : return self._sample_from_clip_pair(vid, clip)

        clip_skeleton3d: np.ndarray = clip['keypoints3d']
        clip_rotmat: np.ndarray = clip['rotmat']
        clip_raw_euler_poses = clip['raw_euler_poses']
        clip_raw_trans = clip['raw_trans']
        clip_audio_raw, clip_audio_raw_sr = clip['audio_raw'], clip['audio_sr']
        clip_audio_raw, clip_audio_raw_sr = self.audio_tokenizer.normalize(torch.from_numpy(clip_audio_raw),
                                                                           clip_audio_raw_sr)

        # clip_word_list: list[list] = clip['words']
        clip_HM3D_joint_vec = clip['HML3D_joints_vec']

        clipt_body_betas = clip['body_betas']
        clip_body_vertices = clip['body_vertices']
        clip_body_faces = clip['body_faces']

        # divide
        aux_info = []
        # sample_skeletons_list = []
        # sample_words_list = []

        sample_skeleton3d_list = []
        sample_rotmat_list = []
        sample_HML3D_joints_list = []
        sample_HML3D_joints_vec_list = []
        sample_vqtokens_list = []
        # sample_audio_list_mels = []
        sample_audio_raw_list = []
        sample_audio_tokens_list = []
        sample_ms_description_list = []
        # sentence_leve_latents_list = []
        # # GPT_3_STR_list = []
        # # GPT_3_Embedding_list = []

        # if self.sentence_level:
        #     self.n_poses = self.sentence_frame_length
        #     self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * self.audio_sampling_rate)

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * self.audio_sampling_rate)

        num_subdivision = math.floor(
            (len(clip_skeleton3d) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        # Sentence level preparation:




        for i in tqdm(range(num_subdivision)):



            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses
            if fin_idx>=len(clip_skeleton3d):
                print("^^^^^^1")
                continue
            sample_skeletons3d = clip_skeleton3d[start_idx:fin_idx]
            sample_rotmat = clip_rotmat[start_idx:fin_idx]
            sample_HML3D_joints_vec =  clip_HM3D_joint_vec[start_idx:fin_idx]
            sample_vqtokens = self.motion_tokenizer.npy263_tokenizer(sample_HML3D_joints_vec)
            sample_vqtokens = sample_vqtokens.cpu().detach().numpy()

            sample_raw_euler_poses = clip_raw_euler_poses[start_idx:fin_idx]
            sample_raw_trans = clip_raw_trans[start_idx:fin_idx]


            sample_body_betas = None # clipt_body_betas
            sample_body_vertices = None # clip_body_vertices[start_idx:fin_idx]
            sample_body_faces = None # clip_body_faces

            # from MotionScript.stmc_renderer.humor import HumorRenderer
            # smpl_renderer = HumorRenderer(20, imw=720, imh=720)
            # from smplx import SMPLX
            # smplx = SMPLX(model_path='utils\\salsa_utils\\SMPLX_DEP\\models_lockedhead\\smplx',
            #               gender="NEUTRAL",
            #               num_betas=16, use_pca=False, use_face_contour=True, flat_hand_mean=True)
            #
            # data = {'poses':  sample_raw_euler_poses}
            # data['trans'] = sample_raw_trans
            # itself = smplx.forward(
            #     # global_orient=torch.from_numpy(data['global_orient']).float(),
            #
            #     global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
            #     body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
            #
            #     transl=torch.from_numpy(data['trans']).float()
            # )
            # smpl_renderer(
            #     sample_HML3D_joints_vec.copy(),
            #     output='smpl_video_path.mp4',
            #     progress_bar=tqdm,
            # )
            sample_bin_ms = None
            if not self.args.is_MDM:
                S, T = 80, 140
                S, T = 0, -1
                input2MotionScript = {
                                    'poses': sample_raw_euler_poses[S:T].copy(),
                                    '3d_keypoints': sample_skeletons3d[S:T],
                                    'trans': sample_raw_trans[S:T],
                                    'body_betas': sample_body_betas,
                                    'body_vertices':   None, # sample_body_vertices[S:T],
                                    'body_faces': sample_body_faces
                                    }
                ablation = ['chronological']

                bining_details_printout, ms_non_agg, ms_agg, s, e  = MS_Salsa.MotionScript_Forward_Salsa(input2MotionScript,
                                                    motion_id=f'Win_{i}',
                                                    ablations=ablation)
                sample_bin_ms = ms_non_agg # We pick the simples plain textual rep.
                # import importlib
                # importlib.reload(MS_Salsa)


            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            # sample_words = self.get_words_in_time_range(word_list=clip_word_list,
            #                                             start_time=subdivision_start_time,
            #                                             end_time=subdivision_end_time)

            # if len(sample_words) < 4:
            #     continue

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton3d) * len(clip_audio_raw[0]))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[:, audio_start:audio_end]
            sample_audiotokens = self.audio_tokenizer.tokenize(sample_audio)
            sample_audiotokens = sample_audiotokens.squeeze().cpu().numpy()
            # mel_chunks = []
            # raw_chunks = []
            # for audio_sub in range(self.audio_sample_length//self.audio_sampling_rate):
            #     audio_chunk = sample_audio[audio_sub*self.audio_sampling_rate: (audio_sub+1)*self.audio_sampling_rate]
            #     signal = librosa.feature.melspectrogram(y=audio_chunk, sr=self.audio_sampling_rate)
            #     signal = librosa.power_to_db(signal, ref=np.max)
            #     mel_chunks.append(signal)
            #     # raw_chunks.append(audio_chunk)
            #     raw_chunks.append(0)
            #     # signal = librosa.amplitude_to_db(signal)


            # Extract dance moves labels

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            def get_dance_moves(motion_info_x):
                DD_vid = motion_info_x['vid']
                get_moves = []
                current_take = DD_vid  # [:15]
                current_role = 'Follower' if 'follower' in DD_vid else 'Leader'
                current_start_time, current_end_time = motion_info_x['start_time'], motion_info_x['end_time']

                def time2seconds(timestamp):
                    from datetime import datetime
                    # timestamp = "00:00:00.000"
                    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")  # Parses timestamp
                    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
                    return total_seconds

                # for i in range(len(self.Dance_moves_annot)):
                #     temp = self.Dance_moves_annot[i]
                #     if temp['ID'] in current_take and \
                #             current_role in temp['role']:
                #
                #         # print(temp['role'])
                #         if current_start_time >= time2seconds(temp['start_time']) and current_end_time <= time2seconds(
                #                 temp['end_time']):
                #             get_moves.append(self.Dance_moves_annot[i])
                for i in range(len(self.Dance_moves_annot)):
                    temp = self.Dance_moves_annot[i]
                    if temp['ID'] in current_take and current_role in temp['role']:
                        move_start_time = time2seconds(temp['start_time'])
                        move_end_time = time2seconds(temp['end_time'])
                        move_duration = move_end_time - move_start_time
                        overlap_start = max(current_start_time, move_start_time)
                        overlap_end = min(current_end_time, move_end_time)

                        if overlap_end > overlap_start:  # Check if there is an overlap
                            overlap_duration = overlap_end - overlap_start
                            overlap_percentage = overlap_duration / move_duration

                            if overlap_percentage > 0.6:  # Check if overlap is more than 60%
                                get_moves.append(self.Dance_moves_annot[i])
                return get_moves

            # dance_moves = get_dance_moves(motion_info)
            # motion_info['dance_moves'] = dance_moves


            sample_skeleton3d_list.append(sample_skeletons3d)
            sample_rotmat_list.append(sample_rotmat)
            sample_HML3D_joints_vec_list.append(sample_HML3D_joints_vec)
            sample_vqtokens_list.append(sample_vqtokens)
            sample_audio_tokens_list.append(sample_audiotokens)

            sample_ms_description_list.append(sample_bin_ms)

            # sample_words_list.append(sample_words)
            # sample_audio_list_mels.append(mel_chunks)
            # sample_audio_list_raws.append(raw_chunks)

            aux_info.append(motion_info)



        if len(sample_skeleton3d_list) > 0:
            with (self.dst_lmdb_env.begin(write=True) as txn):

                if not self.args.is_MDM:
                    for poses_keypoints3d, poses_rotmat, ms_description, \
                        poses_vq_tokens, audio_tokens, aux in \
                            zip(sample_skeleton3d_list, sample_rotmat_list, sample_ms_description_list,
                                sample_vqtokens_list, sample_audio_tokens_list, aux_info): # , GPT_3_Embedding_list):

                        poses_keypoints3d = np.asarray(poses_keypoints3d)
                        poses_rotmat = np.asarray(poses_rotmat)
                        poses_vqtokens = np.asarray(poses_vq_tokens)
                        audio_tokens = np.asarray(audio_tokens)
                        # GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [poses_keypoints3d, poses_rotmat, ms_description, poses_vqtokens, audio_tokens, aux]
                        # v = [words, poses, audio_raws, audio_mels, aux, sentence_leve_latents, GPT_3_Embedding]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

        print()


    def _sample_from_clip_pair(self, vid: str, clip: dict) -> None:
        """Internal function to extract and write skeleton, audio and word data from provided clip.

        Modifies internal state of the object (n_out_samples, n_poses, audio_sample_length).
        #TODO

        Args:
            vid: A string representing the name or id of the clip.
            clip: A dictionary containing the following string keys:
                'poses': A Numpy array of pose/gesture data.
                'audio_raw': A Numpy array of audio data.
                'words': A list of lists. Each internal list contains 3 elements:
                    index 0: A float start time.
                    index 1: A float end time.
                    index 2: A string word.
        """
        clip_skeleton3d_L: np.ndarray = clip['keypoints3d_L']
        clip_rotmat_L: np.ndarray = clip['rotmat_L']
        clip_raw_euler_poses_L = clip['raw_euler_poses_L']
        clip_raw_trans_L = clip['raw_trans_L']

        clip_skeleton3d_F: np.ndarray = clip['keypoints3d_F']
        clip_rotmat_F: np.ndarray = clip['rotmat_F']
        clip_raw_euler_poses_F = clip['raw_euler_poses_F']
        clip_raw_trans_F = clip['raw_trans_F']

        clip_audio_raw, clip_audio_raw_sr = clip['audio_raw'], clip['audio_sr']
        clip_audio_raw, clip_audio_raw_sr = self.audio_tokenizer.normalize(torch.from_numpy(clip_audio_raw),
                                                                           clip_audio_raw_sr)

        # clip_word_list: list[list] = clip['words']
        clip_HM3D_joint_vec_L = clip['HML3D_joints_vec_L']
        clip_HM3D_joint_vec_F = clip['HML3D_joints_vec_F']

        clipt_body_betas = clip['body_betas']
        clip_body_vertices = clip['body_vertices']
        clip_body_faces = clip['body_faces']

        # divide
        aux_info = []



        # Leader's sample list
        sample_skeleton3d_list_L = []
        sample_rotmat_list_L = []
        sample_HML3D_vec_list_L = []
        sample_vqtokens_list_L = []
        sample_ms_description_list_L = []

        # Follower's sample list
        sample_skeleton3d_list_F = []
        sample_rotmat_list_F = []
        sample_HML3D_vec_list_F = []
        sample_vqtokens_list_F = []
        sample_ms_description_list_F = []

        # sample_audio_list_mels = []
        sample_audio_raw_list = []
        sample_audio_tokens_list = []


        # sentence_leve_latents_list = []
        # # GPT_3_STR_list = []
        # # GPT_3_Embedding_list = []

        # if self.sentence_level:
        #     self.n_poses = self.sentence_frame_length
        #     self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * self.audio_sampling_rate)

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * self.audio_sampling_rate)

        num_subdivision = math.floor(
            (len(clip_skeleton3d_L) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        # Sentence level preparation:




        for i in tqdm(range(num_subdivision)):

            # if i>2:break

            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses
            if fin_idx>=len(clip_skeleton3d_L):
                print("^^^^^^1")
                continue
            # Leader's
            sample_skeletons3d_L = clip_skeleton3d_L[start_idx:fin_idx]
            sample_rotmat_L = clip_rotmat_L[start_idx:fin_idx]
            sample_HML3D_joints_vec_L =  clip_HM3D_joint_vec_L[start_idx:fin_idx]
            sample_vqtokens_L = self.motion_tokenizer.npy263_tokenizer(sample_HML3D_joints_vec_L)
            sample_vqtokens_L = sample_vqtokens_L.cpu().detach().numpy()
            sample_raw_euler_poses_L = clip_raw_euler_poses_L[start_idx:fin_idx]
            sample_raw_trans_L = clip_raw_trans_L[start_idx:fin_idx]

            # Follower's
            sample_skeletons3d_F = clip_skeleton3d_F[start_idx:fin_idx]
            sample_rotmat_F = clip_rotmat_F[start_idx:fin_idx]
            sample_HML3D_joints_vec_F =  clip_HM3D_joint_vec_F[start_idx:fin_idx]
            sample_vqtokens_F = self.motion_tokenizer.npy263_tokenizer(sample_HML3D_joints_vec_F)
            sample_vqtokens_F = sample_vqtokens_F.cpu().detach().numpy()
            sample_raw_euler_poses_F = clip_raw_euler_poses_F[start_idx:fin_idx]
            sample_raw_trans_F = clip_raw_trans_F[start_idx:fin_idx]


            sample_body_betas = None # clipt_body_betas
            sample_body_vertices = None # clip_body_vertices[start_idx:fin_idx]
            sample_body_faces = None # clip_body_faces

            # from MotionScript.stmc_renderer.humor import HumorRenderer
            # smpl_renderer = HumorRenderer(20, imw=720, imh=720)
            # from smplx import SMPLX
            # smplx = SMPLX(model_path='utils\\salsa_utils\\SMPLX_DEP\\models_lockedhead\\smplx',
            #               gender="NEUTRAL",
            #               num_betas=16, use_pca=False, use_face_contour=True, flat_hand_mean=True)
            #
            # data = {'poses':  sample_raw_euler_poses}
            # data['trans'] = sample_raw_trans
            # itself = smplx.forward(
            #     # global_orient=torch.from_numpy(data['global_orient']).float(),
            #
            #     global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
            #     body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
            #
            #     transl=torch.from_numpy(data['trans']).float()
            # )
            # smpl_renderer(
            #     sample_HML3D_joints_vec.copy(),
            #     output='smpl_video_path.mp4',
            #     progress_bar=tqdm,
            # )

            if not self.args.is_MDM:
                # S, T = 80, 140
                S, T = 0, -1
                ablation = ['chronological']

                # Leader's
                input2MotionScript = {
                                    'poses': sample_raw_euler_poses_L[S:T].copy(),
                                    '3d_keypoints': sample_skeletons3d_L[S:T],
                                    'trans': sample_raw_trans_L[S:T],
                                    'body_betas': None,         # sample_body_betas,
                                    'body_vertices':   None,    # sample_body_vertices[S:T],
                                    'body_faces': None          # sample_body_faces
                                    }


                bining_details_printout, ms_non_agg_L, ms_agg_L, s, e  = \
                    MS_Salsa.MotionScript_Forward_Salsa(input2MotionScript,
                                                    motion_id=f'Win_{i}',
                                                    ablations=ablation)
                sample_bin_ms_L = ms_agg_L # We pick the simples plain textual rep (non-aggregated).

                # Follower's
                input2MotionScript = {
                    'poses': sample_raw_euler_poses_F[S:T].copy(),
                    '3d_keypoints': sample_skeletons3d_F[S:T],
                    'trans': sample_raw_trans_F[S:T],
                    'body_betas': None,  # sample_body_betas,
                    'body_vertices': None,  # sample_body_vertices[S:T],
                    'body_faces': None  # sample_body_faces
                }
                bining_details_printout, ms_non_agg_F, ms_agg_F, s, e = \
                    MS_Salsa.MotionScript_Forward_Salsa(input2MotionScript,
                                                        motion_id=f'Win_{i}',
                                                        ablations=ablation)

                sample_bin_ms_F = ms_agg_F
                # We pick the simples plain
                # textual rep (non-aggregated through time by setting max_range to zerp).


                # import importlib
                # importlib.reload(MS_Salsa)
            else:
                sample_bin_ms_L = []
                sample_bin_ms_F = []

            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            # sample_words = self.get_words_in_time_range(word_list=clip_word_list,
            #                                             start_time=subdivision_start_time,
            #                                             end_time=subdivision_end_time)

            # if len(sample_words) < 4:
            #     continue

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton3d_L) * len(clip_audio_raw[0]))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[:, audio_start:audio_end]
            sample_audiotokens = self.audio_tokenizer.tokenize(sample_audio)
            sample_audiotokens = sample_audiotokens.squeeze().cpu().numpy()
            # mel_chunks = []
            # raw_chunks = []
            # for audio_sub in range(self.audio_sample_length//self.audio_sampling_rate):
            #     audio_chunk = sample_audio[audio_sub*self.audio_sampling_rate: (audio_sub+1)*self.audio_sampling_rate]
            #     signal = librosa.feature.melspectrogram(y=audio_chunk, sr=self.audio_sampling_rate)
            #     signal = librosa.power_to_db(signal, ref=np.max)
            #     mel_chunks.append(signal)
            #     # raw_chunks.append(audio_chunk)
            #     raw_chunks.append(0)
            #     # signal = librosa.amplitude_to_db(signal)


            # Extract dance moves labels

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            def get_dance_moves(motion_info_x):
                DD_vid = motion_info_x['vid']
                get_moves = []
                current_take = DD_vid  # [:15]
                current_role = 'Follower' if 'follower' in DD_vid else 'Leader'
                current_start_time, current_end_time = motion_info_x['start_time'], motion_info_x['end_time']

                def time2seconds(timestamp):
                    from datetime import datetime
                    # timestamp = "00:00:00.000"
                    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")  # Parses timestamp
                    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
                    return total_seconds

                # for i in range(len(self.Dance_moves_annot)):
                #     temp = self.Dance_moves_annot[i]
                #     if temp['ID'] in current_take and \
                #             current_role in temp['role']:
                #
                #         # print(temp['role'])
                #         if current_start_time >= time2seconds(temp['start_time']) and current_end_time <= time2seconds(
                #                 temp['end_time']):
                #             get_moves.append(self.Dance_moves_annot[i])
                for i in range(len(self.Dance_moves_annot)):
                    temp = self.Dance_moves_annot[i]
                    if temp['ID'] in current_take and current_role in temp['role']:
                        move_start_time = time2seconds(temp['start_time'])
                        move_end_time = time2seconds(temp['end_time'])
                        move_duration = move_end_time - move_start_time
                        overlap_start = max(current_start_time, move_start_time)
                        overlap_end = min(current_end_time, move_end_time)

                        if overlap_end > overlap_start:  # Check if there is an overlap
                            overlap_duration = overlap_end - overlap_start
                            overlap_percentage = overlap_duration / move_duration

                            if overlap_percentage > 0.6:  # Check if overlap is more than 60%
                                get_moves.append(self.Dance_moves_annot[i])
                return get_moves

            # dance_moves = get_dance_moves(motion_info)
            # motion_info['dance_moves'] = dance_moves


            # Leader's
            sample_skeleton3d_list_L.append(sample_skeletons3d_L)
            sample_rotmat_list_L.append(sample_rotmat_L)
            if self.args.is_MDM:
                sample_HML3D_vec_list_L.append(sample_HML3D_joints_vec_L)
            sample_vqtokens_list_L.append(sample_vqtokens_L)
            sample_ms_description_list_L.append(sample_bin_ms_L)

            # Follower's
            sample_skeleton3d_list_F.append(sample_skeletons3d_F)
            sample_rotmat_list_F.append(sample_rotmat_F)
            if self.args.is_MDM:
                sample_HML3D_vec_list_F.append(sample_HML3D_joints_vec_F)
            sample_vqtokens_list_F.append(sample_vqtokens_F)
            sample_ms_description_list_F.append(sample_bin_ms_F)

            sample_audio_tokens_list.append(sample_audiotokens)


            # sample_words_list.append(sample_words)
            # sample_audio_list_mels.append(mel_chunks)
            # sample_audio_list_raws.append(raw_chunks)

            aux_info.append(motion_info)



        if len(sample_skeleton3d_list_L) > 0:
            # with ((((self.dst_lmdb_env.begin(write=True) as txn)))):
            with (self.dst_lmdb_env.begin(write=True) as txn):
                if not self.args.is_MDM:
                    for poses_keypoints3d_L, poses_keypoints3d_F, \
                        poses_rotmat_L, poses_rotmat_F, \
                        ms_description_L, ms_description_F, \
                        poses_vq_tokens_L, poses_vq_tokens_F, \
                         audio_tokens, aux in \
                            zip(sample_skeleton3d_list_L, sample_skeleton3d_list_F,
                                sample_rotmat_list_L, sample_rotmat_list_F,
                                sample_ms_description_list_L, sample_ms_description_list_F,
                                sample_vqtokens_list_L, sample_vqtokens_list_F,
                                sample_audio_tokens_list, aux_info):

                        poses_keypoints3d_L = np.asarray(poses_keypoints3d_L)
                        poses_rotmat_L = np.asarray(poses_rotmat_L)
                        poses_vqtokens_L = np.asarray(poses_vq_tokens_L)

                        poses_keypoints3d_F = np.asarray(poses_keypoints3d_F)
                        poses_rotmat_F = np.asarray(poses_rotmat_F)
                        poses_vqtokens_F = np.asarray(poses_vq_tokens_F)

                        audio_tokens = np.asarray(audio_tokens)
                        # GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [poses_keypoints3d_L, poses_rotmat_L, ms_description_L, poses_vqtokens_L,
                             poses_keypoints3d_F, poses_rotmat_F, ms_description_F, poses_vqtokens_F,
                             audio_tokens, aux]
                        # v = [words, poses, audio_raws, audio_mels, aux, sentence_leve_latents, GPT_3_Embedding]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
                if self.args.is_MDM:
                    for poses_keypoints3d_L, poses_keypoints3d_F, \
                            poses_rotmat_L, poses_rotmat_F, \
                            HML3D_vec_L, HML3D_vec_F, \
                            ms_description_L, ms_description_F, \
                            poses_vq_tokens_L, poses_vq_tokens_F, \
                            audio_tokens, aux in \
                            zip(sample_skeleton3d_list_L, sample_skeleton3d_list_F,
                                sample_rotmat_list_L, sample_rotmat_list_F,
                                sample_HML3D_vec_list_L, sample_HML3D_vec_list_F,
                                sample_ms_description_list_L, sample_ms_description_list_F,
                                sample_vqtokens_list_L, sample_vqtokens_list_F,
                                sample_audio_tokens_list, aux_info):
                        poses_keypoints3d_L = np.asarray(poses_keypoints3d_L)
                        poses_rotmat_L = np.asarray(poses_rotmat_L)
                        HML3D_vec_L = np.asarray(HML3D_vec_L)
                        poses_vqtokens_L = np.asarray(poses_vq_tokens_L)

                        poses_keypoints3d_F = np.asarray(poses_keypoints3d_F)
                        poses_rotmat_F = np.asarray(poses_rotmat_F)
                        HML3D_vec_F = np.asarray(HML3D_vec_F)
                        poses_vqtokens_F = np.asarray(poses_vq_tokens_F)

                        audio_tokens = np.asarray(audio_tokens)
                        # GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [poses_keypoints3d_L, poses_rotmat_L, HML3D_vec_L, ms_description_L, poses_vqtokens_L,
                             poses_keypoints3d_F, poses_rotmat_F, HML3D_vec_F, ms_description_F, poses_vqtokens_F,
                             audio_tokens, aux]
                        # v = [words, poses, audio_raws, audio_mels, aux, sentence_leve_latents, GPT_3_Embedding]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

        print()

# Todo -----------------------------------------------------------------
PAIR2LEVEL = {
    f"pair{i}": level
    for i, level in zip(range(1, 10), ["beginner", "intermediate", "beginner",
                                       "intermediate", "professional", "intermediate",
                                       "professional", "beginner", "professional"], )
}

SALSA_CAPTIONS = {
    "beginner": [
        "A beginner salsa dancer practices simple steps with careful timing.",
        "A novice salsa dancer moves cautiously to the rhythm.",
        "A beginner salsa dancer performs basic footwork with focused effort.",
        "A new salsa dancer follows the beat with steady and controlled movements.",
        "A first-time salsa dancer attempts a slow and structured routine."
    ],
    "intermediate": [
        "An intermediate salsa dancer combines footwork and turns with growing confidence.",
        "A mid-level salsa dancer executes a balanced and expressive routine.",
        "An intermediate dancer performs with more rhythm and body coordination.",
        "A salsa dancer at intermediate level adds flair while maintaining structure.",
        "An intermediate-level salsa dancer blends technical steps with smoother transitions."
    ],
    "professional": [
        "A professional salsa dancer delivers a dynamic and polished performance.",
        "A skilled salsa dancer flows through complex moves with ease.",
        "A professional dancer commands the floor with sharp and expressive motion.",
        "A seasoned salsa dancer performs an intricate routine with confidence.",
        "An expert salsa dancer dazzles with swift, precise, and rhythmic movements."
    ]
}

class Salsa_Dataset(Dataset):
    """Contains information and parameters of a (Trinity) dataset.

    This is a PyTorch Dataset subclass containing information of a Trinity dataset.
    https://trinityspeechgesture.scss.tcd.ie/data/Trinity%20Speech-Gesture%20I/GENEA_Challenge_2020_data_release/

    This class is #TODO.

    Attributes:
        lmdb_dir: A string filepath of the directory containing the actual dataset.
        lmdb_env: A Lmdb object loaded from .mdb files in the lmdb_dir.
        n_poses: An integer number of frames in each clip in the dataset (normally 30 (in 30 fps)).
        subdivision_stride: An integer number of frames between the start of one clip and the start of the next clip (clips can overlap).
        skeleton_resampling_fps: An integer frames per second of clip to use for training (usually downsampled to 20 fps, clips are normally 30 fps).
        n_samples: An integer number of clips/entries in the original dataset.
        lang_model: A 'Vocab' pre-trained word vector representation or None.
        data_mean: A mean calculated from each video in the original dataset.
        data_std: A standard deviation calculcated from each video.
        pairwise_enabled: #TODO
        use_derivative: Boolean to stack gradients to data during training.
        encoded_labeled_poses: #TODO
        rep_learning_dim: An integer dimension of the model (unused).
        rep_learning_checkpoint: A string filepath to saved DAE model checkpoints.
        rep_model: A DAE neural net model loaded from the above checkpoint.
            Models: VQ_Frame, VAE_Network, DAE_Network in DAE_model.py.
    """

    def __init__(
        self,
        args,
        lmdb_dir: str,
        n_poses: int,
        subdivision_stride: int,
        pose_resampling_fps: int,
        # data_mean: list[float],
        # data_std: list[float],
    ):
        """Initialize with dataset location and several parameters.

        The args argument must contain the following keys:
            name: A string name of the model (ex. 'DAE' or 'autoencoder_vq').
            rep_learning_checkpoint: If name is not 'DAE', a string filepath to a saved 'DAE' checkpoint model.
            autoencoder_checkpoint: If sentence level is True, a string filepath to a saved VQVAE checkpoint model.
            sentence_frame_length: An integer number of frames in each clip (for a sentence instead of gesture).
            rep_learning_checkpoint: A string filepath to saved model checkpoints.
            use_derivative: A boolean whether to use derivatives.

        Args:
            args: A configargparse object containing parameters (See above).
            lmdb_dir: A string representing the filepath of the directory containing the actual dataset.
            n_poses: An int representing the number of frames in each clip in the dataset (normally 30 (in 30 fps)).
            subdivision_stride: An int representing the number of frames between the start of one clip and the start of the next clip (clips can overlap).
            pose_resampling_fps: An int representing the frames per second of clip to use for training (usually downsampled to 20 fps, clips are normally 30 fps).
            data_mean: A mean calculated from each video in the original dataset.
            data_std: A standard deviation calculcated from each video in the original dataset.
        """
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.args = args

        print("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + "_cache"
        if self.args.is_MDM:
            preloaded_dir += '_MDM'
        if not os.path.exists(preloaded_dir): # TODO
            data_sampler = DataPreprocessor(
                args,
                lmdb_dir,
                preloaded_dir,
                n_poses,
                subdivision_stride,
                self.skeleton_resampling_fps
            )
            data_sampler.run()
        else:
            print("Found pre-loaded samples from {}".format(preloaded_dir))

        # init lmdb
        self.lmdb_env: lmdb.Environment = lmdb.open(
            preloaded_dir, readonly=True, lock=False
        )
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]



    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The integer size of samples in the dataset.
        """
        return self.n_samples-1 # last item is None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the item at a specific index in the dataset.

        Args:
            idx: An integer index of the item to get.

        Returns:
            A 2-Tuple:
                encoded_poses: A Tensor of pose/gesture data.
                encoded_poses: The same tensor as above.
        """
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:010}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            # sample: Tuple[
            #     np.ndarray, np.ndarray, np.ndarray, dict
            # ] = pyarrow.deserialize(sample)
            # word_seq, pose_seq, audio, aux_info = sample

            # pose_seq_keypoints3d, pose_seq_rotmat, vq_tokens, aux_info = sample
            # pose_seq_keypoints3d, pose_seq_rotmat, ms_desc_bins, vq_tokens, audio_tokens, aux_info = sample
            if not self.args.is_MDM:
                poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                 poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                 audio_tokens, aux_info = sample
            if self.args.is_MDM:
                poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                    poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                    audio_tokens, aux_info = sample
                HML3D_L = torch.from_numpy(HML3D_L).to(self.args.device)
                HML3D_F = torch.from_numpy(HML3D_F).to(self.args.device)

        vq_tokens_L = torch.from_numpy(vq_tokens_L).to(self.args.device)
        vq_tokens_F = torch.from_numpy(vq_tokens_F).to(self.args.device)

        # Todo: this VQ_tokens are already transfered to LLM space (nb_llm_tokens+2 was addded)
        # Todo: consider add the second dancer ---- Done!
        # Todo: Now we need to call the prompt function.

        level = PAIR2LEVEL[(aux_info['vid'][:5]).lower()]
        caption = random.choice(SALSA_CAPTIONS[level])

        audio_tokens = torch.from_numpy(audio_tokens).to(self.args.device)

        # we need to return a one str here.
        if not self.args.is_MDM:
            return level, '-->'.join(ms_desc_L), '-->'.join(ms_des_F), vq_tokens_L, vq_tokens_F, audio_tokens, aux_info
        if self.args.is_MDM:
            return level, HML3D_L, vq_tokens_L, HML3D_F, vq_tokens_F, audio_tokens, aux_info
    def create_similarity_dataset(self, pickle_file: str, labelstxt_file: str) -> None:
        """TODO"""
        # Todo: 1. Thos function gets the pickle file that I made in the clustering.py(or flowgmm) process as well
        # Todo: as the labels text file that I annotated in the Unity application.
        # Todo: 2. Then I will creat those pairs of similarity and dissimilarity
        # Todo: 3. Finally, we store the pairs into the class.
        # Todo: We will use pairwise label and an extra loss in backpropagation process later.

        # 1. call preprocess load
        (
            self.data_rnn,
            self.labels,
            self.pairwise_labels,
            self.data_original,
        ) = self.load_gesture_data(pickle_file, labelstxt_file)

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        self.data_original = (self.data_original - self.data_mean) / std

        target = torch.from_numpy(self.data_original)

        target = target.float()
        # target = target.to(device)
        with torch.no_grad():
            self.encoded_labeled_poses = self.rep_model.encoder(target)
            if self.use_derivative:
                diff = [
                    (
                        self.encoded_labeled_poses[n, :]
                        - self.encoded_labeled_poses[n - 1, :]
                    )
                    for n in range(1, self.encoded_labeled_poses.shape[0])
                ]
                diff.insert(0, torch.zeros_like(self.encoded_labeled_poses[0, :]))
                self.encoded_labeled_poses = torch.cat(
                    (self.encoded_labeled_poses, torch.stack(diff)), dim=2
                )
        self.pairwise_enabled = True
        pass

    def get_labeled_(
        self, count: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO"""
        stack_pairs1 = torch.zeros(
            count,
            self.encoded_labeled_poses.shape[1],
            self.encoded_labeled_poses.shape[2],
        )
        stack_pairs2 = torch.zeros(
            count,
            self.encoded_labeled_poses.shape[1],
            self.encoded_labeled_poses.shape[2],
        )
        stack_labels = torch.zeros(count)
        rnds = random.sample(range(1, len(self.pairwise_labels)), 3)
        k = 0
        for rnd in rnds:
            current_pair = self.pairwise_labels[rnd]
            s1_ = self.encoded_labeled_poses[current_pair[0]]
            s2_ = self.encoded_labeled_poses[current_pair[1]]
            ss_label = current_pair[2]
            stack_pairs1[k, :, :] = s1_
            stack_pairs2[k, :, :] = s2_
            stack_labels[k] = ss_label
            k = k + 1

        return stack_pairs1, stack_pairs2, stack_labels






# Usage example of motion_tokenizer
from options.option_llm import get_args_parser
#
# args = get_args_parser()
# args.save_dir = "./demo"
# args.device = 'cuda:0'
# #
# # object = Motion_tokenizer(args)
# #
# i_motion = np.load('utils/salsa_utils/Salsa_Temp/lmdb_Salsa/SFU_SALSA_EXAMPLE.npy')
# # sample_tokens = object.npy263_tokenizer(i_motion)
# # object.sanity_check(sample_tokens, i_motion)
# # print()
#
#
#
# x = Salsa_dataset(args,
#                     lmdb_dir='utils/salsa_utils/Salsa_Temp/lmdb_Salsa/lmdb_train',
#                     n_poses=160,
#                     subdivision_stride=50,
#                     pose_resampling_fps=20)
# object = Motion_tokenizer(args)
# object.sanity_check(x.__getitem__(111), i_motion[:20])
#
# print()