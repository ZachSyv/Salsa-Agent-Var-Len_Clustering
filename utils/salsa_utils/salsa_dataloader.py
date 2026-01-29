"""create data samples
"""
import os

import sys

import math
import pickle
import os
from typing import Tuple, Optional
from pathlib import Path

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

# WavTokenizer expects decoder/encoder as top-level; add its root to sys.path before importing
_wavtok_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "libs", "WavTokenizer"))
if _wavtok_root not in sys.path:
    sys.path.insert(0, _wavtok_root)

from utils.salsa_utils.libs.WavTokenizer.encoder.utils import convert_audio

from utils.salsa_utils.libs.WavTokenizer.decoder.pretrained import WavTokenizer

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
# import utils.salsa_utils.libs.MotionScript.captioning_motion_Salsa as MS_Salsa  # Commented out for Ubuntu compatibility
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
        
        # Annotation file cache (per take ID)
        self.annotation_cache = {}
        
        # Move class mapping (README Glossary + utils_gesture2vec). "Suzy Q" per README.
        self.MOVE_MAPPING = {
            'Arm lock': ['arm lock'],
            'Basic step': ['basic step', 'side basic step', 'cross back basic step', 'back basic step'],
            'Body Shake': ['body shake'],
            'Body roll': ['body roll'],
            'Change of Directions': ['change of directions', 'swap position'],
            'Check': ['check'],
            'Comb': ['comb'],
            'Copa': ['copa'],
            'Dile que no': ['dile que no'],
            'Hand throw': ['hand throw', 'right hand throw', 'left hand throw', 'double hand throw'],
            'right turn': ['right turn for the', 'right turn'],
            'Drawing circle': ['drawing circle'],
            'Enchufla': ['enchufla'],
            'walks around': ['walks around'],
            'Suzy': ['suzy', 'suzy q'],
            'Hip movement': ['hip movement'],
            'Kicks': ['kicks'],
            'Lasso': ['lasso'],
            'Natural top': ['natural top'],
            'Left turn': ['left turn for the', 'left turn'],
            'Mambo': ['mambo'],
            'Open break': ['open break'],
            'Point': ['point'],
            'Sliding': ['sliding'],
            'Standing': ['standing'],
            'Steps': ['steps', 'advanced footwork', 'footwork variation', 'twisting on beat', 'footwork'],
            'Swing': ['swing'],
            'Walk': ['walk'],
            'XBL': ['xbl'],
            'Siete': ['siete'],
            'Indescribable': ['indescribable', 'misread signals', 'misinterpreted signal', 'failed move', 'mixed signals and failed move'],
            'Markers Swap issue': ['markers swap issue']
        }
        self.ERROR_MAPPING = {
            'Misinterpreted signal': ['misinterpreted', 'misread', 'misunderstand'],
            'Misstep': ['misstep', 'incorrect foot', 'wrong foot'],
            'Mixed signals': ['mixed signals', 'conflicting'],
            'Off beat': ['off beat'],
        }
        
        # Dataset root for annotations: Dataset/compas3d/PairX/PairX_songY_takeZ/*.txt
        _dr = os.environ.get('SALSA_DATASET_ROOT') or os.environ.get('SALSA_DATA_ROOT')
        if _dr:
            self._dataset_root = Path(_dr)
        else:
            self._dataset_root = Path('/localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/Dataset')
        if not self._dataset_root.exists():
            self._dataset_root = Path(__file__).resolve().parents[4] / 'Dataset'
        
        # Lazy loading for InterHuman tokenizers (only load if needed)
        self.interhuman_motion_tokenizer = None
        self.relationship_tokenizer = None
        self.interhuman_normalization_stats = None
        self.relationship_normalization_stats = None

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
            print("\n\nvideo ", counter, "of", total_count, '\n\n')
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

    def _extract_take_id_from_vid(self, vid: str) -> Optional[str]:
        """Extract take ID from vid for annotation lookup.
        
        Vid is e.g. "Pair1_song1_take1_leader,Pair1_song1_take1_follower".
        Take ID = "Pair1_song1_take1". Annotations live only in Dataset/compas3d/PairX/PairX_songY_takeZ/PairX_songY_takeZ.txt.
        """
        if ',' in vid:
            base = vid.split(',')[0].strip()
        else:
            base = vid.strip()
        for suffix in ('_leader', '_leader_subject', '_follower', '_follower_subject'):
            if base.lower().endswith(suffix):
                base = base[:-len(suffix)].rstrip('_')
                break
        if not base or 'pair' not in base.lower().split('_')[0]:
            return None
        parts = base.split('_')
        if len(parts) < 3:
            return None
        return base
    
    def _load_annotation_file_for_take(self, take_id: str, vid: str = '') -> Optional[list]:
        """Load annotation .txt only from the take folder in Dataset/compas3d.
        
        Path: {dataset_root}/compas3d/PairX/PairX_songY_takeZ/PairX_songY_takeZ.txt.
        A take has annotations iff that .txt exists in its folder; we do not use any other source.
        """
        if take_id in self.annotation_cache:
            return self.annotation_cache[take_id]
        
        # e.g. Pair1_song1_take1 -> compas3d/Pair1/Pair1_song1_take1/Pair1_song1_take1.txt
        parts = take_id.split('_')
        if len(parts) < 3 or 'pair' not in parts[0].lower():
            self.annotation_cache[take_id] = None
            return None
        pair_folder = parts[0]
        ann_path = self._dataset_root / 'compas3d' / pair_folder / take_id / f"{take_id}.txt"
        
        if not ann_path.exists():
            self.annotation_cache[take_id] = None
            return None
        
        annotations = []
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) < 8:
                        continue
                    ann_type = parts[0]
                    role = parts[1] if len(parts) > 1 else ''
                    start_time_sec = float(parts[3])
                    end_time_sec = float(parts[5])
                    duration_sec = float(parts[7])
                    description = (parts[8] if len(parts) > 8 else '').strip()
                    annotations.append({
                        'type': ann_type,
                        'role': role,
                        'start_time': start_time_sec,
                        'end_time': end_time_sec,
                        'duration': duration_sec,
                        'description': description,
                        'start_time_str': parts[2],
                        'end_time_str': parts[4],
                        'duration_str': parts[6],
                    })
            self.annotation_cache[take_id] = annotations
            if not getattr(self, '_ann_load_logged', None):
                self._ann_load_logged = set()
            if take_id not in self._ann_load_logged:
                self._ann_load_logged.add(take_id)
                print(f"[Annotations] Loaded {ann_path.name} ({len(annotations)} entries) for take {take_id}")
            return annotations
        except Exception as e:
            print(f"Warning: Error loading annotation file {ann_path}: {e}")
            self.annotation_cache[take_id] = None
            return None
    
    def _extract_move_class(self, description: str) -> Optional[str]:
        """Extract move class from description using MOVE_MAPPING heuristics.
        
        Returns move class name or None if not found.
        """
        if not description:
            return None
        
        description_lower = description.lower()
        
        # Check each move class pattern
        for class_name, patterns in self.MOVE_MAPPING.items():
            if any(pattern in description_lower for pattern in patterns):
                return class_name
        
        # If no match found, return None (will be flagged)
        return None
    
    def _extract_error_class(self, description: str) -> Optional[str]:
        """Extract error class from description using ERROR_MAPPING (README Glossary)."""
        if not description:
            return None
        d = description.lower()
        for class_name, patterns in self.ERROR_MAPPING.items():
            if any(p in d for p in patterns):
                return class_name
        return None
    
    def _match_annotations_to_window(self, annotations: list, start_time: float, end_time: float) -> dict:
        """Match annotations to a time window.
        
        Returns dict with keys: 'moves', 'errors', 'styling_leader', 'styling_follower'
        Each contains list of matched annotation dicts with overlap info.
        """
        matched = {
            'moves': [],
            'errors': [],
            'styling_leader': [],
            'styling_follower': []
        }
        
        if not annotations:
            return matched
        
        for ann in annotations:
            ann_start = ann['start_time']
            ann_end = ann['end_time']
            
            # Check overlap
            overlap_start = max(start_time, ann_start)
            overlap_end = min(end_time, ann_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                ann_duration = ann_end - ann_start
                overlap_percentage = overlap_duration / ann_duration if ann_duration > 0 else 0
                
                # Only include if significant overlap (>30%)
                if overlap_percentage > 0.3:
                    desc = ann.get('description') or ''
                    desc_lower = desc.lower()
                    move_class = self._extract_move_class(desc) if ann['type'] != 'Errors' else None
                    error_class = self._extract_error_class(desc) if ann['type'] == 'Errors' else None
                    ann_info = {
                        'type': ann['type'],
                        'role': ann['role'],
                        'description': desc,
                        'move_class': move_class,
                        'error_class': error_class,
                        'start_time': ann_start,
                        'end_time': ann_end,
                        'duration': ann_duration,
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_percentage': overlap_percentage
                    }
                    
                    if ann['type'] == 'Errors':
                        matched['errors'].append(ann_info)
                    elif ann['type'] == 'Separate_Leader' or (ann['type'] == 'Together' and 'Leader' in ann['role']):
                        if 'styling' in desc_lower or 'man styling' in desc_lower:
                            matched['styling_leader'].append(ann_info)
                        elif desc.strip():
                            matched['moves'].append(ann_info)
                    elif ann['type'] == 'Separate_Follower' or (ann['type'] == 'Together' and 'Follower' in ann['role']):
                        if 'styling' in desc_lower or 'lady styling' in desc_lower:
                            matched['styling_follower'].append(ann_info)
                        elif desc.strip():
                            matched['moves'].append(ann_info)
                    elif ann['type'] == 'Together':
                        if 'styling' in desc_lower:
                            matched['styling_leader'].append(ann_info)
                            matched['styling_follower'].append(ann_info)
                        elif desc.strip():
                            matched['moves'].append(ann_info)
        
        return matched
    
    def _get_annotations_for_window(self, vid: str, start_time: float, end_time: float) -> dict:
        """Get annotations for a time window from the corresponding annotation file.
        
        This is the main function to call during window sampling.
        Returns dict with 'moves', 'errors', 'styling_leader', 'styling_follower'.
        """
        # Extract take ID from vid
        take_id = self._extract_take_id_from_vid(vid)
        if not take_id:
            return {'moves': [], 'errors': [], 'styling_leader': [], 'styling_follower': []}
        
        # Load annotation file for this take (try take_id and vid-based alternates)
        annotations = self._load_annotation_file_for_take(take_id, vid=vid)
        if not annotations:
            return {'moves': [], 'errors': [], 'styling_leader': [], 'styling_follower': []}
        
        # Match annotations to window
        matched = self._match_annotations_to_window(annotations, start_time, end_time)
        
        # Flag non-empty moves without class for user review (skip empty descriptions)
        for move in matched['moves']:
            if move['move_class'] is None and (move.get('description') or '').strip():
                print(f"⚠️  Unclassified move: '{move['description']}' (take: {take_id}, time: {move['start_time']:.2f}-{move['end_time']:.2f}s)")
        
        return matched
    
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
        
        # Lazy import MS_Salsa only when needed (for cache creation; not needed if cache exists)
        global MS_Salsa
        if 'MS_Salsa' not in globals():
            try:
                import utils.salsa_utils.libs.MotionScript.captioning_motion_Salsa as MS_Salsa
            except ImportError:
                raise ImportError("MS_Salsa (MotionScript) is required for cache creation but not available. "
                                "Cache should already exist at the expected path. If cache doesn't exist, "
                                "ensure MotionScript dependencies are installed.")

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


            # Extract dance moves labels from annotation files

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            # Get annotations for this window
            annotations = self._get_annotations_for_window(vid, subdivision_start_time, subdivision_end_time)
            motion_info['dance_moves'] = annotations['moves']
            motion_info['errors'] = annotations['errors']
            motion_info['styling_leader'] = annotations['styling_leader']
            motion_info['styling_follower'] = annotations['styling_follower']


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

    def _load_interhuman_tokenizers(self):
        """Lazy load InterHuman and Relationship VQVAE tokenizers."""
        if self.interhuman_motion_tokenizer is not None:
            return  # Already loaded
        
        import sys
        import os
        # Add motion_representation to path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        motion_rep_path = os.path.join(project_root, 'Salsa-Agent', 'motion_representation')
        if motion_rep_path not in sys.path:
            sys.path.insert(0, motion_rep_path)
        
        from motion_representation.models.motion_model import MotionModel
        
        device = self.args.device if hasattr(self.args, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load InterHuman Motion VQVAE
        interhuman_ckpt_path = os.path.join(motion_rep_path, 'checkpoints_VQVAE_GRU_InterHuman', 'best_checkpoint.pth')
        if not os.path.exists(interhuman_ckpt_path):
            # Try latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(motion_rep_path, 'checkpoints_VQVAE_GRU_InterHuman', 'checkpoint_epoch_*.pth'))
            if checkpoints:
                interhuman_ckpt_path = max(checkpoints, key=os.path.getctime)
            else:
                raise FileNotFoundError(f"InterHuman VQVAE checkpoint not found in {os.path.join(motion_rep_path, 'checkpoints_VQVAE_GRU_InterHuman')}")
        
        print(f"Loading InterHuman Motion VQVAE from {interhuman_ckpt_path}")
        interhuman_ckpt = torch.load(interhuman_ckpt_path, map_location='cpu')
        interhuman_config = interhuman_ckpt['config']
        
        self.interhuman_motion_tokenizer = MotionModel(
            input_dim=interhuman_config['input_dim'],
            hidden_dim=interhuman_config['hidden_dim'],
            num_layers=interhuman_config['num_layers'],
            latent_dim=interhuman_config['latent_dim'],
            seq_len=interhuman_config['seq_len'],
            dropout=interhuman_config['dropout'],
            encoder_type=interhuman_config.get('encoder_type', 'gru'),
            decoder_type=interhuman_config.get('decoder_type', 'gru'),
            use_vqvae=True,
            nb_code=interhuman_config.get('nb_code', 512),
            quantizer=interhuman_config.get('quantizer', 'ema_reset'),
            vq_mu=interhuman_config.get('vq_mu', 0.95),
        ).to(device)
        self.interhuman_motion_tokenizer.load_state_dict(interhuman_ckpt['model_state_dict'])
        self.interhuman_motion_tokenizer.eval()
        
        # Load Relationship VQVAE
        relationship_ckpt_path = os.path.join(motion_rep_path, 'checkpoints_VQVAE_GRU_Relationship', 'best_checkpoint.pth')
        if not os.path.exists(relationship_ckpt_path):
            import glob
            checkpoints = glob.glob(os.path.join(motion_rep_path, 'checkpoints_VQVAE_GRU_Relationship', 'checkpoint_epoch_*.pth'))
            if checkpoints:
                relationship_ckpt_path = max(checkpoints, key=os.path.getctime)
            else:
                raise FileNotFoundError(f"Relationship VQVAE checkpoint not found in {os.path.join(motion_rep_path, 'checkpoints_VQVAE_GRU_Relationship')}")
        
        print(f"Loading Relationship VQVAE from {relationship_ckpt_path}")
        relationship_ckpt = torch.load(relationship_ckpt_path, map_location='cpu')
        relationship_config = relationship_ckpt['config']
        
        self.relationship_tokenizer = MotionModel(
            input_dim=relationship_config['input_dim'],
            hidden_dim=relationship_config['hidden_dim'],
            num_layers=relationship_config['num_layers'],
            latent_dim=relationship_config['latent_dim'],
            seq_len=relationship_config['seq_len'],
            dropout=relationship_config['dropout'],
            encoder_type=relationship_config.get('encoder_type', 'gru'),
            decoder_type=relationship_config.get('decoder_type', 'gru'),
            use_vqvae=True,
            nb_code=relationship_config.get('nb_code', 512),
            quantizer=relationship_config.get('quantizer', 'ema_reset'),
            vq_mu=relationship_config.get('vq_mu', 0.95),
        ).to(device)
        self.relationship_tokenizer.load_state_dict(relationship_ckpt['model_state_dict'])
        self.relationship_tokenizer.eval()
        
        # Load normalization statistics
        # Try multiple possible cache locations (matching visualization app approach)
        possible_cache_dirs = [
            os.path.join(project_root, 'Salsa-Agent', 'dataset_processed_New', 'lmdb_Salsa_pair', 'lmdb_train_interhuman_20frames_cache'),  # Actual location (no /dd/)
            os.path.join(project_root, 'Salsa-Agent', 'dataset_processed_New', 'lmdb_Salsa_pair', 'dd', 'lmdb_train_interhuman_20frames_cache'),  # Alternative location
        ]
        
        interhuman_stats_path = None
        relationship_stats_path = None
        
        for cache_dir in possible_cache_dirs:
            if interhuman_stats_path is None:
                test_path = os.path.join(cache_dir, 'normalization_stats_interhuman.pkl')
                if os.path.exists(test_path):
                    interhuman_stats_path = test_path
                    print(f"Found InterHuman normalization stats at: {interhuman_stats_path}")
            
            if relationship_stats_path is None:
                test_path = os.path.join(cache_dir, 'normalization_stats_relationship.pkl')
                if os.path.exists(test_path):
                    relationship_stats_path = test_path
                    print(f"Found Relationship normalization stats at: {relationship_stats_path}")
            
            if interhuman_stats_path and relationship_stats_path:
                break
        
        if interhuman_stats_path and os.path.exists(interhuman_stats_path):
            with open(interhuman_stats_path, 'rb') as f:
                self.interhuman_normalization_stats = pickle.load(f)
            print(f"Loaded InterHuman normalization stats successfully")
        else:
            print(f"ERROR: InterHuman normalization stats not found!")
            print(f"  Searched in: {possible_cache_dirs}")
            self.interhuman_normalization_stats = None
        
        if relationship_stats_path and os.path.exists(relationship_stats_path):
            with open(relationship_stats_path, 'rb') as f:
                self.relationship_normalization_stats = pickle.load(f)
            print(f"Loaded Relationship normalization stats successfully")
        else:
            print(f"ERROR: Relationship normalization stats not found!")
            print(f"  Searched in: {possible_cache_dirs}")
            self.relationship_normalization_stats = None
        
        print("InterHuman tokenizers loaded successfully!")

    def _convert_to_interhuman_and_tokenize_sequence(self, full_keypoints3d_L, full_rotmat_L,
                                                      full_keypoints3d_F, full_rotmat_F):
        """
        Convert full sequence to InterHuman representation and tokenize window by window.
        Splits sequence into 20-frame windows and tokenizes each separately.
        
        Args:
            full_keypoints3d_L: (T, 22, 3) numpy array - Leader keypoints for full sequence
            full_rotmat_L: (T, 498) numpy array - Leader rotation matrices for full sequence
            full_keypoints3d_F: (T, 22, 3) numpy array - Follower keypoints for full sequence
            full_rotmat_F: (T, 498) numpy array - Follower rotation matrices for full sequence
        
        Returns:
            dict with keys:
                - leader_motion_ih: (T-1, 262) numpy array - Full InterHuman canonicalized motion (concatenated)
                - follower_motion_ih: (T-1, 262) numpy array
                - relationship_features: (T-1, 4) numpy array - [w, z, x, z]
                - leader_tokens: (num_windows,) numpy array - VQ token indices, one per 20-frame window
                - follower_tokens: (num_windows,) numpy array
                - relationship_tokens: (num_windows,) numpy array
                - root_quat_init_L: (4,) numpy array - Frame 0 root quaternion (from first window)
                - root_pos_init_L: (3,) numpy array - Frame 0 root position (from first window)
                - root_quat_init_F: (4,) numpy array
                - root_pos_init_F: (3,) numpy array
        """
        window_size = 20  # Tokenizer expects 20-frame windows
        T = len(full_keypoints3d_L)
        
        # Calculate number of windows
        num_windows = (T + window_size - 1) // window_size  # Ceiling division
        
        # Lists to store results from each window
        leader_motions_ih = []
        follower_motions_ih = []
        relationship_features_list = []
        leader_tokens_list = []
        follower_tokens_list = []
        relationship_tokens_list = []
        
        # Store root transforms from first window only
        root_quat_init_L_frame0 = None
        root_pos_init_L_frame0 = None
        root_quat_init_F_frame0 = None
        root_pos_init_F_frame0 = None
        
        # Process each window
        for w in range(num_windows):
            start_idx = w * window_size
            end_idx = min(start_idx + window_size, T)
            
            # Extract window
            window_keypoints3d_L = full_keypoints3d_L[start_idx:end_idx]
            window_rotmat_L = full_rotmat_L[start_idx:end_idx]
            window_keypoints3d_F = full_keypoints3d_F[start_idx:end_idx]
            window_rotmat_F = full_rotmat_F[start_idx:end_idx]
            
            # Pad last window if needed (shouldn't happen for 100 frames, but handle it)
            if len(window_keypoints3d_L) < window_size:
                # Pad with last frame
                pad_frames = window_size - len(window_keypoints3d_L)
                window_keypoints3d_L = np.concatenate([
                    window_keypoints3d_L,
                    np.tile(window_keypoints3d_L[-1:], (pad_frames, 1, 1))
                ], axis=0)
                window_rotmat_L = np.concatenate([
                    window_rotmat_L,
                    np.tile(window_rotmat_L[-1:], (pad_frames, 1))
                ], axis=0)
                window_keypoints3d_F = np.concatenate([
                    window_keypoints3d_F,
                    np.tile(window_keypoints3d_F[-1:], (pad_frames, 1, 1))
                ], axis=0)
                window_rotmat_F = np.concatenate([
                    window_rotmat_F,
                    np.tile(window_rotmat_F[-1:], (pad_frames, 1))
                ], axis=0)
            
            # Convert and tokenize this window
            window_data = self._convert_to_interhuman_and_tokenize(
                window_keypoints3d_L, window_rotmat_L,
                window_keypoints3d_F, window_rotmat_F
            )
            
            # Store results
            leader_motions_ih.append(window_data['leader_motion_ih'])  # (19, 262)
            follower_motions_ih.append(window_data['follower_motion_ih'])  # (19, 262)
            relationship_features_list.append(window_data['relationship_features'])  # (19, 4)
            leader_tokens_list.append(window_data['leader_tokens'][0])  # scalar
            follower_tokens_list.append(window_data['follower_tokens'][0])  # scalar
            relationship_tokens_list.append(window_data['relationship_tokens'][0])  # scalar
            
            # Store root transforms from first window
            if w == 0:
                root_quat_init_L_frame0 = window_data['root_quat_init_L']
                root_pos_init_L_frame0 = window_data['root_pos_init_L']
                root_quat_init_F_frame0 = window_data['root_quat_init_F']
                root_pos_init_F_frame0 = window_data['root_pos_init_F']
        
        # Concatenate all windows
        leader_motion_ih_full = np.concatenate(leader_motions_ih, axis=0)  # (num_windows*19, 262)
        follower_motion_ih_full = np.concatenate(follower_motions_ih, axis=0)  # (num_windows*19, 262)
        relationship_features_full = np.concatenate(relationship_features_list, axis=0)  # (num_windows*19, 4)
        
        # Convert token lists to arrays
        leader_tokens_array = np.array(leader_tokens_list, dtype=np.int64)  # (num_windows,)
        follower_tokens_array = np.array(follower_tokens_list, dtype=np.int64)  # (num_windows,)
        relationship_tokens_array = np.array(relationship_tokens_list, dtype=np.int64)  # (num_windows,)
        
        return {
            'leader_motion_ih': leader_motion_ih_full.astype(np.float32),  # (num_windows*19, 262)
            'follower_motion_ih': follower_motion_ih_full.astype(np.float32),  # (num_windows*19, 262)
            'relationship_features': relationship_features_full.astype(np.float32),  # (num_windows*19, 4)
            'leader_tokens': leader_tokens_array,  # (num_windows,)
            'follower_tokens': follower_tokens_array,  # (num_windows,)
            'relationship_tokens': relationship_tokens_array,  # (num_windows,)
            'root_quat_init_L': root_quat_init_L_frame0.astype(np.float32),  # (4,)
            'root_pos_init_L': root_pos_init_L_frame0.astype(np.float32),  # (3,)
            'root_quat_init_F': root_quat_init_F_frame0.astype(np.float32),  # (4,)
            'root_pos_init_F': root_pos_init_F_frame0.astype(np.float32),  # (3,)
        }
    
    def _convert_to_interhuman_and_tokenize(self, window_keypoints3d_L, window_rotmat_L, 
                                             window_keypoints3d_F, window_rotmat_F):
        """
        Convert 20-frame window to InterHuman representation and tokenize.
        Follows exact process from motion_representation/data/motion_dataset.py
        
        Args:
            window_keypoints3d_L: (20, 22, 3) numpy array - Leader keypoints
            window_rotmat_L: (20, 498) numpy array - Leader rotation matrices
            window_keypoints3d_F: (20, 22, 3) numpy array - Follower keypoints
            window_rotmat_F: (20, 498) numpy array - Follower rotation matrices
        
        Returns:
            dict with keys:
                - leader_motion_ih: (19, 262) numpy array - InterHuman canonicalized motion
                - follower_motion_ih: (19, 262) numpy array
                - relationship_features: (19, 4) numpy array - [w, z, x, z]
                - leader_tokens: (1,) numpy array - VQ token indices
                - follower_tokens: (1,) numpy array
                - relationship_tokens: (1,) numpy array
                - root_quat_init_L: (4,) numpy array - Frame 0 root quaternion
                - root_pos_init_L: (3,) numpy array - Frame 0 root position
                - root_quat_init_F: (4,) numpy array
                - root_pos_init_F: (3,) numpy array
        """
        # Lazy load tokenizers if not already loaded
        if self.interhuman_motion_tokenizer is None:
            self._load_interhuman_tokenizers()
        
        # Import conversion functions from motion_representation
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        motion_rep_path = os.path.join(project_root, 'Salsa-Agent', 'motion_representation')
        if motion_rep_path not in sys.path:
            sys.path.insert(0, motion_rep_path)
        
        from motion_representation.utils.relationship_features import (
            salsa_to_interhuman, extract_interhuman_relationship_features
        )
        
        device = self.args.device if hasattr(self.args, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Ensure inputs are numpy arrays
        if isinstance(window_keypoints3d_L, torch.Tensor):
            window_keypoints3d_L = window_keypoints3d_L.cpu().numpy()
        if isinstance(window_rotmat_L, torch.Tensor):
            window_rotmat_L = window_rotmat_L.cpu().numpy()
        if isinstance(window_keypoints3d_F, torch.Tensor):
            window_keypoints3d_F = window_keypoints3d_F.cpu().numpy()
        if isinstance(window_rotmat_F, torch.Tensor):
            window_rotmat_F = window_rotmat_F.cpu().numpy()
        
        # Step 1: Convert to InterHuman representation (following motion_dataset.py lines 491-497)
        # Apply +90° rotation to reverse the -90° preprocessing rotation
        motion_L_ih, root_quat_init_L, root_pos_init_L = salsa_to_interhuman(
            window_keypoints3d_L, window_rotmat_L, rotation_deg=90
        )  # (19, 262), (19, 4), (19, 3)
        
        motion_F_ih, root_quat_init_F, root_pos_init_F = salsa_to_interhuman(
            window_keypoints3d_F, window_rotmat_F, rotation_deg=90
        )  # (19, 262), (19, 4), (19, 3)
        
        # Step 2: Extract frame 0 root quaternions and positions (for relationship computation)
        root_quat_init_L_frame0 = root_quat_init_L[0] if root_quat_init_L.ndim > 1 else root_quat_init_L  # (4,)
        root_pos_init_L_frame0 = root_pos_init_L[0] if root_pos_init_L.ndim > 1 else root_pos_init_L  # (3,)
        root_quat_init_F_frame0 = root_quat_init_F[0] if root_quat_init_F.ndim > 1 else root_quat_init_F  # (4,)
        root_pos_init_F_frame0 = root_pos_init_F[0] if root_pos_init_F.ndim > 1 else root_pos_init_F  # (3,)
        
        # Step 3: Extract relationship features (following motion_dataset.py lines 513-522)
        # CRITICAL: Pass copies to avoid in-place modification
        relationship_features = extract_interhuman_relationship_features(
            motion_L_ih.copy(), motion_F_ih.copy(),  # Make copies
            root_quat_init_L_frame0, root_pos_init_L_frame0,
            root_quat_init_F_frame0, root_pos_init_F_frame0,
            root_quat_init_L_all=root_quat_init_L,  # (19, 4)
            root_pos_init_L_all=root_pos_init_L,      # (19, 3)
            root_quat_init_F_all=root_quat_init_F,    # (19, 4)
            root_pos_init_F_all=root_pos_init_F,      # (19, 3)
            return_aligned_follower=False
        )  # (19, 4) - [w, z, x, z]
        
        # Step 4: Normalize motions and relationship features
        # CRITICAL: Normalization must be applied before tokenization!
        if self.interhuman_normalization_stats is not None:
            mean_ih = torch.from_numpy(self.interhuman_normalization_stats['mean']).float()
            std_ih = torch.from_numpy(self.interhuman_normalization_stats['std']).float()
            epsilon = 1e-8
            motion_L_ih_norm = (torch.from_numpy(motion_L_ih).float() - mean_ih) / (std_ih + epsilon)
            motion_F_ih_norm = (torch.from_numpy(motion_F_ih).float() - mean_ih) / (std_ih + epsilon)
            
            # Debug: Check normalization is working (first sample only)
            if not hasattr(self, '_normalization_debug_printed'):
                print(f"\n[DEBUG] Normalization applied to InterHuman motion:")
                print(f"  Original motion_L_ih range: [{motion_L_ih.min():.4f}, {motion_L_ih.max():.4f}]")
                print(f"  Normalized motion_L_ih_norm range: [{motion_L_ih_norm.min():.4f}, {motion_L_ih_norm.max():.4f}]")
                print(f"  Mean shape: {mean_ih.shape}, Std shape: {std_ih.shape}")
                print(f"  Mean sample (first 5): {mean_ih[:5].numpy()}")
                print(f"  Std sample (first 5): {std_ih[:5].numpy()}")
                self._normalization_debug_printed = True
        else:
            print(f"\n[ERROR] InterHuman normalization stats are None! Motion will NOT be normalized before tokenization!")
            print(f"  This will cause all samples to produce the same tokens!")
            motion_L_ih_norm = torch.from_numpy(motion_L_ih).float()
            motion_F_ih_norm = torch.from_numpy(motion_F_ih).float()
        
        if self.relationship_normalization_stats is not None:
            mean_rel = torch.from_numpy(self.relationship_normalization_stats['mean']).float()
            std_rel = torch.from_numpy(self.relationship_normalization_stats['std']).float()
            epsilon = 1e-8
            relationship_features_norm = (torch.from_numpy(relationship_features).float() - mean_rel) / (std_rel + epsilon)
        else:
            relationship_features_norm = torch.from_numpy(relationship_features).float()
        
        # Step 5: Tokenize using VQVAEs
        # InterHuman motion tokenization: (19, 262) -> (1,) token
        with torch.no_grad():
            # Leader motion
            motion_L_tensor = motion_L_ih_norm.unsqueeze(0).to(device)  # (1, 19, 262)
            
            # Debug: Check input to tokenizer (first sample only)
            if not hasattr(self, '_tokenizer_input_debug_printed'):
                print(f"\n[DEBUG] Input to motion tokenizer:")
                print(f"  motion_L_tensor shape: {motion_L_tensor.shape}")
                print(f"  motion_L_tensor range: [{motion_L_tensor.min():.4f}, {motion_L_tensor.max():.4f}]")
                print(f"  motion_L_tensor mean: {motion_L_tensor.mean():.4f}, std: {motion_L_tensor.std():.4f}")
                print(f"  motion_L_tensor sample (first frame, first 10 dims): {motion_L_tensor[0, 0, :10].cpu().numpy()}")
                self._tokenizer_input_debug_printed = True
            
            _, leader_code_idx = self.interhuman_motion_tokenizer.inference_encode(motion_L_tensor)
            # Handle different shapes: (1,) or (1, 1) -> flatten to (1,)
            if leader_code_idx.dim() > 1:
                leader_code_idx = leader_code_idx.flatten()
            leader_tokens = leader_code_idx.cpu().numpy()  # (1,)
            
            # Follower motion
            motion_F_tensor = motion_F_ih_norm.unsqueeze(0).to(device)  # (1, 19, 262)
            _, follower_code_idx = self.interhuman_motion_tokenizer.inference_encode(motion_F_tensor)
            if follower_code_idx.dim() > 1:
                follower_code_idx = follower_code_idx.flatten()
            follower_tokens = follower_code_idx.cpu().numpy()  # (1,)
            
            # Relationship tokenization: (19, 4) -> (1,) token
            relationship_tensor = relationship_features_norm.unsqueeze(0).to(device)  # (1, 19, 4)
            _, relationship_code_idx = self.relationship_tokenizer.inference_encode(relationship_tensor)
            if relationship_code_idx.dim() > 1:
                relationship_code_idx = relationship_code_idx.flatten()
            relationship_tokens = relationship_code_idx.cpu().numpy()  # (1,)
        
        # Return all data
        return {
            'leader_motion_ih': motion_L_ih.astype(np.float32),  # (19, 262)
            'follower_motion_ih': motion_F_ih.astype(np.float32),  # (19, 262)
            'relationship_features': relationship_features.astype(np.float32),  # (19, 4)
            'leader_tokens': leader_tokens.astype(np.int64),  # (1,)
            'follower_tokens': follower_tokens.astype(np.int64),  # (1,)
            'relationship_tokens': relationship_tokens.astype(np.int64),  # (1,)
            'root_quat_init_L': root_quat_init_L_frame0.astype(np.float32),  # (4,)
            'root_pos_init_L': root_pos_init_L_frame0.astype(np.float32),  # (3,)
            'root_quat_init_F': root_quat_init_F_frame0.astype(np.float32),  # (4,)
            'root_pos_init_F': root_pos_init_F_frame0.astype(np.float32),  # (3,)
        }

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

        # InterHuman sample list
        sample_interhuman_data_list = []  # List of dicts with InterHuman data and tokens

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

            # Convert to InterHuman representation and tokenize
            # Use sequence-level function that handles full sequence and tokenizes window by window
            try:
                interhuman_data = self._convert_to_interhuman_and_tokenize_sequence(
                    sample_skeletons3d_L, sample_rotmat_L,
                    sample_skeletons3d_F, sample_rotmat_F
                )
                sample_interhuman_data_list.append(interhuman_data)
            except Exception as e:
                print(f"Warning: Failed to convert to InterHuman for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                # Add None as placeholder to keep list aligned
                sample_interhuman_data_list.append(None)

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
            # Save raw audio as numpy array (convert from torch tensor if needed)
            sample_audio_raw = sample_audio.cpu().numpy() if isinstance(sample_audio, torch.Tensor) else sample_audio
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


            # Extract dance moves labels from annotation files

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            # Get annotations for this window
            annotations = self._get_annotations_for_window(vid, subdivision_start_time, subdivision_end_time)
            motion_info['dance_moves'] = annotations['moves']
            motion_info['errors'] = annotations['errors']
            motion_info['styling_leader'] = annotations['styling_leader']
            motion_info['styling_follower'] = annotations['styling_follower']


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
            sample_audio_raw_list.append(sample_audio_raw)


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
                         audio_tokens, audio_raw, aux, interhuman_data in \
                            zip(sample_skeleton3d_list_L, sample_skeleton3d_list_F,
                                sample_rotmat_list_L, sample_rotmat_list_F,
                                sample_ms_description_list_L, sample_ms_description_list_F,
                                sample_vqtokens_list_L, sample_vqtokens_list_F,
                                sample_audio_tokens_list, sample_audio_raw_list, aux_info,
                                sample_interhuman_data_list):

                        poses_keypoints3d_L = np.asarray(poses_keypoints3d_L)
                        poses_rotmat_L = np.asarray(poses_rotmat_L)
                        poses_vqtokens_L = np.asarray(poses_vq_tokens_L)

                        poses_keypoints3d_F = np.asarray(poses_keypoints3d_F)
                        poses_rotmat_F = np.asarray(poses_rotmat_F)
                        poses_vqtokens_F = np.asarray(poses_vq_tokens_F)

                        audio_tokens = np.asarray(audio_tokens)
                        audio_raw = np.asarray(audio_raw)
                        # GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        # Add InterHuman data if available
                        if interhuman_data is not None:
                            v = [poses_keypoints3d_L, poses_rotmat_L, ms_description_L, poses_vqtokens_L,
                                 poses_keypoints3d_F, poses_rotmat_F, ms_description_F, poses_vqtokens_F,
                                 audio_tokens, audio_raw, aux, interhuman_data]
                        else:
                            # Fallback: no InterHuman data (shouldn't happen, but handle gracefully)
                            v = [poses_keypoints3d_L, poses_rotmat_L, ms_description_L, poses_vqtokens_L,
                                 poses_keypoints3d_F, poses_rotmat_F, ms_description_F, poses_vqtokens_F,
                                 audio_tokens, audio_raw, aux, None]
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
                            audio_tokens, audio_raw, aux, interhuman_data in \
                            zip(sample_skeleton3d_list_L, sample_skeleton3d_list_F,
                                sample_rotmat_list_L, sample_rotmat_list_F,
                                sample_HML3D_vec_list_L, sample_HML3D_vec_list_F,
                                sample_ms_description_list_L, sample_ms_description_list_F,
                                sample_vqtokens_list_L, sample_vqtokens_list_F,
                                sample_audio_tokens_list, sample_audio_raw_list, aux_info,
                                sample_interhuman_data_list):
                        poses_keypoints3d_L = np.asarray(poses_keypoints3d_L)
                        poses_rotmat_L = np.asarray(poses_rotmat_L)
                        HML3D_vec_L = np.asarray(HML3D_vec_L)
                        poses_vqtokens_L = np.asarray(poses_vq_tokens_L)

                        poses_keypoints3d_F = np.asarray(poses_keypoints3d_F)
                        poses_rotmat_F = np.asarray(poses_rotmat_F)
                        HML3D_vec_F = np.asarray(HML3D_vec_F)
                        poses_vqtokens_F = np.asarray(poses_vq_tokens_F)

                        audio_tokens = np.asarray(audio_tokens)
                        audio_raw = np.asarray(audio_raw)
                        # GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        # Add InterHuman data if available
                        if interhuman_data is not None:
                            v = [poses_keypoints3d_L, poses_rotmat_L, HML3D_vec_L, ms_description_L, poses_vqtokens_L,
                                 poses_keypoints3d_F, poses_rotmat_F, HML3D_vec_F, ms_description_F, poses_vqtokens_F,
                                 audio_tokens, audio_raw, aux, interhuman_data]
                        else:
                            # Fallback: no InterHuman data
                            v = [poses_keypoints3d_L, poses_rotmat_L, HML3D_vec_L, ms_description_L, poses_vqtokens_L,
                                 poses_keypoints3d_F, poses_rotmat_F, HML3D_vec_F, ms_description_F, poses_vqtokens_F,
                                 audio_tokens, audio_raw, aux, None]
                        # v = [words, poses, audio_raws, audio_mels, aux, sentence_leve_latents, GPT_3_Embedding]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

        print()

# Todo -----------------------------------------------------------------
# Pair → Proficiency per Dataset README (Dataset/compas3d/README.md table).
PAIR2LEVEL = {
    "pair1": "beginner",
    "pair2": "intermediate",
    "pair3": "beginner",
    "pair4": "intermediate",
    "pair5": "professional",
    "pair6": "intermediate",
    "pair7": "professional",
    "pair8": "beginner",
    "pair9": "professional",
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
        cache_suffix: str = "_cache",
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
        # Use is_MDM flag explicitly: if True, use _MDM cache; if False, use regular cache
        is_MDM = getattr(self.args, 'is_MDM', False)
        preloaded_dir = lmdb_dir + cache_suffix
        if is_MDM:
            preloaded_dir += '_MDM'
        # Use absolute path for cache existence check
        preloaded_dir_abs = os.path.abspath(preloaded_dir)
        print(f"Using cache path (is_MDM={is_MDM}): {preloaded_dir_abs}")
        if not os.path.exists(preloaded_dir_abs): # TODO
            data_sampler = DataPreprocessor(
                args,
                lmdb_dir,
                preloaded_dir_abs,
                n_poses,
                subdivision_stride,
                self.skeleton_resampling_fps
            )
            data_sampler.run()
            preloaded_dir = preloaded_dir_abs  # Use absolute path for opening LMDB
        else:
            print("Found pre-loaded samples from {}".format(preloaded_dir_abs))
            preloaded_dir = preloaded_dir_abs  # Use absolute path for opening LMDB

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
        # Ensure we return at least 0 (handle empty cache gracefully)
        return max(0, self.n_samples - 1)  # last item is None

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
            # Handle both old cache format (without InterHuman) and new format (with InterHuman)
            interhuman_data = None
            if not self.args.is_MDM:
                if len(sample) == 12:
                    # New format with InterHuman data
                    poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                     poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                     audio_tokens, audio_raw, aux_info, interhuman_data = sample
                else:
                    # Old format without InterHuman data
                    poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                     poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                     audio_tokens, audio_raw, aux_info = sample
            if self.args.is_MDM:
                if len(sample) == 14:
                    # New format with InterHuman data
                    poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                        poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                        audio_tokens, audio_raw, aux_info, interhuman_data = sample
                else:
                    # Old format without InterHuman data
                    poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                        poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                        audio_tokens, audio_raw, aux_info = sample
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

        # Convert InterHuman data to tensors if available
        if interhuman_data is not None:
            interhuman_data_tensors = {
                'leader_motion_ih': torch.from_numpy(interhuman_data['leader_motion_ih']).float(),
                'follower_motion_ih': torch.from_numpy(interhuman_data['follower_motion_ih']).float(),
                'relationship_features': torch.from_numpy(interhuman_data['relationship_features']).float(),
                'leader_tokens': torch.from_numpy(interhuman_data['leader_tokens']).long(),
                'follower_tokens': torch.from_numpy(interhuman_data['follower_tokens']).long(),
                'relationship_tokens': torch.from_numpy(interhuman_data['relationship_tokens']).long(),
                'root_quat_init_L': torch.from_numpy(interhuman_data['root_quat_init_L']).float(),
                'root_pos_init_L': torch.from_numpy(interhuman_data['root_pos_init_L']).float(),
                'root_quat_init_F': torch.from_numpy(interhuman_data['root_quat_init_F']).float(),
                'root_pos_init_F': torch.from_numpy(interhuman_data['root_pos_init_F']).float(),
            }
        else:
            interhuman_data_tensors = None

        # Return format: (level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_info, interhuman_data_tensors)
        # Always return motion script strings (ms_desc_L, ms_des_F) for prompt building, regardless of is_MDM
        # HML3D vectors are stored in cache but not needed for LLM training (prompts use motion script + VQ tokens)
        ms_desc_L_str = '-->'.join(ms_desc_L) if isinstance(ms_desc_L, list) else ms_desc_L
        ms_des_F_str = '-->'.join(ms_des_F) if isinstance(ms_des_F, list) else ms_des_F
        return level, ms_desc_L_str, ms_des_F_str, vq_tokens_L, vq_tokens_F, audio_tokens, aux_info, interhuman_data_tensors
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