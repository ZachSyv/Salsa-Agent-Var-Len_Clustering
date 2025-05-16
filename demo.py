import random

import tqdm
from openai import AzureOpenAI
from models.motion_agent import MotionAgent
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import torch
import argparse as demo_argparse
import os
import sys

# If working on windows use the following:
# Path to the FFmpeg bin directory
ffmpeg_path = r'S:\Payam\LAMMA\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin'
import matplotlib as mpl
import shutil
from models.training_utils import *
ffmpeg_path = shutil.which("ffmpeg")
mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
# Check if the path is already in the PATH variable
if ffmpeg_path not in os.environ['PATH']:
    # Add the path to the PATH environment variable
    os.environ['PATH'] += os.pathsep + ffmpeg_path

from utils.salsa_utils.salsa_dataloader import Salsa_Dataset
from utils.salsa_utils.salsa_dataloader import SALSA_CAPTIONS
import pickle
os.chdir('Motion-Agent-Salsa') # to refine the data loader felan.
from utils.salsa_utils.libs.MotionScript.ms_utils_visu import render_HQ_Salsa, render_HQ_Salsa_pair




def motionllm_evaluation_qualitative_pairs(task):


    args = get_args_parser()
    args.save_dir = "./demo/"
    args.device = 'cuda:0'

    if task == 'baseline':
        args.is_baseline = True


    model = MotionLLM(args)

    # Baseline
    if task == 'baseline':

        model.load_model('ckpt/motionllm.pth')
        current_batch_task = 'caption_to_motion'
    elif task == 'solo':
        model.load_model('ckpt/captioning_to_motion_V3/salsa_agent.pth')
        current_batch_task = 'caption_to_motion'
    elif task == "leader_to_follower":
        model.load_model('ckpt/leader_to_follower_V3/salsa_agent.pth')
        current_batch_task = 'leader_to_follower'
    elif task == 'follower_to_leader':
        model.load_model('ckpt/follower_to_leader_V3/salsa_agent.pth')
        current_batch_task = 'follower_to_leader'




    model.llm.eval()
    model.llm.cuda()

    motion_tokens_to_generate = []
    follower_motion_tokens = []
    leader_motion_tokens = []

    H3D_GT_Leader = []
    H3D_GT_Follower = []

    # Pair < Number > _take < Song > _ < Take > _follower_subject.bvh
    test_set = [
        {"Pair": "Pair1", "Level": "Beginner", "SongTake": "take1_1"},
        {"Pair": "Pair2", "Level": "Intermediate", "SongTake": "take1_2"},
        {"Pair": "Pair3", "Level": "Beginner", "SongTake": "take2_1"},
        {"Pair": "Pair4", "Level": "Intermediate", "SongTake": "take2_2"},
        {"Pair": "Pair5", "Level": "Professional", "SongTake": "take3_1"},
        {"Pair": "Pair6", "Level": "Intermediate", "SongTake": "take3_2"},
        {"Pair": "Pair7", "Level": "Professional", "SongTake": "take4_1"},
        {"Pair": "Pair8", "Level": "Beginner", "SongTake": "take4_2"},
        {"Pair": "Pair9", "Level": "Professional", "SongTake": "take1_1"},
    ]

    # now iterate
    for row in test_set:
        Pair = row["Pair"]
        Style = row["Level"].lower()  # e.g. "Beginner" → "beginner"
        my_take = row["SongTake"]
        # Style, Pair = 'beginner', "Pair1"
        # Style, Pair = 'professional', "Pair3"
        # my_take = 'tale1_1'
        s, e = 0, 10
        items = load_data_pair(style=Style, pair=Pair, take=my_take, start_sec=s, end_sec=e)

        args.is_MDM = True # at the inference to get GT from the preprocessed data

        motion_tokens_to_generate = []
        follower_motion_tokens = []
        leader_motion_tokens = []

        H3D_GT_Leader = []
        H3D_GT_Follower = []

        for item in items:
            if not args.is_MDM:
                level, ms_desc_L, ms_des_F, \
                    vq_tokens_L, vq_tokens_F, audio_tokens, aux_info = item

            else:
                level, HML3D_L, vq_tokens_L, HML3D_F, vq_tokens_F, audio_tokens, aux_info = item
                ms_desc_L = ms_des_F = 'a-->b'

            # captions_list = []
            level = Style

            full_prompt, input_ids = build_random_training_instance_salsa_prompt(
                tokenizer=model.tokenizer,
                leader_motion_script_segments=ms_desc_L.split('-->'),
                follower_motion_script_segments=ms_des_F.split('-->'),
                leader_motion_tokens=vq_tokens_L,
                follower_motion_tokens=vq_tokens_F,
                audio_tokens=audio_tokens,
                proficiency_level=level,
                allowed_tasks=[current_batch_task],
                snippet_prob=0.5,
                min_snippet_steps=1,
                max_snippet_steps=4,
                inference=True,
                is_baseline=True
            )



            motion_tokens = model.generate_Payam(full_prompt, input_ids)
            motion_tokens_to_generate.append(motion_tokens)

            leader_motion_tokens.append(vq_tokens_L)
            follower_motion_tokens.append(vq_tokens_F)

            # store GT:
            H3D_GT_Leader.append(HML3D_L)
            H3D_GT_Follower.append(HML3D_F)

            # the_other_motion_tokens.append(vq_tokens_F if current_batch_task=='follower_to_leader' else vq_tokens_F)

        print("Inference completed.\nExporting results...")

        motion_tokens = torch.cat(motion_tokens_to_generate)
        motion = model.net.forward_decoder(motion_tokens)
        motion = model.denormalize(motion.detach().cpu().numpy())
        positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
        # print(motion.shape)

        # Leader:
        leader_motion_tokens = torch.cat(leader_motion_tokens)
        leader_motion = model.net.forward_decoder(leader_motion_tokens)
        leader_motion = model.denormalize(leader_motion.detach().cpu().numpy())
        leader_positions = recover_from_ric(torch.from_numpy(leader_motion).float().cuda(), 22)
        # print(motion.shape)
        # Leader:
        follower_motion_tokens = torch.cat(follower_motion_tokens)
        follower_motion = model.net.forward_decoder(follower_motion_tokens)
        follower_motion = model.denormalize(follower_motion.detach().cpu().numpy())
        follower_positions = recover_from_ric(torch.from_numpy(follower_motion).float().cuda(), 22)

        # Original motion GT
        H3D_GT_Leader = torch.cat(H3D_GT_Leader)
        H3D_GT_Follower = torch.cat(H3D_GT_Follower)

        H3D_GT_Leader_Positions = recover_from_ric(H3D_GT_Leader.unsqueeze(0).float().cuda(), 22)
        H3D_GT_Follower_positions = recover_from_ric(H3D_GT_Leader.unsqueeze(0).float().cuda(), 22)


        iterate = Pair + '_' + my_take
        sav_path = f"./demo/eval4Bermet/{current_batch_task}/{level}/{Pair}_{my_take}"

        os.makedirs(sav_path, exist_ok=True)
        plot_3d_motion(os.path.join(sav_path,
                        f"motionllm_{level}_{iterate}_predicted.mp4"),
                        t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                        title=(f"Generated {'leader' if current_batch_task=='follower_to_leader' else 'follower'}: {Style} {Pair}"),
                        fps=20, radius=4)

        my_dict = {'caption': "infer",
                   'HML3D_vec': motion.squeeze(),
                   'positions': positions.squeeze().detach().cpu().numpy(),
                }
        with open(f"{sav_path}/motionllm_{level}_{iterate}_predicted.pk", "wb") as f:
            pickle.dump(my_dict, f)




        tx = 'input(recon)' if current_batch_task=='leader_to_follower' else 'GT(recon)'
        plot_3d_motion(os.path.join(sav_path,
                                    f"motionllm_{level}_{iterate}_leader_{tx}.mp4"),
                       t2m_kinematic_chain, leader_positions.squeeze().detach().cpu().numpy(),
                       title=(f"Leader {tx}: {Style} {Pair}"),
                       fps=20, radius=4)
        my_dict = {'caption': "infer",
                   'HML3D_vec': leader_motion.squeeze(),
                   'positions': leader_positions.squeeze().detach().cpu().numpy(),
                    }
        with open(f"{sav_path}/motionllm_{level}_{iterate}_leader_{tx}.pk", "wb") as f:
            pickle.dump(my_dict, f)

        tx = 'input(recon)' if current_batch_task=='follower_to_leader' else 'GT(recon)'
        plot_3d_motion(os.path.join(sav_path,
                                    f"motionllm_{level}_{iterate}_follower_{tx}.mp4"),
                       t2m_kinematic_chain, follower_positions.squeeze().detach().cpu().numpy(),
                       title=(f"Follower {tx}: {Style} {Pair}"),
                       fps=20, radius=4)
        my_dict = {'caption': "infer",
                   'HML3D_vec': follower_motion.squeeze(),
                   'positions': follower_positions.squeeze().detach().cpu().numpy(),
                    }
        with open(f"{sav_path}/motionllm_{level}_{iterate}_follower_{tx}.pk", "wb") as f:
            pickle.dump(my_dict, f)


            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_leader_GT.mp4"),
                           t2m_kinematic_chain, H3D_GT_Leader_Positions.squeeze().detach().cpu().numpy(),
                           title=(f"Leader GT: {Style} {Pair}"),
                           fps=20, radius=4)
            my_dict = {'caption': "infer",
                       'HML3D_vec': H3D_GT_Leader.squeeze(),
                       'positions': H3D_GT_Leader_Positions.squeeze().detach().cpu().numpy(),
                        }
            with open(f"{sav_path}/motionllm_{level}_{iterate}_leader_GT.pk", "wb") as f:
                pickle.dump(my_dict, f)


            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_follower_GT.mp4"),
                           t2m_kinematic_chain, H3D_GT_Follower_positions.squeeze().detach().cpu().numpy(),
                           title=(f"Follower GT: {Style} {Pair}"),
                           fps=20, radius=4)
            my_dict = {'caption': "infer",
                       'HML3D_vec': H3D_GT_Follower.squeeze(),
                       'positions': H3D_GT_Follower_positions.squeeze().detach().cpu().numpy(),
                    }
            with open(f"{sav_path}/motionllm_{level}_{iterate}_follower_GT.pk", "wb") as f:
                pickle.dump(my_dict, f)







def load_data_pair(style='beginner', pair="Pair1", take='take1_1', start_sec=0, end_sec=5):
    args = get_args_parser()
    # args.is_MDM = True # to get HumanML3D outputs
    args.is_MDM = True
    train_dataset = Salsa_Dataset(args,
                    lmdb_dir='dataset_processed/lmdb_Salsa_pair/lmdb_train',
                    n_poses=100,
                    subdivision_stride=50,
                    pose_resampling_fps=20)

    item = train_dataset.__getitem__(1)
    print()

    #   Workaround striding
    #   pick style and pair:
    # style = "beginner"
    # pair = "Pair1"
    # start_sec, end_sec = 20, 35
    # find the indices '-->'
    list_of_indecis = []
    list_of_aux = []

    for index in range(len(train_dataset)):
        if not args.is_MDM:
            level, ms_desc_L, ms_des_F, \
                vq_tokens_L, vq_tokens_F, audio_tokens, aux_info = train_dataset.__getitem__(index)
        else:
            level, HML3D_L, vq_tokens_L, HML3D_F, vq_tokens_F, audio_tokens, aux_info = train_dataset.__getitem__(index)
        level = style #todo fixeme: cheating at the inference time due to incorrect lmdb processing at the time.
        if level==style and \
                aux_info['start_time']>=start_sec and \
                aux_info['end_time'] <= end_sec and \
                pair==aux_info['vid'][:5] and \
                take in aux_info['vid']: #
            if len(list_of_aux)>0:
                if aux_info['start_time']>=list_of_aux[-1]['end_time']:
                    list_of_indecis.append(index)
                    list_of_aux.append(aux_info)
            else:
                list_of_indecis.append(index)
                list_of_aux.append(aux_info)

        if len(list_of_indecis)>0 and \
                list_of_aux[-1]['end_time'] == end_sec and \
                list_of_aux[0]['start_time'] == start_sec:
            break



    items = [train_dataset.__getitem__(index) for index in list_of_indecis]
    return items

if __name__ == "__main__":
    demo_parser = demo_argparse.ArgumentParser(description="Run demo with a task name")
    demo_parser.add_argument(
        '--task',
        type=str,
        choices=["follower_to_leader", "leader_to_follower", "caption_to_motion"],
        help="Name of the task"
    )
    demo_args = demo_parser.parse_args()

    if demo_args.task is None:
        print("Please choose a task:")
        print("1. baseline")
        print("2. follower_to_leader")
        print("3. leader_to_follower")
        print("4. caption_to_motion")
        choice = input("Enter the task name or number: ").strip()

        task_map = {
            "1": "baseline",
            "2": "follower_to_leader",
            "3": "leader_to_follower",
            "4": "caption_to_motion"
        }

        task_name = task_map.get(choice, choice if choice in task_map.values() else None)

        if task_name is None:
            print("Invalid choice. Exiting.")
            exit(1)
    else:
        task_name = demo_args.task

    print(f"Running task: {task_name}")

    # motion_agent_demo()


    # motionllm_evaluation_qualitative()

    motionllm_evaluation_qualitative_pairs(task_name)

